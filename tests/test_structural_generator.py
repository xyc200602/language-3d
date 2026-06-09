"""Tests for the structural part auto-generator (Phase 4)."""
from __future__ import annotations

import ast
import pytest

from lang3d.knowledge.parts_catalog import (
    PART_CATALOG,
    format_fc_script,
    get_mounting_interface,
    search_parts,
)
from lang3d.tools.structural_generator import (
    InterfaceRequirement,
    generate_structural_part,
    register_generated_template,
    _infer_shape,
)


class TestInterfaceRequirement:
    """Test InterfaceRequirement dataclass."""

    def test_creation(self):
        req = InterfaceRequirement("nema17_stepper", "front")
        assert req.functional_part_id == "nema17_stepper"
        assert req.anchor == "front"
        assert req.offset == (0, 0, 0)
        assert req.rotation_deg == 0.0

    def test_with_offset(self):
        req = InterfaceRequirement("bearing_608", "back", offset=(0, 5, 0))
        assert req.offset == (0, 5, 0)


class TestShapeInference:
    """Test _infer_shape from anchor sets."""

    def test_single_face_is_plate(self):
        assert _infer_shape(["front"]) == "plate"

    def test_opposite_faces_is_housing(self):
        assert _infer_shape(["front", "back"]) == "housing"

    def test_adjacent_faces_is_bracket(self):
        assert _infer_shape(["front", "top"]) == "bracket"

    def test_top_bottom_is_housing(self):
        assert _infer_shape(["top", "bottom"]) == "housing"

    def test_three_faces_is_housing(self):
        assert _infer_shape(["front", "back", "top"]) == "housing"


class TestSingleInterfaceGeneration:
    """Test generating a part with one interface (NEMA17 on front)."""

    def test_generates_template(self):
        bracket = generate_structural_part(
            name="nema17_bracket",
            interfaces=[
                InterfaceRequirement("nema17_stepper", "front"),
            ],
        )
        assert bracket is not None
        assert bracket.id == "nema17_bracket"
        assert bracket.part_class == "structural"
        assert bracket.category == "structural"

    def test_fc_script_is_valid_python(self):
        bracket = generate_structural_part(
            name="nema17_bracket",
            interfaces=[
                InterfaceRequirement("nema17_stepper", "front"),
            ],
        )
        params = {p.name: p.default for p in bracket.parameters}
        script = format_fc_script(bracket, params)
        try:
            ast.parse(script)
        except SyntaxError as e:
            pytest.fail(f"Generated script is invalid Python: {e}")

    def test_mounting_interface_has_holes(self):
        bracket = generate_structural_part(
            name="nema17_bracket",
            interfaces=[
                InterfaceRequirement("nema17_stepper", "front"),
            ],
        )
        mi = bracket.mounting_interface
        assert mi is not None
        assert len(mi.holes) == 4  # NEMA17 has 4 holes

    def test_mounting_interface_hole_diameter(self):
        bracket = generate_structural_part(
            name="nema17_bracket",
            interfaces=[
                InterfaceRequirement("nema17_stepper", "front"),
            ],
        )
        for hole in bracket.mounting_interface.holes:
            assert hole.diameter == pytest.approx(3.4, abs=0.1)


class TestDualInterfaceGeneration:
    """Test generating a part with two interfaces (NEMA17 front + bearing back)."""

    def test_generates_template(self):
        bracket = generate_structural_part(
            name="motor_bearing_bracket",
            interfaces=[
                InterfaceRequirement("nema17_stepper", "front"),
                InterfaceRequirement("bearing_608", "back"),
            ],
        )
        assert bracket is not None
        assert bracket.id == "motor_bearing_bracket"

    def test_dual_interface_has_more_holes(self):
        bracket = generate_structural_part(
            name="motor_bearing_bracket",
            interfaces=[
                InterfaceRequirement("nema17_stepper", "front"),
                InterfaceRequirement("bearing_608", "back"),
            ],
        )
        mi = bracket.mounting_interface
        assert mi is not None
        # NEMA17: 4 holes + bearing: may have 0 holes (press_fit) → total ≥ 4
        assert len(mi.holes) >= 4

    def test_fc_script_valid_python(self):
        bracket = generate_structural_part(
            name="motor_bearing_bracket",
            interfaces=[
                InterfaceRequirement("nema17_stepper", "front"),
                InterfaceRequirement("bearing_608", "back"),
            ],
        )
        params = {p.name: p.default for p in bracket.parameters}
        script = format_fc_script(bracket, params)
        try:
            ast.parse(script)
        except SyntaxError as e:
            pytest.fail(f"Dual interface script invalid: {e}")


class TestFallbackBehavior:
    """Test graceful handling of unknown functional parts."""

    def test_unknown_interface_produces_warning(self):
        bracket = generate_structural_part(
            name="custom_part",
            interfaces=[
                InterfaceRequirement("unknown_motor_xyz", "front"),
            ],
        )
        # Should still produce a valid template (fallback plate)
        assert bracket is not None
        assert "unknown" in bracket.notes.lower() or bracket.fc_script_template

    def test_no_interfaces_produces_minimal_plate(self):
        bracket = generate_structural_part(
            name="empty_part",
            interfaces=[],
        )
        assert bracket is not None
        assert bracket.fc_script_template


class TestRegistration:
    """Test registering generated templates in the catalog."""

    def test_register_new_template(self):
        bracket = generate_structural_part(
            name="test_reg_bracket",
            interfaces=[
                InterfaceRequirement("nema17_stepper", "front"),
            ],
        )
        result = register_generated_template(bracket)
        assert result is True
        assert "test_reg_bracket" in PART_CATALOG

        # Clean up
        del PART_CATALOG["test_reg_bracket"]

    def test_register_duplicate_returns_false(self):
        bracket = generate_structural_part(
            name="l_bracket",  # Already exists
            interfaces=[
                InterfaceRequirement("nema17_stepper", "front"),
            ],
        )
        result = register_generated_template(bracket)
        assert result is False

    def test_registered_template_searchable(self):
        bracket = generate_structural_part(
            name="test_search_bracket",
            interfaces=[
                InterfaceRequirement("nema17_stepper", "front"),
            ],
        )
        register_generated_template(bracket)

        results = search_parts(query="test_search_bracket")
        assert len(results) >= 1
        assert results[0].id == "test_search_bracket"

        # Clean up
        del PART_CATALOG["test_search_bracket"]


class TestParameterScaling:
    """Test that generated parameters produce valid scripts at boundary values."""

    def test_min_params_valid_script(self):
        bracket = generate_structural_part(
            name="scale_test",
            interfaces=[
                InterfaceRequirement("nema17_stepper", "front"),
            ],
        )
        params = {p.name: p.min_value for p in bracket.parameters}
        script = format_fc_script(bracket, params)
        try:
            ast.parse(script)
        except SyntaxError as e:
            pytest.fail(f"Script with min params invalid: {e}")

    def test_max_params_valid_script(self):
        bracket = generate_structural_part(
            name="scale_test",
            interfaces=[
                InterfaceRequirement("nema17_stepper", "front"),
            ],
        )
        params = {p.name: p.max_value for p in bracket.parameters}
        script = format_fc_script(bracket, params)
        try:
            ast.parse(script)
        except SyntaxError as e:
            pytest.fail(f"Script with max params invalid: {e}")
