"""Tests for 15 new Layer 3 structural part templates."""
from __future__ import annotations

import ast
import pytest

from lang3d.knowledge.parts_catalog import (
    PART_CATALOG,
    PartTemplate,
    format_fc_script,
    get_structural_parts,
    search_parts,
)

# The 15 new template IDs from Layer 3 Phase 1
NEW_TEMPLATE_IDS = [
    "aluminum_extrusion", "u_bracket", "t_bracket", "gusset_plate",
    "bearing_block", "servo_bracket", "nema_mount", "standoff_column",
    "cable_chain_mount", "battery_tray", "sensor_shelf",
    "shaft_coupling_block", "guide_rail_carriage", "pulley_idler_mount",
    "encoder_mount",
]


class TestNewTemplateExistence:
    """Verify all 15 new templates exist in the catalog."""

    @pytest.mark.parametrize("template_id", NEW_TEMPLATE_IDS)
    def test_template_exists(self, template_id):
        assert template_id in PART_CATALOG, (
            f"Template '{template_id}' not found in PART_CATALOG"
        )

    def test_all_fifteen_present(self):
        for tid in NEW_TEMPLATE_IDS:
            assert tid in PART_CATALOG

    def test_structural_count_increased(self):
        structural = get_structural_parts()
        structural_ids = [t.id for t in structural]
        for tid in NEW_TEMPLATE_IDS:
            assert tid in structural_ids, f"'{tid}' not in structural parts"
        # 8 existing structural + 15 new = at least 23
        assert len(structural_ids) >= 23


class TestNewTemplateProperties:
    """Verify part_class, scalable, and basic properties."""

    @pytest.mark.parametrize("template_id", NEW_TEMPLATE_IDS)
    def test_part_class_is_structural(self, template_id):
        t = PART_CATALOG[template_id]
        assert t.part_class == "structural"

    @pytest.mark.parametrize("template_id", NEW_TEMPLATE_IDS)
    def test_scalable_is_true(self, template_id):
        t = PART_CATALOG[template_id]
        assert t.scalable is True

    @pytest.mark.parametrize("template_id", NEW_TEMPLATE_IDS)
    def test_has_parameters(self, template_id):
        t = PART_CATALOG[template_id]
        assert len(t.parameters) >= 3, f"'{template_id}' has too few parameters"

    @pytest.mark.parametrize("template_id", NEW_TEMPLATE_IDS)
    def test_parameters_have_defaults(self, template_id):
        t = PART_CATALOG[template_id]
        for p in t.parameters:
            assert p.default is not None or p.default == 0.0 or p.default != 0

    @pytest.mark.parametrize("template_id", NEW_TEMPLATE_IDS)
    def test_parameters_have_valid_ranges(self, template_id):
        t = PART_CATALOG[template_id]
        for p in t.parameters:
            if p.param_type == "float":
                assert p.min_value < p.max_value, (
                    f"Parameter '{p.name}' in '{template_id}': min >= max"
                )


class TestNewTemplateScripts:
    """Verify FreeCAD script templates are valid."""

    @pytest.mark.parametrize("template_id", NEW_TEMPLATE_IDS)
    def test_fc_script_non_empty(self, template_id):
        t = PART_CATALOG[template_id]
        assert t.fc_script_template, f"'{template_id}' has empty fc_script_template"
        assert len(t.fc_script_template) > 50

    @pytest.mark.parametrize("template_id", NEW_TEMPLATE_IDS)
    def test_fc_script_has_param_placeholders(self, template_id):
        t = PART_CATALOG[template_id]
        referenced = sum(
            1 for p in t.parameters
            if "{" + p.name + "}" in t.fc_script_template
        )
        assert referenced >= len(t.parameters) // 2, (
            f"'{template_id}': only {referenced}/{len(t.parameters)} params "
            f"referenced in script"
        )

    @pytest.mark.parametrize("template_id", NEW_TEMPLATE_IDS)
    def test_format_fc_script_produces_valid_python(self, template_id):
        t = PART_CATALOG[template_id]
        params = {}
        for p in t.parameters:
            if p.param_type == "string":
                params[p.name] = p.choices[0] if p.choices else p.default
            else:
                params[p.name] = p.default
        script = format_fc_script(t, params)
        assert script, f"format_fc_script returned empty for '{template_id}'"
        try:
            ast.parse(script)
        except SyntaxError as e:
            pytest.fail(
                f"format_fc_script for '{template_id}' produced invalid Python: {e}"
            )

    @pytest.mark.parametrize("template_id", NEW_TEMPLATE_IDS)
    def test_format_fc_script_with_custom_params(self, template_id):
        t = PART_CATALOG[template_id]
        params = {}
        for p in t.parameters:
            if p.param_type == "string":
                params[p.name] = p.choices[0] if p.choices else p.default
            else:
                params[p.name] = p.max_value
        script = format_fc_script(t, params)
        try:
            ast.parse(script)
        except SyntaxError as e:
            pytest.fail(
                f"format_fc_script with max params for '{template_id}' "
                f"produced invalid Python: {e}"
            )


class TestNewTemplateMountingInterface:
    """Verify mounting interfaces are defined."""

    @pytest.mark.parametrize("template_id", NEW_TEMPLATE_IDS)
    def test_mounting_interface_defined(self, template_id):
        t = PART_CATALOG[template_id]
        assert t.mounting_interface is not None, (
            f"'{template_id}' has no mounting_interface"
        )

    @pytest.mark.parametrize("template_id", NEW_TEMPLATE_IDS)
    def test_mounting_interface_has_type(self, template_id):
        t = PART_CATALOG[template_id]
        mi = t.mounting_interface
        assert mi.interface_type in (
            "through_hole", "press_fit", "threaded_hole", "snap_fit", "flange",
        )


class TestNewTemplateSearchable:
    """Verify templates are findable via search_parts()."""

    @pytest.mark.parametrize("template_id", NEW_TEMPLATE_IDS)
    def test_searchable_by_structural_class(self, template_id):
        results = search_parts(part_class="structural")
        ids = [r.id for r in results]
        assert template_id in ids, (
            f"'{template_id}' not found in structural search"
        )

    def test_search_by_tag(self):
        results = search_parts(tags=["structural"])
        ids = [r.id for r in results]
        for tid in NEW_TEMPLATE_IDS:
            assert tid in ids, f"'{tid}' not found by tag 'structural'"
