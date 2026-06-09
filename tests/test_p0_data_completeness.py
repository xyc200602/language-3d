"""P0 data completeness tests — XM430 script + MountingInterface coverage.

Covers:
  - P0-1: dynamixel_xm430_w350 fc_script_template is valid
  - P0-2: 11 structural parts have MountingInterface
  - P0-3: 12 functional parts have MountingInterface
  - Coverage metrics
"""

import ast

import pytest

from lang3d.knowledge.parts_catalog import (
    MOUNTING_INTERFACES,
    PART_CATALOG,
    get_mounting_interface,
    format_fc_script,
    resolve_parameters,
)


# ============================================================================
# P0-1: XM430 fc_script_template
# ============================================================================


class TestXM430Script:
    """Validate the DYNAMIXEL XM430-W350-T FreeCAD script."""

    def test_script_nonempty(self):
        t = PART_CATALOG["dynamixel_xm430_w350"]
        assert t.fc_script_template, "fc_script_template must not be empty"

    def test_script_parses(self):
        t = PART_CATALOG["dynamixel_xm430_w350"]
        params = {p.name: p.default for p in t.parameters}
        script = t.fc_script_template.format(**params)
        ast.parse(script)

    def test_script_contains_key_elements(self):
        t = PART_CATALOG["dynamixel_xm430_w350"]
        script = t.fc_script_template
        assert "makeBox" in script
        assert "makeCylinder" in script
        assert "XM430" in script

    def test_script_format_fc_script(self):
        t = PART_CATALOG["dynamixel_xm430_w350"]
        params = {p.name: p.default for p in t.parameters}
        result = format_fc_script(t, params)
        assert result is not None
        assert "import FreeCAD" in result


# ============================================================================
# P0-2: Structural part MountingInterfaces (11 parts)
# ============================================================================

STRUCTURAL_PARTS = [
    "l_bracket",
    "mounting_plate",
    "motor_bracket_u",
    "chassis_plate",
    "corner_bracket",
    "standoff_hex",
    "pcb_mount",
    "battery_holder_18650",
    "wheel_simple",
    "wheel_mecanum",
    "hub_adapter",
]


class TestStructuralInterfaces:
    """Every structural part must have a valid MountingInterface."""

    @pytest.mark.parametrize("part_id", STRUCTURAL_PARTS)
    def test_has_mounting_interface(self, part_id):
        mi = get_mounting_interface(part_id)
        assert mi is not None, f"{part_id} has no MountingInterface"

    @pytest.mark.parametrize("part_id", STRUCTURAL_PARTS)
    def test_interface_type_valid(self, part_id):
        mi = get_mounting_interface(part_id)
        assert mi.interface_type in (
            "through_hole", "threaded_hole", "press_fit",
            "snap_fit", "flange", "shaft",
        )

    @pytest.mark.parametrize("part_id", STRUCTURAL_PARTS)
    def test_holes_have_positive_diameter(self, part_id):
        mi = get_mounting_interface(part_id)
        if mi.holes:
            for h in mi.holes:
                assert h.diameter > 0, (
                    f"{part_id} has hole with non-positive diameter"
                )


# ============================================================================
# P0-3: Functional part MountingInterfaces (12 parts)
# ============================================================================

FUNCTIONAL_PARTS = [
    "motor_tt",
    "motor_jgb37_520",
    "bldc_motor_5010",
    "bldc_motor_2208",
    "linear_bearing_lm8uu",
    "linear_bearing_lm10uu",
    "linear_bearing_lm12uu",
    "limit_switch_kw12",
    "driver_tb6612fng",
    "power_lm2596_buck",
    "gt2_pulley",
    "linear_shaft",
]


class TestFunctionalInterfaces:
    """Every targeted functional part must have a valid MountingInterface."""

    @pytest.mark.parametrize("part_id", FUNCTIONAL_PARTS)
    def test_has_mounting_interface(self, part_id):
        mi = get_mounting_interface(part_id)
        assert mi is not None, f"{part_id} has no MountingInterface"

    @pytest.mark.parametrize("part_id", FUNCTIONAL_PARTS)
    def test_interface_type_valid(self, part_id):
        mi = get_mounting_interface(part_id)
        assert mi.interface_type in (
            "through_hole", "threaded_hole", "press_fit",
            "snap_fit", "flange", "shaft",
        )

    @pytest.mark.parametrize("part_id", FUNCTIONAL_PARTS)
    def test_press_fit_has_bore(self, part_id):
        mi = get_mounting_interface(part_id)
        if mi.interface_type == "press_fit":
            assert mi.bore_diameter > 0, (
                f"{part_id} press_fit must have bore_diameter > 0"
            )

    @pytest.mark.parametrize("part_id", FUNCTIONAL_PARTS)
    def test_through_hole_has_holes(self, part_id):
        mi = get_mounting_interface(part_id)
        if mi.interface_type in ("through_hole", "threaded_hole"):
            assert len(mi.holes) > 0, (
                f"{part_id} {mi.interface_type} must have bolt holes"
            )


# ============================================================================
# Coverage metrics
# ============================================================================


class TestCoverageMetrics:
    """Overall MountingInterface coverage targets."""

    def test_structural_coverage_ge_80(self):
        structural_ids = [
            pid for pid, t in PART_CATALOG.items()
            if t.part_class == "structural"
        ]
        if not structural_ids:
            pytest.skip("No structural parts in catalog")
        covered = sum(
            1 for pid in structural_ids
            if get_mounting_interface(pid) is not None
        )
        coverage = covered / len(structural_ids)
        assert coverage >= 0.80, (
            f"Structural coverage {coverage:.0%} < 80% "
            f"({covered}/{len(structural_ids)})"
        )

    def test_functional_coverage_ge_60(self):
        functional_ids = [
            pid for pid, t in PART_CATALOG.items()
            if t.part_class == "functional"
        ]
        if not functional_ids:
            pytest.skip("No functional parts in catalog")
        covered = sum(
            1 for pid in functional_ids
            if get_mounting_interface(pid) is not None
        )
        coverage = covered / len(functional_ids)
        assert coverage >= 0.60, (
            f"Functional coverage {coverage:.0%} < 60% "
            f"({covered}/{len(functional_ids)})"
        )

    def test_all_functional_fc_scripts_nonempty(self):
        functional_ids = [
            pid for pid, t in PART_CATALOG.items()
            if t.part_class == "functional"
        ]
        empty = [
            pid for pid in functional_ids
            if not PART_CATALOG[pid].fc_script_template
        ]
        assert not empty, (
            f"Functional parts with empty fc_script_template: {empty}"
        )
