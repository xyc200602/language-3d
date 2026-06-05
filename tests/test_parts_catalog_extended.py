"""Tests for extended parts catalog (Task 53).

Covers:
  - New part templates (wheel, mecanum, hub, bracket, standoff, battery, chassis, corner, pcb_mount)
  - CATEGORY_TREE updates
  - Subsystem search
  - Compatible parts query
  - Script generation for new parts
"""

import pytest

from lang3d.knowledge.parts_catalog import (
    CATEGORY_TREE,
    PART_CATALOG,
    PartTemplate,
    search_by_subsystem,
    find_compatible_parts,
    search_parts,
    get_template,
    resolve_parameters,
    format_fc_script,
)


# ============================================================================
# Catalog size and integrity
# ============================================================================


class TestCatalogSize:
    def test_at_least_24_templates(self):
        assert len(PART_CATALOG) >= 24

    def test_new_parts_present(self):
        expected = [
            "wheel_simple", "wheel_mecanum", "hub_adapter",
            "motor_bracket_u", "standoff_hex", "battery_holder_18650",
            "chassis_plate", "corner_bracket", "pcb_mount",
        ]
        for pid in expected:
            assert pid in PART_CATALOG, f"{pid} missing from catalog"

    def test_old_parts_still_present(self):
        old = [
            "socket_head_cap_screw", "hex_nut", "flat_washer",
            "bearing_608", "servo_sg90", "l_bracket",
        ]
        for pid in old:
            assert pid in PART_CATALOG, f"{pid} was removed"


class TestCategoryTree:
    def test_mobile_base_category(self):
        assert "mobile_base" in CATEGORY_TREE
        assert "wheel" in CATEGORY_TREE["mobile_base"]
        assert "chassis" in CATEGORY_TREE["mobile_base"]

    def test_mounting_category(self):
        assert "mounting" in CATEGORY_TREE
        assert "standoff" in CATEGORY_TREE["mounting"]
        assert "battery_holder" in CATEGORY_TREE["mounting"]

    def test_categories_match_templates(self):
        for template in PART_CATALOG.values():
            assert template.category in CATEGORY_TREE or template.subcategory in [
                s for subs in CATEGORY_TREE.values() for s in subs
            ], f"{template.id} has unmatched category '{template.category}'"


# ============================================================================
# Individual new part templates
# ============================================================================


class TestWheelSimple:
    def test_exists(self):
        t = get_template("wheel_simple")
        assert t is not None
        assert t.category == "mobile_base"

    def test_parameters(self):
        t = get_template("wheel_simple")
        names = {p.name for p in t.parameters}
        assert "outer_diameter" in names
        assert "width" in names
        assert "hub_diameter" in names

    def test_script_generation(self):
        t = get_template("wheel_simple")
        params = resolve_parameters(t, {"outer_diameter": 65, "width": 26, "hub_diameter": 5})
        script = format_fc_script(t, params)
        assert "makeCylinder" in script
        assert "65" in script

    def test_standard_sizes(self):
        t = get_template("wheel_simple")
        assert len(t.standard_sizes) >= 2


class TestWheelMecanum:
    def test_exists(self):
        t = get_template("wheel_mecanum")
        assert t is not None

    def test_script_has_rollers(self):
        t = get_template("wheel_mecanum")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "roller" in script.lower() or "fuse" in script


class TestHubAdapter:
    def test_exists(self):
        t = get_template("hub_adapter")
        assert t is not None
        assert t.subcategory == "hub"

    def test_script(self):
        t = get_template("hub_adapter")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "makeCylinder" in script


class TestMotorBracket:
    def test_exists(self):
        t = get_template("motor_bracket_u")
        assert t is not None
        assert t.subcategory == "motor_bracket"

    def test_script(self):
        t = get_template("motor_bracket_u")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "makeBox" in script
        assert "motor_diameter" in t.fc_script_template


class TestStandoff:
    def test_exists(self):
        t = get_template("standoff_hex")
        assert t is not None
        assert t.category == "mounting"

    def test_standard_sizes(self):
        t = get_template("standoff_hex")
        assert len(t.standard_sizes) >= 2


class TestBatteryHolder:
    def test_exists(self):
        t = get_template("battery_holder_18650")
        assert t is not None
        assert t.subcategory == "battery_holder"

    def test_num_cells_param(self):
        t = get_template("battery_holder_18650")
        names = {p.name for p in t.parameters}
        assert "num_cells" in names
        assert "cell_diameter" in names


class TestChassisPlate:
    def test_exists(self):
        t = get_template("chassis_plate")
        assert t is not None
        assert t.subcategory == "chassis"

    def test_grid_params(self):
        t = get_template("chassis_plate")
        names = {p.name for p in t.parameters}
        assert "grid_x" in names
        assert "grid_y" in names


class TestCornerBracket:
    def test_exists(self):
        t = get_template("corner_bracket")
        assert t is not None

    def test_script(self):
        t = get_template("corner_bracket")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "CornerBracket" in script


class TestPCBMount:
    def test_exists(self):
        t = get_template("pcb_mount")
        assert t is not None
        assert t.subcategory == "pcb_mount"


# ============================================================================
# Subsystem search
# ============================================================================


class TestSubsystemSearch:
    def test_mobile_base_returns_parts(self):
        parts = search_by_subsystem("mobile_base")
        assert len(parts) >= 5
        ids = {p.id for p in parts}
        assert "wheel_simple" in ids
        assert "chassis_plate" in ids

    def test_mounting_returns_parts(self):
        parts = search_by_subsystem("mounting")
        assert len(parts) >= 3
        ids = {p.id for p in parts}
        assert "standoff_hex" in ids

    def test_unknown_subsystem_returns_empty(self):
        parts = search_by_subsystem("nonexistent")
        assert parts == []


# ============================================================================
# Compatible parts
# ============================================================================


class TestCompatibleParts:
    def test_wheel_compatible(self):
        compat = find_compatible_parts("wheel_simple")
        ids = {p.id for p in compat}
        # Should include hub_adapter, motor_bracket (same subsystem)
        assert "hub_adapter" in ids or "motor_bracket_u" in ids

    def test_no_self_match(self):
        compat = find_compatible_parts("wheel_simple")
        ids = {p.id for p in compat}
        assert "wheel_simple" not in ids

    def test_unknown_returns_empty(self):
        compat = find_compatible_parts("nonexistent_part")
        assert compat == []

    def test_dimension_match(self):
        # Hub adapter has shaft_diameter → should match coupling
        compat = find_compatible_parts("hub_adapter")
        ids = {p.id for p in compat}
        assert "flexible_coupling" in ids


# ============================================================================
# Search integration
# ============================================================================


class TestSearchIntegration:
    def test_search_wheel(self):
        results = search_parts(query="wheel")
        assert len(results) >= 2
        ids = {r.id for r in results}
        assert "wheel_simple" in ids
        assert "wheel_mecanum" in ids

    def test_search_category_mobile_base(self):
        results = search_parts(category="mobile_base")
        assert len(results) >= 5

    def test_search_chinese(self):
        results = search_parts(query="底盘")
        assert len(results) >= 1
        assert any(r.id == "chassis_plate" for r in results)

    def test_search_tag(self):
        results = search_parts(tags=["18650"])
        assert len(results) >= 1
