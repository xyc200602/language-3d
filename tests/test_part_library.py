"""Tests for the part library management system.

Covers:
1. Data integrity: PartTemplate required fields
2. Search: Chinese/English/category filtering
3. Parameter resolution: defaults, range validation
4. Script generation: placeholder substitution
5. Tool registration: 6 tools in registry
6. Tool execution: search/get/list output
7. Category tree structure
8. Regression: no import errors
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# 1. Data integrity — every PartTemplate has required fields
# ---------------------------------------------------------------------------

class TestCatalogDataIntegrity:
    def test_catalog_has_15_templates(self):
        from lang3d.knowledge.parts_catalog import PART_CATALOG
        assert len(PART_CATALOG) == 15

    def test_all_templates_have_required_fields(self):
        from lang3d.knowledge.parts_catalog import PART_CATALOG
        for tid, t in PART_CATALOG.items():
            assert t.id == tid, f"Template key mismatch: {tid} vs {t.id}"
            assert t.name_en, f"Missing name_en for {tid}"
            assert t.name_cn, f"Missing name_cn for {tid}"
            assert t.category, f"Missing category for {tid}"
            assert t.subcategory, f"Missing subcategory for {tid}"
            assert t.description, f"Missing description for {tid}"
            assert len(t.parameters) > 0, f"No parameters for {tid}"
            assert t.fc_script_template, f"Missing fc_script_template for {tid}"

    def test_all_parameters_have_required_fields(self):
        from lang3d.knowledge.parts_catalog import PART_CATALOG
        for tid, t in PART_CATALOG.items():
            for p in t.parameters:
                assert p.name, f"Missing param name in {tid}"
                assert p.display_name_cn, f"Missing display_name_cn for {p.name} in {tid}"
                assert p.min_value <= p.default <= p.max_value, (
                    f"Default out of range for {p.name} in {tid}: "
                    f"{p.min_value} <= {p.default} <= {p.max_value}"
                )

    def test_all_ids_are_unique(self):
        from lang3d.knowledge.parts_catalog import PART_CATALOG
        ids = [t.id for t in PART_CATALOG.values()]
        assert len(ids) == len(set(ids))

    def test_expected_template_ids(self):
        from lang3d.knowledge.parts_catalog import PART_CATALOG
        expected = {
            "socket_head_cap_screw", "hex_nut", "flat_washer", "hex_bolt",
            "bearing_608", "bearing_623", "bearing_625",
            "servo_sg90", "servo_mg996r", "nema17_stepper",
            "linear_shaft", "flexible_coupling",
            "spur_gear", "l_bracket", "mounting_plate",
        }
        assert set(PART_CATALOG.keys()) == expected


# ---------------------------------------------------------------------------
# 2. Search — Chinese/English/category
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_chinese(self):
        from lang3d.knowledge.parts_catalog import search_parts
        results = search_parts(query="螺钉")
        ids = {t.id for t in results}
        assert "socket_head_cap_screw" in ids
        assert "hex_nut" not in ids

    def test_search_english(self):
        from lang3d.knowledge.parts_catalog import search_parts
        results = search_parts(query="bearing")
        ids = {t.id for t in results}
        assert "bearing_608" in ids
        assert "bearing_623" in ids
        assert "bearing_625" in ids
        assert "socket_head_cap_screw" not in ids

    def test_search_servo(self):
        from lang3d.knowledge.parts_catalog import search_parts
        results = search_parts(query="舵机")
        ids = {t.id for t in results}
        assert "servo_sg90" in ids
        assert "servo_mg996r" in ids

    def test_search_category_fastener(self):
        from lang3d.knowledge.parts_catalog import search_parts
        results = search_parts(category="fastener")
        assert len(results) == 4  # screw, nut, washer, bolt
        for t in results:
            assert t.category == "fastener"

    def test_search_category_bearing(self):
        from lang3d.knowledge.parts_catalog import search_parts
        results = search_parts(category="bearing")
        assert len(results) == 3

    def test_search_category_actuator(self):
        from lang3d.knowledge.parts_catalog import search_parts
        results = search_parts(category="actuator")
        assert len(results) == 3  # sg90, mg996r, nema17

    def test_search_category_gear(self):
        from lang3d.knowledge.parts_catalog import search_parts
        results = search_parts(category="gear")
        assert len(results) == 1
        assert results[0].id == "spur_gear"

    def test_search_empty_query(self):
        from lang3d.knowledge.parts_catalog import search_parts
        results = search_parts(query="")
        assert len(results) == 15  # returns all

    def test_search_no_match(self):
        from lang3d.knowledge.parts_catalog import search_parts
        results = search_parts(query="nonexistent_xyz")
        assert len(results) == 0

    def test_search_subcategory_screw(self):
        from lang3d.knowledge.parts_catalog import search_parts
        results = search_parts(category="screw")
        assert len(results) == 1
        assert results[0].id == "socket_head_cap_screw"

    def test_search_tags(self):
        from lang3d.knowledge.parts_catalog import search_parts
        results = search_parts(tags=["3D printer"])
        ids = {t.id for t in results}
        assert "bearing_625" in ids  # has "3D printer" in tags
        assert "nema17_stepper" in ids  # has "3D printer" in tags


# ---------------------------------------------------------------------------
# 3. Parameter resolution
# ---------------------------------------------------------------------------

class TestParameterResolution:
    def test_defaults_filled(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters
        t = get_template("socket_head_cap_screw")
        params = resolve_parameters(t)
        assert "thread_diameter" in params
        assert "length" in params
        assert params["thread_diameter"] == 3
        assert params["length"] == 10

    def test_override_params(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters
        t = get_template("socket_head_cap_screw")
        params = resolve_parameters(t, {"thread_diameter": 5, "length": 25})
        assert params["thread_diameter"] == 5
        assert params["length"] == 25
        # head_diameter gets default
        assert "head_diameter" in params

    def test_partial_override(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters
        t = get_template("socket_head_cap_screw")
        params = resolve_parameters(t, {"thread_diameter": 8})
        assert params["thread_diameter"] == 8
        assert params["length"] == 10  # default

    def test_out_of_range_raises(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters
        t = get_template("socket_head_cap_screw")
        with pytest.raises(ValueError, match="out of range"):
            resolve_parameters(t, {"thread_diameter": 100})  # max is 30

    def test_below_min_raises(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters
        t = get_template("socket_head_cap_screw")
        with pytest.raises(ValueError, match="out of range"):
            resolve_parameters(t, {"length": -1})

    def test_none_params_uses_defaults(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters
        t = get_template("bearing_608")
        params = resolve_parameters(t, None)
        assert params["inner_diameter"] == 8
        assert params["outer_diameter"] == 22
        assert params["width"] == 7

    def test_standard_size_variant(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters
        t = get_template("socket_head_cap_screw")
        variant = t.standard_sizes[3]  # M5x20
        params = resolve_parameters(t, variant)
        assert params["thread_diameter"] == 5
        assert params["length"] == 20


# ---------------------------------------------------------------------------
# 4. Script generation
# ---------------------------------------------------------------------------

class TestScriptGeneration:
    def test_script_has_no_placeholders(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters, format_fc_script
        t = get_template("socket_head_cap_screw")
        params = resolve_parameters(t, {"thread_diameter": 5, "length": 20, "head_diameter": 8.5})
        script = format_fc_script(t, params)
        assert "{" not in script or "}" not in script or "FreeCAD.Vector" in script
        assert "makeCylinder" in script
        assert "doc" in script

    def test_bearing_script(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters, format_fc_script
        t = get_template("bearing_608")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "OuterRing" in script
        assert "InnerRing" in script
        assert "8" in script  # inner diameter

    def test_gear_script(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters, format_fc_script
        t = get_template("spur_gear")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "SpurGear" in script
        assert "bore_diameter" not in script.lower()  # substituted

    def test_all_templates_generate_valid_script(self):
        from lang3d.knowledge.parts_catalog import PART_CATALOG, resolve_parameters, format_fc_script
        for tid, t in PART_CATALOG.items():
            params = resolve_parameters(t)
            script = format_fc_script(t, params)
            assert "import FreeCAD" in script, f"Missing FreeCAD import in {tid}"
            assert "doc" in script, f"Missing doc in {tid}"


# ---------------------------------------------------------------------------
# 5. Tool registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    def test_six_tools_registered(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.part_library import register_part_library_tools
        registry = ToolRegistry()
        register_part_library_tools(registry)
        tools = registry.list_tools()
        expected = {"part_search", "part_get", "part_generate", "part_list", "part_import", "part_save"}
        assert expected == set(tools)

    def test_tool_definitions(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.part_library import register_part_library_tools
        registry = ToolRegistry()
        register_part_library_tools(registry)
        defs = registry.get_all_definitions()
        assert len(defs) == 6
        for d in defs:
            assert d.name.startswith("part_")
            assert d.description
            assert d.parameters


# ---------------------------------------------------------------------------
# 6. Tool execution (no FreeCAD required for search/get/list)
# ---------------------------------------------------------------------------

class TestToolExecution:
    def _make_registry(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.part_library import register_part_library_tools
        registry = ToolRegistry()
        register_part_library_tools(registry)
        return registry

    def test_part_search_execution(self):
        registry = self._make_registry()
        result = registry.execute("part_search", query="bearing")
        assert "bearing" in result.lower() or "轴承" in result
        assert "608" in result

    def test_part_search_empty(self):
        registry = self._make_registry()
        result = registry.execute("part_search", query="")
        assert "15" in result  # should show all 15

    def test_part_get_execution(self):
        registry = self._make_registry()
        result = registry.execute("part_get", part_id="bearing_608")
        assert "608" in result
        assert "内径" in result or "inner_diameter" in result.lower()

    def test_part_get_not_found(self):
        registry = self._make_registry()
        result = registry.execute("part_get", part_id="nonexistent")
        assert "未找到" in result or "not found" in result.lower()

    def test_part_list_execution(self):
        registry = self._make_registry()
        result = registry.execute("part_list")
        assert "15" in result
        assert "fastener" in result

    def test_part_list_by_category(self):
        registry = self._make_registry()
        result = registry.execute("part_list", category="bearing")
        assert "3" in result  # 3 bearings
        assert "bearing" in result.lower()

    def test_part_list_by_subcategory(self):
        registry = self._make_registry()
        result = registry.execute("part_list", subcategory="servo")
        assert "servo" in result.lower()


# ---------------------------------------------------------------------------
# 7. Category tree
# ---------------------------------------------------------------------------

class TestCategoryTree:
    def test_category_tree_structure(self):
        from lang3d.knowledge.parts_catalog import CATEGORY_TREE
        expected_categories = {"fastener", "bearing", "actuator", "shaft", "gear", "structural"}
        assert set(CATEGORY_TREE.keys()) == expected_categories

    def test_fastener_subcategories(self):
        from lang3d.knowledge.parts_catalog import CATEGORY_TREE
        assert set(CATEGORY_TREE["fastener"]) == {"screw", "nut", "washer", "bolt"}

    def test_template_categories_match_tree(self):
        from lang3d.knowledge.parts_catalog import PART_CATALOG, CATEGORY_TREE
        for t in PART_CATALOG.values():
            assert t.category in CATEGORY_TREE, f"Unknown category: {t.category}"
            assert t.subcategory in CATEGORY_TREE[t.category], (
                f"Unknown subcategory: {t.subcategory} in {t.category}"
            )


# ---------------------------------------------------------------------------
# 8. Regression — no import errors
# ---------------------------------------------------------------------------

class TestRegression:
    def test_import_catalog(self):
        from lang3d.knowledge.parts_catalog import PART_CATALOG, CATEGORY_TREE
        assert PART_CATALOG
        assert CATEGORY_TREE

    def test_import_part_library_tools(self):
        from lang3d.tools.part_library import register_part_library_tools
        assert callable(register_part_library_tools)

    def test_base_category_registered(self):
        from lang3d.tools.base import TOOL_CATEGORIES
        assert "part_library" in TOOL_CATEGORIES
        assert "part_search" in TOOL_CATEGORIES["part_library"]

    def test_step_tool_categories(self):
        from lang3d.tools.base import STEP_TOOL_CATEGORIES
        assert "part_library" in STEP_TOOL_CATEGORIES["modeling"]
