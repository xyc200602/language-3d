"""Tests for the part library management system.

Covers:
1. Data integrity: PartTemplate required fields
2. Search: Chinese/English/category filtering
3. Parameter resolution: defaults, range validation, string type
4. Script generation: placeholder substitution, multi-script selection
5. Tool registration: 7 tools in registry
6. Tool execution: search/get/list output
7. Category tree structure
8. Regression: no import errors
9. PartsStore: JSON persistence
10. Realistic thread/gear scripts
11. Print analysis tool
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

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
            assert hasattr(t, 'quality_levels'), f"Missing quality_levels for {tid}"
            assert hasattr(t, 'fc_script_alternatives'), f"Missing fc_script_alternatives for {tid}"

    def test_all_parameters_have_required_fields(self):
        from lang3d.knowledge.parts_catalog import PART_CATALOG
        for tid, t in PART_CATALOG.items():
            for p in t.parameters:
                assert p.name, f"Missing param name in {tid}"
                assert p.display_name_cn, f"Missing display_name_cn for {p.name} in {tid}"
                assert hasattr(p, 'param_type'), f"Missing param_type for {p.name} in {tid}"
                assert hasattr(p, 'choices'), f"Missing choices for {p.name} in {tid}"
                if p.param_type == "float":
                    assert isinstance(p.default, (int, float)), (
                        f"Float param {p.name} in {tid} has non-numeric default: {p.default}"
                    )
                    assert p.min_value <= p.default <= p.max_value, (
                        f"Default out of range for {p.name} in {tid}: "
                        f"{p.min_value} <= {p.default} <= {p.max_value}"
                    )
                elif p.param_type == "string":
                    assert isinstance(p.default, str), (
                        f"String param {p.name} in {tid} has non-string default: {p.default}"
                    )
                    assert p.choices, f"String param {p.name} in {tid} has no choices"

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

    def test_quality_levels_exist(self):
        from lang3d.knowledge.parts_catalog import PART_CATALOG
        for tid, t in PART_CATALOG.items():
            assert isinstance(t.quality_levels, list), f"quality_levels not a list for {tid}"
            assert len(t.quality_levels) >= 1, f"Empty quality_levels for {tid}"
            assert "simplified" in t.quality_levels, f"simplified not in quality_levels for {tid}"


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

    def test_string_param_valid_choice(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters
        t = get_template("socket_head_cap_screw")
        params = resolve_parameters(t, {"thread_detail": "realistic"})
        assert params["thread_detail"] == "realistic"

    def test_string_param_invalid_choice_raises(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters
        t = get_template("socket_head_cap_screw")
        with pytest.raises(ValueError, match="not in choices"):
            resolve_parameters(t, {"thread_detail": "ultra_hq"})

    def test_string_param_default(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters
        t = get_template("socket_head_cap_screw")
        params = resolve_parameters(t)
        assert params["thread_detail"] == "simplified"

    def test_gear_string_param_tooth_detail(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters
        t = get_template("spur_gear")
        params = resolve_parameters(t, {"tooth_detail": "realistic"})
        assert params["tooth_detail"] == "realistic"
        assert params["pressure_angle"] == 20.0
        assert params["backlash"] == 0.1


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

    def test_realistic_thread_script_selected(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters, format_fc_script
        t = get_template("socket_head_cap_screw")
        params = resolve_parameters(t, {"thread_detail": "realistic", "thread_pitch": 1.0})
        script = format_fc_script(t, params)
        assert "makeHelix" in script
        assert "makePipe" in script
        assert "realistic" in script or "minor_r" in script

    def test_realistic_hex_bolt_script(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters, format_fc_script
        t = get_template("hex_bolt")
        params = resolve_parameters(t, {"thread_detail": "realistic", "thread_pitch": 1.0})
        script = format_fc_script(t, params)
        assert "makeHelix" in script
        assert "makePipe" in script

    def test_realistic_hex_nut_script(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters, format_fc_script
        t = get_template("hex_nut")
        params = resolve_parameters(t, {"thread_detail": "realistic", "thread_pitch": 1.0})
        script = format_fc_script(t, params)
        assert "makeHelix" in script
        assert "makePipe" in script

    def test_simplified_script_backward_compatible(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters, format_fc_script
        t = get_template("socket_head_cap_screw")
        params = resolve_parameters(t, {"thread_detail": "simplified"})
        script = format_fc_script(t, params)
        assert "makeHelix" not in script
        assert "makeCylinder" in script

    def test_realistic_gear_script(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters, format_fc_script
        t = get_template("spur_gear")
        params = resolve_parameters(t, {
            "tooth_detail": "realistic",
            "pressure_angle": 20.0,
            "backlash": 0.1,
        })
        script = format_fc_script(t, params)
        assert "involute" in script.lower()
        assert "makePolygon" in script

    def test_simplified_gear_backward_compatible(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters, format_fc_script
        t = get_template("spur_gear")
        params = resolve_parameters(t, {"tooth_detail": "simplified"})
        script = format_fc_script(t, params)
        assert "involute" not in script.lower()
        assert "makeCylinder" in script

    def test_m6_realistic_thread_no_unreplaced_placeholders(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters, format_fc_script
        t = get_template("socket_head_cap_screw")
        params = resolve_parameters(t, {
            "thread_diameter": 6, "length": 25, "head_diameter": 10,
            "thread_detail": "realistic", "thread_pitch": 1.0,
        })
        script = format_fc_script(t, params)
        # Check no unreplaced parameter placeholders
        import re
        unreplaced = re.findall(r'\{[a-zA-Z_][a-zA-Z0-9_]*\}', script)
        assert len(unreplaced) == 0, f"Unreplaced placeholders: {unreplaced}"


# ---------------------------------------------------------------------------
# 5. Tool registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    def test_eight_tools_registered(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.part_library import register_part_library_tools
        registry = ToolRegistry()
        register_part_library_tools(registry)
        tools = registry.list_tools()
        expected = {
            "part_search", "part_get", "part_generate", "part_list",
            "part_import", "part_save", "part_analyze_print", "part_assemble",
        }
        assert expected == set(tools)

    def test_tool_definitions(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.part_library import register_part_library_tools
        registry = ToolRegistry()
        register_part_library_tools(registry)
        defs = registry.get_all_definitions()
        assert len(defs) == 8
        for d in defs:
            assert d.name.startswith("part_")
            assert d.description
            assert d.parameters

    def test_part_analyze_print_in_category(self):
        from lang3d.tools.base import TOOL_CATEGORIES
        assert "part_analyze_print" in TOOL_CATEGORIES["part_library"]

    def test_part_assemble_in_category(self):
        from lang3d.tools.base import TOOL_CATEGORIES
        assert "part_assemble" in TOOL_CATEGORIES["part_library"]


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

    def test_part_analyze_print_missing_file(self):
        registry = self._make_registry()
        result = registry.execute("part_analyze_print", stl_path="/nonexistent/file.stl")
        assert "错误" in result or "error" in result.lower() or "不存在" in result


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


# ---------------------------------------------------------------------------
# 9. PartsStore — JSON persistence
# ---------------------------------------------------------------------------

class TestPartsStore:
    def _make_store(self, tmp_path):
        from lang3d.tools.part_library import PartsStore
        json_path = tmp_path / "test_parts.json"
        return PartsStore(json_path)

    def test_empty_init(self, tmp_path):
        store = self._make_store(tmp_path)
        assert store.count() == 0
        assert store.list_all() == []

    def test_add_and_list(self, tmp_path):
        from lang3d.knowledge.parts_catalog import GeneratedPart
        store = self._make_store(tmp_path)
        gen = GeneratedPart(template_id="test", name="part1", parameters={"a": 1})
        store.add(gen)
        assert store.count() == 1
        parts = store.list_all()
        assert parts[0].name == "part1"
        assert parts[0].template_id == "test"

    def test_add_persists_to_json(self, tmp_path):
        from lang3d.knowledge.parts_catalog import GeneratedPart
        from lang3d.tools.part_library import PartsStore
        json_path = tmp_path / "test_parts.json"
        store = PartsStore(json_path)
        gen = GeneratedPart(template_id="test", name="part1", parameters={"a": 1})
        store.add(gen)
        # Verify file was written
        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["name"] == "part1"

    def test_load_existing_json(self, tmp_path):
        from lang3d.knowledge.parts_catalog import GeneratedPart
        from lang3d.tools.part_library import PartsStore
        json_path = tmp_path / "test_parts.json"
        # Write existing data
        json_path.write_text(json.dumps([
            {"template_id": "test", "name": "old_part", "parameters": {"x": 5},
             "fcstd_path": "", "stl_path": "", "created_at": "", "print_analysis": {}}
        ]), encoding="utf-8")
        store = PartsStore(json_path)
        assert store.count() == 1
        assert store.get("old_part") is not None

    def test_corrupted_json_returns_empty(self, tmp_path):
        from lang3d.tools.part_library import PartsStore
        json_path = tmp_path / "test_parts.json"
        json_path.write_text("not valid json {{{", encoding="utf-8")
        store = PartsStore(json_path)
        assert store.count() == 0

    def test_remove(self, tmp_path):
        from lang3d.knowledge.parts_catalog import GeneratedPart
        store = self._make_store(tmp_path)
        store.add(GeneratedPart(template_id="t1", name="p1", parameters={}))
        store.add(GeneratedPart(template_id="t2", name="p2", parameters={}))
        assert store.count() == 2
        removed = store.remove("p1")
        assert removed is True
        assert store.count() == 1
        assert store.get("p1") is None
        assert store.get("p2") is not None

    def test_remove_nonexistent(self, tmp_path):
        store = self._make_store(tmp_path)
        removed = store.remove("nonexistent")
        assert removed is False

    def test_get_by_name(self, tmp_path):
        from lang3d.knowledge.parts_catalog import GeneratedPart
        store = self._make_store(tmp_path)
        store.add(GeneratedPart(template_id="t1", name="p1", parameters={"x": 1}))
        found = store.get("p1")
        assert found is not None
        assert found.template_id == "t1"
        assert store.get("missing") is None

    def test_to_dict_from_dict_roundtrip(self):
        from lang3d.knowledge.parts_catalog import GeneratedPart
        original = GeneratedPart(
            template_id="test", name="rt", parameters={"a": 1.0, "b": "x"},
            fcstd_path="/path/to/file.FCStd", stl_path="/path/to/file.stl",
            created_at="2024-01-01T00:00:00",
            print_analysis={"volume": 100, "issues": ["thin wall"]},
        )
        d = original.to_dict()
        restored = GeneratedPart.from_dict(d)
        assert restored.template_id == original.template_id
        assert restored.name == original.name
        assert restored.parameters == original.parameters
        assert restored.fcstd_path == original.fcstd_path
        assert restored.stl_path == original.stl_path
        assert restored.created_at == original.created_at
        assert restored.print_analysis == original.print_analysis

    def test_missing_json_file(self, tmp_path):
        from lang3d.tools.part_library import PartsStore
        json_path = tmp_path / "nonexistent" / "parts.json"
        store = PartsStore(json_path)
        assert store.count() == 0


# ---------------------------------------------------------------------------
# 10. Realistic thread and gear scripts
# ---------------------------------------------------------------------------

class TestRealisticScripts:
    def test_metric_thread_pitch_table(self):
        from lang3d.knowledge.parts_catalog import METRIC_THREAD_PITCH
        assert METRIC_THREAD_PITCH[6] == 1.0
        assert METRIC_THREAD_PITCH[10] == 1.5
        assert METRIC_THREAD_PITCH[3] == 0.5
        assert len(METRIC_THREAD_PITCH) == 10

    def test_socket_head_cap_screw_has_realistic_alternative(self):
        from lang3d.knowledge.parts_catalog import get_template
        t = get_template("socket_head_cap_screw")
        assert "realistic" in t.fc_script_alternatives
        assert "realistic" in t.quality_levels

    def test_hex_bolt_has_realistic_alternative(self):
        from lang3d.knowledge.parts_catalog import get_template
        t = get_template("hex_bolt")
        assert "realistic" in t.fc_script_alternatives
        assert "realistic" in t.quality_levels

    def test_hex_nut_has_realistic_alternative(self):
        from lang3d.knowledge.parts_catalog import get_template
        t = get_template("hex_nut")
        assert "realistic" in t.fc_script_alternatives
        assert "realistic" in t.quality_levels

    def test_spur_gear_has_realistic_alternative(self):
        from lang3d.knowledge.parts_catalog import get_template
        t = get_template("spur_gear")
        assert "realistic" in t.fc_script_alternatives
        assert "realistic" in t.quality_levels

    def test_realistic_script_contains_helix(self):
        from lang3d.knowledge.parts_catalog import get_template
        t = get_template("socket_head_cap_screw")
        assert "makeHelix" in t.fc_script_alternatives["realistic"]

    def test_realistic_gear_script_contains_involute(self):
        from lang3d.knowledge.parts_catalog import get_template
        t = get_template("spur_gear")
        realistic = t.fc_script_alternatives["realistic"]
        assert "involute" in realistic.lower()
        assert "pressure_angle" in realistic or "pressure_angle" in realistic

    def test_thread_templates_have_thread_pitch_param(self):
        from lang3d.knowledge.parts_catalog import PART_CATALOG
        thread_templates = ["socket_head_cap_screw", "hex_bolt", "hex_nut"]
        for tid in thread_templates:
            t = PART_CATALOG[tid]
            param_names = [p.name for p in t.parameters]
            assert "thread_pitch" in param_names, f"Missing thread_pitch in {tid}"
            assert "thread_detail" in param_names, f"Missing thread_detail in {tid}"

    def test_gear_has_new_params(self):
        from lang3d.knowledge.parts_catalog import get_template
        t = get_template("spur_gear")
        param_names = [p.name for p in t.parameters]
        assert "tooth_detail" in param_names
        assert "pressure_angle" in param_names
        assert "backlash" in param_names

    def test_thread_standard_sizes_have_pitch(self):
        from lang3d.knowledge.parts_catalog import PART_CATALOG
        thread_templates = ["socket_head_cap_screw", "hex_bolt", "hex_nut"]
        for tid in thread_templates:
            t = PART_CATALOG[tid]
            for sz in t.standard_sizes:
                assert "thread_pitch" in sz, f"Missing thread_pitch in standard size of {tid}: {sz}"

    def test_gear_standard_sizes_have_new_params(self):
        from lang3d.knowledge.parts_catalog import get_template
        t = get_template("spur_gear")
        for sz in t.standard_sizes:
            assert "pressure_angle" in sz, f"Missing pressure_angle in: {sz}"
            assert "backlash" in sz, f"Missing backlash in: {sz}"

    def test_small_gear_params(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters
        t = get_template("spur_gear")
        params = resolve_parameters(t, {"teeth": 8, "module": 1.0})
        assert params["teeth"] == 8

    def test_large_gear_params(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters
        t = get_template("spur_gear")
        params = resolve_parameters(t, {"teeth": 100, "module": 2.0})
        assert params["teeth"] == 100


# ---------------------------------------------------------------------------
# 11. Print analysis tool
# ---------------------------------------------------------------------------

class TestPrintAnalysisTool:
    def _make_registry(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.part_library import register_part_library_tools
        registry = ToolRegistry()
        register_part_library_tools(registry)
        return registry

    def test_analyze_print_tool_registered(self):
        registry = self._make_registry()
        assert "part_analyze_print" in registry.list_tools()

    def test_analyze_print_definition(self):
        registry = self._make_registry()
        tool = registry.get("part_analyze_print")
        assert tool is not None
        defn = tool.get_definition()
        assert "stl_path" in str(defn.parameters)
        assert "orientation" in str(defn.parameters)

    def test_analyze_print_missing_file(self):
        registry = self._make_registry()
        result = registry.execute("part_analyze_print", stl_path="/nonexistent.stl")
        assert "错误" in result or "不存在" in result

    def test_analyze_print_unsupported_format(self, tmp_path):
        # Create a dummy file with unsupported extension
        dummy = tmp_path / "test.doc"
        dummy.write_text("not a model")
        registry = self._make_registry()
        result = registry.execute("part_analyze_print", stl_path=str(dummy))
        assert "不支持的文件格式" in result or "error" in result.lower()

    def test_tool_count_is_eight(self):
        from lang3d.tools.base import TOOL_CATEGORIES
        part_tools = TOOL_CATEGORIES["part_library"]
        assert len(part_tools) == 8


# ---------------------------------------------------------------------------
# 12. GeneratedPart serialization
# ---------------------------------------------------------------------------

class TestGeneratedPartSerialization:
    def test_to_dict_has_all_fields(self):
        from lang3d.knowledge.parts_catalog import GeneratedPart
        gen = GeneratedPart(
            template_id="test", name="p1", parameters={"a": 1.0},
            fcstd_path="/path.FCStd", stl_path="/path.stl",
            created_at="2024-01-01", print_analysis={"volume": 50},
        )
        d = gen.to_dict()
        assert "template_id" in d
        assert "name" in d
        assert "parameters" in d
        assert "fcstd_path" in d
        assert "stl_path" in d
        assert "created_at" in d
        assert "print_analysis" in d
        assert d["print_analysis"]["volume"] == 50

    def test_from_dict_defaults(self):
        from lang3d.knowledge.parts_catalog import GeneratedPart
        gen = GeneratedPart.from_dict({"template_id": "t", "name": "n"})
        assert gen.template_id == "t"
        assert gen.name == "n"
        assert gen.parameters == {}
        assert gen.fcstd_path == ""
        assert gen.print_analysis == {}

    def test_default_print_analysis_is_empty(self):
        from lang3d.knowledge.parts_catalog import GeneratedPart
        gen = GeneratedPart(template_id="t", name="n", parameters={})
        assert gen.print_analysis == {}


# ---------------------------------------------------------------------------
# 13. format_fc_script multi-script selection
# ---------------------------------------------------------------------------

class TestFormatFcScriptSelection:
    def test_default_uses_simplified(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters, format_fc_script
        t = get_template("socket_head_cap_screw")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        # Default template is simplified (no makeHelix)
        assert "makeCylinder" in script

    def test_realistic_param_selects_alternative(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters, format_fc_script
        t = get_template("socket_head_cap_screw")
        params = resolve_parameters(t, {"thread_detail": "realistic", "thread_pitch": 1.0})
        script = format_fc_script(t, params)
        assert "makeHelix" in script

    def test_gear_tooth_detail_realistic(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters, format_fc_script
        t = get_template("spur_gear")
        params = resolve_parameters(t, {"tooth_detail": "realistic", "pressure_angle": 20.0, "backlash": 0.1})
        script = format_fc_script(t, params)
        assert "involute" in script.lower()

    def test_gear_tooth_detail_simplified(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters, format_fc_script
        t = get_template("spur_gear")
        params = resolve_parameters(t, {"tooth_detail": "simplified"})
        script = format_fc_script(t, params)
        assert "involute" not in script.lower()


# ---------------------------------------------------------------------------
# 14. Assembly tool (B2)
# ---------------------------------------------------------------------------

class TestAssemblyTool:
    def _make_registry(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.part_library import register_part_library_tools
        registry = ToolRegistry()
        register_part_library_tools(registry)
        return registry

    def test_assemble_tool_registered(self):
        registry = self._make_registry()
        assert "part_assemble" in registry.list_tools()

    def test_assemble_tool_definition(self):
        registry = self._make_registry()
        tool = registry.get("part_assemble")
        assert tool is not None
        defn = tool.get_definition()
        assert "assembly_name" in str(defn.parameters)
        assert "parts" in str(defn.parameters)

    def test_assemble_script_generation(self):
        """Verify assembly script is generated correctly (no FreeCAD needed)."""
        from lang3d.tools.part_library import _ASSEMBLY_SCRIPT_TEMPLATE, _PART_IMPORT_TEMPLATES
        # Test STL import template
        assert ".stl" in _PART_IMPORT_TEMPLATES
        assert ".fcstd" in _PART_IMPORT_TEMPLATES
        assert ".step" in _PART_IMPORT_TEMPLATES

        # Test template placeholders
        assert "{assembly_name}" in _ASSEMBLY_SCRIPT_TEMPLATE
        assert "{part_operations}" in _ASSEMBLY_SCRIPT_TEMPLATE

    def test_assemble_missing_file_error(self):
        """Assembly with missing files should return error."""
        registry = self._make_registry()
        result = registry.execute("part_assemble", assembly_name="test_asm", parts=[
            {"file": "/nonexistent/part1.stl", "name": "p1", "position": [0, 0, 0]},
            {"file": "/nonexistent/part2.stl", "name": "p2", "position": [10, 0, 0]},
        ])
        assert "错误" in result or "不存在" in result

    def test_assemble_empty_parts_error(self):
        """Assembly with empty parts list should return error."""
        registry = self._make_registry()
        result = registry.execute("part_assemble", assembly_name="test_asm", parts=[])
        assert "错误" in result or "为空" in result

    def test_assemble_unsupported_format_error(self):
        """Assembly with unsupported file format should return error."""
        registry = self._make_registry()
        import tempfile
        # Create a temp file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test")
            tmppath = f.name
        try:
            result = registry.execute("part_assemble", assembly_name="test_asm", parts=[
                {"file": tmppath, "name": "p1", "position": [0, 0, 0]},
            ])
            assert "不支持" in result or "unsupported" in result.lower()
        finally:
            import os
            os.unlink(tmppath)


# ---------------------------------------------------------------------------
# 15. Bearing realistic scripts (B3)
# ---------------------------------------------------------------------------

class TestBearingRealistic:
    def test_bearing_realistic_script_has_balls(self):
        from lang3d.knowledge.parts_catalog import PART_CATALOG
        bearing_608 = PART_CATALOG["bearing_608"]
        realistic_script = bearing_608.fc_script_alternatives["realistic"]
        assert "makeSphere" in realistic_script
        assert "Ball" in realistic_script

    def test_bearing_realistic_script_has_raceway(self):
        from lang3d.knowledge.parts_catalog import PART_CATALOG
        bearing_608 = PART_CATALOG["bearing_608"]
        realistic_script = bearing_608.fc_script_alternatives["realistic"]
        assert "raceway" in realistic_script.lower() or "revolve" in realistic_script.lower()

    def test_bearing_realistic_script_has_cage(self):
        from lang3d.knowledge.parts_catalog import PART_CATALOG
        bearing_608 = PART_CATALOG["bearing_608"]
        realistic_script = bearing_608.fc_script_alternatives["realistic"]
        assert "Cage" in realistic_script
        assert "pocket" in realistic_script.lower()

    def test_bearing_detail_parameter(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters
        t = get_template("bearing_608")
        params = resolve_parameters(t, {"bearing_detail": "realistic"})
        assert params["bearing_detail"] == "realistic"

    def test_bearing_detail_default_simplified(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters
        t = get_template("bearing_608")
        params = resolve_parameters(t)
        assert params["bearing_detail"] == "simplified"

    def test_ball_count_parameter(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters
        t = get_template("bearing_608")
        params = resolve_parameters(t, {"ball_count": 7})
        assert params["ball_count"] == 7

    def test_ball_count_default_auto(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters
        t = get_template("bearing_608")
        params = resolve_parameters(t)
        assert params["ball_count"] == 0  # 0 means auto-calculate

    def test_all_bearings_have_realistic_alternative(self):
        from lang3d.knowledge.parts_catalog import PART_CATALOG
        bearing_ids = ["bearing_608", "bearing_623", "bearing_625"]
        for bid in bearing_ids:
            t = PART_CATALOG[bid]
            assert "realistic" in t.fc_script_alternatives, f"Missing realistic in {bid}"
            assert "realistic" in t.quality_levels, f"Missing realistic in quality_levels of {bid}"

    def test_all_bearings_have_detail_params(self):
        from lang3d.knowledge.parts_catalog import PART_CATALOG
        bearing_ids = ["bearing_608", "bearing_623", "bearing_625"]
        for bid in bearing_ids:
            t = PART_CATALOG[bid]
            param_names = [p.name for p in t.parameters]
            assert "bearing_detail" in param_names, f"Missing bearing_detail in {bid}"
            assert "ball_count" in param_names, f"Missing ball_count in {bid}"

    def test_bearing_realistic_script_no_unreplaced_placeholders(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters, format_fc_script
        import re
        t = get_template("bearing_608")
        params = resolve_parameters(t, {"bearing_detail": "realistic", "ball_count": 0})
        script = format_fc_script(t, params)
        unreplaced = re.findall(r'\{[a-zA-Z_][a-zA-Z0-9_]*\}', script)
        assert len(unreplaced) == 0, f"Unreplaced placeholders: {unreplaced}"

    def test_bearing_simplified_backward_compatible(self):
        from lang3d.knowledge.parts_catalog import get_template, resolve_parameters, format_fc_script
        t = get_template("bearing_608")
        params = resolve_parameters(t, {"bearing_detail": "simplified"})
        script = format_fc_script(t, params)
        assert "makeSphere" not in script  # simplified has no balls
        assert "OuterRing" in script
        assert "InnerRing" in script
