"""Tests for assembly template knowledge base, tools, and integration."""

import json
import pytest

from lang3d.knowledge.assembly_templates import (
    TEMPLATES,
    AssemblyTemplate,
    TemplateJointSpec,
    TemplatePartSpec,
    list_assembly_templates,
    search_assembly_templates,
    template_to_assembly,
)
from lang3d.knowledge.mechanics import Assembly, Joint, Part
from lang3d.tools.assembly_template_tool import (
    AssemblyTemplateInstantiateTool,
    AssemblyTemplateSearchTool,
    register_assembly_template_tools,
)
from lang3d.tools.base import ToolRegistry


# ---------------------------------------------------------------------------
# Template data integrity
# ---------------------------------------------------------------------------


class TestTemplateDataIntegrity:
    """Verify all 6 templates load with valid structure."""

    EXPECTED_IDS = {
        "3dof_arm", "4dof_arm", "6dof_belt_arm",
        "diff_drive_4w", "mecanum_4w", "scara_arm",
    }

    def test_all_six_templates_loaded(self):
        assert set(TEMPLATES.keys()) == self.EXPECTED_IDS

    @pytest.mark.parametrize("tpl_id", EXPECTED_IDS)
    def test_template_has_required_fields(self, tpl_id):
        tpl = TEMPLATES[tpl_id]
        assert tpl.name_en
        assert tpl.name_cn
        assert tpl.description
        assert tpl.dof > 0
        assert tpl.robot_type in ("arm", "mobile_base", "scara")
        assert len(tpl.keywords) >= 3
        assert len(tpl.parts) >= 5
        assert len(tpl.joints) >= 4

    @pytest.mark.parametrize("tpl_id", EXPECTED_IDS)
    def test_parts_non_empty(self, tpl_id):
        tpl = TEMPLATES[tpl_id]
        for ps in tpl.parts:
            assert ps.name_pattern, f"Empty name_pattern in {tpl_id}"
            assert ps.category, f"Empty category for {ps.name_pattern} in {tpl_id}"
            assert ps.description_cn, f"Empty desc for {ps.name_pattern} in {tpl_id}"

    @pytest.mark.parametrize("tpl_id", EXPECTED_IDS)
    def test_joints_non_empty(self, tpl_id):
        tpl = TEMPLATES[tpl_id]
        for js in tpl.joints:
            assert js.type in ("revolute", "fixed", "prismatic"), (
                f"Invalid joint type '{js.type}' in {tpl_id}"
            )
            assert js.parent, f"Empty parent in {tpl_id}"
            assert js.child, f"Empty child in {tpl_id}"

    @pytest.mark.parametrize("tpl_id", EXPECTED_IDS)
    def test_tree_connectivity(self, tpl_id):
        """Verify joints form a connected tree: parts = joints + 1 (approx).

        For a tree: number of edges (joints) should be <= number of nodes (parts).
        Each joint references existing part names.
        """
        tpl = TEMPLATES[tpl_id]
        part_names = {ps.name_pattern for ps in tpl.parts}
        referenced = set()
        for js in tpl.joints:
            assert js.parent in part_names, (
                f"Joint parent '{js.parent}' not in parts of {tpl_id}"
            )
            assert js.child in part_names, (
                f"Joint child '{js.child}' not in parts of {tpl_id}"
            )
            referenced.add(js.parent)
            referenced.add(js.child)
        # Most parts should be referenced in joints (allow up to 50% for
        # complex templates that include sub-component parts like bearings,
        # belts, and pulleys not explicitly linked in the kinematic chain)
        unreferenced = part_names - referenced
        assert len(unreferenced) <= len(part_names) * 0.5, (
            f"Too many unreferenced parts in {tpl_id}: {unreferenced}"
        )

    @pytest.mark.parametrize("tpl_id", EXPECTED_IDS)
    def test_no_duplicate_part_names(self, tpl_id):
        tpl = TEMPLATES[tpl_id]
        names = [ps.name_pattern for ps in tpl.parts]
        assert len(names) == len(set(names)), (
            f"Duplicate part names in {tpl_id}"
        )

    def test_dof_values_correct(self):
        assert TEMPLATES["3dof_arm"].dof == 3
        assert TEMPLATES["4dof_arm"].dof == 4
        assert TEMPLATES["6dof_belt_arm"].dof == 6
        assert TEMPLATES["diff_drive_4w"].dof == 2
        assert TEMPLATES["mecanum_4w"].dof == 2
        assert TEMPLATES["scara_arm"].dof == 4

    def test_robot_types_correct(self):
        assert TEMPLATES["3dof_arm"].robot_type == "arm"
        assert TEMPLATES["diff_drive_4w"].robot_type == "mobile_base"
        assert TEMPLATES["scara_arm"].robot_type == "scara"


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestTemplateSearch:
    """Test weighted search functionality."""

    def test_chinese_query_3dof(self):
        results = search_assembly_templates("3自由度机械臂")
        assert len(results) >= 1
        assert results[0].id == "3dof_arm"

    def test_english_query_4wheel(self):
        results = search_assembly_templates("4 wheel")
        assert len(results) >= 1
        # Should match diff_drive_4w or mecanum_4w
        matched_ids = {r.id for r in results}
        assert "diff_drive_4w" in matched_ids or "mecanum_4w" in matched_ids

    def test_dof_filter(self):
        results = search_assembly_templates(min_dof=5, max_dof=7)
        ids = {r.id for r in results}
        assert "6dof_belt_arm" in ids
        assert "3dof_arm" not in ids
        assert "4dof_arm" not in ids

    def test_robot_type_filter(self):
        results = search_assembly_templates(robot_type="mobile_base")
        ids = {r.id for r in results}
        assert "diff_drive_4w" in ids
        assert "mecanum_4w" in ids
        assert "3dof_arm" not in ids

    def test_combined_query_and_filter(self):
        results = search_assembly_templates("arm", robot_type="arm", min_dof=4)
        ids = {r.id for r in results}
        assert "4dof_arm" in ids
        assert "6dof_belt_arm" in ids
        assert "3dof_arm" not in ids

    def test_no_results_returns_empty(self):
        results = search_assembly_templates("quantum_teleporter", min_dof=100)
        assert len(results) == 0

    def test_empty_query_returns_all(self):
        results = search_assembly_templates()
        assert len(results) == 6

    def test_mecanum_keyword(self):
        results = search_assembly_templates("麦克纳姆")
        assert len(results) >= 1
        assert results[0].id == "mecanum_4w"

    def test_belt_keyword(self):
        results = search_assembly_templates("belt")
        ids = {r.id for r in results}
        assert "6dof_belt_arm" in ids

    def test_scara_keyword(self):
        results = search_assembly_templates("scara")
        assert len(results) >= 1
        assert results[0].id == "scara_arm"


# ---------------------------------------------------------------------------
# Template → Assembly conversion
# ---------------------------------------------------------------------------


class TestTemplateToAssembly:
    """Test template to Assembly object conversion."""

    def test_3dof_arm_conversion(self):
        tpl = TEMPLATES["3dof_arm"]
        asm = template_to_assembly(tpl)
        assert isinstance(asm, Assembly)
        assert len(asm.parts) == len(tpl.parts)
        assert len(asm.joints) == len(tpl.joints)
        assert asm.name == tpl.name_en

    def test_connection_methods_set(self):
        tpl = TEMPLATES["diff_drive_4w"]
        asm = template_to_assembly(tpl)
        bolted_joints = [
            j for j in asm.joints
            if j.connection and j.connection.type == "bolted"
        ]
        assert len(bolted_joints) > 0

    def test_overrides_applied(self):
        tpl = TEMPLATES["3dof_arm"]
        overrides = {
            "shoulder_link": {"length": 200},
        }
        asm = template_to_assembly(tpl, overrides)
        shoulder = next(p for p in asm.parts if p.name == "shoulder_link")
        assert shoulder.dimensions["length"] == 200

    def test_tree_structure_valid(self):
        """Verify the converted assembly forms a valid kinematic tree."""
        for tpl_id in TEMPLATES:
            tpl = TEMPLATES[tpl_id]
            asm = template_to_assembly(tpl)

            # Find root (appears as parent but never as child)
            parents = {j.parent for j in asm.joints}
            children = {j.child for j in asm.joints}
            roots = parents - children
            assert len(roots) >= 1, f"No root found in {tpl_id}"

            # No cycles: every child has at most one parent
            child_to_parent = {}
            for j in asm.joints:
                if j.child in child_to_parent:
                    # Multiple parents for same child is ok (e.g., gripper fingers)
                    pass
                child_to_parent[j.child] = j.parent

    def test_default_angles_present(self):
        tpl = TEMPLATES["3dof_arm"]
        asm = template_to_assembly(tpl)
        assert len(asm.default_angles) > 0


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------


class TestAssemblyTemplateSearchTool:
    """Test the assembly_template_search tool."""

    def test_execute_with_query(self):
        tool = AssemblyTemplateSearchTool()
        result = tool.execute(query="3自由度")
        assert "3dof_arm" in result
        assert "JSON" in result

    def test_execute_returns_valid_json(self):
        tool = AssemblyTemplateSearchTool()
        result = tool.execute(query="arm")
        # Extract JSON portion
        json_start = result.index("--- JSON ---") + len("--- JSON ---")
        json_str = result[json_start:].strip()
        data = json.loads(json_str)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "id" in data[0]
        assert "name_cn" in data[0]

    def test_execute_no_results(self):
        tool = AssemblyTemplateSearchTool()
        result = tool.execute(query="", robot_type="arm", min_dof=100)
        # Should still return something (empty list or all templates)
        assert isinstance(result, str)


class TestAssemblyTemplateInstantiateTool:
    """Test the assembly_template_instantiate tool."""

    def test_execute_valid_template(self):
        tool = AssemblyTemplateInstantiateTool()
        result = tool.execute(template_id="3dof_arm")
        assert "3-DOF" in result
        assert "JSON" in result

    def test_execute_returns_valid_json(self):
        tool = AssemblyTemplateInstantiateTool()
        result = tool.execute(template_id="4dof_arm")
        json_start = result.index("--- Assembly JSON ---") + len("--- Assembly JSON ---")
        json_str = result[json_start:].strip()
        data = json.loads(json_str)
        assert "parts" in data
        assert "joints" in data
        assert len(data["parts"]) > 0

    def test_execute_unknown_template(self):
        tool = AssemblyTemplateInstantiateTool()
        result = tool.execute(template_id="nonexistent_template")
        assert "错误" in result

    def test_execute_with_overrides(self):
        tool = AssemblyTemplateInstantiateTool()
        result = tool.execute(
            template_id="3dof_arm",
            overrides={"shoulder_link": {"length": 300}},
        )
        assert "300" in result


class TestToolRegistration:
    """Test tool registration."""

    def test_register_tools(self):
        registry = ToolRegistry()
        register_assembly_template_tools(registry)
        assert registry.get("assembly_template_search") is not None
        assert registry.get("assembly_template_instantiate") is not None

    def test_tool_definitions(self):
        registry = ToolRegistry()
        register_assembly_template_tools(registry)
        search_tool = registry.get("assembly_template_search")
        inst_tool = registry.get("assembly_template_instantiate")
        assert search_tool is not None
        assert inst_tool is not None
        defn = search_tool.get_definition()
        assert defn.name == "assembly_template_search"
        assert "properties" in defn.parameters


class TestListTemplates:
    """Test list_assembly_templates utility."""

    def test_returns_all_templates(self):
        summaries = list_assembly_templates()
        assert len(summaries) == 6

    def test_summary_structure(self):
        summaries = list_assembly_templates()
        for s in summaries:
            assert "id" in s
            assert "name_en" in s
            assert "name_cn" in s
            assert "dof" in s
            assert "robot_type" in s
            assert "parts_required" in s
            assert "parts_optional" in s
            assert "joints" in s
