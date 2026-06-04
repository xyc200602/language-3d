"""Tests for quality control tools.

Tests cover:
- Dimensional inspection checklist
- Post-assembly test procedure
- Maintenance guide
- Quality report generation
- Tool registration and execution
"""

from __future__ import annotations

import pytest

from lang3d.knowledge.mechanics import ROBOTIC_ARM_ASSEMBLY
from lang3d.tools.base import ToolRegistry
from lang3d.tools.quality import (
    QualityCheckTool,
    generate_inspection_checklist,
    generate_maintenance_guide,
    generate_quality_report,
    generate_test_procedure,
    register_quality_tools,
)


# ============================================================================
# Test: Inspection checklist
# ============================================================================

class TestInspectionChecklist:
    """Test dimensional inspection checklist generation."""

    @pytest.fixture
    def checklist(self):
        return generate_inspection_checklist(ROBOTIC_ARM_ASSEMBLY)

    def test_has_assembly_name(self, checklist):
        assert checklist["assembly_name"] == "3-DOF Robotic Arm"

    def test_has_all_parts(self, checklist):
        assert len(checklist["parts"]) == len(ROBOTIC_ARM_ASSEMBLY.parts)

    def test_each_part_has_checks(self, checklist):
        for p in checklist["parts"]:
            assert len(p["checks"]) > 0
            assert p["part_name"] in [pp.name for pp in ROBOTIC_ARM_ASSEMBLY.parts]

    def test_check_fields(self, checklist):
        for p in checklist["parts"]:
            for c in p["checks"]:
                assert "dimension" in c
                assert "nominal_mm" in c
                assert "tolerance_mm" in c
                assert "min_mm" in c
                assert "max_mm" in c
                assert "critical" in c

    def test_min_max_bounds(self, checklist):
        for p in checklist["parts"]:
            for c in p["checks"]:
                assert c["min_mm"] < c["nominal_mm"]
                assert c["max_mm"] > c["nominal_mm"]

    def test_has_summary(self, checklist):
        s = checklist["summary"]
        assert s["total_dimensions"] > 0
        assert s["critical_dimensions"] >= 0
        assert s["total_dimensions"] == s["critical_dimensions"] + s["non_critical_dimensions"]

    def test_shaft_dimensions_are_critical(self, checklist):
        for p in checklist["parts"]:
            for c in p["checks"]:
                if "shaft" in c["dimension"].lower():
                    assert c["critical"] is True

    def test_custom_tolerance(self):
        cl = generate_inspection_checklist(ROBOTIC_ARM_ASSEMBLY, tolerance_mm=0.5)
        assert cl["default_tolerance_mm"] == 0.5


# ============================================================================
# Test: Post-assembly test procedure
# ============================================================================

class TestTestProcedure:
    """Test post-assembly test procedure generation."""

    @pytest.fixture
    def procedure(self):
        return generate_test_procedure(ROBOTIC_ARM_ASSEMBLY)

    def test_has_phases(self, procedure):
        assert len(procedure["phases"]) >= 4

    def test_has_visual_inspection(self, procedure):
        names = [p["name"] for p in procedure["phases"]]
        assert "外观检查" in names

    def test_has_mechanical_test(self, procedure):
        names = [p["name"] for p in procedure["phases"]]
        assert "机械范围测试" in names

    def test_has_electrical_test(self, procedure):
        names = [p["name"] for p in procedure["phases"]]
        assert "电气测试" in names

    def test_has_functional_test(self, procedure):
        names = [p["name"] for p in procedure["phases"]]
        assert "功能测试" in names

    def test_has_endurance_test(self, procedure):
        names = [p["name"] for p in procedure["phases"]]
        assert "耐久测试" in names

    def test_mechanical_steps_cover_joints(self, procedure):
        mech_phase = [p for p in procedure["phases"] if p["name"] == "机械范围测试"][0]
        revolute_joints = [j for j in ROBOTIC_ARM_ASSEMBLY.joints if j.type == "revolute"]
        assert len(mech_phase["steps"]) >= len(revolute_joints)

    def test_steps_have_action(self, procedure):
        for phase in procedure["phases"]:
            for step in phase["steps"]:
                assert "action" in step


# ============================================================================
# Test: Maintenance guide
# ============================================================================

class TestMaintenanceGuide:
    """Test maintenance guide generation."""

    @pytest.fixture
    def guide(self):
        return generate_maintenance_guide(ROBOTIC_ARM_ASSEMBLY)

    def test_has_schedules(self, guide):
        assert len(guide["schedules"]) >= 3

    def test_has_weekly(self, guide):
        intervals = [s["interval"] for s in guide["schedules"]]
        assert "每周" in intervals

    def test_has_monthly(self, guide):
        intervals = [s["interval"] for s in guide["schedules"]]
        assert "每月" in intervals

    def test_has_quarterly(self, guide):
        intervals = [s["interval"] for s in guide["schedules"]]
        assert "每季度" in intervals

    def test_has_yearly(self, guide):
        intervals = [s["interval"] for s in guide["schedules"]]
        assert "每年" in intervals

    def test_tasks_have_fields(self, guide):
        for sched in guide["schedules"]:
            for task in sched["tasks"]:
                assert "task" in task
                assert "action" in task

    def test_has_common_issues(self, guide):
        assert len(guide["common_issues"]) > 0

    def test_issues_have_symptom_and_fix(self, guide):
        for issue in guide["common_issues"]:
            assert "symptom" in issue
            assert "cause" in issue
            assert "fix" in issue


# ============================================================================
# Test: Full quality report
# ============================================================================

class TestQualityReport:
    """Test full quality report generation."""

    @pytest.fixture
    def report(self):
        return generate_quality_report(ROBOTIC_ARM_ASSEMBLY)

    def test_has_three_sections(self, report):
        assert "inspection" in report
        assert "test_procedure" in report
        assert "maintenance" in report


# ============================================================================
# Test: Tool registration and execution
# ============================================================================

class TestQualityTool:
    """Test QualityCheckTool."""

    def test_registered(self):
        registry = ToolRegistry()
        register_quality_tools(registry)
        assert "quality_check" in registry.list_tools()

    def test_execute_all(self):
        tool = QualityCheckTool()
        result = tool.execute()
        assert "Quality Report" in result
        assert "尺寸检查清单" in result
        assert "装配后测试流程" in result
        assert "维护指南" in result

    def test_execute_inspection_only(self):
        tool = QualityCheckTool()
        result = tool.execute(section="inspection")
        assert "尺寸检查清单" in result
        assert "装配后测试流程" not in result

    def test_execute_test_only(self):
        tool = QualityCheckTool()
        result = tool.execute(section="test")
        assert "装配后测试流程" in result
        assert "尺寸检查清单" not in result

    def test_execute_maintenance_only(self):
        tool = QualityCheckTool()
        result = tool.execute(section="maintenance")
        assert "维护指南" in result
        assert "尺寸检查清单" not in result

    def test_execute_unknown_assembly(self):
        tool = QualityCheckTool()
        result = tool.execute(assembly_name="nonexistent")
        assert "错误" in result

    def test_execute_json_included(self):
        tool = QualityCheckTool()
        result = tool.execute()
        assert "--- JSON ---" in result
