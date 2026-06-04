"""Tests for production readiness check tools.

Tests cover:
- Assembly integrity checks
- Firmware code checks
- BOM correctness checks
- Tolerance reasonableness checks
- Wiring diagram checks
- Full production check pipeline
- Report formatting
- Tool registration and execution
"""

from __future__ import annotations

import pytest

from lang3d.knowledge.mechanics import Assembly, Joint, Part, ROBOTIC_ARM_ASSEMBLY
from lang3d.tools.base import ToolRegistry
from lang3d.tools.production_check import (
    CheckItem,
    ProductionCheckTool,
    ProductionReport,
    format_production_report,
    register_production_tools,
    run_production_check,
)
from lang3d.tools.code_gen import generate_firmware
from lang3d.tools.bom_gen import generate_bom


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def actuators():
    return ["MG996R", "MG996R", "DS3218", "SG90"]


@pytest.fixture
def sensors():
    return ["AS5600", "LIMIT_SWITCH_MICRO", "MPU6050"]


@pytest.fixture
def report(actuators, sensors):
    return run_production_check(
        ROBOTIC_ARM_ASSEMBLY, actuators, sensors, "esp32",
    )


# ============================================================================
# Test: Assembly integrity checks
# ============================================================================

class TestAssemblyIntegrity:
    """Test assembly structural integrity checks."""

    def test_has_part_count(self, report):
        checks = [c for c in report.checks if c.name == "零件数量"]
        assert len(checks) == 1
        assert checks[0].status == "pass"

    def test_has_joint_count(self, report):
        checks = [c for c in report.checks if c.name == "关节数量"]
        assert len(checks) == 1
        assert checks[0].status == "pass"

    def test_joint_references_valid(self, report):
        checks = [c for c in report.checks if c.name == "关节引用完整性"]
        assert len(checks) == 1
        assert checks[0].status == "pass"

    def test_part_dimensions_present(self, report):
        checks = [c for c in report.checks if c.name == "零件尺寸"]
        assert len(checks) == 1
        assert checks[0].status == "pass"

    def test_empty_assembly_fails(self):
        asm = Assembly("empty", parts=[], joints=[])
        r = run_production_check(asm, [], [], "esp32")
        part_check = [c for c in r.checks if c.name == "零件数量"]
        assert part_check[0].status == "fail"

    def test_invalid_joint_reference_fails(self):
        parts = [Part("A", "struct", "A", dimensions={"h": 10}),
                 Part("B", "struct", "B", dimensions={"h": 10})]
        joints = [Joint("fixed", "A", "NONEXISTENT")]
        asm = Assembly("bad", parts=parts, joints=joints)
        r = run_production_check(asm, [], [], "esp32")
        ref_checks = [c for c in r.checks if "关节引用" in c.name and c.status == "fail"]
        assert len(ref_checks) > 0

    def test_missing_dimensions_warns(self):
        parts = [Part("A", "struct", "A", dimensions={"h": 10}),
                 Part("B", "struct", "B")]  # No dimensions
        joints = [Joint("fixed", "A", "B")]
        asm = Assembly("warn", parts=parts, joints=joints)
        r = run_production_check(asm, [], [], "esp32")
        dim_check = [c for c in r.checks if c.name == "零件尺寸"]
        assert dim_check[0].status == "warning"


# ============================================================================
# Test: Firmware checks
# ============================================================================

class TestFirmwareChecks:
    """Test firmware completeness and validity checks."""

    def test_firmware_files_pass(self, report):
        checks = [c for c in report.checks if c.name == "固件文件完整"]
        assert len(checks) == 1
        assert checks[0].status == "pass"

    def test_main_structure_pass(self, report):
        checks = [c for c in report.checks if c.name == "主程序结构"]
        assert len(checks) == 1
        assert checks[0].status == "pass"

    def test_serial_communication(self, report):
        checks = [c for c in report.checks if c.name == "串口通信"]
        assert len(checks) == 1
        assert checks[0].status == "pass"

    def test_brace_balance(self, report):
        checks = [c for c in report.checks if c.name == "括号平衡"]
        assert len(checks) == 1
        assert checks[0].status == "pass"

    def test_code_lines(self, report):
        checks = [c for c in report.checks if c.name == "代码行数"]
        assert len(checks) == 1
        assert checks[0].status == "pass"

    def test_ik_solver_check(self, report):
        checks = [c for c in report.checks if c.name == "IK 求解器"]
        assert len(checks) == 1

    def test_servo_driver_check(self, report):
        checks = [c for c in report.checks if c.name == "舵机驱动"]
        assert len(checks) == 1

    def test_sensor_driver_check(self, report):
        checks = [c for c in report.checks if c.name == "传感器驱动"]
        assert len(checks) == 1


# ============================================================================
# Test: BOM checks
# ============================================================================

class TestBOMChecks:
    """Test BOM correctness checks."""

    def test_parts_match(self, report):
        checks = [c for c in report.checks if c.name == "零件清单"]
        assert len(checks) == 1
        assert checks[0].status == "pass"

    def test_standard_parts(self, report):
        checks = [c for c in report.checks if c.name == "标准件"]
        assert len(checks) == 1
        assert checks[0].status == "pass"

    def test_electronics_match(self, report):
        checks = [c for c in report.checks if c.name == "电子件匹配"]
        assert len(checks) >= 1

    def test_cost_positive(self, report):
        checks = [c for c in report.checks if c.name == "成本估算"]
        assert len(checks) == 1
        assert checks[0].status == "pass"

    def test_cost_breakdown(self, report):
        checks = [c for c in report.checks if c.name == "成本明细"]
        assert len(checks) == 1


# ============================================================================
# Test: Tolerance checks
# ============================================================================

class TestToleranceChecks:
    """Test tolerance reasonableness checks."""

    def test_dimensions_reasonable(self, report):
        checks = [c for c in report.checks if c.name == "尺寸合理性"]
        assert len(checks) == 1
        assert checks[0].status == "pass"

    def test_zero_dimension_fails(self):
        parts = [Part("A", "struct", "A", dimensions={"diameter": 0}),
                 Part("B", "struct", "B", dimensions={"h": 10})]
        joints = [Joint("fixed", "A", "B")]
        asm = Assembly("zero", parts=parts, joints=joints)
        r = run_production_check(asm, [], [], "esp32")
        zero_checks = [c for c in r.checks if c.status == "fail" and "为零" in c.message]
        assert len(zero_checks) > 0

    def test_huge_dimension_warns(self):
        parts = [Part("A", "struct", "A", dimensions={"length": 5000}),
                 Part("B", "struct", "B", dimensions={"h": 10})]
        joints = [Joint("fixed", "A", "B")]
        asm = Assembly("huge", parts=parts, joints=joints)
        r = run_production_check(asm, [], [], "esp32")
        huge_checks = [c for c in r.checks if "过大" in c.message]
        assert len(huge_checks) > 0


# ============================================================================
# Test: Wiring checks
# ============================================================================

class TestWiringChecks:
    """Test wiring diagram checks."""

    def test_wiring_exists(self, report):
        checks = [c for c in report.checks if c.name == "接线图"]
        assert len(checks) == 1
        assert checks[0].status == "pass"

    def test_gnd_present(self, report):
        checks = [c for c in report.checks if c.name == "接地连接"]
        assert len(checks) == 1
        assert checks[0].status == "pass"

    def test_gpio_present(self, report):
        checks = [c for c in report.checks if c.name == "GPIO 引脚"]
        assert len(checks) == 1
        assert checks[0].status == "pass"


# ============================================================================
# Test: Full production report
# ============================================================================

class TestProductionReport:
    """Test full production report."""

    def test_report_ready(self, report):
        assert report.ready
        assert report.failed == 0

    def test_report_has_checks(self, report):
        assert report.total > 0
        assert report.passed > 0

    def test_report_summary(self, report):
        assert "生产就绪" in report.summary

    def test_report_assembly_name(self, report):
        assert report.assembly_name == ROBOTIC_ARM_ASSEMBLY.name

    def test_report_all_categories(self, report):
        categories = {c.category for c in report.checks}
        assert "assembly" in categories
        assert "code" in categories
        assert "bom" in categories
        assert "tolerance" in categories
        assert "file" in categories


# ============================================================================
# Test: Report formatting
# ============================================================================

class TestReportFormatting:
    """Test production report formatting."""

    def test_has_title(self, report):
        text = format_production_report(report)
        assert "生产准备就绪检查" in text

    def test_has_result(self, report):
        text = format_production_report(report)
        assert "通过" in text or "✓" in text

    def test_has_categories(self, report):
        text = format_production_report(report)
        assert "装配体验证" in text
        assert "固件代码" in text
        assert "物料清单" in text

    def test_has_check_items(self, report):
        text = format_production_report(report)
        for c in report.checks:
            assert c.name in text


# ============================================================================
# Test: Tool registration and execution
# ============================================================================

class TestProductionTool:
    """Test ProductionCheckTool."""

    def test_registered(self):
        registry = ToolRegistry()
        register_production_tools(registry)
        assert "production_check" in registry.list_tools()

    def test_execute_default(self):
        tool = ProductionCheckTool()
        result = tool.execute()
        assert "生产检查" in result
        assert "生产就绪" in result

    def test_execute_with_params(self):
        tool = ProductionCheckTool()
        result = tool.execute(
            actuator_ids=["MG996R", "MG996R", "DS3218", "SG90"],
            sensor_ids=["AS5600", "LIMIT_SWITCH_MICRO", "MPU6050"],
            controller="esp32",
        )
        assert "通过" in result

    def test_execute_markdown(self):
        tool = ProductionCheckTool()
        result = tool.execute()
        assert "--- Markdown ---" in result
        assert "生产准备就绪检查" in result

    def test_execute_unknown_assembly(self):
        tool = ProductionCheckTool()
        result = tool.execute(assembly_name="nonexistent")
        assert "错误" in result

    def test_execute_shows_categories(self):
        tool = ProductionCheckTool()
        result = tool.execute()
        assert "assembly" in result or "装配" in result
        assert "code" in result or "固件" in result
        assert "bom" in result or "物料" in result
