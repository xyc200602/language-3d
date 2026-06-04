"""Tests for assembly guide generation.

Tests cover:
- Guide structure and sections
- Assembly step content
- Wiring instructions
- Calibration and test sections
- Troubleshooting guide
- Tool registration and execution
"""

from __future__ import annotations

import pytest

from lang3d.knowledge.mechanics import ROBOTIC_ARM_ASSEMBLY
from lang3d.tools.assembly_doc import (
    GenAssemblyGuideTool,
    generate_assembly_guide,
    register_assembly_doc_tools,
)
from lang3d.tools.base import ToolRegistry


# ============================================================================
# Test: Guide structure
# ============================================================================

class TestGuideStructure:
    """Test assembly guide structure and sections."""

    @pytest.fixture
    def guide(self):
        return generate_assembly_guide(
            ROBOTIC_ARM_ASSEMBLY,
            actuator_ids=["MG996R", "MG996R", "DS3218", "SG90"],
            sensor_ids=["AS5600", "LIMIT_SWITCH_MICRO"],
        )

    def test_has_title(self, guide):
        assert "# 装配指导书" in guide

    def test_has_parts_section(self, guide):
        assert "## 1. 零件清单" in guide

    def test_has_tools_section(self, guide):
        assert "## 2. 所需工具" in guide

    def test_has_assembly_steps(self, guide):
        assert "## 3. 装配步骤" in guide

    def test_has_wiring_section(self, guide):
        assert "## 4. 接线说明" in guide

    def test_has_calibration(self, guide):
        assert "## 5. 校准步骤" in guide

    def test_has_test_section(self, guide):
        assert "## 6. 测试" in guide

    def test_has_troubleshooting(self, guide):
        assert "## 7. 常见问题排查" in guide

    def test_contains_assembly_name(self, guide):
        assert "3-DOF Robotic Arm" in guide

    def test_guide_is_not_empty(self, guide):
        assert len(guide) > 200


# ============================================================================
# Test: Parts checklist
# ============================================================================

class TestPartsChecklist:
    """Test parts checklist content."""

    @pytest.fixture
    def guide(self):
        return generate_assembly_guide(ROBOTIC_ARM_ASSEMBLY)

    def test_lists_all_parts(self, guide):
        for part in ROBOTIC_ARM_ASSEMBLY.parts:
            assert part.name in guide

    def test_lists_standard_parts(self, guide):
        assert "M3×10 螺丝" in guide
        assert "M3 螺母" in guide
        assert "MR105ZZ 轴承" in guide

    def test_lists_tools(self, guide):
        assert "螺丝刀" in guide
        assert "万用表" in guide
        assert "USB" in guide


# ============================================================================
# Test: Assembly steps
# ============================================================================

class TestAssemblySteps:
    """Test assembly step content."""

    @pytest.fixture
    def guide(self):
        return generate_assembly_guide(
            ROBOTIC_ARM_ASSEMBLY,
            actuator_ids=["MG996R", "MG996R", "DS3218", "SG90"],
        )

    def test_step_count_matches_joints(self, guide):
        # Each joint gets a step
        for j in ROBOTIC_ARM_ASSEMBLY.joints:
            assert j.child in guide
            assert j.parent in guide

    def test_revolute_joint_has_range(self, guide):
        # Revolute joints should show rotation range
        assert "180°" in guide or "180" in guide  # base range

    def test_joint_types_mentioned(self, guide):
        assert "旋转关节" in guide
        assert "固定连接" in guide

    def test_actuator_names_in_steps(self, guide):
        assert "MG996R" in guide
        assert "DS3218" in guide
        assert "SG90" in guide

    def test_anchor_directions_mentioned(self, guide):
        # Should mention top/bottom anchors
        assert "top" in guide.lower() or "bottom" in guide.lower()


# ============================================================================
# Test: Wiring instructions
# ============================================================================

class TestWiringInstructions:
    """Test wiring section content."""

    def test_wiring_with_actuators(self):
        guide = generate_assembly_guide(
            ROBOTIC_ARM_ASSEMBLY,
            actuator_ids=["MG996R", "SG90"],
        )
        assert "GPIO" in guide
        assert "GND" in guide
        assert "VCC" in guide or "V" in guide

    def test_wiring_no_actuators(self):
        guide = generate_assembly_guide(ROBOTIC_ARM_ASSEMBLY)
        # No wiring section if no actuators
        assert "## 4. 接线说明" not in guide

    def test_sensor_wiring(self):
        guide = generate_assembly_guide(
            ROBOTIC_ARM_ASSEMBLY,
            actuator_ids=["MG996R"],
            sensor_ids=["AS5600", "MPU6050"],
        )
        assert "AS5600" in guide
        assert "MPU6050" in guide
        assert "SDA" in guide or "I2C" in guide or "i2c" in guide

    def test_esp32_controller(self):
        guide = generate_assembly_guide(
            ROBOTIC_ARM_ASSEMBLY,
            actuator_ids=["MG996R"],
            controller="esp32",
        )
        assert "ESP32" in guide

    def test_arduino_controller(self):
        guide = generate_assembly_guide(
            ROBOTIC_ARM_ASSEMBLY,
            actuator_ids=["MG996R"],
            controller="arduino",
        )
        assert "ARDUINO" in guide or "arduino" in guide.lower()


# ============================================================================
# Test: Calibration and test sections
# ============================================================================

class TestCalibrationAndTest:
    """Test calibration and test sections."""

    @pytest.fixture
    def guide(self):
        return generate_assembly_guide(
            ROBOTIC_ARM_ASSEMBLY,
            actuator_ids=["MG996R"],
            sensor_ids=["AS5600"],
        )

    def test_calibration_mentions_serial(self, guide):
        assert "串口" in guide or "Serial" in guide or "115200" in guide

    def test_calibration_mentions_home(self, guide):
        assert "归零" in guide or "H" in guide

    def test_test_has_checklist(self, guide):
        assert "[ ]" in guide

    def test_test_checks_joints(self, guide):
        assert "旋转" in guide or "俯仰" in guide or "弯曲" in guide

    def test_sensor_check_in_calibration(self, guide):
        # When sensors are present, S command should be in calibration
        assert "'S'" in guide or "S" in guide


# ============================================================================
# Test: Troubleshooting
# ============================================================================

class TestTroubleshooting:
    """Test troubleshooting section."""

    @pytest.fixture
    def guide(self):
        return generate_assembly_guide(ROBOTIC_ARM_ASSEMBLY)

    def test_has_common_problems(self, guide):
        assert "抖动" in guide or "不动" in guide
        assert "排查" in guide

    def test_has_solutions(self, guide):
        assert "检查" in guide
        assert "电容" in guide or "校准" in guide


# ============================================================================
# Test: Tool registration and execution
# ============================================================================

class TestAssemblyDocTool:
    """Test GenAssemblyGuideTool."""

    def test_registered(self):
        registry = ToolRegistry()
        register_assembly_doc_tools(registry)
        assert "gen_assembly_guide" in registry.list_tools()

    def test_execute_basic(self):
        tool = GenAssemblyGuideTool()
        result = tool.execute()
        assert "Assembly Guide Generated" in result
        assert "3-DOF Robotic Arm" in result

    def test_execute_with_actuators(self):
        tool = GenAssemblyGuideTool()
        result = tool.execute(actuator_ids=["MG996R", "SG90"])
        assert "MG996R" in result
        assert "接线说明" in result

    def test_execute_unknown_assembly(self):
        tool = GenAssemblyGuideTool()
        result = tool.execute(assembly_name="nonexistent")
        assert "错误" in result

    def test_execute_full(self):
        tool = GenAssemblyGuideTool()
        result = tool.execute(
            actuator_ids=["MG996R", "MG996R", "DS3218", "SG90"],
            sensor_ids=["AS5600", "LIMIT_SWITCH_MICRO", "MPU6050"],
            controller="esp32",
        )
        assert "AS5600" in result
        assert "MPU6050" in result
        assert "ESP32" in result
        assert "传感器接线" in result
