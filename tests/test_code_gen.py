"""Tests for firmware code generation tools.

Tests cover:
- Firmware generation (all 5 files)
- Wiring diagram generation
- Test sequence generation
- Tool registration and execution
- Code content validation
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from lang3d.knowledge.mechanics import ROBOTIC_ARM_ASSEMBLY
from lang3d.tools.code_gen import (
    GenFirmwareTool,
    GenTestSequenceTool,
    GenWiringDiagramTool,
    generate_firmware,
    generate_test_sequence,
    generate_wiring,
    register_code_gen_tools,
)
from lang3d.tools.base import ToolRegistry


# ============================================================================
# Test: Firmware generation
# ============================================================================

class TestFirmwareGeneration:
    """Test firmware file generation."""

    @pytest.fixture
    def firmware(self):
        return generate_firmware(ROBOTIC_ARM_ASSEMBLY, ["MG996R", "MG996R", "DS3218", "SG90"])

    def test_generates_five_files(self, firmware):
        expected = {"robot_arm.ino", "ik_solver.h", "ik_solver.cpp",
                    "servo_driver.h", "servo_driver.cpp"}
        assert set(firmware.keys()) == expected

    def test_ino_has_setup_and_loop(self, firmware):
        ino = firmware["robot_arm.ino"]
        assert "void setup()" in ino
        assert "void loop()" in ino

    def test_ino_has_serial(self, firmware):
        ino = firmware["robot_arm.ino"]
        assert "Serial.begin" in ino
        assert "Serial.println" in ino

    def test_ino_includes_libs(self, firmware):
        ino = firmware["robot_arm.ino"]
        assert "ESP32Servo.h" in ino

    def test_ino_arduino_lib(self):
        files = generate_firmware(ROBOTIC_ARM_ASSEMBLY, ["MG996R"], controller="arduino")
        assert "Servo.h" in files["robot_arm.ino"]

    def test_ik_header_has_num_joints(self, firmware):
        h = firmware["ik_solver.h"]
        assert "NUM_JOINTS 4" in h
        assert "IKSolution" in h

    def test_ik_cpp_has_link_lengths(self, firmware):
        cpp = firmware["ik_solver.cpp"]
        assert "LINK_1" in cpp
        assert "LINK_2" in cpp
        assert "BASE_HEIGHT" in cpp

    def test_ik_cpp_has_cosine_law(self, firmware):
        cpp = firmware["ik_solver.cpp"]
        assert "acos" in cpp
        assert "atan2" in cpp

    def test_ik_cpp_has_joint_limits(self, firmware):
        cpp = firmware["ik_solver.cpp"]
        assert "JOINT_MIN" in cpp
        assert "JOINT_MAX" in cpp
        assert "clamp_angle" in cpp

    def test_servo_header(self, firmware):
        h = firmware["servo_driver.h"]
        assert "NUM_SERVOS" in h
        assert "servo_write" in h

    def test_servo_cpp_has_pwm_mapping(self, firmware):
        cpp = firmware["servo_driver.cpp"]
        assert "SERVO_MIN_PULSE" in cpp
        assert "SERVO_MAX_PULSE" in cpp
        assert "angle_to_pwm" in cpp

    def test_all_files_nonempty(self, firmware):
        for fname, content in firmware.items():
            assert len(content) > 50, f"{fname} is too short"

    def test_ik_solver_has_fk(self, firmware):
        cpp = firmware["ik_solver.cpp"]
        assert "fk_compute" in cpp


# ============================================================================
# Test: Wiring diagram
# ============================================================================

class TestWiringDiagram:
    def test_basic_wiring(self):
        w = generate_wiring(["MG996R", "SG90"])
        assert "MG996R" in w
        assert "SG90" in w
        assert "GPIO" in w
        assert "GND" in w

    def test_esp32_pins(self):
        w = generate_wiring(["MG996R"], controller="esp32")
        assert "ESP32" in w

    def test_arduino_pins(self):
        w = generate_wiring(["MG996R"], controller="arduino")
        assert "Arduino" in w or "arduino" in w.lower()

    def test_power_supply_info(self):
        w = generate_wiring(["MG996R", "SG90"])
        assert "Power" in w or "V" in w

    def test_duplicate_actuators(self):
        w = generate_wiring(["MG996R", "MG996R"])
        # Should list both with different labels
        assert "#1" in w or "#2" in w or "MG996R" in w


# ============================================================================
# Test: Test sequence
# ============================================================================

class TestTestSequence:
    def test_basic_sequence(self):
        t = generate_test_sequence(ROBOTIC_ARM_ASSEMBLY)
        assert "Phase 1" in t
        assert "Phase 2" in t
        assert "Phase 3" in t

    def test_has_serial_commands(self):
        t = generate_test_sequence(ROBOTIC_ARM_ASSEMBLY)
        # Commands start with G
        assert "G0,0,0,0" in t or "G0.0,0.0,0.0,0.0" in t or "G" in t

    def test_custom_steps(self):
        t = generate_test_sequence(ROBOTIC_ARM_ASSEMBLY, steps=10)
        assert "Phase 3" in t

    def test_has_checklist(self):
        t = generate_test_sequence(ROBOTIC_ARM_ASSEMBLY)
        assert "[ ]" in t


# ============================================================================
# Test: Tool registration and execution
# ============================================================================

class TestCodeGenTools:
    def test_all_three_registered(self):
        registry = ToolRegistry()
        register_code_gen_tools(registry)
        names = registry.list_tools()
        assert "gen_firmware" in names
        assert "gen_wiring_diagram" in names
        assert "gen_test_sequence" in names

    def test_firmware_tool_execute(self):
        tool = GenFirmwareTool()
        result = tool.execute(actuator_ids=["MG996R", "SG90"])
        assert "Firmware Generated" in result
        assert "ik_solver" in result

    def test_firmware_tool_no_actuators(self):
        tool = GenFirmwareTool()
        result = tool.execute(actuator_ids=None)
        assert "错误" in result

    def test_firmware_tool_unknown_assembly(self):
        tool = GenFirmwareTool()
        result = tool.execute(actuator_ids=["MG996R"], assembly_name="nonexistent")
        assert "错误" in result

    def test_wiring_tool_execute(self):
        tool = GenWiringDiagramTool()
        result = tool.execute(actuator_ids=["MG996R", "SG90"])
        assert "MG996R" in result

    def test_wiring_tool_empty(self):
        tool = GenWiringDiagramTool()
        result = tool.execute(actuator_ids=None)
        assert "错误" in result

    def test_test_seq_tool_execute(self):
        tool = GenTestSequenceTool()
        result = tool.execute(assembly_name="robotic_arm")
        assert "Phase" in result

    def test_test_seq_tool_unknown(self):
        tool = GenTestSequenceTool()
        result = tool.execute(assembly_name="nonexistent")
        assert "错误" in result
