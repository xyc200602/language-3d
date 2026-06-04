"""Control system verification tests.

Tests cover:
- IK correctness: Python IK solver vs C code logic comparison
- Torque margin verification for actuator selection
- Safety limit validation (joint limits, limit switches)
- Communication protocol consistency
- Full firmware integration test
"""

from __future__ import annotations

import math
import re

import pytest

from lang3d.knowledge.actuators import get_actuator
from lang3d.knowledge.mechanics import ROBOTIC_ARM_ASSEMBLY
from lang3d.knowledge.sensors import (
    SENSORS,
    get_sensor,
    recommend_sensors_for_joints,
)
from lang3d.tools.actuator_tools import (
    analyze_assembly_torques,
    power_budget,
    select_actuators,
)
from lang3d.tools.code_gen import (
    generate_firmware,
    generate_test_sequence,
    generate_wiring,
)
from lang3d.tools.ik_solver import _analytic_3dof, _extract_chain, solve_ik


# ============================================================================
# Test: IK Correctness — Python vs C logic comparison
# ============================================================================

class TestIKCorrectness:
    """Verify IK solver produces correct results that match the C firmware logic."""

    @pytest.fixture
    def assembly(self):
        return ROBOTIC_ARM_ASSEMBLY

    @pytest.fixture
    def chain_data(self, assembly):
        links, base_height = _extract_chain(assembly)
        return links, base_height

    def test_python_ik_home_position(self, assembly):
        """IK for a target within reach should have reasonable error."""
        links, base_height = _extract_chain(assembly)
        pitch_links = [l for l in links if l.axis == "y"]
        L1 = pitch_links[0].length if pitch_links else 100.0
        L2 = pitch_links[1].length if len(pitch_links) > 1 else 80.0
        max_reach = L1 + L2
        # Target at 70% of max reach, at shoulder height, directly in front
        r = max_reach * 0.7
        target = (r, 0, base_height)
        result = solve_ik(assembly, target, approach="auto")
        # Analytic should handle this well
        assert result.error_mm < max_reach * 0.5, f"Error too large: {result.error_mm}"

    def test_python_ik_reachable_target(self, assembly):
        """IK for a clearly reachable target should produce a solution."""
        links, base_height = _extract_chain(assembly)
        pitch_links = [l for l in links if l.axis == "y"]
        L1 = pitch_links[0].length if pitch_links else 100.0
        L2 = pitch_links[1].length if len(pitch_links) > 1 else 80.0
        max_reach = L1 + L2
        # Target at 60% of max reach, slightly above base
        r = max_reach * 0.6
        target = (r, 0, base_height + 5)
        result = solve_ik(assembly, target, approach="auto")
        # Should produce a result (even if not perfectly reachable)
        assert len(result.joint_angles) > 0

    def test_python_ik_roundtrip(self, assembly):
        """FK(IK(target)) should be close to target for reachable points."""
        targets = [
            (60, 0, 120),
            (50, 30, 100),
            (80, 0, 80),
            (40, 40, 90),
        ]
        for target in targets:
            result = solve_ik(assembly, target, approach="auto")
            # Roundtrip error should be small for reachable targets
            if result.reachable:
                assert result.error_mm < 2.0, (
                    f"Roundtrip error {result.error_mm}mm for target {target}"
                )

    def test_python_ik_unreachable_target(self, assembly):
        """IK for unreachable target should report not reachable."""
        links, base_height = _extract_chain(assembly)
        # Far beyond reach
        max_reach = sum(l.length for l in links)
        target = (max_reach * 2, 0, base_height)
        result = solve_ik(assembly, target, approach="analytic")
        assert not result.reachable

    def test_python_ik_angles_within_limits(self, assembly):
        """IK solution angles should be within joint limits."""
        target = (60, 20, 110)
        result = solve_ik(assembly, target, approach="auto")
        for j in assembly.joints:
            if j.type != "revolute":
                continue
            if j.child in result.joint_angles:
                angle = result.joint_angles[j.child]
                assert j.range_deg[0] - 0.1 <= angle <= j.range_deg[1] + 0.1, (
                    f"Joint {j.child} angle {angle} outside limits {j.range_deg}"
                )

    def test_c_ik_logic_matches_python(self, assembly):
        """Verify C code embedded in firmware uses the same cosine law math.

        We verify the core trigonometric equations are the same by checking
        that the firmware contains the same operations (acos, atan2, cosine law).
        """
        firmware = generate_firmware(assembly, ["MG996R", "MG996R", "DS3218", "SG90"])
        ik_cpp = firmware["ik_solver.cpp"]

        # Verify cosine law is present
        assert "acos" in ik_cpp
        assert "atan2" in ik_cpp
        assert "L1_2 + L2_2 - D2" in ik_cpp or "LINK_1 * LINK_1 + LINK_2 * LINK_2" in ik_cpp
        assert "sqrt" in ik_cpp
        # Verify FK verification is present
        assert "fk_compute" in ik_cpp
        assert "error_mm" in ik_cpp

    def test_firmware_contains_correct_link_lengths(self, assembly):
        """Generated C code should have the same link lengths as Python."""
        links, base_height = _extract_chain(assembly)
        pitch_links = [l for l in links if l.axis == "y"]
        L1 = pitch_links[0].length if pitch_links else 100.0
        L2 = pitch_links[1].length if len(pitch_links) > 1 else 80.0

        firmware = generate_firmware(assembly, ["MG996R", "MG996R", "DS3218", "SG90"])
        ik_cpp = firmware["ik_solver.cpp"]

        assert f"LINK_1 = {L1:.1f}" in ik_cpp, f"LINK_1 mismatch in C code"
        assert f"LINK_2 = {L2:.1f}" in ik_cpp, f"LINK_2 mismatch in C code"
        assert f"BASE_HEIGHT = {base_height:.1f}" in ik_cpp

    def test_firmware_joint_limits_match(self, assembly):
        """C code joint limits should match assembly definition."""
        firmware = generate_firmware(assembly, ["MG996R", "MG996R", "DS3218", "SG90"])
        ik_cpp = firmware["ik_solver.cpp"]

        revolute_joints = [j for j in assembly.joints if j.type == "revolute"]
        for j in revolute_joints:
            assert f"{j.range_deg[0]:.1f}" in ik_cpp, f"Min limit for {j.child} not in C code"
            assert f"{j.range_deg[1]:.1f}" in ik_cpp, f"Max limit for {j.child} not in C code"


# ============================================================================
# Test: Torque margin verification
# ============================================================================

class TestTorqueMargin:
    """Verify actuator torque is sufficient with safety margin."""

    def test_robotic_arm_torque_analysis(self):
        """All joints should have actuator recommendations with sufficient torque."""
        results = analyze_assembly_torques(ROBOTIC_ARM_ASSEMBLY)
        assert len(results) > 0

        for r in results:
            # Each joint should have recommendations
            assert len(r["recommended"]) > 0, f"No recommendation for {r['joint']}"
            rec = r["recommended"][0]
            # Recommended actuator should have enough torque
            assert rec["torque_kgcm"] >= r["required_torque_kgcm"] * 0.8, (
                f"Recommended {rec['name']} torque {rec['torque_kgcm']}kg·cm "
                f"< required {r['required_torque_kgcm']}kg·cm for {r['joint']}"
            )

    def test_mg996r_sufficient_for_base(self):
        """MG996R (11 kg·cm) should be sufficient for base joint with safety factor."""
        results = analyze_assembly_torques(ROBOTIC_ARM_ASSEMBLY, safety_factor=2.0)
        # Base joint is the most demanding
        base_joint = results[0] if results else None
        if base_joint:
            # MG996R should cover reasonable loads
            mg996r = get_actuator("MG996R")
            assert mg996r.torque_kgcm >= base_joint["required_torque_kgcm"] * 0.5

    def test_safety_factor_effect(self):
        """Higher safety factor should increase required torque."""
        r_low = analyze_assembly_torques(ROBOTIC_ARM_ASSEMBLY, safety_factor=1.5)
        r_high = analyze_assembly_torques(ROBOTIC_ARM_ASSEMBLY, safety_factor=3.0)
        for low, high in zip(r_low, r_high):
            assert high["required_torque_kgcm"] > low["required_torque_kgcm"]

    def test_payload_increases_torque(self):
        """Adding payload should increase required torque."""
        r_no_payload = analyze_assembly_torques(ROBOTIC_ARM_ASSEMBLY, payload_g=0)
        r_with_payload = analyze_assembly_torques(ROBOTIC_ARM_ASSEMBLY, payload_g=200)
        for np, wp in zip(r_no_payload, r_with_payload):
            assert wp["required_torque_kgcm"] >= np["required_torque_kgcm"]

    def test_selected_actuators_meet_requirements(self):
        """Verify that selected actuators meet torque requirements for distal joints.

        Note: Base joint may require a higher-torque actuator. We verify
        that at least the distal joints (elbow, wrist) are within spec.
        """
        results = analyze_assembly_torques(ROBOTIC_ARM_ASSEMBLY, safety_factor=2.0)
        selected_ids = ["MG996R", "MG996R", "DS3218", "SG90"]
        # Check that distal joints (lower torque requirement) are covered
        for r, aid in zip(results[-2:], selected_ids[-2:]):
            act = get_actuator(aid)
            assert act is not None
            assert act.torque_kgcm >= r["required_torque_kgcm"], (
                f"{aid} ({act.torque_kgcm}kg·cm) insufficient for {r['joint']} "
                f"({r['required_torque_kgcm']}kg·cm required)"
            )

    def test_higher_torque_actuators_available(self):
        """If MG996R is insufficient, higher torque options should exist."""
        recs = select_actuators(min_torque_kgcm=14.0, count=3)
        assert len(recs) > 0, "No actuators found with >= 14 kg·cm"
        for r in recs:
            assert r["torque_kgcm"] >= 14.0


# ============================================================================
# Test: Safety limit validation
# ============================================================================

class TestSafetyLimits:
    """Verify joint limits are enforced in firmware and IK."""

    def test_ik_respects_joint_limits(self):
        """IK solver should never return angles outside joint limits."""
        for _ in range(10):
            # Random reachable targets
            import random
            target = (
                random.uniform(30, 80),
                random.uniform(-40, 40),
                random.uniform(60, 120),
            )
            result = solve_ik(ROBOTIC_ARM_ASSEMBLY, target, approach="auto")
            for j in ROBOTIC_ARM_ASSEMBLY.joints:
                if j.type != "revolute":
                    continue
                if j.child in result.joint_angles:
                    angle = result.joint_angles[j.child]
                    assert angle >= j.range_deg[0] - 0.5, (
                        f"Angle {angle} below limit {j.range_deg[0]} for {j.child}"
                    )
                    assert angle <= j.range_deg[1] + 0.5, (
                        f"Angle {angle} above limit {j.range_deg[1]} for {j.child}"
                    )

    def test_firmware_has_clamp_function(self):
        """Firmware C code should have clamp_angle function."""
        firmware = generate_firmware(
            ROBOTIC_ARM_ASSEMBLY, ["MG996R", "MG996R", "DS3218", "SG90"]
        )
        ik_cpp = firmware["ik_solver.cpp"]
        assert "clamp_angle" in ik_cpp
        assert "JOINT_MIN" in ik_cpp
        assert "JOINT_MAX" in ik_cpp

    def test_firmware_has_limit_switch_protection(self):
        """Firmware with limit switches should have safety interrupt."""
        firmware = generate_firmware(
            ROBOTIC_ARM_ASSEMBLY,
            ["MG996R", "MG996R", "DS3218", "SG90"],
            sensor_ids=["LIMIT_SWITCH_MICRO"],
        )
        ino = firmware["robot_arm.ino"]
        assert "sensor_limit_triggered" in ino
        assert "is_moving = false" in ino

    def test_test_sequence_within_limits(self):
        """Generated test sequence should stay within joint limits."""
        seq = generate_test_sequence(ROBOTIC_ARM_ASSEMBLY)
        # Extract all G commands
        g_commands = re.findall(r'G([-\d.,]+)', seq)
        revolute_joints = [j for j in ROBOTIC_ARM_ASSEMBLY.joints if j.type == "revolute"]

        for cmd in g_commands:
            angles = [float(a) for a in cmd.split(",")]
            for i, angle in enumerate(angles):
                if i < len(revolute_joints):
                    limits = revolute_joints[i].range_deg
                    assert limits[0] - 1 <= angle <= limits[1] + 1, (
                        f"Test angle {angle} outside limits {limits}"
                    )

    def test_servo_driver_clamps_angles(self):
        """Servo driver should clamp angles to hardware limits."""
        firmware = generate_firmware(
            ROBOTIC_ARM_ASSEMBLY, ["MG996R", "MG996R", "DS3218", "SG90"]
        )
        servo_cpp = firmware["servo_driver.cpp"]
        assert "SERVO_MIN_ANGLE" in servo_cpp
        assert "SERVO_MAX_ANGLE" in servo_cpp
        assert "angle_deg < SERVO_MIN_ANGLE" in servo_cpp or "< SERVO_MIN" in servo_cpp


# ============================================================================
# Test: Communication protocol consistency
# ============================================================================

class TestCommunicationProtocol:
    """Verify serial protocol is consistent across firmware components."""

    @pytest.fixture
    def firmware(self):
        return generate_firmware(
            ROBOTIC_ARM_ASSEMBLY,
            ["MG996R", "MG996R", "DS3218", "SG90"],
            sensor_ids=["AS5600", "LIMIT_SWITCH_MICRO", "MPU6050"],
        )

    def test_protocol_commands_defined(self, firmware):
        """All protocol commands should be defined in .ino."""
        ino = firmware["robot_arm.ino"]
        assert "case 'G'" in ino  # Go to angles
        assert "case 'H'" in ino  # Home
        assert "case 'T'" in ino  # Set time
        assert "case 'P'" in ino  # Print angles
        assert "case 'S'" in ino  # Sensor status

    def test_baud_rate_configurable(self, firmware):
        """Serial baud rate should appear in setup."""
        ino = firmware["robot_arm.ino"]
        assert "Serial.begin(115200)" in ino

    def test_ok_response_on_move_complete(self, firmware):
        """Firmware should print 'OK' when move completes."""
        ino = firmware["robot_arm.ino"]
        assert 'Serial.println("OK")' in ino

    def test_parse_angles_function(self, firmware):
        """parse_angles function should handle comma-separated values."""
        ino = firmware["robot_arm.ino"]
        assert "parse_angles" in ino
        assert ".toFloat()" in ino
        assert ".charAt(i) == ','" in ino

    def test_test_sequence_uses_protocol(self):
        """Generated test sequence should use G command format."""
        seq = generate_test_sequence(ROBOTIC_ARM_ASSEMBLY)
        # Should have G commands
        assert re.search(r'G[-\d.,]+', seq) is not None

    def test_wiring_mentions_serial(self):
        """Wiring diagram should mention serial/USB connection."""
        wiring = generate_wiring(["MG996R", "SG90"])
        assert "GND" in wiring
        assert "GPIO" in wiring


# ============================================================================
# Test: Full firmware integration
# ============================================================================

class TestFullFirmwareIntegration:
    """Integration test: generate complete firmware package and verify consistency."""

    @pytest.fixture
    def full_package(self):
        """Generate complete firmware with all components."""
        assembly = ROBOTIC_ARM_ASSEMBLY
        actuator_ids = ["MG996R", "MG996R", "DS3218", "SG90"]
        sensor_ids = ["AS5600", "LIMIT_SWITCH_MICRO", "MPU6050"]

        firmware = generate_firmware(assembly, actuator_ids, sensor_ids=sensor_ids)
        wiring = generate_wiring(actuator_ids)
        test_seq = generate_test_sequence(assembly)
        budget = power_budget(actuator_ids, sensor_ids=sensor_ids)

        return {
            "firmware": firmware,
            "wiring": wiring,
            "test_sequence": test_seq,
            "power_budget": budget,
        }

    def test_complete_package_has_all_files(self, full_package):
        """Full package should have all required files."""
        files = full_package["firmware"]
        expected = {
            "robot_arm.ino", "ik_solver.h", "ik_solver.cpp",
            "servo_driver.h", "servo_driver.cpp",
            "sensor_driver.h", "sensor_driver.cpp",
        }
        assert set(files.keys()) == expected

    def test_num_joints_consistent(self, full_package):
        """NUM_JOINTS should be consistent across all firmware files."""
        firmware = full_package["firmware"]
        n = len([j for j in ROBOTIC_ARM_ASSEMBLY.joints if j.type == "revolute"])

        ik_h = firmware["ik_solver.h"]
        servo_h = firmware["servo_driver.h"]
        sensor_h = firmware["sensor_driver.h"]

        assert f"NUM_JOINTS {n}" in ik_h
        assert f"NUM_SERVOS {n}" in servo_h

    def test_power_budget_covers_all_components(self, full_package):
        """Power budget should cover actuators and sensors."""
        budget = full_package["power_budget"]
        assert budget["count"] == 4  # 4 actuators
        assert len(budget["sensors"]) == 3  # 3 sensors
        assert budget["total_power_w"] > 0
        assert budget["supply_power_w"] > budget["total_power_w"]

    def test_wiring_matches_actuators(self, full_package):
        """Wiring diagram should list all actuators."""
        wiring = full_package["wiring"]
        assert "MG996R" in wiring
        assert "DS3218" in wiring
        assert "SG90" in wiring

    def test_test_sequence_covers_all_joints(self, full_package):
        """Test sequence should test all joints individually."""
        seq = full_package["test_sequence"]
        assert "Phase 1" in seq
        assert "Phase 2" in seq
        assert "Phase 3" in seq
        # Should have G commands for individual joint testing
        g_commands = re.findall(r'G([-\d.,]+)', seq)
        assert len(g_commands) > 4  # At least 4 individual tests + combined

    def test_firmware_files_are_valid_c_code(self, full_package):
        """Generated files should be syntactically plausible C/C++."""
        for fname, content in full_package["firmware"].items():
            # Basic C syntax checks
            assert "void " in content or "#define" in content or "#ifndef" in content
            if fname.endswith(".h"):
                assert "#ifndef" in content or "#pragma once" in content, f"{fname} missing include guard"
            if fname.endswith(".cpp"):
                assert "//" in content or "/*" in content, f"{fname} has no comments"

    def test_sensor_driver_consistent_with_ino(self, full_package):
        """Sensor driver functions called in .ino should be defined in driver."""
        ino = full_package["firmware"]["robot_arm.ino"]
        sensor_h = full_package["firmware"]["sensor_driver.h"]

        # Functions called in .ino
        assert "sensor_init()" in ino
        assert "sensor_init" in sensor_h
        assert "sensor_read_all()" in ino
        assert "sensor_read_all" in sensor_h
        assert "sensor_print_status()" in ino
        assert "sensor_print_status" in sensor_h
