"""Tests for DC motor firmware generation (Task 50).

Covers:
  - DCMotorPIDSpec lookup and defaults
  - EncoderSpec lookup
  - gen_motor_driver_code (header + implementation generation)
  - GenMotorDriverTool
  - gen_odometry_code (CPR, mm/tick, Pose update)
  - GenOdometryTool
  - Registration tests
"""

import math

import pytest

from lang3d.knowledge.actuators import (
    DCMotorPIDSpec,
    get_motor_pid_spec,
    get_actuator,
    DC_MOTOR_PID_SPECS,
)
from lang3d.knowledge.sensors import (
    EncoderSpec,
    get_encoder_spec,
    ENCODER_SPECS,
)
from lang3d.tools.code_gen import (
    GenMotorDriverTool,
    GenOdometryTool,
    gen_motor_driver_code,
    gen_odometry_code,
    register_code_gen_tools,
)
from lang3d.tools.base import ToolRegistry


# ============================================================================
# DCMotorPIDSpec
# ============================================================================


class TestDCMotorPIDSpec:
    def test_tt_motor_spec_exists(self):
        spec = get_motor_pid_spec("TT_MOTOR")
        assert spec is not None
        assert spec.kp == 1.2
        assert spec.ki == 0.3
        assert spec.kd == 0.05
        assert spec.gear_ratio == 48.0
        assert spec.encoder_ppr == 7

    def test_jgb37_spec(self):
        spec = get_motor_pid_spec("JGB37_520")
        assert spec is not None
        assert spec.kp == 2.0
        assert spec.max_pwm == 240
        assert spec.sample_period_ms == 10

    def test_ga25_spec(self):
        spec = get_motor_pid_spec("GA25_370_6V")
        assert spec is not None
        assert spec.gear_ratio == 30.0
        assert spec.encoder_ppr == 11

    def test_unknown_motor_returns_none(self):
        assert get_motor_pid_spec("UNKNOWN") is None

    def test_case_insensitive(self):
        assert get_motor_pid_spec("tt_motor") is not None

    def test_all_specs_have_valid_pid(self):
        for motor_id, spec in DC_MOTOR_PID_SPECS.items():
            assert spec.kp > 0, f"{motor_id} has non-positive Kp"
            assert spec.ki >= 0, f"{motor_id} has negative Ki"
            assert spec.kd >= 0, f"{motor_id} has negative Kd"
            assert spec.gear_ratio > 0, f"{motor_id} has non-positive gear_ratio"
            assert spec.max_pwm > 0, f"{motor_id} has non-positive max_pwm"


# ============================================================================
# EncoderSpec
# ============================================================================


class TestEncoderSpec:
    def test_hall_tt_spec(self):
        enc = get_encoder_spec("HALL_TT_7PPR")
        assert enc is not None
        assert enc.ppr == 7
        assert enc.quadrature is True
        assert enc.interface == "gpio_ab"

    def test_as5600_encoder(self):
        enc = get_encoder_spec("AS5600_ENCODER")
        assert enc is not None
        assert enc.ppr == 4096
        assert enc.quadrature is False
        assert enc.interface == "i2c"

    def test_optical_encoder(self):
        enc = get_encoder_spec("OPTICAL_360PPR")
        assert enc is not None
        assert enc.ppr == 360
        assert enc.max_rpm > 0

    def test_unknown_returns_none(self):
        assert get_encoder_spec("NONEXISTENT") is None

    def test_all_specs_have_ppr(self):
        for eid, spec in ENCODER_SPECS.items():
            assert spec.ppr > 0, f"{eid} has non-positive PPR"


# ============================================================================
# gen_motor_driver_code
# ============================================================================


class TestGenMotorDriverCode:
    def test_generates_header_and_cpp(self):
        motors = [
            {"motor_id": "TT_MOTOR", "encoder_id": "HALL_TT_7PPR",
             "pwm_pin": 5, "dir_pin1": 6, "dir_pin2": 7,
             "enc_a_pin": 18, "enc_b_pin": 19},
        ]
        files = gen_motor_driver_code(motors)
        assert "dc_motor_driver.h" in files
        assert "dc_motor_driver.cpp" in files

    def test_header_has_pid_struct(self):
        motors = [{"motor_id": "TT_MOTOR", "encoder_id": "HALL_TT_7PPR",
                    "pwm_pin": 5, "dir_pin1": 6, "dir_pin2": 7,
                    "enc_a_pin": 18, "enc_b_pin": 19}]
        files = gen_motor_driver_code(motors)
        h = files["dc_motor_driver.h"]
        assert "PIDState" in h
        assert "motor_set_speed" in h
        assert "motor_pid_update" in h
        assert "motor_read_encoder" in h

    def test_cpp_has_pid_values(self):
        motors = [{"motor_id": "TT_MOTOR", "encoder_id": "HALL_TT_7PPR",
                    "pwm_pin": 5, "dir_pin1": 6, "dir_pin2": 7,
                    "enc_a_pin": 18, "enc_b_pin": 19}]
        files = gen_motor_driver_code(motors)
        cpp = files["dc_motor_driver.cpp"]
        assert "1.20" in cpp  # Kp for TT_MOTOR
        assert "0.30" in cpp  # Ki
        assert "48.0" in cpp  # Gear ratio

    def test_dual_motor_config(self):
        motors = [
            {"motor_id": "TT_MOTOR", "encoder_id": "HALL_TT_7PPR",
             "pwm_pin": 5, "dir_pin1": 6, "dir_pin2": 7,
             "enc_a_pin": 18, "enc_b_pin": 19},
            {"motor_id": "TT_MOTOR", "encoder_id": "HALL_TT_7PPR",
             "pwm_pin": 8, "dir_pin1": 9, "dir_pin2": 10,
             "enc_a_pin": 20, "enc_b_pin": 21},
        ]
        files = gen_motor_driver_code(motors)
        cpp = files["dc_motor_driver.cpp"]
        assert "encoder_isr_0" in cpp
        assert "encoder_isr_1" in cpp
        assert "NUM_MOTORS 2" in files["dc_motor_driver.h"]

    def test_unknown_motor_uses_defaults(self):
        motors = [{"motor_id": "MYSTERY", "encoder_id": "UNKNOWN",
                    "pwm_pin": 5, "dir_pin1": 6, "dir_pin2": 7,
                    "enc_a_pin": 18, "enc_b_pin": 19}]
        files = gen_motor_driver_code(motors)
        assert "1.0" in files["dc_motor_driver.cpp"]  # Default Kp


# ============================================================================
# GenMotorDriverTool
# ============================================================================


class TestGenMotorDriverTool:
    def test_tool_execution(self):
        tool = GenMotorDriverTool()
        result = tool.execute(
            motors=[
                {"motor_id": "TT_MOTOR", "encoder_id": "HALL_TT_7PPR",
                 "pwm_pin": 5, "dir_pin1": 6, "dir_pin2": 7,
                 "enc_a_pin": 18, "enc_b_pin": 19},
            ]
        )
        assert "[Motor Driver Generated]" in result
        assert "TT_MOTOR" in result
        assert "Kp=1.2" in result

    def test_missing_motors(self):
        tool = GenMotorDriverTool()
        result = tool.execute()
        assert "错误" in result


# ============================================================================
# gen_odometry_code
# ============================================================================


class TestGenOdometryCode:
    def test_generates_code(self):
        code = gen_odometry_code()
        assert "odometry_update" in code
        assert "odometry_reset" in code
        assert "odometry_velocity_to_wheels" in code
        assert "Pose" in code

    def test_correct_cpr(self):
        code = gen_odometry_code(encoder_ppr=7, gear_ratio=48.0, quadrature=True)
        cpr = 7 * 4 * 48  # 1344
        assert str(cpr) in code

    def test_mm_per_tick(self):
        r = 32.5
        cpr = 7 * 4 * 48  # 1344
        expected_mm = 2 * math.pi * r / cpr
        code = gen_odometry_code(wheel_radius_mm=r, encoder_ppr=7, gear_ratio=48.0)
        assert f"{expected_mm:.4f}" in code or f"{expected_mm:.6f}" in code

    def test_wheel_base_in_code(self):
        code = gen_odometry_code(wheel_base_mm=200.0)
        assert "200.00" in code

    def test_non_quadrature(self):
        code = gen_odometry_code(encoder_ppr=11, gear_ratio=30.0, quadrature=False)
        cpr = 11 * 30  # 330 (no x4)
        assert str(cpr) in code


# ============================================================================
# GenOdometryTool
# ============================================================================


class TestGenOdometryTool:
    def test_default_execution(self):
        tool = GenOdometryTool()
        result = tool.execute()
        assert "[Odometry Code Generated]" in result
        assert "CPR:" in result

    def test_custom_params(self):
        tool = GenOdometryTool()
        result = tool.execute(
            wheel_radius_mm=50.0,
            wheel_base_mm=200.0,
            encoder_ppr=11,
            gear_ratio=30.0,
        )
        assert "50" in result
        assert "200" in result


# ============================================================================
# Registration
# ============================================================================


class TestRegistration:
    def test_motor_driver_registered(self):
        registry = ToolRegistry()
        register_code_gen_tools(registry)
        names = [t.name for t in registry._tools.values()]
        assert "gen_motor_driver" in names
        assert "gen_odometry" in names
        assert "gen_firmware" in names

    def test_all_five_tools(self):
        registry = ToolRegistry()
        register_code_gen_tools(registry)
        names = [t.name for t in registry._tools.values()]
        expected = ["gen_firmware", "gen_wiring_diagram", "gen_test_sequence",
                     "gen_motor_driver", "gen_odometry"]
        for name in expected:
            assert name in names, f"{name} not registered"
