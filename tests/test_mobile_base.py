"""Tests for mobile base knowledge & kinematics (Task 46).

Covers: DifferentialDriveKinematics, OmnidirectionalKinematics,
MotorTorqueCalculator, BatterySizingCalculator, WheelBaseCalculator,
templates, motor catalog, tools (mobile_base_design, differential_drive_sim),
planner task type detection.
"""

import json
import math
import pytest

from lang3d.knowledge.mobile_base import (
    BatterySizingCalculator,
    DC_MOTOR_CATALOG,
    DCMotorSpec,
    DifferentialDriveKinematics,
    MobileBaseTemplate,
    MOBILE_BASE_TEMPLATES,
    MotorTorqueCalculator,
    OmnidirectionalKinematics,
    WheelBaseCalculator,
    design_mobile_base,
)
from lang3d.tools.mobile_design import (
    DifferentialDriveSimTool,
    MobileBaseDesignTool,
    register_mobile_design_tools,
)
from lang3d.agent.planner import Planner


# ── DC Motor Catalog ───────────────────────────────────────────


class TestDCMotorCatalog:
    def test_catalog_has_motors(self):
        assert len(DC_MOTOR_CATALOG) >= 4

    def test_jga25_370(self):
        m = DC_MOTOR_CATALOG["JGA25-370"]
        assert m.nominal_voltage == 12
        assert m.rated_torque_kg_cm > 0
        assert "JGA25" in m.name

    def test_nema17(self):
        m = DC_MOTOR_CATALOG["NEMA17"]
        assert m.nominal_voltage == 12
        assert m.weight_g > 0


# ── Differential Drive Kinematics ──────────────────────────────


class TestDifferentialDrive:
    def test_forward_straight(self):
        kin = DifferentialDriveKinematics(wheel_radius_mm=50, track_width_mm=300)
        v, omega = kin.forward_kinematics(100, 100)
        assert v == 100
        assert omega == 0.0

    def test_forward_spin(self):
        kin = DifferentialDriveKinematics(track_width_mm=300)
        v, omega = kin.forward_kinematics(-50, 50)
        assert abs(v) < 0.01  # stationary
        assert omega > 0  # spinning in place

    def test_inverse_roundtrip(self):
        kin = DifferentialDriveKinematics(track_width_mm=300)
        v_left, v_right = kin.inverse_kinematics(200, 0.5)
        v_back, omega_back = kin.forward_kinematics(v_left, v_right)
        assert abs(v_back - 200) < 0.01
        assert abs(omega_back - 0.5) < 0.01

    def test_turning_radius_straight(self):
        kin = DifferentialDriveKinematics()
        r = kin.turning_radius_mm(100, 0.0)
        assert r == float("inf")

    def test_turning_radius_curve(self):
        kin = DifferentialDriveKinematics(track_width_mm=300)
        v, omega = kin.forward_kinematics(80, 100)
        r = kin.turning_radius_mm(v, omega)
        assert r > 0

    def test_max_speed(self):
        kin = DifferentialDriveKinematics(wheel_radius_mm=50, max_rpm=200)
        max_v = kin.max_linear_speed_mm_s()
        assert max_v > 0
        # 200 RPM * circumference(2*pi*50) / 60
        expected = 200 / 60 * 2 * math.pi * 50
        assert abs(max_v - expected) < 1

    def test_rpm_from_speed(self):
        kin = DifferentialDriveKinematics(wheel_radius_mm=50)
        rpm = kin.rpm_from_speed(500)
        assert rpm > 0

    def test_to_dict(self):
        kin = DifferentialDriveKinematics()
        d = kin.to_dict()
        assert "wheel_radius_mm" in d
        assert "max_linear_speed_mm_s" in d


# ── Omnidirectional (Mecanum) Kinematics ──────────────────────


class TestOmnidirectional:
    def test_forward_only(self):
        kin = OmnidirectionalKinematics(wheelbase_mm=300, track_width_mm=300)
        wheels = kin.inverse_kinematics(100, 0, 0)
        assert all(w == 100 for w in wheels)

    def test_spin_only(self):
        kin = OmnidirectionalKinematics(wheelbase_mm=300, track_width_mm=300)
        wheels = kin.inverse_kinematics(0, 0, 1.0)
        # FR and RR positive, FL and RL negative for CCW spin
        assert wheels[1] > 0  # FR
        assert wheels[3] > 0  # RR

    def test_roundtrip(self):
        kin = OmnidirectionalKinematics(wheelbase_mm=300, track_width_mm=300)
        wheels = kin.inverse_kinematics(100, 50, 0.3)
        vx, vy, omega = kin.forward_kinematics(wheels)
        assert abs(vx - 100) < 0.01
        assert abs(vy - 50) < 0.01
        assert abs(omega - 0.3) < 0.01


# ── Motor Torque Calculator ────────────────────────────────────


class TestMotorTorque:
    def test_total_torque_positive(self):
        calc = MotorTorqueCalculator(total_mass_kg=10, wheel_radius_mm=40)
        assert calc.total_torque_nm_per_wheel() > 0

    def test_incline_increases_torque(self):
        flat = MotorTorqueCalculator(max_grade_deg=0)
        slope = MotorTorqueCalculator(max_grade_deg=30)
        assert slope.total_torque_nm_per_wheel() > flat.total_torque_nm_per_wheel()

    def test_select_motor(self):
        calc = MotorTorqueCalculator(total_mass_kg=5, wheel_radius_mm=40)
        motor = calc.select_motor()
        assert motor is not None
        assert motor.rated_torque_kg_cm >= calc.total_torque_kg_cm_per_wheel()

    def test_to_dict(self):
        calc = MotorTorqueCalculator()
        d = calc.to_dict()
        assert "total_torque_kg_cm_per_wheel" in d


# ── Battery Sizing ─────────────────────────────────────────────


class TestBatterySizing:
    def test_basic(self):
        calc = BatterySizingCalculator()
        ah = calc.capacity_ah()
        assert ah > 0

    def test_more_motors_more_capacity(self):
        small = BatterySizingCalculator(motor_count=2)
        large = BatterySizingCalculator(motor_count=4)
        assert large.capacity_ah() > small.capacity_ah()

    def test_safety_factor(self):
        no_sf = BatterySizingCalculator(safety_factor=1.0)
        with_sf = BatterySizingCalculator(safety_factor=1.5)
        assert with_sf.capacity_ah() > no_sf.capacity_ah()

    def test_to_dict(self):
        calc = BatterySizingCalculator()
        d = calc.to_dict()
        assert "capacity_ah" in d
        assert "energy_wh" in d


# ── Wheelbase Calculator ──────────────────────────────────────


class TestWheelbaseCalculator:
    def test_light_robot(self):
        calc = WheelBaseCalculator(payload_kg=1)
        assert calc.wheel_diameter_mm() <= 80

    def test_heavy_robot_larger_wheels(self):
        calc = WheelBaseCalculator(payload_kg=30)
        assert calc.wheel_diameter_mm() >= 100

    def test_mecanum_wider(self):
        diff = WheelBaseCalculator(payload_kg=10, drive_type="differential")
        mec = WheelBaseCalculator(payload_kg=10, drive_type="mecanum")
        assert mec.track_width_mm() >= diff.track_width_mm()

    def test_ground_clearance(self):
        calc = WheelBaseCalculator(payload_kg=20)
        assert calc.ground_clearance_mm() > 0


# ── Templates ──────────────────────────────────────────────────


class TestTemplates:
    def test_differential_4w_exists(self):
        assert "differential_4w" in MOBILE_BASE_TEMPLATES

    def test_differential_2w_exists(self):
        assert "differential_2w" in MOBILE_BASE_TEMPLATES

    def test_mecanum_exists(self):
        assert "mecanum" in MOBILE_BASE_TEMPLATES

    def test_4w_has_parts(self):
        t = MOBILE_BASE_TEMPLATES["differential_4w"]
        assert len(t.parts) >= 5
        assert len(t.joints) >= 4

    def test_template_has_parameters(self):
        for name, t in MOBILE_BASE_TEMPLATES.items():
            assert len(t.parameters) > 0, f"Template {name} has no parameters"


# ── design_mobile_base ─────────────────────────────────────────


class TestDesignMobileBase:
    def test_basic_design(self):
        result = design_mobile_base(payload_kg=5, max_speed_mm_s=300)
        assert "chassis" in result
        assert "motor" in result
        assert "battery" in result
        assert "kinematics" in result

    def test_motor_selected(self):
        result = design_mobile_base(payload_kg=5)
        assert result["motor"]["selected"] != "None found"

    def test_requirements_echoed(self):
        result = design_mobile_base(payload_kg=10, max_speed_mm_s=500, runtime_hours=3)
        assert result["requirements"]["payload_kg"] == 10
        assert result["requirements"]["runtime_hours"] == 3


# ── Tool: mobile_base_design ──────────────────────────────────


class TestMobileBaseDesignTool:
    def test_default_params(self):
        tool = MobileBaseDesignTool()
        result = json.loads(tool.execute())
        assert "chassis" in result
        assert "motor" in result

    def test_custom_params(self):
        tool = MobileBaseDesignTool()
        result = json.loads(tool.execute(
            payload_kg=20, drive_type="mecanum", runtime_hours=4
        ))
        assert result["requirements"]["payload_kg"] == 20
        assert result["requirements"]["drive_type"] == "mecanum"


# ── Tool: differential_drive_sim ──────────────────────────────


class TestDifferentialDriveSimTool:
    def test_straight_line(self):
        tool = DifferentialDriveSimTool()
        result = json.loads(tool.execute(
            v_left_mm_s=100, v_right_mm_s=100, steps=10
        ))
        assert result["v_linear_mm_s"] == 100
        assert result["omega_rad_s"] == 0.0
        # Should move in x direction only
        assert result["final_position"]["x"] > 0
        assert abs(result["final_position"]["y"]) < 0.1

    def test_spin_in_place(self):
        tool = DifferentialDriveSimTool()
        result = json.loads(tool.execute(
            v_left_mm_s=-50, v_right_mm_s=50, steps=20
        ))
        assert abs(result["v_linear_mm_s"]) < 0.01
        assert result["omega_rad_s"] != 0

    def test_curve_trajectory(self):
        tool = DifferentialDriveSimTool()
        result = json.loads(tool.execute(
            v_left_mm_s=80, v_right_mm_s=100, steps=50
        ))
        # Should curve (y changes)
        assert result["final_position"]["y"] != 0


# ── Registration ──────────────────────────────────────────────


class TestRegistration:
    def test_register(self):
        registry = type("MockRegistry", (), {"register": lambda self, t: None})()
        register_mobile_design_tools(registry)

    def test_tool_names(self):
        tools = [MobileBaseDesignTool(), DifferentialDriveSimTool()]
        names = [t.name for t in tools]
        assert "mobile_base_design" in names
        assert "differential_drive_sim" in names


# ── Planner Detection ─────────────────────────────────────────


class TestPlannerMobileBaseDetection:
    @pytest.mark.parametrize("task", [
        "mobile base design",
        "wheel base design",
        "chassis design",
    ])
    def test_mobile_base_detected(self, task):
        assert Planner._detect_task_type(task) == "mobile_base"

    def test_complex_robot_still_highest(self):
        assert Planner._detect_task_type("设计4轮差速底盘移动机器人") == "complex_robot"
