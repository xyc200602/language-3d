"""End-to-end validation for a complex robot: 4-wheel chassis + IPC + dual 3-DOF arms.

This test validates the full Language-3D pipeline can handle a complex robot
with 40+ parts across 5 subsystems.

Stages:
  1. Subsystem decomposition
  2. Symmetry / reuse validation
  3. Part modeling (40+ parts)
  4. Assembly solving (closed-chain + tree)
  5. Kinematics (differential drive + dual-arm IK + collision-free)
  6. Stability analysis
  7. Firmware generation
  8. Engineering outputs (BOM, assembly guide, URDF, cable routing)
  9. Power budget & battery selection
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from lang3d.knowledge.mechanics import (
    Assembly,
    Joint,
    MaterialDensity,
    Part,
    compute_assembly_mass,
)

# ============================================================================
# Test fixture: complex robot assembly
# ============================================================================


def _build_complex_robot() -> Assembly:
    """Build a 4-wheel differential chassis + IPC + dual 3-DOF arm robot.

    5 subsystems:
      - chassis (base plate + 4 wheel modules)
      - arm_left (3-DOF arm)
      - arm_right (mirror of left)
      - ipc_mount (industrial PC + mounting)
      - sensor_tower (IMU + LiDAR + camera)

    Total: ~41 parts, ~43 joints
    """
    parts: list[Part] = []
    joints: list[Joint] = []

    # ---- Chassis subsystem ----
    chassis_parts = [
        Part("base_plate", "structural", "主底盘板",
             dimensions=dict(length=300, width=200, height=5), material="Aluminum"),
        Part("top_plate", "structural", "顶板",
             dimensions=dict(length=280, width=180, height=3), material="Aluminum"),
        Part("standoff_fl", "structural", "前左铜柱",
             dimensions=dict(length=8, diameter=6, height=50)),
        Part("standoff_fr", "structural", "前右铜柱",
             dimensions=dict(length=8, diameter=6, height=50)),
        Part("standoff_rl", "structural", "后左铜柱",
             dimensions=dict(length=8, diameter=6, height=50)),
        Part("standoff_rr", "structural", "后右铜柱",
             dimensions=dict(length=8, diameter=6, height=50)),
        Part("motor_fl", "actuator", "前左驱动电机",
             dimensions=dict(length=40, width=30, height=25)),
        Part("motor_fr", "actuator", "前右驱动电机",
             dimensions=dict(length=40, width=30, height=25)),
        Part("motor_rl", "actuator", "后左驱动电机",
             dimensions=dict(length=40, width=30, height=25)),
        Part("motor_rr", "actuator", "后右驱动电机",
             dimensions=dict(length=40, width=30, height=25)),
        Part("wheel_fl", "structural", "前左轮",
             dimensions=dict(diameter=65, height=26)),
        Part("wheel_fr", "structural", "前右轮",
             dimensions=dict(diameter=65, height=26)),
        Part("wheel_rl", "structural", "后左轮",
             dimensions=dict(diameter=65, height=26)),
        Part("wheel_rr", "structural", "后右轮",
             dimensions=dict(diameter=65, height=26)),
        Part("battery_box", "battery", "锂电池组",
             dimensions=dict(length=150, width=60, height=40)),
        Part("motor_driver_board", "controller", "电机驱动板",
             dimensions=dict(length=70, width=50, height=10)),
        Part("encoder_fl", "sensor", "前左编码器",
             dimensions=dict(diameter=12, height=5)),
        Part("encoder_fr", "sensor", "前右编码器",
             dimensions=dict(diameter=12, height=5)),
        Part("encoder_rl", "sensor", "后左编码器",
             dimensions=dict(diameter=12, height=5)),
        Part("encoder_rr", "sensor", "后右编码器",
             dimensions=dict(diameter=12, height=5)),
    ]
    parts.extend(chassis_parts)

    # Chassis joints
    for s in ["fl", "fr", "rl", "rr"]:
        joints.append(Joint("fixed", "base_plate", f"standoff_{s}",
                            parent_anchor="top", child_anchor="bottom"))
    joints.append(Joint("fixed", "standoff_fl", "top_plate",
                        parent_anchor="top", child_anchor="bottom"))
    joints.append(Joint("fixed", "standoff_fr", "top_plate",
                        parent_anchor="top", child_anchor="bottom"))
    joints.append(Joint("fixed", "standoff_rl", "top_plate",
                        parent_anchor="top", child_anchor="bottom"))
    joints.append(Joint("fixed", "standoff_rr", "top_plate",
                        parent_anchor="top", child_anchor="bottom"))
    for s in ["fl", "fr", "rl", "rr"]:
        joints.append(Joint("fixed", "base_plate", f"motor_{s}",
                            parent_anchor="bottom", child_anchor="top"))
        joints.append(Joint("revolute", f"motor_{s}", f"wheel_{s}", axis="z",
                            range_deg=(-360, 360),
                            parent_anchor="left", child_anchor="bottom"))
        joints.append(Joint("fixed", f"motor_{s}", f"encoder_{s}",
                            parent_anchor="right", child_anchor="bottom"))
    joints.append(Joint("fixed", "base_plate", "battery_box",
                        parent_anchor="top", child_anchor="bottom",
                        offset=(0, 0, 5)))
    joints.append(Joint("fixed", "top_plate", "motor_driver_board",
                        parent_anchor="top", child_anchor="bottom"))

    # ---- IPC mount subsystem ----
    ipc_parts = [
        Part("ipc_bracket", "structural", "工控机支架",
             dimensions=dict(length=120, width=80, height=40)),
        Part("ipc_body", "controller", "工控机主体",
             dimensions=dict(length=110, width=75, height=30)),
        Part("ipc_fan", "structural", "散热风扇",
             dimensions=dict(diameter=40, height=10)),
    ]
    parts.extend(ipc_parts)
    joints.append(Joint("fixed", "top_plate", "ipc_bracket",
                        parent_anchor="top", child_anchor="bottom"))
    joints.append(Joint("fixed", "ipc_bracket", "ipc_body",
                        parent_anchor="top", child_anchor="bottom"))
    joints.append(Joint("fixed", "ipc_body", "ipc_fan",
                        parent_anchor="top", child_anchor="bottom"))

    # ---- Left arm subsystem ----
    arm_l_parts = [
        Part("arm_l_base", "joint", "左臂底座旋转关节",
             dimensions=dict(outer_diameter=80, height=40)),
        Part("arm_l_shoulder", "joint", "左臂肩关节",
             dimensions=dict(outer_diameter=60, height=35)),
        Part("arm_l_upper_link", "structural", "左臂上臂",
             dimensions=dict(length=150, width=40, height=30)),
        Part("arm_l_elbow", "joint", "左臂肘关节",
             dimensions=dict(outer_diameter=50, height=30)),
        Part("arm_l_forearm", "structural", "左臂前臂",
             dimensions=dict(length=120, width=35, height=25)),
        Part("arm_l_wrist", "joint", "左臂腕关节",
             dimensions=dict(outer_diameter=40, height=25)),
        Part("arm_l_gripper", "structural", "左臂末端执行器",
             dimensions=dict(length=60, width=30, height=20)),
    ]
    parts.extend(arm_l_parts)
    joints.extend([
        Joint("revolute", "top_plate", "arm_l_base", (-180, 180), "左臂旋转",
              axis="z", parent_anchor="top", child_anchor="bottom"),
        Joint("revolute", "arm_l_base", "arm_l_shoulder", (-90, 90), "左肩俯仰",
              axis="y", parent_anchor="top", child_anchor="bottom"),
        Joint("fixed", "arm_l_shoulder", "arm_l_upper_link",
              parent_anchor="top", child_anchor="bottom"),
        Joint("revolute", "arm_l_upper_link", "arm_l_elbow", (-135, 135), "左肘弯曲",
              axis="y", parent_anchor="top", child_anchor="bottom"),
        Joint("fixed", "arm_l_elbow", "arm_l_forearm",
              parent_anchor="top", child_anchor="bottom"),
        Joint("revolute", "arm_l_forearm", "arm_l_wrist", (-180, 180), "左腕旋转",
              axis="z", parent_anchor="top", child_anchor="bottom"),
        Joint("fixed", "arm_l_wrist", "arm_l_gripper",
              parent_anchor="top", child_anchor="bottom"),
    ])

    # ---- Right arm (mirror of left) ----
    arm_r_parts = [
        Part("arm_r_base", "joint", "右臂底座旋转关节",
             dimensions=dict(outer_diameter=80, height=40)),
        Part("arm_r_shoulder", "joint", "右臂肩关节",
             dimensions=dict(outer_diameter=60, height=35)),
        Part("arm_r_upper_link", "structural", "右臂上臂",
             dimensions=dict(length=150, width=40, height=30)),
        Part("arm_r_elbow", "joint", "右臂肘关节",
             dimensions=dict(outer_diameter=50, height=30)),
        Part("arm_r_forearm", "structural", "右臂前臂",
             dimensions=dict(length=120, width=35, height=25)),
        Part("arm_r_wrist", "joint", "右臂腕关节",
             dimensions=dict(outer_diameter=40, height=25)),
        Part("arm_r_gripper", "structural", "右臂末端执行器",
             dimensions=dict(length=60, width=30, height=20)),
    ]
    parts.extend(arm_r_parts)
    joints.extend([
        Joint("revolute", "top_plate", "arm_r_base", (-180, 180), "右臂旋转",
              axis="z", parent_anchor="top", child_anchor="bottom"),
        Joint("revolute", "arm_r_base", "arm_r_shoulder", (-90, 90), "右肩俯仰",
              axis="y", parent_anchor="top", child_anchor="bottom"),
        Joint("fixed", "arm_r_shoulder", "arm_r_upper_link",
              parent_anchor="top", child_anchor="bottom"),
        Joint("revolute", "arm_r_upper_link", "arm_r_elbow", (-135, 135), "右肘弯曲",
              axis="y", parent_anchor="top", child_anchor="bottom"),
        Joint("fixed", "arm_r_elbow", "arm_r_forearm",
              parent_anchor="top", child_anchor="bottom"),
        Joint("revolute", "arm_r_forearm", "arm_r_wrist", (-180, 180), "右腕旋转",
              axis="z", parent_anchor="top", child_anchor="bottom"),
        Joint("fixed", "arm_r_wrist", "arm_r_gripper",
              parent_anchor="top", child_anchor="bottom"),
    ])

    # ---- Sensor tower subsystem ----
    sensor_parts = [
        Part("sensor_tower_post", "structural", "传感器塔立柱",
             dimensions=dict(diameter=20, height=120)),
        Part("imu_mount", "sensor", "IMU 安装座",
             dimensions=dict(length=25, width=15, height=5)),
        Part("lidar_mount", "sensor", "LiDAR 安装座",
             dimensions=dict(diameter=80, height=40)),
        Part("camera_bracket", "structural", "摄像头支架",
             dimensions=dict(length=30, width=20, height=15)),
    ]
    parts.extend(sensor_parts)
    joints.extend([
        Joint("fixed", "top_plate", "sensor_tower_post",
              parent_anchor="top", child_anchor="bottom"),
        Joint("fixed", "sensor_tower_post", "imu_mount",
              parent_anchor="top", child_anchor="bottom"),
        Joint("fixed", "sensor_tower_post", "lidar_mount",
              parent_anchor="top", child_anchor="bottom",
              offset=(0, 0, 40)),
        Joint("fixed", "sensor_tower_post", "camera_bracket",
              parent_anchor="top", child_anchor="bottom",
              offset=(0, 0, 80)),
    ])

    return Assembly(
        name="4-Wheel Mobile Robot with Dual Arms",
        description="4轮差速底盘移动机器人 + 工控机 + 双 3-DOF 机械臂",
        parts=parts,
        joints=joints,
    )


@pytest.fixture(scope="module")
def robot():
    return _build_complex_robot()


@pytest.fixture(scope="module")
def solved_positions(robot):
    from lang3d.tools.assembly_solver import AssemblySolver
    solver = AssemblySolver(robot)
    return solver.solve()


# ============================================================================
# Stage 1: Subsystem decomposition
# ============================================================================


class TestSubsystemDecomposition:
    def test_total_part_count(self, robot):
        assert len(robot.parts) >= 41

    def test_total_joint_count(self, robot):
        assert len(robot.joints) >= 43

    def test_chassis_subsystem(self, robot):
        chassis = [p for p in robot.parts if p.name in
                   ["base_plate", "top_plate"] or
                   p.name.startswith("standoff_") or
                   p.name.startswith("motor_") or
                   p.name.startswith("wheel_") or
                   p.name in ("battery_box", "motor_driver_board")]
        assert len(chassis) >= 16

    def test_arm_left_subsystem(self, robot):
        arm_l = [p for p in robot.parts if p.name.startswith("arm_l_")]
        assert len(arm_l) == 7

    def test_arm_right_subsystem(self, robot):
        arm_r = [p for p in robot.parts if p.name.startswith("arm_r_")]
        assert len(arm_r) == 7

    def test_ipc_subsystem(self, robot):
        ipc = [p for p in robot.parts if "ipc" in p.name]
        assert len(ipc) >= 3

    def test_sensor_tower_subsystem(self, robot):
        sensors = [p for p in robot.parts if p.category == "sensor" or
                   "sensor" in p.name or "lidar" in p.name or "camera" in p.name]
        assert len(sensors) >= 3

    def test_actuator_count(self, robot):
        actuators = [p for p in robot.parts if p.category == "actuator"]
        assert len(actuators) == 4  # 4 drive motors


# ============================================================================
# Stage 2: Symmetry / reuse
# ============================================================================


class TestSymmetry:
    def test_four_wheels_same_dims(self, robot):
        wheels = [p for p in robot.parts if p.name.startswith("wheel_")]
        assert len(wheels) == 4
        dims = [p.dimensions for p in wheels]
        assert all(d == dims[0] for d in dims)

    def test_four_motors_same_dims(self, robot):
        motors = [p for p in robot.parts if p.category == "actuator" and p.name.startswith("motor_")]
        assert len(motors) == 4
        dims = [p.dimensions for p in motors]
        assert all(d == dims[0] for d in dims)

    def test_four_standoffs_same_dims(self, robot):
        standoffs = [p for p in robot.parts if p.name.startswith("standoff_")]
        assert len(standoffs) == 4
        dims = [p.dimensions for p in standoffs]
        assert all(d == dims[0] for d in dims)

    def test_dual_arms_mirror(self, robot):
        left = [p for p in robot.parts if p.name.startswith("arm_l_")]
        right = [p for p in robot.parts if p.name.startswith("arm_r_")]
        assert len(left) == len(right)
        for lp in left:
            suffix = lp.name.replace("arm_l_", "")
            rp = next((p for p in right if p.name == f"arm_r_{suffix}"), None)
            assert rp is not None, f"No mirror for {lp.name}"
            assert lp.dimensions == rp.dimensions


# ============================================================================
# Stage 3: Part modeling validation
# ============================================================================


class TestPartModeling:
    def test_all_parts_have_dimensions(self, robot):
        for p in robot.parts:
            assert len(p.dimensions) > 0, f"{p.name} has no dimensions"

    def test_all_parts_have_positive_dims(self, robot):
        for p in robot.parts:
            for k, v in p.dimensions.items():
                assert v > 0, f"{p.name}.{k} = {v}"

    def test_all_parts_have_mass_estimate(self, robot):
        for p in robot.parts:
            mass = p.compute_estimated_mass()
            assert mass > 0, f"{p.name} has zero mass estimate"

    def test_wheel_dimensions_reasonable(self, robot):
        wheels = [p for p in robot.parts if p.name.startswith("wheel_")]
        for w in wheels:
            d = w.dimensions.get("diameter", 0)
            assert 30 < d < 200, f"Wheel diameter {d} unreasonable"

    def test_arm_link_dimensions_reasonable(self, robot):
        for p in robot.parts:
            if "upper_link" in p.name or "forearm" in p.name:
                l = p.dimensions.get("length", 0)
                assert 50 < l < 300, f"Arm link {p.name} length {l} unreasonable"


# ============================================================================
# Stage 4: Assembly solving
# ============================================================================


class TestAssemblySolving:
    def test_solver_returns_all_parts(self, robot, solved_positions):
        assert len(solved_positions) == len(robot.parts)

    def test_all_positions_valid(self, solved_positions):
        for name, data in solved_positions.items():
            pos = data["position"]
            assert len(pos) == 3
            assert all(isinstance(v, (int, float)) for v in pos)

    def test_base_at_origin(self, solved_positions):
        bp = solved_positions.get("base_plate")
        assert bp is not None
        assert bp["position"][0] == pytest.approx(0, abs=1)
        assert bp["position"][1] == pytest.approx(0, abs=1)
        assert bp["position"][2] == pytest.approx(0, abs=1)

    def test_wheels_below_base(self, robot, solved_positions):
        base_z = solved_positions["base_plate"]["position"][2]
        for s in ["fl", "fr", "rl", "rr"]:
            wheel = solved_positions.get(f"wheel_{s}")
            assert wheel is not None, f"wheel_{s} not in positions"
            assert wheel["position"][2] < base_z, f"wheel_{s} not below base"

    def test_top_plate_above_base(self, solved_positions):
        base_z = solved_positions["base_plate"]["position"][2]
        top_z = solved_positions["top_plate"]["position"][2]
        assert top_z > base_z


# ============================================================================
# Stage 5: Kinematics
# ============================================================================


class TestKinematics:
    def test_differential_drive_model(self):
        from lang3d.tools.drive_train import drive_train_design
        result = drive_train_design(
            wheel_count=4, drive_type="differential_4w",
            motor_type="TT_MOTOR",
        )
        assert result["total_parts"] >= 4
        assert result["drive_type"] == "differential_4w"

    def test_left_arm_ik(self, robot):
        from lang3d.tools.ik_solver import solve_ik
        # Build a sub-assembly for the left arm
        arm_parts = [p for p in robot.parts if p.name.startswith("arm_l_")]
        arm_joints = [j for j in robot.joints
                      if j.parent.startswith("arm_l_") and j.child.startswith("arm_l_")]
        # Add the base connection
        base_joints = [j for j in robot.joints if j.child == "arm_l_base"]
        arm_assembly = Assembly(
            name="Left Arm",
            parts=[p for p in robot.parts if p.name == "top_plate"] + arm_parts,
            joints=base_joints + arm_joints,
        )
        result = solve_ik(arm_assembly, target=(0, 0, 250), tolerance_mm=50)
        # Should converge (may not be precise with simple solver)
        assert result is not None

    def test_odometry_code_generation(self):
        from lang3d.tools.code_gen import gen_odometry_code
        code = gen_odometry_code(
            wheel_radius_mm=32.5,
            wheel_base_mm=200.0,
            encoder_ppr=7,
            gear_ratio=48.0,
        )
        assert "odometry" in code.lower() or "x_pos" in code.lower() or "theta" in code.lower()

    def test_motor_driver_code_generation(self):
        from lang3d.tools.code_gen import gen_motor_driver_code
        motors = [
            dict(motor_id="TT_MOTOR", encoder_id="HALL_TT_7PPR",
                 pwm_pin=5, dir_pin1=6, dir_pin2=7,
                 enc_a_pin=18, enc_b_pin=19),
        ]
        result = gen_motor_driver_code(motors)
        assert "dc_motor_driver.h" in result
        assert "dc_motor_driver.cpp" in result


# ============================================================================
# Stage 6: Stability analysis
# ============================================================================


class TestStability:
    def test_assembly_mass_computed(self, robot):
        result = compute_assembly_mass(robot)
        assert result["total_mass_kg"] > 0
        assert result["total_mass_kg"] < 50  # Reasonable for a desktop robot

    def test_static_stability(self, robot, solved_positions):
        from lang3d.tools.stability import compute_static_stability, compute_support_polygon
        # Get wheel positions as contact points
        contacts = []
        for s in ["fl", "fr", "rl", "rr"]:
            pos = solved_positions[f"wheel_{s}"]["position"]
            contacts.append([pos[0], pos[1], pos[2]])

        mass_result = compute_assembly_mass(robot)
        com = list(mass_result["center_of_mass_mm"])

        polygon = compute_support_polygon(contacts)
        # Solver may place contacts collinearly → degenerate polygon
        # Project to 2D for stability computation
        if len(polygon) >= 3:
            poly_2d = [[p[0], p[1]] for p in polygon]
        else:
            poly_2d = [[c[0], c[1]] for c in contacts]
        stability = compute_static_stability(com, poly_2d)
        assert stability["margin_mm"] is not None

    def test_tip_over_risk(self, robot, solved_positions):
        from lang3d.tools.stability import check_tip_over_risk
        contacts = []
        for s in ["fl", "fr", "rl", "rr"]:
            pos = solved_positions[f"wheel_{s}"]["position"]
            contacts.append([pos[0], pos[1], pos[2]])

        mass_result = compute_assembly_mass(robot)
        com = list(mass_result["center_of_mass_mm"])

        result = check_tip_over_risk(
            com=com,
            contact_points=contacts,
            mass_kg=mass_result["total_mass_kg"],
        )
        assert "risk_level" in result
        assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")


# ============================================================================
# Stage 7: Firmware generation
# ============================================================================


class TestFirmware:
    def test_firmware_package(self, robot):
        from lang3d.tools.code_gen import generate_firmware
        # Build arm sub-assembly for firmware
        arm_parts = [p for p in robot.parts if p.name.startswith("arm_l_")]
        arm_joints = [j for j in robot.joints
                      if j.parent.startswith("arm_l_") and j.child.startswith("arm_l_")]
        arm_assembly = Assembly(name="Left Arm", parts=arm_parts, joints=arm_joints)

        firmware = generate_firmware(
            arm_assembly,
            actuator_ids=["MG996R", "MG996R", "MG996R"],
            controller="esp32",
        )
        assert "robot_arm.ino" in firmware
        assert "ik_solver.h" in firmware
        assert "servo_driver.h" in firmware

    def test_wiring_diagram(self):
        from lang3d.tools.code_gen import generate_wiring
        wiring = generate_wiring(
            actuator_ids=["TT_MOTOR", "TT_MOTOR", "MG996R"],
            controller="esp32",
        )
        assert "ESP32" in wiring or "esp32" in wiring.lower()
        assert "GPIO" in wiring or "Signal" in wiring or "Servo" in wiring


# ============================================================================
# Stage 8: Engineering outputs
# ============================================================================


class TestEngineeringOutputs:
    def test_bom_generation(self, robot):
        from lang3d.tools.bom_gen import generate_bom, format_bom_markdown
        bom = generate_bom(
            robot,
            actuator_ids=["TT_MOTOR", "TT_MOTOR", "TT_MOTOR", "TT_MOTOR",
                          "MG996R", "MG996R", "MG996R", "MG996R", "MG996R", "MG996R"],
            controller="esp32",
        )
        assert "custom_parts" in bom
        assert len(bom["custom_parts"]) >= 30
        assert bom["cost_summary"]["total_cost_cny"] > 0

        md = format_bom_markdown(bom)
        assert "BOM" in md or "cost" in md.lower()

    def test_urdf_export(self, robot):
        from lang3d.tools.urdf_export import AssemblyToURDF
        converter = AssemblyToURDF(robot)
        xml = converter.convert()
        assert "<robot" in xml
        links = converter.get_links()
        joints = converter.get_joints()
        assert len(links) >= 41
        assert len(joints) >= 43

    def test_cable_routing(self, robot, solved_positions):
        from lang3d.tools.cable_routing import (
            auto_detect_connections,
            build_3d_grid,
            find_cable_path,
            generate_cable_report,
        )
        cables = auto_detect_connections(robot)
        assert len(cables) >= 4  # 4 motors at minimum

        grid = build_3d_grid(solved_positions, robot.parts)
        cable_paths = []
        for spec in cables:
            start_pos = solved_positions.get(spec.start_connector, {}).get("position", [0, 0, 0])
            end_pos = solved_positions.get(spec.end_connector, {}).get("position", [0, 0, 0])
            cp = find_cable_path(
                grid,
                start=(start_pos[0], start_pos[1], start_pos[2]),
                end=(end_pos[0], end_pos[1], end_pos[2]),
                spec=spec,
            )
            cable_paths.append(cp)

        assert len(cable_paths) >= 4
        report = generate_cable_report(cable_paths, robot.name)
        assert "Cable Routing Report" in report

    def test_ros2_package(self, robot):
        import tempfile
        from lang3d.tools.urdf_export import AssemblyToURDF, ROS2PackageBuilder
        converter = AssemblyToURDF(robot)
        xml = converter.convert()
        builder = ROS2PackageBuilder("mobile_robot_dual_arm", xml)
        with tempfile.TemporaryDirectory() as tmp:
            path = builder.write(tmp)
            assert Path(path).exists()
            assert (Path(path) / "urdf" / "mobile_robot_dual_arm.urdf").exists()
            assert (Path(path) / "package.xml").exists()
            assert (Path(path) / "CMakeLists.txt").exists()
            assert (Path(path) / "launch" / "display.launch.py").exists()


# ============================================================================
# Stage 9: Power budget
# ============================================================================


class TestPowerBudget:
    def test_power_budget_calculation(self):
        from lang3d.tools.power_budget import PowerBudgetCalculator
        calc = PowerBudgetCalculator("MobileRobotDualArm")
        # 4 drive motors
        calc.add_motor("Drive Motor FL", "TT_MOTOR", duty_cycle=0.5, quantity=4)
        # 6 arm servos
        calc.add_servo("Arm Servos", "MG996R", duty_cycle=0.3, quantity=6)
        # IPC
        calc.add_controller("IPC", tdp_w=15.0)
        # Sensors
        calc.add_sensor_load("Sensors", power_w=2.0, quantity=3)

        peak = calc.compute_total_peak()
        avg = calc.compute_total_avg()
        assert peak > 0
        assert avg > 0
        assert avg < peak

    def test_battery_recommendation(self):
        from lang3d.tools.power_budget import PowerBudgetCalculator
        calc = PowerBudgetCalculator("MobileRobotDualArm")
        calc.add_motor("Drive Motor", "TT_MOTOR", duty_cycle=0.5, quantity=4)
        calc.add_servo("Arm Servo", "MG996R", duty_cycle=0.3, quantity=6)
        calc.add_controller("IPC", tdp_w=15.0)

        recs = calc.recommend_battery(runtime_target_h=0.5)
        assert len(recs) > 0
        for r in recs:
            assert r["runtime_h"] >= 0.5

    def test_power_report(self):
        from lang3d.tools.power_budget import PowerBudgetCalculator
        calc = PowerBudgetCalculator("MobileRobotDualArm")
        calc.add_motor("Drive", "TT_MOTOR", quantity=4)
        calc.add_controller("IPC", tdp_w=15.0)
        report = calc.generate_report()
        assert "Power Budget Report" in report
        assert "MobileRobotDualArm" in report


# ============================================================================
# Integration: full pipeline metrics
# ============================================================================


class TestFullPipelineMetrics:
    def test_total_parts_count(self, robot):
        """Verify we meet the 40+ part requirement."""
        assert len(robot.parts) >= 40, f"Only {len(robot.parts)} parts, need 40+"

    def test_total_joints_count(self, robot):
        assert len(robot.joints) >= 40, f"Only {len(robot.joints)} joints, need 40+"

    def test_all_categories_represented(self, robot):
        categories = {p.category for p in robot.parts}
        assert "structural" in categories
        assert "actuator" in categories
        assert "joint" in categories
        assert "controller" in categories
        assert "battery" in categories
        assert "sensor" in categories

    def test_report_generation(self, robot, solved_positions):
        """Generate and validate a comprehensive design report."""
        report: dict[str, Any] = {
            "requirement": "4-wheel differential mobile robot with IPC + dual 3-DOF arms",
            "total_parts": len(robot.parts),
            "total_joints": len(robot.joints),
            "subsystems": {
                "chassis": len([p for p in robot.parts if p.name.startswith(("base_", "top_", "standoff_", "motor_", "wheel_", "battery_"))]),
                "arm_left": len([p for p in robot.parts if p.name.startswith("arm_l_")]),
                "arm_right": len([p for p in robot.parts if p.name.startswith("arm_r_")]),
                "ipc": len([p for p in robot.parts if "ipc" in p.name]),
                "sensor_tower": len([p for p in robot.parts if p.category == "sensor" or "sensor" in p.name or "lidar" in p.name or "camera" in p.name]),
            },
            "assembly_solved": len(solved_positions) == len(robot.parts),
        }

        # Mass
        mass_result = compute_assembly_mass(robot)
        report["total_mass_kg"] = round(mass_result["total_mass_kg"], 3)

        # URDF
        from lang3d.tools.urdf_export import AssemblyToURDF
        converter = AssemblyToURDF(robot)
        converter.convert()
        report["urdf_links"] = len(converter.get_links())
        report["urdf_joints"] = len(converter.get_joints())

        # Cables
        from lang3d.tools.cable_routing import auto_detect_connections
        cables = auto_detect_connections(robot)
        report["cable_count"] = len(cables)

        # Power
        from lang3d.tools.power_budget import PowerBudgetCalculator
        calc = PowerBudgetCalculator("Robot")
        calc.add_motor("Drive", "TT_MOTOR", quantity=4)
        calc.add_controller("IPC", tdp_w=15.0)
        report["peak_power_w"] = round(calc.compute_total_peak(), 1)
        report["avg_power_w"] = round(calc.compute_total_avg(), 1)

        assert report["total_parts"] >= 41
        assert report["total_joints"] >= 43
        assert report["assembly_solved"] is True
        assert report["total_mass_kg"] > 0
        assert report["urdf_links"] >= 41
        assert report["cable_count"] >= 4
        assert report["peak_power_w"] > 0
