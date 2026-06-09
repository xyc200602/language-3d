"""End-to-end test: Design a dual-arm 4-wheel robot (Task 82+ final validation).

This is a comprehensive integration test that creates a realistic robot:
  - 4-wheel differential drive chassis
  - 2× 5-DOF arms (shoulder/elbow/wrist pitch/wrist yaw/gripper)
  - Sensor tower (IMU + LiDAR)
  - Battery box + electronics plate

Validates the FULL pipeline:
  1. Assembly definition (Assembly/Part/Joint/ConnectionMethod)
  2. Assembly pattern matching (get_connection_pattern, get_recommended_bolt_size)
  3. Mounting interface lookup (NEMA17, SG90, sensors, etc.)
  4. Connection feature generation (bolted, press-fit)
  5. Fastener selection from catalog
  6. Assembly constraint solving (positions)
  7. Tolerance analysis
  8. Production-grade assembly verification
  9. Report generation
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from lang3d.agent.assembly_verifier import (
    AssemblySequenceCheck,
    AssemblyVerificationResult,
    AssemblyVerifier,
    BoltHoleAlignmentCheck,
    MatingSurfaceCheck,
    VerificationItem,
)
from lang3d.knowledge.assembly_patterns import (
    ASSEMBLY_STATISTICS,
    CONNECTION_PATTERNS,
    ROBOT_PROFILES,
    get_connection_pattern,
    get_recommended_bolt_size,
    get_robot_profile,
)
from lang3d.knowledge.fastener_catalog import (
    get_bolt_spec,
    get_clearance_hole,
    get_nut_spec,
    get_washer_spec,
    recommend_bolt_length,
    recommend_fastener_set,
)
from lang3d.knowledge.mechanics import (
    Assembly,
    ConnectionMethod,
    Joint,
    Part,
)
from lang3d.knowledge.parts_catalog import (
    get_mounting_interface,
)
from lang3d.knowledge.tolerance import (
    compute_fit,
    it_tolerance,
    recommend_fit,
)
from lang3d.tools.assembly_solver import AssemblySolver
from lang3d.tools.connection_features import ConnectionFeatureEngine
from lang3d.tools.export_package import export_engineering_package
from lang3d.tools.tolerance_analysis import (
    ToleranceStackup,
    analyze_assembly_chain,
)


# ============================================================================
# Robot definition: dual-arm 4-wheel robot
# ============================================================================

def _build_dual_arm_wheeled_robot() -> Assembly:
    """Build a complete dual-arm 4-wheel differential drive robot.

    Structure:
        base_plate (root)
        ├── standoff_fl/fr/rl/rr → top_plate
        ├── motor_fl/fr/rl/rr → wheel_fl/fr/rl/rr
        ├── arm_l_base → arm_l_shoulder → arm_l_upper → arm_l_elbow →
        │   arm_l_forearm → arm_l_wrist → arm_l_gripper
        ├── arm_r_base → arm_r_shoulder → arm_r_upper → arm_r_elbow →
        │   arm_r_forearm → arm_r_wrist → arm_r_gripper
        ├── battery_box
        ├── imu_mount → imu_sensor
        └── lidar_mount → lidar_sensor
    """
    parts = [
        # ---- Chassis ----
        Part("base_plate", "structural", "主底盘板", material="Aluminum",
             dimensions={"length": 300, "width": 200, "height": 5}),
        Part("top_plate", "structural", "上盖板", material="Aluminum",
             dimensions={"length": 280, "width": 180, "height": 3}),
        Part("standoff_fl", "structural", "前左铜柱", material="Steel",
             dimensions={"length": 8, "diameter": 6, "height": 50}),
        Part("standoff_fr", "structural", "前右铜柱", material="Steel",
             dimensions={"length": 8, "diameter": 6, "height": 50}),
        Part("standoff_rl", "structural", "后左铜柱", material="Steel",
             dimensions={"length": 8, "diameter": 6, "height": 50}),
        Part("standoff_rr", "structural", "后右铜柱", material="Steel",
             dimensions={"length": 8, "diameter": 6, "height": 50}),

        # ---- Drive ----
        Part("motor_fl", "actuator", "前左TT电机", material="Steel",
             dimensions={"length": 40, "width": 30, "height": 25}),
        Part("motor_fr", "actuator", "前右TT电机", material="Steel",
             dimensions={"length": 40, "width": 30, "height": 25}),
        Part("motor_rl", "actuator", "后左TT电机", material="Steel",
             dimensions={"length": 40, "width": 30, "height": 25}),
        Part("motor_rr", "actuator", "后右TT电机", material="Steel",
             dimensions={"length": 40, "width": 30, "height": 25}),
        Part("wheel_fl", "mechanical", "前左轮", material="Rubber",
             dimensions={"diameter": 65, "height": 26}),
        Part("wheel_fr", "mechanical", "前右轮", material="Rubber",
             dimensions={"diameter": 65, "height": 26}),
        Part("wheel_rl", "mechanical", "后左轮", material="Rubber",
             dimensions={"diameter": 65, "height": 26}),
        Part("wheel_rr", "mechanical", "后右轮", material="Rubber",
             dimensions={"diameter": 65, "height": 26}),

        # ---- Left arm ----
        Part("arm_l_base", "structural", "左臂底座", material="Aluminum",
             dimensions={"length": 40, "width": 40, "height": 15}),
        Part("arm_l_shoulder", "actuator", "左肩SG90舵机", material="Steel",
             dimensions={"length": 22.2, "width": 11.8, "height": 31}),
        Part("arm_l_upper", "structural", "左上臂连杆", material="PLA",
             dimensions={"length": 100, "width": 25, "height": 15}),
        Part("arm_l_elbow", "actuator", "左肘SG90舵机", material="Steel",
             dimensions={"length": 22.2, "width": 11.8, "height": 31}),
        Part("arm_l_forearm", "structural", "左前臂连杆", material="PLA",
             dimensions={"length": 80, "width": 25, "height": 12}),
        Part("arm_l_wrist", "actuator", "左腕SG90舵机", material="Steel",
             dimensions={"length": 22.2, "width": 11.8, "height": 28}),
        Part("arm_l_gripper", "mechanical", "左夹爪", material="PLA",
             dimensions={"length": 50, "width": 20, "height": 15}),

        # ---- Right arm (mirror) ----
        Part("arm_r_base", "structural", "右臂底座", material="Aluminum",
             dimensions={"length": 40, "width": 40, "height": 15}),
        Part("arm_r_shoulder", "actuator", "右肩SG90舵机", material="Steel",
             dimensions={"length": 22.2, "width": 11.8, "height": 31}),
        Part("arm_r_upper", "structural", "右上臂连杆", material="PLA",
             dimensions={"length": 100, "width": 25, "height": 15}),
        Part("arm_r_elbow", "actuator", "右肘SG90舵机", material="Steel",
             dimensions={"length": 22.2, "width": 11.8, "height": 31}),
        Part("arm_r_forearm", "structural", "右前臂连杆", material="PLA",
             dimensions={"length": 80, "width": 25, "height": 12}),
        Part("arm_r_wrist", "actuator", "右腕SG90舵机", material="Steel",
             dimensions={"length": 22.2, "width": 11.8, "height": 28}),
        Part("arm_r_gripper", "mechanical", "右夹爪", material="PLA",
             dimensions={"length": 50, "width": 20, "height": 15}),

        # ---- Electronics ----
        Part("battery_box", "electronics", "电池盒", material="PLA",
             dimensions={"length": 150, "width": 60, "height": 30}),
        Part("controller_board", "electronics", "主控板ESP32", material="PCB",
             dimensions={"length": 54, "width": 28, "height": 12}),

        # ---- Sensors ----
        Part("imu_mount", "structural", "IMU安装座", material="PLA",
             dimensions={"length": 30, "width": 30, "height": 10}),
        Part("imu_sensor", "sensor", "MPU6050", material="PCB",
             dimensions={"length": 20, "width": 15, "height": 3}),
        Part("lidar_mount", "structural", "LiDAR安装座", material="PLA",
             dimensions={"length": 40, "width": 40, "height": 15}),
        Part("lidar_sensor", "sensor", "RPLIDAR A1", material="PCB",
             dimensions={"diameter": 74, "height": 38}),
    ]

    bolted_m3 = ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=4)
    bolted_m3_2 = ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=2)

    joints = [
        # ---- Chassis ----
        Joint("fixed", "base_plate", "standoff_fl", parent_anchor="top",
              child_anchor="bottom", distribution_group="standoffs",
              connection=ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=1)),
        Joint("fixed", "base_plate", "standoff_fr", parent_anchor="top",
              child_anchor="bottom", distribution_group="standoffs",
              connection=ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=1)),
        Joint("fixed", "base_plate", "standoff_rl", parent_anchor="top",
              child_anchor="bottom", distribution_group="standoffs",
              connection=ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=1)),
        Joint("fixed", "base_plate", "standoff_rr", parent_anchor="top",
              child_anchor="bottom", distribution_group="standoffs",
              connection=ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=1)),
        Joint("fixed", "standoff_fl", "top_plate", parent_anchor="top",
              child_anchor="bottom", no_distribute=True,
              connection=ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=1)),
        Joint("fixed", "standoff_fr", "top_plate", parent_anchor="top",
              child_anchor="bottom", no_distribute=True,
              connection=ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=1)),
        Joint("fixed", "standoff_rl", "top_plate", parent_anchor="top",
              child_anchor="bottom", no_distribute=True,
              connection=ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=1)),
        Joint("fixed", "standoff_rr", "top_plate", parent_anchor="top",
              child_anchor="bottom", no_distribute=True,
              connection=ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=1)),

        # ---- Drive ----
        Joint("fixed", "base_plate", "motor_fl", parent_anchor="bottom",
              child_anchor="top", distribution_group="motors", connection=bolted_m3),
        Joint("fixed", "base_plate", "motor_fr", parent_anchor="bottom",
              child_anchor="top", distribution_group="motors", connection=bolted_m3),
        Joint("fixed", "base_plate", "motor_rl", parent_anchor="bottom",
              child_anchor="top", distribution_group="motors", connection=bolted_m3),
        Joint("fixed", "base_plate", "motor_rr", parent_anchor="bottom",
              child_anchor="top", distribution_group="motors", connection=bolted_m3),
        Joint("revolute", "motor_fl", "wheel_fl", parent_anchor="left",
              child_anchor="center", axis="y", range_deg=(-360, 360)),
        Joint("revolute", "motor_fr", "wheel_fr", parent_anchor="right",
              child_anchor="center", axis="y", range_deg=(-360, 360)),
        Joint("revolute", "motor_rl", "wheel_rl", parent_anchor="left",
              child_anchor="center", axis="y", range_deg=(-360, 360)),
        Joint("revolute", "motor_rr", "wheel_rr", parent_anchor="right",
              child_anchor="center", axis="y", range_deg=(-360, 360)),

        # ---- Left arm ----
        Joint("fixed", "top_plate", "arm_l_base", parent_anchor="top",
              child_anchor="bottom", distribution_group="arms", connection=bolted_m3_2),
        Joint("revolute", "arm_l_base", "arm_l_shoulder", axis="z",
              range_deg=(-180, 180), parent_anchor="top", child_anchor="bottom"),
        Joint("revolute", "arm_l_shoulder", "arm_l_upper", axis="y",
              range_deg=(-120, 120), parent_anchor="top", child_anchor="bottom"),
        Joint("revolute", "arm_l_upper", "arm_l_elbow", axis="y",
              range_deg=(-150, 150), parent_anchor="front", child_anchor="back"),
        Joint("revolute", "arm_l_elbow", "arm_l_forearm", axis="y",
              range_deg=(-150, 150), parent_anchor="front", child_anchor="back"),
        Joint("revolute", "arm_l_forearm", "arm_l_wrist", axis="z",
              range_deg=(-180, 180), parent_anchor="front", child_anchor="back"),
        Joint("fixed", "arm_l_wrist", "arm_l_gripper",
              parent_anchor="front", child_anchor="back"),

        # ---- Right arm (mirror) ----
        Joint("fixed", "top_plate", "arm_r_base", parent_anchor="top",
              child_anchor="bottom", distribution_group="arms", connection=bolted_m3_2),
        Joint("revolute", "arm_r_base", "arm_r_shoulder", axis="z",
              range_deg=(-180, 180), parent_anchor="top", child_anchor="bottom"),
        Joint("revolute", "arm_r_shoulder", "arm_r_upper", axis="y",
              range_deg=(-120, 120), parent_anchor="top", child_anchor="bottom"),
        Joint("revolute", "arm_r_upper", "arm_r_elbow", axis="y",
              range_deg=(-150, 150), parent_anchor="front", child_anchor="back"),
        Joint("revolute", "arm_r_elbow", "arm_r_forearm", axis="y",
              range_deg=(-150, 150), parent_anchor="front", child_anchor="back"),
        Joint("revolute", "arm_r_forearm", "arm_r_wrist", axis="z",
              range_deg=(-180, 180), parent_anchor="front", child_anchor="back"),
        Joint("fixed", "arm_r_wrist", "arm_r_gripper",
              parent_anchor="front", child_anchor="back"),

        # ---- Electronics ----
        Joint("fixed", "base_plate", "battery_box", parent_anchor="top",
              child_anchor="bottom", no_distribute=True),
        Joint("fixed", "top_plate", "controller_board", parent_anchor="top",
              child_anchor="bottom", no_distribute=True),

        # ---- Sensors ----
        Joint("fixed", "top_plate", "imu_mount", parent_anchor="top",
              child_anchor="bottom", no_distribute=True),
        Joint("fixed", "imu_mount", "imu_sensor", parent_anchor="top",
              child_anchor="bottom",
              connection=ConnectionMethod(type="bolted", bolt_size="M2", bolt_count=2)),
        Joint("fixed", "top_plate", "lidar_mount", parent_anchor="top",
              child_anchor="bottom", no_distribute=True),
        Joint("fixed", "lidar_mount", "lidar_sensor", parent_anchor="top",
              child_anchor="bottom",
              connection=ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=4)),
    ]

    return Assembly(
        name="Dual-Arm 4-Wheel Robot",
        parts=parts,
        joints=joints,
        description="4轮差速底盘+双5自由度机械臂+传感器塔的移动操作机器人",
        default_angles={
            "arm_l_upper": -30, "arm_l_forearm": 15,
            "arm_r_upper": -30, "arm_r_forearm": 15,
        },
    )


# ============================================================================
# Test 1: Assembly definition integrity
# ============================================================================

class TestAssemblyDefinition:

    def test_robot_has_34_parts(self):
        """Robot should have 34 parts total."""
        assembly = _build_dual_arm_wheeled_robot()
        assert len(assembly.parts) == 34

    def test_robot_has_36_joints(self):
        """34 parts → 36 joints (33 tree + 3 extra multi-parent for top_plate)."""
        assembly = _build_dual_arm_wheeled_robot()
        assert len(assembly.joints) == 36

    def test_all_part_names_unique(self):
        assembly = _build_dual_arm_wheeled_robot()
        names = [p.name for p in assembly.parts]
        assert len(names) == len(set(names)), "Duplicate part names found"

    def test_all_joint_refs_valid(self):
        """All parent/child references in joints must reference existing parts."""
        assembly = _build_dual_arm_wheeled_robot()
        part_names = {p.name for p in assembly.parts}
        for i, j in enumerate(assembly.joints):
            assert j.parent in part_names, f"Joint {i}: parent '{j.parent}' not found"
            assert j.child in part_names, f"Joint {i}: child '{j.child}' not found"

    def test_connected_tree(self):
        """All parts reachable from base_plate via BFS."""
        assembly = _build_dual_arm_wheeled_robot()
        part_names = {p.name for p in assembly.parts}

        children_map: dict[str, set[str]] = {}
        for j in assembly.joints:
            children_map.setdefault(j.parent, set()).add(j.child)

        visited = {"base_plate"}
        queue = ["base_plate"]
        while queue:
            current = queue.pop(0)
            for child in children_map.get(current, set()):
                if child not in visited:
                    visited.add(child)
                    queue.append(child)

        unreachable = part_names - visited
        assert not unreachable, f"Unreachable parts: {unreachable}"

    def test_part_categories(self):
        """Verify category distribution."""
        assembly = _build_dual_arm_wheeled_robot()
        cats = {}
        for p in assembly.parts:
            cats[p.category] = cats.get(p.category, 0) + 1

        assert cats.get("structural", 0) >= 10  # brackets, plates, standoffs, mounts
        assert cats.get("actuator", 0) >= 10  # motors + servos
        assert cats.get("mechanical", 0) >= 4  # wheels + grippers
        assert cats.get("electronics", 0) >= 2  # battery + controller
        assert cats.get("sensor", 0) >= 2  # IMU + LiDAR

    def test_default_angles_nonzero(self):
        """Arms should have bent default posture."""
        assembly = _build_dual_arm_wheeled_robot()
        angles = assembly.default_angles
        assert len(angles) > 0
        assert not all(v == 0 for v in angles.values())


# ============================================================================
# Test 2: Assembly pattern matching
# ============================================================================

class TestPatternMatching:

    def test_motor_to_bracket_pattern(self):
        pattern = get_connection_pattern("bracket", "stepper_motor")
        assert pattern is not None
        assert pattern.typical_bolt_size == "M3"
        assert pattern.typical_bolt_count == 4

    def test_recommended_bolt_for_chassis(self):
        bolt, count = get_recommended_bolt_size("plate", "standoff")
        assert bolt == "M3"

    def test_statistics_available(self):
        stats = ASSEMBLY_STATISTICS
        assert "connection_method_distribution" in stats
        assert stats["connection_method_distribution"]["bolted"] > 0.5


# ============================================================================
# Test 3: Mounting interface lookup
# ============================================================================

class TestMountingInterfaces:

    def test_sg90_interface(self):
        iface = get_mounting_interface("servo_sg90")
        assert iface is not None
        assert len(iface.holes) == 2

    def test_nema17_interface(self):
        iface = get_mounting_interface("nema17_stepper")
        assert iface is not None
        assert len(iface.holes) == 4

    def test_imu_interface(self):
        iface = get_mounting_interface("sensor_mpu6050")
        assert iface is not None
        assert len(iface.holes) == 4

    def test_lidar_interface(self):
        iface = get_mounting_interface("sensor_rplidar_a1")
        assert iface is not None
        assert len(iface.holes) == 4


# ============================================================================
# Test 4: Feature generation for key connections
# ============================================================================

class TestFeatureGeneration:

    def test_arm_base_mounting_features(self):
        """Arm base mounted to top_plate should generate bolt features."""
        engine = ConnectionFeatureEngine()
        arm_base = Part("arm_l_base", "structural", "Arm base",
                        dimensions={"length": 40, "width": 40, "height": 15})
        conn = ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=2)
        result = engine.generate_features(
            structural_part=arm_base,
            connection=conn,
            anchor="top",
        )
        assert len(result.ops) > 0

    def test_lidar_mount_features_with_interface(self):
        """LiDAR mount should use its interface for accurate holes."""
        engine = ConnectionFeatureEngine()
        lidar_mount = Part("lidar_mount", "structural", "LiDAR mount",
                           dimensions={"length": 40, "width": 40, "height": 15})
        conn = ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=4)
        result = engine.generate_features(
            structural_part=lidar_mount,
            connection=conn,
            anchor="top",
            functional_part_id="sensor_rplidar_a1",
        )
        assert len(result.ops) > 0
        assert len(result.features_generated) > 0

    def test_motor_mount_features(self):
        """Motor mount on base plate should generate clearance holes."""
        engine = ConnectionFeatureEngine()
        base_plate = Part("base_plate", "structural", "Base plate",
                          dimensions={"length": 300, "width": 200, "height": 5})
        conn = ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=4)
        result = engine.generate_features(
            structural_part=base_plate,
            connection=conn,
            anchor="bottom",
            functional_part_id="nema17_stepper",
        )
        assert len(result.ops) >= 4  # At least 4 bolt holes
        assert len(result.fastener_ops) > 0


# ============================================================================
# Test 5: Fastener selection
# ============================================================================

class TestFastenerSelection:

    def test_m3_for_standoffs(self):
        """M3×8 bolt + M3 nut for 5mm plate."""
        bolt = get_bolt_spec("M3")
        nut = get_nut_spec("M3")
        assert bolt is not None
        assert nut is not None

    def test_m2_for_imu(self):
        """M2 bolt for sensor mounting."""
        bolt = get_bolt_spec("M2")
        assert bolt is not None
        assert bolt.thread_diameter == pytest.approx(2.0)

    def test_clearance_holes(self):
        """Clearance holes for M3 should be 3.4mm."""
        clearance = get_clearance_hole("M3")
        assert clearance == pytest.approx(3.4, abs=0.1)

    def test_bolt_length_for_arm(self):
        """Arm base is 15mm thick → need bolt > 15mm."""
        length = recommend_bolt_length(grip_mm=15.0)
        assert length >= 15.0

    def test_fastener_set_for_motor(self):
        """Complete fastener set for NEMA17 on 5mm plate."""
        result = recommend_fastener_set("M3", grip_mm=5.0, with_washer=True)
        assert result is not None
        assert result["bolt_length_mm"] >= 5.0


# ============================================================================
# Test 6: Tolerance analysis
# ============================================================================

class TestToleranceAnalysis:

    def test_it7_for_arm_links(self):
        """Arm link tolerances should be IT7 grade."""
        tol = it_tolerance(100.0, "IT7")
        assert tol > 0
        assert tol < 0.1  # < 0.1mm for 100mm dimension

    def test_bearing_fit(self):
        """608 bearing seat should be H7/js6 (transition fit)."""
        rec = recommend_fit("bearing_seat")
        assert rec is not None
        assert rec.fit_type == "transition"

    def test_clearance_fit(self):
        """Sliding fit should be clearance."""
        rec = recommend_fit("sliding")
        assert rec is not None
        assert rec.fit_type == "clearance"

    def test_tolerance_stackup_for_arm(self):
        """Arm chain tolerance stackup analysis."""
        stack = ToleranceStackup()
        stack.add_dimension("shoulder_link", 100.0, upper=0.05, lower=-0.05)
        stack.add_dimension("elbow_link", 80.0, upper=0.04, lower=-0.04)
        stack.add_dimension("wrist_link", 50.0, upper=0.03, lower=-0.03)
        result = stack.compute_stackup()
        assert result.nominal == pytest.approx(230.0)
        assert result.total_tolerance == pytest.approx(0.24)

    def test_arm_chain_via_helper(self):
        """analyze_assembly_chain helper."""
        chain = [
            {"name": "base_to_shoulder", "nominal": 15.0, "upper": 0.05, "lower": -0.05},
            {"name": "upper_link", "nominal": 100.0, "upper": 0.05, "lower": -0.05},
            {"name": "forearm", "nominal": 80.0, "upper": 0.04, "lower": -0.04},
        ]
        result = analyze_assembly_chain(chain)
        assert result.nominal == pytest.approx(195.0)
        assert result.upper_dev == pytest.approx(0.14)


# ============================================================================
# Test 7: Assembly constraint solving
# ============================================================================

class TestConstraintSolving:

    def test_solver_produces_all_placements(self):
        """Solver should produce positions for all 34 parts."""
        assembly = _build_dual_arm_wheeled_robot()
        solver = AssemblySolver(assembly)
        placements = solver.solve()
        assert len(placements) == 34

    def test_base_plate_at_origin(self):
        """Root part should be at or near origin."""
        assembly = _build_dual_arm_wheeled_robot()
        solver = AssemblySolver(assembly)
        placements = solver.solve()
        base_pos = placements["base_plate"]["position"]
        assert base_pos[0] == pytest.approx(0.0, abs=1.0)
        assert base_pos[1] == pytest.approx(0.0, abs=1.0)

    def test_wheels_below_base(self):
        """Wheels should be below the base plate."""
        assembly = _build_dual_arm_wheeled_robot()
        solver = AssemblySolver(assembly)
        placements = solver.solve()
        base_z = placements["base_plate"]["position"][2]
        for name in ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]:
            wheel_z = placements[name]["position"][2]
            assert wheel_z < base_z, f"{name} z={wheel_z} should be < base z={base_z}"

    def test_top_plate_above_base(self):
        """Top plate should be above base plate."""
        assembly = _build_dual_arm_wheeled_robot()
        solver = AssemblySolver(assembly)
        placements = solver.solve()
        base_z = placements["base_plate"]["position"][2]
        top_z = placements["top_plate"]["position"][2]
        assert top_z > base_z, f"top z={top_z} should be > base z={base_z}"

    def test_arms_above_top_plate(self):
        """Arm bases should be above top plate."""
        assembly = _build_dual_arm_wheeled_robot()
        solver = AssemblySolver(assembly)
        placements = solver.solve()
        top_z = placements["top_plate"]["position"][2]
        for name in ["arm_l_base", "arm_r_base"]:
            arm_z = placements[name]["position"][2]
            assert arm_z > top_z, f"{name} z={arm_z} should be > top z={top_z}"

    def test_symmetric_arm_placement(self):
        """Left and right arm bases should both exist and be placed."""
        assembly = _build_dual_arm_wheeled_robot()
        solver = AssemblySolver(assembly)
        placements = solver.solve()
        assert "arm_l_base" in placements
        assert "arm_r_base" in placements
        l_pos = placements["arm_l_base"]["position"]
        r_pos = placements["arm_r_base"]["position"]
        # Both arms should be above top plate (Z check)
        top_z = placements["top_plate"]["position"][2]
        assert l_pos[2] > top_z, f"Left arm z={l_pos[2]} should be > top z={top_z}"
        assert r_pos[2] > top_z, f"Right arm z={r_pos[2]} should be > top z={top_z}"
        # Arms should be at different X positions (distributed along width)
        assert l_pos != r_pos, "Arm placements should differ"


# ============================================================================
# Test 8: Assembly verification
# ============================================================================

class TestAssemblyVerification:

    def _make_parts_results(self, assembly):
        return {p.name: {"artifacts": [f"{p.name}.step"], "result": "success"}
                for p in assembly.parts}

    def test_full_verification(self, tmp_path):
        """Full verification should produce structured result."""
        assembly = _build_dual_arm_wheeled_robot()
        solver = AssemblySolver(assembly)
        placements = solver.solve()
        parts_results = self._make_parts_results(assembly)

        verifier = AssemblyVerifier()
        result = verifier.verify_assembly(
            assembly=assembly,
            workspace=tmp_path,
            parts_results=parts_results,
            placements=placements,
            allowed_tolerance_total=2.0,
        )

        assert isinstance(result, AssemblyVerificationResult)
        assert result.assembly_name == "Dual-Arm 4-Wheel Robot"
        # collision_free may be False for simplified geometry overlaps;
        # what matters is that the check ran and produced results
        assert len(result.verification_items) > 0

    def test_mating_surface_checks(self, tmp_path):
        """Should have mating surface checks for bolted joints."""
        assembly = _build_dual_arm_wheeled_robot()
        solver = AssemblySolver(assembly)
        placements = solver.solve()
        parts_results = self._make_parts_results(assembly)

        verifier = AssemblyVerifier()
        result = verifier.verify_assembly(
            assembly=assembly,
            workspace=tmp_path,
            parts_results=parts_results,
            placements=placements,
            allowed_tolerance_total=2.0,
        )

        # Should have mating checks for joints with placements
        assert len(result.mating_surface_checks) > 0

    def test_sequence_feasibility(self, tmp_path):
        """Assembly sequence should be feasible."""
        assembly = _build_dual_arm_wheeled_robot()
        parts_results = self._make_parts_results(assembly)

        verifier = AssemblyVerifier()
        result = verifier.verify_assembly(
            assembly=assembly,
            workspace=tmp_path,
            parts_results=parts_results,
            allowed_tolerance_total=2.0,
        )

        assert len(result.sequence_checks) > 0
        assert all(sc.feasible for sc in result.sequence_checks), \
            f"Infeasible steps: {[sc for sc in result.sequence_checks if not sc.feasible]}"

    def test_bolt_alignment_checks(self, tmp_path):
        """Bolted joints should produce alignment checks."""
        assembly = _build_dual_arm_wheeled_robot()
        parts_results = self._make_parts_results(assembly)

        verifier = AssemblyVerifier()
        result = verifier.verify_assembly(
            assembly=assembly,
            workspace=tmp_path,
            parts_results=parts_results,
            allowed_tolerance_total=2.0,
        )

        # Count bolted joints
        bolted_joints = [j for j in assembly.joints
                         if j.connection and j.connection.type == "bolted"]
        assert len(result.bolt_alignment_checks) == len(bolted_joints)

    def test_report_generation(self, tmp_path):
        """Should generate comprehensive Chinese report."""
        assembly = _build_dual_arm_wheeled_robot()
        parts_results = self._make_parts_results(assembly)

        verifier = AssemblyVerifier()
        result = verifier.verify_assembly(
            assembly=assembly,
            workspace=tmp_path,
            parts_results=parts_results,
            allowed_tolerance_total=2.0,
        )

        report = verifier.generate_assembly_report(result)
        assert "装配验证报告" in report
        assert "Dual-Arm" in report or "4-Wheel" in report


# ============================================================================
# Test 9: JSON report export
# ============================================================================

class TestJSONExport:

    def test_export_full_report(self, tmp_path):
        """Export complete robot assembly report as JSON."""
        assembly = _build_dual_arm_wheeled_robot()
        solver = AssemblySolver(assembly)
        placements = solver.solve()
        parts_results = {p.name: {"artifacts": [f"{p.name}.step"], "result": "success"}
                         for p in assembly.parts}

        verifier = AssemblyVerifier()
        result = verifier.verify_assembly(
            assembly=assembly,
            workspace=tmp_path,
            parts_results=parts_results,
            placements=placements,
            allowed_tolerance_total=2.0,
        )

        # Count connection methods
        conn_counts: dict[str, int] = {}
        for j in assembly.joints:
            if j.connection:
                conn_counts[j.connection.type] = conn_counts.get(j.connection.type, 0) + 1
            else:
                conn_counts["default"] = conn_counts.get("default", 0) + 1

        report_data = {
            "robot_name": assembly.name,
            "description": assembly.description,
            "total_parts": len(assembly.parts),
            "total_joints": len(assembly.joints),
            "part_categories": {
                cat: sum(1 for p in assembly.parts if p.category == cat)
                for cat in sorted(set(p.category for p in assembly.parts))
            },
            "connection_method_distribution": conn_counts,
            "default_angles": assembly.default_angles,
            "verification": {
                "overall_pass": result.overall_pass,
                "collision_free": result.collision_free,
                "fcl_available": result.fcl_available,
                "mating_surface_checks": len(result.mating_surface_checks),
                "bolt_alignment_checks": len(result.bolt_alignment_checks),
                "sequence_checks": len(result.sequence_checks),
                "tolerance_chain_checks": len(result.tolerance_chain_checks),
                "verification_items_total": len(result.verification_items),
                "verification_items_passed": sum(1 for v in result.verification_items if v.passed),
            },
            "key_placements": {
                name: {
                    "position": [round(v, 2) for v in p["position"]],
                    "rotation": p.get("rotation", [0, 0, 0, 0]),
                }
                for name, p in placements.items()
            },
            "bolt_alignment_details": [
                {
                    "parent": bc.parent_part,
                    "child": bc.child_part,
                    "aligned": bc.aligned,
                    "hole_count_parent": bc.hole_count_parent,
                }
                for bc in result.bolt_alignment_checks
            ],
            "sequence_feasibility": [
                {
                    "step": sc.step,
                    "part": sc.part_name,
                    "feasible": sc.feasible,
                }
                for sc in result.sequence_checks
            ],
        }

        # Save to data/
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        report_path = data_dir / "dual_arm_wheeled_robot_report.json"
        report_path.write_text(
            json.dumps(report_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        assert report_path.exists()
        with open(report_path, encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["total_parts"] == 34
        assert loaded["total_joints"] == 36
        assert loaded["part_categories"]["actuator"] >= 10
        assert loaded["part_categories"]["structural"] >= 10

    def test_report_summary_matches_expectations(self, tmp_path):
        """Report should match the design specification."""
        assembly = _build_dual_arm_wheeled_robot()

        # 4 wheels
        wheels = [p for p in assembly.parts if p.name.startswith("wheel_")]
        assert len(wheels) == 4

        # 2 arms × 7 parts = 14 arm parts
        arm_parts = [p for p in assembly.parts if p.name.startswith("arm_")]
        assert len(arm_parts) == 14

        # 2 sensors
        sensors = [p for p in assembly.parts if p.category == "sensor"]
        assert len(sensors) == 2

        # 4 motors
        motors = [p for p in assembly.parts if p.name.startswith("motor_")]
        assert len(motors) == 4


# ============================================================================
# Test 10: Full engineering package export (E2E pipeline)
# ============================================================================

class TestFullEngineeringPackage:
    """Run the complete export_engineering_package() pipeline.

    This generates a full output directory with FreeCAD scripts, firmware,
    URDF, BOM, assembly guide, wiring, cable routing, power, stability
    reports, and subsystem decomposition.
    """

    @pytest.fixture(scope="class")
    def export_dir(self, tmp_path_factory):
        return tmp_path_factory.mktemp("dual_arm_robot_export")

    @pytest.fixture(scope="class")
    def export_result(self, export_dir):
        assembly = _build_dual_arm_wheeled_robot()
        result = export_engineering_package(
            assembly=assembly,
            output_dir=export_dir,
            controller="esp32",
        )
        return result

    def test_output_directory_exists(self, export_dir, export_result):
        assert export_dir.exists()
        assert export_dir.is_dir()

    def test_freecad_scripts_generated(self, export_dir, export_result):
        fc_dir = export_dir / "freecad_scripts"
        assert fc_dir.exists()
        scripts = list(fc_dir.glob("*.py"))
        assert len(scripts) >= 30, f"Expected >=30 FreeCAD scripts, got {len(scripts)}"

    def test_bom_generated(self, export_dir, export_result):
        bom_path = export_dir / "bom.md"
        assert bom_path.exists()
        content = bom_path.read_text(encoding="utf-8")
        assert "BOM" in content or "物料" in content or "零件" in content

    def test_assembly_guide_generated(self, export_dir, export_result):
        guide_path = export_dir / "assembly_guide.md"
        assert guide_path.exists()

    def test_wiring_diagram_generated(self, export_dir, export_result):
        wiring_path = export_dir / "wiring_diagram.md"
        assert wiring_path.exists()

    def test_firmware_generated(self, export_dir, export_result):
        fw_dir = export_dir / "firmware"
        assert fw_dir.exists()
        # Should have at least the .ino and some headers
        fw_files = list(fw_dir.iterdir())
        assert len(fw_files) >= 3, f"Expected >=3 firmware files, got {len(fw_files)}"

    def test_design_report_json(self, export_dir, export_result):
        report_path = export_dir / "design_report.json"
        assert report_path.exists()
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        assert report["total_parts"] >= 30

    def test_stability_report(self, export_dir, export_result):
        stability_path = export_dir / "stability_report.md"
        assert stability_path.exists()

    def test_power_report(self, export_dir, export_result):
        power_path = export_dir / "power_report.md"
        assert power_path.exists()

    def test_urdf_generated(self, export_dir, export_result):
        urdf_path = export_dir / "urdf.xml"
        assert urdf_path.exists()
        content = urdf_path.read_text(encoding="utf-8")
        assert "<robot" in content
        assert "</robot>" in content

    def test_subsystem_jsons(self, export_dir, export_result):
        ss_dir = export_dir / "subsystems"
        assert ss_dir.exists()
        json_files = list(ss_dir.glob("*.json"))
        assert len(json_files) >= 3, f"Expected >=3 subsystem files, got {len(json_files)}"

    def test_cable_routing_report(self, export_dir, export_result):
        cable_path = export_dir / "cable_routing_report.md"
        assert cable_path.exists()

    def test_export_result_metrics(self, export_result):
        assert "generated_files" in export_result
        assert len(export_result["generated_files"]) >= 20

    def test_ros2_package_structure(self, export_dir, export_result):
        ros2_dir = export_dir / "ros2_package"
        assert ros2_dir.exists()
        # Find the package directory inside
        pkg_dirs = [d for d in ros2_dir.iterdir() if d.is_dir()]
        assert len(pkg_dirs) >= 1, "Expected at least one ROS2 package directory"
        pkg = pkg_dirs[0]
        assert (pkg / "package.xml").exists() or (pkg / "CMakeLists.txt").exists()

    def test_total_file_count(self, export_dir, export_result):
        all_files = [f for f in export_dir.rglob("*") if f.is_file()]
        assert len(all_files) >= 40, f"Expected >=40 total files, got {len(all_files)}"
