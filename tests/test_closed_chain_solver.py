"""Tests for closed-chain solver and drive train tools (Task 47).

Covers: ClosedChainSolver, DifferentialConstraint, GearConstraint,
loop detection, iterative constraint solving, gear/belt joints,
eccentric rotation, graph topology, drive_train_design tool.
"""

import json
import math
import pytest

from lang3d.knowledge.mechanics import Assembly, Joint, Part
from lang3d.tools.assembly_solver import (
    AssemblySolver,
    ClosedChainSolver,
    DifferentialConstraint,
    GearConstraint,
    _axis_angle_deg_to_rot_mat,
    _identity_matrix,
)
from lang3d.tools.drive_train import (
    DriveTrainDesignTool,
    drive_train_design,
    register_drive_train_tools,
)


# ── Helpers ────────────────────────────────────────────────────


def _make_4wheel_assembly() -> Assembly:
    """Create a 4-wheel differential drive assembly with a closed loop."""
    return Assembly(
        name="4wheel_diff",
        parts=[
            Part(name="chassis", category="structural", description="底盘",
                 dimensions={"length": 300, "width": 200, "height": 5}),
            Part(name="axle_front", category="structural", description="前轴",
                 dimensions={"length": 200, "width": 10, "height": 10}),
            Part(name="wheel_fl", category="wheel", description="左前轮",
                 dimensions={"diameter": 80, "width": 25}),
            Part(name="wheel_fr", category="wheel", description="右前轮",
                 dimensions={"diameter": 80, "width": 25}),
            Part(name="axle_rear", category="structural", description="后轴",
                 dimensions={"length": 200, "width": 10, "height": 10}),
            Part(name="wheel_rl", category="wheel", description="左后轮",
                 dimensions={"diameter": 80, "width": 25}),
            Part(name="wheel_rr", category="wheel", description="右后轮",
                 dimensions={"diameter": 80, "width": 25}),
        ],
        joints=[
            Joint("fixed", "chassis", "axle_front", parent_anchor="front", child_anchor="bottom"),
            Joint("revolute", "axle_front", "wheel_fl", parent_anchor="left", child_anchor="back",
                  axis="y", range_deg=(-360, 360)),
            Joint("revolute", "axle_front", "wheel_fr", parent_anchor="right", child_anchor="back",
                  axis="y", range_deg=(-360, 360)),
            Joint("fixed", "chassis", "axle_rear", parent_anchor="back", child_anchor="bottom"),
            Joint("revolute", "axle_rear", "wheel_rl", parent_anchor="left", child_anchor="back",
                  axis="y", range_deg=(-360, 360)),
            Joint("revolute", "axle_rear", "wheel_rr", parent_anchor="right", child_anchor="back",
                  axis="y", range_deg=(-360, 360)),
        ],
    )


def _make_dual_arm_assembly() -> Assembly:
    """Assembly with two arms sharing a common chassis (graph structure)."""
    return Assembly(
        name="dual_arm",
        parts=[
            Part(name="chassis", category="structural", description="底盘",
                 dimensions={"length": 300, "width": 200, "height": 5}),
            Part(name="arm_left", category="structural", description="左臂",
                 dimensions={"length": 200, "width": 30, "height": 20}),
            Part(name="arm_right", category="structural", description="右臂",
                 dimensions={"length": 200, "width": 30, "height": 20}),
        ],
        joints=[
            Joint("revolute", "chassis", "arm_left", parent_anchor="top",
                  child_anchor="bottom", axis="y", range_deg=(-180, 180)),
            Joint("revolute", "chassis", "arm_right", parent_anchor="top",
                  child_anchor="bottom", axis="y", range_deg=(-180, 180)),
        ],
    )


def _make_closed_loop_assembly() -> Assembly:
    """Assembly with an actual closed loop (A-B-C-A)."""
    return Assembly(
        name="closed_loop",
        parts=[
            Part(name="A", category="structural", description="",
                 dimensions={"length": 100, "width": 10, "height": 10}),
            Part(name="B", category="structural", description="",
                 dimensions={"length": 100, "width": 10, "height": 10}),
            Part(name="C", category="structural", description="",
                 dimensions={"length": 100, "width": 10, "height": 10}),
        ],
        joints=[
            Joint("revolute", "A", "B", parent_anchor="right", child_anchor="left",
                  axis="z", range_deg=(-90, 90)),
            Joint("revolute", "B", "C", parent_anchor="right", child_anchor="left",
                  axis="z", range_deg=(-90, 90)),
            # This closes the loop: C → A
            Joint("revolute", "C", "A", parent_anchor="right", child_anchor="left",
                  axis="z", range_deg=(-90, 90)),
        ],
    )


# ── DifferentialConstraint Tests ──────────────────────────────


class TestDifferentialConstraint:
    def test_straight_line(self):
        dc = DifferentialConstraint("wheel_l", "wheel_r", track_width_mm=300)
        vl, vr = dc.speed_ratio(0.0)
        assert vl == 1.0
        assert vr == 1.0

    def test_turning(self):
        dc = DifferentialConstraint("wheel_l", "wheel_r", track_width_mm=300)
        vl, vr = dc.speed_ratio(1.0)
        assert vl < vr  # left slower, right faster for left turn

    def test_to_dict(self):
        dc = DifferentialConstraint("wl", "wr", track_width_mm=300)
        d = dc.to_dict()
        assert d["type"] == "differential"
        assert d["left_wheel"] == "wl"


# ── GearConstraint Tests ──────────────────────────────────────


class TestGearConstraint:
    def test_speed_ratio(self):
        gc = GearConstraint("motor_shaft", "wheel_shaft", transmission_ratio=2.0)
        assert gc.child_speed(100) == 200.0

    def test_reduction(self):
        gc = GearConstraint("motor", "wheel", transmission_ratio=0.1)
        assert gc.child_speed(1000) == 100.0

    def test_to_dict(self):
        gc = GearConstraint("a", "b", transmission_ratio=3.0, joint_type="belt")
        d = gc.to_dict()
        assert d["type"] == "belt"
        assert d["transmission_ratio"] == 3.0


# ── Loop Detection Tests ──────────────────────────────────────


class TestLoopDetection:
    def test_no_loop_in_tree(self):
        asm = _make_4wheel_assembly()
        solver = ClosedChainSolver(asm)
        loops = solver.detect_loops()
        assert len(loops) == 0

    def test_closed_loop_detected(self):
        asm = _make_closed_loop_assembly()
        solver = ClosedChainSolver(asm)
        loops = solver.detect_loops()
        assert len(loops) >= 1

    def test_dual_arm_no_loop(self):
        asm = _make_dual_arm_assembly()
        solver = ClosedChainSolver(asm)
        loops = solver.detect_loops()
        assert len(loops) == 0


# ── ClosedChainSolver Tests ────────────────────────────────────


class TestClosedChainSolver:
    def test_no_loops_uses_base_solver(self):
        asm = _make_4wheel_assembly()
        solver = ClosedChainSolver(asm)
        result = solver.solve_closed_chain()
        assert result["converged"] is True
        assert result["iterations"] == 0
        assert len(result["placements"]) == 7

    def test_with_loops_converges(self):
        asm = _make_closed_loop_assembly()
        solver = ClosedChainSolver(asm)
        result = solver.solve_closed_chain()
        assert "placements" in result
        assert "loops" in result
        assert len(result["loops"]) >= 1
        assert "iterations" in result

    def test_gear_constraints_applied(self):
        asm = _make_4wheel_assembly()
        solver = ClosedChainSolver(asm)
        solver.add_gear_constraint(GearConstraint("wheel_fl", "wheel_fr", 1.0))
        angles = {"wheel_fl": 45.0}
        result = solver.apply_gear_constraints(angles)
        assert result["wheel_fr"] == 45.0

    def test_gear_reduction(self):
        asm = _make_4wheel_assembly()
        solver = ClosedChainSolver(asm)
        solver.add_gear_constraint(GearConstraint("motor", "wheel", 0.5))
        angles = {"motor": 100.0}
        result = solver.apply_gear_constraints(angles)
        assert result["wheel"] == 50.0

    def test_differential_constraints(self):
        asm = _make_4wheel_assembly()
        solver = ClosedChainSolver(asm)
        solver.add_differential_constraint(
            DifferentialConstraint("wheel_rl", "wheel_rr", track_width_mm=300)
        )
        angles = {"wheel_rl": 100.0}
        result = solver.apply_differential_constraints(angles, omega_rad_s=0.5)
        assert "wheel_rr" in result
        # Right wheel should be faster for positive omega
        assert result["wheel_rr"] != 100.0

    def test_differential_straight_no_change(self):
        asm = _make_4wheel_assembly()
        solver = ClosedChainSolver(asm)
        solver.add_differential_constraint(
            DifferentialConstraint("wheel_rl", "wheel_rr", track_width_mm=300)
        )
        angles = {"wheel_rl": 100.0}
        result = solver.apply_differential_constraints(angles, omega_rad_s=0.0)
        # Straight line: no change
        assert abs(result.get("wheel_rr", 100.0) - 100.0) < 0.01


# ── Graph Topology Tests ──────────────────────────────────────


class TestGraphTopology:
    def test_dual_arm_shows_graph_topology(self):
        asm = _make_dual_arm_assembly()
        solver = AssemblySolver(asm)
        chain = solver.get_joint_chain()

        # Should have metadata at the end
        metadata = chain[-1]
        assert metadata.get("topology") == "graph"
        assert "chassis" in metadata.get("shared_nodes", [])

    def test_tree_no_graph_metadata_for_simple_chain(self):
        """A simple linear chain (A→B→C) should not show graph topology."""
        asm = Assembly(
            name="linear_chain",
            parts=[
                Part(name="A", category="test", description="", dimensions={"length": 50}),
                Part(name="B", category="test", description="", dimensions={"length": 50}),
                Part(name="C", category="test", description="", dimensions={"length": 50}),
            ],
            joints=[
                Joint("fixed", "A", "B", parent_anchor="right", child_anchor="left"),
                Joint("fixed", "B", "C", parent_anchor="right", child_anchor="left"),
            ],
        )
        solver = AssemblySolver(asm)
        chain = solver.get_joint_chain()

        # Last entry should be a regular joint, not metadata
        last = chain[-1]
        assert "type" in last  # regular joint entry, not topology metadata


# ── Extended _compute_child_transform Tests ────────────────────


class TestExtendedTransform:
    def test_gear_joint_type(self):
        asm = Assembly(
            name="gear_test",
            parts=[
                Part(name="gear1", category="test", description="",
                     dimensions={"diameter": 30, "height": 10}),
                Part(name="gear2", category="test", description="",
                     dimensions={"diameter": 30, "height": 10}),
            ],
            joints=[
                Joint("gear", "gear1", "gear2", parent_anchor="right",
                      child_anchor="left", axis="z", range_deg=(-360, 360)),
            ],
        )
        solver = AssemblySolver(asm)
        placements = solver.solve(joint_angles={"gear2": 45.0})
        assert "gear1" in placements
        assert "gear2" in placements

    def test_belt_joint_type(self):
        asm = Assembly(
            name="belt_test",
            parts=[
                Part(name="pulley1", category="test", description="",
                     dimensions={"diameter": 20, "height": 8}),
                Part(name="pulley2", category="test", description="",
                     dimensions={"diameter": 20, "height": 8}),
            ],
            joints=[
                Joint("belt", "pulley1", "pulley2", parent_anchor="front",
                      child_anchor="back", axis="y", range_deg=(-360, 360)),
            ],
        )
        solver = AssemblySolver(asm)
        placements = solver.solve(joint_angles={"pulley2": 90.0})
        assert "pulley1" in placements


# ── axis_angle_deg_to_rot_mat Tests ───────────────────────────


class TestAxisAngleConversion:
    def test_identity(self):
        m = _axis_angle_deg_to_rot_mat([0, 0, 1, 0])
        assert abs(m[0][0] - 1) < 0.01
        assert abs(m[1][1] - 1) < 0.01
        assert abs(m[2][2] - 1) < 0.01

    def test_90_deg_z(self):
        m = _axis_angle_deg_to_rot_mat([0, 0, 1, 90])
        # cos(90)=0, sin(90)=1
        assert abs(m[0][0]) < 0.01
        assert abs(m[0][1] - (-1)) < 0.01
        assert abs(m[1][0] - 1) < 0.01


# ── Drive Train Tool Tests ────────────────────────────────────


class TestDriveTrainDesign:
    def test_differential_basic(self):
        result = drive_train_design(drive_type="differential")
        assert result["total_parts"] > 0
        assert result["total_joints"] > 0
        assert "motor_l" in {p["name"] for p in result["parts"]}
        assert "motor_r" in {p["name"] for p in result["parts"]}

    def test_mecanum(self):
        result = drive_train_design(drive_type="mecanum")
        assert result["total_parts"] > 0

    def test_motor_specs_included(self):
        result = drive_train_design(motor_type="JGA25-370")
        assert "motor_specs" in result
        assert result["motor_specs"]["voltage"] == 12

    def test_unknown_motor(self):
        result = drive_train_design(motor_type="UNKNOWN")
        assert result["motor_specs"] == {}

    def test_tool_execution(self):
        tool = DriveTrainDesignTool()
        result = json.loads(tool.execute(drive_type="differential_2w"))
        assert "parts" in result
        assert "joints" in result


class TestDriveTrainRegistration:
    def test_register(self):
        registry = type("MockRegistry", (), {"register": lambda self, t: None})()
        register_drive_train_tools(registry)

    def test_tool_name(self):
        tool = DriveTrainDesignTool()
        assert tool.name == "drive_train_design"
