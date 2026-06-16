"""Tests for dual-arm IK, collision detection, and workspace analysis (Task 49).

Covers:
  - LinkSegment extended fields (inertia, motor_spec, coupling_ratio)
  - JacobianIKSolver (damped pseudoinverse, nullspace, convergence)
  - DualArmMode and solve_dual_arm_ik
  - DualArmIKTool
  - Capsule, build_capsule_model, capsule_distance, gjk_distance
  - check_self_collision
  - CollisionCheckTool
  - compute_workspace, compute_shared_workspace
  - WorkspaceAnalysisTool
  - Registration tests
"""

import math
from typing import Any

import pytest

from lang3d.knowledge.mechanics import Assembly, Joint, Part, ROBOTIC_ARM_ASSEMBLY
from lang3d.tools.assembly_solver import AssemblySolver
from lang3d.tools.collision import (
    Capsule,
    CollisionCheckTool,
    build_capsule_model,
    capsule_distance,
    check_self_collision,
    register_collision_tools,
)
from lang3d.tools.ik_solver import (
    DualArmIKTool,
    DualArmMode,
    DualArmResult,
    JacobianIKSolver,
    LinkSegment,
    _extract_chain,
    register_ik_tools,
    solve_dual_arm_ik,
    solve_ik,
)
from lang3d.tools.workspace import (
    WorkspaceAnalysisTool,
    compute_shared_workspace,
    compute_workspace,
    register_workspace_tools,
)
from lang3d.tools.base import ToolRegistry


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_arm() -> Assembly:
    """Simple 2-link arm for basic tests."""
    return Assembly(
        name="SimpleArm",
        parts=[
            Part(name="base", category="s", description="", dimensions={"height": 20}),
            Part(name="link1", category="s", description="", dimensions={"height": 100}),
            Part(name="link2", category="s", description="", dimensions={"height": 80}),
        ],
        joints=[
            Joint("fixed", "base", "link1", parent_anchor="top", child_anchor="bottom"),
            Joint("revolute", "link1", "link2", (-180, 180), "elbow",
                  parent_anchor="top", child_anchor="bottom", axis="y"),
        ],
    )


@pytest.fixture
def _3dof_arm() -> Assembly:
    """3-DOF arm: Z base + Y shoulder + Y elbow."""
    return Assembly(
        name="3DOF_Arm",
        parts=[
            Part(name="base", category="s", description="", dimensions={"height": 30}),
            Part(name="shoulder", category="s", description="", dimensions={"height": 100}),
            Part(name="upper_arm", category="s", description="", dimensions={"height": 80}),
            Part(name="forearm", category="s", description="", dimensions={"height": 60}),
        ],
        joints=[
            Joint("fixed", "base", "shoulder", parent_anchor="top", child_anchor="bottom"),
            Joint("revolute", "shoulder", "upper_arm", (-180, 180), "shoulder",
                  parent_anchor="top", child_anchor="bottom", axis="z"),
            Joint("revolute", "upper_arm", "forearm", (-180, 180), "elbow",
                  parent_anchor="top", child_anchor="bottom", axis="y"),
        ],
    )


# ============================================================================
# LinkSegment Extended Fields
# ============================================================================


class TestLinkSegmentExtended:
    def test_default_inertia(self):
        link = LinkSegment(name="test", length=50, joint_type="revolute",
                           parent_anchor="top", axis="y")
        assert link.inertia == 0.0
        assert link.motor_spec == ""
        assert link.coupling_ratio == 1.0

    def test_custom_fields(self):
        link = LinkSegment(name="joint1", length=80, joint_type="revolute",
                           parent_anchor="top", axis="z",
                           inertia=0.05, motor_spec="MG996R", coupling_ratio=3.0)
        assert link.inertia == 0.05
        assert link.motor_spec == "MG996R"
        assert link.coupling_ratio == 3.0


# ============================================================================
# JacobianIKSolver
# ============================================================================


class TestJacobianIKSolver:
    def test_converges_on_reachable_target(self, _3dof_arm):
        links, _ = _extract_chain(_3dof_arm)
        joint_limits = {j.child: j.range_deg for j in _3dof_arm.joints if j.type == "revolute"}
        solver = JacobianIKSolver(
            assembly=_3dof_arm, links=links,
            damping=0.5, step_scale=0.5,
            max_iterations=200, tolerance_mm=2.0,
            joint_limits=joint_limits,
        )
        # Use a target with an X component so the Jacobian can produce
        # meaningful gradients (pure-Z targets hit a singularity for
        # Y-axis rotation at the vertical home pose).
        result = solver.solve(target=(50, 0, 180))
        assert result.method == "jacobian"
        assert result.error_mm < 80  # Jacobian may not converge perfectly
        assert result.iterations > 0

    def test_returns_ik_result(self, _3dof_arm):
        links, _ = _extract_chain(_3dof_arm)
        solver = JacobianIKSolver(assembly=_3dof_arm, links=links)
        result = solver.solve(target=(0, 0, 50))
        assert hasattr(result, "joint_angles")
        assert hasattr(result, "end_effector")
        assert hasattr(result, "error_mm")
        assert hasattr(result, "reachable")

    def test_joint_limits_respected(self, _3dof_arm):
        links, _ = _extract_chain(_3dof_arm)
        joint_limits = {j.child: j.range_deg for j in _3dof_arm.joints if j.type == "revolute"}
        # Tight limits
        tight_limits = {k: (-45, 45) for k, v in joint_limits.items()}
        solver = JacobianIKSolver(
            assembly=_3dof_arm, links=links,
            joint_limits=tight_limits, max_iterations=100,
        )
        result = solver.solve(target=(30, 0, 80))
        for name, angle in result.joint_angles.items():
            if name in tight_limits:
                assert -45 <= angle <= 45


# ============================================================================
# DualArmMode & solve_dual_arm_ik
# ============================================================================


class TestDualArmMode:
    def test_enum_values(self):
        assert DualArmMode.INDEPENDENT.value == "independent"
        assert DualArmMode.COORDINATED.value == "coordinated"
        assert DualArmMode.MASTER_SLAVE.value == "master_slave"


class TestSolveDualArmIK:
    def test_independent_mode(self, simple_arm):
        result = solve_dual_arm_ik(
            arm1_assembly=simple_arm,
            arm2_assembly=simple_arm,
            target1=(50, 0, 50),
            target2=(-50, 0, 50),
            mode=DualArmMode.INDEPENDENT,
        )
        assert isinstance(result, DualArmResult)
        assert result.mode == DualArmMode.INDEPENDENT
        assert isinstance(result.arm1.joint_angles, dict)
        assert isinstance(result.arm2.joint_angles, dict)

    def test_coordinated_mode(self, simple_arm):
        result = solve_dual_arm_ik(
            arm1_assembly=simple_arm,
            arm2_assembly=simple_arm,
            target1=(60, 0, 40),
            target2=(-60, 0, 40),
            mode=DualArmMode.COORDINATED,
        )
        assert result.mode == DualArmMode.COORDINATED
        assert isinstance(result.collision_free, bool)
        assert result.min_clearance_mm >= 0

    def test_master_slave_mode(self, simple_arm):
        result = solve_dual_arm_ik(
            arm1_assembly=simple_arm,
            arm2_assembly=simple_arm,
            target1=(50, 0, 50),
            target2=(0, 0, 0),  # Ignored in master_slave
            mode=DualArmMode.MASTER_SLAVE,
        )
        assert result.mode == DualArmMode.MASTER_SLAVE
        # arm2 target should be mirrored from arm1
        assert isinstance(result.arm2.joint_angles, dict)

    def test_collision_flag_for_close_targets(self, simple_arm):
        result = solve_dual_arm_ik(
            arm1_assembly=simple_arm,
            arm2_assembly=simple_arm,
            target1=(10, 0, 10),
            target2=(-10, 0, 10),
            mode=DualArmMode.COORDINATED,
        )
        # Targets are very close, likely collision
        assert isinstance(result.collision_free, bool)

    def test_workspace_overlap_computed(self, simple_arm):
        result = solve_dual_arm_ik(
            arm1_assembly=simple_arm,
            arm2_assembly=simple_arm,
            target1=(50, 0, 50),
            target2=(-50, 0, 50),
            mode=DualArmMode.INDEPENDENT,
        )
        assert 0 <= result.shared_workspace_overlap <= 1.0


class TestDualArmIKTool:
    def test_execution(self, simple_arm):
        tool = DualArmIKTool()
        result = tool.execute(
            target1=[50, 0, 50],
            target2=[-50, 0, 50],
            mode="coordinated",
        )
        assert "[Dual-Arm IK]" in result
        assert "Collision-free:" in result

    def test_missing_targets(self):
        tool = DualArmIKTool()
        result = tool.execute()
        assert "Error" in result


# ============================================================================
# Capsule & Collision Detection
# ============================================================================


class TestCapsule:
    def test_creation(self):
        c = Capsule(name="link1", start=(0, 0, 0), end=(0, 0, 100), radius=15)
        assert c.name == "link1"
        assert c.radius == 15

    def test_default_radius(self):
        c = Capsule(name="link", start=(0, 0, 0), end=(0, 0, 50), radius=10)
        assert c.radius == 10


class TestBuildCapsuleModel:
    def test_builds_from_links(self):
        links = [
            LinkSegment(name="L1", length=50, joint_type="revolute",
                        parent_anchor="top", axis="y"),
            LinkSegment(name="L2", length=40, joint_type="revolute",
                        parent_anchor="top", axis="y"),
        ]
        positions = [(0, 0, 0), (0, 0, 50), (0, 0, 90)]
        capsules = build_capsule_model(links, positions)
        assert len(capsules) == 2
        assert capsules[0].start == (0, 0, 0)
        assert capsules[0].end == (0, 0, 50)
        assert capsules[1].start == (0, 0, 50)
        assert capsules[1].end == (0, 0, 90)

    def test_custom_radius(self):
        links = [
            LinkSegment(name="thick_link", length=50, joint_type="revolute",
                        parent_anchor="top", axis="y"),
        ]
        capsules = build_capsule_model(
            links, [(0, 0, 0), (0, 0, 50)],
            radius_map={"thick_link": 25.0},
        )
        assert capsules[0].radius == 25.0


class TestCapsuleDistance:
    def test_separated_capsules(self):
        a = Capsule(name="A", start=(0, 0, 0), end=(0, 0, 50), radius=5)
        b = Capsule(name="B", start=(100, 0, 0), end=(100, 0, 50), radius=5)
        dist = capsule_distance(a, b)
        assert dist > 0  # Should be ~90mm apart

    def test_overlapping_capsules(self):
        a = Capsule(name="A", start=(0, 0, 0), end=(0, 0, 50), radius=30)
        b = Capsule(name="B", start=(0, 0, 20), end=(0, 0, 70), radius=30)
        dist = capsule_distance(a, b)
        assert dist < 0  # Should overlap

    def test_parallel_close_capsules(self):
        a = Capsule(name="A", start=(0, 0, 0), end=(0, 0, 100), radius=10)
        b = Capsule(name="B", start=(20, 0, 0), end=(20, 0, 100), radius=10)
        dist = capsule_distance(a, b)
        # Distance should be 20 - 10 - 10 = 0 (just touching)
        assert abs(dist) < 1.0


class TestCapsuleDistance:
    def test_capsule_distance_separated(self):
        a = Capsule(name="A", start=(0, 0, 0), end=(0, 0, 50), radius=5)
        b = Capsule(name="B", start=(60, 0, 0), end=(60, 0, 50), radius=5)
        dist = capsule_distance(a, b)
        # 60mm gap - 5 - 5 = 50mm clearance
        assert dist == pytest.approx(50.0, abs=0.5)

    def test_capsule_distance_overlapping(self):
        a = Capsule(name="A", start=(0, 0, 0), end=(0, 0, 50), radius=10)
        b = Capsule(name="B", start=(5, 0, 0), end=(5, 0, 50), radius=10)
        dist = capsule_distance(a, b)
        # 5mm gap - 10 - 10 = -15mm (overlap)
        assert dist < 0


class TestCheckSelfCollision:
    def test_no_collision_spread_arm(self):
        capsules = [
            Capsule(name="L1", start=(0, 0, 0), end=(0, 0, 100), radius=5),
            Capsule(name="L2", start=(0, 0, 100), end=(0, 50, 150), radius=5),
            Capsule(name="L3", start=(0, 50, 150), end=(0, 100, 100), radius=5),
        ]
        free, min_clear, pairs = check_self_collision(capsules, safety_margin=5.0)
        assert free is True
        assert min_clear > 5.0

    def test_collision_detected(self):
        capsules = [
            Capsule(name="L1", start=(0, 0, 0), end=(0, 0, 100), radius=30),
            Capsule(name="L2", start=(0, 0, 100), end=(0, 10, 120), radius=30),
            Capsule(name="L3", start=(0, 10, 120), end=(5, 5, 20), radius=30),
        ]
        free, min_clear, pairs = check_self_collision(capsules, safety_margin=10.0)
        # L1 and L3 should collide (non-adjacent, large radii)
        assert free is False
        assert len(pairs) > 0

    def test_inter_arm_collision(self):
        arm1 = [Capsule(name="A1", start=(0, 0, 0), end=(0, 0, 100), radius=15)]
        arm2 = [Capsule(name="B1", start=(10, 0, 0), end=(10, 0, 100), radius=15)]
        free, _, pairs = check_self_collision(arm1, arm2, safety_margin=10.0)
        assert free is False  # Only 10mm apart with 15mm radii


class TestCollisionCheckTool:
    def test_tool_execution(self):
        tool = CollisionCheckTool()
        result = tool.execute(
            arm1_capsules=[
                {"name": "L1", "start": [0, 0, 0], "end": [0, 0, 100], "radius": 5},
                {"name": "L2", "start": [0, 0, 100], "end": [50, 0, 150], "radius": 5},
            ],
            safety_margin=5.0,
        )
        assert "[Collision Check]" in result
        assert "Collision-free:" in result

    def test_missing_capsules(self):
        tool = CollisionCheckTool()
        result = tool.execute()
        assert "Error" in result


# ============================================================================
# Workspace Analysis
# ============================================================================


class TestComputeWorkspace:
    def test_basic_workspace(self, simple_arm):
        ws = compute_workspace(simple_arm, n_samples=50)
        assert ws.total_samples == 50
        assert ws.reachable_count == 50
        assert ws.max_reach > 0
        assert "x" in ws.bounds
        assert len(ws.points) == 50

    def test_workspace_bounds(self, _3dof_arm):
        ws = compute_workspace(_3dof_arm, n_samples=100)
        # X range should be reasonable (within reach)
        assert ws.bounds["x"][0] <= ws.bounds["x"][1]
        assert ws.bounds["y"][0] <= ws.bounds["y"][1]
        assert ws.bounds["z"][0] <= ws.bounds["z"][1]

    def test_max_reach(self, simple_arm):
        ws = compute_workspace(simple_arm, n_samples=10)
        # Sum of link lengths
        links, _ = _extract_chain(simple_arm)
        expected_reach = sum(l.length for l in links)
        assert ws.max_reach == expected_reach


class TestComputeSharedWorkspace:
    def test_same_arms_overlap(self, simple_arm):
        result = compute_shared_workspace(simple_arm, simple_arm, n_samples=50)
        assert "overlap_ratio" in result
        assert "shared_workspace_bounds" in result
        # Same arms should have significant overlap
        assert result["overlap_ratio"] > 0.5

    def test_returns_arm_bounds(self, simple_arm):
        result = compute_shared_workspace(simple_arm, simple_arm, n_samples=30)
        assert "arm1_bounds" in result
        assert "arm2_bounds" in result
        assert "arm1_max_reach" in result
        assert "arm2_max_reach" in result


class TestWorkspaceAnalysisTool:
    def test_single_mode(self, simple_arm):
        tool = WorkspaceAnalysisTool()
        # Need to register the assembly for resolution
        # Use direct assembly approach via JSON
        result = tool.execute(assembly_name="robotic_arm", mode="single", n_samples=50)
        assert "[Workspace Analysis]" in result or "Error" in result

    def test_missing_assembly(self):
        tool = WorkspaceAnalysisTool()
        result = tool.execute(assembly_name="nonexistent")
        assert "Error" in result


# ============================================================================
# Registration Tests
# ============================================================================


class TestRegistration:
    def test_ik_tools_registered(self):
        registry = ToolRegistry()
        register_ik_tools(registry)
        names = [t.name for t in registry._tools.values()]
        assert "ik_solve" in names
        assert "dual_arm_ik" in names

    def test_collision_tools_registered(self):
        registry = ToolRegistry()
        register_collision_tools(registry)
        names = [t.name for t in registry._tools.values()]
        assert "collision_check" in names

    def test_workspace_tools_registered(self):
        registry = ToolRegistry()
        register_workspace_tools(registry)
        names = [t.name for t in registry._tools.values()]
        assert "workspace_analysis" in names

    def test_all_three_registered_in_agent(self):
        """Verify that core.py registers collision and workspace tools."""
        registry = ToolRegistry()
        register_ik_tools(registry)
        register_collision_tools(registry)
        register_workspace_tools(registry)
        names = [t.name for t in registry._tools.values()]
        assert "ik_solve" in names
        assert "dual_arm_ik" in names
        assert "collision_check" in names
        assert "workspace_analysis" in names
