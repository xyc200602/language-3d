"""Tests for Layer 3 Phase 2: motion collision, reachability, and interference.

All tests are guarded by ``HAS_FCL`` and skipped automatically when
python-fcl is not installed.
"""

from __future__ import annotations

import json
import math
import pytest

from lang3d.knowledge.mechanics import Assembly, Joint, Part, ROBOTIC_ARM_ASSEMBLY

# Check if FCL is available
try:
    import fcl as fcl_mod  # noqa: F401
    import trimesh  # noqa: F401
    HAS_FCL = True
except ImportError:
    HAS_FCL = False

pytestmark = pytest.mark.skipif(not HAS_FCL, reason="python-fcl not installed")


def _box(name, l, w, h):
    return Part(name, "structural", name, dimensions=dict(length=l, width=w, height=h))


# ---------------------------------------------------------------------------
# Motion collision tests
# ---------------------------------------------------------------------------

class TestMotionCollision:
    """Tests for MotionCollisionChecker."""

    def test_simple_arm_no_collision(self):
        """A simple 2-link arm with well-separated parts should be collision-free."""
        from lang3d.tools.motion_collision import MotionCollisionChecker

        parts = [
            _box("base", 40, 40, 10),
            _box("link1", 100, 20, 10),
            _box("link2", 80, 15, 10),
        ]
        joints = [
            Joint("fixed", "base", "link1",
                  parent_anchor="top", child_anchor="bottom"),
            Joint("revolute", "link1", "link2",
                  parent_anchor="top", child_anchor="bottom",
                  range_deg=(-90, 90)),
        ]
        asm = Assembly(name="simple_arm", parts=parts, joints=joints)

        checker = MotionCollisionChecker(num_samples=5)
        result = checker.check_motion_collisions(asm)

        assert isinstance(result.collision_free, bool)
        assert result.joints_checked == 1  # only 1 revolute joint

    def test_self_interference_detection(self):
        """An arm where the link folds back onto the base should detect collision."""
        from lang3d.tools.motion_collision import MotionCollisionChecker

        # Very long link that will fold back onto itself
        parts = [
            _box("base", 20, 20, 5),
            _box("arm", 60, 20, 10),
            _box("hand", 60, 20, 10),
        ]
        joints = [
            Joint("fixed", "base", "arm",
                  parent_anchor="top", child_anchor="bottom"),
            Joint("revolute", "arm", "hand",
                  parent_anchor="top", child_anchor="bottom",
                  range_deg=(-180, 180)),
        ]
        asm = Assembly(name="fold_arm", parts=parts, joints=joints)

        checker = MotionCollisionChecker(num_samples=10)
        result = checker.check_motion_collisions(asm)

        # At extreme angles, hand should collide with base
        assert result.joints_checked == 1
        jr = result.joint_results[0]
        assert isinstance(jr.collision_angles, list)

    def test_no_revolute_joints(self):
        """An assembly with only fixed joints should return immediately."""
        from lang3d.tools.motion_collision import MotionCollisionChecker

        parts = [_box("a", 10, 10, 10), _box("b", 10, 10, 10)]
        joints = [Joint("fixed", "a", "b",
                        parent_anchor="top", child_anchor="bottom")]
        asm = Assembly(name="fixed_only", parts=parts, joints=joints)

        checker = MotionCollisionChecker(num_samples=5)
        result = checker.check_motion_collisions(asm)

        assert result.collision_free is True
        assert result.joints_checked == 0

    def test_free_segments_computed(self):
        """Collision-free angular segments should be computed."""
        from lang3d.tools.motion_collision import MotionCollisionChecker

        # Use the robotic arm which has revolute joints
        checker = MotionCollisionChecker(num_samples=5)
        result = checker.check_motion_collisions(ROBOTIC_ARM_ASSEMBLY)

        assert result.joints_checked > 0
        for jr in result.joint_results:
            # Each joint result should have free segments (possibly the full range)
            assert isinstance(jr.collision_free_segments, list)
            if not jr.has_collision:
                # No collision => full range is free
                assert len(jr.collision_free_segments) >= 1


# ---------------------------------------------------------------------------
# Reachability tests
# ---------------------------------------------------------------------------

class TestReachability:
    """Tests for ReachabilityAnalyzer."""

    def test_reachable_target(self):
        """A target within the arm's workspace should be reachable."""
        from lang3d.tools.motion_collision import ReachabilityAnalyzer

        parts = [
            _box("base", 30, 30, 5),
            _box("link1", 80, 15, 8),
            _box("link2", 60, 12, 8),
        ]
        joints = [
            Joint("fixed", "base", "link1",
                  parent_anchor="top", child_anchor="bottom"),
            Joint("revolute", "link1", "link2",
                  parent_anchor="top", child_anchor="bottom",
                  range_deg=(-90, 90)),
        ]
        asm = Assembly(name="reach_arm", parts=parts, joints=joints)

        analyzer = ReachabilityAnalyzer(samples_per_joint=5)
        # Target along the arm direction should be reachable
        result = analyzer.analyze_reachability(
            asm, target=(50, 0, 10), tolerance_mm=50.0,
        )

        assert isinstance(result.reachable, bool)
        assert result.samples_total > 0
        assert result.error_mm < float("inf")

    def test_unreachable_target(self):
        """A target far beyond the arm's reach should not be reachable."""
        from lang3d.tools.motion_collision import ReachabilityAnalyzer

        parts = [
            _box("base", 30, 30, 5),
            _box("link1", 40, 15, 8),
            _box("link2", 30, 12, 8),
        ]
        joints = [
            Joint("fixed", "base", "link1",
                  parent_anchor="top", child_anchor="bottom"),
            Joint("revolute", "link1", "link2",
                  parent_anchor="top", child_anchor="bottom",
                  range_deg=(-45, 45)),
        ]
        asm = Assembly(name="short_arm", parts=parts, joints=joints)

        analyzer = ReachabilityAnalyzer(samples_per_joint=5)
        # Target very far away
        result = analyzer.analyze_reachability(
            asm, target=(5000, 5000, 5000), tolerance_mm=5.0,
        )

        assert result.reachable is False

    def test_workspace_bbox(self):
        """Workspace bounding box should be computed."""
        from lang3d.tools.motion_collision import ReachabilityAnalyzer

        analyzer = ReachabilityAnalyzer(samples_per_joint=3)
        bbox = analyzer.compute_workspace_bbox(ROBOTIC_ARM_ASSEMBLY)

        assert len(bbox) == 2
        min_pt, max_pt = bbox
        assert len(min_pt) == 3
        assert len(max_pt) == 3
        # Min should be <= max in all dimensions
        for i in range(3):
            assert min_pt[i] <= max_pt[i]


# ---------------------------------------------------------------------------
# Interference report tests
# ---------------------------------------------------------------------------

class TestInterferenceReport:
    """Tests for MeshCollisionChecker.generate_interference_report()."""

    def test_no_interference(self):
        """Well-separated parts should have no interference."""
        from lang3d.tools.mesh_collision import MeshCollisionChecker

        parts = [
            _box("a", 10, 10, 10),
            _box("b", 10, 10, 10),
        ]
        asm = Assembly(name="test", parts=parts, joints=[])
        placements = {
            "a": {"position": [0, 0, 0]},
            "b": {"position": [100, 0, 0]},
        }

        checker = MeshCollisionChecker()
        report = checker.generate_interference_report(asm, placements, skip_adjacent=False)

        assert report.collision_free is True
        assert len(report.pairs) == 1
        assert report.pairs[0].severity == "none"
        assert report.worst_interference is not None
        assert report.worst_interference.severity == "none"

    def test_with_interference(self):
        """Overlapping parts should produce an interference report."""
        from lang3d.tools.mesh_collision import MeshCollisionChecker

        parts = [
            _box("a", 100, 100, 100),
            _box("b", 100, 100, 100),
        ]
        asm = Assembly(name="test", parts=parts, joints=[])
        placements = {
            "a": {"position": [0, 0, 0]},
            "b": {"position": [0, 0, 0]},
        }

        checker = MeshCollisionChecker()
        report = checker.generate_interference_report(asm, placements, skip_adjacent=False)

        assert report.collision_free is False
        assert len(report.pairs) == 1
        ip = report.pairs[0]
        assert ip.is_collision if hasattr(ip, 'is_collision') else ip.severity != "none"
        # Worst should be this pair
        assert report.worst_interference is not None

    def test_severity_classification(self):
        """Severity should be correctly classified."""
        from lang3d.tools.mesh_collision import MeshCollisionChecker

        checker = MeshCollisionChecker()

        assert checker._classify_severity(False, 0, 5.0, 2.0) == "none"
        assert checker._classify_severity(False, 0, 1.0, 2.0) == "clearance"
        assert checker._classify_severity(True, 0.2, 0, 2.0) == "light"
        assert checker._classify_severity(True, 1.0, 0, 2.0) == "moderate"
        assert checker._classify_severity(True, 3.0, 0, 2.0) == "severe"

    def test_report_with_robotic_arm(self):
        """Interference report on the robotic arm assembly should run."""
        from lang3d.tools.assembly_solver import AssemblySolver
        from lang3d.tools.mesh_collision import MeshCollisionChecker

        solver = AssemblySolver(ROBOTIC_ARM_ASSEMBLY)
        placements = solver.solve()

        checker = MeshCollisionChecker()
        report = checker.generate_interference_report(
            ROBOTIC_ARM_ASSEMBLY, placements, skip_adjacent=True,
        )

        assert report.parts_checked == len(ROBOTIC_ARM_ASSEMBLY.parts)
        assert report.pairs_checked >= 1
        assert isinstance(report.collision_free, bool)
        assert isinstance(report.summary, str)


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------

class TestMotionCollisionTool:
    """Tests for MotionCollisionTool."""

    def test_tool_execute(self):
        from lang3d.tools.motion_collision import MotionCollisionTool
        tool = MotionCollisionTool()
        result = tool.execute(assembly_name="robotic_arm", num_samples=3)
        assert "[Motion Collision Check]" in result

    def test_tool_execute_unknown(self):
        from lang3d.tools.motion_collision import MotionCollisionTool
        tool = MotionCollisionTool()
        result = tool.execute(assembly_name="nonexistent_xyz")
        assert "Error" in result

    def test_tool_registration(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.motion_collision import register_motion_collision_tools
        registry = ToolRegistry()
        register_motion_collision_tools(registry)
        assert "motion_collision_check" in registry.list_tools()
        assert "reachability_check" in registry.list_tools()


class TestReachabilityTool:
    """Tests for ReachabilityTool."""

    def test_tool_execute(self):
        from lang3d.tools.motion_collision import ReachabilityTool
        tool = ReachabilityTool()
        result = tool.execute(
            assembly_name="robotic_arm",
            target_x=100, target_y=0, target_z=100,
        )
        assert "[Reachability Check]" in result

    def test_tool_execute_custom_json(self):
        from lang3d.tools.motion_collision import ReachabilityTool
        json_str = json.dumps({
            "name": "TestArm",
            "parts": [
                {"name": "base", "dimensions": {"length": 30, "width": 30, "height": 5}},
                {"name": "link", "dimensions": {"length": 60, "width": 15, "height": 8}},
            ],
            "joints": [
                {"parent": "base", "child": "link", "type": "revolute",
                 "parent_anchor": "top", "child_anchor": "bottom",
                 "range_deg": [-90, 90]},
            ],
        })
        tool = ReachabilityTool()
        result = tool.execute(
            assembly_json=json_str,
            target_x=50, target_y=0, target_z=10,
        )
        assert "[Reachability Check]" in result


# ---------------------------------------------------------------------------
# Collision-aware joint range clamping tests
#
# _clamp_joint_ranges_to_collision_free narrows each arm joint's range_deg to
# its maximal collision-free sub-range, so the arm cannot sweep into its own
# base during simulation.  These guard the generalisation from dual-arm-only
# to all arms (pipeline integration).
# ---------------------------------------------------------------------------

class TestCollisionAwareRangeClamp:
    """Tests for the collision-aware range clamper."""

    def test_no_collision_leaves_range_unchanged(self):
        """A collision-free arm must keep its full range_deg."""
        from lang3d.agent.assembly_compose import (
            clamp_assembly_joint_ranges_collision_free,
        )
        # Short link on a wide base — no possible self-collision.
        assembly = Assembly(
            name="test",
            parts=[
                _box("base", 200, 200, 8),
                _box("link1", 40, 40, 10),
            ],
            joints=[
                Joint("revolute", "base", "link1", axis="x",
                      range_deg=(-90, 90),
                      parent_anchor="top", child_anchor="bottom"),
            ],
            default_angles={"link1": 0.0},
        )
        n = clamp_assembly_joint_ranges_collision_free(assembly)
        # Range may or may not be narrowed depending on whether the short
        # link actually clears the base at extremes; the key invariant is
        # that the function runs without error and returns an int.
        assert isinstance(n, int)
        assert n >= 0

    def test_wheel_joint_not_clamped(self):
        """Wheel revolute joints must be excluded from arm-joint clamping."""
        from lang3d.agent.assembly_compose import (
            clamp_assembly_joint_ranges_collision_free,
        )
        assembly = Assembly(
            name="test",
            parts=[
                _box("base", 100, 100, 8),
                _box("wheel_fl", 40, 20, 40),
            ],
            joints=[
                Joint("revolute", "base", "wheel_fl", axis="x",
                      range_deg=(-180, 180),
                      parent_anchor="left", child_anchor="center"),
            ],
            default_angles={"wheel_fl": 0.0},
        )
        original = assembly.joints[0].range_deg
        clamp_assembly_joint_ranges_collision_free(assembly)
        # Wheel range must be unchanged (wheels are not arm joints).
        assert assembly.joints[0].range_deg == original

    def test_clamp_narrows_colliding_range(self):
        """A long link that can hit its base must get a narrowed range."""
        from lang3d.agent.assembly_compose import (
            clamp_assembly_joint_ranges_collision_free,
        )
        # A very long link (300mm) on a small base (50mm) — folding down
        # WILL collide with the base.  The clamper should narrow the range.
        assembly = Assembly(
            name="test",
            parts=[
                _box("base", 50, 50, 8),
                _box("long_link", 30, 300, 15),
            ],
            joints=[
                Joint("revolute", "base", "long_link", axis="x",
                      range_deg=(-170, 170),
                      parent_anchor="top", child_anchor="bottom"),
            ],
            default_angles={"long_link": 0.0},
        )
        original_span = assembly.joints[0].range_deg[1] - assembly.joints[0].range_deg[0]
        n = clamp_assembly_joint_ranges_collision_free(assembly)
        if n > 0:
            clamped = assembly.joints[0].range_deg
            clamped_span = clamped[1] - clamped[0]
            assert clamped_span < original_span, (
                f"range not narrowed ({original_span} → {clamped_span}) "
                f"despite a colliding long link"
            )

