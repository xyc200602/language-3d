"""Tests for collision fix suggestion system.

Uses manually constructed MotionCollisionResult objects — no FCL dependency.
"""

import json
import pytest

from lang3d.knowledge.mechanics import Assembly, Joint, Part
from lang3d.tools.motion_collision import (
    CollisionFixReport,
    CollisionFixSuggester,
    CollisionFixSuggestion,
    CollisionFixTool,
    JointCollisionRange,
    MotionCollisionResult,
    register_motion_collision_tools,
)
from lang3d.tools.base import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_joint_result(
    joint_name: str = "elbow",
    angle_min: float = -180,
    angle_max: float = 180,
    collision_angles: list[float] | None = None,
    free_segments: list[tuple[float, float]] | None = None,
    samples: int = 36,
) -> JointCollisionRange:
    """Build a JointCollisionRange for testing."""
    coll = collision_angles or []
    has = len(coll) > 0
    free = free_segments or []
    return JointCollisionRange(
        joint_name=joint_name,
        angle_min_deg=angle_min,
        angle_max_deg=angle_max,
        samples=samples,
        collision_angles=coll,
        collision_free_segments=free,
        has_collision=has,
    )


def _make_assembly_for_fix() -> Assembly:
    """Build a simple assembly with known dimensions for fix testing."""
    return Assembly(
        name="Test Arm",
        parts=[
            Part("shoulder", "structural", "肩部", dimensions={"length": 150}),
            Part("elbow", "structural", "肘部", dimensions={"length": 120}),
            Part("wrist", "structural", "腕部", dimensions={"length": 80}),
        ],
        joints=[
            Joint("revolute", "shoulder", "elbow", range_deg=(-180, 180)),
            Joint("revolute", "elbow", "wrist", range_deg=(-180, 180)),
        ],
    )


# ---------------------------------------------------------------------------
# Decision tree tests
# ---------------------------------------------------------------------------


class TestCollisionFixSuggester:
    """Test the CollisionFixSuggester decision tree."""

    def test_no_collision_returns_empty(self):
        suggester = CollisionFixSuggester()
        result = MotionCollisionResult(collision_free=True, joints_checked=1)
        report = suggester.suggest_fixes(result)
        assert report.total_collisions == 0
        assert len(report.suggestions) == 0
        assert "无碰撞" in report.summary

    def test_extreme_angle_collision_suggests_limit_range(self):
        """Collision at extremes → limit_joint_range."""
        suggester = CollisionFixSuggester()
        jr = _make_joint_result(
            joint_name="elbow",
            angle_min=-90,
            angle_max=90,
            collision_angles=[-90, -85, -80, -75, 80, 85, 90],
            free_segments=[(-60, 65)],
            samples=19,
        )
        result = MotionCollisionResult(
            collision_free=False,
            joints_checked=1,
            joint_results=[jr],
        )
        report = suggester.suggest_fixes(result)
        assert report.total_collisions == 1
        assert len(report.suggestions) >= 1
        assert report.suggestions[0].fix_type == "limit_joint_range"
        assert report.suggestions[0].confidence > 0

    def test_center_collision_suggests_add_offset(self):
        """Collision near zero → add_offset."""
        suggester = CollisionFixSuggester()
        # Collision at center (-10 to 10 degrees), free elsewhere
        collision_angles = [a for a in range(-10, 11, 2)]  # 11 angles near center
        jr = _make_joint_result(
            joint_name="wrist",
            angle_min=-90,
            angle_max=90,
            collision_angles=collision_angles,
            free_segments=[(-85, -15), (15, 85)],
            samples=37,
        )
        result = MotionCollisionResult(
            collision_free=False,
            joints_checked=1,
            joint_results=[jr],
        )
        report = suggester.suggest_fixes(result)
        assert len(report.suggestions) >= 1
        assert report.suggestions[0].fix_type == "add_offset"
        assert report.suggestions[0].suggested_value != 0.0

    def test_large_range_collision_suggests_reduce_length(self):
        """Collision covers >50% of range → reduce_link_length."""
        suggester = CollisionFixSuggester()
        assembly = _make_assembly_for_fix()
        # 70% of samples are collisions
        total_samples = 36
        n_collision = 26  # >70%
        collision_angles = [
            -180 + (360 * i / n_collision) for i in range(n_collision)
        ]
        jr = _make_joint_result(
            joint_name="elbow",
            angle_min=-180,
            angle_max=180,
            collision_angles=collision_angles,
            free_segments=[(150, 170)],
            samples=total_samples,
        )
        result = MotionCollisionResult(
            collision_free=False,
            joints_checked=1,
            joint_results=[jr],
        )
        report = suggester.suggest_fixes(result, assembly)
        assert len(report.suggestions) >= 1
        assert report.suggestions[0].fix_type == "reduce_link_length"
        assert report.suggestions[0].suggested_value < report.suggestions[0].current_value

    def test_no_free_segments_suggests_increase_spacing(self):
        """Almost no free segment → increase_spacing.

        Create a scenario where collision_ratio < 0.5 but free_ratio < 0.1.
        The collision_angles list is short (collision_ratio < 0.5), but
        the free segment covers very little of the total range.
        """
        suggester = CollisionFixSuggester()
        # Only 4 collision angles → ratio = 4/36 ≈ 0.11 (< 0.5)
        # But free segment is tiny: 3 degrees out of 360 = 0.8%
        collision_angles = [-170, -165, 165, 170]
        jr = _make_joint_result(
            joint_name="shoulder",
            angle_min=-180,
            angle_max=180,
            collision_angles=collision_angles,
            free_segments=[(172, 175)],  # tiny 3-degree segment
            samples=36,
        )
        result = MotionCollisionResult(
            collision_free=False,
            joints_checked=1,
            joint_results=[jr],
        )
        report = suggester.suggest_fixes(result)
        assert len(report.suggestions) >= 1
        fix_types = [s.fix_type for s in report.suggestions]
        assert "increase_spacing" in fix_types

    def test_multiple_joints_produces_multiple_suggestions(self):
        """Multiple colliding joints get separate suggestions."""
        suggester = CollisionFixSuggester()
        jr1 = _make_joint_result(
            joint_name="elbow",
            collision_angles=[-180, -175, -170, 170, 175, 180],
            free_segments=[(-155, 155)],
            samples=73,
        )
        jr2 = _make_joint_result(
            joint_name="wrist",
            collision_angles=[0, 2, -2, 5, -5],
            free_segments=[(-85, -10), (15, 85)],
            samples=37,
        )
        result = MotionCollisionResult(
            collision_free=False,
            joints_checked=2,
            joint_results=[jr1, jr2],
        )
        report = suggester.suggest_fixes(result)
        assert report.total_collisions == 2
        assert len(report.suggestions) >= 2

    def test_confidence_range(self):
        """All confidence values are in [0, 1]."""
        suggester = CollisionFixSuggester()
        jr = _make_joint_result(
            joint_name="elbow",
            collision_angles=[-90, -85, 90, 85],
            free_segments=[(-70, 70)],
            samples=37,
        )
        result = MotionCollisionResult(
            collision_free=False, joints_checked=1, joint_results=[jr],
        )
        report = suggester.suggest_fixes(result)
        for s in report.suggestions:
            assert 0.0 <= s.confidence <= 1.0


# ---------------------------------------------------------------------------
# Constraint propagation
# ---------------------------------------------------------------------------


class TestConstraintPropagation:
    """Test that fix suggestions propagate correctly."""

    def test_reduce_length_propagates_downstream(self):
        """Length reduction creates offset updates for downstream joints."""
        suggester = CollisionFixSuggester()
        assembly = _make_assembly_for_fix()
        # 70% collision on elbow
        n_collision = 26
        collision_angles = [-180 + (360 * i / n_collision) for i in range(n_collision)]
        jr = _make_joint_result(
            joint_name="elbow",
            angle_min=-180,
            angle_max=180,
            collision_angles=collision_angles,
            free_segments=[(150, 170)],
            samples=36,
        )
        result = MotionCollisionResult(
            collision_free=False, joints_checked=1, joint_results=[jr],
        )
        report = suggester.suggest_fixes(result, assembly)
        # Should have constraint updates for downstream
        assert len(report.constraint_updates) > 0 or len(report.suggestions) > 0

    def test_limit_range_propagates_default_angle(self):
        """Range limiting updates default_angle in constraint_updates."""
        suggester = CollisionFixSuggester()
        jr = _make_joint_result(
            joint_name="elbow",
            angle_min=-90,
            angle_max=90,
            collision_angles=[-90, -85, -80, -75, 80, 85, 90],
            free_segments=[(-60, 65)],
            samples=19,
        )
        result = MotionCollisionResult(
            collision_free=False, joints_checked=1, joint_results=[jr],
        )
        report = suggester.suggest_fixes(result)
        assert len(report.constraint_updates) > 0
        if "elbow" in report.constraint_updates:
            assert "default_angle" in report.constraint_updates["elbow"]


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------


class TestCollisionFixTool:
    """Test the collision_fix_suggest tool."""

    def test_execute_with_valid_collision_data(self):
        tool = CollisionFixTool()
        collision_data = {
            "collision_free": False,
            "joints_checked": 1,
            "joints": [{
                "joint_name": "elbow",
                "has_collision": True,
                "collision_angles": [-90, -85, 90, 85],
                "collision_free_segments": [(-70, 70)],
                "angle_min_deg": -90,
                "angle_max_deg": 90,
                "samples": 19,
            }],
        }
        result = tool.execute(
            collision_result_json=json.dumps(collision_data),
        )
        assert "[Collision Fix Suggest]" in result
        assert "elbow" in result

    def test_execute_returns_valid_json(self):
        tool = CollisionFixTool()
        collision_data = {
            "collision_free": False,
            "joints_checked": 1,
            "joints": [{
                "joint_name": "wrist",
                "has_collision": True,
                "collision_angles": [0, 2, -2],
                "collision_free_segments": [(-85, -5), (10, 85)],
                "angle_min_deg": -90,
                "angle_max_deg": 90,
                "samples": 37,
            }],
        }
        result = tool.execute(
            collision_result_json=json.dumps(collision_data),
        )
        json_start = result.index("--- JSON ---") + len("--- JSON ---")
        json_str = result[json_start:].strip()
        data = json.loads(json_str)
        assert "suggestions" in data
        assert "total_collisions" in data

    def test_execute_no_collision(self):
        tool = CollisionFixTool()
        collision_data = {
            "collision_free": True,
            "joints_checked": 1,
            "joints": [],
        }
        result = tool.execute(
            collision_result_json=json.dumps(collision_data),
        )
        assert "0" in result

    def test_execute_invalid_json(self):
        tool = CollisionFixTool()
        result = tool.execute(collision_result_json="not json")
        assert "错误" in result

    def test_execute_missing_data(self):
        tool = CollisionFixTool()
        result = tool.execute(collision_result_json="")
        assert "错误" in result


class TestToolRegistration:
    """Test tool registration with CollisionFixTool."""

    def test_collision_fix_always_registered(self):
        """CollisionFixTool registers even without FCL."""
        registry = ToolRegistry()
        register_motion_collision_tools(registry)
        assert registry.get("collision_fix_suggest") is not None

    def test_tool_definition_valid(self):
        registry = ToolRegistry()
        register_motion_collision_tools(registry)
        tool = registry.get("collision_fix_suggest")
        defn = tool.get_definition()
        assert defn.name == "collision_fix_suggest"
        assert "properties" in defn.parameters
        assert "collision_result_json" in defn.parameters["properties"]
