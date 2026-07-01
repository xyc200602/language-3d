"""Tests for the collision-aware pitch range clamping in assembly sanitizers.

These guard the fix for the "arm sweeps through the base plate during motion"
defect (2026-07-01).  The sanitizer now computes a geometric pitch limit
from the arm's height above the base and downstream link length, then
verifies the COMBINED worst-case configuration (all pitch joints at their
positive limits) stays above the base plate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lang3d.knowledge.mechanics import Assembly, Joint, Part
from lang3d.tools.assembly_gen.sanitizers import (
    _compute_safe_pitch_range,
    _count_downstream_pitch_joints,
    _downstream_link_length,
    _ensure_arm_default_angles,
    _joint_height_above_base,
)


def _box(name: str, l: float, w: float, h: float) -> Part:
    return Part(name, "structural", name, dimensions=dict(length=l, width=w, height=h))


def _simple_arm() -> Assembly:
    """A minimal 2-DOF arm: base_plate → shoulder → link → elbow → forearm."""
    return Assembly(
        name="test_arm",
        parts=[
            _box("base_plate", 200, 150, 8),
            _box("shoulder_servo", 40, 20, 40),
            _box("upper_link", 120, 30, 15),
            _box("elbow_servo", 40, 20, 40),
            _box("lower_link", 100, 25, 12),
        ],
        joints=[
            Joint("revolute", "base_plate", "shoulder_servo", axis="z",
                  range_deg=(-90, 90), parent_anchor="top", child_anchor="bottom"),
            Joint("fixed", "shoulder_servo", "upper_link",
                  parent_anchor="front", child_anchor="back"),
            Joint("revolute", "upper_link", "elbow_servo", axis="x",
                  range_deg=(-180, 180), parent_anchor="front", child_anchor="back"),
            Joint("fixed", "elbow_servo", "lower_link",
                  parent_anchor="front", child_anchor="back"),
        ],
        default_angles={},
    )


class TestGeometricPitchLimit:
    """Tests for _compute_safe_pitch_range and helpers."""

    def test_downstream_link_length_sums_links(self):
        arm = _simple_arm()
        L = _downstream_link_length(arm.parts, arm.joints, "elbow_servo")
        assert L == pytest.approx(100.0)  # only lower_link downstream

    def test_joint_height_above_base(self):
        arm = _simple_arm()
        H = _joint_height_above_base(arm.parts, arm.joints, "elbow_servo")
        # elbow is downstream of shoulder (40mm) + base (8mm) + upper_link (horizontal, 0)
        assert H > 40.0  # at least the servo + base height

    def test_count_downstream_pitch_joints(self):
        arm = _simple_arm()
        # elbow has 0 downstream pitch joints
        assert _count_downstream_pitch_joints(arm.joints, "elbow_servo") == 0
        # shoulder has 1 downstream pitch joint (elbow)
        assert _count_downstream_pitch_joints(arm.joints, "shoulder_servo") == 1

    def test_safe_range_returns_none_for_no_geometry(self):
        """A joint with no downstream links returns None (can't compute)."""
        arm = Assembly(
            name="t", parts=[_box("base", 10, 10, 10)],
            joints=[], default_angles={},
        )
        assert _compute_safe_pitch_range(arm.parts, arm.joints, "base") is None

    def test_safe_range_is_bounded(self):
        """The safe pitch range should be reasonable (not ±180°)."""
        arm = _simple_arm()
        geo = _compute_safe_pitch_range(arm.parts, arm.joints, "elbow_servo")
        assert geo is not None
        back, fwd = geo
        # Forward fold should be less than 90° (can't fold past vertical safely)
        assert fwd < 90.0
        assert fwd > 0.0
        assert back > 0.0


class TestCombinedPitchSafety:
    """Tests that _ensure_arm_default_angles prevents arm-base interpenetration."""

    def test_home_within_range_after_clamp(self):
        """After sanitizing, all home angles must be within their range_deg."""
        arm = _simple_arm()
        # Give the shoulder a wide range + home at a dangerous angle.
        arm.joints[1].range_deg = (-90, 90)
        arm.joints[2].range_deg = (-180, 180)
        arm.default_angles = {"shoulder_servo": -20.0, "elbow_servo": 30.0}
        result = _ensure_arm_default_angles(arm)
        for j in result.joints:
            if j.type != "revolute" or not j.range_deg:
                continue
            home = result.default_angles.get(j.child)
            if home is not None:
                lo, hi = j.range_deg
                assert lo <= home <= hi, (
                    f"{j.child} home={home} outside range [{lo}, {hi}]"
                )

    def test_arm_cannot_reach_below_base(self):
        """After sanitizing, no joint combination drives the gripper below base.

        This is the DEFINITIVE test: solve the assembly at every corner of the
        joint range (all pitch joints at min, max, and home) and assert the
        end-effector stays above the base plate top.
        """
        from lang3d.tools.assembly_solver import AssemblySolver

        # Build a realistic arm that WOULD penetrate at wide ranges.
        arm = Assembly(
            name="test",
            parts=[
                _box("base_plate", 200, 150, 8),
                _box("base_yaw_servo", 40, 20, 40),
                _box("yaw_link", 40, 40, 20),
                _box("shoulder_pitch_servo", 40, 20, 40),
                _box("upper_arm_link", 120, 30, 15),
                _box("elbow_pitch_servo", 40, 20, 40),
                _box("lower_arm_link", 100, 25, 12),
                _box("wrist_roll_servo", 40, 20, 30),
                _box("gripper_base", 30, 40, 25),
                _box("gripper_finger_left", 50, 10, 20),
            ],
            joints=[
                Joint("revolute", "base_plate", "base_yaw_servo", axis="z",
                      range_deg=(-90, 90), parent_anchor="top", child_anchor="bottom"),
                Joint("fixed", "base_yaw_servo", "yaw_link",
                      parent_anchor="front", child_anchor="back"),
                Joint("revolute", "yaw_link", "shoulder_pitch_servo", axis="x",
                      range_deg=(-90, 90), parent_anchor="front", child_anchor="back"),
                Joint("fixed", "shoulder_pitch_servo", "upper_arm_link",
                      parent_anchor="front", child_anchor="back"),
                Joint("revolute", "upper_arm_link", "elbow_pitch_servo", axis="x",
                      range_deg=(-180, 180), parent_anchor="front", child_anchor="back"),
                Joint("fixed", "elbow_pitch_servo", "lower_arm_link",
                      parent_anchor="front", child_anchor="back"),
                Joint("revolute", "lower_arm_link", "wrist_roll_servo", axis="y",
                      range_deg=(-180, 180), parent_anchor="front", child_anchor="back"),
                Joint("fixed", "wrist_roll_servo", "gripper_base",
                      parent_anchor="front", child_anchor="back"),
                Joint("fixed", "gripper_base", "gripper_finger_left",
                      parent_anchor="front", child_anchor="back"),
            ],
            default_angles={
                "base_yaw_servo": 0.0,
                "shoulder_pitch_servo": -20.0,
                "elbow_pitch_servo": 30.0,
                "wrist_roll_servo": 0.0,
            },
        )

        arm = _ensure_arm_default_angles(arm)

        # Base plate top Z.
        base = next(p for p in arm.parts if p.name == "base_plate")
        base_top = base.dimensions["height"] / 2.0

        # Check ALL corner combinations of pitch joints.
        pitch_joints = [
            j for j in arm.joints
            if j.axis == "x" and j.type == "revolute"
        ]
        import itertools
        configs = []
        for j in pitch_joints:
            lo, hi = j.range_deg
            home = arm.default_angles.get(j.child, 0.0)
            configs.append([lo, home, hi])

        worst_z = float("inf")
        for combo in itertools.product(*configs):
            da = dict(arm.default_angles)
            for j, val in zip(pitch_joints, combo):
                da[j.child] = val
            trial = Assembly(
                name="t", parts=arm.parts, joints=arm.joints,
                default_angles=da,
            )
            pos = AssemblySolver(trial).solve()
            for nm in ("gripper_finger_left", "gripper_base", "lower_arm_link"):
                if nm in pos:
                    z = pos[nm]["position"][2]
                    worst_z = min(worst_z, z)

        assert worst_z >= base_top, (
            f"End-effector reached Z={worst_z:.1f}mm, below base top "
            f"({base_top}mm) — arm CAN interpenetrate the base during motion"
        )
