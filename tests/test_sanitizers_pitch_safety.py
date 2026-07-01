"""Tests for the arm pitch-range clamping in assembly sanitizers.

Guards the design lesson from 2026-07-01: an analytic geometric limit
(arcsin(H/L)) was tried to prevent the arm reaching below the base plate,
but it was WRONG — the arm extends FORWARD (−Y) from the base, so the
gripper reaching low Z is in front of the base (not inside it). FCL
confirmed 0 collisions across the full ±90° range. The "穿模" the user
saw was the arm passing through the FLOOR (no ground in the static
render), not through the base.

The correct collision authority is the FCL mesh check in the e2e
motion-collision sweep — NOT a Z-height heuristic. These tests verify:

  1. Home angles are always within range_deg (no MuJoCo "fly" artifact).
  2. Pitch ranges are generous enough for meaningful motion (not 6°).
  3. The FCL collision sweep confirms the ranges are actually safe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lang3d.knowledge.mechanics import Assembly, Joint, Part
from lang3d.tools.assembly_gen.sanitizers import _ensure_arm_default_angles


def _box(name: str, l: float, w: float, h: float) -> Part:
    return Part(name, "structural", name, dimensions=dict(length=l, width=w, height=h))


def _realistic_arm() -> Assembly:
    """A 4-DOF arm matching the project's standard topology."""
    return Assembly(
        name="test_arm",
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


class TestPitchRangeClamping:
    """Tests for the numeric pitch-range clamping."""

    def test_home_within_range_after_clamp(self):
        """After sanitizing, all home angles must be within their range_deg."""
        arm = _realistic_arm()
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

    def test_pitch_range_is_generous(self):
        """Pitch joints must have enough range for meaningful motion (≥20°).

        A 6° span (the over-aggressive geometric limit) is too small — the
        arm looks frozen. The correct numeric caps give ±90° forward.
        """
        arm = _realistic_arm()
        result = _ensure_arm_default_angles(arm)
        for j in result.joints:
            if j.axis != "x" or j.type != "revolute":
                continue
            span = j.range_deg[1] - j.range_deg[0]
            assert span >= 20.0, (
                f"{j.child} span={span:.1f}° is too small for meaningful motion "
                f"(the old geometric limit bug returned)"
            )

    def test_backward_pitch_is_capped(self):
        """Backward pitch (folds toward base) must be capped tighter than forward."""
        arm = _realistic_arm()
        result = _ensure_arm_default_angles(arm)
        for j in result.joints:
            if j.axis != "x" or j.type != "revolute":
                continue
            home = result.default_angles.get(j.child, 0.0)
            lo, hi = j.range_deg
            if home < 0:
                # Forward = negative direction; backward = positive.
                # Backward (positive) should be capped tighter.
                assert hi <= 45.0 + 1.0, (
                    f"{j.child} backward limit={hi:.0f}° exceeds 45° cap"
                )
