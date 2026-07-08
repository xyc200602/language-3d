"""Tests for the humanoid-topology and wrist-roll range clamping in sanitizers.

Background (external audit + humanoid_2leg_2arm e2e failures):
The generic arm pitch caps (±90° forward) were calibrated on fixed-base
arms. On a humanoid torso+pelvis, the shoulder's forward-pitch extreme
sweeps the upper arm through the hip/pelvis volume — a self-collision
absent on fixed-base arms. The humanoid case had 3 motion-collision
failures (left/right shoulder_pitch at +90°, left_wrist_roll at ±180°).

These tests verify the topology-aware fixes:

1. **Humanoid detection**: pelvis + thigh/shin/knee parts trigger the
   tighter forward pitch cap (60°, not 90°).
2. **Fixed-base unaffected**: an arm WITHOUT pelvis/leg parts keeps the
   generous ±90° forward cap.
3. **Wrist-roll clamp**: a wrist-roll joint (axis=y, name contains
   "wrist"/"roll") is clamped to ±120° regardless of topology, so the
   gripper can't rotate into the forearm at ±180°.
4. **Home-within-range invariant**: after clamping, home angles remain
   inside range_deg (no MuJoCo "fly" artifact).
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
    _ensure_arm_default_angles,
    apply_default_connection_methods,
)


def _box(name: str, l: float, w: float, h: float) -> Part:
    return Part(name, "structural", name, dimensions=dict(length=l, width=w, height=h))


def _humanoid_arm() -> Assembly:
    """A minimal humanoid: pelvis + one leg chain + one arm chain.

    Mirrors the topology of humanoid_2leg_2arm (which failed with 3
    motion collisions on shoulder_pitch +90° and wrist_roll ±180°).
    One leg + one arm is enough to trigger the topology detector — the
    detector keys on PRESENCE of pelvis+leg parts, not on symmetry.
    """
    return Assembly(
        name="humanoid_test",
        parts=[
            _box("pelvis", 120, 80, 40),
            _box("torso_link", 100, 60, 120),
            # Left leg chain
            _box("left_hip_yaw", 40, 40, 40),
            _box("left_thigh_link", 50, 50, 150),
            _box("left_knee_pitch", 40, 40, 40),
            _box("left_shin_link", 40, 40, 140),
            _box("left_ankle_pitch", 40, 40, 30),
            _box("left_foot", 80, 50, 20),
            # Left arm chain
            _box("left_shoulder_pitch", 40, 40, 40),
            _box("left_upper_arm", 40, 40, 120),
            _box("left_elbow_pitch", 40, 40, 40),
            _box("left_lower_arm", 35, 35, 100),
            _box("left_wrist_roll", 35, 35, 30),
            _box("gripper_base", 30, 40, 25),
        ],
        joints=[
            # Torso
            Joint("fixed", "pelvis", "torso_link",
                  parent_anchor="top", child_anchor="bottom"),
            # Left leg
            Joint("revolute", "pelvis", "left_hip_yaw", axis="z",
                  range_deg=(-90, 90), parent_anchor="bottom", child_anchor="top"),
            Joint("fixed", "left_hip_yaw", "left_thigh_link",
                  parent_anchor="front", child_anchor="back"),
            Joint("revolute", "left_thigh_link", "left_knee_pitch", axis="x",
                  range_deg=(-90, 45), parent_anchor="bottom", child_anchor="top"),
            Joint("fixed", "left_knee_pitch", "left_shin_link",
                  parent_anchor="bottom", child_anchor="top"),
            Joint("revolute", "left_shin_link", "left_ankle_pitch", axis="x",
                  range_deg=(-45, 45), parent_anchor="bottom", child_anchor="top"),
            Joint("fixed", "left_ankle_pitch", "left_foot",
                  parent_anchor="front", child_anchor="back"),
            # Left arm (off torso)
            Joint("revolute", "torso_link", "left_shoulder_pitch", axis="x",
                  range_deg=(-45, 90), parent_anchor="top", child_anchor="bottom"),
            Joint("fixed", "left_shoulder_pitch", "left_upper_arm",
                  parent_anchor="front", child_anchor="back"),
            Joint("revolute", "left_upper_arm", "left_elbow_pitch", axis="x",
                  range_deg=(-45, 0), parent_anchor="front", child_anchor="back"),
            Joint("fixed", "left_elbow_pitch", "left_lower_arm",
                  parent_anchor="front", child_anchor="back"),
            Joint("revolute", "left_lower_arm", "left_wrist_roll", axis="y",
                  range_deg=(-180, 180), parent_anchor="front", child_anchor="back"),
            Joint("fixed", "left_wrist_roll", "gripper_base",
                  parent_anchor="front", child_anchor="back"),
        ],
        default_angles={
            "left_shoulder_pitch": 40.0,
            "left_elbow_pitch": 0.0,
            "left_wrist_roll": 0.0,
            "left_knee_pitch": -30.0,
        },
    )


def _fixed_base_arm() -> Assembly:
    """A fixed-base arm with NO pelvis/leg parts — the original calibration
    target. The humanoid caps must NOT apply here."""
    return Assembly(
        name="fixed_arm",
        parts=[
            _box("base_plate", 200, 150, 8),
            _box("shoulder_pitch", 40, 40, 40),
            _box("upper_arm", 40, 40, 120),
            _box("wrist_roll", 35, 35, 30),
        ],
        joints=[
            Joint("revolute", "base_plate", "shoulder_pitch", axis="x",
                  range_deg=(-90, 90), parent_anchor="top", child_anchor="bottom"),
            Joint("fixed", "shoulder_pitch", "upper_arm",
                  parent_anchor="front", child_anchor="back"),
            Joint("revolute", "upper_arm", "wrist_roll", axis="y",
                  range_deg=(-180, 180), parent_anchor="front", child_anchor="back"),
        ],
        default_angles={"shoulder_pitch": -20.0, "wrist_roll": 0.0},
    )


# ---------------------------------------------------------------------------
# Humanoid shoulder pitch: tighter forward cap
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHumanoidShoulderPitchCap:
    def test_humanoid_forward_pitch_capped_below_90(self):
        """On a humanoid, the shoulder's forward (+) extreme must be < 90°.

        The +90° extreme is where the upper arm swings into the pelvis/hip
        (the humanoid-specific collision). The topology-aware cap tightens
        this to 60°.
        """
        humanoid = _humanoid_arm()
        result = _ensure_arm_default_angles(humanoid)
        shoulder = next(
            j for j in result.joints if j.child == "left_shoulder_pitch"
        )
        lo, hi = shoulder.range_deg
        # Forward (positive, since home=40>0) must be capped at the humanoid
        # 60° cap, NOT the generic 90° cap.
        assert hi <= 60.0 + 0.5, (
            f"humanoid shoulder forward limit={hi:.0f}° — should be ≤60° "
            f"(generic 90° lets the arm clip the pelvis)"
        )

    def test_fixed_base_keeps_generous_90_cap(self):
        """A fixed-base arm (no pelvis/legs) keeps the ±90° forward cap.

        The humanoid caps must NOT regress the original calibration target.
        """
        arm = _fixed_base_arm()
        result = _ensure_arm_default_angles(arm)
        shoulder = next(
            j for j in result.joints if j.child == "shoulder_pitch"
        )
        lo, hi = shoulder.range_deg
        # Home is -20 (< 0), so forward = negative direction. The forward
        # extreme (lo) should keep the generic -90° cap.
        assert lo >= -90.0 - 0.5, (
            f"fixed-base shoulder forward limit={lo:.0f}° — humanoid cap "
            f"wrongly applied to a non-humanoid arm"
        )


# ---------------------------------------------------------------------------
# Wrist-roll clamp (topology-independent)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWristRollClamp:
    def test_wrist_roll_clamped_to_120(self):
        """A ±180° wrist-roll must be clamped to ±120° (avoid forearm clip)."""
        humanoid = _humanoid_arm()
        result = _ensure_arm_default_angles(humanoid)
        wrist = next(j for j in result.joints if j.child == "left_wrist_roll")
        lo, hi = wrist.range_deg
        assert lo >= -120.0 - 0.5, f"wrist lo={lo:.0f}° — should be ≥-120°"
        assert hi <= 120.0 + 0.5, f"wrist hi={hi:.0f}° — should be ≤120°"

    def test_wrist_roll_clamp_applies_on_fixed_base_too(self):
        """The wrist-roll clamp is topology-independent — it applies to
        fixed-base arms too (the forearm geometry is the same)."""
        arm = _fixed_base_arm()
        result = _ensure_arm_default_angles(arm)
        wrist = next(j for j in result.joints if j.child == "wrist_roll")
        lo, hi = wrist.range_deg
        assert hi <= 120.0 + 0.5, f"fixed-base wrist hi={hi:.0f}° — should be ≤120°"


# ---------------------------------------------------------------------------
# Home-within-range invariant (the MuJoCo-fly guard)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_humanoid_home_angles_within_range():
    """After all clamping, every home angle must lie inside its range_deg.

    This is the load-bearing invariant for MuJoCo stability: if home is
    outside range, the PD controller yanks the joint to the limit on step 1,
    causing the 'fly' artifact.
    """
    humanoid = _humanoid_arm()
    result = _ensure_arm_default_angles(humanoid)
    for j in result.joints:
        if j.type != "revolute" or not j.range_deg:
            continue
        home = result.default_angles.get(j.child)
        if home is None:
            continue
        lo, hi = j.range_deg
        assert lo - 0.5 <= home <= hi + 0.5, (
            f"{j.child} home={home} outside range [{lo}, {hi}] — "
            f"MuJoCo will fly on step 1"
        )


# ---------------------------------------------------------------------------
# Adhesive connection triggers (sensor + foot)
# ---------------------------------------------------------------------------


def _assembly_with_adhesive_parts() -> Assembly:
    """A minimal assembly with a sensor (camera) and a foot pad.

    Mirrors the humanoid_2leg_2arm topology where these connection
    methods should fire: camera_module (category=sensor) bonded to a
    torso bracket, and left_foot bonded to an ankle.
    """
    return Assembly(
        name="adhesive_test",
        parts=[
            Part("torso_link", "structural", "torso",
                 dimensions=dict(length=60, width=40, height=200)),
            Part("camera_module", "sensor", "camera",
                 dimensions=dict(length=30, width=30, height=20)),
            Part("ankle_link", "structural", "ankle",
                 dimensions=dict(length=40, width=30, height=30)),
            Part("left_foot", "structural", "foot pad",
                 dimensions=dict(length=100, width=60, height=10)),
            Part("base_bracket", "structural", "bracket",
                 dimensions=dict(length=50, width=50, height=8)),
        ],
        joints=[
            # Sensor → should be adhesive
            Joint("fixed", "torso_link", "camera_module",
                  parent_anchor="front", child_anchor="back"),
            # Foot pad → should be adhesive
            Joint("fixed", "ankle_link", "left_foot",
                  parent_anchor="bottom", child_anchor="top"),
            # Normal structural → should stay bolted
            Joint("fixed", "base_bracket", "torso_link",
                  parent_anchor="top", child_anchor="bottom"),
        ],
        default_angles={},
    )


@pytest.mark.unit
class TestAdhesiveConnectionTriggers:
    def test_sensor_joint_gets_adhesive(self):
        """A sensor (category=sensor) on a fixed joint should default to
        adhesive, not bolted — sensors bond to brackets in real robotics."""
        asm = _assembly_with_adhesive_parts()
        apply_default_connection_methods(asm.joints, parts=asm.parts)
        cam_joint = next(j for j in asm.joints if j.child == "camera_module")
        assert cam_joint.connection is not None, "sensor joint has no connection"
        assert cam_joint.connection.type == "adhesive", (
            f"sensor joint type={cam_joint.connection.type} — should be adhesive"
        )
        assert cam_joint.connection.adhesive_type == "epoxy"

    def test_foot_joint_gets_adhesive(self):
        """A foot pad (name contains 'foot') on a fixed joint should default
        to adhesive — rubber/TPU pads bond to plates (cf. ANYmal B)."""
        asm = _assembly_with_adhesive_parts()
        apply_default_connection_methods(asm.joints, parts=asm.parts)
        foot_joint = next(j for j in asm.joints if j.child == "left_foot")
        assert foot_joint.connection is not None, "foot joint has no connection"
        assert foot_joint.connection.type == "adhesive", (
            f"foot joint type={foot_joint.connection.type} — should be adhesive"
        )

    def test_structural_joint_stays_bolted(self):
        """A plain structural fixed joint (not sensor/foot) must still
        get bolted — the adhesive rules must not over-trigger."""
        asm = _assembly_with_adhesive_parts()
        apply_default_connection_methods(asm.joints, parts=asm.parts)
        struct_joint = next(j for j in asm.joints if j.child == "torso_link")
        assert struct_joint.connection is not None
        assert struct_joint.connection.type == "bolted", (
            f"structural joint type={struct_joint.connection.type} — "
            f"adhesive rule over-triggered"
        )

