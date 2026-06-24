"""Tests for the pipeline-level arm geometry fix (flat bar → 3D bent arm).

Covers the four layers of the systematic fix:

  - Layer 2 (sanitizer): ``_fix_arm_chain_anchors`` now FLIPS front/back →
    top/bottom (the opposite of its old behaviour), and switches pitch axis
    x → y.  Motor mounts (back/front, fixed), bearings (center/center), and
    prismatic fingers are left untouched.
  - Layer 2 (sanitizer): ``_ensure_arm_default_angles`` injects a zig-zag
    bend when the LLM emitted all-zero default_angles, and is a no-op when
    non-zero angles already exist.
  - Layer 3 (prevalidation): ``_geometric_prevalidation`` flags an arm whose
    Z span is tiny relative to its horizontal span ("too flat").
  - Layer 4 (examples): the 3 few-shot examples use top/bottom + axis y for
    pitch joints, and keep motor mounts / bearings unchanged.
"""

from __future__ import annotations

import pytest

from lang3d.knowledge.mechanics import Assembly, Joint, Part
from lang3d.tools.assembly_generator import (
    _ensure_arm_default_angles,
    _fix_arm_chain_anchors,
    _geometric_prevalidation,
    _is_joint_like,
    _is_link_like,
    _normalize_gripper_fingers,
    _validate_proportions,
    EXAMPLE_ARM_STANDALONE,
    EXAMPLE_5DOF_ARM_REALISTIC,
    EXAMPLE_6DOF_BELT_DRIVE_ARM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box(name: str, length=80, width=25, height=15, category="structural") -> Part:
    return Part(
        name=name,
        category=category,
        description=name,
        dimensions={"length": length, "width": width, "height": height},
    )


def _cyl(name: str, diameter=36, height=30, category="actuator") -> Part:
    return Part(
        name=name,
        category=category,
        description=name,
        dimensions={"diameter": diameter, "height": height},
    )


def _make_flat_arm_parts() -> list[Part]:
    """Parts for a 4-DOF arm (same naming as the real e2e failure)."""
    return [
        _box("base_plate", length=200, width=150, height=8),
        _cyl("shoulder_joint", diameter=40, height=35),
        _box("shoulder_link", length=120, width=25, height=15),
        _cyl("elbow_joint", diameter=36, height=30),
        _box("elbow_link", length=100, width=25, height=15),
        _cyl("wrist_joint", diameter=28, height=28),
        _box("wrist_link", length=60, width=20, height=12),
    ]


# ---------------------------------------------------------------------------
# _is_joint_like / _is_link_like
# ---------------------------------------------------------------------------

def test_is_joint_like_detects_servo_housing_joint():
    assert _is_joint_like("shoulder_joint")
    assert _is_joint_like("elbow_servo")
    assert _is_joint_like("wrist_housing")
    assert _is_joint_like("base_motor")


def test_is_joint_like_rejects_plain_link():
    assert not _is_joint_like("shoulder_link")
    assert not _is_joint_like("upper_arm")


def test_is_link_like_detects_link_arm():
    assert _is_link_like("shoulder_link")
    assert _is_link_like("forearm")
    assert _is_link_like("upper_arm")


# ---------------------------------------------------------------------------
# _fix_arm_chain_anchors — the FLIP
# ---------------------------------------------------------------------------

def test_fix_arm_chain_anchors_flips_top_bottom_to_front_back():
    """Legacy top/bottom+y arm joints must be converted to front/back+x."""
    parts = _make_flat_arm_parts()
    joints = [
        Joint("revolute", "base_plate", "shoulder_joint",
              axis="z", parent_anchor="top", child_anchor="bottom"),
        Joint("revolute", "shoulder_joint", "shoulder_link",
              axis="y", parent_anchor="top", child_anchor="bottom",
              range_deg=(-120, 120)),
        Joint("revolute", "shoulder_link", "elbow_joint",
              axis="y", parent_anchor="top", child_anchor="bottom",
              range_deg=(-150, 150)),
        Joint("fixed", "elbow_joint", "elbow_link",
              parent_anchor="top", child_anchor="bottom"),
        Joint("revolute", "elbow_link", "wrist_joint",
              axis="y", parent_anchor="top", child_anchor="bottom",
              range_deg=(-150, 150)),
    ]
    _fix_arm_chain_anchors(joints, parts)

    # Every legacy top/bottom arm-chain joint should now be front/back.
    for j in joints[1:]:
        assert j.parent_anchor == "front", f"{j.parent}->{j.child} parent_anchor"
        assert j.child_anchor == "back", f"{j.parent}->{j.child} child_anchor"
    # Pitch axes y → x.
    assert joints[1].axis == "x"
    assert joints[2].axis == "x"
    assert joints[4].axis == "x"
    # Base yaw joint (top/bottom, axis z) is untouched.
    assert joints[0].parent_anchor == "top"
    assert joints[0].axis == "z"


def test_fix_arm_chain_anchors_leaves_motor_mounts_untouched():
    """A fixed back/front motor mount inside a housing must NOT be converted."""
    parts = [
        _box("shoulder_housing", length=50, width=40, height=45),
        _cyl("shoulder_motor", diameter=42, height=40),
    ]
    joints = [
        Joint("revolute", "base", "shoulder_housing",
              axis="z", parent_anchor="top", child_anchor="bottom"),
        Joint("revolute", "base", "elbow_joint",
              axis="x", parent_anchor="front", child_anchor="back"),
        # The motor mount we care about — must stay back/front.
        Joint("fixed", "shoulder_housing", "shoulder_motor",
              parent_anchor="back", child_anchor="front"),
    ]
    _fix_arm_chain_anchors(joints, parts)
    mm = joints[2]
    assert mm.parent_anchor == "back"
    assert mm.child_anchor == "front"


def test_fix_arm_chain_anchors_leaves_bearings_untouched():
    """center/center bearing press-fits must NOT be converted."""
    parts = [
        _box("shoulder_housing", length=50, width=40, height=45),
        _cyl("bearing_shoulder", diameter=22, height=7, category="bearing"),
        _cyl("elbow_joint", diameter=36, height=30),
    ]
    joints = [
        Joint("revolute", "base", "shoulder_housing",
              axis="z", parent_anchor="top", child_anchor="bottom"),
        Joint("revolute", "base", "elbow_joint",
              axis="x", parent_anchor="front", child_anchor="back"),
        Joint("fixed", "shoulder_housing", "bearing_shoulder",
              parent_anchor="center", child_anchor="center"),
    ]
    _fix_arm_chain_anchors(joints, parts)
    bearing = joints[2]
    assert bearing.parent_anchor == "center"
    assert bearing.child_anchor == "center"


def test_fix_arm_chain_anchors_leaves_prismatic_untouched():
    """Prismatic gripper fingers must NOT be converted here."""
    parts = [
        _box("gripper_base", length=28, width=50, height=32),
        _box("gripper_finger_left", length=60, width=10, height=28),
        _box("elbow_joint", length=40, width=35, height=40),
    ]
    joints = [
        Joint("revolute", "base", "elbow_joint",
              axis="x", parent_anchor="front", child_anchor="back"),
        Joint("revolute", "base", "wrist_joint",
              axis="x", parent_anchor="front", child_anchor="back"),
        Joint("prismatic", "gripper_base", "gripper_finger_left",
              axis="x", parent_anchor="front", child_anchor="back"),
    ]
    _fix_arm_chain_anchors(joints, parts)
    finger = joints[2]
    assert finger.parent_anchor == "front"
    assert finger.child_anchor == "back"


def test_fix_arm_chain_anchors_leaves_z_axis_untouched():
    """A front/back revolute with axis=z (wrist roll) is left untouched."""
    parts = _make_flat_arm_parts()
    joints = [
        Joint("revolute", "base_plate", "shoulder_joint",
              axis="z", parent_anchor="top", child_anchor="bottom"),
        Joint("revolute", "shoulder_joint", "shoulder_link",
              axis="x", parent_anchor="front", child_anchor="back"),
        Joint("revolute", "wrist_link", "wrist_rotate",
              axis="z", parent_anchor="front", child_anchor="back",
              range_deg=(-180, 180)),
    ]
    _fix_arm_chain_anchors(joints, parts)
    # The front/back+z joint stays as-is (only top/bottom+y gets converted).
    wrist_roll = joints[2]
    assert wrist_roll.parent_anchor == "front"
    assert wrist_roll.child_anchor == "back"
    assert wrist_roll.axis == "z"


# ---------------------------------------------------------------------------
# _ensure_arm_default_angles
# ---------------------------------------------------------------------------

def _make_arm_assembly(default_angles=None) -> Assembly:
    parts = _make_flat_arm_parts()
    joints = [
        Joint("revolute", "base_plate", "shoulder_joint",
              axis="z", parent_anchor="top", child_anchor="bottom",
              range_deg=(-180, 180)),
        Joint("revolute", "shoulder_joint", "shoulder_link",
              axis="y", parent_anchor="top", child_anchor="bottom",
              range_deg=(-120, 120)),
        Joint("revolute", "shoulder_link", "elbow_joint",
              axis="y", parent_anchor="top", child_anchor="bottom",
              range_deg=(-150, 150)),
        Joint("revolute", "elbow_joint", "elbow_link",
              axis="y", parent_anchor="top", child_anchor="bottom",
              range_deg=(-150, 150)),
    ]
    return Assembly(
        name="test_arm",
        parts=parts,
        joints=joints,
        default_angles=dict(default_angles or {}),
    )


def test_ensure_arm_default_angles_injects_when_all_zero():
    asm = _make_arm_assembly(default_angles={})
    result = _ensure_arm_default_angles(asm)
    angles = result.default_angles
    # Should have injected something non-zero.
    assert any(abs(float(v)) > 1e-6 for v in angles.values())
    # Base yaw (first revolute, axis=z) should be 0.
    assert angles.get("shoulder_joint") == 0.0
    # Pitch joints should all tilt the same direction (negative = upward
    # in the front/back convention) so the arm rises in Z.
    vals = [angles["shoulder_link"], angles["elbow_joint"], angles["elbow_link"]]
    assert abs(vals[0]) > 5.0
    assert abs(vals[1]) > 5.0
    assert abs(vals[2]) > 5.0
    # All same sign (negative = upward tilt) — reinforcing, not cancelling.
    assert all(v < 0 for v in vals), f"all pitch should be negative, got {vals}"


def test_ensure_arm_default_angles_respects_existing_nonzero():
    """LLM-provided non-zero angles are clamped; sign forced negative (upward)."""
    asm = _make_arm_assembly(default_angles={"shoulder_link": -42.0})
    result = _ensure_arm_default_angles(asm)
    # Shoulder clamped to ±35° (first-pitch cap) and kept negative (upward).
    val = result.default_angles.get("shoulder_link")
    assert val is not None and val < 0, "shoulder_link must be negative (upward)"
    assert abs(val) <= 35.0, f"shoulder_link must be clamped to ±35°, got {val}"


def test_ensure_arm_default_angles_no_op_for_non_arm():
    """A wheeled robot (no link parts, < 2 revolute) should not be touched."""
    parts = [
        _box("base_plate", length=300, width=200, height=5),
        _cyl("motor_fl", diameter=30, height=40),
        _cyl("wheel_fl", diameter=65, height=26, category="mechanical"),
    ]
    joints = [
        Joint("fixed", "base_plate", "motor_fl"),
        Joint("revolute", "motor_fl", "wheel_fl", axis="y"),
    ]
    asm = Assembly(name="wheeled", parts=parts, joints=joints, default_angles={})
    result = _ensure_arm_default_angles(asm)
    # No link-like part → no injection.
    assert result.default_angles == {}


def test_ensure_arm_default_angles_clamps_to_range():
    """Injected angles must stay within each joint's range_deg."""
    asm = _make_arm_assembly(default_angles={})
    # Tighten the ranges to verify clamping.
    for j in asm.joints:
        j.range_deg = (-20, 20)
    result = _ensure_arm_default_angles(asm)
    angles = result.default_angles
    name_to_joint = {j.child: j for j in asm.joints}
    for child, angle in angles.items():
        lo, hi = name_to_joint[child].range_deg
        assert lo - 0.5 <= angle <= hi + 0.5, f"{child}={angle} not in [{lo},{hi}]"


def test_ensure_arm_default_angles_fills_per_joint_with_partial_nonzero():
    """A stray non-zero value must NOT cause all other pitch joints to be skipped.

    Regression: previously the sanitizer bailed entirely if ANY single
    default_angle was non-zero.  The LLM frequently gives only the wrist roll
    a non-zero value while leaving every pitch joint at 0 — the arm then
    renders as a straight column.
    """
    asm = _make_arm_assembly(default_angles={"elbow_joint": 45.0})
    result = _ensure_arm_default_angles(asm)
    angles = result.default_angles
    # All pitch angles are forced to negative (upward tilt, same sign) and
    # clamped: shoulder ±25°, subsequent pitches ±30°.
    ej = angles.get("elbow_joint")
    assert ej is not None and abs(ej) > 5.0, f"elbow_joint must be non-zero, got {ej}"
    assert ej < 0, f"elbow_joint must be negative (upward), got {ej}"
    assert abs(ej) <= 40.0, f"elbow_joint must be clamped to ±40°, got {ej}"
    # The other pitch joints (zero/missing) are now filled with bends.
    assert abs(angles["shoulder_link"]) > 5.0
    assert abs(angles["elbow_link"]) > 5.0


def test_ensure_arm_default_angles_cleans_non_revolute_keys():
    """default_angles keys that are not revolute-joint children must be removed."""
    asm = _make_arm_assembly(
        default_angles={"gripper_base": 0.0, "wrist_link": 108.0},
    )
    result = _ensure_arm_default_angles(asm)
    angles = result.default_angles
    # gripper_base is a fixed-joint child → removed.
    assert "gripper_base" not in angles
    # Pitch joints that were zero are now filled.
    assert abs(angles.get("shoulder_link", 0.0)) > 5.0


# ---------------------------------------------------------------------------
# _normalize_gripper_fingers — prismatic connection_method + mimic_joint
# ---------------------------------------------------------------------------

def test_normalize_gripper_fingers_clears_prismatic_connection():
    """Prismatic finger joints must not carry connection_method='bolted'."""
    from lang3d.knowledge.mechanics import ConnectionMethod
    parts = [
        _box("gripper_base", length=28, width=50, height=32),
        _box("gripper_finger_left", length=60, width=10, height=28),
        _box("gripper_finger_right", length=60, width=10, height=28),
    ]
    joints = [
        Joint("prismatic", "gripper_base", "gripper_finger_left",
              axis="y", parent_anchor="center", child_anchor="center",
              connection=ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=2)),
        Joint("prismatic", "gripper_base", "gripper_finger_right",
              axis="y", parent_anchor="center", child_anchor="center",
              connection=ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=2)),
    ]
    asm = Assembly(name="gripper_test", parts=parts, joints=joints, default_angles={})
    from lang3d.tools.assembly_generator import _normalize_gripper_fingers
    result = _normalize_gripper_fingers(asm)
    for j in result.joints:
        if j.type == "prismatic":
            assert j.connection is None, (
                f"Prismatic {j.parent}->{j.child} should have no connection_method"
            )


def test_normalize_gripper_fingers_injects_mimic_joint():
    """Right finger must mimic left finger with multiplier=-1 for antagonistic grip."""
    parts = [
        _box("gripper_base", length=28, width=50, height=32),
        _box("gripper_finger_left", length=60, width=10, height=28),
        _box("gripper_finger_right", length=60, width=10, height=28),
    ]
    joints = [
        Joint("prismatic", "gripper_base", "gripper_finger_left",
              axis="y", parent_anchor="center", child_anchor="center"),
        Joint("prismatic", "gripper_base", "gripper_finger_right",
              axis="y", parent_anchor="center", child_anchor="center"),
    ]
    asm = Assembly(name="gripper_test", parts=parts, joints=joints, default_angles={})
    from lang3d.tools.assembly_generator import _normalize_gripper_fingers
    result = _normalize_gripper_fingers(asm)
    left = [j for j in result.joints if j.child == "gripper_finger_left"][0]
    right = [j for j in result.joints if j.child == "gripper_finger_right"][0]
    # Right finger mimics left.
    assert right.mimic_joint == "gripper_finger_left"
    assert right.mimic_multiplier == -1.0
    # Left finger is the driver — no mimic.
    assert left.mimic_joint == ""


def test_normalize_gripper_fingers_preserves_existing_mimic():
    """If the LLM already set a mimic_joint, the sanitizer must not overwrite."""
    parts = [
        _box("gripper_base", length=28, width=50, height=32),
        _box("gripper_finger_left", length=60, width=10, height=28),
        _box("gripper_finger_right", length=60, width=10, height=28),
    ]
    joints = [
        Joint("prismatic", "gripper_base", "gripper_finger_left",
              axis="y", parent_anchor="center", child_anchor="center"),
        Joint("prismatic", "gripper_base", "gripper_finger_right",
              axis="y", parent_anchor="center", child_anchor="center",
              mimic_joint="gripper_finger_left", mimic_multiplier=-1.0),
    ]
    asm = Assembly(name="gripper_test", parts=parts, joints=joints, default_angles={})
    from lang3d.tools.assembly_generator import _normalize_gripper_fingers
    result = _normalize_gripper_fingers(asm)
    right = [j for j in result.joints if j.child == "gripper_finger_right"][0]
    assert right.mimic_joint == "gripper_finger_left"
    assert right.mimic_multiplier == -1.0


# ---------------------------------------------------------------------------
# _geometric_prevalidation — arm-too-flat detection
# ---------------------------------------------------------------------------

def test_geometric_prevalidation_flags_flat_arm():
    """A flat arm (small Z span, large Y span) must be flagged."""
    parts = [{"name": n} for n in (
        "base_plate", "shoulder_joint", "shoulder_link",
        "elbow_joint", "elbow_link", "wrist_joint", "wrist_link",
    )]
    # Flat bar: Z span ~20mm, Y span ~500mm.
    ys = [0, 0, -100, -200, -300, -400, -500]
    positions = {}
    for n, y in zip(
        ("base_plate", "shoulder_joint", "shoulder_link",
         "elbow_joint", "elbow_link", "wrist_joint", "wrist_link"),
        ys,
    ):
        positions[n] = {"position": [0, y, 10]}

    problems = _geometric_prevalidation(parts, positions)
    flat_problems = [p for p in problems if "flat" in p.lower() or "horizontal" in p.lower()]
    assert flat_problems, f"Expected a too-flat/horizontal problem, got: {problems}"


def test_geometric_prevalidation_passes_3d_arm():
    """A properly 3D arm (large Z span) should not trigger the flat check."""
    parts = [{"name": n} for n in (
        "base_plate", "shoulder_joint", "shoulder_link",
        "elbow_joint", "elbow_link", "wrist_joint", "wrist_link",
    )]
    # 3D bent arm: Z span ~300mm, Y span ~150mm.
    positions = {
        "base_plate": {"position": [0, 0, 0]},
        "shoulder_joint": {"position": [0, 0, 40]},
        "shoulder_link": {"position": [0, 30, 140]},
        "elbow_joint": {"position": [0, 50, 240]},
        "elbow_link": {"position": [0, 20, 320]},
        "wrist_joint": {"position": [0, -10, 280]},
        "wrist_link": {"position": [0, -30, 220]},
    }
    problems = _geometric_prevalidation(parts, positions)
    flat_problems = [p for p in problems if "flat" in p.lower() or "horizontal" in p.lower()]
    assert not flat_problems, f"Unexpected flat flag on 3D arm: {flat_problems}"


def test_geometric_prevalidation_ignores_non_arm_assemblies():
    """A wheeled robot with few arm-keyword parts should not trigger arm check."""
    parts = [{"name": n} for n in ("base_plate", "motor_fl", "wheel_fl")]
    positions = {
        "base_plate": {"position": [0, 0, 30]},
        "motor_fl": {"position": [-100, 80, 10]},
        "wheel_fl": {"position": [-100, 80, 30]},
    }
    problems = _geometric_prevalidation(parts, positions)
    flat_problems = [p for p in problems if "flat" in p.lower() or "horizontal" in p.lower()]
    assert not flat_problems


def test_geometric_prevalidation_flags_missing_gripper_fingers():
    """An arm with fewer than 2 finger parts must be flagged."""
    parts = [{"name": n} for n in (
        "base_plate", "shoulder_joint", "shoulder_link",
        "elbow_joint", "elbow_link", "wrist_joint", "wrist_link",
        "gripper_base",
    )]
    positions = {
        "base_plate": {"position": [0, 0, 0]},
        "shoulder_joint": {"position": [0, 0, 40]},
        "shoulder_link": {"position": [0, 30, 140]},
        "elbow_joint": {"position": [0, 50, 240]},
        "elbow_link": {"position": [0, 20, 320]},
        "wrist_joint": {"position": [0, -10, 280]},
        "wrist_link": {"position": [0, -30, 220]},
        "gripper_base": {"position": [0, -40, 200]},
    }
    problems = _geometric_prevalidation(parts, positions)
    gripper_problems = [p for p in problems if "gripper" in p.lower() or "finger" in p.lower()]
    assert gripper_problems, (
        f"Expected a missing-gripper-fingers problem, got: {problems}"
    )


def test_geometric_prevalidation_flags_close_fingers():
    """Two fingers < 25mm apart must be flagged as fused block."""
    parts = [{"name": n} for n in (
        "base_plate", "shoulder_joint", "shoulder_link",
        "elbow_joint", "elbow_link", "wrist_joint", "wrist_link",
        "gripper_base", "gripper_finger_left", "gripper_finger_right",
    )]
    positions = {
        "base_plate": {"position": [0, 0, 0]},
        "shoulder_joint": {"position": [0, 0, 40]},
        "shoulder_link": {"position": [0, 30, 140]},
        "elbow_joint": {"position": [0, 50, 240]},
        "elbow_link": {"position": [0, 20, 320]},
        "wrist_joint": {"position": [0, -10, 280]},
        "wrist_link": {"position": [0, -30, 220]},
        "gripper_base": {"position": [0, -40, 200]},
        # Fingers only 10mm apart → fused block
        "gripper_finger_left": {"position": [0, -45, 200]},
        "gripper_finger_right": {"position": [0, -35, 200]},
    }
    problems = _geometric_prevalidation(parts, positions)
    close_problems = [p for p in problems if "apart" in p.lower() or "fuse" in p.lower()]
    assert close_problems, (
        f"Expected a fingers-too-close problem, got: {problems}"
    )


def test_geometric_prevalidation_passes_well_separated_fingers():
    """Two fingers >= 25mm apart should NOT trigger the closeness check."""
    parts = [{"name": n} for n in (
        "base_plate", "shoulder_joint", "shoulder_link",
        "elbow_joint", "elbow_link", "wrist_joint", "wrist_link",
        "gripper_base", "gripper_finger_left", "gripper_finger_right",
    )]
    positions = {
        "base_plate": {"position": [0, 0, 0]},
        "shoulder_joint": {"position": [0, 0, 40]},
        "shoulder_link": {"position": [0, 30, 140]},
        "elbow_joint": {"position": [0, 50, 240]},
        "elbow_link": {"position": [0, 20, 320]},
        "wrist_joint": {"position": [0, -10, 280]},
        "wrist_link": {"position": [0, -30, 220]},
        "gripper_base": {"position": [0, -40, 200]},
        # Fingers 90mm apart → clearly separated
        "gripper_finger_left": {"position": [0, -85, 216]},
        "gripper_finger_right": {"position": [0, 5, 216]},
    }
    problems = _geometric_prevalidation(parts, positions)
    close_problems = [p for p in problems if "apart" in p.lower() or "fuse" in p.lower()]
    assert not close_problems, (
        f"Unexpected fingers-too-close flag on well-separated fingers: {close_problems}"
    )


# ---------------------------------------------------------------------------


# Few-shot examples — vertical pattern assertions
# ---------------------------------------------------------------------------

def test_example_arm_standalone_uses_horizontal_anchors():
    """EXAMPLE_ARM_STANDALONE arm-chain joints use front/back, pitch axis=x."""
    import json
    data = json.loads(EXAMPLE_ARM_STANDALONE)
    joints = data["joints"]
    arm_joints = [
        j for j in joints
        if j.get("parent_anchor") in ("top", "front")
        and "gripper_finger" not in j["child"]
    ]
    # Every arm-chain joint (non-prismatic, non-servo-mount, non-base-yaw) is front/back.
    for j in arm_joints:
        if j.get("type") == "prismatic":
            continue
        if j["parent"] == "gripper_base" and j["child"] == "gripper_servo":
            continue
        # Base yaw (axis=z) correctly keeps top/bottom.
        if j.get("axis") == "z":
            continue
        assert j["parent_anchor"] == "front", f"{j['parent']}->{j['child']}"
        assert j["child_anchor"] == "back", f"{j['parent']}->{j['child']}"
    # Shoulder/elbow pitch joints use axis=x (horizontal convention).
    pitch_children = {"shoulder_link", "elbow_joint"}
    for j in joints:
        if j["child"] in pitch_children and j.get("type") == "revolute":
            assert j["axis"] == "x", f"{j['child']} axis should be x"
    # default_angles has non-zero values.
    assert any(abs(float(v)) > 1e-6 for v in data["default_angles"].values())


def test_example_5dof_keeps_motor_mounts_back_front():
    """Motor mount joints (back/front, fixed) must remain unchanged in 5DOF."""
    import json
    data = json.loads(EXAMPLE_5DOF_ARM_REALISTIC)
    motor_mounts = [
        j for j in data["joints"]
        if j.get("type") == "fixed"
        and j.get("parent_anchor") == "back"
        and j.get("child_anchor") == "front"
    ]
    # There are several motor mounts and they all stay back/front.
    assert len(motor_mounts) >= 3
    for j in motor_mounts:
        assert "motor" in j["child"].lower()


def test_example_5dof_pitch_joints_are_horizontal():
    """5DOF pitch joints use front/back + axis=x."""
    import json
    data = json.loads(EXAMPLE_5DOF_ARM_REALISTIC)
    for j in data["joints"]:
        if j.get("type") == "revolute" and j.get("axis") == "x":
            assert j["parent_anchor"] == "front", f"{j['parent']}->{j['child']}"
            assert j["child_anchor"] == "back"


def test_example_6dof_pitch_joints_are_horizontal():
    """6DOF pitch joints use front/back + axis=x."""
    import json
    data = json.loads(EXAMPLE_6DOF_BELT_DRIVE_ARM)
    pitch_joints = [
        j for j in data["joints"]
        if j.get("type") == "revolute" and j.get("axis") == "x"
    ]
    assert len(pitch_joints) >= 3
    for j in pitch_joints:
        assert j["parent_anchor"] == "front"
        assert j["child_anchor"] == "back"
    # Bearings still center/center.
    bearings = [
        j for j in data["joints"]
        if j.get("parent_anchor") == "center"
    ]
    assert len(bearings) >= 3


def test_examples_default_angles_nonzero():
    """All three arm examples have at least one non-zero default_angle."""
    import json
    for example in (
        EXAMPLE_ARM_STANDALONE,
        EXAMPLE_5DOF_ARM_REALISTIC,
        EXAMPLE_6DOF_BELT_DRIVE_ARM,
    ):
        data = json.loads(example)
        assert any(
            abs(float(v)) > 1e-6 for v in data["default_angles"].values()
        ), f"{data['name']} has all-zero default_angles"


# ---------------------------------------------------------------------------
# Structural quality fix tests (gripper gap, center anchor, proportions)
# ---------------------------------------------------------------------------

def test_gripper_fingers_gap_within_base():
    """Gripper finger centers must stay within the base width so they
    connect to the rail grooves instead of floating in space."""
    base = _box("gripper_base", length=28, width=50, height=32)
    lf = _box("gripper_finger_left", length=60, width=10, height=28)
    rf = _box("gripper_finger_right", length=60, width=10, height=28)
    asm = Assembly(
        name="test",
        parts=[base, lf, rf],
        joints=[
            Joint("prismatic", "gripper_base", "gripper_finger_left",
                  parent_anchor="center", child_anchor="center"),
            Joint("prismatic", "gripper_base", "gripper_finger_right",
                  parent_anchor="center", child_anchor="center"),
        ],
    )
    asm = _normalize_gripper_fingers(asm)
    left = [j for j in asm.joints if j.child == "gripper_finger_left"][0]
    right = [j for j in asm.joints if j.child == "gripper_finger_right"][0]

    # Gap must be within the base half-width (25mm for w=50)
    gap = abs(left.offset[0])
    assert gap <= 22.0, f"Gap {gap}mm exceeds base half-width — fingers float"

    # z_lift must be 0 — fingers at base center height, not floating above
    assert left.offset[2] == 0.0, f"Left z_lift should be 0, got {left.offset[2]}"
    assert right.offset[2] == 0.0, f"Right z_lift should be 0, got {right.offset[2]}"


def test_fixed_joint_center_anchor_no_overlap():
    """Center+face anchor should push child outside parent, not embed it."""
    from lang3d.tools.assembly_solver import AssemblySolver

    parent = _box("parent_block", length=60, width=40, height=20)
    child = _box("child_cap", length=50, width=35, height=10)
    asm = Assembly(
        name="test",
        parts=[parent, child],
        joints=[
            Joint("fixed", "parent_block", "child_cap",
                  parent_anchor="top", child_anchor="center"),
        ],
    )
    solver = AssemblySolver(asm)
    positions = solver.solve()

    parent_z = positions["parent_block"]["position"][2]
    child_z = positions["child_cap"]["position"][2]
    parent_top = parent_z + 10  # half of height=20

    # Child center should be ABOVE parent top face by at least child_half (5mm)
    assert child_z >= parent_top + 4.0, (
        f"Child Z={child_z:.1f} should be >= {parent_top + 5:.1f} "
        f"(parent top + child half-extent) — parts are intersecting"
    )


def test_center_center_bearing_unchanged():
    """Center/center anchor (bearing press-fit) should NOT be offset."""
    from lang3d.tools.assembly_solver import AssemblySolver

    housing = _box("housing", length=40, width=40, height=20)
    bearing = _cyl("bearing", diameter=22, height=7)
    asm = Assembly(
        name="test",
        parts=[housing, bearing],
        joints=[
            Joint("fixed", "housing", "bearing",
                  parent_anchor="center", child_anchor="center"),
        ],
    )
    solver = AssemblySolver(asm)
    positions = solver.solve()

    h_z = positions["housing"]["position"][2]
    b_z = positions["bearing"]["position"][2]

    # Center/center: bearing should be concentric with housing (same Z)
    assert abs(b_z - h_z) < 1.0, (
        f"Bearing Z={b_z:.1f} should equal housing Z={h_z:.1f} "
        f"for center/center press-fit"
    )


def test_proportion_validation_rejects_oversize_gripper_width():
    """gripper_base wider than 1.8x parent link should raise, not clamp.

    P1-1: per CLAUDE.md ("不要在代码里加 hack 让 LLM/外部输入看起来对"),
    disproportionate dimensions must be fed back to the LLM via a
    RuntimeError so it can regenerate, not silently clamped.
    """
    wrist = _box("wrist_link", length=60, width=20, height=12)
    gripper = _box("gripper_base", length=28, width=80, height=32)
    asm = Assembly(
        name="test",
        parts=[wrist, gripper],
        joints=[
            Joint("fixed", "wrist_link", "gripper_base",
                  parent_anchor="front", child_anchor="back"),
        ],
    )
    with pytest.raises(RuntimeError, match="gripper_base"):
        _validate_proportions(asm)
    # Dimensions must be left UNCHANGED so the error message reports the
    # original LLM values (no silent clamp-and-pretend).
    assert gripper.dimensions["width"] == 80


def test_proportion_validation_rejects_extreme_link_ratio():
    """Consecutive links with length ratio > 3.0 should raise."""
    link1 = _box("upper_link", length=150, width=25, height=15)
    link2 = _box("lower_link", length=30, width=25, height=15)
    asm = Assembly(
        name="test",
        parts=[link1, link2],
        joints=[
            Joint("fixed", "upper_link", "lower_link",
                  parent_anchor="front", child_anchor="back"),
        ],
    )
    with pytest.raises(RuntimeError, match="ratio"):
        _validate_proportions(asm)
    assert link2.dimensions["length"] == 30  # unchanged


def test_proportion_validation_rejects_thin_link_vs_joint():
    """Link dwarfed by a joint cylinder should raise, not be widened.

    Regression for the joint-link intersection seen in the 4dof_arm render:
    shoulder_joint (diameter=40) connecting to shoulder_link (25x15) left
    the 20 mm joint radius extending well past the 15 mm link height, so the
    joint visually "swallowed" the link.  The validator now reports this
    instead of silently enlarging the link cross-section.
    """
    joint = _cyl("shoulder_joint", diameter=40, height=35)
    link = _box("shoulder_link", length=120, width=25, height=15)
    asm = Assembly(
        name="test",
        parts=[joint, link],
        joints=[
            Joint("revolute", "shoulder_joint", "shoulder_link",
                  parent_anchor="front", child_anchor="back"),
        ],
    )
    with pytest.raises(RuntimeError, match="shoulder_link"):
        _validate_proportions(asm)
    # Width 25 < 22? no, 25 > 22 ok. Height 15 < 20 -> violation.
    assert link.dimensions["height"] == 15  # unchanged


def test_proportion_validation_link_to_joint_reverse():
    """Same joint/link rule applies when the link is the parent."""
    link = _box("wrist_link", length=60, width=20, height=12)
    joint = _cyl("wrist_joint", diameter=28, height=28)
    asm = Assembly(
        name="test",
        parts=[link, joint],
        joints=[
            Joint("revolute", "wrist_link", "wrist_joint",
                  parent_anchor="front", child_anchor="back"),
        ],
    )
    # 0.55x28 = 15.4 mm: link width 20 > 15.4 ok.
    # 0.50x28 = 14 mm: link height 12 < 14 -> violation -> raise.
    with pytest.raises(RuntimeError, match="wrist_link"):
        _validate_proportions(asm)
    assert link.dimensions["height"] == 12  # unchanged


def test_proportion_validation_accepts_coherent_proportions():
    """A well-proportioned assembly must NOT raise (positive control).

    Guards against the validator becoming overly strict and rejecting
    valid designs after the P1-1 clamp->raise conversion.
    """
    joint = _cyl("shoulder_joint", diameter=30, height=30)
    link = _box("shoulder_link", length=120, width=22, height=18)
    asm = Assembly(
        name="test",
        parts=[joint, link],
        joints=[
            Joint("revolute", "shoulder_joint", "shoulder_link",
                  parent_anchor="front", child_anchor="back"),
        ],
    )
    # 0.55x30 = 16.5 mm: width 22 > 16.5 ok, height 18 > 15 ok.
    # No gripper_base, no consecutive links -> only Check 3 applies.
    result = _validate_proportions(asm)
    assert result is asm  # returned unchanged

# ---------------------------------------------------------------------------
# Regression: validation errors must enter the LLM retry loop
# ---------------------------------------------------------------------------


def test_validate_assembly_error_enters_retry_loop(monkeypatch, tmp_path):
    """When _validate_assembly raises RuntimeError, the VLM loop must
    catch it and route the error into problems_history (so the LLM gets
    a chance to regenerate), instead of letting the exception escape
    and kill the pipeline.

    Regression guard for the bug that caused 4wheel_dual_arm to die on
    "Joint #16: child not in parts list" before reaching Phase 2.
    """
    from lang3d.tools import assembly_generator as ag
    from lang3d.knowledge.mechanics import Assembly, Part, Joint

    # Construct an assembly that _validate_assembly will reject:
    # joint references a child ("ghost") that is not in parts list.
    bad = Assembly(
        name="bad",
        parts=[
            Part(
                name="base",
                category="link",
                description="base",
                dimensions={"length": 10, "width": 10, "height": 10},
            ),
        ],
        joints=[
            Joint(type="fixed", parent="base", child="ghost"),
        ],
    )

    # Stub out the LLM entry point so no real API call is made.
    monkeypatch.setattr(ag, "generate_assembly_from_nl", lambda **kw: bad)

    # max_rounds=1: round 1 generates bad -> validation fails -> continue
    # -> loop exits with passed=False. The key assertion is that no
    # exception escapes generate_assembly_with_vlm_loop.
    # api_key="dummy" bypasses the GLM_API_KEY gate; no real API call
    # happens because generate_assembly_from_nl is stubbed.
    result = ag.generate_assembly_with_vlm_loop(
        description="test",
        output_dir=str(tmp_path),
        max_rounds=1,
        api_key="dummy",
    )

    assert result["passed"] is False
    # The validation error must appear in problems_history (this is the
    # feedback channel the LLM would see on the next round if max_rounds>1).
    flat_problems = [
        p for plist in result["problems_history"] for p in plist
    ]
    assert any("validation error" in p.lower() for p in flat_problems), (
        f"Expected validation error in problems_history, got: {flat_problems}"
    )
    assert any("ghost" in p for p in flat_problems), (
        f"Expected 'ghost' (the bad child name) in error message, got: {flat_problems}"
    )


# ---------------------------------------------------------------------------
# P1: VLM gripper false-alarm filter (geometric ground-truth arbitration)
# ---------------------------------------------------------------------------


class TestGripperFalseAlarmFilter:
    """_is_gripper_false_alarm identifies VLM gripper misreads to override.

    When _geometric_prevalidation confirms fingers are separated (>= 25mm),
    VLM complaints like "solid block / no separated prongs" are false alarms
    and must be removable so a geometrically-correct gripper is not rejected.
    """

    def test_matches_actual_round3_false_alarms(self):
        from lang3d.tools.assembly_generator import _is_gripper_false_alarm
        # Exact strings observed in vlm_responses.json round 3.
        actual = [
            "Tip of the arm does not have two clearly separated parallel prongs",
            "End effector is a solid block",
            "No gripper at the tip of the arm",
            "End effector is a single chunky mass with no visible gap",
        ]
        for p in actual:
            assert _is_gripper_false_alarm(p), f"should match: {p}"

    def test_does_not_match_structural_problems(self):
        from lang3d.tools.assembly_generator import _is_gripper_false_alarm
        # Real structural problems must be preserved (not filtered).
        for p in [
            "Parts floating in mid-air with no support",
            "Parts intersecting / overlapping each other",
            "Shoulder joint overlapping with base plate",
            "Arm pointing in impossible directions",
        ]:
            assert not _is_gripper_false_alarm(p), f"should NOT match: {p}"

    def test_does_not_match_orientation_problems(self):
        from lang3d.tools.assembly_generator import _is_gripper_false_alarm
        # A gripper with wrong orientation is a REAL problem — the filter
        # must not swallow it.  These phrases lack the fusion structure
        # keywords, so they correctly do not match.
        for p in [
            "Cylinder oriented along wrong axis",
            "Gripper fingers point backward",
            "End effector is rotated 90 degrees",
        ]:
            assert not _is_gripper_false_alarm(p), f"should NOT match: {p}"

    def test_double_condition_prevents_false_positives(self):
        from lang3d.tools.assembly_generator import _is_gripper_false_alarm
        # "tip" alone (context) without a fusion pattern is not a false alarm.
        assert not _is_gripper_false_alarm("The tip is too far from the base")
        # "solid block" alone (pattern) without gripper context is not either.
        assert not _is_gripper_false_alarm("Base plate is a solid block of aluminum")


class TestWheelFalseAlarmFilter:
    """_is_wheel_false_alarm filters VLM "wheel orientation" hallucinations.

    GLM-4.6V mistakes the cylindrical servo housings of a fixed-base arm
    (base_yaw_servo Ø40) for wheels and reports them as "oriented vertically,
    should be horizontal to roll on ground". On an arm with no wheel parts,
    this is a pure hallucination and must be filtered — otherwise the Fixer
    re-orients servos to "fix" non-existent wheels, corrupting the arm.
    """

    ARM_PARTS = [
        {"name": "base_plate"}, {"name": "base_yaw_servo"},
        {"name": "pitch_link_1"}, {"name": "gripper_finger_left"},
    ]
    WHEELED_PARTS = [
        {"name": "base_plate"}, {"name": "wheel_fr"},
        {"name": "wheel_rl"}, {"name": "arm_l_link"},
    ]

    def test_filters_wheel_orientation_on_arm(self):
        from lang3d.tools.assembly_generator import _is_wheel_false_alarm
        for p in [
            "Orange cylindrical parts (likely wheels) oriented vertically",
            "Wheels have incorrect orientation, axis not perpendicular to ground",
            "Wheels should be horizontal to roll on the ground",
        ]:
            assert _is_wheel_false_alarm(p, self.ARM_PARTS), f"should filter: {p}"

    def test_keeps_collision_report_mentioning_wheel(self):
        """A real collision text that happens to contain "wheel" must NOT be
        filtered — the bare word isn't an orientation complaint."""
        from lang3d.tools.assembly_generator import _is_wheel_false_alarm
        assert not _is_wheel_false_alarm(
            "base_plate and wheel overlap by 65x26x5mm", self.ARM_PARTS)

    def test_keeps_complaint_when_wheels_genuinely_exist(self):
        """If the assembly really has wheel parts, orientation complaints
        are legitimate and must be kept."""
        from lang3d.tools.assembly_generator import _is_wheel_false_alarm
        assert not _is_wheel_false_alarm(
            "Wheels have incorrect orientation", self.WHEELED_PARTS)

    def test_unrelated_problems_not_matched(self):
        from lang3d.tools.assembly_generator import _is_wheel_false_alarm
        assert not _is_wheel_false_alarm("Parts collide and intersect", self.ARM_PARTS)
        assert not _is_wheel_false_alarm("COM outside support polygon", self.ARM_PARTS)


class TestGeometricArbitration:
    """When geometry confirms the gripper is fine, VLM false alarms are dropped.

    Covers the assembly_generator.py arbitration logic: geo_problems
    empty (fingers OK) → remove gripper false alarms → if nothing remains, pass.
    """

    @staticmethod
    def _patch_pipeline(monkeypatch, geo_result, vlm_json, tmp_path):
        """Monkeypatch the VLM pipeline's side-effectful calls.

        _vlm_check_assembly does function-local imports of
        render_assembly_from_positions and GLMBackend, so we must patch the
        SOURCE modules (not the assembly_generator namespace copy).
        """
        import lang3d.tools.assembly_generator as ag
        import lang3d.tools.vtk_renderer as vr
        import lang3d.models.glm as glm

        monkeypatch.setattr(ag, "_geometric_prevalidation", lambda *a, **k: geo_result)
        monkeypatch.setattr(
            vr, "render_assembly_from_positions",
            lambda **kw: [str(tmp_path / "fake_view.png")],
        )
        (tmp_path / "fake_view.png").write_bytes(b"")

        class _FakeBackend:
            def __init__(self, *a, **k):
                pass
            def vision(self, *a, **k):
                return vlm_json

        monkeypatch.setattr(glm, "GLMBackend", _FakeBackend)

    def test_geometry_clean_removes_finger_false_alarms(self, monkeypatch, tmp_path):
        """VLM reports only finger false alarms + geometry clean → pass."""
        vlm = ('{"passed": false, "problems": '
               '["End effector is a solid block", '
               '"Tip of the arm does not have two clearly separated parallel prongs"]}')
        self._patch_pipeline(monkeypatch, [], vlm, tmp_path)

        import lang3d.tools.assembly_generator as ag
        passed, problems = ag._vlm_check_assembly(
            positions={"gripper_finger_left": {"position": [0, 0, 0]},
                       "gripper_finger_right": {"position": [40, 0, 0]}},
            parts=[{"name": "gripper_finger_left", "dimensions": {}, "category": ""},
                   {"name": "gripper_finger_right", "dimensions": {}, "category": ""}],
            render_dir=str(tmp_path),
            api_key="dummy",
            base_url="dummy",
            real_stl_dir=str(tmp_path),  # skip _generate_preview_stls
        )
        assert passed is True
        assert problems == []

    def test_real_problems_survive_arbitration(self, monkeypatch, tmp_path):
        """Finger false alarm + a real problem → still fails, real problem kept.

        Note (updated 2026-06-22, B+C arbitration): a "floating" VLM report
        is now ALSO filtered when the joint graph is connected (Check 7).
        So to test that real problems survive, we must either (a) give a
        genuinely disconnected assembly (Check 7 flags it → not filtered),
        or (b) use a non-floating real problem like "intersecting".  This
        test uses (b) — a collision problem that no false-alarm filter
        matches, so it survives arbitration.
        """
        vlm = ('{"passed": false, "problems": '
               '["End effector is a solid block", '
               '"Parts intersecting (brown cylinder overlapping red cube)"]}')
        self._patch_pipeline(monkeypatch, [], vlm, tmp_path)

        import lang3d.tools.assembly_generator as ag
        passed, problems = ag._vlm_check_assembly(
            positions={}, parts=[],
            render_dir=str(tmp_path), api_key="dummy", base_url="dummy",
            real_stl_dir=str(tmp_path),
        )
        assert passed is False
        # The collision problem survives (no filter matches it).
        assert any("intersecting" in p.lower() for p in problems)
        # The gripper "solid block" false alarm was filtered by geometry
        # (empty parts → Check 5 has < 2 fingers but is_arm is False since
        # no arm keywords → no gripper problem; still, the false-alarm
        # filter catches "solid block" with "effector" context word).
        assert not any("solid block" in p.lower() for p in problems)

    def test_geometric_problem_forces_fail(self, monkeypatch, tmp_path):
        """HARD geometric problem (collision) → fail regardless of VLM.

        Note (updated 2026-06-22): soft pose warnings ("Arm too flat") no
        longer force FAIL — they are advisory.  This test now uses a hard
        collision problem to verify that real geometry failures still block.
        """
        vlm = '{"passed": true, "problems": []}'
        self._patch_pipeline(
            monkeypatch,
            ["Parts 'a' and 'b' overlap by 20x14x28mm — physically intersect"],
            vlm, tmp_path)

        import lang3d.tools.assembly_generator as ag
        passed, problems = ag._vlm_check_assembly(
            positions={}, parts=[],
            render_dir=str(tmp_path), api_key="dummy", base_url="dummy",
            real_stl_dir=str(tmp_path),
        )
        assert passed is False
        assert any("intersect" in p.lower() for p in problems)


class TestObbOverlapDetection:
    """Check 6 must use rotated (OBB-aware) world AABBs, not crude centre
    distance.  A long thin finger rotated across its sibling's volume has
    centres 32mm apart but boxes overlapping ~40mm — the old centre-distance
    heuristic missed this entirely."""

    def test_rotated_finger_overlap_detected(self):
        # Two 60mm-long fingers offset ±16 in X (centres 32mm apart) but
        # rotated ~73° about a compound axis — exactly the e2e failure case.
        # Centres are far apart yet the rotated boxes intersect.
        parts = [
            {"name": "gripper_finger_left",
             "dimensions": {"length": 60, "width": 14, "height": 28},
             "category": "mechanical"},
            {"name": "gripper_finger_right",
             "dimensions": {"length": 60, "width": 14, "height": 28},
             "category": "mechanical"},
        ]
        positions = {
            # Rotated 73° about axis (-0.86,-0.36,0.27) — same as real solver
            # output that caused the missed collision.
            "gripper_finger_left": {
                "position": [-13, -193, 366],
                "rotation": [-0.86, -0.36, 0.27, 73]},
            "gripper_finger_right": {
                "position": [13, -175, 370],
                "rotation": [-0.86, -0.36, 0.27, 73]},
        }
        # No joints → the pair is non-adjacent → must be checked.
        problems = _geometric_prevalidation(parts, positions, joints=[])
        overlap = [p for p in problems if "overlap" in p.lower()]
        assert overlap, (
            f"Rotated finger overlap must be detected; got: {problems}"
        )

    def test_well_separated_fingers_not_flagged(self):
        # Same fingers but offset ±40 (centres 80mm apart) — no overlap.
        parts = [
            {"name": "gripper_finger_left",
             "dimensions": {"length": 60, "width": 14, "height": 28}},
            {"name": "gripper_finger_right",
             "dimensions": {"length": 60, "width": 14, "height": 28}},
        ]
        positions = {
            "gripper_finger_left": {"position": [-40, 0, 0]},
            "gripper_finger_right": {"position": [40, 0, 0]},
        }
        problems = _geometric_prevalidation(parts, positions, joints=[])
        assert not any("overlap" in p.lower() for p in problems), (
            f"Separated fingers should not be flagged; got: {problems}"
        )

    def test_adjacent_parts_not_flagged(self):
        # Parent-child pair touching at a joint must be skipped.
        parts = [
            {"name": "base", "dimensions": {"length": 50, "width": 50, "height": 5}},
            {"name": "pillar", "dimensions": {"length": 10, "width": 10, "height": 40}},
        ]
        positions = {
            "base": {"position": [0, 0, 0]},
            "pillar": {"position": [0, 0, 22]},  # sits on base
        }
        joints = [{"parent": "base", "child": "pillar"}]
        problems = _geometric_prevalidation(parts, positions, joints=joints)
        assert not any("overlap" in p.lower() for p in problems)
