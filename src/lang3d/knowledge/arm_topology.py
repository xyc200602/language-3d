"""Parametric single-arm topology generator (2-7 DOF).

Replaces the three hand-written example JSONs (``EXAMPLE_ARM_STANDALONE`` /
``EXAMPLE_5DOF_ARM_REALISTIC`` / ``EXAMPLE_6DOF_BELT_DRIVE_ARM``) for the
LLM few-shot prompt.  Instead of the legacy keyword detector
(``is_5dof``/``is_6dof``/``else→4dof``) that fell back to a 4-DOF exemplar
for every non-5/6 request, ``build_arm_example(n_dof)`` synthesises a
correctly-structured arm for *any* requested degree count.

Topology convention (must match the system-prompt rules in
``assembly_generator.py`` rules 11/12/18 and the solver's
``_revolute_axis``):

  base_plate ──[rev z, top/bottom]──> base_yaw_servo      # the ONLY z yaw
                 │
                 └─[rev x, front/back]─> link₁            # pitch (shoulder)
                       │
                       └─[rev x, front/back]─> link₂     # pitch (elbow)
                             │
                             └─ ...                        # more pitch
                                   │
                                   └─[rev y, front/back]─> wrist_link  # roll
                                         │
                                         └─ gripper (4 fixed/prismatic parts)

The kinematic rules the solver already enforces (``assembly_solver.py`` is
fully DOF-agnostic — it walks the parent/child tree and accumulates
rotations, so a 7-DOF chain solves identically to a 4-DOF one):

* base yaw = the ONLY ``top/bottom`` joint, ``axis="z"`` (spins about vertical)
* every arm-segment joint = ``front/back`` so the link ``length`` becomes the
  axis-to-axis distance the IK solver needs
* pitch joints = ``axis="x"`` (perpendicular to the arm direction → vertical bend)
* wrist roll = ``axis="y"`` (along the arm direction → spins the end effector)
* ``default_angles`` give a bent zig-zag pose (never all-zero, or the arm
  collapses into a straight rod)

All dimensions come from the existing validated ranges (links 60-200mm,
servos ~28-40mm diameter) so the proportions sanitizer
(``_validate_proportions``) passes.
"""

from __future__ import annotations

import json
import math
from typing import Any, Literal


# ---------------------------------------------------------------------------
# DOF → joint-role schedule
# ---------------------------------------------------------------------------
#
# Each arm is built from a fixed repertoire of joint *roles*. The number of
# pitch joints grows with DOF; the wrist gets roll/yaw once DOF allows a
# wrist. This mirrors how real arms are laid out (more DOF = more wrist
# articulation, not arbitrarily long upper arms).
#
# role: ("pitch"|"yaw"|"roll", range_deg)
_YAW = ("yaw", (-180.0, 180.0))      # base rotation, axis z
_PITCH = ("pitch", (-150.0, 150.0))  # up/down bend, axis x
_ROLL = ("roll", (-180.0, 180.0))    # spin about arm axis, axis y


def _joint_schedule(n_dof: int) -> list[tuple[str, tuple[float, float]]]:
    """Map a DOF count to the ordered list of joint roles (excluding base).

    The first role is always the base yaw (a yaw that sits on the plate).
    The remaining DOF-1 are split between pitch (reach) and wrist
    articulation (roll/pitch). More DOF → richer wrist, because that is what
    extra DOF buys a real arm (a 6-DOF arm reaches with 3 and orients with 3).

    IMPORTANT: only the base role is ``yaw`` (axis z). Wrist orientation is
    done with ``pitch`` (axis x) and ``roll`` (axis y) only — never a second
    z-yaw, because the system-prompt rule mandates "the ONLY top/bottom z
    joint is the base" and a second z-yaw on a front/back anchor would spin
    the horizontal arm about a nonsensical axis.
    """
    if n_dof < 2:
        n_dof = 2
    if n_dof > 7:
        n_dof = 7

    # Base yaw always consumes 1 DOF.
    roles: list[tuple[str, tuple[float, float]]] = [_YAW]

    # Pitch joints provide reach. We always want at least 2 (shoulder+elbow)
    # so the arm can fold — a single-pitch arm is just a stick.
    remaining = n_dof - 1
    n_pitch = min(remaining, 2)
    remaining -= n_pitch
    for _ in range(n_pitch):
        roles.append(_PITCH)

    # Extra DOF beyond shoulder+elbow go to the wrist. A spherical wrist
    # is pitch→roll→pitch (or roll→pitch→roll); we cycle in that order so
    # the end effector gains orientation freedom without duplicating the
    # base z-yaw.
    if remaining >= 1:
        wrist_cycle = [_ROLL, _PITCH, _ROLL]
        for i in range(remaining):
            roles.append(wrist_cycle[i % len(wrist_cycle)])

    return roles


# ---------------------------------------------------------------------------
# Zig-zag default-angle sequence (parameterised)
# ---------------------------------------------------------------------------

# A natural reaching pose: shoulder down a lot, elbow up, wrist back to level.
# The legacy sanitizer hard-coded [-45,-30,+15,-10] for 3+ pitch joints and
# [-45,+30] for exactly 2 (assembly_generator.py:1796). This generalises to
# any count by cycling the pattern, keeping the arm extended rather than
# curled.
_ZIGZAG_BASE = [-45.0, 30.0, -15.0, 10.0]


def zigzag_angles(n_pitch_joints: int) -> list[float]:
    """Return a sign-alternating default-angle sequence for *n* pitch joints.

    For >4 pitch joints the base pattern repeats (modulo sign preservation),
    which keeps every arm extended outward instead of folding on itself.
    """
    if n_pitch_joints <= 0:
        return []
    if n_pitch_joints == 1:
        return [-35.0]
    out: list[float] = []
    for i in range(n_pitch_joints):
        out.append(_ZIGZAG_BASE[i % len(_ZIGZAG_BASE)])
    return out


# ---------------------------------------------------------------------------
# Dimension heuristics (from validated EXAMPLE_ARM_STANDALONE ranges)
# ---------------------------------------------------------------------------

# Servo diameters shrink down the chain (base heaviest, wrist lightest).
_SERVO_DIMS = [
    {"diameter": 40, "height": 35},  # base / shoulder
    {"diameter": 36, "height": 30},  # elbow
    {"diameter": 30, "height": 28},  # upper wrist
    {"diameter": 28, "height": 26},  # distal wrist
]

# Link lengths decrease down the arm (shoulder longest, wrist shortest).
_LINK_LENGTHS = [120, 100, 80, 60]
_LINK_CROSS = {"width": 25, "height": 15}

_BASE_PLATE = {"length": 200, "width": 150, "height": 8}
_GRIPPER_BASE = {"length": 28, "width": 50, "height": 32}
_GRIPPER_SERVO = {"length": 23, "width": 12, "height": 22}
_FINGER = {"length": 60, "width": 14, "height": 28}


def _servo_dims(idx: int) -> dict[str, int]:
    return dict(_SERVO_DIMS[min(idx, len(_SERVO_DIMS) - 1)])


def _link_dims(idx: int) -> dict[str, int]:
    base = dict(_LINK_CROSS)
    base["length"] = _LINK_LENGTHS[min(idx, len(_LINK_LENGTHS) - 1)]
    return base


# ---------------------------------------------------------------------------
# axis / anchor mapping per role
# ---------------------------------------------------------------------------

_ROLE_AXIS: dict[str, str] = {"pitch": "x", "yaw": "z", "roll": "y"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_arm_example(n_dof: int) -> str:
    """Build a complete assembly-JSON string for an *n_dof* single arm.

    Returns JSON text matching the schema of ``EXAMPLE_ARM_STANDALONE``:
    ``{name, description, default_angles, parts[], joints[]}``.  The result
    is a standalone arm on a base plate with a two-finger gripper — directly
    consumable by ``generate_assembly_from_nl`` as the few-shot example.
    """
    n_dof = max(2, min(int(n_dof), 7))
    schedule = _joint_schedule(n_dof)

    parts: list[dict[str, Any]] = [
        {"name": "base_plate", "category": "structural",
         "description": "底座安装板", "material": "Aluminum",
         "dimensions": dict(_BASE_PLATE)},
    ]
    joints: list[dict[str, Any]] = []
    default_angles: dict[str, float] = {}

    prev_part = "base_plate"
    pitch_idx = 0  # counts pitch joints for zig-zag + dims

    # Build the joint→link chain. Each role becomes a servo (revolute/fixed)
    # followed by a link. The base yaw mounts on the plate via top/bottom;
    # every later joint is front/back.
    for role_i, (role, rng) in enumerate(schedule):
        axis = _ROLE_AXIS[role]
        is_base = role_i == 0
        servo_name = "base_yaw_servo" if is_base else f"{role}_servo_{role_i}"
        link_name = f"{role}_link_{role_i}"

        parts.append({
            "name": servo_name, "category": "actuator",
            "description": f"{'底座旋转' if is_base else role}舵机",
            "material": "Steel",
            "dimensions": _servo_dims(pitch_idx if role == "pitch" else role_i),
        })
        parts.append({
            "name": link_name, "category": "structural",
            "description": f"{role}连杆", "material": "Aluminum",
            "dimensions": _link_dims(pitch_idx if role == "pitch" else role_i),
        })

        # servo joint: parent→servo (revolute about the role axis)
        joints.append({
            "type": "revolute", "parent": prev_part, "child": servo_name,
            "axis": axis, "range_deg": list(rng),
            "parent_anchor": "top" if is_base else "front",
            "child_anchor": "bottom" if is_base else "back",
        })
        # link joint: servo→link (fixed — the link is rigidly bolted to the
        # servo output, same convention as EXAMPLE_ARM_STANDALONE)
        joints.append({
            "type": "fixed", "parent": servo_name, "child": link_name,
            "parent_anchor": "front", "child_anchor": "back",
        })

        # default angle only on pitch joints (yaw/roll stay 0 in the home pose
        # so the arm points forward and the gripper is symmetric — matches
        # the _ensure_arm_default_angles sanitizer expectation).
        if role == "pitch":
            pitch_idx += 1

        prev_part = link_name

    # Apply zig-zag default angles to the pitch joints. Only pitch (axis=x)
    # joints get a non-zero home angle so the arm folds into a reaching pose
    # instead of lying flat; yaw/roll stay 0 so the arm points forward and
    # the gripper stays symmetric (matches _ensure_arm_default_angles).
    n_pitch = sum(1 for r, _ in schedule if r == "pitch")
    zz = zigzag_angles(n_pitch)
    default_angles: dict[str, float] = {}
    pi = 0
    for j in joints:
        if j["type"] == "revolute" and j.get("axis") == "x":
            default_angles[j["child"]] = zz[pi] if pi < len(zz) else 0.0
            pi += 1

    # --- gripper (constant 4-part structure, same as the 4-DOF exemplar) ---
    parts.extend([
        {"name": "gripper_base", "category": "mechanical",
         "description": "夹爪基座(含直线导轨槽和舵机安装座)", "material": "PLA",
         "dimensions": dict(_GRIPPER_BASE)},
        {"name": "gripper_servo", "category": "actuator",
         "description": "夹爪驱动舵机SG90", "material": "Steel",
         "dimensions": dict(_GRIPPER_SERVO)},
        {"name": "gripper_finger_left", "category": "mechanical",
         "description": "夹爪左手指(含滑动导轨和L形指尖)", "material": "PLA",
         "dimensions": dict(_FINGER)},
        {"name": "gripper_finger_right", "category": "mechanical",
         "description": "夹爪右手指(含滑动导轨和L形指尖)", "material": "PLA",
         "dimensions": dict(_FINGER)},
    ])
    joints.extend([
        {"type": "fixed", "parent": prev_part, "child": "gripper_base",
         "parent_anchor": "front", "child_anchor": "back"},
        {"type": "fixed", "parent": "gripper_base", "child": "gripper_servo",
         "parent_anchor": "top", "child_anchor": "bottom",
         "connection_method": "bolted",
         "connection_detail": {"bolt_size": "M2", "bolt_count": 2}},
        {"type": "prismatic", "parent": "gripper_base",
         "child": "gripper_finger_left", "axis": "x", "range_deg": [-8, 12],
         "parent_anchor": "front", "child_anchor": "back", "offset": [-16, 0, 0]},
        {"type": "prismatic", "parent": "gripper_base",
         "child": "gripper_finger_right", "axis": "x", "range_deg": [-8, 12],
         "parent_anchor": "front", "child_anchor": "back", "offset": [16, 0, 0],
         "mimic_joint": "gripper_finger_left", "mimic_multiplier": -1.0,
         "mimic_offset": 0},
    ])

    assembly = {
        "name": f"{n_dof}dof_robot_arm",
        "description": f"{n_dof}自由度单机械臂",
        "default_angles": default_angles,
        "parts": parts,
        "joints": joints,
    }
    return json.dumps(assembly, ensure_ascii=False, indent=2)


def parse_dof(description: str) -> int | None:
    """Extract an explicit DOF count from a natural-language description.

    Recognises Chinese numerals (两/二/三...七), Arabic digits followed by a
    DOF unit (自由度/dof/轴/axis/joint), and bare digit forms like "6dof".
    Returns ``None`` when no DOF is stated (the caller then defaults to 4).

    This replaces the keyword-substring detector that matched "5"/"6"
    anywhere in the text (false-positiving on "5V", "6 wheels", etc.).
    """
    if not description:
        return None
    t = description.lower()

    # "N自由度" / "N dof" / "N轴" / "N-axis" / "N关节"
    import re
    m = re.search(r"(\d+)\s*(?:自由度|dof|-?axis|轴|关节|joint)", t)
    if m:
        n = int(m.group(1))
        if 2 <= n <= 7:
            return n
    # "Ndof" / "N-dof" (no space, common in prompts)
    m = re.search(r"\b(\d)\s*-?dof\b", t)
    if m:
        n = int(m.group(1))
        if 2 <= n <= 7:
            return n

    # Chinese numerals preceding 自由度/轴/dof
    cn_map = {"两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7}
    for cn, val in cn_map.items():
        if cn in description and any(
            u in description for u in ("自由度", "轴", "dof", "关节")
        ):
            return val
    return None
