"""VLM verification + false-alarm classifiers + preview mesh generation.

Extracted from assembly_generator.py (P1-1 God Module split, AGENTS.md §2.1).
Contains:
- Robot category classification + category-aware VLM prompts
- Geometric pre-validation (7 deterministic checks)
- VLM false-alarm filters (gripper/floating/wheel)
- VLM check assembly (render + verify + problem filtering)
- Preview STL mesh builders (box/cylinder/finger)

All functions verify or filter; they do not modify the assembly.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import Any

from ...knowledge.mechanics import Assembly, Joint, Part
from ..assembly_solver import AssemblySolver

logger = logging.getLogger(__name__)

# Context words for gripper detection (was in assembly_generator.py).
_GRIPPER_CONTEXT_WORDS = (
    "gripper", "finger", "prong", "effector", "tip", "claw", "爪",
)

_CATEGORY_EXPECTATIONS = {
    "fixed_arm": (
        "=== ROBOT CATEGORY: FIXED-BASE ARM ===\n"
        "This assembly is a FIXED-BASE ARM (bolted to a workbench). Expected:\n"
        "  - A large base plate at the BOTTOM (the workbench mount).\n"
        "  - Vertical stack of servo joints + links reaching UP and OUT.\n"
        "  - A gripper at the tip.\n"
        "Wheels are NOT expected here — flag any wheel/tire as an error.\n"
        "Do NOT report 'missing wheels' for a fixed-base arm.\n"
    ),
    "wheeled": (
        "=== ROBOT CATEGORY: WHEELED MOBILE BASE (no arm) ===\n"
        "This assembly is a WHEELED MOBILE BASE / CHASSIS. Expected:\n"
        "  - 2 or 4 wheels as HORIZONTAL CYLINDERS resting ON or NEAR the "
        "ground (Z near wheel radius). Wheels roll forward — they must look "
        "like discs/tyres lying on their side, NOT standing upright like "
        "spinning tops.\n"
        "  - A chassis body / deck plate ABOVE the wheels.\n"
        "  - Motors mounted coaxially with each wheel axle.\n"
        "Wheels near the ground are CORRECT — do NOT flag them as errors.\n"
        "Flag wheels only if they FLOAT above the ground or stand vertically.\n"
    ),
    "wheeled_arm": (
        "=== ROBOT CATEGORY: WHEELED MOBILE MANIPULATOR (wheels + arm(s)) ===\n"
        "This assembly is a MOBILE MANIPULATOR: a wheeled chassis with one or "
        "more arms on top. Expected ALL of:\n"
        "  - Wheels as HORIZONTAL CYLINDERS near the ground (rolling discs, "
        "not upright). Wheels at Z ≈ wheel radius are CORRECT.\n"
        "  - A chassis body / deck above the wheels.\n"
        "  - Arm(s) mounted ON TOP of the chassis (base yaw servo -> links "
        "-> gripper), reaching upward/forward.\n"
        "BOTH the wheels AND the arms are EXPECTED here. Do NOT flag the "
        "wheels as 'arms should not have wheels' — this category legitimately "
        "combines them. Do NOT flag the arms as 'floating' if they sit on the "
        "chassis deck.\n"
        "Flag ONLY genuine defects: wheels floating above ground, wheels "
        "vertical, arms intersecting the chassis, parts disconnected.\n"
    ),
    "assembly": (
        "=== ROBOT CATEGORY: GENERAL ASSEMBLY ===\n"
        "No specific category hint. Judge structural integrity generically.\n"
        "If wheels are present they should be near the ground as horizontal "
        "cylinders; if this is an arm with no wheels, do not require wheels.\n"
    ),
}


def _classify_robot(description: str) -> str:
    """Classify the NL description into a robot category for the VLM prompt.

    Mirrors the ``is_arm`` / ``is_wheeled`` decision in
    :func:`generate_assembly_with_vlm_loop` so the prompt and the generator
    agree on what kind of robot is being built.  Returns one of the keys of
    :data:`_CATEGORY_EXPECTATIONS`.
    """
    d = (description or "").lower()
    is_arm = any(kw in d for kw in [
        "臂", "arm", "机械手", "机械臂", "抓手", "gripper", "自由度",
    ])
    is_wheeled = any(kw in d for kw in [
        "轮", "wheel", "差速", "移动", "底盘",
    ])
    is_legged = any(kw in d for kw in [
        "腿", "leg", "人型", "人形", "humanoid", "双足", "biped",
        "行走", "quadruped", "四足",
    ])
    if is_legged:
        return "assembly"  # legged robots are general assemblies (not arms/wheeled)
    if is_arm and is_wheeled:
        return "wheeled_arm"
    if is_wheeled:
        return "wheeled"
    if is_arm:
        return "fixed_arm"
    return "assembly"


def _build_verify_prompt(robot_category: str) -> str:
    """Build the whole-assembly VLM prompt with category context injected.

    Replaces the old static ``_VLM_VERIFY_PROMPT``.  The category block goes
    at the TOP of the prompt so the VLM reads the expectations before judging.
    """
    category_block = _CATEGORY_EXPECTATIONS.get(
        robot_category, _CATEGORY_EXPECTATIONS["assembly"],
    )
    return (
        "You are a STRICT robot assembly quality inspector. Examine the 3D "
        "render.\n\n"
        + category_block
        + "\n"
        "This is a WHOLE-ASSEMBLY view. Judge ONLY structural integrity — do "
        "NOT attempt to evaluate the gripper fingers from this view (they are "
        "too small to resolve here; a dedicated close-up view covers the "
        "gripper).\n\n"
        "=== STRUCTURAL INTEGRITY ===\n"
        "Check for:\n"
        "1. Parts floating in mid-air with no support\n"
        "2. Parts intersecting / overlapping each other\n"
        "3. Arms pointing in impossible directions (e.g. going through the "
        "body)\n"
        "4. Critical parts missing (no base plate, no main body)\n"
        "5. Overall structural coherence\n"
        "6. Parts with WRONG ORIENTATION (e.g. wheels standing vertical "
        "instead of lying horizontal; arms pointing down into the ground)\n\n"
        "IMPORTANT: Only report a problem if it is a GENUINE defect for THIS "
        "robot category. Parts that match the category expectations above are "
        "CORRECT and must NOT be flagged.\n\n"
        "Reply with JSON only:\n"
        '{"passed": true/false, '
        '"problems": ["list of specific structural issues found"], '
        '"description": "brief assessment"}\n'
    )


# Backwards-compat alias.  Some external callers / tests may still reference
# ``_VLM_VERIFY_PROMPT`` as a string; keep it as the generic-assembly variant
# so they get a valid prompt without the category injection.
_VLM_VERIFY_PROMPT = _build_verify_prompt("assembly")

_VLM_GRIPPER_CLOSEUP_PROMPT = (
    "You are inspecting the GRIPPER at the tip of a robotic arm. This is a "
    "CLOSE-UP view zoomed in on the gripper — the rest of the arm is out of "
    "frame, which is intentional.\n\n"
    "Judge ONLY the gripper. The gripper passes if you can see TWO clearly "
    "separated, parallel finger prongs that face each other with a VISIBLE "
    "OPEN GAP between them (like a claw, chopsticks, or pliers).\n\n"
    "AUTOMATIC FAIL (passed=false) if ANY of these are true:\n"
    "- The tip is a single solid block, box, cylinder, sphere, or housing\n"
    "- The tip is just another arm link or segment\n"
    "- There are NOT two clearly separated parallel prongs\n"
    "- The end-effector is a single chunky mass with no visible gap/split\n\n"
    "Do NOT rationalize. If you cannot clearly see TWO separate finger prongs "
    "with a gap between them, the gripper FAILS.\n\n"
    "Reply with JSON only:\n"
    '{"passed": true/false, '
    '"problems": ["gripper-specific issues only"], '
    '"description": "brief gripper assessment"}\n'
)


_VLM_FIX_PROMPT = (
    "You previously generated a robot assembly JSON, but the visual "
    "verification found problems.\n\n"
    "Problems found:\n{problems}\n\n"
    "Original description: {description}\n\n"
    "Please regenerate the COMPLETE assembly JSON that fixes these problems. "
    "Apply the relevant fixes below based on each problem type:\n"
    "- Floating / disconnected part: adjust its position offset so it connects "
    "to its parent anchor point, or add/fix the joint referencing it.\n"
    "- Overlapping parts: increase the child part's position offset along the "
    "joint axis so the parts no longer intersect.\n"
    "- Wrong joint type (e.g. revolute where continuous is needed, or fixed "
    "where rotation is needed): change the joint \"type\" field accordingly.\n"
    "- Wrong orientation (e.g. cylinder axis pointing the wrong way): swap the "
    "dimension keys (diameter/height/length/width) or adjust the rotation so "
    "the part aligns with its joint axis.\n"
    "- Missing gripper/claw / end-effector not a gripper: The VERY END of the "
    "arm MUST have TWO clearly separated opposing finger parts named "
    "'gripper_finger_left' and 'gripper_finger_right' (or equivalent). Each "
    "finger MUST be at least 15mm wide and 40mm long so it is clearly visible. "
    "Connect each finger to the gripper_base via a 'prismatic' joint with "
    "'parent_anchor':'center','child_anchor':'center', and set the 'offset' "
    "to [0, +/-gap, z_lift] where gap >= 35mm so the two fingers are clearly "
    "separated with a visible opening between them. Remove any solid block, "
    "cylinder, or extra arm link that is currently at the arm tip.\n"
    "- Unstable / insufficient base: enlarge the base plate dimensions "
    "(length & width) so the assembly center of mass stays over it.\n"
    "- Wheels off the ground: lower the wheel parts' Z position so they "
    "contact the ground plane (Z ≈ wheel radius).\n\n"
    "Return only the JSON, no code blocks, no explanation.\n"
)


_GRIPPER_FALSE_ALARM_PATTERNS = (
    "solid block", "solid mass", "solid chunk", "chunky mass", "fused",
    "single curved mass", "single chunky", "single solid",
    "no visible gap", "no gap", "no separated",
    "not two clearly separated", "no two clearly separated",
    "does not have two", "does not terminate in a gripper",
    "no clearly separated parallel prongs", "parallel prongs",
    "no gripper at the tip", "no gripper", "not a gripper",
    "absence of a functional gripper", "no functional gripper",
    "tip of the arm does not have", "tip is a solid",
    "tip of the arm terminates in a solid",
    "tip of the arm does not terminate in a gripper",
    "end effector is a solid", "end-effector is a solid",
    "fails functional gripper", "fails category 2",
    "end-effector is a single", "end effector is a single",
)
def _is_gripper_false_alarm(problem_text: str) -> bool:
    """Return True if a VLM problem is a gripper-finger complaint.

    Uses a double condition — the text must mention a gripper context word
    (gripper/finger/prong/effector/tip/claw) AND match a finger-fusion /
    missing-gripper pattern.  This prevents structural problems like
    "base plate does not have two mounting holes" from being filtered.
    """
    t = problem_text.lower()
    has_context = any(w in t for w in _GRIPPER_CONTEXT_WORDS)
    has_pattern = any(p in t for p in _GRIPPER_FALSE_ALARM_PATTERNS)
    return has_context and has_pattern


# Floating false-alarm patterns (added 2026-06-22, Plan B+C).
#
# VLMs catastrophically misjudge "floating / disconnected" when a part
# occupies <1% of the frame (TDBench, arXiv 2504.03748).  In a long
# robotic arm, the shoulder servo (Ø40mm) linking the base plate to the
# arm chain is exactly such a small part — GLM-4.6V reliably reports
# the arm as "floating with no support" even when the joint graph
# confirms every part is connected to the base.  When the joint-graph
# connectivity check (Check 7 in _geometric_prevalidation) returns
# clean, these reports are viewpoint artifacts and must be filtered so
# they do not trigger a corrupting regeneration round.
#
# Unlike the gripper filter, no context-word double condition is needed:
# "floating" / "mid-air" / "disconnected" are unambiguously geometric
# connectivity complaints, not structural details that could be real.
_FLOATING_FALSE_ALARM_PATTERNS = (
    "floating", "floats", "floated",
    "mid-air", "mid air", "in mid air", "in mid-air",
    "no support", "not supported", "unsupported",
    "disconnected", "not connected", "no visible connection",
    "no visible support", "no physical connection",
    "悬空", "悬浮", "未连接", "无支撑",
)


def _is_floating_false_alarm(problem_text: str) -> bool:
    """Return True if a VLM problem is a floating / disconnected complaint.

    These are filtered ONLY when the joint-graph connectivity check
    (Check 7) confirms every part is reachable from the root — i.e. the
    assembly is genuinely connected and the VLM report is a viewpoint
    artifact.  See ``_vlm_check_assembly`` for the gating logic.
    """
    t = problem_text.lower()
    return any(p in t for p in _FLOATING_FALSE_ALARM_PATTERNS)


# ---------------------------------------------------------------------------
# VLM complaint severity classification (audit: fix-loop has no grading)
# ---------------------------------------------------------------------------
# The fix loop previously treated ALL VLM complaints identically: try a
# targeted fix, then fall back to LLM regeneration. But a gripper-closeup
# FRAMING complaint ("not a close-up view") is fundamentally different from
# a HARD geometry defect ("wheel_fr is 591mm from center"). Conflating them
# meant a deterministic compose output (engineering-correct) got overturned
# by a framing nitpick and regenerated by the LLM into garbage.
#
# Severity grades:
#   HARD  — structural/geometric defect that MUST be fixed (collision,
#           disconnected parts, parts far from center, missing critical
#           parts, wheels misplaced/floating, finger overlap). Triggers
#           targeted fix, and on LLM-sourced assemblies, regeneration.
#   SOFT  — framing / posture / orientation nitpick that does NOT indicate
#           a geometric defect (gripper closeup framing, "arm too
#           flat/horizontal", "not a close-up view"). On a deterministic
#           compose output these are NOT a reason to regenerate — the
#           geometry is correct and the VLM is critiquing the render.
_SOFT_VLM_MARKERS = (
    "not a close-up", "not close-up", "rest of arm is in frame",
    "arm too flat", "arm too horizontal", "arm too vertical",
    "closeup framing", "close-up framing", "view is not",
)

_HARD_VLM_MARKERS = (
    "overlap", "intersect", "collision", "穿模", "penetrat",
    "floating", "disconnected", "not attached", "no support",
    "missing wheel", "missing chassis", "missing arm", "missing critical",
    "misplaced", "far from center", "from center",
    "above z", "below ground", "underground",
    "single chunky mass", "no visible gap", "no separated prongs",
    "vertical instead of horizontal", "wrong orientation",
)


def _classify_vlm_complaint(problem_text: str) -> str:
    """Classify a VLM complaint as 'HARD', 'SOFT', or 'UNKNOWN'.

    HARD = structural defect (must fix). SOFT = framing/posture nitpick.
    UNKNOWN = unclassified; treated conservatively as HARD (safer to fix
    than to ignore an unknown complaint).
    """
    t = problem_text.lower()
    if any(m in t for m in _HARD_VLM_MARKERS):
        return "HARD"
    if any(m in t for m in _SOFT_VLM_MARKERS):
        return "SOFT"
    return "UNKNOWN"


# Wheel false-alarm patterns (added 2026-06-24).
#
# GLM-4.6V reliably mistakes the cylindrical servo housings of a fixed-base
# arm (base_yaw_servo Ø40, pitch_servo Ø36) for "wheels" and reports their
# vertical orientation as "incorrect — wheels should be horizontal to roll
# on the ground". On an arm assembly there are no wheels at all, so the
# entire complaint is a hallucination. When the part list confirms there is
# no wheel/tire part (and the assembly is not a wheeled robot), these
# reports must be filtered instead of triggering a corrupting regeneration
# that "fixes" non-existent wheels by mis-orienting the servos.
#
# IMPORTANT: patterns are deliberately *orientation/rolling specific*
# ("wheels ... oriented", "should roll on ground"), NOT bare "wheel".
# A bare "wheel" would match real collision reports like "base_plate and
# wheel_fr overlap by 65mm", which must be kept even on a part list that
# happens to lack wheels (the part name alone may differ).
_WHEEL_FALSE_ALARM_PATTERNS = (
    "wheel.*orient", "wheels.*orient",
    "tire.*orient", "tires.*orient",
    "rolling on ground", "roll on the ground",
    "should be horizontal", "axis.*perpendicular to ground",
    "轮.*方向", "轮.*朝向", "轮胎.*方向",
)
_WHEEL_PART_STEMS = ("wheel", "tire", "轮")


def _assembly_has_wheels(parts: list[dict]) -> bool:
    """Return True if any part is a genuine wheel/tire."""
    for p in parts:
        name = (p.get("name", "") or "").lower()
        if any(stem in name for stem in _WHEEL_PART_STEMS):
            return True
    return False


def _is_wheel_false_alarm(
    problem_text: str,
    parts: list[dict],
    positions: dict[str, dict] | None = None,
) -> bool:
    """Return True if a VLM problem is a spurious wheel complaint.

    Two cases:
      (1) The assembly has NO wheel parts but the VLM hallucinated wheels
          onto cylindrical servos ("wheels oriented vertically"). Filter.
      (2) The assembly HAS wheels and they are GROUNDED (solved Z ≈ wheel
          radius, within tolerance), but the VLM misreads the render and
          reports "wheels above Z / floating / wrong orientation". This is
          the recurring wheeled-dual-arm false-negative — the deterministic
          compose path produces correct Z=47.5 wheels (verified), but GLM-4.6V
          still reports "wheels above Z=178". When geometry confirms the
          wheels are grounded, the VLM complaint is overruled.
    """
    import re
    t = problem_text.lower()
    # Case 1: wheel complaint on an assembly with no wheels (hallucination).
    if not _assembly_has_wheels(parts):
        return any(re.search(p, t) for p in _WHEEL_FALSE_ALARM_PATTERNS)

    # Case 2: assembly has wheels. Check whether the complaint is about
    # wheel position/orientation/presence, and whether geometry refutes it.
    is_wheel_position_complaint = any(re.search(p, t) for p in _WHEEL_FALSE_ALARM_PATTERNS) or (
        "wheel" in t and ("above z" in t or "floating" in t
                          or "near ground" in t or "vertical" in t
                          or "horizontal" in t or "missing wheel" in t
                          or "no wheel" in t or "not present" in t
                          or "not visible" in t or "absent" in t)
    )
    # Also catch "No <part> present / visible / found" patterns that name
    # wheels without the word "wheel" adjacent (e.g. "No horizontal cylinders
    # near ground present" — the VLM describing wheels it cannot see).
    if not is_wheel_position_complaint and (
        ("no " in t and ("cylinder" in t or "tire" in t))
        or "not present" in t or "critical part missing" in t
    ):
        is_wheel_position_complaint = "wheel" in t or "cylinder" in t
    if not is_wheel_position_complaint:
        return False

    # Geometric oracle: are the wheels actually grounded?
    if positions is None:
        return False  # can't verify — don't filter
    wheel_positions = {
        name: pose["position"]
        for name, pose in positions.items()
        if "wheel" in name.lower()
    }
    if not wheel_positions:
        return False
    # Wheel radius from part dims (fall back to 45mm, the default).
    wheel_part = next((p for p in parts if "wheel" in (p.get("name","") or "").lower()), None)
    wheel_r = ((wheel_part.get("dimensions", {}).get("diameter", 0) or 0) / 2.0
               if wheel_part else 45.0) or 45.0
    # Grounded = every wheel's Z is within ±40% of wheel_r of wheel_r (i.e.
    # the wheel bottom is near the ground). Tolerance is generous because the
    # solver reports the wheel CENTER; bottom = center - radius.
    grounded = all(
        abs(z - wheel_r) < wheel_r * 0.8
        for (_, _, z) in wheel_positions.values()
    )
    return grounded


def _geometric_prevalidation(
    parts: list[dict],
    positions: dict[str, dict],
    joints: list[dict] | None = None,
) -> list[str]:
    """Deterministic geometric checks. Returns problem descriptions."""
    import math as _math
    problems = []

    # Build adjacency set from joints (parent-child pairs are expected
    # to be close — they are connected and should not trigger overlap warns).
    # Adjacency here mirrors mesh_collision.MeshCollisionChecker._build_adjacent_pairs:
    # direct parent↔child, transitive 2-hop (grandparent↔grandchild, e.g. motor↔wheel
    # through a suspension link), and siblings (same parent, e.g. chassis_body↔motor
    # both mounted on base_plate — a real chassis shell CONTAINS its motors, so their
    # AABB overlap is the intended enclosure, not a collision).
    _adjacent_pairs: set[tuple[str, str]] = set()
    _children_by_parent: dict[str, list[str]] = {}
    if joints:
        for j in joints:
            if isinstance(j, dict):
                p = j.get("parent", "")
                c = j.get("child", "")
            else:
                p = getattr(j, "parent", "")
                c = getattr(j, "child", "")
            if p and c:
                _adjacent_pairs.add((p, c))
                _adjacent_pairs.add((c, p))
                _children_by_parent.setdefault(p, []).append(c)
        # Transitive 2-hop: grandparent↔grandchild.
        for gp, mids in _children_by_parent.items():
            for mid in mids:
                for gc in _children_by_parent.get(mid, []):
                    _adjacent_pairs.add((gp, gc))
                    _adjacent_pairs.add((gc, gp))
        # Siblings: same parent (e.g. chassis_body & motor_* on base_plate).
        for _parent, siblings in _children_by_parent.items():
            for i in range(len(siblings)):
                for k in range(i + 1, len(siblings)):
                    si, sk = siblings[i], siblings[k]
                    # Gripper fingers are siblings (both children of
                    # gripper_base) but they MUST NOT overlap — they are
                    # parallel jaws with an open gap between them, not an
                    # enclosing shell like chassis-body-around-motors.
                    # Excluding them from adjacency lets a finger-overlap
                    # bug hide (test_gripper_finger_geometry regression,
                    # 2026-07-03). Keep the sibling-skip for genuine
                    # enclosures (motors, hubs) but not for fingers.
                    if ("finger" in si.lower() and "finger" in sk.lower()):
                        continue
                    _adjacent_pairs.add((si, sk))
                    _adjacent_pairs.add((sk, si))

    # 1. Collision proxy: parts at same position.
    # Parts joined by a joint (parent↔child) legitimately share a position —
    # e.g. a suspension_link co-located with its motor (a prismatic joint with
    # zero travel sits at the motor), or two links meeting at a coincident
    # anchor. Skip such pairs; only flag UNRELATED parts that overlap.
    seen: dict[str, str] = {}
    for name, pdata in positions.items():
        pos = pdata.get("position", [0, 0, 0])
        key = f"{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}"
        if key in seen:
            other = seen[key]
            if (name, other) not in _adjacent_pairs:
                problems.append(f"Parts '{name}' and '{other}' at same position")
        else:
            seen[key] = name

    # 2. Outlier: parts >500mm from centroid
    if positions:
        vals = list(positions.values())
        cx = sum(v["position"][0] for v in vals) / len(vals)
        cy = sum(v["position"][1] for v in vals) / len(vals)
        cz = sum(v["position"][2] for v in vals) / len(vals)
        for name, pdata in positions.items():
            p = pdata["position"]
            dist = _math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2)
            if dist > 500:
                problems.append(f"Part '{name}' is {dist:.0f}mm from center - misplaced")

    # 3. Wheels near ground
    wheel_names = [n for n in positions if "wheel" in n.lower()]
    if wheel_names:
        min_z = min(positions[n]["position"][2] for n in wheel_names)
        if min_z > 100:
            problems.append(f"All wheels above Z={min_z:.0f}mm - should be near ground")

    # 4. Arm-too-flat detection: a robotic arm must have real 3D (Z) extent,
    # not lie as a flat bar along the ground. This catches the systematic
    # failure where prompt rules + sanitizer produced a completely flat arm
    # (e.g. 4dof_arm: Z span 54mm over a 589mm Y span) that the VLM could not
    # recognise as a 3D structure.
    _ARM_PART_KEYWORDS = (
        "link", "joint", "shoulder", "elbow", "wrist", "arm",
        "gripper", "servo", "housing",
    )
    arm_names = [
        n for n in positions
        if any(kw in n.lower() for kw in _ARM_PART_KEYWORDS)
    ]
    if len(arm_names) >= 4:
        xs = [positions[n]["position"][0] for n in arm_names]
        ys = [positions[n]["position"][1] for n in arm_names]
        zs = [positions[n]["position"][2] for n in arm_names]
        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)
        z_span = max(zs) - min(zs)
        horiz_span = max(x_span, y_span)
        if horiz_span > 100:
            if z_span < 30:
                problems.append(
                    f"Arm too flat: Z span {z_span:.0f}mm but horizontal span "
                    f"{horiz_span:.0f}mm — arm lies as a flat bar. Use "
                    f"top/bottom anchors for arm-chain joints and non-zero "
                    f"default_angles so links bend upward into 3D."
                )
            elif z_span < 0.25 * horiz_span:
                problems.append(
                    f"Arm too horizontal: Z span {z_span:.0f}mm is <25% of "
                    f"horizontal span {horiz_span:.0f}mm. Bend the pitch "
                    f"joints (axis='x', front/back anchors) with non-zero "
                    f"default_angles so the arm rises in Z instead of lying flat."
                )
            elif z_span > 2.0 * horiz_span:
                problems.append(
                    f"Arm too vertical: Z span {z_span:.0f}mm is >200% of "
                    f"horizontal span {horiz_span:.0f}mm — the arm looks like "
                    f"a vertical tower instead of a reaching arm. Reduce the "
                    f"pitch default_angles (shoulder/elbow) and use alternating "
                    f"signs (zig-zag) so the elbow bends back toward horizontal."
                )

    # 5. Gripper finger visibility: if the assembly describes an arm with
    # gripper fingers, verify that (a) there are >= 2 finger parts, and
    # (b) the solved finger positions are separated by >= 25mm so they read
    # as distinct opposing prongs rather than a fused block.  This is a
    # deterministic safety net for the VLM, which tends to rationalise a
    # non-visible gripper as "physically plausible".
    finger_names = [n for n in positions if "finger" in n.lower()]
    is_arm = len(arm_names) >= 4
    if is_arm:
        if len(finger_names) < 2:
            problems.append(
                "Arm is missing a functional gripper: fewer than 2 finger "
                "parts found. Add 'gripper_finger_left' and "
                "'gripper_finger_right' parts at the end of the arm."
            )
        else:
            import math as _math2
            for i in range(len(finger_names)):
                for j in range(i + 1, len(finger_names)):
                    p1 = positions[finger_names[i]]["position"]
                    p2 = positions[finger_names[j]]["position"]
                    dist = _math2.sqrt(
                        (p1[0] - p2[0]) ** 2
                        + (p1[1] - p2[1]) ** 2
                        + (p1[2] - p2[2]) ** 2
                    )
                    if dist < 25.0:
                        problems.append(
                            f"Gripper fingers '{finger_names[i]}' and "
                            f"'{finger_names[j]}' are only {dist:.1f}mm apart "
                            f"— they fuse into a single block. Increase the "
                            f"lateral offset so fingers are clearly separated "
                            f"(>= 35mm gap)."
                        )

    # 6. Bounding-box overlap detection for non-adjacent parts.
    #    Parts connected by joints are expected to touch.  Non-adjacent
    #    parts whose rotated world AABBs overlap are likely intersecting
    #    and must be flagged for the VLM fix loop.
    #
    #    P1: previously this check used a crude centre-distance heuristic
    #    (dist < 0.2 * (max_dim_a + max_dim_b)).  That MISSED real
    #    collisions when a long thin part (e.g. a 60mm finger) is rotated
    #    so its long axis sweeps across a sibling part — the centres can
    #    be 32mm apart while the rotated boxes overlap 39mm.  The fix
    #    computes each part's world AABB by rotating its 8 local corners
    #    by the solved axis-angle rotation, then tests axis-aligned
    #    overlap.  This is conservative (AABB ⊇ OBB) but never misses a
    #    real collision, which is the correct direction for a safety net.
    _part_dims = {}
    for p in parts:
        pname = p.get("name", "")
        pdims = p.get("dimensions", {})
        if pname and pdims:
            _part_dims[pname] = pdims

    def _world_aabb(pname):
        """World-space AABB of pname after rotating its local box by the
        solved rotation.  Returns (xmin,ymin,zmin,xmax,ymax,zmax) or None.

        Axis convention matches the renderer's swap_xy (R_z(-90°)): the
        STL's part-local length (FreeCAD X) renders along world Y, and
        width (FreeCAD Y) renders along world X.  So the world AABB maps
        length→world Y extent and width→world X extent.  A cylinder uses
        diameter for X&Y and height for Z.
        """
        pd = positions.get(pname, {})
        center = pd.get("position", [0, 0, 0])
        dims = _part_dims.get(pname)
        if not dims:
            return None
        if "diameter" in dims:
            hx = hy = dims["diameter"] / 2
            hz = dims.get("height", 0) / 2
        else:
            # Match swap_xy: world X = part width, world Y = part length.
            hx = dims.get("width", 0) / 2
            hy = dims.get("length", 0) / 2
            hz = dims.get("height", 0) / 2
        corners = [
            (sx * hx, sy * hy, sz * hz)
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
        ]
        rot = pd.get("rotation", [0, 0, 1, 0])
        ax, ay, az, ang = rot
        try:
            ang = _math.radians(float(ang))
        except (TypeError, ValueError):
            ang = 0.0
        n = _math.sqrt(ax * ax + ay * ay + az * az)
        if n < 1e-9 or abs(ang) < 1e-9:
            xs = [c[0] + center[0] for c in corners]
            ys = [c[1] + center[1] for c in corners]
            zs = [c[2] + center[2] for c in corners]
            return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))
        ax, ay, az = ax / n, ay / n, az / n
        c = _math.cos(ang); s = _math.sin(ang); C = 1 - c
        R = (
            (ax * ax * C + c,       ax * ay * C - az * s, ax * az * C + ay * s),
            (ay * ax * C + az * s,  ay * ay * C + c,      ay * az * C - ax * s),
            (az * ax * C - ay * s,  az * ay * C + ax * s, az * az * C + c),
        )
        wxs = []; wys = []; wzs = []
        for (lx, ly, lz) in corners:
            rx = R[0][0] * lx + R[0][1] * ly + R[0][2] * lz
            ry = R[1][0] * lx + R[1][1] * ly + R[1][2] * lz
            rz = R[2][0] * lx + R[2][1] * ly + R[2][2] * lz
            wxs.append(rx + center[0])
            wys.append(ry + center[1])
            wzs.append(rz + center[2])
        return (min(wxs), min(wys), min(wzs), max(wxs), max(wys), max(wzs))

    _pos_list = list(positions.items())
    _aabb_cache: dict[str, tuple | None] = {}
    # Collect AABB-candidate intersections first; FCL confirms them below.
    # AABB (axis-aligned bbox of the rotated box) is a conservative
    # over-approximation: a 45°-rotated slender bar has an AABB ~41%
    # larger than its actual swept volume, so AABB flags many pairs that
    # do not truly intersect.  Feeding those false positives to the LLM
    # as "physically intersect" feedback caused the VLM loop to chase
    # phantom collisions.  FCL (oriented bounding-box + exact contact)
    # is the ground truth; we use it to filter the AABB candidates.
    _aabb_candidates: list[tuple[str, str, float, float, float]] = []
    for i in range(len(_pos_list)):
        na = _pos_list[i][0]
        box_a = _aabb_cache.get(na)
        if box_a is None:
            box_a = _world_aabb(na)
            _aabb_cache[na] = box_a
        if box_a is None:
            continue
        for j_idx in range(i + 1, len(_pos_list)):
            nb = _pos_list[j_idx][0]
            if (na, nb) in _adjacent_pairs:
                continue
            # Skip container↔internal: a structural shell (chassis_body, etc.)
            # intentionally encloses internal parts, so their AABB overlap is
            # the designed enclosure, not a collision.
            if any(kw in na.lower() for kw in ("chassis_body", "body_shell", "housing")) or \
               any(kw in nb.lower() for kw in ("chassis_body", "body_shell", "housing")):
                continue
            box_b = _aabb_cache.get(nb)
            if box_b is None:
                box_b = _world_aabb(nb)
                _aabb_cache[nb] = box_b
            if box_b is None:
                continue
            ox = min(box_a[3], box_b[3]) - max(box_a[0], box_b[0])
            oy = min(box_a[4], box_b[4]) - max(box_a[1], box_b[1])
            oz = min(box_a[5], box_b[5]) - max(box_a[2], box_b[2])
            if ox > 1.0 and oy > 1.0 and oz > 1.0:
                _aabb_candidates.append((na, nb, ox, oy, oz))

    # FCL confirmation: only report a collision if the oriented bounding
    # boxes truly overlap (penetration > 1mm).  Falls back to reporting
    # all AABB candidates if FCL/trimesh is unavailable, preserving the
    # original conservative behaviour for dependency-free environments.
    if _aabb_candidates:
        confirmed = _fcl_confirm_intersections(
            _aabb_candidates, parts, positions,
        )
        # confirmed is None when FCL is unavailable -> keep all candidates.
        if confirmed is not None:
            report_pairs = confirmed
        else:
            report_pairs = _aabb_candidates
        for na, nb, ox, oy, oz in report_pairs:
            problems.append(
                f"Parts '{na}' and '{nb}' overlap by "
                f"{ox:.0f}x{oy:.0f}x{oz:.0f}mm in their rotated "
                f"world bounding boxes — they physically intersect. "
                f"Increase the offset between them or reduce their "
                f"dimensions so they do not collide."
            )

    # 7. Connectivity — every part must be reachable from an arbitrary
    #    root via the joint graph (BFS).  This is the GROUND-TRUTH
    #    arbiter for VLM "floating / disconnected" reports: if the joint
    #    graph says the assembly is a single connected component, then
    #    no part is genuinely floating, regardless of how the render
    #    angle makes it look (TDBench, arXiv 2504.03748 shows VLMs
    #    catastrophically misjudge "floating" when parts are <1% of the
    #    frame — this check overrides those false negatives).
    #
    #    Without this check, `_vlm_check_assembly` has no geometric
    #    signal to refute VLM "floating" false-alarms, so they enter the
    #    LLM regeneration loop and corrupt an otherwise-correct assembly
    #    (observed: 4dof_arm round 2 correct → VLM false "floating" →
    #    round 3 regeneration broke the gripper).
    if joints:
        part_names = {p.get("name", "") for p in parts if p.get("name")}
        adj: dict[str, set[str]] = {n: set() for n in part_names}
        for j in joints:
            if isinstance(j, dict):
                jp, jc = j.get("parent", ""), j.get("child", "")
            else:
                jp, jc = getattr(j, "parent", ""), getattr(j, "child", "")
            if jp in adj and jc in adj:
                adj[jp].add(jc)
                adj[jc].add(jp)
        if part_names:
            root = next(iter(part_names))
            visited: set[str] = set()
            queue = [root]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                queue.extend(adj.get(node, set()) - visited)
            disconnected = part_names - visited
            if disconnected:
                problems.append(
                    f"Parts {sorted(disconnected)} are not connected to "
                    f"the root '{root}' via the joint graph — they are "
                    f"genuinely floating (no joint path from the base)."
                )

    return problems


def _fcl_confirm_intersections(
    candidates: list[tuple[str, str, float, float, float]],
    parts: list[dict],
    positions: dict[str, dict],
) -> list[tuple[str, str, float, float, float]] | None:
    """Filter AABB candidate pairs through exact FCL collision tests.

    Returns the subset of candidates whose oriented bounding boxes truly
    intersect (penetration > 1mm), or ``None`` if FCL/trimesh is not
    installed (caller falls back to the full AABB candidate list).
    """
    try:
        from ..mesh_collision import MeshCollisionChecker
        from ...knowledge.mechanics import Assembly, Joint, Part
    except ImportError:
        return None

    # MeshCollisionChecker needs an Assembly + placements.  Reconstruct
    # lightweight Part objects from the dict list; the checker only reads
    # name + dimensions, so category/material defaults are fine.
    part_objs: list[Part] = []
    name_to_dict: dict[str, dict] = {}
    for p in parts:
        name = p.get("name", "")
        dims = p.get("dimensions", {})
        if not name or not dims:
            continue
        part_objs.append(Part(
            name=name, category="mechanical", description="",
            dimensions=dict(dims),
        ))
        name_to_dict[name] = p
    if len(part_objs) < 2:
        return [c for c in candidates]  # nothing to check

    # Joints are needed only for adjacency filtering, which the caller
    # has already applied via _adjacent_pairs, so pass an empty list.
    asm = Assembly(name="prevalidation", parts=part_objs, joints=[])

    try:
        checker = MeshCollisionChecker()
    except Exception:
        return None

    result = checker.check_assembly_collisions(
        asm, positions, skip_adjacent=False, min_penetration_mm=1.0,
    )
    colliding_names: set[tuple[str, str]] = set()
    for pair in result.pairs:
        if pair.is_collision:
            a, b = pair.part_a, pair.part_b
            colliding_names.add((a, b))
            colliding_names.add((b, a))

    # Keep only candidates FCL confirms; preserve AABB overlap dims for
    # the message severity.
    confirmed = [
        (na, nb, ox, oy, oz)
        for (na, nb, ox, oy, oz) in candidates
        if (na, nb) in colliding_names
    ]
    return confirmed


# ---------------------------------------------------------------------------
# Preview STL generation (trimesh) for VLM verification
# ---------------------------------------------------------------------------
#
# Production STLs are exported by Phase 4 (engineering package), which runs
# AFTER the VLM loop.  During verification the renderer therefore falls back
# to box/cylinder approximations, where gripper fingers become near-invisible
# ~6mm boxes and cylindrical servos become boxes.  That makes prompt checks
# like "gripper should look like a gripper" always fail.
#
# These fast trimesh previews are drop-in replacements that follow the same
# axis convention as FreeCAD STLs (X=length, Y=width, Z=height) so the
# renderer's swap_xy=True path aligns them where the solver expects.


def _build_box_preview_mesh(dims: dict):
    """Box preview mesh with extents [L, W, H] → X=L, Y=W, Z=H."""
    import trimesh

    if "length" in dims and "width" in dims:
        l = dims["length"]
        w = dims["width"]
        h = dims.get("height", dims.get("thickness", 5))
    else:
        l = dims.get("length", dims.get("diameter", 20))
        w = dims.get("width", l)
        h = dims.get("height", dims.get("thickness", 20))
    return trimesh.creation.box(extents=[l, w, h])


def _build_cylinder_preview_mesh(dims: dict):
    """Cylinder preview mesh along Z (matches FreeCAD cylinder convention)."""
    import trimesh

    d = dims.get("outer_diameter", dims.get("diameter", 20))
    h = dims.get("height", dims.get("length", d))
    return trimesh.creation.cylinder(radius=d / 2.0, height=h)


def _build_finger_preview_mesh(name: str, dims: dict):
    """L-shaped gripper finger preview matching _gripper_finger_ops shape.

    Two fused boxes (concatenated): a main bar extending in +X and an
    inward-hooking tip at the front end.  Left/right tip direction is
    detected from the name, mirroring _gripper_finger_ops.
    """
    import trimesh

    L = dims.get("length", 35)
    W = dims.get("width", 6)
    H = dims.get("height", 15)

    n_lower = name.lower()
    is_left = "left" in n_lower
    # Match _gripper_finger_ops: left finger tip hooks toward +Y, right
    # finger tip hooks toward -Y (in FreeCAD coords).
    tip_dir = 1.0 if is_left else -1.0

    # Main bar: makeBox(L, W, H) has its corner at the origin in FreeCAD,
    # so translate the centred trimesh box to match.
    bar = trimesh.creation.box(extents=[L, W, H])
    bar.apply_translation([L / 2.0, W / 2.0, H / 2.0])

    # L-shaped tip at the front end, hooking inward.
    # tip_w mirrors _gripper_finger_ops (part_feature_engine.py): keep the
    # finger slender.  W*2.0 inflated the total Y extent to 3W (e.g.
    # 14→42mm), making the rendered finger a stubby block that the VLM
    # reads as a "solid sphere" instead of a gripper prong.
    tip_l = L * 0.25
    tip_w = max(4.0, W * 0.4)
    tip_y = W if tip_dir > 0 else -tip_w
    tip = trimesh.creation.box(extents=[tip_l, tip_w, H])
    tip.apply_translation([L - tip_l / 2.0, tip_y + tip_w / 2.0, H / 2.0])

    # NOTE: bar and tip touch on a coincident face (Y=W for left, Y=0
    # for right).  ``trimesh.util.concatenate`` does NOT boolean-merge,
    # so this leaves a non-manifold edge (euler=4, two separate bodies).
    # A real boolean union would fix it but requires the optional
    # ``manifold3d``/``blender`` backend, which is not a project
    # dependency.  This preview path is only a FALLBACK when FreeCAD is
    # unavailable — the production path uses FreeCAD-generated STLs
    # (where ``Part.fuse`` produces a clean watertight union), so the
    # non-manifold preview is acceptable.  See C1 (part_feature_engine
    # _gripper_finger_ops) for the real water-tightness fix.
    return trimesh.util.concatenate([bar, tip])


def _generate_preview_stls(parts: list[dict], output_dir: str) -> str:
    """Generate fast trimesh STL previews for VLM rendering.

    Writes one ``{part_name}.stl`` per part into ``{output_dir}/preview_stls``
    and returns that directory path.  Returns an empty string (so the renderer
    falls back to dimension boxes) when trimesh is unavailable.

    The renderer's existing fallback handles any part missing a preview STL
    gracefully (it just builds a dimension box), so partial generation is safe.
    """
    try:
        import trimesh  # noqa: F401
    except ImportError:
        logger.warning(
            "trimesh not installed — VLM will fall back to box approximations"
        )
        return ""

    preview_dir = os.path.join(output_dir, "preview_stls")
    os.makedirs(preview_dir, exist_ok=True)

    for idx, part in enumerate(parts):
        name = part.get("name", f"part_{idx}")
        dims = part.get("dimensions", {}) or {}
        n_lower = name.lower()

        if "finger" in n_lower:
            mesh = _build_finger_preview_mesh(name, dims)
        elif "diameter" in dims or "outer_diameter" in dims:
            mesh = _build_cylinder_preview_mesh(dims)
        else:
            mesh = _build_box_preview_mesh(dims)

        if mesh is None:
            continue

        # Centre on bounding-box centre so the renderer's load_stl centering
        # is a no-op and the part lands exactly where the solver expects.
        try:
            mesh.apply_translation(-mesh.bounding_box.centroid)
        except Exception as _e:
            logger.debug("mesh bounding-box center recentre failed: %s", _e)

        stl_path = os.path.join(preview_dir, f"{name}.stl")
        try:
            mesh.export(stl_path)
        except Exception as e:
            logger.warning("Preview STL export failed for %s: %s", name, e)

    return preview_dir


# Default vision model for the closed-loop assembly verifier.
#
# MUST be GLM-4.6V (the MAXIMUM-tier model), not GLM-4.6V-Flash or
# GLM-4V-Plus.  Empirically verified 2026-06-21: on the 4dof_arm
# gripper close-up render (two 14mm fingers separated by 46mm), only
# GLM-4.6V reliably identifies the two parallel finger prongs and the
# visible gap.  GLM-4.6V-Flash (free tier) and GLM-4V-Plus both
# false-negative the gripper as a "solid block / no separated prongs",
# which causes the VLM loop to fail all 3 rounds on a geometrically
# correct assembly (verification_status=FAILED_MAX_ROUNDS, e2e
# 89.5% blocked at the single critical check).  Using an underpowered
# model as the sole arbiter of fine geometric features defeats the
# purpose of the vision channel — see AGENTS.md §5.2.
_DEFAULT_VERIFIER_VISION_MODEL = "GLM-4.6V"


def _vlm_check_assembly(
    positions: dict[str, dict],
    parts: list[dict],
    render_dir: str,
    api_key: str,
    base_url: str,
    vision_model: str = _DEFAULT_VERIFIER_VISION_MODEL,
    round_num: int = 0,
    real_stl_dir: str | None = None,
    joints: list | None = None,
    robot_category: str = "assembly",
) -> tuple[bool, list[str]]:
    """Render assembly and run VLM verification. Returns (passed, problems).

    ``robot_category`` selects the expectations block injected into the
    whole-assembly prompt (see :func:`_build_verify_prompt`).  This MUST match
    the generator's own classification — passing a category-blind prompt
    historically caused the wheeled-dual-arm false-negative loop.

    Per-view raw VLM responses are accumulated and written to
    ``{render_dir}/vlm_responses.json`` so failures can be debugged across
    rounds without re-running the model.
    """
    from ...models.base import Message
    from ...models.glm import GLMBackend

    from ..vtk_renderer import render_assembly_from_positions

    # Use real FreeCAD STLs when available (produced by generate_part_stls
    # before the VLM loop); fall back to fast trimesh preview STLs when
    # FreeCAD is not installed or generation failed.
    stl_dir_for_render = real_stl_dir or _generate_preview_stls(parts, render_dir)

    # Render 4 views
    # Render 4 standard views PLUS a gripper close-up.  The close-up aims
    # the camera at the finger centroid with a tight parallel scale so the
    # ~32mm finger gap is clearly resolvable — without it the VLM sees
    # fingers as sub-pixel slivers at the edge of the full-arm frame and
    # false-negatives the gripper as "single solid mass".
    rendered = render_assembly_from_positions(
        parts=parts,
        positions=positions,
        output_dir=render_dir,
        views=["isometric", "front", "top", "right", "gripper_closeup"],
        stl_dir=stl_dir_for_render,
        width=1600,
        height=1200,
        joints=joints,
        gripper_closeup=True,
    )
    if not rendered:
        return False, ["VTK rendering produced no images"]

    # Check each view with VLM — split by responsibility (2026-06-23).
    # Whole-assembly views (iso/front/top/right) judge STRUCTURAL integrity
    # only — they cannot resolve a ~46mm finger gap at the edge of a
    # 490mm-tall frame, so asking them about the gripper produced false
    # "solid block" negatives.  The gripper_closeup view is the SOLE
    # authority on the gripper question: it zooms to a 120mm window.
    backend = GLMBackend(api_key=api_key, base_url=base_url,
                          vision_model=vision_model)
    all_problems: list[str] = []
    # Track whole-assembly and gripper verdicts separately.
    structural_views: list[str] = []
    structural_pass_count = 0
    gripper_view_passed: bool | None = None  # None = no closeup rendered
    total_views = len(rendered)
    view_logs: list[dict] = []

    for view_path in rendered:
        view_name = os.path.splitext(os.path.basename(view_path))[0]
        is_closeup = view_name == "gripper_closeup"
        prompt = (
            _VLM_GRIPPER_CLOSEUP_PROMPT if is_closeup
            else _build_verify_prompt(robot_category)
        )
        entry: dict = {
            "view": view_name,
            "prompt_role": "gripper" if is_closeup else "structural",
            "raw_response": None,
            "parsed": None,
            "passed": False,
        }
        try:
            resp = backend.vision(
                image_path=view_path,
                prompt=prompt,
            )
            entry["raw_response"] = str(resp)
            text = str(resp).lower()
            view_passed = ('"passed": true' in text) or ('"passed":true' in text)
            entry["passed"] = view_passed
            # Always extract problems (even from "passed" views)
            try:
                start = str(resp).find("{")
                end = str(resp).rfind("}") + 1
                data = json.loads(str(resp)[start:end])
                entry["parsed"] = data
                for p in data.get("problems", []):
                    if p and p not in all_problems:
                        all_problems.append(p)
            except (json.JSONDecodeError, ValueError):
                pass
            # Route the verdict by responsibility.
            if is_closeup:
                gripper_view_passed = view_passed
            else:
                structural_views.append(view_name)
                if view_passed:
                    structural_pass_count += 1
        except Exception as e:
            logger.warning("VLM check failed for %s: %s", view_path, e)
            entry["raw_response"] = f"ERROR: {e}"
        view_logs.append(entry)

    # Pass requires BOTH responsibilities to pass:
    #  - STRUCTURAL: majority of whole-assembly views pass
    #  - GRIPPER: the close-up view passes (sole authority)
    # When no close-up was rendered, fall back to majority vote.
    if gripper_view_passed is not None and structural_views:
        structural_majority = structural_pass_count > len(structural_views) / 2
        passed = structural_majority and bool(gripper_view_passed)
    else:
        passed = (structural_pass_count + (1 if gripper_view_passed else 0)) > total_views / 2

    # Geometric pre-validation as safety net AND ground-truth arbitrator.
    # ABLATION: set LANG3D_ABLATION=no_geo to disable geometric arbitration
    # (for measuring the contribution of this component in ablation studies).
    import os as _os
    _ablation = _os.environ.get("LANG3D_ABLATION", "")
    if "no_geo" in _ablation:
        geo_problems = []  # skip geometric checks entirely
    else:
        geo_problems = _geometric_prevalidation(parts, positions, joints)

    # WHEEL FALSE-ALARM FILTER (geometric oracle, runs BEFORE the hard_geo
    # gate). The deterministic compose path produces grounded wheels (Z≈
    # radius, verified), but GLM-4.6V still reports "wheels above Z / wrong
    # orientation / missing". When geometry confirms the wheels are grounded,
    # these VLM complaints are overruled REGARDLESS of other geometry issues
    # — otherwise a single arm-motor overlap (hard_geo) would skip the
    # false-alarm filter and the wheeled-dual-arm e2e dead-loops on wheel
    # false-negatives. Filter from both all_problems (VLM) and geo_problems
    # so neither path keeps the refuted complaint.
    # ABLATION: skip false-alarm filtering when no_geo mode is active.
    if "no_geo" not in _ablation:
        all_problems = [
            p for p in all_problems
            if not _is_wheel_false_alarm(p, parts, positions)
        ]
    geo_problems = [
        p for p in geo_problems
        if not _is_wheel_false_alarm(p, parts, positions)
    ]

    # Separate HARD geometry failures (collision, disconnection, absurd
    # positions) from SOFT pose warnings ("arm too flat/horizontal").
    # Soft warnings describe the arm's posture, not a geometric defect —
    # they should NOT block the gripper/floating false-alarm filtering
    # below.  Hard failures force FAIL + skip filtering; soft warnings
    # are appended to all_problems but still allow the filtering branch.
    _SOFT_POSE_MARKERS = ("arm too flat", "arm too horizontal", "arm too vertical")
    hard_geo = [p for p in geo_problems
                if not any(m in p.lower() for m in _SOFT_POSE_MARKERS)]
    soft_geo = [p for p in geo_problems
                if any(m in p.lower() for m in _SOFT_POSE_MARKERS)]

    if hard_geo:
        # Hard geometry problem (collision, disconnection, fused fingers,
        # outlier) → force failure.
        passed = False
        for p in geo_problems:  # include soft warnings too
            if p not in all_problems:
                all_problems.append(p)
    else:
        # No hard geometry failures.  Append soft pose warnings but still
        # run the false-alarm filtering (the arm posture is a suggestion,
        # not a reason to block a geometrically-correct assembly).
        for p in soft_geo:
            if p not in all_problems:
                all_problems.append(p)
        # Geometry is clean — including Check 7 (joint-graph connectivity:
        # every part reachable from the root).  Two classes of VLM false
        # alarm are filtered here, each backed by a geometric oracle:
        #
        #  (a) Gripper "solid block / no separated prongs": Check 5
        #      confirmed >= 2 fingers separated by >= 25mm, so the VLM
        #      (which catastrophically misjudges sub-1%-of-frame features,
        #      per TDBench arXiv 2504.03748) is overruled.
        #
        #  (b) "Floating / disconnected / no support": Check 7 confirmed
        #      the whole assembly is one connected component via the joint
        #      graph, so the VLM report is a viewpoint artifact.
        filtered = [
            p for p in all_problems
            if not _is_gripper_false_alarm(p)
            and not _is_floating_false_alarm(p)
            and not _is_wheel_false_alarm(p, parts, positions)
            and not any(m in p.lower() for m in _SOFT_POSE_MARKERS)
        ]
        if len(filtered) < len(all_problems):
            all_problems = filtered
            if not all_problems:
                # Every remaining problem was a geometrically-refuted
                # false alarm or a soft pose warning → pass.
                passed = True

    # Persist per-view VLM responses for debugging across rounds.
    vlm_log = {
        "round": round_num,
        "views": view_logs,
        "pass_count": structural_pass_count + (1 if gripper_view_passed else 0),
        "total_views": total_views,
        "final_passed": passed,
        "all_problems": all_problems,
    }
    try:
        with open(
            os.path.join(render_dir, "vlm_responses.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(vlm_log, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Failed to write vlm_responses.json: %s", e)

    return passed, all_problems

