"""Assembly sanitizers — post-generation correction functions.

Extracted from assembly_generator.py (P1-1 God Module split, AGENTS.md §2.1).
These functions normalize, validate, and correct LLM-generated assemblies:
- Arm chain anchor fixing
- Wheel position normalization
- Gripper finger symmetry
- Proportion validation
- Default angle injection + workspace-safe joint limits
- Assembly connectivity enforcement

All functions operate on Assembly/Part/Joint objects from knowledge.mechanics.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

from ...knowledge.mechanics import Assembly, ConnectionMethod, Joint, Part
from ...knowledge.fastener_catalog import get_torque
from ...knowledge.arm_topology import build_arm_example, parse_dof, zigzag_angles
from ...knowledge.mobile_base_gen import build_wheeled_base, parse_drive_type
from ..assembly_solver import ANCHOR_DIM_KEYS

logger = logging.getLogger(__name__)

def apply_default_connection_methods(joints: list, parts: list | None = None) -> None:
    """Assign a default ``ConnectionMethod`` to joints that lack one.

    Dispatches by part category (when ``parts`` is provided) on top of the
    existing anchor-geometry rule:

    - **Bearing** parent or child (any joint type) → ``press_fit`` H7/js6.
      Bearings are never bolted; they press into housings.
    - **Sensor** child (category=sensor, fixed joint) → ``adhesive`` (epoxy
      bond). Cameras/IMUs/LiDARs bond to brackets, not bolted through PCBs.
    - **Foot** child (name contains "foot", fixed joint) → ``adhesive``
      (epoxy bond). Rubber/TPU foot pads bond to plates (cf. ANYmal B).
    - **Servo** child (name contains "servo", e.g. SG90) → ``bolted M2×2``
      with ``hole_type="threaded_hole"`` — SG90 servos have tapped holes
      for self-tapping M2 screws, not through holes.
    - **Actuator** child (motors like NEMA17) → ``bolted M3×4`` with
      ``hole_type="threaded_hole"`` — motor flanges have tapped holes.
    - Other fixed/revolute joints with face anchors → ``bolted M3×4``
      ``through_hole`` (structural bracket mounting).
    - Revolute with center/center anchors → ``press_fit`` (bearing seat).
    - Prismatic → null (sliding fit, not a fastening).

    Mutates *joints* in place.  ``parts`` is optional for backward
    compatibility; without it the function falls back to the original
    geometry-only dispatch.
    """
    _face_anchors = {"front", "back", "top", "bottom", "left", "right"}
    _parts_by_name = {p.name: p for p in parts} if parts else {}

    def _category(name: str) -> str:
        p = _parts_by_name.get(name)
        return (p.category or "").lower() if p else ""

    def _is_servo(name: str) -> bool:
        return "servo" in name.lower()

    def _bolted(size: str, count: int, hole_type: str) -> ConnectionMethod:
        return ConnectionMethod(
            type="bolted", bolt_size=size, bolt_count=count,
            hole_type=hole_type, torque_nm=get_torque(size, "PLA"),
        )

    for joint in joints:
        if joint.connection is not None:
            continue

        child_cat = _category(joint.child)
        parent_cat = _category(joint.parent)

        # Bearing → always press_fit regardless of joint type
        if child_cat == "bearing" or parent_cat == "bearing":
            if joint.type in ("fixed", "revolute"):
                joint.connection = ConnectionMethod(
                    type="press_fit", interference_mm=0.02,
                )
                logger.debug(
                    "Defaulted joint %s->%s to press_fit (bearing seat)",
                    joint.parent, joint.child,
                )
                continue

        # Adhesive bonds (not bolted): sensors and foot pads.
        # These connection methods have working geometry generators
        # (connection_features.py) but were never triggered by the default
        # rule set, leaving snap_fit/adhesive as dead dispatch entries.
        if joint.type == "fixed":
            if child_cat == "sensor":
                # Cameras/IMUs/LiDARs bond to brackets with epoxy or
                # double-sided tape in real robotics — bolting through a
                # PCB or sensor housing is not standard practice.
                joint.connection = ConnectionMethod(
                    type="adhesive", adhesive_type="epoxy",
                    bond_area_mm2=0.0,  # computed by geometry generator
                )
                logger.debug(
                    "Defaulted sensor joint %s->%s to adhesive (epoxy bond)",
                    joint.parent, joint.child,
                )
                continue
            if "foot" in joint.child.lower():
                # Rubber/TPU foot pads bond to metal/carbon plates
                # (e.g. ANYmal B foot-shell bonding — assembly_patterns.py
                # RobotAssemblyProfile declares "adhesive": 4 for feet).
                joint.connection = ConnectionMethod(
                    type="adhesive", adhesive_type="epoxy",
                    bond_area_mm2=0.0,
                )
                logger.debug(
                    "Defaulted foot-pad joint %s->%s to adhesive (epoxy bond)",
                    joint.parent, joint.child,
                )
                continue

        if joint.type == "fixed":
            if _is_servo(joint.child):
                # SG90-style servo: M2 into tapped holes
                joint.connection = _bolted("M2", 2, "threaded_hole")
            elif child_cat == "actuator":
                # Larger motors (NEMA17 etc.): M3 into tapped flange
                joint.connection = _bolted("M3", 4, "threaded_hole")
            else:
                # Structural fixed joint: through hole + nut
                joint.connection = _bolted("M3", 4, "through_hole")
            logger.debug(
                "Defaulted fixed joint %s->%s to %s %s (%s)",
                joint.parent, joint.child,
                joint.connection.bolt_size,
                joint.connection.hole_type,
                joint.connection.type,
            )
        elif joint.type == "revolute":
            uses_face_anchor = (
                joint.parent_anchor in _face_anchors
                or joint.child_anchor in _face_anchors
            )
            if uses_face_anchor:
                if _is_servo(joint.child):
                    joint.connection = _bolted("M2", 2, "threaded_hole")
                elif child_cat == "actuator":
                    joint.connection = _bolted("M3", 4, "threaded_hole")
                else:
                    joint.connection = _bolted("M3", 4, "through_hole")
                logger.debug(
                    "Defaulted revolute joint %s->%s to bolted %s %s "
                    "(face anchor %s/%s)",
                    joint.parent, joint.child,
                    joint.connection.bolt_size,
                    joint.connection.hole_type,
                    joint.parent_anchor, joint.child_anchor,
                )
            else:
                # Center/center: bearing press-fit into a housing bore.
                joint.connection = ConnectionMethod(
                    type="press_fit", interference_mm=0.01,
                )
                logger.debug(
                    "Defaulted revolute joint %s->%s to press_fit "
                    "(bearing seat, center anchors)",
                    joint.parent, joint.child,
                )
        elif joint.type == "prismatic":
            # Sliding interface is not a fastening method; null is intentional.
            logger.info(
                "Prismatic joint %s->%s has no connection_method "
                "(sliding fit, expected)",
                joint.parent, joint.child,
            )

    # Safety: clear connections on ALL prismatic joints.  Sliding
    # interfaces must never have bolted/press-fit fasteners — a bolt
    # through a rail would prevent sliding.  The LLM sometimes marks
    # non-gripper prismatic joints as "bolted"; this ensures they are
    # always null.
    for joint in joints:
        if joint.type == "prismatic" and joint.connection is not None:
            logger.info(
                "Safety: cleared %s connection on prismatic joint "
                "%s->%s (sliding fit)",
                joint.connection.type, joint.parent, joint.child,
            )
            joint.connection = None


def _is_link_like(name: str) -> bool:
    """Check if a part name looks like an arm link (extends horizontally)."""
    n = name.lower()
    return any(p in n for p in _LINK_PATTERNS)


def _is_end_effector(name: str) -> bool:
    """Check if a part name is an end effector."""
    n = name.lower()
    return "end_effector" in n or "gripper" in n or "effector" in n


def _is_joint_like(name: str) -> bool:
    """Check if a part name looks like an arm joint/housing/servo (rotary node).

    Used by ``_fix_arm_chain_anchors`` to decide whether a top/bottom joint is
    part of the arm kinematic chain (should be normalised to front/back) versus
    a motor mount inside a housing (should stay back/front).
    """
    n = name.lower()
    return any(p in n for p in _JOINT_PATTERNS)


# Pattern constants (defined here so the helper functions above resolve).
# Were at module level in the original assembly_generator.py.
_LINK_PATTERNS = ("link", "arm", "forearm", "upper_arm", "bracket")
_JOINT_PATTERNS = ("joint", "support", "housing", "servo", "motor")


def _fix_arm_chain_anchors(joints: list[Joint], parts: list[Part]) -> None:
    """Normalise arm-chain joints to the clean horizontal (front/back) convention.

    The target arm geometry extends horizontally so each link's ``length``
    dimension positions the next pitch axis: pitch joints use ``front/back``
    anchors with ``axis="x"`` (see the 4dof_arm template in
    assembly_templates.py). The base yaw is the only ``top/bottom`` joint.

    Two LLM patterns are fixed:

    1. **top/bottom + axis=y → front/back + axis=x**: Legacy prompt rules
       told the LLM to stack links vertically via top/bottom anchors; that
       built the arm as a tower of thin plates whose ``length`` dimension
       never participated in positioning, collapsing IK link lengths and
       producing a vertical column that does not move like a real arm.
       Converted to the clean pitch convention. base yaw (axis=z) and wrist
       roll are left untouched.

    2. **top/top → top/bottom**: When the LLM uses ``child_anchor="top"`` the
       solver places the child's top face at the parent's top face, so the
       child extends DOWNWARD — the arm folds back on itself like an
       accordion (workspace collapses to ~47mm instead of ~200mm).

    Conservative filters — the following joints are LEFT UNTOUCHED:

    * prismatic joints (gripper fingers are handled by
      ``_normalize_gripper_fingers``).
    * joints already using top/bottom.
    * center/center joints (bearing press-fits).
    * bottom/top joints (motor mounted under the base).
    * fixed joints with parent_anchor="back" / child_anchor="front" (motor
      mounted behind a housing face — e.g. NEMA17 on the BCN3D MOVEO).
    * joints where neither parent nor child is a joint/link/effector-like part
      (avoids converting unrelated structural brackets).
    """
    if len(joints) < 3:
        return

    for i, joint in enumerate(joints):
        # Prismatic fingers are normalized separately.
        if joint.type == "prismatic":
            continue

        parent_name = joint.parent.lower()
        child_name = joint.child.lower()

        # At least one endpoint must be an arm-chain part (joint/housing/servo/
        # motor/link/effector). This avoids rewriting unrelated brackets.
        if not (_is_joint_like(parent_name) or _is_joint_like(child_name)
                or _is_link_like(parent_name) or _is_link_like(child_name)
                or _is_end_effector(parent_name) or _is_end_effector(child_name)):
            continue

        # --- Pattern 2: top/top → top/bottom (fix fold-back) ---
        # child_anchor="top" makes the solver place the child hanging
        # downward from the parent's top, collapsing the arm. Fix first so
        # the subsequent pitch-normalisation can still apply. No `continue`
        # — fall through to Pattern 1.
        if joint.parent_anchor == "top" and joint.child_anchor == "top":
            logger.info(
                "Fixing arm-chain joint %d ('%s'→'%s'): top/top → top/bottom"
                " (child was folding back)",
                i, joint.parent, joint.child,
            )
            joint.child_anchor = "bottom"

        # --- Pattern 1: legacy top/bottom arm-chain joints → clean front/back ---
        # The clean arm convention uses front/back so each link's `length`
        # dimension positions the next pitch axis and the arm extends
        # horizontally as a real, movable arm. Convert any legacy top/bottom
        # arm-chain joint (pitch revolute with axis=y, or fixed link
        # connectors with axis=auto). base yaw (axis=z) is left untouched.
        # The gripper-servo mount (servo atop gripper_base) is preserved.
        is_gripper_servo_mount = (
            "servo" in child_name and "grip" in parent_name
        )
        if (joint.parent_anchor == "top" and joint.child_anchor == "bottom"
                and joint.axis in ("y", "auto")
                and not is_gripper_servo_mount):
            old_axis = joint.axis
            joint.parent_anchor = "front"
            joint.child_anchor = "back"
            if old_axis == "y":
                joint.axis = "x"
            logger.info(
                "Normalising arm-chain joint %d ('%s'→'%s'): top/bottom → "
                "front/back (clean convention, axis %s→%s)",
                i, joint.parent, joint.child, old_axis, joint.axis,
            )
            continue


def _find_best_parent(part_name: str, part_names: set[str], visited: set[str]) -> str | None:
    """Find the best parent for an orphaned part using prefix-based heuristics.

    Maps common component prefixes to their natural parent patterns:
    sensor_* → sensor_tower, camera_* → sensor_tower, lidar_* → sensor_tower,
    imu_* → sensor_tower, battery_* → base_plate, pcb_* → top_plate,
    controller_* → top_plate, power_* → base_plate, servo_* → matching arm link,
    arm_* → matching base, gripper_* → matching wrist/link.
    """
    n = part_name.lower()

    # Prefix-to-parent mapping with candidate patterns
    prefix_map: dict[str, list[str]] = {
        "sensor_": ["sensor_tower", "sensor_mount", "top_plate", "base_plate"],
        "camera_": ["sensor_tower", "sensor_mount", "top_plate", "base_plate"],
        "lidar_": ["sensor_tower", "sensor_mount", "top_plate", "base_plate"],
        "imu_": ["sensor_tower", "sensor_mount", "top_plate", "base_plate"],
        "battery_": ["base_plate", "bottom_plate", "chassis", "top_plate"],
        "pcb_": ["top_plate", "base_plate", "main_board"],
        "controller_": ["top_plate", "base_plate", "main_board"],
        "power_": ["base_plate", "bottom_plate", "battery_box"],
        "servo_": ["base_plate", "top_plate"],
        "arm_": ["base_plate", "top_plate"],
        "gripper_": ["wrist_link", "wrist", "end_effector"],
    }

    for prefix, candidates in prefix_map.items():
        if n.startswith(prefix):
            for candidate in candidates:
                if candidate in visited:
                    return candidate
            return None

    return None


def _ensure_connected(assembly: Assembly, part_names: set[str]) -> None:
    """Auto-fix: connect orphaned parts to the nearest reachable parent.

    When the LLM generates fewer joints than needed, some parts have no
    parent in the joint tree.  This function detects them and adds fixed
    joints to the nearest structural part that IS reachable from the root.
    """
    # Build parent→children map from existing joints
    children_map: dict[str, set[str]] = {}
    child_to_parent: dict[str, str] = {}
    for j in assembly.joints:
        children_map.setdefault(j.parent, set()).add(j.child)
        child_to_parent[j.child] = j.parent

    # BFS from the true kinematic root.
    # The root is the part that is NEVER a joint child (nothing parents onto
    # it) — e.g. base_footprint in a Husky chassis, which is the parent of
    # base_plate but never a child. Hard-coding "base_plate" as the BFS root
    # was wrong: it left base_footprint unvisited, so this auto-fixer added a
    # spurious base_plate→base_footprint joint, creating a cycle (no root in
    # the URDF → MuJoCo "URDF body not found" parse failure). If multiple
    # parts have no parent, prefer base_footprint/base_plate for stability.
    children_of_joints = {j.child for j in assembly.joints}
    candidate_roots = part_names - children_of_joints
    if not candidate_roots:
        # Cycle (every part is a child) — fall back to base_plate/first part.
        root = "base_plate" if "base_plate" in part_names else assembly.parts[0].name
    elif "base_footprint" in candidate_roots:
        root = "base_footprint"
    elif "base_plate" in candidate_roots:
        root = "base_plate"
    else:
        root = next(iter(candidate_roots))
    visited = {root}
    queue = [root]
    while queue:
        current = queue.pop(0)
        for child in children_map.get(current, set()):
            if child not in visited:
                visited.add(child)
                queue.append(child)

    unconnected = part_names - visited
    if not unconnected:
        return

    logger.info("Auto-fixing %d unconnected parts: %s", len(unconnected), unconnected)

    # For each unconnected part, find the best parent and add a fixed joint
    for part_name in unconnected:
        # Find a suitable parent: prefer base_plate, then first reachable structural part
        parent = root
        # Check if the part name hints at its natural parent
        n = part_name.lower()
        if n.startswith("motor_"):
            # Motors go under base_plate bottom
            assembly.joints.append(Joint(
                type="fixed", parent="base_plate", child=part_name,
                parent_anchor="bottom", child_anchor="top",
                distribution_group="motors",
            ))
            continue
        elif n.startswith("wheel_"):
            # Wheels go on motors: wheel_fl → motor_fl
            suffix = n.replace("wheel_", "")
            motor_name = f"motor_{suffix}"
            if motor_name in part_names:
                assembly.joints.append(Joint(
                    type="revolute", parent=motor_name, child=part_name,
                    axis="y", range_deg=(-360, 360),
                    parent_anchor="left", child_anchor="center",
                    no_distribute=True,
                ))
                continue
        elif n.startswith("encoder_"):
            suffix = n.replace("encoder_", "")
            motor_name = f"motor_{suffix}"
            if motor_name in part_names:
                parent = motor_name
        else:
            best = _find_best_parent(part_name, part_names, visited)
            if best is not None:
                parent = best

        # Default: attach to root with a fixed joint
        assembly.joints.append(Joint(
            type="fixed", parent=parent, child=part_name,
            parent_anchor="top", child_anchor="bottom",
        ))
        logger.info("  Auto-connected '%s' -> '%s'", part_name, parent)


def _raise_on_wheel_in_arm(assembly: Assembly) -> None:
    """P1-1: detect hallucinated wheel parts in arm assemblies and raise.

    Per CLAUDE.md: "LLM 给出离谱尺寸/位置时，应该报错让 LLM 重试，
    而不是悄悄修正".  Previously ``_strip_wheel_parts`` silently deleted
    wheel/motor_mount parts, hiding the error from the VLM feedback loop
    so the LLM never learned to stop generating them.

    This function raises ``RuntimeError`` so the error enters
    ``problems_history`` via ``_validate_assembly``'s pattern, giving the
    LLM a chance to regenerate without wheels.
    """
    wheel_keywords = ("wheel", "motor_mount", "电机座", "轮")
    found = [
        p.name for p in assembly.parts
        if any(kw in p.name.lower() for kw in wheel_keywords)
    ]
    if found:
        raise RuntimeError(
            f"Arm assembly contains wheel/motor_mount parts that should "
            f"not exist in a fixed-base arm: {found}. Remove these parts "
            f"and their joints — a fixed-base arm has only base_plate, "
            f"joints (housings), links, and end_effector."
        )


def _strip_wheel_parts(assembly: Assembly) -> Assembly:
    """Remove wheel and wheel-motor parts from the assembly in-place.

    .. deprecated:: P1-1
       Silent deletion of LLM-hallucinated parts violates the CLAUDE.md
       principle "不要在代码里加 hack 让 LLM/外部输入看起来对".  Use
       :func:`_raise_on_wheel_in_arm` instead — it feeds the error back
       into the VLM retry loop so the LLM can correct itself.

    The LLM sometimes hallucinates wheel/motor_mount parts for fixed-base arms,
    which causes VLM verification failures (overlapping parts, wrong wheel
    orientation feedback that confuses the regeneration loop).

    This sanitizer strips any part whose name matches wheel keywords and
    removes their associated joints. Returns the (mutated) assembly.
    """
    wheel_keywords = ("wheel", "motor_mount", "电机座", "轮")
    removed_names: set[str] = set()
    kept_parts: list[Part] = []
    for p in assembly.parts:
        name_lower = p.name.lower()
        if any(kw in name_lower for kw in wheel_keywords):
            removed_names.add(p.name)
            logger.info(
                "Sanitizer: removed wheel part '%s' from arm assembly", p.name
            )
        else:
            kept_parts.append(p)
    if not removed_names:
        return assembly
    kept_joints: list[Joint] = []
    for j in assembly.joints:
        if j.parent in removed_names or j.child in removed_names:
            logger.info(
                "Sanitizer: removed joint '%s' -> '%s' (references stripped part)",
                j.parent, j.child,
            )
        else:
            kept_joints.append(j)
    assembly.parts = kept_parts
    assembly.joints = kept_joints
    logger.info(
        "Sanitizer: stripped %d wheel part(s), %d parts / %d joints remain",
        len(removed_names), len(assembly.parts), len(assembly.joints),
    )
    return assembly


def _normalize_wheel_positions(assembly: Assembly) -> Assembly:
    """Fix wheel joint offsets ONLY when the solved wheels are actually wrong.

    The LLM frequently emits wheel joint ``offset`` values that overlap the
    four wheels (``wheel_fr``/``wheel_rr`` collide) or fling them far outside
    the chassis. BUT ``build_wheeled_base`` / ``compose_dual_arm_assembly``
    already produce correct wheel layouts — and a previous revision of this
    sanitizer UNCONDITIONALLY overwrote those correct offsets, breaking the
    2026-06-26 working run (wheels went from Z=47.6 ground-contact to Z=96
    floating). Regression confirmed by re-solving build_wheeled_base with the
    old sanitizer.

    So this is now CONDITIONAL: solve the assembly first, measure the actual
    wheel positions, and only override when there is a genuine defect —
    overlapping wheels, or wheels far from a sensible 4-corner layout. A
    correct layout is left untouched.
    """
    wheel_joint_suffixes = ("_fl", "_fr", "_rl", "_rr")
    wheel_joints = {
        j.child[-3:].lower(): j for j in assembly.joints
        if j.child[-3:].lower() in wheel_joint_suffixes
        and "wheel" in j.child.lower()
    }
    if len(wheel_joints) < 4:
        return assembly  # not a 4-wheel layout

    base = next(
        (p for p in assembly.parts
         if "base" in p.name.lower() and "plate" in p.name.lower()),
        None,
    )
    if base is None:
        return assembly
    bd = base.dimensions
    base_l = float(bd.get("length", 0) or 0)
    base_w = float(bd.get("width", 0) or 0)
    if base_l < 50 or base_w < 50:
        return assembly  # base dims unreliable

    # Wheel radius for overlap/ground checks.
    wheel_part = next(
        (p for p in assembly.parts if "wheel" in p.name.lower()), None,
    )
    wheel_r = float(
        (wheel_part.dimensions.get("diameter", 0) or 0) / 2.0
        if wheel_part else 45.0
    ) or 45.0

    # --- Solve and inspect the ACTUAL wheel positions before touching anything.
    try:
        from ..assembly_solver import AssemblySolver
        solved = AssemblySolver(assembly).solve()
    except Exception:
        return assembly  # can't verify — don't risk corrupting the layout

    wheel_pos = {
        suf: solved[j.child]["position"]
        for suf, j in wheel_joint_suffixes_map(wheel_joints).items()
        if j.child in solved
    }
    if len(wheel_pos) < 4:
        return assembly

    # Defect 1: any pair of wheels overlaps (centers closer than wheel diameter).
    import itertools
    min_pair_dist = min(
        ((wheel_pos[a][0]-wheel_pos[b][0])**2
         + (wheel_pos[a][1]-wheel_pos[b][1])**2)**0.5
        for a, b in itertools.combinations(wheel_pos, 2)
    )
    # Defect 2: wheels not near the ground (Z far from wheel_r).
    avg_z = sum(p[2] for p in wheel_pos.values()) / 4.0
    z_bad = abs(avg_z - wheel_r) > max(wheel_r * 0.6, 30.0)

    if min_pair_dist >= (wheel_r * 1.5) and not z_bad:
        # Layout is fine — do NOT touch it (preserves correct chassis builds).
        return assembly

    logger.info(
        "Sanitizer: wheel layout defective (min_pair_dist=%.0fmm, avg_Z=%.0f "
        "vs wheel_r=%.0f) — overriding offsets to canonical 4-corner layout",
        min_pair_dist, avg_z, wheel_r,
    )

    # Canonical corner offsets (solver X=lateral, Y=forward/back, Z=up).
    half_w = base_w / 2.0
    half_l = base_l / 2.0
    fy = half_l * 0.78
    corners = {"_fl": (-1, +1), "_fr": (+1, +1), "_rl": (-1, -1), "_rr": (+1, -1)}

    fixed = []
    for suf, (xs, ys) in corners.items():
        j = wheel_joints.get(suf)
        if j is None:
            continue
        # Wheel sits coaxial with its parent motor (shared axle). Z=0 relative
        # to the motor center; the motor's own position sets the axle height.
        new_offset = [xs * half_w, ys * fy, 0.0]
        if j.offset != new_offset:
            old = list(j.offset) if j.offset else None
            j.offset = new_offset
            j.parent_anchor = "center"
            j.child_anchor = "center"
            fixed.append((j.child, old, new_offset))

    if fixed:
        logger.info(
            "Sanitizer: reset %d wheel joint offset(s) to canonical 4-corner "
            "layout (base %.0f×%.0f): %s",
            len(fixed), base_l, base_w,
            [(c, [round(v, 1) for v in (o or [])], [round(v, 1) for v in n])
             for c, o, n in fixed],
        )
    return assembly


def wheel_joint_suffixes_map(wheel_joints: dict) -> dict:
    """Helper: return the suffix->joint dict as-is (kept for clarity)."""
    return wheel_joints


def _normalize_gripper_fingers(assembly: Assembly) -> Assembly:
    """Ensure gripper fingers are symmetrically separated, anchored at center.

    The LLM often emits left/right finger joints with face anchors such as
    ``"front"``/``"back"`` (or inconsistent pairs).  Face anchors contribute a
    non-zero positional offset via ``_anchor_offset_for_part`` (e.g. ±Y for
    front/back).  When that anchor offset is then *added* to the explicit
    lateral ``offset`` (±X) by the solver, the resulting 3D displacement can
    exceed the ``_clamp_child_offset`` threshold and be scaled down — but
    worse, in the 4dof_arm audit the LLM's "front"/"back" anchors produced a
    ±Y displacement that, combined with the ±X finger offset, summed to a
    ~477mm vector which ``_clamp_child_offset`` truncated to ~330mm and the
    URDF exporter then emitted as a **322mm** joint origin for
    ``gripper_finger_left``.  The fingers ended up far from the gripper base.

    Root cause: anchor (rotational face) and offset (lateral position) both
    move the child, so they compound.  The fix is to make the anchor
    contribute **rotation only** by forcing ``"center"`` for both parent and
    child anchors.  ``_anchor_offset_for_part(part, "center")`` returns
    ``(0, 0, 0)``, so the solver computes ``child_center = parent_center +
    rot @ offset`` — offset becomes the sole position determinant and the
    intended symmetric ±X gap is preserved exactly.

    This sanitizer:
    1. Detects left/right finger pairs by name.
    2. Forces ``parent_anchor == child_anchor == "center"`` on both joints
       (root-cause fix for the 4dof_arm 322mm URDF origin).
    3. Sets ``no_distribute=True`` to prevent auto-distribution.
    4. Sets explicit lateral (**Y**) offsets perpendicular to the finger
       length, and switches the prismatic axis to ``"y"`` so the grip
       opens/closes in the correct direction.
    """
    finger_left_kw = ("finger_left", "left_finger", "left_gripper",
                      "gripper_left", "左爪", "左指", "左夹", "左手指")
    finger_right_kw = ("finger_right", "right_finger", "right_gripper",
                       "gripper_right", "右爪", "右指", "右夹", "右手指")

    parts_by_name = {p.name: p for p in assembly.parts}

    left_name = None
    right_name = None
    for p in assembly.parts:
        nl = p.name.lower()
        if left_name is None and any(kw in nl for kw in finger_left_kw):
            left_name = p.name
        if right_name is None and any(kw in nl for kw in finger_right_kw):
            right_name = p.name

    if not left_name or not right_name:
        return assembly

    left_joint = None
    right_joint = None
    for j in assembly.joints:
        if j.child == left_name:
            left_joint = j
        elif j.child == right_name:
            right_joint = j

    if not left_joint or not right_joint:
        return assembly

    # L1 fix: anchors contribute rotation only — offset is the sole position
    # determinant.  Prevents anchor (front/back -> +/-Y displacement) + offset
    # (+/-X) from compounding into the 300mm+ URDF origins observed in the
    # 4dof_arm audit (gripper_finger_left measured at 322mm).  With "center"
    # anchors, ``_anchor_offset_for_part(part, "center")`` returns (0,0,0), so
    # the solver computes child_center = parent_center + rot*offset — exactly
    # the intended symmetric geometry.
    for j in (left_joint, right_joint):
        j.parent_anchor = "center"
        j.child_anchor = "center"

    # Disable auto-distribution so explicit offsets are the sole lateral factor
    left_joint.no_distribute = True
    right_joint.no_distribute = True

    # Compute the lateral gap between the two fingers.
    #
    # Coordinate convention — TWO frames (fixed 2026-06-22):
    #
    #   PART-LOCAL (FreeCAD makeBox, the STL mesh):
    #     finger length → X,  finger width → Y,  height → Z
    #
    #   SOLVER/WORLD (where this offset lives; what the renderer sees after
    #   swap_xy R_z(-90°) maps part-local X → world -Y, Y → world +X):
    #     forward (front/back) → Y    (ANCHOR_DIRECTIONS: front=(0,-1,0))
    #     lateral (left/right) → X    (ANCHOR_DIRECTIONS: left=(-1,0,0))
    #     up/down             → Z
    #
    # The finger offset is in WORLD coords: gap on X (lateral, so fingers
    # straddle the arm centreline) and forward on Y (so both fingers
    # protrude ahead of the base).  After swap_xy the finger STL's length
    # (part-local X) renders along world Y — aligned with the forward
    # offset — and width (part-local Y) renders along world X — aligned
    # with the gap.  So the two fingers appear as parallel bars extending
    # forward, separated left/right.  This reads as a gripper.
    #
    # PREVIOUS BUG (pre-2026-06-22): offset was (forward_x, ±gap, 0) —
    # forward on X, gap on Y.  After swap_xy both fingers landed on the
    # same side of the arm (world -Y), separated along the arm-length
    # axis — they read as two extra links, not a gripper.
    #
    # Geometric invariant: gap > finger_width guarantees the world-X AABBs
    # (finger width on world X after swap) do not overlap.  6mm clearance.
    parent_part = parts_by_name.get(left_joint.parent)
    gap = 22.0
    base_length = 28.0
    finger_w = 14.0
    finger_l = 60.0
    if parent_part and parent_part.dimensions:
        w = parent_part.dimensions.get("width",
                    parent_part.dimensions.get("depth", 50))
        base_length = parent_part.dimensions.get("length", 28.0)
        finger_part = parts_by_name.get(left_joint.child)
        if finger_part and finger_part.dimensions:
            finger_w = finger_part.dimensions.get("width", 14.0)
            finger_l = finger_part.dimensions.get("length", 60.0)
        # gap must exceed finger width so AABBs separate on Y.  Cap by the
        # parent base width so fingers stay within the gripper footprint.
        min_gap = finger_w + 6.0          # guarantee > width + 6mm grip gap
        max_gap = w / 2.0 - 2.0           # finger stays within base
        if max_gap < min_gap:
            # Base too narrow to fit both fingers inside — prefer the
            # geometric invariant (no intersection) over footprint fit.
            gap = min_gap
        else:
            gap = max(min_gap, min(min_gap * 1.25, max_gap))

    # Prismatic axis is X (lateral): fingers slide toward/away from each
    # other along X (open/close the grip), perpendicular to the forward Y
    # direction.  This matches the gap axis so closing the grip moves each
    # finger toward the centreline on X.
    for j in (left_joint, right_joint):
        j.axis = "x"

    # Push fingers forward along Y so the main bar fully protrudes beyond
    # the gripper base face — but WHICH direction is "forward" depends on
    # where the parent chain attaches to the gripper base.  The fingers
    # must point AWAY from the arm (the parent link), not back into it.
    #
    # The arm-side link (e.g. wrist_link) connects to gripper_base via a
    # joint whose child_anchor names the face it mounts on.  If the parent
    # attaches on the 'back' face, fingers go to 'front' (+X); if on
    # 'front', fingers go to 'back' (-X).  This was previously hardcoded
    # to -X, which drove the fingers back into the wrist_link whenever the
    # arm attached on the back face (the 4dof_arm topology) — causing the
    # wrist_link/gripper_finger intersection that the VLM loop could never
    # resolve.
    parent_face = None
    for j in assembly.joints:
        if j.child == left_joint.parent and j.child_anchor != "center":
            parent_face = j.child_anchor
            break
    # ANCHOR_DIRECTIONS (assembly_solver): front=(0,-1,0), back=(0,1,0).
    # Fingers protrude ALONG Y (forward/back), gap is on X (lateral).
    # Parent on back (+Y)  -> fingers toward front (-Y), away from arm.
    # Parent on front (-Y) -> fingers toward back (+Y), away from arm.
    # This convention matches the renderer's swap_xy (R_z(-90°)): the
    # finger STL has length on FreeCAD-X, which swap_xy maps to world -Y,
    # so the finger's long edge visually extends along world Y — the same
    # axis as the forward offset.  Gap on X means the two fingers straddle
    # the arm centreline left/right (world X), which reads as a gripper
    # in the render.  (The old convention put forward on X and gap on Y,
    # which after swap_xy placed both fingers on the same side of the arm
    # — they read as two extra links lined up, not a gripper.)
    if parent_face == "back":
        forward_sign = -1.0    # arm at +Y → fingers to -Y (front)
    elif parent_face == "front":
        forward_sign = +1.0    # arm at -Y → fingers to +Y (back)
    else:
        forward_sign = -1.0    # default: fingers forward (-Y)
    forward_y = forward_sign * (base_length / 2.0 + finger_l / 2.0)
    left_joint.offset = (-gap, forward_y, 0.0)
    right_joint.offset = (gap, forward_y, 0.0)

    # Dynamic range clamp: prevent finger collision.
    # The closing displacement moves both fingers toward center (mimic=-1).
    # Max safe close = gap - finger_w/2 - 1mm_margin.
    # At this displacement, inner faces have >= 2mm clearance.
    max_close = gap - finger_w / 2.0 - 1.0
    for j in (left_joint, right_joint):
        if j.type == "prismatic" and j.range_deg:
            lo, hi = j.range_deg
            hi = min(hi, max_close)
            lo = min(lo, -1.0)  # ensure at least 1mm opening range
            j.range_deg = (lo, hi)
            logger.info(
                "Sanitizer: clamped gripper finger %s range to "
                "(%.1f, %.1f) mm (gap=%.1f, finger_w=%.1f)",
                j.child, lo, hi, gap, finger_w,
            )

    # Clear connection_method on prismatic finger joints — sliding interfaces
    # are not fastenings.  The LLM frequently marks them "bolted" which is
    # mechanically wrong (you cannot bolt a sliding finger to the rail) and
    # causes the CAD feature engine to generate spurious bolt holes on the
    # linear rail grooves.
    for j in (left_joint, right_joint):
        if j.type == "prismatic" and j.connection is not None:
            logger.info(
                "Sanitizer: cleared connection_method on prismatic joint "
                "%s->%s (sliding fit, not a fastening)",
                j.parent, j.child,
            )
            j.connection = None

    # Wire the right finger to mimic the left (antagonistic grip).
    # Without this, the URDF exporter emits two independent prismatic joints,
    # so opening/closing one finger does not move the other — the gripper
    # cannot actually grasp.  mimic_multiplier=-1 makes them move symmetrically
    # toward/away from centre.
    if not left_joint.mimic_joint and not right_joint.mimic_joint:
        right_joint.mimic_joint = left_joint.child
        right_joint.mimic_multiplier = -1.0
        right_joint.mimic_offset = 0.0
        logger.info(
            "Sanitizer: set %s to mimic %s (multiplier=-1.0) for "
            "antagonistic grip",
            right_joint.child, left_joint.child,
        )

    logger.info(
        "Sanitizer: normalized gripper fingers '%s'/'%s' — "
        "anchors=center/center, axis=x, gap=±%.1fmm (X), z_lift=0",
        left_name, right_name, gap,
    )
    return assembly


def _validate_proportions(assembly: Assembly) -> Assembly:
    """Validate part proportions and raise on physically bad ratios.

    P1-1: previously this sanitizer SILENTLY CLAMPED disproportionate
    dimensions (gripper width, link length, link cross-section) so the
    rendered assembly "looked right".  Per CLAUDE.md ("不要在代码里加
    hack 让 LLM/外部输入看起来对"), clamp-and-pretend masks the real
    data-quality issue from the VLM retry loop so the LLM never learns
    to produce coherent dimensions.

    Now the function COLLECTS every proportion violation and raises a
    single RuntimeError describing all of them, so the error enters
    ``problems_history`` and the LLM gets a chance to regenerate with
    corrected dimensions.  Returns ``assembly`` unchanged when valid.

    Checks:
    1. gripper_base width ≤ 1.8 × parent link width
    2. Consecutive link length ratio < 3.0
    3. link cross-section ≥ 0.55 × joint diameter (width) / 0.50× (height)
    """
    problems: list[str] = []
    parts_by_name = {p.name: p for p in assembly.parts}

    for joint in assembly.joints:
        parent = parts_by_name.get(joint.parent)
        child = parts_by_name.get(joint.child)
        if not parent or not child:
            continue
        if not parent.dimensions or not child.dimensions:
            continue

        parent_w = parent.dimensions.get("width", 0)
        child_w = child.dimensions.get("width", 0)
        parent_l = parent.dimensions.get("length", 0)
        child_l = child.dimensions.get("length", 0)

        # Check 1: gripper_base width should not dwarf the parent link.
        # P1 correction: the original 1.8x threshold was too tight — a
        # real gripper base houses a servo (SG90 = 22mm wide) plus linear
        # guide rails plus finger mounts, so it is naturally 2-2.5x the
        # wrist link width.  3.0x captures grossly oversized grippers
        # (e.g. 90mm gripper on a 20mm wrist) without rejecting the
        # standard SG90 grip-per-base (50mm on a 22mm wrist = 2.27x).
        child_nl = child.name.lower()
        if ("gripper" in child_nl and "base" in child_nl
                and parent_w > 0 and child_w > 0):
            max_w = parent_w * 3.0
            if child_w > max_w:
                problems.append(
                    f"gripper_base '{child.name}' width {child_w:.0f}mm > "
                    f"3.0x parent '{parent.name}' width {parent_w:.0f}mm "
                    f"(limit {max_w:.0f}mm); reduce the gripper width or "
                    f"widen the parent link"
                )

        # Check 2: consecutive link length ratio
        parent_nl = parent.name.lower()
        if (parent_l > 0 and child_l > 0
                and "link" in parent_nl and "link" in child_nl):
            ratio = max(parent_l, child_l) / min(parent_l, child_l)
            if ratio > 3.0:
                problems.append(
                    f"consecutive links '{parent.name}' ({parent_l:.0f}mm) "
                    f"and '{child.name}' ({child_l:.0f}mm) have length "
                    f"ratio {ratio:.1f} > 3.0; make adjacent link lengths "
                    f"comparable (ratio < 3.0)"
                )

        # Check 3: joint-link cross-section consistency.
        # Joint cylinders (with "diameter") are often much fatter than the
        # links they connect to (e.g. diameter=40 vs link 25×15).  When the
        # joint is centred on the link's end face, the joint body extends
        # well beyond the link profile on all sides, making it look like the
        # joint "swallows" the link — visually read as parts intersecting.
        # Enforce: link cross-section ≥ 0.55 × joint diameter in both width
        # and height.  This keeps the link profile visually comparable to the
        # joint so the connection looks clean rather than overlapping.
        parent_d = parent.dimensions.get("diameter", 0)
        child_d = child.dimensions.get("diameter", 0)
        link_part = None
        joint_d = 0
        joint_name = ""
        # The cross-section rule targets ARM LINKS (the bars between joints),
        # not chassis/base plates.  A base_plate is "structural" but is a
        # thin slab by design (prompt allows 3-8mm) — applying the arm-link
        # 0.50×joint-diameter rule to it rejects every legitimate base
        # plate (8mm < 0.5×40mm = 20mm).  Restrict to parts whose name
        # actually reads as an arm link.
        def _is_arm_link(pt) -> bool:
            nl = pt.name.lower()
            return (
                pt.category in ("structural", "link")
                and ("link" in nl or "arm" in nl)
                and not any(b in nl for b in (
                    "base", "plate", "chassis", "foot", "mount",
                ))
            )
        if parent_d > 0 and "joint" in parent_nl and _is_arm_link(child):
            link_part = child
            joint_d = parent_d
            joint_name = parent.name
        elif child_d > 0 and "joint" in child_nl and _is_arm_link(parent):
            # parent is the link, child is the joint
            link_part = parent
            joint_d = child_d
            joint_name = child.name
        if link_part is not None and joint_d > 0:
            min_w = joint_d * 0.55
            min_h = joint_d * 0.50
            link_w = link_part.dimensions.get("width", 0)
            link_h = link_part.dimensions.get("height", 0)
            if link_w > 0 and link_w < min_w:
                problems.append(
                    f"link '{link_part.name}' width {link_w:.0f}mm < "
                    f"0.55x joint '{joint_name}' diameter {joint_d:.0f}mm "
                    f"(need >= {min_w:.0f}mm); the joint visually swallows "
                    f"the link — widen the link"
                )
            if link_h > 0 and link_h < min_h:
                problems.append(
                    f"link '{link_part.name}' height {link_h:.0f}mm < "
                    f"0.50x joint '{joint_name}' diameter {joint_d:.0f}mm "
                    f"(need >= {min_h:.0f}mm); the joint visually swallows "
                    f"the link — increase the link height"
                )

    if problems:
        raise RuntimeError(
            "Proportion validation failed: " + "; ".join(problems)
        )

    return assembly


def _ensure_arm_default_angles(assembly: Assembly) -> Assembly:
    """Inject non-zero default_angles for arm pitch joints that lack them.

    Even with prompt rules asking for bent postures, the LLM frequently emits
    all-zero default_angles for the pitch joints.  Combined with vertical
    top/bottom anchors, zero angles stack every part into a single straight
    column with no 3D extent — the VLM then sees "1 part" and the
    motion-collision sweep flags self collisions because the links overlap
    end-to-end.

    This sanitizer operates **per joint**, not all-or-nothing:

    1. **Clean**: remove default_angles entries whose key is NOT the child of a
       revolute joint.  The LLM sometimes emits entries for fixed-joint
       children (e.g. ``gripper_base``) or random structural parts — these are
       meaningless (fixed joints cannot rotate) and pollute the pose.
    2. **Preserve**: for each revolute joint where the LLM supplied a non-zero
       angle, keep it unchanged.
    3. **Inject**: for each revolute pitch joint that is zero or missing,
       synthesise a natural zig-zag bend by alternating the sign.

    A single stray non-zero value (e.g. the LLM setting only the wrist roll)
    no longer causes the sanitizer to skip every other pitch joint.
    """
    existing = dict(assembly.default_angles or {})

    # Build the set of child names that belong to revolute joints.
    revolute_children = {j.child for j in assembly.joints if j.type == "revolute"}

    # --- Clean: strip keys that are not revolute joint children. ---
    cleaned: dict[str, float] = {}
    removed: list[str] = []
    for k, v in existing.items():
        if k in revolute_children:
            try:
                cleaned[k] = float(v)
            except (TypeError, ValueError):
                removed.append(k)
        else:
            removed.append(k)
    if removed:
        logger.info(
            "Sanitizer: removed non-revolute default_angles keys: %s",
            removed,
        )

    # Zero out roll/yaw default angles that are NOT the base yaw joint.
    # Only pitch joints (axis=x, the up/down bend) and the base yaw
    # (first axis=z joint, turning the whole arm left/right) should
    # carry a non-zero default — they define the arm's reach pose.
    # Other roll (axis=y, spinning the end effector) and yaw joints
    # rotate the END of the arm, which breaks gripper symmetry in the
    # default pose (a 35° wrist roll tilts the whole gripper sideways,
    # making the two fingers project asymmetrically in world space).
    # The LLM frequently emits non-zero wrist roll; zero it here.
    revolute_joints_tmp = [
        j for j in assembly.joints if j.type == "revolute"
    ]
    # Base yaw = the FIRST z-axis joint of EACH arm chain (not just the
    # global first). A dual-arm assembly has two base yaws (arm_l + arm_r),
    # both carrying the collision-aware splay — zeroing the right one
    # destroyed the symmetric park pose and made the arms asymmetric. Detect
    # base-yaw children per arm-prefix so every arm keeps its base yaw.
    import re as _re_yaw
    def _arm_yaw_prefix(name: str) -> str:
        m = _re_yaw.match(r"(arm_[lr]|left_|right_)", name.lower())
        return m.group(1) if m else ""
    base_yaw_children: set[str] = set()
    _seen_yaw_prefixes: set[str] = set()
    for j in revolute_joints_tmp:
        if j.axis == "z":
            pref = _arm_yaw_prefix(j.child)
            if pref not in _seen_yaw_prefixes:
                _seen_yaw_prefixes.add(pref)
                base_yaw_children.add(j.child)
            elif not pref:
                # No-prefix arm: only the very first z joint is base yaw.
                if not base_yaw_children:
                    base_yaw_children.add(j.child)
    zeroed_roll = []
    for j in revolute_joints_tmp:
        if j.axis in ("y", "z") and j.child not in base_yaw_children:
            cur = cleaned.get(j.child)
            if cur is not None and abs(float(cur)) > 1e-6:
                zeroed_roll.append((j.child, cur))
                cleaned[j.child] = 0.0
    if zeroed_roll:
        logger.info(
            "Sanitizer: zeroed non-base roll/yaw default_angles "
            "(they tilt the gripper in the default pose): %s",
            zeroed_roll,
        )

    # Detect an arm-like assembly: at least 2 revolute joints and at least 1
    # link-like structural part.
    revolute_joints = revolute_joints_tmp
    has_link = any(
        _is_link_like(p.name) or _is_end_effector(p.name)
        for p in assembly.parts
    )
    if len(revolute_joints) < 2 or not has_link:
        assembly.default_angles = cleaned
        return assembly

    # Helper: identify ARM pitch joints while EXCLUDING drive-train joints.
    # Wheels/tires are axis=x (the axle runs along X for differential drive)
    # and motors sit on the same axle — they are NOT arm bend joints. Treating
    # them as pitch joints gave wheels spurious ±35° angles (rotated render)
    # AND pushed the real arm joints to higher pitch_idx, skipping the
    # shoulder's "force negative" branch so the arm drooped. Defined early so
    # both the zig-zag check and the pitch_children list use the same filter.
    def _is_arm_pitch_joint(j) -> bool:
        if j.axis != "x" or j.type != "revolute":
            return False
        c = j.child.lower()
        if any(c.startswith(p) for p in ("wheel_", "tire_")):
            return False
        if "motor" in c:
            return False
        return True

    # --- Anchor-consistency check (clean arm convention). ---
    # Under the clean convention, pitch joints (axis=x) must use front/back
    # anchors so the link's `length` dimension positions the next axis. A
    # pitch joint on top/bottom anchors means the LLM ignored the rule; the
    # IK link lengths still come from part dimensions, but the solver will
    # stack the arm into a vertical column. Warn (non-blocking) so this is
    # visible in logs without rejecting the assembly.
    mismatched = [
        j.child for j in revolute_joints
        if j.axis == "x"
        and {j.parent_anchor, j.child_anchor} != {"front", "back"}
        and j.parent_anchor in ("top", "bottom")
    ]
    if mismatched:
        logger.warning(
            "Sanitizer: %d arm pitch joint(s) use top/bottom anchors instead "
            "of front/back (clean convention): %s. Link lengths remain correct "
            "(read from part dimensions) but the solver may stack the arm "
            "vertically.",
            len(mismatched), mismatched,
        )

    # --- Per-joint fill: inject bends for pitch joints that are zero/missing. ---
    injected: dict[str, float] = dict(cleaned)
    pitch_index = 0
    filled: list[tuple[str, float]] = []

    # --- Over-fold detection: an arm whose pitch joints all bend the same
    # direction curls up on itself, crushing the end-effector into the
    # base and making the gripper impossible to resolve in renders (the
    # VLM then reports "gripper missing").  The LLM frequently emits
    # -35/-35/-35 (cumulative -105°), folding the arm into a tight coil.
    # A natural arm pose alternates direction (zig-zag): -45/-30/+15.
    # If the cumulative same-sign pitch exceeds 90°, override with a
    # canonical zig-zag so the arm extends enough to expose the gripper.
    # Only pitch joints (axis=x) are considered — yaw (axis=z) and roll
    # are not "bend" joints and must be left alone. Wheels/motors are also
    # axis=x (axle along X) but are drive-train parts, NOT arm bends —
    # exclude them (same filter as pitch_children above).
    pitch_joints_arm = [
        j for j in revolute_joints if _is_arm_pitch_joint(j) and j.child
    ]
    pitch_vals = [
        float(injected.get(j.child, 0.0) or 0.0) for j in pitch_joints_arm
    ]
    nonzero_pitch = [v for v in pitch_vals if abs(v) > 1e-6]
    _overrode_to_zigzag = False
    if len(nonzero_pitch) >= 2:
        all_same_sign = (
            all(v < -1e-6 for v in nonzero_pitch)
            or all(v > 1e-6 for v in nonzero_pitch)
        )
        cumulative = sum(nonzero_pitch)
        # Threshold scales with joint count: 2 pitch joints folding
        # -30°/-30° = -60° already curls the forearm back; 3 joints
        # need more room.  ~25° per joint is the "natural bend" ceiling
        # before the arm starts folding on itself.
        fold_threshold = 25.0 * len(nonzero_pitch)
        if all_same_sign and abs(cumulative) > fold_threshold:
            logger.warning(
                "Sanitizer: arm over-folded (all pitch joints same sign, "
                "cumulative %.0f° > %.0f° threshold for %d joints). "
                "Overriding with zig-zag so the gripper is visible: %s",
                cumulative, fold_threshold, len(nonzero_pitch),
                [j.child for j in pitch_joints_arm],
            )
            # Zig-zag template: alternate sign so the arm extends
            # outward instead of curling in.  Generalised to any pitch-joint
            # count via knowledge/arm_topology.zigzag_angles (the legacy
            # hard-coded [-45,30] / [-45,-30,15,-10] wrapped arbitrarily for
            # >4 pitch joints, producing a curled pose on 7-DOF arms).
            n = len(pitch_joints_arm)
            _zigzag_seq = zigzag_angles(n)
            for idx, j in enumerate(pitch_joints_arm):
                injected[j.child] = _zigzag_seq[idx % len(_zigzag_seq)]
            _overrode_to_zigzag = True
    for j in revolute_joints:
        # Skip drive-train joints (wheels/tires/motors) — they are axis=x like
        # arm pitch joints but must NOT get a bend angle (a rotated wheel
        # renders as a tilted disc). Only fill arm-chain joints.
        if not _is_arm_pitch_joint(j) and j.axis != "z" and j.axis != "y":
            continue
        # Base yaw (axis=z): clamp to ±10° for a SINGLE arm so it points
        # forward. BUT for a wheeled DUAL-arm assembly, the base yaw carries
        # the collision-aware splay (±30° outward) configured by
        # _configure_collision_aware_dual_arm — zeroing it would destroy the
        # anti-collision park pose and make the arms overlap. Detect dual-arm
        # (both arm_l_ and arm_r_ prefixes present) and PRESERVE those yaws.
        _is_dual_arm = (
            any(p.name.startswith("arm_l_") for p in assembly.parts)
            and any(p.name.startswith("arm_r_") for p in assembly.parts)
        )
        if j.axis == "z" and pitch_index == 0 and not _is_dual_arm:
            # Base yaw: the default (home) pose should point the arm
            # straight forward (yaw = 0).  Any non-zero yaw rotates the
            # whole arm, which carries through to the gripper and makes
            # the two fingers project asymmetrically in world space
            # (a 10° yaw offsets the ±20mm gap into a 39mm world-Y
            # difference).  The VLM verifies the home pose from a world
            # view, so symmetry matters here.  Yaw for reaching different
            # directions is exercised by the workspace/motion-sweep
            # checks, not baked into the default pose.
            injected[j.child] = 0.0
            pitch_index += 1
            continue

        # Roll joints (axis=y, spinning the end effector) must stay at 0°
        # in the default pose — a non-zero roll tilts the gripper and
        # breaks finger symmetry.  They were zeroed in the Clean phase
        # above; skip the fill so we don't reinject a value.
        if j.axis == "y":
            if injected.get(j.child) is None:
                injected[j.child] = 0.0
            continue

        current = injected.get(j.child)
        if current is not None and abs(float(current)) > 1e-6:
            # LLM explicitly gave a non-zero angle for this joint — keep it.
            pitch_index += 1
            continue

        # Compute a bend from the joint's range.  Cap at 35 degrees — larger
        # values (e.g. 90° from 30% of a 300° range) fold the arm back on
        # itself, causing motion-collision sweep failures and COM instability.
        lo, hi = j.range_deg if j.range_deg else (-120.0, 120.0)
        try:
            lo_f, hi_f = float(lo), float(hi)
        except (TypeError, ValueError):
            lo_f, hi_f = -120.0, 120.0
        span = hi_f - lo_f
        magnitude = max(15.0, min(abs(span) * 0.20, 35.0))
        # Clamp into the legal range so the angle is realisable.
        magnitude = min(magnitude, abs(span) / 2.0 - 1.0) if span > 2.0 else 15.0
        if magnitude < 5.0:
            magnitude = 15.0
        # Alternate sign to produce a zig-zag (natural-looking) posture.
        sign = -1.0 if (pitch_index % 2 == 0) else 1.0
        angle = sign * magnitude
        # Keep inside [lo, hi].
        angle = max(lo_f + 1.0, min(hi_f - 1.0, angle))
        injected[j.child] = round(angle, 1)
        filled.append((j.child, round(angle, 1)))
        pitch_index += 1

    if filled:
        logger.info(
            "Sanitizer: injected default_angles bends for arm '%s': %s",
            assembly.name, filled,
        )

    # --- Rising-arm pose (front/back convention). ---
    # With front/back anchors + axis=x, 0° = the link lies horizontal
    # (extending forward in -Y).  Pitch angles tilt each link upward.
    # For the arm to rise in Z (look 3D, not flat) ALL pitch joints must
    # tilt the SAME direction (negative = upward) so their effects
    # reinforce rather than cancel.  This is the opposite of the old
    # zig-zag logic, which was designed for the top/bottom convention
    # (where the arm starts vertical and alternating signs create bends).
    #
    # Each pitch is clamped to a moderate magnitude so the arm doesn't
    # fold back on itself (too steep) or stay flat (too horizontal).
    #
    # SKIPPED when the over-fold detector already applied a zig-zag: the
    # zig-zag alternates signs on purpose (to extend the arm and expose
    # the gripper), so forcing all-same-sign here would undo it and fold
    # the arm right back. The two pose strategies are mutually exclusive.
    # IMPORTANT: wheels are also axis=x (the axle runs along X for a
    # differential drive), so they must be EXCLUDED — otherwise they get
    # mis-treated as arm pitch joints, given spurious ±35° angles (the
    # wheels render rotated), AND they push the real arm joints to higher
    # pitch_idx so the shoulder's "force negative" branch is skipped and
    # the arm gets forced positive (drooping). The _is_arm_pitch_joint
    # helper (defined above, near the early-return) excludes wheel_/tire_/
    # motor_ so both the zig-zag check and this list stay drive-train-free.
    pitch_children = [
        j.child for j in revolute_joints
        if _is_arm_pitch_joint(j) and j.child in injected
    ]
    range_limit: dict[str, float] = {}
    for j in revolute_joints:
        if _is_arm_pitch_joint(j) and j.range_deg:
            try:
                lo_r, hi_r = float(j.range_deg[0]), float(j.range_deg[1])
                range_limit[j.child] = min(abs(lo_r), abs(hi_r)) - 1.0
            except (TypeError, ValueError):
                pass
    adjusted: list[tuple[str, float, float]] = []
    # Group pitch joints per-arm so EACH arm's first pitch joint (the
    # shoulder) is treated as shoulder (force negative = tilt up), not just
    # the global first joint. Without this, a dual-arm assembly treats the
    # right arm's shoulder as an elbow (pitch_idx > 0) and forces it positive
    # → the right arm droops while the left rises ("一个向上一个向下").
    # Arm prefix = the leading "arm_l" / "arm_r" / "left_" / "right_" token;
    # fall back to "" (treat as one chain) if no prefix.
    import re as _re
    def _arm_prefix(name: str) -> str:
        m = _re.match(r"(arm_[lr]|left_|right_)", name.lower())
        return m.group(1) if m else ""
    _seen_shoulders: set[str] = set()
    pitch_idx = 0
    for child in pitch_children:
        val = float(injected[child])
        # Is this the FIRST pitch joint of its arm chain? → shoulder.
        pref = _arm_prefix(child)
        is_shoulder = pref not in _seen_shoulders
        _seen_shoulders.add(pref)
        # Moderate cap so the arm tilts up without folding back.
        if is_shoulder:
            cap = 35.0      # Shoulder: sets the overall reach angle.
            min_mag = 30.0  # Minimum shoulder tilt for a working-arm look.
        else:
            cap = 40.0      # Subsequent pitches reinforce the rise.
            min_mag = 30.0  # Minimum elbow/wrist tilt.
        rl = range_limit.get(child)
        if rl is not None and rl > 0:
            cap = min(cap, rl)
        clamped = max(-cap, min(cap, val))
        if is_shoulder:
            # Shoulder (first pitch of THIS arm): force negative (upward tilt)
            # so the arm rises in Z.  This is the only joint that MUST be
            # same-sign for the arm to point up rather than lie flat. Applied
            # per-arm so a dual-arm assembly tilts BOTH shoulders up.
            if clamped > 0:
                clamped = -clamped
            if abs(clamped) < min_mag:
                clamped = -min(min_mag, cap)
        else:
            # Subsequent pitch joints (elbow, wrist): force POSITIVE
            # (opposite of shoulder).  The home pose must show an
            # EXTENDED arm, not a folded one.  The LLM systematically
            # emits same-sign angles (e.g. shoulder -30 + elbow -35),
            # which fold the arm into a coil and crush the gripper into
            # the base.  Forcing elbow/wrist positive guarantees the arm
            # extends outward regardless of what the LLM emitted.
            # Previous "preserve sign for large values" logic still
            # produced folds when the LLM gave -35 (>= min_mag), so the
            # pose ended up coiled in every E2E run.
            clamped = abs(clamped)  # force positive
            if clamped < min_mag:
                clamped = min_mag
        if abs(clamped - val) > 0.05:
            adjusted.append((child, val, clamped))
            injected[child] = round(clamped, 1)
        pitch_idx += 1

    if adjusted:
        logger.info(
            "Sanitizer: rising-arm default_angles for '%s': %s",
            assembly.name,
            [(c, f"{old:.0f}->{new:.0f}") for c, old, new in adjusted],
        )

    assembly.default_angles = injected

    # --- Base-plate sizing (COM stability, added 2026-06-22). ---
    # A 4-DOF arm with ~400mm of link reach generates a center of mass that
    # projects ~100mm forward of the base origin in the home pose.  When the
    # LLM emits a small base plate (e.g. 200×150), the support polygon
    # (±length/2 = ±100mm) lands exactly on the COM edge, so the
    # com_stability check fails by 1-2mm on a geometrically-correct arm.
    # Ensure the base plate's LENGTH (solver Y, the arm reach direction) is
    # at least 60% of the total arm link reach so the support polygon has
    # margin.  Width (solver X) is left alone — lateral COM offset is small.
    #
    # WHEELED-ROBOT GUARD: a wheeled chassis's base_plate is the CHASSIS deck,
    # sized by mobile_base_gen from real UGV proportions (Husky-class), with
    # wheels attached as children. Enlarging it here to chase an arm-COM
    # offset blows the deck up to 1176/1400mm (observed in
    # data/runs/wheeled_arm/20260627_*), which pushes the child wheels up to
    # Z≈248mm (floating ~200mm off the ground) and outward to XY≈246mm — the
    # "wheels above Z=248 / parts misplaced" VLM failures. Wheeled robots are
    # supported by their WHEELS (ground contact), not by a bench base plate,
    # so the fixed-arm COM-on-plate assumption does not apply. Skip entirely
    # when the assembly has wheels.
    has_wheels = any(
        "wheel" in p.name.lower() for p in assembly.parts
    )
    base_part = next(
        (p for p in assembly.parts
         if "base" in p.name.lower() and "plate" in p.name.lower()),
        None,
    )
    link_parts = [
        p for p in assembly.parts
        if _is_link_like(p.name) or "joint" in p.name.lower()
    ]
    if base_part and link_parts and not has_wheels:
        # Effective forward COM offset of a zig-zag arm. Real solves show the
        # COM projects ~0.30× the raw link-sum forward of the base (measured:
        # 400mm links → COM Y≈-110 to -132mm across LLM-generated poses).
        # Using 0.32 with a margin covers the observed variance.
        #
        # AXIS MAPPING (verified against AssemblySolver + assembly_verifier):
        #   - The arm reaches along solver -Y (front/back anchors).
        #   - base LENGTH is the Y (forward) dimension; WIDTH is X (lateral).
        #   - assembly_verifier.build_support_polygon uses (cy ± length/2)
        #     for the forward extent, so LENGTH must cover the COM offset.
        #   - An earlier revision enlarged WIDTH by mistake (treating width as
        #     Y), which left the forward support edge too short → com_stability
        #     failed at margin -10.6mm. Enlarge LENGTH here.
        total_reach = sum(
            float(p.dimensions.get("length", 0) or 0)
            + float(p.dimensions.get("diameter", 0) or 0)
            for p in link_parts
        )
        com_forward_mm = total_reach * 0.40
        # Support polygon must extend past the COM by a 35mm stability margin
        # (LLM pose variance can shift COM ~20mm run-to-run, and the verifier
        # uses a slightly different mass model than this estimate), so base
        # LENGTH (solver Y, the reach direction) >= 2*(COM + margin).
        min_base_length = max(320.0, 2.0 * (com_forward_mm + 35.0))
        cur_length = float(base_part.dimensions.get("length", 0) or 0)
        if 0 < cur_length < min_base_length:
            base_part.dimensions["length"] = min_base_length
            logger.info(
                "Sanitizer: enlarged base_plate '%s' length %.0f → %.0fmm "
                "so the support polygon covers the arm COM (forward≈%.0fmm "
                "along Y, need length≥%.0fmm)",
                base_part.name, cur_length, min_base_length,
                com_forward_mm, min_base_length,
            )

    # Clamp arm revolute joint ranges to a physically-reasonable workspace
    # so the e2e motion-collision sweep does not articulate into a
    # self-collision extreme. A ±180° base yaw or ±150° elbow is a servo
    # spec limit, not a usable workspace — at those extremes the arm hits
    # the chassis or the other arm (穿模). Real robots ship with narrower
    # software limits than the raw servo range.
    #
    # **Design lesson (2026-07-01)**: a previous attempt used an analytic
    # geometric limit (arcsin(H/L)) to prevent the arm reaching below the
    # base plate.  This was WRONG — the arm extends FORWARD (−Y) from the
    # base, so the gripper reaching low Z is in front of the base (Y=−269mm,
    # base Y=[−75,75]), NOT inside it.  FCL confirmed 0 collisions across
    # the FULL range [-90,+90].  The "穿模" the user saw was the arm passing
    # through the FLOOR (no ground in the static render), not through the
    # base.  The correct collision authority is the FCL mesh check in the
    # e2e motion-collision sweep — NOT a Z-height heuristic.  Numeric caps
    # remain as a coarse sanity bound; the FCL sweep is the real gate.
    #
    # Base yaw (axis=z) is clamped ASYMMETRICALLY: the home angle's sign
    # sets the allowed direction, and the range must not cross 0° (the
    # midline).
    _ARM_PITCH_CAP = 90.0   # forward (downward reach) — generous, FCL gates actual collision
    _ARM_PITCH_BACK_CAP = 45.0  # backward (folds toward base) — tighter, avoids self-collision
    _ARM_YAW_CAP = 90.0

    # --- Humanoid-topology detection ---
    # A humanoid (pelvis + leg chain + arm chain) has a fundamentally
    # different self-collision structure than a fixed-base arm: the arm's
    # shoulder swings through an arc that intersects the wide pelvis and
    # the leg chain, which does not exist on a fixed base. The generic
    # ±90° forward pitch cap (calibrated on fixed-base arms) lets the
    # humanoid shoulder reach +90° where the upper arm clips into the
    # pelvis/hip — observed on humanoid_2leg_2arm (3 motion-collision
    # failures on left/right shoulder_pitch). Detect the topology and
    # apply a tighter forward cap so the arm cannot swing into the body.
    # This is a topology-aware cap, not a model patch: it keys on the
    # PRESENCE of a pelvis + leg chain (the structural cause), not on
    # the joint name.
    part_names = {p.name.lower() for p in assembly.parts}
    has_pelvis = any("pelvis" in n or "hip" in n for n in part_names)
    has_leg_chain = any(
        "thigh" in n or "shin" in n or "knee" in n for n in part_names
    )
    is_humanoid_topology = has_pelvis and has_leg_chain
    # Humanoid shoulder can fold back (negative pitch, arm down) but the
    # forward (positive) extreme must clear the pelvis: cap at +60° so the
    # upper arm stays lateral rather than swinging down into the hips.
    _HUMANOID_PITCH_FWD_CAP = 60.0
    # A wrist-roll joint near the end-effector rarely needs ±180°; full
    # rotation lets the gripper cable/housing clip the forearm. Cap at
    # ±120° (a generous tool-roll range that still clears the forearm).
    _WRIST_ROLL_CAP = 120.0

    for j in revolute_joints:
        # --- Wrist-roll clamp (topology-independent) ---
        # Apply BEFORE the arm-pitch/yaw branch so axis=y wrist joints are
        # bounded even though they're skipped by _is_arm_pitch_joint.
        if (
            j.type == "revolute"
            and j.axis == "y"
            and j.range_deg
            and ("wrist" in j.child.lower() or "roll" in j.child.lower())
        ):
            lo, hi = float(j.range_deg[0]), float(j.range_deg[1])
            new_lo = max(lo, -_WRIST_ROLL_CAP)
            new_hi = min(hi, _WRIST_ROLL_CAP)
            if new_hi - new_lo >= 10.0 and (new_lo, new_hi) != (lo, hi):
                logger.info(
                    "Sanitizer: clamped wrist-roll '%s' [%.0f, %.0f] → "
                    "[%.0f, %.0f] (avoid forearm self-collision at ±180°)",
                    j.child, lo, hi, new_lo, new_hi,
                )
                j.range_deg = (new_lo, new_hi)
            continue  # wrist-roll handled; don't also run pitch/yaw logic

        if not _is_arm_pitch_joint(j) and j.axis != "z":
            continue
        if not j.range_deg:
            continue
        lo, hi = float(j.range_deg[0]), float(j.range_deg[1])
        home = float(injected.get(j.child, 0.0) or 0.0)
        if j.axis == "z":
            # Base yaw: keep on the home side of the midline.
            if home >= 0:
                new_lo, new_hi = max(lo, 0.0), min(hi, _ARM_YAW_CAP)
            else:
                new_lo, new_hi = max(lo, -_ARM_YAW_CAP), min(hi, 0.0)
        else:
            # Pitch: asymmetric numeric cap. Forward (home sign) is generous
            # (±90°) so the arm can reach down to pick up objects. Backward
            # (folds toward base, opposite sign) is tighter (±45°) to avoid
            # the arm folding back into its own servo stack.  The real
            # collision gate is the FCL motion-collision sweep downstream.
            #
            # Humanoid override: on a humanoid torso+pelvis, the forward
            # pitch extreme sweeps the upper arm through the hip/pelvis
            # volume (a self-collision absent on fixed-base arms). Tighten
            # the forward cap so the arm stays lateral to the body.
            fwd_cap = (
                _HUMANOID_PITCH_FWD_CAP if is_humanoid_topology
                else _ARM_PITCH_CAP
            )
            if home >= 0:
                new_lo, new_hi = max(lo, -_ARM_PITCH_BACK_CAP), min(hi, fwd_cap)
            else:
                new_lo, new_hi = max(lo, -fwd_cap), min(hi, _ARM_PITCH_BACK_CAP)
        if new_hi - new_lo < 10.0:
            continue  # range already tiny — don't collapse it
        if (new_lo, new_hi) != (lo, hi):
            topology_note = (
                "humanoid fwd-cap" if (is_humanoid_topology and j.axis != "z")
                else "yaw midline-safe" if j.axis == "z"
                else f"cap ±{_ARM_PITCH_CAP:.0f}° fwd/±{_ARM_PITCH_BACK_CAP:.0f}° back"
            )
            logger.info(
                "Sanitizer: clamped arm joint '%s' range [%.0f, %.0f] → "
                "[%.0f, %.0f] (home %.0f, %s, avoid workspace-extreme "
                "self-collision)",
                j.child, lo, hi, new_lo, new_hi, home, topology_note,
            )
            j.range_deg = (new_lo, new_hi)

    # --- Home-within-range guarantee ---
    # After all clamping, every revolute joint's home angle MUST be inside
    # its range_deg.  If home is outside, MuJoCo's PD controller yanks the
    # arm to the nearest limit on step 1, causing "fly" artifacts and
    # physics instability.  Clamp home into range as a final safety net.
    for j in assembly.joints:
        if j.type != "revolute" or not j.range_deg:
            continue
        lo, hi = float(j.range_deg[0]), float(j.range_deg[1])
        home = injected.get(j.child)
        if home is None:
            continue
        home = float(home)
        if home < lo:
            injected[j.child] = lo
            assembly.default_angles[j.child] = lo
        elif home > hi:
            injected[j.child] = hi
            assembly.default_angles[j.child] = hi

    return assembly


def _validate_assembly(assembly: Assembly) -> None:
    """Validate an Assembly for basic correctness.

    Raises RuntimeError for critical issues, logs warnings for minor ones.
    """
    if not assembly.parts:
        raise RuntimeError("Assembly has no parts")

    part_names = {p.name for p in assembly.parts}

    # Check all joints reference existing parts
    for i, joint in enumerate(assembly.joints):
        if joint.parent not in part_names:
            raise RuntimeError(
                f"Joint #{i}: parent '{joint.parent}' not in parts list"
            )
        if joint.child not in part_names:
            raise RuntimeError(
                f"Joint #{i}: child '{joint.child}' not in parts list"
            )

    # Check joint.type is valid (LLM sometimes hallucinates 'rotary' etc.)
    _VALID_JOINT_TYPES = {"fixed", "revolute", "prismatic", "continuous"}
    for i, joint in enumerate(assembly.joints):
        if joint.type not in _VALID_JOINT_TYPES:
            raise RuntimeError(
                f"Joint #{i} ('{joint.description}'): invalid type "
                f"'{joint.type}'. Must be one of {sorted(_VALID_JOINT_TYPES)}"
            )

    # Check range_deg well-formed for movable joints
    for i, joint in enumerate(assembly.joints):
        if joint.type in ("revolute", "continuous", "prismatic"):
            if not joint.range_deg or len(joint.range_deg) != 2:
                raise RuntimeError(
                    f"Joint #{i} ('{joint.description}'): range_deg missing "
                    f"or not a 2-tuple"
                )
            lo, hi = joint.range_deg
            if lo >= hi:
                raise RuntimeError(
                    f"Joint #{i} ('{joint.description}'): range_deg "
                    f"({lo}, {hi}) invalid, min must be < max"
                )
            if abs(lo) > 360 or abs(hi) > 360:
                raise RuntimeError(
                    f"Joint #{i} ('{joint.description}'): range_deg "
                    f"({lo}, {hi}) exceeds +/-360 degrees"
                )

    # Check all parts are connected (reachable from root via joints)
    # Auto-fix: connect any orphaned parts to the nearest reachable parent
    if assembly.joints:
        _ensure_connected(assembly, part_names)

    # DOF-completeness: when the assembly name carries an explicit "<N>dof"
    # prefix (the Architect names arms like "6dof_industrial_ball_wrist_arm"),
    # verify the generated revolute-joint count matches. A recurring failure
    # (6dof_arm: 4/9 historical runs) is the LLM merging two wrist joints
    # into one housing, producing N-1 revolute joints for an N-DOF request.
    # Without this check the under-DOF assembly passed silently (joint_count
    # only checks a minimum). Raise so the Fixer regenerates with the missing
    # joint called out explicitly.
    #
    # Only revolute/continuous joints count as arm-pose DOF — prismatic joints
    # are gripper fingers (a parallel-jaw gripper's linear motion is not part
    # of the arm's kinematic DOF), and mimic joints are coupled (never a DOF).
    _dof_m = re.search(r"(\d+)\s*dof", assembly.name, re.IGNORECASE)
    if _dof_m:
        expected_dof = int(_dof_m.group(1))
        # Count arm-pose revolute joints, EXCLUDING wheels (a wheel is a
        # revolute locomotion joint, not a manipulation DOF) and gripper
        # prismatic fingers (linear gripper motion, not arm pose).
        _WHEEL_KEYWORDS = ("wheel", "轮")
        actual_dof = sum(
            1 for j in assembly.joints
            if j.type in ("revolute", "continuous")
            and not any(kw in (j.child or "").lower() for kw in _WHEEL_KEYWORDS)
        )
        if actual_dof < expected_dof:
            raise RuntimeError(
                f"DOF mismatch: assembly '{assembly.name}' requests "
                f"{expected_dof} DOF but only {actual_dof} revolute joints "
                f"found. The LLM likely merged two joints into one housing "
                f"(a recurring 6dof failure) — regenerate with "
                f"{expected_dof} distinct revolute joints."
            )

    # Check dimensions
    for part in assembly.parts:
        if not part.dimensions:
            logger.warning("Part '%s' has no dimensions", part.name)
        else:
            for key, val in part.dimensions.items():
                if val <= 0:
                    raise RuntimeError(
                        f"Part '{part.name}' dimension '{key}' = {val} "
                        f"(must be > 0)"
                    )

    # Check anchor-dimension compatibility
    _parts_by_name = {p.name: p for p in assembly.parts}
    for joint in assembly.joints:
        for part_name, anchor in [(joint.parent, joint.parent_anchor), (joint.child, joint.child_anchor)]:
            part = _parts_by_name.get(part_name)
            if part and anchor in ("front", "back", "left", "right"):
                dim_keys = ANCHOR_DIM_KEYS.get(anchor, [])
                has_match = any(k in part.dimensions for k in dim_keys)
                if not has_match and not any(
                    k in part.dimensions for k in ("diameter", "outer_diameter")
                ):
                    logger.warning(
                        "Joint '%s': part '%s' uses anchor '%s' but has no matching dimensions %s",
                        joint.description, part_name, anchor, dim_keys,
                    )

    logger.info(
        "Assembly '%s' validated: %d parts, %d joints",
        assembly.name, len(assembly.parts), len(assembly.joints),
    )

    # DOF sanity check (warning only, non-blocking)
    revolute_count = sum(1 for j in assembly.joints if j.type == "revolute")
    if revolute_count == 0:
        logger.warning("Assembly has 0 revolute DOF — all joints fixed")
    elif revolute_count > 8:
        logger.warning(
            "Assembly has %d revolute DOF — verify design intent", revolute_count,
        )

    # P0-4: check joint offsets against the 3.0× max-dimension bound the
    # solver uses in _clamp_child_offset.  Without this, extreme offsets
    # pass validation, pass the VLM loop (because default_angles bend the
    # arm and keep the offset within bounds), then crash the all-zero
    # home-pose solve at export time — killing the entire engineering
    # package output.  Raising here feeds the error back to the LLM via
    # the VLM retry loop so it can regenerate with corrected offsets.
    for i, joint in enumerate(assembly.joints):
        if not joint.offset:
            continue
        parent_part = _parts_by_name.get(joint.parent)
        child_part = _parts_by_name.get(joint.child)
        if not parent_part or not child_part:
            continue
        parent_max = max(parent_part.dimensions.values()) if parent_part.dimensions else 0
        child_max = max(child_part.dimensions.values()) if child_part.dimensions else 0
        if parent_max < 1 and child_max < 1:
            continue
        offset_mag = math.sqrt(sum(c ** 2 for c in joint.offset))
        max_allowed = 3.0 * (parent_max + child_max)
        if max_allowed < 1.0:
            max_allowed = 500.0
        if offset_mag > max_allowed:
            raise RuntimeError(
                f"Joint #{i} ('{joint.description}'): offset "
                f"{joint.offset} (magnitude {offset_mag:.1f}mm) exceeds "
                f"3.0× (parent+child) = {max_allowed:.1f}mm. "
                f"The offset is physically inconsistent with the part "
                f"dimensions; reduce the offset or increase part sizes."
            )

