"""Deterministic dual-arm assembly composer (ArtiCAD Assembly Agent).

Replaces the LLM-driven "generate the whole dual-arm robot in one prompt"
approach — which produced mis-placed arms, wheels re-parented onto the arm
chain, and parts flung 600mm from centre — with a deterministic composition
that the LLM cannot get wrong because it involves no LLM.

This mirrors ArtiCAD's insight (arXiv 2604.10992 §4.4): once the Design
Agent fixes the topology and connectors, *assembly reduces to deterministic
frame alignment*. The chassis and each arm are generated correctly in
isolation (by ``mobile_base_gen`` and ``arm_topology`` respectively); this
module just bolts them together with SE(3) transforms.

Composition strategy (ArtiCAD "Derive Mechanism"):
  * One arm is the source (from ``arm_topology.build_arm_example``).
  * The left arm = source with all names prefixed ``arm_l_``.
  * The right arm = source with names prefixed ``arm_r_`` and its lateral
    offsets mirrored (Y → -Y), so the two arms face outward symmetrically.
  * Each arm's root is re-parented onto the chassis ``top_plate`` via a
    ``fixed`` joint with a lateral offset (±mount_offset_mm), grouped under
    ``distribution_group="arms"`` so the solver knows they are a pair.

No LLM, no VLM, no guessing — just geometry.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any


def _validate_arm_chassis_proportion(chassis: dict, arm: dict) -> None:
    """Raise if an arm is out of proportion with the chassis it mounts on.

    Real wheeled dual-arm robots (HSR, TIAGo++) keep the arm base ≤ ~50% of the
    deck area and the arm reach within ~1.5× the chassis diagonal (see
    docs/references/wheeled_dual_arm_proportions.md). A desktop-scale arm
    (200×150 base) bolted onto a ~120mm chassis violates both. We raise so the
    caller regenerates the arm at mobile scale rather than silently shipping a
    mis-proportioned robot or (worse) rescaling real servos.

    Structural parts (links, base plates) are legitimately sizeable by design,
    so the thresholds here are loose enough that a correctly-profiled mobile arm
    passes, while a desktop-profile arm on a small chassis fails loudly.
    """
    # The deck is whatever _arm_mount.deck names (base_plate for Husky-style
    # chassis; top_plate for legacy). Fall back to either name so the guard
    # works across both structures.
    deck_name = chassis.get("_arm_mount", {}).get("deck", "top_plate")
    deck_part = next(
        (p for p in chassis.get("parts", [])
         if p["name"] in (deck_name, "top_plate", "base_plate")),
        None,
    )
    arm_base = next(
        (p for p in arm.get("parts", []) if p["name"] == "base_plate"), None,
    )
    if deck_part is None or arm_base is None:
        return  # can't check without both; let the solver surface it

    dd = deck_part["dimensions"]
    ad = arm_base["dimensions"]
    deck_area = dd.get("length", 0) * dd.get("width", 0)
    arm_area = ad.get("length", 0) * ad.get("width", 0)
    if deck_area <= 0:
        return

    # Arm base should not exceed ~60% of deck area (allows some overhang for a
    # correctly-sized mobile arm, fails a desktop arm that's bigger than deck).
    if arm_area > deck_area * 0.6:
        raise ValueError(
            f"Arm base ({ad.get('length')}×{ad.get('width')}={arm_area:.0f}mm²) "
            f"is larger than 60% of the deck area "
            f"({dd.get('length')}×{dd.get('width')}={deck_area:.0f}mm²). "
            f"This is a desktop-scale arm on a mobile chassis — regenerate the "
            f"arm with build_arm_example(n_dof, profile='mobile')."
        )

    # Arm reach (sum of link lengths) vs chassis diagonal. A real mobile arm
    # reaches 1-3× the chassis diagonal (HSR≈1.05; a 7-DOF arm with a wrist
    # naturally reaches more). The desktop profile reaches 5-8×, which is the
    # failure we catch. 4× is the loose bound: allows valid high-DOF mobile
    # arms, rejects desktop-scale arms bolted to a small chassis.
    link_lengths = [
        p["dimensions"].get("length", 0)
        for p in arm.get("parts", [])
        if p["name"].endswith("_link_0") or "_link_" in p["name"]
    ]
    arm_reach = sum(link_lengths)
    chassis_diag = math.hypot(dd.get("length", 0), dd.get("width", 0))
    if chassis_diag > 0 and arm_reach > chassis_diag * 4.0:
        raise ValueError(
            f"Arm reach ({arm_reach:.0f}mm over {len(link_lengths)} links) "
            f"exceeds 4× the chassis diagonal ({chassis_diag:.0f}mm). "
            f"Use profile='mobile' for shorter links."
        )


def compose_dual_arm_assembly(
    chassis_json: str,
    arm_json: str,
    arm_dof: int = 4,
    mount_offset_mm: float = 70.0,
    configure_avoidance: bool = True,
) -> str:
    """Bolt two mirrored copies of an arm onto a wheeled chassis.

    Args:
        chassis_json: output of ``mobile_base_gen.build_wheeled_base``.
            Must contain a ``top_plate`` part and an ``_arm_mount`` metadata
            block (the deck arms attach to and the lateral spread).
        arm_json: output of ``arm_topology.build_arm_example``. A standalone
            single arm rooted at its own ``base_plate``.
        arm_dof: the arm's DOF (used only for the assembly name/description).
        mount_offset_mm: left↔right spread of the two arm bases on the deck.
            Defaults to 70mm; overridden by the chassis ``_arm_mount.offset_mm``
            when present.

    Returns:
        A single assembly-JSON string: chassis parts + two prefixed,
        mirrored arm copies + mounting joints, forming one connected tree
        rooted at the chassis ``base_plate``.
    """
    chassis = json.loads(chassis_json)
    arm = json.loads(arm_json)

    # Proportion guard (AGENTS.md §1.2 + docs/references/
    # wheeled_dual_arm_proportions.md): a real wheeled dual-arm robot has the
    # arm base SMALLER than the deck and the arm reach within ~1.5× the chassis
    # footprint. Bolting a desktop-scale arm (200×150 base) onto a small chassis
    # produces arms bigger than the robot — the failure we are fixing. We RAISE
    # rather than silently rescale, because rescaling would be the prohibited
    # "shortcut" and the caller (the chassis expert) must regenerate the arm at
    # the correct mobile scale (profile="mobile").
    _validate_arm_chassis_proportion(chassis, arm)

    mount = chassis.get("_arm_mount", {})
    deck = mount.get("deck", "top_plate")
    spread = mount.get("offset_mm", mount_offset_mm)

    # Collect chassis parts/joints, but DROP the _arm_mount metadata key
    # (it's generator-internal, not part of the assembly schema).
    parts: list[dict[str, Any]] = [p for p in chassis["parts"]]
    joints: list[dict[str, Any]] = list(chassis["joints"])

    # The arm's own base_plate becomes an arm-specific mounting plate; we
    # prefix every arm part name and re-root the arm onto the chassis deck.
    arm_root = "base_plate"  # the arm's root part name
    # Arms mount on the TOP FACE of the chassis shell (chassis_body), raised
    # by deck_top_z so the arm base sits ON the shell top — clear of the
    # shell volume. Mounting on the deck beneath the shell had the shell
    # enclose the arm bases (the穿模 defect). deck_top_z is the shell's
    # top-face height above the deck CENTER; the arm base half-height is
    # added so the arm base bottom = shell top face.
    arm_base_h = 6.0  # arm base_plate height (mobile profile)
    _deck_top_z = mount.get("deck_top_z", 0.0)
    _arm_z = _deck_top_z + arm_base_h / 2.0
    for side, sign in (("l", -1.0), ("r", 1.0)):
        prefix = f"arm_{side}_"
        arm_parts, arm_joints = _copy_arm_prefixed(arm, prefix, arm_root)
        parts.extend(arm_parts)

        # Re-root: the arm's (now prefixed) base_plate mounts on the deck.
        # Use center/center anchors so the solver applies NO face offset and
        # NO half-extent push-out — the arm base lands EXACTLY at deck_center
        # + the explicit offset. The Z offset raises the arm base to sit ON
        # the shell top face (deck_top_z), so it cannot pierce the shell.
        joints.append({
            "type": "fixed",
            "parent": deck,
            "child": prefix + arm_root,
            "parent_anchor": "center",
            "child_anchor": "center",
            "offset": [sign * spread, 0.0, _arm_z],
            "distribution_group": "arms",
            "no_distribute": True,
        })
        joints.extend(arm_joints)

    # Merge default_angles (arm joints keep their bend; prefix the keys).
    merged_angles: dict[str, float] = {}
    for side in ("l", "r"):
        prefix = f"arm_{side}_"
        for k, v in arm.get("default_angles", {}).items():
            merged_angles[prefix + k] = v
    # chassis has no revolute angles (wheels spin freely, no home pose).

    # --- Collision-aware dual-arm configuration (multi-arm avoidance) ---
    # A dual-arm robot with two identical arms at the same default pose has
    # OVERLAPPING workspaces — both arms splay forward into the same -Y
    # region and collide on any yaw toward each other (the project
    # expectation: "多机械臂的话自主避障").  The physically-correct fix is
    # NOT to widen the chassis (that hides the issue) but to configure the
    # arms into a collision-free COORDINATED pose, the way every real
    # dual-arm robot (PR2, TIAGo++, HSR) ships from the factory:
    #   1. Symmetric outward splay — left arm yawed toward -X, right toward
    #      +X, so the arms reach OUTWARD instead of stacking forward.
    #   2. Soft-limit each joint's range_deg to its collision-free reachable
    #      region — a joint's declared range must reflect what the robot can
    #      physically reach without self-collision, not the bare servo spec
    #      (a yaw that hits the other arm at +90° is not a usable +90°).
    # This is generic (adapts to any arm length / chassis width via the
    # collision checker's real feedback), never hard-codes angles.
    #
    # ``configure_avoidance`` defaults True (production delivers a
    # collision-free dual-arm), but unit tests of composition STRUCTURE
    # pass False to avoid the per-joint collision search (which is
    # intentionally heavy — it runs the solver+FCL checker many times).
    if configure_avoidance:
        _configure_collision_aware_dual_arm(parts, joints, merged_angles)

    assembly = {
        "name": f"dual_arm_{arm_dof}dof_{chassis.get('name', 'wheeled')}",
        "description": f"双{arm_dof}自由度臂轮式机器人",
        "default_angles": merged_angles,
        "parts": parts,
        "joints": joints,
    }
    return json.dumps(assembly, ensure_ascii=False, indent=2)


def _copy_arm_prefixed(
    arm: dict[str, Any], prefix: str, arm_root: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deep-copy an arm's parts/joints with every name prefixed.

    The arm's internal parent/child references are rewritten to the prefixed
    names so the copied arm is a self-contained kinematic subtree. External
    re-rooting (onto the chassis deck) is done by the caller.

    Returns (prefixed_parts, prefixed_joints). The arm_root part is kept
    (it becomes the arm's mounting plate on the deck).
    """
    parts = []
    name_map: dict[str, str] = {}
    for p in arm["parts"]:
        new_p = copy.deepcopy(p)
        new_name = prefix + p["name"]
        name_map[p["name"]] = new_name
        new_p["name"] = new_name
        parts.append(new_p)

    joints = []
    for j in arm["joints"]:
        new_j = copy.deepcopy(j)
        new_j["parent"] = name_map.get(j["parent"], j["parent"])
        new_j["child"] = name_map.get(j["child"], j["child"])
        # NOTE: we do NOT mirror the arm's internal X offsets here.
        # The arm is mounted on the chassis as a rigid, un-mirrored subtree:
        # the lateral placement (left vs right) is handled entirely by the
        # mounting joint's ±spread offset (see compose_dual_arm_assembly).
        # Mirroring internal offsets used to collapse the right arm's gripper
        # — the finger prismatic offsets (±16mm) flipped sign, driving both
        # fingers inward to a 12.4mm overlap instead of a 37.6mm gap. Real
        # dual-arm robots (TIAGo++, HSR) have both grippers oriented the same
        # way (grippers open forward), not mirrored.
        # Keep mimic_joint references consistent within the copied arm.
        if "mimic_joint" in new_j and new_j["mimic_joint"]:
            new_j["mimic_joint"] = name_map.get(new_j["mimic_joint"], new_j["mimic_joint"])
        joints.append(new_j)

    return parts, joints


# ---------------------------------------------------------------------------
# Multi-arm collision-aware configuration (the "自主避障" deliverable)
# ---------------------------------------------------------------------------


def _to_assembly(parts: list[dict], joints: list[dict],
                 default_angles: dict[str, float]) -> "Assembly":
    """Build a mechanics.Assembly from raw dicts (for solver/collision)."""
    from ..knowledge.mechanics import Assembly, Joint, Part
    jf = ("type", "parent", "child", "axis", "parent_anchor", "child_anchor",
          "offset", "range_deg", "no_distribute", "distribution_group",
          "mimic_joint", "mimic_multiplier", "mimic_offset")
    return Assembly(
        name="dual_arm_config",
        parts=[Part(name=p["name"], category=p.get("category", ""),
                    description=p.get("description", ""),
                    material=p.get("material", "PLA"),
                    dimensions=p["dimensions"]) for p in parts],
        joints=[Joint(**{k: v for k, v in j.items() if k in jf}) for j in joints],
        default_angles=default_angles,
    )


def _cross_arm_collision_count(
    parts: list[dict], joints: list[dict], default_angles: dict[str, float],
) -> int:
    """Count arm_l↔arm_r collisions at the given pose (static check).

    Returns 0 when the two arms do not interpenetrate.  Uses the project's
    trimesh+FCL checker; degrades to "no info" (returns 0) if FCL is absent
    so a missing optional dep never blocks composition.
    """
    try:
        from ..tools.mesh_collision import MeshCollisionChecker, HAS_FCL
        from ..tools.assembly_solver import AssemblySolver
    except ImportError:
        return 0
    if not HAS_FCL:
        return 0
    checker = MeshCollisionChecker()
    asm = _to_assembly(parts, joints, default_angles)
    pos = AssemblySolver(asm).solve()
    res = checker.check_assembly_collisions(
        asm, pos, skip_adjacent=True, min_penetration_mm=0.5,
    )
    n = 0
    for c in res.pairs:
        if not c.is_collision:
            continue
        a, b = c.part_a, c.part_b
        if ("arm_l" in a and "arm_r" in b) or ("arm_r" in a and "arm_l" in b):
            n += 1
    return n


def _any_collision_count(
    parts: list[dict], joints: list[dict], default_angles: dict[str, float],
) -> int:
    """Count ALL non-adjacent collisions at the given pose (static check).

    Unlike ``_cross_arm_collision_count`` (which counts only arm_l↔arm_r
    overlaps), this counts EVERY collision the project checker would —
    including an arm colliding with itself, the chassis, or the battery.
    Used by Phase 2's per-joint soft-limit search: a joint limit must avoid
    ALL self-collisions, not just inter-arm ones.  (An elbow that rams its
    own shoulder at +150° is just as unusable as one that rams the other
    arm.)  Degrades to 0 when FCL is absent.
    """
    try:
        from ..tools.mesh_collision import MeshCollisionChecker, HAS_FCL
        from ..tools.assembly_solver import AssemblySolver
    except ImportError:
        return 0
    if not HAS_FCL:
        return 0
    checker = MeshCollisionChecker()
    asm = _to_assembly(parts, joints, default_angles)
    pos = AssemblySolver(asm).solve()
    res = checker.check_assembly_collisions(
        asm, pos, skip_adjacent=True, min_penetration_mm=0.5,
    )
    return sum(1 for c in res.pairs if c.is_collision)


def _configure_collision_aware_dual_arm(
    parts: list[dict], joints: list[dict], default_angles: dict[str, float],
) -> None:
    """Configure a dual-arm assembly into a collision-free coordinated pose.

    Two-phase, generic (no hard-coded angles — driven by the real collision
    checker):

    Phase 1 — symmetric outward splay of the base yaw joints.  Searches
    increasing |yaw| offsets (left arm toward -X, right toward +X) for the
    smallest splay that yields ZERO static arm↔arm collision.  This mirrors
    how a real dual-arm robot is parked: arms splayed outward, not stacked.

    Phase 2 — per-joint soft-limit search.  For each revolute arm joint,
    binary-searches the collision-free sub-range of its declared range_deg
    and rewrites range_deg to that reachable region.  A joint whose servo
    spec says ±180° but which physically hits the other arm at +90° gets a
    soft limit matching what the robot can actually reach.  This is the
    mechanical+software limit every shipped robot enforces.

    Mutates *default_angles* (Phase 1) and each joint's ``range_deg``
    (Phase 2) in place.  Silent no-op (with a logged reason) when FCL is
    unavailable or no dual-arm revolute joints exist — composition must
    never fail because an optional dep is missing.
    """
    import logging
    log = logging.getLogger("lang3d.assembly_compose")

    # Identify the two arms' base-yaw (z-axis, top/bottom mount) joints.
    arm_yaws: dict[str, str] = {}  # side -> joint child name
    for j in joints:
        child = j.get("child", "")
        for side in ("l", "r"):
            if (child.startswith(f"arm_{side}_") and child.endswith("_base_yaw_servo")
                    and j.get("type") == "revolute" and j.get("axis") == "z"):
                arm_yaws[side] = child
    if len(arm_yaws) < 2:
        return  # not a dual-arm-zaw assembly; nothing to coordinate

    try:
        from ..tools.mesh_collision import HAS_FCL
    except ImportError:
        HAS_FCL = False
    if not HAS_FCL:
        log.info("dual-arm collision config skipped (python-fcl not installed)")
        return

    # ---- Phase 1: symmetric outward splay of base yaw ----
    # Search |yaw| from 30° up to 90° in 15° steps; take the first that
    # yields zero cross-arm collision.  Left arm yaws negative (toward -X,
    # outward), right arm yaws positive (toward +X, outward).
    chosen_splay: float | None = None
    for splay in (30.0, 45.0, 60.0, 75.0, 90.0):
        trial = dict(default_angles)
        trial[arm_yaws["l"]] = -splay
        trial[arm_yaws["r"]] = +splay
        if _cross_arm_collision_count(parts, joints, trial) == 0:
            chosen_splay = splay
            break
    if chosen_splay is None:
        log.warning(
            "dual-arm splay search found no collision-free pose up to 90°; "
            "arms may overlap at every park angle (consider wider spread)"
        )
        return
    default_angles[arm_yaws["l"]] = -chosen_splay
    default_angles[arm_yaws["r"]] = +chosen_splay
    log.info(
        "dual-arm collision-aware park pose: base_yaw splay ±%d° "
        "(left -%d°, right +%d°)", chosen_splay, chosen_splay, chosen_splay,
    )

    # ---- Phase 2: per-joint soft-limit search ----
    # Delegate to the generic collision-aware range clamper (shared with
    # the single-arm and post-solver pipeline path).
    _clamp_joint_ranges_to_collision_free(
        parts, joints, default_angles,
        joint_filter=lambda j: (
            j.get("child", "").startswith("arm_l_")
            or j.get("child", "").startswith("arm_r_")
        ),
    )


def _clamp_joint_ranges_to_collision_free(
    parts: list[dict],
    joints: list[dict],
    default_angles: dict[str, float],
    *,
    joint_filter=None,
) -> int:
    """Rewrite each revolute joint's ``range_deg`` to its collision-free subset.

    For every revolute joint passing *joint_filter* (default: all arm joints,
    i.e. non-wheel revolute joints), binary-search the maximal collision-free
    sub-range of its declared ``range_deg`` that contains the joint's home
    angle, and rewrite ``range_deg`` in place.  This is the
    mechanical+software limit every shipped robot enforces — a joint whose
    servo spec says ±180° but which rams its own base at +90° gets a soft
    limit matching what the arm can actually reach without interpenetration.

    Used by:
      - ``_configure_collision_aware_dual_arm`` Phase 2 (dual-arm, at compose)
      - ``AssemblyPipeline.run_solver`` (ALL arms, post-solve pre-export)

    Args:
        joint_filter: predicate ``(joint_dict) -> bool`` selecting which
            revolute joints to clamp.  Defaults to all revolute joints whose
            child is NOT a wheel/tire/motor (i.e. arm joints).

    Returns the number of joints whose range was narrowed.  Silent no-op
    (returns 0) when FCL is unavailable — clamping is a refinement, not a
    gate; generation must never fail because an optional dep is missing.
    """
    import logging
    log = logging.getLogger("lang3d.assembly_compose")

    try:
        from ..tools.mesh_collision import HAS_FCL
    except ImportError:
        HAS_FCL = False
    if not HAS_FCL:
        log.debug("collision-aware range clamp skipped (python-fcl not installed)")
        return 0

    if joint_filter is None:
        # Default: arm joints = revolute, not wheels/tires/motors.
        def joint_filter(j: dict) -> bool:
            if j.get("type") != "revolute":
                return False
            child = j.get("child", "").lower()
            return not any(
                child.startswith(p) for p in ("wheel_", "tire_", "motor_")
            )

    n_clamped = 0
    for j in joints:
        if not joint_filter(j):
            continue
        child = j["child"]
        lo, hi = j.get("range_deg", [-180.0, 180.0])
        if hi - lo < 1.0:
            continue
        home = default_angles.get(child, (lo + hi) / 2.0)
        # Binary-search the maximal collision-free [a, b] ⊆ [lo, hi] with
        # home ∈ [a, b].  We search each side independently: expand a
        # downward from home and b upward from home as long as the swept
        # midpoint stays collision-free.  This is conservative (a coarser
        # search than full range N-body) but matches the e2e joint-sweep.
        a = _expand_collision_free(
            parts, joints, default_angles, child, home, lo, step=-15.0,
            check_fn=_any_collision_count,
        )
        b = _expand_collision_free(
            parts, joints, default_angles, child, home, hi, step=+15.0,
            check_fn=_any_collision_count,
        )
        # ENDPOINT VALIDATION: the expand-from-home search checks angles at
        # 15° intervals from home, but the motion-collision sweep samples
        # the FINAL range uniformly (7 points incl. both endpoints). The
        # original lo/hi extremes (far from home) were never re-checked by
        # the clamper — if the LLM specified a colliding extreme (e.g.
        # shoulder at -90°), it survived into the output range and the
        # sweep then caught it (the most common motion_collision_sweep
        # failure). Verify the endpoints a and b are themselves collision-
        # free; if not, walk them inward until they are.
        check_fn = _any_collision_count
        while a < home:
            trial = dict(default_angles); trial[child] = a
            if check_fn(parts, joints, trial) == 0:
                break
            a += 15.0  # walk toward home until clear
        while b > home:
            trial = dict(default_angles); trial[child] = b
            if check_fn(parts, joints, trial) == 0:
                break
            b -= 15.0  # walk toward home until clear
        if (b - a) >= 10.0 and (b - a) < (hi - lo):
            old = j["range_deg"]
            j["range_deg"] = [round(a, 1), round(b, 1)]
            n_clamped += 1
            log.info(
                "collision-aware range clamp: %s [%.1f, %.1f] → [%.1f, %.1f]",
                child, old[0], old[1], a, b,
            )
    return n_clamped


def clamp_assembly_joint_ranges_collision_free(assembly, *, joint_filter=None) -> int:
    """Pipeline entry point: clamp an ``Assembly`` object's joint ranges in place.

    Thin adapter over :func:`_clamp_joint_ranges_to_collision_free` that converts
    the typed ``Assembly`` (Part/Joint dataclasses) into the list-of-dicts the
    compose-path helpers expect, runs the clamp, then writes the narrowed
    ``range_deg`` back onto the live Joint objects.  This is the single-arm /
    post-solver analogue of ``_configure_collision_aware_dual_arm`` Phase 2.

    Args:
        assembly: a ``knowledge.mechanics.Assembly`` with ``.parts`` (Part
            dataclasses), ``.joints`` (Joint dataclasses), ``.default_angles``.
        joint_filter: optional predicate on a joint-dict; defaults to all arm
            revolute joints (non-wheel/tire/motor).

    Returns the number of joints whose range was narrowed.  No-op when FCL is
    unavailable.
    """
    from dataclasses import asdict

    # Convert typed Parts/Joints to plain dicts for the compose-path helpers.
    parts_d = [asdict(p) for p in assembly.parts]
    joints_d = []
    for j in assembly.joints:
        jd = asdict(j)
        # range_deg is a tuple in the dataclass; helpers expect a list.
        if isinstance(jd.get("range_deg"), tuple):
            jd["range_deg"] = list(jd["range_deg"])
        # connection is an enum; drop fields the Joint constructor rejects.
        jd.pop("connection", None)
        joints_d.append(jd)
    da = dict(assembly.default_angles)

    n = _clamp_joint_ranges_to_collision_free(
        parts_d, joints_d, da, joint_filter=joint_filter,
    )

    # Write narrowed ranges back onto the live Joint objects.
    if n:
        by_child = {jd["child"]: jd for jd in joints_d if "range_deg" in jd}
        for j in assembly.joints:
            jd = by_child.get(j.child)
            if jd and jd["range_deg"] != list(j.range_deg):
                j.range_deg = tuple(jd["range_deg"])
    return n


def _expand_collision_free(
    parts: list[dict], joints: list[dict], default_angles: dict[str, float],
    joint_child: str, start: float, bound: float, step: float,
    check_fn=None,
) -> float:
    """Expand a joint's collision-free limit from *start* toward *bound*.

    Returns the furthest angle (multiple of |step|) between start and bound
    at which the assembly (with this joint set to that angle and all others
    at default) is collision-free according to *check_fn* (defaults to the
    ALL-collision counter — a joint limit must avoid every self-collision,
    not just inter-arm ones).  Moves in *step*-sized jumps (5°): coarse
    enough to stay fast, fine enough for a usable soft limit.
    """
    if check_fn is None:
        check_fn = _any_collision_count
    best = start
    ang = start + step
    while (step > 0 and ang <= bound) or (step < 0 and ang >= bound):
        trial = dict(default_angles)
        trial[joint_child] = ang
        if check_fn(parts, joints, trial) == 0:
            best = ang
            ang += step
        else:
            break
    return best
