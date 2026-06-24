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
from typing import Any


def compose_dual_arm_assembly(
    chassis_json: str,
    arm_json: str,
    arm_dof: int = 4,
    mount_offset_mm: float = 70.0,
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
    for side, sign in (("l", -1.0), ("r", 1.0)):
        prefix = f"arm_{side}_"
        arm_parts, arm_joints = _copy_arm_prefixed(arm, prefix, arm_root)
        parts.extend(arm_parts)

        # Re-root: the arm's (now prefixed) base_plate mounts on the deck.
        # fixed joint, top→bottom so the arm stands on the deck; lateral
        # offset ±spread along X (left/right). distribution_group="arms"
        # tells the solver this is a symmetric pair.
        joints.append({
            "type": "fixed",
            "parent": deck,
            "child": prefix + arm_root,
            "parent_anchor": "top",
            "child_anchor": "bottom",
            "offset": [sign * spread, 0.0, 0.0],
            "distribution_group": "arms",
        })
        joints.extend(arm_joints)

    # Merge default_angles (arm joints keep their bend; prefix the keys).
    merged_angles: dict[str, float] = {}
    for side in ("l", "r"):
        prefix = f"arm_{side}_"
        for k, v in arm.get("default_angles", {}).items():
            merged_angles[prefix + k] = v
    # chassis has no revolute angles (wheels spin freely, no home pose).

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
        # Mirror lateral offset for the right arm so the two arms face
        # outward symmetrically. The arm's internal X offsets (e.g. gripper
        # finger spread) flip sign on the right side.
        if prefix == "arm_r_" and "offset" in new_j and new_j["offset"]:
            off = list(new_j["offset"])
            off[0] = -off[0]
            new_j["offset"] = off
        # Keep mimic_joint references consistent within the copied arm.
        if "mimic_joint" in new_j and new_j["mimic_joint"]:
            new_j["mimic_joint"] = name_map.get(new_j["mimic_joint"], new_j["mimic_joint"])
        joints.append(new_j)

    return parts, joints
