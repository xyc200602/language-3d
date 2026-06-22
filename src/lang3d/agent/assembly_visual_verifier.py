"""Assembly-level visual verification with VLM closed-loop correction.

Implements the CADCodeVerify (ICLR 2025) iterative verification pattern:
solve assembly → render multi-angle screenshots → VLM evaluates visual
correctness → detect problems → generate correction feedback → re-solve
→ loop (max 3 rounds).
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..knowledge.mechanics import Assembly, Joint

logger = logging.getLogger(__name__)


class ProblemType(str, Enum):
    COLLISION = "collision"
    FLOATING = "floating"
    WRONG_ORIENTATION = "wrong_orientation"
    UNREASONABLE_LAYOUT = "unreasonable_layout"
    # New types added 2026-06-18 to capture the actual failure modes
    # observed in e2e VLM feedback (was: everything got mapped to
    # UNREASONABLE_LAYOUT or COLLISION, which made correction routing
    # too coarse to actually fix anything).
    GRIPPER_INVISIBLE = "gripper_invisible"      # "end effector is a solid block"
    FINGER_OVERLAP = "finger_overlap"            # "fingers overlap by 38x41x23mm"
    PLATE_OVERLAP = "plate_overlap"              # "base_plate and top_plate overlap 580mm"
    MISSING_PART = "missing_part"                # "no base plate" / "critical parts missing"


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class LayoutProblem:
    """A detected layout problem from VLM visual verification."""

    problem_type: ProblemType
    severity: Severity
    description: str
    affected_parts: list[str] = field(default_factory=list)
    suggestion: str = ""
    # Structured correction detail returned by VLM (or heuristic).
    # Keys vary by correction type (see apply_corrections).
    correction: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssemblyVisualVerificationResult:
    """Result of an assembly visual verification round."""

    passed: bool = False
    problems: list[LayoutProblem] = field(default_factory=list)
    vlm_response: str = ""
    round_number: int = 0
    corrections_applied: list[dict[str, Any]] = field(default_factory=list)


def _build_assembly_prompt(
    assembly: Assembly,
    expected_layout: str = "",
    screenshot_paths: list[str] | None = None,
    positions: dict[str, dict] | None = None,
) -> str:
    """Build the VLM prompt for assembly-level verification.

    Follows the CADCodeVerify pattern: describe what you see, compare with
    expectations, structured output.
    """
    parts_summary = "\n".join(
        f"  - {p.name} ({p.category}): {p.description}"
        for p in assembly.parts
    )
    joints_summary = "\n".join(
        f"  - {j.parent} ← {j.type} → {j.child}"
        for j in assembly.joints
    )

    prompt = (
        "You are a 3D CAD assembly verification expert.\n\n"
        "Step 1: Describe what you see in the 3D viewport images.\n"
        "Look at the spatial arrangement of all parts carefully.\n\n"
        "Step 2: Check for the following assembly problems:\n"
        "  - **Collision**: Parts overlapping or intersecting\n"
        "  - **Floating**: Parts not connected to anything (unsupported)\n"
        "  - **Wrong orientation**: Parts rotated incorrectly (e.g. wheels should be vertical cylinders, not horizontal)\n"
        "  - **Unreasonable layout**: Parts too far apart or in illogical positions\n\n"
        "Step 3: Compare with the expected layout description.\n\n"
        f"Assembly: {assembly.name} ({len(assembly.parts)} parts)\n"
        f"Parts:\n{parts_summary}\n\n"
        f"Joints:\n{joints_summary}\n\n"
    )

    # Include current positions so the VLM can propose precise corrections
    if positions:
        pos_lines = []
        for name, pdata in positions.items():
            pos = pdata.get("position", [0, 0, 0])
            pos_lines.append(f"  {name}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        prompt += "Current solved positions (mm):\n" + "\n".join(pos_lines) + "\n\n"

    if expected_layout:
        prompt += f"Expected layout: {expected_layout}\n\n"

    prompt += (
        'Respond with a JSON object:\n'
        '{"passed": true/false, '
        '"problems": [{"type": "collision|floating|wrong_orientation|unreasonable_layout", '
        '"severity": "high|medium|low", '
        '"description": "...", '
        '"affected_parts": ["part1", "part2"], '
        '"suggestion": "...", '
        '"correction": {"type": "reposition|distribution_group", '
        '"joint_child": "part_name", '
        '"delta_xyz": [dx, dy, dz], '
        '"distribution_group": "group_name"}}], '
        '"overall_assessment": "..."}\n'
        '\n'
        'For "correction" use one of:\n'
        '  - reposition: {"type": "reposition", "joint_child": "<child part>", '
        '"delta_xyz": [dx_mm, dy_mm, dz_mm]}  — large position adjustment\n'
        '  - distribution_group: {"type": "distribution_group", "joint_child": "<child part>", '
        '"distribution_group": "<group>"}  — assign to a new sibling group\n'
    )
    return prompt


def _parse_layout_problems(vlm_response: str) -> list[LayoutProblem]:
    """Parse VLM JSON response into LayoutProblem list."""
    problems: list[LayoutProblem] = []

    # Try to extract JSON from response
    json_str = vlm_response
    # Try code block extraction
    if "```json" in json_str:
        start = json_str.index("```json") + 7
        end = json_str.index("```", start)
        json_str = json_str[start:end].strip()
    elif "```" in json_str:
        start = json_str.index("```") + 3
        end = json_str.index("```", start)
        json_str = json_str[start:end].strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Try to find { ... } in the response
        start = vlm_response.find("{")
        end = vlm_response.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(vlm_response[start:end])
            except json.JSONDecodeError:
                logger.warning("Failed to parse VLM response as JSON")
                return problems
        else:
            return problems

    raw_problems = data.get("problems", [])
    for rp in raw_problems:
        try:
            problems.append(LayoutProblem(
                problem_type=ProblemType(rp.get("type", "unreasonable_layout")),
                severity=Severity(rp.get("severity", "medium")),
                description=rp.get("description", ""),
                affected_parts=rp.get("affected_parts", []),
                suggestion=rp.get("suggestion", ""),
                correction=rp.get("correction", {}),
            ))
        except (ValueError, KeyError):
            continue

    return problems


# ---------------------------------------------------------------------------
# Free-text problem classifier (added 2026-06-18)
#
# The main loop historically passed `list[str]` of VLM problems straight into
# the LLM fix-prompt — which then regenerated the *whole* assembly.  This
# classifier turns those strings into structured LayoutProblem objects so
# `_generate_constraint_corrections` can route them to deterministic fixes
# (offset bumps, anchor flips, finger scaling) instead of an LLM redo.
# ---------------------------------------------------------------------------

import re as _re

_FINGER_NAMES = ("gripper_finger_left", "gripper_finger_right",
                 "finger_l", "finger_r", "finger_left", "finger_right")
_GRIPPER_KEYWORDS = (
    "gripper", "end effector", "end-effector", "夹爪", "抓手",
    "prongs", "parallel prongs", "two finger", "two separate",
)
_SOLID_BLOCK_KEYWORDS = (
    "solid block", "single chunky", "not a gripper",
    "no functional gripper", "missing gripper",
    "no two clearly separated", "no visible gap",
    "no gripper at", "no parallel prongs",
)
_PLATE_KEYWORDS = ("plate", "底盘", "底板", "chassis", "base_plate", "top_plate")
_OVERLAP_KEYWORDS = ("overlap", "intersect", "重叠", "交叉")
# Floating / disconnected complaints (added 2026-06-22, Plan B+C).
# These classify VLM "floating" reports into ProblemType.FLOATING so the
# targeted-fix path can route them.  NOTE: when the joint-graph
# connectivity check (assembly_generator._geometric_prevalidation Check 7)
# confirms the assembly is fully connected, these reports are filtered as
# false alarms BEFORE reaching classify_problems — so this route only
# fires for GENUINE floating (disconnected joint graph), where add_joint
# is the correct fix.
_FLOATING_KEYWORDS = (
    "floating", "mid-air", "mid air", "disconnected",
    "not connected", "no support", "no visible support",
    "no physical connection", "悬空", "悬浮", "未连接",
)


def classify_problem_text(text: str, assembly: Assembly) -> LayoutProblem:
    """Best-effort mapping of a free-text VLM problem to a structured LayoutProblem.

    The classifier is intentionally keyword-driven (no LLM call) so that the
    targeted-fix path is deterministic and cheap.  When no specific type
    matches, the problem is returned as ``UNREASONABLE_LAYOUT`` so the
    fallback LLM regeneration still kicks in.
    """
    t = (text or "").lower()
    affected: list[str] = []

    # --- Finger overlap (specific, high-value fix) ---
    # Example from production: "Parts 'gripper_finger_left' and
    # 'gripper_finger_right' overlap by 38x41x23mm in their rotated
    # world bounding boxes".
    if any(k in t for k in _OVERLAP_KEYWORDS):
        # Pull quoted part names
        quoted = _re.findall(r"['\"]([a-zA-Z_][\w]*)['\"]", text or "")
        # Pull a trailing "XxYxZmm" or "Xmm" dimension if present
        dim_match = _re.search(r"(\d+(?:\.\d+)?)\s*(?:x\s*\d+(?:\.\d+)?)*\s*mm", t)
        finger_quoted = [q for q in quoted if "finger" in q.lower()]
        plate_quoted = [q for q in quoted if "plate" in q.lower()]
        if finger_quoted:
            return LayoutProblem(
                problem_type=ProblemType.FINGER_OVERLAP,
                severity=Severity.HIGH,
                description=text,
                affected_parts=finger_quoted,
                correction={
                    "type": "finger_overlap",
                    "penetration_mm": float(dim_match.group(1)) if dim_match else 0.0,
                },
            )
        if plate_quoted:
            return LayoutProblem(
                problem_type=ProblemType.PLATE_OVERLAP,
                severity=Severity.HIGH,
                description=text,
                affected_parts=plate_quoted,
                correction={"type": "plate_overlap"},
            )
        # Generic overlap → collision
        return LayoutProblem(
            problem_type=ProblemType.COLLISION,
            severity=Severity.HIGH,
            description=text,
            affected_parts=quoted,
        )

    # --- Floating / disconnected (Plan B+C, 2026-06-22) ---
    # NOTE: this route only fires for GENUINE floating — when the
    # joint-graph connectivity check in _geometric_prevalidation (Check 7)
    # confirms the assembly is connected, floating reports are filtered
    # as false alarms in _vlm_check_assembly BEFORE reaching here.  So
    # by the time classify_problems sees a floating text, the assembly
    # really does have a disconnected component, and add_joint (the
    # FLOATING correction in _generate_constraint_corrections) is the
    # right fix.
    if any(k in t for k in _FLOATING_KEYWORDS):
        # Try to identify which part is floating by name-quotations.
        quoted = _re.findall(r"['\"]([a-zA-Z_][\w]*)['\"]", text or "")
        return LayoutProblem(
            problem_type=ProblemType.FLOATING,
            severity=Severity.HIGH,
            description=text,
            affected_parts=quoted,
            correction={"type": "floating"},
        )

    # --- Missing part ---
    if "no base plate" in t or "missing" in t or "absent" in t or "no base" in t:
        for p in _PLATE_KEYWORDS:
            if p in t:
                affected = [p for p in assembly.parts if "plate" in p.name.lower()
                            or "chassis" in p.name.lower()]
                return LayoutProblem(
                    problem_type=ProblemType.MISSING_PART,
                    severity=Severity.HIGH,
                    description=text,
                    affected_parts=[p.name for p in affected] or ["base_plate"],
                    correction={"type": "missing_part"},
                )

    # --- Gripper invisible / solid block ---
    if any(k in t for k in _SOLID_BLOCK_KEYWORDS) or (
        any(k in t for k in _GRIPPER_KEYWORDS) and
        any(k in t for k in ("prong", "finger", "gap", "爪"))
    ):
        fingers = [p.name for p in assembly.parts
                   if "finger" in p.name.lower()]
        return LayoutProblem(
            problem_type=ProblemType.GRIPPER_INVISIBLE,
            severity=Severity.HIGH,
            description=text,
            affected_parts=fingers,
            correction={"type": "gripper_invisible"},
        )

    # --- Generic fallback ---
    return LayoutProblem(
        problem_type=ProblemType.UNREASONABLE_LAYOUT,
        severity=Severity.MEDIUM,
        description=text,
    )


def classify_problems(
    problem_texts: list[str],
    assembly: Assembly,
) -> list[LayoutProblem]:
    """Convert a list of free-text problems to structured LayoutProblems."""
    return [classify_problem_text(t, assembly) for t in problem_texts]


def _generate_constraint_corrections(
    problems: list[LayoutProblem],
    assembly: Assembly,
) -> list[dict[str, Any]]:
    """Convert visual problems into constraint correction suggestions.

    Returns a list of correction dicts with keys:
    - joint_index: index in assembly.joints
    - correction_type: "offset" | "angle" | "attachment" | "reposition" | "distribution_group"
    - value: the corrected value
    - reason: why this correction was made
    """
    corrections: list[dict[str, Any]] = []

    for problem in problems:
        # --- VLM-suggested structured correction takes priority ---
        vlm_corr = problem.correction
        if vlm_corr and vlm_corr.get("type") in ("reposition", "distribution_group"):
            target_part = vlm_corr.get("joint_child", "")
            if not target_part and problem.affected_parts:
                target_part = problem.affected_parts[0]
            for i, joint in enumerate(assembly.joints):
                if joint.child == target_part:
                    entry: dict[str, Any] = {
                        "joint_index": i,
                        "correction_type": vlm_corr["type"],
                        "reason": f"{problem.problem_type.value}: {problem.description}",
                    }
                    if vlm_corr["type"] == "reposition":
                        entry["delta_xyz"] = vlm_corr.get("delta_xyz", [0, 0, 0])
                    elif vlm_corr["type"] == "distribution_group":
                        entry["distribution_group"] = vlm_corr.get("distribution_group", "")
                    corrections.append(entry)
                    break
            continue  # skip heuristic for this problem

        # --- Heuristic fallback ---
        if problem.problem_type == ProblemType.COLLISION:
            # Move parts apart by adjusting offset
            for part_name in problem.affected_parts:
                for i, joint in enumerate(assembly.joints):
                    if joint.child == part_name or joint.parent == part_name:
                        # Skip mimic-linked joints: nudging one finger's offset
                        # breaks the antagonist pair's synchronised motion
                        # (right finger mimics left; editing either desyncs).
                        is_mimic_linked = bool(getattr(joint, "mimic_joint", "")) or any(
                            getattr(j, "mimic_joint", "") == joint.child
                            for j in assembly.joints
                        )
                        if is_mimic_linked:
                            logger.info(
                                "Skipping collision correction for mimic-linked "
                                "joint %s->%s",
                                joint.parent, joint.child,
                            )
                            break
                        corrections.append({
                            "joint_index": i,
                            "correction_type": "offset",
                            "value": 5.0,  # mm offset to resolve collision
                            "reason": f"Collision: {problem.description}",
                        })
                        break

        elif problem.problem_type == ProblemType.FLOATING:
            # Suggest fixed joint for floating parts
            for part_name in problem.affected_parts:
                # Check if this part has any joint
                has_joint = any(
                    j.child == part_name or j.parent == part_name
                    for j in assembly.joints
                )
                if not has_joint:
                    corrections.append({
                        "part_name": part_name,
                        "correction_type": "add_joint",
                        "value": "fixed",
                        "reason": f"Floating part: {problem.description}",
                    })

        elif problem.problem_type == ProblemType.WRONG_ORIENTATION:
            for part_name in problem.affected_parts:
                for i, joint in enumerate(assembly.joints):
                    if joint.child == part_name:
                        corrections.append({
                            "joint_index": i,
                            "correction_type": "angle",
                            "value": 90.0,  # degree rotation
                            "reason": f"Wrong orientation: {problem.description}",
                        })
                        break

        # ------------------------------------------------------------------
        # New structured problem types (2026-06-18)
        # Each one routes to a *deterministic* correction that
        # `apply_corrections` knows how to apply — no LLM round-trip.
        # ------------------------------------------------------------------
        elif problem.problem_type == ProblemType.FINGER_OVERLAP:
            # Increase the joint offset of every finger part along its
            # distribution axis so the two fingertips separate.  The
            # exact direction (Y for left/right fingers) is handled by
            # `apply_corrections` → ctype="finger_spread".
            target_parts = problem.affected_parts or [
                j.child for j in assembly.joints
                if "finger" in j.child.lower()
            ]
            for part_name in target_parts:
                for i, joint in enumerate(assembly.joints):
                    if joint.child == part_name:
                        corrections.append({
                            "joint_index": i,
                            "correction_type": "finger_spread",
                            "penetration_mm": problem.correction.get(
                                "penetration_mm", 0.0),
                            "reason": f"Finger overlap: {problem.description}",
                        })
                        break

        elif problem.problem_type == ProblemType.GRIPPER_INVISIBLE:
            # "VLM can't see the gripper" is a RENDERING problem, not a
            # geometry problem.  The VLM loop already renders a dedicated
            # gripper_closeup view (render_assembly_from_positions with
            # gripper_closeup=True) that zooms in on the finger centroid.
            # If the close-up still can't resolve the fingers, scaling the
            # dimensions up will NOT help — it creates a runaway feedback
            # loop (60→96→153.6mm in two rounds, factor 1.6²) that makes
            # the fingers larger than the arm links, causing proportion
            # failures and wrist_link/finger intersections.
            #
            # Instead of mutating geometry, just record the problem so the
            # LLM-regeneration fallback path (which rewrites the assembly
            # JSON from scratch) can address the root cause — typically a
            # folded arm pose or genuinely fused fingers, neither of which
            # is fixed by scaling.
            logger.info(
                "GRIPPER_INVISIBLE reported but NOT scaling finger "
                "dimensions (would cause runaway growth). Close-up view "
                "and LLM regeneration will handle visibility: %s",
                problem.description[:120],
            )

        elif problem.problem_type == ProblemType.PLATE_OVERLAP:
            # Two stacked plates reported overlapping.  Push the child
            # plate up by the parent plate's height + a small gap.
            # `apply_corrections` looks up the parent's dim-height.
            parts_in_play = problem.affected_parts or []
            if len(parts_in_play) >= 2:
                child_name = parts_in_play[-1]
                for i, joint in enumerate(assembly.joints):
                    if joint.child == child_name:
                        corrections.append({
                            "joint_index": i,
                            "correction_type": "plate_z_separation",
                            "reason": f"Plate overlap: {problem.description}",
                        })
                        break

        elif problem.problem_type == ProblemType.MISSING_PART:
            # A part name was reported missing.  If it actually exists in
            # `assembly.parts` (the usual case — VLM mis-counts), no-op.
            # If it really is absent, flag for the LLM fallback path.
            wanted = problem.affected_parts or ["base_plate"]
            for w in wanted:
                already = any(p.name == w for p in assembly.parts)
                if not already:
                    corrections.append({
                        "correction_type": "rebuild_needed",
                        "reason": f"Truly missing part {w}: {problem.description}",
                    })

    return corrections


def _apply_finger_spread_to_joint(
    joint: Any, parts: list[Any]
) -> None:
    """Push one gripper finger joint apart from its twin along Y.

    Used by ``apply_corrections`` for the ``finger_spread`` correction type,
    and exposed at module level so it can be unit-tested in isolation (the
    VLM-loop bug was that this logic never actually ran — wrong offset
    component + an always-false idempotency guard).

    Convention (must match ``_normalize_gripper_fingers`` in
    assembly_generator.py): the two fingers separate on **Y** (the width
    axis), perpendicular to the bar length on X.

    The target offset is derived from geometry, not from the VLM-reported
    penetration (which caused runaway offsets on 2026-06-18): each finger
    centre needs ``finger_width + clearance`` from the origin so the inner
    faces clear each other.  Idempotent — a finger already past the target
    is left alone.
    """
    finger_width = 14.0  # fallback
    for p in parts:
        if p.name == joint.child:
            finger_width = float(p.dimensions.get("width", 14.0) or 14.0)
            break
    clearance = 6.0  # mm between inner faces when closed
    target_offset = min(finger_width + clearance, 40.0)
    sign = -1.0 if "left" in joint.child.lower() else 1.0
    cur = joint.offset or (0, 0, 0)
    # Idempotent: only push out if the current Y magnitude is still below
    # the target.  Symmetric for left (-) and right (+) fingers.
    if abs(cur[1]) < target_offset:
        joint.offset = (
            cur[0],
            sign * target_offset,
            cur[2],
        )
    joint.no_distribute = True


def apply_corrections(
    assembly: Assembly,
    corrections: list[dict[str, Any]],
) -> Assembly:
    """Apply constraint corrections to an assembly and return a modified copy.

    Does not mutate the original assembly.
    """
    # Deep copy assembly data
    import copy
    new_parts = [copy.deepcopy(p) for p in assembly.parts]
    new_joints = [copy.deepcopy(j) for j in assembly.joints]

    for corr in corrections:
        ctype = corr.get("correction_type", "")
        if "joint_index" in corr:
            idx = corr["joint_index"]
            if idx < len(new_joints):
                joint = new_joints[idx]
                if ctype == "offset":
                    # Prismatic joints (gripper fingers) have sanitizer-
                    # normalised offsets; VLM repositioning them inflates the
                    # offset and produces absurd URDF origins (4dof audit:
                    # finger offset grew to 330mm). Skip prismatic joints.
                    if joint.type == "prismatic":
                        continue
                    # Adjust position by adding Z offset
                    current_offset = joint.offset or (0, 0, 0)
                    joint.offset = (
                        current_offset[0],
                        current_offset[1],
                        current_offset[2] + corr["value"],
                    )
                elif ctype == "reposition":
                    # Same prismatic guard as offset corrections.
                    if joint.type == "prismatic":
                        continue
                    # Large position correction via delta_xyz applied to offset
                    current_offset = joint.offset or (0, 0, 0)
                    dx, dy, dz = corr.get("delta_xyz", [0, 0, 0])
                    joint.offset = (
                        current_offset[0] + dx,
                        current_offset[1] + dy,
                        current_offset[2] + dz,
                    )
                elif ctype == "distribution_group":
                    # Assign the joint to a new sibling group
                    joint.distribution_group = corr.get("distribution_group", "")
                    # If this joint was previously excluded from distribution,
                    # re-enable it so the new group takes effect.
                    if joint.no_distribute:
                        joint.no_distribute = False
                elif ctype == "finger_spread":
                    # Push the finger apart from its twin along Y.  Logic
                    # lives in _apply_finger_spread_to_joint so it can be
                    # unit-tested directly (the VLM-loop bug was precisely
                    # that this correction silently never fired).
                    _apply_finger_spread_to_joint(joint, new_parts)
                elif ctype == "plate_z_separation":
                    # Find the parent plate's height and push the child
                    # plate up to sit cleanly above it (plus a 2mm gap).
                    parent_name = joint.parent
                    parent_height = 0.0
                    for p in new_parts:
                        if p.name == parent_name:
                            parent_height = (
                                p.dimensions.get("height", 0.0)
                                or p.dimensions.get("thickness", 0.0)
                                or 0.0
                            )
                            break
                    needed = parent_height + 2.0
                    cur = joint.offset or (0, 0, 0)
                    joint.offset = (cur[0], cur[1], cur[2] + needed)
        elif ctype == "add_joint":
            # Add a fixed joint for floating parts
            part_name = corr["part_name"]
            # Find the nearest structural part to attach to
            parent = "base_plate"
            for p in new_parts:
                if p.category == "structural" and p.name != part_name:
                    parent = p.name
                    break
            new_joints.append(Joint(
                type="fixed",
                parent=parent,
                child=part_name,
                parent_anchor="top",
                child_anchor="bottom",
            ))
        elif ctype == "scale_part":
            # Scale a single part's dimensions by a multiplicative factor.
            # Used to make invisible gripper fingers visible to the VLM.
            part_name = corr.get("part_name", "")
            factor = float(corr.get("factor", 1.0))
            for p in new_parts:
                if p.name == part_name:
                    new_dims = {}
                    for k, v in p.dimensions.items():
                        try:
                            new_dims[k] = float(v) * factor
                        except (TypeError, ValueError):
                            new_dims[k] = v
                    p.dimensions = new_dims
                    break
        # "rebuild_needed" is a no-op here — it's a signal to the caller
        # that the LLM fallback path must be taken.

    return Assembly(
        name=assembly.name,
        parts=new_parts,
        joints=new_joints,
        description=assembly.description,
        default_angles=dict(assembly.default_angles),
        total_mass=assembly.total_mass,
        center_of_mass=assembly.center_of_mass,
        inertia_tensor=assembly.inertia_tensor,
    )


def verify_assembly_visual(
    assembly: Assembly,
    positions: dict[str, dict],
    model_backend: Any = None,
    expected_layout: str = "",
    max_iterations: int = 3,
    detail_level: str = "detailed",
) -> AssemblyVisualVerificationResult:
    """Main entry: run closed-loop assembly visual verification.

    For each iteration:
    1. Render assembly → screenshots (if FreeCAD + GUI available)
    2. Send screenshots + prompt to VLM
    3. Parse response into LayoutProblem list
    4. If passed or max iterations reached, return result
    5. Otherwise, generate corrections and re-solve

    Args:
        assembly: The assembly to verify
        positions: Solved positions from assembly solver
        model_backend: VLM model backend (GLMBackend, etc.)
        expected_layout: Text description of expected layout
        max_iterations: Maximum verification iterations (default 3)
        detail_level: VLM analysis detail level

    Returns:
        AssemblyVisualVerificationResult with final state
    """
    iteration_history: list[AssemblyVisualVerificationResult] = []
    current_assembly = assembly
    current_positions = positions

    for round_num in range(1, max_iterations + 1):
        logger.info("Assembly visual verification round %d/%d", round_num, max_iterations)

        # Build prompt (include current positions so VLM can propose precise fixes)
        prompt = _build_assembly_prompt(
            current_assembly, expected_layout,
            positions=current_positions,
        )

        # Attempt VLM verification if backend is available
        vlm_response = ""
        screenshots: list[str] = []

        if model_backend is not None:
            try:
                # Render assembly screenshots using matplotlib
                # Use a persistent temp dir that lives until after VLM calls
                render_tmpdir = tempfile.mkdtemp(prefix="vlm_verify_")
                try:
                    screenshots = _render_to_dir(
                        current_assembly, current_positions, render_tmpdir,
                    )
                    if screenshots:
                        # Send the isometric view to VLM (most informative)
                        # If detailed, also send front and top views
                        views_to_check = [screenshots[0]]  # isometric
                        if detail_level in ("detailed", "maximum") and len(screenshots) >= 3:
                            views_to_check = screenshots[:3]  # iso + front + top

                        all_responses = []
                        for ss_path in views_to_check:
                            resp = model_backend.vision(
                                ss_path,
                                prompt,
                                max_tokens=4096 if detail_level == "detailed" else 2048,
                            )
                            all_responses.append(resp)
                        vlm_response = "\n\n---\n\n".join(all_responses)
                    else:
                        # No screenshots — fall back to heuristic
                        logger.warning("No screenshots generated, using heuristic verification")
                        vlm_response = _heuristic_verification(current_assembly, current_positions)
                finally:
                    # Clean up temp files
                    import shutil
                    shutil.rmtree(render_tmpdir, ignore_errors=True)
            except Exception as e:
                logger.warning("VLM verification failed: %s", e)
                vlm_response = f'{{"passed": false, "problems": [], "overall_assessment": "VLM error: {e}"}}'
        else:
            # No backend: use heuristic-only verification
            vlm_response = _heuristic_verification(current_assembly, current_positions)

        # Parse problems
        problems = _parse_layout_problems(vlm_response)
        # Fail-closed: only pass when the VLM explicitly declares passed:true
        # AND no problems were parsed. Previously `passed = len(problems) == 0`
        # which mis-judged parse failures (empty list) as PASSED, bypassing
        # the quality gate whenever the VLM returned malformed output.
        has_explicit_pass = (
            '"passed": true' in vlm_response or '"passed":true' in vlm_response
        )
        passed = has_explicit_pass and len(problems) == 0

        result = AssemblyVisualVerificationResult(
            passed=passed,
            problems=problems,
            vlm_response=vlm_response,
            round_number=round_num,
        )
        iteration_history.append(result)

        if passed:
            logger.info("Assembly visual verification PASSED on round %d", round_num)
            result.corrections_applied = []
            return result

        # Generate and apply corrections for next iteration
        corrections = _generate_constraint_corrections(problems, current_assembly)
        result.corrections_applied = corrections

        if round_num < max_iterations and corrections:
            logger.info(
                "Applying %d corrections for round %d",
                len(corrections),
                round_num + 1,
            )
            current_assembly = apply_corrections(current_assembly, corrections)

            # Re-solve assembly positions
            try:
                from ..tools.assembly_solver import AssemblySolver
                solver = AssemblySolver(current_assembly)
                current_positions = solver.solve()
            except Exception as e:
                logger.warning("Re-solve failed: %s", e)
                break

    # Return last result (max iterations reached)
    if iteration_history:
        return iteration_history[-1]
    return AssemblyVisualVerificationResult(
        passed=False,
        vlm_response="No iterations completed",
        round_number=0,
    )


def _render_to_dir(
    assembly: Assembly,
    positions: dict[str, dict],
    output_dir: str,
) -> list[str]:
    """Render assembly using VTK offscreen (primary) or matplotlib (fallback).

    Tries VTK first for high-quality Phong-shaded renders with real STL
    geometry when available. Falls back to matplotlib if VTK is not installed.
    Saves PNGs to output_dir. Returns list of screenshot file paths.
    """
    # --- Try VTK first (high quality, headless, deterministic) ---
    try:
        return _render_vtk(assembly, positions, output_dir)
    except ImportError:
        pass
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("VTK render failed, falling back to matplotlib: %s", e)

    # --- Fallback: matplotlib (crude but always works) ---
    return _render_matplotlib(assembly, positions, output_dir)


def _render_vtk(
    assembly: Assembly,
    positions: dict[str, dict],
    output_dir: str,
) -> list[str]:
    """Render assembly using the shared VTK pipeline.

    Delegates to ``vtk_renderer.render_assembly_from_positions`` so this
    path gets every enhancement the main VLM-loop render gets: trimesh
    preview STLs (L-shaped fingers, not flat boxes), gripper close-up
    view, finger symmetrisation, fastener rendering, feature-edge
    extraction, and the arm-link tint system.  The previous in-line
    implementation built its own box/cylinder approximation and skipped
    all of these, so the export_package VLM path (triggered when
    verification_status != PASSED) always saw degraded geometry and the
    vision model reliably mis-classified the gripper as a solid block.
    """
    import os
    from ..tools.vtk_renderer import render_assembly_from_positions

    parts_dicts = [
        {
            "name": p.name,
            "dimensions": dict(p.dimensions),
            "category": getattr(p, "category", ""),
        }
        for p in assembly.parts
    ]
    stl_dir = getattr(assembly, "_stl_dir", "") or None
    os.makedirs(output_dir, exist_ok=True)
    return render_assembly_from_positions(
        parts=parts_dicts,
        positions=positions,
        output_dir=output_dir,
        stl_dir=stl_dir,
        joints=list(assembly.joints),
        views=["isometric", "front", "top", "right", "gripper_closeup"],
        gripper_closeup=True,
        width=1600,
        height=1200,
    )


def _render_matplotlib(
    assembly: Assembly,
    positions: dict[str, dict],
    output_dir: str,
) -> list[str]:
    """Render assembly using matplotlib (crude fallback)."""
    screenshots: list[str] = []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import numpy as np

        # Subsystem colors (RGBA)
        _SUBSYS_COLORS = {
            "arm_left": (0.85, 0.30, 0.20, 0.8),
            "arm_right": (0.20, 0.75, 0.30, 0.8),
            "ipc": (0.75, 0.55, 0.10, 0.8),
            "sensor_tower": (0.60, 0.20, 0.75, 0.8),
        }

        def _get_subsystem(name: str) -> str:
            if name.startswith("arm_l_"): return "arm_left"
            if name.startswith("arm_r_"): return "arm_right"
            if "ipc" in name: return "ipc"
            if name.startswith(("sensor_", "imu_", "lidar_", "camera_")): return "sensor_tower"
            return "chassis"

        def _box_faces(l, w, h, pos):
            x, y, z = pos
            v = np.array([
                [x-l/2,y-w/2,z],[x+l/2,y-w/2,z],[x+l/2,y+w/2,z],[x-l/2,y+w/2,z],
                [x-l/2,y-w/2,z+h],[x+l/2,y-w/2,z+h],[x+l/2,y+w/2,z+h],[x-l/2,y+w/2,z+h],
            ])
            return [
                [v[0],v[1],v[2],v[3]], [v[4],v[5],v[6],v[7]],
                [v[0],v[1],v[5],v[4]], [v[2],v[3],v[7],v[6]],
                [v[0],v[3],v[7],v[4]], [v[1],v[2],v[6],v[5]],
            ]

        def _cyl_faces(r, h, pos, n=12):
            x, y, z = pos
            bottom, top = [], []
            for i in range(n):
                a = 2*np.pi*i/n
                bx, by = x+r*np.cos(a), y+r*np.sin(a)
                bottom.append([bx,by,z])
                top.append([bx,by,z+h])
            faces = []
            for i in range(n):
                j = (i+1) % n
                faces.append([bottom[i],bottom[j],top[j],top[i]])
            faces.append(bottom)
            faces.append(top)
            return faces

        def _render_view(ax, elev, azim, title):
            ax.cla()
            ax.set_title(title, fontsize=14, fontweight='bold')
            for p in assembly.parts:
                dims = p.dimensions
                pos = positions.get(p.name, {}).get("position", [0,0,0])
                sub = _get_subsystem(p.name)
                color = _SUBSYS_COLORS.get(sub, (0.20, 0.40, 0.80, 0.7))

                if "diameter" in dims or "outer_diameter" in dims:
                    r = dims.get("diameter", dims.get("outer_diameter", 10)) / 2
                    h = dims.get("height", 10)
                    faces = _cyl_faces(r, h, pos)
                else:
                    l = dims.get("length", 10)
                    w = dims.get("width", 10)
                    h = dims.get("height", 10)
                    faces = _box_faces(l, w, h, pos)

                poly = Poly3DCollection(faces, alpha=0.75)
                poly.set_facecolor(color[:3])
                poly.set_edgecolor((0.2, 0.2, 0.2, 0.3))
                poly.set_linewidth(0.3)
                ax.add_collection3d(poly)

            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim(-200, 200)
            ax.set_ylim(-150, 150)
            ax.set_zlim(-80, 280)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')

        views = [
            (25, 45, "isometric"),
            (0, 0, "front"),
            (90, 0, "top"),
            (0, 270, "right"),
        ]

        for elev, azim, vname in views:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            _render_view(ax, elev, azim, f"Assembly - {vname.title()} View")
            img_path = str(Path(output_dir) / f"assembly_{vname}.png")
            fig.savefig(img_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            if Path(img_path).exists():
                    screenshots.append(img_path)

    except Exception as e:
        logger.warning("Matplotlib render failed: %s", e)

    return screenshots


def _heuristic_verification(
    assembly: Assembly,
    positions: dict[str, dict],
) -> str:
    """Run heuristic checks when VLM is not available.

    Checks for basic issues: floating parts, extreme positions, etc.
    """
    problems = []

    # Check for parts without positions
    positioned_parts = set(positions.keys())
    for part in assembly.parts:
        if part.name not in positioned_parts:
            problems.append({
                "type": "floating",
                "severity": "high",
                "description": f"Part '{part.name}' has no position",
                "affected_parts": [part.name],
                "suggestion": f"Add a joint for {part.name}",
            })

    # Check for extreme positions (> 500mm from origin)
    for name, pos_data in positions.items():
        pos = pos_data.get("position", [0, 0, 0])
        dist = sum(x**2 for x in pos) ** 0.5
        if dist > 500:
            problems.append({
                "type": "unreasonable_layout",
                "severity": "medium",
                "description": f"Part '{name}' is {dist:.0f}mm from origin",
                "affected_parts": [name],
                "suggestion": f"Review constraints for {name}",
            })

    # Check for parts at same position (potential collision)
    pos_map: dict[str, str] = {}
    for name, pos_data in positions.items():
        pos = pos_data.get("position", [0, 0, 0])
        key = f"{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}"
        if key in pos_map:
            problems.append({
                "type": "collision",
                "severity": "high",
                "description": f"'{name}' and '{pos_map[key]}' at same position",
                "affected_parts": [name, pos_map[key]],
                "suggestion": "Adjust joint offsets to separate parts",
            })
        else:
            pos_map[key] = name

    passed = len(problems) == 0
    return json.dumps({
        "passed": passed,
        "problems": problems,
        "overall_assessment": f"Heuristic check: {len(problems)} issues found",
    })
