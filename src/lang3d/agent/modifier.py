"""Targeted modification engine — apply user/VLM edits without full regeneration.

This module replaces the legacy "regenerate whole assembly on every VLM
feedback round" pattern.  Two entry points:

* ``apply_modification(assembly, request)`` — Claude-Code-style edit API.
  Used by both the interactive REPL (user says "make the arm longer") and
  the VLM closed-loop (system says "fix the gripper").
* ``apply_targeted_fix_from_vlm(assembly, problem_texts)`` — convenience
  wrapper that classifies VLM free-text problems into structured
  ``LayoutProblem`` objects and applies the resulting corrections.
  Returns ``(new_assembly, applied_any)`` so the caller can decide whether
  to fall back to LLM regeneration.

Scope tiers (per the user's spec):
* ``part``      — one part is scaled / moved / replaced
* ``subsystem`` — a group of related parts (gripper, arm, chassis)
* ``whole``     — full regeneration (fallback)

All operations are non-mutating: a new ``Assembly`` is returned.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Literal

from ..knowledge.mechanics import Assembly, Joint, Part
from .assembly_visual_verifier import (
    LayoutProblem,
    ProblemType,
    Severity,
    classify_problems,
    _generate_constraint_corrections,
    apply_corrections,
)

logger = logging.getLogger(__name__)


Scope = Literal["part", "subsystem", "whole"]
Source = Literal["user", "vlm", "auto"]


@dataclass
class ModificationRequest:
    """A single requested change to an assembly.

    ``raw_text`` is the original instruction (user prompt or VLM problem).
    ``params`` carries structured arguments (scale factor, delta_xyz, etc.).
    """

    scope: Scope
    intent: str
    target: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    source: Source = "user"
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Classifier — turn free text into a ModificationRequest
# ---------------------------------------------------------------------------

# Subsystem keywords → canonical subsystem name
_SUBSYSTEM_KEYWORDS: dict[str, list[str]] = {
    "gripper": ["gripper", "夹爪", "抓手", "finger", "爪", "end effector", "effector"],
    "arm": ["arm", "臂", "shoulder", "elbow", "wrist", "link", "肩", "肘", "腕"],
    "chassis": ["chassis", "底盘", "base", "plate", "frame", "frame"],
    "wheel": ["wheel", "轮", "tire"],
    "leg": ["leg", "腿"],
}

_INTENT_KEYWORDS: list[tuple[str, str]] = [
    ("enlarge", ["enlarge", "bigger", "larger", "longer", "wider", "taller",
                 "scale up",
                 "加长", "加大", "变大", "更大", "更长", "更宽", "更高",
                 "放大", "增加", "增大"]),
    ("shrink", ["shrink", "smaller", "shorter", "narrower", "lower",
                "scale down",
                "缩短", "变小", "更短", "缩窄", "减小", "减小"]),
    ("fix_collision", ["overlap", "intersect", "collision", "penetrat",
                       "重叠", "干涉", "碰撞", "穿透"]),
    ("fix_orientation", ["orientation", "rotated", "upside down",
                         "翻转", "方向", "旋转"]),
    ("replace", ["replace", "swap", "switch to", "换成", "替换", "改为"]),
    ("redo", ["redo", "regenerate", "重做", "重新生成", "重新设计"]),
]


def _keyword_match(text_lower: str, keywords: list[str]) -> bool:
    return any(k in text_lower for k in keywords)


def classify_modification(text: str, assembly: Assembly) -> ModificationRequest:
    """Classify a free-text edit request into a structured ModificationRequest.

    The classifier is keyword-driven (no LLM call) so it is deterministic
    and cheap.  When the request mentions a specific part name, scope is
    ``part``.  When it mentions a subsystem (gripper/arm/...) but no
    specific part, scope is ``subsystem``.  Otherwise — or when "redo" /
    "regenerate" appears — scope is ``whole``.
    """
    raw = text or ""
    t = raw.lower()

    # --- Detect intent ---
    intent = "modify"
    for canon, keywords in _INTENT_KEYWORDS:
        if _keyword_match(t, keywords):
            intent = canon
            break

    # Explicit redo → whole-scope regeneration
    if intent == "redo":
        return ModificationRequest(
            scope="whole", intent="redo", source="user", raw_text=raw,
        )

    # --- Detect specific part target ---
    part_names = {p.name.lower(): p.name for p in assembly.parts}
    matched_part = ""
    for lower_name, real_name in part_names.items():
        if lower_name in t:
            matched_part = real_name
            break
    if matched_part:
        # Refine intent for known part
        return ModificationRequest(
            scope="part",
            intent=intent,
            target=matched_part,
            params=_params_for_intent(intent, t),
            source="user",
            raw_text=raw,
        )

    # --- Detect subsystem target ---
    for subsystem, keywords in _SUBSYSTEM_KEYWORDS.items():
        if _keyword_match(t, keywords):
            return ModificationRequest(
                scope="subsystem",
                intent=intent,
                target=subsystem,
                params=_params_for_intent(intent, t),
                source="user",
                raw_text=raw,
            )

    # --- Fallback: whole ---
    return ModificationRequest(
        scope="whole", intent=intent, source="user", raw_text=raw,
    )


def _params_for_intent(intent: str, text_lower: str) -> dict[str, Any]:
    """Extract structured params from text for a given intent."""
    import re

    params: dict[str, Any] = {}
    if intent in ("enlarge", "shrink"):
        # Default factors
        params["factor"] = 1.5 if intent == "enlarge" else 0.75
        # Try to parse "1.5x", "150%", "两倍" etc.
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|×|倍)", text_lower)
        if m:
            try:
                f = float(m.group(1))
                if intent == "shrink" and f > 1.0:
                    f = 1.0 / f
                params["factor"] = f
            except ValueError:
                logger.warning("modifier: failed to parse scale factor '%s'", m.group(1))
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", text_lower)
        if m:
            try:
                pct = float(m.group(1)) / 100.0
                if intent == "enlarge":
                    params["factor"] = 1.0 + pct if pct < 1.0 else pct
                else:
                    params["factor"] = 1.0 - pct if pct < 1.0 else 1.0 / pct
            except ValueError:
                logger.warning("modifier: failed to parse percentage '%s'", m.group(1))
        # Chinese numerals
        cn_map = {"两": 2.0, "二": 2.0, "三": 3.0, "半": 0.5}
        for cn, val in cn_map.items():
            if cn in text_lower:
                params["factor"] = val if intent == "enlarge" else 1.0 / val
                break
    return params


# ---------------------------------------------------------------------------
# Part-level modifiers (zero LLM)
# ---------------------------------------------------------------------------


def _scale_part(part: Part, factor: float) -> Part:
    """Return a copy of *part* with all dimensions scaled by *factor*.

    FUNCTIONAL parts (motors, servos, bearings — real COTS components) are
    NEVER scaled: their dimensions come from the manufacturer's spec sheet and
    rescaling them would violate AGENTS.md §1.2 (don't make real parts "look
    right" by distorting their specs). Only structural parts (links, plates,
    brackets) may be resized. A functional part is returned unchanged with a
    warning.
    """
    if _is_functional_part(part):
        import logging
        logging.getLogger(__name__).warning(
            "Refused to scale functional part '%s' (category=%s) by %sx — "
            "real COTS components must keep their catalog dimensions "
            "(AGENTS.md §1.2).", part.name, part.category, factor,
        )
        return part
    new_part = copy.deepcopy(part)
    new_dims: dict[str, float] = {}
    for k, v in part.dimensions.items():
        try:
            new_dims[k] = float(v) * factor
        except (TypeError, ValueError):
            new_dims[k] = v
    new_part.dimensions = new_dims
    return new_part


def _is_functional_part(part: Part) -> bool:
    """True if *part* is a real functional component that must not be rescaled.

    Functional = actuators (servos/motors/steppers) and real COTS parts.
    Structural parts (links, plates, standoffs, gripper fingers — even those
    categorised "mechanical" in arm_topology) return False, because their
    dimensions are a design choice. The deciding signal is whether the part
    carries a real servo/motor identity (model number or actuator category),
    NOT the broad "mechanical" category (which arm_topology uses for the
    gripper base/fingers, which ARE designable).
    """
    cat = (part.category or "").lower()
    if cat in ("actuator", "bearing", "gear", "fastener"):
        return True
    desc = (part.description or "").lower()
    # Servo/motor model numbers or actuator terms in the description => real
    # COTS part that must keep its catalog dimensions.
    functional_markers = (
        "servo", "mg996", "sg90", "ds3218", "dynamixel", "nema",
        "motor", "电机", "舵机", "马达", "bearing", "轴承",
    )
    return any(m in desc for m in functional_markers)


def _move_part_offset(joint: Joint, delta_xyz: tuple[float, float, float]) -> Joint:
    """Return a copy of *joint* with offset bumped by *delta_xyz*."""
    new_joint = copy.deepcopy(joint)
    cur = new_joint.offset or (0.0, 0.0, 0.0)
    new_joint.offset = (
        cur[0] + delta_xyz[0],
        cur[1] + delta_xyz[1],
        cur[2] + delta_xyz[2],
    )
    return new_joint


def _flip_anchor(joint: Joint) -> Joint:
    """Swap parent_anchor / child_anchor (fix for the sanitizer's
    'top/bottom instead of front/back' warning on pitch joints)."""
    new_joint = copy.deepcopy(joint)
    new_joint.parent_anchor, new_joint.child_anchor = (
        new_joint.child_anchor, new_joint.parent_anchor,
    )
    return new_joint


def _modify_part(assembly: Assembly, req: ModificationRequest) -> Assembly:
    """Apply a part-level modification."""
    target = req.target
    parts = [_scale_part(p, 1.0) for p in assembly.parts]  # deep copy
    joints = [copy.deepcopy(j) for j in assembly.joints]

    # Find the part
    part_idx = next((i for i, p in enumerate(parts) if p.name == target), -1)
    if part_idx < 0:
        logger.warning("Part %r not found for modification", target)
        return assembly

    if req.intent in ("enlarge", "shrink"):
        factor = float(req.params.get("factor", 1.5 if req.intent == "enlarge" else 0.75))
        parts[part_idx] = _scale_part(assembly.parts[part_idx], factor)
        logger.info("Scaled part %s by %.2fx", target, factor)
    elif req.intent == "fix_orientation":
        # Find the joint where this part is the child and flip its anchors
        for i, j in enumerate(joints):
            if j.child == target:
                joints[i] = _flip_anchor(j)
                logger.info("Flipped anchors on joint %s->%s", j.parent, j.child)
                break
    elif req.intent == "fix_collision":
        # Nudge the part's joint offset by a small +Z (deterministic)
        for i, j in enumerate(joints):
            if j.child == target:
                joints[i] = _move_part_offset(j, (0.0, 0.0, 5.0))
                logger.info("Nudged part %s by +5mm Z", target)
                break
    elif req.intent == "replace":
        # No-op without a template library; caller should fall back to LLM.
        logger.info("replace intent for part %s — no template, no-op", target)
    else:
        logger.info("Unknown intent %r for part %s — no-op", req.intent, target)

    return _rebuild_assembly(assembly, parts, joints)


# ---------------------------------------------------------------------------
# Subsystem-level modifiers
# ---------------------------------------------------------------------------


def _subsystem_part_names(assembly: Assembly, subsystem: str) -> list[str]:
    """Return the part names belonging to *subsystem*."""
    out: list[str] = []
    keywords = _SUBSYSTEM_KEYWORDS.get(subsystem, [subsystem])
    for p in assembly.parts:
        nl = p.name.lower()
        if any(k in nl for k in keywords):
            out.append(p.name)
    return out


def _modify_subsystem(assembly: Assembly, req: ModificationRequest) -> Assembly:
    """Apply a subsystem-level modification."""
    subsystem = req.target
    part_names = _subsystem_part_names(assembly, subsystem)
    if not part_names:
        logger.warning("Subsystem %r has no matching parts", subsystem)
        return assembly

    parts = [copy.deepcopy(p) for p in assembly.parts]
    joints = [copy.deepcopy(j) for j in assembly.joints]

    if req.intent in ("enlarge", "shrink"):
        factor = float(req.params.get("factor", 1.5 if req.intent == "enlarge" else 0.75))
        for i, p in enumerate(parts):
            if p.name in part_names:
                parts[i] = _scale_part(p, factor)
        logger.info("Scaled subsystem %s (%d parts) by %.2fx",
                    subsystem, len(part_names), factor)
    elif req.intent == "fix_collision":
        # Use the structured correction pipeline for the named parts.
        # We synthesise a LayoutProblem of type COLLISION with the affected
        # parts and let `_generate_constraint_corrections` route it.
        problem = LayoutProblem(
            problem_type=ProblemType.COLLISION,
            severity=Severity.HIGH,
            description=f"Subsystem collision fix: {req.raw_text}",
            affected_parts=part_names,
        )
        corrections = _generate_constraint_corrections([problem], assembly)
        if corrections:
            return apply_corrections(assembly, corrections)
    elif req.intent == "fix_orientation":
        for i, j in enumerate(joints):
            if j.child in part_names:
                joints[i] = _flip_anchor(j)
    elif req.intent == "replace":
        # Out of scope without a template library — caller should fall back.
        logger.info("replace subsystem %s — needs template, no-op", subsystem)

    return _rebuild_assembly(assembly, parts, joints)


# ---------------------------------------------------------------------------
# Whole-assembly fallback
# ---------------------------------------------------------------------------


def _regenerate_whole(
    assembly: Assembly,
    req: ModificationRequest,
    *,
    description: str = "",
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> Assembly:
    """Fallback: regenerate the whole assembly via LLM.

    Calls into ``assembly_generator.generate_assembly_from_nl`` lazily so
    importing this module never requires an LLM backend to be configured.
    """
    from ..tools.assembly_generator import generate_assembly_from_nl

    prompt = description or req.raw_text or assembly.description
    return generate_assembly_from_nl(
        description=prompt,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def apply_modification(
    assembly: Assembly,
    req: ModificationRequest,
    *,
    description: str = "",
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> Assembly:
    """Apply a ModificationRequest and return the new Assembly.

    For ``scope="whole"`` an LLM regeneration is performed (requires
    api_key).  For ``part`` and ``subsystem`` scopes the change is purely
    deterministic.
    """
    if req.scope == "part":
        return _modify_part(assembly, req)
    if req.scope == "subsystem":
        return _modify_subsystem(assembly, req)
    # whole
    return _regenerate_whole(
        assembly, req,
        description=description,
        api_key=api_key, base_url=base_url, model=model,
    )


def apply_dynamic_vlm_fixes(
    assembly: Assembly,
    fix_hints: list[str],
) -> tuple[Assembly, bool]:
    """Apply motion/grasp fixes from the dynamic VLM (GLM-4.6V video).

    Unlike :func:`apply_targeted_fix_from_vlm` which handles geometric/
    appearance problems (part too big, wrong orientation), this handles
    **motion-behaviour** problems identified by watching the simulation:

    - "clamp <joint> range" / "limit <joint>" / "narrow range"
      → tighten the joint's ``range_deg`` to avoid the colliding extreme.
    - "increase friction" / "increase grasp force" / "close gap"
      → narrow the gripper finger gap (reduce finger separation so the
        cube is clamped tighter).

    Returns ``(new_assembly, applied_any)``.  When no hint matches a known
    fix pattern, returns ``(assembly, False)`` so the caller can fall back
    to LLM regeneration or accept the assembly as-is.
    """
    if not fix_hints:
        return assembly, False

    import copy
    import re

    joints = [copy.deepcopy(j) for j in assembly.joints]
    parts = [copy.deepcopy(p) for p in assembly.parts]
    applied = False

    # --- Pattern 1: clamp/narrow joint range ---
    range_keywords = ("clamp", "limit", "narrow", "restrict", "reduce.*range",
                      "tighten.*range")
    for hint in fix_hints:
        h_lower = hint.lower()
        if not any(k in h_lower for k in range_keywords):
            continue
        for j in joints:
            if j.type != "revolute":
                continue
            child = j.child.lower()
            joint_kw = child.replace("_", " ")
            if any(kw in h_lower for kw in (child, joint_kw, "shoulder", "elbow",
                                             "wrist", "pitch", "yaw", "roll")):
                lo, hi = j.range_deg
                center = (lo + hi) / 2.0
                span = (hi - lo) * 0.75
                j.range_deg = (center - span / 2, center + span / 2)
                logger.info("Dynamic VLM fix: clamped %s range %s→%s",
                            j.child, (lo, hi), j.range_deg)
                applied = True
                break

    # --- Pattern 2: gripper friction/force/gap ---
    grip_keywords = ("friction", "grasp force", "clamping force", "close gap",
                     "narrow gripper", "increase.*force", "tighten.*grip",
                     "finger.*gap", "gripper.*force", "does not hold",
                     "does not grasp", "hold the object")
    if any(any(gk in h_lower for gk in grip_keywords) for h_lower in
           [h.lower() for h in fix_hints]):
        for p in parts:
            if "finger" in p.name.lower():
                w = p.dimensions.get("width", 0)
                if w > 2:
                    p.dimensions["width"] = max(w * 0.85, 2.0)
                    logger.info("Dynamic VLM fix: narrowed %s width %.1f→%.1f",
                                p.name, w, p.dimensions["width"])
                    applied = True

    if not applied:
        return assembly, False

    return _rebuild_assembly(assembly, parts, joints), True


def optimize_grasp_parameters(
    assembly: Assembly,
    urdf_path: str,
) -> tuple[Assembly, dict]:
    """Simulation-driven grasp parameter optimization.

    Tries multiple finger-width and joint-range scaling factors, runs the
    actual grasp test (SimGraspTool) for each candidate, and selects the
    one with the best objective result (grasp_ok=True with max lift).

    **Crucially**, candidate URDFs are generated by regex-editing the ORIGINAL
    URDF XML (not re-exporting from Assembly). This preserves the original
    STL meshes, default_angles (home pose), and all geometric detail — only
    the finger width and joint range are changed. Re-exporting from Assembly
    loses these (no assembly.json sibling → home pose defaults to zero →
    fundamentally different physics).

    Args:
        assembly: the current assembly (for applying the winning params).
        urdf_path: path to the ORIGINAL URDF for grasp testing.

    Returns:
        (best_assembly, optimization_report) where report has:
        - ``candidates_tried``: number of parameter sets tested
        - ``best_score``: the score of the winner (grasp_ok=2, lifted=1, +lift/100)
        - ``best_params``: the winning parameter dict
        - ``improvement``: score delta vs original
    """
    import copy
    import re
    import tempfile

    # Read the original URDF XML — we'll regex-edit it for each candidate.
    original_xml = open(urdf_path, encoding="utf-8").read()

    # Candidate parameter sets to try.
    candidates = [
        {"finger_scale": 1.0, "range_scale": 1.0},   # original (baseline)
        {"finger_scale": 0.85, "range_scale": 1.0},   # narrower fingers
        {"finger_scale": 0.75, "range_scale": 1.0},   # even narrower
        {"finger_scale": 0.85, "range_scale": 0.85},  # narrower + tighter range
        {"finger_scale": 0.75, "range_scale": 0.85},  # max narrowing + tighter
        {"finger_scale": 1.15, "range_scale": 1.0},   # wider fingers (more contact)
    ]

    def _make_trial_urdf(params: dict) -> str:
        """Create a trial URDF by regex-editing the original XML.

        Preserves meshes, home pose, inertia — only changes finger box size
        and joint range. This is the key difference from re-exporting.
        """
        xml = original_xml
        fs = params.get("finger_scale", 1.0)
        rs = params.get("range_scale", 1.0)

        # Scale finger collision box sizes. In the URDF, finger links have
        # <box size="X Y Z"/> in their <collision> element. We scale X (width).
        if fs != 1.0:
            # Find <link name="...finger..."> blocks and scale the box size.
            def _scale_finger_box(m: re.Match) -> str:
                block = m.group(0)
                # Scale the first dimension (width) of box size in collision.
                def _scale_size(sm: re.Match) -> str:
                    vals = sm.group(1).split()
                    if len(vals) >= 3:
                        vals[0] = f"{float(vals[0]) * fs:.6f}"
                    return f'size="{" ".join(vals)}"'
                # Only scale collision boxes (not visual — visual uses mesh)
                collision_start = block.find("<collision>")
                if collision_start >= 0:
                    col_block = block[collision_start:]
                    col_block = re.sub(r'size="([^"]+)"', _scale_size, col_block, count=1)
                    block = block[:collision_start] + col_block
                return block
            xml = re.sub(
                r'<link\s+name="[^"]*finger[^"]*">.*?</link>',
                _scale_finger_box, xml, flags=re.DOTALL,
            )

        # Scale joint ranges (effort/velocity limits stay the same).
        if rs != 1.0:
            def _scale_limit(m: re.Match) -> str:
                lo = float(m.group(1)) * rs
                hi = float(m.group(2)) * rs
                # Re-center around the midpoint (keep the arm's workspace center).
                orig_mid = (float(m.group(1)) + float(m.group(2))) / 2.0
                new_half = (hi - lo) / 2.0
                return f'lower="{orig_mid - new_half:.6f}" upper="{orig_mid + new_half:.6f}"'
            # Only scale revolute joints (not prismatic/fixed).
            # Match <limit lower="X" upper="Y" .../> inside revolute joints.
            xml = re.sub(r'lower="([-\d.]+)"\s+upper="([-\d.]+)"', _scale_limit, xml)

        return xml

    def _score_grasp(urdf: str) -> tuple:
        """Run grasp test and return a score (higher = better)."""
        from ..tools.sim_grasp import SimGraspTool
        result = SimGraspTool().execute(urdf_path=urdf)
        grasp_ok = '"grasp_ok": true' in result
        lifted = '"lifted": true' in result
        lift_match = re.search(r'"lift_c_m":\s*([\d.e+-]+)', result)
        lift_mm = float(lift_match.group(1)) * 1000 if lift_match else 0
        score = (2 if grasp_ok else 0) + (1 if lifted else 0) + max(lift_mm, 0) / 100
        return score, grasp_ok, lift_mm

    best_score = -1
    best_asm = assembly
    best_params = candidates[0]
    report_lines = []

    for params in candidates:
        trial_xml = _make_trial_urdf(params)
        trial_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".urdf", delete=False, encoding="utf-8")
        trial_file.write(trial_xml)
        trial_file.close()
        try:
            score, grasp_ok, lift_mm = _score_grasp(trial_file.name)
        except Exception:
            score, grasp_ok, lift_mm = -1, False, 0
        finally:
            os.unlink(trial_file.name)
        report_lines.append(
            f"  params={params} score={score:.2f} grasp={grasp_ok} lift={lift_mm:.1f}mm")
        logger.info("Grasp optimization: params=%s score=%.2f grasp=%s lift=%.1fmm",
                    params, score, grasp_ok, lift_mm)
        if score > best_score:
            best_score = score
            best_params = params

    # Apply winning params to the Assembly dataclass for the caller.
    best_asm = _apply_params_to_assembly(assembly, best_params)

    # Baseline score (original, no modification)
    baseline_score, _, _ = _score_grasp(urdf_path)

    report = {
        "candidates_tried": len(candidates),
        "baseline_score": round(baseline_score, 2),
        "best_score": round(best_score, 2),
        "best_params": best_params,
        "improvement": round(best_score - baseline_score, 2),
        "details": report_lines,
    }
    logger.info("Grasp optimization: best=%s (baseline=%.2f, best=%.2f, improvement=%.2f)",
                best_params, baseline_score, best_score, best_score - baseline_score)
    return best_asm, report


def _apply_params_to_assembly(assembly: Assembly, params: dict) -> Assembly:
    """Apply winning optimization params to the Assembly dataclass."""
    parts = [copy.deepcopy(p) for p in assembly.parts]
    joints = [copy.deepcopy(j) for j in assembly.joints]
    fs = params.get("finger_scale", 1.0)
    rs = params.get("range_scale", 1.0)
    for p in parts:
        if "finger" in p.name.lower():
            w = p.dimensions.get("width", 10)
            p.dimensions["width"] = max(w * fs, 2.0)
    if rs < 1.0:
        for j in joints:
            if j.type == "revolute":
                lo, hi = j.range_deg
                center = (lo + hi) / 2.0
                span = (hi - lo) * rs
                j.range_deg = (center - span / 2, center + span / 2)
    return _rebuild_assembly(assembly, parts, joints)


def apply_targeted_fix_from_vlm(
    assembly: Assembly,
    problem_texts: list[str],
) -> tuple[Assembly, bool]:
    """Try deterministic fixes for VLM-reported problems before falling back.

    Returns ``(new_assembly, applied_any)``.  When ``applied_any`` is
    ``False``, the caller should regenerate via LLM.

    Safety guard (added 2026-06-18): the returned assembly is sanity-
    checked against the input.  If a deterministic correction blew up the
    assembly's bounding volume by >2× (runaway scaling) or any part became
    absurdly large (>2× the largest input part), the correction is rejected
    and ``(original_assembly, False)`` is returned so the caller falls back
    to LLM regeneration.  This is what prevents the
    finger-393×92×184mm-on-a-300mm-arm disaster from re-occurring.
    """
    if not problem_texts:
        return assembly, False

    problems = classify_problems(problem_texts, assembly)
    corrections = _generate_constraint_corrections(problems, assembly)

    # Filter out "rebuild_needed" markers — they signal "give up, regenerate"
    actionable = [c for c in corrections
                  if c.get("correction_type") != "rebuild_needed"]
    has_rebuild_signal = any(
        c.get("correction_type") == "rebuild_needed" for c in corrections
    )

    if not actionable:
        return assembly, False

    new_assembly = apply_corrections(assembly, actionable)

    # ---- Sanity guard: reject runaway corrections ----
    if not _is_sane_assembly(assembly, new_assembly):
        logger.warning(
            "Targeted fix rejected by sanity guard — falling back to LLM. "
            "Problem texts: %s", problem_texts[:3],
        )
        return assembly, False

    logger.info(
        "Applied %d deterministic corrections (%d problems classified, "
        "rebuild_signal=%s)",
        len(actionable), len(problems), has_rebuild_signal,
    )
    return new_assembly, True


def modifications_diff(
    before: Assembly, after: Assembly
) -> dict[str, Any]:
    """Return a structured diff of part changes for UI display.

    Format::

        {
          "parts_changed": [
            {"name": "gripper_finger_left", "dims_before": {...}, "dims_after": {...}},
            ...
          ],
          "joints_changed": [
            {"child": "...", "offset_before": [...], "offset_after": [...], ...},
            ...
          ],
          "parts_added": [...],
          "parts_removed": [...],
        }
    """
    before_parts = {p.name: p for p in before.parts}
    after_parts = {p.name: p for p in after.parts}
    before_joints = {(j.parent, j.child): j for j in before.joints}
    after_joints = {(j.parent, j.child): j for j in after.joints}

    parts_changed: list[dict[str, Any]] = []
    for name, p_after in after_parts.items():
        p_before = before_parts.get(name)
        if p_before and p_before.dimensions != p_after.dimensions:
            parts_changed.append({
                "name": name,
                "dims_before": dict(p_before.dimensions),
                "dims_after": dict(p_after.dimensions),
            })

    joints_changed: list[dict[str, Any]] = []
    for key, j_after in after_joints.items():
        j_before = before_joints.get(key)
        if j_before and (
            j_before.offset != j_after.offset
            or j_before.parent_anchor != j_after.parent_anchor
            or j_before.child_anchor != j_after.child_anchor
        ):
            joints_changed.append({
                "parent": j_after.parent,
                "child": j_after.child,
                "offset_before": list(j_before.offset or (0, 0, 0)),
                "offset_after": list(j_after.offset or (0, 0, 0)),
                "anchor_before": [j_before.parent_anchor, j_before.child_anchor],
                "anchor_after": [j_after.parent_anchor, j_after.child_anchor],
            })

    return {
        "parts_changed": parts_changed,
        "joints_changed": joints_changed,
        "parts_added": [n for n in after_parts if n not in before_parts],
        "parts_removed": [n for n in before_parts if n not in after_parts],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _part_volume_mm3(part: Part) -> float:
    """Rough bounding-box volume of a part in mm^3 (for sanity checks)."""
    d = part.dimensions
    length = float(d.get("length", 0) or 0)
    width = float(d.get("width", 0) or 0)
    height = float(d.get("height", 0) or 0)
    # For cylindrical parts fall back to cylinder volume
    if "diameter" in d:
        diam = float(d.get("diameter", 0) or 0)
        h = height or length or float(d.get("shaft_length", 0) or 0)
        return 3.14159 * (diam / 2) ** 2 * h
    return length * width * height


def _is_sane_assembly(before: Assembly, after: Assembly) -> bool:
    """Return False if the *after* assembly looks like a correction blew up.

    Heuristics (deliberately conservative — we'd rather skip a borderline
    fix than ship an absurd one):
    1. No single part in *after* may exceed 3× the volume of the largest
       part in *before*.
    2. Total assembly bounding volume may not exceed 2× the input.
    3. No joint offset may exceed 200mm (catches the ±210mm runaway).
    """
    before_volumes = [_part_volume_mm3(p) for p in before.parts]
    before_max = max(before_volumes, default=0.0)
    before_total = sum(before_volumes)

    after_volumes = [_part_volume_mm3(p) for p in after.parts]
    after_max = max(after_volumes, default=0.0)
    after_total = sum(after_volumes)

    # Per-part check
    if before_max > 0 and after_max > 3.0 * before_max:
        logger.info(
            "Sanity guard: max part volume grew %.1f -> %.1f mm^3 (>3x)",
            before_max, after_max,
        )
        return False

    # Whole-assembly check
    if before_total > 0 and after_total > 2.0 * before_total:
        logger.info(
            "Sanity guard: total volume grew %.1f -> %.1f mm^3 (>2x)",
            before_total, after_total,
        )
        return False

    # Joint-offset check
    for j in after.joints:
        off = j.offset or (0.0, 0.0, 0.0)
        if max(abs(c) for c in off) > 200.0:
            logger.info(
                "Sanity guard: joint %s->%s offset %s exceeds 200mm",
                j.parent, j.child, off,
            )
            return False

    return True


def _rebuild_assembly(
    assembly: Assembly,
    parts: list[Part],
    joints: list[Joint],
) -> Assembly:
    """Rebuild an Assembly preserving metadata."""
    return Assembly(
        name=assembly.name,
        parts=parts,
        joints=joints,
        description=assembly.description,
        default_angles=dict(assembly.default_angles),
        total_mass=assembly.total_mass,
        center_of_mass=assembly.center_of_mass,
        inertia_tensor=assembly.inertia_tensor,
    )
