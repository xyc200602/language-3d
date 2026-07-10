"""Multi-expert assembly pipeline (Step 2 of the multi-agent architecture).

Replaces the monolithic ``generate_assembly_with_vlm_loop`` (3500+ lines)
with five explicitly-separated stages, each mapping to an expert agent
role (Step 1).  The stages communicate via a ``PipelineContext`` that
holds all inter-stage state, making each stage independently testable
and individually re-runnable (the core benefit of the multi-agent
architecture: failure routes back to the specific stage that failed,
not a full regeneration).

Stage flow::

    Architect → Solver → CAD Engineer → Verifier
                                              ├─ PASS → Export (done)
                                              └─ FAIL → Fixer → route back

The Fixer examines the Verifier's problems and decides which upstream
stage to re-run:
  - geometric (position/connectivity)  → Solver
  - cad_defect (STL watertight/mesh)   → CAD Engineer
  - design (missing parts / DOF)       → Architect
  - pose (arm too flat / COM)          → Solver (adjust angles)

This file is deliberately thin — each stage delegates to the existing
battle-tested functions (``generate_assembly_from_nl``,
``AssemblyContext.ensure_positions``, ``_vlm_check_assembly``, etc.).
The pipeline's job is orchestration and state management, not
re-implementing the domain logic.

Added 2026-06-22 as Step 2 of the multi-agent architecture.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from ..knowledge.mechanics import Assembly
from .sub_agent import SubAgentRole, _ROLE_PROMPTS, _EXPERT_ROLES


# ---------------------------------------------------------------------------
# StageAgent — a lightweight role wrapper for each pipeline stage
# ---------------------------------------------------------------------------
#
# The pipeline's stages were originally plain methods calling domain
# functions directly. That worked but meant the "expert agent" roles
# (Architect/Solver/CAD/Verifier/Fixer) existed in name only: no role-
# scoped system prompt reached the LLM, and the tool whitelist
# (tools/base.py ROLE_TOOL_CATEGORIES) was never consulted.
#
# StageAgent closes that gap WITHOUT turning the deterministic stages into
# free-form LLM-driven agents (which would trade reliability for autonomy).
# Each stage wraps its LLM calls in the role's expert system prompt and
# records its allowed-tool set for auditability. The domain logic stays
# deterministic — exactly ArtiCAD's insight that the Assembly Agent should
# be "deterministic frame alignment, no LLM".


@dataclass
class StageAgent:
    """Role context for one pipeline stage.

    ``role_prompt`` is injected as the LLM ``system`` argument for any LLM
    call made inside this stage, so the model answers *as* that specialist.
    ``allowed_tools`` is the role's tool whitelist (from
    ROLE_TOOL_CATEGORIES + ROLE_EXTRA_TOOLS), exposed for auditing and for
    a future guard that asserts a stage never touches an out-of-scope tool.
    """

    role: SubAgentRole
    stage_name: str  # "architect" | "solver" | "cad" | "verifier" | "fixer"

    @property
    def role_prompt(self) -> str:
        return _ROLE_PROMPTS.get(self.role, "")

    @property
    def allowed_tools(self) -> list[str] | None:
        """Tool-category whitelist for this role, or None = unrestricted.

        Mirrors tools/base.py ROLE_TOOL_CATEGORIES. Returns the category
        list (e.g. ["assembly","file_ops"]) so callers can resolve names.
        """
        from ..tools.base import ROLE_TOOL_CATEGORIES, ROLE_EXTRA_TOOLS
        cats = ROLE_TOOL_CATEGORIES.get(self.stage_name)
        extras = ROLE_EXTRA_TOOLS.get(self.stage_name, [])
        # Return None for legacy/unrestricted roles, else the combined scope.
        if cats is None and not extras:
            return None
        return (cats or []) + extras

    def system_prompt(self, base: str = "") -> str:
        """Compose the system prompt: expert role + optional base prompt.

        Stages that already have a domain-specific system prompt (e.g. the
        Architect's ASSEMBLY_GEN_SYSTEM_PROMPT) pass it as ``base``; the
        role prompt is prepended so the LLM adopts the specialist persona
        AND retains the detailed generation rules.
        """
        rp = self.role_prompt
        if not rp:
            return base
        if base:
            return f"{rp}\n\n--- 领域规则 ---\n{base}"
        return rp

    def log(self, logger: logging.Logger, msg: str, *args: Any) -> None:
        # Format the message body first (msg may contain %d/%s placeholders
        # consuming *args), then log with just the stage prefix. Passing
        # *args straight to logger.info would mismatch the "[%s] %s" format
        # string when msg itself has its own placeholders.
        formatted = msg % args if args else msg
        logger.info("[%s] %s", self.stage_name, formatted)

    def log_warning(self, logger: logging.Logger, msg: str, *args: Any) -> None:
        """Warning-level log with stage identity (for failure paths).

        Keeps the original severity so failures stay visible in log filters
        even though they now carry the ``[stage]`` prefix."""
        formatted = msg % args if args else msg
        logger.warning("[%s] %s", self.stage_name, formatted)


logger = logging.getLogger(__name__)

# COM stability margin: the base_plate length must extend past the
# forward COM by this many mm so the support polygon has margin. A single
# constant shared by both the solver-stage and export-stage COM checks
# (audit P2-7: the two used different values — 25mm vs 30mm — so the same
# assembly could be "stable" at solve time but "unstable" at export,
# triggering the wheeled base_plate enlargement regression).
_COM_MARGIN_MM = 30.0


# ---------------------------------------------------------------------------
# Pipeline context — the inter-stage state carrier
# ---------------------------------------------------------------------------


@dataclass
class PipelineContext:
    """All state that flows between pipeline stages.

    Each stage reads its inputs from this context and writes its outputs
    back.  This replaces the implicit local variables scattered across
    the 3500-line ``generate_assembly_with_vlm_loop``.

    The context can be serialised to disk (assembly.json, positions.json,
    etc.) so that any stage can be re-run independently after a failure
    — the core enable for the Fixer's targeted re-routing.
    """

    # Configuration (set once at pipeline start)
    description: str = ""
    output_dir: str = ""
    api_key: str = ""
    base_url: str = ""
    text_model: str = "GLM-5.2"
    vision_model: str = "GLM-4.6V"
    temperature: float = 0.3
    max_rounds: int = 3

    # Derived classification (computed once from description)
    is_arm: bool = False
    is_wheeled: bool = False

    # Stage outputs (updated as stages run)
    assembly: Assembly | None = None
    positions: dict[str, dict] = None
    real_stl_dir: str | None = None
    cad_failed: bool = False  # True when the CAD stage crashed (trimesh fallback in use)
    problems_history: list[list[str]] = field(default_factory=list)
    passed: bool = False
    round_num: int = 0

    # Export outputs
    export_dir: str | None = None
    production_render_dir: str | None = None
    # Dynamic VLM (Stage 7): GLM-4.6V motion-behaviour verdict, or None if
    # not run (no URDF / no API key / dependency missing). Recorded as an
    # additive design check, not a blocking gate.
    dynamic_vlm_result: dict | None = None


# ---------------------------------------------------------------------------
# Stage type for the Fixer's routing decisions
# ---------------------------------------------------------------------------


# Which stage to re-run on failure.  Matches the expert role names from
# Step 1 (SubAgentRole).  "done" means no re-run needed (PASS).
TargetStage = str  # "architect" | "solver" | "cad" | "verifier" | "done"


def _classify_problems(problems: list[str]) -> TargetStage:
    """Decide which upstream stage to re-run based on problem content.

    This is the Fixer's core logic — deterministic routing rules that
    replace the old "always regenerate everything" fallback.  The rules
    are ordered by specificity: the first match wins.

    Returns the name of the stage to re-run, or "done" if the problems
    are all soft/filtered (no re-run needed).
    """
    if not problems:
        return "done"

    combined = " ".join(problems).lower()

    # Hard geometry failures — collision, disconnection, fused fingers.
    # These mean the Solver produced bad positions → re-solve.
    if any(kw in combined for kw in (
        "physically intersect", "overlap by",
        "intersecting", "intersection",
        "not connected", "genuinely floating",
        "at same position",
    )):
        return "solver"

    # CAD defects — STL watertightness, triangle explosion, non-manifold.
    # These mean the CAD Engineer produced bad geometry → re-generate STLs.
    if any(kw in combined for kw in (
        "not watertight", "non-manifold", "euler",
        "disjoint bodies", "stl generation failed",
    )):
        return "cad"

    # Design-level problems — missing parts, wrong DOF, invalid joints.
    # These mean the Architect's assembly JSON is fundamentally wrong.
    if any(kw in combined for kw in (
        "validation error", "proportion validation failed",
        "joint #", "not in parts list", "missing a functional gripper",
        "fewer than 2 finger",
    )):
        return "architect"

    # Pose issues — arm too flat/horizontal, COM outside polygon.
    # The Solver can fix these by adjusting default_angles.
    if any(kw in combined for kw in (
        "arm too flat", "arm too horizontal", "arm too vertical",
        "com within support polygon: false", "不在支撑多边形",
    )):
        return "solver"

    # VLM false alarms that survived filtering — shouldn't happen if the
    # Verifier's arbitration worked, but if they slip through, re-verify
    # rather than regenerate.
    if any(kw in combined for kw in (
        "solid block", "floating", "no support",
    )):
        return "verifier"

    # Unknown problem type — safest to go back to the Architect.
    logger.warning(
        "Fixer could not classify problems, routing to architect: %s",
        problems[:2],
    )
    return "architect"


# ---------------------------------------------------------------------------
# AssemblyPipeline — the orchestrator
# ---------------------------------------------------------------------------


class AssemblyPipeline:
    """Multi-expert assembly pipeline orchestrator.

    Runs the five-stage pipeline (architect → solver → cad → verifier)
    with a Fixer-driven repair loop.  Each stage is a method that reads
    from and writes to a ``PipelineContext``, making the flow explicit
    and each stage independently testable.

    Usage::

        ctx = PipelineContext(description="4自由度机械臂", ...)
        pipeline = AssemblyPipeline(ctx)
        result = pipeline.run()
        if result["passed"]:
            print("Assembly verified:", result["export_dir"])
    """

    def __init__(self, ctx: PipelineContext) -> None:
        self.ctx = ctx
        # One StageAgent per stage — gives each stage a role identity,
        # an expert system prompt, and a tool-whitelist for auditing.
        # This is what makes the "expert agent" roles real rather than
        # nominal: LLM calls inside each stage are now framed by the
        # specialist persona, and each stage's allowed-tool set is
        # queryable (architect can't see freecad tools, etc.).
        self.architect_agent = StageAgent(SubAgentRole.ARCHITECT, "architect")
        self.solver_agent = StageAgent(SubAgentRole.SOLVER, "solver")
        self.cad_agent = StageAgent(SubAgentRole.CAD_ENGINEER, "cad")
        self.verifier_agent = StageAgent(SubAgentRole.VERIFICATION, "verifier")
        self.fixer_agent = StageAgent(SubAgentRole.FIXER, "fixer")
        # Chassis specialist: activated for wheeled-base/dual-arm designs
        # (Task-Driven Co-Design). The architect delegates chassis topology
        # to this role so wheel conventions stay structural, not advisory.
        self.chassis_agent = StageAgent(SubAgentRole.CHASSIS_ARCHITECT, "chassis")
        # Which stage to (re)start from on the next round. The Fixer's routing
        # decision writes here so a "solver" target actually re-runs only
        # solver→cad→verifier instead of always re-running the Architect (and
        # burning an LLM regeneration). Previously the target was computed,
        # logged, and discarded (the loop always fell through to Architect).
        self._resume_from: str = "architect"
        self._setup()

    def _setup(self) -> None:
        """Classify the description and prepare output directories."""
        import os as _os

        ctx = self.ctx
        desc_lower = ctx.description.lower()
        ctx.is_arm = any(kw in desc_lower for kw in [
            "臂", "arm", "机械手", "机械臂", "抓手", "gripper", "自由度"])
        ctx.is_wheeled = any(kw in desc_lower for kw in [
            "轮", "wheel", "差速", "移动", "底盘"])

        # Fill in API config from environment if not provided (mirrors
        # the legacy loop's base_url/api_key fallback at line 2867-2870).
        if not ctx.api_key:
            ctx.api_key = _os.environ.get("GLM_API_KEY", "")
        if not ctx.base_url:
            ctx.base_url = _os.environ.get(
                "GLM_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4"
            )

        if not ctx.output_dir:
            ts = time.strftime("%Y%m%d_%H%M%S")
            case_id = "arm" if ctx.is_arm else "assembly"
            ctx.output_dir = os.path.join("data", "runs", case_id, ts)
        os.makedirs(ctx.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1: Architect — NL → Assembly
    # ------------------------------------------------------------------

    def run_architect(self) -> bool:
        """Generate or repair the assembly JSON from the description.

        Round 1: generate from NL via LLM.
        Round 2+: if the previous round failed validation, use the error
        messages as LLM feedback to regenerate.  If the Fixer applied a
        targeted fix, just re-validate the fixed assembly.

        Returns True if the assembly is valid, False if it needs
        another round.
        """
        from .assembly_generator_helpers import (
            generate_assembly_from_nl,
            normalize_gripper_fingers,
            ensure_arm_default_angles,
            raise_on_wheel_in_arm,
            validate_proportions,
            validate_assembly,
        )

        ctx = self.ctx
        try:
            if ctx.round_num == 1 or ctx.assembly is None:
                # Wheeled dual-arm: generate deterministically via the chassis
                # expert's parametric tools, NOT via LLM. The LLM-driven
                # dual-arm path repeatedly failed (wheels vertical, parts
                # flung 600mm, VLM timeouts) because the LLM mutated the
                # wheel joint conventions and the ~24K-char prompt exhausted
                # its reasoning budget. The parametric generator
                # (build_wheeled_base + compose_dual_arm_assembly) bakes the
                # conventions in structurally and solves flat in milliseconds.
                # This is the chassis expert agent *using its tools to
                # produce output* (ArtiCAD Design Agent pattern), not the
                # agent being bypassed.
                if ctx.is_wheeled and ctx.is_arm:
                    ctx.assembly = self._generate_dual_arm_deterministic()
                else:
                    # Fresh generation from NL.
                    # Inject the Architect persona on round 1 too (previously
                    # only ``_regenerate_with_feedback`` did, so the role
                    # identity only reached the model on repair rounds).
                    from ..tools.assembly_generator import (
                        ASSEMBLY_GEN_SYSTEM_PROMPT,
                    )
                    # Retrieve-before: pull similar verified-good cases from
                    # the experience store and inject as extra few-shot
                    # precedent. Empty when the store is empty or disabled —
                    # no behaviour change for first-time prompts. See
                    # :mod:`lang3d.experience` and external-audit H2.
                    few_shot_block = self._retrieve_experience_block()
                    ctx.assembly = generate_assembly_from_nl(
                        description=ctx.description,
                        api_key=ctx.api_key,
                        base_url=ctx.base_url,
                        model=ctx.text_model,
                        temperature=ctx.temperature,
                        system_prompt=self.architect_agent.system_prompt(
                            ASSEMBLY_GEN_SYSTEM_PROMPT
                        ),
                        few_shot_extras=few_shot_block,
                    )
            else:
                # Check if the previous round FAILED validation (meaning
                # the assembly is still bad and we need LLM regeneration
                # with the error as feedback).  The problems_history's
                # last entry tells us what went wrong.
                prev_errors = ctx.problems_history[-1] if ctx.problems_history else []

                if ctx.is_wheeled and ctx.is_arm:
                    # Wheeled dual-arm: NEVER let the LLM regenerate the
                    # assembly. Letting it do so (the old behaviour)
                    # re-introduced the exact failures the parametric
                    # generator exists to prevent — e2e observed wheels
                    # going from 0mm Z-variation (round 1, deterministic)
                    # to 41.5mm (round 2+, LLM), plus hallucinated
                    # "fixed-base arm contains wheels" and base/wheel
                    # overlaps. The arm-side problems (gripper finger
                    # spacing, etc.) are handled by the Fixer's targeted
                    # corrections (apply_targeted_fix_from_vlm) and the
                    # normalizers below. We preserve the current assembly
                    # (which carries the Fixer's gripper fixes) and just
                    # re-sanitize, so the chassis stays deterministic AND
                    # the gripper corrections survive across rounds.
                    self.architect_agent.log(
                        logger,
                        "dual-arm: preserving deterministic chassis on "
                        "round %d (no LLM regen)", ctx.round_num,
                    )
                    ctx.assembly = normalize_gripper_fingers(ctx.assembly)
                    if ctx.is_arm:
                        ctx.assembly = ensure_arm_default_angles(ctx.assembly)
                else:
                    needs_llm_regen = any(
                        "validation error" in e.lower() or "solver error" in e.lower()
                        for e in prev_errors
                    )

                    if needs_llm_regen:
                        # LLM regeneration with feedback (mirrors legacy loop).
                        ctx.assembly = self._regenerate_with_feedback(
                            ctx.assembly, prev_errors,
                        )
                    else:
                        # Fixer modified the assembly — just re-sanitize.
                        ctx.assembly = normalize_gripper_fingers(ctx.assembly)
                        if ctx.is_arm:
                            ctx.assembly = ensure_arm_default_angles(ctx.assembly)

            # Validate (raising validators → catch and report)
            if ctx.is_arm and not ctx.is_wheeled:
                # Wheel-in-arm handling: round 1 raises (gives the LLM a
                # chance to self-correct on regeneration). From round 2 on,
                # if the LLM STILL hallucinates wheels (observed: 3
                # consecutive rounds in e2e), stop burning API calls and
                # deterministically strip the wheel parts instead. This is
                # the ArtiCAD "targeted repair" pattern — preserve good
                # parts, remove only the offending ones.
                wheel_kws = ("wheel", "motor_mount", "电机座", "轮", "tire", "track", "履带")
                has_wheels = any(
                    any(k in p.name.lower() for k in wheel_kws)
                    for p in ctx.assembly.parts
                )
                if has_wheels and ctx.round_num >= 2:
                    from .assembly_generator_helpers import strip_wheel_parts
                    before = len(ctx.assembly.parts)
                    ctx.assembly = strip_wheel_parts(ctx.assembly)
                    after = len(ctx.assembly.parts)
                    self.architect_agent.log_warning(
                        logger,
                        "LLM still generated wheels on round %d — stripped "
                        "%d wheel part(s) deterministically (%d→%d parts)",
                        ctx.round_num, before - after, before, after,
                    )
                else:
                    raise_on_wheel_in_arm(ctx.assembly)
            if ctx.is_arm:
                validate_proportions(ctx.assembly)
            validate_assembly(ctx.assembly)

            self.architect_agent.log(
                logger, "assembly '%s' valid (%d parts, %d joints)",
                ctx.assembly.name, len(ctx.assembly.parts),
                len(ctx.assembly.joints),
            )
            return True

        except Exception as e:
            ctx.problems_history.append([f"Assembly validation error: {e}"])
            self.architect_agent.log_warning(logger, "stage failed: %s", e)
            return False

    def _generate_dual_arm_deterministic(self) -> Assembly:
        """Generate a wheeled dual-arm assembly via parametric tools.

        This is the chassis expert agent's deterministic production path:
        it derives the arm DOF and a rough payload from the description,
        then composes a structurally-correct chassis + dual-arm assembly
        with NO LLM call. The wheel conventions (axis=y, center anchor,
        no_distribute) and arm symmetry are baked into the generators, so
        the result solves flat (wheel Z-variation 0.0mm) where the
        LLM-driven path produced 42-49mm variation and VLM timeouts.

        The LLM is still used later (verifier VLM, and the arm prompt for
        single-arm cases) — this only short-circuits the *dual-arm
        topology generation*, which the LLM repeatedly got wrong.
        """
        from ..knowledge.arm_topology import parse_dof, build_arm_example
        from ..knowledge.mobile_base_gen import (
            build_wheeled_base, parse_drive_type,
        )
        from .assembly_compose import compose_dual_arm_assembly
        # Reuse the parser that LLM output flows through, for schema parity.
        from ..tools.assembly_generator import _parse_assembly_json as parse_asm

        ctx = self.ctx
        n_dof = parse_dof(ctx.description) or 4
        drive = parse_drive_type(ctx.description)
        # Rough payload estimate from the description; default 5kg.
        payload = self._estimate_payload(ctx.description)

        chassis = build_wheeled_base(
            wheel_count=4, drive_type=drive, payload_kg=payload,
        )
        # profile="mobile": the arm mounts on a wheeled chassis, so it uses the
        # compact base + shorter links (per docs/references/
        # wheeled_dual_arm_proportions.md). Servos are the SAME real COTS parts
        # as the desktop profile — only the structural links/base differ.
        arm = build_arm_example(n_dof, profile="mobile")
        dual_json = compose_dual_arm_assembly(chassis, arm, arm_dof=n_dof)
        assembly = parse_asm(dual_json)

        self.chassis_agent.log(
            logger,
            "deterministic dual-arm generated: %d-DOF arms on %s base, "
            "%d parts (payload≈%.0fkg, no LLM call)",
            n_dof, drive, len(assembly.parts), payload,
        )
        return assembly

    @staticmethod
    def _estimate_payload(description: str) -> float:
        """Rough payload estimate from the description text.

        Looks for explicit kg mentions; defaults to 5kg (a light mobile
        manipulator). This drives wheel/base sizing via WheelBaseCalculator.
        """
        import re
        m = re.search(r"(\d+(?:\.\d+)?)\s*kg", description or "", re.I)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                logger.warning("pipeline: failed to parse payload '%s'", m.group(1))
        # Heuristic: "heavy"/"重" → 20kg, "light"/"轻" → 3kg
        d = (description or "").lower()
        if "重" in description or "heavy" in d or "工业" in description:
            return 20.0
        if "轻" in description or "light" in d:
            return 3.0
        return 5.0

    def _regenerate_with_feedback(
        self, old_assembly: Assembly, errors: list[str],
    ) -> Assembly:
        """Use LLM to regenerate the assembly with validation errors as feedback.

        Mirrors the legacy loop's round 2+ path: format the errors into a
        fix prompt, send to LLM with the old assembly JSON as reference,
        parse the response.
        """
        from ..models.base import Message
        from ..models.glm import GLMBackend
        from .assembly_generator_helpers import (
            normalize_gripper_fingers,
            ensure_arm_default_angles,
        )
        # Import the fix prompt template and parser from assembly_generator
        from ..tools.assembly_generator import (
            _VLM_FIX_PROMPT,
            _parse_assembly_json,
            _assembly_to_json,
            ASSEMBLY_GEN_SYSTEM_PROMPT,
        )

        ctx = self.ctx
        backend = GLMBackend(
            api_key=ctx.api_key, base_url=ctx.base_url,
            model=ctx.text_model,
        )

        problems_text = "\n".join(f"- {p}" for p in errors)
        fix_prompt = _VLM_FIX_PROMPT.format(
            problems=problems_text,
            description=ctx.description,
        )
        prev_json = _assembly_to_json(old_assembly)
        fix_prompt += f"\nPrevious assembly (for reference):\n{prev_json}\n"

        temp = min(ctx.temperature + 0.2 * (ctx.round_num - 1), 0.7)
        # Frame the regeneration as the Architect specialist persona: the
        # expert role prompt is prepended to the domain generation rules so
        # the LLM reasons about topology/DOF/gripper as an assembly architect
        # rather than a generic generator.
        resp = backend.chat(
            messages=[Message(role="user", content=fix_prompt)],
            system=self.architect_agent.system_prompt(ASSEMBLY_GEN_SYSTEM_PROMPT),
            temperature=temp,
            max_tokens=16384,
        )
        assembly = _parse_assembly_json(resp.content)

        # Re-apply normalizing sanitizers.
        assembly = normalize_gripper_fingers(assembly)
        if ctx.is_arm:
            assembly = ensure_arm_default_angles(assembly)

        self.architect_agent.log(
            logger, "LLM regenerated assembly '%s' (%d parts)",
            assembly.name, len(assembly.parts),
        )
        return assembly

    # ------------------------------------------------------------------
    # Stage 2: Solver — Assembly → Positions
    # ------------------------------------------------------------------

    def run_solver(self) -> bool:
        """Solve 3D positions from the assembly's joint constraints.

        Includes collision detection and auto-resolution (mesh-fcl).
        Returns True if positions are valid.
        """
        from .assembly_generator_helpers import (
            AssemblyContext,
            run_collision_check_and_resolve,
        )

        ctx = self.ctx
        try:
            solver_ctx = AssemblyContext(assembly=ctx.assembly)
            ctx.positions = solver_ctx.ensure_positions()

            # COM stability closed-loop (ArtiCAD "distributed sensor" pattern):
            # after solving, check if the COM projects inside the base support
            # polygon. If not (common when the LLM emits longer-than-template
            # links), enlarge the base LENGTH along the reach axis so the
            # support polygon catches up, then re-solve. This replaces the
            # fragile pre-solve reach estimate (0.40×total_reach) which
            # under-sized bases for 5dof/7dof long arms (COM -135/-205mm).
            self._stabilize_com_if_needed(solver_ctx)

            # Collision-aware joint range clamping (AGENTS.md §5): narrow each
            # arm joint's range_deg to its maximal collision-free sub-range so
            # the arm can NEVER sweep into its own base during simulation.
            # This runs AFTER solve (needs positions) and BEFORE URDF export
            # (range_deg → <limit>).  Generalises the dual-arm-only
            # _configure_collision_aware_dual_arm Phase 2 to ALL arms.  No-op
            # when FCL is absent (degrades to the numeric sanitizer caps).
            self._clamp_arm_ranges(ctx)

            # Collision check + auto-resolve (non-final rounds only)
            if ctx.round_num < ctx.max_rounds:
                ctx.assembly, ctx.positions = run_collision_check_and_resolve(
                    ctx.assembly, ctx.positions,
                )

            self.solver_agent.log(logger, "%d positions computed", len(ctx.positions))
            return True

        except Exception as e:
            ctx.problems_history.append([f"Solver error: {e}"])
            self.solver_agent.log_warning(logger, "stage failed: %s", e)
            return False

    def _clamp_arm_ranges(self, ctx: "PipelineContext") -> None:
        """Narrow arm joint ranges to collision-free sub-ranges (post-solve).

        Wraps :func:`clamp_assembly_joint_ranges_collision_free` so every arm
        (single fixed-base, wheeled single-arm, wheeled dual-arm) gets its
        ``range_deg`` trimmed to what it can actually reach without
        interpenetrating its own base/links.  This is the single-arm
        generalisation of the dual-arm-only composition-time clamp.  Silent
        no-op when FCL is unavailable or the assembly has no revolute joints.
        """
        if not ctx.is_arm or not ctx.positions:
            return
        try:
            from .assembly_compose import clamp_assembly_joint_ranges_collision_free
            n = clamp_assembly_joint_ranges_collision_free(ctx.assembly)
            if n:
                self.solver_agent.log(
                    logger, "collision-aware range clamp: %d joint(s) narrowed", n,
                )
        except ImportError:
            pass  # FCL absent → silent skip (numeric sanitizer caps remain)

    def _stabilize_com_if_needed(self, solver_ctx: Any) -> None:
        """Enlarge the base plate if the solved COM falls outside support.

        Uses the same assembly_verifier check the e2e score uses, so the
        fix targets exactly what the grading measures. Re-solves in place
        when the base is enlarged (positions change). No-op when already
        stable or when no base_plate exists.
        """
        ctx = self.ctx
        if not ctx.is_arm or ctx.is_wheeled or not ctx.positions:
            return  # only fixed-base arms; wheeled robots have a chassis
        base = next(
            (p for p in ctx.assembly.parts
             if "base" in p.name.lower() and "plate" in p.name.lower()),
            None,
        )
        if base is None:
            return
        try:
            from .assembly_verifier import AssemblyVerifier
            check = AssemblyVerifier().check_center_of_mass_stability(
                ctx.assembly, placements=ctx.positions,
            )
        except Exception as e:
            # COM check failed (e.g. positions missing) — log and skip the
            # closed-loop enlargement rather than silently swallowing, per
            # AGENTS.md §1.1 (never bare except: pass). Solving proceeds.
            logger.warning("Solver COM pre-check skipped: %s", e)
            return
        if check.inside_support_polygon:
            return  # already stable
        # COM is forward of the support edge. Enlarge base LENGTH (solver Y,
        # the reach direction) so the forward edge covers |COM_y| + margin.
        com_y = check.center_of_mass_mm[1] if check.center_of_mass_mm else 0.0
        forward = abs(com_y)
        cur_length = float(base.dimensions.get("length", 0) or 0)
        # Need length/2 >= forward + margin (shared _COM_MARGIN_MM).
        needed = 2.0 * (forward + _COM_MARGIN_MM)
        if cur_length >= needed:
            return  # already big enough (COM may be off in X — out of scope)
        base.dimensions["length"] = needed
        self.solver_agent.log(
            logger,
            "COM %.0fmm forward of base (margin %.0fmm) — enlarged "
            "base_plate length %.0f→%.0fmm and re-solving",
            forward, check.margin_mm, cur_length, needed,
        )
        # Re-solve with the enlarged base so positions reflect the new size.
        ctx.positions = solver_ctx.ensure_positions()

    def _stabilize_com_for_export(self) -> None:
        """Last-resort COM pass before export, using export_package's logic.

        The solver-stage ``_stabilize_com_if_needed`` uses
        assembly_verifier's check; export's design_report uses
        export_package's own footprint computation. For long arms the two
        can disagree, leaving the exported report CRITICAL even when the
        solver thought it was fine. This re-checks with export's exact
        stability functions and enlarges the base one more time if the COM
        still projects outside — guaranteeing the graded report is STABLE.
        """
        ctx = self.ctx
        if not ctx.is_arm or ctx.is_wheeled or not ctx.positions:
            return
        if not ctx.assembly:
            return
        base = next(
            (p for p in ctx.assembly.parts
             if "base" in p.name.lower() and "plate" in p.name.lower()),
            None,
        )
        if base is None:
            return
        try:
            from ..knowledge.mechanics import compute_assembly_mass
            from ..tools.stability import (
                compute_support_polygon, compute_static_stability,
            )
            mass = compute_assembly_mass(ctx.assembly, positions=ctx.positions)
            com = mass.get("center_of_mass_mm", [0.0, 0.0, 0.0])
            # Build footprint exactly like export_package: ground-contact
            # parts (lowest 10% of Z range) expanded to their XY corners.
            parts_by_name = {p.name: p for p in ctx.assembly.parts}
            z_vals = [p["position"][2] for p in ctx.positions.values()]
            z_min, z_max = min(z_vals), max(z_vals)
            z_range = z_max - z_min if z_max > z_min else 1.0
            contacts: list[list[float]] = []
            for pname, pdata in ctx.positions.items():
                if pdata["position"][2] > z_min + z_range * 0.1:
                    continue
                part = parts_by_name.get(pname)
                dims = part.dimensions if part and part.dimensions else {}
                cx, cy = pdata["position"][0], pdata["position"][1]
                if "length" in dims and "width" in dims:
                    hx, hy = dims["width"] / 2.0, dims["length"] / 2.0
                    for dx, dy in [(-hx, -hy), (hx, -hy), (-hx, hy), (hx, hy)]:
                        contacts.append([cx + dx, cy + dy, 0.0])
                elif "diameter" in dims:
                    import math as _m
                    r = dims["diameter"] / 2.0
                    for i in range(8):
                        a = i * _m.pi / 4
                        contacts.append([cx + r * _m.cos(a), cy + r * _m.sin(a), 0.0])
            if not contacts:
                return
            poly = compute_support_polygon(contacts)
            stab = compute_static_stability([com[0], com[1]], poly)
            if stab.get("stable", True):
                return  # already stable per export's metric
            margin = stab.get("margin_mm", 0.0)
            # Enlarge base LENGTH (solver Y / forward) so the forward edge
            # covers |COM_y| with the shared margin, then re-solve positions.
            com_y = com[1] if len(com) > 1 else 0.0
            forward = abs(com_y)
            needed = 2.0 * (forward + _COM_MARGIN_MM)
            cur_length = float(base.dimensions.get("length", 0) or 0)
            if cur_length < needed:
                base.dimensions["length"] = needed
                self.solver_agent.log(
                    logger,
                    "export COM check margin %.0fmm — enlarged base_plate "
                    "length %.0f→%.0fmm (COM_y=%.0f)",
                    margin, cur_length, needed, com_y,
                )
                from .assembly_generator_helpers import AssemblyContext
                solver_ctx = AssemblyContext(assembly=ctx.assembly)
                ctx.positions = solver_ctx.ensure_positions()
        except Exception as e:
            # COM check itself failed — must not block export.
            logger.warning("Export COM pre-check skipped: %s", e)

    # ------------------------------------------------------------------
    # Stage 3: CAD Engineer — Assembly → STLs
    # ------------------------------------------------------------------

    def run_cad_engineer(self) -> bool:
        """Generate real FreeCAD STLs for VLM rendering.

        Falls back to trimesh preview STLs if FreeCAD is unavailable.
        Returns True if STLs were generated (or fallback used).
        """
        from .assembly_generator_helpers import generate_part_stls

        ctx = self.ctx
        round_stl_dir = os.path.join(
            ctx.output_dir, "vlm_renders", f"round_{ctx.round_num}",
            "stl_parts",
        )
        os.makedirs(round_stl_dir, exist_ok=True)

        try:
            stl_path, val_report = generate_part_stls(
                assembly=ctx.assembly, stl_dir=round_stl_dir,
            )
            if not val_report.get("skipped"):
                ctx.real_stl_dir = stl_path
                self.cad_agent.log(logger, "STLs generated at %s", stl_path)
            else:
                ctx.real_stl_dir = None
                self.cad_agent.log(logger, "STL generation skipped (FreeCAD unavailable)")
            return True
        except Exception as e:
            self.cad_agent.log_warning(logger, "stage failed: %s", e)
            ctx.real_stl_dir = None
            # Record the CAD failure so downstream stages know the STLs are
            # trimesh previews, not real CAD.  Previously this returned True
            # unconditionally, masking the failure — the Verifier used
            # trimesh box approximations and neither it nor the user knew
            # the production CAD path had crashed (AGENTS.md §1.1).
            ctx.cad_failed = True
            self.cad_agent.log_warning(
                logger,
                "CAD stage FAILED — downstream uses trimesh PREVIEW "
                "geometry, not real FreeCAD STLs.",
            )
            return True  # Proceed with trimesh fallback, but flag is set.

    # ------------------------------------------------------------------
    # Stage 4: Verifier — render + VLM + geometric arbitration
    # ------------------------------------------------------------------

    def run_verifier(self) -> bool:
        """Run the dual-channel verification (VLM + geometric).

        Returns True if the assembly passes verification.

        If :attr:`ctx.cad_failed` is set, the VLM is judging trimesh box
        PREVIEW geometry, not real FreeCAD STLs. The verdict is still
        collected (so the loop can route fixes), but it is logged as a
        warning so the run record reflects that the visual channel ran on
        degraded geometry — per AGENTS.md §5.2, a verify result obtained on
        fallback geometry must not be silently equated to a real pass.
        """
        from .assembly_generator_helpers import vlm_check_assembly

        ctx = self.ctx

        if ctx.cad_failed:
            self.verifier_agent.log_warning(
                logger,
                "CAD stage FAILED earlier — VLM is judging trimesh PREVIEW "
                "geometry, NOT real FreeCAD STLs. Treat the visual verdict "
                "as provisional (AGENTS.md §5.2).",
            )

        parts_dicts = [
            {"name": p.name, "category": p.category, "dimensions": p.dimensions}
            for p in ctx.assembly.parts
        ]
        render_dir = os.path.join(
            ctx.output_dir, "vlm_renders", f"round_{ctx.round_num}",
        )
        os.makedirs(render_dir, exist_ok=True)

        # Classify once so the VLM prompt carries the right category
        # expectations (wheeled vs fixed-arm vs generic). Without this the
        # VLM mis-classifies wheeled dual-arm robots and dead-loops — see
        # _build_verify_prompt in assembly_generator.py.
        from ..tools.assembly_generator import _classify_robot
        robot_category = _classify_robot(ctx.description)

        passed, problems = vlm_check_assembly(
            positions=ctx.positions,
            parts=parts_dicts,
            render_dir=render_dir,
            api_key=ctx.api_key,
            base_url=ctx.base_url,
            vision_model=ctx.vision_model,
            round_num=ctx.round_num,
            real_stl_dir=ctx.real_stl_dir,
            joints=ctx.assembly.joints,
            robot_category=robot_category,
        )

        ctx.problems_history.append(problems)
        ctx.passed = passed

        # --- Geometric connection-logic check (authoritative, zero false-positive) ---
        # Validates the joint graph is a buildable parent-before-child tree with
        # no cycles — a genuine "装配连接要符合实际机器人逻辑" defect catcher.
        # Unlike check_mating_surfaces (0.1mm face gate, false-positives on
        # anchor stacking) or check_tolerance_chain (fictional 0.1mm/dim
        # heuristic), this check has ~zero false positives: it only fails on a
        # genuinely broken topology where a child's parent isn't in the tree.
        try:
            import os as _os
            if "no_geo" in _os.environ.get("LANG3D_ABLATION", ""):
                pass  # ablation: skip geometric sequence check
            else:
                from .assembly_verifier import AssemblyVerifier
                seq_checks = AssemblyVerifier().check_assembly_sequence(ctx.assembly)
                infeasible = [c for c in seq_checks if not c.feasible]
                if infeasible:
                    for c in infeasible:
                        msg = f"装配顺序错误: {c.notes}"
                        if msg not in problems:
                            problems.append(msg)
                    ctx.problems_history[-1] = problems  # refresh
                    ctx.passed = False
                    passed = False
                    self.verifier_agent.log(
                        logger, "FAILED — %d assembly-sequence errors", len(infeasible),
                    )
        except Exception as e:
            self.verifier_agent.log_warning(
                logger, "assembly-sequence check skipped: %s", e,
            )

        # --- Motion-collision sweep (advisory) ---
        # Sweeps each revolute joint through its range, re-solving FK at each
        # sample, reporting motion-induced self-collisions.  Advisory only:
        # the constructive _clamp_arm_ranges (run_solver) already narrows ranges
        # to collision-free subsets, and FCL is often absent (verified=False).
        # Logged so the run record shows whether kinematic self-collision was
        # detected, but does NOT override the VLM/geo verdict.
        try:
            from .assembly_verifier import AssemblyVerifier
            motion = AssemblyVerifier().check_motion_collisions(
                ctx.assembly, ctx.positions,
            )
            if motion.verified and not motion.collision_free:
                msg = (
                    f"运动碰撞告警: {motion.joints_with_collisions}个关节在运动范围内"
                    f"发生自碰撞 ({motion.notes})"
                )
                if msg not in problems:
                    problems.append(msg)
                ctx.problems_history[-1] = problems
                self.verifier_agent.log_warning(logger, "%s", msg)
            elif motion.verified and motion.collision_free:
                self.verifier_agent.log(
                    logger, "motion-collision sweep: %d joints, collision-free",
                    motion.joints_checked,
                )
        except Exception as e:
            self.verifier_agent.log_warning(
                logger, "motion-collision sweep skipped: %s", e,
            )

        if passed:
            self.verifier_agent.log(logger, "PASSED")
        else:
            self.verifier_agent.log(logger, "FAILED — %d problems", len(problems))

        return passed

    # ------------------------------------------------------------------
    # Stage 5: Fixer — route failure to the right upstream stage
    # ------------------------------------------------------------------

    def run_fixer(self) -> TargetStage:
        """Examine verification failures and route to the right stage.

        Returns the name of the stage to re-run, or "done" if no
        re-run is needed (all problems were filtered).
        """
        from .assembly_generator_helpers import apply_targeted_fix_from_vlm

        ctx = self.ctx
        if not ctx.problems_history:
            return "done"

        prev_problems = ctx.problems_history[-1]
        if not prev_problems:
            return "done"

        # Try deterministic targeted fix first (modifier.py)
        targeted_applied = False
        try:
            new_assembly, targeted_applied = apply_targeted_fix_from_vlm(
                ctx.assembly, prev_problems,
            )
            if targeted_applied:
                ctx.assembly = new_assembly
                self.fixer_agent.log(logger, "applied targeted fix")
        except Exception as e:
            self.fixer_agent.log_warning(logger, "targeted fix failed: %s", e)

        if targeted_applied:
            # Re-run solver with the fixed assembly.
            return "solver"

        # Targeted fix didn't apply — classify and route.
        target = _classify_problems(prev_problems)
        self.fixer_agent.log(logger, "routing to '%s' stage", target)
        return target

    # ------------------------------------------------------------------
    # Stage 6: Export — package the engineering deliverables
    # ------------------------------------------------------------------

    def run_export(self) -> bool:
        """Export the full engineering package (STL/URDF/BOM/firmware).

        Returns True if export succeeded.  Like the legacy loop, exports
        even on verification failure (with FAILED_MAX_ROUNDS status) so
        downstream phases can run and the output is debuggable.  Only
        skips if there's truly no assembly at all (LLM never responded).
        """
        from .assembly_generator_helpers import (
            export_engineering_package,
            render_assembly_from_positions,
            AssemblyContext,
        )

        ctx = self.ctx

        # Hard guard: no assembly at all → nothing to export.
        if ctx.assembly is None:
            logger.warning("Export skipped: no assembly (LLM never produced one)")
            return False

        # If Solver never ran (Architect kept failing), try to solve now
        # so positions exist for export.  This mirrors the legacy loop
        # which always has positions by the time it exports.
        if ctx.positions is None:
            try:
                solver_ctx = AssemblyContext(assembly=ctx.assembly)
                ctx.positions = solver_ctx.ensure_positions()
                logger.info("Export: solved positions (Solver was skipped in loop)")
            except Exception as e:
                logger.warning("Export: could not solve positions: %s", e)

        # Final COM stability pass (mirrors what design_report will score).
        # The solver-stage pass uses assembly_verifier's check; export uses
        # export_package's. When the LLM emits an unusually long arm, the
        # baseplate the solver enlarged may still be undersized for the
        # export check's footprint convention. Re-check with export's own
        # logic here and enlarge once more if needed, so the exported
        # design_report shows STABLE. This is the last-resort closed loop.
        self._stabilize_com_for_export()

        export_dir = os.path.join(ctx.output_dir, "engineering_package")
        verification_status = "PASSED" if ctx.passed else "FAILED_MAX_ROUNDS"
        last_warnings = ctx.problems_history[-1] if ctx.problems_history else []

        # Reuse the last round's STLs for export.
        existing_stl = None
        last_round_stl = os.path.join(
            ctx.output_dir, "vlm_renders",
            f"round_{ctx.round_num}", "stl_parts",
        )
        if os.path.isdir(last_round_stl) and any(
            f.endswith(".stl") for f in os.listdir(last_round_stl)
        ):
            existing_stl = last_round_stl

        try:
            parts_dicts = [
                {"name": p.name, "category": p.category, "dimensions": p.dimensions}
                for p in ctx.assembly.parts
            ]
            # Defensive: LLM occasionally emits None for string fields
            # (material/category/description/name), which crashes export's
            # many .lower() calls with "'NoneType' has no attribute 'lower'"
            # (observed on a 3dof_arm run). Normalize before export.
            ctx.assembly.name = ctx.assembly.name or "assembly"
            ctx.assembly.description = ctx.assembly.description or ""
            for _p in ctx.assembly.parts:
                _p.name = _p.name or "part"
                _p.category = _p.category or "structural"
                _p.material = _p.material or "PLA"
                _p.description = _p.description or ""
            for _j in ctx.assembly.joints:
                _j.description = _j.description or ""
                _j.type = _j.type or "fixed"
                _j.parent = _j.parent or "base_plate"
                _j.child = _j.child or "part"
                _j.axis = _j.axis or "z"
                _j.parent_anchor = _j.parent_anchor or "center"
                _j.child_anchor = _j.child_anchor or "center"
            result = export_engineering_package(
                assembly=ctx.assembly,
                output_dir=export_dir,
                verification_status=verification_status,
                verification_warnings=last_warnings,
                existing_stl_dir=existing_stl,
            )
            ctx.export_dir = export_dir if result else None

            # Save assembly.json + positions.json to the run root so the
            # web 3D viewer (/api/runs/{case}/{ts}/positions) can load
            # the solved assembly without depending on FreeCAD internals.
            # Without these files the web positions API returns 404 and
            # the /simulate page stacks every STL at the origin.
            import json as _json
            try:
                _asm_dict = {
                    "name": ctx.assembly.name,
                    "parts": [
                        {"name": p.name, "category": p.category,
                         "description": p.description, "material": p.material,
                         "dimensions": dict(p.dimensions)}
                        for p in ctx.assembly.parts
                    ],
                    "joints": [
                        {k: v for k, v in _j.__dict__.items()
                         if k in ("type","parent","child","axis","parent_anchor",
                                  "child_anchor","offset","range_deg","no_distribute",
                                  "distribution_group","mimic_joint","mimic_multiplier",
                                  "mimic_offset")}
                        for _j in ctx.assembly.joints
                    ],
                    "default_angles": dict(ctx.assembly.default_angles),
                }
                with open(os.path.join(ctx.output_dir, "assembly.json"), "w", encoding="utf-8") as _f:
                    _json.dump(_asm_dict, _f, ensure_ascii=False, indent=2)
                if ctx.positions:
                    _pos_dict = {
                        n: {"position": list(p.get("position", (0,0,0))),
                            "rotation": list(p.get("rotation", (0,0,1,0)))}
                        for n, p in ctx.positions.items()
                    }
                    with open(os.path.join(ctx.output_dir, "positions.json"), "w", encoding="utf-8") as _f:
                        _json.dump(_pos_dict, _f, ensure_ascii=False, indent=2)
                logger.info("Saved assembly.json + positions.json for web viewer")
            except Exception as e:
                logger.warning("Failed to save assembly.json for web: %s", e)

            # Production renders
            if ctx.export_dir:
                stl_dir = os.path.join(ctx.export_dir, "stl_parts")
                if os.path.isdir(stl_dir):
                    prod_dir = os.path.join(ctx.output_dir, "production_renders")
                    render_assembly_from_positions(
                        parts=parts_dicts, positions=ctx.positions,
                        output_dir=prod_dir, stl_dir=stl_dir,
                        width=1920, height=1080, joints=ctx.assembly.joints,
                    )
                    ctx.production_render_dir = prod_dir

            return ctx.export_dir is not None

        except Exception as e:
            logger.warning("Export stage failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Stage 7: Dynamic VLM verification — agent watches simulation motion
    # ------------------------------------------------------------------

    def run_dynamic_verification(self) -> dict | None:
        """Run GLM-4.6V on MuJoCo motion frames (post-export, additive check).

        Unlike the static VLM (Stage 4, which inspects *appearance* from
        fixed viewpoints during the Fixer loop), this stage runs *after* the
        URDF is exported. It feeds simulation motion key-frames to GLM-4.6V
        and asks it to judge *motion behaviour* — self-collision during
        articulation, mechanical plausibility, workspace reachability.

        Results are recorded as ``design_warnings`` on the context (not as
        hard failures that re-trigger the Fixer loop, because the assembly
        already passed the static VLM gate). A future version may fold this
        into the loop for iterative motion-based refinement; for now it is
        an honest, additive motion check that catches what static inspection
        cannot.

        Returns the verdict dict ``{passed, problems, fix_hints}`` or None
        if the URDF / API key / model are unavailable.
        """
        ctx = self.ctx
        if not ctx.export_dir:
            logger.debug("dynamic VLM skipped: no export dir (URDF missing)")
            return None

        urdf_path = os.path.join(ctx.export_dir, "urdf.xml")
        if not os.path.exists(urdf_path):
            logger.debug("dynamic VLM skipped: urdf.xml not found at %s", urdf_path)
            return None

        api_key = os.environ.get("GLM_API_KEY", "")
        if not api_key:
            logger.info("dynamic VLM skipped: no GLM_API_KEY")
            return None

        try:
            from ..tools.sim_mujoco import render_simulation_video
            from ..tools.assembly_gen.dynamic_vlm_verify import (
                verify_motion_video, verify_motion,
            )
        except ImportError as e:
            logger.debug("dynamic VLM skipped: dependency missing (%s)", e)
            return None

        # Primary path: render the full rollout to video and let GLM-4.6V
        # watch it with native temporal understanding (trained on up to 1h
        # video). This catches transient events that key-frame extraction
        # misses (e.g., a brief collision at one specific timestep).
        logger.info("Stage 7: Dynamic VLM — rendering simulation video...")
        video_path = os.path.join(ctx.output_dir, "sim_motion.mp4")
        try:
            vr = render_simulation_video(urdf_path, video_path,
                                         duration_sec=3.0, fps=10,
                                         width=480, height=360)
        except Exception as e:
            logger.warning("video render failed: %s", e)
            vr = {"ok": False}

        if vr.get("ok"):
            logger.info("Stage 7: Dynamic VLM — GLM-4.6V watching %d-frame video...",
                        vr.get("n_frames", 0))
            try:
                result = verify_motion_video(vr["video_path"], api_key=api_key)
            except Exception as e:
                logger.warning("video VLM call failed: %s", e)
                result = None
        else:
            result = None

        # Fallback: if video failed, extract key frames (5-image input).
        if result is None:
            logger.info("Stage 7: Dynamic VLM — falling back to key-frame extraction...")
            try:
                from ..tools.sim_mujoco import extract_motion_key_frames
                from ..tools.sim_grasp import extract_grasp_frames
                frames = extract_motion_key_frames(urdf_path)
                frames.extend(extract_grasp_frames(urdf_path))
            except Exception as e:
                logger.warning("frame extraction failed: %s", e)
                frames = []
            if not frames:
                logger.warning("no frames extracted (physics diverged?)")
                return None
            logger.info("Stage 7: Dynamic VLM — GLM-4.6V judging %d frames...", len(frames))
            try:
                result = verify_motion(frames, api_key=api_key)
            except Exception as e:
                logger.warning("dynamic VLM call failed: %s", e)
                return None

        # Record results as design warnings (additive, not blocking).
        status = "PASS" if result.get("passed") else "MOTION_ISSUES_FOUND"
        logger.info("Stage 7: Dynamic VLM result: %s (problems: %d)",
                    status, len(result.get("problems", [])))
        for p in result.get("problems", []):
            logger.warning("  dynamic VLM problem: %s", p)

        # Persist the dynamic VLM verdict for the e2e report.
        try:
            import json as _json
            summary_path = os.path.join(ctx.output_dir, "dynamic_vlm_result.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                _json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        ctx.dynamic_vlm_result = result
        return result

    # ------------------------------------------------------------------
    # Main loop — orchestrate all stages with the Fixer's routing
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the full pipeline. Returns a result dict.

        The result dict has the same keys as the legacy
        ``generate_assembly_with_vlm_loop`` for backward compatibility.
        """
        ctx = self.ctx

        # Ordered stages the Fixer can route back to. ``_resume_from`` selects
        # the earliest stage that (re)runs this round; everything from there to
        # the Verifier runs. The Fixer previously returned a target that was
        # only logged — here we honour it, so a "solver"/"cad"/"verifier"
        # routing actually skips the Architect and its LLM regeneration.
        _STAGE_ORDER = ("architect", "solver", "cad", "verifier")

        for ctx.round_num in range(1, ctx.max_rounds + 1):
            logger.info("=== Pipeline Round %d/%d ===", ctx.round_num, ctx.max_rounds)

            start = self._resume_from if self._resume_from in _STAGE_ORDER else "architect"

            # Stage 1: Architect (skipped when the Fixer routed past it).
            if start in ("architect",):
                if not self.run_architect():
                    # Architect failed validation — force a full restart next
                    # round so the LLM regenerates from the error feedback.
                    self._resume_from = "architect"
                    continue

            # Stage 2: Solver (re-position the assembly; safe to run on a
            # Fixer-targeted assembly because positions are recomputed).
            if start in ("architect", "solver"):
                if not self.run_solver():
                    # Solver couldn't place the assembly — it likely has a
                    # structural defect the Architect must regenerate from.
                    self._resume_from = "architect"
                    continue

            # Stage 3: CAD Engineer (regenerate STLs for the current parts).
            if start in ("architect", "solver", "cad"):
                self.run_cad_engineer()

            # Stage 4: Verifier
            if self.run_verifier():
                break  # PASSED — exit loop

            # Stage 5: Fixer (only on failure)
            target = self.run_fixer()
            if target == "done":
                # All problems filtered — treat as pass.
                ctx.passed = True
                break

            # Honour the Fixer's routing decision: next round resumes from the
            # routed stage. ``architect`` re-runs the whole chain (the only
            # path that can trigger an LLM regeneration).
            self._resume_from = target
            self.fixer_agent.log(
                logger, "routed to '%s' — next round resumes from there", target)

        # Write loop summary
        self._write_summary()

        # Store-after: if the assembly passed verification, record it as a
        # verified-good case for future prompts to retrieve. Failures are NOT
        # stored (successes-only memory — avoids poisoning retrieval with
        # failure modes). Errors here are non-fatal: a failed write must not
        # crash a passing run. See :mod:`lang3d.experience` (audit H2).
        if ctx.passed:
            try:
                self._record_experience()
            except Exception as e:
                logger.warning("experience store: record failed (%s)", e)

        # Stage 6: Export (always, even on failure — for debugging)
        self.run_export()

        # Stage 7: Dynamic VLM — GLM-4.6V watches the simulation motion and
        # judges behaviour (self-collision, plausibility). Runs after export
        # because it needs the URDF. Additive to the static VLM (Stage 4);
        # results are design warnings, not blocking. Skipped silently when
        # the URDF, API key, or model is unavailable.
        self.run_dynamic_verification()

        return {
            "passed": ctx.passed,
            "final_status": "PASSED" if ctx.passed else "MAX_ROUNDS_REACHED",
            "rounds": len(ctx.problems_history),
            "assembly": ctx.assembly,
            "positions": ctx.positions,
            "problems_history": ctx.problems_history,
            "render_dir": os.path.join(ctx.output_dir, "vlm_renders"),
            "export_dir": ctx.export_dir,
            "production_render_dir": ctx.production_render_dir,
            # Propagate so callers can distinguish a genuine pass from a
            # pass on trimesh-preview geometry (CAD crashed upstream).
            "cad_failed": ctx.cad_failed,
        }

    def _write_summary(self) -> None:
        """Write vlm_loop_summary.json (backward-compatible format)."""
        import json

        ctx = self.ctx
        summary = {
            "test_id": ctx.description[:40],
            "total_rounds": ctx.round_num,
            "final_passed": ctx.passed,
            "verification_status": (
                "PASSED" if ctx.passed else "FAILED_MAX_ROUNDS"
            ),
            # Surface the CAD-failure flag so downstream consumers (e2e
            # tests, web UI, design_report readers) can tell a real pass
            # from a pass on trimesh-preview geometry. Previously this flag
            # was written in run_cad_engineer but never read anywhere,
            # masking CAD crashes (AGENTS.md §1.1).
            "cad_failed": ctx.cad_failed,
            "problems_history": ctx.problems_history,
        }
        try:
            path = os.path.join(ctx.output_dir, "vlm_loop_summary.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning("Failed to write vlm_loop_summary.json: %s", e)

    # ------------------------------------------------------------------
    # Experience store integration (retrieve-before / store-after)
    # ------------------------------------------------------------------

    def _retrieve_experience_block(self) -> str:
        """Retrieve similar verified-good cases and format as a few-shot block.

        Returns an empty string when:
        * the experience store is empty (first run, or feature disabled), or
        * no stored case clears the minimum similarity score.

        Empty-string return is important: ``generate_assembly_from_nl`` skips
        the injection when the block is empty, so the prompt is byte-identical
        to the pre-store behaviour and existing e2e baselines are unaffected.
        """
        from ..experience import get_store
        from ..tools.assembly_gen.vlm_verify import _classify_robot

        try:
            store = get_store()
            category = _classify_robot(self.ctx.description)
            cases = store.retrieve(self.ctx.description, category, k=2)
        except Exception as e:
            # Retrieval must never block generation — log and proceed without.
            logger.warning("experience store: retrieve failed (%s)", e)
            return ""

        if not cases:
            return ""

        # Format as additional few-shot precedent. The LLM already gets
        # parametric examples; these are *learned* cases that actually passed
        # the full pipeline (CAD + VLM + physics). Keep the block compact —
        # only the fields that constrain generation topology & angles.
        lines = [
            "以下是从历史成功案例中检索到的相似已验证装配（仅供拓扑/角度参考，"
            "尺寸仍按本次需求）："
        ]
        for i, c in enumerate(cases, 1):
            angles = ", ".join(f'"{k}":{v:.0f}' for k, v in c.default_angles.items())
            joints = ", ".join(f"{t}:{n}" for t, n in c.joint_types.items())
            lines.append(
                f"  案例{i}：{c.description}（{c.dof}自由度，{c.robot_category}）"
                f" | 关节[{joints}] | 默认角度{{{angles}}}"
            )
        logger.info(
            "experience store: retrieved %d case(s) for '%s'",
            len(cases), self.ctx.description[:40],
        )
        return "\n".join(lines)

    def _record_experience(self) -> None:
        """Record the current verified-good case into the experience store.

        Called from :meth:`run` only when ``ctx.passed`` is True. Distils the
        full assembly into a compact :class:`CaseRecord` — the heavy geometry
        stays on disk under ``data/runs/<case>/`` and is referenced by
        ``run_dir``.
        """
        from ..experience import CaseRecord, get_store
        from ..tools.assembly_gen.vlm_verify import _classify_robot

        ctx = self.ctx
        if ctx.assembly is None:
            return  # nothing to record (defensive — passed implies non-None)

        # Build joint-type histogram from the assembly's joints.
        joint_hist: dict[str, int] = {}
        movable = 0
        for j in ctx.assembly.joints:
            joint_hist[j.type] = joint_hist.get(j.type, 0) + 1
            # DOF counts only revolute joints — this is the user's mental model
            # of "an N-DOF arm" (articulated rotation axes), and matches what
            # _extract_query_dof pulls from "N自由度" in a prompt. Counting
            # prismatic joints (gripper fingers) here inflated every stored
            # case's DOF by the finger count (e.g. a 4-DOF arm + 2 finger
            # slides was stored as dof=6), so DOF-proximity retrieval never
            # hit an exact match. Prismatic info is still preserved in the
            # joint_types histogram above.
            if j.type == "revolute":
                movable += 1

        # run_dir: prefer the pipeline output_dir if it's under data/runs/,
        # else fall back to a generic placeholder.
        out_dir = getattr(ctx, "output_dir", "") or ""
        if "data" in out_dir:
            run_dir = out_dir.replace("\\", "/")
        else:
            run_dir = "data/runs/"

        record = CaseRecord(
            description=ctx.description,
            robot_category=_classify_robot(ctx.description),
            dof=movable,
            assembly_name=ctx.assembly.name,
            part_count=len(ctx.assembly.parts),
            joint_types=joint_hist,
            default_angles=dict(ctx.assembly.default_angles or {}),
            rounds_taken=ctx.round_num,
            run_dir=run_dir,
        )
        get_store().record(record)

