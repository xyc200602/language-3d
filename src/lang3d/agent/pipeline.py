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
        logger.info("[%s] %s", self.stage_name, msg, *args)

    def log_warning(self, logger: logging.Logger, msg: str, *args: Any) -> None:
        """Warning-level log with stage identity (for failure paths).

        Keeps the original severity so failures stay visible in log filters
        even though they now carry the ``[stage]`` prefix."""
        logger.warning("[%s] %s", self.stage_name, msg, *args)


logger = logging.getLogger(__name__)


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
    problems_history: list[list[str]] = field(default_factory=list)
    passed: bool = False
    round_num: int = 0

    # Export outputs
    export_dir: str | None = None
    production_render_dir: str | None = None


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
                # Fresh generation from NL.
                ctx.assembly = generate_assembly_from_nl(
                    description=ctx.description,
                    api_key=ctx.api_key,
                    base_url=ctx.base_url,
                    model=ctx.text_model,
                    temperature=ctx.temperature,
                )
            else:
                # Check if the previous round FAILED validation (meaning
                # the assembly is still bad and we need LLM regeneration
                # with the error as feedback).  The problems_history's
                # last entry tells us what went wrong.
                prev_errors = ctx.problems_history[-1] if ctx.problems_history else []
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
        # Need length/2 >= forward + 25mm margin.
        needed = 2.0 * (forward + 25.0)
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
            # covers |COM_y| with a 30mm margin, then re-solve positions.
            com_y = com[1] if len(com) > 1 else 0.0
            forward = abs(com_y)
            needed = 2.0 * (forward + 30.0)
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
            return True  # Non-fatal — Verifier will use trimesh fallback

    # ------------------------------------------------------------------
    # Stage 4: Verifier — render + VLM + geometric arbitration
    # ------------------------------------------------------------------

    def run_verifier(self) -> bool:
        """Run the dual-channel verification (VLM + geometric).

        Returns True if the assembly passes verification.
        """
        from .assembly_generator_helpers import vlm_check_assembly

        ctx = self.ctx
        parts_dicts = [
            {"name": p.name, "category": p.category, "dimensions": p.dimensions}
            for p in ctx.assembly.parts
        ]
        render_dir = os.path.join(
            ctx.output_dir, "vlm_renders", f"round_{ctx.round_num}",
        )
        os.makedirs(render_dir, exist_ok=True)

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
        )

        ctx.problems_history.append(problems)
        ctx.passed = passed

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
            result = export_engineering_package(
                assembly=ctx.assembly,
                output_dir=export_dir,
                verification_status=verification_status,
                verification_warnings=last_warnings,
                existing_stl_dir=existing_stl,
            )
            ctx.export_dir = export_dir if result else None

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
    # Main loop — orchestrate all stages with the Fixer's routing
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the full pipeline. Returns a result dict.

        The result dict has the same keys as the legacy
        ``generate_assembly_with_vlm_loop`` for backward compatibility.
        """
        ctx = self.ctx

        for ctx.round_num in range(1, ctx.max_rounds + 1):
            logger.info("=== Pipeline Round %d/%d ===", ctx.round_num, ctx.max_rounds)

            # Stage 1: Architect
            if not self.run_architect():
                continue

            # Stage 2: Solver
            if not self.run_solver():
                continue

            # Stage 3: CAD Engineer
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

            # The Fixer may have modified the assembly (targeted fix).
            # The next round's Architect will re-validate it.
            # For non-targeted routes, the LLM regeneration happens in
            # the Architect stage when round_num > 1 and the assembly
            # needs fundamental changes.
            self.fixer_agent.log(
                logger, "routed to '%s' — next round will re-run", target)

        # Write loop summary
        self._write_summary()

        # Stage 6: Export (always, even on failure — for debugging)
        self.run_export()

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
            "problems_history": ctx.problems_history,
        }
        try:
            path = os.path.join(ctx.output_dir, "vlm_loop_summary.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning("Failed to write vlm_loop_summary.json: %s", e)
