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
    text_model: str = "GLM-4.6"
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
                raise_on_wheel_in_arm(ctx.assembly)
            if ctx.is_arm:
                validate_proportions(ctx.assembly)
            validate_assembly(ctx.assembly)

            logger.info(
                "Architect: assembly '%s' valid (%d parts, %d joints)",
                ctx.assembly.name, len(ctx.assembly.parts),
                len(ctx.assembly.joints),
            )
            return True

        except Exception as e:
            ctx.problems_history.append([f"Assembly validation error: {e}"])
            logger.warning("Architect stage failed: %s", e)
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
        resp = backend.chat(
            messages=[Message(role="user", content=fix_prompt)],
            system=ASSEMBLY_GEN_SYSTEM_PROMPT,
            temperature=temp,
            max_tokens=16384,
        )
        assembly = _parse_assembly_json(resp.content)

        # Re-apply normalizing sanitizers.
        assembly = normalize_gripper_fingers(assembly)
        if ctx.is_arm:
            assembly = ensure_arm_default_angles(assembly)

        logger.info(
            "Architect: LLM regenerated assembly '%s' (%d parts)",
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

            # Collision check + auto-resolve (non-final rounds only)
            if ctx.round_num < ctx.max_rounds:
                ctx.assembly, ctx.positions = run_collision_check_and_resolve(
                    ctx.assembly, ctx.positions,
                )

            logger.info("Solver: %d positions computed", len(ctx.positions))
            return True

        except Exception as e:
            ctx.problems_history.append([f"Solver error: {e}"])
            logger.warning("Solver stage failed: %s", e)
            return False

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
                logger.info("CAD Engineer: STLs generated at %s", stl_path)
            else:
                ctx.real_stl_dir = None
                logger.info("CAD Engineer: STL generation skipped (FreeCAD unavailable)")
            return True
        except Exception as e:
            logger.warning("CAD Engineer stage failed: %s", e)
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
            logger.info("Verifier: PASSED")
        else:
            logger.info("Verifier: FAILED — %d problems", len(problems))

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
                logger.info("Fixer: applied targeted fix")
        except Exception as e:
            logger.warning("Fixer: targeted fix failed: %s", e)

        if targeted_applied:
            # Re-run solver with the fixed assembly.
            return "solver"

        # Targeted fix didn't apply — classify and route.
        target = _classify_problems(prev_problems)
        logger.info("Fixer: routing to '%s' stage", target)
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
            logger.info("Fixer routed to '%s' — next round will re-run", target)

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
