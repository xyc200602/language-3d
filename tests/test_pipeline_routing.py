"""Unit tests for the AssemblyPipeline selective re-run routing (Change 3).

Historically the Fixer's routing decision (returned by ``run_fixer``) was
computed, logged, and discarded: the ``run()`` loop always re-ran the
Architect on the next round regardless of whether the Fixer said
``solver``/``cad``/``verifier``. Change 3 makes ``run()`` honour the target
via the ``_resume_from`` attribute, so a ``solver`` target re-runs only
solver→cad→verifier and skips the Architect (and its LLM regeneration).

These tests pin that behaviour with a fake pipeline subclass that records
which stages actually executed. They run with no LLM / no FreeCAD.
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestPipelineSelectiveRerun:
    """The ``run()`` loop must honour the Fixer's routing target."""

    def _make_pipeline(self, fixer_targets, verifier_pass_on_round):
        """Build a fake AssemblyPipeline that records its stage calls.

        ``fixer_targets``: list of targets the fake ``run_fixer`` returns,
            one per round (the loop runs ``len(fixer_targets)+1`` rounds at
            most, but we cap max_rounds explicitly).
        ``verifier_pass_on_round``: the 1-based round index on which the
            fake verifier returns True (PASSED) and breaks the loop.
        """
        from lang3d.agent.pipeline import AssemblyPipeline, PipelineContext

        ctx = PipelineContext(description="2自由度机械臂", max_rounds=10)
        ctx.is_arm = True
        ctx.is_wheeled = False

        class FakePipeline(AssemblyPipeline):
            def __init__(self_inner, ctx):
                # Skip the real __init__'s heavy _setup (env reads, makedirs)
                # but still install the StageAgent attributes the loop logs
                # through. We mirror __init__ minus _setup.
                self_inner.ctx = ctx
                from lang3d.agent.pipeline import StageAgent
                from lang3d.agent.sub_agent import SubAgentRole
                self_inner.architect_agent = StageAgent(SubAgentRole.ARCHITECT, "architect")
                self_inner.solver_agent = StageAgent(SubAgentRole.SOLVER, "solver")
                self_inner.cad_agent = StageAgent(SubAgentRole.CAD_ENGINEER, "cad")
                self_inner.verifier_agent = StageAgent(SubAgentRole.VERIFICATION, "verifier")
                self_inner.fixer_agent = StageAgent(SubAgentRole.FIXER, "fixer")
                self_inner.chassis_agent = StageAgent(SubAgentRole.CHASSIS_ARCHITECT, "chassis")
                self_inner._resume_from = "architect"
                self_inner.calls: list[str] = []
                self_inner._fixer_targets = list(fixer_targets)
                self_inner._verifier_pass_on = verifier_pass_on_round

            def run_architect(self):
                self.calls.append("architect")
                return True

            def run_solver(self):
                self.calls.append("solver")
                return True

            def run_cad_engineer(self):
                self.calls.append("cad")
                return True

            def run_verifier(self):
                self.calls.append("verifier")
                return self.ctx.round_num == self._verifier_pass_on

            def run_fixer(self):
                self.calls.append("fixer")
                if self._fixer_targets:
                    return self._fixer_targets.pop(0)
                return "done"

            def _write_summary(self):
                pass

            def run_export(self):
                pass

        return FakePipeline(ctx)

    def test_architect_route_runs_all_stages(self):
        """Fixer routing to 'architect' re-runs the whole chain (the only
        path that can trigger an LLM regeneration)."""
        pipe = self._make_pipeline(fixer_targets=["architect"], verifier_pass_on_round=2)
        pipe.run()
        # Round 1: architect→solver→cad→verifier→fixer(→architect)
        # Round 2: architect→solver→cad→verifier(PASS)
        assert pipe.calls == [
            "architect", "solver", "cad", "verifier", "fixer",
            "architect", "solver", "cad", "verifier",
        ]

    def test_solver_route_skips_architect(self):
        """Bug-b guard: routing to 'solver' must NOT re-run the Architect."""
        pipe = self._make_pipeline(fixer_targets=["solver"], verifier_pass_on_round=2)
        pipe.run()
        # Round 1: architect→solver→cad→verifier→fixer(→solver)
        # Round 2: solver→cad→verifier(PASS)  ← architect SKIPPED
        assert pipe.calls == [
            "architect", "solver", "cad", "verifier", "fixer",
            "solver", "cad", "verifier",
        ]

    def test_cad_route_skips_architect_and_solver(self):
        """Routing to 'cad' re-runs only cad→verifier."""
        pipe = self._make_pipeline(fixer_targets=["cad"], verifier_pass_on_round=2)
        pipe.run()
        assert pipe.calls == [
            "architect", "solver", "cad", "verifier", "fixer",
            "cad", "verifier",
        ]

    def test_verifier_route_skips_architect_solver_cad(self):
        """Routing to 'verifier' re-runs only the verifier (e.g. a VLM false
        alarm that survived filtering and just needs re-checking)."""
        pipe = self._make_pipeline(fixer_targets=["verifier"], verifier_pass_on_round=2)
        pipe.run()
        assert pipe.calls == [
            "architect", "solver", "cad", "verifier", "fixer",
            "verifier",
        ]

    def test_resume_from_resets_to_architect_after_solver_failure(self):
        """When the solver fails, the loop must force the next round back to
        the Architect (a solver failure implies a structural defect the
        Architect must regenerate from), not stay on 'solver'."""
        from lang3d.agent.pipeline import AssemblyPipeline, PipelineContext

        ctx = PipelineContext(description="2自由度机械臂", max_rounds=3)
        ctx.is_arm = True

        class P(AssemblyPipeline):
            def __init__(self_inner, ctx):
                self_inner.ctx = ctx
                from lang3d.agent.pipeline import StageAgent
                from lang3d.agent.sub_agent import SubAgentRole
                self_inner.architect_agent = StageAgent(SubAgentRole.ARCHITECT, "architect")
                self_inner.solver_agent = StageAgent(SubAgentRole.SOLVER, "solver")
                self_inner.cad_agent = StageAgent(SubAgentRole.CAD_ENGINEER, "cad")
                self_inner.verifier_agent = StageAgent(SubAgentRole.VERIFICATION, "verifier")
                self_inner.fixer_agent = StageAgent(SubAgentRole.FIXER, "fixer")
                self_inner.chassis_agent = StageAgent(SubAgentRole.CHASSIS_ARCHITECT, "chassis")
                self_inner._resume_from = "architect"
                self_inner.calls = []

            def run_architect(self):
                self.calls.append("architect")
                return True

            def run_solver(self):
                self.calls.append("solver")
                # Always fail → each round the loop must come back to architect.
                return False

            def _write_summary(self):
                pass

            def run_export(self):
                pass

        pipe = P(ctx)
        pipe.run()
        # Each round: architect(ok)→solver(fail)→continue, and next round
        # resumes from architect (not solver).
        assert pipe.calls == ["architect", "solver"] * ctx.max_rounds
