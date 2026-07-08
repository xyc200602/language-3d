"""Integration tests for the pipeline ↔ experience-store wiring.

These cover the two hooks added in response to external-audit finding H2
(ArtiCAD has a self-evolving experience store; Language-3D did not):

1. **Store-after** — when ``run_verifier`` passes, the verified-good case is
   recorded into the experience store.
2. **Failure isolation** — when verification fails, *nothing* is recorded
   (the store is a successes-only memory).
3. **Retrieve-before** — ``_retrieve_experience_block`` returns an empty
   string when the store is empty (so the prompt is byte-identical to the
   pre-store behaviour) and a non-empty block when the store has matches.

These run with no LLM / no FreeCAD, using the same FakePipeline pattern as
``test_pipeline_routing.py``. The real ``_record_experience`` /
``_retrieve_experience_block`` methods are exercised — only the surrounding
stage methods are stubbed.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers — build a minimal fake pipeline that exercises the REAL hooks.
# ---------------------------------------------------------------------------


def _make_fake_pipeline(
    ctx,
    *,
    verifier_passes: bool,
    assembly=None,
):
    """Construct a fake AssemblyPipeline whose stage methods are stubbed but
    whose ``_record_experience`` and ``_retrieve_experience_block`` are left
    REAL (inherited unchanged). This is what actually tests the wiring."""
    from lang3d.agent.pipeline import AssemblyPipeline
    from lang3d.agent.pipeline import StageAgent
    from lang3d.agent.sub_agent import SubAgentRole

    class FakePipeline(AssemblyPipeline):
        def __init__(self_inner, ctx):
            self_inner.ctx = ctx
            self_inner.architect_agent = StageAgent(SubAgentRole.ARCHITECT, "architect")
            self_inner.solver_agent = StageAgent(SubAgentRole.SOLVER, "solver")
            self_inner.cad_agent = StageAgent(SubAgentRole.CAD_ENGINEER, "cad")
            self_inner.verifier_agent = StageAgent(SubAgentRole.VERIFICATION, "verifier")
            self_inner.fixer_agent = StageAgent(SubAgentRole.FIXER, "fixer")
            self_inner.chassis_agent = StageAgent(SubAgentRole.CHASSIS_ARCHITECT, "chassis")
            self_inner._resume_from = "architect"
            self_inner._verifier_passes = verifier_passes

        def run_architect(self):
            # Plant the assembly so _record_experience has something to read.
            self.ctx.assembly = assembly
            return True

        def run_solver(self):
            return True

        def run_cad_engineer(self):
            pass

        def run_verifier(self):
            # Set ctx.passed to match — the real run() gates store-after on it.
            self.ctx.passed = self._verifier_passes
            return self._verifier_passes

        def run_fixer(self):
            # On a passing run, the verifier returns True and the loop breaks
            # before ever calling run_fixer. On a failing run, return a
            # routing target (NOT "done", which would set ctx.passed=True at
            # pipeline.py line ~1244) so the loop exhausts max_rounds with
            # ctx.passed still False.
            return "verifier"

        def _write_summary(self):
            pass

        def run_export(self):
            pass

    return FakePipeline(ctx)


def _make_assembly(name: str = "4dof_robotic_arm", dof: int = 4):
    """Build a minimal but real Assembly for _record_experience to distil."""
    from lang3d.knowledge.mechanics import Assembly, Part, Joint

    parts = [
        Part(name="base_plate", category="structure", description="base",
             dimensions={"length": 100, "width": 100, "height": 10}),
        Part(name="servo_1", category="actuator", description="servo",
             dimensions={"diameter": 20, "height": 40}),
    ]
    joints = [
        Joint(type="revolute", parent="base_plate", child="servo_1",
              range_deg=[-90, 90], axis="z"),
    ]
    return Assembly(
        name=name,
        parts=parts,
        joints=joints,
        description="4自由度机械臂",
        default_angles={"servo_1": 0.0},
    )


# ---------------------------------------------------------------------------
# Store-after: success records, failure does not.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_passed_run_records_to_experience_store(tmp_path: Path) -> None:
    """A passing run must record its case into the store (the H2 fix)."""
    from lang3d.agent.pipeline import PipelineContext
    from lang3d.experience.store import reset_store_for_tests

    # Isolate the store in tmp — get_store() singleton must point here.
    store = reset_store_for_tests(tmp_path / "exp")
    assert store.stats() == {}  # empty to start

    ctx = PipelineContext(description="4自由度机械臂", max_rounds=3)
    ctx.is_arm = True
    ctx.output_dir = str(tmp_path / "run")

    pipe = _make_fake_pipeline(ctx, verifier_passes=True, assembly=_make_assembly())
    pipe.run()

    # The store should now have one fixed_arm case.
    stats = store.stats()
    assert stats.get("fixed_arm") == 1
    cases = store.all_cases()
    assert cases[0].description == "4自由度机械臂"
    assert cases[0].dof == 1  # the one revolute joint
    assert cases[0].assembly_name == "4dof_robotic_arm"


@pytest.mark.unit
def test_dof_excludes_prismatic_finger_joints(tmp_path: Path) -> None:
    """DOF must count only revolute joints, not prismatic gripper fingers.

    A user saying "4自由度机械臂" means 4 articulated rotation axes. The prior
    code counted all non-fixed joints, so a 4-revolute arm + 2 prismatic
    finger slides was stored as dof=6 — breaking DOF-proximity retrieval
    (query_dof=4 never matched case.dof=6, not even the ±1 near bonus).
    Prismatic info is preserved in joint_types; it just must not inflate dof.
    """
    from lang3d.agent.pipeline import PipelineContext
    from lang3d.experience.store import reset_store_for_tests
    from lang3d.knowledge.mechanics import Assembly, Part, Joint

    store = reset_store_for_tests(tmp_path / "exp")

    # 4-revolute arm + 2 prismatic fingers + 1 fixed = the real 4dof_arm shape
    parts = [Part(name=f"p{i}", category="structure", description="x",
                  dimensions={"length": 10, "width": 10, "height": 10})
             for i in range(7)]
    joints = [
        Joint(type="revolute", parent="p0", child="p1", range_deg=[-90, 90], axis="z"),
        Joint(type="revolute", parent="p1", child="p2", range_deg=[-90, 90], axis="y"),
        Joint(type="revolute", parent="p2", child="p3", range_deg=[-90, 90], axis="y"),
        Joint(type="revolute", parent="p3", child="p4", range_deg=[-180, 180], axis="z"),
        Joint(type="prismatic", parent="p4", child="p5", range_deg=[0, 30], axis="y"),
        Joint(type="prismatic", parent="p4", child="p6", range_deg=[0, 30], axis="y"),
        Joint(type="fixed", parent="p0", child="p0", range_deg=[0, 0], axis="z"),
    ]
    assembly = Assembly(name="4dof_with_gripper", parts=parts, joints=joints,
                        description="4自由度机械臂", default_angles={})

    ctx = PipelineContext(description="4自由度机械臂", max_rounds=3)
    ctx.is_arm = True
    ctx.output_dir = str(tmp_path / "run")
    pipe = _make_fake_pipeline(ctx, verifier_passes=True, assembly=assembly)
    pipe.run()

    cases = store.all_cases()
    assert len(cases) == 1
    # dof == 4 (revolute only), NOT 6 (which would include the 2 finger slides)
    assert cases[0].dof == 4
    # prismatic joints are still recorded in the histogram
    assert cases[0].joint_types.get("prismatic") == 2
    assert cases[0].joint_types.get("revolute") == 4


@pytest.mark.unit
def test_failed_run_does_not_record(tmp_path: Path) -> None:
    """A failing run must NOT poison the store with failure cases."""
    from lang3d.agent.pipeline import PipelineContext
    from lang3d.experience.store import reset_store_for_tests

    store = reset_store_for_tests(tmp_path / "exp")

    ctx = PipelineContext(description="4自由度机械臂", max_rounds=3)
    ctx.is_arm = True
    ctx.output_dir = str(tmp_path / "run")

    pipe = _make_fake_pipeline(ctx, verifier_passes=False, assembly=_make_assembly())
    pipe.run()

    # Store stays empty — only successes are recorded.
    assert store.stats() == {}
    assert store.all_cases() == []


# ---------------------------------------------------------------------------
# Retrieve-before: empty store => no injection.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_retrieve_returns_empty_when_store_empty(tmp_path: Path) -> None:
    """``_retrieve_experience_block`` returns '' when no cases are stored.

    This is the critical backward-compat invariant: an empty store must not
    change the prompt. ``generate_assembly_from_nl`` skips injection when the
    block is empty (verified in test_prompt_injection.py-equivalent inline).
    """
    from lang3d.agent.pipeline import AssemblyPipeline, PipelineContext
    from lang3d.experience.store import reset_store_for_tests

    reset_store_for_tests(tmp_path / "exp")

    ctx = PipelineContext(description="4自由度机械臂", max_rounds=1)
    ctx.is_arm = True

    # Bypass __init__ — we only need the method, which reads self.ctx.
    pipe = AssemblyPipeline.__new__(AssemblyPipeline)
    pipe.ctx = ctx

    block = pipe._retrieve_experience_block()
    assert block == ""


@pytest.mark.unit
def test_retrieve_returns_block_when_store_has_match(tmp_path: Path) -> None:
    """After a case is stored, a similar prompt retrieves a non-empty block."""
    from lang3d.agent.pipeline import AssemblyPipeline, PipelineContext
    from lang3d.experience import CaseRecord
    from lang3d.experience.store import reset_store_for_tests

    store = reset_store_for_tests(tmp_path / "exp")
    # Seed the store with a verified-good arm case.
    store.record(CaseRecord(
        description="4自由度机械臂",
        robot_category="fixed_arm",
        dof=4,
        assembly_name="4dof_robotic_arm",
        part_count=12,
        joint_types={"revolute": 4, "fixed": 8},
        default_angles={"joint_1": 0.0, "joint_2": -45.0},
        rounds_taken=1,
        run_dir="data/runs/4dof_arm/x",
    ))

    ctx = PipelineContext(description="4自由度机械臂", max_rounds=1)
    ctx.is_arm = True
    pipe = AssemblyPipeline.__new__(AssemblyPipeline)
    pipe.ctx = ctx

    block = pipe._retrieve_experience_block()
    assert block != ""
    assert "4自由度机械臂" in block
    # The block must be a few-shot-style hint, not raw JSON.
    assert "历史成功案例" in block or "案例" in block
