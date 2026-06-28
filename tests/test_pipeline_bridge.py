"""Unit tests for the AssemblyGenerateTool → AssemblyPipeline bridge (Change 1).

``AssemblyGenerateTool.execute`` is the single production entry point for
assembly generation (the live Orchestrator/Executor path calls it as the
``assembly_generate`` tool). Change 1 redirects it from the legacy
monolithic ``generate_assembly_with_vlm_loop`` to the multi-expert
``AssemblyPipeline``, with a fallback to the legacy loop if the pipeline
raises.

These tests pin both branches:
- pipeline succeeds → legacy loop is NOT called
- pipeline raises   → legacy loop IS called (fallback)

No LLM / no FreeCAD. Everything is mocked.
"""

from __future__ import annotations

import json

import pytest


@pytest.mark.unit
class TestAssemblyGenerateToolBridge:
    """The production tool must route to AssemblyPipeline, with legacy
    fallback on pipeline failure."""

    def _ok_result(self):
        """A minimal dict shaped like AssemblyPipeline.run()'s return."""
        from lang3d.knowledge.mechanics import Assembly, Part
        asm = Assembly(
            name="t",
            parts=[Part(name="base_plate", category="structural",
                        description="底座", dimensions={})],
            joints=[],
        )
        return {
            "passed": True,
            "final_status": "PASSED",
            "rounds": 1,
            "assembly": asm,
            "positions": {},
            "problems_history": [["ok"]],
            "render_dir": "/tmp/r",
            "export_dir": "/tmp/e",
            "production_render_dir": None,
        }

    def _make_tool(self):
        from lang3d.tools.assembly_generator import AssemblyGenerateTool
        return AssemblyGenerateTool()

    def test_pipeline_success_does_not_call_legacy_loop(self, monkeypatch):
        """When the pipeline returns a result, the legacy monolithic loop
        must NOT run (it would be a wasted double-generation)."""
        import lang3d.tools.assembly_generator as ag
        import lang3d.agent.pipeline as pipeline_mod

        pipeline_called = {"n": 0}
        legacy_called = {"n": 0}

        class FakePipeline:
            def __init__(self_inner, ctx):
                pass

            def run(self_inner):
                pipeline_called["n"] += 1
                return self._ok_result()

        monkeypatch.setattr(pipeline_mod, "AssemblyPipeline", FakePipeline)
        # If the bridge incorrectly falls through, this would be invoked.
        def legacy_should_not_run(**kwargs):
            legacy_called["n"] += 1
            return self._ok_result()
        monkeypatch.setattr(ag, "generate_assembly_with_vlm_loop", legacy_should_not_run)

        out = self._make_tool().execute(description="4自由度机械臂")
        assert pipeline_called["n"] == 1
        assert legacy_called["n"] == 0, "legacy loop must not run on pipeline success"
        data = json.loads(out)
        assert data["passed"] is True
        assert data["assembly_name"] == "t"
        assert data["part_count"] == 1
        assert data["export_dir"] == "/tmp/e"

    def test_pipeline_failure_falls_back_to_legacy_loop(self, monkeypatch):
        """When the pipeline raises, the error propagates (fail-loud) unless
        LANG3D_LEGACY_FALLBACK=1 is set (audit P0-5: the old silent auto-
        fallback was a black hole that bypassed all pipeline improvements).
        With the env var set, the legacy loop runs as before."""
        import lang3d.tools.assembly_generator as ag
        import lang3d.agent.pipeline as pipeline_mod

        legacy_called = {"n": 0}

        class ExplodingPipeline:
            def __init__(self_inner, ctx):
                pass

            def run(self_inner):
                raise RuntimeError("pipeline boom")

        monkeypatch.setattr(pipeline_mod, "AssemblyPipeline", ExplodingPipeline)

        def legacy_fallback(**kwargs):
            legacy_called["n"] += 1
            return self._ok_result()
        monkeypatch.setattr(ag, "generate_assembly_with_vlm_loop", legacy_fallback)

        # Without LANG3D_LEGACY_FALLBACK: error propagates (fail-loud).
        monkeypatch.delenv("LANG3D_LEGACY_FALLBACK", raising=False)
        with pytest.raises(RuntimeError, match="pipeline boom"):
            self._make_tool().execute(description="4自由度机械臂")
        assert legacy_called["n"] == 0, "legacy must NOT run without opt-in"

        # With LANG3D_LEGACY_FALLBACK=1: legacy runs (opt-in fallback).
        monkeypatch.setenv("LANG3D_LEGACY_FALLBACK", "1")
        out = self._make_tool().execute(description="4自由度机械臂")
        assert legacy_called["n"] == 1, "legacy loop must run when opted in"
        data = json.loads(out)
        assert data["passed"] is True

    def test_missing_description_returns_error(self):
        out = self._make_tool().execute()
        assert out.startswith("Error:")
        assert "description" in out

    def test_both_paths_failing_returns_error(self, monkeypatch):
        """With LANG3D_LEGACY_FALLBACK=1, if BOTH the pipeline and the legacy
        loop raise, the tool returns an error string (the Executor contract is
        'tool returns str'). Without the env var, only the pipeline runs and
        its error propagates (audit P0-5: fail-loud, not silent fallback)."""
        import lang3d.tools.assembly_generator as ag
        import lang3d.agent.pipeline as pipeline_mod

        class ExplodingPipeline:
            def __init__(self_inner, ctx):
                pass

            def run(self_inner):
                raise RuntimeError("pipeline boom")

        monkeypatch.setattr(pipeline_mod, "AssemblyPipeline", ExplodingPipeline)
        monkeypatch.setattr(
            ag, "generate_assembly_with_vlm_loop",
            lambda **kw: (_ for _ in ()).throw(RuntimeError("legacy boom")),
        )

        # Without opt-in: pipeline error propagates directly.
        monkeypatch.delenv("LANG3D_LEGACY_FALLBACK", raising=False)
        with pytest.raises(RuntimeError, match="pipeline boom"):
            self._make_tool().execute(description="4自由度机械臂")

        # With opt-in: both fail → legacy boom → tool returns error string.
        monkeypatch.setenv("LANG3D_LEGACY_FALLBACK", "1")
        out = self._make_tool().execute(description="4自由度机械臂")
        assert out.startswith("Error:")
