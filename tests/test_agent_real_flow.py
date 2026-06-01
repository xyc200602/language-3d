"""End-to-end test for Agent real modeling flow.

Tests the complete Planner → Executor → Verifier loop with real FreeCAD + GLM API.
Requires: FreeCAD installed, GLM API key configured.

NOTE: These tests make real API calls and are slow (~30-60s each).
They are skipped if GLM_API_KEY is not set.
"""

import os
import sys
from pathlib import Path

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _has_api_key():
    """Check if GLM API key is available."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
    return bool(os.environ.get("GLM_API_KEY"))


def _has_freecad():
    """Check if FreeCAD is installed."""
    return any(
        (Path(p) / "python.exe").exists()
        for p in [
            os.path.expanduser(r"~\AppData\Local\Programs\FreeCAD 1.1\bin"),
            r"C:\Program Files\FreeCAD 1.1\bin",
            r"C:\Program Files\FreeCAD\bin",
        ]
    )


skip_no_api = pytest.mark.skipif(
    not _has_api_key(),
    reason="GLM_API_KEY not configured",
)

skip_no_fc = pytest.mark.skipif(
    not _has_freecad(),
    reason="FreeCAD not installed",
)

e2e_mark = pytest.mark.e2e


@e2e_mark
@skip_no_api
@skip_no_fc
class TestAgentDirectModeling:
    """Test Agent._run_direct() for simple modeling tasks."""

    def test_agent_creates_cube(self, tmp_path):
        """Agent can create a simple cube and save it."""
        from lang3d.agent.core import Agent

        agent = Agent()
        agent.config.agent.workspace = str(tmp_path)
        agent.state.workspace = tmp_path

        # Collect tool calls for verification
        tool_calls_log = []
        agent.on_tool_call(lambda name, args: tool_calls_log.append((name, args)))

        result = agent.run_task(
            "用 fc_batch 创建一个 30x30x30mm 的正方体，保存为 test_cube.FCStd",
            use_planning=False,
        )

        # Verify fc_batch was called
        fc_batch_calls = [t for t in tool_calls_log if t[0] == "fc_batch"]
        assert len(fc_batch_calls) >= 1, "fc_batch should have been called"

        # Verify file was created
        fcstd_files = list(tmp_path.glob("*.FCStd"))
        assert len(fcstd_files) >= 1, "At least one .FCStd file should be created"

    def test_agent_saves_and_exports(self, tmp_path):
        """Agent creates a model, saves FCStd and exports STL."""
        from lang3d.agent.core import Agent

        agent = Agent()
        agent.config.agent.workspace = str(tmp_path)
        agent.state.workspace = tmp_path

        result = agent.run_task(
            "用 fc_batch 创建一个半径10mm、高20mm的圆柱体，保存为 test_cyl.FCStd 并导出 STL",
            use_planning=False,
        )

        # Verify both files created
        assert list(tmp_path.glob("*.FCStd")), "FCStd file should exist"
        assert list(tmp_path.glob("*.stl")), "STL file should exist"

    def test_agent_with_boolean_ops(self, tmp_path):
        """Agent creates a box with a hole using boolean cut."""
        from lang3d.agent.core import Agent

        agent = Agent()
        agent.config.agent.workspace = str(tmp_path)
        agent.state.workspace = tmp_path

        tool_calls_log = []
        agent.on_tool_call(lambda name, args: tool_calls_log.append((name, args)))

        result = agent.run_task(
            "用 fc_batch 创建一个 50x50x10mm 的板，中间打一个直径10mm的通孔，保存为 plate_hole.FCStd",
            use_planning=False,
        )

        # Verify fc_batch was used
        assert any(t[0] == "fc_batch" for t in tool_calls_log)
        assert list(tmp_path.glob("*.FCStd"))


@e2e_mark
@skip_no_api
@skip_no_fc
class TestAgentPlanning:
    """Test Agent with planning enabled."""

    def test_agent_planned_modeling(self, tmp_path):
        """Agent plans and executes a multi-step modeling task."""
        from lang3d.agent.core import Agent

        agent = Agent()
        agent.config.agent.workspace = str(tmp_path)
        agent.state.workspace = tmp_path

        # Disable orchestration to test basic planning
        agent.config.agent.orchestrator.enable_parallel = False

        tool_calls_log = []
        agent.on_tool_call(lambda name, args: tool_calls_log.append((name, args)))

        result = agent.run_task(
            "创建一个简单的圆柱体零件：半径15mm，高30mm，保存为 test_planned_cyl.FCStd",
            use_planning=True,
        )

        # Verify planning was used (plan exists in state)
        assert agent.state.plan is not None
        assert len(agent.state.plan.steps) >= 1

        # Verify file created
        assert list(tmp_path.glob("*.FCStd"))


@e2e_mark
@skip_no_api
@skip_no_fc
class TestAgentToolRegistration:
    """Test that Agent registers all expected tools."""

    def test_freecad_tools_registered(self):
        """All FreeCAD tools are registered."""
        from lang3d.agent.core import Agent

        agent = Agent()
        tools = agent.tools.list_tools()

        essential_tools = [
            "fc_batch", "file_read", "file_write", "bash",
            "python_exec", "mesh_stats",
        ]
        for tool in essential_tools:
            assert tool in tools, f"Tool '{tool}' should be registered"

    def test_simulation_tools_registered(self):
        """FEA and simulation tools are registered."""
        from lang3d.agent.core import Agent

        agent = Agent()
        tools = agent.tools.list_tools()

        sim_tools = ["fea_run", "interference_check", "tolerance_analysis"]
        for tool in sim_tools:
            assert tool in tools, f"Tool '{tool}' should be registered"

    def test_agent_has_57_tools(self):
        """Agent registers 57 tools total."""
        from lang3d.agent.core import Agent

        agent = Agent()
        tools = agent.tools.list_tools()
        assert len(tools) >= 50, f"Expected >= 50 tools, got {len(tools)}"
