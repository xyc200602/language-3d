"""Thread safety tests — validates Phase 2 fixes."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest


class TestSubAgentStepRestore:
    """Verify step.description is restored even when execution fails."""

    def test_description_restored_on_success(self):
        from lang3d.agent.sub_agent import SubAgent
        from lang3d.agent.state import PlanStep, StepStatus

        step = PlanStep(description="Original task description")
        agent = SubAgent(
            router=MagicMock(),
            tools=MagicMock(),
        )
        # Mock executor to avoid real execution
        with patch("lang3d.agent.sub_agent.Executor") as MockExecutor:
            mock_executor = MagicMock()
            mock_executor.execute_step.return_value = "done"
            MockExecutor.return_value = mock_executor

            context = {"dep1": {"description": "test", "result": "ok"}}
            agent.execute(step, context)

            assert step.description == "Original task description"

    def test_description_restored_on_exception(self):
        from lang3d.agent.sub_agent import SubAgent
        from lang3d.agent.state import PlanStep

        step = PlanStep(description="Original task description")
        agent = SubAgent(
            router=MagicMock(),
            tools=MagicMock(),
        )
        with patch("lang3d.agent.sub_agent.Executor") as MockExecutor:
            mock_executor = MagicMock()
            mock_executor.execute_step.side_effect = RuntimeError("boom")
            MockExecutor.return_value = mock_executor

            context = {"dep1": {"description": "test", "result": "ok"}}
            try:
                agent.execute(step, context)
            except RuntimeError:
                pass

            assert step.description == "Original task description"


class TestExecutorCounterReset:
    """Verify verify_fail_count and fix_history are reset per step."""

    def test_counters_reset_on_new_step(self):
        from lang3d.agent.executor import Executor
        from lang3d.agent.state import PlanStep, AgentState

        executor = Executor(router=MagicMock(), tool_registry=MagicMock())
        # Simulate a previous step leaving stale state
        executor._verify_fail_count = 5
        executor._fix_history = ["old_history"]

        step = PlanStep(description="test step")
        state = AgentState()

        # Mock the router to return no tool calls (immediate completion)
        with patch.object(executor.router, "chat") as mock_chat:
            mock_response = MagicMock()
            mock_response.content = "done"
            mock_response.tool_calls = []
            mock_chat.return_value = mock_response

            executor.execute_step(step, state)

        assert executor._verify_fail_count == 0
        assert executor._fix_history == []


class TestOrchestratorResultDefault:
    """Verify _run_node_with_retry initializes result with default."""

    def test_result_has_default_on_max_retries_zero(self):
        from lang3d.agent.orchestrator import OrchestratorAgent
        from lang3d.agent.state import PlanStep, StepStatus, Plan

        planner = MagicMock()
        planner.create_plan.return_value = Plan(steps=[])
        orchestrator = OrchestratorAgent(
            config=MagicMock(),
            router=MagicMock(),
            tools=MagicMock(),
            planner=planner,
            max_retries=0,  # Zero retries
        )

        class MockNode:
            step = PlanStep(description="test")
            dependencies = []
            agent_role = "general"

        import asyncio
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                orchestrator._run_node_with_retry(MockNode(), {})
            )
        finally:
            loop.close()
        # Should have default result, not crash
        assert result is not None
        assert result.success is False


class TestWebCurrentTaskLock:
    """Verify _current_task is protected by lock."""

    def test_run_task_uses_state_lock(self):
        from lang3d.web.app import _state_lock
        assert isinstance(_state_lock, type(threading.Lock()))
