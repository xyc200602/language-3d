"""Tests for SubAgent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from lang3d.agent.sub_agent import SubAgent, SubAgentResult, SubAgentRole
from lang3d.agent.state import PlanStep, StepStatus


class TestSubAgentRole:
    def test_role_values(self):
        assert SubAgentRole.MODELING == "modeling"
        assert SubAgentRole.VISION == "vision"
        assert SubAgentRole.GUI == "gui"
        assert SubAgentRole.VERIFICATION == "verification"
        assert SubAgentRole.GENERAL == "general"


class TestSubAgentSystemPrompt:
    def test_modeling_prompt(self):
        agent = SubAgent(role=SubAgentRole.MODELING)
        prompt = agent.get_system_prompt()
        assert "建模" in prompt
        assert "FreeCAD" in prompt

    def test_vision_prompt(self):
        agent = SubAgent(role=SubAgentRole.VISION)
        prompt = agent.get_system_prompt()
        assert "视觉" in prompt

    def test_general_prompt(self):
        agent = SubAgent(role=SubAgentRole.GENERAL)
        prompt = agent.get_system_prompt()
        assert "通用" in prompt or "任务" in prompt


class TestSubAgentExecute:
    def test_execute_without_init_returns_error(self):
        agent = SubAgent(role=SubAgentRole.GENERAL)
        step = PlanStep(description="test step")
        result = agent.execute(step)
        assert not result.success
        assert "not initialized" in result.error

    def test_execute_returns_result(self):
        mock_router = MagicMock()
        mock_tools = MagicMock()

        # Simulate LLM returning a text response (no tool calls)
        mock_response = MagicMock()
        mock_response.content = "Step completed successfully"
        mock_response.tool_calls = []
        mock_router.chat.return_value = mock_response

        agent = SubAgent(
            role=SubAgentRole.GENERAL,
            router=mock_router,
            tools=mock_tools,
        )
        step = PlanStep(description="test step")
        result = agent.execute(step)

        assert isinstance(result, SubAgentResult)
        assert result.agent_id == agent.agent_id
        assert result.step_id == step.id
        assert result.success
        assert step.status == StepStatus.COMPLETED

    def test_execute_with_context(self):
        mock_router = MagicMock()
        mock_tools = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Done"
        mock_response.tool_calls = []
        mock_router.chat.return_value = mock_response

        agent = SubAgent(
            role=SubAgentRole.GENERAL,
            router=mock_router,
            tools=mock_tools,
        )
        step = PlanStep(description="assemble parts")
        context = {
            "agent-1": {
                "description": "build base",
                "result": "base_plate.FCStd created",
                "artifacts": ["/tmp/base_plate.FCStd"],
            }
        }
        result = agent.execute(step, context=context)
        assert result.success
        # Verify context was injected into step description passed to LLM
        call_args = mock_router.chat.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
        user_msg = messages[0].content if messages else ""
        # The executor prefixes the step description, but the context should be there
        assert "前置任务结果" in user_msg
        assert "agent-1" in user_msg


class TestSubAgentCallbacks:
    def test_tool_call_callback(self):
        mock_router = MagicMock()
        mock_tools = MagicMock()

        # Simulate a tool call then completion
        tool_call_response = MagicMock()
        tool_call_response.content = ""
        tc = MagicMock()
        tc.id = "tc-1"
        tc.name = "bash"
        tc.arguments = {"command": "echo hello"}
        tool_call_response.tool_calls = [tc]

        final_response = MagicMock()
        final_response.content = "Done"
        final_response.tool_calls = []

        mock_router.chat.side_effect = [tool_call_response, final_response]
        mock_tools.execute.return_value = "hello"
        mock_tools.get_all_definitions.return_value = []

        agent = SubAgent(
            role=SubAgentRole.GENERAL,
            router=mock_router,
            tools=mock_tools,
        )

        calls = []
        agent.on_tool_call(lambda aid, name, args: calls.append((aid, name)))
        step = PlanStep(description="test")
        agent.execute(step)

        assert len(calls) == 1
        assert calls[0][0] == agent.agent_id
        assert calls[0][1] == "bash"
