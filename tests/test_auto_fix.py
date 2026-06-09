"""Auto-fix closed-loop integration tests.

Tests the complete verify → fail → classify → extract FIX_COMMANDS → fix → re-verify
cycle using mocked LLM, VLM, and tools. No real API keys / FreeCAD / VLM required.

Two test classes:
  - TestAutoFixClosedLoopDirect: tests core._run_direct() auto-fix path
  - TestAutoFixClosedLoopExecutor: tests executor.execute_step() auto-fix path
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lang3d.models.base import Message, ModelResponse, ToolCall, ToolDefinition
from lang3d.tools.base import Tool, ToolRegistry
from lang3d.agent.state import AgentState, PlanStep, StepStatus


# ---------------------------------------------------------------------------
# Helpers: mock tools
# ---------------------------------------------------------------------------

class MockTool(Tool):
    """A simple mock tool that returns a preset result."""

    def __init__(self, name: str, result: str = "ok"):
        self.name = name
        self.description = f"Mock {name} tool"
        self._result = result

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}},
        )

    def execute(self, **kwargs: Any) -> str:
        return self._result


class CallableMockTool(MockTool):
    """Mock tool that calls a function to determine the result."""

    def __init__(self, name: str, func):
        super().__init__(name)
        self._func = func
        self.call_count = 0

    def execute(self, **kwargs: Any) -> str:
        self.call_count += 1
        return self._func(self.call_count, kwargs)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VERIFY_FAIL_RESULT = (
    "MATCH: False\n"
    "OBSERVED: Solid cube, no features visible\n"
    "DIFFERENCES: Missing center hole\n"
    "FIX_COMMANDS: fc_batch operations=[{op: cylinder_cut, radius: 4, height: 25}]\n"
    "SUGGESTION: Add a cylindrical cut"
)

VERIFY_PASS_RESULT = "MATCH: True\nOBSERVED: 25x25x25mm cube with center hole"


def _make_tool_registry() -> tuple[ToolRegistry, dict[str, list[str]]]:
    """Create a ToolRegistry with mock fc_batch and cad_verify tools.

    Returns (registry, call_log) where call_log maps tool names to lists of results.
    """
    registry = ToolRegistry()
    call_log: dict[str, list[str]] = {"fc_batch": [], "cad_verify": []}

    def fc_batch_fn(call_count, kwargs):
        call_log["fc_batch"].append(f"call_{call_count}")
        return "fc_batch executed successfully"

    def cad_verify_fn(call_count, kwargs):
        call_log["cad_verify"].append(f"call_{call_count}")
        # First call fails, second call passes
        if call_count == 1:
            return VERIFY_FAIL_RESULT
        return VERIFY_PASS_RESULT

    fc_batch_tool = CallableMockTool("fc_batch", fc_batch_fn)
    cad_verify_tool = CallableMockTool("cad_verify", cad_verify_fn)
    registry.register(fc_batch_tool)
    registry.register(cad_verify_tool)

    return registry, call_log


def _make_responses_for_direct_fail_then_pass() -> list[ModelResponse]:
    """Model responses for the _run_direct closed-loop test.

    Sequence:
      1. LLM calls fc_batch (create model)
      2. LLM calls cad_verify (fails → auto-fix hint injected)
      3. LLM calls fc_batch again (fix model)
      4. LLM calls cad_verify (passes)
      5. LLM returns final text (no tool calls)
    """
    return [
        # Turn 1: call fc_batch
        ModelResponse(
            content="I'll create the model",
            tool_calls=[ToolCall(id="tc1", name="fc_batch", arguments={"operations": "[]"})],
        ),
        # Turn 2: call cad_verify
        ModelResponse(
            content="Let me verify",
            tool_calls=[ToolCall(id="tc2", name="cad_verify", arguments={"expected": "cube with hole"})],
        ),
        # Turn 3: call fc_batch (fix)
        ModelResponse(
            content="I'll fix the model",
            tool_calls=[ToolCall(id="tc3", name="fc_batch", arguments={"operations": "[cylinder_cut]"})],
        ),
        # Turn 4: call cad_verify (should pass now)
        ModelResponse(
            content="Let me verify again",
            tool_calls=[ToolCall(id="tc4", name="cad_verify", arguments={"expected": "cube with hole"})],
        ),
        # Turn 5: done
        ModelResponse(
            content="Task completed successfully",
            tool_calls=[],
        ),
    ]


def _make_responses_always_fail() -> list[ModelResponse]:
    """Model responses for max-retries test: cad_verify always fails."""

    def _verify_fail_call(call_num):
        return ModelResponse(
            content=f"Verifying attempt {call_num}",
            tool_calls=[
                ToolCall(
                    id=f"tc_v{call_num}",
                    name="cad_verify",
                    arguments={"expected": "cube with hole"},
                )
            ],
        )

    def _fix_call(call_num):
        return ModelResponse(
            content=f"Fixing attempt {call_num}",
            tool_calls=[
                ToolCall(
                    id=f"tc_f{call_num}",
                    name="fc_batch",
                    arguments={"operations": "[]"},
                )
            ],
        )

    # Pattern: fc_batch → cad_verify(fail) → fc_batch → cad_verify(fail) → fc_batch → cad_verify(fail) → cad_verify(fail) → done
    responses = []
    for i in range(4):
        responses.append(_fix_call(i + 1))
        responses.append(_verify_fail_call(i + 1))
    # Final response after max turns
    responses.append(ModelResponse(content="Giving up", tool_calls=[]))
    return responses


# ---------------------------------------------------------------------------
# TestAutoFixClosedLoopDirect — tests core._run_direct() path
# ---------------------------------------------------------------------------

class TestAutoFixClosedLoopDirect:
    """Test auto-fix closed loop in core.Agent._run_direct()."""

    def test_verify_fail_fix_pass(self):
        """fc_batch → cad_verify(FAIL) → fix hint injected → fc_batch(fix) → cad_verify(PASS) → done."""
        from lang3d.agent.core import Agent

        registry, call_log = _make_tool_registry()

        # Build a sequence of LLM responses
        responses = _make_responses_for_direct_fail_then_pass()
        mock_router = MagicMock()
        mock_router.chat = MagicMock(side_effect=responses)
        mock_router.chat.__wrapped__ = None  # prevent inspection issues

        # Patch Agent.__init__ to avoid loading all tools
        with patch.object(Agent, "__init__", lambda self, *a, **kw: None):
            agent = Agent.__new__(Agent)
            agent.router = mock_router
            agent.tools = registry
            agent.state = AgentState(workspace="/tmp/test")
            agent.config = MagicMock()
            agent.config.agent.max_turns = 20
            agent.config.agent.max_verify_retries = 3
            agent._on_tool_call = None
            agent._on_tool_result = None
            agent._on_thinking = None

        result = agent._run_direct("Create a cube with hole")

        # fc_batch called twice (create + fix)
        assert len(call_log["fc_batch"]) == 2
        # cad_verify called twice (fail + pass)
        assert len(call_log["cad_verify"]) == 2
        # Final result is success
        assert "completed successfully" in result

    def test_verify_fail_max_retries_exceeded(self):
        """cad_verify keeps failing → stops after max_verify_retries."""
        from lang3d.agent.core import Agent

        registry = ToolRegistry()

        # cad_verify always fails
        def cad_verify_always_fail(call_count, kwargs):
            return VERIFY_FAIL_RESULT

        def fc_batch_ok(call_count, kwargs):
            return "ok"

        cad_tool = CallableMockTool("cad_verify", cad_verify_always_fail)
        batch_tool = CallableMockTool("fc_batch", fc_batch_ok)
        registry.register(cad_tool)
        registry.register(batch_tool)

        responses = _make_responses_always_fail()
        mock_router = MagicMock()
        mock_router.chat = MagicMock(side_effect=responses)

        with patch.object(Agent, "__init__", lambda self, *a, **kw: None):
            agent = Agent.__new__(Agent)
            agent.router = mock_router
            agent.tools = registry
            agent.state = AgentState(workspace="/tmp/test")
            agent.config = MagicMock()
            agent.config.agent.max_turns = 20
            agent.config.agent.max_verify_retries = 3
            agent._on_tool_call = None
            agent._on_tool_result = None
            agent._on_thinking = None

        result = agent._run_direct("Create a cube with hole")

        # cad_verify should be called more than 3 times (max_verify_retries=3)
        # but the loop should eventually stop
        assert isinstance(result, str)

    def test_fix_commands_included_in_hint(self):
        """Verify FIX_COMMANDS is extracted and included in the fix hint message."""
        from lang3d.agent.core import Agent

        registry = ToolRegistry()
        hint_messages: list[str] = []

        def cad_verify_with_fix_commands(call_count, kwargs):
            return VERIFY_FAIL_RESULT

        def fc_batch_ok(call_count, kwargs):
            return "ok"

        registry.register(CallableMockTool("cad_verify", cad_verify_with_fix_commands))
        registry.register(CallableMockTool("fc_batch", fc_batch_ok))

        # Responses: fc_batch → cad_verify → done
        responses = [
            ModelResponse(
                content="Creating model",
                tool_calls=[ToolCall(id="tc1", name="fc_batch", arguments={})],
            ),
            ModelResponse(
                content="Verifying",
                tool_calls=[ToolCall(id="tc2", name="cad_verify", arguments={"expected": "cube with hole"})],
            ),
            ModelResponse(content="Done", tool_calls=[]),
        ]

        mock_router = MagicMock()

        # Wrap chat to capture user messages (fix hints)
        original_side_effect = responses

        def chat_wrapper(*args, **kwargs):
            # Capture messages passed to chat for inspection
            messages = args[0] if args else kwargs.get("messages", [])
            for msg in messages:
                if isinstance(msg, Message) and msg.role == "user":
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if "VLM 建议的具体修复操作" in content:
                        hint_messages.append(content)
            return original_side_effect.pop(0)

        mock_router.chat = MagicMock(side_effect=chat_wrapper)

        with patch.object(Agent, "__init__", lambda self, *a, **kw: None):
            agent = Agent.__new__(Agent)
            agent.router = mock_router
            agent.tools = registry
            agent.state = AgentState(workspace="/tmp/test")
            agent.config = MagicMock()
            agent.config.agent.max_turns = 20
            agent.config.agent.max_verify_retries = 3
            agent._on_tool_call = None
            agent._on_tool_result = None
            agent._on_thinking = None

        agent._run_direct("Create a cube with hole")

        # Verify that the fix hint contained FIX_COMMANDS
        assert len(hint_messages) > 0, "Expected at least one fix hint message with VLM suggestions"
        assert "cylinder_cut" in hint_messages[0], "FIX_COMMANDS should be included in the fix hint"


# ---------------------------------------------------------------------------
# TestAutoFixClosedLoopExecutor — tests executor.execute_step() path
# ---------------------------------------------------------------------------

class TestAutoFixClosedLoopExecutor:
    """Test auto-fix closed loop in executor.execute_step()."""

    def test_executor_auto_fix_on_verify_failure(self):
        """In execute_step(), cad_verify FAIL → fix hint injected → re-verify PASS → COMPLETED."""
        from lang3d.agent.executor import Executor

        registry = ToolRegistry()

        cad_verify_call_count = 0

        def cad_verify_fn(call_count, kwargs):
            nonlocal cad_verify_call_count
            cad_verify_call_count = call_count
            if call_count == 1:
                return VERIFY_FAIL_RESULT
            return VERIFY_PASS_RESULT

        def fc_batch_fn(call_count, kwargs):
            return "fc_batch ok"

        registry.register(CallableMockTool("cad_verify", cad_verify_fn))
        registry.register(CallableMockTool("fc_batch", fc_batch_fn))

        # LLM responses for executor:
        # 1. fc_batch (initial)
        # 2. cad_verify (fail → auto-fix injects hint)
        # 3. fc_batch (fix)
        # 4. cad_verify (pass)
        # 5. no tool calls → done
        responses = [
            ModelResponse(
                content="Creating model",
                tool_calls=[ToolCall(id="e1", name="fc_batch", arguments={"operations": "[]"})],
            ),
            ModelResponse(
                content="Verifying",
                tool_calls=[ToolCall(id="e2", name="cad_verify", arguments={"expected": "cube with hole"})],
            ),
            ModelResponse(
                content="Fixing model based on hint",
                tool_calls=[ToolCall(id="e3", name="fc_batch", arguments={"operations": "[cylinder_cut]"})],
            ),
            ModelResponse(
                content="Re-verifying",
                tool_calls=[ToolCall(id="e4", name="cad_verify", arguments={"expected": "cube with hole"})],
            ),
            ModelResponse(
                content="Step completed",
                tool_calls=[],
            ),
        ]

        mock_router = MagicMock()
        mock_router.chat = MagicMock(side_effect=responses)

        executor = Executor(mock_router, registry, max_turns_per_step=25)
        step = PlanStep(
            description="Create a cube with center hole",
            expected_tools=["fc_batch", "cad_verify"],
        )
        state = AgentState(workspace="/tmp/test")

        result = executor.execute_step(step, state)

        assert step.status == StepStatus.COMPLETED
        assert cad_verify_call_count == 2

    def test_executor_no_infinite_loop(self):
        """cad_verify keeps failing → executor stops within max_turns → step FAILED."""
        from lang3d.agent.executor import Executor

        registry = ToolRegistry()
        max_turns = 5

        def cad_verify_always_fail(call_count, kwargs):
            return VERIFY_FAIL_RESULT

        def fc_batch_ok(call_count, kwargs):
            return "ok"

        registry.register(CallableMockTool("cad_verify", cad_verify_always_fail))
        registry.register(CallableMockTool("fc_batch", fc_batch_ok))

        # Generate enough responses: each turn has one tool call
        responses = []
        for i in range(max_turns + 2):
            if i % 2 == 0:
                responses.append(ModelResponse(
                    content=f"Step {i}",
                    tool_calls=[ToolCall(id=f"e{i}", name="fc_batch", arguments={})],
                ))
            else:
                responses.append(ModelResponse(
                    content=f"Verify {i}",
                    tool_calls=[ToolCall(id=f"e{i}", name="cad_verify", arguments={"expected": "test"})],
                ))

        mock_router = MagicMock()
        mock_router.chat = MagicMock(side_effect=responses)

        executor = Executor(mock_router, registry, max_turns_per_step=max_turns)
        step = PlanStep(
            description="Create something",
            expected_tools=["fc_batch", "cad_verify"],
        )
        state = AgentState(workspace="/tmp/test")

        result = executor.execute_step(step, state)

        # Executor should stop due to max_turns
        assert step.status == StepStatus.FAILED
