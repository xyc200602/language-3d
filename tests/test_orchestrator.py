"""Tests for OrchestratorAgent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from lang3d.agent.dag import TaskDAG
from lang3d.agent.orchestrator import OrchestratorAgent
from lang3d.agent.state import Plan, PlanStep, StepStatus
from lang3d.agent.sub_agent import SubAgentResult
from lang3d.knowledge.mechanics import ROBOTIC_ARM_ASSEMBLY


def _make_orchestrator() -> OrchestratorAgent:
    """Create an OrchestratorAgent with mocked dependencies."""
    mock_config = MagicMock()
    mock_router = MagicMock()
    mock_tools = MagicMock()
    mock_planner = MagicMock()
    mock_reflector = MagicMock()
    mock_reflector.reflect.return_value = "Try again with different approach"

    return OrchestratorAgent(
        config=mock_config,
        router=mock_router,
        tools=mock_tools,
        planner=mock_planner,
        reflector=mock_reflector,
        workspace="/tmp/test_workspace",
        max_parallel=2,
        max_retries=2,
    )


class TestShouldOrchestrate:
    def test_too_few_steps(self):
        orch = _make_orchestrator()
        plan = Plan(
            goal="test",
            steps=[PlanStep(description=f"step {i}") for i in range(3)],
        )
        assert not orch.should_orchestrate("test", plan)

    def test_enough_modeling_steps(self):
        orch = _make_orchestrator()
        plan = Plan(
            goal="build arm",
            steps=[
                PlanStep(description="build base", expected_tools=["fc_batch"]),
                PlanStep(description="build arm", expected_tools=["fc_batch"]),
                PlanStep(description="build joint", expected_tools=["fc_batch"]),
                PlanStep(description="assemble", expected_tools=["bash"]),
            ],
        )
        assert orch.should_orchestrate("build arm", plan)

    def test_not_enough_modeling(self):
        orch = _make_orchestrator()
        plan = Plan(
            goal="test",
            steps=[
                PlanStep(description="step 1", expected_tools=["bash"]),
                PlanStep(description="step 2", expected_tools=["file_write"]),
                PlanStep(description="step 3", expected_tools=["python_exec"]),
                PlanStep(description="step 4", expected_tools=["bash"]),
            ],
        )
        assert not orch.should_orchestrate("test", plan)


class TestCollectContext:
    def test_collects_from_results(self):
        orch = _make_orchestrator()
        step_a = PlanStep(description="build base")
        step_b = PlanStep(description="build arm")

        dag = TaskDAG()
        dag.add_step(step_a)
        dag.add_step(step_b, dependencies=[step_a.id])
        orch._dag = dag

        # Simulate step_a completed
        result_a = SubAgentResult(
            agent_id="sub-1",
            step_id=step_a.id,
            success=True,
            result="base created",
            artifacts=["/tmp/base.FCStd"],
        )
        orch._results[step_a.id] = result_a

        node_b = dag.get_node(step_b.id)
        context = orch._collect_context(node_b)

        assert step_a.id in context
        assert context[step_a.id]["result"] == "base created"
        assert "/tmp/base.FCStd" in context[step_a.id]["artifacts"]


class TestSpawnSubAgent:
    def test_spawns_with_correct_role(self):
        from lang3d.agent.dag import DAGNode

        orch = _make_orchestrator()
        node = DAGNode(
            step=PlanStep(description="model part"),
            agent_role="modeling",
        )
        sub = orch._spawn_sub_agent(node)
        assert sub.role.value == "modeling"

    def test_default_role(self):
        from lang3d.agent.dag import DAGNode

        orch = _make_orchestrator()
        node = DAGNode(
            step=PlanStep(description="generic task"),
            agent_role="general",
        )
        sub = orch._spawn_sub_agent(node)
        assert sub.role.value == "general"


class TestNoPlaceholderAssemblyVerification:
    """The orchestrator must NOT carry placeholder assembly verification.

    The prior design ran a fake verification against a hardcoded
    ROBOTIC_ARM_ASSEMBLY teaching fixture for any task containing "机械臂",
    which did not reflect the user's actual task and was misleading. Real
    assembly verification lives in AssemblyPipeline. These tests guard
    against the placeholder creeping back in.
    """

    def test_no_try_get_assembly_method(self):
        orch = _make_orchestrator()
        assert not hasattr(orch, "_try_get_assembly")

    def test_no_run_assembly_verification_method(self):
        orch = _make_orchestrator()
        assert not hasattr(orch, "_run_assembly_verification")


class TestRunTask:
    def test_orchestrated_run(self):
        """Test full orchestrated run with mocked LLM."""
        orch = _make_orchestrator()

        # Create a simple plan
        step_a = PlanStep(description="build base", expected_tools=["fc_batch"])
        step_b = PlanStep(description="build arm", expected_tools=["fc_batch"])
        plan = Plan(goal="build robot", steps=[step_a, step_b])

        orch.planner.create_plan = MagicMock(return_value=plan)

        # Mock LLM to return completion immediately
        mock_response = MagicMock()
        mock_response.content = "Done"
        mock_response.tool_calls = []
        orch.router.chat.return_value = mock_response

        thinking_log = []
        orch.on_thinking(lambda t: thinking_log.append(t))

        result = orch.run_task("设计一个简单的2零件组件")

        assert "编排任务完成" in result or "步骤成功" in result


class TestDAGProperty:
    def test_dag_none_initially(self):
        orch = _make_orchestrator()
        assert orch.dag is None

    def test_active_agents_initially_empty(self):
        orch = _make_orchestrator()
        assert len(orch.active_agents) == 0


class TestMaxRetriesZeroNoCrash:
    """BUG 6: sub_agent undefined when max_retries=0."""

    def test_max_retries_zero_no_crash(self):
        """When max_retries=0, _run_node_with_retry should still return a result."""
        from lang3d.agent.dag import DAGNode
        import asyncio

        orch = _make_orchestrator()
        orch.max_retries = 0

        step = PlanStep(description="test step")
        node = DAGNode(step=step)

        result = asyncio.run(orch._run_node_with_retry(node, {}))
        assert result is not None
        assert result.success is False
        assert result.error == "No attempts made"


class TestNoDoubleCountingAttempts:
    """BUG 7: attempts should not be double-counted by the orchestrator."""

    def test_no_double_counting_attempts(self):
        """The orchestrator should NOT set step.attempts = attempt + 1.
        Only SubAgent.execute() manages the attempt counter."""
        from lang3d.agent.dag import DAGNode
        import asyncio

        orch = _make_orchestrator()
        orch.max_retries = 2

        step = PlanStep(description="test step")
        node = DAGNode(step=step)

        # Simulate SubAgent that increments step.attempts like the real one
        fail_result = SubAgentResult(
            agent_id="test-agent", step_id=step.id,
            success=False, error="fail",
        )

        class FakeSubAgent:
            agent_id = "test-agent"
            async def execute_async(self, s, ctx):
                # Mimic SubAgent.execute() which does step.attempts += 1
                s.attempts += 1
                return fail_result

        orch._spawn_sub_agent = MagicMock(return_value=FakeSubAgent())
        orch.reflector.reflect = MagicMock(return_value="try again")

        result = asyncio.run(orch._run_node_with_retry(node, {}))

        # With max_retries=2, FakeSubAgent.execute_async is called 2 times
        # Each call increments step.attempts by 1 → total should be exactly 2
        # If orchestrator still double-counted, it would be higher
        assert step.attempts == 2


class TestOrchestratorUsesProvidedPlan:
    """BUG 5: Orchestrator should use an existing plan when provided."""

    def test_orchestrator_uses_provided_plan(self):
        orch = _make_orchestrator()

        step_a = PlanStep(description="custom step A")
        step_b = PlanStep(description="custom step B")
        provided_plan = Plan(goal="custom goal", steps=[step_a, step_b])

        # Should NOT call planner.create_plan when plan is provided
        orch.planner.create_plan = MagicMock(side_effect=AssertionError("Should not be called"))

        mock_response = MagicMock()
        mock_response.content = "Done"
        mock_response.tool_calls = []
        orch.router.chat.return_value = mock_response

        result = orch.run_task("some task", plan=provided_plan)

        # Verify planner.create_plan was NOT called (it would have raised)
        orch.planner.create_plan.assert_not_called()
