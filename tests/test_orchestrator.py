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


class TestTryGetAssembly:
    def test_robotic_arm(self):
        orch = _make_orchestrator()
        assembly = orch._try_get_assembly("设计一个3自由度机械臂")
        assert assembly is not None
        assert assembly.name == "3-DOF Robotic Arm"

    def test_no_match(self):
        orch = _make_orchestrator()
        assembly = orch._try_get_assembly("写一个Python脚本")
        assert assembly is None


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
