"""Integration tests for multi-agent orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lang3d.agent.assembly_verifier import AssemblyVerifier
from lang3d.agent.dag import TaskDAG
from lang3d.agent.message_bus import AgentMessage, MessageBus
from lang3d.agent.orchestrator import OrchestratorAgent
from lang3d.agent.shared_registry import SharedToolRegistry
from lang3d.agent.state import Plan, PlanStep, StepStatus
from lang3d.agent.sub_agent import SubAgent, SubAgentResult, SubAgentRole
from lang3d.knowledge.mechanics import ROBOTIC_ARM_ASSEMBLY, Assembly, Joint, Part


def _mock_llm_response(content="Done", tool_calls=None):
    """Create a mock LLM response."""
    resp = MagicMock()
    resp.content = content
    resp.tool_calls = tool_calls or []
    return resp


def _mock_orchestrator() -> OrchestratorAgent:
    """Create an orchestrator with mocked LLM."""
    mock_router = MagicMock()
    mock_tools = MagicMock()
    mock_tools.get_all_definitions.return_value = []
    mock_tools.execute.return_value = "ok"

    mock_planner = MagicMock()
    mock_reflector = MagicMock()
    mock_reflector.reflect.return_value = "Try different approach"

    return OrchestratorAgent(
        config=MagicMock(),
        router=mock_router,
        tools=mock_tools,
        planner=mock_planner,
        reflector=mock_reflector,
        workspace="/tmp/test_workspace",
        max_parallel=2,
        max_retries=2,
    )


class TestOrchestrationDAGDecomposition:
    """Test that the orchestrator correctly decomposes tasks into DAGs."""

    def test_robotic_arm_plan_creates_dag(self):
        orch = _mock_orchestrator()

        # Simulate planner returning steps for robotic arm parts
        steps = [
            PlanStep(description=f"创建 {p.name}", expected_tools=["fc_batch", "cad_verify"])
            for p in ROBOTIC_ARM_ASSEMBLY.parts[:4]
        ]
        steps.append(PlanStep(description="装配验证", expected_tools=["bash"]))
        plan = Plan(goal="设计机械臂", steps=steps)

        # Add dependencies based on assembly joints
        base_plate = steps[0]
        servo_holder = steps[1]
        base_joint = steps[2]
        shoulder_link = steps[3]

        base_joint.dependencies = [base_plate.id]
        shoulder_link.dependencies = [base_joint.id]

        dag = TaskDAG.from_plan(plan, assembly=ROBOTIC_ARM_ASSEMBLY)
        groups = dag.parallel_groups()

        # Should have multiple waves (at least 2, given dependencies)
        assert len(groups) >= 2

    def test_dag_with_assembly_joints_infers_deps(self):
        """Dependencies should be inferred from assembly joints."""
        base = Part(name="base", category="structural", description="base", dimensions={})
        arm = Part(name="arm", category="structural", description="arm", dimensions={})
        hand = Part(name="hand", category="structural", description="hand", dimensions={})

        assembly = Assembly(
            name="test",
            parts=[base, arm, hand],
            joints=[
                Joint("fixed", "base", "arm", description="base-arm"),
                Joint("revolute", "arm", "hand", description="arm-hand"),
            ],
        )

        steps = [
            PlanStep(description="Create base part"),
            PlanStep(description="Create arm part"),
            PlanStep(description="Create hand part"),
        ]

        plan = Plan(goal="build test", steps=steps)
        dag = TaskDAG.from_plan(plan, assembly=assembly)

        groups = dag.parallel_groups()
        # base should be in wave 0, arm in wave 1 (depends on base), hand in wave 2 (depends on arm)
        assert len(groups) >= 2


class TestParallelWaveExecution:
    """Test that waves execute correctly."""

    def test_independent_steps_run_in_parallel(self):
        """Independent steps should be in the same wave."""
        a = PlanStep(description="task A")
        b = PlanStep(description="task B")
        c = PlanStep(description="task C")

        dag = TaskDAG()
        dag.add_step(a)
        dag.add_step(b)
        dag.add_step(c)

        groups = dag.parallel_groups()
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_sequential_steps_run_in_separate_waves(self):
        a = PlanStep(description="A")
        b = PlanStep(description="B")
        c = PlanStep(description="C")

        dag = TaskDAG()
        dag.add_step(a)
        dag.add_step(b, dependencies=[a.id])
        dag.add_step(c, dependencies=[b.id])

        groups = dag.parallel_groups()
        assert len(groups) == 3

    def test_mixed_parallelism(self):
        a = PlanStep(description="A")
        b = PlanStep(description="B")
        c = PlanStep(description="C")
        d = PlanStep(description="D")

        dag = TaskDAG()
        dag.add_step(a)
        dag.add_step(b)
        dag.add_step(c, dependencies=[a.id, b.id])
        dag.add_step(d, dependencies=[c.id])

        groups = dag.parallel_groups()
        assert len(groups) == 3
        assert len(groups[0]) == 2  # A, B
        assert len(groups[1]) == 1  # C
        assert len(groups[2]) == 1  # D


class TestContextPassing:
    """Test that context is correctly passed between agents."""

    def test_context_includes_predecessor_results(self):
        orch = _mock_orchestrator()

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
            result="base_plate.FCStd created",
            artifacts=["/tmp/base_plate.FCStd"],
        )
        orch._results[step_a.id] = result_a

        # Collect context for step_b
        node_b = dag.get_node(step_b.id)
        context = orch._collect_context(node_b)

        assert step_a.id in context
        assert context[step_a.id]["result"] == "base_plate.FCStd created"
        assert "/tmp/base_plate.FCStd" in context[step_a.id]["artifacts"]

    def test_multiple_dependencies_context(self):
        orch = _mock_orchestrator()

        a = PlanStep(description="A")
        b = PlanStep(description="B")
        c = PlanStep(description="C")

        dag = TaskDAG()
        dag.add_step(a)
        dag.add_step(b)
        dag.add_step(c, dependencies=[a.id, b.id])
        orch._dag = dag

        orch._results[a.id] = SubAgentResult(
            agent_id="s1", step_id=a.id, success=True, result="A done"
        )
        orch._results[b.id] = SubAgentResult(
            agent_id="s2", step_id=b.id, success=True, result="B done"
        )

        node_c = dag.get_node(c.id)
        context = orch._collect_context(node_c)

        assert a.id in context
        assert b.id in context
        assert context[a.id]["result"] == "A done"
        assert context[b.id]["result"] == "B done"


class TestFailureRetry:
    """Test failure handling and retry."""

    def test_dag_mark_failed_cascades(self):
        """When a step fails, dependents should be skipped."""
        a = PlanStep(description="A")
        b = PlanStep(description="B")
        c = PlanStep(description="C")

        dag = TaskDAG()
        dag.add_step(a)
        dag.add_step(b, dependencies=[a.id])
        dag.add_step(c, dependencies=[b.id])

        skipped = dag.mark_failed(a.id)
        assert b.id in skipped
        assert c.id in skipped
        assert dag.get_node(b.id).step.status == StepStatus.SKIPPED
        assert dag.get_node(c.id).step.status == StepStatus.SKIPPED

    def test_partial_failure_doesnt_affect_independent(self):
        a = PlanStep(description="A")
        b = PlanStep(description="B")
        c = PlanStep(description="C")

        dag = TaskDAG()
        dag.add_step(a)
        dag.add_step(b)
        dag.add_step(c, dependencies=[b.id])

        skipped = dag.mark_failed(a.id)
        assert len(skipped) == 0
        assert b.status == StepStatus.PENDING
        assert c.status == StepStatus.PENDING


class TestAssemblyVerificationIntegration:
    """Test assembly verification with the robotic arm."""

    def test_verify_with_no_files(self, tmp_path):
        verifier = AssemblyVerifier()
        result = verifier.verify_assembly(ROBOTIC_ARM_ASSEMBLY, tmp_path)

        assert not result.overall_pass
        assert len(result.part_checks) == len(ROBOTIC_ARM_ASSEMBLY.parts)
        assert all(not pc.exists for pc in result.part_checks)

    def test_verify_with_all_files(self, tmp_path):
        # Create dummy files for all parts
        for part in ROBOTIC_ARM_ASSEMBLY.parts:
            (tmp_path / f"{part.name}.fcstd").write_text("dummy")

        verifier = AssemblyVerifier()
        result = verifier.verify_assembly(ROBOTIC_ARM_ASSEMBLY, tmp_path)

        assert all(pc.exists for pc in result.part_checks)
        # May still fail on fit checks
        report = AssemblyVerifier.generate_assembly_report(result)
        assert "零件检查" in report


class TestMessageBusIntegration:
    """Test message bus with sub-agent communication."""

    def test_artifacts_propagation(self):
        bus = MessageBus()

        # Simulate agent publishing artifacts
        bus.publish(AgentMessage(
            sender="sub-base",
            type="artifact",
            payload="/tmp/base_plate.FCStd",
        ))
        bus.publish(AgentMessage(
            sender="sub-arm",
            type="artifact",
            payload="/tmp/shoulder_link.FCStd",
        ))

        artifacts = bus.get_artifacts()
        assert len(artifacts) == 2
        assert "/tmp/base_plate.FCStd" in artifacts

    def test_tool_call_tracking(self):
        bus = MessageBus()

        bus.publish(AgentMessage(
            sender="sub-1",
            type="tool_call",
            payload={"name": "fc_batch", "args": {"steps": 3}},
        ))
        bus.publish(AgentMessage(
            sender="sub-2",
            type="tool_call",
            payload={"name": "cad_verify", "args": {"expected": "box"}},
        ))

        sub1_tools = bus.get_messages(agent_id="sub-1", type="tool_call")
        assert len(sub1_tools) == 1
        assert sub1_tools[0].payload["name"] == "fc_batch"


class TestSharedRegistryConcurrency:
    """Test that SharedToolRegistry handles concurrent access."""

    def test_concurrent_executions(self):
        import threading
        from lang3d.tools.base import ToolRegistry

        inner = ToolRegistry()

        # Register a simple tool
        class CountTool:
            name = "count"
            description = "counter"

            def get_definition(self):
                from lang3d.models.base import ToolDefinition
                return ToolDefinition(name="count", description="counter", parameters={})

            def execute(self, **kwargs):
                return "counted"

        inner.register(CountTool())
        shared = SharedToolRegistry(inner)

        results = []
        errors = []

        def worker():
            try:
                r = shared.execute("count")
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 20
        assert all(r == "counted" for r in results)
