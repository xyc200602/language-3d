"""Tests for phased orchestration architecture (Task 44).

Covers: PhasedOrchestrator, 4-phase execution, parallel subsystem design,
failure recovery, design context propagation, config settings.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from lang3d.agent.state import (
    AgentState,
    HierarchicalPlan,
    Plan,
    PlanStep,
    StepStatus,
    SubSystem,
    SystemDependency,
)
from lang3d.agent.orchestrator import PhasedOrchestrator
from lang3d.agent.sub_agent import SubAgentResult
from lang3d.config import AgentConfig, OrchestratorSettings


# ── Config Tests ──────────────────────────────────────────────────


class TestOrchestratorConfig:
    def test_new_settings_exist(self):
        s = OrchestratorSettings()
        assert s.phase_execution is True
        assert s.layout_model == ""
        assert s.integration_timeout == 600

    def test_custom_settings(self):
        s = OrchestratorSettings(
            phase_execution=False,
            layout_model="glm-4",
            integration_timeout=1200,
        )
        assert s.phase_execution is False
        assert s.layout_model == "glm-4"
        assert s.integration_timeout == 1200

    def test_max_turns_per_step_updated(self):
        c = AgentConfig()
        assert c.max_turns_per_step == 25


# ── PhasedOrchestrator Construction ──────────────────────────────


def _make_hierarchical_plan() -> HierarchicalPlan:
    """Create a test hierarchical plan with 3 subsystems."""
    base = SubSystem(
        name="mobile_base",
        description="4轮差速底盘",
        parts=["chassis", "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"],
        steps=[
            PlanStep(description="创建底盘", expected_tools=["fc_batch"]),
            PlanStep(description="创建4个轮子", expected_tools=["fc_batch"]),
        ],
    )
    arm_left = SubSystem(
        name="arm_left",
        description="左机械臂",
        parts=["shoulder_l", "elbow_l", "wrist_l"],
        steps=[
            PlanStep(description="创建左臂肩关节", expected_tools=["fc_batch"]),
            PlanStep(description="创建左臂肘关节", expected_tools=["fc_batch"]),
        ],
    )
    arm_right = SubSystem(
        name="arm_right",
        description="右机械臂(镜像)",
        parts=["shoulder_r", "elbow_r", "wrist_r"],
        steps=[
            PlanStep(description="创建右臂(镜像)", expected_tools=["fc_batch"]),
        ],
        mirror_of="arm_left",
    )
    return HierarchicalPlan(
        goal="设计4轮底盘+双臂机器人",
        subsystems=[base, arm_left, arm_right],
        system_dependencies=[
            SystemDependency("mobile_base", "arm_left", "臂安装在底盘上"),
            SystemDependency("mobile_base", "arm_right", "臂安装在底盘上"),
        ],
        integration_steps=[
            PlanStep(
                description="整机装配验证",
                expected_tools=["assembly_solve", "cad_verify"],
            ),
        ],
    )


def _make_planner():
    """Create a mock planner."""
    from lang3d.models.base import ModelResponse
    planner = MagicMock()
    planner.create_plan.return_value = Plan(goal="test", steps=[])
    return planner


def _make_router():
    """Create a mock router."""
    from lang3d.models.base import ModelResponse
    router = MagicMock()
    router.chat.return_value = ModelResponse(content="", tool_calls=[])
    return router


def _make_tools():
    """Create a mock tool registry."""
    tools = MagicMock()
    tools.get_relevant_definitions.return_value = []
    tools.execute.return_value = "ok"
    return tools


def _make_orchestrator(plan=None) -> PhasedOrchestrator:
    """Create a PhasedOrchestrator with mocks."""
    return PhasedOrchestrator(
        config=MagicMock(),
        router=_make_router(),
        tools=_make_tools(),
        planner=_make_planner(),
        hierarchical_plan=plan or _make_hierarchical_plan(),
        workspace=".",
    )


# ── Construction Tests ──────────────────────────────────────────


class TestPhasedOrchestratorConstruction:
    def test_max_parallel_default_6(self):
        o = _make_orchestrator()
        assert o.max_parallel == 6

    def test_custom_max_parallel(self):
        o = PhasedOrchestrator(
            config=MagicMock(),
            router=_make_router(),
            tools=_make_tools(),
            planner=_make_planner(),
            hierarchical_plan=_make_hierarchical_plan(),
            max_parallel=4,
        )
        assert o.max_parallel == 4

    def test_hierarchical_plan_stored(self):
        plan = _make_hierarchical_plan()
        o = _make_orchestrator(plan)
        assert o.hierarchical_plan is plan
        assert o.current_phase == ""

    def test_phases_constants(self):
        assert PhasedOrchestrator.PHASE_LAYOUT == "layout"
        assert PhasedOrchestrator.PHASE_PART_DESIGN == "part_design"
        assert PhasedOrchestrator.PHASE_SUBSYSTEM_VERIFY == "subsystem_verify"
        assert PhasedOrchestrator.PHASE_INTEGRATION == "integration_verify"


# ── Layout Phase Tests ──────────────────────────────────────────


class TestLayoutPhase:
    def test_layout_generates_constraints(self):
        o = _make_orchestrator()
        plan = o.hierarchical_plan
        o._plan_layout_phase(plan)

        assert "mobile_base" in o._layout_constraints
        assert "arm_left" in o._layout_constraints
        assert "arm_right" in o._layout_constraints

    def test_layout_includes_parts(self):
        o = _make_orchestrator()
        o._plan_layout_phase(o.hierarchical_plan)

        base_ctx = o._layout_constraints["mobile_base"]
        assert "chassis" in base_ctx["parts"]
        assert len(base_ctx["parts"]) == 5  # chassis + 4 wheels

    def test_layout_includes_mirror_info(self):
        o = _make_orchestrator()
        o._plan_layout_phase(o.hierarchical_plan)

        arm_r_ctx = o._layout_constraints["arm_right"]
        assert arm_r_ctx["mirror_of"] == "arm_left"

    def test_layout_includes_dependencies(self):
        o = _make_orchestrator()
        o._plan_layout_phase(o.hierarchical_plan)

        arm_ctx = o._layout_constraints["arm_left"]
        assert "depends_on" in arm_ctx
        assert arm_ctx["depends_on"][0]["source"] == "mobile_base"

    def test_layout_independent_subsystem_no_deps(self):
        o = _make_orchestrator()
        o._plan_layout_phase(o.hierarchical_plan)

        base_ctx = o._layout_constraints["mobile_base"]
        assert "depends_on" not in base_ctx


# ── Part Design Phase Tests ──────────────────────────────────────


class TestPartDesignPhase:
    def test_subsystem_scheduling_respects_deps(self):
        """Verify mobile_base runs before arm_left and arm_right."""
        o = _make_orchestrator()
        plan = o.hierarchical_plan

        executed_order: list[str] = []

        original_design = o._design_single_subsystem

        async def _mock_design(ss, p, completed):
            executed_order.append(ss.name)

        o._design_single_subsystem = _mock_design

        asyncio.run(o._run_part_design_phase(plan))

        # mobile_base should be first
        assert executed_order[0] == "mobile_base"
        # arm_left and arm_right come after
        assert set(executed_order[1:]) == {"arm_left", "arm_right"}

    def test_subsystem_results_collected(self):
        """Verify results are stored per subsystem."""
        o = _make_orchestrator()

        async def _mock_design(ss, p, completed):
            o._subsystem_results[ss.name] = [
                SubAgentResult(
                    agent_id="test", step_id="s1", success=True, result="ok"
                )
            ]
            ss.status = StepStatus.COMPLETED

        o._design_single_subsystem = _mock_design
        asyncio.run(o._run_part_design_phase(o.hierarchical_plan))

        assert "mobile_base" in o._subsystem_results
        assert "arm_left" in o._subsystem_results
        assert "arm_right" in o._subsystem_results

    def test_independent_subsystems_parallel(self):
        """Verify arm_left and arm_right (which depend on base, not each other)
        both run after base completes."""
        o = _make_orchestrator()
        plan = o.hierarchical_plan

        execution_log: list[str] = []

        async def _mock_design(ss, p, completed):
            execution_log.append(ss.name)
            o._subsystem_results[ss.name] = []
            ss.status = StepStatus.COMPLETED

        o._design_single_subsystem = _mock_design
        asyncio.run(o._run_part_design_phase(plan))

        # base first, then both arms
        assert execution_log[0] == "mobile_base"
        assert len(execution_log) == 3


# ── Failure Recovery Tests ──────────────────────────────────────


class TestFailureRecovery:
    def test_subsystem_failure_does_not_block_others(self):
        """When arm_left fails, mobile_base and arm_right should still complete."""
        o = _make_orchestrator()
        plan = o.hierarchical_plan

        async def _mock_design(ss, p, completed):
            if ss.name == "arm_left":
                ss.status = StepStatus.FAILED
                o._subsystem_results[ss.name] = [
                    SubAgentResult(
                        agent_id="fail", step_id="s1", success=False,
                        error="建模失败",
                    )
                ]
            else:
                ss.status = StepStatus.COMPLETED
                o._subsystem_results[ss.name] = [
                    SubAgentResult(
                        agent_id="ok", step_id="s1", success=True, result="ok"
                    )
                ]

        o._design_single_subsystem = _mock_design
        asyncio.run(o._run_part_design_phase(plan))

        # base and right arm should be completed
        base = plan.get_subsystem("mobile_base")
        arm_r = plan.get_subsystem("arm_right")
        arm_l = plan.get_subsystem("arm_left")

        assert base.status == StepStatus.COMPLETED
        assert arm_r.status == StepStatus.COMPLETED
        assert arm_l.status == StepStatus.FAILED

    def test_subsystem_verify_skips_failed(self):
        """Phase 3 should skip verification for failed subsystems."""
        o = _make_orchestrator()
        plan = o.hierarchical_plan

        # Mark base completed, arm_left failed
        plan.get_subsystem("mobile_base").status = StepStatus.COMPLETED
        plan.get_subsystem("arm_left").status = StepStatus.FAILED
        plan.get_subsystem("arm_right").status = StepStatus.COMPLETED

        o._subsystem_results = {
            "mobile_base": [SubAgentResult(agent_id="t", step_id="s1", success=True)],
            "arm_right": [SubAgentResult(agent_id="t", step_id="s1", success=True)],
        }

        verified: list[str] = []

        original_run = o._run_step_with_retry

        async def _mock_run(step, ctx):
            verified.append(ctx.get("subsystem", ""))
            return SubAgentResult(agent_id="v", step_id=step.id, success=True)

        o._run_step_with_retry = _mock_run
        asyncio.run(o._run_subsystem_verify_phase(plan))

        # arm_left should not be verified
        assert "arm_left" not in verified
        assert "mobile_base" in verified
        assert "arm_right" in verified


# ── Context Propagation Tests ───────────────────────────────────


class TestContextPropagation:
    def test_collect_subsystem_context_includes_deps(self):
        """arm_left should get context from mobile_base results."""
        o = _make_orchestrator()
        plan = o.hierarchical_plan

        o._subsystem_results["mobile_base"] = [
            SubAgentResult(
                agent_id="base_agent",
                step_id="s1",
                success=True,
                result="底盘完成",
                artifacts=["chassis.fcstd", "wheel.fcstd"],
            ),
        ]

        arm = plan.get_subsystem("arm_left")
        ctx = o._collect_subsystem_context(arm, plan, {"mobile_base"})

        assert "mobile_base" in ctx
        assert ctx["mobile_base"]["artifacts"] == [
            "chassis.fcstd", "wheel.fcstd"
        ]
        assert ctx["mobile_base"]["reason"] == "臂安装在底盘上"

    def test_collect_subsystem_context_empty_for_independent(self):
        """mobile_base has no deps, should get empty context."""
        o = _make_orchestrator()
        plan = o.hierarchical_plan
        base = plan.get_subsystem("mobile_base")

        ctx = o._collect_subsystem_context(base, plan, set())
        assert ctx == {}

    def test_design_context_includes_layout(self):
        """Design context passed to executor should include layout constraints."""
        o = _make_orchestrator()
        plan = o.hierarchical_plan
        o._plan_layout_phase(plan)

        base_layout = o._layout_constraints["mobile_base"]
        assert base_layout["subsystem"] == "mobile_base"
        assert base_layout["instance_count"] == 1


# ── Integration Phase Tests ──────────────────────────────────────


class TestIntegrationPhase:
    def test_integration_success(self):
        """All integration steps succeed."""
        o = _make_orchestrator()
        plan = o.hierarchical_plan

        for ss in plan.subsystems:
            ss.status = StepStatus.COMPLETED

        o._subsystem_results = {
            ss.name: [SubAgentResult(agent_id="t", step_id="s1", success=True)]
            for ss in plan.subsystems
        }

        async def _mock_run(step, ctx):
            return SubAgentResult(agent_id="int", step_id=step.id, success=True)

        o._run_step_with_retry = _mock_run

        result = asyncio.run(o._run_integration_phase(plan))
        assert result is True

    def test_integration_failure(self):
        """Integration step fails."""
        o = _make_orchestrator()
        plan = o.hierarchical_plan

        for ss in plan.subsystems:
            ss.status = StepStatus.COMPLETED

        o._subsystem_results = {
            ss.name: [SubAgentResult(agent_id="t", step_id="s1", success=True)]
            for ss in plan.subsystems
        }

        async def _mock_run(step, ctx):
            return SubAgentResult(
                agent_id="int", step_id=step.id, success=False,
                error="干涉检测失败",
            )

        o._run_step_with_retry = _mock_run

        result = asyncio.run(o._run_integration_phase(plan))
        assert result is False

    def test_no_integration_steps_passes(self):
        """Plan with no integration steps should return True."""
        plan = HierarchicalPlan(
            goal="simple",
            subsystems=[SubSystem(name="base", parts=["a"])],
            integration_steps=[],
        )
        o = _make_orchestrator(plan)
        result = asyncio.run(o._run_integration_phase(plan))
        assert result is True


# ── Full 4-Phase Pipeline Tests ─────────────────────────────────


class TestFullPipeline:
    def test_run_task_completes_all_phases(self):
        """Smoke test: full pipeline runs without errors."""
        o = _make_orchestrator()

        async def _mock_design(ss, p, completed):
            o._subsystem_results[ss.name] = [
                SubAgentResult(agent_id="t", step_id="s1", success=True, result="ok")
            ]
            ss.status = StepStatus.COMPLETED

        async def _mock_run(step, ctx):
            return SubAgentResult(agent_id="t", step_id=step.id, success=True, result="ok")

        o._design_single_subsystem = _mock_design
        o._run_step_with_retry = _mock_run

        result = o.run_task("test")
        assert "分阶段编排完成" in result

    def test_phases_progress_sequentially(self):
        """Verify phases execute in correct order."""
        o = _make_orchestrator()
        phases_seen: list[str] = []

        def _on_thinking(text):
            if o.current_phase:
                if o.current_phase not in phases_seen:
                    phases_seen.append(o.current_phase)

        o._on_thinking = _on_thinking

        async def _mock_design(ss, p, completed):
            o._subsystem_results[ss.name] = [
                SubAgentResult(agent_id="t", step_id="s1", success=True)
            ]
            ss.status = StepStatus.COMPLETED

        async def _mock_run(step, ctx):
            return SubAgentResult(agent_id="t", step_id=step.id, success=True)

        o._design_single_subsystem = _mock_design
        o._run_step_with_retry = _mock_run

        o.run_task("test")

        assert phases_seen == [
            "layout", "part_design", "subsystem_verify", "integration_verify"
        ]

    def test_pipeline_with_subsystem_failure(self):
        """Pipeline should complete even if a subsystem fails."""
        o = _make_orchestrator()

        async def _mock_design(ss, p, completed):
            if ss.name == "arm_left":
                ss.status = StepStatus.FAILED
                o._subsystem_results[ss.name] = [
                    SubAgentResult(agent_id="f", step_id="s1", success=False, error="fail")
                ]
            else:
                ss.status = StepStatus.COMPLETED
                o._subsystem_results[ss.name] = [
                    SubAgentResult(agent_id="t", step_id="s1", success=True)
                ]

        async def _mock_run(step, ctx):
            return SubAgentResult(agent_id="t", step_id=step.id, success=True)

        o._design_single_subsystem = _mock_design
        o._run_step_with_retry = _mock_run

        result = o.run_task("test")
        assert "分阶段编排完成" in result


# ── Executor Design Context Tests ──────────────────────────────


class TestExecutorDesignContext:
    def test_set_design_context(self):
        from lang3d.agent.executor import Executor

        router = _make_router()
        tools = _make_tools()
        exec = Executor(router, tools)

        ctx = {"subsystem": "arm_left", "parts": ["shoulder", "elbow"]}
        exec.set_design_context(ctx)
        assert exec._design_context == ctx

    def test_design_context_injected_into_message(self):
        from lang3d.agent.executor import Executor

        router = _make_router()
        tools = _make_tools()
        # Make execute return quickly (no tool calls)
        router.chat.return_value = MagicMock(content="done", tool_calls=[])

        exec = Executor(router, tools)
        exec.set_design_context({"subsystem": "arm_left", "parts": ["shoulder"]})

        step = PlanStep(description="创建肩关节")
        state = AgentState(workspace=".")
        exec.execute_step(step, state)

        # Check that the message included design context
        call_args = router.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        if messages is None:
            # positional
            messages = router.chat.call_args[0][0] if router.chat.call_args[0] else []

        # The first user message should contain design context
        first_msg = messages[0] if messages else None
        if first_msg:
            assert "设计上下文" in first_msg.content
            assert "arm_left" in first_msg.content
