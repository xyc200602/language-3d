"""Tests for hierarchical planning architecture (Task 43).

Covers: SubSystem, HierarchicalPlan, symmetry detection,
task type detection, and plan data structures.
"""

import json
import pytest

from lang3d.agent.state import (
    AgentState,
    HierarchicalPlan,
    Plan,
    PlanStep,
    StepStatus,
    SubSystem,
    SystemDependency,
)
from lang3d.agent.planner import (
    COMPLEX_ROBOT_KEYWORDS,
    SYMMETRY_PATTERNS,
    Planner,
)


# ── StepStatus ──────────────────────────────────────────────────


class TestStepStatus:
    def test_new_statuses_exist(self):
        assert StepStatus.BLOCKED == "blocked"
        assert StepStatus.WAITING == "waiting"

    def test_all_statuses(self):
        expected = {"pending", "in_progress", "completed", "failed", "skipped", "blocked", "waiting"}
        actual = {s.value for s in StepStatus}
        assert actual == expected


# ── SubSystem ───────────────────────────────────────────────────


class TestSubSystem:
    def test_defaults(self):
        ss = SubSystem()
        assert ss.name == ""
        assert ss.parts == []
        assert ss.joints == []
        assert ss.steps == []
        assert ss.status == StepStatus.PENDING
        assert ss.mirror_of == ""
        assert ss.instance_count == 1

    def test_with_data(self):
        ss = SubSystem(
            name="mobile_base",
            description="4轮差速底盘",
            parts=["chassis", "wheel"],
            instance_count=4,
        )
        assert ss.name == "mobile_base"
        assert len(ss.parts) == 2
        assert ss.instance_count == 4

    def test_progress(self):
        ss = SubSystem(
            name="test",
            steps=[
                PlanStep(status=StepStatus.COMPLETED),
                PlanStep(status=StepStatus.COMPLETED),
                PlanStep(status=StepStatus.PENDING),
            ],
        )
        assert ss.progress() == (2, 3)

    def test_progress_empty(self):
        ss = SubSystem(name="empty")
        assert ss.progress() == (0, 0)


# ── SystemDependency ───────────────────────────────────────────


class TestSystemDependency:
    def test_creation(self):
        dep = SystemDependency(
            source="mobile_base",
            target="arm_left",
            reason="臂安装在底盘上",
        )
        assert dep.source == "mobile_base"
        assert dep.target == "arm_left"
        assert dep.reason == "臂安装在底盘上"


# ── HierarchicalPlan ───────────────────────────────────────────


class TestHierarchicalPlan:
    def _make_plan(self) -> HierarchicalPlan:
        base = SubSystem(
            name="mobile_base",
            parts=["chassis", "wheel"],
            steps=[
                PlanStep(description="创建底盘"),
                PlanStep(description="创建轮子"),
            ],
        )
        arm = SubSystem(
            name="arm_left",
            parts=["shoulder", "elbow"],
            steps=[PlanStep(description="创建肩关节")],
            mirror_of="",
        )
        arm_r = SubSystem(
            name="arm_right",
            parts=["shoulder", "elbow"],
            steps=[PlanStep(description="创建右臂(镜像)")],
            mirror_of="arm_left",
        )
        return HierarchicalPlan(
            goal="设计4轮底盘+双臂机器人",
            subsystems=[base, arm, arm_r],
            system_dependencies=[
                SystemDependency("mobile_base", "arm_left", "安装接口"),
                SystemDependency("mobile_base", "arm_right", "安装接口"),
            ],
            integration_steps=[
                PlanStep(description="整机装配验证"),
            ],
        )

    def test_get_subsystem(self):
        plan = self._make_plan()
        assert plan.get_subsystem("mobile_base") is not None
        assert plan.get_subsystem("arm_left") is not None
        assert plan.get_subsystem("nonexistent") is None

    def test_total_parts(self):
        plan = self._make_plan()
        assert plan.total_parts() == 6  # 2 + 2 + 2

    def test_all_steps(self):
        plan = self._make_plan()
        steps = plan.all_steps()
        assert len(steps) == 5  # 2 + 1 + 1 + 1 (integration)

    def test_progress(self):
        plan = self._make_plan()
        # Mark first step of first subsystem as completed
        plan.subsystems[0].steps[0].status = StepStatus.COMPLETED
        assert plan.progress() == (1, 5)

    def test_to_flat_plan(self):
        plan = self._make_plan()
        flat = plan.to_flat_plan()
        assert isinstance(flat, Plan)
        assert flat.goal == plan.goal
        assert len(flat.steps) == 5

    def test_system_dependencies(self):
        plan = self._make_plan()
        assert len(plan.system_dependencies) == 2
        assert plan.system_dependencies[0].source == "mobile_base"
        assert plan.system_dependencies[1].target == "arm_right"


# ── Task Type Detection ────────────────────────────────────────


class TestTaskTypeDetection:
    @pytest.mark.parametrize("task", [
        "设计一个4轮差速底盘移动机器人",
        "带有双臂的移动平台",
        "工控机安装在底盘上",
        "设计一个AGV巡检机器人",
        "4 wheel mobile robot with dual arm",
        "differential drive mobile base",
    ])
    def test_complex_robot_detected(self, task):
        assert Planner._detect_task_type(task) == "complex_robot"

    @pytest.mark.parametrize("task", [
        "创建一个30x30x10的带孔平板",
        "创建一个方块",
    ])
    def test_simple_task_not_complex(self, task):
        assert Planner._detect_task_type(task) != "complex_robot"

    def test_assembly_still_detected(self):
        assert Planner._detect_task_type("装配一个机械臂") == "assembly"

    def test_part_usage_still_detected(self):
        assert Planner._detect_task_type("生成M4螺钉标准件") == "part_usage"

    def test_slicing_still_detected(self):
        assert Planner._detect_task_type("切片分析这个STL") == "slicing"


# ── Symmetry Detection ────────────────────────────────────────


class TestSymmetryDetection:
    def test_4_wheel_detected(self):
        result = Planner.detect_symmetry("设计一个4轮差速底盘机器人")
        instances = [s for s in result if s["type"] == "instance"]
        assert len(instances) >= 1
        wheel = instances[0]
        assert wheel["count"] == 4
        assert "wheel_fl" in wheel["targets"]

    def test_dual_arm_detected(self):
        result = Planner.detect_symmetry("带双臂的移动平台")
        mirrors = [s for s in result if s["type"] == "mirror"]
        assert len(mirrors) >= 1
        arm = mirrors[0]
        assert arm["source"] == "arm_left"
        assert arm["target"] == "arm_right"

    def test_no_symmetry_simple_task(self):
        result = Planner.detect_symmetry("创建一个方块")
        assert result == []

    def test_dual_leg_detected(self):
        result = Planner.detect_symmetry("设计一个双腿机器人")
        mirrors = [s for s in result if s["type"] == "mirror"]
        assert len(mirrors) >= 1

    def test_2_wheel_detected(self):
        result = Planner.detect_symmetry("2轮平衡车")
        instances = [s for s in result if s["type"] == "instance"]
        assert len(instances) >= 1
        assert instances[0]["count"] == 2


# ── AgentState with HierarchicalPlan ───────────────────────────


class TestAgentStateWithHierarchy:
    def test_plan_accepts_hierarchical(self, tmp_path):
        state = AgentState(workspace=tmp_path)
        hp = HierarchicalPlan(
            goal="test",
            subsystems=[SubSystem(name="base", parts=["a", "b"])],
        )
        state.plan = hp
        assert isinstance(state.plan, HierarchicalPlan)
        assert state.plan.total_parts() == 2

    def test_plan_still_accepts_flat(self, tmp_path):
        state = AgentState(workspace=tmp_path)
        state.plan = Plan(goal="test", steps=[PlanStep(description="step1")])
        assert isinstance(state.plan, Plan)
        assert len(state.plan.steps) == 1


# ── Parsing Tests ──────────────────────────────────────────────


class TestHierarchicalParsing:
    """Test the parsing of LLM-style JSON responses."""

    def _make_planner(self):
        """Create a Planner with a mock router."""
        from unittest.mock import MagicMock
        from lang3d.models.base import ModelResponse

        router = MagicMock()
        router.chat.return_value = ModelResponse(
            content="",
            tool_calls=[],
        )
        return Planner(router)

    def test_parse_valid_hierarchical_json(self):
        planner = self._make_planner()
        response = json.dumps({
            "subsystems": [
                {
                    "name": "mobile_base",
                    "description": "4轮底盘",
                    "parts": ["chassis", "wheel"],
                    "joints": [],
                    "steps": [
                        {"description": "创建底盘", "expected_tools": ["fc_batch"], "verification": "文件存在"},
                    ],
                    "mirror_of": "",
                    "instance_count": 1,
                },
                {
                    "name": "arm_right",
                    "description": "右臂",
                    "parts": ["shoulder_r"],
                    "steps": [],
                    "mirror_of": "arm_left",
                    "instance_count": 1,
                },
            ],
            "system_dependencies": [
                {"source": "mobile_base", "target": "arm_right", "reason": "安装"},
            ],
            "integration_steps": [
                {"description": "整机装配", "expected_tools": ["assembly_solve"], "verification": "无干涉"},
            ],
        })

        plan = planner._parse_hierarchical_response("test", response, [])
        assert isinstance(plan, HierarchicalPlan)
        assert len(plan.subsystems) == 2
        assert plan.subsystems[1].mirror_of == "arm_left"
        assert len(plan.system_dependencies) == 1
        assert len(plan.integration_steps) == 1

    def test_parse_with_markdown_fences(self):
        planner = self._make_planner()
        response = "```json\n" + json.dumps({
            "subsystems": [{"name": "base", "parts": ["a"], "steps": []}],
            "system_dependencies": [],
            "integration_steps": [],
        }) + "\n```"

        plan = planner._parse_hierarchical_response("test", response, [])
        assert len(plan.subsystems) == 1
        assert plan.subsystems[0].name == "base"

    def test_parse_invalid_json_fallback(self):
        planner = self._make_planner()
        response = "This is not JSON at all"

        plan = planner._parse_hierarchical_response("test", response, [])
        assert len(plan.subsystems) == 1
        assert plan.subsystems[0].name == "main"

    def test_symmetry_applied_to_parsed_plan(self):
        planner = self._make_planner()
        response = json.dumps({
            "subsystems": [
                {"name": "arm_right", "parts": ["shoulder_r"], "steps": [], "mirror_of": ""},
            ],
            "system_dependencies": [],
            "integration_steps": [],
        })
        symmetry = [{"source": "arm_left", "target": "arm_right", "type": "mirror", "count": 1}]

        plan = planner._parse_hierarchical_response("test", response, symmetry)
        assert plan.subsystems[0].mirror_of == "arm_left"

    def test_empty_subsystems_handled(self):
        planner = self._make_planner()
        response = json.dumps({
            "subsystems": [],
            "system_dependencies": [],
            "integration_steps": [],
        })

        plan = planner._parse_hierarchical_response("test", response, [])
        assert len(plan.subsystems) == 0
        assert plan.total_parts() == 0


# ── Integration: _should_use_hierarchical ──────────────────────


class TestShouldUseHierarchical:
    def test_complex_robot_triggers(self):
        from lang3d.agent.core import Agent
        assert Agent._should_use_hierarchical("设计4轮差速底盘移动机器人") is True
        assert Agent._should_use_hierarchical("dual arm mobile manipulator") is True

    def test_simple_task_does_not_trigger(self):
        from lang3d.agent.core import Agent
        assert Agent._should_use_hierarchical("创建一个30x30的方块") is False
        assert Agent._should_use_hierarchical("导出STL文件") is False
