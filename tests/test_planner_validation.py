"""Tests for planner validation and few-shot examples."""

from __future__ import annotations

from lang3d.agent.planner import PLANNER_EXAMPLES, Planner
from lang3d.agent.state import Plan, PlanStep


class TestPlannerExamples:
    """Few-shot example existence tests."""

    def test_assembly_example_exists(self):
        assert "assembly" in PLANNER_EXAMPLES
        assert "装配" in PLANNER_EXAMPLES["assembly"] or "assembly" in PLANNER_EXAMPLES["assembly"].lower()

    def test_single_part_example_exists(self):
        assert "single_part" in PLANNER_EXAMPLES
        assert "fc_batch" in PLANNER_EXAMPLES["single_part"]


class TestDetectTaskType:
    """Task type detection tests."""

    def test_assembly_detection(self):
        assert Planner._detect_task_type("装配一个3自由度机械臂") == "assembly"

    def test_assembly_english(self):
        assert Planner._detect_task_type("Create an assembly of robotic arm") == "assembly"

    def test_single_part_detection(self):
        assert Planner._detect_task_type("创建一个30x30x30mm的方块") == "single_part"

    def test_single_part_default(self):
        assert Planner._detect_task_type("write a python script") == "single_part"


class TestValidatePlan:
    """Plan validation and auto-fix tests."""

    def test_adds_cad_verify_to_fc_batch_steps(self):
        step = PlanStep(
            description="Create a box",
            expected_tools=["fc_batch"],
            verification="File exists",
        )
        plan = Plan(goal="test", steps=[step])
        Planner._validate_plan(plan)
        assert "cad_verify" in step.expected_tools

    def test_does_not_duplicate_cad_verify(self):
        step = PlanStep(
            description="Create a box",
            expected_tools=["fc_batch", "cad_verify"],
            verification="File exists",
        )
        plan = Plan(goal="test", steps=[step])
        Planner._validate_plan(plan)
        assert step.expected_tools.count("cad_verify") == 1

    def test_adds_verification_to_modeling_steps(self):
        step = PlanStep(
            description="使用 fc_batch 创建零件",
            expected_tools=["fc_batch", "cad_verify"],
            verification="",
        )
        plan = Plan(goal="test", steps=[step])
        Planner._validate_plan(plan)
        assert step.verification != ""
        assert "cad_verify" in step.verification

    def test_preserves_existing_verification(self):
        step = PlanStep(
            description="Create model",
            expected_tools=["fc_batch", "cad_verify"],
            verification="Custom verification check",
        )
        plan = Plan(goal="test", steps=[step])
        Planner._validate_plan(plan)
        assert step.verification == "Custom verification check"

    def test_non_modeling_step_unchanged(self):
        step = PlanStep(
            description="Read configuration file",
            expected_tools=["file_read"],
            verification="File content not empty",
        )
        plan = Plan(goal="test", steps=[step])
        Planner._validate_plan(plan)
        assert "cad_verify" not in step.expected_tools
        assert step.verification == "File content not empty"
