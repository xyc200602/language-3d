"""Tests for agent core components."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from lang3d.agent.state import AgentState, Plan, PlanStep, StepStatus
from lang3d.config import Config, ModelConfig


def test_plan_step_status():
    step = PlanStep(description="test", status=StepStatus.PENDING)
    assert step.status == StepStatus.PENDING
    step.status = StepStatus.COMPLETED
    assert step.status == StepStatus.COMPLETED


def test_plan_current_step():
    plan = Plan(
        goal="test",
        steps=[
            PlanStep(description="step 1", status=StepStatus.COMPLETED),
            PlanStep(description="step 2", status=StepStatus.PENDING),
            PlanStep(description="step 3", status=StepStatus.PENDING),
        ],
    )
    current = plan.current_step()
    assert current is not None
    assert current.description == "step 2"


def test_plan_progress():
    plan = Plan(
        goal="test",
        steps=[
            PlanStep(description="step 1", status=StepStatus.COMPLETED),
            PlanStep(description="step 2", status=StepStatus.COMPLETED),
            PlanStep(description="step 3", status=StepStatus.PENDING),
        ],
    )
    completed, total = plan.progress()
    assert completed == 2
    assert total == 3


def test_agent_state_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        state = AgentState(workspace=tmpdir)
        state.plan = Plan(
            goal="test goal",
            steps=[
                PlanStep(description="step 1", status=StepStatus.COMPLETED),
                PlanStep(description="step 2", status=StepStatus.IN_PROGRESS),
            ],
        )
        state.add_tool_call("bash", {"command": "echo hello"}, "hello\n")

        save_path = Path(tmpdir) / "state.json"
        state.save(save_path)

        loaded = AgentState.load(save_path)
        assert loaded.session_id == state.session_id
        assert loaded.plan is not None
        assert loaded.plan.goal == "test goal"
        assert len(loaded.plan.steps) == 2
        assert len(loaded.tool_history) == 1


def test_config_defaults():
    config = Config()
    assert config.default_backend == "glm"
    assert config.agent.max_turns == 50


def test_config_from_dict():
    config = Config(
        glm=ModelConfig(api_key="test-key", base_url="https://example.com", model="glm-4"),
        default_backend="glm",
    )
    assert config.glm.api_key == "test-key"
    assert config.glm.model == "glm-4"
