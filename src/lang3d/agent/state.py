"""Agent state management and persistence."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in an execution plan."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = ""
    expected_tools: list[str] = field(default_factory=list)
    verification: str = ""
    status: StepStatus = StepStatus.PENDING
    result: str = ""
    attempts: int = 0
    dependencies: list[str] = field(default_factory=list)
    assigned_agent: str = ""


@dataclass
class Plan:
    """An execution plan with multiple steps."""

    goal: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def current_step(self) -> PlanStep | None:
        for step in self.steps:
            if step.status in (StepStatus.PENDING, StepStatus.IN_PROGRESS):
                return step
        return None

    def progress(self) -> tuple[int, int]:
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        return completed, len(self.steps)


class AgentState:
    """Manages the state of an agent session."""

    def __init__(self, workspace: str | Path | None = None) -> None:
        self.session_id: str = uuid.uuid4().hex[:12]
        self.created_at: str = datetime.now().isoformat()
        self.workspace = Path(workspace) if workspace else Path.cwd()
        self.plan: Plan | None = None
        self.conversation: list[dict[str, Any]] = []
        self.tool_history: list[dict[str, Any]] = []
        self.screenshots: list[str] = []
        self.metadata: dict[str, Any] = {}

    def add_tool_call(self, name: str, args: dict, result: str) -> None:
        self.tool_history.append({
            "name": name,
            "arguments": args,
            "result": result[:2000],  # Truncate large results
            "timestamp": datetime.now().isoformat(),
        })

    def add_screenshot(self, path: str) -> None:
        self.screenshots.append(path)

    def save(self, path: Path | None = None) -> None:
        """Save state to JSON file."""
        save_path = path or self.workspace / ".lang3d_state.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "workspace": str(self.workspace),
            "plan": {
                "goal": self.plan.goal,
                "steps": [
                    {
                        "id": s.id,
                        "description": s.description,
                        "expected_tools": s.expected_tools,
                        "verification": s.verification,
                        "status": s.status.value,
                        "result": s.result,
                        "attempts": s.attempts,
                        "dependencies": s.dependencies,
                        "assigned_agent": s.assigned_agent,
                    }
                    for s in self.plan.steps
                ],
            }
            if self.plan
            else None,
            "tool_history": self.tool_history[-100:],  # Keep last 100
            "metadata": self.metadata,
        }
        save_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> AgentState:
        """Load state from JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        state = cls(workspace=data.get("workspace"))
        state.session_id = data["session_id"]
        state.created_at = data["created_at"]
        state.tool_history = data.get("tool_history", [])
        state.metadata = data.get("metadata", {})

        if plan_data := data.get("plan"):
            state.plan = Plan(
                goal=plan_data["goal"],
                steps=[
                    PlanStep(
                        id=s["id"],
                        description=s["description"],
                        expected_tools=s.get("expected_tools", []),
                        verification=s.get("verification", ""),
                        status=StepStatus(s["status"]),
                        result=s.get("result", ""),
                        attempts=s.get("attempts", 0),
                        dependencies=s.get("dependencies", []),
                        assigned_agent=s.get("assigned_agent", ""),
                    )
                    for s in plan_data["steps"]
                ],
            )

        return state
