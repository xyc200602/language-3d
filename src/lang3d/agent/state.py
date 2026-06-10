"""Agent state management and persistence."""

from __future__ import annotations

import json
import os
import tempfile
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
    BLOCKED = "blocked"
    WAITING = "waiting"


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
        relevant = [s for s in self.steps if s.status != StepStatus.SKIPPED]
        completed = sum(1 for s in relevant if s.status == StepStatus.COMPLETED)
        return completed, len(relevant)


# --- Hierarchical Plan (Phase E: Task 43) ---


@dataclass
class SubSystem:
    """A subsystem within a complex robot design.

    Examples: mobile_base, arm_left, arm_right, ipc_mount, sensor_tower.
    Each subsystem has its own isolated set of plan steps.
    """

    name: str = ""
    description: str = ""
    parts: list[str] = field(default_factory=list)
    joints: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    steps: list[PlanStep] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    # Symmetry: if non-empty, this subsystem is a mirrored/instanced copy
    mirror_of: str = ""  # name of the source subsystem (e.g. "arm_left")
    instance_count: int = 1  # how many identical copies (e.g. 4 for wheels)
    # Interface constraints with other subsystems
    interface_points: dict[str, Any] = field(default_factory=dict)

    def progress(self) -> tuple[int, int]:
        relevant = [s for s in self.steps if s.status != StepStatus.SKIPPED]
        completed = sum(1 for s in relevant if s.status == StepStatus.COMPLETED)
        return completed, len(relevant)


@dataclass
class SystemDependency:
    """Dependency between two subsystems (e.g. arm depends on base for mounting)."""

    source: str = ""  # subsystem that must complete first
    target: str = ""  # subsystem that depends on source
    reason: str = ""  # why the dependency exists (e.g. "mounting interface")


@dataclass
class HierarchicalPlan:
    """A hierarchical plan that organizes steps into subsystems.

    Used for complex robot designs with 15+ parts that need to be decomposed
    into manageable subsystems (mobile base, arms, electronics, etc.).
    """

    goal: str = ""
    subsystems: list[SubSystem] = field(default_factory=list)
    system_dependencies: list[SystemDependency] = field(default_factory=list)
    integration_steps: list[PlanStep] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_subsystem(self, name: str) -> SubSystem | None:
        for ss in self.subsystems:
            if ss.name == name:
                return ss
        return None

    def all_steps(self) -> list[PlanStep]:
        """Collect all steps across all subsystems and integration."""
        steps: list[PlanStep] = []
        for ss in self.subsystems:
            steps.extend(ss.steps)
        steps.extend(self.integration_steps)
        return steps

    def current_step(self) -> PlanStep | None:
        """Return the first PENDING or IN_PROGRESS step across all subsystems."""
        for step in self.all_steps():
            if step.status in (StepStatus.PENDING, StepStatus.IN_PROGRESS):
                return step
        return None

    def total_parts(self) -> int:
        return sum(len(ss.parts) for ss in self.subsystems)

    def progress(self) -> tuple[int, int]:
        completed = 0
        total = 0
        for ss in self.subsystems:
            c, t = ss.progress()
            completed += c
            total += t
        c, t = len([s for s in self.integration_steps if s.status == StepStatus.COMPLETED]), len(self.integration_steps)
        completed += c
        total += t
        return completed, total

    def to_flat_plan(self) -> Plan:
        """Convert to a flat Plan for backward compatibility."""
        return Plan(goal=self.goal, steps=self.all_steps())


@dataclass
class CheckpointInfo:
    """Metadata for a saved checkpoint."""

    checkpoint_id: str
    label: str
    step_id: str
    step_description: str
    created_at: str
    path: str  # path to the checkpoint JSON file


_MAX_CHECKPOINTS = 20


class AgentState:
    """Manages the state of an agent session."""

    def __init__(self, workspace: str | Path | None = None) -> None:
        self.session_id: str = uuid.uuid4().hex[:12]
        self.created_at: str = datetime.now().isoformat()
        self.workspace = Path(workspace) if workspace else Path.cwd()
        self.plan: Plan | HierarchicalPlan | None = None
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
        plan_data = None
        if self.plan:
            if isinstance(self.plan, HierarchicalPlan):
                plan_data = self._serialize_hierarchical_plan(self.plan)
            else:
                plan_data = {
                    "type": "flat",
                    "goal": self.plan.goal,
                    "steps": [self._serialize_step(s) for s in self.plan.steps],
                }
        data = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "workspace": str(self.workspace),
            "plan": plan_data,
            "conversation": self.conversation[-100:],
            "screenshots": self.screenshots[-50:],
            "tool_history": self.tool_history[-100:],  # Keep last 100
            "metadata": self.metadata,
        }
        data_str = json.dumps(data, indent=2, ensure_ascii=False)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(save_path.parent), suffix=".tmp", prefix=".lang3d_state_"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(data_str)
            os.replace(tmp_path, str(save_path))
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def checkpoint(self, label: str, step_id: str = "", step_description: str = "") -> CheckpointInfo:
        """Save a timestamped snapshot of the current state.

        Snapshots are stored under ``<workspace>/.lang3d/checkpoints/``.
        Older checkpoints are pruned automatically when the count exceeds
        ``_MAX_CHECKPOINTS``.
        """
        cp_dir = self.workspace / ".lang3d" / "checkpoints"
        cp_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cp_id = f"cp_{ts}"
        cp_path = cp_dir / f"{cp_id}.json"

        self.save(path=cp_path)

        info = CheckpointInfo(
            checkpoint_id=cp_id,
            label=label,
            step_id=step_id,
            step_description=step_description,
            created_at=datetime.now().isoformat(),
            path=str(cp_path),
        )

        # Write sidecar metadata (atomic)
        meta_path = cp_dir / f"{cp_id}_meta.json"
        meta_str = json.dumps({
            "checkpoint_id": info.checkpoint_id,
            "label": info.label,
            "step_id": info.step_id,
            "step_description": info.step_description,
            "created_at": info.created_at,
            "path": info.path,
        }, indent=2, ensure_ascii=False)
        tmp_fd, tmp_meta = tempfile.mkstemp(
            dir=str(cp_dir), suffix=".tmp", prefix=".lang3d_meta_"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(meta_str)
            os.replace(tmp_meta, str(meta_path))
        except Exception:
            Path(tmp_meta).unlink(missing_ok=True)
            raise

        # Auto-prune oldest checkpoints
        metas = sorted(cp_dir.glob("*_meta.json"), key=lambda p: p.name)
        if len(metas) > _MAX_CHECKPOINTS:
            for old_meta in metas[: len(metas) - _MAX_CHECKPOINTS]:
                old_cp = Path(old_meta.stem.replace("_meta", "") + ".json")
                old_meta.unlink(missing_ok=True)
                (cp_dir / old_cp).unlink(missing_ok=True)

        return info

    def list_checkpoints(self) -> list[CheckpointInfo]:
        """Return checkpoints newest-first for the current workspace."""
        cp_dir = self.workspace / ".lang3d" / "checkpoints"
        if not cp_dir.exists():
            return []

        results: list[CheckpointInfo] = []
        for meta_path in sorted(cp_dir.glob("*_meta.json"), reverse=True):
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                results.append(CheckpointInfo(
                    checkpoint_id=data["checkpoint_id"],
                    label=data.get("label", ""),
                    step_id=data.get("step_id", ""),
                    step_description=data.get("step_description", ""),
                    created_at=data.get("created_at", ""),
                    path=data.get("path", ""),
                ))
            except (json.JSONDecodeError, KeyError):
                continue
        return results

    @classmethod
    def restore_checkpoint(cls, workspace: str | Path, checkpoint_id: str) -> AgentState:
        """Load a specific checkpoint as a new AgentState.

        Args:
            workspace: The workspace directory containing ``.lang3d/checkpoints/``.
            checkpoint_id: The checkpoint identifier (e.g. ``cp_20250610_143000_123456``).

        Returns:
            A fresh AgentState restored from the checkpoint snapshot.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        cp_path = Path(workspace) / ".lang3d" / "checkpoints" / f"{checkpoint_id}.json"
        if not cp_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")
        return cls.load(path=cp_path)

    @staticmethod
    def _serialize_step(s: PlanStep) -> dict[str, Any]:
        return {
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

    @staticmethod
    def _serialize_hierarchical_plan(plan: HierarchicalPlan) -> dict[str, Any]:
        return {
            "type": "hierarchical",
            "goal": plan.goal,
            "subsystems": [
                {
                    "name": ss.name,
                    "description": ss.description,
                    "parts": ss.parts,
                    "joints": ss.joints,
                    "constraints": ss.constraints,
                    "steps": [AgentState._serialize_step(s) for s in ss.steps],
                    "status": ss.status.value,
                    "mirror_of": ss.mirror_of,
                    "instance_count": ss.instance_count,
                    "interface_points": ss.interface_points,
                }
                for ss in plan.subsystems
            ],
            "system_dependencies": [
                {"source": d.source, "target": d.target, "reason": d.reason}
                for d in plan.system_dependencies
            ],
            "integration_steps": [AgentState._serialize_step(s) for s in plan.integration_steps],
        }

    @classmethod
    def load(cls, path: Path) -> AgentState:
        """Load state from JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        state = cls(workspace=data.get("workspace"))
        state.session_id = data.get("session_id", state.session_id)
        state.created_at = data.get("created_at", state.created_at)
        state.tool_history = data.get("tool_history", [])
        state.metadata = data.get("metadata", {})
        state.conversation = data.get("conversation", [])
        state.screenshots = data.get("screenshots", [])

        if plan_data := data.get("plan"):
            plan_type = plan_data.get("type", "flat")
            if plan_type == "hierarchical":
                state.plan = cls._deserialize_hierarchical_plan(plan_data)
            else:
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

    @staticmethod
    def _deserialize_step(s: dict[str, Any]) -> PlanStep:
        return PlanStep(
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

    @classmethod
    def _deserialize_hierarchical_plan(cls, data: dict[str, Any]) -> HierarchicalPlan:
        subsystems = []
        for ss_data in data.get("subsystems", []):
            ss = SubSystem(
                name=ss_data["name"],
                description=ss_data.get("description", ""),
                parts=ss_data.get("parts", []),
                joints=ss_data.get("joints", []),
                constraints=ss_data.get("constraints", {}),
                steps=[cls._deserialize_step(s) for s in ss_data.get("steps", [])],
                status=StepStatus(ss_data.get("status", "pending")),
                mirror_of=ss_data.get("mirror_of", ""),
                instance_count=ss_data.get("instance_count", 1),
                interface_points=ss_data.get("interface_points", {}),
            )
            subsystems.append(ss)

        system_deps = [
            SystemDependency(
                source=d.get("source", ""),
                target=d.get("target", ""),
                reason=d.get("reason", ""),
            )
            for d in data.get("system_dependencies", [])
        ]

        return HierarchicalPlan(
            goal=data.get("goal", ""),
            subsystems=subsystems,
            system_dependencies=system_deps,
            integration_steps=[cls._deserialize_step(s) for s in data.get("integration_steps", [])],
        )
