"""Lightweight sub-agent for parallel task execution."""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from ..models.router import ModelRouter, TaskType
from ..tools.base import ToolRegistry
from .executor import Executor
from .state import AgentState, PlanStep, StepStatus

logger = logging.getLogger(__name__)


# Known CAD file extensions that should be cleaned up on step failure
_CAD_EXTENSIONS = (".fcstd", ".step", ".stl", ".obj", ".py")


def cleanup_artifacts(artifacts: list[str], workspace: str) -> None:
    """Move artifact files produced by a failed step to a backup directory.

    Files are moved to ``<workspace>/.lang3d/backups/`` with a timestamp
    prefix instead of being permanently deleted.  Backups older than 7
    days are purged automatically.
    """
    ws = Path(workspace)
    backup_dir = ws / ".lang3d" / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    ts_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

    for artifact_path in artifacts:
        p = Path(artifact_path)
        # Only back up files with known CAD extensions
        if p.suffix.lower() not in _CAD_EXTENSIONS:
            continue
        # Resolve relative paths against workspace
        if not p.is_absolute():
            p = ws / p
        try:
            if p.exists():
                backup_name = f"{ts_prefix}_{p.name}"
                shutil.move(str(p), str(backup_dir / backup_name))
                logger.info("Backed up artifact: %s -> %s", p, backup_dir / backup_name)
        except OSError:
            logger.warning("Failed to back up artifact: %s", p)

    # Auto-purge backups older than 7 days
    cutoff = datetime.now() - timedelta(days=7)
    for old_file in backup_dir.iterdir():
        try:
            if old_file.is_file() and old_file.stat().st_mtime < cutoff.timestamp():
                old_file.unlink()
                logger.info("Purged old backup: %s", old_file)
        except OSError:
            pass


class SubAgentRole(str, Enum):
    """Role specialization for sub-agents."""

    MODELING = "modeling"
    VISION = "vision"
    GUI = "gui"
    VERIFICATION = "verification"
    GENERAL = "general"


@dataclass
class SubAgentResult:
    """Result from a sub-agent execution."""

    agent_id: str
    step_id: str
    success: bool
    result: str = ""
    artifacts: list[str] = field(default_factory=list)
    error: str = ""
    tool_history: list[dict[str, Any]] = field(default_factory=list)


_ROLE_PROMPTS: dict[SubAgentRole, str] = {
    SubAgentRole.MODELING: (
        "你是一个专门的 3D 建模子 Agent。你的任务是使用 FreeCAD 工具完成特定的零件建模。\n"
        "规范：\n"
        "- 使用 mm 作为单位\n"
        "- 使用 fc_batch 一次性完成多步建模\n"
        "- 建模完成后必须调用 cad_verify 验证\n"
        "- 如果验证不通过，根据 DIFFERENCES 修正模型\n"
        "- 将生成的模型文件保存到工作目录\n"
        "- 完成后报告生成的文件路径"
    ),
    SubAgentRole.VISION: (
        "你是一个专门的视觉分析子 Agent。你的任务是使用 VLM 工具进行视觉验证和分析。\n"
        "规范：\n"
        "- 使用 cad_verify 或 vlm_analyze 验证模型\n"
        "- 详细描述观察到的内容\n"
        "- 报告任何不一致之处"
    ),
    SubAgentRole.GUI: (
        "你是一个专门的 GUI 操作子 Agent。你的任务是通过 GUI 自动化操作 CAD 软件。\n"
        "规范：\n"
        "- 使用 gui_* 工具操作界面\n"
        "- 使用 vlm_locate 定位 UI 元素\n"
        "- 精确操作，每步确认结果"
    ),
    SubAgentRole.VERIFICATION: (
        "你是一个专门的验证子 Agent。你的任务是检查建模结果是否满足要求。\n"
        "规范：\n"
        "- 检查文件是否存在\n"
        "- 验证模型尺寸和形状\n"
        "- 检查配合公差\n"
        "- 生成验证报告"
    ),
    SubAgentRole.GENERAL: (
        "你是一个通用的任务执行子 Agent。按照指示完成指定的任务步骤。"
    ),
}


class SubAgent:
    """Lightweight agent that executes a single plan step."""

    def __init__(
        self,
        agent_id: str | None = None,
        role: SubAgentRole = SubAgentRole.GENERAL,
        router: ModelRouter | None = None,
        tools: ToolRegistry | None = None,
        workspace: str | None = None,
        max_turns: int = 10,
    ) -> None:
        self.agent_id = agent_id or f"sub-{uuid.uuid4().hex[:6]}"
        self.role = role
        self.router = router
        self.tools = tools
        self.workspace = workspace or "."
        self.max_turns = max_turns
        self._state = AgentState(workspace=self.workspace)

        # Callbacks
        self._on_tool_call: Callable[[str, str, dict], None] | None = None
        self._on_tool_result: Callable[[str, str, str], None] | None = None
        self._on_thinking: Callable[[str, str], None] | None = None

    def on_tool_call(self, callback: Callable[[str, str, dict], None]) -> None:
        """Set tool call callback: (agent_id, tool_name, args)."""
        self._on_tool_call = callback

    def on_tool_result(self, callback: Callable[[str, str, str], None]) -> None:
        """Set tool result callback: (agent_id, tool_name, result)."""
        self._on_tool_result = callback

    def on_thinking(self, callback: Callable[[str, str], None]) -> None:
        """Set thinking callback: (agent_id, text)."""
        self._on_thinking = callback

    def get_system_prompt(self) -> str:
        """Get the system prompt based on the agent's role."""
        return _ROLE_PROMPTS.get(self.role, _ROLE_PROMPTS[SubAgentRole.GENERAL])

    def execute(
        self,
        step: PlanStep,
        context: dict[str, Any] | None = None,
    ) -> SubAgentResult:
        """Execute a plan step synchronously.

        Args:
            step: The plan step to execute.
            context: Optional context from predecessor agents, mapping
                     agent_id -> {result, artifacts, ...}.

        Returns:
            SubAgentResult with execution outcome.
        """
        if self.router is None or self.tools is None:
            return SubAgentResult(
                agent_id=self.agent_id,
                step_id=step.id,
                success=False,
                error="SubAgent not initialized with router and tools",
            )

        step.status = StepStatus.IN_PROGRESS
        step.attempts += 1
        if not step.assigned_agent:
            step.assigned_agent = self.agent_id

        # Inject context into step description for the executor
        original_desc = step.description
        if context:
            context_text = self._format_context(context)
            step.description = f"{original_desc}\n\n{context_text}"

        try:
            executor = Executor(
                self.router,
                self.tools,
                max_turns_per_step=self.max_turns,
            )

            # Execute with callbacks that include agent_id
            def _on_tool_call(name: str, args: dict) -> None:
                if self._on_tool_call:
                    self._on_tool_call(self.agent_id, name, args)

            def _on_tool_result(name: str, result: str) -> None:
                if self._on_tool_result:
                    self._on_tool_result(self.agent_id, name, result)

            def _on_thinking(text: str) -> None:
                if self._on_thinking:
                    self._on_thinking(self.agent_id, text)

            result_text = executor.execute_step(
                step,
                self._state,
                on_tool_call=_on_tool_call,
                on_tool_result=_on_tool_result,
                on_thinking=_on_thinking,
            )
        finally:
            # Always restore original description even on exception
            step.description = original_desc

        success = step.status == StepStatus.COMPLETED

        # Extract artifacts from tool history
        artifacts = self._extract_artifacts()

        return SubAgentResult(
            agent_id=self.agent_id,
            step_id=step.id,
            success=success,
            result=result_text,
            artifacts=artifacts,
            error="" if success else result_text,
            tool_history=[
                {"name": t.get("name", "unknown"), "arguments": t.get("arguments", {})}
                for t in self._state.tool_history
            ],
        )

    async def execute_async(
        self,
        step: PlanStep,
        context: dict[str, Any] | None = None,
    ) -> SubAgentResult:
        """Execute a plan step asynchronously using asyncio.to_thread."""
        return await asyncio.to_thread(self.execute, step, context)

    def _format_context(self, context: dict[str, Any]) -> str:
        """Format context from predecessor agents into text for injection."""
        parts = ["## 前置任务结果"]
        for agent_id, agent_result in context.items():
            if isinstance(agent_result, dict):
                desc = agent_result.get("description", "")
                res = agent_result.get("result", "")
                arts = agent_result.get("artifacts", [])
                parts.append(f"### Agent {agent_id} ({desc})")
                if res:
                    parts.append(f"结果：{res[:500]}")
                if arts:
                    parts.append(f"生成的文件：{', '.join(arts)}")
            elif isinstance(agent_result, SubAgentResult):
                parts.append(f"### Agent {agent_id}")
                parts.append(f"结果：{agent_result.result[:500]}")
                if agent_result.artifacts:
                    parts.append(f"生成的文件：{', '.join(agent_result.artifacts)}")
        return "\n".join(parts)

    def _extract_artifacts(self) -> list[str]:
        """Extract file paths generated during execution from tool history."""
        artifacts: list[str] = []
        for entry in self._state.tool_history:
            result = entry.get("result", "")
            # Look for file paths in results
            args = entry.get("arguments", {})
            if "path" in args:
                artifacts.append(args["path"])
            elif "file_path" in args:
                artifacts.append(args["file_path"])
            elif "output_path" in args:
                artifacts.append(args["output_path"])
            # Check result text for common file patterns
            for line in result.split("\n"):
                line = line.strip()
                if line.endswith((".fcstd", ".step", ".stl", ".obj", ".py")):
                    # Extract path from line
                    for ext in (".fcstd", ".step", ".stl", ".obj", ".py"):
                        if ext in line:
                            idx = line.find(ext)
                            path = line[: idx + len(ext)].split()[-1]
                            artifacts.append(path)
                            break
        return list(dict.fromkeys(artifacts))  # Deduplicate preserving order
