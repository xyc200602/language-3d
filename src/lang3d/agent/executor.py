"""Step executor - drives each plan step through the agent loop."""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

from ..models.base import Message, ModelResponse, ToolCall
from ..models.router import ModelRouter, TaskType
from ..tools.base import ToolRegistry
from .context import truncate_messages, truncate_tool_result
from .fix_strategy import classify_failure, check_convergence, extract_fix_commands, generate_fix_hint
from .state import AgentState, PlanStep, StepStatus


EXECUTOR_SYSTEM_PROMPT = """你是一个执行助手。你需要完成指定的步骤任务。

视觉感知策略：
- 建模后使用 fc_open_gui + cad_verify 验证模型
- 如果 cad_verify 返回 match=false，根据差异修正模型
- 使用 gui_* 工具操作 FreeCAD GUI（旋转、缩放、点击菜单）
- 使用 screen_analyze 快速查看当前屏幕状态

3D 建模规范：
- 使用 mm 作为单位
- 使用 fc_batch 一次完成多步建模
- 建模完成后必须验证

请使用工具完成任务，完成后简要说明结果。"""


class Executor:
    """Executes individual plan steps by driving the agent loop."""

    def __init__(
        self,
        router: ModelRouter,
        tool_registry: ToolRegistry,
        max_turns_per_step: int = 25,
    ) -> None:
        self.router = router
        self.tools = tool_registry
        self.max_turns = max_turns_per_step
        self._design_context: dict[str, Any] | None = None

    def set_design_context(self, context: dict[str, Any] | None) -> None:
        """Inject design context (subsystem info, interface constraints, etc.)."""
        self._design_context = context

    @staticmethod
    def _infer_step_type(step: PlanStep) -> str:
        """Infer step type from step's expected_tools and description."""
        desc_lower = step.description.lower()
        tools_lower = [t.lower() for t in step.expected_tools]

        # Check for simulation keywords
        if any(k in desc_lower for k in ("fea", "应力", "有限元", "结构分析")):
            return "simulation"
        if any(k in desc_lower for k in ("cfd", "流体", "流场")):
            return "cfd"
        if any(k in desc_lower for k in ("运动", "motion", "轨迹", "关节")):
            return "motion"
        if any(k in desc_lower for k in ("切片", "slice", "g-code", "gcode", "打印", "3d print")):
            return "slicing"

        # Check by tools
        if any("fc_" in t or "cad" in t or "part_" in t for t in tools_lower):
            return "modeling"
        if any("slice_" in t for t in tools_lower):
            return "slicing"
        if any(t in ("cad_verify", "vlm_analyze") for t in tools_lower):
            return "verification"
        if any(t in ("file_read", "file_write", "file_edit", "bash") for t in tools_lower):
            if not any("fc_" in t for t in tools_lower):
                return "file_ops"

        # Check description for modeling keywords
        if any(k in desc_lower for k in ("建模", "创建", "模型", "model")):
            return "modeling"
        if any(k in desc_lower for k in ("验证", "verify", "检查")):
            return "verification"

        return "general"

    def execute_step(
        self,
        step: PlanStep,
        state: AgentState,
        *,
        on_tool_call: Callable[[str, dict], None] | None = None,
        on_tool_result: Callable[[str, str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> str:
        """Execute a single plan step.

        Returns the final result string.
        """
        step.status = StepStatus.IN_PROGRESS
        step.attempts += 1
        # Explicitly initialize per-step state (thread safety)
        self._verify_fail_count = 0
        self._fix_history = []

        messages: list[Message] = [
            Message(
                role="user",
                content=f"请完成以下任务步骤：\n\n{step.description}"
                + (f"\n\n验证条件：{step.verification}" if step.verification else ""),
            )
        ]

        # Add reflection from previous attempts if any
        if step.attempts > 1:
            messages[0].content += f"\n\n（这是第 {step.attempts} 次尝试，之前的尝试失败了）"

        # Inject design context if available (subsystem, interface constraints, etc.)
        if self._design_context:
            ctx_lines = ["\n\n## 设计上下文"]
            for key, val in self._design_context.items():
                if isinstance(val, (list, dict)):
                    ctx_lines.append(f"- {key}: {val}")
                else:
                    ctx_lines.append(f"- {key}: {val}")
            messages[0].content += "\n".join(ctx_lines)

        step_type = self._infer_step_type(step)

        for turn in range(self.max_turns):
            tools = self.tools.get_relevant_definitions(step_type, extra_tools=step.expected_tools)
            response = self.router.chat(
                messages=messages,
                tools=tools,
                system=EXECUTOR_SYSTEM_PROMPT,
                task_type=TaskType.CODE_GENERATION,
                temperature=0.4,
            )

            if on_thinking and response.content:
                try:
                    on_thinking(response.content)
                except Exception:
                    pass

            # If no tool calls, the agent is done
            if not response.tool_calls:
                step.status = StepStatus.COMPLETED
                step.result = response.content
                return response.content

            # Process tool calls
            assistant_msg = Message(
                role="assistant",
                content=response.content,
                tool_calls=[
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in response.tool_calls
                ],
            )
            messages.append(assistant_msg)

            # Execute each tool call
            for tc in response.tool_calls:
                if on_tool_call:
                    try:
                        on_tool_call(tc.name, tc.arguments)
                    except Exception:
                        pass

                result = self.tools.execute(tc.name, **tc.arguments)
                state.add_tool_call(tc.name, tc.arguments, result)

                # Truncate large tool results
                result = truncate_tool_result(result)

                if on_tool_result:
                    try:
                        on_tool_result(tc.name, result)
                    except UnicodeEncodeError:
                        # VLM may return Unicode chars that can't encode in GBK
                        try:
                            on_tool_result(
                                tc.name,
                                result.encode("utf-8", errors="replace").decode("utf-8"),
                            )
                        except Exception:
                            pass
                    except Exception:
                        pass

                messages.append(
                    Message(
                        role="tool",
                        content=result,
                        tool_call_id=tc.id,
                    )
                )

                # Auto-fix: detect cad_verify MATCH:False and inject fix hint
                if tc.name == "cad_verify" and "MATCH: False" in result:
                    self._verify_fail_count = getattr(self, "_verify_fail_count", 0) + 1
                    max_verify_retries = 3  # same as core._run_direct default
                    if self._verify_fail_count <= max_verify_retries:
                        expected = tc.arguments.get("expected", "")
                        fix_ctx = classify_failure(result, expected)
                        fix_commands = extract_fix_commands(result)
                        self._fix_history = getattr(self, "_fix_history", [])
                        if check_convergence(self._fix_history, result):
                            fix_hint = (
                                "[系统提示] 检测到修复陷入循环（连续多次失败原因相似）。"
                                "请尝试完全不同的建模方法，或删除当前模型从头开始重建。"
                            )
                        else:
                            fix_hint = generate_fix_hint(fix_ctx, fix_commands=fix_commands)
                        self._fix_history.append(result)
                        messages.append(Message(role="user", content=fix_hint))
                elif tc.name == "cad_verify" and "MATCH: True" in result:
                    # Reset counter on success
                    self._verify_fail_count = 0

            # Apply sliding window when messages grow too large
            if len(messages) > 12:
                messages = truncate_messages(messages, keep_first=1, keep_last=2)

        # Max turns reached
        step.status = StepStatus.FAILED
        step.result = "Max turns reached without completion"
        return step.result
