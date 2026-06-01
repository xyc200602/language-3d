"""Step executor - drives each plan step through the agent loop."""

from __future__ import annotations

from typing import Any, Callable

from ..models.base import Message, ModelResponse, ToolCall
from ..models.router import ModelRouter, TaskType
from ..tools.base import ToolRegistry
from .state import AgentState, PlanStep, StepStatus


EXECUTOR_SYSTEM_PROMPT = """你是一个执行助手。你需要完成指定的步骤任务。

工具分类：
- 文件操作：file_read, file_write, file_edit, file_search, file_glob, list_dir
- 命令执行：bash, python_exec
- 屏幕截取：screen_capture, window_capture, list_windows
- VLM 视觉分析：vlm_analyze, screen_analyze, window_analyze, cad_verify
- FreeCAD 建模：fc_batch (多步建模), fc_open_gui, fc_close_gui, fc_set_camera, fc_*
- GUI 自动化：gui_click, gui_type, gui_hotkey, gui_press_key, gui_screenshot, gui_drag, gui_scroll, gui_mouse_pos
- 仿真分析：fea_run, fea_visualize, fea_vlm_analyze, interference_check, tolerance_analysis, motion_sim, motion_range, motion_trajectory, motion_vlm_analyze
- CFD 流体分析：cfd_run, cfd_vlm_analyze

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
        max_turns_per_step: int = 10,
    ) -> None:
        self.router = router
        self.tools = tool_registry
        self.max_turns = max_turns_per_step

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

        for turn in range(self.max_turns):
            tools = self.tools.get_all_definitions()
            response = self.router.chat(
                messages=messages,
                tools=tools,
                system=EXECUTOR_SYSTEM_PROMPT,
                task_type=TaskType.CODE_GENERATION,
                temperature=0.4,
            )

            if on_thinking and response.content:
                on_thinking(response.content)

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
                    on_tool_call(tc.name, tc.arguments)

                result = self.tools.execute(tc.name, **tc.arguments)
                state.add_tool_call(tc.name, tc.arguments, result)

                if on_tool_result:
                    on_tool_result(tc.name, result)

                messages.append(
                    Message(
                        role="tool",
                        content=result,
                        tool_call_id=tc.id,
                    )
                )

        # Max turns reached
        step.status = StepStatus.FAILED
        step.result = "Max turns reached without completion"
        return step.result
