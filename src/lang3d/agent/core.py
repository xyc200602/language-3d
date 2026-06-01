"""Agent core engine - the main Observe-Think-Act loop."""

from __future__ import annotations

from typing import Any, Callable

from ..config import Config
from ..models.base import Message, ModelResponse, ToolCall
from ..models.router import ModelRouter, TaskType
from ..tools.base import ToolRegistry
from ..tools.bash import register_bash_tools
from ..tools.file_ops import register_file_tools
from .executor import Executor
from .planner import Planner
from .reflector import Reflector
from .state import AgentState, Plan, PlanStep, StepStatus
from .verifier import Verifier


AGENT_SYSTEM_PROMPT = """你是 Language-3D Agent，一个自主编程和 3D 建模助手。

你的能力：
1. 读写文件、执行命令
2. 通过截屏和视觉模型观察屏幕
3. 控制 CAD 软件进行 3D 建模（FreeCAD 或 SolidWorks）
4. 将复杂任务分解为步骤并逐步执行
5. 自我验证和纠错

CAD 工具前缀：
- fc_* : FreeCAD 工具（免费开源，推荐优先使用）
- sw_* : SolidWorks 工具（需要安装 SolidWorks）
- gui_* : GUI 自动化操作（点击、输入、快捷键、拖拽、滚动）

3D 建模工作流（必须遵循）：
1. 规划：分析任务，确定需要的建模步骤
2. 建模：使用 fc_batch 一次性完成多步建模操作
3. 验证：建模完成后必须调用 cad_verify 验证结果
4. 修正：如果 cad_verify 返回 match=false，根据 DIFFERENCES 和 SUGGESTION 修正模型
5. 再验证：修正后再次调用 cad_verify 确认

cad_verify 验证策略：
- 建模后使用 fc_open_gui 打开 FreeCAD GUI 查看模型
- 调用 cad_verify 并传入预期模型描述
- 检查返回的 MATCH 字段：true=通过，false=需要修正
- 如果不匹配，根据 FIX_COMMANDS 字段修正
- 使用 detail="standard" 进行常规验证（快速准确）
- 使用 detail="detailed" 进行需要详细描述的复杂验证

仿真分析工具：
- fea_run：对 .FCStd 文档运行 FEA 结构分析（网格→边界条件→CalculiX求解）
- fea_visualize：在 FreeCAD GUI 中显示应力/位移云图
- fea_vlm_analyze：截取云图 + VLM 解读应力分布，返回安全评估
- interference_check：检查零件干涉（布尔交集检测重叠体积）
- tolerance_analysis：蒙特卡洛公差分析（纯 Python，不需要 FreeCAD）
- motion_sim：运动仿真（正向运动学、范围检查、轨迹规划）
- motion_range：关节运动范围和可达空间分析
- motion_trajectory：关节空间线性插值轨迹规划
- motion_vlm_analyze：截取运动可视化 + VLM 分析

CFD 流体分析工具：
- cfd_run：运行 OpenFOAM CFD 分析（网格→边界条件→求解→结果）
- cfd_vlm_analyze：截取 CFD 可视化 + VLM 解读流场

仿真工作流：
1. 建模 → 保存 .FCStd
2. fea_run 运行结构分析
3. fea_vlm_analyze 用 VLM 解读应力云图
4. 如果 unsafe，根据 SUGGESTION/FIX_COMMANDS 修改模型
5. 重新 fea_run → fea_vlm_analyze 验证改进效果

工具使用策略：
- fc_batch：优先用于多步建模，一次调用完成整个建模流程
- cad_verify：建模完成后必须调用，传入详细的预期描述
- fc_open_gui / fc_close_gui：用于查看模型和验证
- fc_menu：通过 VLM 视觉定位点击 FreeCAD 菜单/按钮（用于 GUI 自动化）
- fc_menu_workflow：执行预定义 GUI 操作流程（new_part, add_box, save_file 等）
- vlm_locate：先定位 UI 元素坐标，再用 gui_click 点击（自动化的前置步骤）
- gui_* 工具：用于操作 FreeCAD GUI（旋转视图、点击菜单等）
- detail 级别：fast(快速), standard(准确), detailed(详细), maximum(最全面)

fc_menu vs fc_batch 选择策略：
- fc_batch（API 方式）：精确、快速、无需 GUI。适合参数化建模。
- fc_menu（GUI 方式）：视觉驱动、交互式。适合工作台特定功能、复杂对话框、API 未暴露的操作。
- 优先使用 fc_batch，仅在 API 无法实现时使用 fc_menu。

工作原则：
- 每次只做一步，确认结果后再继续
- 遇到错误时分析原因，尝试修复
- 使用工具前确认操作是安全的
- 保持简洁，不过度解释
- 3D 建模时使用毫米(mm)作为单位
- 建模完成后必须验证（cad_verify 或 vlm_analyze）

当前工作目录：{workspace}"""


class Agent:
    """Main agent class implementing the Observe-Think-Act loop."""

    def __init__(self, config: Config | None = None) -> None:
        from ..config import load_config

        self.config = config or load_config()
        self.router = ModelRouter(self.config)
        self.tools = ToolRegistry()
        self.state = AgentState(workspace=self.config.agent.workspace)

        # Initialize components
        self.planner = Planner(self.router)
        self.executor = Executor(self.router, self.tools)
        self.verifier = Verifier(self.router)
        self.reflector = Reflector(self.router)

        # Register built-in tools
        register_file_tools(self.tools)
        register_bash_tools(self.tools)

        # Register optional tools
        try:
            from ..tools.screen import register_screen_tools
            from ..tools.vlm import register_vlm_tools
            from ..tools.cad_utils import register_cad_utils
            from ..tools.python_exec import register_python_tools
            from ..tools.gui_action import register_gui_action_tools

            register_screen_tools(self.tools, screenshot_dir=self.config.agent.screenshot_dir)
            register_vlm_tools(self.tools, self.router, screenshot_dir=self.config.agent.screenshot_dir)
            register_cad_utils(self.tools)
            register_python_tools(self.tools)
            register_gui_action_tools(self.tools, screenshot_dir=self.config.agent.screenshot_dir)

            # Register FreeCAD menu tools (depend on VLM + GUI tools above)
            try:
                from ..tools.fc_menu import register_fc_menu_tools
                from ..tools.vlm import VLMLocateTool
                from ..tools.gui_action import (
                    GUIClickTool, GUITypeTool, GUIPressKeyTool,
                )

                _locate = VLMLocateTool(self.router, screenshot_dir=self.config.agent.screenshot_dir)
                _click = GUIClickTool()
                _type = GUITypeTool()
                _press = GUIPressKeyTool()
                register_fc_menu_tools(
                    self.tools,
                    locate_tool=_locate,
                    click_tool=_click,
                    type_tool=_type,
                    press_key_tool=_press,
                )
            except ImportError:
                pass
        except ImportError:
            pass

        # Register SolidWorks tools (may fail if SW not installed, that's OK)
        try:
            from ..tools.solidworks import register_solidworks_tools
            register_solidworks_tools(self.tools)
        except Exception:
            pass

        # Register FreeCAD tools (may fail if FreeCAD not installed, that's OK)
        try:
            from ..tools.freecad import register_freecad_tools
            register_freecad_tools(self.tools)
        except Exception:
            pass

        # Register simulation tools (FEA, tolerance, interference, motion)
        try:
            from ..tools.simulation import register_simulation_tools
            register_simulation_tools(
                self.tools,
                router=self.router,
                screenshot_dir=self.config.agent.screenshot_dir,
            )
        except Exception:
            pass

        # Register CFD tools (OpenFOAM)
        try:
            from ..tools.cfd import register_cfd_tools
            register_cfd_tools(
                self.tools,
                router=self.router,
                screenshot_dir=self.config.agent.screenshot_dir,
            )
        except Exception:
            pass

        # Callbacks for UI
        self._on_tool_call: Callable[[str, dict], None] | None = None
        self._on_tool_result: Callable[[str, str], None] | None = None
        self._on_thinking: Callable[[str], None] | None = None
        self._on_plan_update: Callable[[Any], None] | None = None
        self._on_step_update: Callable[[PlanStep], None] | None = None

    def on_tool_call(self, callback: Callable[[str, dict], None]) -> None:
        self._on_tool_call = callback

    def on_tool_result(self, callback: Callable[[str, str], None]) -> None:
        self._on_tool_result = callback

    def on_thinking(self, callback: Callable[[str], None]) -> None:
        self._on_thinking = callback

    def on_plan_update(self, callback: Callable[[Any], None]) -> None:
        self._on_plan_update = callback

    def on_step_update(self, callback: Callable[[PlanStep], None]) -> None:
        self._on_step_update = callback

    def connect_web_panel(self) -> None:
        """Connect agent callbacks to the web monitoring panel."""
        try:
            from ..web.app import (
                add_log,
                add_tool_call as web_add_tool_call,
                add_vlm_result,
                set_thinking,
                update_agent_state,
            )

            self.on_tool_call(lambda name, args: web_add_tool_call(name, args))
            self.on_tool_result(lambda name, result: self._web_on_result(name, result, add_vlm_result))
            self.on_thinking(lambda text: set_thinking(text))
            self._web_connected = True
        except ImportError:
            pass

    def _web_on_result(self, name: str, result: str, add_vlm_result: Any) -> None:
        """Handle tool results for web panel VLM tracking."""
        if name in ("vlm_analyze", "cad_verify", "vlm_locate",
                     "screen_analyze", "window_analyze", "fea_vlm_analyze",
                     "motion_vlm_analyze", "cfd_vlm_analyze"):
            # Extract the prompt from recent tool calls in state
            prompt = ""
            for tc_name, tc_args, _ in reversed(self.state.tool_history):
                if tc_name == name:
                    prompt = str(tc_args.get("prompt", tc_args.get("expected", "")))
                    break
            add_vlm_result(name, prompt, result)

    def run_task(self, task: str, *, use_planning: bool = True, use_orchestration: bool = True) -> str:
        """Run a task, optionally with planning and multi-agent orchestration.

        Returns the final result.
        """
        self.state = AgentState(workspace=self.config.agent.workspace)

        if use_planning:
            plan = self.planner.create_plan(task)
            self.state.plan = plan
            if use_orchestration and self._should_orchestrate(task, plan):
                return self._run_with_orchestration(task, plan)
            else:
                return self._run_with_planning(task)
        else:
            return self._run_direct(task)

    def _should_orchestrate(self, task: str, plan: Plan) -> bool:
        """Heuristic: orchestrate when >= 4 steps with >= 3 modeling steps."""
        if len(plan.steps) < 4:
            return False
        if not self.config.agent.orchestrator.enable_parallel:
            return False

        modeling_steps = 0
        for step in plan.steps:
            tools_lower = [t.lower() for t in step.expected_tools]
            if any("fc_" in t or "cad" in t for t in tools_lower):
                modeling_steps += 1

        return modeling_steps >= 3

    def _run_with_orchestration(self, task: str, plan: Plan) -> str:
        """Delegate to OrchestratorAgent for multi-agent parallel execution."""
        from .assembly_verifier import AssemblyVerifier
        from .orchestrator import OrchestratorAgent

        orchestrator = OrchestratorAgent(
            config=self.config,
            router=self.router,
            tools=self.tools,
            planner=self.planner,
            verifier=AssemblyVerifier(),
            reflector=self.reflector,
            workspace=self.config.agent.workspace,
            max_parallel=self.config.agent.orchestrator.max_parallel_agents,
            max_retries=self.config.agent.orchestrator.max_retries_per_step,
        )

        # Wire callbacks
        if self._on_thinking:
            orchestrator.on_thinking(self._on_thinking)
        if self._on_tool_call:
            orchestrator.on_tool_call(self._on_tool_call)
        if self._on_tool_result:
            orchestrator.on_tool_result(self._on_tool_result)

        # The orchestrator creates its own plan, so pass the task
        # (the plan we already have was used for the should_orchestrate check)
        return orchestrator.run_task(task)

    def _run_with_planning(self, task: str) -> str:
        """Run a task with full planning pipeline."""
        # Phase 1: Plan (already created in run_task if called from there)
        if self._on_thinking:
            try:
                self._on_thinking("正在分析任务并制定计划...")
            except Exception:
                pass

        if self.state.plan is None:
            self.state.plan = self.planner.create_plan(task)
        if self._on_plan_update:
            self._on_plan_update(self.state.plan)

        # Phase 2: Execute each step
        max_retries = 3
        while True:
            step = self.state.plan.current_step()
            if step is None:
                break

            if self._on_step_update:
                self._on_step_update(step)

            result = self.executor.execute_step(
                step,
                self.state,
                on_tool_call=self._on_tool_call,
                on_tool_result=self._on_tool_result,
                on_thinking=self._on_thinking,
            )

            # Phase 3: Verify
            if step.status == StepStatus.COMPLETED:
                success, msg = self.verifier.verify_step(step, result)
                if success:
                    step.status = StepStatus.COMPLETED
                    if self._on_step_update:
                        self._on_step_update(step)
                else:
                    step.status = StepStatus.FAILED
                    step.result = f"验证失败: {msg}"

            # Phase 4: Reflect and retry if failed
            if step.status == StepStatus.FAILED and step.attempts < max_retries:
                reflection = self.reflector.reflect(
                    self.state.plan, step, result, self.state.tool_history
                )
                if self._on_thinking:
                    try:
                        self._on_thinking(f"反思：{reflection}")
                    except Exception:
                        pass

                # Get replacement step from planner
                new_step = self.planner.replan_from_failure(
                    self.state.plan, step, result, reflection
                )
                if new_step:
                    # Insert after current step
                    idx = self.state.plan.steps.index(step)
                    self.state.plan.steps.insert(idx + 1, new_step)
                    if self._on_plan_update:
                        self._on_plan_update(self.state.plan)
            elif step.status == StepStatus.FAILED:
                if self._on_step_update:
                    self._on_step_update(step)

        # Save final state
        self.state.save()

        completed, total = self.state.plan.progress()
        return f"任务完成：{completed}/{total} 步骤成功"

    def _run_direct(self, task: str) -> str:
        """Run a task directly without planning (simple chat mode)."""
        messages: list[Message] = [Message(role="user", content=task)]
        verify_fail_count = 0
        max_verify_retries = 3

        # Update web panel status
        try:
            from ..web.app import update_agent_state, add_log
            update_agent_state(status="running")
            add_log(f"Task started: {task[:100]}", level="info")
        except ImportError:
            pass

        for _ in range(self.config.agent.max_turns):
            tools = self.tools.get_all_definitions()
            response = self.router.chat(
                messages=messages,
                tools=tools,
                system=AGENT_SYSTEM_PROMPT.format(workspace=self.state.workspace),
                task_type=TaskType.CHAT,
            )

            if not response.tool_calls:
                # Task completed
                try:
                    from ..web.app import update_agent_state, add_log
                    update_agent_state(status="complete")
                    add_log(f"Task completed: {response.content[:100]}", level="success")
                except ImportError:
                    pass
                return response.content

            # Add assistant message with tool calls
            messages.append(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=[
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in response.tool_calls
                    ],
                )
            )

            # Execute tool calls
            for tc in response.tool_calls:
                if self._on_tool_call:
                    try:
                        self._on_tool_call(tc.name, tc.arguments)
                    except Exception:
                        pass

                result = self.tools.execute(tc.name, **tc.arguments)
                self.state.add_tool_call(tc.name, tc.arguments, result)

                if self._on_tool_result:
                    try:
                        self._on_tool_result(tc.name, result)
                    except UnicodeEncodeError:
                        # VLM may return Unicode chars (e.g. braille) that
                        # can't be encoded in Windows GBK terminals
                        try:
                            self._on_tool_result(tc.name, result.encode("utf-8", errors="replace").decode("utf-8"))
                        except Exception:
                            pass
                    except Exception:
                        pass

                messages.append(
                    Message(role="tool", content=result, tool_call_id=tc.id)
                )

                # Auto-fix: detect cad_verify mismatch and inject fix prompt
                if tc.name == "cad_verify" and "MATCH: False" in result:
                    verify_fail_count += 1
                    if verify_fail_count <= max_verify_retries:
                        fix_hint = (
                            f"[系统提示] cad_verify 检测到模型不匹配（第 {verify_fail_count} 次）。"
                            f"请分析 DIFFERENCES 和 SUGGESTION，使用 fc_batch 修正模型，然后重新验证。"
                        )
                        messages.append(Message(role="user", content=fix_hint))
                        if self._on_thinking:
                            try:
                                self._on_thinking(
                                    f"自动修复：cad_verify 不匹配，注入修正提示（第 {verify_fail_count} 次）"
                                )
                            except Exception:
                                pass

        return "达到最大对话轮数限制"

    def chat(self, message: str, history: list[Message] | None = None) -> ModelResponse:
        """Simple chat without tool use."""
        messages = history or []
        messages.append(Message(role="user", content=message))

        response = self.router.chat(
            messages=messages,
            system=AGENT_SYSTEM_PROMPT.format(workspace=self.state.workspace),
            task_type=TaskType.CHAT,
        )

        return response
