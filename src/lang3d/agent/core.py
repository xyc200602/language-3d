"""Agent core engine - the main Observe-Think-Act loop."""

from __future__ import annotations

from typing import Any, Callable

from ..config import Config
from ..models.base import Message, ModelResponse, ToolCall
from ..models.router import ModelRouter, TaskType
from ..tools.base import ToolRegistry
from ..tools.bash import register_bash_tools
from ..tools.file_ops import register_file_tools
from .context import truncate_messages, truncate_tool_result
from .executor import Executor
from .fix_strategy import classify_failure, check_convergence, generate_fix_hint
from .planner import Planner
from .reflector import Reflector
from .state import AgentState, Plan, PlanStep, StepStatus
from .verifier import Verifier


AGENT_SYSTEM_PROMPT = """你是 Language-3D Agent，一个自主编程和 3D 建模助手。

CAD 工具前缀：fc_*（FreeCAD）, sw_*（SolidWorks）, gui_*（GUI 自动化）
零件库工具前缀：part_*（搜索/获取/生成/导入/保存标准零件）
装配工具：assembly_solve（约束求解），ik_solve（逆向运动学）
切片工具前缀：slice_*（3D 打印切片）

3D 建模工作流：规划 → 建模 → 验证 → 修正
1. 规划（Plan）：分析任务，确定建模步骤
2. 建模（Model）：优先用 fc_batch 一次性完成多步建模
3. 验证（Verify）：根据复杂度选择验证方式（见下方策略）
4. 修正（Fix）：若验证不通过，根据反馈修正模型

验证策略（按零件复杂度选择）：
- Level 1-2（简单零件：正方体/圆柱体/平板）：用 fc_batch 的 volume_check 快速验证
  - 检查体积、尺寸是否在预期范围内，无需 VLM，耗时 <5s
  - 示例：fc_batch operations=[..., volume_check + checks: dimensions/tolerance_mm]
- Level 3+（复杂零件：带孔/倒角/多特征）：用 cad_verify 多角度验证
  - 默认 angles=isometric,front,top（已内置）
  - detail="detailed" 复杂验证
- 装配体验证：用 interference_check（碰撞检测）替代 cad_verify

cad_verify 验证策略：
- 检查返回 MATCH 字段：true=通过，false=需修正
- 不匹配时根据 FIX_COMMANDS 修正模型
- 默认使用 isometric,front,top 三角度验证

零件库工作流：
- part_search(query)：搜索标准件（螺钉/轴承/舵机/齿轮等）
- part_get(part_id)：查看零件参数和标准尺寸
- part_generate(part_id, parameters)：生成参数化零件文件（.FCStd + .STL）
- part_list(category)：按类别浏览零件
- 需要标准件时优先用零件库，避免从零建模

装配工作流：
- 定义 Assembly（parts + joints），设置 parent_anchor/child_anchor/axis
- assembly_solve：自动计算每个零件的全局位置
- part_assemble + assembly_definition：自动定位组装
- ik_solve(target)：求解逆运动学，得到各关节角度

切片工作流：slice_model → slice_analyze → slice_vlm_analyze
- slice_model(stl_path)：STL → G-code 切片（支持打印机/材料/质量预设）
- slice_analyze(gcode_path)：解析 G-code 获取打印统计（时间/材料/成本）
- slice_preview_layers(gcode_path)：提取每层数据（Z高度/挤出/行程）
- slice_vlm_analyze(gcode_path)：截图 + VLM 评估打印质量

工具优先级：fc_batch > fc_menu > gui_*。仿真用 fea_run/fea_vlm_analyze，CFD 用 cfd_run/cfd_vlm_analyze，运动用 motion_sim 等。
detail 级别：fast, standard, detailed, maximum
单位：mm（毫米）。

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

        # Register part library tools (standard parts catalog)
        try:
            from ..tools.part_library import register_part_library_tools
            register_part_library_tools(self.tools)
        except Exception:
            pass

        # Register assembly solver tools (constraint-based auto-positioning)
        try:
            from ..tools.assembly_solver import register_assembly_solver_tools
            register_assembly_solver_tools(self.tools)
        except Exception:
            pass

        # Register IK solver tools (inverse kinematics)
        try:
            from ..tools.ik_solver import register_ik_tools
            register_ik_tools(self.tools)
        except Exception:
            pass

        # Register slicing tools (PrusaSlicer/OrcaSlicer)
        try:
            from ..tools.slicing import register_slicing_tools
            register_slicing_tools(
                self.tools,
                router=self.router,
                screenshot_dir=self.config.agent.screenshot_dir,
            )
        except Exception:
            pass

        # Register actuator tools (servo/motor selection and analysis)
        try:
            from ..tools.actuator_tools import register_actuator_tools
            register_actuator_tools(self.tools)
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
                     "motion_vlm_analyze", "cfd_vlm_analyze", "slice_vlm_analyze"):
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

    def _broadcast_plan(self) -> None:
        """Push the current plan (if any) to the web panel as JSON."""
        try:
            from ..web.app import update_agent_state
        except ImportError:
            return
        if self.state.plan is None:
            return
        plan = self.state.plan
        payload = {
            "goal": plan.goal,
            "steps": [
                {
                    "id": s.id,
                    "description": s.description,
                    "expected_tools": list(s.expected_tools),
                    "verification": s.verification,
                    "status": s.status.value if hasattr(s.status, "value") else str(s.status),
                    "result": s.result,
                    "attempts": s.attempts,
                    "dependencies": list(s.dependencies),
                    "assigned_agent": s.assigned_agent,
                }
                for s in plan.steps
            ],
        }
        try:
            completed, total = plan.progress()
            update_agent_state(plan=payload, progress={"completed": completed, "total": total})
        except Exception:
            try:
                update_agent_state(plan=payload)
            except Exception:
                pass

    def _run_with_planning(self, task: str) -> str:
        """Run a task with full planning pipeline."""
        # Phase 1: Plan (already created in run_task if called from there)
        try:
            from ..web.app import add_log, update_agent_state
            update_agent_state(status="running")
            add_log(f"Task started (planning): {task[:120]}", level="info")
        except ImportError:
            pass

        if self._on_thinking:
            try:
                self._on_thinking("正在分析任务并制定计划...")
            except Exception:
                pass

        if self.state.plan is None:
            self.state.plan = self.planner.create_plan(task)
        if self._on_plan_update:
            self._on_plan_update(self.state.plan)
        self._broadcast_plan()

        # Phase 2: Execute each step
        max_retries = getattr(self.config.agent, "max_plan_retries", 3)
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

            # Broadcast progress after every step
            self._broadcast_plan()

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
        try:
            from ..web.app import add_log, update_agent_state
            update_agent_state(status="complete")
            add_log(f"Task completed: {completed}/{total} steps succeeded", level="success")
        except ImportError:
            pass
        return f"任务完成：{completed}/{total} 步骤成功"

    def _run_direct(self, task: str) -> str:
        """Run a task directly without planning (simple chat mode)."""
        messages: list[Message] = [Message(role="user", content=task)]
        verify_fail_count = 0
        max_verify_retries = getattr(self.config.agent, "max_verify_retries", 3)
        fix_history: list[str] = []

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

                # Truncate large tool results
                result = truncate_tool_result(result)

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

                # Smart auto-fix: classify failure and generate targeted hint
                if tc.name == "cad_verify" and "MATCH: False" in result:
                    verify_fail_count += 1
                    if verify_fail_count <= max_verify_retries:
                        expected = tc.arguments.get("expected", "")
                        fix_ctx = classify_failure(result, expected)

                        # Check for convergence (stuck in loop)
                        if check_convergence(fix_history, result):
                            fix_hint = (
                                "[系统提示] 检测到修复陷入循环（连续多次失败原因相似）。"
                                "请尝试完全不同的建模方法，或删除当前模型从头开始重建。"
                            )
                        else:
                            fix_ctx.fix_history = fix_history
                            fix_hint = generate_fix_hint(fix_ctx)

                        fix_history.append(result)
                        messages.append(Message(role="user", content=fix_hint))
                        if self._on_thinking:
                            try:
                                self._on_thinking(
                                    f"智能修复：{fix_ctx.failure_type.value}，注入定向提示（第 {verify_fail_count} 次）"
                                )
                            except Exception:
                                pass

            # Apply sliding window when messages grow too large
            if len(messages) > 12:
                messages = truncate_messages(messages, keep_first=1, keep_last=2)

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
