"""Agent core engine - the main Observe-Think-Act loop."""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

from ..config import Config
from ..models.base import Message, ModelResponse, ToolCall
from ..models.router import ModelRouter, TaskType
from ..tools.base import ToolError, ToolRegistry
from ..tools.bash import register_bash_tools
from ..tools.file_ops import register_file_tools
from .context import truncate_messages, truncate_tool_result
from .executor import Executor, execute_tool_calls
from .planner import Planner
from .reflector import Reflector
from .state import AgentState, HierarchicalPlan, Plan, PlanStep, StepStatus
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
- assembly_vlm_solve：装配视觉验证闭环（solve → render → VLM → fix → re-solve，最多 3 轮）
  - 检测碰撞、悬空、错误朝向、不合理布局
  - 自动修正约束并重新求解
  - 适合 10+ 零件复杂装配的质量验证
- part_assemble + assembly_definition：自动定位组装
- ik_solve(target)：求解逆运动学，得到各关节角度
- mesh_collision_check：网格碰撞检测（trimesh + FCL），检查零件间干涉和穿透

双臂协调工具：
- dual_arm_ik(arm1, arm2, target1, target2, mode)：双臂协调逆运动学求解
  - mode=independent：两臂独立求解
  - mode=coordinated：协调求解+碰撞检测（推荐）
  - mode=master_slave：从臂镜像跟随主臂
- collision_check(arm1_capsules, arm2_capsules, safety_margin)：胶囊体碰撞检测
  - 使用 Capsule(线段+半径) 简化模型 + GJK 距离算法
  - safety_margin 默认 10mm，低于此距离触发碰撞警告
- workspace_analysis(assembly, mode, n_samples)：工作空间分析
  - mode=single：单臂可达范围 Monte Carlo 采样
  - mode=dual：双臂共享工作区域重叠分析
  - 返回 bounds、max_reach、overlap_ratio
- 双臂设计工作流：workspace_analysis → dual_arm_ik(coordinated) → collision_check → 稳定性验证

切片工作流：slice_model → slice_analyze → slice_vlm_analyze
- slice_model(stl_path)：STL → G-code 切片（支持打印机/材料/质量预设）
- slice_analyze(gcode_path)：解析 G-code 获取打印统计（时间/材料/成本）
- slice_preview_layers(gcode_path)：提取每层数据（Z高度/挤出/行程）
- slice_vlm_analyze(gcode_path)：截图 + VLM 评估打印质量

直流电机固件生成：
- gen_motor_driver(motors)：生成 PID 速度/位置控制 + 编码器反馈 C 代码
  - motors=[dict(motor_id, encoder_id, pwm_pin, dir_pin1, dir_pin2, enc_a_pin, enc_b_pin)]
  - 自动从 DC_MOTOR_PID_SPECS 匹配 Kp/Ki/Kd 参数
  - 支持 TT/GA25-370/JGB37-520/NEMA17 等电机
- gen_odometry(wheel_radius, wheel_base, encoder_ppr, gear_ratio)：生成差速底盘里程计
  - 自动计算 CPR 和 mm/tick
  - 含 Pose(x,y,θ) 更新和 velocity_to_wheels 速度分解
- 工作流：gen_motor_driver → gen_odometry → gen_firmware（电机底盘模式）

功率预算分析：
- power_budget(actuator_ids, mode)：系统功耗估算 + 电池选型
  - 自动计算峰值/平均功耗（电机/舵机/控制器/传感器）
  - 电池推荐：LiPo/LiFePO4/18650，按运行时间排序
  - mode=report 输出 Markdown 报告，mode=json 输出结构化数据
  - 支持自定义 duty_cycle、controller 功耗、safety_factor

高级建模操作（fc_batch 内）：
- sweep：沿路径扫掠截面（弹簧/螺纹/弯管）。参数：profile(circle/rectangle), profile_radius, path_type(helix/circle/line), pitch, height, helix_radius, turns
- loft：在多个截面之间过渡（锥形过渡/支架/喷嘴）。参数：profiles([dict(type,radius,center)...]), radius1, radius2, height, solid, ruled
- polar_pattern：圆周阵列（螺栓孔圆/风扇叶片/叶轮）。参数：object, count, angle(默认360), axis, center
- linear_pattern：线性阵列（散热片/格栅/齿条）。参数：object, count, spacing, direction
- mirror：镜像特征（右臂=左臂镜像）。参数：object, plane(XY/YZ/XZ)
- shell：抽壳（盒子/容器/外壳）。参数：object, thickness, faces_to_remove(list)
- draft：拔模斜度（注塑/3D打印脱模）。参数：object, angle(度), direction, faces(list)
- create_sketch：创建 2D 草图（电机支架/传感器座/外壳轮廓）。参数：name, plane(XY/XZ/YZ), elements([dict(type,...)]) — type 可选 point/line/circle/arc/rectangle/polygon
- extrude_sketch：草图拉伸为 3D 实体。参数：sketch, height, direction(x/y/z), midplane, reverse
- revolve_sketch：草图旋转为实体（轴/套筒/手轮）。参数：sketch, axis(x/y/z), angle(度), base
- pocket：草图挖槽（减材操作/减重孔/走线槽）。参数：sketch, target, depth, through_all, direction
何时用草图 vs 体素：简单零件用 make_box/cylinder + boolean；复杂轮廓（L型/多边形/弧形）用 create_sketch + extrude_sketch
FreeCAD 子进程超时已从 60s 提升至 300s 以支持复杂操作。

工具优先级：fc_batch > fc_menu > gui_*。仿真用 fea_run/fea_vlm_analyze，CFD 用 cfd_run/cfd_vlm_analyze，运动用 motion_sim 等。
detail 级别：fast, standard, detailed, maximum
单位：mm（毫米）。

URDF 物理仿真验证：用 sim_mujoco（MuJoCo 加载 URDF + 物理 + 关节能动性测试）。生成完整工程包后必须运行 sim_mujoco 验证：
- 结构验证（PASS）：URDF 可加载，mesh 可解析，质量/惯性合理，关节可驱动
- 物理稳定性（WARN/PASS）：PD 控制下能否保持初始姿态；不稳定通常意味着 URDF 需要 <actuator> 定义或控制器调参
自动修复：sim_mujoco 会把 URDF 中相对 mesh 路径（"meshes/X.stl"）改写为绝对路径以解决 urdf/ 子目录解析问题。
何时用：export_package 生成 ROS2 包后；或直接对任意 URDF 文件做加载检查。

抓取验证：用 sim_grasp（在夹爪间放立方体，闭合手指，测试能否抓起）。三阶段测试：
- Phase A（零重力）：闭合手指，看几何上能否夹紧立方体（接触数 ≥2 = 几何可行）
- Phase B（加重力）：测试摩擦+夹持力能否抵抗重力（滑落 <5mm = 夹持力足够）
- Phase C（抬升）：测试夹爪能跟随机械臂抬起（抬升 >5mm = 抬升成功）
何时用：sim_mujoco 结构验证通过后；用于验证夹爪装配是否真能抓东西。

质量属性工具：
- compute_part_mass：根据尺寸和材料计算零件质量（kg）
- compute_com：计算多零件质心（加权平均）
- compute_inertia：计算简单形状惯性张量（box/cylinder/sphere）
- compute_assembly_properties：计算装配体总质量、质心、惯性张量（平行轴定理）
- fc_batch compute_mass 操作：调用 FreeCAD 获取精确体积→质量
用途：稳定性分析、电机力矩计算、URDF 导出的基础数据

底盘设计工具：
- mobile_base_design(requirements)：从需求（载重/速度/坡度/续航）自动推导底盘参数（轮径/轮距/电机选型/电池容量/零件清单）
- differential_drive_sim(v_left, v_right, dt)：差速运动学仿真（轨迹预测）
底盘模板：differential_4w（差速4轮）、differential_2w（差速2轮）、mecanum（全向轮）
工作流：mobile_base_design → 建模 → compute_mass → 装配 → 仿真验证

驱动系统与闭环装配：
- drive_train_design(wheel_count, drive_type, motor_type)：自动生成电机→联轴器→轮子的驱动链装配定义
- assembly_solve 支持闭环运动链（4轮差速底盘的闭环约束）
- 闭环求解器使用 Newton-Raphson 迭代，支持齿轮传动比、差速器约束
- get_joint_chain() 返回树状/图状运动学结构（如双臂共享底盘节点）

稳定性分析：
- stability_analysis(com, contact_points, mode) → 稳定性分析
  - mode=full：综合评估（静态+动态+Force-Angle+翻倒风险）
  - mode=static：静态稳定裕度（质心到支撑多边形边缘最短距离）
  - mode=dynamic：ZMP 动态稳定性（零力矩点）
  - mode=report：生成 Markdown 稳定性报告
- 高重心设计（工控机+双臂在上方）必须通过稳定性检查
- 工作流：compute_assembly_properties → stability_analysis → 调整设计

复杂机器人设计工作流（子系统分解模式）：
当任务包含多个子系统（底盘+臂+电子设备等）时，使用分层规划：
1. 系统分解：将任务拆分为独立子系统（如 mobile_base / arm_left / arm_right / ipc_mount）
2. 对称性复用：4 个轮子只建模 1 次（instance_count=4），双臂只建模一侧（mirror_of）
3. 子系统并行设计：每个子系统独立建模、独立验证
4. 接口约束传递：子系统间通过安装接口约束关联（如底盘顶面螺孔→工控机支架底面）
5. 整机集成验证：装配→干涉检查→稳定性分析→运动学验证

ROS URDF 导出（urdf_export 工具）：
- 从装配体自动生成 ROS2 URDF/Xacro 文件
- 包含 link 几何（STL 网格）、joint 定义（类型/轴/限位）、质量/惯性属性
- 自动检测差速驱动/机械臂类型并添加 Gazebo 插件
- 生成完整 ROS2 包结构：urdf/ + meshes/ + launch/ + config/ + package.xml + CMakeLists.txt
- 参数：assembly_name（必需）、mode=xml|package、output_dir、package_name
- 工作流：装配体建模 → urdf_export(mode=package) → ROS2 启动仿真

电缆走线规划（cable_routing 工具）：
- 自动检测零件间电缆连接（actuator→controller、sensor→controller、battery→power）
- 3D 体素网格 A* 路径搜索 + Chaikin 平滑 + 弯曲半径约束验证
- 输出电缆走线报告：电缆清单、路径长度、弯曲检查、固定点建议
- 参数：assembly_name、mode=report|json、resolution（体素精度，默认 5mm）

工程包导出（export_package 工具）：
- 一键导出完整工程包：FreeCAD 建模脚本 + URDF/ROS2 包 + BOM + 装配指导 + 固件代码 + 接线图 + 电缆走线 + 功率预算 + 稳定性分析 + 设计报告
- 支持内置装配体（complex_robot、4w_dual_arm）或自定义 assembly_json
- 参数：assembly_name、assembly_json、output_dir、actuator_ids、controller、components
- CLI 命令：/export [assembly_name] [output_dir]

当前工作目录：{workspace}"""


class Agent:
    """Main agent class implementing the Observe-Think-Act loop."""

    def __init__(self, config: Config | None = None) -> None:
        from ..config import load_config

        self.config = config or load_config()
        self.router = ModelRouter(self.config)
        self.tools = ToolRegistry()
        self.state = AgentState(workspace=self.config.agent.workspace)

        # Enforce workspace boundary for file tools
        from ..tools.file_ops import FileOps
        FileOps.set_workspace(self.config.agent.workspace)

        # Initialize components
        self.planner = Planner(self.router)
        self.executor = Executor(self.router, self.tools)
        self.verifier = Verifier(self.router)
        self.reflector = Reflector(self.router)

        # Register built-in tools
        from ..tools.file_ops import register_file_tools
        from ..tools.bash import register_bash_tools
        register_file_tools(self.tools)
        register_bash_tools(self.tools)

        # Register all optional tools via auto-discovery
        # Pre-build fc_menu dependencies (needs VLM + GUI tool instances)
        fc_menu_deps = None
        try:
            from ..tools.vlm import VLMLocateTool
            from ..tools.gui_action import GUIClickTool, GUITypeTool, GUIPressKeyTool
            fc_menu_deps = {
                "locate_tool": VLMLocateTool(self.router, screenshot_dir=self.config.agent.screenshot_dir),
                "click_tool": GUIClickTool(),
                "type_tool": GUITypeTool(),
                "press_key_tool": GUIPressKeyTool(),
            }
        except Exception:
            pass

        from ..tools import discover_and_register
        discover_and_register(
            self.tools,
            router=self.router,
            screenshot_dir=self.config.agent.screenshot_dir,
            fc_menu_deps=fc_menu_deps,
        )

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
            for tc in reversed(self.state.tool_history):
                if tc.get("name") == name:
                    args = tc.get("arguments", {})
                    prompt = str(args.get("prompt", args.get("expected", "")))
                    break
            add_vlm_result(name, prompt, result)

    def _cleanup_failed_step_artifacts(self, step: PlanStep) -> None:
        """Extract and clean up artifact files from a failed step."""
        try:
            from .sub_agent import cleanup_artifacts
            # Collect artifact paths from recent tool history related to this step
            artifacts: list[str] = []
            for entry in self.state.tool_history:
                result = entry.get("result", "")
                args = entry.get("arguments", {})
                for key in ("path", "file_path", "output_path"):
                    if key in args:
                        artifacts.append(args[key])
                for line in result.split("\n"):
                    line = line.strip()
                    if line.endswith((".fcstd", ".step", ".stl", ".obj", ".py")):
                        for ext in (".fcstd", ".step", ".stl", ".obj", ".py"):
                            if ext in line:
                                idx = line.find(ext)
                                path = line[: idx + len(ext)].split()[-1]
                                artifacts.append(path)
                                break
            if artifacts:
                cleanup_artifacts(artifacts, str(self.state.workspace))
        except Exception:
            logger.warning("Failed to clean up artifacts for step %s", step.id, exc_info=True)

    def _insert_step_into_hierarchical_plan(
        self, failed_step: PlanStep, new_step: PlanStep
    ) -> None:
        """Insert a replacement step after the failed step in the correct subsystem."""
        plan = self.state.plan
        if not isinstance(plan, HierarchicalPlan):
            raise TypeError(f"Expected HierarchicalPlan, got {type(plan).__name__}")
        # Find which subsystem (or integration_steps) owns the failed step
        for ss in plan.subsystems:
            if failed_step in ss.steps:
                idx = ss.steps.index(failed_step)
                ss.steps.insert(idx + 1, new_step)
                return
        # Check integration_steps
        if failed_step in plan.integration_steps:
            idx = plan.integration_steps.index(failed_step)
            plan.integration_steps.insert(idx + 1, new_step)
            return
        # Fallback: append to integration_steps
        plan.integration_steps.append(new_step)

    def run_task(self, task: str, *, use_planning: bool = True, use_orchestration: bool = True) -> str:
        """Run a task, optionally with planning and multi-agent orchestration.

        Returns the final result.
        """
        self.state = AgentState(workspace=self.config.agent.workspace)

        if use_planning:
            # Check if this is a complex robot task that needs hierarchical planning
            if self._should_use_hierarchical(task):
                plan = self.planner.create_hierarchical_plan(task)
                self.state.plan = plan
                if use_orchestration:
                    from .orchestrator import PhasedOrchestrator
                    phased = PhasedOrchestrator(
                        config=self.config,
                        router=self.router,
                        tools=self.tools,
                        planner=self.planner,
                        hierarchical_plan=plan,
                        workspace=self.config.agent.workspace,
                        max_parallel=self.config.agent.orchestrator.max_parallel_agents,
                        max_retries=self.config.agent.orchestrator.max_retries_per_step,
                    )
                    self._wire_callbacks(phased)
                    return phased.run_task(task)
                else:
                    return self._run_with_planning(task)
            else:
                plan = self.planner.create_plan(task)
                self.state.plan = plan
                if use_orchestration and self._should_orchestrate(task, plan):
                    return self._run_with_orchestration(task, plan)
                else:
                    return self._run_with_planning(task)
        else:
            return self._run_direct(task)

    @staticmethod
    def _should_use_hierarchical(task: str) -> bool:
        """Determine if a task needs hierarchical (subsystem-level) planning.

        Triggers when the task description matches complex robot patterns
        (multiple subsystems, wheels, dual arms, etc.).
        """
        from .planner import COMPLEX_ROBOT_KEYWORDS
        task_lower = task.lower()
        return any(kw in task_lower for kw in COMPLEX_ROBOT_KEYWORDS)

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
        self._wire_callbacks(orchestrator)

        # The orchestrator can reuse the plan we already created
        return orchestrator.run_task(task, plan)

    def _wire_callbacks(self, orchestrator: Any) -> None:
        """Wire UI callbacks onto an orchestrator or phased orchestrator."""
        if self._on_thinking:
            orchestrator.on_thinking(self._on_thinking)
        if self._on_tool_call:
            orchestrator.on_tool_call(self._on_tool_call)
        if self._on_tool_result:
            orchestrator.on_tool_result(self._on_tool_result)

    def _broadcast_plan(self) -> None:
        """Push the current plan (if any) to the web panel as JSON."""
        try:
            from ..web.app import update_agent_state
        except ImportError:
            return
        if self.state.plan is None:
            return
        plan = self.state.plan
        # Get steps from plan (flat) or all_steps (hierarchical)
        if isinstance(plan, HierarchicalPlan):
            all_steps = plan.all_steps()
        else:
            all_steps = plan.steps
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
                for s in all_steps
            ],
        }
        # Add subsystem info if this is a hierarchical plan
        if isinstance(plan, HierarchicalPlan):
            payload["subsystems"] = [
                {
                    "name": ss.name,
                    "description": ss.description,
                    "parts": ss.parts,
                    "mirror_of": ss.mirror_of,
                    "instance_count": ss.instance_count,
                    "status": ss.status.value if hasattr(ss.status, "value") else str(ss.status),
                }
                for ss in plan.subsystems
            ]
            payload["system_dependencies"] = [
                {"source": d.source, "target": d.target, "reason": d.reason}
                for d in plan.system_dependencies
            ]
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
        global_retry_count = 0
        max_global_retries = max_retries * 10  # Safety: prevent infinite retry loops
        max_plan_size = 50  # Maximum number of steps to prevent unbounded growth

        while True:
            step = self.state.plan.current_step()
            if step is None:
                break

            # Safety: check plan hasn't grown too large
            if isinstance(self.state.plan, HierarchicalPlan):
                step_count = len(self.state.plan.all_steps())
            else:
                step_count = len(self.state.plan.steps)
            if step_count > max_plan_size:
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
            if step.status == StepStatus.FAILED and step.attempts < max_retries and global_retry_count < max_global_retries:
                global_retry_count += 1

                # Clean up artifact files from the failed step
                self._cleanup_failed_step_artifacts(step)

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
                    # Mark failed step as SKIPPED before inserting replacement
                    step.status = StepStatus.SKIPPED
                    # Insert after current step in the actual plan structure
                    if isinstance(self.state.plan, HierarchicalPlan):
                        self._insert_step_into_hierarchical_plan(step, new_step)
                    else:
                        idx = self.state.plan.steps.index(step)
                        self.state.plan.steps.insert(idx + 1, new_step)
                    if self._on_plan_update:
                        self._on_plan_update(self.state.plan)
            elif step.status == StepStatus.FAILED:
                # Clean up artifact files from the failed step
                self._cleanup_failed_step_artifacts(step)
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
            tools = self.tools.get_direct_definitions(task)
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

            # Execute tool calls via shared helper
            verify_fail_count, fix_history = execute_tool_calls(
                response.tool_calls,
                tools=self.tools,
                state=self.state,
                messages=messages,
                verify_fail_count=verify_fail_count,
                fix_history=fix_history,
                max_verify_retries=max_verify_retries,
                on_tool_call=self._on_tool_call,
                on_tool_result=self._on_tool_result,
                on_thinking=self._on_thinking,
                assistant_content=response.content,
            )

            # Apply sliding window when messages grow too large
            if len(messages) > 12:
                messages = truncate_messages(messages, keep_first=1, keep_last=2)

        return "达到最大对话轮数限制"

    def chat(self, message: str, history: list[Message] | None = None) -> ModelResponse:
        """Simple chat without tool use."""
        messages = list(history) if history else []
        messages.append(Message(role="user", content=message))

        response = self.router.chat(
            messages=messages,
            system=AGENT_SYSTEM_PROMPT.format(workspace=self.state.workspace),
            task_type=TaskType.CHAT,
        )

        return response
