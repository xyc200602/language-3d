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
from .fix_strategy import classify_failure, check_convergence, extract_fix_commands, generate_fix_hint
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

        # Register fastener model tools (ISO/DIN standard hardware)
        try:
            from ..tools.fastener_model import register_fastener_tools
            register_fastener_tools(self.tools)
        except Exception:
            pass

        # Register assembly solver tools (constraint-based auto-positioning)
        try:
            from ..tools.assembly_solver import register_assembly_solver_tools
            register_assembly_solver_tools(self.tools)
        except Exception:
            pass

        # Register mating constraint solver (Task 77)
        try:
            from ..tools.mating_constraint import constraint_solve_tool_factory
            _def, _cls = constraint_solve_tool_factory()
            self.tools.register(_cls())
        except Exception:
            pass

        # Register assembly auto-matcher (Task 78)
        try:
            from ..tools.assembly_matcher import assembly_match_tool_factory
            _def2, _cls2 = assembly_match_tool_factory()
            self.tools.register(_cls2())
        except Exception:
            pass

        # Register assembly VLM verification tools
        try:
            from ..tools.assembly_vlm import AssemblyVLMSolveTool
            self.tools.register(AssemblyVLMSolveTool())
        except Exception:
            pass

        # Register IK solver tools (inverse kinematics)
        try:
            from ..tools.ik_solver import register_ik_tools
            register_ik_tools(self.tools)
        except Exception:
            pass

        # Register collision detection tools (capsule + GJK)
        try:
            from ..tools.collision import register_collision_tools
            register_collision_tools(self.tools)
        except Exception:
            pass

        # Register mesh collision tools (trimesh + FCL)
        try:
            from ..tools.mesh_collision import register_mesh_collision_tools
            register_mesh_collision_tools(self.tools)
        except Exception:
            pass

        # Register workspace analysis tools (Monte Carlo sampling)
        try:
            from ..tools.workspace import register_workspace_tools
            register_workspace_tools(self.tools)
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

        # Register power budget tools (system power analysis + battery selection)
        try:
            from ..tools.power_budget import register_power_budget_tools
            register_power_budget_tools(self.tools)
        except Exception:
            pass

        # Register code generation tools (firmware, wiring, test sequence)
        try:
            from ..tools.code_gen import register_code_gen_tools
            register_code_gen_tools(self.tools)
        except Exception:
            pass

        # Register BOM generation tools (bill of materials)
        try:
            from ..tools.bom_gen import register_bom_tools
            register_bom_tools(self.tools)
        except Exception:
            pass

        # Register assembly documentation tools (assembly guide)
        try:
            from ..tools.assembly_doc import register_assembly_doc_tools
            register_assembly_doc_tools(self.tools)
        except Exception:
            pass

        # Register print optimization tools (3D print settings)
        try:
            from ..tools.print_optimize import register_print_optimize_tools
            register_print_optimize_tools(self.tools)
        except Exception:
            pass

        # Register quality control tools (inspection, test, maintenance)
        try:
            from ..tools.quality import register_quality_tools
            register_quality_tools(self.tools)
        except Exception:
            pass

        # Register iteration design tools (change impact, redesign)
        try:
            from ..tools.iteration import register_iteration_tools
            register_iteration_tools(self.tools)
        except Exception:
            pass

        # Register scheme comparison tools
        try:
            from ..tools.scheme_compare import register_scheme_tools
            register_scheme_tools(self.tools)
        except Exception:
            pass

        # Register production readiness check tools
        try:
            from ..tools.production_check import register_production_tools
            register_production_tools(self.tools)
        except Exception:
            pass

        # Register mass properties tools (part mass, COM, inertia, assembly)
        try:
            from ..tools.mass_properties import register_mass_properties_tools
            register_mass_properties_tools(self.tools)
        except Exception:
            pass

        # Register mobile base design tools (chassis, drive kinematics)
        try:
            from ..tools.mobile_design import register_mobile_design_tools
            register_mobile_design_tools(self.tools)
        except Exception:
            pass

        # Register drive train tools (motor → wheel assembly)
        try:
            from ..tools.drive_train import register_drive_train_tools
            register_drive_train_tools(self.tools)
        except Exception:
            pass

        # Register stability analysis tools (COM, support polygon, Force-Angle)
        try:
            from ..tools.stability import register_stability_tools
            register_stability_tools(self.tools)
        except Exception:
            pass

        # Register URDF export tools
        try:
            from ..tools.urdf_export import register_urdf_tools
            register_urdf_tools(self.tools)
        except Exception:
            pass

        # Register cable routing tools
        try:
            from ..tools.cable_routing import register_cable_routing_tools
            register_cable_routing_tools(self.tools)
        except Exception:
            pass

        # Register export package tools (engineering package export)
        try:
            from ..tools.export_package import register_export_package_tools
            register_export_package_tools(self.tools)
        except Exception:
            pass

        # Register assembly generator tools (NL → assembly definition)
        try:
            from ..tools.assembly_generator import register_assembly_generator_tools
            register_assembly_generator_tools(self.tools)
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
            # Check if this is a complex robot task that needs hierarchical planning
            if self._should_use_hierarchical(task):
                plan = self.planner.create_hierarchical_plan(task)
                self.state.plan = plan
                # Flatten for backward-compatible execution
                # (Phase 44 will add true phased execution via PhasedOrchestrator)
                flat_plan = plan.to_flat_plan()
                if use_orchestration and self._should_orchestrate(task, flat_plan):
                    return self._run_with_orchestration(task, flat_plan)
                else:
                    self.state.plan = flat_plan
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
                        fix_commands = extract_fix_commands(result)

                        # Check for convergence (stuck in loop)
                        if check_convergence(fix_history, result):
                            fix_hint = (
                                "[系统提示] 检测到修复陷入循环（连续多次失败原因相似）。"
                                "请尝试完全不同的建模方法，或删除当前模型从头开始重建。"
                            )
                        else:
                            fix_ctx.fix_history = fix_history
                            fix_hint = generate_fix_hint(fix_ctx, fix_commands=fix_commands)

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
