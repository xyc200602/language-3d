"""Task planner - breaks down high-level tasks into executable steps."""

from __future__ import annotations

import json
from typing import Any

from ..models.base import Message
from ..models.router import ModelRouter, TaskType
from .state import Plan, PlanStep, HierarchicalPlan, SubSystem, SystemDependency


PLANNER_SYSTEM_PROMPT = """你是一个任务规划专家。你的职责是将用户的高层任务分解为可执行的原子步骤。

每个步骤必须包含：
1. description: 清晰描述要做什么
2. expected_tools: 预期使用的工具列表（可选：file_read, file_write, file_edit, bash, python_exec, screen_capture, vlm_analyze）
3. verification: 如何验证这一步是否成功完成

输出格式要求：返回 JSON 数组，每个元素包含 description、expected_tools、verification 字段。
只输出 JSON，不要包含其他文字。

示例：
[
  {
    "description": "创建 Python 项目目录结构",
    "expected_tools": ["file_write", "bash"],
    "verification": "目录存在且包含 __init__.py"
  }
]
"""

DAG_PLANNER_SYSTEM_PROMPT = """你是一个任务规划专家，专门用于创建带有依赖关系的任务 DAG。

给定一个复杂任务，你需要：
1. 将任务分解为可执行的原子步骤
2. 确定每个步骤之间的依赖关系
3. 标识哪些步骤可以并行执行

每个步骤必须包含：
1. description: 清晰描述要做什么
2. expected_tools: 预期使用的工具列表
3. verification: 如何验证这一步是否成功完成
4. dependencies: 这一步依赖的前置步骤编号列表（0-indexed，空列表表示无依赖）

输出格式要求：返回 JSON 数组，每个元素包含 description、expected_tools、verification、dependencies 字段。
只输出 JSON，不要包含其他文字。

示例（3自由度机械臂）：
[
  {
    "description": "创建底座板（base_plate）",
    "expected_tools": ["fc_batch", "cad_verify"],
    "verification": "base_plate.FCStd 文件存在",
    "dependencies": []
  },
  {
    "description": "创建舵机安装座（servo_holder）",
    "expected_tools": ["fc_batch", "cad_verify"],
    "verification": "servo_holder.FCStd 文件存在",
    "dependencies": []
  },
  {
    "description": "创建底座旋转关节外壳（base_joint_housing）",
    "expected_tools": ["fc_batch", "cad_verify"],
    "verification": "base_joint_housing.FCStd 文件存在",
    "dependencies": [0]
  },
  {
    "description": "装配所有零件并验证",
    "expected_tools": ["fc_batch", "cad_verify"],
    "verification": "装配模型完整",
    "dependencies": [1, 2]
  }
]
"""


# Few-shot examples keyed by task type
PLANNER_EXAMPLES: dict[str, str] = {
    "assembly": """
示例（装配任务 - 3自由度机械臂）：
[
  {
    "description": "创建底座板（base_plate），100x80x10mm，四角有M4安装孔",
    "expected_tools": ["fc_batch", "cad_verify"],
    "verification": "base_plate.FCStd 文件存在且 cad_verify 通过"
  },
  {
    "description": "创建舵机安装座（servo_holder），适配 SG90 舵机",
    "expected_tools": ["fc_batch", "cad_verify"],
    "verification": "servo_holder.FCStd 文件存在且 cad_verify 通过"
  },
  {
    "description": "装配所有零件并验证配合关系",
    "expected_tools": ["fc_batch", "cad_verify"],
    "verification": "装配模型完整且无干涉"
  }
]
""",
    "single_part": """
示例（单个零件 - 带中心孔的方块）：
[
  {
    "description": "使用 fc_batch 创建 30x30x30mm 方块，中心打 10mm 通孔",
    "expected_tools": ["fc_batch", "cad_verify"],
    "verification": "方块模型文件存在且 cad_verify 验证通过"
  }
]
""",
    "part_usage": """
示例（使用零件库 - 含标准件的装配）：
[
  {
    "description": "搜索零件库中适合的螺钉类型（M4/M5）",
    "expected_tools": ["part_search"],
    "verification": "找到匹配的螺钉模板"
  },
  {
    "description": "生成 M4x20 内六角螺钉和匹配的六角螺母",
    "expected_tools": ["part_generate", "cad_verify"],
    "verification": "螺钉和螺母文件生成成功"
  },
  {
    "description": "使用 fc_batch 创建安装板（80x60x5mm，四角 M4 孔）",
    "expected_tools": ["fc_batch", "cad_verify"],
    "verification": "安装板文件存在且尺寸正确"
  },
  {
    "description": "装配螺钉、螺母和安装板，验证配合关系",
    "expected_tools": ["fc_batch", "cad_verify"],
    "verification": "装配模型完整且无干涉"
  }
]
""",
    "slicing": """
示例（3D 打印切片）：
[
  {
    "description": "使用 slice_model 将 STL 文件切片为 G-code（PLA/标准质量）",
    "expected_tools": ["slice_model"],
    "verification": "G-code 文件生成成功"
  },
  {
    "description": "使用 slice_analyze 分析 G-code 打印统计信息",
    "expected_tools": ["slice_analyze"],
    "verification": "获取打印时间、材料用量等信息"
  },
  {
    "description": "使用 slice_preview_layers 查看各层详情",
    "expected_tools": ["slice_preview_layers"],
    "verification": "获取每层 Z 高度和挤出数据"
  }
]
""",
    "complex_robot": """
示例（复杂机器人 - 4轮差速底盘 + 双臂 + 工控机）：
将任务分解为子系统（subsystems），每个子系统独立设计。

必须输出以下 JSON 格式：
{
  "subsystems": [
    {
      "name": "mobile_base",
      "description": "4轮差速移动底盘",
      "parts": ["chassis_plate", "wheel_fl", "wheel_fr", "wheel_bl", "wheel_br", "motor_fl", "motor_fr", "motor_bl", "motor_br", "motor_bracket"],
      "joints": ["wheel_fl_joint", "wheel_fr_joint", "wheel_bl_joint", "wheel_br_joint"],
      "steps": [
        {"description": "创建底盘板 chassis_plate（200x150x5mm）", "expected_tools": ["fc_batch"], "verification": "文件存在"},
        {"description": "创建轮子 wheel（直径60mm，仅建模1次）", "expected_tools": ["fc_batch"], "verification": "文件存在"},
        {"description": "创建电机支架 motor_bracket", "expected_tools": ["fc_batch"], "verification": "文件存在"},
        {"description": "装配底盘子系统", "expected_tools": ["assembly_solve"], "verification": "装配完整"}
      ]
    },
    {
      "name": "arm_left",
      "description": "左侧3-DOF机械臂",
      "parts": ["shoulder_link_l", "upper_arm_l", "forearm_l", "gripper_l", "servo_l1", "servo_l2", "servo_l3"],
      "steps": [...]
    }
  ],
  "system_dependencies": [
    {"source": "mobile_base", "target": "ipc_mount", "reason": "工控机安装在底盘上"},
    {"source": "mobile_base", "target": "arm_left", "reason": "左臂安装在底盘上"},
    {"source": "mobile_base", "target": "arm_right", "reason": "右臂安装在底盘上"}
  ],
  "symmetry": [
    {"source": "arm_left", "target": "arm_right", "type": "mirror"},
    {"source": "wheel", "target": ["wheel_fl", "wheel_fr", "wheel_bl", "wheel_br"], "type": "instance", "count": 4}
  ],
  "integration_steps": [
    {"description": "整机装配验证", "expected_tools": ["assembly_solve", "interference_check"], "verification": "无干涉"},
    {"description": "稳定性分析", "expected_tools": [], "verification": "质心在支撑多边形内"}
  ]
}
""",
    "mobile_base": """
示例（移动底盘设计 - 4轮差速底盘）：
[
  {"description": "使用 mobile_base_design 工具根据需求参数推导底盘设计参数", "expected_tools": ["mobile_base_design"], "verification": "返回电机选型和电池容量"},
  {"description": "根据设计参数创建底盘框架零件", "expected_tools": ["fc_batch"], "verification": "底盘文件存在且体积正确"},
  {"description": "创建驱动轮和电机支架零件", "expected_tools": ["fc_batch"], "verification": "所有轮子和支架文件存在"},
  {"description": "装配底盘所有零件并验证", "expected_tools": ["assembly_solve", "cad_verify"], "verification": "装配无干涉"},
  {"description": "计算装配体质量和质心位置", "expected_tools": ["compute_assembly_properties"], "verification": "质心在合理范围内"},
  {"description": "差速运动学仿真验证转弯性能", "expected_tools": ["differential_drive_sim"], "verification": "轨迹合理"}
]
""",
}
COMPLEX_ROBOT_KEYWORDS = [
    "四轮", "4轮", "轮式", "差速", "移动底盘", "移动机器人",
    "底盘", "麦克纳姆", "mecanum", "全向轮",
    "双臂", "2臂", "两臂", "两个机械臂",
    "多臂", "多关节", "多自由度",
    "工控机", "ipc", "工业pc",
    "移动平台", "agv", "移动基座",
    "四足", "4足", "轮腿",
    "巡检机器人", "仓储机器人", "服务机器人",
    "robotic vehicle", "mobile robot", "wheeled robot",
    "differential drive", "omnidirectional",
    "dual arm", "two arm", "mobile manipulator",
    "4 wheel", "four wheel",
]


# Keywords for detecting complex robot tasks# Patterns for symmetry detection
SYMMETRY_PATTERNS: list[dict[str, Any]] = [
    {
        "name": "wheel_4",
        "indicators": ["四轮", "4轮", "4个轮", "四个轮", "4 wheel", "four wheel"],
        "template": {"part_name": "wheel", "count": 4, "positions": ["fl", "fr", "bl", "br"]},
    },
    {
        "name": "wheel_2",
        "indicators": ["两轮", "2轮", "2个轮", "双轮", "2 wheel"],
        "template": {"part_name": "wheel", "count": 2, "positions": ["left", "right"]},
    },
    {
        "name": "dual_arm",
        "indicators": ["双臂", "两臂", "2臂", "两个臂", "左右臂", "dual arm", "two arm"],
        "template": {"source": "arm_left", "target": "arm_right", "type": "mirror"},
    },
    {
        "name": "dual_leg",
        "indicators": ["双腿", "两腿", "2腿", "两个腿", "dual leg", "bipedal"],
        "template": {"source": "leg_left", "target": "leg_right", "type": "mirror"},
    },
]


class Planner:
    """Breaks down tasks into executable plans."""

    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    @staticmethod
    def _detect_task_type(task: str) -> str:
        """Detect the task type from the task description."""
        task_lower = task.lower()
        # Complex robot keywords (highest priority)
        for kw in COMPLEX_ROBOT_KEYWORDS:
            if kw in task_lower:
                return "complex_robot"
        # Mobile base keywords
        mobile_base_keywords = [
            "底盘设计", "移动底盘", "差速底盘", "轮式底盘",
            "agv底盘", "巡检底盘", "mecanum底盘",
            "mobile base", "wheel base", "chassis design",
        ]
        for kw in mobile_base_keywords:
            if kw in task_lower:
                return "mobile_base"
        assembly_keywords = [
            "装配", "组装", "assembly", "多个零件", "机械臂",
            "机器人", "关节", "连接", "底座",
        ]
        for kw in assembly_keywords:
            if kw in task_lower:
                return "assembly"
        # Part library keywords
        part_keywords = [
            "零件库", "标准件", "螺钉", "螺栓", "螺母", "垫圈", "轴承",
            "舵机", "步进电机", "齿轮", "联轴器", "光轴",
            "part library", "standard part", "screw", "bolt", "nut",
            "bearing", "servo", "stepper", "gear", "shaft",
        ]
        for kw in part_keywords:
            if kw in task_lower:
                return "part_usage"
        # Slicing keywords
        slicing_keywords = [
            "切片", "slice", "g-code", "gcode", "3d打印", "3d print",
            "打印", "slicer", "切片器", "切片分析",
        ]
        for kw in slicing_keywords:
            if kw in task_lower:
                return "slicing"
        return "single_part"

    def create_plan(self, task: str, context: str = "") -> Plan:
        """Create an execution plan for a task."""
        user_message = f"任务：{task}"
        if context:
            user_message += f"\n\n上下文信息：\n{context}"

        # Inject few-shot examples based on task type
        task_type = self._detect_task_type(task)
        system_prompt = PLANNER_SYSTEM_PROMPT
        example = PLANNER_EXAMPLES.get(task_type)
        if example:
            system_prompt = system_prompt + "\n" + example

        response = self.router.chat(
            messages=[Message(role="user", content=user_message)],
            system=system_prompt,
            task_type=TaskType.PLANNING,
            temperature=0.5,
        )

        steps = self._parse_plan_response(response.content)
        plan = Plan(goal=task, steps=steps)
        self._validate_plan(plan)
        return plan

    def replan_from_failure(
        self,
        plan: Plan,
        failed_step: PlanStep,
        error: str,
        reflection: str = "",
    ) -> PlanStep | None:
        """Generate a replacement step when a step fails."""
        prompt = f"""原计划目标：{plan.goal}

失败的步骤：{failed_step.description}
错误信息：{error}
{f'分析：{reflection}' if reflection else ''}

请提供一个替代步骤来完成这个目标。只输出一个 JSON 对象，包含 description、expected_tools、verification 字段。"""

        response = self.router.chat(
            messages=[Message(role="user", content=prompt)],
            system=PLANNER_SYSTEM_PROMPT,
            task_type=TaskType.PLANNING,
            temperature=0.5,
        )

        steps = self._parse_plan_response(response.content)
        return steps[0] if steps else None

    def _parse_plan_response(self, content: str) -> list[PlanStep]:
        """Parse the LLM response into PlanStep objects."""
        # Try to extract JSON from the response
        text = content.strip()

        # Remove markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (code fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON array in the text
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    # If all else fails, create a single step from the text
                    return [PlanStep(description=content, expected_tools=[], verification="手动验证")]
            else:
                return [PlanStep(description=content, expected_tools=[], verification="手动验证")]

        steps: list[PlanStep] = []
        for item in data:
            if isinstance(item, dict):
                steps.append(
                    PlanStep(
                        description=item.get("description", str(item)),
                        expected_tools=item.get("expected_tools", []),
                        verification=item.get("verification", ""),
                    )
                )

        return steps

    @staticmethod
    def _validate_plan(plan: Plan) -> None:
        """Validate and auto-fix plan steps.

        - For modeling steps using fc_batch, add cad_verify to expected_tools.
        - For modeling steps missing verification, add a cad_verify-based check.
        """
        for step in plan.steps:
            tools_lower = [t.lower() for t in step.expected_tools]

            # Add cad_verify to modeling steps that use fc_batch but lack verification tool
            if any("fc_batch" in t or "fc_menu" in t for t in tools_lower):
                if "cad_verify" not in tools_lower:
                    step.expected_tools.append("cad_verify")

            # Add verification text for modeling steps that lack it
            has_modeling = any(
                "fc_" in t or "建模" in step.description or "创建" in step.description
                for t in tools_lower
            )
            if has_modeling and not step.verification:
                step.verification = "模型文件存在且 cad_verify 通过"

    def create_dag_plan(
        self,
        task: str,
        context: str = "",
        assembly: Any | None = None,
    ) -> tuple[Plan, dict[str, list[str]]]:
        """Create a plan with explicit step dependencies for DAG execution.

        Args:
            task: The task description.
            context: Optional additional context.
            assembly: Optional Assembly object for dependency inference.

        Returns:
            Tuple of (Plan, dependencies_dict) where dependencies_dict maps
            step_id -> list of dependent step_ids.
        """
        user_message = f"任务：{task}"
        if context:
            user_message += f"\n\n上下文信息：\n{context}"

        response = self.router.chat(
            messages=[Message(role="user", content=user_message)],
            system=DAG_PLANNER_SYSTEM_PROMPT,
            task_type=TaskType.PLANNING,
            temperature=0.5,
        )

        steps, dep_indices = self._parse_dag_response(response.content)
        plan = Plan(goal=task, steps=steps)

        # Convert index-based dependencies to ID-based
        dependencies_dict: dict[str, list[str]] = {}
        for step_idx, dep_idxs in dep_indices.items():
            step = steps[step_idx]
            dep_ids = [steps[i].id for i in dep_idxs if i < len(steps)]
            step.dependencies = dep_ids
            dependencies_dict[step.id] = dep_ids

        # If assembly provided, enhance dependencies from joints
        if assembly is not None:
            name_to_step_id: dict[str, str] = {}
            for step in steps:
                for part in getattr(assembly, "parts", []):
                    if part.name in step.description.lower() or part.name.replace("_", " ") in step.description.lower():
                        name_to_step_id[part.name] = step.id
                        break

            for joint in getattr(assembly, "joints", []):
                child = getattr(joint, "child", "")
                parent = getattr(joint, "parent", "")
                child_step_id = name_to_step_id.get(child)
                parent_step_id = name_to_step_id.get(parent)
                if child_step_id and parent_step_id:
                    if parent_step_id not in dependencies_dict.get(child_step_id, []):
                        step = next(s for s in steps if s.id == child_step_id)
                        if parent_step_id not in step.dependencies:
                            step.dependencies.append(parent_step_id)
                        dependencies_dict.setdefault(child_step_id, []).append(parent_step_id)

        return plan, dependencies_dict

    def _parse_dag_response(
        self, content: str
    ) -> tuple[list[PlanStep], dict[int, list[int]]]:
        """Parse LLM response with dependency information.

        Returns (steps, dep_indices) where dep_indices maps step index -> dep indices.
        """
        text = content.strip()

        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    return [PlanStep(description=content, expected_tools=[], verification="手动验证")], {}
            else:
                return [PlanStep(description=content, expected_tools=[], verification="手动验证")], {}

        steps: list[PlanStep] = []
        dep_indices: dict[int, list[int]] = {}

        for idx, item in enumerate(data):
            if isinstance(item, dict):
                steps.append(
                    PlanStep(
                        description=item.get("description", str(item)),
                        expected_tools=item.get("expected_tools", []),
                        verification=item.get("verification", ""),
                    )
                )
                deps = item.get("dependencies", [])
                if deps:
                    dep_indices[idx] = [int(d) for d in deps]

        return steps, dep_indices

    # --- Hierarchical Planning (Phase E: Task 43) ---

    HIERARCHICAL_PLANNER_PROMPT = """你是一个复杂机器人系统设计专家。你的职责是将高层机器人设计任务分解为多个子系统，每个子系统独立设计。

分解规则：
1. 按功能分组（驱动系统、机械臂、电子设备、传感器），不要按零件类型分组
2. 每个子系统应包含 3-10 个零件，独立可验证
3. 对称零件只需在源子系统建模一次（如 4 个轮子→只建模 1 次，标记 instance_count=4）
4. 镜像零件只需建模一侧（如双臂→只建模左臂，右臂标记 mirror_of="arm_left"）
5. 子系统之间通过接口约束关联（如底盘顶面螺孔位置→工控机支架底面）

输出 JSON 格式：
{
  "subsystems": [
    {
      "name": "子系统英文标识（如 mobile_base）",
      "description": "子系统中文描述",
      "parts": ["part_name_1", "part_name_2"],
      "joints": ["joint_name_1"],
      "steps": [
        {"description": "步骤描述", "expected_tools": ["fc_batch"], "verification": "验证方式"}
      ],
      "mirror_of": "",
      "instance_count": 1
    }
  ],
  "system_dependencies": [
    {"source": "mobile_base", "target": "arm_left", "reason": "臂安装在底盘上"}
  ],
  "integration_steps": [
    {"description": "整机装配验证", "expected_tools": ["assembly_solve"], "verification": "无干涉"}
  ]
}

只输出 JSON，不要包含其他文字。"""

    @staticmethod
    def detect_symmetry(task: str) -> list[dict[str, Any]]:
        """Detect symmetric parts from task description.

        Returns a list of symmetry descriptors, each containing:
        - source: the part/subsystem to model once
        - targets: the instances/mirrors to create
        - type: "instance" (identical copies) or "mirror" (mirrored copies)
        - count: number of copies
        """
        task_lower = task.lower()
        detected: list[dict[str, Any]] = []

        for pattern in SYMMETRY_PATTERNS:
            for indicator in pattern["indicators"]:
                if indicator in task_lower:
                    tmpl = pattern["template"]
                    if pattern["name"] in ("wheel_4", "wheel_2"):
                        detected.append({
                            "source": tmpl["part_name"],
                            "targets": [f"{tmpl['part_name']}_{p}" for p in tmpl["positions"]],
                            "type": "instance",
                            "count": tmpl["count"],
                        })
                    elif pattern["name"] in ("dual_arm", "dual_leg"):
                        prefix = pattern["name"].replace("dual_", "")
                        detected.append({
                            "source": f"{prefix}_left",
                            "target": f"{prefix}_right",
                            "type": "mirror",
                            "count": 1,
                        })
                    break  # Only match once per pattern

        return detected

    def create_hierarchical_plan(self, task: str, context: str = "") -> HierarchicalPlan:
        """Create a hierarchical plan decomposed into subsystems.

        Used for complex robot designs with 15+ parts. The LLM decomposes
        the task into subsystems, each with its own isolated steps.
        """
        symmetry = self.detect_symmetry(task)
        symmetry_hint = ""
        if symmetry:
            lines = ["检测到对称性，请遵循以下规则："]
            for s in symmetry:
                if s["type"] == "instance":
                    lines.append(f"- {s['source']} 只建模1次，然后实例化 {s['count']} 个副本：{', '.join(s['targets'])}")
                elif s["type"] == "mirror":
                    lines.append(f"- {s['source']} 只建模1次，{s['target']} 为镜像复制（mirror_of=\"{s['source']}\"）")
            symmetry_hint = "\n".join(lines)

        user_message = f"任务：{task}"
        if context:
            user_message += f"\n\n上下文信息：\n{context}"
        if symmetry_hint:
            user_message += f"\n\n{symmetry_hint}"

        # Inject complex_robot example
        system_prompt = self.HIERARCHICAL_PLANNER_PROMPT
        example = PLANNER_EXAMPLES.get("complex_robot", "")
        if example:
            system_prompt += "\n" + example

        response = self.router.chat(
            messages=[Message(role="user", content=user_message)],
            system=system_prompt,
            task_type=TaskType.PLANNING,
            temperature=0.5,
        )

        return self._parse_hierarchical_response(task, response.content, symmetry)

    def _parse_hierarchical_response(
        self, task: str, content: str, symmetry: list[dict[str, Any]],
    ) -> HierarchicalPlan:
        """Parse LLM response into a HierarchicalPlan."""
        text = content.strip()

        # Remove markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        # Try to parse JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    # Fallback: create a single subsystem
                    return HierarchicalPlan(
                        goal=task,
                        subsystems=[SubSystem(
                            name="main",
                            description=task,
                            steps=[PlanStep(description=content)],
                        )],
                    )
            else:
                return HierarchicalPlan(
                    goal=task,
                    subsystems=[SubSystem(
                        name="main",
                        description=task,
                        steps=[PlanStep(description=content)],
                    )],
                )

        # Parse subsystems
        subsystems: list[SubSystem] = []
        for ss_data in data.get("subsystems", []):
            if not isinstance(ss_data, dict):
                continue
            steps = []
            for step_data in ss_data.get("steps", []):
                if isinstance(step_data, dict):
                    steps.append(PlanStep(
                        description=step_data.get("description", ""),
                        expected_tools=step_data.get("expected_tools", []),
                        verification=step_data.get("verification", ""),
                    ))
            subsystems.append(SubSystem(
                name=ss_data.get("name", "unknown"),
                description=ss_data.get("description", ""),
                parts=ss_data.get("parts", []),
                joints=ss_data.get("joints", []),
                steps=steps,
                mirror_of=ss_data.get("mirror_of", ""),
                instance_count=ss_data.get("instance_count", 1),
            ))

        # Apply detected symmetry if LLM didn't fill it in
        if symmetry:
            for s in symmetry:
                if s["type"] == "mirror":
                    # Find the target subsystem and set mirror_of
                    target_name = s.get("target", "")
                    source_name = s.get("source", "")
                    for ss in subsystems:
                        if ss.name == target_name and not ss.mirror_of:
                            ss.mirror_of = source_name
                elif s["type"] == "instance":
                    # Find the source part's subsystem and set instance_count
                    source = s.get("source", "")
                    for ss in subsystems:
                        if source in ss.parts or source in ss.name:
                            if ss.instance_count <= 1:
                                ss.instance_count = s.get("count", 1)

        # Parse system dependencies
        sys_deps: list[SystemDependency] = []
        for dep_data in data.get("system_dependencies", []):
            if isinstance(dep_data, dict):
                sys_deps.append(SystemDependency(
                    source=dep_data.get("source", ""),
                    target=dep_data.get("target", ""),
                    reason=dep_data.get("reason", ""),
                ))

        # Parse integration steps
        integration_steps: list[PlanStep] = []
        for step_data in data.get("integration_steps", []):
            if isinstance(step_data, dict):
                integration_steps.append(PlanStep(
                    description=step_data.get("description", ""),
                    expected_tools=step_data.get("expected_tools", []),
                    verification=step_data.get("verification", ""),
                ))

        plan = HierarchicalPlan(
            goal=task,
            subsystems=subsystems,
            system_dependencies=sys_deps,
            integration_steps=integration_steps,
        )

        # Validate: add cad_verify to modeling steps
        self._validate_plan(plan.to_flat_plan())

        return plan
