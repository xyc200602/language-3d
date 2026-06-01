"""Task planner - breaks down high-level tasks into executable steps."""

from __future__ import annotations

import json
from typing import Any

from ..models.base import Message
from ..models.router import ModelRouter, TaskType
from .state import Plan, PlanStep


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


class Planner:
    """Breaks down tasks into executable plans."""

    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    def create_plan(self, task: str, context: str = "") -> Plan:
        """Create an execution plan for a task."""
        user_message = f"任务：{task}"
        if context:
            user_message += f"\n\n上下文信息：\n{context}"

        response = self.router.chat(
            messages=[Message(role="user", content=user_message)],
            system=PLANNER_SYSTEM_PROMPT,
            task_type=TaskType.PLANNING,
            temperature=0.5,
        )

        steps = self._parse_plan_response(response.content)
        return Plan(goal=task, steps=steps)

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
