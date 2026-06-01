"""Reflector - analyzes failures and generates improvement suggestions."""

from __future__ import annotations

import re

from ..models.base import Message
from ..models.router import ModelRouter, TaskType
from .state import Plan, PlanStep


REFLECTOR_SYSTEM_PROMPT = """你是一个自我反思专家。当一个执行步骤失败时，你需要：
1. 分析失败的根本原因
2. 提出具体的修复建议
3. 建议下一步行动

对于 VLM 视觉验证失败的场合：
- 如果 cad_verify 返回 match=false，根据 DIFFERENCES 分析问题
- 根据 FIX_COMMANDS 生成具体的修正操作
- 建议使用 fc_batch 重新建模

保持简洁，用中文回答。"""


class Reflector:
    """Analyzes failures and generates improvement suggestions."""

    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    def reflect(
        self,
        plan: Plan,
        step: PlanStep,
        error: str,
        tool_history: list[dict] | None = None,
    ) -> str:
        """Analyze a failure and generate reflection.

        Returns a reflection string with analysis and suggestions.
        """
        # Check if this is a VLM verification failure
        vlm_context = self._extract_vlm_feedback(error, tool_history)

        # Build context about what happened
        recent_tools = ""
        if tool_history:
            last_5 = tool_history[-5:]
            recent_tools = "\n".join(
                f"- {t['name']}({list(t.get('arguments', {}).keys())}): "
                f"{t.get('result', '')[:200]}"
                for t in last_5
            )

        vlm_section = ""
        if vlm_context:
            vlm_section = f"""
## VLM 视觉验证反馈
{vlm_context}

请根据 VLM 反馈分析问题，并建议具体的修正操作（如 fc_batch 操作列表）。"""

        prompt = f"""## 任务目标
{plan.goal}

## 失败的步骤
{step.description}

## 错误信息
{error}

## 之前的工具调用
{recent_tools or '（无）'}

## 步骤已尝试次数
{step.attempts}
{vlm_section}

请分析失败原因并提出修复建议。"""

        response = self.router.chat(
            messages=[Message(role="user", content=prompt)],
            system=REFLECTOR_SYSTEM_PROMPT,
            task_type=TaskType.REASONING,
            max_tokens=1024,
            temperature=0.5,
        )

        return response.content

    def _extract_vlm_feedback(
        self,
        error: str,
        tool_history: list[dict] | None = None,
    ) -> str:
        """Extract VLM verification feedback from error/result history."""
        parts = []

        # Check error string itself for VLM verification output
        if "MATCH: False" in error or "MATCH: false" in error or '"match": false' in error:
            parts.append(f"验证结果: 模型不匹配\n{error[:500]}")

        # Check tool history for cad_verify results
        if tool_history:
            for t in reversed(tool_history[-3:]):
                if t["name"] == "cad_verify":
                    result = t.get("result", "")
                    if "MATCH: False" in result or "MATCH: false" in result or '"match": false' in result:
                        parts.append(f"cad_verify 结果:\n{result[:500]}")

        return "\n".join(parts) if parts else ""
