"""Verifier - checks whether step results meet expectations."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from ..models.base import Message
from ..models.router import ModelRouter, TaskType
from .state import PlanStep


class Verifier:
    """Verifies that step execution results match expectations."""

    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    def verify_step(self, step: PlanStep, execution_result: str) -> tuple[bool, str]:
        """Verify that a step completed successfully.

        Returns (success, message).
        """
        checks = []

        # Built-in verification heuristics
        if step.verification:
            heuristics_result = self._check_heuristics(step, execution_result)
            checks.append(heuristics_result)

        # If execution result contains errors
        # Exclude common false-positive phrases
        _error_exclude = {
            "no error", "error_log", "error_message",
            "error handling", "error_code: 0", "error_code:0",
            "exit code: 0",
        }
        if "Error:" in execution_result or "error" in execution_result.lower():
            error_lines = [
                line for line in execution_result.split("\n")
                if "error" in line.lower()
                and not any(phrase in line.lower() for phrase in _error_exclude)
            ]
            if error_lines and "exit code: 0" not in execution_result.lower():
                return False, f"Execution contained errors: {error_lines[0][:200]}"

        if not checks:
            # Default: if no explicit errors, consider it successful
            return True, "Step completed (no specific verification defined)"

        # All heuristic checks must pass
        all_passed = all(passed for passed, _ in checks)
        messages = [msg for _, msg in checks]
        return all_passed, "; ".join(messages)

    def _check_heuristics(self, step: PlanStep, result: str) -> tuple[bool, str]:
        """Run heuristic checks based on verification description."""
        import re as _re
        verification = step.verification.lower()

        # Check file existence (Chinese + English)
        if any(kw in verification for kw in ("文件存在", "file exists", "目录存在", "directory exists", "file created")):
            path_pattern = _re.compile(r'[\w./\\:-]+\.\w+|[./\\][\w./\\:-]+')
            for match in path_pattern.finditer(step.description):
                candidate = match.group()
                if Path(candidate).exists():
                    return True, f"File/directory found: {candidate}"

        # Check for successful code execution (Chinese + English)
        if any(kw in verification for kw in ("运行成功", "executes successfully", "runs successfully", "completed")):
            if "Exit code: 0" in result or ("error" not in result.lower()):
                return True, "Execution appears successful"

        # Use LLM for complex verification (Chinese + English)
        if any(kw in verification for kw in ["正确", "匹配", "包含", "符合",
                                              "correct", "matches", "contains", "matches pattern"]):
            return self._llm_verify(step, result)

        # No heuristic matched — try LLM; if that fails too, fail-safe
        llm_result = self._llm_verify(step, result)
        if llm_result is not None:
            return llm_result
        return False, f"No specific heuristic matched"

    def _llm_verify(self, step: PlanStep, result: str) -> tuple[bool, str]:
        """Use LLM to verify the result."""
        prompt = f"""请验证以下步骤是否成功完成。

步骤描述：{step.description}
验证条件：{step.verification}
执行结果（前1000字）：
{result[:1000]}

请回答：通过 或 失败，并简要说明原因。"""

        try:
            response = self.router.chat(
                messages=[Message(role="user", content=prompt)],
                task_type=TaskType.REASONING,
                max_tokens=256,
                temperature=0.3,
            )
            content = response.content.strip()
            content_lower = content.lower()
            fail_indicators = ["验证失败", "verification failed", "失败，", "失败。", "失败\n", "步骤失败"]
            pass_indicators = ["通过", "成功", "passed", "success", "succeeded"]
            has_fail = any(ind in content_lower for ind in fail_indicators)
            has_pass = any(ind in content_lower for ind in pass_indicators)
            if has_fail:
                return False, content
            if has_pass:
                return True, content
            # No clear indicators — fail-safe
            return False, content
        except Exception as e:
            # Fail-closed (AGENTS.md §1.1: never silently assume pass on
            # error). Previously this returned True on exception, which
            # would let a network blip pass a bad step. The pipeline's
            # production path uses assembly_verifier.py (not this method),
            # but any caller of verify_step() deserves a safe default.
            logger.warning("LLM verification failed (assuming FAIL): %s", e)
            return False, f"LLM verification unavailable, assumed FAIL: {e}"
