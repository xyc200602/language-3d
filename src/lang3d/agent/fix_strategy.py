"""Failure classification engine and targeted fix strategies.

Classifies cad_verify failures into specific types and generates
targeted repair hints instead of generic prompts.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureType(str, Enum):
    MISSING_FEATURE = "missing_feature"
    WRONG_DIMENSION = "wrong_dimension"
    WRONG_POSITION = "wrong_position"
    WRONG_SHAPE = "wrong_shape"
    EXTRA_FEATURE = "extra_feature"
    ASSEMBLY_ERROR = "assembly_error"
    UNKNOWN = "unknown"


# Keyword patterns for failure classification
_MISSING_KEYWORDS = [
    "missing", "缺少", "缺失", "not found", "不存在", "未找到",
    "no hole", "没有孔", "没有槽", "no slot", "no chamfer",
    "no fillet", "没有倒角", "没有圆角",
]
_DIMENSION_KEYWORDS = [
    "dimension", "尺寸", "size", "大小", "too big", "too small",
    "宽度", "高度", "长度", "width", "height", "length",
    "thickness", "厚度", "diameter", "直径", "radius", "半径",
    "mm", "not mm", "incorrect mm", "wrong mm",
]
_POSITION_KEYWORDS = [
    "position", "位置", "offset", "偏移", "misalign", "不对齐",
    "偏心", "center", "居中", "located", "坐标",
]
_SHAPE_KEYWORDS = [
    "shape", "形状", "wrong shape", "geometry", "几何",
    "instead of", "而不是", "should be", "应该是",
    "cube.*cylinder", "cylinder.*cube", "box.*sphere",
]
_EXTRA_KEYWORDS = [
    "extra", "多余", "additional", "unexpected", "unexpected",
    "不应有", "多余的",
]
_ASSEMBLY_KEYWORDS = [
    "assembly", "装配", "组装", "mate", "align", "constraint",
    "干涉", "interference", "overlap", "重叠", "间隙", "gap",
    "joint", "连接",
]


@dataclass
class FixContext:
    """Context for a fix attempt."""

    failure_type: FailureType = FailureType.UNKNOWN
    description: str = ""
    target_feature: str = ""
    expected_value: str = ""
    actual_value: str = ""
    fix_history: list[str] = field(default_factory=list)


def classify_failure(cad_verify_result: str, expected: str = "") -> FixContext:
    """Classify a cad_verify failure result into a specific failure type.

    Uses keyword-based heuristic classification.
    """
    text = (cad_verify_result + " " + expected).lower()

    def _score(keywords: list[str]) -> int:
        return sum(1 for kw in keywords if kw in text)

    scores = {
        FailureType.MISSING_FEATURE: _score(_MISSING_KEYWORDS),
        FailureType.WRONG_DIMENSION: _score(_DIMENSION_KEYWORDS),
        FailureType.WRONG_POSITION: _score(_POSITION_KEYWORDS),
        FailureType.WRONG_SHAPE: _score(_SHAPE_KEYWORDS),
        FailureType.EXTRA_FEATURE: _score(_EXTRA_KEYWORDS),
        FailureType.ASSEMBLY_ERROR: _score(_ASSEMBLY_KEYWORDS),
    }

    best_type = max(scores, key=scores.get)
    if scores[best_type] == 0:
        best_type = FailureType.UNKNOWN

    # Extract key info from DIFFERENCES field
    target_feature = ""
    expected_value = ""
    actual_value = ""
    for line in cad_verify_result.split("\n"):
        if "DIFFERENCES:" in line or "DIFFERENCES：" in line:
            diff_text = line.split(":", 1)[-1].strip() if ":" in line else line.split("：", 1)[-1].strip()
            target_feature = diff_text
        if "OBSERVED:" in line or "OBSERVED：" in line:
            actual_value = line.split(":", 1)[-1].strip() if ":" in line else line.split("：", 1)[-1].strip()

    return FixContext(
        failure_type=best_type,
        description=cad_verify_result[:500],
        target_feature=target_feature,
        expected_value=expected[:200],
        actual_value=actual_value,
    )


# Targeted fix hints per failure type (in Chinese for the agent)
_FIX_HINTS: dict[FailureType, str] = {
    FailureType.MISSING_FEATURE: (
        "检测到缺少特征。请检查 DIFFERENCES 中描述的缺失部分，"
        "使用 fc_batch 添加对应的特征（如打孔、开槽、倒角、圆角等）。"
        "确保添加操作基于正确的参考面和位置。"
    ),
    FailureType.WRONG_DIMENSION: (
        "检测到尺寸不正确。请检查具体的尺寸差异，"
        "使用 fc_batch 修改对应特征的参数（长度、宽度、高度、直径、半径等）。"
        "注意单位为毫米(mm)。"
    ),
    FailureType.WRONG_POSITION: (
        "检测到位置偏移。请检查特征的定位参数，"
        "使用 fc_batch 调整特征的位置（偏移量、参考点、对齐方式）。"
        "可能需要修改草图(Sketch)的约束。"
    ),
    FailureType.WRONG_SHAPE: (
        "检测到形状不正确。请重新检查建模步骤，"
        "可能需要删除当前特征并使用正确的几何体重新创建。"
        "确认使用了正确的建模方法（拉伸/旋转/扫掠/放样）。"
    ),
    FailureType.EXTRA_FEATURE: (
        "检测到多余的特征。请检查模型是否有不应存在的几何体，"
        "使用 fc_batch 删除多余的特征或几何体。"
    ),
    FailureType.ASSEMBLY_ERROR: (
        "检测到装配问题。请检查零件之间的配合关系，"
        "确认约束类型和参考面是否正确。"
        "检查是否有干涉或间隙问题。"
    ),
    FailureType.UNKNOWN: (
        "cad_verify 检测到模型不匹配。请仔细分析 DIFFERENCES 和 SUGGESTION，"
        "使用 fc_batch 修正模型，然后重新验证。"
    ),
}


def extract_fix_commands(result: str) -> str:
    """Extract FIX_COMMANDS from a cad_verify result string.

    Handles both single-angle format (FIX_COMMANDS: ...) and
    multi-angle format (looking for fix_commands in raw VLM output).

    Returns empty string for "None", "null", or missing FIX_COMMANDS.
    """
    for line in result.split("\n"):
        stripped = line.strip()
        if stripped.startswith("FIX_COMMANDS:"):
            value = stripped[len("FIX_COMMANDS:"):].strip()
            if value.lower() in ("none", "null", ""):
                return ""
            return value

    # Try to find fix_commands in JSON-like structures within raw VLM output
    match = re.search(r'"fix_commands"\s*:\s*"([^"]+)"', result)
    if match:
        value = match.group(1).strip()
        if value.lower() in ("none", "null", ""):
            return ""
        return value

    return ""


def generate_fix_hint(ctx: FixContext, fix_commands: str = "") -> str:
    """Generate a targeted fix hint based on failure type.

    Args:
        ctx: The failure context with classification info.
        fix_commands: Optional FIX_COMMANDS extracted from VLM cad_verify result.
    """
    base_hint = _FIX_HINTS.get(ctx.failure_type, _FIX_HINTS[FailureType.UNKNOWN])
    parts = [f"[系统提示] {base_hint}"]

    if ctx.target_feature:
        parts.append(f"具体差异：{ctx.target_feature}")
    if ctx.actual_value:
        parts.append(f"当前状态：{ctx.actual_value}")
    if ctx.expected_value:
        parts.append(f"期望结果：{ctx.expected_value}")
    if fix_commands:
        parts.append(f"VLM 建议的具体修复操作：\n{fix_commands}")

    return "\n".join(parts)


def parse_cad_verify_failure(result: str, expected: str = "") -> tuple[FixContext, str]:
    """Convenience function: classify failure and extract FIX_COMMANDS in one call.

    Returns:
        A tuple of (FixContext, fix_commands_string).
    """
    ctx = classify_failure(result, expected)
    fix_commands = extract_fix_commands(result)
    return ctx, fix_commands


def check_convergence(previous_fixes: list[str], current_result: str, threshold: float = 0.8) -> bool:
    """Detect if the fix loop is stuck (converging on the same failure).

    Returns True if the current result is too similar to previous fixes,
    indicating the agent is stuck in a loop.
    """
    if not previous_fixes:
        return False

    current = current_result[:500].lower().strip()
    for prev in previous_fixes[-3:]:
        prev_normalized = prev[:500].lower().strip()
        ratio = difflib.SequenceMatcher(None, current, prev_normalized).ratio()
        if ratio > threshold:
            return True

    return False
