"""Agent tool for part accessory recommendations."""

from __future__ import annotations

import json
from typing import Any

from ..knowledge.part_recommendations import (
    RecommendationResult,
    compute_compatibility_score,
    get_recommendations,
    get_recommendations_for_assembly,
)
from ..models.base import ToolDefinition
from .base import Tool


class PartRecommendTool(Tool):
    """Recommend compatible accessories for a part or entire assembly."""

    name = "part_recommend"
    description = (
        "零件推荐：根据选定零件推荐兼容配件（螺丝、支架、轴承等）。"
        "支持单零件推荐和全装配体批量推荐。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "part_id": {
                        "type": "string",
                        "description": "零件目录ID（如 'servo_sg90', 'nema17_stepper'），"
                                       "与 assembly_json 二选一",
                    },
                    "assembly_json": {
                        "type": "string",
                        "description": "装配体 JSON 定义，用于全装配体批量推荐",
                    },
                    "check_compatibility": {
                        "type": "string",
                        "description": "检查两个零件的兼容性评分，格式: 'part_a,part_b'",
                    },
                },
                "required": [],
            },
        )

    def execute(
        self,
        *,
        part_id: str = "",
        assembly_json: str = "",
        check_compatibility: str = "",
        **kwargs: Any,
    ) -> str:
        # Compatibility check mode
        if check_compatibility and "," in check_compatibility:
            parts = check_compatibility.split(",", 1)
            score = compute_compatibility_score(parts[0].strip(), parts[1].strip())
            return (
                f"[Part Recommend] 兼容性评分:\n"
                f"  {parts[0].strip()} ↔ {parts[1].strip()}: {score:.2f}\n"
                f"  (0.0=不兼容, 1.0=完美匹配)"
            )

        # Assembly-level recommendation
        if assembly_json:
            return self._recommend_for_assembly(assembly_json)

        # Single part recommendation
        if part_id:
            return self._recommend_single(part_id)

        return "[Part Recommend] 请提供 part_id 或 assembly_json 参数"

    def _recommend_single(self, part_id: str) -> str:
        result = get_recommendations(part_id)

        lines = [
            f"[Part Recommend] 零件: {part_id}",
            f"匹配分数: {result.compatibility_score:.2f}",
            f"推荐配件数: {len(result.recommended_parts)}",
            "",
            result.summary,
        ]

        if result.recommended_parts:
            lines.append("")
            lines.append("--- 推荐配件 ---")
            for rec in result.recommended_parts:
                req = "必需" if rec.required else "可选"
                lines.append(
                    f"  {rec.part_catalog_id} ×{rec.quantity} "
                    f"({req}) — {rec.reason}"
                )

        lines.append("")
        lines.append("--- JSON ---")
        lines.append(json.dumps({
            "trigger_part": result.trigger_part,
            "compatibility_score": result.compatibility_score,
            "recommended_parts": [
                {
                    "part_catalog_id": r.part_catalog_id,
                    "quantity": r.quantity,
                    "reason": r.reason,
                    "required": r.required,
                }
                for r in result.recommended_parts
            ],
        }, ensure_ascii=False, indent=2))

        return "\n".join(lines)

    def _recommend_for_assembly(self, assembly_json: str) -> str:
        from .assembly_solver import _parse_assembly_json

        try:
            asm = _parse_assembly_json(assembly_json)
        except Exception as e:
            return f"[Part Recommend] 错误：无法解析装配体 JSON - {e}"

        results = get_recommendations_for_assembly(asm)

        # Aggregate all recommendations
        aggregate: dict[str, dict[str, Any]] = {}
        for res in results:
            for rec in res.recommended_parts:
                if rec.part_catalog_id in aggregate:
                    aggregate[rec.part_catalog_id]["quantity"] += rec.quantity
                else:
                    aggregate[rec.part_catalog_id] = {
                        "part_catalog_id": rec.part_catalog_id,
                        "quantity": rec.quantity,
                        "reason": rec.reason,
                        "required": rec.required,
                    }

        lines = [
            f"[Part Recommend] 装配体: {asm.name}",
            f"触发零件数: {len(results)}",
            f"去重后推荐配件: {len(aggregate)}",
            "",
            "--- 推荐配件清单 ---",
        ]

        for pid, info in sorted(aggregate.items()):
            req = "必需" if info["required"] else "可选"
            lines.append(
                f"  {pid} ×{info['quantity']} ({req}) — {info['reason']}"
            )

        lines.append("")
        lines.append("--- JSON ---")
        lines.append(json.dumps({
            "assembly_name": asm.name,
            "total_recommended": len(aggregate),
            "recommendations": list(aggregate.values()),
        }, ensure_ascii=False, indent=2))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_part_recommend_tools(registry: Any) -> None:
    """Register part recommendation tool."""
    registry.register(PartRecommendTool())
