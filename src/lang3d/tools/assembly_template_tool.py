"""Agent tools for searching and instantiating assembly templates."""

from __future__ import annotations

import json
from typing import Any

from ..knowledge.assembly_templates import (
    AssemblyTemplate,
    list_assembly_templates,
    search_assembly_templates,
    template_to_assembly,
)
from ..models.base import ToolDefinition
from .base import Tool


# ---------------------------------------------------------------------------
# Tool 1: Search assembly templates
# ---------------------------------------------------------------------------


class AssemblyTemplateSearchTool(Tool):
    """Search the assembly template knowledge base."""

    name = "assembly_template_search"
    description = (
        "搜索装配体模板库：根据关键词、机器人类型、自由度范围查找匹配的模板。"
        "返回模板摘要列表，包含名称、DOF、零件数等信息。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词（中文或英文，如 '3自由度机械臂'、'4 wheel'）",
                    },
                    "robot_type": {
                        "type": "string",
                        "description": "机器人类型过滤: 'arm' | 'mobile_base' | 'scara'",
                    },
                    "min_dof": {
                        "type": "integer",
                        "description": "最小自由度（默认 0）",
                    },
                    "max_dof": {
                        "type": "integer",
                        "description": "最大自由度（默认 99）",
                    },
                },
                "required": [],
            },
        )

    def execute(
        self,
        *,
        query: str = "",
        robot_type: str = "",
        min_dof: int = 0,
        max_dof: int = 99,
        **kwargs: Any,
    ) -> str:
        results = search_assembly_templates(
            query=query,
            robot_type=robot_type,
            min_dof=min_dof,
            max_dof=max_dof,
        )

        if not results:
            # Fall back to listing all templates
            summaries = list_assembly_templates()
            if not summaries:
                return "[Assembly Template Search] 没有可用的模板"

            lines = [
                "[Assembly Template Search] 无精确匹配，显示所有模板：",
                "",
            ]
        else:
            summaries = []
            for tpl in results:
                req = sum(1 for p in tpl.parts if not p.optional)
                opt = sum(1 for p in tpl.parts if p.optional)
                summaries.append({
                    "id": tpl.id,
                    "name_en": tpl.name_en,
                    "name_cn": tpl.name_cn,
                    "dof": tpl.dof,
                    "robot_type": tpl.robot_type,
                    "parts_required": req,
                    "parts_optional": opt,
                    "joints": len(tpl.joints),
                })
            lines = [
                f"[Assembly Template Search] 找到 {len(results)} 个模板：",
                "",
            ]

        for s in summaries:
            lines.append(
                f"  {s['id']}: {s['name_cn']} ({s['name_en']}) — "
                f"DOF={s['dof']}, {s['parts_required']}+{s['parts_optional']}零件, "
                f"{s['joints']}关节, type={s['robot_type']}"
            )

        lines.append("")
        lines.append("--- JSON ---")
        lines.append(json.dumps(summaries, ensure_ascii=False, indent=2))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2: Instantiate a template as an Assembly
# ---------------------------------------------------------------------------


class AssemblyTemplateInstantiateTool(Tool):
    """Instantiate an assembly template as a runtime Assembly object."""

    name = "assembly_template_instantiate"
    description = (
        "实例化装配体模板：将选定的模板转换为完整的 Assembly 对象，"
        "可选传入零件尺寸覆盖参数。返回装配体定义 JSON。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "template_id": {
                        "type": "string",
                        "description": "模板 ID（如 '3dof_arm', '6dof_belt_arm'）",
                    },
                    "overrides": {
                        "type": "object",
                        "description": "零件尺寸覆盖 {part_name: {dim_name: value}}",
                    },
                },
                "required": ["template_id"],
            },
        )

    def execute(
        self,
        *,
        template_id: str = "",
        overrides: dict[str, dict[str, float]] | None = None,
        **kwargs: Any,
    ) -> str:
        from ..knowledge.assembly_templates import TEMPLATES

        if template_id not in TEMPLATES:
            available = ", ".join(sorted(TEMPLATES.keys()))
            return (
                f"[Assembly Template] 错误：未知模板 '{template_id}'\n"
                f"可用模板: {available}"
            )

        tpl = TEMPLATES[template_id]
        asm = template_to_assembly(tpl, overrides)

        # Build JSON representation
        asm_dict = {
            "name": asm.name,
            "description": asm.description,
            "parts": [
                {
                    "name": p.name,
                    "category": p.category,
                    "description": p.description,
                    "material": p.material,
                    "dimensions": p.dimensions,
                }
                for p in asm.parts
            ],
            "joints": [
                {
                    "type": j.type,
                    "parent": j.parent,
                    "child": j.child,
                    "range_deg": list(j.range_deg),
                    "parent_anchor": j.parent_anchor,
                    "child_anchor": j.child_anchor,
                    "axis": j.axis,
                }
                for j in asm.joints
            ],
            "default_angles": asm.default_angles,
        }

        lines = [
            f"[Assembly Template Instantiate] 模板: {tpl.name_cn}",
            f"零件数: {len(asm.parts)}, 关节数: {len(asm.joints)}",
            "",
            "--- Assembly JSON ---",
            json.dumps(asm_dict, ensure_ascii=False, indent=2),
        ]

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_assembly_template_tools(registry: Any) -> None:
    """Register assembly template search and instantiate tools."""
    registry.register(AssemblyTemplateSearchTool())
    registry.register(AssemblyTemplateInstantiateTool())
