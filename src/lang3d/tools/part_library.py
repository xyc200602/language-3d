"""Part library tools — search, generate, import, and manage standard parts."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..knowledge.parts_catalog import (
    PART_CATALOG,
    CATEGORY_TREE,
    GeneratedPart,
    PartTemplate,
    format_fc_script,
    get_all_templates,
    get_template,
    resolve_parameters,
    search_parts,
)
from ..models.base import ToolDefinition
from .base import Tool


# ---------------------------------------------------------------------------
# In-memory store for generated / imported parts
# ---------------------------------------------------------------------------

_generated_parts: list[GeneratedPart] = []


def _workspace_dir() -> str:
    """Return a workspace directory for saving generated parts."""
    # Try agent workspace if available
    try:
        from ..config import load_config
        cfg = load_config()
        ws = cfg.agent.workspace
        if ws and Path(ws).exists():
            parts_dir = Path(ws) / "parts_library"
            parts_dir.mkdir(parents=True, exist_ok=True)
            return str(parts_dir)
    except Exception:
        pass
    # Fallback: data/parts_library
    parts_dir = Path.cwd() / "data" / "parts_library"
    parts_dir.mkdir(parents=True, exist_ok=True)
    return str(parts_dir)


def _run_freecad_script_local(script: str, timeout: int = 60) -> str:
    """Execute a FreeCAD script via the freecad module's subprocess bridge."""
    from .freecad import _run_freecad_script
    return _run_freecad_script(script, timeout=timeout)


# ---------------------------------------------------------------------------
# Tool: part_search
# ---------------------------------------------------------------------------

class PartSearchTool(Tool):
    name = "part_search"
    description = "搜索标准零件库（支持中文/英文关键词、类别、标签筛选）"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词（中英文均可，如 '螺钉'、'bearing'、'舵机'）",
                    },
                    "category": {
                        "type": "string",
                        "description": "零件类别：fastener, bearing, actuator, shaft, gear, structural",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "标签过滤",
                    },
                },
                "required": [],
            },
        )

    def execute(self, *, query: str = "", category: str | None = None,
                tags: list[str] | None = None, **kwargs: Any) -> str:
        results = search_parts(query=query, category=category, tags=tags)
        if not results:
            return "未找到匹配的零件。可用的类别：" + ", ".join(CATEGORY_TREE.keys())

        lines = [f"找到 {len(results)} 个零件："]
        for t in results:
            standard = ""
            if t.standard_sizes:
                first = t.standard_sizes[0]
                dims = "x".join(str(v) for v in first.values())
                standard = f" (标准尺寸: {dims})"
            lines.append(
                f"  [{t.id}] {t.name_cn} ({t.name_en}) — {t.category}/{t.subcategory}{standard}"
            )
            lines.append(f"    {t.description[:80]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: part_get
# ---------------------------------------------------------------------------

class PartGetTool(Tool):
    name = "part_get"
    description = "获取零件模板详情（参数定义、标准尺寸、描述）"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "part_id": {
                        "type": "string",
                        "description": "零件模板ID（如 socket_head_cap_screw, bearing_608）",
                    },
                },
                "required": ["part_id"],
            },
        )

    def execute(self, *, part_id: str, **kwargs: Any) -> str:
        template = get_template(part_id)
        if template is None:
            available = ", ".join(sorted(PART_CATALOG.keys()))
            return f"未找到零件 '{part_id}'。可用零件：{available}"

        lines = [
            f"零件: {template.name_cn} ({template.name_en})",
            f"ID: {template.id}",
            f"类别: {template.category}/{template.subcategory}",
            f"描述: {template.description}",
            f"默认材料: {template.material_default}",
            f"标签: {', '.join(template.tags)}",
            "",
            "参数：",
        ]
        for p in template.parameters:
            fixed_mark = " [固定]" if p.fixed else ""
            lines.append(
                f"  {p.display_name_cn} ({p.name}): 默认={p.default}{p.unit}, "
                f"范围=[{p.min_value}, {p.max_value}], 步进={p.step}{fixed_mark}"
            )

        if template.standard_sizes:
            lines.append("")
            lines.append("标准尺寸：")
            for i, size in enumerate(template.standard_sizes):
                dims = ", ".join(f"{k}={v}" for k, v in size.items())
                lines.append(f"  [{i}] {dims}")

        if template.notes:
            lines.append(f"\n备注: {template.notes}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: part_generate
# ---------------------------------------------------------------------------

class PartGenerateTool(Tool):
    name = "part_generate"
    description = "从零件模板生成参数化3D模型文件（.FCStd + .STL）"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "part_id": {
                        "type": "string",
                        "description": "零件模板ID",
                    },
                    "parameters": {
                        "type": "object",
                        "description": "参数覆盖（如 {\"thread_diameter\": 5, \"length\": 20}）。未指定参数使用默认值。",
                    },
                    "variant_index": {
                        "type": "integer",
                        "description": "使用标准尺寸表中的第N个变体（覆盖parameters）",
                    },
                },
                "required": ["part_id"],
            },
        )

    def execute(
        self,
        *,
        part_id: str,
        parameters: dict[str, Any] | None = None,
        variant_index: int | None = None,
        **kwargs: Any,
    ) -> str:
        template = get_template(part_id)
        if template is None:
            return f"错误：未找到零件模板 '{part_id}'"

        # Resolve parameters: variant_index overrides individual parameters
        if variant_index is not None:
            if variant_index < 0 or variant_index >= len(template.standard_sizes):
                return (
                    f"错误：variant_index {variant_index} 超出范围 "
                    f"(0-{len(template.standard_sizes) - 1})"
                )
            parameters = template.standard_sizes[variant_index]

        try:
            resolved = resolve_parameters(template, parameters)
        except ValueError as e:
            return f"参数错误：{e}"

        # Generate FreeCAD script
        try:
            model_script = format_fc_script(template, resolved)
        except (KeyError, ValueError) as e:
            return f"脚本模板错误：{e}"

        # Build output paths
        ws = _workspace_dir()
        param_desc = "_".join(f"{k}{int(v) if v == int(v) else v}" for k, v in resolved.items())
        safe_name = f"{part_id}_{param_desc}"
        fcstd_path = str(Path(ws) / f"{safe_name}.FCStd")
        stl_path = str(Path(ws) / f"{safe_name}.stl")

        # Append save + export operations
        full_script = model_script + f"""
# Save document
import os
os.makedirs(r'{Path(fcstd_path).parent}', exist_ok=True)
doc.saveAs(r'{fcstd_path}')
print(f"Saved: {fcstd_path}")

# Export STL
import Mesh
_export_list = [o for o in doc.Objects if hasattr(o, 'Shape')]
if _export_list:
    Mesh.export(_export_list, r'{stl_path}')
    print(f"STL: {stl_path} ({os.path.getsize(r'{stl_path}'):,} bytes)")
"""

        # Execute via FreeCAD
        try:
            output = _run_freecad_script_local(full_script, timeout=120)
        except RuntimeError as e:
            return f"FreeCAD 执行错误：{e}"
        except Exception as e:
            return f"执行错误：{e}"

        # Record generated part
        gen = GeneratedPart(
            template_id=part_id,
            name=safe_name,
            parameters=resolved,
            fcstd_path=fcstd_path,
            stl_path=stl_path,
            created_at=datetime.now().isoformat(),
        )
        _generated_parts.append(gen)

        result_lines = [
            f"零件生成成功：{template.name_cn} ({template.name_en})",
            f"FCStd: {fcstd_path}",
        ]
        if Path(stl_path).exists():
            size_kb = Path(stl_path).stat().st_size // 1024
            result_lines.append(f"STL: {stl_path} ({size_kb} KB)")
        result_lines.append(f"参数：{json.dumps(resolved, ensure_ascii=False)}")
        if output:
            # Include relevant output lines
            for line in output.splitlines():
                if line.strip():
                    result_lines.append(f"  {line}")

        return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# Tool: part_list
# ---------------------------------------------------------------------------

class PartListTool(Tool):
    name = "part_list"
    description = "列出零件库中的模板（可按类别筛选）"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "筛选类别：fastener, bearing, actuator, shaft, gear, structural",
                    },
                    "subcategory": {
                        "type": "string",
                        "description": "筛选子类别：screw, nut, washer, bolt, ball_bearing, servo, stepper, linear, coupling, spur, bracket, plate",
                    },
                },
                "required": [],
            },
        )

    def execute(
        self,
        *,
        category: str | None = None,
        subcategory: str | None = None,
        **kwargs: Any,
    ) -> str:
        templates = get_all_templates()
        if category:
            templates = [t for t in templates if t.category.lower() == category.lower()]
        if subcategory:
            templates = [t for t in templates if t.subcategory.lower() == subcategory.lower()]

        if not templates:
            lines = ["零件库为空或没有匹配的零件。"]
        else:
            # Group by category
            by_cat: dict[str, list[PartTemplate]] = {}
            for t in templates:
                by_cat.setdefault(t.category, []).append(t)

            lines = [f"共 {len(templates)} 个零件模板："]
            for cat, cat_templates in sorted(by_cat.items()):
                lines.append(f"\n【{cat}】")
                for t in cat_templates:
                    lines.append(
                        f"  [{t.id}] {t.name_cn} ({t.name_en}) — {t.subcategory}"
                    )
                    if t.standard_sizes:
                        lines.append(f"    {len(t.standard_sizes)} 个标准尺寸")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: part_import
# ---------------------------------------------------------------------------

class PartImportTool(Tool):
    name = "part_import"
    description = "导入已有模型文件到零件库（记录元数据以便复用）"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要导入的文件路径（.FCStd, .STL, .STEP）",
                    },
                    "name": {
                        "type": "string",
                        "description": "零件名称",
                    },
                    "category": {
                        "type": "string",
                        "description": "类别（可选）",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "标签列表（可选）",
                    },
                },
                "required": ["path", "name"],
            },
        )

    def execute(
        self,
        *,
        path: str,
        name: str,
        category: str = "custom",
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        src = Path(path)
        if not src.exists():
            return f"错误：文件不存在 '{path}'"

        if src.suffix.lower() not in (".fcstd", ".stl", ".step", ".stp", ".obj"):
            return f"错误：不支持的文件格式 '{src.suffix}'"

        # Copy to parts library
        ws = _workspace_dir()
        dest = Path(ws) / f"{name}{src.suffix}"
        if dest != src:
            import shutil
            shutil.copy2(str(src), str(dest))

        gen = GeneratedPart(
            template_id="custom",
            name=name,
            parameters={},
            fcstd_path=str(dest) if dest.suffix.lower() == ".fcstd" else "",
            stl_path=str(dest) if dest.suffix.lower() == ".stl" else "",
            created_at=datetime.now().isoformat(),
        )
        _generated_parts.append(gen)

        tag_str = f"，标签：{', '.join(tags)}" if tags else ""
        return f"已导入：{name} ({dest.name})，类别：{category}{tag_str}"


# ---------------------------------------------------------------------------
# Tool: part_save
# ---------------------------------------------------------------------------

class PartSaveTool(Tool):
    name = "part_save"
    description = "保存当前模型文件到零件库（从工作目录复制）"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "零件名称（保存到 parts_library 目录）",
                    },
                    "fcstd_path": {
                        "type": "string",
                        "description": "要保存的 .FCStd 文件路径",
                    },
                    "category": {
                        "type": "string",
                        "description": "类别（可选，默认 'custom'）",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "标签列表（可选）",
                    },
                },
                "required": ["name", "fcstd_path"],
            },
        )

    def execute(
        self,
        *,
        name: str,
        fcstd_path: str,
        category: str = "custom",
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        src = Path(fcstd_path)
        if not src.exists():
            return f"错误：文件不存在 '{fcstd_path}'"

        ws = _workspace_dir()
        dest = Path(ws) / f"{name}{src.suffix}"
        if dest != src:
            import shutil
            shutil.copy2(str(src), str(dest))

        gen = GeneratedPart(
            template_id="custom",
            name=name,
            parameters={},
            fcstd_path=str(dest),
            stl_path="",
            created_at=datetime.now().isoformat(),
        )
        _generated_parts.append(gen)

        tag_str = f"，标签：{', '.join(tags)}" if tags else ""
        return f"已保存到零件库：{name} → {dest}{tag_str}"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_part_library_tools(registry: Any) -> None:
    """Register all part library tools."""
    tools = [
        PartSearchTool(),
        PartGetTool(),
        PartGenerateTool(),
        PartListTool(),
        PartImportTool(),
        PartSaveTool(),
    ]
    for tool in tools:
        registry.register(tool)
