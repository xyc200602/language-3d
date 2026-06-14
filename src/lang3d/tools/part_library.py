"""Part library tools — search, generate, import, and manage standard parts."""

from __future__ import annotations

import json
import logging
import os
import threading
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
    get_functional_parts,
    get_structural_parts,
    get_fastener_parts,
    get_template,
    resolve_parameters,
    search_parts,
    validate_functional_params,
)
from ..models.base import ToolDefinition
from .base import Tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON-persisted store for generated / imported parts
# ---------------------------------------------------------------------------

class PartsStore:
    """Persistent store for generated parts, backed by a JSON file."""

    def __init__(self, json_path: str | Path) -> None:
        self._path = Path(json_path)
        self._parts: list[GeneratedPart] = []
        self._load()

    def _load(self) -> None:
        """Load from JSON file. Tolerates missing or corrupted files."""
        if not self._path.exists():
            self._parts = []
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._parts = [GeneratedPart.from_dict(item) for item in data]
            else:
                self._parts = []
        except (json.JSONDecodeError, TypeError, KeyError):
            self._parts = []

    def _save(self) -> None:
        """Persist current state to JSON file."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = [p.to_dict() for p in self._parts]
            self._path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as e:
            logger.warning("Failed to save parts store to %s: %s", self._path, e)

    def add(self, part: GeneratedPart) -> None:
        """Add a generated part and persist."""
        self._parts.append(part)
        self._save()

    def remove(self, name: str) -> bool:
        """Remove a part by name. Returns True if found and removed."""
        before = len(self._parts)
        self._parts = [p for p in self._parts if p.name != name]
        if len(self._parts) < before:
            self._save()
            return True
        return False

    def get(self, name: str) -> GeneratedPart | None:
        """Get a part by name."""
        for p in self._parts:
            if p.name == name:
                return p
        return None

    def list_all(self) -> list[GeneratedPart]:
        """Return all stored parts."""
        return list(self._parts)

    def count(self) -> int:
        """Return number of stored parts."""
        return len(self._parts)


# Module-level store instance (lazy-initialized, thread-safe)
_parts_store: PartsStore | None = None
_parts_store_lock = threading.Lock()


def _get_parts_store() -> PartsStore:
    """Get or create the singleton PartsStore.

    Thread-safe via double-checked locking.  The PartsStore is constructed
    fully inside the lock before assignment, so no other thread can observe
    a partially-initialised instance.
    """
    global _parts_store
    if _parts_store is None:
        with _parts_store_lock:
            if _parts_store is None:
                ws = _workspace_dir()
                json_path = Path(ws) / "generated_parts.json"
                # Construct fully before assigning to the global, so that
                # no thread ever sees a partially-initialised PartsStore.
                store = PartsStore(json_path)
                _parts_store = store
    return _parts_store


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
                    "part_class": {
                        "type": "string",
                        "description": "零件分类：functional(功能件-不可缩放), structural(结构件-可缩放), fastener(紧固件-标准尺寸)",
                    },
                },
                "required": [],
            },
        )

    def execute(self, *, query: str = "", category: str | None = None,
                tags: list[str] | None = None, part_class: str | None = None,
                **kwargs: Any) -> str:
        results = search_parts(query=query, category=category, tags=tags, part_class=part_class)
        if not results:
            return "未找到匹配的零件。可用的类别：" + ", ".join(CATEGORY_TREE.keys())

        lines = [f"找到 {len(results)} 个零件："]
        for t in results:
            cls_icon = {"functional": "[F]", "structural": "[S]", "fastener": "[H]"}.get(t.part_class, "[?]")
            standard = ""
            if t.standard_sizes:
                first = t.standard_sizes[0]
                dims = "x".join(str(v) for v in first.values())
                standard = f" (标准尺寸: {dims})"
            scalable_tag = "可缩放" if t.scalable else "固定尺寸"
            model_info = f" ({t.model_number})" if t.model_number else ""
            lines.append(
                f"  {cls_icon} [{t.id}] {t.name_cn} ({t.name_en}){model_info} — "
                f"{t.category}/{t.subcategory} [{scalable_tag}]{standard}"
            )
            lines.append(f"    {t.description[:80]}")
        lines.append("")
        lines.append("图例: [F]=功能件(电机/舵机/轴承,尺寸固定) [S]=结构件(可缩放) [H]=紧固件(标准规格)")
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

        # For functional parts, warn if parameters deviate from real specs
        if template.part_class == "functional" and parameters:
            warnings = validate_functional_params(part_id, parameters)
            if warnings:
                # Auto-correct: fall back to standard sizes for functional parts
                if template.standard_sizes:
                    best = template.standard_sizes[0]
                    corrected = {**best}
                    corrected.update(parameters)  # user overrides
                    # Re-validate
                    new_warnings = validate_functional_params(part_id, corrected)
                    if new_warnings:
                        return (
                            f"警告：功能件 '{template.name_en}' ({template.model_number}) "
                            f"的尺寸不可自由缩放。\n"
                            + "\n".join(warnings)
                            + f"\n\n标准尺寸：{json.dumps(best, ensure_ascii=False)}"
                            + f"\n请使用 variant_index 选择标准型号，或使用标准尺寸参数。"
                        )

        # For functional parts without parameters, auto-select standard size
        if template.part_class == "functional" and not parameters and variant_index is None:
            if template.standard_sizes:
                variant_index = 0  # Use first standard size by default

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
        _get_parts_store().add(gen)

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
        _get_parts_store().add(gen)

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
        _get_parts_store().add(gen)

        tag_str = f"，标签：{', '.join(tags)}" if tags else ""
        return f"已保存到零件库：{name} → {dest}{tag_str}"


# ---------------------------------------------------------------------------
# Tool: part_analyze_print
# ---------------------------------------------------------------------------

class PartAnalyzePrintTool(Tool):
    name = "part_analyze_print"
    description = "分析STL/FCStd文件的3D打印可行性（悬垂、壁厚、打印方向）"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "stl_path": {
                        "type": "string",
                        "description": "STL或FCStd文件路径",
                    },
                    "orientation": {
                        "type": "string",
                        "description": "打印方向: auto, xy, xz, yz",
                    },
                },
                "required": ["stl_path"],
            },
        )

    def execute(
        self,
        *,
        stl_path: str,
        orientation: str = "auto",
        **kwargs: Any,
    ) -> str:
        src = Path(stl_path)
        if not src.exists():
            return f"错误：文件不存在 '{stl_path}'"

        ext = src.suffix.lower()
        if ext not in (".stl", ".fcstd"):
            return f"错误：不支持的文件格式 '{ext}'，仅支持 STL 和 FCStd"

        analysis = _run_print_analysis(str(src), orientation)
        return analysis


def _run_print_analysis(file_path: str, orientation: str = "auto") -> str:
    """Run 3D print analysis via FreeCAD subprocess."""
    script = f'''
import FreeCAD, Part, Mesh, math, json

path = r"{file_path}"
ext = path.lower().split(".")[-1]

if ext == "stl":
    mesh = Mesh.read(path)
elif ext == "fcstd":
    doc = FreeCAD.openDocument(path)
    shapes = []
    for obj in doc.Objects:
        if hasattr(obj, "Shape") and obj.Shape is not None:
            try:
                shapes.append(obj.Shape)
            except Exception:
                pass
    if shapes:
        import MeshPart
        compound = Part.makeCompound(shapes) if len(shapes) > 1 else shapes[0]
        mesh = MeshPart.meshFromShape(compound, LinearDeflection=0.5, AngularDeflection=0.523599)
    else:
        mesh = Mesh.Mesh()
else:
    print("PRINT_ANALYSIS_JSON:error: unsupported format")
    import sys
    sys.exit(1)

# Bounding box
bb = mesh.BoundBox
bbox = {{"x": bb.XLength, "y": bb.YLength, "z": bb.ZLength}}

# Volume and area via Part.Shape
try:
    shape = Part.Shape(mesh.topology)
    volume = shape.Volume
    area = shape.Area
except Exception:
    volume = 0
    area = 0

# Overhang analysis: check facet normals vs Z-axis
overhang_count = 0
total_facets = mesh.CountFacets
bottom_area = 0.0
z_min = bb.ZMin

for i in range(total_facets):
    facet = mesh.getFacet(i)
    normal = facet.Normal
    # Angle with Z-axis (0,0,1)
    cos_angle = normal.z / max(normal.Length, 1e-10)
    angle_deg = math.degrees(math.acos(max(-1, min(1, cos_angle))))
    # Overhang: normal pointing more than 45 degrees from vertical
    if angle_deg > 135:  # more than 45deg from vertical downward
        overhang_count += 1
    # Bottom face: Z near ZMin and normal pointing down
    avg_z = (facet.p1.z + facet.p2.z + facet.p3.z) / 3
    if avg_z <= z_min + 0.5 and normal.z < -0.5:
        tri_area = facet.Area
        bottom_area += tri_area

overhang_ratio = overhang_count / max(total_facets, 1)

# Minimum wall thickness (approximate as min bbox dimension)
min_wall = min(bb.XLength, bb.YLength, bb.ZLength)

# Recommended orientation: check XY, XZ, YZ bottom areas
orientations = {{
    "xy": bottom_area,
    "xz": 0,
    "yz": 0,
}}
# For XZ orientation, X-length is height, so bottom is YZ plane
for i in range(total_facets):
    facet = mesh.getFacet(i)
    normal = facet.Normal
    avg_y = (facet.p1.y + facet.p2.y + facet.p3.y) / 3
    if avg_y <= bb.YMin + 0.5 and normal.y < -0.5:
        orientations["xz"] += facet.Area
    avg_x = (facet.p1.x + facet.p2.x + facet.p3.x) / 3
    if avg_x <= bb.XMin + 0.5 and normal.x < -0.5:
        orientations["yz"] += facet.Area

if orientation == "auto":
    best_orient = max(orientations, key=orientations.get)
else:
    best_orient = orientation

# Material estimate (PLA, 20% infill, density 1.24 g/cm3)
material_grams = volume * 0.2 * 0.00124 if volume > 0 else 0

# Issues list
issues = []
if overhang_ratio > 0.3:
    issues.append("高悬垂比例 (>30%)，需要支撑结构")
if min_wall < 0.8:
    issues.append("最小壁厚过薄 (<0.8mm)，可能无法打印")
if bottom_area < 10:
    issues.append("底面接触面积过小，可能需要支撑或调整方向")

result = {{
    "bounding_box": bbox,
    "volume_mm3": round(volume, 2),
    "surface_area_mm2": round(area, 2),
    "overhang_ratio": round(overhang_ratio, 3),
    "overhang_facets": overhang_count,
    "total_facets": total_facets,
    "bottom_area_mm2": round(bottom_area, 2),
    "min_wall_thickness_mm": round(min_wall, 2),
    "recommended_orientation": best_orient,
    "orientation_areas": {{k: round(v, 2) for k, v in orientations.items()}},
    "material_estimate_grams": round(material_grams, 2),
    "issues": issues,
    "printable": len(issues) == 0,
}}
print("PRINT_ANALYSIS_JSON:" + json.dumps(result, ensure_ascii=False))
'''

    try:
        output = _run_freecad_script_local(script, timeout=120)
    except RuntimeError as e:
        return f"FreeCAD 执行错误：{e}"
    except Exception as e:
        return f"执行错误：{e}"

    # Parse output for JSON
    for line in output.splitlines():
        if line.strip().startswith("PRINT_ANALYSIS_JSON:"):
            json_str = line.strip()[len("PRINT_ANALYSIS_JSON:"):]
            return f"3D打印可行性分析：\n{json_str}"

    return f"分析完成（无法解析JSON结果）\n{output}"


# ---------------------------------------------------------------------------
# Tool: part_assemble
# ---------------------------------------------------------------------------

_ASSEMBLY_SCRIPT_TEMPLATE = """\
import FreeCAD, Part, Mesh, os, math

doc = FreeCAD.newDocument("{assembly_name}")

{part_operations}

doc.recompute()

# Save document
os.makedirs(r"{output_dir}", exist_ok=True)
doc.saveAs(r"{fcstd_path}")
print(f"Assembly saved: {fcstd_path}")

{stl_export}
"""

_PART_IMPORT_TEMPLATES = {
    ".stl": """\
# Import STL: {name}
mesh_{idx} = Mesh.read(r"{file}")
shape_{idx} = Part.Shape(mesh_{idx}.topology)
rot_{idx} = FreeCAD.Rotation(FreeCAD.Vector({rax}, {ray}, {raz}), {rangle})
placement_{idx} = FreeCAD.Placement(FreeCAD.Vector({tx}, {ty}, {tz}), rot_{idx})
obj_{idx} = doc.addObject("Part::Feature", "{name}")
shape_{idx}.Placement = placement_{idx}
obj_{idx}.Shape = shape_{idx}
""",
    ".fcstd": """\
# Import FCStd: {name}
_import_doc_{idx} = FreeCAD.openDocument(r"{file}")
_shapes_{idx} = []
for _o in _import_doc_{idx}.Objects:
    if hasattr(_o, 'Shape') and _o.Shape is not None:
        try:
            _shapes_{idx}.append(_o.Shape)
        except Exception:
            pass
if _shapes_{idx}:
    _compound_{idx} = Part.makeCompound(_shapes_{idx}) if len(_shapes_{idx}) > 1 else _shapes_{idx}[0]
    rot_{idx} = FreeCAD.Rotation(FreeCAD.Vector({rax}, {ray}, {raz}), {rangle})
    placement_{idx} = FreeCAD.Placement(FreeCAD.Vector({tx}, {ty}, {tz}), rot_{idx})
    obj_{idx} = doc.addObject("Part::Feature", "{name}")
    _compound_{idx}.Placement = placement_{idx}
    obj_{idx}.Shape = _compound_{idx}
FreeCAD.closeDocument(_import_doc_{idx}.Name)
""",
    ".step": """\
# Import STEP: {name}
_import_shape_{idx} = Part.Shape()
_import_shape_{idx}.read(r"{file}")
rot_{idx} = FreeCAD.Rotation(FreeCAD.Vector({rax}, {ray}, {raz}), {rangle})
placement_{idx} = FreeCAD.Placement(FreeCAD.Vector({tx}, {ty}, {tz}), rot_{idx})
obj_{idx} = doc.addObject("Part::Feature", "{name}")
_import_shape_{idx}.Placement = placement_{idx}
obj_{idx}.Shape = _import_shape_{idx}
""",
}


class PartAssembleTool(Tool):
    name = "part_assemble"
    description = "将多个已生成零件组装到一个 FreeCAD 文档中，按配合关系自动定位"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "assembly_name": {
                        "type": "string",
                        "description": "装配体名称（用于文档和文件命名）",
                    },
                    "parts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file": {"type": "string", "description": "零件文件路径 (.stl/.fcstd/.step)"},
                                "name": {"type": "string", "description": "零件名称"},
                                "position": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "description": "位置 [x, y, z]，默认 [0,0,0]",
                                },
                                "rotation": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "description": "旋转轴+角 [ax, ay, az, angle_deg]，默认 [0,0,1,0]",
                                },
                            },
                            "required": ["file", "name"],
                        },
                        "description": "零件列表，每项包含 file, name, position, rotation",
                    },
                    "assembly_definition": {
                        "type": "string",
                        "description": "装配体定义（内置名称如 'robotic_arm'，或 JSON 字符串）。提供时自动计算零件位置，忽略 parts 中的 position/rotation。",
                    },
                    "joint_angles": {
                        "type": "object",
                        "description": "关节角度映射 {零件名: 角度(度)}，仅在 assembly_definition 模式下使用",
                    },
                    "output_format": {
                        "type": "string",
                        "description": "输出格式: fcstd 或 stl（默认 fcstd）",
                    },
                },
                "required": ["assembly_name", "parts"],
            },
        )

    def execute(
        self,
        *,
        assembly_name: str,
        parts: list[dict[str, Any]],
        assembly_definition: str = "",
        joint_angles: dict[str, float] | None = None,
        output_format: str = "fcstd",
        **kwargs: Any,
    ) -> str:
        if not parts:
            return "错误：零件列表为空"

        # If assembly_definition provided, auto-compute positions via solver
        if assembly_definition:
            parts = self._apply_solver_positions(parts, assembly_definition, joint_angles)

        # Validate all part files exist
        missing = []
        for p in parts:
            fpath = Path(p["file"])
            if not fpath.exists():
                missing.append(p["file"])
        if missing:
            return f"错误：以下零件文件不存在：\n" + "\n".join(f"  - {f}" for f in missing)

        ws = _workspace_dir()
        fcstd_path = str(Path(ws) / f"{assembly_name}.FCStd")
        stl_path = str(Path(ws) / f"{assembly_name}.stl")

        # Build part operations
        part_ops: list[str] = []
        for idx, part in enumerate(parts):
            fpath = Path(part["file"])
            ext = fpath.suffix.lower()
            pos = part.get("position", [0, 0, 0])
            rot = part.get("rotation", [0, 0, 1, 0])
            name = part.get("name", f"Part{idx}")

            if ext not in _PART_IMPORT_TEMPLATES:
                return f"错误：不支持的文件格式 '{ext}'（零件: {name}）"

            template = _PART_IMPORT_TEMPLATES[ext]
            op = template.format(
                idx=idx,
                file=str(fpath),
                name=name,
                tx=pos[0] if len(pos) > 0 else 0,
                ty=pos[1] if len(pos) > 1 else 0,
                tz=pos[2] if len(pos) > 2 else 0,
                rax=rot[0] if len(rot) > 0 else 0,
                ray=rot[1] if len(rot) > 1 else 0,
                raz=rot[2] if len(rot) > 2 else 1,
                rangle=rot[3] if len(rot) > 3 else 0,
            )
            part_ops.append(op)

        stl_export = ""
        if output_format == "stl":
            stl_export = f"""\
_export_list = [o for o in doc.Objects if hasattr(o, 'Shape')]
if _export_list:
    Mesh.export(_export_list, r'{stl_path}')
    print(f"STL: {stl_path} ({os.path.getsize(r'{stl_path}'):,} bytes)")
"""

        script = _ASSEMBLY_SCRIPT_TEMPLATE.format(
            assembly_name=assembly_name,
            part_operations="\n".join(part_ops),
            output_dir=ws,
            fcstd_path=fcstd_path,
            stl_export=stl_export,
        )

        try:
            output = _run_freecad_script_local(script, timeout=120)
        except RuntimeError as e:
            return f"FreeCAD 执行错误：{e}"
        except Exception as e:
            return f"执行错误：{e}"

        result_lines = [
            f"装配体生成成功：{assembly_name}",
            f"零件数量：{len(parts)}",
            f"FCStd: {fcstd_path}",
        ]
        if output_format == "stl" and Path(stl_path).exists():
            size_kb = Path(stl_path).stat().st_size // 1024
            result_lines.append(f"STL: {stl_path} ({size_kb} KB)")

        if output:
            for line in output.splitlines():
                if line.strip():
                    result_lines.append(f"  {line}")

        return "\n".join(result_lines)

    def _apply_solver_positions(
        self,
        parts: list[dict[str, Any]],
        assembly_definition: str,
        joint_angles: dict[str, float] | None,
    ) -> list[dict[str, Any]]:
        """Use assembly solver to compute positions for parts."""
        try:
            from .assembly_solver import _resolve_assembly
            assembly = _resolve_assembly(assembly_definition, "")
            if assembly is None:
                return parts

            from .assembly_solver import AssemblySolver
            solver = AssemblySolver(assembly)
            placements = solver.solve(joint_angles=joint_angles)

            # Map part names to their solver-computed placements
            for part in parts:
                name = part.get("name", "")
                if name in placements:
                    p = placements[name]
                    part["position"] = p["position"]
                    part["rotation"] = p["rotation"]
        except Exception as e:
            logger.warning("Solver position overlay failed, using manual positions: %s", e)
        return parts


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
        PartAnalyzePrintTool(),
        PartAssembleTool(),
    ]
    for tool in tools:
        registry.register(tool)
