"""Fastener model generation — FreeCAD script generators for standard hardware.

Generates FreeCAD Python scripts (as operation dicts or raw scripts) for:
- Socket head cap screws (DIN 912) — simplified and realistic (with hex socket)
- Hex nuts (DIN 934) — simplified and realistic (with hex profile)
- Flat washers (DIN 125)
- Spring washers (DIN 127) — split ring approximation
- Heat-set thread inserts — knurled brass cylinder
- Set screws (DIN 916)

Uses dimension data from ``knowledge.fastener_catalog`` — the authoritative source.
"""

from __future__ import annotations

from typing import Any

from ..knowledge.fastener_catalog import (
    BOLT_SPECS,
    NUT_SPECS,
    WASHER_SPECS,
    SPRING_WASHER_SPECS,
    THREAD_INSERT_SPECS,
    SET_SCREW_SPECS,
    get_bolt_spec,
    get_nut_spec,
    get_washer_spec,
    get_thread_insert_spec,
    get_set_screw_spec,
    recommend_bolt_length,
)
from ..models.base import ToolDefinition
from .base import Tool


# ============================================================================
# FreeCAD script generators
# ============================================================================


def generate_bolt_ops(
    size: str,
    length: float,
    *,
    style: str = "socket_head",
    quality: str = "simplified",
) -> list[dict]:
    """Generate FreeCAD operation dicts for a bolt.

    Args:
        size: Metric size, e.g. 'M3'
        length: Bolt length (shank + thread) in mm
        style: 'socket_head' (DIN 912), 'hex_head', 'countersunk'
        quality: 'simplified' (smooth cylinders) or 'realistic' (hex socket recess)

    Returns:
        List of FreeCAD operation dicts.
    """
    spec = get_bolt_spec(size)
    if spec is None:
        return []

    ops: list[dict] = []
    name = f"bolt_{size}_{int(length)}"

    if style == "socket_head":
        # Head
        ops.append({
            "type": "make_cylinder",
            "radius": spec.head_diameter / 2,
            "height": spec.head_height,
            "name": f"{name}_head",
        })
        # Shank
        ops.append({
            "type": "make_cylinder",
            "radius": spec.thread_diameter / 2,
            "height": length,
            "name": f"{name}_shank",
        })
        ops.append({
            "type": "move",
            "object": f"{name}_shank",
            "dx": 0, "dy": 0, "dz": spec.head_height,
        })
        # Fuse head + shank
        ops.append({
            "type": "boolean",
            "operation": "union",
            "object1": f"{name}_head",
            "object2": f"{name}_shank",
            "result_name": name,
        })

        if quality == "realistic":
            # Hex socket recess in head
            socket_r = spec.socket_width / 2
            ops.append({
                "type": "make_cylinder",
                "radius": socket_r,
                "height": spec.socket_depth,
                "name": f"{name}_socket",
            })
            ops.append({
                "type": "boolean",
                "operation": "cut",
                "object1": name,
                "object2": f"{name}_socket",
                "result_name": f"{name}_cut1",
            })
            ops.append({"type": "delete_object", "object": name})
            ops.append({"type": "delete_object", "object": f"{name}_socket"})
            name = f"{name}_cut1"

    elif style == "hex_head":
        # Hex head approximated as cylinder (outer_radius = corners)
        hex_r = spec.head_diameter / 2 * 1.15
        ops.append({
            "type": "make_cylinder",
            "radius": hex_r,
            "height": spec.head_height,
            "name": f"{name}_head",
        })
        ops.append({
            "type": "make_cylinder",
            "radius": spec.thread_diameter / 2,
            "height": length,
            "name": f"{name}_shank",
        })
        ops.append({
            "type": "move",
            "object": f"{name}_shank",
            "dx": 0, "dy": 0, "dz": spec.head_height,
        })
        ops.append({
            "type": "boolean",
            "operation": "union",
            "object1": f"{name}_head",
            "object2": f"{name}_shank",
            "result_name": name,
        })

    elif style == "countersunk":
        # Countersunk head — cone + cylinder
        ops.append({
            "type": "make_cone",
            "radius1": spec.thread_diameter / 2,
            "radius2": spec.head_diameter / 2,
            "height": spec.head_height,
            "name": f"{name}_head",
        })
        ops.append({
            "type": "make_cylinder",
            "radius": spec.thread_diameter / 2,
            "height": length,
            "name": f"{name}_shank",
        })
        ops.append({
            "type": "move",
            "object": f"{name}_shank",
            "dx": 0, "dy": 0, "dz": spec.head_height,
        })
        ops.append({
            "type": "boolean",
            "operation": "union",
            "object1": f"{name}_head",
            "object2": f"{name}_shank",
            "result_name": name,
        })

    return ops


def generate_nut_ops(
    size: str,
    *,
    quality: str = "simplified",
) -> list[dict]:
    """Generate FreeCAD operation dicts for a hex nut.

    Args:
        size: Metric size, e.g. 'M3'
        quality: 'simplified' (cylinder with hole) or 'realistic' (hex profile)

    Returns:
        List of FreeCAD operation dicts.
    """
    spec = get_nut_spec(size)
    if spec is None:
        return []

    name = f"nut_{size}"
    ops: list[dict] = []

    if quality == "simplified":
        # Cylinder approximation of hex
        ops.append({
            "type": "cylinder_with_hole",
            "outer_radius": spec.outer_radius,
            "inner_radius": spec.thread_diameter / 2,
            "height": spec.height,
            "name": name,
        })
    else:
        # Realistic: hex profile via raw script
        ops.append({
            "type": "raw_script",
            "script": _hex_nut_script(size, spec),
        })

    return ops


def generate_washer_ops(size: str) -> list[dict]:
    """Generate FreeCAD operation dicts for a flat washer."""
    spec = get_washer_spec(size)
    if spec is None:
        return []

    name = f"washer_{size}"
    return [{
        "type": "cylinder_with_hole",
        "outer_radius": spec.outer_diameter / 2,
        "inner_radius": spec.inner_diameter / 2,
        "height": spec.thickness,
        "name": name,
    }]


def generate_spring_washer_ops(size: str) -> list[dict]:
    """Generate FreeCAD operation dicts for a spring (lock) washer.

    Approximated as a split ring (cylinder with gap).
    """
    spec = SPRING_WASHER_SPECS.get(size)
    if spec is None:
        return []

    name = f"spring_washer_{size}"

    # Simplified: cylinder with hole + a gap cut
    script = f'''
import FreeCAD, Part, math

id_ = {spec.inner_diameter / 2}
od_ = {spec.outer_diameter / 2}
t = {spec.thickness}
gap_angle = 15  # degrees

# Full ring
ring = Part.makeCylinder(od_, t)
bore = Part.makeCylinder(id_, t + 2)
bore.translate(FreeCAD.Vector(0, 0, -1))
ring = ring.cut(bore)

# Cut a gap (wedge)
gap_w = od_ * 0.15
gap = Part.makeBox(gap_w, od_ * 2, t + 2)
gap.translate(FreeCAD.Vector(od_ * 0.85, -od_, -1))
ring = ring.cut(gap)

obj = doc.addObject("Part::Feature", "{name}")
obj.Shape = ring
doc.recompute()
'''
    return [{"type": "raw_script", "script": script}]


def generate_thread_insert_ops(size: str) -> list[dict]:
    """Generate FreeCAD operation dicts for a heat-set thread insert.

    Brass cylinder with knurled exterior.
    """
    spec = get_thread_insert_spec(size)
    if spec is None:
        return []

    name = f"thread_insert_{size}"

    script = f'''
import FreeCAD, Part, math

od = {spec.outer_diameter}
length = {spec.length}
thread_d = {spec.thread_diameter}
knurl_depth = 0.3  # mm

# Main body cylinder
body = Part.makeCylinder(od / 2, length)

# Internal thread hole
hole = Part.makeCylinder(thread_d / 2, length + 2)
hole.translate(FreeCAD.Vector(0, 0, -1))
body = body.cut(hole)

# Knurl grooves (simplified as longitudinal slots)
n_slots = max(12, int(od * math.pi / 2))
for i in range(n_slots):
    angle = 2 * math.pi * i / n_slots
    x = (od / 2 - knurl_depth / 2) * math.cos(angle)
    y = (od / 2 - knurl_depth / 2) * math.sin(angle)
    slot = Part.makeCylinder(knurl_depth / 2, length)
    slot.translate(FreeCAD.Vector(x, y, 0))
    body = body.cut(slot)

obj = doc.addObject("Part::Feature", "{name}")
obj.Shape = body
doc.recompute()
'''
    return [{"type": "raw_script", "script": script}]


def generate_set_screw_ops(size: str, length: float) -> list[dict]:
    """Generate FreeCAD operation dicts for a set screw (grub screw)."""
    spec = get_set_screw_spec(size)
    if spec is None:
        return []

    name = f"set_screw_{size}_{int(length)}"
    ops: list[dict] = []

    # Plain cylinder (thread)
    ops.append({
        "type": "make_cylinder",
        "radius": spec.thread_diameter / 2,
        "height": length,
        "name": name,
    })

    # Hex socket on one end
    ops.append({
        "type": "make_cylinder",
        "radius": spec.socket_width / 2,
        "height": spec.socket_depth,
        "name": f"{name}_socket",
    })
    ops.append({
        "type": "boolean",
        "operation": "cut",
        "object1": name,
        "object2": f"{name}_socket",
        "result_name": f"{name}_cut",
    })
    ops.append({"type": "delete_object", "object": name})
    ops.append({"type": "delete_object", "object": f"{name}_socket"})

    return ops


# ============================================================================
# Realistic hex nut script (for quality="realistic")
# ============================================================================


def _hex_nut_script(size: str, spec: Any) -> str:
    """Generate a FreeCAD script for a realistic hex nut profile."""
    w = spec.width_across_flats
    h = spec.height
    r_thread = spec.thread_diameter / 2
    # Hex corners: point-to-point distance = w / cos(30°)
    r_hex = w / 2

    return f'''
import FreeCAD, Part, math

w = {w}
h = {h}
r_thread = {r_thread}
r_hex = w / 2
r_corner = r_hex / math.cos(math.radians(30))

# Build hex profile via 6-sided polygon wire
points = []
for i in range(6):
    angle = math.radians(30 + i * 60)
    x = r_corner * math.cos(angle)
    y = r_corner * math.sin(angle)
    points.append(FreeCAD.Vector(x, y, 0))
points.append(points[0])  # close

wire = Part.makePolygon(points)
face = Part.Face(wire)

# Extrude to height
hex_prism = face.extrude(FreeCAD.Vector(0, 0, h))

# Thread hole
hole = Part.makeCylinder(r_thread, h + 2)
hole.translate(FreeCAD.Vector(0, 0, -1))
nut = hex_prism.cut(hole)

# Chamfer both faces
try:
    nut = nut.makeChamfer(0.3, [_e for _e in nut.Edges if _e.Length < r_thread * 4][:12])
except Exception as _e:
    logger.debug("nut chamfer failed (cosmetic: %s", _e)

obj = doc.addObject("Part::Feature", "nut_{size}")
obj.Shape = nut
doc.recompute()
'''


# ============================================================================
# Agent tool
# ============================================================================


class FastenerModelTool(Tool):
    """Tool for generating standard fastener 3D models."""

    name = "fastener_model"
    description = (
        "生成 ISO/DIN 标准紧固件 3D 模型（螺栓/螺母/垫圈/热嵌螺母/紧定螺钉）。"
        "输入紧固件类型和规格，输出 FreeCAD 建模脚本。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "fastener_type": {
                    "type": "string",
                    "description": (
                        "紧固件类型: bolt(螺栓), nut(螺母), washer(平垫圈), "
                        "spring_washer(弹簧垫圈), thread_insert(热嵌螺母), "
                        "set_screw(紧定螺钉), fastener_set(完整螺栓组=螺栓+螺母+垫圈)"
                    ),
                },
                "size": {
                    "type": "string",
                    "description": "螺纹规格: M2/M2.5/M3/M4/M5/M6/M8/M10/M12",
                },
                "length": {
                    "type": "number",
                    "description": "螺栓长度(mm)，仅 bolt/set_screw/fastener_set 需要",
                },
                "style": {
                    "type": "string",
                    "description": "螺栓头型式: socket_head(圆柱头)/hex_head(六角头)/countersunk(沉头)",
                },
                "quality": {
                    "type": "string",
                    "description": "建模质量: simplified(简化)/realistic(真实含六角槽)",
                },
                "grip_thickness": {
                    "type": "number",
                    "description": "夹持厚度(mm)，用于自动推荐螺栓长度。fastener_set 模式需要。",
                },
                "with_washer": {
                    "type": "boolean",
                    "description": "是否包含垫圈（fastener_set 模式），默认 true",
                },
                "with_thread_insert": {
                    "type": "boolean",
                    "description": "是否使用热嵌螺母替代普通螺母（3D打印推荐），默认 false",
                },
            },
        )

    def execute(self, *, fastener_type: str, size: str, **kwargs: Any) -> str:
        """Execute fastener model generation."""
        from ..knowledge.fastener_catalog import (
            get_bolt_spec,
            recommend_bolt_length,
            recommend_fastener_set,
        )

        # Validate size
        spec = get_bolt_spec(size)
        if spec is None and fastener_type != "fastener_set":
            return f"错误: 不支持的规格 '{size}'。可选: {', '.join(BOLT_SPECS.keys())}"

        ftype = fastener_type.lower()
        quality = kwargs.get("quality", "simplified")
        style = kwargs.get("style", "socket_head")
        length = kwargs.get("length", 0.0)
        grip = kwargs.get("grip_thickness", 0.0)
        with_washer = kwargs.get("with_washer", True)
        with_insert = kwargs.get("with_thread_insert", False)

        if ftype == "bolt":
            if length <= 0:
                length = recommend_bolt_length(grip if grip > 0 else 10.0)
            ops = generate_bolt_ops(size, length, style=style, quality=quality)
            return f"已生成 {size} 螺栓(长度{length}mm, {style}, {quality})，共 {len(ops)} 个操作"

        elif ftype == "nut":
            ops = generate_nut_ops(size, quality=quality)
            return f"已生成 {size} 螺母({quality})，共 {len(ops)} 个操作"

        elif ftype == "washer":
            ops = generate_washer_ops(size)
            return f"已生成 {size} 平垫圈，共 {len(ops)} 个操作"

        elif ftype == "spring_washer":
            ops = generate_spring_washer_ops(size)
            return f"已生成 {size} 弹簧垫圈，共 {len(ops)} 个操作"

        elif ftype == "thread_insert":
            ops = generate_thread_insert_ops(size)
            return f"已生成 {size} 热嵌螺母，共 {len(ops)} 个操作"

        elif ftype == "set_screw":
            if length <= 0:
                length = recommend_bolt_length(grip if grip > 0 else 5.0)
            ops = generate_set_screw_ops(size, length)
            return f"已生成 {size} 紧定螺钉(长度{length}mm)，共 {len(ops)} 个操作"

        elif ftype == "fastener_set":
            result = recommend_fastener_set(
                size,
                grip if grip > 0 else 10.0,
                with_washer=with_washer,
                use_thread_insert=with_insert,
            )
            if result is None:
                return f"错误: 不支持的规格 '{size}'"

            # Generate all ops
            all_ops: list[dict] = []
            bolt_len = result["bolt_length_mm"]
            all_ops.extend(generate_bolt_ops(size, bolt_len, quality=quality))
            if with_insert:
                all_ops.extend(generate_thread_insert_ops(size))
            else:
                all_ops.extend(generate_nut_ops(size, quality=quality))
            if with_washer:
                all_ops.extend(generate_washer_ops(size))

            lines = [
                f"=== {size} 紧固件组 ===",
                f"  螺栓: {size}×{bolt_len}mm",
                f"  {'热嵌螺母' if with_insert else '螺母'}: {size}",
                f"  垫圈: {f'{size} 平垫圈' if with_washer else '无'}",
                f"  推荐扭矩: {result['torque_nm']}N·m (PLA)",
                f"  间隙孔: Ø{result['clearance_hole_mm']}mm",
                f"  共 {len(all_ops)} 个 FreeCAD 操作，{result['parts_count']} 个零件",
            ]
            return "\n".join(lines)

        else:
            return (
                f"错误: 未知紧固件类型 '{fastener_type}'。"
                f"可选: bolt, nut, washer, spring_washer, thread_insert, set_screw, fastener_set"
            )


class FastenerQueryTool(Tool):
    """Tool for querying standard fastener dimensions."""

    name = "fastener_query"
    description = (
        "查询 ISO/DIN 标准紧固件尺寸数据（螺栓头径/螺母对边/垫圈外径/间隙孔/攻丝孔/"
        "推荐扭矩/推荐螺栓长度）。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "query_type": {
                    "type": "string",
                    "description": (
                        "查询类型: bolt_spec(螺栓规格), nut_spec(螺母规格), "
                        "washer_spec(垫圈规格), thread_insert_spec(热嵌螺母规格), "
                        "clearance_hole(间隙孔径), tap_hole(攻丝孔径), "
                        "torque(推荐扭矩), bolt_length(推荐螺栓长度), "
                        "fastener_set(完整推荐), available_sizes(可用规格列表)"
                    ),
                },
                "size": {
                    "type": "string",
                    "description": "螺纹规格: M3/M4/M5/M6 等",
                },
                "material": {
                    "type": "string",
                    "description": "材料(用于扭矩推荐): PLA/ABS/Steel/Aluminum",
                },
                "grip_mm": {
                    "type": "number",
                    "description": "夹持厚度(mm)，bolt_length 和 fastener_set 查询需要",
                },
            },
        )

    def execute(self, *, query_type: str, **kwargs: Any) -> str:
        """Execute fastener dimension query."""
        from ..knowledge.fastener_catalog import (
            get_bolt_spec,
            get_nut_spec,
            get_washer_spec,
            get_thread_insert_spec,
            get_clearance_hole,
            get_tap_hole,
            get_torque,
            recommend_bolt_length,
            recommend_fastener_set,
            list_available_sizes,
        )

        qt = query_type.lower()
        size = kwargs.get("size", "")

        if qt == "bolt_spec":
            spec = get_bolt_spec(size)
            if spec is None:
                return f"未知规格: {size}"
            return (
                f"{spec.size} 螺栓 (DIN 912):\n"
                f"  螺纹直径: {spec.thread_diameter}mm\n"
                f"  螺距: {spec.pitch}mm\n"
                f"  头部直径: {spec.head_diameter}mm\n"
                f"  头部高度: {spec.head_height}mm\n"
                f"  内六角对边: {spec.socket_width}mm\n"
                f"  内六角深度: {spec.socket_depth}mm"
            )

        elif qt == "nut_spec":
            spec = get_nut_spec(size)
            if spec is None:
                return f"未知规格: {size}"
            return (
                f"{spec.size} 螺母 (DIN 934):\n"
                f"  螺纹直径: {spec.thread_diameter}mm\n"
                f"  对边宽度: {spec.width_across_flats}mm\n"
                f"  高度: {spec.height}mm\n"
                f"  对角宽度: {spec.width_across_corners:.1f}mm"
            )

        elif qt == "washer_spec":
            spec = get_washer_spec(size)
            if spec is None:
                return f"未知规格: {size}"
            return (
                f"{spec.size} 平垫圈 (DIN 125):\n"
                f"  内径: {spec.inner_diameter}mm\n"
                f"  外径: {spec.outer_diameter}mm\n"
                f"  厚度: {spec.thickness}mm"
            )

        elif qt == "thread_insert_spec":
            spec = get_thread_insert_spec(size)
            if spec is None:
                return f"未知规格: {size}"
            return (
                f"{spec.size} 热嵌螺母:\n"
                f"  内螺纹: {spec.thread_diameter}mm\n"
                f"  外径(滚花): {spec.outer_diameter}mm\n"
                f"  长度: {spec.length}mm\n"
                f"  安装孔径: {spec.install_hole_diameter}mm\n"
                f"  最小壁厚: {spec.min_wall_thickness}mm"
            )

        elif qt == "clearance_hole":
            fit = kwargs.get("fit", "normal")
            d = get_clearance_hole(size, fit)
            return f"{size} 间隙孔({fit}配合): Ø{d}mm"

        elif qt == "tap_hole":
            d = get_tap_hole(size)
            return f"{size} 攻丝孔: Ø{d}mm"

        elif qt == "torque":
            material = kwargs.get("material", "PLA")
            t = get_torque(size, material)
            return f"{size} 推荐扭矩({material}): {t}N·m"

        elif qt == "bolt_length":
            grip = kwargs.get("grip_mm", 10)
            length = recommend_bolt_length(float(grip))
            return f"夹持{grip}mm → 推荐 {size} 螺栓长度: {length}mm"

        elif qt == "fastener_set":
            grip = kwargs.get("grip_mm", 10)
            result = recommend_fastener_set(size, float(grip))
            if result is None:
                return f"未知规格: {size}"
            lines = [f"{size} 紧固件组推荐 (夹持{grip}mm):"]
            for k, v in result.items():
                if isinstance(v, dict):
                    lines.append(f"  {k}:")
                    for sk, sv in v.items():
                        lines.append(f"    {sk}: {sv}")
                else:
                    lines.append(f"  {k}: {v}")
            return "\n".join(lines)

        elif qt == "available_sizes":
            sizes = list_available_sizes()
            return f"可用规格: {', '.join(sizes)}"

        else:
            return f"未知查询类型: {query_type}"


# ============================================================================
# Registration
# ============================================================================


def register_fastener_tools(registry: Any) -> None:
    """Register all fastener tools with the agent tool registry."""
    registry.register(FastenerModelTool())
    registry.register(FastenerQueryTool())
