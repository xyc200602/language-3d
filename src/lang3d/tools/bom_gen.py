"""Bill of Materials (BOM) generation for robotic assemblies.

Generates a complete BOM including:
  - Custom parts (3D printed parts with material, dimensions, print params)
  - Standard parts (screws, nuts, bearings)
  - Electronics (servos, controller, sensors, wiring)
  - Cost estimate

Tools:
  gen_bom - Generate BOM for an assembly
"""

from __future__ import annotations

import json
from typing import Any

from ..knowledge.actuators import get_actuator
from ..knowledge.mechanics import Assembly, Part, STANDARD_SCREWS
from ..knowledge.sensors import get_sensor
from ..knowledge.fastener_catalog import recommend_bolt_length
from ..models.base import ToolDefinition
from .assembly_solver import _resolve_assembly
from .base import Tool


# ---------------------------------------------------------------------------
# BOM generation
# ---------------------------------------------------------------------------

# PLA material cost per gram (CNY)
PLA_COST_PER_G = 0.15

# Default 3D print parameters
DEFAULT_PRINT_PARAMS = {
    "layer_height": 0.2,
    "infill": 20,
    "wall_thickness": 1.2,
    "material": "PLA",
}


def generate_bom(
    assembly: Assembly,
    actuator_ids: list[str] | None = None,
    sensor_ids: list[str] | None = None,
    controller: str = "esp32",
) -> dict[str, Any]:
    """Generate a Bill of Materials for an assembly.

    Args:
        assembly: The mechanical assembly.
        actuator_ids: Actuator IDs used in the assembly.
        sensor_ids: Sensor IDs used in the assembly.
        controller: Controller type (esp32/arduino).

    Returns a structured BOM dict with sections:
      custom_parts, standard_parts, electronics, cost_summary
    """
    bom: dict[str, Any] = {
        "assembly_name": assembly.name,
        "custom_parts": [],
        "standard_parts": [],
        "electronics": [],
        "cost_summary": {},
    }

    # --- Custom parts (3D printed) ---
    for part in assembly.parts:
        entry = _part_to_bom_entry(part)
        bom["custom_parts"].append(entry)

    # --- Standard parts (inferred from assembly notes) ---
    bom["standard_parts"] = _infer_standard_parts(assembly)

    # --- Electronics ---
    if actuator_ids:
        for aid in actuator_ids:
            a = get_actuator(aid)
            if a:
                bom["electronics"].append({
                    "type": "actuator",
                    "id": a.id,
                    "name": a.name,
                    "specs": f"{a.torque_kgcm} kg·cm, {a.voltage}V",
                    "price_cny": a.price_cny,
                })

    if sensor_ids:
        for sid in sensor_ids:
            s = get_sensor(sid)
            if s:
                bom["electronics"].append({
                    "type": "sensor",
                    "id": s.id,
                    "name": s.name,
                    "specs": f"{s.interface}, {s.resolution}",
                    "price_cny": s.price_cny,
                })

    # Controller
    ctrl_name = "ESP32 开发板" if controller == "esp32" else "Arduino Uno"
    ctrl_price = 35 if controller == "esp32" else 25
    bom["electronics"].append({
        "type": "controller",
        "id": controller,
        "name": ctrl_name,
        "specs": f"{controller.upper()} MCU",
        "price_cny": ctrl_price,
    })

    # Breadboard + wires
    bom["electronics"].append({
        "type": "accessory",
        "id": "breadboard",
        "name": "面包板 + 杜邦线套装",
        "specs": "400 孔面包板 + 20cm 杜邦线",
        "price_cny": 10,
    })

    # --- Cost summary ---
    custom_cost = sum(p.get("estimated_cost_cny", 0) for p in bom["custom_parts"])
    std_cost = sum(p.get("price_cny", 0) * p.get("quantity", 1) for p in bom["standard_parts"])
    elec_cost = sum(e.get("price_cny", 0) for e in bom["electronics"])

    bom["cost_summary"] = {
        "custom_parts_cost_cny": round(custom_cost, 1),
        "standard_parts_cost_cny": round(std_cost, 1),
        "electronics_cost_cny": round(elec_cost, 1),
        "total_cost_cny": round(custom_cost + std_cost + elec_cost, 1),
        "num_custom_parts": len(bom["custom_parts"]),
        "num_standard_parts": sum(p.get("quantity", 1) for p in bom["standard_parts"]),
        "num_electronics": len(bom["electronics"]),
    }

    return bom


def _part_to_bom_entry(part: Part) -> dict[str, Any]:
    """Convert a Part to a BOM entry with cost estimate."""
    # Estimate volume (mm³)
    dims = part.dimensions
    vol_mm3 = 1.0
    if "diameter" in dims and ("height" in dims or "thickness" in dims):
        # Cylinder (disc)
        r = dims["diameter"] / 2
        h = dims.get("height", dims.get("thickness", 5))
        vol_mm3 = 3.14159 * r * r * h
    elif "outer_diameter" in dims and ("height" in dims or "thickness" in dims):
        # Hollow cylinder (estimate wall thickness)
        r_out = dims["outer_diameter"] / 2
        wall = dims.get("wall_thickness", 3)
        r_in = r_out - wall
        h = dims.get("height", dims.get("thickness", 5))
        vol_mm3 = 3.14159 * (r_out * r_out - r_in * r_in) * h
    elif all(k in dims for k in ("length", "width", "height")):
        vol_mm3 = dims["length"] * dims["width"] * dims["height"]
    elif "length" in dims and "width" in dims:
        vol_mm3 = dims["length"] * dims["width"] * dims.get("thickness", 5)

    # Material-aware density (kg/m³ → g/mm³ = kg/m³ × 1e-6)
    density_g_mm3 = part.effective_density() * 1e-6
    weight_g = vol_mm3 * density_g_mm3
    cost = weight_g * PLA_COST_PER_G

    return {
        "name": part.name,
        "category": part.category,
        "description": part.description,
        "material": part.material,
        "dimensions": part.dimensions,
        "estimated_weight_g": round(weight_g, 1),
        "estimated_cost_cny": round(cost, 2),
        "print_params": DEFAULT_PRINT_PARAMS,
        "quantity": 1,
    }


def _infer_standard_parts(assembly: Assembly) -> list[dict[str, Any]]:
    """Infer standard parts needed from assembly joints and ConnectionMethod data."""
    std_parts: list[dict[str, Any]] = []

    revolute_joints = [j for j in assembly.joints if j.type == "revolute"]
    fixed_joints = [j for j in assembly.joints if j.type == "fixed"]

    # Per-joint fastener inference from ConnectionMethod
    for joint in assembly.joints:
        cm = getattr(joint, 'connection', None)
        if cm is None:
            # Fallback: default M3×10 per joint
            _add_default_fasteners(std_parts)
            continue

        if cm.type == "bolted":
            bolt_size = cm.bolt_size or "M3"
            bolt_count = cm.bolt_count or 4

            # Estimate grip from part dimensions if available
            grip = _estimate_joint_grip(assembly, joint)
            bolt_length = recommend_bolt_length(grip)
            _add_bolt_nut_washer(std_parts, bolt_size, bolt_length, bolt_count)

        elif cm.type == "set_screw":
            size = getattr(cm, 'set_screw_size', None) or cm.bolt_size or "M3"
            _add_set_screw(std_parts, size, 1)

        elif cm.type == "dowel_pin":
            _add_dowel_pins(std_parts)

    # Scan notes for specific screw mentions
    for part in assembly.parts:
        notes_lower = part.notes.lower()
        if "m6" in notes_lower:
            std_parts.append({
                "type": "screw",
                "name": "M6×16 螺丝",
                "spec": "M6×16 不锈钢内六角",
                "quantity": 4,
                "unit_price_cny": 0.3,
                "price_cny": 0.3,
            })

    # Bearings for revolute joints
    if revolute_joints:
        std_parts.append({
            "type": "bearing",
            "name": "MR105ZZ 轴承",
            "spec": "5×10×4mm 微型法兰轴承",
            "quantity": len(revolute_joints) * 2,
            "unit_price_cny": 1.5,
            "price_cny": 1.5,
        })

    # Consolidate duplicate entries
    return _consolidate_parts(std_parts)


def _estimate_joint_grip(assembly: Assembly, joint) -> float:
    """Estimate grip length for a joint from part dimensions."""
    parts_by_name = {p.name: p for p in assembly.parts}
    parent = parts_by_name.get(joint.parent)
    child = parts_by_name.get(joint.child)

    grip = 0.0
    for p in (parent, child):
        if p is None:
            continue
        for k in ("thickness", "height"):
            if k in p.dimensions:
                grip += p.dimensions[k]
                break
    return max(grip, 3.0) if grip > 0 else 10.0


def _add_default_fasteners(std_parts: list[dict[str, Any]]) -> None:
    """Add default M3×10 fastener set."""
    std_parts.append({
        "type": "screw",
        "name": "M3×10 螺丝",
        "spec": "M3×10 不锈钢十字盘头",
        "quantity": 4,
        "unit_price_cny": 0.1,
        "price_cny": 0.1,
    })
    std_parts.append({
        "type": "nut",
        "name": "M3 螺母",
        "spec": "M3 不锈钢六角螺母",
        "quantity": 4,
        "unit_price_cny": 0.05,
        "price_cny": 0.05,
    })


def _add_bolt_nut_washer(
    std_parts: list[dict[str, Any]],
    bolt_size: str,
    bolt_length: float,
    count: int,
) -> None:
    """Add bolt + nut + washer standard parts."""
    length_str = f"{bolt_length:.0f}" if bolt_length == int(bolt_length) else f"{bolt_length}"
    std_parts.append({
        "type": "screw",
        "name": f"{bolt_size}×{length_str} 螺丝",
        "spec": f"{bolt_size}×{length_str} 不锈钢内六角",
        "quantity": count,
        "unit_price_cny": 0.1,
        "price_cny": 0.1,
    })
    std_parts.append({
        "type": "nut",
        "name": f"{bolt_size} 螺母",
        "spec": f"{bolt_size} 不锈钢六角螺母",
        "quantity": count,
        "unit_price_cny": 0.05,
        "price_cny": 0.05,
    })
    std_parts.append({
        "type": "washer",
        "name": f"{bolt_size} 平垫圈",
        "spec": f"{bolt_size} 不锈钢平垫圈",
        "quantity": count,
        "unit_price_cny": 0.03,
        "price_cny": 0.03,
    })


def _add_set_screw(
    std_parts: list[dict[str, Any]],
    size: str,
    count: int,
) -> None:
    """Add set screw / grub screw."""
    std_parts.append({
        "type": "screw",
        "name": f"{size} 紧定螺钉",
        "spec": f"{size} 不锈钢内六角紧定螺钉",
        "quantity": count,
        "unit_price_cny": 0.08,
        "price_cny": 0.08,
    })


def _add_dowel_pins(std_parts: list[dict[str, Any]]) -> None:
    """Add dowel pins."""
    std_parts.append({
        "type": "pin",
        "name": "D5×20 定位销",
        "spec": "Ø5×20mm 碳钢定位销",
        "quantity": 2,
        "unit_price_cny": 0.5,
        "price_cny": 0.5,
    })


def _consolidate_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge duplicate standard part entries by name."""
    by_name: dict[str, dict[str, Any]] = {}
    for p in parts:
        name = p["name"]
        if name in by_name:
            by_name[name]["quantity"] += p["quantity"]
        else:
            by_name[name] = dict(p)
    return list(by_name.values())


def format_bom_markdown(bom: dict[str, Any]) -> str:
    """Format BOM as a Markdown document."""
    lines = [
        f"# BOM — {bom['assembly_name']}",
        "",
    ]

    # Custom parts
    lines.append("## 自定义零件（3D 打印）")
    lines.append("")
    lines.append(f"| # | 名称 | 材料 | 尺寸 | 重量(g) | 成本(元) | 数量 |")
    lines.append("|---|------|------|------|---------|---------|------|")
    for i, p in enumerate(bom["custom_parts"], 1):
        dims_str = "×".join(f"{v}" for v in p["dimensions"].values())
        lines.append(
            f"| {i} | {p['name']} | {p['material']} | {dims_str} | "
            f"{p['estimated_weight_g']} | {p['estimated_cost_cny']} | {p['quantity']} |"
        )
    lines.append("")

    # Standard parts
    if bom["standard_parts"]:
        lines.append("## 标准件")
        lines.append("")
        lines.append(f"| # | 名称 | 规格 | 数量 | 单价(元) |")
        lines.append("|---|------|------|------|---------|")
        for i, p in enumerate(bom["standard_parts"], 1):
            lines.append(
                f"| {i} | {p['name']} | {p['spec']} | {p['quantity']} | {p['unit_price_cny']} |"
            )
        lines.append("")

    # Electronics
    if bom["electronics"]:
        lines.append("## 电子件")
        lines.append("")
        lines.append(f"| # | 名称 | 规格 | 价格(元) |")
        lines.append("|---|------|------|---------|")
        for i, e in enumerate(bom["electronics"], 1):
            lines.append(
                f"| {i} | {e['name']} | {e['specs']} | {e['price_cny']} |"
            )
        lines.append("")

    # Cost summary
    cs = bom["cost_summary"]
    lines.append("## 成本汇总")
    lines.append("")
    lines.append(f"| 项目 | 费用(元) |")
    lines.append("|------|---------|")
    lines.append(f"| 自定义零件 (3D 打印) | {cs['custom_parts_cost_cny']} |")
    lines.append(f"| 标准件 | {cs['standard_parts_cost_cny']} |")
    lines.append(f"| 电子件 | {cs['electronics_cost_cny']} |")
    lines.append(f"| **总计** | **{cs['total_cost_cny']}** |")
    lines.append("")
    lines.append(f"- 自定义零件数: {cs['num_custom_parts']}")
    lines.append(f"- 标准件总数: {cs['num_standard_parts']}")
    lines.append(f"- 电子件数: {cs['num_electronics']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class GenBOMTool(Tool):
    name = "gen_bom"
    description = (
        "生成物料清单（BOM）：自定义零件 + 标准件 + 电子件 + 成本估算。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "assembly_name": {"type": "string", "description": "装配体名称"},
                "actuator_ids": {
                    "type": "array", "items": {"type": "string"},
                    "description": "执行器 ID 列表",
                },
                "sensor_ids": {
                    "type": "array", "items": {"type": "string"},
                    "description": "传感器 ID 列表",
                },
                "controller": {
                    "type": "string", "enum": ["esp32", "arduino"],
                    "description": "控制器类型",
                },
            }, "required": []},
        )

    def execute(self, *, assembly_name: str = "robotic_arm",
                actuator_ids: list[str] | None = None,
                sensor_ids: list[str] | None = None,
                controller: str = "esp32",
                **kwargs: Any) -> str:
        assembly = _resolve_assembly(assembly_name, "")
        if assembly is None:
            return f"错误：未找到装配体 '{assembly_name}'"

        bom = generate_bom(assembly, actuator_ids, sensor_ids, controller)
        md = format_bom_markdown(bom)

        lines = [
            f"[BOM Generated] {assembly.name}",
            "",
            md,
            "",
            "--- JSON ---",
            json.dumps(bom, ensure_ascii=False, indent=2),
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_bom_tools(registry: Any) -> None:
    """Register BOM generation tools."""
    registry.register(GenBOMTool())
