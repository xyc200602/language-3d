"""3D print optimization for robotic assemblies.

Provides:
  - Print orientation optimization (minimize supports, maximize strength)
  - Tolerance compensation (shrinkage + clearance adjustments)
  - Print parameter recommendations (layer height, infill, walls)
  - Batch plate packing (arrange multiple parts on build plate)

Tools:
  print_optimize - Optimize 3D print settings for assembly parts
"""

from __future__ import annotations

import json
import math
from typing import Any

from ..knowledge.mechanics import Assembly, Part, PRINT_TOLERANCES
from ..models.base import ToolDefinition
from .assembly_solver import _resolve_assembly
from .base import Tool


# ---------------------------------------------------------------------------
# Print parameters
# ---------------------------------------------------------------------------

# Common 3D printer build volumes (mm)
BUILD_VOLUMES: dict[str, dict[str, float]] = {
    "ender3": {"x": 220, "y": 220, "z": 250},
    "prusa_mk3": {"x": 250, "y": 210, "z": 210},
    "bambu_x1c": {"x": 256, "y": 256, "z": 256},
    "custom": {"x": 200, "y": 200, "z": 200},
}

# Material shrinkage rates (%)
SHRINKAGE: dict[str, float] = {
    "PLA": 0.3,
    "ABS": 0.8,
    "PETG": 0.4,
    "TPU": 0.5,
    "NYLON": 1.2,
}

# Print quality presets
QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    "draft": {
        "layer_height": 0.3,
        "infill": 10,
        "wall_loops": 2,
        "top_bottom_layers": 3,
        "speed": 80,
    },
    "standard": {
        "layer_height": 0.2,
        "infill": 20,
        "wall_loops": 3,
        "top_bottom_layers": 4,
        "speed": 60,
    },
    "high": {
        "layer_height": 0.12,
        "infill": 30,
        "wall_loops": 4,
        "top_bottom_layers": 5,
        "speed": 40,
    },
}


# ---------------------------------------------------------------------------
# Orientation optimization
# ---------------------------------------------------------------------------

def optimize_orientation(part: Part) -> dict[str, Any]:
    """Recommend print orientation for a part.

    Strategy:
      - Minimize overhangs (keep largest flat face on build plate)
      - Align functional features (shafts, holes) vertically for strength
      - Avoid supports on mating surfaces

    Returns dict with orientation angles and reasoning.
    """
    dims = part.dimensions
    notes = part.notes.lower()

    # Determine part shape
    is_cylindrical = "diameter" in dims or "outer_diameter" in dims
    is_box = all(k in dims for k in ("length", "width", "height"))

    recommendation: dict[str, Any] = {
        "part_name": part.name,
        "rotation_x": 0,
        "rotation_y": 0,
        "rotation_z": 0,
        "reasoning": [],
        "support_needed": False,
        "support_volume_cm3": 0,
    }

    if is_cylindrical:
        # Print cylindrical parts standing up (axis vertical)
        # This gives best layer adhesion for axial loads
        recommendation["reasoning"].append("圆柱体竖直放置：轴竖直方向，层粘合最佳")
        # Check if hollow (joint housing) — standing up needs support for overhang
        if "wall_thickness" in dims:
            recommendation["support_needed"] = True
            recommendation["support_volume_cm3"] = 0.5
            recommendation["reasoning"].append("内部空腔需要支撑材料")
    elif is_box:
        # Print box parts on largest face
        l, w, h = dims.get("length", 0), dims.get("width", 0), dims.get("height", 0)
        # Lay on the face with the largest area
        if h <= l and h <= w:
            recommendation["reasoning"].append(f"平放：最小高度 {h}mm 在 Z 方向，减少层数")
        else:
            recommendation["rotation_x"] = 90
            recommendation["reasoning"].append(f"侧放：将 {l}mm 长边水平放置")

    # Check for features that need special orientation
    if "孔" in notes or "hole" in notes:
        recommendation["reasoning"].append("孔特征：确保孔轴线竖直（避免悬垂桥接）")
    if "轴承" in notes or "bearing" in notes:
        recommendation["reasoning"].append("轴承座：轴承配合面应竖直打印，确保圆度")
    if "走线" in notes or "中空" in notes:
        recommendation["support_needed"] = True
        recommendation["support_volume_cm3"] = 1.0
        recommendation["reasoning"].append("内部走线通道需要支撑")

    return recommendation


# ---------------------------------------------------------------------------
# Tolerance compensation
# ---------------------------------------------------------------------------

def apply_tolerance_compensation(
    dimensions: dict[str, float],
    fit_type: str = "sliding_fit",
    material: str = "PLA",
) -> dict[str, Any]:
    """Apply tolerance compensation to part dimensions.

    Adjusts dimensions for:
      - Material shrinkage (scale up before printing)
      - Fit clearance (subtract/add for mating surfaces)

    Args:
        dimensions: Original dimensions {name: value_mm}.
        fit_type: "tight_fit", "sliding_fit", "loose_fit", "bearing_fit".
        material: Print material (PLA, ABS, PETG, etc).

    Returns dict with original, compensated, and adjustments.
    """
    shrinkage_pct = SHRINKAGE.get(material, 0.3)
    shrinkage_factor = 1.0 + shrinkage_pct / 100.0

    clearance = PRINT_TOLERANCES.get(fit_type, 0.3)

    compensated = {}
    adjustments = {}

    for name, value in dimensions.items():
        # Apply shrinkage compensation (scale up)
        shrinkage_comp = value * shrinkage_factor

        # For diameter/radius/shaft dimensions: subtract clearance (make smaller)
        # For hole/bore dimensions: add clearance (make larger)
        is_external = any(kw in name.lower() for kw in ["diameter", "width", "length", "thickness", "outer", "shaft"])
        is_internal = any(kw in name.lower() for kw in ["tap_hole", "clearance", "inner", "bore"])

        if is_external and not is_internal:
            # External dimension: subtract clearance for fit
            final = shrinkage_comp - clearance
            adjustments[name] = {
                "original": value,
                "shrinkage_comp": round(shrinkage_comp, 3),
                "clearance": round(-clearance, 3),
                "final": round(final, 3),
            }
            compensated[name] = round(final, 3)
        elif is_internal:
            # Internal dimension: add clearance for fit
            final = shrinkage_comp + clearance
            adjustments[name] = {
                "original": value,
                "shrinkage_comp": round(shrinkage_comp, 3),
                "clearance": round(clearance, 3),
                "final": round(final, 3),
            }
            compensated[name] = round(final, 3)
        else:
            # Neutral dimension: just apply shrinkage
            compensated[name] = round(shrinkage_comp, 3)
            adjustments[name] = {
                "original": value,
                "shrinkage_comp": round(shrinkage_comp, 3),
                "clearance": 0,
                "final": round(shrinkage_comp, 3),
            }

    return {
        "material": material,
        "shrinkage_pct": shrinkage_pct,
        "fit_type": fit_type,
        "clearance_mm": clearance,
        "original": dimensions,
        "compensated": compensated,
        "adjustments": adjustments,
    }


# ---------------------------------------------------------------------------
# Print parameter recommendation
# ---------------------------------------------------------------------------

def recommend_print_params(
    part: Part,
    quality: str = "standard",
) -> dict[str, Any]:
    """Recommend print parameters for a part.

    Args:
        part: The part to print.
        quality: "draft", "standard", or "high".

    Returns print parameter dict.
    """
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["standard"])
    notes = part.notes.lower()

    params = dict(preset)
    params["material"] = part.material

    # Adjust for functional requirements
    if "轴承" in notes or "bearing" in notes:
        params["infill"] = max(params["infill"], 40)
        params["wall_loops"] = max(params["wall_loops"], 4)
        params["note"] = "轴承座零件：增加填充和壁厚以提高强度和圆度"
    elif "关节" in notes or "joint" in notes:
        params["infill"] = max(params["infill"], 30)
        params["note"] = "关节零件：中等填充，确保运动精度"
    elif "轻量" in notes or "light" in notes or "中空" in notes:
        params["infill"] = min(params["infill"], 15)
        params["note"] = "轻量化零件：低填充率，壁厚提供主要强度"

    # Estimate print time (very rough)
    dims = part.dimensions
    vol = 1.0
    if all(k in dims for k in ("length", "width", "height")):
        vol = dims["length"] * dims["width"] * dims["height"]
    elif "diameter" in dims and ("height" in dims or "thickness" in dims):
        r = dims["diameter"] / 2
        h = dims.get("height", dims.get("thickness", 5))
        vol = math.pi * r * r * h
    elif "outer_diameter" in dims and ("height" in dims or "thickness" in dims):
        r = dims["outer_diameter"] / 2
        wall = dims.get("wall_thickness", 3)
        h = dims.get("height", dims.get("thickness", 5))
        vol = math.pi * (r * r - (r - wall) ** 2) * h

    # Rough time estimate: vol_mm3 * infill% / speed_mm3_per_s
    effective_vol = vol * (params["infill"] / 100.0) * 0.5 + vol * 0.5  # walls + infill
    mm3_per_s = params["speed"] * 0.4 * params["layer_height"]  # nozzle 0.4mm
    if mm3_per_s > 0:
        params["estimated_print_time_min"] = round(effective_vol / mm3_per_s / 60, 0)
    else:
        params["estimated_print_time_min"] = 0

    return params


# ---------------------------------------------------------------------------
# Batch plate packing
# ---------------------------------------------------------------------------

def pack_parts_on_plate(
    parts: list[Part],
    printer: str = "ender3",
) -> dict[str, Any]:
    """Arrange multiple parts on a build plate.

    Simple greedy packing algorithm: sort by footprint area (largest first),
    place in a grid pattern on the build plate.

    Args:
        parts: List of parts to pack.
        printer: Printer model key.

    Returns packing result with positions and plate usage.
    """
    plate = BUILD_VOLUMES.get(printer, BUILD_VOLUMES["ender3"])

    # Calculate footprint for each part
    part_footprints: list[dict[str, Any]] = []
    for part in parts:
        dims = part.dimensions
        if all(k in dims for k in ("length", "width")):
            fw, fd = dims["length"], dims["width"]
        elif all(k in dims for k in ("length", "height")):
            fw, fd = dims["length"], dims["height"]
        elif "diameter" in dims:
            fw = fd = dims["diameter"]
        elif "outer_diameter" in dims:
            fw = fd = dims["outer_diameter"]
        else:
            fw = fd = 20  # default

        # Add margin (3mm each side)
        margin = 6
        fw += margin
        fd += margin

        part_footprints.append({
            "name": part.name,
            "footprint_x": round(fw, 1),
            "footprint_y": round(fd, 1),
            "area": round(fw * fd, 1),
        })

    # Sort by area (largest first)
    part_footprints.sort(key=lambda p: p["area"], reverse=True)

    # Greedy grid packing
    positions: list[dict[str, Any]] = []
    x_cursor = 0
    y_cursor = 0
    row_height = 0
    plates_needed = 1

    for pf in part_footprints:
        # Check if fits in current row
        if x_cursor + pf["footprint_x"] > plate["x"]:
            # Move to next row
            x_cursor = 0
            y_cursor += row_height
            row_height = 0

        # Check if fits on current plate
        if y_cursor + pf["footprint_y"] > plate["y"]:
            # Need new plate
            plates_needed += 1
            x_cursor = 0
            y_cursor = 0
            row_height = 0

        positions.append({
            "part": pf["name"],
            "x": round(x_cursor, 1),
            "y": round(y_cursor, 1),
            "plate": plates_needed,
        })

        x_cursor += pf["footprint_x"]
        row_height = max(row_height, pf["footprint_y"])

    total_part_area = sum(p["area"] for p in part_footprints)
    plate_area = plate["x"] * plate["y"]
    utilization = min(total_part_area / (plate_area * plates_needed), 1.0)

    return {
        "printer": printer,
        "plate_size": plate,
        "plates_needed": plates_needed,
        "part_count": len(parts),
        "positions": positions,
        "plate_utilization": round(utilization * 100, 1),
    }


# ---------------------------------------------------------------------------
# Full optimization for assembly
# ---------------------------------------------------------------------------

def optimize_assembly_print(
    assembly: Assembly,
    quality: str = "standard",
    material: str = "PLA",
    printer: str = "ender3",
) -> dict[str, Any]:
    """Run all print optimizations for an assembly's parts.

    Returns comprehensive optimization report.
    """
    result: dict[str, Any] = {
        "assembly_name": assembly.name,
        "quality": quality,
        "material": material,
        "printer": printer,
        "parts": [],
    }

    total_time = 0
    total_support = 0

    for part in assembly.parts:
        orientation = optimize_orientation(part)
        params = recommend_print_params(part, quality)
        tolerance = apply_tolerance_compensation(part.dimensions, "sliding_fit", material)

        result["parts"].append({
            "name": part.name,
            "orientation": orientation,
            "print_params": params,
            "tolerance": tolerance,
        })

        total_time += params.get("estimated_print_time_min", 0)
        total_support += orientation.get("support_volume_cm3", 0)

    # Batch packing
    packing = pack_parts_on_plate(assembly.parts, printer)
    result["packing"] = packing

    result["summary"] = {
        "total_parts": len(assembly.parts),
        "total_print_time_min": round(total_time, 0),
        "total_support_cm3": round(total_support, 1),
        "plates_needed": packing["plates_needed"],
        "plate_utilization_pct": packing["plate_utilization"],
    }

    return result


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class PrintOptimizeTool(Tool):
    name = "print_optimize"
    description = (
        "优化 3D 打印设置：打印方向、公差补偿、参数推荐、批量排版。"
        "输出每个零件的最佳打印参数和排布方案。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "assembly_name": {"type": "string", "description": "装配体名称"},
                "quality": {
                    "type": "string", "enum": ["draft", "standard", "high"],
                    "description": "打印质量（默认 standard）",
                },
                "material": {
                    "type": "string", "description": "打印材料（默认 PLA）",
                },
                "printer": {
                    "type": "string", "description": "打印机型号（ender3/prusa_mk3/bambu_x1c）",
                },
            }, "required": []},
        )

    def execute(self, *, assembly_name: str = "robotic_arm",
                quality: str = "standard", material: str = "PLA",
                printer: str = "ender3",
                **kwargs: Any) -> str:
        assembly = _resolve_assembly(assembly_name, "")
        if assembly is None:
            return f"错误：未找到装配体 '{assembly_name}'"

        result = optimize_assembly_print(assembly, quality, material, printer)

        lines = [
            f"[Print Optimization] {assembly.name}",
            f"Quality: {quality}, Material: {material}, Printer: {printer}",
            "",
            "--- Per-Part Optimization ---",
        ]
        for p in result["parts"]:
            lines.append(f"  {p['name']}:")
            o = p["orientation"]
            lines.append(f"    Orientation: rx={o['rotation_x']}° ry={o['rotation_y']}° rz={o['rotation_z']}°")
            lines.append(f"    Support: {'Yes' if o['support_needed'] else 'No'}")
            pp = p["print_params"]
            lines.append(f"    Params: {pp['layer_height']}mm layer, {pp['infill']}% infill, ~{pp.get('estimated_print_time_min', 0):.0f}min")

        s = result["summary"]
        lines.append("")
        lines.append("--- Summary ---")
        lines.append(f"  Total parts: {s['total_parts']}")
        lines.append(f"  Total print time: ~{s['total_print_time_min']:.0f} min")
        lines.append(f"  Plates needed: {s['plates_needed']}")
        lines.append(f"  Plate utilization: {s['plate_utilization_pct']}%")

        lines.append("")
        lines.append("--- JSON ---")
        lines.append(json.dumps(result, ensure_ascii=False, indent=2))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_print_optimize_tools(registry: Any) -> None:
    """Register print optimization tools."""
    registry.register(PrintOptimizeTool())
