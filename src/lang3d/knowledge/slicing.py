"""3D printing slicing knowledge base.

Provides printer, material, and quality presets, plus G-code parsing utilities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Printer presets
# ---------------------------------------------------------------------------

PRINTER_PRESETS: dict[str, dict] = {
    "prusa_mk3s": {
        "bed_x": 250, "bed_y": 210, "bed_z": 210,
        "nozzle": 0.4, "name": "Prusa i3 MK3S+",
    },
    "ender_3": {
        "bed_x": 220, "bed_y": 220, "bed_z": 250,
        "nozzle": 0.4, "name": "Creality Ender 3",
    },
    "bambu_p1s": {
        "bed_x": 256, "bed_y": 256, "bed_z": 256,
        "nozzle": 0.4, "name": "Bambu Lab P1S",
    },
    "generic": {
        "bed_x": 200, "bed_y": 200, "bed_z": 200,
        "nozzle": 0.4, "name": "Generic FFF Printer",
    },
}

# ---------------------------------------------------------------------------
# Material presets
# ---------------------------------------------------------------------------

MATERIAL_PRESETS: dict[str, dict] = {
    "pla": {
        "temp": 200, "bed_temp": 60, "diameter": 1.75,
        "density": 1.24, "cost_per_kg": 20, "name": "PLA",
    },
    "abs": {
        "temp": 240, "bed_temp": 100, "diameter": 1.75,
        "density": 1.04, "cost_per_kg": 25, "name": "ABS",
    },
    "petg": {
        "temp": 230, "bed_temp": 80, "diameter": 1.75,
        "density": 1.27, "cost_per_kg": 22, "name": "PETG",
    },
    "tpu": {
        "temp": 220, "bed_temp": 50, "diameter": 1.75,
        "density": 1.21, "cost_per_kg": 30, "name": "TPU",
    },
}

# ---------------------------------------------------------------------------
# Quality presets
# ---------------------------------------------------------------------------

QUALITY_PRESETS: dict[str, dict] = {
    "draft": {
        "layer_height": 0.3, "infill": 10, "perimeters": 2,
        "top_solid_layers": 3, "bottom_solid_layers": 3,
    },
    "standard": {
        "layer_height": 0.2, "infill": 20, "perimeters": 3,
        "top_solid_layers": 4, "bottom_solid_layers": 4,
    },
    "high": {
        "layer_height": 0.12, "infill": 30, "perimeters": 4,
        "top_solid_layers": 6, "bottom_solid_layers": 6,
    },
}

# ---------------------------------------------------------------------------
# G-code statistics parsing patterns
# ---------------------------------------------------------------------------

# PrusaSlicer / OrcaSlicer comment patterns
GCODE_STAT_PATTERNS: dict[str, re.Pattern] = {
    # Total print time
    "print_time_total": re.compile(
        r";\s*(?:estimated printing time|total print time).*?(\d+)h?\s*(\d+)m(?:\s*(\d+)s)?",
        re.IGNORECASE,
    ),
    "print_time_simple": re.compile(
        r";\s*(?:estimated printing time|total print time)\s*=\s*(\d+)",
        re.IGNORECASE,
    ),
    # Filament used
    "filament_mm": re.compile(
        r";\s*filament used\s*\[?mm\]?\s*[:=]\s*([\d.]+)",
        re.IGNORECASE,
    ),
    "filament_g": re.compile(
        r";\s*filament used\s*\[?g\]?\s*[:=]\s*([\d.]+)",
        re.IGNORECASE,
    ),
    "filament_cm3": re.compile(
        r";\s*filament used\s*\[?cm3?\]?\s*[:=]\s*([\d.]+)",
        re.IGNORECASE,
    ),
    # Cost
    "cost": re.compile(
        r";\s*(?:total filament cost|cost)\s*[:=]\s*([\d.]+)",
        re.IGNORECASE,
    ),
    # Layer count (PrusaSlicer style)
    "total_layers": re.compile(
        r";\s*(?:total layers|layer count)\s*[:=]\s*(\d+)",
        re.IGNORECASE,
    ),
    # Support material
    "support_material": re.compile(
        r";\s*support material\s*[:=]\s*(yes|no|true|false)",
        re.IGNORECASE,
    ),
}


# ---------------------------------------------------------------------------
# Merge preset parameters
# ---------------------------------------------------------------------------

def merge_params(
    printer: str = "generic",
    material: str = "pla",
    quality: str = "standard",
    **overrides: object,
) -> dict:
    """Merge printer + material + quality presets with user overrides.

    Returns a flat dict with all slicing parameters.
    """
    printer_preset = PRINTER_PRESETS.get(printer, PRINTER_PRESETS["generic"])
    material_preset = MATERIAL_PRESETS.get(material, MATERIAL_PRESETS["pla"])
    quality_preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["standard"])

    params: dict = {}
    params.update(printer_preset)
    params.update(material_preset)
    params.update(quality_preset)

    # Preserve preset names separately (avoid name collision)
    params["printer_name"] = printer_preset.get("name", "Generic")
    params["material_name"] = material_preset.get("name", "PLA")

    # Apply user overrides (skip None values)
    for key, value in overrides.items():
        if value is not None:
            params[key] = value

    return params


# ---------------------------------------------------------------------------
# G-code parsing utilities
# ---------------------------------------------------------------------------

def parse_gcode_stats(gcode_path: str) -> dict:
    """Parse G-code file comments to extract print statistics.

    Returns a dict with keys: print_time_s, filament_mm, filament_g,
    filament_cm3, cost, total_layers, has_supports, has_brim.
    """
    stats: dict = {
        "print_time_s": 0,
        "filament_mm": 0.0,
        "filament_g": 0.0,
        "filament_cm3": 0.0,
        "cost": 0.0,
        "total_layers": 0,
        "has_supports": False,
        "has_brim": False,
        "print_time_h": 0,
        "print_time_m": 0,
    }

    try:
        with open(gcode_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except (OSError, FileNotFoundError):
        return stats

    # Parse print time
    m = GCODE_STAT_PATTERNS["print_time_total"].search(content)
    if m:
        hours = int(m.group(1) or 0)
        minutes = int(m.group(2) or 0)
        seconds = int(m.group(3) or 0)
        stats["print_time_h"] = hours
        stats["print_time_m"] = minutes
        stats["print_time_s"] = hours * 3600 + minutes * 60 + seconds

    # Parse filament usage
    for key, pattern in [
        ("filament_mm", GCODE_STAT_PATTERNS["filament_mm"]),
        ("filament_g", GCODE_STAT_PATTERNS["filament_g"]),
        ("filament_cm3", GCODE_STAT_PATTERNS["filament_cm3"]),
    ]:
        m = pattern.search(content)
        if m:
            stats[key] = float(m.group(1))

    # Parse cost
    m = GCODE_STAT_PATTERNS["cost"].search(content)
    if m:
        stats["cost"] = float(m.group(1))

    # Parse total layers from comments
    m = GCODE_STAT_PATTERNS["total_layers"].search(content)
    if m:
        stats["total_layers"] = int(m.group(1))

    # If no layer count from comments, count Z changes
    if stats["total_layers"] == 0:
        z_heights: set[float] = set()
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("G1 ") or line.startswith("G0 "):
                z_match = re.search(r"Z([\d.]+)", line)
                if z_match:
                    z_heights.add(float(z_match.group(1)))
        stats["total_layers"] = len(z_heights)

    # Detect supports
    m = GCODE_STAT_PATTERNS["support_material"].search(content)
    if m and m.group(1).lower() in ("yes", "true"):
        stats["has_supports"] = True
    elif "support" in content.lower() and "; support" in content.lower():
        stats["has_supports"] = True

    # Detect brim
    if re.search(r";\s*brim", content, re.IGNORECASE):
        stats["has_brim"] = True

    return stats


def parse_gcode_layers(gcode_path: str) -> list[dict]:
    """Parse G-code to extract per-layer data.

    Each layer dict contains: layer_number, z_height, extrusion_length,
    travel_length, line_count.
    """
    layers: list[dict] = []
    current_z = -1.0
    current_layer: dict | None = None
    layer_num = 0

    try:
        with open(gcode_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except (OSError, FileNotFoundError):
        return layers

    for line in lines:
        line = line.strip()
        if not line or line.startswith(";"):
            continue

        # Detect Z change (layer change)
        if line.startswith("G1 ") or line.startswith("G0 "):
            z_match = re.search(r"Z([\d.]+)", line)
            if z_match:
                z = float(z_match.group(1))
                if z != current_z and z > 0:
                    # Save previous layer
                    if current_layer is not None:
                        layers.append(current_layer)
                    current_z = z
                    layer_num += 1
                    current_layer = {
                        "layer_number": layer_num,
                        "z_height": z,
                        "extrusion_length": 0.0,
                        "travel_length": 0.0,
                        "line_count": 0,
                    }

        if current_layer is None:
            continue

        current_layer["line_count"] += 1

        # Parse E (extrusion) value
        e_match = re.search(r"E([\d.]+)", line)
        if e_match:
            current_layer["extrusion_length"] += float(e_match.group(1))

        # Parse travel (G0 or G1 without E)
        if (line.startswith("G0 ") or line.startswith("G1 ")) and "E" not in line:
            x_match = re.search(r"X([\d.]+)", line)
            y_match = re.search(r"Y([\d.]+)", line)
            if x_match and y_match:
                current_layer["travel_length"] += 1  # Approximate

    # Don't forget last layer
    if current_layer is not None:
        layers.append(current_layer)

    return layers


def parse_gcode_bounds(gcode_path: str) -> dict:
    """Parse G-code to extract print bounds (min/max X, Y, Z coordinates).

    Returns dict with min_x, max_x, min_y, max_y, min_z, max_z.
    X/Y bounds are from extrusion moves only; Z bounds from all moves.
    """
    bounds = {
        "min_x": float("inf"), "max_x": float("-inf"),
        "min_y": float("inf"), "max_y": float("-inf"),
        "min_z": float("inf"), "max_z": float("-inf"),
    }

    try:
        with open(gcode_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except (OSError, FileNotFoundError):
        return {}

    prev_e = 0.0
    for line in lines:
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if not (line.startswith("G0 ") or line.startswith("G1 ")):
            continue

        # Track Z from all moves (layer changes happen without extrusion)
        z_match = re.search(r"Z([\d.]+)", line)
        if z_match:
            z = float(z_match.group(1))
            bounds["min_z"] = min(bounds["min_z"], z)
            bounds["max_z"] = max(bounds["max_z"], z)

        # Track X/Y only from extrusion moves for accurate print bounds
        e_match = re.search(r"E([\d.]+)", line)
        if e_match:
            e_val = float(e_match.group(1))
            if e_val <= prev_e:
                continue
            prev_e = e_val
        else:
            continue

        x_match = re.search(r"X([\d.]+)", line)
        y_match = re.search(r"Y([\d.]+)", line)

        if x_match:
            x = float(x_match.group(1))
            bounds["min_x"] = min(bounds["min_x"], x)
            bounds["max_x"] = max(bounds["max_x"], x)
        if y_match:
            y = float(y_match.group(1))
            bounds["min_y"] = min(bounds["min_y"], y)
            bounds["max_y"] = max(bounds["max_y"], y)

    # Replace infinities with 0
    for k in bounds:
        if bounds[k] == float("inf") or bounds[k] == float("-inf"):
            bounds[k] = 0.0

    return bounds
