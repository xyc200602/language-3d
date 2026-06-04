"""3D printing slicing tools: STL to G-code pipeline.

Tools:
  slice_model           - Slice STL file to G-code via PrusaSlicer/OrcaSlicer
  slice_analyze         - Analyze existing G-code for print statistics
  slice_preview_layers  - Extract per-layer data from G-code
  slice_vlm_analyze     - VLM visual analysis of sliced model
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from ..knowledge.slicing import (
    MATERIAL_PRESETS,
    PRINTER_PRESETS,
    QUALITY_PRESETS,
    merge_params,
    parse_gcode_bounds,
    parse_gcode_layers,
    parse_gcode_stats,
)
from ..models.base import ToolDefinition
from .base import Tool


# ---------------------------------------------------------------------------
# Slicer discovery
# ---------------------------------------------------------------------------

_WINDOWS_PRUSA_PATHS = [
    r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe",
    r"C:\Program Files (x86)\Prusa3D\PrusaSlicer\prusa-slicer-console.exe",
]

_WINDOWS_ORCA_PATHS = [
    r"C:\Program Files\OrcaSlicer\orca-slicer.exe",
    r"C:\Program Files (x86)\OrcaSlicer\orca-slicer.exe",
]

_WINDOWS_PRUSA_GUI_PATHS = [
    r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer.exe",
    r"C:\Program Files (x86)\Prusa3D\PrusaSlicer\prusa-slicer.exe",
]


def _find_slicer() -> str | None:
    """Discover slicer CLI executable (PrusaSlicer -> OrcaSlicer)."""
    # 1. Environment variable
    env_path = os.environ.get("PRUSASLICER_PATH", "")
    if env_path and Path(env_path).exists():
        return env_path

    # 2. Common install paths
    for p in _WINDOWS_PRUSA_PATHS:
        if Path(p).exists():
            return p

    # 3. PATH lookup
    found = shutil.which("prusa-slicer-console")
    if found:
        return found

    # 4. Try OrcaSlicer
    env_path = os.environ.get("ORCASLICER_PATH", "")
    if env_path and Path(env_path).exists():
        return env_path

    for p in _WINDOWS_ORCA_PATHS:
        if Path(p).exists():
            return p

    found = shutil.which("orca-slicer")
    if found:
        return found

    return None


def _find_slicer_gui() -> str | None:
    """Discover slicer GUI executable for VLM analysis."""
    env_path = os.environ.get("PRUSASLICER_GUI_PATH", "")
    if env_path and Path(env_path).exists():
        return env_path

    for p in _WINDOWS_PRUSA_GUI_PATHS:
        if Path(p).exists():
            return p

    return None


# ---------------------------------------------------------------------------
# CLI command builders
# ---------------------------------------------------------------------------

def _build_prusa_command(
    stl_path: str,
    output_path: str,
    params: dict,
) -> list[str]:
    """Build PrusaSlicer CLI command."""
    cmd = [
        _find_slicer() or "prusa-slicer-console",
        "--export-gcode",
        "-o", output_path,
        "--nozzle-diameter", str(params.get("nozzle", 0.4)),
        "--filament-diameter", str(params.get("diameter", 1.75)),
        "--temperature", str(params.get("temp", 200)),
        "--bed-temperature", str(params.get("bed_temp", 60)),
        "--layer-height", str(params.get("layer_height", 0.2)),
        "--fill-density", str(params.get("infill", 20) / 100.0),
        "--perimeters", str(params.get("perimeters", 3)),
        "--top-solid-layers", str(params.get("top_solid_layers", 4)),
        "--bottom-solid-layers", str(params.get("bottom_solid_layers", 4)),
    ]

    # Support
    supports = params.get("supports", "auto")
    if supports == "no":
        cmd.extend(["--support-material", "0"])
    elif supports == "yes":
        cmd.extend(["--support-material", "1"])
    elif supports == "buildplate_only":
        cmd.extend(["--support-material", "1", "--support-material-buildplate-only", "1"])
    # "auto" is default, no flag needed

    # Brim
    if params.get("brim", False):
        cmd.extend(["--brim", "1"])

    # Print bed center
    cmd.extend([
        "--center", f"{params.get('bed_x', 200) / 2},{params.get('bed_y', 200) / 2}",
    ])

    cmd.append(stl_path)
    return cmd


def _build_orca_command(
    stl_path: str,
    output_path: str,
    params: dict,
) -> list[str]:
    """Build OrcaSlicer CLI command."""
    slicer_path = os.environ.get("ORCASLICER_PATH", "")
    if not slicer_path or not Path(slicer_path).exists():
        # Try to find it
        for p in _WINDOWS_ORCA_PATHS:
            if Path(p).exists():
                slicer_path = p
                break

    cmd = [
        slicer_path or "orca-slicer",
        "--export-gcode",
        "-o", output_path,
        "--nozzle-diameter", str(params.get("nozzle", 0.4)),
        "--filament-diameter", str(params.get("diameter", 1.75)),
        "--temperature", str(params.get("temp", 200)),
        "--bed-temperature", str(params.get("bed_temp", 60)),
        "--layer-height", str(params.get("layer_height", 0.2)),
        "--fill-density", str(params.get("infill", 20) / 100.0),
        "--perimeters", str(params.get("perimeters", 3)),
        stl_path,
    ]

    return cmd


# ---------------------------------------------------------------------------
# Tool: slice_model
# ---------------------------------------------------------------------------

class SliceModelTool(Tool):
    """Slice an STL file into G-code using PrusaSlicer or OrcaSlicer."""

    name = "slice_model"
    description = "Slice an STL file into G-code for 3D printing. Uses PrusaSlicer (preferred) or OrcaSlicer."

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "stl_path": {
                        "type": "string",
                        "description": "Path to the STL file to slice",
                    },
                    "printer": {
                        "type": "string",
                        "description": "Printer preset: prusa_mk3s, ender_3, bambu_p1s, or generic",
                        "default": "generic",
                        "enum": list(PRINTER_PRESETS.keys()),
                    },
                    "material": {
                        "type": "string",
                        "description": "Material preset: pla, abs, petg, or tpu",
                        "default": "pla",
                        "enum": list(MATERIAL_PRESETS.keys()),
                    },
                    "quality": {
                        "type": "string",
                        "description": "Quality preset: draft, standard, or high",
                        "default": "standard",
                        "enum": list(QUALITY_PRESETS.keys()),
                    },
                    "layer_height": {
                        "type": "number",
                        "description": "Override layer height (mm), e.g. 0.2",
                    },
                    "infill": {
                        "type": "integer",
                        "description": "Override infill percentage (0-100)",
                    },
                    "supports": {
                        "type": "string",
                        "description": "Support strategy: auto, yes, no, buildplate_only",
                        "default": "auto",
                        "enum": ["auto", "yes", "no", "buildplate_only"],
                    },
                    "brim": {
                        "type": "boolean",
                        "description": "Add brim to the print",
                        "default": False,
                    },
                    "output_path": {
                        "type": "string",
                        "description": "G-code output path (default: same directory as STL)",
                    },
                },
                "required": ["stl_path"],
            },
        )

    def execute(self, **kwargs: Any) -> str:
        stl_path = kwargs.get("stl_path", "")
        if not stl_path:
            return "Error: stl_path is required"

        stl = Path(stl_path)
        if not stl.exists():
            return f"Error: STL file not found: {stl_path}"
        if stl.suffix.lower() not in (".stl", ".obj"):
            return f"Error: Unsupported file format: {stl.suffix}"

        # Find slicer
        slicer = _find_slicer()
        if slicer is None:
            return (
                "Error: No slicer found. Install PrusaSlicer or OrcaSlicer, "
                "or set PRUSASLICER_PATH / ORCASLICER_PATH environment variable."
            )

        # Merge parameters
        params = merge_params(
            printer=kwargs.get("printer", "generic"),
            material=kwargs.get("material", "pla"),
            quality=kwargs.get("quality", "standard"),
            layer_height=kwargs.get("layer_height"),
            infill=kwargs.get("infill"),
            supports=kwargs.get("supports", "auto"),
            brim=kwargs.get("brim", False),
        )

        # Determine output path
        output_path = kwargs.get("output_path", "")
        if not output_path:
            output_path = str(stl.with_suffix(".gcode"))

        # Build command
        cmd = _build_prusa_command(str(stl), output_path, params)

        # Execute
        try:
            timeout = 300  # 5 minutes default
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Error: Slicing timed out after {timeout}s"
        except FileNotFoundError:
            return f"Error: Slicer executable not found: {cmd[0]}"

        if proc.returncode != 0:
            return (
                f"Error: Slicing failed (exit code {proc.returncode})\n"
                f"stderr: {proc.stderr[:500]}\n"
                f"stdout: {proc.stdout[:500]}"
            )

        # Check output
        if not Path(output_path).exists():
            return f"Error: G-code file was not generated at {output_path}"

        # Parse stats from generated G-code
        stats = parse_gcode_stats(output_path)

        result = {
            "status": "success",
            "gcode_path": output_path,
            "gcode_size_kb": Path(output_path).stat().st_size // 1024,
            "slicer": Path(slicer).name,
            "printer": params.get("printer_name", "generic"),
            "material": params.get("material_name", "PLA"),
            "quality_preset": kwargs.get("quality", "standard"),
            "layer_height": params.get("layer_height", 0.2),
            "infill_percent": params.get("infill", 20),
            "stats": stats,
        }

        return json.dumps(result, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool: slice_analyze
# ---------------------------------------------------------------------------

class SliceAnalyzeTool(Tool):
    """Analyze a G-code file for print statistics."""

    name = "slice_analyze"
    description = "Analyze an existing G-code file to extract print statistics (time, material, cost, layers)."

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "gcode_path": {
                        "type": "string",
                        "description": "Path to the G-code file to analyze",
                    },
                },
                "required": ["gcode_path"],
            },
        )

    def execute(self, **kwargs: Any) -> str:
        gcode_path = kwargs.get("gcode_path", "")
        if not gcode_path:
            return "Error: gcode_path is required"

        gcode = Path(gcode_path)
        if not gcode.exists():
            return f"Error: G-code file not found: {gcode_path}"
        if gcode.suffix.lower() not in (".gcode", ".g", ".gc", ".ngc"):
            return f"Warning: File does not appear to be a G-code file: {gcode.suffix}"

        # Parse stats
        stats = parse_gcode_stats(gcode_path)

        # Parse bounds
        bounds = parse_gcode_bounds(gcode_path)

        # Calculate estimated cost if we have filament_g
        material = "pla"  # Default
        material_info = MATERIAL_PRESETS.get(material, {})
        filament_g = stats.get("filament_g", 0.0)
        if filament_g > 0 and material_info:
            estimated_cost = filament_g / 1000.0 * material_info.get("cost_per_kg", 20)
        else:
            estimated_cost = stats.get("cost", 0.0)

        # Format print time
        total_s = stats.get("print_time_s", 0)
        hours = total_s // 3600
        minutes = (total_s % 3600) // 60
        seconds = total_s % 60

        result = {
            "gcode_path": gcode_path,
            "file_size_kb": gcode.stat().st_size // 1024,
            "print_time": {
                "total_seconds": total_s,
                "formatted": f"{hours}h {minutes}m {seconds}s",
            },
            "filament": {
                "length_mm": round(stats.get("filament_mm", 0.0), 1),
                "weight_g": round(filament_g, 1),
                "volume_cm3": round(stats.get("filament_cm3", 0.0), 2),
            },
            "estimated_cost": round(estimated_cost, 2),
            "layers": stats.get("total_layers", 0),
            "has_supports": stats.get("has_supports", False),
            "has_brim": stats.get("has_brim", False),
            "print_bounds": bounds,
        }

        return json.dumps(result, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool: slice_preview_layers
# ---------------------------------------------------------------------------

class SlicePreviewLayersTool(Tool):
    """Extract per-layer data from a G-code file."""

    name = "slice_preview_layers"
    description = "Extract per-layer information from G-code (Z height, extrusion, travel per layer)."

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "gcode_path": {
                        "type": "string",
                        "description": "Path to the G-code file",
                    },
                    "layer_range": {
                        "type": "string",
                        "description": 'Layer range: "all", "1-10", "5,10,15"',
                        "default": "all",
                    },
                },
                "required": ["gcode_path"],
            },
        )

    def execute(self, **kwargs: Any) -> str:
        gcode_path = kwargs.get("gcode_path", "")
        if not gcode_path:
            return "Error: gcode_path is required"

        gcode = Path(gcode_path)
        if not gcode.exists():
            return f"Error: G-code file not found: {gcode_path}"

        layer_range = kwargs.get("layer_range", "all")

        # Parse all layers
        all_layers = parse_gcode_layers(gcode_path)

        if not all_layers:
            return json.dumps({
                "total_layers": 0,
                "layers": [],
                "message": "No layer data found in G-code",
            }, ensure_ascii=False)

        # Filter by range
        if layer_range == "all":
            filtered = all_layers
        elif "-" in layer_range and "," not in layer_range:
            # Range: "1-10"
            parts = layer_range.split("-")
            try:
                start = int(parts[0])
                end = int(parts[1])
                filtered = [l for l in all_layers if start <= l["layer_number"] <= end]
            except (ValueError, IndexError):
                filtered = all_layers
        elif "," in layer_range:
            # Specific layers: "5,10,15"
            try:
                selected = {int(x.strip()) for x in layer_range.split(",")}
                filtered = [l for l in all_layers if l["layer_number"] in selected]
            except ValueError:
                filtered = all_layers
        else:
            filtered = all_layers

        result = {
            "total_layers": len(all_layers),
            "returned_layers": len(filtered),
            "layers": filtered,
        }

        return json.dumps(result, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool: slice_vlm_analyze
# ---------------------------------------------------------------------------

class SliceVLMAnalyzeTool(Tool):
    """VLM visual analysis of sliced model via slicer GUI screenshot."""

    name = "slice_vlm_analyze"
    description = "Open slicer GUI with G-code, capture screenshot, and analyze with VLM."

    def __init__(
        self,
        router: Any = None,
        screenshot_dir: str = "",
    ) -> None:
        self.router = router
        self.screenshot_dir = screenshot_dir

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "gcode_path": {
                        "type": "string",
                        "description": "Path to the G-code file to visually analyze",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Custom analysis prompt for VLM",
                        "default": "Analyze this 3D print slice preview. Check for potential issues like insufficient supports, thin walls, overhangs, and print quality concerns.",
                    },
                    "detail": {
                        "type": "string",
                        "description": "VLM detail level: fast, standard, detailed, maximum",
                        "default": "standard",
                        "enum": ["fast", "standard", "detailed", "maximum"],
                    },
                },
                "required": ["gcode_path"],
            },
        )

    def execute(self, **kwargs: Any) -> str:
        gcode_path = kwargs.get("gcode_path", "")
        if not gcode_path:
            return "Error: gcode_path is required"

        gcode = Path(gcode_path)
        if not gcode.exists():
            return f"Error: G-code file not found: {gcode_path}"

        if self.router is None:
            return "Error: VLM router not available"

        # Find slicer GUI
        gui_path = _find_slicer_gui()
        if gui_path is None:
            return "Error: No slicer GUI found. Install PrusaSlicer or set PRUSASLICER_GUI_PATH."

        prompt = kwargs.get(
            "prompt",
            "Analyze this 3D print slice preview. Check for potential issues.",
        )
        detail = kwargs.get("detail", "standard")

        try:
            # Launch slicer GUI with G-code
            proc = subprocess.Popen(
                [gui_path, "--gcode", str(gcode)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Wait for GUI to load
            time.sleep(5)

            # Capture screenshot
            try:
                from .screen import WindowCaptureTool
                capture = WindowCaptureTool(screenshot_dir=self.screenshot_dir)
                # Try to find the slicer window
                screenshot_result = capture.execute(
                    title="PrusaSlicer",
                    save=True,
                )
            except Exception as e:
                proc.terminate()
                return f"Error capturing screenshot: {e}"

            # Extract screenshot path from result
            screenshot_path = ""
            for line in screenshot_result.splitlines():
                if "Saved:" in line or "path" in line.lower():
                    parts = line.split("Saved:")
                    if len(parts) > 1:
                        screenshot_path = parts[1].strip()
                    break

            # If no screenshot path found, try to find latest screenshot
            if not screenshot_path and self.screenshot_dir:
                ss_dir = Path(self.screenshot_dir)
                if ss_dir.exists():
                    screenshots = sorted(ss_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if screenshots:
                        screenshot_path = str(screenshots[0])

            # Close slicer
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

            if not screenshot_path or not Path(screenshot_path).exists():
                return "Error: Could not capture slicer screenshot"

            # VLM analysis
            from ..models.router import VisionDetail
            try:
                detail_level = VisionDetail(detail)
            except ValueError:
                detail_level = VisionDetail.STANDARD

            vlm_result = self.router.vision(
                image_path=screenshot_path,
                prompt=prompt,
                detail=detail_level,
            )

            result = {
                "gcode_path": str(gcode),
                "screenshot_path": screenshot_path,
                "vlm_analysis": vlm_result,
            }

            return json.dumps(result, indent=2, ensure_ascii=False)

        except Exception as e:
            return f"Error during VLM analysis: {e}"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_slicing_tools(
    registry: Any,
    router: Any = None,
    screenshot_dir: str = "",
) -> None:
    """Register all slicing tools with the tool registry."""
    registry.register(SliceModelTool())
    registry.register(SliceAnalyzeTool())
    registry.register(SlicePreviewLayersTool())
    registry.register(SliceVLMAnalyzeTool(
        router=router,
        screenshot_dir=screenshot_dir,
    ))
