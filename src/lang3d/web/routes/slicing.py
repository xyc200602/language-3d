"""Slicing routes: slice model, analyze G-code, view layers.

Extracted from web/app.py (P1-1 God Module split, AGENTS.md §2.1).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Shared paths from app.py
# _get_data_root() — deferred to avoid circular import
_DATA_ROOT = None
def _get_data_root():
    global _DATA_ROOT
    if _DATA_ROOT is None:
        from ..app import DATA_ROOT
        _DATA_ROOT = _get_data_root()
    return _DATA_ROOT

@router.post("/api/slice")
async def api_slice(payload: dict[str, Any]) -> JSONResponse:
    """Submit a slicing task: STL → G-code."""
    stl_path = payload.get("stl_path", "")
    if not stl_path:
        raise HTTPException(status_code=400, detail="Missing 'stl_path'")

    ws = _workspace_root()
    resolved = _resolve_safe(stl_path, ws)
    if resolved is None:
        resolved = _resolve_safe(stl_path, _get_data_root())
    if resolved is None or not resolved.exists():
        raise HTTPException(status_code=404, detail="STL file not found or access denied")

    try:
        from ..tools.slicing import SliceModelTool
    except ImportError:
        raise HTTPException(status_code=503, detail="Slicing tools not available")

    tool = SliceModelTool()
    result_str = tool.execute(
        stl_path=str(resolved),
        printer=payload.get("printer", "generic"),
        material=payload.get("material", "pla"),
        quality=payload.get("quality", "standard"),
        layer_height=payload.get("layer_height"),
        infill=payload.get("infill"),
        supports=payload.get("supports", "auto"),
        brim=payload.get("brim", False),
        output_path=payload.get("output_path"),
    )

    # Try to parse JSON result
    try:
        result_data = json.loads(result_str)
    except json.JSONDecodeError:
        result_data = {"raw": result_str}

    return JSONResponse(result_data)


@router.post("/api/slice/analyze")
async def api_slice_analyze(payload: dict[str, Any]) -> JSONResponse:
    """Analyze an existing G-code file."""
    gcode_path = payload.get("gcode_path", "")
    if not gcode_path:
        raise HTTPException(status_code=400, detail="Missing 'gcode_path'")

    ws = _workspace_root()
    resolved = _resolve_safe(gcode_path, ws)
    if resolved is None:
        resolved = _resolve_safe(gcode_path, _get_data_root())
    if resolved is None or not resolved.exists():
        raise HTTPException(status_code=404, detail="G-code file not found or access denied")

    try:
        from ..tools.slicing import SliceAnalyzeTool
    except ImportError:
        raise HTTPException(status_code=503, detail="Slicing tools not available")

    tool = SliceAnalyzeTool()
    result_str = tool.execute(gcode_path=str(resolved))

    try:
        result_data = json.loads(result_str)
    except json.JSONDecodeError:
        result_data = {"raw": result_str}

    return JSONResponse(result_data)


@router.post("/api/slice/layers")
async def api_slice_layers(payload: dict[str, Any]) -> JSONResponse:
    """Extract per-layer data from a G-code file."""
    gcode_path = payload.get("gcode_path", "")
    if not gcode_path:
        raise HTTPException(status_code=400, detail="Missing 'gcode_path'")

    ws = _workspace_root()
    resolved = _resolve_safe(gcode_path, ws)
    if resolved is None:
        resolved = _resolve_safe(gcode_path, _get_data_root())
    if resolved is None or not resolved.exists():
        raise HTTPException(status_code=404, detail="G-code file not found or access denied")

    try:
        from ..tools.slicing import SlicePreviewLayersTool
    except ImportError:
        raise HTTPException(status_code=503, detail="Slicing tools not available")

    tool = SlicePreviewLayersTool()
    result_str = tool.execute(
        gcode_path=str(resolved),
        layer_range=payload.get("layer_range", "all"),
    )

    try:
        result_data = json.loads(result_str)
    except json.JSONDecodeError:
        result_data = {"raw": result_str}

    return JSONResponse(result_data)


# ---------------------------------------------------------------------------
# Routes: complex robot design (Task 58)
# ---------------------------------------------------------------------------

# Module-level cache for design calculations (avoids repeated solve).
_design_cache: dict[str, Any] = {}


def _get_default_robot_data() -> dict[str, Any]:
    """Build and cache the default complex_robot assembly data."""
    if "robot" in _design_cache:
        return _design_cache["robot"]

    from ..tools.export_package import build_complex_robot, _build_subsystems
    from ..tools.assembly_solver import AssemblySolver
    from ..knowledge.mechanics import compute_assembly_mass

    assembly = build_complex_robot()
    solver = AssemblySolver(assembly)
    positions = solver.solve()
    mass_result = compute_assembly_mass(assembly)
    subsystems = _build_subsystems(assembly, positions)

    data = {
        "assembly": assembly,
        "positions": positions,
        "mass_result": mass_result,
        "subsystems": subsystems,
    }
    _design_cache["robot"] = data
    return data

