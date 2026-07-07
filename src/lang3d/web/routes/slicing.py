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

# Helpers resolved dynamically from app.py each call (keeps @patch live
# in tests; avoids the caching bug that broke test_async_convert).
def _get_data_root():
    from ..app import DATA_ROOT
    return DATA_ROOT

def _workspace_root():
    from ..app import _workspace_root as _wr
    return _wr()

def _resolve_safe(path, ws):
    from ..app import _resolve_safe as _rs
    return _rs(path, ws)

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
        from ...tools.slicing import SliceModelTool
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
        from ...tools.slicing import SliceAnalyzeTool
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
        from ...tools.slicing import SlicePreviewLayersTool
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

