"""Routes for STEP/FCStd conversion (sync + async).

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

# Deferred imports from app.py (avoid circular dependency: app.py includes
# this router at module level). Resolved on first use.
_app_helpers = None
def _get_app_helpers():
    global _app_helpers
    if _app_helpers is None:
        from ..app import DATA_ROOT, _workspace_root, _resolve_safe
        _app_helpers = (DATA_ROOT, _workspace_root, _resolve_safe)
    return _app_helpers

@router.get("/api/convert-step")
async def api_convert_step(path: str = Query(...)) -> JSONResponse:
    """Convert a STEP file to STL using FreeCAD (server-side). Returns the
    relative path of the generated STL within the workspace, so the client
    can load it via /api/file."""
    freecad = _find_freecad()
    if freecad is None:
        raise HTTPException(status_code=503, detail="FreeCAD not available on server")
    ws = _get_app_helpers()[1]()
    src = _get_app_helpers()[2](path, ws)
    if src is None or not src.exists() or src.suffix.lower() not in {".step", ".stp"}:
        raise HTTPException(status_code=404, detail="STEP file not found or access denied")

    # Cache converted STL next to the source (with .stl extension)
    cache = src.with_suffix(".preview.stl")
    # Reuse if fresh (newer than source)
    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        try:
            rel = str(cache.relative_to(ws)).replace("\\", "/")
            return JSONResponse({"stl_path": rel, "cached": True})
        except ValueError:
            pass

    # Run FreeCAD conversion — use json.dumps to prevent path injection
    import json as _json
    script = (
        "import sys, json, FreeCAD, Part, Mesh, MeshPart\n"
        f"shape = Part.Shape()\n"
        f"shape.read({_json.dumps(str(src))})\n"
        f"mesh = MeshPart.meshFromShape(shape, LinearDeflection=0.5, AngularDeflection=0.523599)\n"
        f"mesh.write({_json.dumps(str(cache))})\n"
        f"print('OK', mesh.CountPoints, mesh.CountFacets)\n"
    )
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(script)
        script_path = tf.name
    try:
        proc = subprocess.run(
            [freecad, script_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=60,
        )
        if proc.returncode != 0 or not cache.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Conversion failed: {(proc.stderr or proc.stdout)[:500]}"
            )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Conversion timed out")
    finally:
        try:
            Path(script_path).unlink()
        except OSError:
            pass

    try:
        rel = str(cache.relative_to(ws)).replace("\\", "/")
    except ValueError:
        rel = str(cache)
    return JSONResponse({"stl_path": rel, "cached": False})


@router.get("/api/convert-fcstd")
async def api_convert_fcstd(path: str = Query(...)) -> JSONResponse:
    """Convert a FreeCAD .FCStd document to STL using FreeCADCmd (server-side).
    Merges all visible solid objects (Part::Feature, PartDesign::Body, etc.)
    into a single mesh. Returns the relative path of the generated STL."""
    freecad = _find_freecad()
    if freecad is None:
        raise HTTPException(status_code=503, detail="FreeCAD not available on server")
    ws = _get_app_helpers()[1]()
    src = _get_app_helpers()[2](path, ws)
    if src is None or not src.exists() or src.suffix.lower() != ".fcstd":
        raise HTTPException(status_code=404, detail="FCStd file not found or access denied")

    cache = src.with_suffix(".preview.stl")
    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        try:
            rel = str(cache.relative_to(ws)).replace("\\", "/")
            return JSONResponse({"stl_path": rel, "cached": True})
        except ValueError:
            pass

    # FreeCAD script: open document, collect all visible shapes, merge, tessellate, write STL.
    # Use a temp log file to capture output for diagnostics.
    import tempfile
    import json as _json2
    script = (
        "import sys, json, FreeCAD, FreeCADGui, Part, Mesh, MeshPart\n"
        f"doc = FreeCAD.openDocument({_json2.dumps(str(src))})\n"
        "shapes = []\n"
        "for obj in doc.Objects:\n"
        "    # Skip hidden / helper objects\n"
        "    if hasattr(obj, 'Visibility') and not obj.Visibility:\n"
        "        continue\n"
        "    shape = None\n"
        "    if hasattr(obj, 'Shape') and obj.Shape is not None:\n"
        "        try:\n"
        "            shape = obj.Shape\n"
        "        except Exception:\n"
        "            shape = None\n"
        "    if shape is not None and not shape.isNull():\n"
        "        shapes.append(shape)\n"
        "if not shapes:\n"
        "    print('ERR no shapes')\n"
        "    sys.exit(1)\n"
        "compound = Part.makeCompound(shapes) if len(shapes) > 1 else shapes[0]\n"
        f"mesh = MeshPart.meshFromShape(compound, LinearDeflection=0.5, AngularDeflection=0.523599)\n"
        f"mesh.write({_json2.dumps(str(cache))})\n"
        f"print('OK', mesh.CountPoints, mesh.CountFacets, 'shapes', len(shapes))\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(script)
        script_path = tf.name
    import subprocess
    try:
        proc = subprocess.run(
            [freecad, script_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=120,
        )
        if proc.returncode != 0 or not cache.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Conversion failed: {(proc.stderr or proc.stdout)[:500]}"
            )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Conversion timed out")
    finally:
        try:
            Path(script_path).unlink()
        except OSError:
            pass

    try:
        rel = str(cache.relative_to(ws)).replace("\\", "/")
    except ValueError:
        rel = str(cache)
    return JSONResponse({"stl_path": rel, "cached": False})


# ---------------------------------------------------------------------------
# Routes: async conversion (Step 4 — engineering reliability)
# ---------------------------------------------------------------------------

_convert_lock = threading.Lock()
_convert_queue: dict[str, dict[str, Any]] = {}


def _run_conversion(job_id: str, src_path: Path, output_path: Path, src_ext: str) -> None:
    """Execute FreeCAD conversion in a background thread."""
    freecad = _find_freecad()
    if freecad is None:
        with _convert_lock:
            _convert_queue[job_id]["status"] = "failed"
            _convert_queue[job_id]["error"] = "FreeCAD not available"
        return

    with _convert_lock:
        _convert_queue[job_id]["status"] = "running"

    try:
        # Use JSON sidecar to pass paths safely (avoids f-string injection)
        import json as _json
        paths_json = _json.dumps({"src": str(src_path), "out": str(output_path)})
        if src_ext == ".step" or src_ext == ".stp":
            script = (
                "import sys, json, FreeCAD, Part, Mesh, MeshPart\n"
                f"_paths = json.loads({paths_json})\n"
                "shape = Part.Shape()\n"
                "shape.read(_paths['src'])\n"
                "mesh = MeshPart.meshFromShape(shape, LinearDeflection=0.5, AngularDeflection=0.523599)\n"
                "mesh.write(_paths['out'])\n"
                "print('OK', mesh.CountPoints, mesh.CountFacets)\n"
            )
        elif src_ext == ".fcstd":
            script = (
                "import sys, json, FreeCAD, Part, Mesh, MeshPart\n"
                f"_paths = json.loads({paths_json})\n"
                "doc = FreeCAD.openDocument(_paths['src'])\n"
                "shapes = []\n"
                "for obj in doc.Objects:\n"
                "    if hasattr(obj, 'Visibility') and not obj.Visibility:\n"
                "        continue\n"
                "    if hasattr(obj, 'Shape') and obj.Shape is not None:\n"
                "        try:\n"
                "            shapes.append(obj.Shape)\n"
                "        except Exception:\n"
                "            pass\n"
                "if not shapes:\n"
                "    print('ERR no shapes')\n"
                "    sys.exit(1)\n"
                "compound = Part.makeCompound(shapes) if len(shapes) > 1 else shapes[0]\n"
                "mesh = MeshPart.meshFromShape(compound, LinearDeflection=0.5, AngularDeflection=0.523599)\n"
                "mesh.write(_paths['out'])\n"
                "print('OK', mesh.CountPoints, mesh.CountFacets)\n"
            )
        else:
            raise ValueError(f"Unsupported format: {src_ext}")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
            tf.write(script)
            script_path = tf.name

        try:
            proc = subprocess.run(
                [freecad, script_path],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=120,
            )
            if proc.returncode != 0 or not output_path.exists():
                err_msg = (proc.stderr or proc.stdout)[:500]
                with _convert_lock:
                    _convert_queue[job_id]["status"] = "failed"
                    _convert_queue[job_id]["error"] = f"Conversion failed: {err_msg}"
                return
        except subprocess.TimeoutExpired:
            with _convert_lock:
                _convert_queue[job_id]["status"] = "failed"
                _convert_queue[job_id]["error"] = "Conversion timed out"
            return
        finally:
            try:
                Path(script_path).unlink()
            except OSError:
                pass

        ws = _get_app_helpers()[1]()
        try:
            rel = str(output_path.relative_to(ws)).replace("\\", "/")
        except ValueError:
            rel = str(output_path)

        with _convert_lock:
            _convert_queue[job_id]["status"] = "done"
            _convert_queue[job_id]["result"] = {"stl_path": rel, "cached": False}

    except Exception as e:
        with _convert_lock:
            _convert_queue[job_id]["status"] = "failed"
            _convert_queue[job_id]["error"] = str(e)


@router.post("/api/convert-async")
async def api_convert_async(path: str = Query(...), format: str = Query("stl")) -> JSONResponse:
    """Submit an asynchronous conversion task. Returns a job_id for polling."""
    ws = _get_app_helpers()[1]()
    src = _get_app_helpers()[2](path, ws)
    if src is None or not src.exists():
        raise HTTPException(status_code=404, detail="File not found or access denied")

    ext = src.suffix.lower()
    if ext not in {".step", ".stp", ".fcstd"}:
        raise HTTPException(status_code=400, detail=f"Unsupported source format: {ext}")

    freecad = _find_freecad()
    if freecad is None:
        raise HTTPException(status_code=503, detail="FreeCAD not available on server")

    output_path = src.with_suffix(".preview.stl")

    # If already cached, return immediately
    if output_path.exists() and output_path.stat().st_mtime >= src.stat().st_mtime:
        try:
            rel = str(output_path.relative_to(ws)).replace("\\", "/")
        except ValueError:
            rel = str(output_path)
        job_id = str(uuid.uuid4())[:8]
        with _convert_lock:
            _convert_queue[job_id] = {
                "status": "done",
                "result": {"stl_path": rel, "cached": True},
            }
        return JSONResponse({"job_id": job_id, "status": "done", "stl_path": rel, "cached": True})

    job_id = str(uuid.uuid4())[:8]
    with _convert_lock:
        _convert_queue[job_id] = {"status": "pending", "created_at": _time.time()}
        # Cleanup old entries (keep last 100)
        if len(_convert_queue) > 100:
            now = _time.time()
            old_keys = [
                k for k, v in _convert_queue.items()
                if v.get("status") in ("done", "failed") and now - v.get("created_at", 0) > 3600
            ]
            for k in old_keys:
                del _convert_queue[k]

    thread = threading.Thread(
        target=_run_conversion,
        args=(job_id, src, output_path, ext),
        daemon=True,
        name=f"convert-{job_id}",
    )
    thread.start()

    return JSONResponse({"job_id": job_id, "status": "pending"})


@router.get("/api/convert-status")
async def api_convert_status(job_id: str = Query(...)) -> JSONResponse:
    """Query the status of an asynchronous conversion task."""
    with _convert_lock:
        entry = _convert_queue.get(job_id)

    if entry is None:
        raise HTTPException(status_code=404, detail="Job not found")

    resp: dict[str, Any] = {"status": entry["status"]}
    if "result" in entry:
        resp["result"] = entry["result"]
    if "error" in entry:
        resp["error"] = entry["error"]
    return JSONResponse(resp)


# ---------------------------------------------------------------------------
# Routes: file browsing (Step 5)
# ---------------------------------------------------------------------------

_FILE_TYPE_MAP = {
    ".fcstd": "freecad", ".fcmacro": "freecad",
    ".stl": "model3d", ".step": "model3d", ".stp": "model3d", ".obj": "model3d",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image", ".bmp": "image",
    ".py": "code", ".js": "code", ".html": "code", ".css": "code", ".json": "code",
    ".csv": "data", ".xml": "data", ".yaml": "data", ".yml": "data", ".toml": "data",
    ".txt": "text", ".md": "text", ".log": "text",
}


def _classify_file(p: Path) -> str:
    return _FILE_TYPE_MAP.get(p.suffix.lower(), "other")

