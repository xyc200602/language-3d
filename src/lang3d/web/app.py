"""FastAPI web monitoring panel."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import subprocess
import tempfile
import threading
import time as _time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Language-3D Agent Monitor")

# Project root (…/language-3d) — used to build a safe data-root for file serving.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATA_ROOT_CANDIDATES = [
    _PROJECT_ROOT / "data",
    Path.home() / "Desktop" / "language-3d" / "data",
]
DATA_ROOT = next((p for p in _DATA_ROOT_CANDIDATES if p.exists()), _PROJECT_ROOT / "data")

# In-memory state (updated by the agent)
_agent_state: dict[str, Any] = {
    "status": "idle",
    "plan": None,
    "logs": [],
    "screenshots": [],
    "tool_calls": [],
    "vlm_results": [],
    "thinking": "",
    "thinking_history": [],
    "sub_agents": [],
    "dag": None,
    "session_start": _time.time(),
    "session_id": "",
}
_websockets: list[WebSocket] = []

# Task runner state
_agent_instance: Any = None  # Registered Agent instance
_task_thread: threading.Thread | None = None
_task_stop_flag = threading.Event()
_task_history: list[dict[str, Any]] = []
_current_task: dict[str, Any] | None = None

# Lock for state mutations that involve compound read-modify-write
_state_lock = threading.Lock()

# Previous snapshot used for delta WebSocket updates.
_last_snapshot: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Path safety helpers
# ---------------------------------------------------------------------------

def _resolve_safe(rel_or_abs: str, root: Path) -> Path | None:
    """Resolve a path inside `root`. Return None if it escapes the root."""
    if not rel_or_abs:
        return None
    p = Path(rel_or_abs)
    if not p.is_absolute():
        candidate = (root / p).resolve()
    else:
        candidate = p.resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _workspace_root() -> Path:
    """Return the effective workspace root for the registered agent, or DATA_ROOT."""
    if _agent_instance is not None:
        try:
            ws = getattr(_agent_instance.state, "workspace", None)
            if ws is not None:
                ws_path = Path(ws)
                if ws_path.exists():
                    return ws_path
        except Exception:
            pass
    return DATA_ROOT


# ---------------------------------------------------------------------------
# State update API (used by the agent)
# ---------------------------------------------------------------------------

def update_agent_state(**kwargs: Any) -> None:
    """Update the shared agent state and broadcast to WebSocket clients."""
    with _state_lock:
        _agent_state.update(kwargs)
        _agent_state["timestamp"] = _time.strftime("%H:%M:%S")
    broadcast_state()


def add_log(message: str, level: str = "info") -> None:
    """Add a log entry."""
    with _state_lock:
        _agent_state["logs"] = _agent_state.get("logs", [])
        _agent_state["logs"].append({
            "message": message,
            "level": level,
            "time": _time.strftime("%H:%M:%S"),
        })
        _agent_state["logs"] = _agent_state["logs"][-200:]
    broadcast_state()


def add_tool_call(name: str, args: dict, result: str = "") -> None:
    """Record a tool call for the timeline."""
    with _state_lock:
        _agent_state["tool_calls"] = _agent_state.get("tool_calls", [])
        entry = {
            "name": name,
            "args": args,
            "result_preview": result[:200] if result else "",
            "time": _time.strftime("%H:%M:%S"),
            "timestamp": _time.time(),
        }
        _agent_state["tool_calls"].append(entry)
        _agent_state["tool_calls"] = _agent_state["tool_calls"][-100:]

    arg_preview = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3])
    add_log(f"Tool: {name}({arg_preview})", level="tool")


def add_vlm_result(tool: str, prompt: str, result: str, image_path: str = "") -> None:
    """Record a VLM analysis result."""
    with _state_lock:
        _agent_state["vlm_results"] = _agent_state.get("vlm_results", [])
        entry = {
            "tool": tool,
            "prompt": prompt[:200],
            "result": result[:500],
            "image_path": image_path,
            "time": _time.strftime("%H:%M:%S"),
        }
        _agent_state["vlm_results"].append(entry)
        _agent_state["vlm_results"] = _agent_state["vlm_results"][-50:]
    broadcast_state()


def set_thinking(text: str) -> None:
    """Update the current agent thinking text and append to history."""
    with _state_lock:
        _agent_state["thinking"] = text
        history = _agent_state.setdefault("thinking_history", [])
        if text and (not history or history[-1].get("text") != text):
            history.append({
                "text": text,
                "time": _time.strftime("%H:%M:%S"),
            })
            # Keep last 100 entries
            _agent_state["thinking_history"] = history[-100:]
    broadcast_state()


def update_sub_agent(agent_id: str, status: str, step: str = "") -> None:
    """Update or add a sub-agent status entry."""
    with _state_lock:
        agents = _agent_state.get("sub_agents", [])
        found = False
        for entry in agents:
            if entry.get("agent_id") == agent_id:
                entry["status"] = status
                if step:
                    entry["step"] = step
                entry["time"] = _time.strftime("%H:%M:%S")
                found = True
                break
        if not found:
            agents.append({
                "agent_id": agent_id,
                "status": status,
                "step": step,
                "time": _time.strftime("%H:%M:%S"),
            })
        _agent_state["sub_agents"] = agents
    broadcast_state()


def update_dag(dag_data: dict[str, Any]) -> None:
    """Update the DAG visualization data."""
    _agent_state["dag"] = dag_data
    broadcast_state()


# ---------------------------------------------------------------------------
# Agent registration & task execution (Step 2)
# ---------------------------------------------------------------------------

def set_agent_instance(agent: Any) -> None:
    """Register an Agent instance so the web panel can submit tasks."""
    global _agent_instance
    _agent_instance = agent
    try:
        _agent_state["session_id"] = getattr(agent.state, "session_id", "")
    except Exception:
        pass
    try:
        ws = getattr(agent.state, "workspace", None)
        if ws:
            add_log(f"Agent registered (workspace={Path(ws)})", level="info")
    except Exception:
        pass


def _run_task_background(task: str, mode: str = "run") -> None:
    """Background-thread task executor."""
    global _current_task
    _task_stop_flag.clear()
    _current_task = {
        "task": task,
        "mode": mode,
        "started_at": datetime.now().isoformat(),
        "status": "running",
    }
    update_agent_state(status="running")
    add_log(f"Task started ({mode}): {task[:120]}", level="info")
    try:
        if _agent_instance is None:
            raise RuntimeError("No agent registered")
        if mode == "direct":
            result = _agent_instance.run_task(task, use_planning=False)
        else:
            result = _agent_instance.run_task(task, use_planning=True)
        _current_task["status"] = "complete"
        _current_task["result"] = result[:500] if isinstance(result, str) else str(result)[:500]
        update_agent_state(status="complete")
        add_log(f"Task completed: {_current_task['result'][:120]}", level="success")
    except Exception as e:
        _current_task["status"] = "error"
        _current_task["error"] = str(e)
        update_agent_state(status="error")
        add_log(f"Task error: {e}", level="error")
    finally:
        _current_task["finished_at"] = datetime.now().isoformat()
        _task_history.append(dict(_current_task))
        # Keep last 50 entries
        del _task_history[:-50]
        _current_task = None


# ---------------------------------------------------------------------------
# WebSocket broadcast (delta updates — Step 8)
# ---------------------------------------------------------------------------

def _snapshot_state() -> dict[str, Any]:
    """Return a JSON-serializable snapshot of the current state."""
    with _state_lock:
        return json.loads(json.dumps(_agent_state, ensure_ascii=False, default=str))


def _diff(prev: dict[str, Any], curr: dict[str, Any]) -> dict[str, Any]:
    """Compute a shallow diff. Returns empty dict if nothing changed."""
    out: dict[str, Any] = {}
    for k, v in curr.items():
        if k not in prev or prev[k] != v:
            out[k] = v
    # Always include a fresh timestamp so clients know we're alive
    if "timestamp" not in out:
        out["timestamp"] = curr.get("timestamp", _time.strftime("%H:%M:%S"))
    return out


async def _send_json(ws: WebSocket, payload: dict[str, Any]) -> None:
    try:
        await ws.send_text(json.dumps(payload, ensure_ascii=False, default=str))
    except Exception:
        pass


async def broadcast_state_async() -> None:
    """Broadcast current state (delta) to all connected WebSocket clients."""
    global _last_snapshot
    snapshot = _snapshot_state()
    if _last_snapshot is None:
        payload = snapshot
    else:
        payload = _diff(_last_snapshot, snapshot)
        if len(payload) <= 1:  # only timestamp changed
            return
    _last_snapshot = snapshot
    for ws in _websockets[:]:
        await _send_json(ws, payload)


def broadcast_state() -> None:
    """Synchronous wrapper for broadcasting."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(broadcast_state_async())
    except RuntimeError:
        # No event loop — fall back to a new one (rare, e.g. tests)
        pass


# ---------------------------------------------------------------------------
# Routes: HTML & static
# ---------------------------------------------------------------------------

@app.get("/")
async def index() -> HTMLResponse:
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Language-3D Agent Monitor</h1><p>Static files not found</p>")


# ---------------------------------------------------------------------------
# Routes: read-only inspection
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def get_status() -> JSONResponse:
    return JSONResponse(_agent_state)


@app.get("/api/screenshots")
async def get_screenshots() -> JSONResponse:
    return JSONResponse({"screenshots": _agent_state.get("screenshots", [])})


@app.get("/api/tool-calls")
async def get_tool_calls() -> JSONResponse:
    return JSONResponse({"tool_calls": _agent_state.get("tool_calls", [])})


@app.get("/api/vlm-results")
async def get_vlm_results() -> JSONResponse:
    return JSONResponse({"vlm_results": _agent_state.get("vlm_results", [])})


@app.get("/api/screenshot-gallery")
async def get_screenshot_gallery() -> JSONResponse:
    """Return all screenshots with thumbnails for the gallery."""
    screenshots = _agent_state.get("screenshots", [])
    gallery = []
    for s in screenshots:
        path = s.get("path", "") if isinstance(s, dict) else str(s)
        if path and Path(path).exists():
            size_kb = Path(path).stat().st_size // 1024
            gallery.append({
                "path": path,
                "size_kb": size_kb,
                "name": Path(path).name,
                "time": s.get("time", "") if isinstance(s, dict) else "",
            })
    return JSONResponse({"gallery": gallery, "total": len(gallery)})


@app.get("/api/agents")
async def get_agents() -> JSONResponse:
    return JSONResponse({"agents": _agent_state.get("sub_agents", [])})


@app.get("/api/dag")
async def get_dag() -> JSONResponse:
    return JSONResponse({"dag": _agent_state.get("dag")})


# ---------------------------------------------------------------------------
# Routes: file serving (Step 1 — fixes the missing endpoint bug)
# ---------------------------------------------------------------------------

@app.get("/api/screenshot-file")
async def get_screenshot_file(path: str = Query(..., description="Absolute path of the screenshot")) -> FileResponse:
    """Serve a screenshot file from the data directory."""
    # Accept either absolute (must be under data/) or relative to data/.
    p = Path(path)
    if p.is_absolute():
        try:
            p.resolve().relative_to(DATA_ROOT.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Path outside data directory")
    else:
        p = (DATA_ROOT / p).resolve()
        try:
            p.relative_to(DATA_ROOT.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Path outside data directory")

    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    mime, _ = mimetypes.guess_type(str(p))
    return FileResponse(str(p), media_type=mime or "application/octet-stream")


@app.get("/api/file")
async def get_file(path: str = Query(..., description="Path relative to workspace or absolute under data/")) -> FileResponse:
    """Serve any file from the workspace/data directory."""
    ws = _workspace_root()
    resolved = _resolve_safe(path, ws)
    if resolved is None:
        # Try the global DATA_ROOT
        resolved = _resolve_safe(path, DATA_ROOT)
    if resolved is None or not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found or access denied")
    mime, _ = mimetypes.guess_type(str(resolved))
    return FileResponse(str(resolved), media_type=mime or "application/octet-stream")


# ---------------------------------------------------------------------------
# Routes: task submission (Step 2)
# ---------------------------------------------------------------------------

@app.post("/api/run-task")
async def api_run_task(payload: dict[str, Any]) -> JSONResponse:
    """Submit a task to the registered agent (background execution)."""
    global _task_thread
    if _agent_instance is None:
        raise HTTPException(status_code=503, detail="No agent registered. Start the CLI first.")
    if _task_thread is not None and _task_thread.is_alive():
        raise HTTPException(status_code=409, detail="A task is already running")
    task = (payload.get("task") or "").strip()
    mode = payload.get("mode", "run")
    if mode not in ("run", "direct"):
        mode = "run"
    if not task:
        raise HTTPException(status_code=400, detail="Missing 'task'")
    thread = threading.Thread(
        target=_run_task_background, args=(task, mode), daemon=True, name="lang3d-task"
    )
    _task_thread = thread
    thread.start()
    return JSONResponse({"status": "started", "task": task, "mode": mode})


@app.post("/api/stop-task")
async def api_stop_task() -> JSONResponse:
    """Request the current task to stop (cooperative)."""
    if _task_thread is None or not _task_thread.is_alive():
        return JSONResponse({"status": "idle", "message": "No running task"})
    _task_stop_flag.set()
    add_log("Stop requested by user", level="info")
    return JSONResponse({"status": "stop_requested"})


@app.get("/api/is-running")
async def api_is_running() -> JSONResponse:
    running = _task_thread is not None and _task_thread.is_alive()
    return JSONResponse({"running": running, "current": _current_task})


@app.get("/api/task-history")
async def api_task_history() -> JSONResponse:
    return JSONResponse({"history": _task_history, "total": len(_task_history)})


# ---------------------------------------------------------------------------
# Routes: 3D model listing (Step 4)
# ---------------------------------------------------------------------------

_MODEL_EXTS = {".stl", ".step", ".stp", ".obj", ".fcstd"}
_PREVIEWABLE_EXTS = {".stl", ".obj"}


def _find_freecad() -> str | None:
    """Locate FreeCADCmd executable for server-side conversions (STEP → STL)."""
    candidates = [
        "C:/Users/xyc/AppData/Local/Programs/FreeCAD 1.1/bin/freecadcmd.exe",
        "C:/Program Files/FreeCAD 1.1/bin/freecadcmd.exe",
        "C:/Program Files/FreeCAD 1.0/bin/freecadcmd.exe",
        "C:/Program Files/FreeCAD/bin/freecadcmd.exe",
        "/usr/bin/freecadcmd",
        "/usr/local/bin/freecadcmd",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    # Try PATH
    import shutil
    return shutil.which("freecadcmd") or shutil.which("FreeCADCmd")


@app.get("/api/models")
async def api_models() -> JSONResponse:
    """List 3D model files in the workspace. Marks each as `previewable` if
    the browser can render it directly (STL/OBJ) or via server-side conversion
    (STEP → STL when FreeCAD is available)."""
    ws = _workspace_root()
    freecad = _find_freecad()
    models: list[dict[str, Any]] = []
    if ws.exists():
        for p in ws.rglob("*"):
            if p.suffix.lower() not in _MODEL_EXTS or not p.is_file():
                continue
            try:
                stat = p.stat()
            except OSError:
                continue
            ext = p.suffix.lower()
            size = stat.st_size
            previewable = ext in _PREVIEWABLE_EXTS and size > 0
            convertible = ext in {".step", ".stp", ".fcstd"} and freecad is not None
            models.append({
                "name": p.name,
                "path": str(p),
                "rel_path": str(p.relative_to(ws)).replace("\\", "/"),
                "size_kb": size // 1024,
                "size_bytes": size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "ext": ext.lstrip("."),
                "previewable": previewable,
                "convertible": convertible,
                "empty": size == 0,
            })
    # Sort: previewable first, then convertible, then others;
    # within each bucket, newest first.
    def _sort_key(m: dict[str, Any]) -> tuple:
        rank = 0 if m["previewable"] else 1 if m["convertible"] else 2
        try:
            mtime = datetime.fromisoformat(m["modified"]).timestamp()
        except Exception:
            mtime = 0.0
        return (rank, -mtime)

    models.sort(key=_sort_key)
    return JSONResponse({
        "models": models,
        "total": len(models),
        "root": str(ws),
        "freecad_available": freecad is not None,
    })


@app.get("/api/convert-step")
async def api_convert_step(path: str = Query(...)) -> JSONResponse:
    """Convert a STEP file to STL using FreeCAD (server-side). Returns the
    relative path of the generated STL within the workspace, so the client
    can load it via /api/file."""
    freecad = _find_freecad()
    if freecad is None:
        raise HTTPException(status_code=503, detail="FreeCAD not available on server")
    ws = _workspace_root()
    src = _resolve_safe(path, ws)
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

    # Run FreeCAD conversion
    script = (
        "import sys, FreeCAD, Part, Mesh, MeshPart\n"
        f"shape = Part.Shape()\n"
        f"shape.read(r'{src}')\n"
        # Use MeshPart.meshFromShape (more reliable than shape.tessellate)
        f"mesh = MeshPart.meshFromShape(shape, LinearDeflection=0.5, AngularDeflection=0.523599)\n"
        # Write STL using Mesh.write (auto-detects extension)
        f"mesh.write(r'{cache}')\n"
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


@app.get("/api/convert-fcstd")
async def api_convert_fcstd(path: str = Query(...)) -> JSONResponse:
    """Convert a FreeCAD .FCStd document to STL using FreeCADCmd (server-side).
    Merges all visible solid objects (Part::Feature, PartDesign::Body, etc.)
    into a single mesh. Returns the relative path of the generated STL."""
    freecad = _find_freecad()
    if freecad is None:
        raise HTTPException(status_code=503, detail="FreeCAD not available on server")
    ws = _workspace_root()
    src = _resolve_safe(path, ws)
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
    script = (
        "import sys, FreeCAD, FreeCADGui, Part, Mesh, MeshPart\n"
        f"doc = FreeCAD.openDocument(r'{src}')\n"
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
        f"mesh.write(r'{cache}')\n"
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
        if src_ext == ".step" or src_ext == ".stp":
            script = (
                "import sys, FreeCAD, Part, Mesh, MeshPart\n"
                f"shape = Part.Shape()\n"
                f"shape.read(r'{src_path}')\n"
                f"mesh = MeshPart.meshFromShape(shape, LinearDeflection=0.5, AngularDeflection=0.523599)\n"
                f"mesh.write(r'{output_path}')\n"
                f"print('OK', mesh.CountPoints, mesh.CountFacets)\n"
            )
        elif src_ext == ".fcstd":
            script = (
                "import sys, FreeCAD, Part, Mesh, MeshPart\n"
                f"doc = FreeCAD.openDocument(r'{src_path}')\n"
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
                f"mesh = MeshPart.meshFromShape(compound, LinearDeflection=0.5, AngularDeflection=0.523599)\n"
                f"mesh.write(r'{output_path}')\n"
                f"print('OK', mesh.CountPoints, mesh.CountFacets)\n"
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

        ws = _workspace_root()
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


@app.post("/api/convert-async")
async def api_convert_async(path: str = Query(...), format: str = Query("stl")) -> JSONResponse:
    """Submit an asynchronous conversion task. Returns a job_id for polling."""
    ws = _workspace_root()
    src = _resolve_safe(path, ws)
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

    thread = threading.Thread(
        target=_run_conversion,
        args=(job_id, src, output_path, ext),
        daemon=True,
        name=f"convert-{job_id}",
    )
    thread.start()

    return JSONResponse({"job_id": job_id, "status": "pending"})


@app.get("/api/convert-status")
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


@app.get("/api/browse")
async def api_browse(
    path: str = Query("", description="Directory relative to workspace"),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
) -> JSONResponse:
    """Browse files in a workspace subdirectory."""
    ws = _workspace_root()
    if path:
        target = _resolve_safe(path, ws)
        if target is None or not target.exists():
            raise HTTPException(status_code=404, detail="Directory not found")
    else:
        target = ws

    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Not a directory")

    try:
        rel = str(target.relative_to(ws)).replace("\\", "/")
    except ValueError:
        rel = str(target)

    entries: list[dict[str, Any]] = []
    parent_rel = ""
    if target != ws:
        try:
            parent = target.parent
            if parent == ws or ws in parent.resolve().parents or parent.resolve() == ws.resolve():
                parent_rel = str(parent.relative_to(ws)).replace("\\", "/")
                if parent_rel == ".":
                    parent_rel = ""
        except Exception:
            parent_rel = ""

    try:
        children = sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")

    for child in children:
        try:
            stat = child.stat()
        except OSError:
            continue
        if child.is_dir():
            entries.append({
                "name": child.name,
                "path": str(child),
                "rel_path": str(child.relative_to(ws)).replace("\\", "/"),
                "type": "folder",
                "size_kb": 0,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        else:
            entries.append({
                "name": child.name,
                "path": str(child),
                "rel_path": str(child.relative_to(ws)).replace("\\", "/"),
                "type": _classify_file(child),
                "size_kb": stat.st_size // 1024,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

    total = len(entries)
    start = (page - 1) * page_size
    end = start + page_size
    paged = entries[start:end]
    return JSONResponse({
        "path": rel or ".",
        "abs_path": str(target),
        "parent": parent_rel,
        "entries": paged,
        "total": total,
        "page": page,
        "page_size": page_size,
    })


# ---------------------------------------------------------------------------
# Routes: session history (Step 6)
# ---------------------------------------------------------------------------

@app.get("/api/sessions")
async def api_sessions() -> JSONResponse:
    """List all persisted agent sessions (.lang3d_state.json files)."""
    ws = _workspace_root()
    sessions: list[dict[str, Any]] = []
    if ws.exists():
        for p in ws.rglob(".lang3d_state.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            plan = data.get("plan") or {}
            steps = plan.get("steps") or []
            completed = sum(1 for s in steps if s.get("status") == "completed")
            sessions.append({
                "session_id": data.get("session_id", ""),
                "created_at": data.get("created_at", ""),
                "workspace": data.get("workspace", ""),
                "goal": plan.get("goal", ""),
                "step_count": len(steps),
                "completed": completed,
                "tool_calls": len(data.get("tool_history") or []),
                "file": str(p),
            })
    sessions.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return JSONResponse({"sessions": sessions, "total": len(sessions)})


@app.get("/api/session/{session_id}")
async def api_session_detail(session_id: str) -> JSONResponse:
    """Load a full session by id."""
    ws = _workspace_root()
    if ws.exists():
        for p in ws.rglob(".lang3d_state.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if data.get("session_id") == session_id:
                return JSONResponse(data)
    raise HTTPException(status_code=404, detail="Session not found")


# ---------------------------------------------------------------------------
# Routes: part library
# ---------------------------------------------------------------------------

@app.get("/api/parts/catalog")
async def api_parts_catalog(
    query: str = Query("", description="Search keyword"),
    category: str = Query("", description="Filter by category"),
) -> JSONResponse:
    """List/search part templates in the catalog."""
    try:
        from ..knowledge.parts_catalog import (
            CATEGORY_TREE,
            get_all_templates,
            search_parts,
        )
    except ImportError:
        raise HTTPException(status_code=503, detail="Part library not available")

    if query or category:
        results = search_parts(
            query=query,
            category=category if category else None,
        )
    else:
        results = get_all_templates()

    templates = []
    for t in results:
        templates.append({
            "id": t.id,
            "name_en": t.name_en,
            "name_cn": t.name_cn,
            "category": t.category,
            "subcategory": t.subcategory,
            "description": t.description,
            "tags": t.tags,
            "material_default": t.material_default,
            "parameters": [
                {
                    "name": p.name,
                    "display_name_cn": p.display_name_cn,
                    "unit": p.unit,
                    "default": p.default,
                    "min_value": p.min_value,
                    "max_value": p.max_value,
                    "step": p.step,
                    "fixed": p.fixed,
                    "param_type": p.param_type,
                    "choices": p.choices,
                }
                for p in t.parameters
            ],
            "standard_sizes": t.standard_sizes,
            "notes": t.notes,
            "quality_levels": t.quality_levels,
        })

    return JSONResponse({
        "templates": templates,
        "total": len(templates),
        "categories": CATEGORY_TREE,
    })


@app.get("/api/parts/template/{part_id}")
async def api_parts_template(part_id: str) -> JSONResponse:
    """Get detailed info for a single part template."""
    try:
        from ..knowledge.parts_catalog import get_template
    except ImportError:
        raise HTTPException(status_code=503, detail="Part library not available")

    template = get_template(part_id)
    if template is None:
        raise HTTPException(status_code=404, detail=f"Part template '{part_id}' not found")

    return JSONResponse({
        "id": template.id,
        "name_en": template.name_en,
        "name_cn": template.name_cn,
        "category": template.category,
        "subcategory": template.subcategory,
        "description": template.description,
        "tags": template.tags,
        "material_default": template.material_default,
        "parameters": [
            {
                "name": p.name,
                "display_name_cn": p.display_name_cn,
                "unit": p.unit,
                "default": p.default,
                "min_value": p.min_value,
                "max_value": p.max_value,
                "step": p.step,
                "fixed": p.fixed,
                "param_type": p.param_type,
                "choices": p.choices,
            }
            for p in template.parameters
        ],
        "standard_sizes": template.standard_sizes,
        "notes": template.notes,
        "quality_levels": template.quality_levels,
    })


@app.post("/api/parts/generate")
async def api_parts_generate(payload: dict[str, Any]) -> JSONResponse:
    """Generate a parametric part on the server."""
    part_id = payload.get("part_id", "")
    parameters = payload.get("parameters")
    variant_index = payload.get("variant_index")

    if not part_id:
        raise HTTPException(status_code=400, detail="Missing 'part_id'")

    try:
        from ..knowledge.parts_catalog import get_template, resolve_parameters, format_fc_script
    except ImportError:
        raise HTTPException(status_code=503, detail="Part library not available")

    template = get_template(part_id)
    if template is None:
        raise HTTPException(status_code=404, detail=f"Part template '{part_id}' not found")

    # Resolve parameters
    try:
        if variant_index is not None:
            idx = int(variant_index)
            if idx < 0 or idx >= len(template.standard_sizes):
                raise HTTPException(status_code=400, detail="variant_index out of range")
            parameters = template.standard_sizes[idx]
        resolved = resolve_parameters(template, parameters)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Generate script
    try:
        model_script = format_fc_script(template, resolved)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Script template error: {e}")

    # Build output paths
    ws = _workspace_root()
    parts_dir = ws / "parts_library"
    parts_dir.mkdir(parents=True, exist_ok=True)
    param_desc = "_".join(f"{k}{int(v) if v == int(v) else v}" for k, v in resolved.items())
    safe_name = f"{part_id}_{param_desc}"
    fcstd_path = parts_dir / f"{safe_name}.FCStd"
    stl_path = parts_dir / f"{safe_name}.stl"

    full_script = model_script + f"""
import os
os.makedirs(r'{parts_dir}', exist_ok=True)
doc.saveAs(r'{fcstd_path}')
print(f"Saved: {fcstd_path}")
import Mesh
_export_list = [o for o in doc.Objects if hasattr(o, 'Shape')]
if _export_list:
    Mesh.export(_export_list, r'{stl_path}')
    print(f"STL: {stl_path} ({os.path.getsize(r'{stl_path}'):,} bytes)")
"""

    # Try to run FreeCAD
    freecad = _find_freecad()
    fc_python = None
    if freecad:
        # Find FreeCAD Python from same directory
        fc_python = str(Path(freecad).parent / "python.exe")
        if not Path(fc_python).exists():
            fc_python = None

    # Also try the FreeCAD tool's finder
    if not fc_python:
        try:
            from ..tools.freecad import _find_freecad_python
            fc_python = _find_freecad_python()
        except Exception:
            pass

    if not fc_python:
        raise HTTPException(status_code=503, detail="FreeCAD not available for part generation")

    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(full_script)
        script_path = tf.name

    try:
        proc = subprocess.run(
            [fc_python, "-c", f"exec(open(r'{script_path}').read())"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=120,
        )
        if proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"FreeCAD error: {(proc.stderr or proc.stdout)[:500]}"
            )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="FreeCAD generation timed out")
    finally:
        try:
            Path(script_path).unlink()
        except OSError:
            pass

    result: dict[str, Any] = {
        "part_id": part_id,
        "name": template.name_cn,
        "parameters": resolved,
        "fcstd_path": str(fcstd_path),
    }
    if stl_path.exists():
        result["stl_path"] = str(stl_path)
        result["stl_size_kb"] = stl_path.stat().st_size // 1024
        # Make relative to workspace for web serving
        try:
            result["stl_rel"] = str(stl_path.relative_to(ws)).replace("\\", "/")
        except ValueError:
            pass

    return JSONResponse(result)


@app.get("/api/parts/generated")
async def api_parts_generated() -> JSONResponse:
    """List all generated/imported parts with file existence checks."""
    try:
        from ..tools.part_library import _get_parts_store
    except ImportError:
        raise HTTPException(status_code=503, detail="Part library not available")

    store = _get_parts_store()
    parts = []
    for p in store.list_all():
        entry = p.to_dict()
        entry["fcstd_exists"] = Path(p.fcstd_path).exists() if p.fcstd_path else False
        entry["stl_exists"] = Path(p.stl_path).exists() if p.stl_path else False
        parts.append(entry)

    return JSONResponse({"parts": parts, "total": len(parts)})


@app.delete("/api/parts/generated/{name}")
async def api_parts_generated_delete(name: str) -> JSONResponse:
    """Delete a generated part record by name."""
    try:
        from ..tools.part_library import _get_parts_store
    except ImportError:
        raise HTTPException(status_code=503, detail="Part library not available")

    store = _get_parts_store()
    removed = store.remove(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Part '{name}' not found in store")
    return JSONResponse({"status": "deleted", "name": name})


@app.post("/api/parts/analyze")
async def api_parts_analyze(payload: dict[str, Any]) -> JSONResponse:
    """Run 3D print feasibility analysis on an STL/FCStd file."""
    stl_path = payload.get("stl_path", "")
    orientation = payload.get("orientation", "auto")

    if not stl_path:
        raise HTTPException(status_code=400, detail="Missing 'stl_path'")

    # Validate file exists and is in workspace
    ws = _workspace_root()
    resolved = _resolve_safe(stl_path, ws)
    if resolved is None or not resolved.exists():
        raise HTTPException(status_code=404, detail="File not found or access denied")

    try:
        from ..tools.part_library import _run_print_analysis
        result_str = _run_print_analysis(str(resolved), orientation)
    except ImportError:
        raise HTTPException(status_code=503, detail="Part library not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Try to extract JSON from the result
    analysis_data = {"raw": result_str}
    for line in result_str.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                analysis_data = json.loads(line)
            except json.JSONDecodeError:
                pass
            break

    return JSONResponse({"analysis": analysis_data, "stl_path": str(resolved)})


@app.post("/api/parts/assemble")
async def api_parts_assemble(payload: dict[str, Any]) -> JSONResponse:
    """Assemble multiple parts into a single FreeCAD document."""
    assembly_name = payload.get("assembly_name", "")
    parts = payload.get("parts", [])
    output_format = payload.get("output_format", "fcstd")

    if not assembly_name:
        raise HTTPException(status_code=400, detail="Missing 'assembly_name'")
    if not parts:
        raise HTTPException(status_code=400, detail="Missing 'parts' list")

    # Validate all part files exist
    ws = _workspace_root()
    validated_parts = []
    for p in parts:
        fpath = p.get("file", "")
        if not fpath:
            raise HTTPException(status_code=400, detail="Each part must have a 'file' field")
        resolved = _resolve_safe(fpath, ws)
        if resolved is None:
            resolved = _resolve_safe(fpath, DATA_ROOT)
        if resolved is None or not resolved.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {fpath}")
        validated_parts.append({
            "file": str(resolved),
            "name": p.get("name", Path(fpath).stem),
            "position": p.get("position", [0, 0, 0]),
            "rotation": p.get("rotation", [0, 0, 1, 0]),
        })

    try:
        from ..tools.part_library import PartAssembleTool
    except ImportError:
        raise HTTPException(status_code=503, detail="Part library not available")

    tool = PartAssembleTool()
    result_str = tool.execute(
        assembly_name=assembly_name,
        parts=validated_parts,
        output_format=output_format,
    )

    if result_str.startswith("错误"):
        raise HTTPException(status_code=500, detail=result_str)

    # Find generated files
    parts_dir = ws / "parts_library"
    fcstd_file = parts_dir / f"{assembly_name}.FCStd"
    stl_file = parts_dir / f"{assembly_name}.stl"

    result: dict[str, Any] = {
        "assembly_name": assembly_name,
        "part_count": len(validated_parts),
        "output_format": output_format,
        "result": result_str,
    }
    if fcstd_file.exists():
        result["fcstd_path"] = str(fcstd_file)
        try:
            result["fcstd_rel"] = str(fcstd_file.relative_to(ws)).replace("\\", "/")
        except ValueError:
            pass
    if stl_file.exists():
        result["stl_path"] = str(stl_file)
        result["stl_size_kb"] = stl_file.stat().st_size // 1024

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Routes: slicing
# ---------------------------------------------------------------------------

@app.post("/api/slice")
async def api_slice(payload: dict[str, Any]) -> JSONResponse:
    """Submit a slicing task: STL → G-code."""
    stl_path = payload.get("stl_path", "")
    if not stl_path:
        raise HTTPException(status_code=400, detail="Missing 'stl_path'")

    ws = _workspace_root()
    resolved = _resolve_safe(stl_path, ws)
    if resolved is None:
        resolved = _resolve_safe(stl_path, DATA_ROOT)
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


@app.post("/api/slice/analyze")
async def api_slice_analyze(payload: dict[str, Any]) -> JSONResponse:
    """Analyze an existing G-code file."""
    gcode_path = payload.get("gcode_path", "")
    if not gcode_path:
        raise HTTPException(status_code=400, detail="Missing 'gcode_path'")

    ws = _workspace_root()
    resolved = _resolve_safe(gcode_path, ws)
    if resolved is None:
        resolved = _resolve_safe(gcode_path, DATA_ROOT)
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


@app.post("/api/slice/layers")
async def api_slice_layers(payload: dict[str, Any]) -> JSONResponse:
    """Extract per-layer data from a G-code file."""
    gcode_path = payload.get("gcode_path", "")
    if not gcode_path:
        raise HTTPException(status_code=400, detail="Missing 'gcode_path'")

    ws = _workspace_root()
    resolved = _resolve_safe(gcode_path, ws)
    if resolved is None:
        resolved = _resolve_safe(gcode_path, DATA_ROOT)
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


@app.get("/api/design/hierarchy")
async def api_design_hierarchy() -> JSONResponse:
    """Return subsystem decomposition for the complex robot design."""
    try:
        data = _get_default_robot_data()
        assembly = data["assembly"]
        subsystems = data["subsystems"]

        ss_list = []
        for name, parts in subsystems.items():
            ss_list.append({
                "name": name,
                "parts": parts,
                "part_count": len(parts),
                "status": "completed",
            })

        return JSONResponse({
            "assembly_name": assembly.name,
            "total_parts": len(assembly.parts),
            "total_joints": len(assembly.joints),
            "subsystems": ss_list,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/design/assembly-tree")
async def api_design_assembly_tree() -> JSONResponse:
    """Return the assembly hierarchy tree (parent -> children) from joints."""
    try:
        data = _get_default_robot_data()
        assembly = data["assembly"]

        # Build adjacency: parent -> list of (child, joint_type)
        children_map: dict[str, list[dict[str, Any]]] = {}
        all_children: set[str] = set()
        for j in assembly.joints:
            children_map.setdefault(j.parent, []).append({
                "name": j.child,
                "joint_type": j.type,
            })
            all_children.add(j.child)

        # Root = part that is never a child
        roots = [p.name for p in assembly.parts if p.name not in all_children]
        root_name = roots[0] if roots else assembly.parts[0].name

        def _build_tree(name: str) -> dict[str, Any]:
            node: dict[str, Any] = {"name": name}
            kids = children_map.get(name, [])
            if kids:
                node["children"] = [
                    {**_build_tree(k["name"]), "joint_type": k["joint_type"]}
                    for k in kids
                ]
            return node

        tree = _build_tree(root_name)
        return JSONResponse({"tree": tree})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/design/stability")
async def api_design_stability() -> JSONResponse:
    """Return stability analysis for the complex robot design."""
    try:
        data = _get_default_robot_data()
        assembly = data["assembly"]
        positions = data["positions"]
        mass_result = data["mass_result"]

        from ..tools.stability import (
            compute_support_polygon,
            compute_static_stability,
            check_tip_over_risk,
        )

        contacts = []
        for s in ["fl", "fr", "rl", "rr"]:
            pos = positions.get(f"wheel_{s}", {}).get("position", [0, 0, 0])
            contacts.append([pos[0], pos[1], pos[2]])

        com = list(mass_result["center_of_mass_mm"])
        polygon = compute_support_polygon(contacts)
        poly_2d = [[p[0], p[1]] for p in (polygon if len(polygon) >= 3 else contacts)]
        static_stab = compute_static_stability(com, poly_2d)
        tip_risk = check_tip_over_risk(
            com=com, contact_points=contacts,
            mass_kg=mass_result["total_mass_kg"],
        )

        return JSONResponse({
            "total_mass_kg": round(mass_result["total_mass_kg"], 3),
            "center_of_mass_mm": com,
            "support_polygon": {
                "vertices": len(polygon),
                "area_mm2": round(static_stab.get("polygon_area_mm2", 0), 1),
            },
            "static_stability": {
                "stable": static_stab.get("stable", False),
                "margin_mm": round(static_stab.get("margin_mm", 0), 1),
            },
            "tip_over_risk": {
                "risk_level": tip_risk.get("risk_level", "unknown"),
                "min_stability_margin_mm": round(
                    tip_risk.get("min_stability_margin_mm", 0), 1
                ),
            },
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/design/power-budget")
async def api_design_power_budget() -> JSONResponse:
    """Return power budget analysis for the complex robot design."""
    try:
        from ..tools.power_budget import PowerBudgetCalculator

        calc = PowerBudgetCalculator("MobileRobotDualArm")
        calc.add_motor("Drive Motor", "TT_MOTOR", duty_cycle=0.5, quantity=4)
        calc.add_servo("Arm Servos", "MG996R", duty_cycle=0.3, quantity=6)
        calc.add_controller("IPC", tdp_w=15.0)
        calc.add_sensor_load("Sensors", power_w=2.0, quantity=3)

        consumers = [
            {
                "name": c.name,
                "category": c.category,
                "peak_power_w": c.peak_power_w,
                "avg_power_w": c.avg_power_w,
                "duty_cycle": c.duty_cycle,
                "quantity": c.quantity,
            }
            for c in calc.consumers
        ]

        batt_recs = calc.recommend_battery(runtime_target_h=0.5)
        battery_recommendations = [
            {
                "name": r.get("battery").name if hasattr(r.get("battery"), "name") else str(r.get("battery")),
                "runtime_h": r.get("runtime_h"),
                "margin_pct": r.get("margin_pct"),
            }
            for r in batt_recs[:3]
        ]

        return JSONResponse({
            "consumers": consumers,
            "peak_power_w": round(calc.compute_total_peak(), 1),
            "avg_power_w": round(calc.compute_total_avg(), 1),
            "battery_recommendations": battery_recommendations,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    global _last_snapshot
    await websocket.accept()
    _websockets.append(websocket)
    try:
        # Send a full snapshot on connect; reset delta baseline for this client.
        snapshot = _snapshot_state()
        await _send_json(websocket, snapshot)
        # Keep connection alive, ignore inbound text
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if websocket in _websockets:
            _websockets.remove(websocket)
        # If this was the last client, reset the delta baseline
        if not _websockets:
            _last_snapshot = None


# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def run_server(host: str = "0.0.0.0", port: int = 8765) -> None:
    """Run the web server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)
