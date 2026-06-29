"""FastAPI web monitoring panel."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import subprocess
import tempfile
import threading
import time as _time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Language-3D Agent Monitor")

# Optional API key authentication middleware
_API_KEY: str | None = os.environ.get("LANG3D_API_KEY")


@app.middleware("http")
async def api_key_middleware(request, call_next):
    """Require API key for ALL endpoints when LANG3D_API_KEY is set."""
    if _API_KEY is None:
        return await call_next(request)
    # Allow static files and index page without auth
    if request.url.path in ("/", "/static", "/index.html"):
        return await call_next(request)
    key = request.headers.get("X-API-Key", "")
    if key != _API_KEY:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
    return await call_next(request)

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
_state_lock = threading.RLock()

# Simple rate limiter for /api/run-task
_rate_limit_timestamps: list[float] = []

# Previous snapshot used for delta WebSocket updates.
_last_snapshot: dict[str, Any] | None = None

# Captured event loop for cross-thread broadcast
_event_loop: asyncio.AbstractEventLoop | None = None


def set_event_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    global _event_loop
    _event_loop = loop


@app.on_event("startup")
async def _capture_event_loop():
    set_event_loop(asyncio.get_running_loop())


@app.on_event("shutdown")
async def _release_event_loop():
    set_event_loop(None)


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
        except Exception as e:  # agent state access — log, don't crash the UI
            logger.debug("workspace_root lookup failed: %s", e)
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
    with _state_lock:
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
    except Exception as e:  # agent.state shape varies across versions
        logger.debug("register_agent: session_id read failed: %s", e)
    try:
        ws = getattr(agent.state, "workspace", None)
        if ws:
            add_log(f"Agent registered (workspace={Path(ws)})", level="info")
    except Exception as e:
        logger.debug("register_agent: workspace read failed: %s", e)


def _run_task_background(task: str, mode: str = "run") -> None:
    """Background-thread task executor."""
    global _current_task
    _task_stop_flag.clear()
    with _state_lock:
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

        if _task_stop_flag.is_set():
            raise RuntimeError("Task cancelled before start")

        if mode == "direct":
            result = _agent_instance.run_task(task, use_planning=False)
        else:
            result = _agent_instance.run_task(task, use_planning=True)

        if _task_stop_flag.is_set():
            add_log("Task was stopped by user", level="warning")
        with _state_lock:
            if _current_task:
                _current_task["status"] = "complete"
                _current_task["result"] = result[:500] if isinstance(result, str) else str(result)[:500]
        update_agent_state(status="complete")
        with _state_lock:
            add_log(f"Task completed: {_current_task.get('result', '')[:120]}", level="success")
    except Exception as e:
        with _state_lock:
            if _current_task:
                _current_task["status"] = "error"
                _current_task["error"] = str(e)
        update_agent_state(status="error")
        add_log(f"Task error: {e}", level="error")
    finally:
        with _state_lock:
            if _current_task:
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
    except Exception as e:  # client disconnected mid-broadcast — expected, log debug
        logger.debug("websocket send failed (client gone?): %s", e)


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
    global _event_loop
    if _event_loop is not None and _event_loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast_state_async(), _event_loop)
    else:
        try:
            loop = asyncio.get_running_loop()
            asyncio.ensure_future(broadcast_state_async())
        except RuntimeError:
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


@app.get("/simulate")
async def simulate_page() -> HTMLResponse:
    """Run viewer / interactive modification page (added 2026-06-18)."""
    html_path = Path(__file__).parent / "static" / "simulate.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>simulate.html not found</h1>", status_code=404)


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


# ---------------------------------------------------------------------------
# Routes: run inspection & interactive modification (added 2026-06-18)
#
# These endpoints expose the new ``data/runs/<case>/<ts>/`` layout and the
# IterativeSession API to the web UI.
# ---------------------------------------------------------------------------


@app.get("/api/runs")
async def list_runs() -> JSONResponse:
    """List all run directories under ``data/runs/``.

    Returns ``{"runs": [{"case": "4dof_arm", "timestamp": "...",
                         "path": "...", "assembly": {...} | null}, ...]}``
    """
    runs_root = DATA_ROOT / "runs"
    out: list[dict[str, Any]] = []
    if not runs_root.is_dir():
        return JSONResponse({"runs": [], "runs_root": str(runs_root)})

    for case_dir in sorted(runs_root.iterdir()):
        if not case_dir.is_dir():
            continue
        for ts_dir in sorted(case_dir.iterdir()):
            if not ts_dir.is_dir():
                continue
            asm_path = ts_dir / "assembly.json"
            entry: dict[str, Any] = {
                "case": case_dir.name,
                "timestamp": ts_dir.name,
                "path": str(ts_dir),
                "has_assembly": asm_path.exists(),
            }
            if asm_path.exists():
                try:
                    asm = json.loads(asm_path.read_text(encoding="utf-8"))
                    entry["assembly"] = {
                        "name": asm.get("name", ""),
                        "parts": len(asm.get("parts", [])),
                        "joints": len(asm.get("joints", [])),
                        "description": asm.get("description", ""),
                    }
                except (OSError, ValueError):
                    pass
            out.append(entry)
    return JSONResponse({"runs": out, "runs_root": str(runs_root)})


@app.get("/api/runs/{case}/{ts}")
async def get_run(case: str, ts: str) -> JSONResponse:
    """Return the assembly JSON for a specific run."""
    run_dir = DATA_ROOT / "runs" / case / ts
    asm_path = run_dir / "assembly.json"
    if not asm_path.exists():
        return JSONResponse({"error": f"Run not found: {case}/{ts}"},
                            status_code=404)
    try:
        asm = json.loads(asm_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return JSONResponse({"error": f"Failed to read assembly: {e}"},
                            status_code=500)
    return JSONResponse({
        "case": case, "timestamp": ts, "path": str(run_dir),
        "assembly": asm,
    })


@app.get("/api/runs/{case}/{ts}/stls")
async def list_stls(case: str, ts: str) -> JSONResponse:
    """List STL files in a run's engineering_package/stl_parts/."""
    run_dir = DATA_ROOT / "runs" / case / ts
    stl_dir = run_dir / "engineering_package" / "stl_parts"
    if not stl_dir.is_dir():
        return JSONResponse({"stls": [], "stl_dir": str(stl_dir)})
    stls = []
    for f in sorted(stl_dir.iterdir()):
        if f.suffix.lower() == ".stl":
            stls.append({
                "name": f.stem,
                "filename": f.name,
                "size_bytes": f.stat().st_size,
            })
    return JSONResponse({"stls": stls, "stl_dir": str(stl_dir)})


@app.post("/api/runs/{case}/{ts}/modify")
async def modify_run(case: str, ts: str, request: Request) -> JSONResponse:
    """Apply a natural-language modification request to a run in-place.

    Body: ``{"request": "把夹爪加长50%"}``
    Returns: ``{"scope": ..., "intent": ..., "diff": {...}, "applied": bool}``
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    request_text = (body or {}).get("request", "").strip()
    if not request_text:
        return JSONResponse({"error": "Missing 'request' field"},
                            status_code=400)

    run_dir = DATA_ROOT / "runs" / case / ts
    if not (run_dir / "assembly.json").exists():
        return JSONResponse({"error": f"Run not found: {case}/{ts}"},
                            status_code=404)

    try:
        from ..interactive import IterativeSession
        session = IterativeSession(str(run_dir))
        result = session.apply(request_text)
        session.save()  # in-place
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/runs/{case}/{ts}/simulate")
async def simulate_run(case: str, ts: str) -> JSONResponse:
    """Run headless MuJoCo simulation on the run's URDF.

    Returns a JSON report with load/physics/joint-test outcomes.  Use the
    CLI's ``/iter <folder>`` + ``/sim`` for the interactive viewer.
    """
    run_dir = DATA_ROOT / "runs" / case / ts
    urdf = run_dir / "engineering_package" / "urdf.xml"
    if not urdf.exists():
        return JSONResponse({"error": f"URDF not found: {urdf}"},
                            status_code=404)
    try:
        from ..tools.sim_mujoco import SimMujocoTool
        report_text = SimMujocoTool().execute(
            urdf_path=str(urdf),
            mode="validate",
            duration_sec=1.0,
            interactive=False,  # never block the web server
        )
        return JSONResponse({"report": report_text, "urdf": str(urdf)})
    except ImportError as e:
        return JSONResponse({"error": f"mujoco not installed: {e}"},
                            status_code=503)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/runs/{case}/{ts}/positions")
async def run_positions(case: str, ts: str) -> JSONResponse:
    """Return per-part world positions (mm) for the home pose.

    Loads ``assembly.json`` → ``Assembly`` → ``AssemblySolver.solve()`` so
    the web 3D viewer can place each STL at its solved location instead of
    stacking every part at the origin.  Output::

        {"positions": {part_name: {"position": [x,y,z], "rotation": [...]}}}
    """
    run_dir = DATA_ROOT / "runs" / case / ts

    # Fast path: pre-solved positions.json (written by pipeline.run_export)
    pos_file = run_dir / "positions.json"
    if pos_file.exists():
        try:
            return JSONResponse({"positions": json.loads(pos_file.read_text("utf-8"))})
        except Exception:
            pass  # fall through to assembly.json solve

    # Fallback: assembly.json → solve
    asm_file = run_dir / "assembly.json"
    if not asm_file.exists():
        return JSONResponse({"error": "Run not found"}, status_code=404)
    try:
        from ..tools.assembly_generator import _parse_assembly_json
        from ..tools.assembly_solver import AssemblySolver

        assembly = _parse_assembly_json(asm_file.read_text(encoding="utf-8"))
        positions = AssemblySolver(assembly).solve()
        # Trim to JSON-serialisable plain lists.
        out: dict[str, Any] = {}
        for name, pose in positions.items():
            out[name] = {
                "position": list(pose.get("position", (0.0, 0.0, 0.0))),
                "rotation": list(pose.get("rotation", (0.0, 0.0, 0.0, 0.0))),
            }
        return JSONResponse({"positions": out})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/runs/{case}/{ts}/animate")
async def animate_run(case: str, ts: str) -> JSONResponse:
    """Record a short physics rollout for browser playback.

    Returns JOINT ANGLE sequences per frame (not body poses).  The
    frontend calls /fk_positions per frame to forward-kinematics every
    part's correct world position from the joint angles — so ALL parts
    (including fixed-joint-merged links) move correctly along their DOF.
    """
    run_dir = DATA_ROOT / "runs" / case / ts
    urdf = run_dir / "engineering_package" / "urdf.xml"
    if not urdf.exists():
        return JSONResponse({"error": f"URDF not found: {urdf}"},
                            status_code=404)
    try:
        from ..tools.sim_mujoco import record_joint_motion
        result = record_joint_motion(str(urdf), duration_sec=3.0, fps=15)
        if not result.get("ok"):
            return JSONResponse({"error": result.get("error", "animation failed")},
                                status_code=500)
        return JSONResponse(result)
    except ImportError as e:
        return JSONResponse({"error": f"mujoco not installed: {e}"},
                            status_code=503)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/runs/{case}/{ts}/fk_positions")
async def fk_positions(case: str, ts: str, request: Request) -> JSONResponse:
    """Forward-kinematics: given joint angles, return all part positions.

    The web animation calls this for each frame with the joint angles from
    /animate.  This solves the assembly with those angles and returns the
    world position + rotation of EVERY part (all 13, not just 6 MuJoCo
    bodies).  This is the correct way to animate — every part follows the
    kinematic chain exactly.
    """
    import json as _json
    body = await request.json()
    angles = body.get("angles", {})

    run_dir = DATA_ROOT / "runs" / case / ts
    asm_file = run_dir / "assembly.json"
    if not asm_file.exists():
        return JSONResponse({"error": "Run not found"}, status_code=404)

    try:
        from ..tools.assembly_generator import _parse_assembly_json
        from ..tools.assembly_solver import AssemblySolver

        assembly = _parse_assembly_json(asm_file.read_text("utf-8"))
        # Merge provided angles with defaults.
        merged = dict(assembly.default_angles)
        # MuJoCo joint names may differ from assembly part names; match
        # by looking up the child part name embedded in the joint name.
        for jname, jval in angles.items():
            # MuJoCo joint names like "base_plate_to_base_yaw_servo"
            # → the child is "base_yaw_servo" (last segment after "to_").
            if "_to_" in jname:
                child = jname.rsplit("_to_", 1)[-1]
                merged[child] = float(jval)
            else:
                merged[jname] = float(jval)

        positions = AssemblySolver(assembly).solve(joint_angles=merged)
        out: dict[str, Any] = {}
        for name, pose in positions.items():
            out[name] = {
                "position": list(pose.get("position", (0.0, 0.0, 0.0))),
                "rotation": list(pose.get("rotation", (0.0, 0.0, 1.0, 0.0))),
            }
        return JSONResponse({"positions": out})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/runs/{case}/{ts}/render")
async def render_run(case: str, ts: str) -> JSONResponse:
    """Trigger a fresh VTK render of the assembly.

    Registered as a POST route — previously this function existed but had no
    ``@app.post`` decorator, so the frontend's "Re-render" button
    (simulate.html) hit a 404 and silently failed.
    """
    run_dir = DATA_ROOT / "runs" / case / ts
    if not (run_dir / "assembly.json").exists():
        return JSONResponse({"error": "Run not found"}, status_code=404)
    try:
        from ..interactive import IterativeSession
        session = IterativeSession(str(run_dir))
        out = session.render()
        if out is None:
            return JSONResponse({"error": "Renderer unavailable"},
                                status_code=503)
        return JSONResponse({"render_dir": str(out)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


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
    global _task_thread, _rate_limit_timestamps
    # Rate limit: max 5 task submissions per 60 seconds
    now = _time.time()
    _rate_limit_timestamps = [t for t in _rate_limit_timestamps if now - t < 60]
    if len(_rate_limit_timestamps) >= 5:
        raise HTTPException(status_code=429, detail="Too many task submissions. Wait 60 seconds.")
    _rate_limit_timestamps.append(now)

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
                "type": _get_classify_file()(child),
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


# ---------------------------------------------------------------------------
# Route modules (P1-1 split) — registered via APIRouter.
# ---------------------------------------------------------------------------
from .routes import convert as _convert_routes
from .routes import parts as _parts_routes
from .routes import slicing as _slicing_routes
from .routes import design as _design_routes
# _classify_file is used by /api/browse (kept in app.py) but defined in
# routes/convert.py. Deferred import to avoid circular dependency
# (convert.py imports DATA_ROOT from app.py).
_classify_file = None
def _get_classify_file():
    global _classify_file
    if _classify_file is None:
        from .routes.convert import _classify_file as _cf
        _classify_file = _cf
    return _classify_file

app.include_router(_convert_routes.router)
app.include_router(_parts_routes.router)
app.include_router(_slicing_routes.router)
app.include_router(_design_routes.router)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    global _last_snapshot
    # WebSocket authentication: require api_key query param when API key is configured.
    # Browsers cannot set custom headers on WebSocket handshakes, so query param is
    # the standard mechanism. Note: this may leak into server/proxy access logs.
    if _API_KEY is not None:
        ws_key = websocket.query_params.get("api_key", "")
        if ws_key != _API_KEY:
            await websocket.close(code=1008, reason="Invalid or missing API key")
            return
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
        pass  # normal client disconnect — no action needed
    except Exception as e:  # unexpected socket error — log, then clean up
        logger.debug("websocket listener exited: %s", e)
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


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the web server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)
