"""FastAPI web monitoring panel."""

from __future__ import annotations

import asyncio
import base64
import json
import time as _time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Language-3D Agent Monitor")

# In-memory state (updated by the agent)
_agent_state: dict[str, Any] = {
    "status": "idle",
    "plan": None,
    "logs": [],
    "screenshots": [],
    "tool_calls": [],
    "vlm_results": [],
    "thinking": "",
    "sub_agents": [],
    "dag": None,
}
_websockets: list[WebSocket] = []


def update_agent_state(**kwargs: Any) -> None:
    """Update the shared agent state and broadcast to WebSocket clients."""
    _agent_state.update(kwargs)
    _agent_state["timestamp"] = _time.strftime("%H:%M:%S")
    broadcast_state()


def add_log(message: str, level: str = "info") -> None:
    """Add a log entry."""
    _agent_state["logs"] = _agent_state.get("logs", [])
    _agent_state["logs"].append({
        "message": message,
        "level": level,
        "time": _time.strftime("%H:%M:%S"),
    })
    # Keep last 200 entries
    _agent_state["logs"] = _agent_state["logs"][-200:]
    broadcast_state()


def add_tool_call(name: str, args: dict, result: str = "") -> None:
    """Record a tool call for the timeline."""
    _agent_state["tool_calls"] = _agent_state.get("tool_calls", [])
    entry = {
        "name": name,
        "args": args,
        "result_preview": result[:200] if result else "",
        "time": _time.strftime("%H:%M:%S"),
        "timestamp": _time.time(),
    }
    _agent_state["tool_calls"].append(entry)
    # Keep last 100 tool calls
    _agent_state["tool_calls"] = _agent_state["tool_calls"][-100:]

    # Add to logs as well
    arg_preview = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3])
    add_log(f"Tool: {name}({arg_preview})", level="tool")
    broadcast_state()


def add_vlm_result(tool: str, prompt: str, result: str, image_path: str = "") -> None:
    """Record a VLM analysis result."""
    _agent_state["vlm_results"] = _agent_state.get("vlm_results", [])
    entry = {
        "tool": tool,
        "prompt": prompt[:200],
        "result": result[:500],
        "image_path": image_path,
        "time": _time.strftime("%H:%M:%S"),
    }
    _agent_state["vlm_results"].append(entry)
    # Keep last 50 VLM results
    _agent_state["vlm_results"] = _agent_state["vlm_results"][-50:]
    broadcast_state()


def set_thinking(text: str) -> None:
    """Update the current agent thinking text."""
    _agent_state["thinking"] = text
    broadcast_state()


def update_sub_agent(agent_id: str, status: str, step: str = "") -> None:
    """Update or add a sub-agent status entry."""
    agents = _agent_state.get("sub_agents", [])
    # Find existing or create new
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


async def broadcast_state_async() -> None:
    """Broadcast current state to all connected WebSocket clients."""
    data = json.dumps(_agent_state, ensure_ascii=False)
    for ws in _websockets[:]:
        try:
            await ws.send_text(data)
        except Exception:
            _websockets.remove(ws)


def broadcast_state() -> None:
    """Synchronous wrapper for broadcasting."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(broadcast_state_async())
    except RuntimeError:
        pass


@app.get("/")
async def index() -> HTMLResponse:
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Language-3D Agent Monitor</h1><p>Static files not found</p>")


@app.get("/api/status")
async def get_status() -> JSONResponse:
    return JSONResponse(_agent_state)


@app.get("/api/screenshots")
async def get_screenshots() -> JSONResponse:
    screenshots = _agent_state.get("screenshots", [])
    return JSONResponse({"screenshots": screenshots})


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
    """Return active sub-agents and their status."""
    return JSONResponse({"agents": _agent_state.get("sub_agents", [])})


@app.get("/api/dag")
async def get_dag() -> JSONResponse:
    """Return DAG visualization data."""
    return JSONResponse({"dag": _agent_state.get("dag")})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    _websockets.append(websocket)
    try:
        # Send current state on connect
        await websocket.send_text(json.dumps(_agent_state, ensure_ascii=False))
        # Keep connection alive
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _websockets.remove(websocket)


# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def run_server(host: str = "0.0.0.0", port: int = 8765) -> None:
    """Run the web server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)
