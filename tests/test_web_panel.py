"""Tests for web monitoring panel."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from lang3d.web.app import (
    add_log,
    add_tool_call,
    add_vlm_result,
    app,
    run_server,
    set_thinking,
    update_agent_state,
)


class TestWebAppState:
    def test_initial_state(self):
        from lang3d.web import app as web_app
        state = web_app._agent_state
        assert state["status"] == "idle"
        assert state["logs"] == []
        assert state["tool_calls"] == []
        assert state["vlm_results"] == []

    def test_update_agent_state(self):
        update_agent_state(status="running", plan={"steps": []})
        from lang3d.web import app as web_app
        assert web_app._agent_state["status"] == "running"
        # Reset
        web_app._agent_state["status"] = "idle"

    def test_add_log(self):
        from lang3d.web import app as web_app
        initial_count = len(web_app._agent_state["logs"])
        add_log("Test message", level="info")
        assert len(web_app._agent_state["logs"]) == initial_count + 1
        # Reset
        web_app._agent_state["logs"] = []

    def test_add_log_with_timestamp(self):
        from lang3d.web import app as web_app
        add_log("Timestamp test", level="tool")
        last_log = web_app._agent_state["logs"][-1]
        assert "time" in last_log
        assert last_log["message"] == "Timestamp test"
        # Reset
        web_app._agent_state["logs"] = []

    def test_add_tool_call(self):
        from lang3d.web import app as web_app
        add_tool_call("fc_batch", {"operations": []}, result="OK")
        assert len(web_app._agent_state["tool_calls"]) >= 1
        tc = web_app._agent_state["tool_calls"][-1]
        assert tc["name"] == "fc_batch"
        assert tc["result_preview"] == "OK"
        # Reset
        web_app._agent_state["tool_calls"] = []

    def test_add_vlm_result(self):
        from lang3d.web import app as web_app
        add_vlm_result(
            tool="cad_verify",
            prompt="Expected: a cube",
            result="MATCH: True",
            image_path="/test/img.png",
        )
        assert len(web_app._agent_state["vlm_results"]) >= 1
        vr = web_app._agent_state["vlm_results"][-1]
        assert vr["tool"] == "cad_verify"
        assert "MATCH: True" in vr["result"]
        # Reset
        web_app._agent_state["vlm_results"] = []

    def test_set_thinking(self):
        from lang3d.web import app as web_app
        set_thinking("Analyzing task...")
        assert web_app._agent_state["thinking"] == "Analyzing task..."
        # Reset
        web_app._agent_state["thinking"] = ""

    def test_log_limit(self):
        from lang3d.web import app as web_app
        web_app._agent_state["logs"] = []
        for i in range(250):
            add_log(f"Log {i}")
        assert len(web_app._agent_state["logs"]) == 200
        # Reset
        web_app._agent_state["logs"] = []

    def test_tool_call_limit(self):
        from lang3d.web import app as web_app
        web_app._agent_state["tool_calls"] = []
        for i in range(150):
            add_tool_call("tool", {"i": i}, "")
        assert len(web_app._agent_state["tool_calls"]) == 100
        # Reset
        web_app._agent_state["tool_calls"] = []

    def test_vlm_result_limit(self):
        from lang3d.web import app as web_app
        web_app._agent_state["vlm_results"] = []
        for i in range(60):
            add_vlm_result("vlm_analyze", f"prompt {i}", f"result {i}")
        assert len(web_app._agent_state["vlm_results"]) == 50
        # Reset
        web_app._agent_state["vlm_results"] = []


class TestWebAPIEndpoints:
    def test_app_exists(self):
        assert app is not None
        assert app.title == "Language-3D Agent Monitor"

    def test_status_endpoint(self):
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "logs" in data
        assert "tool_calls" in data
        assert "vlm_results" in data

    def test_screenshots_endpoint(self):
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/api/screenshots")
        assert response.status_code == 200
        data = response.json()
        assert "screenshots" in data

    def test_tool_calls_endpoint(self):
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/api/tool-calls")
        assert response.status_code == 200
        data = response.json()
        assert "tool_calls" in data

    def test_vlm_results_endpoint(self):
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/api/vlm-results")
        assert response.status_code == 200
        data = response.json()
        assert "vlm_results" in data

    def test_gallery_endpoint(self):
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/api/screenshot-gallery")
        assert response.status_code == 200
        data = response.json()
        assert "gallery" in data
        assert "total" in data

    def test_index_endpoint(self):
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert "Language-3D" in response.text


class TestAgentWebIntegration:
    def test_connect_web_panel(self):
        from lang3d.agent.core import Agent
        from lang3d.web.app import add_tool_call, add_vlm_result

        agent = Agent.__new__(Agent)
        agent._on_tool_call = None
        agent._on_tool_result = None
        agent._on_thinking = None
        agent.state = MagicMock()
        agent.state.tool_history = []

        agent.connect_web_panel()
        assert agent._on_tool_call is not None
        assert agent._on_tool_result is not None
        assert agent._on_thinking is not None
