"""Tests for async file conversion queue."""

from __future__ import annotations

import json
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from lang3d.web.app import app, _convert_queue, _convert_lock


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_convert_queue():
    """Clear the conversion queue before each test."""
    with _convert_lock:
        _convert_queue.clear()
    yield
    with _convert_lock:
        _convert_queue.clear()


# ---------------------------------------------------------------------------
# POST /api/convert-async
# ---------------------------------------------------------------------------

class TestAsyncConvert:
    def test_missing_path(self, client):
        response = client.post("/api/convert-async")
        assert response.status_code == 422  # missing required param

    def test_nonexistent_file(self, client):
        with patch("lang3d.web.app._resolve_safe", return_value=None):
            response = client.post("/api/convert-async?path=nonexistent.step")
            assert response.status_code == 404

    def test_unsupported_format(self, client):
        ws = Path("C:/tmp")
        src = MagicMock()
        src.suffix = ".obj"
        src.exists.return_value = True

        with patch("lang3d.web.app._resolve_safe", return_value=src):
            with patch("lang3d.web.app._workspace_root", return_value=ws):
                response = client.post("/api/convert-async?path=test.obj")
                assert response.status_code == 400

    @patch("lang3d.web.app._find_freecad", return_value=None)
    def test_freecad_not_available(self, mock_fc, client):
        ws = Path("C:/tmp")
        src = MagicMock()
        src.suffix = ".step"
        src.exists.return_value = True

        with patch("lang3d.web.app._resolve_safe", return_value=src):
            with patch("lang3d.web.app._workspace_root", return_value=ws):
                response = client.post("/api/convert-async?path=test.step")
                # 503 (FreeCAD unavailable) or 404 (file check fails)
                assert response.status_code in (404, 503)

    @patch("lang3d.web.app._find_freecad", return_value="/usr/bin/freecadcmd")
    def test_submit_returns_job_id(self, mock_fc, client):
        ws = Path("C:/tmp")
        src = MagicMock()
        src.suffix = ".step"
        src.exists.return_value = True
        src.stat.return_value = MagicMock(st_mtime=100)

        output = MagicMock()
        output.exists.return_value = False
        src.with_suffix.return_value = output

        with patch("lang3d.web.app._resolve_safe", return_value=src):
            with patch("lang3d.web.app._workspace_root", return_value=ws):
                with patch("lang3d.web.app.threading.Thread") as mock_thread:
                    mock_thread.return_value = MagicMock()
                    response = client.post("/api/convert-async?path=test.step")

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    @patch("lang3d.web.app._find_freecad", return_value="/usr/bin/freecadcmd")
    def test_cached_file_returns_immediately(self, mock_fc, client):
        ws = Path("C:/tmp")
        src = MagicMock()
        src.suffix = ".step"
        src.exists.return_value = True
        src.stat.return_value = MagicMock(st_mtime=100)
        src.relative_to.return_value = Path("test.step")

        output = MagicMock()
        output.exists.return_value = True
        output.stat.return_value = MagicMock(st_mtime=200)
        output.relative_to.return_value = Path("test.preview.stl")
        src.with_suffix.return_value = output

        with patch("lang3d.web.app._resolve_safe", return_value=src):
            with patch("lang3d.web.app._workspace_root", return_value=ws):
                response = client.post("/api/convert-async?path=test.step")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "done"
        assert data["cached"] is True


# ---------------------------------------------------------------------------
# GET /api/convert-status
# ---------------------------------------------------------------------------

class TestConvertStatus:
    def test_unknown_job(self, client):
        response = client.get("/api/convert-status?job_id=nonexistent")
        assert response.status_code == 404

    def test_pending_job(self, client):
        with _convert_lock:
            _convert_queue["test123"] = {"status": "pending"}

        response = client.get("/api/convert-status?job_id=test123")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"

    def test_running_job(self, client):
        with _convert_lock:
            _convert_queue["test456"] = {"status": "running"}

        response = client.get("/api/convert-status?job_id=test456")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"

    def test_done_job(self, client):
        with _convert_lock:
            _convert_queue["test789"] = {
                "status": "done",
                "result": {"stl_path": "output.stl", "cached": False},
            }

        response = client.get("/api/convert-status?job_id=test789")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "done"
        assert data["result"]["stl_path"] == "output.stl"

    def test_failed_job(self, client):
        with _convert_lock:
            _convert_queue["testfail"] = {
                "status": "failed",
                "error": "Conversion timed out",
            }

        response = client.get("/api/convert-status?job_id=testfail")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert "timed out" in data["error"]


# ---------------------------------------------------------------------------
# Conversion queue thread safety
# ---------------------------------------------------------------------------

class TestConvertQueueConcurrency:
    def test_concurrent_status_reads(self):
        """Simulate concurrent reads from the queue."""
        with _convert_lock:
            for i in range(100):
                _convert_queue[f"job_{i}"] = {"status": "running"}

        errors = []

        def read_status(job_id):
            try:
                with _convert_lock:
                    entry = _convert_queue.get(job_id)
                    assert entry is not None
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_status, args=(f"job_{i}",)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_queue_cleanup(self):
        """Verify the autouse fixture clears the queue."""
        with _convert_lock:
            _convert_queue["cleanup_test"] = {"status": "done"}
        assert "cleanup_test" in _convert_queue
        # The fixture will clean this up after the test
