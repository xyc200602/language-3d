"""Tests for the web run-viewer animation/position routes.

Covers ``GET /api/runs/{case}/{ts}/positions`` (real solver, no mujoco)
and ``POST /api/runs/{case}/{ts}/animate`` (record_motion mocked so the
test stays fast and mujoco-free).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lang3d.web.app import app

client = pytest.importorskip("fastapi.testclient").TestClient(app)

# A real completed run shipped under data/runs/ — used for /positions.
_REAL_CASE = "4dof_arm"
_REAL_TS = "20260624_172515"


def test_positions_returns_per_part_placement() -> None:
    """/positions yields a world position (mm) for each assembled part."""
    r = client.get(f"/api/runs/{_REAL_CASE}/{_REAL_TS}/positions")
    assert r.status_code == 200
    data = r.json()
    assert "positions" in data
    pos = data["positions"]
    assert len(pos) >= 6  # the 4dof arm has 11 parts
    # base_plate sits at the origin.
    assert pos["base_plate"]["position"] == [0.0, 0.0, 0.0]
    # Every entry has a 3-element position list.
    for name, pose in pos.items():
        assert len(pose["position"]) == 3


def test_positions_missing_run_404() -> None:
    """An unknown case/timestamp returns 404, not a 500."""
    r = client.get("/api/runs/nope/00000000_000000/positions")
    assert r.status_code == 404


def test_animate_returns_motion_frames() -> None:
    """/animate forwards the record_motion frame series to the client."""
    fake = {
        "ok": True,
        "bodies": ["link_a", "link_b"],
        "fps": 30,
        "duration_sec": 1.0,
        "frames": [
            {"t": 0.0, "poses": [[0, 0, 0, 1, 0, 0, 0]] * 2},
            {"t": 0.03, "poses": [[0.01, 0, 0, 1, 0, 0, 0]] * 2},
        ],
    }
    with patch("lang3d.tools.sim_mujoco.record_motion", return_value=fake):
        r = client.post(f"/api/runs/{_REAL_CASE}/{_REAL_TS}/animate")
    assert r.status_code == 200
    data = r.json()
    assert data["bodies"] == ["link_a", "link_b"]
    assert len(data["frames"]) == 2


def test_animate_missing_urdf_404() -> None:
    """A run without an engineering_package/urdf.xml returns 404."""
    with patch("lang3d.tools.sim_mujoco.record_motion") as mock_rec:
        r = client.post("/api/runs/nope/00000000_000000/animate")
    assert r.status_code == 404
    mock_rec.assert_not_called()


def test_animate_propagates_record_failure() -> None:
    """When record_motion reports ok=False, the route returns 500."""
    with patch("lang3d.tools.sim_mujoco.record_motion",
               return_value={"ok": False, "error": "boom", "frames": []}):
        r = client.post(f"/api/runs/{_REAL_CASE}/{_REAL_TS}/animate")
    assert r.status_code == 500
    assert r.json()["error"] == "boom"
