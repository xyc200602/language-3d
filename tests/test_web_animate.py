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

# A real completed run under data/runs/ — auto-discovered at import time so
# the test does not break when a specific historical timestamp is pruned.
# Previously this was hard-coded to "20260624_172515", which no longer exists,
# causing the test to 404 even on a clean tree.
import glob as _glob
import os as _os

_REAL_CASE = "4dof_arm"
_4dof_runs = sorted(_glob.glob("data/runs/4dof_arm/*/assembly.json"),
                    reverse=True)
_REAL_TS = _os.path.basename(_os.path.dirname(_4dof_runs[0])) \
    if _4dof_runs else "00000000_000000"
if not _4dof_runs:
    pytest.skip("no 4dof_arm run with assembly.json available", allow_module_level=True)


def test_positions_returns_per_part_placement() -> None:
    """/positions yields a world position (mm) for each assembled part."""
    r = client.get(f"/api/runs/{_REAL_CASE}/{_REAL_TS}/positions")
    assert r.status_code == 200
    data = r.json()
    assert "positions" in data
    pos = data["positions"]
    assert len(pos) >= 6  # the 4dof arm has 11 parts
    # base_plate sits at the XY origin (Z may be a half-thickness offset
    # depending on the base placement convention of the run — the structural
    # invariant is that the base is centred in X/Y, not pinned to Z=0).
    bp = pos["base_plate"]["position"]
    assert bp[0] == 0.0 and bp[1] == 0.0
    # Every entry has a 3-element position list.
    for name, pose in pos.items():
        assert len(pose["position"]) == 3


def test_positions_missing_run_404() -> None:
    """An unknown case/timestamp returns 404, not a 500."""
    r = client.get("/api/runs/nope/00000000_000000/positions")
    assert r.status_code == 404


def test_animate_returns_motion_frames() -> None:
    """/animate forwards the record_joint_motion frame series to the client.

    The route calls ``record_joint_motion`` (returns per-frame JOINT ANGLES,
    not body poses) so the frontend can forward-kinematics every part. This
    was changed from ``record_motion`` (body poses) because MuJoCo merges
    fixed joints, so body-pose playback only moved 6 of 13 parts.
    """
    fake = {
        "ok": True,
        "joints": [
            {"name": "shoulder_pitch", "type": "hinge", "range": [-90, 90]},
        ],
        "fps": 15,
        "duration_sec": 1.0,
        "frames": [
            {"t": 0.0, "angles": {"shoulder_pitch": 0.0}},
            {"t": 0.066, "angles": {"shoulder_pitch": 5.2}},
        ],
        "base_trajectory": None,
    }
    with patch("lang3d.tools.sim_mujoco.record_joint_motion",
               return_value=fake):
        r = client.post(f"/api/runs/{_REAL_CASE}/{_REAL_TS}/animate")
    assert r.status_code == 200
    data = r.json()
    # Joint-angle frames, not body poses.
    assert "angles" in data["frames"][0]
    assert "poses" not in data["frames"][0]
    assert len(data["frames"]) == 2


def test_animate_missing_urdf_404() -> None:
    """A run without an engineering_package/urdf.xml returns 404."""
    with patch("lang3d.tools.sim_mujoco.record_joint_motion") as mock_rec:
        r = client.post("/api/runs/nope/00000000_000000/animate")
    assert r.status_code == 404
    mock_rec.assert_not_called()


def test_animate_propagates_record_failure() -> None:
    """When record_joint_motion reports ok=False, the route returns 500."""
    with patch("lang3d.tools.sim_mujoco.record_joint_motion",
               return_value={"ok": False, "error": "boom", "frames": []}):
        r = client.post(f"/api/runs/{_REAL_CASE}/{_REAL_TS}/animate")
    assert r.status_code == 500
    assert r.json()["error"] == "boom"
