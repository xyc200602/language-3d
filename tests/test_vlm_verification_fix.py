"""Tests for VLM verification fix — geometric pre-validation."""

import pytest

from lang3d.tools.assembly_generator import _geometric_prevalidation


def _pos(x: float, y: float, z: float) -> dict:
    return {"position": [x, y, z]}


class TestGeometricPrevalidation:
    def test_detects_coincident_parts(self):
        """Two parts at the same position should be flagged."""
        parts = [
            {"name": "a", "dimensions": {"length": 10}},
            {"name": "b", "dimensions": {"length": 10}},
        ]
        positions = {
            "a": _pos(0, 0, 0),
            "b": _pos(0, 0, 0),
        }
        problems = _geometric_prevalidation(parts, positions)
        assert any("same position" in p for p in problems)

    def test_detects_outlier_parts(self):
        """Part >500mm from centroid should be flagged."""
        parts = [
            {"name": "a", "dimensions": {"length": 10}},
            {"name": "b", "dimensions": {"length": 10}},
        ]
        positions = {
            "a": _pos(0, 0, 0),
            "b": _pos(1200, 0, 0),
        }
        problems = _geometric_prevalidation(parts, positions)
        assert any("misplaced" in p or "center" in p for p in problems)

    def test_detects_wheels_too_high(self):
        """All wheels above Z=100 should be flagged."""
        parts = [
            {"name": "wheel_fl", "dimensions": {"diameter": 65}},
            {"name": "wheel_fr", "dimensions": {"diameter": 65}},
        ]
        positions = {
            "wheel_fl": _pos(-50, -50, 200),
            "wheel_fr": _pos(50, -50, 200),
        }
        problems = _geometric_prevalidation(parts, positions)
        assert any("wheel" in p.lower() and "ground" in p.lower() for p in problems)

    def test_no_problems_for_valid_positions(self):
        """Normal positions should produce no problems."""
        parts = [
            {"name": "base_plate", "dimensions": {"length": 300, "width": 200, "height": 5}},
            {"name": "wheel_fl", "dimensions": {"diameter": 65}},
            {"name": "wheel_fr", "dimensions": {"diameter": 65}},
        ]
        positions = {
            "base_plate": _pos(0, 0, 30),
            "wheel_fl": _pos(-80, -60, 0),
            "wheel_fr": _pos(80, -60, 0),
        }
        problems = _geometric_prevalidation(parts, positions)
        assert len(problems) == 0
