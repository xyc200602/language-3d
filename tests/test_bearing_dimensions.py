"""Bearing dimension calculation tests."""

from __future__ import annotations

import pytest

from lang3d.knowledge.tolerance import press_fit_bore_diameter, bearing_seat_diameter


class TestPressFitBore:
    def test_standard_bearing_bore(self):
        """Press-fit bore for a 10mm outer diameter should return (min, nom, max)."""
        result = press_fit_bore_diameter(10.0)
        assert isinstance(result, tuple)
        assert len(result) == 3
        min_bore, nominal, max_bore = result
        assert nominal == 10.0  # Nominal equals outer_d for default
        assert min_bore <= nominal <= max_bore

    def test_large_shaft(self):
        result = press_fit_bore_diameter(50.0)
        assert isinstance(result, tuple)
        assert len(result) == 3


class TestBearingSeat:
    def test_standard_seat(self):
        """Bearing seat for a 30mm bearing OD should return (min, nom, max)."""
        result = bearing_seat_diameter(30.0)  # Typical 6200 bearing OD
        assert isinstance(result, tuple)
        assert len(result) == 3
        min_seat, nominal, max_seat = result
        assert nominal == 30.0

    def test_large_bearing(self):
        result = bearing_seat_diameter(72.0)  # Typical 6204 bearing OD
        assert isinstance(result, tuple)
        assert len(result) == 3
