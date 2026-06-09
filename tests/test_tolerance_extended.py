"""Tolerance tests — validates Phase 4 boundary condition fixes."""

from __future__ import annotations

import math

import pytest

from lang3d.knowledge.tolerance import _size_range_index, _BASIC_SIZE_RANGES


class TestSizeRangeIndex:
    """Test _size_range_index boundary conditions."""

    def test_lower_bound_first_range(self):
        """The first range (1-3) should include 1.0."""
        idx = _size_range_index(1.0)
        assert idx == 0

    def test_boundary_3mm(self):
        """3.0mm is the upper bound of first range (1-3], inclusive."""
        idx = _size_range_index(3.0)
        assert idx == 0

    def test_just_above_3mm(self):
        """3.001mm should be in second range (3-6]."""
        idx = _size_range_index(3.001)
        assert idx == 1

    def test_boundary_6mm(self):
        """6.0mm is the upper bound of second range (3-6], inclusive."""
        idx = _size_range_index(6.0)
        assert idx == 1

    def test_mid_range(self):
        """10.0 should be in third range (6-10], inclusive upper."""
        idx = _size_range_index(10.0)
        assert idx == 2

    def test_upper_boundary(self):
        """500.0 should be in last range."""
        idx = _size_range_index(500.0)
        assert idx == len(_BASIC_SIZE_RANGES) - 1

    def test_below_range(self):
        """0.5 should clamp to first range."""
        idx = _size_range_index(0.5)
        assert idx == 0

    def test_above_range(self):
        """600 should clamp to last range."""
        idx = _size_range_index(600)
        assert idx == len(_BASIC_SIZE_RANGES) - 1


class TestComputeFit:
    """Test ISO fit computation."""

    def test_clearance_fit_basic(self):
        from lang3d.knowledge.tolerance import compute_fit
        # H7/g6 is a clearance fit for 20mm shaft
        result = compute_fit(20.0, "IT7", "IT6", hole_deviation="H", shaft_deviation="g")
        assert result is not None
        assert result.fit_type == "clearance"

    def test_interference_fit(self):
        from lang3d.knowledge.tolerance import compute_fit
        # H7/p6 is an interference fit
        result = compute_fit(20.0, "IT7", "IT6", hole_deviation="H", shaft_deviation="p")
        assert result is not None
        assert result.fit_type == "interference"
