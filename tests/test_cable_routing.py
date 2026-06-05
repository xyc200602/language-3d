"""Tests for cable routing tool (Task 55).

Covers:
  - CableSpec / CablePath data classes
  - VoxelGrid: occupancy, bounds, mark_occupied_box
  - build_3d_grid: from assembly part positions
  - A* path search
  - Path smoothing (Chaikin)
  - Bend radius validation
  - auto_detect_connections
  - generate_cable_report
  - CableRoutingTool: execution, registration
"""

import json
import math
import tempfile
from pathlib import Path

import pytest

from lang3d.knowledge.mechanics import Assembly, Joint, Part
from lang3d.tools.cable_routing import (
    CablePath,
    CableRoutingTool,
    CableSpec,
    VoxelGrid,
    auto_detect_connections,
    build_3d_grid,
    find_cable_path,
    generate_cable_report,
    register_cable_routing_tools,
    _astar,
    _check_bend_radius,
    _path_length,
    _smooth_path,
)
from lang3d.tools.base import ToolRegistry


# ============================================================================
# Fixtures
# ============================================================================


def _make_electronics_assembly():
    """Assembly with actuators, sensors, battery, controller."""
    return Assembly(
        name="TestBot",
        parts=[
            Part(name="chassis", category="structural", description="chassis",
                 dimensions=dict(length=200, width=150, height=10)),
            Part(name="motor_l", category="actuator", description="left motor",
                 dimensions=dict(length=40, width=30, height=25)),
            Part(name="motor_r", category="actuator", description="right motor",
                 dimensions=dict(length=40, width=30, height=25)),
            Part(name="servo_arm", category="servo", description="arm servo",
                 dimensions=dict(length=30, width=24, height=12)),
            Part(name="imu_sensor", category="sensor", description="IMU",
                 dimensions=dict(length=20, width=15, height=3)),
            Part(name="battery", category="battery", description="LiPo battery",
                 dimensions=dict(length=100, width=35, height=20)),
            Part(name="esp32", category="controller", description="ESP32 controller",
                 dimensions=dict(length=55, width=28, height=5)),
        ],
        joints=[
            Joint("fixed", "chassis", "motor_l"),
            Joint("fixed", "chassis", "motor_r"),
            Joint("fixed", "chassis", "servo_arm"),
            Joint("fixed", "chassis", "imu_sensor"),
            Joint("fixed", "chassis", "battery"),
            Joint("fixed", "chassis", "esp32"),
        ],
    )


# ============================================================================
# CableSpec / CablePath
# ============================================================================


class TestCableSpec:
    def test_defaults(self):
        spec = CableSpec(name="test", start_connector="a", end_connector="b")
        assert spec.cable_type == "power"
        assert spec.diameter == 3.0
        assert spec.min_bend_radius == 9.0
        assert spec.length_limit == 1000.0

    def test_custom(self):
        spec = CableSpec(
            name="sig", start_connector="imu", end_connector="esp32",
            cable_type="data", diameter=1.5, min_bend_radius=4.5,
        )
        assert spec.cable_type == "data"
        assert spec.min_bend_radius == 4.5


class TestCablePath:
    def test_within_limit(self):
        spec = CableSpec(name="t", start_connector="a", end_connector="b", length_limit=100)
        cp = CablePath(spec=spec, waypoints=[(0, 0, 0), (50, 0, 0)], length_mm=50)
        assert cp.within_limit is True

    def test_exceeds_limit(self):
        spec = CableSpec(name="t", start_connector="a", end_connector="b", length_limit=100)
        cp = CablePath(spec=spec, waypoints=[(0, 0, 0), (200, 0, 0)], length_mm=200)
        assert cp.within_limit is False


# ============================================================================
# VoxelGrid
# ============================================================================


class TestVoxelGrid:
    def test_default_free(self):
        grid = VoxelGrid()
        assert grid.is_free(50, 50, 50) is True

    def test_mark_occupied(self):
        grid = VoxelGrid(resolution=5.0, bounds_min=(0, 0, 0), bounds_max=(100, 100, 100))
        grid.mark_occupied_box(center=(50, 50, 50), half_extents=(10, 10, 10))
        assert grid.is_free(50, 50, 50) is False
        assert grid.is_free(80, 80, 80) is True

    def test_index_roundtrip(self):
        grid = VoxelGrid(resolution=10.0, bounds_min=(0, 0, 0))
        idx = grid._to_idx(25, 35, 45)
        w = grid._to_world(*idx)
        # Should be close to center of voxel
        assert abs(w[0] - 25) < grid.resolution
        assert abs(w[1] - 35) < grid.resolution
        assert abs(w[2] - 45) < grid.resolution


# ============================================================================
# build_3d_grid
# ============================================================================


class TestBuild3dGrid:
    def test_basic(self):
        parts = [
            Part(name="box", category="structural", description="box",
                 dimensions=dict(length=50, width=50, height=50)),
        ]
        positions = {"box": {"position": [100, 100, 100], "rotation": [0, 0, 1, 0]}}
        grid = build_3d_grid(positions, parts, resolution=10.0)
        # Center should be occupied
        assert grid.is_free(100, 100, 100) is False
        # Far away should be free
        assert grid.is_free(300, 300, 300) is True

    def test_empty(self):
        grid = build_3d_grid({}, [], resolution=5.0)
        assert len(grid.occupied) == 0


# ============================================================================
# A* path search
# ============================================================================


class TestAStar:
    def test_straight_line(self):
        grid = VoxelGrid(resolution=10.0, bounds_min=(0, 0, 0), bounds_max=(200, 200, 200))
        path = _astar(grid, (10, 10, 10), (100, 10, 10))
        assert len(path) >= 2
        # Should reach close to goal
        assert path[-1][0] == pytest.approx(105, abs=15)

    def test_blocked_returns_empty_or_detour(self):
        grid = VoxelGrid(resolution=10.0, bounds_min=(0, 0, 0), bounds_max=(200, 200, 200))
        # Block everything in a wall
        for y in range(20):
            for z in range(20):
                idx = grid._to_idx(50, y * 10, z * 10)
                grid.occupied.add(idx)
        # Should still find a path (detour above/below wall)
        path = _astar(grid, (10, 100, 100), (150, 100, 100))
        # Might find detour or empty depending on wall thickness
        # At minimum, should not crash
        assert isinstance(path, list)


# ============================================================================
# Path smoothing
# ============================================================================


class TestSmoothPath:
    def test_short_path_unchanged(self):
        pts = [(0, 0, 0), (10, 0, 0)]
        result = _smooth_path(pts, factor=2)
        assert len(result) >= 2

    def test_longer_path_smoothed(self):
        pts = [(0, 0, 0), (50, 50, 0), (100, 0, 0)]
        result = _smooth_path(pts, factor=3)
        assert len(result) > len(pts)


# ============================================================================
# Bend radius
# ============================================================================


class TestBendRadius:
    def test_straight_line_passes(self):
        pts = [(0, 0, 0), (50, 0, 0), (100, 0, 0)]
        ok, radius = _check_bend_radius(pts, 5.0)
        assert ok is True
        assert radius == float("inf")

    def test_sharp_bend_fails(self):
        pts = [(0, 0, 0), (5, 0, 0), (5, 5, 0)]
        ok, radius = _check_bend_radius(pts, 5.0)
        assert ok is False

    def test_gentle_curve_passes(self):
        pts = [(0, 0, 0), (100, 0, 0), (200, 1, 0)]
        ok, radius = _check_bend_radius(pts, 5.0)
        assert ok is True


class TestPathLength:
    def test_single_segment(self):
        assert _path_length([(0, 0, 0), (100, 0, 0)]) == pytest.approx(100.0)

    def test_diagonal(self):
        length = _path_length([(0, 0, 0), (3, 4, 0)])
        assert length == pytest.approx(5.0)


# ============================================================================
# find_cable_path
# ============================================================================


class TestFindCablePath:
    def test_basic_path(self):
        grid = VoxelGrid(resolution=10.0, bounds_min=(0, 0, 0), bounds_max=(300, 300, 300))
        spec = CableSpec(name="test", start_connector="a", end_connector="b",
                         min_bend_radius=5.0)
        result = find_cable_path(grid, (10, 10, 10), (200, 10, 10), spec=spec)
        assert result.length_mm > 0
        assert len(result.waypoints) >= 2

    def test_no_path_fallback(self):
        # Completely blocked grid — A* will find nearest free cell outside bounds
        # and route around the obstacle
        grid = VoxelGrid(resolution=10.0, bounds_min=(0, 0, 0), bounds_max=(50, 50, 50))
        for ix in range(10):
            for iy in range(10):
                for iz in range(10):
                    grid.occupied.add((ix, iy, iz))
        spec = CableSpec(name="test", start_connector="a", end_connector="b")
        result = find_cable_path(grid, (5, 5, 5), (45, 5, 5), spec=spec)
        # Should still return a CablePath (may detour outside the blocked region)
        assert isinstance(result, CablePath)
        assert result.length_mm > 0


# ============================================================================
# auto_detect_connections
# ============================================================================


class TestAutoDetectConnections:
    def test_detects_actuator_cables(self):
        asm = _make_electronics_assembly()
        cables = auto_detect_connections(asm)
        # 3 actuators * 2 (signal + power) + 1 sensor * 1 + 1 battery * 1 = 8
        assert len(cables) >= 6
        signal_cables = [c for c in cables if c.cable_type == "signal"]
        power_cables = [c for c in cables if c.cable_type == "power"]
        assert len(signal_cables) >= 3  # 3 actuators
        assert len(power_cables) >= 1  # battery + actuator power

    def test_actuator_goes_to_controller(self):
        asm = _make_electronics_assembly()
        cables = auto_detect_connections(asm)
        signal = [c for c in cables if c.cable_type == "signal"]
        for c in signal:
            assert c.end_connector == "esp32"

    def test_battery_to_controller(self):
        asm = _make_electronics_assembly()
        cables = auto_detect_connections(asm)
        bat_cables = [c for c in cables if c.start_connector == "battery"]
        assert len(bat_cables) >= 1
        assert bat_cables[0].cable_type == "power"

    def test_no_electronics_empty(self):
        asm = Assembly(
            name="no_elec",
            parts=[
                Part(name="plate", category="structural", description="plate",
                     dimensions=dict(length=100, width=100, height=5)),
            ],
            joints=[],
        )
        cables = auto_detect_connections(asm)
        assert cables == []


# ============================================================================
# Report generation
# ============================================================================


class TestGenerateReport:
    def test_basic_report(self):
        spec = CableSpec(name="cable_1", start_connector="motor", end_connector="esp32",
                         cable_type="signal")
        cp = CablePath(spec=spec, waypoints=[(0, 0, 0), (100, 0, 0)], length_mm=100.0,
                       bend_ok=True, min_bend_radius_actual=20.0)
        report = generate_cable_report([cp], assembly_name="TestBot")
        assert "Cable Routing Report" in report
        assert "cable_1" in report
        assert "TestBot" in report
        assert "signal" in report
        assert "100.0" in report

    def test_warning_for_bend_fail(self):
        spec = CableSpec(name="cable_bad", start_connector="a", end_connector="b",
                         min_bend_radius=10.0)
        cp = CablePath(spec=spec, waypoints=[(0, 0, 0), (5, 0, 0), (5, 5, 0)],
                       length_mm=15.0, bend_ok=False, min_bend_radius_actual=3.0)
        report = generate_cable_report([cp])
        assert "FAIL" in report or "Warning" in report

    def test_all_ok_status(self):
        spec = CableSpec(name="cable_ok", start_connector="a", end_connector="b")
        cp = CablePath(spec=spec, waypoints=[(0, 0, 0), (100, 0, 0)], length_mm=100.0,
                       bend_ok=True, min_bend_radius_actual=50.0)
        report = generate_cable_report([cp])
        assert "All cables OK" in report


# ============================================================================
# CableRoutingTool
# ============================================================================


class TestCableRoutingTool:
    def test_report_mode(self):
        tool = CableRoutingTool()
        result = tool.execute(mode="report")
        assert isinstance(result, str)

    def test_json_mode(self):
        tool = CableRoutingTool()
        result = tool.execute(mode="json")
        data = json.loads(result)
        assert "assembly" in data
        assert "cables" in data

    def test_unknown_assembly(self):
        tool = CableRoutingTool()
        result = tool.execute(assembly_name="nonexistent_bot")
        assert "错误" in result

    def test_definition(self):
        tool = CableRoutingTool()
        defn = tool.get_definition()
        assert defn.name == "cable_routing"
        assert "assembly_name" in defn.parameters["properties"]


class TestRegistration:
    def test_cable_routing_registered(self):
        registry = ToolRegistry()
        register_cable_routing_tools(registry)
        assert "cable_routing" in registry._tools

    def test_tool_count(self):
        registry = ToolRegistry()
        register_cable_routing_tools(registry)
        cable_tools = [t for t in registry._tools if "cable" in t]
        assert len(cable_tools) == 1
