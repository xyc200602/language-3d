"""Cable / wire harness routing planner.

Public API:
  - CableSpec        : cable specification (start, end, diameter, bend radius)
  - CablePath        : routing result (waypoints, length, bend check)
  - build_3d_grid    : build a voxel occupancy grid from assembly part positions
  - find_cable_path  : A* path search → B-spline smoothing → bend validation
  - auto_detect_connections : infer cable connections from part categories
  - generate_cable_report   : Markdown routing report
  - CableRoutingTool : Agent tool
  - register_cable_routing_tools : registration helper
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

from ..knowledge.mechanics import Assembly, Part
from ..models.base import ToolDefinition
from .base import Tool

# ============================================================================
# Data classes
# ============================================================================


@dataclass
class CableSpec:
    """Specification for a single cable / wire."""

    name: str
    start_connector: str  # Part name where cable starts
    end_connector: str    # Part name where cable ends
    cable_type: str = "power"  # "power", "signal", "data", "encoder"
    diameter: float = 3.0       # mm
    min_bend_radius: float = 9.0  # mm (typically 3× diameter)
    length_limit: float = 1000.0  # mm (max allowed length)
    voltage: float = 5.0       # V (for power cables)
    color: str = "red"         # wire color suggestion


@dataclass
class CablePath:
    """Result of routing a single cable."""

    spec: CableSpec
    waypoints: list[tuple[float, float, float]]  # 3D points in mm
    length_mm: float = 0.0
    bend_ok: bool = True
    min_bend_radius_actual: float = float("inf")
    fixed_points: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def within_limit(self) -> bool:
        return self.length_mm <= self.spec.length_limit


# ============================================================================
# 3D voxel grid
# ============================================================================


@dataclass
class VoxelGrid:
    """Simple 3D occupancy grid for path planning.

    Each voxel is `resolution` mm on a side.
    `occupied` is a set of (ix, iy, iz) integer tuples.
    """

    resolution: float = 5.0  # mm per voxel
    bounds_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_max: tuple[float, float, float] = (500.0, 500.0, 500.0)
    occupied: set[tuple[int, int, int]] = field(default_factory=set)

    def _to_idx(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        return (
            int((x - self.bounds_min[0]) / self.resolution),
            int((y - self.bounds_min[1]) / self.resolution),
            int((z - self.bounds_min[2]) / self.resolution),
        )

    def _to_world(self, ix: int, iy: int, iz: int) -> tuple[float, float, float]:
        return (
            ix * self.resolution + self.bounds_min[0] + self.resolution / 2,
            iy * self.resolution + self.bounds_min[1] + self.resolution / 2,
            iz * self.resolution + self.bounds_min[2] + self.resolution / 2,
        )

    def is_free(self, x: float, y: float, z: float) -> bool:
        return self._to_idx(x, y, z) not in self.occupied

    def mark_occupied_box(
        self,
        center: tuple[float, float, float],
        half_extents: tuple[float, float, float],
        margin: float = 2.0,
    ) -> None:
        """Mark voxels inside a box as occupied."""
        for axis in range(3):
            lo = int((center[axis] - half_extents[axis] - margin - self.bounds_min[axis]) / self.resolution)
            hi = int((center[axis] + half_extents[axis] + margin - self.bounds_min[axis]) / self.resolution) + 1
            if axis == 0:
                range_x = range(max(0, lo), hi)
            elif axis == 1:
                range_y = range(max(0, lo), hi)
            else:
                range_z = range(max(0, lo), hi)
        for ix in range_x:  # type: ignore[possibly-undefined]
            for iy in range_y:  # type: ignore
                for iz in range_z:  # type: ignore
                    self.occupied.add((ix, iy, iz))


def build_3d_grid(
    part_positions: dict[str, dict[str, Any]],
    parts: list[Part],
    resolution: float = 5.0,
    margin: float = 2.0,
) -> VoxelGrid:
    """Build a voxel grid marking part volumes as occupied.

    Args:
        part_positions: output of AssemblySolver.solve() — {name: {position: [x,y,z], ...}}
        parts: list of Part objects (for dimensions)
        resolution: voxel size in mm
        margin: clearance around parts in mm

    Returns:
        VoxelGrid with occupied cells.
    """
    # Compute bounds from all part positions
    all_pos = [p["position"] for p in part_positions.values()]
    if not all_pos:
        return VoxelGrid(resolution=resolution)

    xs = [p[0] for p in all_pos]
    ys = [p[1] for p in all_pos]
    zs = [p[2] for p in all_pos]

    grid = VoxelGrid(
        resolution=resolution,
        bounds_min=(min(xs) - 200, min(ys) - 200, min(zs) - 200),
        bounds_max=(max(xs) + 200, max(ys) + 200, max(zs) + 200),
    )

    part_by_name = {p.name: p for p in parts}
    for name, placement in part_positions.items():
        pos = placement["position"]
        part = part_by_name.get(name)
        if part is None:
            continue
        dims = part.dimensions
        hx = dims.get("length", dims.get("diameter", 20)) / 2.0
        hy = dims.get("width", dims.get("diameter", 20)) / 2.0
        hz = dims.get("height", dims.get("thickness", 10)) / 2.0
        grid.mark_occupied_box(
            center=(pos[0], pos[1], pos[2]),
            half_extents=(hx, hy, hz),
            margin=margin,
        )

    return grid


# ============================================================================
# A* path search
# ============================================================================


# 26-connected neighbors (3D)
_NEIGHBORS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]


def _astar(
    grid: VoxelGrid,
    start: tuple[float, float, float],
    goal: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    """A* search on the voxel grid. Returns list of world-coordinate waypoints."""
    s_idx = grid._to_idx(*start)
    g_idx = grid._to_idx(*goal)

    if g_idx in grid.occupied:
        # Goal is inside a part — try nearby free cell
        g_idx = _find_nearest_free(grid, g_idx)
        if g_idx is None:
            return []

    if s_idx in grid.occupied:
        s_idx = _find_nearest_free(grid, s_idx)
        if s_idx is None:
            return []

    open_set: list[tuple[float, float, tuple[int, int, int]]] = []
    heapq.heappush(open_set, (0.0, 0.0, s_idx))
    came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    g_score: dict[tuple[int, int, int], float] = {s_idx: 0.0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)

        if current == g_idx:
            # Reconstruct path
            path = []
            node = current
            while node in came_from:
                path.append(grid._to_world(*node))
                node = came_from[node]
            path.append(grid._to_world(*s_idx))
            path.reverse()
            path.append(grid._to_world(*g_idx))
            return path

        for dx, dy, dz in _NEIGHBORS:
            nb = (current[0] + dx, current[1] + dy, current[2] + dz)
            if nb in grid.occupied:
                continue
            step_cost = math.sqrt(dx * dx + dy * dy + dz * dz) * grid.resolution
            tentative = cost + step_cost
            if tentative < g_score.get(nb, float("inf")):
                g_score[nb] = tentative
                came_from[nb] = current
                h = _heuristic(nb, g_idx, grid.resolution)
                heapq.heappush(open_set, (tentative + h, tentative, nb))

    return []  # no path found


def _heuristic(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    res: float,
) -> float:
    """Euclidean heuristic."""
    return math.sqrt(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    ) * res


def _find_nearest_free(
    grid: VoxelGrid,
    idx: tuple[int, int, int],
    max_radius: int = 5,
) -> tuple[int, int, int] | None:
    """Find the nearest free voxel to an occupied one."""
    for r in range(1, max_radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    nb = (idx[0] + dx, idx[1] + dy, idx[2] + dz)
                    if nb not in grid.occupied:
                        return nb
    return None


# ============================================================================
# B-spline smoothing (simplified)
# ============================================================================


def _smooth_path(
    waypoints: list[tuple[float, float, float]],
    factor: int = 3,
) -> list[tuple[float, float, float]]:
    """Simple path smoothing via Chaikin's corner cutting algorithm."""
    if len(waypoints) < 3:
        return waypoints

    pts = list(waypoints)
    for _ in range(factor):
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            q = (
                0.75 * p0[0] + 0.25 * p1[0],
                0.75 * p0[1] + 0.25 * p1[1],
                0.75 * p0[2] + 0.25 * p1[2],
            )
            r = (
                0.25 * p0[0] + 0.75 * p1[0],
                0.25 * p0[1] + 0.75 * p1[1],
                0.25 * p0[2] + 0.75 * p1[2],
            )
            new_pts.append(q)
            new_pts.append(r)
        new_pts.append(pts[-1])
        pts = new_pts

    return pts


# ============================================================================
# Bend radius validation
# ============================================================================


def _check_bend_radius(
    waypoints: list[tuple[float, float, float]],
    min_radius: float,
) -> tuple[bool, float]:
    """Check if all bends along the path satisfy min bend radius.

    Returns (ok, actual_min_radius).
    """
    if len(waypoints) < 3:
        return True, float("inf")

    min_actual = float("inf")
    for i in range(1, len(waypoints) - 1):
        p0 = waypoints[i - 1]
        p1 = waypoints[i]
        p2 = waypoints[i + 1]

        v1 = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        v2 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])

        len1 = math.sqrt(sum(c * c for c in v1))
        len2 = math.sqrt(sum(c * c for c in v2))
        if len1 < 1e-6 or len2 < 1e-6:
            continue

        dot = sum(a * b for a, b in zip(v1, v2))
        cos_angle = max(-1.0, min(1.0, dot / (len1 * len2)))
        angle = math.acos(cos_angle)

        if angle < 1e-6:
            continue

        # Approximate bend radius = segment_length / (2 * sin(angle/2))
        half_angle = angle / 2.0
        if half_angle < 1e-6:
            continue
        radius = min(len1, len2) / (2.0 * math.sin(half_angle))
        if radius < min_actual:
            min_actual = radius

    return min_actual >= min_radius, min_actual


# ============================================================================
# Path length calculation
# ============================================================================


def _path_length(waypoints: list[tuple[float, float, float]]) -> float:
    """Compute total path length in mm."""
    total = 0.0
    for i in range(1, len(waypoints)):
        dx = waypoints[i][0] - waypoints[i - 1][0]
        dy = waypoints[i][1] - waypoints[i - 1][1]
        dz = waypoints[i][2] - waypoints[i - 1][2]
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total


# ============================================================================
# find_cable_path — main routing function
# ============================================================================


def find_cable_path(
    grid: VoxelGrid,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    spec: CableSpec | None = None,
    smooth_iterations: int = 3,
) -> CablePath:
    """Find a cable path from start to end through free space.

    Args:
        grid: Voxel occupancy grid (from build_3d_grid)
        start: (x, y, z) start position in mm
        end: (x, y, z) end position in mm
        spec: Cable specification (for bend radius check)
        smooth_iterations: Chaikin smoothing iterations

    Returns:
        CablePath with waypoints, length, bend check results.
    """
    min_radius = spec.min_bend_radius if spec else 9.0

    raw_path = _astar(grid, start, end)
    if not raw_path:
        return CablePath(
            spec=spec or CableSpec(name="unknown", start_connector="", end_connector=""),
            waypoints=[start, end],
            length_mm=_path_length([start, end]),
            bend_ok=False,
            min_bend_radius_actual=0.0,
        )

    smoothed = _smooth_path(raw_path, factor=smooth_iterations)
    length = _path_length(smoothed)
    bend_ok, min_bend = _check_bend_radius(smoothed, min_radius)

    # Suggest fixed points (clamp locations) every ~100mm
    fixed: list[tuple[float, float, float]] = []
    if smoothed:
        fixed.append(smoothed[0])
        dist = 0.0
        for i in range(1, len(smoothed)):
            seg_len = math.sqrt(
                sum(
                    (smoothed[i][c] - smoothed[i - 1][c]) ** 2
                    for c in range(3)
                )
            )
            dist += seg_len
            if dist >= 100.0:
                fixed.append(smoothed[i])
                dist = 0.0
        if fixed[-1] != smoothed[-1]:
            fixed.append(smoothed[-1])

    return CablePath(
        spec=spec or CableSpec(name="unknown", start_connector="", end_connector=""),
        waypoints=smoothed,
        length_mm=round(length, 1),
        bend_ok=bend_ok,
        min_bend_radius_actual=round(min_bend, 1),
        fixed_points=fixed,
    )


# ============================================================================
# Auto-detect cable connections
# ============================================================================


def auto_detect_connections(assembly: Assembly) -> list[CableSpec]:
    """Infer cable connections from part categories.

    Rules:
      - actuator parts → controller (signal + power)
      - sensor parts → controller (signal)
      - battery parts → power distribution (power)
    """
    cables: list[CableSpec] = []
    cable_id = 0

    actuators = [p for p in assembly.parts if p.category.lower() in ("actuator", "servo", "motor")]
    sensors = [p for p in assembly.parts if p.category.lower() in ("sensor",)]
    batteries = [p for p in assembly.parts if p.category.lower() in ("battery", "power")]
    controllers = [p for p in assembly.parts if p.category.lower() in ("controller", "electronics", "pcb")]

    # Default controller: first controller, or a generic target
    ctrl = controllers[0].name if controllers else "controller"

    # Actuator → controller
    for act in actuators:
        cable_id += 1
        cables.append(CableSpec(
            name=f"cable_{cable_id}_{act.name}_signal",
            start_connector=act.name,
            end_connector=ctrl,
            cable_type="signal",
            diameter=2.0,
            min_bend_radius=6.0,
            voltage=5.0,
            color="yellow",
        ))
        cable_id += 1
        cables.append(CableSpec(
            name=f"cable_{cable_id}_{act.name}_power",
            start_connector=act.name,
            end_connector=batteries[0].name if batteries else "power_supply",
            cable_type="power",
            diameter=3.5,
            min_bend_radius=10.5,
            voltage=act.dimensions.get("voltage", 6.0),
            color="red",
        ))

    # Sensor → controller
    for sens in sensors:
        cable_id += 1
        cables.append(CableSpec(
            name=f"cable_{cable_id}_{sens.name}_data",
            start_connector=sens.name,
            end_connector=ctrl,
            cable_type="data",
            diameter=1.5,
            min_bend_radius=4.5,
            color="blue",
        ))

    # Battery → controller
    for bat in batteries:
        cable_id += 1
        cables.append(CableSpec(
            name=f"cable_{cable_id}_{bat.name}_power_main",
            start_connector=bat.name,
            end_connector=ctrl,
            cable_type="power",
            diameter=4.0,
            min_bend_radius=12.0,
            voltage=bat.dimensions.get("voltage", 12.0),
            color="red",
        ))

    return cables


# ============================================================================
# Report generation
# ============================================================================


def generate_cable_report(
    cable_paths: list[CablePath],
    assembly_name: str = "",
) -> str:
    """Generate a Markdown cable routing report."""
    lines = [
        f"# Cable Routing Report",
        f"",
        f"Assembly: **{assembly_name}**" if assembly_name else "",
        f"",
        f"## Cable Summary",
        f"",
        f"| # | Cable | Type | From → To | Length (mm) | Bend OK | Min Radius |",
        f"|---|-------|------|-----------|-------------|---------|------------|",
    ]

    for i, cp in enumerate(cable_paths, 1):
        bend_str = "Yes" if cp.bend_ok else "FAIL"
        lines.append(
            f"| {i} | {cp.spec.name} | {cp.spec.cable_type} | "
            f"{cp.spec.start_connector} → {cp.spec.end_connector} | "
            f"{cp.length_mm:.1f} | {bend_str} | {cp.min_bend_radius_actual:.1f} mm |"
        )

    lines.append("")
    lines.append("## Cable Details")
    lines.append("")

    for cp in cable_paths:
        within = "OK" if cp.within_limit else f"EXCEEDS {cp.spec.length_limit:.0f}mm"
        lines.append(f"### {cp.spec.name}")
        lines.append(f"- Type: {cp.spec.cable_type}")
        lines.append(f"- Diameter: {cp.spec.diameter:.1f} mm")
        lines.append(f"- Length: {cp.length_mm:.1f} mm ({within})")
        lines.append(f"- Bend radius: {cp.min_bend_radius_actual:.1f} mm (min required: {cp.spec.min_bend_radius:.1f} mm)")
        lines.append(f"- Waypoints: {len(cp.waypoints)}")
        lines.append(f"- Suggested clamps: {len(cp.fixed_points)}")
        lines.append("")

    # Warnings
    warnings: list[str] = []
    for cp in cable_paths:
        if not cp.bend_ok:
            warnings.append(f"- {cp.spec.name}: bend radius {cp.min_bend_radius_actual:.1f}mm < min {cp.spec.min_bend_radius:.1f}mm")
        if not cp.within_limit:
            warnings.append(f"- {cp.spec.name}: length {cp.length_mm:.1f}mm exceeds limit {cp.spec.length_limit:.0f}mm")

    if warnings:
        lines.append("## Warnings")
        lines.append("")
        lines.extend(warnings)
    else:
        lines.append("## Status: All cables OK")

    return "\n".join(lines)


# ============================================================================
# Agent Tool
# ============================================================================


class CableRoutingTool(Tool):
    """Plan cable routing for an assembly."""

    name = "cable_routing"
    description = "Plan cable/wire routing paths between electronic components in an assembly"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "assembly_name": {
                        "type": "string",
                        "description": "Name of the assembly to route cables for",
                    },
                    "mode": {
                        "type": "string",
                        "description": "Output mode: 'report' (Markdown) or 'json' (structured data). Default: report",
                    },
                    "resolution": {
                        "type": "number",
                        "description": "Voxel grid resolution in mm (default: 5.0)",
                    },
                },
                "required": [],
            },
        )

    def execute(
        self,
        *,
        assembly_name: str = "",
        mode: str = "report",
        resolution: float = 5.0,
        **kwargs: Any,
    ) -> str:
        # If no assembly name, do auto-detect demo
        from ..knowledge.mechanics import ROBOTIC_ARM_ASSEMBLY
        assembly = ROBOTIC_ARM_ASSEMBLY
        if assembly_name:
            found = _find_assembly(assembly_name)
            if found is None:
                return f"错误：未找到装配体 '{assembly_name}'"
            assembly = found

        cables = auto_detect_connections(assembly)

        if not cables:
            return "未检测到需要电缆连接的零件（需要 actuator/sensor/battery/controller 类型零件）"

        # Compute part positions using assembly solver
        part_positions = _get_part_positions(assembly)

        # Build grid
        grid = build_3d_grid(part_positions, assembly.parts, resolution=resolution)

        # Route each cable
        cable_paths: list[CablePath] = []
        for spec in cables:
            start_pos = part_positions.get(spec.start_connector, {}).get("position", [0, 0, 0])
            end_pos = part_positions.get(spec.end_connector, {}).get("position", [0, 0, 0])
            cp = find_cable_path(
                grid,
                start=(start_pos[0], start_pos[1], start_pos[2]),
                end=(end_pos[0], end_pos[1], end_pos[2]),
                spec=spec,
            )
            cable_paths.append(cp)

        if mode == "json":
            import json
            data = {
                "assembly": assembly.name,
                "cables": [
                    {
                        "name": cp.spec.name,
                        "type": cp.spec.cable_type,
                        "from": cp.spec.start_connector,
                        "to": cp.spec.end_connector,
                        "length_mm": cp.length_mm,
                        "bend_ok": cp.bend_ok,
                        "min_bend_radius": cp.min_bend_radius_actual,
                        "waypoints": len(cp.waypoints),
                        "clamps": len(cp.fixed_points),
                    }
                    for cp in cable_paths
                ],
            }
            return json.dumps(data, ensure_ascii=False, indent=2)

        return generate_cable_report(cable_paths, assembly_name=assembly.name)


# ============================================================================
# Helpers
# ============================================================================


def _find_assembly(name: str) -> Assembly | None:
    from ..knowledge.mechanics import ROBOTIC_ARM_ASSEMBLY
    if ROBOTIC_ARM_ASSEMBLY.name == name:
        return ROBOTIC_ARM_ASSEMBLY
    return None


def _get_part_positions(assembly: Assembly) -> dict[str, dict[str, Any]]:
    """Solve assembly to get part positions."""
    try:
        from .assembly_solver import AssemblySolver
        solver = AssemblySolver(assembly)
        return solver.solve()
    except Exception:
        # Fallback: arrange parts linearly
        positions: dict[str, dict[str, Any]] = {}
        z = 0.0
        for p in assembly.parts:
            h = p.dimensions.get("height", p.dimensions.get("thickness", 20))
            positions[p.name] = {
                "position": [0.0, 0.0, z],
                "rotation": [0.0, 0.0, 1.0, 0.0],
            }
            z += h
        return positions


# ============================================================================
# Registration
# ============================================================================


def register_cable_routing_tools(registry: Any) -> None:
    """Register cable routing tools."""
    registry.register(CableRoutingTool())
