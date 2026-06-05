"""Workspace analysis for robotic arms.

Provides:
  - compute_workspace: Monte Carlo sampling of reachable points
  - compute_shared_workspace: Dual-arm shared workspace overlap analysis
  - WorkspaceAnalysisTool: Agent tool for workspace queries
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any

from ..knowledge.mechanics import Assembly, Joint
from ..models.base import ToolDefinition
from .assembly_solver import AssemblySolver
from .base import Tool
from .ik_solver import LinkSegment, _extract_chain, solve_ik


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WorkspacePoint:
    """A single reachable point in the workspace."""
    x: float
    y: float
    z: float
    joint_angles: dict[str, float]
    manipulability: float  # 0..1, estimate of dexterity at this point


@dataclass
class WorkspaceResult:
    """Result of workspace analysis."""
    points: list[WorkspacePoint]
    total_samples: int
    reachable_count: int
    reachability_ratio: float
    bounds: dict[str, list[float]]  # {"x": [min, max], "y": ..., "z": ...}
    center: list[float]
    max_reach: float


# ---------------------------------------------------------------------------
# Workspace computation
# ---------------------------------------------------------------------------

def compute_workspace(
    assembly: Assembly,
    n_samples: int = 1000,
    seed: int = 42,
    tolerance_mm: float = 5.0,
) -> WorkspaceResult:
    """Compute workspace using Monte Carlo sampling.

    Randomly samples joint configurations, runs FK to get end-effector
    positions, and records reachable points.

    Args:
        assembly: The robot arm assembly.
        n_samples: Number of random configurations to sample.
        seed: Random seed for reproducibility.
        tolerance_mm: IK tolerance for reachability check.

    Returns:
        WorkspaceResult with reachable points and statistics.
    """
    rng = random.Random(seed)
    links, base_height = _extract_chain(assembly)
    if not links:
        return WorkspaceResult(
            points=[], total_samples=n_samples, reachable_count=0,
            reachability_ratio=0.0,
            bounds={"x": [0, 0], "y": [0, 0], "z": [0, 0]},
            center=[0, 0, 0], max_reach=0.0,
        )

    # Get joint limits
    joint_limits: dict[str, tuple[float, float]] = {}
    for j in assembly.joints:
        joint_limits[j.child] = j.range_deg

    # Compute max reach (sum of all link lengths)
    max_reach = sum(l.length for l in links)

    solver = AssemblySolver(assembly)

    # Find end-effector name
    ee_name = links[-1].name
    for j in assembly.joints:
        if j.parent == links[-1].name and j.type == "fixed":
            ee_name = j.child
            break

    points: list[WorkspacePoint] = []
    x_vals, y_vals, z_vals = [], [], []

    for _ in range(n_samples):
        # Sample random joint angles
        angles: dict[str, float] = {}
        for link in links:
            lim = joint_limits.get(link.name, (-180, 180))
            angles[link.name] = rng.uniform(lim[0], lim[1])

        # Forward kinematics
        placements = solver.solve(joint_angles=angles)
        ee_pos = placements.get(ee_name, {}).get("position", [0, 0, 0])

        x_vals.append(ee_pos[0])
        y_vals.append(ee_pos[1])
        z_vals.append(ee_pos[2])

        # Simple manipulability estimate: distance from workspace boundary
        dist = math.sqrt(ee_pos[0] ** 2 + ee_pos[1] ** 2 + (ee_pos[2] - base_height) ** 2)
        manip = max(0.0, 1.0 - dist / max_reach) if max_reach > 0 else 0.0

        points.append(WorkspacePoint(
            x=round(ee_pos[0], 2),
            y=round(ee_pos[1], 2),
            z=round(ee_pos[2], 2),
            joint_angles={k: round(v, 2) for k, v in angles.items()},
            manipulability=round(manip, 4),
        ))

    bounds = {
        "x": [round(min(x_vals), 2), round(max(x_vals), 2)] if x_vals else [0, 0],
        "y": [round(min(y_vals), 2), round(max(y_vals), 2)] if y_vals else [0, 0],
        "z": [round(min(z_vals), 2), round(max(z_vals), 2)] if z_vals else [0, 0],
    }
    center = [
        round((bounds["x"][0] + bounds["x"][1]) / 2, 2),
        round((bounds["y"][0] + bounds["y"][1]) / 2, 2),
        round((bounds["z"][0] + bounds["z"][1]) / 2, 2),
    ]

    return WorkspaceResult(
        points=points,
        total_samples=n_samples,
        reachable_count=len(points),
        reachability_ratio=1.0,  # All FK samples produce a point
        bounds=bounds,
        center=center,
        max_reach=round(max_reach, 2),
    )


def compute_shared_workspace(
    arm1_assembly: Assembly,
    arm2_assembly: Assembly,
    n_samples: int = 500,
    grid_size: float = 10.0,
) -> dict[str, Any]:
    """Compute the shared workspace between two arms.

    Samples both workspaces and finds overlapping reachable regions
    using grid-based voxelization.

    Args:
        arm1_assembly: First arm assembly.
        arm2_assembly: Second arm assembly.
        n_samples: Samples per arm.
        grid_size: Voxel grid size in mm for overlap computation.

    Returns:
        Dict with shared_workspace_bounds, overlap_ratio, arm1_bounds, arm2_bounds.
    """
    ws1 = compute_workspace(arm1_assembly, n_samples=n_samples)
    ws2 = compute_workspace(arm2_assembly, n_samples=n_samples)

    def _voxelize(points: list[WorkspacePoint], grid: float) -> set[tuple[int, int, int]]:
        voxels = set()
        for p in points:
            vx = int(p.x / grid)
            vy = int(p.y / grid)
            vz = int(p.z / grid)
            voxels.add((vx, vy, vz))
        return voxels

    vox1 = _voxelize(ws1.points, grid_size)
    vox2 = _voxelize(ws2.points, grid_size)

    shared = vox1 & vox2
    total = vox1 | vox2
    overlap_ratio = len(shared) / max(len(total), 1)

    # Compute shared bounds
    if shared:
        sx = [v[0] * grid_size for v in shared]
        sy = [v[1] * grid_size for v in shared]
        sz = [v[2] * grid_size for v in shared]
        shared_bounds = {
            "x": [round(min(sx), 2), round(max(sx), 2)],
            "y": [round(min(sy), 2), round(max(sy), 2)],
            "z": [round(min(sz), 2), round(max(sz), 2)],
        }
    else:
        shared_bounds = {"x": [0, 0], "y": [0, 0], "z": [0, 0]}

    return {
        "arm1_bounds": ws1.bounds,
        "arm2_bounds": ws2.bounds,
        "arm1_max_reach": ws1.max_reach,
        "arm2_max_reach": ws2.max_reach,
        "shared_workspace_bounds": shared_bounds,
        "overlap_ratio": round(overlap_ratio, 4),
        "shared_voxels": len(shared),
        "total_voxels": len(total),
        "arm1_samples": len(ws1.points),
        "arm2_samples": len(ws2.points),
    }


# ---------------------------------------------------------------------------
# Tool: workspace_analysis
# ---------------------------------------------------------------------------


class WorkspaceAnalysisTool(Tool):
    """Analyze robot arm workspace and reachability."""

    name = "workspace_analysis"
    description = (
        "工作空间分析：计算机器人臂的可达空间范围、"
        "双臂共享工作区域、可达性热图数据。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "assembly_name": {
                        "type": "string",
                        "description": "装配体名称或 JSON",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["single", "dual"],
                        "description": "单臂(single)或双臂(dual)分析",
                    },
                    "arm2_name": {
                        "type": "string",
                        "description": "第二臂装配体名称（dual 模式必需）",
                    },
                    "n_samples": {
                        "type": "integer",
                        "description": "采样数量（默认 500）",
                    },
                },
                "required": ["assembly_name"],
            },
        )

    def execute(
        self,
        *,
        assembly_name: str = "robotic_arm",
        mode: str = "single",
        arm2_name: str = "",
        n_samples: int = 500,
        **kwargs: Any,
    ) -> str:
        from .assembly_solver import _resolve_assembly
        assembly = _resolve_assembly(assembly_name, "")
        if assembly is None:
            return f"Error: assembly '{assembly_name}' not found"

        if mode == "dual":
            if not arm2_name:
                arm2_name = assembly_name
            a2 = _resolve_assembly(arm2_name, "")
            if a2 is None:
                return f"Error: assembly '{arm2_name}' not found"
            result = compute_shared_workspace(assembly, a2, n_samples=n_samples)

            lines = [
                "[Workspace Analysis - Dual Arm]",
                f"Arm 1 reach: {result['arm1_max_reach']:.1f} mm",
                f"Arm 2 reach: {result['arm2_max_reach']:.1f} mm",
                f"Overlap ratio: {result['overlap_ratio']:.1%}",
                f"Shared voxels: {result['shared_voxels']} / {result['total_voxels']}",
                f"Shared bounds X: {result['shared_workspace_bounds']['x']}",
                f"Shared bounds Y: {result['shared_workspace_bounds']['y']}",
                f"Shared bounds Z: {result['shared_workspace_bounds']['z']}",
                "",
                "--- JSON ---",
                json.dumps(result, ensure_ascii=False, indent=2),
            ]
            return "\n".join(lines)

        # Single arm
        ws = compute_workspace(assembly, n_samples=n_samples)
        lines = [
            "[Workspace Analysis]",
            f"Samples: {ws.total_samples}",
            f"Max reach: {ws.max_reach:.1f} mm",
            f"Bounds X: {ws.bounds['x']}",
            f"Bounds Y: {ws.bounds['y']}",
            f"Bounds Z: {ws.bounds['z']}",
            f"Center: {ws.center}",
        ]

        lines.append("")
        lines.append("--- JSON ---")
        # Don't dump all points (too many), just summary
        summary = {
            "total_samples": ws.total_samples,
            "reachable_count": ws.reachable_count,
            "reachability_ratio": ws.reachability_ratio,
            "bounds": ws.bounds,
            "center": ws.center,
            "max_reach": ws.max_reach,
        }
        lines.append(json.dumps(summary, ensure_ascii=False, indent=2))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_workspace_tools(registry: Any) -> None:
    """Register workspace analysis tools."""
    registry.register(WorkspaceAnalysisTool())
