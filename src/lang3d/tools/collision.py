"""Collision detection for robotic arms using capsule models and GJK algorithm.

Provides:
  - Capsule: simple geometric body (line segment + radius)
  - build_capsule_model: convert arm links to capsule list
  - gjk_distance: Gilbert-Johnson-Keerthi distance algorithm
  - check_self_collision: detect inter-arm and intra-arm collisions
  - CollisionCheckTool: Agent tool for collision queries
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from ..models.base import ToolDefinition
from .base import Tool


# ---------------------------------------------------------------------------
# Capsule geometry
# ---------------------------------------------------------------------------

@dataclass
class Capsule:
    """A capsule (swept sphere) defined by a line segment and radius.

    Used as a simplified collision body for robot links.
    """
    name: str
    start: tuple[float, float, float]  # (x, y, z) of segment start
    end: tuple[float, float, float]    # (x, y, z) of segment end
    radius: float                      # capsule radius in mm


def build_capsule_model(
    links: list[Any],
    joint_positions: list[tuple[float, float, float]],
    default_radius: float = 15.0,
    radius_map: dict[str, float] | None = None,
) -> list[Capsule]:
    """Build capsule collision model from arm links and joint positions.

    Each link becomes a capsule from its parent joint to its child joint,
    with a configurable radius (default 15mm for typical robot arm links).

    Args:
        links: List of LinkSegment objects (must have .name attribute).
        joint_positions: List of (x, y, z) positions for each joint.
        default_radius: Default capsule radius in mm.
        radius_map: Optional per-link radius overrides {link_name: radius}.

    Returns:
        List of Capsule objects, one per link.
    """
    radius_map = radius_map or {}
    capsules: list[Capsule] = []

    for i, link in enumerate(links):
        start = joint_positions[i] if i < len(joint_positions) else (0, 0, 0)
        end = joint_positions[i + 1] if i + 1 < len(joint_positions) else start
        radius = radius_map.get(link.name, default_radius)
        capsules.append(Capsule(
            name=link.name,
            start=start,
            end=end,
            radius=radius,
        ))

    return capsules


# ---------------------------------------------------------------------------
# GJK distance algorithm (simplified for capsule-capsule)
# ---------------------------------------------------------------------------

def _support_point(
    capsule: Capsule, direction: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Find the farthest point on a capsule in the given direction."""
    d_len = math.sqrt(
        (capsule.end[0] - capsule.start[0]) ** 2 +
        (capsule.end[1] - capsule.start[1]) ** 2 +
        (capsule.end[2] - capsule.start[2]) ** 2
    )
    if d_len < 1e-10:
        # Degenerate capsule (point)
        base = capsule.start
    else:
        # Project direction onto segment axis
        dx = capsule.end[0] - capsule.start[0]
        dy = capsule.end[1] - capsule.start[1]
        dz = capsule.end[2] - capsule.start[2]
        t = (direction[0] * dx + direction[1] * dy + direction[2] * dz) / (d_len * d_len)
        t = max(0.0, min(1.0, t))
        base = (
            capsule.start[0] + t * dx,
            capsule.start[1] + t * dy,
            capsule.start[2] + t * dz,
        )
    # Add radius in direction
    dir_len = math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
    if dir_len < 1e-10:
        return base
    return (
        base[0] + direction[0] / dir_len * capsule.radius,
        base[1] + direction[1] / dir_len * capsule.radius,
        base[2] + direction[2] / dir_len * capsule.radius,
    )


def _vec_sub(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_add(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_scale(v: tuple[float, ...], s: float) -> tuple[float, float, float]:
    return (v[0] * s, v[1] * s, v[2] * s)


def _closest_point_on_segment(
    p: tuple[float, float, float],
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Find the closest point on segment AB to point P."""
    ab = _vec_sub(b, a)
    ap = _vec_sub(p, a)
    ab_len_sq = ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2
    if ab_len_sq < 1e-12:
        return a
    t = max(0.0, min(1.0, (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / ab_len_sq))
    return (a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2])


def capsule_distance(a: Capsule, b: Capsule) -> float:
    """Compute the minimum distance between two capsules.

    Returns negative value if capsules overlap.
    """
    # Find closest points between the two line segments
    # Use iterative approach: alternate closest point computation
    pa = a.start
    pb = b.start
    for _ in range(10):
        pb_new = _closest_point_on_segment(pa, b.start, b.end)
        pa_new = _closest_point_on_segment(pb_new, a.start, a.end)
        pa = pa_new
        pb = pb_new

    dist = math.sqrt(
        (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 + (pa[2] - pb[2]) ** 2
    )
    return dist - a.radius - b.radius


def gjk_distance(capsule_a: Capsule, capsule_b: Capsule) -> float:
    """GJK-inspired distance between two capsules.

    Uses the capsule_distance function which computes segment-segment
    distance minus radii. Returns negative if overlapping.
    """
    return capsule_distance(capsule_a, capsule_b)


def check_self_collision(
    arm1_capsules: list[Capsule],
    arm2_capsules: list[Capsule] | None = None,
    safety_margin: float = 10.0,
) -> tuple[bool, float, list[dict[str, Any]]]:
    """Check for collisions between arm capsules.

    Args:
        arm1_capsules: Capsules for arm 1.
        arm2_capsules: Optional capsules for arm 2 (inter-arm check).
        safety_margin: Minimum safe clearance in mm.

    Returns:
        (collision_free, min_clearance, collision_pairs)
        where collision_pairs is a list of {link_a, link_b, distance}.
    """
    min_clearance = float("inf")
    collision_free = True
    collision_pairs: list[dict[str, Any]] = []

    def _check_pair(ca: Capsule, cb: Capsule) -> None:
        nonlocal min_clearance, collision_free
        # Skip adjacent links (they always share a joint)
        dist = capsule_distance(ca, cb)
        if dist < min_clearance:
            min_clearance = dist
        if dist < safety_margin:
            collision_free = False
            collision_pairs.append({
                "link_a": ca.name,
                "link_b": cb.name,
                "distance_mm": round(dist, 2),
                "status": "collision" if dist < 0 else "warning",
            })

    # Intra-arm collision (non-adjacent pairs)
    n = len(arm1_capsules)
    for i in range(n):
        for j in range(i + 2, n):  # skip adjacent (i+1)
            _check_pair(arm1_capsules[i], arm1_capsules[j])

    # Inter-arm collision
    if arm2_capsules:
        for ca in arm1_capsules:
            for cb in arm2_capsules:
                _check_pair(ca, cb)

    return collision_free, round(min_clearance, 2), collision_pairs


# ---------------------------------------------------------------------------
# Tool: collision_check
# ---------------------------------------------------------------------------


class CollisionCheckTool(Tool):
    """Check for collisions between robotic arm links."""

    name = "collision_check"
    description = (
        "碰撞检测：使用胶囊体模型检测机器人臂的自碰撞和双臂间碰撞。"
        "支持 GJK 距离算法和安全边距设置。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "arm1_capsules": {
                        "type": "array",
                        "description": "左臂胶囊体列表 [{name, start:[x,y,z], end:[x,y,z], radius}]",
                        "items": {"type": "object"},
                    },
                    "arm2_capsules": {
                        "type": "array",
                        "description": "右臂胶囊体列表（可选，用于双臂碰撞检测）",
                        "items": {"type": "object"},
                    },
                    "safety_margin": {
                        "type": "number",
                        "description": "安全间距 mm（默认 10）",
                    },
                },
                "required": ["arm1_capsules"],
            },
        )

    def execute(
        self,
        *,
        arm1_capsules: list[dict] | None = None,
        arm2_capsules: list[dict] | None = None,
        safety_margin: float = 10.0,
        **kwargs: Any,
    ) -> str:
        if arm1_capsules is None:
            return "Error: arm1_capsules is required"

        def _parse_capsules(data: list[dict]) -> list[Capsule]:
            result = []
            for c in data:
                result.append(Capsule(
                    name=c.get("name", ""),
                    start=tuple(c.get("start", [0, 0, 0])),
                    end=tuple(c.get("end", [0, 0, 0])),
                    radius=c.get("radius", 15.0),
                ))
            return result

        c1 = _parse_capsules(arm1_capsules)
        c2 = _parse_capsules(arm2_capsules) if arm2_capsules else None

        free, min_clear, pairs = check_self_collision(c1, c2, safety_margin)

        lines = [
            f"[Collision Check]",
            f"Collision-free: {'Yes' if free else 'NO'}",
            f"Min clearance: {min_clear:.2f} mm",
            f"Safety margin: {safety_margin:.1f} mm",
            f"Collision pairs: {len(pairs)}",
        ]
        for p in pairs:
            lines.append(f"  {p['link_a']} <-> {p['link_b']}: {p['distance_mm']:.2f} mm [{p['status']}]")

        lines.append("")
        lines.append("--- JSON ---")
        lines.append(json.dumps({
            "collision_free": free,
            "min_clearance_mm": min_clear,
            "safety_margin": safety_margin,
            "collision_pairs": pairs,
        }, ensure_ascii=False, indent=2))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_collision_tools(registry: Any) -> None:
    """Register collision detection tools."""
    registry.register(CollisionCheckTool())
