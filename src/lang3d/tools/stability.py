"""Stability analysis tools — support polygon, Force-Angle, ZMP, tip-over risk.

Task 48: Implements comprehensive stability analysis for robots and assemblies.
Key algorithms:
- Graham scan for convex hull (support polygon)
- Static stability margin (COM projection to polygon edge distance)
- Force-Angle stability measure (Papadopoulos, 1996)
- Zero Moment Point (ZMP) for dynamic stability
- Integrated tip-over risk assessment
"""

from __future__ import annotations

import json
import math
from typing import Any

from ..models.base import ToolDefinition
from .base import Tool


# ============================================================================
# Convex Hull — Graham Scan
# ============================================================================

def compute_support_polygon(
    contact_points: list[list[float]],
) -> list[list[float]]:
    """Compute the convex hull of ground contact points using Graham scan.

    Args:
        contact_points: List of [x, y] or [x, y, z] points (z ignored).

    Returns:
        Convex hull vertices in counter-clockwise order.
    """
    if not contact_points:
        return []

    # Project to 2D (x, y)
    pts = [(p[0], p[1]) for p in contact_points]

    if len(pts) <= 2:
        return [list(p) for p in pts]

    # Find lowest-y point (leftmost if tie)
    pivot = min(pts, key=lambda p: (p[1], p[0]))

    def _polar_angle(p: tuple[float, float]) -> float:
        return math.atan2(p[1] - pivot[1], p[0] - pivot[0])

    def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    sorted_pts = sorted(pts, key=lambda p: (_polar_angle(p), (p[0] - pivot[0])**2 + (p[1] - pivot[1])**2))

    hull: list[tuple[float, float]] = []
    for p in sorted_pts:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)

    return [list(p) for p in hull]


# ============================================================================
# Static Stability
# ============================================================================

def compute_static_stability(
    com: list[float],
    support_polygon: list[list[float]],
) -> dict[str, Any]:
    """Compute static stability margin.

    Measures the minimum distance from the COM projection (x, y) to the
    nearest edge of the support polygon. Positive = stable, negative = unstable.

    Args:
        com: Center of mass [x, y, z] (z used for height info only).
        support_polygon: Convex hull vertices [[x,y], ...].

    Returns:
        Dict with stability margin, COM projection, and polygon info.
    """
    if not support_polygon:
        return {"stable": False, "margin_mm": -1, "error": "No support polygon"}

    com_2d = (com[0], com[1])
    n = len(support_polygon)

    # Check if COM is inside the polygon
    def _point_in_polygon(px: float, py: float, poly: list[list[float]]) -> bool:
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i][0], poly[i][1]
            xj, yj = poly[j][0], poly[j][1]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    is_inside = _point_in_polygon(com_2d[0], com_2d[1], support_polygon)

    # Minimum distance to polygon edges
    min_dist = float("inf")
    for i in range(n):
        x1, y1 = support_polygon[i]
        x2, y2 = support_polygon[(i + 1) % n]
        # Distance from point to line segment
        dx, dy = x2 - x1, y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12:
            dist = math.sqrt((com_2d[0] - x1)**2 + (com_2d[1] - y1)**2)
        else:
            t = max(0, min(1, ((com_2d[0] - x1) * dx + (com_2d[1] - y1) * dy) / seg_len_sq))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist = math.sqrt((com_2d[0] - proj_x)**2 + (com_2d[1] - proj_y)**2)
        min_dist = min(min_dist, dist)

    margin = min_dist if is_inside else -min_dist

    return {
        "stable": is_inside,
        "margin_mm": round(margin, 2),
        "com_projection": [round(com_2d[0], 2), round(com_2d[1], 2)],
        "polygon_area_mm2": round(_polygon_area(support_polygon), 2),
        "polygon_vertices": len(support_polygon),
    }


def _polygon_area(poly: list[list[float]]) -> float:
    """Shoelace formula for polygon area."""
    n = len(poly)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i][0] * poly[j][1]
        area -= poly[j][0] * poly[i][1]
    return abs(area) / 2.0


# ============================================================================
# Force-Angle Stability Measure (Papadopoulos, 1996)
# ============================================================================

def compute_force_angle_stability(
    com: list[float],
    contacts: list[list[float]],
    forces: list[list[float]] | None = None,
    gravity: float = 9.81,
    mass_kg: float = 10.0,
) -> dict[str, Any]:
    """Compute Force-Angle stability metric.

    The Force-Angle measure evaluates stability by computing the angle between
    the resultant force vector and the tip-over axis at each contact edge.
    Positive = stable, negative = tip-over imminent.

    Args:
        com: Center of mass [x, y, z] in mm.
        contacts: Ground contact points [[x, y, z], ...] in mm.
        forces: External forces at contacts [[fx, fy, fz], ...] in N. If None, gravity only.
        gravity: Gravitational acceleration in m/s².
        mass_kg: Total mass in kg.

    Returns:
        Dict with Force-Angle metric, per-edge angles, and stability status.
    """
    if len(contacts) < 2:
        return {"metric": 0, "stable": True, "error": "Need >= 2 contact points"}

    # Gravity force vector (in N)
    fg = [0, 0, -mass_kg * gravity]

    n = len(contacts)
    per_edge: list[dict[str, Any]] = []
    min_metric = float("inf")

    for i in range(n):
        j = (i + 1) % n
        ci = contacts[i]
        cj = contacts[j]

        # Tip-over axis: line from contact i to contact j (projected to horizontal)
        axis = [cj[0] - ci[0], cj[1] - ci[1], 0]
        axis_len = math.sqrt(sum(a**2 for a in axis))
        if axis_len < 1e-9:
            continue
        axis = [a / axis_len for a in axis]

        # Vector from axis midpoint to COM
        mid = [(ci[k] + cj[k]) / 2 for k in range(3)]
        r_com = [com[k] - mid[k] for k in range(3)]

        # Moment arm: cross product of r_com with axis
        cross = [
            r_com[1] * axis[2] - r_com[2] * axis[1],
            r_com[2] * axis[0] - r_com[0] * axis[2],
            r_com[0] * axis[1] - r_com[1] * axis[0],
        ]

        # Add external force at this contact if provided
        total_force = list(fg)
        if forces:
            for k in range(3):
                total_force[k] += forces[i][k] if i < len(forces) else 0

        # Angle between total force and the moment arm direction
        cross_len = math.sqrt(sum(c**2 for c in cross))
        force_len = math.sqrt(sum(f**2 for f in total_force))

        if cross_len < 1e-9 or force_len < 1e-9:
            angle_metric = 0
        else:
            cos_angle = sum(cross[k] * total_force[k] for k in range(3)) / (cross_len * force_len)
            cos_angle = max(-1, min(1, cos_angle))
            angle_metric = math.acos(cos_angle)

        per_edge.append({
            "edge": [i, j],
            "angle_deg": round(math.degrees(angle_metric), 2),
        })
        min_metric = min(min_metric, angle_metric)

    # Positive metric = stable
    stable = min_metric > 0
    return {
        "metric_deg": round(math.degrees(min_metric), 2) if min_metric != float("inf") else 0,
        "stable": stable,
        "per_edge": per_edge,
        "num_contact_edges": n,
    }


# ============================================================================
# Zero Moment Point (ZMP)
# ============================================================================

def compute_zmp(
    com: list[float],
    mass_kg: float,
    linear_accel: list[float] | None = None,
    angular_accel: list[float] | None = None,
    com_height_mm: float | None = None,
    gravity: float = 9.81,
) -> dict[str, Any]:
    """Compute Zero Moment Point (ZMP) for dynamic stability.

    ZMP is the point on the ground where the net moment of inertial and
    gravity forces is zero. If ZMP is inside support polygon, the robot
    is dynamically stable.

    Args:
        com: Center of mass [x, y, z] in mm.
        mass_kg: Total mass.
        linear_accel: Linear acceleration [ax, ay, az] in m/s².
        angular_accel: Angular acceleration [αx, αy, αz] in rad/s².
        com_height_mm: Height of COM above ground. If None, uses com[2].
        gravity: Gravitational acceleration.

    Returns:
        ZMP coordinates and distance from COM projection.
    """
    h = com_height_mm if com_height_mm is not None else com[2]
    ax = linear_accel[0] if linear_accel else 0.0
    ay = linear_accel[1] if linear_accel else 0.0

    # ZMP formula (simplified for flat ground):
    # zmp_x = com_x - (h / g) * ax
    # zmp_y = com_y - (h / g) * ay
    g = gravity
    # h in mm, ax/ay in m/s², g in m/s² → ax/g dimensionless → result in mm
    zmp_x = com[0] - h * ax / g
    zmp_y = com[1] - h * ay / g

    # Distance from COM projection to ZMP
    dx = zmp_x - com[0]
    dy = zmp_y - com[1]
    dist = math.sqrt(dx * dx + dy * dy)

    return {
        "zmp_mm": [round(zmp_x, 2), round(zmp_y, 2)],
        "com_projection_mm": [round(com[0], 2), round(com[1], 2)],
        "offset_mm": round(dist, 2),
        "com_height_mm": h,
        "linear_accel": [ax, ay, linear_accel[2] if linear_accel and len(linear_accel) > 2 else 0],
    }


# ============================================================================
# Tip-Over Risk Assessment
# ============================================================================

def check_tip_over_risk(
    com: list[float],
    contact_points: list[list[float]],
    mass_kg: float = 10.0,
    linear_accel: list[float] | None = None,
    safety_threshold_mm: float = 20.0,
) -> dict[str, Any]:
    """Comprehensive tip-over risk assessment.

    Combines static stability, Force-Angle metric, and dynamic ZMP analysis
    to produce an overall risk rating.

    Returns:
        Dict with risk level, static/dynamic metrics, and recommendations.
    """
    # 1. Support polygon
    polygon = compute_support_polygon(contact_points)

    # 2. Static stability
    static = compute_static_stability(com, polygon)

    # 3. Force-Angle
    fa = compute_force_angle_stability(com, contact_points, mass_kg=mass_kg)

    # 4. ZMP (dynamic)
    zmp = compute_zmp(com, mass_kg, linear_accel=linear_accel)

    # 5. Risk assessment
    risk_score = 0  # 0 = safe, higher = riskier
    risk_factors: list[str] = []

    if not static["stable"]:
        risk_score += 100
        risk_factors.append("COM在支撑多边形外")
    elif static["margin_mm"] < safety_threshold_mm:
        risk_score += 30
        risk_factors.append(f"静态稳定裕度不足 ({static['margin_mm']:.1f}mm < {safety_threshold_mm}mm)")

    if zmp["offset_mm"] > safety_threshold_mm * 2:
        risk_score += 20
        risk_factors.append(f"ZMP偏移过大 ({zmp['offset_mm']:.1f}mm)")

    if fa.get("metric_deg", 90) < 15:
        risk_score += 25
        risk_factors.append(f"Force-Angle裕度不足 ({fa['metric_deg']:.1f}°)")

    # Risk level
    if risk_score >= 100:
        risk_level = "CRITICAL"
    elif risk_score >= 50:
        risk_level = "HIGH"
    elif risk_score >= 20:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_factors": risk_factors,
        "static_stability": static,
        "force_angle": fa,
        "zmp": zmp,
        "support_polygon": polygon,
    }


# ============================================================================
# Stability Report Generator
# ============================================================================

def generate_stability_report(
    com: list[float],
    contact_points: list[list[float]],
    mass_kg: float = 10.0,
    linear_accel: list[float] | None = None,
    safety_threshold_mm: float = 20.0,
) -> str:
    """Generate a human-readable stability analysis report."""
    risk = check_tip_over_risk(
        com, contact_points, mass_kg, linear_accel, safety_threshold_mm
    )

    lines = [
        "# 稳定性分析报告",
        "",
        f"**质心位置**: ({com[0]:.1f}, {com[1]:.1f}, {com[2]:.1f}) mm",
        f"**总质量**: {mass_kg:.2f} kg",
        f"**接触点数**: {len(contact_points)}",
        f"**风险等级**: {risk['risk_level']} (分数: {risk['risk_score']})",
        "",
        "## 静态稳定性",
        f"- 稳定: {'是' if risk['static_stability']['stable'] else '否'}",
        f"- 稳定裕度: {risk['static_stability']['margin_mm']:.1f} mm",
        f"- 支撑多边形面积: {risk['static_stability']['polygon_area_mm2']:.0f} mm²",
        "",
        "## Force-Angle 稳定性",
        f"- 最小角度: {risk['force_angle'].get('metric_deg', 'N/A')}°",
        f"- 稳定: {'是' if risk['force_angle']['stable'] else '否'}",
        "",
        "## ZMP 分析",
        f"- ZMP: ({risk['zmp']['zmp_mm'][0]:.1f}, {risk['zmp']['zmp_mm'][1]:.1f}) mm",
        f"- COM投影: ({risk['zmp']['com_projection_mm'][0]:.1f}, {risk['zmp']['com_projection_mm'][1]:.1f}) mm",
        f"- 偏移: {risk['zmp']['offset_mm']:.1f} mm",
        "",
    ]

    if risk["risk_factors"]:
        lines.append("## 风险因素")
        for f in risk["risk_factors"]:
            lines.append(f"- {f}")
        lines.append("")

    # Recommendations
    lines.append("## 建议")
    if risk["risk_level"] == "CRITICAL":
        lines.append("- **必须重新设计**：质心在支撑多边形外，存在翻倒风险")
        lines.append("- 建议加宽底盘或降低重心")
    elif risk["risk_level"] == "HIGH":
        lines.append("- 建议加宽轮距或增加底盘质量")
        lines.append("- 考虑降低工控机/电池安装高度")
    elif risk["risk_level"] == "MEDIUM":
        lines.append("- 稳定性可接受但裕度较小")
        lines.append("- 建议验证动态工况下的稳定性")
    else:
        lines.append("- 稳定性良好，设计合理")

    return "\n".join(lines)


# ============================================================================
# Tool: stability_analysis
# ============================================================================

class StabilityAnalysisTool(Tool):
    """Comprehensive stability analysis for robots and assemblies."""

    name = "stability_analysis"
    description = (
        "分析机器人或装配体的稳定性。计算支撑多边形、静态稳定裕度、"
        "Force-Angle 指标、ZMP，并生成翻倒风险评估报告。"
        "输入质心坐标和接触点，输出稳定性分析结果。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "com": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "质心坐标 [x, y, z] mm",
                    },
                    "contact_points": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "description": "地面接触点 [[x,y,z], ...] mm",
                    },
                    "mass_kg": {
                        "type": "number",
                        "description": "总质量 kg (默认 10)",
                    },
                    "linear_accel": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "线加速度 [ax, ay, az] m/s² (可选)",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["full", "static", "dynamic", "report"],
                        "description": "分析模式: full(默认)/static/dynamic/report",
                    },
                },
                "required": ["com", "contact_points"],
            },
        )

    def execute(
        self,
        *,
        com: list[float],
        contact_points: list[list[float]],
        mass_kg: float = 10.0,
        linear_accel: list[float] | None = None,
        mode: str = "full",
        **kwargs: Any,
    ) -> str:
        if mode == "report":
            return generate_stability_report(com, contact_points, mass_kg, linear_accel)

        if mode == "static":
            polygon = compute_support_polygon(contact_points)
            result = compute_static_stability(com, polygon)
            return json.dumps(result, indent=2, ensure_ascii=False)

        if mode == "dynamic":
            result = compute_zmp(com, mass_kg, linear_accel=linear_accel)
            return json.dumps(result, indent=2, ensure_ascii=False)

        # mode == "full"
        result = check_tip_over_risk(com, contact_points, mass_kg, linear_accel)
        return json.dumps(result, indent=2, ensure_ascii=False)


def register_stability_tools(registry: Any) -> None:
    """Register stability analysis tools."""
    registry.register(StabilityAnalysisTool())
