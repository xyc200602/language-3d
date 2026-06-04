"""Tests for stability analysis tools (Task 48).

Covers: convex hull (Graham scan), static stability, Force-Angle metric,
ZMP computation, tip-over risk assessment, report generation, tool execution.
"""

import json
import math
import pytest

from lang3d.tools.stability import (
    StabilityAnalysisTool,
    check_tip_over_risk,
    compute_force_angle_stability,
    compute_static_stability,
    compute_support_polygon,
    compute_zmp,
    generate_stability_report,
    register_stability_tools,
)


# ── Convex Hull Tests ──────────────────────────────────────────


class TestConvexHull:
    def test_triangle(self):
        pts = [[0, 0], [100, 0], [50, 50]]
        hull = compute_support_polygon(pts)
        assert len(hull) == 3

    def test_square(self):
        pts = [[0, 0], [100, 0], [100, 100], [0, 100]]
        hull = compute_support_polygon(pts)
        assert len(hull) == 4

    def test_interior_points_excluded(self):
        pts = [[0, 0], [100, 0], [100, 100], [0, 100], [50, 50]]
        hull = compute_support_polygon(pts)
        assert len(hull) == 4

    def test_collinear_points(self):
        pts = [[0, 0], [50, 0], [100, 0]]
        hull = compute_support_polygon(pts)
        assert len(hull) == 2  # degenerate

    def test_single_point(self):
        hull = compute_support_polygon([[50, 50]])
        assert len(hull) == 1

    def test_empty(self):
        hull = compute_support_polygon([])
        assert hull == []

    def test_3d_points_projected(self):
        pts = [[0, 0, 10], [100, 0, 10], [50, 100, 10]]
        hull = compute_support_polygon(pts)
        assert len(hull) == 3

    def test_many_points(self):
        # Pentagon with interior points
        import math
        pts = []
        for i in range(5):
            angle = 2 * math.pi * i / 5
            pts.append([100 * math.cos(angle), 100 * math.sin(angle)])
        pts.append([0, 0])  # interior
        pts.append([20, 20])  # interior
        hull = compute_support_polygon(pts)
        assert len(hull) == 5


# ── Static Stability Tests ─────────────────────────────────────


class TestStaticStability:
    def test_com_inside_square(self):
        poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
        result = compute_static_stability([50, 50, 50], poly)
        assert result["stable"] is True
        assert result["margin_mm"] > 0

    def test_com_outside_square(self):
        poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
        result = compute_static_stability([200, 50, 50], poly)
        assert result["stable"] is False
        assert result["margin_mm"] < 0

    def test_com_near_edge(self):
        poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
        result = compute_static_stability([98, 50, 50], poly)
        assert result["stable"] is True
        assert result["margin_mm"] < 5  # very close to edge

    def test_polygon_area(self):
        poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
        result = compute_static_stability([50, 50, 50], poly)
        assert abs(result["polygon_area_mm2"] - 10000) < 1

    def test_empty_polygon(self):
        result = compute_static_stability([50, 50, 50], [])
        assert result["stable"] is False


# ── Force-Angle Tests ──────────────────────────────────────────


class TestForceAngle:
    def test_stable_configuration(self):
        com = [50, 50, 100]
        contacts = [[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]]
        result = compute_force_angle_stability(com, contacts, mass_kg=10)
        assert result["stable"] is True
        assert result["metric_deg"] > 0

    def test_needs_multiple_contacts(self):
        result = compute_force_angle_stability([50, 50, 100], [[0, 0, 0]])
        assert "error" in result

    def test_centered_vs_offset_com(self):
        com_center = [50, 50, 100]
        com_offset = [90, 50, 100]  # near edge
        contacts = [[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]]
        r_center = compute_force_angle_stability(com_center, contacts, mass_kg=10)
        r_offset = compute_force_angle_stability(com_offset, contacts, mass_kg=10)
        # Offset COM should have smaller minimum Force-Angle metric
        assert r_offset["metric_deg"] < r_center["metric_deg"]


# ── ZMP Tests ──────────────────────────────────────────────────


class TestZMP:
    def test_stationary(self):
        com = [50, 50, 100]
        result = compute_zmp(com, mass_kg=10, linear_accel=[0, 0, 0])
        # Stationary: ZMP should be at COM projection
        assert abs(result["zmp_mm"][0] - 50) < 1
        assert abs(result["zmp_mm"][1] - 50) < 1
        assert result["offset_mm"] < 1

    def test_acceleration_shifts_zmp(self):
        com = [50, 50, 100]
        result = compute_zmp(com, mass_kg=10, linear_accel=[2, 0, 0])
        # Forward acceleration should shift ZMP backward
        assert result["zmp_mm"][0] != 50
        assert result["offset_mm"] > 0

    def test_no_accel(self):
        com = [100, 200, 300]
        result = compute_zmp(com, mass_kg=5)
        assert result["zmp_mm"] == [100, 200]
        assert result["offset_mm"] < 0.01

    def test_custom_height(self):
        com = [50, 50, 200]
        result = compute_zmp(com, mass_kg=10, com_height_mm=100, linear_accel=[1, 0, 0])
        assert result["com_height_mm"] == 100


# ── Tip-Over Risk Tests ────────────────────────────────────────


class TestTipOverRisk:
    def test_safe_configuration(self):
        com = [50, 50, 50]
        contacts = [[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]]
        result = check_tip_over_risk(com, contacts, mass_kg=5)
        assert result["risk_level"] in ("LOW", "MEDIUM")
        assert result["risk_score"] < 50

    def test_unstable_configuration(self):
        com = [500, 50, 200]  # way outside
        contacts = [[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]]
        result = check_tip_over_risk(com, contacts, mass_kg=10)
        assert result["risk_level"] in ("CRITICAL", "HIGH")
        assert not result["static_stability"]["stable"]

    def test_high_com_with_accel(self):
        com = [50, 50, 300]
        contacts = [[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]]
        result = check_tip_over_risk(
            com, contacts, mass_kg=20, linear_accel=[3, 0, 0]
        )
        # High COM + acceleration = higher risk
        assert result["risk_level"] in ("MEDIUM", "HIGH", "CRITICAL")

    def test_has_all_sections(self):
        com = [50, 50, 100]
        contacts = [[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]]
        result = check_tip_over_risk(com, contacts)
        assert "static_stability" in result
        assert "force_angle" in result
        assert "zmp" in result
        assert "support_polygon" in result


# ── Report Generation Tests ────────────────────────────────────


class TestReportGeneration:
    def test_report_is_string(self):
        com = [50, 50, 100]
        contacts = [[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]]
        report = generate_stability_report(com, contacts)
        assert isinstance(report, str)
        assert "稳定性分析报告" in report
        assert "风险等级" in report

    def test_report_contains_sections(self):
        com = [50, 50, 100]
        contacts = [[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]]
        report = generate_stability_report(com, contacts)
        assert "静态稳定性" in report
        assert "Force-Angle" in report
        assert "ZMP" in report
        assert "建议" in report


# ── Tool Execution Tests ──────────────────────────────────────


class TestStabilityAnalysisTool:
    def test_full_mode(self):
        tool = StabilityAnalysisTool()
        result = json.loads(tool.execute(
            com=[50, 50, 100],
            contact_points=[[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]],
            mass_kg=10,
            mode="full",
        ))
        assert "risk_level" in result
        assert "static_stability" in result

    def test_static_mode(self):
        tool = StabilityAnalysisTool()
        result = json.loads(tool.execute(
            com=[50, 50, 100],
            contact_points=[[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]],
            mode="static",
        ))
        assert "stable" in result
        assert "margin_mm" in result

    def test_dynamic_mode(self):
        tool = StabilityAnalysisTool()
        result = json.loads(tool.execute(
            com=[50, 50, 100],
            contact_points=[[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]],
            mass_kg=10,
            linear_accel=[1, 0, 0],
            mode="dynamic",
        ))
        assert "zmp_mm" in result

    def test_report_mode(self):
        tool = StabilityAnalysisTool()
        result = tool.execute(
            com=[50, 50, 100],
            contact_points=[[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]],
            mass_kg=10,
            mode="report",
        )
        assert isinstance(result, str)
        assert "稳定性分析报告" in result


# ── Registration Test ──────────────────────────────────────────


class TestRegistration:
    def test_register(self):
        registry = type("MockRegistry", (), {"register": lambda self, t: None})()
        register_stability_tools(registry)

    def test_tool_name(self):
        tool = StabilityAnalysisTool()
        assert tool.name == "stability_analysis"
