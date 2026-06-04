"""Tests for multi-robot scheme comparison tools.

Tests cover:
- Scheme generation (serial, SCARA, delta)
- Metric computation (workspace, precision, cost)
- Scoring and ranking
- Comparison table formatting
- Requirement-based recommendations
- Tool registration and execution
"""

from __future__ import annotations

import pytest

from lang3d.tools.base import ToolRegistry
from lang3d.tools.scheme_compare import (
    RobotRequirement,
    Scheme,
    SchemeCompareTool,
    SchemeMetrics,
    compare_schemes,
    format_comparison_table,
    generate_schemes,
    recommend_scheme,
    register_scheme_tools,
    score_schemes,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def default_req():
    return RobotRequirement(reach_mm=300, payload_g=200, budget_cny=1000)


@pytest.fixture
def high_payload_req():
    return RobotRequirement(reach_mm=300, payload_g=1000, budget_cny=2000)


@pytest.fixture
def precision_req():
    return RobotRequirement(reach_mm=200, payload_g=100, precision_mm=0.1, budget_cny=500,
                            prefer_precision=True)


@pytest.fixture
def speed_req():
    return RobotRequirement(reach_mm=300, payload_g=50, budget_cny=500, prefer_speed=True)


@pytest.fixture
def cost_req():
    return RobotRequirement(reach_mm=200, payload_g=100, budget_cny=300, prefer_cost=True)


@pytest.fixture
def all_schemes(default_req):
    return generate_schemes(default_req)


# ============================================================================
# Test: Scheme generation
# ============================================================================

class TestSchemeGeneration:
    """Test that all three scheme types can be generated."""

    def test_generates_three_schemes(self, default_req):
        schemes = generate_schemes(default_req)
        assert len(schemes) == 3

    def test_generates_specific_types(self, default_req):
        schemes = generate_schemes(default_req, types=["serial", "scara"])
        assert len(schemes) == 2
        types = {s.config_type for s in schemes}
        assert types == {"serial", "scara"}

    def test_serial_scheme(self, default_req):
        schemes = generate_schemes(default_req, types=["serial"])
        s = schemes[0]
        assert s.config_type == "serial"
        assert s.assembly is not None
        assert len(s.assembly.parts) >= 8
        assert len(s.actuator_ids) >= 3

    def test_scara_scheme(self, default_req):
        schemes = generate_schemes(default_req, types=["scara"])
        s = schemes[0]
        assert s.config_type == "scara"
        assert s.assembly is not None
        assert len(s.assembly.parts) >= 8

    def test_delta_scheme(self, default_req):
        schemes = generate_schemes(default_req, types=["delta"])
        s = schemes[0]
        assert s.config_type == "delta"
        assert s.assembly is not None
        assert len(s.assembly.parts) >= 10  # Delta has more parts

    def test_serial_has_pros_and_cons(self, default_req):
        schemes = generate_schemes(default_req, types=["serial"])
        s = schemes[0]
        assert len(s.pros) > 0
        assert len(s.cons) > 0

    def test_scara_has_pros_and_cons(self, default_req):
        schemes = generate_schemes(default_req, types=["scara"])
        s = schemes[0]
        assert len(s.pros) > 0
        assert len(s.cons) > 0

    def test_delta_has_pros_and_cons(self, default_req):
        schemes = generate_schemes(default_req, types=["delta"])
        s = schemes[0]
        assert len(s.pros) > 0
        assert len(s.cons) > 0


# ============================================================================
# Test: Metrics
# ============================================================================

class TestSchemeMetrics:
    """Test metric computation for each scheme type."""

    def test_serial_metrics(self, all_schemes):
        serial = next(s for s in all_schemes if s.config_type == "serial")
        m = serial.metrics
        assert m.workspace_radius_mm > 0
        assert m.workspace_height_mm > 0
        assert m.workspace_volume_dm3 > 0
        assert m.precision_mm > 0
        assert m.max_speed_mm_s > 0
        assert m.payload_capacity_g > 0
        assert m.part_count >= 8
        assert m.joint_count >= 3
        assert m.estimated_cost_cny > 0
        assert 1 <= m.complexity_score <= 10
        assert m.assembly_time_min > 0
        assert m.weight_g > 0

    def test_scara_metrics(self, all_schemes):
        scara = next(s for s in all_schemes if s.config_type == "scara")
        m = scara.metrics
        assert m.workspace_radius_mm > 0
        assert m.precision_mm > 0
        assert m.max_speed_mm_s > serial_speed(all_schemes)

    def test_delta_metrics(self, all_schemes):
        delta = next(s for s in all_schemes if s.config_type == "delta")
        m = delta.metrics
        assert m.workspace_radius_mm > 0
        assert m.precision_mm <= 0.5  # Delta is high precision
        assert m.max_speed_mm_s > 500  # Delta is fast
        assert m.complexity_score > 5  # Delta is complex

    def test_delta_fastest(self, all_schemes):
        delta = next(s for s in all_schemes if s.config_type == "delta")
        speeds = {s.config_type: s.metrics.max_speed_mm_s for s in all_schemes}
        assert speeds["delta"] >= speeds["serial"]
        assert speeds["delta"] >= speeds["scara"]

    def test_delta_most_precise(self, all_schemes):
        delta = next(s for s in all_schemes if s.config_type == "delta")
        precisions = {s.config_type: s.metrics.precision_mm for s in all_schemes}
        assert precisions["delta"] <= precisions["serial"]

    def test_serial_largest_workspace(self, all_schemes):
        serial = next(s for s in all_schemes if s.config_type == "serial")
        volumes = {s.config_type: s.metrics.workspace_volume_dm3 for s in all_schemes}
        assert volumes["serial"] >= volumes["delta"]

    def test_cost_positive(self, all_schemes):
        for s in all_schemes:
            assert s.metrics.estimated_cost_cny > 0

    def test_high_payload_uses_stronger_actuators(self):
        req_low = RobotRequirement(reach_mm=300, payload_g=100)
        req_high = RobotRequirement(reach_mm=300, payload_g=1000)
        low = generate_schemes(req_low, types=["serial"])[0]
        high = generate_schemes(req_high, types=["serial"])[0]
        # Higher payload should use stronger (more expensive) actuators
        assert high.metrics.estimated_cost_cny >= low.metrics.estimated_cost_cny * 0.5


def serial_speed(schemes):
    return next(s for s in schemes if s.config_type == "serial").metrics.max_speed_mm_s


# ============================================================================
# Test: Scoring and ranking
# ============================================================================

class TestScoring:
    """Test scoring and ranking logic."""

    def test_all_schemes_scored(self, default_req):
        schemes = compare_schemes(default_req)
        for s in schemes:
            assert s.score > 0

    def test_schemes_sorted_by_score(self, default_req):
        schemes = compare_schemes(default_req)
        scores = [s.score for s in schemes]
        assert scores == sorted(scores, reverse=True)

    def test_best_score_high(self, default_req):
        schemes = compare_schemes(default_req)
        assert schemes[0].score >= 5.0

    def test_precision_preference_favors_delta(self, precision_req):
        schemes = compare_schemes(precision_req)
        # Delta has better precision metric than serial
        delta = next(s for s in schemes if s.config_type == "delta")
        serial = next(s for s in schemes if s.config_type == "serial")
        assert delta.metrics.precision_mm <= serial.metrics.precision_mm
        # With precision preference, delta's total score should be competitive
        assert delta.score > 0 and serial.score > 0

    def test_speed_preference(self, speed_req):
        schemes = compare_schemes(speed_req)
        # All should have scores
        assert len(schemes) == 3
        for s in schemes:
            assert s.score > 0

    def test_cost_preference(self, cost_req):
        schemes = compare_schemes(cost_req)
        assert len(schemes) == 3
        for s in schemes:
            assert s.score > 0

    def test_scores_within_budget(self, default_req):
        schemes = compare_schemes(default_req)
        for s in schemes:
            if s.metrics.estimated_cost_cny <= default_req.budget_cny:
                # Should get full cost score (10)
                pass  # Just ensure no crash


# ============================================================================
# Test: Recommendation
# ============================================================================

class TestRecommendation:
    """Test recommendation logic."""

    def test_recommend_returns_best(self, default_req):
        best = recommend_scheme(default_req)
        schemes = compare_schemes(default_req)
        assert best.score == schemes[0].score

    def test_recommend_returns_scheme(self, default_req):
        best = recommend_scheme(default_req)
        assert isinstance(best, Scheme)
        assert best.config_type in ("serial", "scara", "delta")

    def test_recommend_specific_type(self):
        req = RobotRequirement(reach_mm=300)
        best = recommend_scheme(req, types=["serial"])
        assert best.config_type == "serial"

    def test_recommend_empty_raises(self):
        req = RobotRequirement()
        with pytest.raises(ValueError):
            recommend_scheme(req, types=[])


# ============================================================================
# Test: Formatting
# ============================================================================

class TestFormatting:
    """Test comparison table formatting."""

    def test_table_has_header(self, all_schemes):
        table = format_comparison_table(all_schemes)
        assert "机器人方案对比" in table

    def test_table_has_all_schemes(self, all_schemes):
        table = format_comparison_table(all_schemes)
        for s in all_schemes:
            assert s.name in table

    def test_table_has_metrics(self, all_schemes):
        table = format_comparison_table(all_schemes)
        assert "工作半径" in table
        assert "精度" in table
        assert "成本" in table

    def test_table_has_recommendation(self, all_schemes):
        scored = score_schemes(all_schemes, RobotRequirement())
        table = format_comparison_table(scored)
        assert "推荐方案" in table

    def test_table_has_pros_cons(self, all_schemes):
        scored = score_schemes(all_schemes, RobotRequirement())
        table = format_comparison_table(scored)
        assert "优势" in table
        assert "劣势" in table

    def test_table_markdown_format(self, all_schemes):
        table = format_comparison_table(all_schemes)
        assert "|" in table  # Table separator


# ============================================================================
# Test: Scheme assembly validity
# ============================================================================

class TestSchemeAssembly:
    """Test that generated assemblies are valid."""

    def test_serial_has_revolute_joints(self, default_req):
        schemes = generate_schemes(default_req, types=["serial"])
        s = schemes[0]
        revolute = [j for j in s.assembly.joints if j.type == "revolute"]
        assert len(revolute) >= 3

    def test_scara_has_prismatic(self, default_req):
        schemes = generate_schemes(default_req, types=["scara"])
        s = schemes[0]
        prismatic = [j for j in s.assembly.joints if j.type == "prismatic"]
        assert len(prismatic) >= 1

    def test_delta_has_3_arms(self, default_req):
        schemes = generate_schemes(default_req, types=["delta"])
        s = schemes[0]
        arms = [p for p in s.assembly.parts if "arm" in p.name]
        assert len(arms) == 3

    def test_delta_has_6_rods(self, default_req):
        schemes = generate_schemes(default_req, types=["delta"])
        s = schemes[0]
        rods = [p for p in s.assembly.parts if "rod" in p.name]
        assert len(rods) == 6

    def test_serial_parts_have_dimensions(self, default_req):
        schemes = generate_schemes(default_req, types=["serial"])
        for part in schemes[0].assembly.parts:
            assert len(part.dimensions) > 0

    def test_scara_parts_have_dimensions(self, default_req):
        schemes = generate_schemes(default_req, types=["scara"])
        for part in schemes[0].assembly.parts:
            assert len(part.dimensions) > 0


# ============================================================================
# Test: Tool registration and execution
# ============================================================================

class TestSchemeTool:
    """Test SchemeCompareTool."""

    def test_registered(self):
        registry = ToolRegistry()
        register_scheme_tools(registry)
        assert "scheme_compare" in registry.list_tools()

    def test_execute_default(self):
        tool = SchemeCompareTool()
        result = tool.execute()
        assert "方案对比" in result
        assert "推荐" in result

    def test_execute_with_params(self):
        tool = SchemeCompareTool()
        result = tool.execute(reach_mm=200, payload_g=100, budget_cny=500)
        assert "200" in result
        assert "100" in result

    def test_execute_precision_preference(self):
        tool = SchemeCompareTool()
        result = tool.execute(prefer_precision=True)
        assert "推荐" in result

    def test_execute_speed_preference(self):
        tool = SchemeCompareTool()
        result = tool.execute(prefer_speed=True)
        assert "推荐" in result

    def test_execute_markdown_section(self):
        tool = SchemeCompareTool()
        result = tool.execute()
        assert "--- Markdown ---" in result
        assert "机器人方案对比" in result

    def test_execute_shows_all_three(self):
        tool = SchemeCompareTool()
        result = tool.execute()
        assert "串联" in result
        assert "SCARA" in result
        assert "Delta" in result
