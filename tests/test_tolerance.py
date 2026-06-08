"""Tests for ISO tolerance system and tolerance stackup analysis.

Covers:
- ISO IT tolerance table lookup (IT5-IT12 across basic size ranges)
- Shaft fundamental deviations (f, g, h, js, k, p)
- Hole fundamental deviations (F, G, H, Js, K, P)
- Fit computation (clearance, transition, interference)
- Fit recommendations by application
- Tolerance-aware dimension helpers
- ToleranceStackup worst-case accumulation
- Integration with connection_features (tolerance-aware hole diameters)
"""

from __future__ import annotations

import pytest

from lang3d.knowledge.tolerance import (
    FitRecommendation,
    FitResult,
    bearing_seat_diameter,
    compute_fit,
    hole_deviations,
    it_tolerance,
    press_fit_bore_diameter,
    recommend_fit,
    shaft_deviations,
    tolerance_hole_diameter,
    tolerance_shaft_diameter,
    FIT_RECOMMENDATIONS,
)
from lang3d.tools.tolerance_analysis import (
    StackupResult,
    ToleranceDimension,
    ToleranceStackup,
    analyze_assembly_chain,
)


# ============================================================================
# 1. IT tolerance table
# ============================================================================

class TestITTolerance:
    """Standard IT grade tolerance values."""

    def test_it7_10mm(self):
        """IT7 for 10mm nominal = 15 µm = 0.015 mm."""
        assert it_tolerance(10.0, "IT7") == pytest.approx(0.015, abs=1e-4)

    def test_it6_30mm(self):
        """IT6 for 30mm nominal = 13 µm = 0.013 mm."""
        assert it_tolerance(30.0, "IT6") == pytest.approx(0.013, abs=1e-4)

    def test_it8_50mm(self):
        """IT8 for 50mm nominal = 39 µm = 0.039 mm."""
        assert it_tolerance(50.0, "IT8") == pytest.approx(0.039, abs=1e-4)

    def test_it12_3mm(self):
        """IT12 for 3mm nominal = 100 µm = 0.100 mm."""
        assert it_tolerance(3.0, "IT12") == pytest.approx(0.100, abs=1e-4)

    def test_it5_100mm(self):
        """IT5 for 100mm nominal = 15 µm = 0.015 mm."""
        assert it_tolerance(100.0, "IT5") == pytest.approx(0.015, abs=1e-4)

    def test_boundary_3mm_belongs_to_1_3(self):
        """3mm exactly should belong to 1-3 range (upper inclusive)."""
        assert it_tolerance(3.0, "IT7") == pytest.approx(0.010, abs=1e-4)

    def test_just_above_3mm(self):
        """3.1mm should be in 3-6 range."""
        assert it_tolerance(3.1, "IT7") == pytest.approx(0.012, abs=1e-4)

    def test_it_increases_with_grade(self):
        """Higher IT grades give larger tolerances."""
        for d in [5, 10, 25, 50, 100]:
            t5 = it_tolerance(d, "IT5")
            t7 = it_tolerance(d, "IT7")
            t12 = it_tolerance(d, "IT12")
            assert t5 < t7 < t12

    def test_it_increases_with_size(self):
        """Larger nominal diameters have larger tolerances for same grade."""
        t1 = it_tolerance(5.0, "IT7")
        t2 = it_tolerance(50.0, "IT7")
        t3 = it_tolerance(200.0, "IT7")
        assert t1 < t2 < t3

    def test_invalid_grade_raises(self):
        with pytest.raises(ValueError, match="Unknown IT grade"):
            it_tolerance(10.0, "IT99")

    def test_out_of_range_clamps(self):
        """Values outside 1-500mm should clamp to nearest range."""
        # Very small → same as 1-3 range
        t_small = it_tolerance(0.5, "IT7")
        t_range = it_tolerance(2.0, "IT7")
        assert t_small == t_range
        # Very large → same as 400-500 range
        t_large = it_tolerance(600.0, "IT7")
        t_max = it_tolerance(450.0, "IT7")
        assert t_large == t_max


# ============================================================================
# 2. Shaft fundamental deviations
# ============================================================================

class TestShaftDeviations:

    def test_h_shaft_es_is_zero(self):
        es, ei = shaft_deviations(10.0, "h", "IT6")
        assert es == pytest.approx(0.0)
        assert ei < 0  # ei = -IT

    def test_g_shaft_negative_es(self):
        """Shaft g has small negative upper deviation."""
        es, ei = shaft_deviations(10.0, "g", "IT6")
        assert es < 0
        assert ei < es  # lower deviation is more negative

    def test_f_shaft_larger_negative_es(self):
        """Shaft f has larger negative es than g."""
        es_f, _ = shaft_deviations(10.0, "f", "IT6")
        es_g, _ = shaft_deviations(10.0, "g", "IT6")
        assert es_f < es_g  # f is more negative

    def test_js_shaft_symmetric(self):
        """Shaft js has symmetric ±IT/2 deviation."""
        es, ei = shaft_deviations(10.0, "js", "IT6")
        it = it_tolerance(10.0, "IT6")
        assert es == pytest.approx(it / 2)
        assert ei == pytest.approx(-it / 2)

    def test_k_shaft_positive_ei(self):
        """Shaft k has small positive lower deviation (for IT ≤ 8)."""
        es, ei = shaft_deviations(10.0, "k", "IT6")
        assert ei >= 0
        assert es == pytest.approx(ei + it_tolerance(10.0, "IT6"))

    def test_p_shaft_positive_ei(self):
        """Shaft p has positive lower deviation (interference)."""
        es, ei = shaft_deviations(10.0, "p", "IT6")
        assert ei > 0
        assert es > 0

    def test_unknown_shaft_raises(self):
        with pytest.raises(ValueError, match="Unknown shaft deviation"):
            shaft_deviations(10.0, "z", "IT6")


# ============================================================================
# 3. Hole fundamental deviations
# ============================================================================

class TestHoleDeviations:

    def test_h_hole_ei_is_zero(self):
        es, ei = hole_deviations(10.0, "H", "IT7")
        assert ei == pytest.approx(0.0)
        assert es > 0  # ES = +IT

    def test_g_hole_positive_ei(self):
        """Hole G has small positive lower deviation."""
        es, ei = hole_deviations(10.0, "G", "IT7")
        assert ei > 0
        assert es > ei

    def test_js_hole_symmetric(self):
        es, ei = hole_deviations(10.0, "Js", "IT7")
        it = it_tolerance(10.0, "IT7")
        assert es == pytest.approx(it / 2)
        assert ei == pytest.approx(-it / 2)

    def test_p_hole_negative_es(self):
        """Hole P has negative upper deviation (interference)."""
        es, ei = hole_deviations(10.0, "P", "IT7")
        assert es < 0
        assert ei < es  # more negative

    def test_k_hole_near_zero(self):
        """Hole K has near-zero or slightly negative upper deviation."""
        es, ei = hole_deviations(10.0, "K", "IT7")
        assert es <= 0  # transition fit

    def test_unknown_hole_raises(self):
        with pytest.raises(ValueError, match="Unknown hole deviation"):
            hole_deviations(10.0, "Z", "IT7")


# ============================================================================
# 4. Fit computation
# ============================================================================

class TestComputeFit:

    def test_h7_g6_clearance(self):
        """H7/g6 should be a clearance fit."""
        result = compute_fit(10.0, "IT7", "IT6", "H", "g")
        assert result.fit_type == "clearance"
        assert result.max_clearance > 0
        assert result.min_clearance > 0
        assert result.max_interference == 0.0

    def test_h7_h6_clearance(self):
        """H7/h6 should be a clearance fit (close)."""
        result = compute_fit(10.0, "IT7", "IT6", "H", "h")
        assert result.fit_type == "clearance"
        assert result.min_clearance >= 0

    def test_h8_f7_clearance(self):
        """H8/f7 should be a clearance fit (running/sliding)."""
        result = compute_fit(10.0, "IT8", "IT7", "H", "f")
        assert result.fit_type == "clearance"
        assert result.max_clearance > result.min_clearance

    def test_h7_k6_transition(self):
        """H7/k6 should be a transition fit."""
        result = compute_fit(10.0, "IT7", "IT6", "H", "k")
        assert result.fit_type == "transition"
        assert result.max_clearance > 0
        assert result.max_interference > 0

    def test_h7_js6_transition(self):
        """H7/js6 should be a transition fit (bearing seat)."""
        result = compute_fit(10.0, "IT7", "IT6", "H", "js")
        assert result.fit_type == "transition"

    def test_h7_p6_interference(self):
        """H7/p6 should be an interference fit."""
        result = compute_fit(10.0, "IT7", "IT6", "H", "p")
        assert result.fit_type == "interference"
        assert result.max_interference > 0
        assert result.min_interference >= 0  # borderline case at 10mm
        assert result.max_clearance <= 0

    def test_h7_p6_22mm(self):
        """H7/p6 for 22mm (608 bearing OD) should be interference."""
        result = compute_fit(22.0, "IT7", "IT6", "H", "p")
        assert result.fit_type == "interference"
        assert result.max_interference > 0

    def test_deviations_populated(self):
        result = compute_fit(10.0, "IT7", "IT6", "H", "g")
        assert result.hole_es > 0  # H: ES = +IT
        assert result.hole_ei == 0.0  # H: EI = 0
        assert result.shaft_es < 0  # g: negative es
        assert result.shaft_ei < 0  # g: even more negative


# ============================================================================
# 5. Fit recommendations
# ============================================================================

class TestFitRecommendations:

    def test_bearing_seat(self):
        rec = recommend_fit("bearing_seat")
        assert rec is not None
        assert rec.code == "H7/js6"
        assert rec.fit_type == "transition"

    def test_sliding(self):
        rec = recommend_fit("sliding")
        assert rec is not None
        assert rec.code == "H8/f7"
        assert rec.fit_type == "clearance"

    def test_locating(self):
        rec = recommend_fit("locating")
        assert rec is not None
        assert rec.code == "H7/g6"

    def test_press(self):
        rec = recommend_fit("press")
        assert rec is not None
        assert rec.code == "H7/p6"
        assert rec.fit_type == "interference"

    def test_snug(self):
        rec = recommend_fit("snug")
        assert rec is not None
        assert rec.code == "H7/h6"

    def test_transition(self):
        rec = recommend_fit("transition")
        assert rec is not None
        assert rec.code == "H7/k6"

    def test_unknown_returns_none(self):
        assert recommend_fit("nonexistent") is None

    def test_all_recommendations_have_valid_codes(self):
        for key, rec in FIT_RECOMMENDATIONS.items():
            assert rec.hole_deviation.isupper() or rec.hole_deviation == "Js"
            assert rec.shaft_deviation.islower() or rec.shaft_deviation == "js"


# ============================================================================
# 6. Tolerance-aware dimension helpers
# ============================================================================

class TestToleranceDimensions:

    def test_bearing_seat_diameter(self):
        """Bearing seat for 22mm OD should have H7 tolerance."""
        min_d, nom, max_d = bearing_seat_diameter(22.0)
        assert nom == 22.0
        assert min_d <= nom <= max_d
        assert max_d - min_d > 0

    def test_press_fit_bore_with_interference(self):
        """Press-fit bore should be smaller than nominal OD."""
        min_d, nom, max_d = press_fit_bore_diameter(22.0, 0.05)
        assert nom == pytest.approx(22.0 - 0.05)
        assert min_d <= nom <= max_d

    def test_press_fit_bore_default(self):
        """Without custom interference, use H7/p6."""
        min_d, nom, max_d = press_fit_bore_diameter(22.0)
        assert nom == 22.0  # H7 hole nominal = OD
        assert max_d > min_d

    def test_tolerance_hole_diameter(self):
        min_d, nom, max_d = tolerance_hole_diameter(10.0, "locating")
        # H hole: EI=0, ES=+IT → min_d = nom, max_d > nom
        assert min_d == pytest.approx(nom)
        assert max_d > nom

    def test_tolerance_shaft_diameter(self):
        min_d, nom, max_d = tolerance_shaft_diameter(10.0, "locating")
        # g shaft: es < 0, ei < 0 → both min_d and max_d < nom
        assert min_d < max_d
        assert max_d < nom  # clearance shaft always below nominal


# ============================================================================
# 7. Tolerance stackup
# ============================================================================

class TestToleranceStackup:

    def test_simple_two_part_stackup(self):
        stack = ToleranceStackup()
        stack.add_dimension("plate", 6.0, upper=0.1, lower=-0.1)
        stack.add_dimension("motor", 47.0, upper=0.2, lower=-0.2)
        result = stack.compute_stackup()
        assert result.nominal == pytest.approx(53.0)
        assert result.upper_dev == pytest.approx(0.3)
        assert result.lower_dev == pytest.approx(-0.3)
        assert result.max_value == pytest.approx(53.3)
        assert result.min_value == pytest.approx(52.7)

    def test_negative_direction(self):
        """Negative direction dimensions subtract from nominal."""
        stack = ToleranceStackup()
        stack.add_dimension("total", 100.0, upper=0.5, lower=-0.5)
        stack.add_dimension("cut", 20.0, upper=0.2, lower=-0.2, direction="-")
        result = stack.compute_stackup()
        assert result.nominal == pytest.approx(80.0)
        # Worst-case: total max - cut min, total min - cut max
        # upper = 0.5 + 0.2 = 0.7
        # lower = -0.5 + (-0.2) = -0.7
        assert result.upper_dev == pytest.approx(0.7)
        assert result.lower_dev == pytest.approx(-0.7)

    def test_it_grade_dimension(self):
        """Dimension with IT grade auto-computes tolerance."""
        stack = ToleranceStackup()
        stack.add_dimension("shaft", 10.0, it_grade="IT7")
        result = stack.compute_stackup()
        it = it_tolerance(10.0, "IT7")
        assert result.upper_dev == pytest.approx(it / 2)
        assert result.lower_dev == pytest.approx(-it / 2)

    def test_check_acceptable_pass(self):
        stack = ToleranceStackup()
        stack.add_dimension("a", 10.0, upper=0.05, lower=-0.05)
        stack.add_dimension("b", 20.0, upper=0.05, lower=-0.05)
        assert stack.check_acceptable(allowed_total=0.2) is True

    def test_check_acceptable_fail(self):
        stack = ToleranceStackup()
        stack.add_dimension("a", 10.0, upper=0.1, lower=-0.1)
        stack.add_dimension("b", 20.0, upper=0.1, lower=-0.1)
        assert stack.check_acceptable(allowed_total=0.3) is False

    def test_check_acceptable_upper_limit(self):
        stack = ToleranceStackup()
        stack.add_dimension("a", 10.0, upper=0.1, lower=-0.05)
        assert stack.check_acceptable(allowed_upper=0.2) is True
        assert stack.check_acceptable(allowed_upper=0.05) is False

    def test_clear(self):
        stack = ToleranceStackup()
        stack.add_dimension("a", 10.0)
        assert stack.dimension_count == 1
        stack.clear()
        assert stack.dimension_count == 0

    def test_empty_stackup(self):
        stack = ToleranceStackup()
        result = stack.compute_stackup()
        assert result.nominal == 0.0
        assert result.total_tolerance == 0.0

    def test_dimension_count(self):
        stack = ToleranceStackup()
        stack.add_dimension("a", 1.0)
        stack.add_dimension("b", 2.0)
        stack.add_dimension("c", 3.0)
        assert stack.dimension_count == 3

    def test_result_dimensions_list(self):
        stack = ToleranceStackup()
        stack.add_dimension("a", 10.0, upper=0.1, lower=-0.1)
        stack.add_dimension("b", 20.0, upper=0.2, lower=-0.2)
        result = stack.compute_stackup()
        assert len(result.dimensions) == 2
        assert result.dimensions[0].name == "a"
        assert result.dimensions[1].name == "b"

    def test_result_as_dict(self):
        stack = ToleranceStackup()
        stack.add_dimension("a", 10.0, upper=0.1, lower=-0.1)
        d = stack.compute_stackup().as_dict()
        assert "nominal" in d
        assert "upper_dev" in d
        assert "total_tolerance" in d


# ============================================================================
# 8. analyze_assembly_chain helper
# ============================================================================

class TestAnalyzeAssemblyChain:

    def test_basic_chain(self):
        chain = [
            {"name": "plate", "nominal": 5.0, "upper": 0.1, "lower": -0.1},
            {"name": "bracket", "nominal": 40.0, "upper": 0.2, "lower": -0.2},
        ]
        result = analyze_assembly_chain(chain)
        assert result.nominal == pytest.approx(45.0)
        assert result.upper_dev == pytest.approx(0.3)

    def test_chain_with_it_grade(self):
        chain = [
            {"name": "bore", "nominal": 22.0, "it_grade": "IT7"},
        ]
        result = analyze_assembly_chain(chain)
        it = it_tolerance(22.0, "IT7")
        assert result.upper_dev == pytest.approx(it / 2)


# ============================================================================
# 9. Integration: connection_features tolerance helpers
# ============================================================================

class TestConnectionFeaturesTolerance:

    def test_clearance_hole_with_tolerance(self):
        from lang3d.tools.connection_features import get_clearance_hole_with_tolerance
        nom, min_d, max_d = get_clearance_hole_with_tolerance("M3")
        assert nom == pytest.approx(3.4)  # ISO 273 normal fit
        assert min_d == nom  # nominal is the lower bound
        assert max_d > nom  # IT12 tolerance adds upper band

    def test_clearance_hole_loose_fit(self):
        from lang3d.tools.connection_features import get_clearance_hole_with_tolerance
        nom, min_d, max_d = get_clearance_hole_with_tolerance("M3", "loose")
        assert nom == pytest.approx(3.9)  # ISO 273 loose fit

    def test_clearance_hole_m5(self):
        from lang3d.tools.connection_features import get_clearance_hole_with_tolerance
        nom, min_d, max_d = get_clearance_hole_with_tolerance("M5")
        assert nom == pytest.approx(5.5)  # ISO 273 M5 normal
        assert max_d > nom
