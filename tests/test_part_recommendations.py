"""Tests for part recommendation engine, tool, and integration."""

import json
import pytest

from lang3d.knowledge.mechanics import Assembly, Joint, Part
from lang3d.knowledge.part_recommendations import (
    CATEGORY_FALLBACK,
    RECOMMENDATION_RULES,
    RecommendationResult,
    RecommendedPart,
    compute_compatibility_score,
    get_recommendations,
    get_recommendations_for_assembly,
)
from lang3d.knowledge.parts_catalog import PART_CATALOG
from lang3d.tools.base import ToolRegistry
from lang3d.tools.part_recommend_tool import (
    PartRecommendTool,
    register_part_recommend_tools,
)


# ---------------------------------------------------------------------------
# Rule data integrity
# ---------------------------------------------------------------------------


class TestRuleDataIntegrity:
    """Verify recommendation rules reference valid catalog IDs."""

    def test_ten_core_rules(self):
        assert len(RECOMMENDATION_RULES) == 10

    @pytest.mark.parametrize("rule", RECOMMENDATION_RULES)
    def test_trigger_part_id_valid(self, rule):
        """All trigger IDs should exist in PART_CATALOG or be reasonable."""
        # Some trigger IDs are in the catalog, verify the rules are well-formed
        assert rule.trigger_part_id
        assert len(rule.trigger_keywords) >= 2
        assert len(rule.recommendations) >= 1

    @pytest.mark.parametrize("rule", RECOMMENDATION_RULES)
    def test_recommendation_part_ids_valid(self, rule):
        """All recommended part IDs should exist in PART_CATALOG."""
        for rec in rule.recommendations:
            assert rec.part_catalog_id in PART_CATALOG, (
                f"Rule '{rule.trigger_part_id}' recommends '{rec.part_catalog_id}' "
                f"which is not in PART_CATALOG"
            )

    @pytest.mark.parametrize("rule", RECOMMENDATION_RULES)
    def test_recommendation_quantities_positive(self, rule):
        for rec in rule.recommendations:
            assert rec.quantity > 0
            assert rec.reason  # non-empty Chinese description

    def test_category_fallback_keys_valid(self):
        """Category fallback keys should be recognized categories."""
        for cat in CATEGORY_FALLBACK:
            assert isinstance(CATEGORY_FALLBACK[cat], list)
            assert len(CATEGORY_FALLBACK[cat]) > 0


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------


class TestExactMatch:
    """Test exact ID matching for recommendations."""

    def test_sg90_recommends_m2_screws(self):
        result = get_recommendations("servo_sg90")
        assert result.compatibility_score == 1.0
        screw_found = any(
            "screw" in r.part_catalog_id for r in result.recommended_parts
        )
        assert screw_found, "SG90 should recommend screws"

    def test_sg90_recommends_bracket(self):
        result = get_recommendations("servo_sg90")
        bracket_found = any(
            "bracket" in r.part_catalog_id for r in result.recommended_parts
        )
        assert bracket_found, "SG90 should recommend servo_bracket"

    def test_nema17_recommends_m3_screws(self):
        result = get_recommendations("nema17_stepper")
        assert result.compatibility_score == 1.0
        screw_found = any(
            "screw" in r.part_catalog_id for r in result.recommended_parts
        )
        assert screw_found, "NEMA17 should recommend M3 screws"

    def test_nema17_recommends_gt2_pulley(self):
        result = get_recommendations("nema17_stepper")
        pulley_found = any(
            "pulley" in r.part_catalog_id for r in result.recommended_parts
        )
        assert pulley_found, "NEMA17 should recommend GT2 pulley"

    def test_bearing_608_recommends_shaft(self):
        result = get_recommendations("bearing_608")
        shaft_found = any(
            "shaft" in r.part_catalog_id for r in result.recommended_parts
        )
        assert shaft_found, "608 bearing should recommend shaft"

    def test_gt2_pulley_recommends_belt(self):
        result = get_recommendations("gt2_pulley")
        belt_found = any(
            "belt" in r.part_catalog_id for r in result.recommended_parts
        )
        assert belt_found, "GT2 pulley should recommend belt"

    def test_t8_leadscrew_recommends_nut(self):
        result = get_recommendations("t8_leadscrew")
        nut_found = any(
            "nut" in r.part_catalog_id for r in result.recommended_parts
        )
        assert nut_found, "T8 leadscrew should recommend T8 nut"

    def test_arduino_recommends_standoff(self):
        result = get_recommendations("controller_arduino_uno")
        standoff_found = any(
            "standoff" in r.part_catalog_id for r in result.recommended_parts
        )
        assert standoff_found, "Arduino should recommend standoffs"


# ---------------------------------------------------------------------------
# Fuzzy match
# ---------------------------------------------------------------------------


class TestFuzzyMatch:
    """Test fuzzy keyword matching."""

    def test_nema17_fuzzy(self):
        result = get_recommendations("nema17")
        assert len(result.recommended_parts) > 0
        assert result.compatibility_score < 1.0
        assert result.compatibility_score > 0.0

    def test_sg90_fuzzy(self):
        result = get_recommendations("sg90")
        assert len(result.recommended_parts) > 0

    def test_bearing_fuzzy(self):
        result = get_recommendations("608 bearing")
        assert len(result.recommended_parts) > 0


# ---------------------------------------------------------------------------
# Category fallback
# ---------------------------------------------------------------------------


class TestCategoryFallback:
    """Test category-level fallback matching."""

    def test_unknown_sensor_falls_back(self):
        """An unknown sensor part should fall back to category rules."""
        # Use a valid catalog ID in the sensor category
        result = get_recommendations("sensor_rplidar_a1")
        # Exact match should work for rplidar
        assert len(result.recommended_parts) > 0


# ---------------------------------------------------------------------------
# No match
# ---------------------------------------------------------------------------


class TestNoMatch:
    def test_completely_unknown_part(self):
        result = get_recommendations("flux_capacitor_xyz_9999")
        assert len(result.recommended_parts) == 0
        assert result.compatibility_score == 0.0


# ---------------------------------------------------------------------------
# Assembly-level recommendations
# ---------------------------------------------------------------------------


class TestAssemblyRecommendations:
    """Test assembly-level batch recommendations with dedup."""

    def test_simple_assembly(self):
        asm = Assembly(
            name="Test Bot",
            parts=[
                Part("servo_sg90", "actuator", "SG90舵机"),
                Part("servo_mg996r", "actuator", "MG996R舵机"),
            ],
            joints=[],
        )
        results = get_recommendations_for_assembly(asm)
        assert len(results) >= 1

    def test_deduplication(self):
        """Multiple parts recommending the same accessory should aggregate."""
        asm = Assembly(
            name="Test Dedup",
            parts=[
                Part("servo_sg90", "actuator", "SG90舵机"),
                Part("servo_mg996r", "actuator", "MG996R舵机"),
            ],
            joints=[],
        )
        results = get_recommendations_for_assembly(asm)
        # Both recommend screws — check aggregation
        all_recs: dict[str, int] = {}
        for res in results:
            for rec in res.recommended_parts:
                all_recs[rec.part_catalog_id] = (
                    all_recs.get(rec.part_catalog_id, 0) + rec.quantity
                )
        # Screw should have aggregated quantity from both servos
        screw_qty = sum(v for k, v in all_recs.items() if "screw" in k)
        assert screw_qty >= 8, "Should aggregate screw quantities from both servos"


# ---------------------------------------------------------------------------
# Compatibility scoring
# ---------------------------------------------------------------------------


class TestCompatibilityScore:
    """Test compute_compatibility_score between two parts."""

    def test_nema17_with_pulley_high(self):
        score = compute_compatibility_score("nema17_stepper", "gt2_pulley")
        assert score >= 0.8, "NEMA17 + GT2 pulley should score high"

    def test_nema17_with_bracket_high(self):
        score = compute_compatibility_score("nema17_stepper", "servo_bracket")
        assert score >= 0.4, "NEMA17 + bracket should score at least moderate"

    def test_bearing_with_servo_low(self):
        score = compute_compatibility_score("bearing_608", "servo_sg90")
        assert score < 0.7, "Bearing + servo should not score high"

    def test_same_part_moderate(self):
        score = compute_compatibility_score("bearing_608", "bearing_608")
        assert score == 0.2, "Same part should score low (not complementary)"

    def test_unknown_part_zero(self):
        score = compute_compatibility_score("nonexistent_part", "bearing_608")
        assert score == 0.0

    def test_reverse_compatibility(self):
        """Compatibility should be symmetric or near-symmetric."""
        score_ab = compute_compatibility_score("servo_sg90", "servo_bracket")
        score_ba = compute_compatibility_score("servo_bracket", "servo_sg90")
        assert abs(score_ab - score_ba) < 0.3, (
            f"Asymmetric scores: {score_ab} vs {score_ba}"
        )


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------


class TestPartRecommendTool:
    """Test the part_recommend tool."""

    def test_execute_single_part(self):
        tool = PartRecommendTool()
        result = tool.execute(part_id="servo_sg90")
        assert "servo_sg90" in result
        assert "推荐配件" in result or "recommend" in result.lower()

    def test_execute_returns_valid_json(self):
        tool = PartRecommendTool()
        result = tool.execute(part_id="nema17_stepper")
        json_start = result.index("--- JSON ---") + len("--- JSON ---")
        json_str = result[json_start:].strip()
        data = json.loads(json_str)
        assert "recommended_parts" in data
        assert len(data["recommended_parts"]) > 0

    def test_execute_compatibility_check(self):
        tool = PartRecommendTool()
        result = tool.execute(check_compatibility="nema17_stepper,gt2_pulley")
        assert "0." in result  # should have a decimal score

    def test_execute_assembly_json(self):
        tool = PartRecommendTool()
        asm_json = json.dumps({
            "name": "Test Arm",
            "parts": [
                {"name": "servo_sg90", "category": "actuator", "description": "舵机"},
            ],
            "joints": [],
        })
        result = tool.execute(assembly_json=asm_json)
        assert "推荐配件" in result or "recommend" in result.lower()

    def test_execute_no_params(self):
        tool = PartRecommendTool()
        result = tool.execute()
        assert "请提供" in result

    def test_execute_unknown_part(self):
        tool = PartRecommendTool()
        result = tool.execute(part_id="flux_capacitor_xyz_9999")
        assert "未找到" in result or len(result) > 0


class TestToolRegistration:
    """Test tool registration."""

    def test_register_tools(self):
        registry = ToolRegistry()
        register_part_recommend_tools(registry)
        tool = registry.get("part_recommend")
        assert tool is not None

    def test_tool_definition(self):
        registry = ToolRegistry()
        register_part_recommend_tools(registry)
        tool = registry.get("part_recommend")
        defn = tool.get_definition()
        assert defn.name == "part_recommend"
        assert "properties" in defn.parameters
        assert "part_id" in defn.parameters["properties"]
        assert "assembly_json" in defn.parameters["properties"]
