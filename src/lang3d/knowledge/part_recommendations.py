"""Part recommendation engine — suggests compatible accessories for selected parts.

Provides:
  - 10 core recommendation rules derived from CONNECTION_PATTERNS and PART_CATALOG.
  - Category-level fallback rules for generic matching.
  - Three-level matching: exact ID → fuzzy keyword → category fallback.
  - Assembly-level batch recommendation with deduplication.
  - Compatibility scoring between any two parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .mechanics import Assembly, Part


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RecommendedPart:
    """A recommended accessory part."""

    part_catalog_id: str
    quantity: int
    reason: str  # Chinese
    required: bool = True


@dataclass
class RecommendationRule:
    """A rule that maps a trigger part to a list of recommended accessories."""

    trigger_part_id: str
    trigger_keywords: list[str]
    recommendations: list[RecommendedPart]
    rationale: str


@dataclass
class RecommendationResult:
    """Result of a recommendation query."""

    trigger_part: str
    recommended_parts: list[RecommendedPart]
    compatibility_score: float
    summary: str


# ---------------------------------------------------------------------------
# Recommendation rules (10 core rules)
# ---------------------------------------------------------------------------

RECOMMENDATION_RULES: list[RecommendationRule] = [
    RecommendationRule(
        trigger_part_id="servo_sg90",
        trigger_keywords=["sg90", "微型舵机", "micro servo"],
        recommendations=[
            RecommendedPart("socket_head_cap_screw", 4, "SG90法兰M2安装螺丝", required=True),
            RecommendedPart("servo_bracket", 1, "SG90配套舵机支架", required=True),
        ],
        rationale="SG90微型舵机法兰M2孔安装，需M2螺丝+舵机支架",
    ),
    RecommendationRule(
        trigger_part_id="servo_mg996r",
        trigger_keywords=["mg996r", "大扭力舵机", "金属齿轮"],
        recommendations=[
            RecommendedPart("socket_head_cap_screw", 4, "MG996R法兰M2.5安装螺丝", required=True),
            RecommendedPart("servo_bracket", 1, "MG996R配套舵机支架", required=True),
        ],
        rationale="MG996R法兰M2.5孔安装，需M2.5螺丝+舵机支架",
    ),
    RecommendationRule(
        trigger_part_id="nema17_stepper",
        trigger_keywords=["nema17", "步进电机", "42mm", "stepper"],
        recommendations=[
            RecommendedPart("socket_head_cap_screw", 4, "NEMA17标准31mm孔距M3安装螺丝", required=True),
            RecommendedPart("gt2_pulley", 1, "NEMA17配套GT2同步轮(5mm轴孔)", required=True),
            RecommendedPart("bearing_608", 2, "关节轴承(内径8mm)", required=True),
        ],
        rationale="NEMA17标准31mm孔距M3安装，需M3螺丝+GT2同步轮+608轴承",
    ),
    RecommendationRule(
        trigger_part_id="bearing_608",
        trigger_keywords=["608", "轴承", "bearing", "8mm"],
        recommendations=[
            RecommendedPart("linear_shaft", 1, "608轴承配套8mm光轴"),
            RecommendedPart("bearing_block", 1, "608轴承座"),
        ],
        rationale="608轴承内径8mm，需配套8mm轴+轴承座",
    ),
    RecommendationRule(
        trigger_part_id="dynamixel_xm430_w350",
        trigger_keywords=["dynamixel", "xm430", "ROBOTIS", "智能舵机"],
        recommendations=[
            RecommendedPart("socket_head_cap_screw", 8, "XM430 M2.5安装螺丝(本体+法兰)", required=True),
        ],
        rationale="DYNAMIXEL XM430-W350 ROBOTIS标准框架，M2.5螺栓安装",
    ),
    RecommendationRule(
        trigger_part_id="sensor_rplidar_a1",
        trigger_keywords=["rplidar", "激光雷达", "lidar"],
        recommendations=[
            RecommendedPart("socket_head_cap_screw", 4, "RPLIDAR标准M3安装螺丝", required=True),
            RecommendedPart("mounting_plate", 1, "RPLIDAR安装板"),
        ],
        rationale="RPLIDAR标准M3安装孔位，需M3螺丝+安装板",
    ),
    RecommendationRule(
        trigger_part_id="gt2_pulley",
        trigger_keywords=["gt2", "同步轮", "timing pulley"],
        recommendations=[
            RecommendedPart("gt2_belt", 1, "GT2配套同步带", required=True),
            RecommendedPart("socket_head_cap_screw", 1, "同步轮紧定螺钉"),
        ],
        rationale="GT2同步轮需配套同步带+紧定螺钉固定",
    ),
    RecommendationRule(
        trigger_part_id="linear_bearing_lm8uu",
        trigger_keywords=["lm8uu", "直线轴承", "linear bearing", "8mm"],
        recommendations=[
            RecommendedPart("linear_shaft", 1, "LM8UU配套8mm直线轴", required=True),
            RecommendedPart("shaft_support", 2, "直线轴支撑座(两端)"),
        ],
        rationale="LM8UU直线轴承内径8mm，需8mm直线轴+支撑座",
    ),
    RecommendationRule(
        trigger_part_id="controller_arduino_uno",
        trigger_keywords=["arduino", "uno", "控制器", "controller"],
        recommendations=[
            RecommendedPart("socket_head_cap_screw", 4, "Arduino标准M3安装螺丝", required=True),
            RecommendedPart("standoff_column", 4, "Arduino安装铜柱(M3×4)"),
        ],
        rationale="Arduino UNO标准M3安装孔位，需M3螺丝+铜柱支撑",
    ),
    RecommendationRule(
        trigger_part_id="t8_leadscrew",
        trigger_keywords=["t8", "丝杠", "leadscrew", "梯形螺纹"],
        recommendations=[
            RecommendedPart("t8_nut", 1, "T8配套丝杠螺母", required=True),
            RecommendedPart("bearing_608", 2, "丝杠两端支撑轴承(8mm内径)"),
        ],
        rationale="T8丝杠配套螺母+两端608轴承支撑",
    ),
]


# ---------------------------------------------------------------------------
# Category-level fallback rules
# ---------------------------------------------------------------------------

CATEGORY_FALLBACK: dict[str, list[RecommendedPart]] = {
    "bearing": [
        RecommendedPart("linear_shaft", 1, "轴承配套光轴(根据内径选择)"),
        RecommendedPart("bearing_block", 1, "轴承座"),
    ],
    "actuator": [
        RecommendedPart("socket_head_cap_screw", 4, "执行器安装螺丝(根据规格选择)"),
        RecommendedPart("servo_bracket", 1, "执行器安装支架"),
    ],
    "controller": [
        RecommendedPart("socket_head_cap_screw", 4, "控制器安装螺丝"),
        RecommendedPart("standoff_column", 4, "控制器安装铜柱"),
    ],
    "sensor": [
        RecommendedPart("socket_head_cap_screw", 2, "传感器安装螺丝"),
        RecommendedPart("mounting_plate", 1, "传感器安装板"),
    ],
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def get_recommendations(
    part_id: str,
    assembly_context: Assembly | None = None,
) -> RecommendationResult:
    """Get accessory recommendations for a single part.

    Matching priority: exact ID → fuzzy keyword → category fallback.

    Args:
        part_id: Part catalog ID or name to look up.
        assembly_context: Optional assembly for context-aware recommendations.

    Returns:
        A RecommendationResult with matched recommendations.
    """
    part_id_lower = part_id.lower().strip()

    # --- Level 1: Exact ID match ---
    for rule in RECOMMENDATION_RULES:
        if rule.trigger_part_id == part_id_lower:
            return RecommendationResult(
                trigger_part=part_id,
                recommended_parts=list(rule.recommendations),
                compatibility_score=1.0,
                summary=f"精确匹配 '{part_id}'：{rule.rationale}",
            )

    # --- Level 2: Fuzzy keyword match ---
    best_match: RecommendationRule | None = None
    best_score = 0.0

    for rule in RECOMMENDATION_RULES:
        for kw in rule.trigger_keywords:
            kw_lower = kw.lower()
            if part_id_lower in kw_lower or kw_lower in part_id_lower:
                score = len(part_id_lower) / max(len(kw_lower), len(part_id_lower))
                if score > best_score:
                    best_score = score
                    best_match = rule

    if best_match and best_score > 0.3:
        return RecommendationResult(
            trigger_part=part_id,
            recommended_parts=list(best_match.recommendations),
            compatibility_score=round(best_score * 0.9, 2),
            summary=f"模糊匹配 '{part_id}' → '{best_match.trigger_part_id}'：{best_match.rationale}",
        )

    # --- Level 3: Category fallback ---
    from .parts_catalog import PART_CATALOG

    tpl = PART_CATALOG.get(part_id_lower)
    if tpl:
        category = tpl.category
        if category in CATEGORY_FALLBACK:
            return RecommendationResult(
                trigger_part=part_id,
                recommended_parts=list(CATEGORY_FALLBACK[category]),
                compatibility_score=0.5,
                summary=f"类别匹配 '{part_id}' (category={category})",
            )

    # No match
    return RecommendationResult(
        trigger_part=part_id,
        recommended_parts=[],
        compatibility_score=0.0,
        summary=f"未找到 '{part_id}' 的推荐配件",
    )


def get_recommendations_for_assembly(
    assembly: Assembly,
) -> list[RecommendationResult]:
    """Get recommendations for all parts in an assembly.

    Deduplicates and aggregates quantities for parts recommended multiple times.

    Args:
        assembly: The assembly to analyze.

    Returns:
        A list of RecommendationResult, one per trigger part with recommendations.
    """
    # Track all recommended parts for deduplication
    aggregate: dict[str, RecommendedPart] = {}
    results: list[RecommendationResult] = []

    for part in assembly.parts:
        result = get_recommendations(part.name, assembly)
        if not result.recommended_parts:
            continue

        # Aggregate quantities
        for rec in result.recommended_parts:
            if rec.part_catalog_id in aggregate:
                existing = aggregate[rec.part_catalog_id]
                existing.quantity += rec.quantity
            else:
                aggregate[rec.part_catalog_id] = RecommendedPart(
                    part_catalog_id=rec.part_catalog_id,
                    quantity=rec.quantity,
                    reason=rec.reason,
                    required=rec.required,
                )

        results.append(result)

    return results


def compute_compatibility_score(part_a: str, part_b: str) -> float:
    """Compute a 0.0-1.0 compatibility score between two parts.

    Uses catalog metadata, mounting interfaces, and recommendation rules
    to estimate how well two parts work together.

    Args:
        part_a: First part catalog ID.
        part_b: Second part catalog ID.

    Returns:
        Score from 0.0 (incompatible) to 1.0 (perfect match).
    """
    from .parts_catalog import PART_CATALOG

    tpl_a = PART_CATALOG.get(part_a)
    tpl_b = PART_CATALOG.get(part_b)

    if not tpl_a or not tpl_b:
        return 0.0

    score = 0.0

    # Check if part_b is recommended for part_a
    rec_a = get_recommendations(part_a)
    for rec in rec_a.recommended_parts:
        if rec.part_catalog_id == part_b:
            return 0.9

    # Check reverse
    rec_b = get_recommendations(part_b)
    for rec in rec_b.recommended_parts:
        if rec.part_catalog_id == part_a:
            score = max(score, 0.8)

    # Category compatibility matrix
    compat_matrix: dict[str, dict[str, float]] = {
        "actuator": {"structural": 0.6, "fastener": 0.5, "bearing": 0.4},
        "bearing": {"shaft": 0.7, "structural": 0.5},
        "structural": {"fastener": 0.5, "structural": 0.3},
        "controller": {"fastener": 0.5, "mounting": 0.6},
        "sensor": {"fastener": 0.5, "mounting": 0.6},
        "transmission": {"actuator": 0.6, "bearing": 0.5},
    }

    cat_a = tpl_a.category
    cat_b = tpl_b.category

    if cat_a in compat_matrix and cat_b in compat_matrix[cat_a]:
        score = max(score, compat_matrix[cat_a][cat_b])
    if cat_b in compat_matrix and cat_a in compat_matrix[cat_b]:
        score = max(score, compat_matrix[cat_b][cat_a])

    # Same category = lower score (not complementary)
    if cat_a == cat_b:
        score = max(score, 0.2)

    return score
