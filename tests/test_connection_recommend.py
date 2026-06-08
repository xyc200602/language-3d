"""Tests for recommend_connection() — automatic ConnectionMethod recommendation.

Covers:
1. Explicit connection not overridden
2. Bracket + NEMA17 → bolted M3×4 (CONNECTION_PATTERNS match)
3. Housing + bearing → press_fit (CONNECTION_PATTERNS match)
4. Bracket + DYNAMIXEL XM430 → bolted M2.5×4 (MountingInterface match)
5. Two plates → bolted M3×4 (CONNECTION_PATTERNS or heuristic)
6. Unknown category fallback → bolted M3×2
7. Sensor + bracket → bolted M2.5×4 (heuristic)
"""

from __future__ import annotations

import pytest

from lang3d.knowledge.assembly_patterns import (
    _infer_bolt_size_from_hole_diameter,
    _match_category_to_pattern_type,
    recommend_connection,
)
from lang3d.knowledge.mechanics import ConnectionMethod, Joint, Part


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def bracket() -> Part:
    return Part(
        name="motor_bracket",
        category="structural",
        description="NEMA17 motor bracket",
        dimensions={"width": 60, "height": 50, "thickness": 5},
    )


@pytest.fixture
def nema17() -> Part:
    return Part(
        name="nema17_motor",
        category="actuator",
        description="NEMA17 stepper motor 42BYGH",
        dimensions={"length": 42.3, "width": 42.3, "height": 38.0},
    )


@pytest.fixture
def housing() -> Part:
    return Part(
        name="joint_housing",
        category="structural",
        description="Joint housing for bearing",
        dimensions={"width": 50, "height": 40, "depth": 30},
    )


@pytest.fixture
def bearing_608() -> Part:
    return Part(
        name="bearing_608",
        category="bearing",
        description="608-2RS bearing",
        dimensions={"diameter": 22, "height": 7},
    )


@pytest.fixture
def plate() -> Part:
    return Part(
        name="top_plate",
        category="structural",
        description="Top plate",
        dimensions={"width": 100, "length": 100, "height": 3},
    )


@pytest.fixture
def dynamixel_xm430() -> Part:
    return Part(
        name="dynamixel_xm430_w350_body",
        category="actuator",
        description="DYNAMIXEL XM430-W350-T servo",
        dimensions={"body_width": 28, "body_height": 46.5, "body_depth": 34},
    )


@pytest.fixture
def sensor() -> Part:
    return Part(
        name="imu_sensor",
        category="sensor",
        description="IMU sensor module",
        dimensions={"width": 25, "length": 25, "height": 5},
    )


@pytest.fixture
def unknown_part() -> Part:
    return Part(
        name="mystery_device",
        category="custom_category",
        description="Something completely novel",
        dimensions={"width": 10, "height": 10},
    )


def _fixed_joint(parent: str, child: str) -> Joint:
    return Joint("fixed", parent, child)


# ============================================================================
# Test: Helper functions
# ============================================================================

class TestHelperFunctions:

    def test_infer_bolt_size_m2(self):
        assert _infer_bolt_size_from_hole_diameter(2.0) == "M2"

    def test_infer_bolt_size_m25(self):
        assert _infer_bolt_size_from_hole_diameter(2.8) == "M2.5"

    def test_infer_bolt_size_m3(self):
        assert _infer_bolt_size_from_hole_diameter(3.4) == "M3"

    def test_infer_bolt_size_m4(self):
        assert _infer_bolt_size_from_hole_diameter(4.5) == "M4"

    def test_infer_bolt_size_m5(self):
        assert _infer_bolt_size_from_hole_diameter(5.5) == "M5"

    def test_infer_bolt_size_m6(self):
        assert _infer_bolt_size_from_hole_diameter(6.5) == "M6"

    def test_match_category_actuator_stepper(self):
        p = Part(name="nema17_motor", category="actuator",
                 description="NEMA17 stepper motor")
        assert _match_category_to_pattern_type(p) == "stepper_motor"

    def test_match_category_actuator_servo(self):
        p = Part(name="dynamixel_servo", category="actuator",
                 description="DYNAMIXEL XM430 servo")
        assert _match_category_to_pattern_type(p) == "servo"

    def test_match_category_structural_bracket(self):
        p = Part(name="motor_bracket", category="structural",
                 description="Motor bracket L-shape")
        assert _match_category_to_pattern_type(p) == "bracket"

    def test_match_category_bearing(self):
        p = Part(name="bearing_608", category="bearing",
                 description="608 bearing")
        assert _match_category_to_pattern_type(p) == "bearing"

    def test_match_category_unknown(self):
        p = Part(name="weird_thing", category="exotic",
                 description="Unknown type")
        assert _match_category_to_pattern_type(p) == "exotic"


# ============================================================================
# Test: Layer 1 — Explicit connection not overridden
# ============================================================================

class TestExplicitConnection:

    def test_explicit_connection_not_overridden(self, bracket, nema17):
        """If joint.connection is already set, recommend_connection returns it."""
        explicit = ConnectionMethod(type="bolted", bolt_size="M5", bolt_count=6)
        j = Joint("fixed", bracket.name, nema17.name, connection=explicit)

        result = recommend_connection(bracket, nema17, j)
        assert result is explicit
        assert result.bolt_size == "M5"
        assert result.bolt_count == 6

    def test_explicit_press_fit_preserved(self, housing, bearing_608):
        """Explicit press_fit connection should not be changed."""
        explicit = ConnectionMethod(type="press_fit", interference_mm=0.08)
        j = Joint("fixed", housing.name, bearing_608.name, connection=explicit)

        result = recommend_connection(housing, bearing_608, j)
        assert result is explicit
        assert result.type == "press_fit"
        assert result.interference_mm == 0.08


# ============================================================================
# Test: Layer 2 — MountingInterface matching (DYNAMIXEL)
# ============================================================================

class TestMountingInterfaceMatch:

    def test_bracket_to_dynamixel_m2_5(self, bracket, dynamixel_xm430):
        """DYNAMIXEL XM430 body has 4×M2.5 threaded holes in MOUNTING_INTERFACES."""
        j = _fixed_joint(bracket.name, dynamixel_xm430.name)

        result = recommend_connection(bracket, dynamixel_xm430, j)
        assert result.type == "bolted"
        assert result.bolt_size == "M2.5"
        assert result.bolt_count == 4


# ============================================================================
# Test: Layer 3 — CONNECTION_PATTERNS matching
# ============================================================================

class TestConnectionPatternMatch:

    def test_bracket_to_nema17(self, bracket, nema17):
        """Bracket + NEMA17 stepper motor → bolted M3×4."""
        j = _fixed_joint(bracket.name, nema17.name)

        result = recommend_connection(bracket, nema17, j)
        assert result.type == "bolted"
        assert result.bolt_size == "M3"
        assert result.bolt_count == 4

    def test_housing_to_bearing(self, housing, bearing_608):
        """Housing + bearing → press_fit."""
        j = _fixed_joint(housing.name, bearing_608.name)

        result = recommend_connection(housing, bearing_608, j)
        assert result.type == "press_fit"
        assert result.interference_mm > 0

    def test_two_plates(self, plate):
        """Two plates → bolted M3×4 (plate_to_plate pattern)."""
        other_plate = Part(
            name="bottom_plate",
            category="structural",
            description="Bottom plate",
            dimensions={"width": 100, "length": 100, "height": 3},
        )
        j = _fixed_joint(plate.name, other_plate.name)

        result = recommend_connection(plate, other_plate, j)
        assert result.type == "bolted"
        assert result.bolt_size == "M3"
        assert result.bolt_count == 4


# ============================================================================
# Test: Layer 4 — Heuristic fallback
# ============================================================================

class TestHeuristicFallback:

    def test_sensor_to_bracket(self, sensor, bracket):
        """Sensor + structural → bolted M2.5×4 (heuristic)."""
        j = _fixed_joint(bracket.name, sensor.name)

        result = recommend_connection(bracket, sensor, j)
        assert result.type == "bolted"
        assert result.bolt_size == "M2.5"
        assert result.bolt_count == 4

    def test_unknown_fallback(self, unknown_part):
        """Unknown category → fallback bolted M3×2."""
        other = Part(name="some_other", category="also_unknown",
                     description="Mystery")
        j = _fixed_joint(unknown_part.name, other.name)

        result = recommend_connection(unknown_part, other, j)
        assert result.type == "bolted"
        assert result.bolt_size == "M3"
        assert result.bolt_count == 2

    def test_structural_to_structural_fallback(self):
        """Two structural parts with no pattern match → bolted M3×4."""
        a = Part(name="custom_struct_a", category="structural",
                 description="Custom structure A")
        b = Part(name="custom_struct_b", category="structural",
                 description="Custom structure B")
        j = _fixed_joint(a.name, b.name)

        result = recommend_connection(a, b, j)
        assert result.type == "bolted"
        assert result.bolt_size == "M3"
        assert result.bolt_count == 4

    def test_bearing_press_fit_fallback(self, bearing_608):
        """Bearing to unknown → press_fit."""
        other = Part(name="something", category="custom",
                     description="Something")
        j = _fixed_joint(other.name, bearing_608.name)

        result = recommend_connection(other, bearing_608, j)
        assert result.type == "press_fit"

    def test_prismatic_joint_structural(self):
        """Prismatic joint → bolted M3×4."""
        a = Part(name="rail", category="structural",
                 description="Linear rail")
        b = Part(name="carriage", category="structural",
                 description="Carriage block")
        j = Joint("prismatic", a.name, b.name)

        result = recommend_connection(a, b, j)
        assert result.type == "bolted"
        assert result.bolt_count == 4
