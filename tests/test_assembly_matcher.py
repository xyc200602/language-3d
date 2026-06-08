"""Tests for assembly_matcher — auto-matching two parts with connection methods.

Covers:
- Functional vs structural part identification (NEMA17, servo, bearing, category-based)
- MountingInterface-driven feature generation (bolted + press-fit)
- Heuristic fallback when no interface exists
- Fastener selection for bolted connections
- Constraint type setup for each connection method
- Validation / verification of match results
- Edge cases: both parts structural, unknown categories, etc.
"""

from __future__ import annotations

import pytest

from lang3d.knowledge.mechanics import ConnectionMethod, Part
from lang3d.tools.assembly_matcher import (
    AssemblyMatcher,
    AssemblyMatchResult,
    FastenerSelection,
    _opposite_face,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bracket(**dims) -> Part:
    defaults = {"length": 60, "width": 60, "thickness": 5}
    defaults.update(dims)
    return Part("bracket", "structural", "L-bracket", dimensions=defaults)


def _make_nema17() -> Part:
    return Part(
        "nema17_stepper", "stepper", "NEMA17 stepper motor",
        dimensions={"length": 42, "width": 42, "height": 47},
    )


def _make_nema23() -> Part:
    return Part(
        "nema23_stepper", "stepper", "NEMA23 stepper motor",
        dimensions={"length": 56, "width": 56, "height": 76},
    )


def _make_sg90() -> Part:
    return Part(
        "servo_sg90", "servo", "SG90 micro servo",
        dimensions={"length": 23, "width": 12.2, "height": 29},
    )


def _make_bearing_608() -> Part:
    return Part(
        "bearing_608", "bearing", "608 ball bearing",
        dimensions={"outer_diameter": 22, "inner_diameter": 8, "width": 7},
    )


def _make_plate(**dims) -> Part:
    defaults = {"length": 100, "width": 100, "thickness": 6}
    defaults.update(dims)
    return Part("plate", "structural", "mounting plate", dimensions=defaults)


def _make_sensor() -> Part:
    return Part(
        "sensor_mpu6050", "sensor", "MPU6050 IMU",
        dimensions={"length": 20, "width": 15, "height": 3},
    )


def _make_generic(name: str = "block", category: str = "structural") -> Part:
    return Part(name, category, "generic part", dimensions={"length": 30, "width": 30, "thickness": 4})


def _bolted(size: str = "M3") -> ConnectionMethod:
    return ConnectionMethod(type="bolted", bolt_size=size)


def _press_fit() -> ConnectionMethod:
    return ConnectionMethod(type="press_fit")


def _snap_fit() -> ConnectionMethod:
    return ConnectionMethod(type="snap_fit")


# ===========================================================================
# 1. Functional part identification
# ===========================================================================

class TestIdentifyFunctionalPart:
    """Step 1: determine which part is functional vs structural."""

    def test_nema17_is_functional(self):
        matcher = AssemblyMatcher()
        bracket = _make_bracket()
        motor = _make_nema17()
        func, struct, fid = matcher._identify_functional_part(bracket, motor)
        assert func is motor
        assert struct is bracket
        assert fid == "nema17_stepper"

    def test_nema23_is_functional(self):
        matcher = AssemblyMatcher()
        bracket = _make_bracket()
        motor = _make_nema23()
        func, struct, fid = matcher._identify_functional_part(bracket, motor)
        assert func is motor
        assert fid == "nema23_stepper"

    def test_order_does_not_matter(self):
        """Functional part should be detected regardless of argument order."""
        matcher = AssemblyMatcher()
        motor = _make_nema17()
        bracket = _make_bracket()
        # motor first
        func1, _, _ = matcher._identify_functional_part(motor, bracket)
        # motor second
        func2, _, _ = matcher._identify_functional_part(bracket, motor)
        assert func1 is motor
        assert func2 is motor

    def test_servo_by_catalog_id(self):
        matcher = AssemblyMatcher()
        plate = _make_plate()
        servo = _make_sg90()
        func, struct, fid = matcher._identify_functional_part(plate, servo)
        assert func is servo
        assert fid == "servo_sg90"

    def test_bearing_by_catalog_id(self):
        matcher = AssemblyMatcher()
        housing = _make_plate()
        bearing = _make_bearing_608()
        func, struct, fid = matcher._identify_functional_part(housing, bearing)
        assert func is bearing
        assert fid == "bearing_608"

    def test_category_fallback_motor(self):
        """Part with 'stepper' category should be detected even without catalog entry."""
        matcher = AssemblyMatcher()
        unknown_motor = Part("custom_motor", "stepper", "custom motor",
                             dimensions={"length": 50, "width": 50, "height": 60})
        bracket = _make_bracket()
        func, struct, fid = matcher._identify_functional_part(bracket, unknown_motor)
        assert func is unknown_motor
        assert fid is None  # no catalog match, only category match

    def test_category_fallback_sensor(self):
        matcher = AssemblyMatcher()
        sensor = Part("my_sensor", "sensor", "custom sensor",
                      dimensions={"length": 20, "width": 20, "height": 5})
        plate = _make_plate()
        func, _, _ = matcher._identify_functional_part(plate, sensor)
        assert func is sensor

    def test_both_structural_returns_none(self):
        matcher = AssemblyMatcher()
        a = _make_generic("block_a")
        b = _make_generic("block_b")
        func, struct, fid = matcher._identify_functional_part(a, b)
        assert func is None
        assert struct is None
        assert fid is None


# ===========================================================================
# 2. MountingInterface-driven feature generation
# ===========================================================================

class TestInterfaceFeatureGeneration:
    """Step 2a: generate features from MountingInterface."""

    def test_nema17_generates_holes_and_bore(self):
        matcher = AssemblyMatcher()
        result = matcher.auto_match(_make_bracket(), _make_nema17(), _bolted())
        assert len(result.structural_ops) > 0
        # Should have features_summary mentioning holes and bore
        assert any("mounting holes" in f for f in result.features_summary)
        assert any("bore" in f for f in result.features_summary)

    def test_nema17_four_holes(self):
        """NEMA17 has 4 mounting holes at ±15.5mm pattern."""
        matcher = AssemblyMatcher()
        result = matcher.auto_match(_make_bracket(), _make_nema17(), _bolted())
        hole_ops = [o for o in result.structural_ops if "hole" in o.get("name", "")]
        assert len(hole_ops) >= 4

    def test_nema23_four_holes(self):
        matcher = AssemblyMatcher()
        result = matcher.auto_match(_make_bracket(), _make_nema23(), _bolted("M5"))
        assert len(result.structural_ops) > 0
        hole_ops = [o for o in result.structural_ops if "hole" in o.get("name", "")]
        assert len(hole_ops) >= 4

    def test_servo_sg90_generates_pocket(self):
        """SG90 has a body pocket in its MountingInterface."""
        matcher = AssemblyMatcher()
        result = matcher.auto_match(_make_plate(), _make_sg90(), _bolted("M2"))
        assert any("pocket" in f for f in result.features_summary)

    def test_bearing_608_press_fit_bore(self):
        """608 bearing uses press-fit, should generate bore."""
        matcher = AssemblyMatcher()
        result = matcher.auto_match(_make_plate(), _make_bearing_608(), _press_fit())
        assert any("bore" in f.lower() or "press" in f.lower() for f in result.features_summary)

    def test_interface_used_in_result(self):
        matcher = AssemblyMatcher()
        result = matcher.auto_match(_make_bracket(), _make_nema17(), _bolted())
        assert "nema17_stepper" in result.interface_used
        assert "through_hole" in result.interface_used

    def test_fastener_ops_generated_for_bolted(self):
        """Bolted connections should produce fastener model ops."""
        matcher = AssemblyMatcher()
        result = matcher.auto_match(_make_bracket(), _make_nema17(), _bolted())
        assert len(result.fastener_ops) > 0


# ===========================================================================
# 3. Heuristic fallback
# ===========================================================================

class TestHeuristicFallback:
    """Step 2b: when no MountingInterface exists, use heuristic."""

    def test_both_structural_uses_heuristic(self):
        matcher = AssemblyMatcher()
        a = _make_generic("block_a")
        b = _make_generic("block_b")
        result = matcher.auto_match(a, b, _bolted())
        assert result.interface_used == "heuristic"

    def test_unknown_functional_uses_heuristic(self):
        """Functional part detected by category but no catalog entry → heuristic."""
        matcher = AssemblyMatcher()
        motor = Part("custom_stepper", "stepper", "custom",
                     dimensions={"length": 42, "width": 42, "height": 40})
        bracket = _make_bracket()
        result = matcher.auto_match(bracket, motor, _bolted())
        # Category match gives func_id=None, so no interface → heuristic
        assert result.interface_used == "heuristic"


# ===========================================================================
# 4. Fastener selection
# ===========================================================================

class TestFastenerSelection:
    """Step 3: select bolt length, nut, washer."""

    def test_bolted_selects_fastener(self):
        matcher = AssemblyMatcher()
        result = matcher.auto_match(_make_bracket(), _make_nema17(), _bolted("M3"))
        sel = result.fastener_selection
        assert sel.bolt_size == "M3"
        assert sel.bolt_length > 0

    def test_bolted_m5_selects_fastener(self):
        matcher = AssemblyMatcher()
        result = matcher.auto_match(_make_bracket(), _make_nema23(), _bolted("M5"))
        sel = result.fastener_selection
        assert sel.bolt_size == "M5"
        assert sel.bolt_length > 0

    def test_press_fit_no_fastener(self):
        """Press-fit connections should not select fasteners."""
        matcher = AssemblyMatcher()
        result = matcher.auto_match(_make_plate(), _make_bearing_608(), _press_fit())
        assert result.fastener_selection.bolt_size == ""
        assert result.fastener_selection.bolt_length == 0.0

    def test_fastener_summary_included(self):
        matcher = AssemblyMatcher()
        result = matcher.auto_match(_make_bracket(), _make_nema17(), _bolted())
        assert any("Fasteners" in f for f in result.features_summary)

    def test_default_bolt_size_is_m3(self):
        """If no bolt_size specified, should default to M3."""
        conn = ConnectionMethod(type="bolted")  # bolt_size defaults to "M3"
        matcher = AssemblyMatcher()
        result = matcher.auto_match(_make_bracket(), _make_nema17(), conn)
        assert result.fastener_selection.bolt_size == "M3"


# ===========================================================================
# 5. Constraint setup
# ===========================================================================

class TestConstraintSetup:
    """Step 4: determine mating constraint type."""

    def test_bolted_gives_coincident(self):
        matcher = AssemblyMatcher()
        result = matcher.auto_match(_make_bracket(), _make_nema17(), _bolted())
        assert result.constraint_type == "coincident"

    def test_press_fit_gives_concentric(self):
        matcher = AssemblyMatcher()
        result = matcher.auto_match(_make_plate(), _make_bearing_608(), _press_fit())
        assert result.constraint_type == "concentric"

    def test_snap_fit_gives_coincident(self):
        matcher = AssemblyMatcher()
        result = matcher.auto_match(_make_generic(), _make_generic("other"), _snap_fit())
        assert result.constraint_type == "coincident"

    def test_anchor_sets_parent_entity(self):
        matcher = AssemblyMatcher()
        result = matcher.auto_match(_make_bracket(), _make_nema17(), _bolted(), anchor="top")
        assert result.parent_entity == ("face", "top")
        assert result.child_entity == ("face", "bottom")

    def test_anchor_left_gives_right_child(self):
        matcher = AssemblyMatcher()
        result = matcher.auto_match(_make_bracket(), _make_nema17(), _bolted(), anchor="left")
        assert result.parent_entity == ("face", "left")
        assert result.child_entity == ("face", "right")


# ===========================================================================
# 6. Verification
# ===========================================================================

class TestVerification:
    """Step 5: validate the match result."""

    def test_valid_nema17_bolted(self):
        matcher = AssemblyMatcher()
        result = matcher.auto_match(_make_bracket(), _make_nema17(), _bolted())
        assert result.valid is True

    def test_bolted_no_holes_gives_warning(self):
        """If no hole ops generated for bolted connection, should warn."""
        matcher = AssemblyMatcher()
        result = AssemblyMatchResult()
        result.fastener_selection.bolt_size = "M3"
        result.structural_ops = []  # no holes
        matcher._verify_match(_bolted(), result)
        assert any("No structural features" in w for w in result.warnings)

    def test_bolted_no_bolt_size_invalid(self):
        matcher = AssemblyMatcher()
        result = AssemblyMatchResult()
        result.structural_ops = [{"name": "hole", "type": "circle"}]
        matcher._verify_match(_bolted(), result)
        assert result.valid is False
        assert any("Bolt size not determined" in w for w in result.warnings)

    def test_press_fit_no_bore_warns(self):
        matcher = AssemblyMatcher()
        result = AssemblyMatchResult()
        result.structural_ops = [{"name": "chamfer"}]  # no bore
        matcher._verify_match(_press_fit(), result)
        assert any("press-fit bore" in w for w in result.warnings)


# ===========================================================================
# 7. Helper functions
# ===========================================================================

class TestHelpers:

    @pytest.mark.parametrize("face,expected", [
        ("top", "bottom"),
        ("bottom", "top"),
        ("left", "right"),
        ("right", "left"),
        ("front", "back"),
        ("back", "front"),
    ])
    def test_opposite_face(self, face, expected):
        assert _opposite_face(face) == expected

    def test_opposite_face_unknown(self):
        assert _opposite_face("diagonal") == "bottom"


# ===========================================================================
# 8. Integration / end-to-end
# ===========================================================================

class TestEndToEnd:
    """Full pipeline tests covering multiple scenarios."""

    def test_nema17_to_bracket_bolted(self):
        """Classic scenario: mount NEMA17 onto L-bracket with M3 bolts."""
        matcher = AssemblyMatcher()
        result = matcher.auto_match(
            _make_bracket(length=80, width=60, thickness=6),
            _make_nema17(),
            _bolted("M3"),
            anchor="top",
        )
        assert result.structural_part_name == "bracket"
        assert result.functional_part_name == "nema17_stepper"
        assert "nema17_stepper" in result.interface_used
        assert len(result.structural_ops) >= 4  # at least 4 holes
        assert result.fastener_selection.bolt_size == "M3"
        assert result.fastener_selection.bolt_length > 0
        assert result.constraint_type == "coincident"
        assert result.valid is True

    def test_servo_to_plate_bolted(self):
        """Mount SG90 servo onto plate with M2 bolts."""
        matcher = AssemblyMatcher()
        result = matcher.auto_match(
            _make_plate(length=50, width=40, thickness=4),
            _make_sg90(),
            _bolted("M2"),
            anchor="top",
        )
        assert result.functional_part_name == "servo_sg90"
        assert len(result.structural_ops) > 0
        assert result.fastener_selection.bolt_size == "M2"

    def test_bearing_press_fit_into_housing(self):
        """Press-fit 608 bearing into plate."""
        matcher = AssemblyMatcher()
        result = matcher.auto_match(
            _make_plate(length=40, width=40, thickness=8),
            _make_bearing_608(),
            _press_fit(),
        )
        assert result.functional_part_name == "bearing_608"
        assert result.constraint_type == "concentric"
        # No bolt fasteners for press-fit
        assert result.fastener_selection.bolt_size == ""

    def test_sensor_to_plate_bolted(self):
        """Mount MPU6050 sensor onto plate."""
        matcher = AssemblyMatcher()
        result = matcher.auto_match(
            _make_plate(length=40, width=30, thickness=3),
            _make_sensor(),
            _bolted("M2"),
        )
        assert result.functional_part_name == "sensor_mpu6050"
        assert len(result.structural_ops) > 0

    def test_two_generic_parts_bolted(self):
        """Two unknown structural parts → heuristic path."""
        matcher = AssemblyMatcher()
        a = _make_generic("frame_piece")
        b = _make_generic("gusset")
        result = matcher.auto_match(a, b, _bolted("M3"))
        assert result.interface_used == "heuristic"
        # Should still produce a valid result
        assert result.valid is True
