"""Tests for the connection feature engine (Task 71).

Tests cover:
1. Query helpers (clearance holes, torque, fastener dimensions)
2. Each connection type feature generation
3. Auto-layout bolt positions
4. Fastener model generation
5. Integration with part_feature_engine via joints
6. Assembly-level feature generation
"""

from __future__ import annotations

import math
import pytest

from lang3d.knowledge.mechanics import (
    BoltHole,
    ConnectionMethod,
    Joint,
    Part,
    Assembly,
)
from lang3d.tools.connection_features import (
    ConnectionFeatureEngine,
    ConnectionFeatureResult,
    get_clearance_hole,
    get_torque_recommendation,
    get_bolt_head_dims,
    get_nut_dims,
    get_washer_dims,
    get_thread_insert_dims,
    get_bearing_spec,
    merge_connection_ops,
    generate_assembly_connection_features,
    _pick_structural_part,
)
from lang3d.tools.part_feature_engine import generate_ops


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def engine() -> ConnectionFeatureEngine:
    return ConnectionFeatureEngine()


@pytest.fixture
def bracket() -> Part:
    """L-bracket: structural part that receives bolt holes."""
    return Part(
        name="motor_bracket",
        category="structural",
        description="NEMA17 motor bracket",
        material="PLA",
        dimensions={"length": 50, "width": 40, "height": 8},
    )


@pytest.fixture
def nema17() -> Part:
    """NEMA17 motor: functional part (should NOT receive holes)."""
    return Part(
        name="nema17_motor",
        category="actuator",
        description="NEMA17 stepper motor",
        material="Steel",
        dimensions={"length": 42.3, "width": 42.3, "height": 40},
    )


@pytest.fixture
def housing() -> Part:
    """Bearing housing: structural part with bearing seat."""
    return Part(
        name="bearing_housing_608",
        category="structural",
        description="608 bearing housing",
        material="PLA",
        dimensions={
            "outer_diameter": 30,
            "height": 10,
            "bore_diameter": 22.0,
        },
    )


@pytest.fixture
def bolted_connection() -> ConnectionMethod:
    return ConnectionMethod(
        type="bolted",
        bolt_size="M3",
        bolt_count=4,
        torque_nm=0.3,
    )


@pytest.fixture
def press_fit_connection() -> ConnectionMethod:
    return ConnectionMethod(
        type="press_fit",
        interference_mm=0.05,
    )


@pytest.fixture
def snap_fit_connection() -> ConnectionMethod:
    return ConnectionMethod(
        type="snap_fit",
        snap_count=2,
        snap_force_n=5.0,
    )


@pytest.fixture
def adhesive_connection() -> ConnectionMethod:
    return ConnectionMethod(
        type="adhesive",
        adhesive_type="epoxy",
        bond_area_mm2=500,
    )


@pytest.fixture
def welded_connection() -> ConnectionMethod:
    return ConnectionMethod(
        type="welded",
        weld_type="butt",
    )


@pytest.fixture
def magnetic_connection() -> ConnectionMethod:
    return ConnectionMethod(type="magnetic")


# ============================================================================
# 1. Query helpers
# ============================================================================


class TestClearanceHoles:
    def test_m3_normal(self):
        assert get_clearance_hole("M3") == 3.4

    def test_m4_normal(self):
        assert get_clearance_hole("M4") == 4.5

    def test_m6_normal(self):
        assert get_clearance_hole("M6") == 6.6

    def test_m3_close(self):
        assert get_clearance_hole("M3", "close") == 3.2

    def test_m3_loose(self):
        assert get_clearance_hole("M3", "loose") == 3.9

    def test_unknown_size_fallback(self):
        result = get_clearance_hole("M99")
        assert result == 99.4  # 99 + 0.4


class TestTorqueRecommendation:
    def test_pla_m3(self):
        assert get_torque_recommendation("M3", "PLA") == 0.3

    def test_steel_m3(self):
        assert get_torque_recommendation("M3", "Steel") == 1.0

    def test_aluminum_m4(self):
        assert get_torque_recommendation("M4", "Aluminum") == 2.5


class TestFastenerDimensions:
    def test_bolt_head_m3(self):
        head_d, head_h = get_bolt_head_dims("M3")
        assert head_d == 5.5
        assert head_h == 3.0

    def test_nut_m3(self):
        w, h = get_nut_dims("M3")
        assert w == 5.5
        assert h == 2.4

    def test_washer_m3(self):
        id_, od, t = get_washer_dims("M3")
        assert id_ == 3.2
        assert od == 7.0
        assert t == 0.5

    def test_thread_insert_m3(self):
        od, length, install_d = get_thread_insert_dims("M3")
        assert od == 4.6
        assert length == 5.6
        assert install_d == 4.7

    def test_bearing_spec_608(self):
        spec = get_bearing_spec("608")
        assert spec is not None
        id_, od, w = spec
        assert id_ == 8.0
        assert od == 22.0
        assert w == 7.0

    def test_bearing_spec_unknown(self):
        assert get_bearing_spec("XYZ") is None


# ============================================================================
# 2. Bolted connection features
# ============================================================================


class TestBoltedFeatures:
    def test_generates_clearance_holes(self, engine, bracket, bolted_connection):
        result = engine.generate_features(bracket, bolted_connection, "top")
        assert len(result.ops) > 0
        # Should have cylinder operations for holes
        cyl_ops = [op for op in result.ops if op.get("type") == "make_cylinder"]
        assert len(cyl_ops) >= 4  # 4 clearance holes

    def test_generates_counterbores(self, engine, bracket, bolted_connection):
        result = engine.generate_features(bracket, bolted_connection, "top")
        # Counterbores should also be cylinders
        all_ops = result.ops
        # At least 8 cylinders: 4 holes + 4 counterbores
        cyl_ops = [op for op in all_ops if op.get("type") == "make_cylinder"]
        assert len(cyl_ops) >= 8

    def test_generates_fastener_models(self, engine, bracket, bolted_connection):
        result = engine.generate_features(bracket, bolted_connection, "top")
        assert len(result.fastener_ops) > 0
        # Each set has bolt head + shank + washer + nut
        assert len(result.fastener_ops) >= 4 * 5  # 4 sets × ~5 ops each

    def test_features_generated_description(self, engine, bracket, bolted_connection):
        result = engine.generate_features(bracket, bolted_connection, "top")
        assert len(result.features_generated) >= 2
        assert "M3" in result.features_generated[0]
        assert "clearance" in result.features_generated[0].lower()

    def test_custom_bolt_holes(self, engine, bracket):
        """Explicit bolt hole positions."""
        conn = ConnectionMethod(
            type="bolted",
            bolt_size="M4",
            bolt_count=2,
            bolt_holes=[
                BoltHole(position=(0.25, 0.5, 0), diameter=4.5, bolt_size="M4"),
                BoltHole(position=(0.75, 0.5, 0), diameter=4.5, bolt_size="M4"),
            ],
        )
        result = engine.generate_features(bracket, conn, "top")
        assert len(result.ops) > 0
        assert "M4" in result.features_generated[0]


class TestBoltedAutoLayout:
    def test_1_bolt(self, engine, bracket):
        conn = ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=1)
        result = engine.generate_features(bracket, conn, "top")
        cyl_ops = [op for op in result.ops if op.get("type") == "make_cylinder"]
        # 1 hole + 1 counterbore = 2 cylinders
        assert len(cyl_ops) >= 2

    def test_2_bolts(self, engine, bracket):
        conn = ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=2)
        result = engine.generate_features(bracket, conn, "top")
        cyl_ops = [op for op in result.ops if op.get("type") == "make_cylinder"]
        assert len(cyl_ops) >= 4  # 2 holes + 2 counterbores

    def test_6_bolts_grid(self, engine, bracket):
        conn = ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=6)
        result = engine.generate_features(bracket, conn, "top")
        cyl_ops = [op for op in result.ops if op.get("type") == "make_cylinder"]
        assert len(cyl_ops) >= 12  # 6 holes + 6 counterbores


# ============================================================================
# 3. Press-fit connection features
# ============================================================================


class TestPressFitFeatures:
    def test_generates_bore_and_shoulder(self, engine, housing, press_fit_connection):
        result = engine.generate_features(housing, press_fit_connection, "top")
        assert len(result.ops) > 0
        # Should have bore cylinder and shoulder cylinder
        cyl_ops = [op for op in result.ops if op.get("type") == "make_cylinder"]
        assert len(cyl_ops) >= 2

    def test_bore_smaller_than_nominal(self, engine, housing, press_fit_connection):
        """Press-fit bore should be smaller than bearing OD by interference."""
        result = engine.generate_features(housing, press_fit_connection, "top")
        desc = result.features_generated[0]
        # The bore diameter should be 22.0 - 0.05 = 21.95mm
        assert "21.95" in desc

    def test_warning_when_no_bore_spec(self, engine):
        """Should warn if bore diameter cannot be inferred."""
        part = Part(
            name="unknown_part",
            category="structural",
            description="No bore spec",
            dimensions={"length": 20, "width": 20, "height": 10},
        )
        conn = ConnectionMethod(type="press_fit", interference_mm=0.05)
        result = engine.generate_features(part, conn, "top")
        assert len(result.warnings) > 0
        assert "bore" in result.warnings[0].lower()


# ============================================================================
# 4. Snap-fit connection features
# ============================================================================


class TestSnapFitFeatures:
    def test_generates_snap_hooks(self, engine, bracket, snap_fit_connection):
        result = engine.generate_features(bracket, snap_fit_connection, "top")
        assert len(result.ops) > 0
        box_ops = [op for op in result.ops if op.get("type") == "make_box"]
        # Each snap has a hook box + undercut box
        assert len(box_ops) >= 4  # 2 snaps × 2 boxes each

    def test_custom_snap_count(self, engine, bracket):
        conn = ConnectionMethod(type="snap_fit", snap_count=4, snap_force_n=10.0)
        result = engine.generate_features(bracket, conn, "front")
        box_ops = [op for op in result.ops if op.get("type") == "make_box"]
        assert len(box_ops) >= 8  # 4 snaps × 2 boxes each


# ============================================================================
# 5. Adhesive connection features
# ============================================================================


class TestAdhesiveFeatures:
    def test_generates_grooves(self, engine, bracket, adhesive_connection):
        result = engine.generate_features(bracket, adhesive_connection, "top")
        assert len(result.ops) > 0
        box_ops = [op for op in result.ops if op.get("type") == "make_box"]
        assert len(box_ops) >= 1  # At least one groove

    def test_bond_area_in_description(self, engine, bracket, adhesive_connection):
        result = engine.generate_features(bracket, adhesive_connection, "top")
        assert len(result.features_generated) > 0
        assert "bond" in result.features_generated[0].lower() or "groove" in result.features_generated[0].lower()


# ============================================================================
# 6. Welded connection features
# ============================================================================


class TestWeldedFeatures:
    def test_butt_weld_generates_bevel(self, engine, bracket, welded_connection):
        result = engine.generate_features(bracket, welded_connection, "top")
        assert len(result.ops) > 0
        assert any("bevel" in desc.lower() or "groove" in desc.lower()
                    for desc in result.features_generated)

    def test_fillet_weld_no_prep(self, engine, bracket):
        conn = ConnectionMethod(type="welded", weld_type="fillet")
        result = engine.generate_features(bracket, conn, "top")
        assert "no joint preparation" in result.features_generated[0].lower()


# ============================================================================
# 7. Magnetic connection features
# ============================================================================


class TestMagneticFeatures:
    def test_generates_magnet_pocket(self, engine, bracket, magnetic_connection):
        result = engine.generate_features(bracket, magnetic_connection, "top")
        assert len(result.ops) > 0
        cyl_ops = [op for op in result.ops if op.get("type") == "make_cylinder"]
        assert len(cyl_ops) >= 1
        assert "magnet" in result.features_generated[0].lower()


# ============================================================================
# 8. Unknown connection type
# ============================================================================


class TestUnknownConnection:
    def test_unknown_type_warning(self, engine, bracket):
        conn = ConnectionMethod(type="dovetail")
        result = engine.generate_features(bracket, conn, "top")
        assert len(result.warnings) > 0
        assert "Unknown" in result.warnings[0]


# ============================================================================
# 9. Fastener model generation
# ============================================================================


class TestFastenerModels:
    def test_fastener_set_completeness(self, engine):
        ops = engine._generate_fastener_set("M3", 8.0, 0)
        # Should have: head cylinder, shank cylinder, shank move, fuse,
        # washer cyl_with_hole, nut cyl_with_hole
        assert len(ops) >= 6

    def test_bolt_length_selection(self):
        assert ConnectionFeatureEngine._select_bolt_length(5.0) == 6.0
        assert ConnectionFeatureEngine._select_bolt_length(7.0) == 8.0
        assert ConnectionFeatureEngine._select_bolt_length(12.0) == 12.0  # exact match
        assert ConnectionFeatureEngine._select_bolt_length(13.0) == 16.0  # round up
        assert ConnectionFeatureEngine._select_bolt_length(60.0) == 60.0


# ============================================================================
# 10. Merge connection ops
# ============================================================================


class TestMergeOps:
    def test_merge_adds_boolean_cuts(self):
        base_ops = [
            {"type": "new_doc", "name": "test"},
            {"type": "make_box", "length": 10, "width": 10, "height": 5, "name": "test"},
            {"type": "export_stl", "object": "test", "name": "test", "path": "/tmp/test.stl"},
        ]
        connection_ops = [
            {"type": "make_cylinder", "radius": 1.5, "height": 10, "name": "test_bolt_hole"},
            {"type": "move", "object": "test_bolt_hole", "dx": 5, "dy": 5, "dz": -2},
        ]
        merged = merge_connection_ops(base_ops, connection_ops, "test")
        # Should have boolean cut operations
        bool_cuts = [op for op in merged if op.get("type") == "boolean" and op.get("operation") == "cut"]
        assert len(bool_cuts) >= 1
        # Export should still be present
        exports = [op for op in merged if op.get("type") == "export_stl"]
        assert len(exports) == 1

    def test_merge_empty_connection_ops(self):
        base_ops = [
            {"type": "new_doc", "name": "test"},
            {"type": "export_stl", "object": "test", "name": "test", "path": "/tmp/test.stl"},
        ]
        merged = merge_connection_ops(base_ops, [], "test")
        assert len(merged) == len(base_ops)


# ============================================================================
# 11. Part picking logic
# ============================================================================


class TestPickStructuralPart:
    def test_functional_vs_structural(self):
        motor = Part("motor", "actuator", "Motor")
        bracket = Part("bracket", "structural", "Bracket")
        assert _pick_structural_part(motor, bracket) == bracket

    def test_both_structural(self):
        plate = Part("plate", "structural", "Plate")
        link = Part("link", "structural", "Link")
        # When both are structural, returns first (parent)
        result = _pick_structural_part(plate, link)
        assert result == plate

    def test_sensor_vs_bracket(self):
        sensor = Part("imu_sensor", "sensor", "IMU")
        mount = Part("mounting_plate", "structural", "Mount")
        assert _pick_structural_part(sensor, mount) == mount

    def test_name_based_detection(self):
        """Parts with motor/servo in name are treated as functional."""
        servo = Part("arm_servo_holder", "structural", "Servo holder")
        bracket = Part("L_bracket", "structural", "Bracket")
        # 'servo' in name should make it functional
        result = _pick_structural_part(servo, bracket)
        assert result == bracket


# ============================================================================
# 12. Assembly-level feature generation
# ============================================================================


class TestAssemblyFeatures:
    def test_generates_features_for_all_joints(self, engine):
        bracket = Part("bracket", "structural", "Bracket",
                       dimensions={"length": 50, "width": 40, "height": 8})
        motor = Part("motor", "actuator", "Motor",
                     dimensions={"length": 42, "width": 42, "height": 40})
        joint = Joint(
            type="fixed",
            parent="motor",
            child="bracket",
            parent_anchor="bottom",
            child_anchor="top",
            connection=ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=4),
        )
        results = generate_assembly_connection_features([bracket, motor], [joint])
        # Should generate features on the bracket (structural part)
        assert "bracket" in results
        assert len(results["bracket"].ops) > 0

    def test_skips_joints_without_connection(self):
        part_a = Part("a", "structural", "A", dimensions={"length": 10, "width": 10, "height": 5})
        part_b = Part("b", "structural", "B", dimensions={"length": 10, "width": 10, "height": 5})
        joint = Joint(type="fixed", parent="a", child="b")
        results = generate_assembly_connection_features([part_a, part_b], [joint])
        assert len(results) == 0


# ============================================================================
# 13. Integration with part_feature_engine
# ============================================================================


class TestPartFeatureEngineIntegration:
    def test_generate_ops_without_joints(self):
        """Backward compatible: no joints → same behavior as before."""
        part = Part("test_bracket", "structural", "Test",
                    dimensions={"length": 30, "width": 20, "height": 5})
        ops = generate_ops(part)
        assert any(op.get("type") == "new_doc" for op in ops)
        assert any(op.get("type") == "export_stl" for op in ops)

    def test_generate_ops_with_joints(self):
        """With joints → connection features merged in."""
        bracket = Part("test_bracket", "structural", "Test bracket",
                       dimensions={"length": 50, "width": 40, "height": 8})
        joint = Joint(
            type="fixed",
            parent="test_bracket",
            child="some_motor",
            parent_anchor="top",
            child_anchor="bottom",
            connection=ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=4),
        )
        ops = generate_ops(bracket, joints=[joint])
        # Should have more ops than without joints
        ops_no_joints = generate_ops(bracket)
        assert len(ops) > len(ops_no_joints)
        # Should have boolean cut operations for the holes
        cuts = [op for op in ops if op.get("type") == "boolean"
                and op.get("operation") == "cut"]
        assert len(cuts) >= 1

    def test_features_generated_flag(self):
        """ConnectionMethod.features_generated should be set after processing."""
        bracket = Part("test_bracket", "structural", "Test",
                       dimensions={"length": 50, "width": 40, "height": 8})
        conn = ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=4)
        assert conn.features_generated is False
        joint = Joint(
            type="fixed",
            parent="test_bracket",
            child="motor",
            parent_anchor="top",
            child_anchor="bottom",
            connection=conn,
        )
        generate_ops(bracket, joints=[joint])
        assert conn.features_generated is True


# ============================================================================
# 14. Edge cases
# ============================================================================


class TestEdgeCases:
    def test_zero_bolt_count_defaults_to_4(self, engine, bracket):
        conn = ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=0)
        result = engine.generate_features(bracket, conn, "top")
        # Should default to 4 bolts
        assert len(result.ops) > 0

    def test_different_anchor_faces(self, engine, bracket, bolted_connection):
        """Should work for all 6 anchor faces."""
        for anchor in ("top", "bottom", "left", "right", "front", "back"):
            result = engine.generate_features(bracket, bolted_connection, anchor)
            assert len(result.ops) > 0, f"Failed for anchor={anchor}"

    def test_cylindrical_part(self, engine, housing, press_fit_connection):
        """Press-fit on cylindrical part should work."""
        result = engine.generate_features(housing, press_fit_connection, "top")
        assert len(result.ops) > 0

    def test_connection_method_describe(self):
        """ConnectionMethod.describe() should work for all types."""
        for t in ("bolted", "press_fit", "snap_fit", "adhesive", "welded", "magnetic"):
            conn = ConnectionMethod(type=t)
            desc = conn.describe()
            assert isinstance(desc, str)
            assert len(desc) > 0


# ============================================================================
# Phase 2: Dowel pin alignment features
# ============================================================================


class TestDowelPinFeatures:
    """Test dowel pin feature generation."""

    def test_dowel_pin_generates_ops(self, engine, bracket):
        conn = ConnectionMethod(type="dowel_pin")
        result = engine.generate_features(bracket, conn, "top")
        assert len(result.ops) > 0
        assert len(result.fastener_ops) > 0

    def test_dowel_pin_creates_holes(self, engine, bracket):
        conn = ConnectionMethod(type="dowel_pin")
        result = engine.generate_features(bracket, conn, "top")
        # Should have 2 holes (default 2 pins)
        make_cyl_ops = [op for op in result.ops if op.get("type") == "make_cylinder"]
        assert len(make_cyl_ops) >= 2

    def test_dowel_pin_creates_pin_models(self, engine, bracket):
        conn = ConnectionMethod(type="dowel_pin")
        result = engine.generate_features(bracket, conn, "top")
        # Should have 2 pin models
        pin_ops = [op for op in result.fastener_ops if op.get("type") == "make_cylinder"]
        assert len(pin_ops) >= 2

    def test_dowel_pin_feature_description(self, engine, bracket):
        conn = ConnectionMethod(type="dowel_pin")
        result = engine.generate_features(bracket, conn, "top")
        assert len(result.features_generated) > 0
        assert "dowel" in result.features_generated[0].lower()

    def test_dowel_pin_on_different_faces(self, engine, bracket):
        conn = ConnectionMethod(type="dowel_pin")
        for anchor in ("top", "front", "left"):
            result = engine.generate_features(bracket, conn, anchor)
            assert len(result.ops) > 0, f"Failed for anchor={anchor}"
            assert len(result.fastener_ops) > 0, f"No pin models for anchor={anchor}"

    def test_dowel_pin_describe(self):
        conn = ConnectionMethod(type="dowel_pin")
        desc = conn.describe()
        assert isinstance(desc, str)
        assert len(desc) > 0
