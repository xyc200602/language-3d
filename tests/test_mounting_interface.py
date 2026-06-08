"""Tests for Task 76: Mounting interface standardization."""

import pytest

from src.lang3d.knowledge.parts_catalog import (
    BoltHole,
    AlignmentFeature,
    MountingInterface,
    MOUNTING_INTERFACES,
    PART_CATALOG,
    auto_match_interface,
    get_mounting_interface,
    get_template,
)
from src.lang3d.knowledge.mechanics import ConnectionMethod, Part
from src.lang3d.tools.connection_features import ConnectionFeatureEngine


# =====================================================================
# 1. Data model tests
# =====================================================================

class TestBoltHole:

    def test_defaults(self):
        bh = BoltHole(x=10, y=20, diameter=3.4)
        assert bh.x == 10
        assert bh.y == 20
        assert bh.diameter == 3.4
        assert bh.depth == 0.0
        assert bh.direction == (0.0, 0.0, -1.0)
        assert bh.hole_type == "through_hole"


class TestMountingInterface:

    def test_defaults(self):
        mi = MountingInterface(interface_type="through_hole")
        assert mi.interface_type == "through_hole"
        assert mi.holes == []
        assert mi.bore_diameter == 0.0
        assert mi.press_fit_interference == 0.0

    def test_with_holes(self):
        mi = MountingInterface(
            interface_type="through_hole",
            holes=[
                BoltHole(x=-15, y=-15, diameter=3.4),
                BoltHole(x=15, y=-15, diameter=3.4),
                BoltHole(x=-15, y=15, diameter=3.4),
                BoltHole(x=15, y=15, diameter=3.4),
            ],
        )
        assert len(mi.holes) == 4


class TestPartTemplateHasInterface:

    def test_mounting_interface_field_exists(self):
        t = get_template("nema17_stepper")
        assert hasattr(t, "mounting_interface")

    def test_default_is_none(self):
        t = get_template("l_bracket")
        assert t.mounting_interface is None


# =====================================================================
# 2. Interface registration tests
# =====================================================================

class TestInterfaceRegistration:

    EXPECTED_INTERFACE_PARTS = [
        "nema17_stepper", "nema23_stepper",
        "servo_sg90", "servo_mg996r", "servo_ds3218",
        "bearing_608", "bearing_623", "bearing_625",
        "sensor_rplidar_a1", "sensor_mpu6050", "sensor_esp32_cam",
        "encoder_as5600",
        "driver_l298n",
        "controller_arduino_uno", "controller_arduino_nano",
        "controller_esp32_devkit",
    ]

    def test_interface_count(self):
        assert len(MOUNTING_INTERFACES) == 16

    def test_all_expected_interfaces_registered(self):
        for part_id in self.EXPECTED_INTERFACE_PARTS:
            assert part_id in MOUNTING_INTERFACES, f"Missing interface: {part_id}"

    def test_get_mounting_interface_returns_interface(self):
        mi = get_mounting_interface("nema17_stepper")
        assert mi is not None
        assert isinstance(mi, MountingInterface)

    def test_get_mounting_interface_returns_none_for_unknown(self):
        mi = get_mounting_interface("nonexistent_part")
        assert mi is None


# =====================================================================
# 3. NEMA17 interface
# =====================================================================

class TestNEMA17Interface:

    def test_hole_count(self):
        mi = get_mounting_interface("nema17_stepper")
        assert len(mi.holes) == 4

    def test_hole_spacing(self):
        mi = get_mounting_interface("nema17_stepper")
        xs = {h.x for h in mi.holes}
        ys = {h.y for h in mi.holes}
        assert -15.5 in xs and 15.5 in xs
        assert -15.5 in ys and 15.5 in ys

    def test_hole_diameter(self):
        mi = get_mounting_interface("nema17_stepper")
        for h in mi.holes:
            assert h.diameter == 3.4

    def test_bore_diameter(self):
        mi = get_mounting_interface("nema17_stepper")
        assert mi.bore_diameter == 23.0

    def test_interface_type(self):
        mi = get_mounting_interface("nema17_stepper")
        assert mi.interface_type == "through_hole"


# =====================================================================
# 4. NEMA23 interface
# =====================================================================

class TestNEMA23Interface:

    def test_hole_count(self):
        mi = get_mounting_interface("nema23_stepper")
        assert len(mi.holes) == 4

    def test_larger_than_nema17(self):
        mi17 = get_mounting_interface("nema17_stepper")
        mi23 = get_mounting_interface("nema23_stepper")
        assert mi23.holes[0].diameter >= mi17.holes[0].diameter
        assert mi23.bore_diameter > mi17.bore_diameter


# =====================================================================
# 5. Servo interfaces
# =====================================================================

class TestServoInterfaces:

    def test_sg90_two_holes(self):
        mi = get_mounting_interface("servo_sg90")
        assert len(mi.holes) == 2
        assert mi.pocket_width == 11.8
        assert mi.pocket_height == 22.2

    def test_mg996r_four_holes(self):
        mi = get_mounting_interface("servo_mg996r")
        assert len(mi.holes) == 4
        assert mi.pocket_width == 19.7

    def test_ds3218_four_holes(self):
        mi = get_mounting_interface("servo_ds3218")
        assert len(mi.holes) == 4
        assert mi.pocket_width == 20.0

    def test_servo_interface_type(self):
        for pid in ["servo_sg90", "servo_mg996r", "servo_ds3218"]:
            mi = get_mounting_interface(pid)
            assert mi.interface_type == "through_hole"
            assert mi.pocket_width > 0


# =====================================================================
# 6. Bearing interfaces
# =====================================================================

class TestBearingInterfaces:

    def test_608_press_fit(self):
        mi = get_mounting_interface("bearing_608")
        assert mi.interface_type == "press_fit"
        assert mi.bore_diameter == 22.0
        assert mi.press_fit_interference > 0

    def test_623_press_fit(self):
        mi = get_mounting_interface("bearing_623")
        assert mi.interface_type == "press_fit"
        assert mi.bore_diameter == 10.0

    def test_625_press_fit(self):
        mi = get_mounting_interface("bearing_625")
        assert mi.interface_type == "press_fit"
        assert mi.bore_diameter == 16.0

    def test_bearing_has_shoulder(self):
        mi = get_mounting_interface("bearing_608")
        assert mi.shoulder_diameter > mi.bore_diameter

    def test_bearing_dimensions_match_catalog(self):
        """Bearing interface bore diameter should match catalog outer diameter."""
        mi = get_mounting_interface("bearing_608")
        t = get_template("bearing_608")
        od_param = next((p for p in t.parameters if p.name == "outer_diameter"), None)
        assert od_param is not None
        assert mi.bore_diameter == od_param.default


# =====================================================================
# 7. auto_match_interface tests
# =====================================================================

class TestAutoMatchInterface:

    def test_nema17_generates_holes(self):
        mi = get_mounting_interface("nema17_stepper")
        ops = auto_match_interface(
            structural_dims={"length": 50, "width": 50, "thickness": 5},
            interface=mi,
            anchor="top",
        )
        hole_ops = [o for o in ops if o["name"].startswith("mount_hole")]
        assert len(hole_ops) == 4
        for op in hole_ops:
            assert op["operation"] == "cut"
            assert op["radius"] == pytest.approx(3.4 / 2.0)

    def test_nema17_generates_bore(self):
        mi = get_mounting_interface("nema17_stepper")
        ops = auto_match_interface(
            structural_dims={"length": 50, "width": 50, "thickness": 5},
            interface=mi,
            anchor="top",
        )
        bore_ops = [o for o in ops if o["name"] == "center_bore"]
        assert len(bore_ops) == 1
        assert bore_ops[0]["radius"] == pytest.approx(23.0 / 2.0)

    def test_servo_generates_pocket(self):
        mi = get_mounting_interface("servo_sg90")
        ops = auto_match_interface(
            structural_dims={"length": 60, "width": 40, "thickness": 3},
            interface=mi,
            anchor="top",
        )
        pocket_ops = [o for o in ops if o["name"] == "body_pocket"]
        assert len(pocket_ops) == 1
        assert pocket_ops[0]["width"] == pytest.approx(11.8)
        assert pocket_ops[0]["depth"] == pytest.approx(22.2)

    def test_bearing_generates_press_fit_bore(self):
        mi = get_mounting_interface("bearing_608")
        ops = auto_match_interface(
            structural_dims={"length": 30, "width": 30, "thickness": 10},
            interface=mi,
            anchor="top",
        )
        pf_ops = [o for o in ops if o["name"] == "press_fit_bore"]
        assert len(pf_ops) == 1
        # Bore should be slightly smaller than bearing OD
        assert pf_ops[0]["radius"] < mi.bore_diameter / 2.0

    def test_bearing_generates_shoulder(self):
        mi = get_mounting_interface("bearing_608")
        ops = auto_match_interface(
            structural_dims={"length": 30, "width": 30, "thickness": 10},
            interface=mi,
            anchor="top",
        )
        sh_ops = [o for o in ops if o["name"] == "shoulder"]
        assert len(sh_ops) == 1

    def test_empty_ops_for_no_features(self):
        mi = MountingInterface(interface_type="snap_fit")
        ops = auto_match_interface(
            structural_dims={"length": 30, "width": 30, "thickness": 3},
            interface=mi,
        )
        assert ops == []

    def test_offset_applied(self):
        mi = get_mounting_interface("nema17_stepper")
        ops1 = auto_match_interface(
            structural_dims={"length": 50, "width": 50, "thickness": 5},
            interface=mi,
            anchor="top",
            offset_x=10,
        )
        ops2 = auto_match_interface(
            structural_dims={"length": 50, "width": 50, "thickness": 5},
            interface=mi,
            anchor="top",
            offset_x=0,
        )
        # With offset, hole positions should differ
        h1 = next(o for o in ops1 if o["name"] == "mount_hole_0")
        h2 = next(o for o in ops2 if o["name"] == "mount_hole_0")
        assert h1["x"] != h2["x"]


# =====================================================================
# 8. ConnectionFeatureEngine integration with MountingInterface
# =====================================================================

class TestConnectionFeatureEngineWithInterface:

    def test_engine_uses_interface_when_provided(self):
        engine = ConnectionFeatureEngine()
        bracket = Part("test_bracket", "bracket", "test",
                        dimensions={"length": 50, "width": 50, "thickness": 5})
        conn = ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=4)
        result = engine.generate_features(
            structural_part=bracket,
            connection=conn,
            anchor="top",
            functional_part_id="nema17_stepper",
        )
        assert len(result.ops) > 0
        assert any("mount_hole" in o["name"] for o in result.ops)

    def test_engine_falls_back_without_interface(self):
        engine = ConnectionFeatureEngine()
        bracket = Part("test_bracket", "bracket", "test",
                        dimensions={"length": 50, "width": 50, "thickness": 5})
        conn = ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=4)
        result = engine.generate_features(
            structural_part=bracket,
            connection=conn,
            anchor="top",
            functional_part_id="nonexistent",
        )
        # Should fall back to heuristic and still generate features
        assert len(result.ops) > 0

    def test_engine_bearing_press_fit(self):
        engine = ConnectionFeatureEngine()
        housing = Part("test_housing", "housing", "test",
                        dimensions={"length": 30, "width": 30, "thickness": 10})
        conn = ConnectionMethod(type="press_fit")
        result = engine.generate_features(
            structural_part=housing,
            connection=conn,
            anchor="top",
            functional_part_id="bearing_608",
        )
        assert len(result.ops) > 0
        assert any("press_fit" in o["name"] for o in result.ops)

    def test_engine_servo_with_pocket(self):
        engine = ConnectionFeatureEngine()
        bracket = Part("servo_bracket", "bracket", "test",
                        dimensions={"length": 60, "width": 40, "thickness": 3})
        conn = ConnectionMethod(type="bolted", bolt_size="M2.5", bolt_count=2)
        result = engine.generate_features(
            structural_part=bracket,
            connection=conn,
            anchor="top",
            functional_part_id="servo_sg90",
        )
        assert len(result.ops) > 0
        assert any("body_pocket" in o["name"] for o in result.ops)

    def test_engine_backward_compatible_no_part_id(self):
        """Without functional_part_id, engine should work as before."""
        engine = ConnectionFeatureEngine()
        bracket = Part("test_bracket", "bracket", "test",
                        dimensions={"length": 50, "width": 50, "thickness": 5})
        conn = ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=4)
        result = engine.generate_features(
            structural_part=bracket,
            connection=conn,
            anchor="top",
        )
        assert len(result.ops) > 0


# =====================================================================
# 9. Cross-module consistency
# =====================================================================

class TestCrossModuleConsistency:

    def test_all_functional_actuators_have_interfaces(self):
        """All functional actuators in catalog should have mounting interfaces."""
        actuator_ids = ["nema17_stepper", "nema23_stepper",
                         "servo_sg90", "servo_mg996r", "servo_ds3218"]
        for pid in actuator_ids:
            assert get_mounting_interface(pid) is not None, f"Missing interface: {pid}"

    def test_all_functional_bearings_have_interfaces(self):
        bearing_ids = ["bearing_608", "bearing_623", "bearing_625"]
        for pid in bearing_ids:
            assert get_mounting_interface(pid) is not None, f"Missing interface: {pid}"

    def test_hole_positions_within_face(self):
        """All hole positions should be within reasonable range of the part face."""
        for part_id, mi in MOUNTING_INTERFACES.items():
            t = get_template(part_id)
            if t is None:
                continue
            # Get the face dimensions from the first standard size
            if t.standard_sizes:
                dims = t.standard_sizes[0]
            else:
                dims = {p.name: p.default for p in t.parameters}
            for h in mi.holes:
                # Hole positions should be within ±100mm of center (reasonable)
                assert abs(h.x) < 100, f"{part_id}: hole x={h.x} out of range"
                assert abs(h.y) < 100, f"{part_id}: hole y={h.y} out of range"
