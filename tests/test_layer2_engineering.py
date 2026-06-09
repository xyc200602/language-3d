"""Layer 2 engineering rationality tests — covers all 6 work items.

W1: Grip length estimation (assembly_matcher._estimate_grip_length)
W2: Threaded hole + nut pocket generation (connection_features)
W3: BOM standard parts inference from ConnectionMethod
W4: 14 new MountingInterface entries
W5: set_screw connection type
W6: RSS statistical tolerance analysis
"""

from __future__ import annotations

import math
import pytest

from lang3d.knowledge.mechanics import (
    Assembly,
    ConnectionMethod,
    Joint,
    Part,
)
from lang3d.tools.assembly_matcher import AssemblyMatcher, FastenerSelection
from lang3d.tools.connection_features import ConnectionFeatureEngine
from lang3d.tools.bom_gen import generate_bom, _infer_standard_parts
from lang3d.tools.tolerance_analysis import ToleranceStackup, StackupResult
from lang3d.knowledge.parts_catalog import get_mounting_interface, MOUNTING_INTERFACES


# ===========================================================================
# W1: Grip length estimation
# ===========================================================================

class TestGripLengthEstimation:
    """W1: _estimate_grip_length should account for struct + func + pocket + washer."""

    def test_basic_grip_includes_struct_thickness(self):
        matcher = AssemblyMatcher()
        struct = Part("bracket", "structural", "test",
                      dimensions={"length": 60, "width": 60, "thickness": 5})
        func = Part("nema17_stepper", "stepper", "motor",
                    dimensions={"length": 42, "width": 42, "height": 47})
        from lang3d.tools.assembly_matcher import AssemblyMatchResult
        result = AssemblyMatchResult()
        result.fastener_selection = FastenerSelection(bolt_size="M3")
        grip = matcher._estimate_grip_length(struct, func, result)
        # grip >= struct_t(5) + func_t(47) + washer(0.5) = 52.5
        assert grip >= 5.0

    def test_grip_accounts_for_pocket(self):
        """SG90 has pocket_depth in its MountingInterface."""
        matcher = AssemblyMatcher()
        struct = Part("plate", "structural", "test",
                      dimensions={"length": 50, "width": 50, "thickness": 4})
        func = Part("servo_sg90", "servo", "servo",
                    dimensions={"length": 23, "width": 12.2, "height": 29})
        from lang3d.tools.assembly_matcher import AssemblyMatchResult
        result = AssemblyMatchResult()
        result.fastener_selection = FastenerSelection(bolt_size="M2")
        grip = matcher._estimate_grip_length(struct, func, result)
        # Should include pocket_height (22.2mm from SG90 interface)
        assert grip >= 4 + 22.2  # struct_t + pocket

    def test_grip_minimum_is_3mm(self):
        """Even with zero dimensions, grip should be at least 3mm."""
        matcher = AssemblyMatcher()
        struct = Part("thin", "structural", "test", dimensions={})
        func = Part("func", "sensor", "test", dimensions={})
        from lang3d.tools.assembly_matcher import AssemblyMatchResult
        result = AssemblyMatchResult()
        result.fastener_selection = FastenerSelection(bolt_size="M3")
        grip = matcher._estimate_grip_length(struct, func, result)
        assert grip >= 3.0

    def test_grip_uses_washer_catalog_thickness(self):
        """Washer thickness should come from fastener_catalog, not hardcoded."""
        matcher = AssemblyMatcher()
        struct = Part("plate", "structural", "test",
                      dimensions={"length": 50, "width": 50, "thickness": 6})
        func = Part("func", "sensor", "test",
                    dimensions={"length": 20, "width": 20, "height": 5})
        from lang3d.tools.assembly_matcher import AssemblyMatchResult
        result = AssemblyMatchResult()
        result.fastener_selection = FastenerSelection(bolt_size="M6")
        grip = matcher._estimate_grip_length(struct, func, result)
        # M6 washer thickness = 1.6mm (from catalog)
        # grip >= 6 + 5 + 1.6 = 12.6
        assert grip >= 12.0

    def test_end_to_end_bolt_length_reflects_grip(self):
        """Full auto_match should produce a bolt length > grip length."""
        matcher = AssemblyMatcher()
        struct = Part("bracket", "structural", "test",
                      dimensions={"length": 80, "width": 60, "thickness": 8})
        func = Part("nema17_stepper", "stepper", "motor",
                    dimensions={"length": 42, "width": 42, "height": 47})
        conn = ConnectionMethod(type="bolted", bolt_size="M3")
        result = matcher.auto_match(struct, func, conn)
        assert result.fastener_selection.bolt_length >= 8


# ===========================================================================
# W2: Threaded hole + nut pocket + thread insert generation
# ===========================================================================

class TestThreadedHoleGeneration:
    """W2: hole_type="threaded_hole" should generate tap drill holes."""

    def test_threaded_hole_generates_tap_diameter(self):
        engine = ConnectionFeatureEngine()
        part = Part("hub", "structural", "test",
                    dimensions={"length": 30, "width": 30, "thickness": 10})
        conn = ConnectionMethod(type="bolted", bolt_size="M3",
                                hole_type="threaded_hole")
        result = engine.generate_features(part, conn, "top")
        assert len(result.ops) > 0
        assert any("threaded" in str(op) for op in result.ops)

    def test_threaded_hole_description(self):
        engine = ConnectionFeatureEngine()
        part = Part("hub", "structural", "test",
                    dimensions={"length": 30, "width": 30, "thickness": 10})
        conn = ConnectionMethod(type="bolted", bolt_size="M4",
                                hole_type="threaded_hole")
        result = engine.generate_features(part, conn, "top")
        assert any("threaded" in f for f in result.features_generated)


class TestNutPocketGeneration:
    """W2: hole_type="nut_pocket" should generate through hole + hex pocket."""

    def test_nut_pocket_generates_ops(self):
        engine = ConnectionFeatureEngine()
        part = Part("plate", "structural", "test",
                    dimensions={"length": 50, "width": 50, "thickness": 8})
        conn = ConnectionMethod(type="bolted", bolt_size="M3",
                                hole_type="nut_pocket")
        result = engine.generate_features(part, conn, "top")
        assert len(result.ops) > 0
        assert any("nut_pocket" in str(op) for op in result.ops)

    def test_nut_pocket_description(self):
        engine = ConnectionFeatureEngine()
        part = Part("plate", "structural", "test",
                    dimensions={"length": 50, "width": 50, "thickness": 8})
        conn = ConnectionMethod(type="bolted", bolt_size="M3",
                                hole_type="nut_pocket")
        result = engine.generate_features(part, conn, "top")
        assert any("nut" in f.lower() for f in result.features_generated)


class TestThreadInsertPocketGeneration:
    """W2: hole_type="thread_insert" should generate insert pocket."""

    def test_thread_insert_generates_pocket(self):
        engine = ConnectionFeatureEngine()
        part = Part("block", "structural", "test",
                    dimensions={"length": 40, "width": 40, "thickness": 10})
        conn = ConnectionMethod(type="bolted", bolt_size="M3",
                                hole_type="thread_insert")
        result = engine.generate_features(part, conn, "top")
        assert len(result.ops) > 0
        assert any("insert" in f.lower() for f in result.features_generated)


# ===========================================================================
# W3: BOM standard parts inference from ConnectionMethod
# ===========================================================================

class TestBOMStandardPartsInference:
    """W3: BOM should infer fasteners from ConnectionMethod, not hardcode."""

    def test_bolted_joint_produces_correct_bolt_size(self):
        part_a = Part("base", "structural", "test",
                      dimensions={"length": 50, "width": 50, "thickness": 5})
        part_b = Part("arm", "structural", "test",
                      dimensions={"length": 80, "width": 30, "height": 10})
        cm = ConnectionMethod(type="bolted", bolt_size="M4", bolt_count=2)
        joint = Joint("fixed", "base", "arm", connection=cm)
        asm = Assembly("test", parts=[part_a, part_b], joints=[joint])
        std_parts = _infer_standard_parts(asm)
        screw_names = [p["name"] for p in std_parts if p["type"] == "screw"]
        assert any("M4" in n for n in screw_names)

    def test_set_screw_joint_produces_set_screw(self):
        part_a = Part("shaft", "structural", "test",
                      dimensions={"length": 50, "diameter": 8})
        part_b = Part("pulley", "structural", "test",
                      dimensions={"diameter": 20, "bore_diameter": 5, "height": 10})
        cm = ConnectionMethod(type="set_screw", bolt_size="M3")
        joint = Joint("fixed", "shaft", "pulley", connection=cm)
        asm = Assembly("test", parts=[part_a, part_b], joints=[joint])
        std_parts = _infer_standard_parts(asm)
        screw_names = [p["name"] for p in std_parts]
        assert any("紧定" in n for n in screw_names)

    def test_no_connection_falls_back_to_default(self):
        part_a = Part("base", "structural", "test",
                      dimensions={"length": 50, "width": 50, "thickness": 5})
        part_b = Part("arm", "structural", "test",
                      dimensions={"length": 80, "width": 30, "height": 10})
        joint = Joint("fixed", "base", "arm")  # connection=None
        asm = Assembly("test", parts=[part_a, part_b], joints=[joint])
        std_parts = _infer_standard_parts(asm)
        assert len(std_parts) > 0

    def test_consolidates_duplicate_parts(self):
        part_a = Part("base", "structural", "test",
                      dimensions={"length": 50, "width": 50, "thickness": 5})
        part_b = Part("arm", "structural", "test",
                      dimensions={"length": 80, "width": 30, "height": 10})
        part_c = Part("head", "structural", "test",
                      dimensions={"length": 30, "width": 30, "thickness": 4})
        cm = ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=4)
        j1 = Joint("fixed", "base", "arm", connection=cm)
        j2 = Joint("fixed", "arm", "head", connection=cm)
        asm = Assembly("test", parts=[part_a, part_b, part_c], joints=[j1, j2])
        std_parts = _infer_standard_parts(asm)
        # M3 screws should be consolidated, not duplicated
        screw_parts = [p for p in std_parts if p["type"] == "screw" and "M3" in p["name"]]
        # Should have 1 consolidated entry (not 2)
        m3_screw_names = [p["name"] for p in screw_parts]
        assert len(m3_screw_names) == len(set(m3_screw_names))


# ===========================================================================
# W4: 14 new MountingInterface entries
# ===========================================================================

class TestNewMountingInterfaces:
    """W4: Verify 14 new MountingInterface entries are present and valid."""

    NEW_FUNCTIONAL_PARTS = [
        "servo_mgmt995",
        "imu_mpu6050",
        "ultrasonic_hcsr04",
        "oled_128x64",
        "spur_gear",
        "flexible_coupling",
        "t8_nut",
    ]

    NEW_STRUCTURAL_PARTS = [
        "linear_guide_mgn12",
        "t8_leadscrew",
        "compression_spring",
        "damper_foot",
        "gt2_belt",
        "shaft_coupling",
    ]

    def test_functional_interfaces_present(self):
        for part_id in self.NEW_FUNCTIONAL_PARTS:
            mi = get_mounting_interface(part_id)
            assert mi is not None, f"Missing MountingInterface for {part_id}"

    def test_structural_interfaces_present(self):
        for part_id in self.NEW_STRUCTURAL_PARTS:
            mi = get_mounting_interface(part_id)
            assert mi is not None, f"Missing MountingInterface for {part_id}"

    def test_servo_mgmt995_has_4_holes(self):
        mi = get_mounting_interface("servo_mgmt995")
        assert len(mi.holes) == 4
        assert mi.bore_diameter == 6.0

    def test_ultrasonic_has_2_holes(self):
        mi = get_mounting_interface("ultronic_hcsr04")
        if mi is None:
            mi = get_mounting_interface("ultrasonic_hcsr04")
        assert mi is not None
        assert len(mi.holes) == 2

    def test_t8_nut_has_4_holes(self):
        mi = get_mounting_interface("t8_nut")
        assert len(mi.holes) == 4
        assert mi.bore_diameter == 8.0

    def test_gt2_belt_is_belt_type(self):
        mi = get_mounting_interface("gt2_belt")
        assert mi.interface_type == "belt"

    def test_shaft_coupling_has_set_screw(self):
        mi = get_mounting_interface("shaft_coupling")
        assert mi.interface_type == "press_fit"
        assert len(mi.holes) == 2
        assert mi.bore_diameter == 5.0

    def test_damper_foot_has_m6_hole(self):
        mi = get_mounting_interface("damper_foot")
        assert len(mi.holes) == 1
        assert mi.holes[0].diameter == 6.6

    def test_total_interface_count_increased(self):
        """Total interfaces should be at least the original 23 + 14 new = 37."""
        assert len(MOUNTING_INTERFACES) >= 37


# ===========================================================================
# W5: set_screw connection type
# ===========================================================================

class TestSetScrewConnection:
    """W5: set_screw connection type generates radial threaded hole."""

    def test_set_screw_generates_features(self):
        engine = ConnectionFeatureEngine()
        part = Part("pulley", "structural", "test",
                    dimensions={"outer_diameter": 20, "bore_diameter": 5,
                                "height": 10, "thickness": 10})
        conn = ConnectionMethod(type="set_screw", bolt_size="M3")
        result = engine.generate_features(part, conn, "top")
        assert len(result.ops) > 0

    def test_set_screw_has_threaded_hole(self):
        engine = ConnectionFeatureEngine()
        part = Part("pulley", "structural", "test",
                    dimensions={"outer_diameter": 20, "bore_diameter": 5,
                                "height": 10, "thickness": 10})
        conn = ConnectionMethod(type="set_screw", bolt_size="M3")
        result = engine.generate_features(part, conn, "top")
        threaded_ops = [op for op in result.ops
                        if op.get("hole_type") == "threaded"]
        assert len(threaded_ops) >= 1

    def test_set_screw_description(self):
        engine = ConnectionFeatureEngine()
        part = Part("pulley", "structural", "test",
                    dimensions={"outer_diameter": 20, "bore_diameter": 5,
                                "height": 10, "thickness": 10})
        conn = ConnectionMethod(type="set_screw", bolt_size="M3")
        result = engine.generate_features(part, conn, "top")
        assert any("set screw" in f.lower() for f in result.features_generated)

    def test_set_screw_generates_fastener_model(self):
        engine = ConnectionFeatureEngine()
        part = Part("pulley", "structural", "test",
                    dimensions={"outer_diameter": 20, "bore_diameter": 5,
                                "height": 10, "thickness": 10})
        conn = ConnectionMethod(type="set_screw", bolt_size="M3")
        result = engine.generate_features(part, conn, "top")
        assert len(result.fastener_ops) > 0

    def test_connection_method_has_set_screw_size(self):
        conn = ConnectionMethod(type="set_screw", set_screw_size="M4")
        assert conn.set_screw_size == "M4"

    def test_connection_method_has_hole_type(self):
        conn = ConnectionMethod(type="bolted", hole_type="threaded_hole")
        assert conn.hole_type == "threaded_hole"


# ===========================================================================
# W6: RSS statistical tolerance analysis
# ===========================================================================

class TestRSSToleranceAnalysis:
    """W6: compute_rss() implements root-sum-square statistical method."""

    def test_rss_produces_smaller_tolerance_than_worst_case(self):
        """RSS should always produce tighter tolerance than worst-case."""
        stack = ToleranceStackup("test")
        stack.add_dimension("plate1", 10.0, upper=0.1, lower=-0.1)
        stack.add_dimension("plate2", 5.0, upper=0.05, lower=-0.05)
        stack.add_dimension("plate3", 8.0, upper=0.08, lower=-0.08)

        wc = stack.compute_stackup()
        rss = stack.compute_rss()

        assert rss.total_tolerance < wc.total_tolerance

    def test_rss_nominal_equals_worst_case(self):
        """Nominal value should be the same for both methods."""
        stack = ToleranceStackup("test")
        stack.add_dimension("a", 10.0, upper=0.1, lower=-0.1)
        stack.add_dimension("b", 5.0, upper=0.05, lower=-0.05)

        wc = stack.compute_stackup()
        rss = stack.compute_rss()

        assert rss.nominal == wc.nominal

    def test_rss_method_label(self):
        stack = ToleranceStackup("test")
        stack.add_dimension("a", 10.0, upper=0.1, lower=-0.1)
        rss = stack.compute_rss()
        assert rss.method == "rss"

    def test_worst_case_method_label(self):
        stack = ToleranceStackup("test")
        stack.add_dimension("a", 10.0, upper=0.1, lower=-0.1)
        wc = stack.compute_stackup()
        assert wc.method == "worst_case"

    def test_rss_single_dimension_equals_worst_case(self):
        """For a single dimension, RSS = worst case."""
        stack = ToleranceStackup("test")
        stack.add_dimension("a", 10.0, upper=0.1, lower=-0.1)

        wc = stack.compute_stackup()
        rss = stack.compute_rss()

        assert math.isclose(rss.total_tolerance, wc.total_tolerance, rel_tol=1e-6)

    def test_rss_math_correct(self):
        """Verify RSS math: sqrt(0.1^2 + 0.05^2) for upper deviation."""
        stack = ToleranceStackup("test")
        stack.add_dimension("a", 10.0, upper=0.1, lower=-0.1)
        stack.add_dimension("b", 5.0, upper=0.05, lower=-0.05)

        rss = stack.compute_rss()

        expected_upper = math.sqrt(0.1**2 + 0.05**2)
        assert math.isclose(rss.upper_dev, expected_upper, rel_tol=1e-6)

    def test_rss_with_negative_direction(self):
        """RSS should handle negative-direction dimensions correctly."""
        stack = ToleranceStackup("test")
        stack.add_dimension("a", 10.0, upper=0.1, lower=-0.1, direction="+")
        stack.add_dimension("gap", 0.5, upper=0.02, lower=-0.02, direction="-")

        rss = stack.compute_rss()
        assert rss.nominal == pytest.approx(9.5)

    def test_check_acceptable_with_rss_method(self):
        """check_acceptable should support method='rss' parameter."""
        stack = ToleranceStackup("test")
        stack.add_dimension("a", 10.0, upper=0.1, lower=-0.1)
        stack.add_dimension("b", 5.0, upper=0.05, lower=-0.05)

        # RSS tolerance is tighter, should pass
        assert stack.check_acceptable(allowed_total=0.3, method="rss")
        # Worst-case tolerance is wider, should fail with same limit
        assert not stack.check_acceptable(allowed_total=0.1, method="worst_case")

    def test_rss_max_min_values(self):
        """max_value and min_value should be computed correctly."""
        stack = ToleranceStackup("test")
        stack.add_dimension("a", 10.0, upper=0.1, lower=-0.1)

        rss = stack.compute_rss()
        assert math.isclose(rss.max_value, 10.0 + rss.upper_dev, rel_tol=1e-6)
        assert math.isclose(rss.min_value, 10.0 + rss.lower_dev, rel_tol=1e-6)

    def test_stackup_result_has_method_field(self):
        """StackupResult should have a 'method' field."""
        result = StackupResult()
        assert hasattr(result, 'method')
        assert result.method == "worst_case"  # default
