"""Tests for the fastener catalog and model generation (Task 72).

Tests cover:
1. Fastener dimension data queries (bolt/nut/washer/insert specs)
2. Clearance hole / tap hole / torque recommendations
3. Bolt length recommendation
4. Fastener set recommendation
5. FreeCAD model generation for each fastener type
6. Agent tool execution
7. Integration with connection_features (delegation works)
"""

from __future__ import annotations

import pytest

from lang3d.knowledge.fastener_catalog import (
    BOLT_SPECS,
    NUT_SPECS,
    WASHER_SPECS,
    SPRING_WASHER_SPECS,
    THREAD_INSERT_SPECS,
    SET_SCREW_SPECS,
    CLEARANCE_HOLE_SPECS,
    TAP_HOLE_SPECS,
    STANDARD_BOLT_LENGTHS,
    TORQUE_PLA,
    TORQUE_STEEL,
    BoltSpec,
    NutSpec,
    WasherSpec,
    ThreadInsertSpec,
    DowelPinSpec,
    DOWEL_PIN_SPECS,
    SPRING_PIN_SPECS,
    get_bolt_spec,
    get_nut_spec,
    get_washer_spec,
    get_spring_washer_spec,
    get_thread_insert_spec,
    get_set_screw_spec,
    get_clearance_hole,
    get_tap_hole,
    get_torque,
    recommend_bolt_length,
    recommend_fastener_set,
    list_available_sizes,
    validate_size,
    get_dowel_pin_spec,
    get_spring_pin_spec,
    recommend_dowel_pin,
)
from lang3d.tools.fastener_model import (
    generate_bolt_ops,
    generate_nut_ops,
    generate_washer_ops,
    generate_spring_washer_ops,
    generate_thread_insert_ops,
    generate_set_screw_ops,
    FastenerModelTool,
    FastenerQueryTool,
)


# ============================================================================
# 1. Bolt spec data
# ============================================================================


class TestBoltSpecs:
    def test_m3_dimensions(self):
        spec = get_bolt_spec("M3")
        assert spec is not None
        assert spec.thread_diameter == 3.0
        assert spec.pitch == 0.5
        assert spec.head_diameter == 5.5
        assert spec.head_height == 3.0
        assert spec.socket_width == 2.5

    def test_m6_dimensions(self):
        spec = get_bolt_spec("M6")
        assert spec.thread_diameter == 6.0
        assert spec.pitch == 1.0
        assert spec.head_diameter == 10.0

    def test_all_sizes_present(self):
        expected = ["M2", "M2.5", "M3", "M4", "M5", "M6", "M8", "M10", "M12"]
        for size in expected:
            assert get_bolt_spec(size) is not None, f"Missing {size}"

    def test_unknown_size_returns_none(self):
        assert get_bolt_spec("M99") is None

    def test_computed_properties(self):
        spec = get_bolt_spec("M4")
        assert spec.thread_radius == 2.0
        assert spec.head_radius == 3.5


class TestNutSpecs:
    def test_m3_dimensions(self):
        spec = get_nut_spec("M3")
        assert spec.width_across_flats == 5.5
        assert spec.height == 2.4
        assert spec.thread_diameter == 3.0

    def test_outer_radius(self):
        spec = get_nut_spec("M4")
        assert spec.outer_radius == pytest.approx(spec.width_across_corners / 2)

    def test_all_sizes(self):
        for size in ["M2", "M3", "M4", "M5", "M6", "M8", "M10", "M12"]:
            assert get_nut_spec(size) is not None


class TestWasherSpecs:
    def test_m3_dimensions(self):
        spec = get_washer_spec("M3")
        assert spec.inner_diameter == 3.2
        assert spec.outer_diameter == 7.0
        assert spec.thickness == 0.5

    def test_radii(self):
        spec = get_washer_spec("M6")
        assert spec.outer_radius == 6.0
        assert spec.inner_radius == pytest.approx(3.2)


class TestThreadInsertSpecs:
    def test_m3_dimensions(self):
        spec = get_thread_insert_spec("M3")
        assert spec.outer_diameter == 4.6
        assert spec.length == 5.6
        assert spec.install_hole_diameter == 4.7
        assert spec.min_wall_thickness == 2.0

    def test_available_sizes(self):
        for size in ["M2", "M2.5", "M3", "M4", "M5", "M6"]:
            assert get_thread_insert_spec(size) is not None


class TestSetScrewSpecs:
    def test_m3(self):
        spec = get_set_screw_spec("M3")
        assert spec.thread_diameter == 3.0
        assert spec.socket_width == 2.5


class TestSpringWasherSpecs:
    def test_m3(self):
        spec = get_spring_washer_spec("M3")
        assert spec is not None
        assert spec.inner_diameter < spec.outer_diameter


# ============================================================================
# 2. Query functions
# ============================================================================


class TestClearanceHoles:
    def test_normal_fit(self):
        assert get_clearance_hole("M3") == 3.4
        assert get_clearance_hole("M6") == 6.6

    def test_close_fit(self):
        assert get_clearance_hole("M3", "close") == 3.2

    def test_loose_fit(self):
        assert get_clearance_hole("M3", "loose") == 3.9

    def test_unknown_fallback(self):
        result = get_clearance_hole("M99")
        assert result == pytest.approx(99.4)


class TestTapHoles:
    def test_m3(self):
        assert get_tap_hole("M3") == 2.5

    def test_m6(self):
        assert get_tap_hole("M6") == 5.0

    def test_unknown(self):
        assert get_tap_hole("M99") == 0.0


class TestTorque:
    def test_pla_m3(self):
        assert get_torque("M3", "PLA") == 0.3

    def test_steel_m3(self):
        assert get_torque("M3", "Steel") == 1.0

    def test_aluminum_m4(self):
        assert get_torque("M4", "Aluminum") == 2.5


class TestBoltLengthRecommendation:
    def test_exact_match(self):
        assert recommend_bolt_length(10.0) == 10.0

    def test_round_up(self):
        assert recommend_bolt_length(11.0) == 12.0

    def test_small(self):
        assert recommend_bolt_length(3.0) == 4.0

    def test_large(self):
        result = recommend_bolt_length(110.0)
        assert result >= 110.0


class TestFastenerSetRecommendation:
    def test_m3_set(self):
        result = recommend_fastener_set("M3", 8.0)
        assert result is not None
        assert "bolt_length_mm" in result
        assert "clearance_hole_mm" in result
        assert result["clearance_hole_mm"] == 3.4
        assert result["parts_count"] >= 3  # bolt + nut + washer

    def test_with_thread_insert(self):
        result = recommend_fastener_set("M3", 8.0, use_thread_insert=True)
        assert "thread_insert" in result

    def test_unknown_size(self):
        assert recommend_fastener_set("M99", 8.0) is None

    def test_without_washer(self):
        result = recommend_fastener_set("M3", 8.0, with_washer=False)
        assert "washer" not in result


class TestUtilityFunctions:
    def test_list_available_sizes(self):
        sizes = list_available_sizes()
        assert "M1.6" in sizes
        assert "M3" in sizes
        assert "M6" in sizes
        assert len(sizes) == 10

    def test_validate_size(self):
        assert validate_size("M3") is True
        assert validate_size("M99") is False


# ============================================================================
# 3. Model generation
# ============================================================================


class TestBoltModelGeneration:
    def test_socket_head_simplified(self):
        ops = generate_bolt_ops("M3", 12.0, style="socket_head", quality="simplified")
        assert len(ops) >= 4  # head + shank + move + fuse
        types = [op["type"] for op in ops]
        assert "make_cylinder" in types
        assert "boolean" in types

    def test_socket_head_realistic(self):
        ops = generate_bolt_ops("M3", 12.0, quality="realistic")
        # Should have additional hex socket cut
        assert len(ops) >= 7

    def test_hex_head(self):
        ops = generate_bolt_ops("M4", 16.0, style="hex_head")
        assert len(ops) >= 4

    def test_countersunk(self):
        ops = generate_bolt_ops("M3", 12.0, style="countersunk")
        types = [op["type"] for op in ops]
        assert "make_cone" in types

    def test_unknown_size(self):
        ops = generate_bolt_ops("M99", 12.0)
        assert ops == []


class TestNutModelGeneration:
    def test_simplified(self):
        ops = generate_nut_ops("M3", quality="simplified")
        assert len(ops) == 1
        assert ops[0]["type"] == "cylinder_with_hole"

    def test_realistic(self):
        ops = generate_nut_ops("M3", quality="realistic")
        assert len(ops) == 1
        assert ops[0]["type"] == "raw_script"

    def test_unknown_size(self):
        assert generate_nut_ops("M99") == []


class TestWasherModelGeneration:
    def test_flat_washer(self):
        ops = generate_washer_ops("M3")
        assert len(ops) == 1
        assert ops[0]["type"] == "cylinder_with_hole"
        assert ops[0]["outer_radius"] == 3.5  # 7.0/2

    def test_unknown_size(self):
        assert generate_washer_ops("M99") == []


class TestSpringWasherGeneration:
    def test_m3(self):
        ops = generate_spring_washer_ops("M3")
        assert len(ops) == 1
        assert ops[0]["type"] == "raw_script"
        assert "gap" in ops[0]["script"]


class TestThreadInsertGeneration:
    def test_m3(self):
        ops = generate_thread_insert_ops("M3")
        assert len(ops) == 1
        assert ops[0]["type"] == "raw_script"
        assert "knurl" in ops[0]["script"].lower()

    def test_unknown(self):
        assert generate_thread_insert_ops("M99") == []


class TestSetScrewGeneration:
    def test_m3(self):
        ops = generate_set_screw_ops("M3", 6.0)
        assert len(ops) >= 3  # cylinder + socket + boolean cut


# ============================================================================
# 4. Agent tool execution
# ============================================================================


class TestFastenerModelTool:
    @pytest.fixture
    def tool(self):
        return FastenerModelTool()

    def test_tool_name(self, tool):
        assert tool.name == "fastener_model"

    def test_tool_definition(self, tool):
        defn = tool.get_definition()
        assert defn.name == "fastener_model"
        assert "fastener_type" in defn.parameters
        assert "size" in defn.parameters

    def test_execute_bolt(self, tool):
        result = tool.execute(fastener_type="bolt", size="M3", length=12)
        assert "M3" in result
        assert "12" in result

    def test_execute_nut(self, tool):
        result = tool.execute(fastener_type="nut", size="M4")
        assert "M4" in result

    def test_execute_washer(self, tool):
        result = tool.execute(fastener_type="washer", size="M3")
        assert "M3" in result

    def test_execute_spring_washer(self, tool):
        result = tool.execute(fastener_type="spring_washer", size="M3")
        assert "弹簧垫圈" in result

    def test_execute_thread_insert(self, tool):
        result = tool.execute(fastener_type="thread_insert", size="M3")
        assert "热嵌螺母" in result

    def test_execute_set_screw(self, tool):
        result = tool.execute(fastener_type="set_screw", size="M3", length=6)
        assert "紧定螺钉" in result

    def test_execute_fastener_set(self, tool):
        result = tool.execute(
            fastener_type="fastener_set", size="M3", grip_thickness=8.0,
        )
        assert "紧固件组" in result
        assert "间隙孔" in result

    def test_execute_fastener_set_with_insert(self, tool):
        result = tool.execute(
            fastener_type="fastener_set", size="M3",
            grip_thickness=8.0, with_thread_insert=True,
        )
        assert "热嵌螺母" in result

    def test_execute_unknown_type(self, tool):
        result = tool.execute(fastener_type="rivet", size="M3")
        assert "错误" in result

    def test_execute_unknown_size(self, tool):
        result = tool.execute(fastener_type="bolt", size="M99", length=10)
        assert "错误" in result


class TestFastenerQueryTool:
    @pytest.fixture
    def tool(self):
        return FastenerQueryTool()

    def test_tool_name(self, tool):
        assert tool.name == "fastener_query"

    def test_query_bolt_spec(self, tool):
        result = tool.execute(query_type="bolt_spec", size="M3")
        assert "5.5" in result  # head diameter
        assert "DIN 912" in result

    def test_query_nut_spec(self, tool):
        result = tool.execute(query_type="nut_spec", size="M3")
        assert "5.5" in result  # width across flats
        assert "DIN 934" in result

    def test_query_washer_spec(self, tool):
        result = tool.execute(query_type="washer_spec", size="M3")
        assert "7.0" in result  # outer diameter

    def test_query_thread_insert(self, tool):
        result = tool.execute(query_type="thread_insert_spec", size="M3")
        assert "4.6" in result  # outer diameter

    def test_query_clearance_hole(self, tool):
        result = tool.execute(query_type="clearance_hole", size="M3")
        assert "3.4" in result

    def test_query_tap_hole(self, tool):
        result = tool.execute(query_type="tap_hole", size="M3")
        assert "2.5" in result

    def test_query_torque(self, tool):
        result = tool.execute(query_type="torque", size="M3", material="PLA")
        assert "0.3" in result

    def test_query_bolt_length(self, tool):
        result = tool.execute(query_type="bolt_length", size="M3", grip_mm=8)
        assert "推荐" in result

    def test_query_fastener_set(self, tool):
        result = tool.execute(query_type="fastener_set", size="M3", grip_mm=8)
        assert "推荐" in result

    def test_query_available_sizes(self, tool):
        result = tool.execute(query_type="available_sizes")
        assert "M3" in result
        assert "M12" in result

    def test_query_unknown_type(self, tool):
        result = tool.execute(query_type="unknown_query")
        assert "未知" in result


# ============================================================================
# 5. Cross-module integration
# ============================================================================


class TestConnectionFeaturesIntegration:
    """Verify connection_features delegates to fastener_catalog correctly."""

    def test_clearance_hole_delegation(self):
        from lang3d.tools.connection_features import get_clearance_hole as cf_hole
        assert cf_hole("M3") == get_clearance_hole("M3") == 3.4

    def test_torque_delegation(self):
        from lang3d.tools.connection_features import get_torque_recommendation
        assert get_torque_recommendation("M3", "PLA") == get_torque("M3", "PLA")

    def test_bolt_head_dims_delegation(self):
        from lang3d.tools.connection_features import get_bolt_head_dims
        hd, hh = get_bolt_head_dims("M4")
        spec = get_bolt_spec("M4")
        assert hd == spec.head_diameter
        assert hh == spec.head_height

    def test_nut_dims_delegation(self):
        from lang3d.tools.connection_features import get_nut_dims
        w, h = get_nut_dims("M4")
        spec = get_nut_spec("M4")
        assert w == spec.width_across_flats
        assert h == spec.height

    def test_washer_dims_delegation(self):
        from lang3d.tools.connection_features import get_washer_dims
        id_, od, t = get_washer_dims("M4")
        spec = get_washer_spec("M4")
        assert id_ == spec.inner_diameter
        assert od == spec.outer_diameter
        assert t == spec.thickness

    def test_thread_insert_dims_delegation(self):
        from lang3d.tools.connection_features import get_thread_insert_dims
        od, length, install = get_thread_insert_dims("M3")
        spec = get_thread_insert_spec("M3")
        assert od == spec.outer_diameter
        assert length == spec.length
        assert install == spec.install_hole_diameter


class TestRegistration:
    """Verify tools can be registered without errors."""

    def test_register_fastener_tools(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.fastener_model import register_fastener_tools
        registry = ToolRegistry()
        register_fastener_tools(registry)
        defs = registry.get_all_definitions()
        names = [d.name for d in defs]
        assert "fastener_model" in names
        assert "fastener_query" in names


# ============================================================================
# 6. Dowel pin specifications (Phase 2)
# ============================================================================


class TestDowelPinSpecs:
    """Test dowel pin data and query functions."""

    def test_dowel_pin_specs_count(self):
        assert len(DOWEL_PIN_SPECS) == 15

    def test_spring_pin_specs_count(self):
        assert len(SPRING_PIN_SPECS) == 5

    def test_d5x20_dimensions(self):
        spec = get_dowel_pin_spec("D5x20")
        assert spec is not None
        assert spec.diameter == 5.0
        assert spec.length == 20.0
        assert spec.tolerance == "m6"

    def test_d8x40_dimensions(self):
        spec = get_dowel_pin_spec("D8x40")
        assert spec is not None
        assert spec.diameter == 8.0
        assert spec.length == 40.0

    def test_slip_fit_hole(self):
        spec = get_dowel_pin_spec("D5x20")
        # H7 slip-fit: nominal + 0.01mm
        assert spec.hole_diameter_slip == pytest.approx(5.01)

    def test_press_fit_hole(self):
        spec = get_dowel_pin_spec("D5x20")
        # Press-fit: nominal - 0.01mm
        assert spec.hole_diameter_press == pytest.approx(4.99)

    def test_all_dowel_pins_have_valid_sizes(self):
        for size, spec in DOWEL_PIN_SPECS.items():
            assert spec.diameter > 0
            assert spec.length > 0
            assert spec.tolerance == "m6"

    def test_all_spring_pins_have_spring_tolerance(self):
        for size, spec in SPRING_PIN_SPECS.items():
            assert spec.tolerance == "spring"

    def test_unknown_dowel_pin_returns_none(self):
        assert get_dowel_pin_spec("D99x100") is None

    def test_unknown_spring_pin_returns_none(self):
        assert get_spring_pin_spec("SP99x100") is None

    def test_spring_pin_spec(self):
        spec = get_spring_pin_spec("SP5x20")
        assert spec is not None
        assert spec.diameter == 5.0
        assert spec.length == 20.0

    def test_recommend_dowel_pin_normal(self):
        pin = recommend_dowel_pin("normal", 10.0)
        assert pin is not None
        assert pin.length >= 10.0

    def test_recommend_dowel_pin_fine(self):
        pin = recommend_dowel_pin("fine", 25.0)
        assert pin is not None
        assert pin.length >= 25.0

    def test_recommend_dowel_pin_coarse(self):
        pin = recommend_dowel_pin("coarse", 15.0)
        assert pin is not None
        assert pin.length >= 15.0

    def test_recommend_dowel_pin_large_thickness(self):
        pin = recommend_dowel_pin("normal", 100.0)
        # May or may not find one that fits 100mm
        # Just verify it doesn't crash
        assert pin is None or pin.length >= 100.0
