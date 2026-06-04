"""Tests for 3D print optimization.

Tests cover:
- Print orientation optimization
- Tolerance compensation
- Print parameter recommendations
- Batch plate packing
- Full assembly optimization
- Tool registration and execution
"""

from __future__ import annotations

import math

import pytest

from lang3d.knowledge.mechanics import ROBOTIC_ARM_ASSEMBLY, Part, PRINT_TOLERANCES
from lang3d.tools.base import ToolRegistry
from lang3d.tools.print_optimize import (
    BUILD_VOLUMES,
    QUALITY_PRESETS,
    SHRINKAGE,
    PrintOptimizeTool,
    apply_tolerance_compensation,
    optimize_assembly_print,
    optimize_orientation,
    pack_parts_on_plate,
    recommend_print_params,
    register_print_optimize_tools,
)


# ============================================================================
# Test: Orientation optimization
# ============================================================================

class TestOrientationOptimization:
    """Test print orientation recommendations."""

    def test_cylindrical_part(self):
        part = Part("test_shaft", "joint", "圆柱轴", dimensions={"diameter": 20, "height": 50})
        result = optimize_orientation(part)
        assert result["part_name"] == "test_shaft"
        assert len(result["reasoning"]) > 0

    def test_box_part(self):
        part = Part("test_box", "structural", "盒子", dimensions={"length": 60, "width": 40, "height": 20})
        result = optimize_orientation(part)
        assert result["part_name"] == "test_box"
        assert len(result["reasoning"]) > 0

    def test_hollow_cylinder_needs_support(self):
        part = Part("test_housing", "joint", "关节外壳",
                     dimensions={"outer_diameter": 60, "height": 35, "wall_thickness": 4})
        result = optimize_orientation(part)
        assert result["support_needed"] is True

    def test_solid_box_no_support(self):
        part = Part("test_plate", "structural", "平板",
                     dimensions={"length": 100, "width": 50, "height": 5})
        result = optimize_orientation(part)
        assert result["support_needed"] is False

    def test_part_with_hole(self):
        part = Part("test_bearing", "joint", "轴承座",
                     dimensions={"outer_diameter": 40, "height": 15},
                     notes="需要轴承安装孔")
        result = optimize_orientation(part)
        assert any("孔" in r for r in result["reasoning"])

    def test_part_with_channel(self):
        part = Part("test_link", "structural", "连杆",
                     dimensions={"length": 150, "width": 40, "height": 30},
                     notes="内部走线通道")
        result = optimize_orientation(part)
        assert result["support_needed"] is True

    def test_all_robotic_arm_parts(self):
        for part in ROBOTIC_ARM_ASSEMBLY.parts:
            result = optimize_orientation(part)
            assert result["part_name"] == part.name
            assert len(result["reasoning"]) > 0


# ============================================================================
# Test: Tolerance compensation
# ============================================================================

class TestToleranceCompensation:
    """Test tolerance compensation calculations."""

    def test_pla_shrinkage(self):
        result = apply_tolerance_compensation({"length": 100}, material="PLA")
        # PLA shrinks 0.3%, so compensated = 100 * 1.003 - 0.3 = 100.0
        assert result["shrinkage_pct"] == 0.3

    def test_abs_shrinkage(self):
        result = apply_tolerance_compensation({"length": 100}, material="ABS")
        assert result["shrinkage_pct"] == 0.8

    def test_compensated_larger_for_shrinkage(self):
        dims = {"width": 50}
        result = apply_tolerance_compensation(dims, fit_type="tight_fit", material="ABS")
        # ABS shrinkage factor = 1.008
        adj = result["adjustments"]["width"]
        assert adj["shrinkage_comp"] > 50  # Scaled up

    def test_sliding_fit_has_clearance(self):
        result = apply_tolerance_compensation(
            {"outer_diameter": 30},
            fit_type="sliding_fit",
            material="PLA",
        )
        clearance = PRINT_TOLERANCES["sliding_fit"]
        assert result["clearance_mm"] == clearance

    def test_tight_fit_less_clearance(self):
        result_tight = apply_tolerance_compensation({"diameter": 30}, fit_type="tight_fit")
        result_loose = apply_tolerance_compensation({"diameter": 30}, fit_type="loose_fit")
        assert result_tight["clearance_mm"] < result_loose["clearance_mm"]

    def test_all_dimensions_compensated(self):
        dims = {"length": 100, "width": 50, "height": 30}
        result = apply_tolerance_compensation(dims)
        assert set(result["compensated"].keys()) == set(dims.keys())

    def test_adjustments_have_all_fields(self):
        result = apply_tolerance_compensation({"length": 100})
        adj = result["adjustments"]["length"]
        assert "original" in adj
        assert "shrinkage_comp" in adj
        assert "clearance" in adj
        assert "final" in adj


# ============================================================================
# Test: Print parameter recommendations
# ============================================================================

class TestPrintParams:
    """Test print parameter recommendations."""

    def test_standard_quality(self):
        part = Part("test", "structural", "测试", dimensions={"length": 50, "width": 30, "height": 20})
        params = recommend_print_params(part, "standard")
        assert params["layer_height"] == 0.2
        assert params["infill"] == 20
        assert params["wall_loops"] == 3

    def test_draft_quality(self):
        part = Part("test", "structural", "测试", dimensions={"length": 50, "width": 30, "height": 20})
        params = recommend_print_params(part, "draft")
        assert params["layer_height"] == 0.3
        assert params["infill"] == 10

    def test_high_quality(self):
        part = Part("test", "structural", "测试", dimensions={"length": 50, "width": 30, "height": 20})
        params = recommend_print_params(part, "high")
        assert params["layer_height"] == 0.12
        assert params["infill"] == 30

    def test_bearing_part_gets_higher_infill(self):
        part = Part("test_bearing", "joint", "测试",
                     dimensions={"outer_diameter": 40, "height": 15},
                     notes="轴承座零件")
        params = recommend_print_params(part, "standard")
        assert params["infill"] >= 40

    def test_lightweight_part_gets_lower_infill(self):
        part = Part("test_link", "structural", "测试",
                     dimensions={"length": 120, "width": 35, "height": 25},
                     notes="轻量化设计，内部中空")
        params = recommend_print_params(part, "standard")
        assert params["infill"] <= 15

    def test_has_estimated_time(self):
        part = Part("test", "structural", "测试", dimensions={"length": 50, "width": 30, "height": 20})
        params = recommend_print_params(part, "standard")
        assert "estimated_print_time_min" in params
        assert params["estimated_print_time_min"] >= 0

    def test_material_set(self):
        part = Part("test", "structural", "测试", dimensions={"length": 50}, material="PETG")
        params = recommend_print_params(part)
        assert params["material"] == "PETG"


# ============================================================================
# Test: Batch plate packing
# ============================================================================

class TestPlatePacking:
    """Test batch plate packing algorithm."""

    def test_single_part_fits(self):
        parts = [Part("small", "test", "小件", dimensions={"length": 30, "width": 30, "height": 10})]
        result = pack_parts_on_plate(parts, "ender3")
        assert result["plates_needed"] == 1
        assert result["part_count"] == 1

    def test_robotic_arm_parts(self):
        result = pack_parts_on_plate(ROBOTIC_ARM_ASSEMBLY.parts, "ender3")
        assert result["part_count"] == len(ROBOTIC_ARM_ASSEMBLY.parts)
        assert result["plates_needed"] >= 1
        assert len(result["positions"]) == len(ROBOTIC_ARM_ASSEMBLY.parts)

    def test_positions_have_coordinates(self):
        parts = [Part("a", "t", "a", dimensions={"length": 30, "width": 30, "height": 10}),
                 Part("b", "t", "b", dimensions={"length": 40, "width": 20, "height": 10})]
        result = pack_parts_on_plate(parts, "ender3")
        for pos in result["positions"]:
            assert "part" in pos
            assert "x" in pos
            assert "y" in pos
            assert "plate" in pos

    def test_utilization_positive(self):
        result = pack_parts_on_plate(ROBOTIC_ARM_ASSEMBLY.parts, "ender3")
        assert result["plate_utilization"] > 0

    def test_different_printers(self):
        for printer in ["ender3", "prusa_mk3", "bambu_x1c"]:
            result = pack_parts_on_plate(ROBOTIC_ARM_ASSEMBLY.parts, printer)
            assert result["printer"] == printer
            assert "plate_size" in result

    def test_parts_dont_overlap(self):
        parts = [
            Part(f"p{i}", "t", f"part{i}", dimensions={"length": 30, "width": 30, "height": 10})
            for i in range(4)
        ]
        result = pack_parts_on_plate(parts, "ender3")
        # Check no two parts have the same position
        positions = [(p["x"], p["y"]) for p in result["positions"]]
        assert len(positions) == len(set(positions))

    def test_empty_parts_list(self):
        result = pack_parts_on_plate([], "ender3")
        assert result["part_count"] == 0
        assert result["plates_needed"] == 1


# ============================================================================
# Test: Full assembly optimization
# ============================================================================

class TestFullAssemblyOptimization:
    """Test complete assembly print optimization."""

    @pytest.fixture
    def result(self):
        return optimize_assembly_print(ROBOTIC_ARM_ASSEMBLY)

    def test_has_all_parts(self, result):
        assert len(result["parts"]) == len(ROBOTIC_ARM_ASSEMBLY.parts)

    def test_each_part_has_sections(self, result):
        for p in result["parts"]:
            assert "orientation" in p
            assert "print_params" in p
            assert "tolerance" in p

    def test_has_packing(self, result):
        assert "packing" in result

    def test_has_summary(self, result):
        s = result["summary"]
        assert s["total_parts"] == len(ROBOTIC_ARM_ASSEMBLY.parts)
        assert s["total_print_time_min"] >= 0
        assert s["plates_needed"] >= 1
        assert 0 < s["plate_utilization_pct"] <= 100

    def test_different_qualities(self):
        r_draft = optimize_assembly_print(ROBOTIC_ARM_ASSEMBLY, quality="draft")
        r_high = optimize_assembly_print(ROBOTIC_ARM_ASSEMBLY, quality="high")
        # Draft should be faster (lower time estimate due to thicker layers)
        assert r_draft["summary"]["total_print_time_min"] <= r_high["summary"]["total_print_time_min"]


# ============================================================================
# Test: Tool registration and execution
# ============================================================================

class TestPrintOptimizeTool:
    """Test PrintOptimizeTool."""

    def test_registered(self):
        registry = ToolRegistry()
        register_print_optimize_tools(registry)
        assert "print_optimize" in registry.list_tools()

    def test_execute_basic(self):
        tool = PrintOptimizeTool()
        result = tool.execute()
        assert "Print Optimization" in result
        assert "3-DOF Robotic Arm" in result

    def test_execute_high_quality(self):
        tool = PrintOptimizeTool()
        result = tool.execute(quality="high")
        assert "0.12" in result

    def test_execute_custom_printer(self):
        tool = PrintOptimizeTool()
        result = tool.execute(printer="bambu_x1c")
        assert "bambu_x1c" in result

    def test_execute_unknown_assembly(self):
        tool = PrintOptimizeTool()
        result = tool.execute(assembly_name="nonexistent")
        assert "错误" in result

    def test_execute_json_included(self):
        tool = PrintOptimizeTool()
        result = tool.execute()
        assert "--- JSON ---" in result
        assert '"parts"' in result
