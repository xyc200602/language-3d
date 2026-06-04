"""Tests for BOM (Bill of Materials) generation.

Tests cover:
- BOM structure and completeness
- Custom parts cost estimation
- Standard parts inference
- Electronics listing
- Cost summary
- Markdown formatting
- Tool registration and execution
"""

from __future__ import annotations

import pytest

from lang3d.knowledge.mechanics import ROBOTIC_ARM_ASSEMBLY
from lang3d.tools.base import ToolRegistry
from lang3d.tools.bom_gen import (
    GenBOMTool,
    format_bom_markdown,
    generate_bom,
    register_bom_tools,
)


# ============================================================================
# Test: BOM structure
# ============================================================================

class TestBOMStructure:
    """Test BOM output structure and completeness."""

    @pytest.fixture
    def bom(self):
        return generate_bom(
            ROBOTIC_ARM_ASSEMBLY,
            actuator_ids=["MG996R", "MG996R", "DS3218", "SG90"],
            sensor_ids=["AS5600", "LIMIT_SWITCH_MICRO"],
        )

    def test_has_required_sections(self, bom):
        assert "custom_parts" in bom
        assert "standard_parts" in bom
        assert "electronics" in bom
        assert "cost_summary" in bom
        assert "assembly_name" in bom

    def test_assembly_name(self, bom):
        assert bom["assembly_name"] == "3-DOF Robotic Arm"

    def test_custom_parts_count(self, bom):
        assert len(bom["custom_parts"]) == len(ROBOTIC_ARM_ASSEMBLY.parts)

    def test_custom_part_fields(self, bom):
        for part in bom["custom_parts"]:
            assert "name" in part
            assert "category" in part
            assert "material" in part
            assert "dimensions" in part
            assert "estimated_weight_g" in part
            assert "estimated_cost_cny" in part
            assert "print_params" in part
            assert "quantity" in part

    def test_standard_parts_not_empty(self, bom):
        assert len(bom["standard_parts"]) > 0

    def test_standard_part_fields(self, bom):
        for sp in bom["standard_parts"]:
            assert "name" in sp
            assert "spec" in sp
            assert "quantity" in sp
            assert sp["quantity"] > 0

    def test_electronics_not_empty(self, bom):
        assert len(bom["electronics"]) > 0

    def test_electronics_include_actuators(self, bom):
        actuator_entries = [e for e in bom["electronics"] if e["type"] == "actuator"]
        assert len(actuator_entries) == 4  # MG996R x2 + DS3218 + SG90

    def test_electronics_include_sensors(self, bom):
        sensor_entries = [e for e in bom["electronics"] if e["type"] == "sensor"]
        assert len(sensor_entries) == 2  # AS5600 + LIMIT_SWITCH_MICRO

    def test_electronics_include_controller(self, bom):
        ctrl = [e for e in bom["electronics"] if e["type"] == "controller"]
        assert len(ctrl) == 1
        assert ctrl[0]["id"] == "esp32"

    def test_electronics_include_accessories(self, bom):
        acc = [e for e in bom["electronics"] if e["type"] == "accessory"]
        assert len(acc) >= 1


# ============================================================================
# Test: Cost estimation
# ============================================================================

class TestCostEstimation:
    """Test BOM cost calculations."""

    @pytest.fixture
    def bom(self):
        return generate_bom(
            ROBOTIC_ARM_ASSEMBLY,
            actuator_ids=["MG996R", "MG996R", "DS3218", "SG90"],
        )

    def test_custom_parts_have_positive_cost(self, bom):
        for part in bom["custom_parts"]:
            assert part["estimated_cost_cny"] > 0

    def test_custom_parts_have_positive_weight(self, bom):
        for part in bom["custom_parts"]:
            assert part["estimated_weight_g"] > 0

    def test_total_cost_positive(self, bom):
        assert bom["cost_summary"]["total_cost_cny"] > 0

    def test_total_equals_sum_of_parts(self, bom):
        cs = bom["cost_summary"]
        expected = cs["custom_parts_cost_cny"] + cs["standard_parts_cost_cny"] + cs["electronics_cost_cny"]
        assert abs(cs["total_cost_cny"] - expected) < 1.0

    def test_electronics_cost_is_sum(self, bom):
        elec_total = sum(e["price_cny"] for e in bom["electronics"])
        assert abs(bom["cost_summary"]["electronics_cost_cny"] - elec_total) < 1.0

    def test_custom_cost_is_sum(self, bom):
        custom_total = sum(p["estimated_cost_cny"] for p in bom["custom_parts"])
        assert abs(bom["cost_summary"]["custom_parts_cost_cny"] - custom_total) < 1.0

    def test_counts_match(self, bom):
        cs = bom["cost_summary"]
        assert cs["num_custom_parts"] == len(bom["custom_parts"])
        assert cs["num_electronics"] == len(bom["electronics"])


# ============================================================================
# Test: Standard parts inference
# ============================================================================

class TestStandardPartsInference:
    """Test automatic standard parts inference."""

    @pytest.fixture
    def bom(self):
        return generate_bom(ROBOTIC_ARM_ASSEMBLY)

    def test_has_screws(self, bom):
        screws = [p for p in bom["standard_parts"] if p["type"] == "screw"]
        assert len(screws) > 0

    def test_has_nuts(self, bom):
        nuts = [p for p in bom["standard_parts"] if p["type"] == "nut"]
        assert len(nuts) > 0

    def test_has_bearings(self, bom):
        bearings = [p for p in bom["standard_parts"] if p["type"] == "bearing"]
        assert len(bearings) > 0

    def test_screw_quantity_proportional_to_joints(self, bom):
        screws = [p for p in bom["standard_parts"] if p["type"] == "screw"]
        joint_count = len(ROBOTIC_ARM_ASSEMBLY.joints)
        total_screws = sum(s["quantity"] for s in screws)
        assert total_screws >= joint_count * 4

    def test_bearing_quantity_proportional_to_revolute(self, bom):
        bearings = [p for p in bom["standard_parts"] if p["type"] == "bearing"]
        revolute_count = len([j for j in ROBOTIC_ARM_ASSEMBLY.joints if j.type == "revolute"])
        total_bearings = sum(b["quantity"] for b in bearings)
        assert total_bearings >= revolute_count * 2


# ============================================================================
# Test: Markdown formatting
# ============================================================================

class TestBOMMarkdown:
    """Test BOM Markdown output."""

    @pytest.fixture
    def bom_md(self):
        bom = generate_bom(
            ROBOTIC_ARM_ASSEMBLY,
            actuator_ids=["MG996R", "SG90"],
            sensor_ids=["AS5600"],
        )
        return format_bom_markdown(bom)

    def test_has_title(self, bom_md):
        assert "# BOM" in bom_md

    def test_has_sections(self, bom_md):
        assert "## 自定义零件" in bom_md
        assert "## 标准件" in bom_md
        assert "## 电子件" in bom_md
        assert "## 成本汇总" in bom_md

    def test_has_table_headers(self, bom_md):
        assert "| 名称 |" in bom_md
        assert "|---|" in bom_md

    def test_has_total_cost(self, bom_md):
        assert "**总计**" in bom_md

    def test_contains_part_names(self, bom_md):
        for part in ROBOTIC_ARM_ASSEMBLY.parts:
            assert part.name in bom_md


# ============================================================================
# Test: Controller variants
# ============================================================================

class TestBOMControllerVariants:
    """Test BOM with different controllers."""

    def test_esp32_controller(self):
        bom = generate_bom(ROBOTIC_ARM_ASSEMBLY, controller="esp32")
        ctrl = [e for e in bom["electronics"] if e["type"] == "controller"]
        assert len(ctrl) == 1
        assert "ESP32" in ctrl[0]["name"]

    def test_arduino_controller(self):
        bom = generate_bom(ROBOTIC_ARM_ASSEMBLY, controller="arduino")
        ctrl = [e for e in bom["electronics"] if e["type"] == "controller"]
        assert len(ctrl) == 1
        assert "Arduino" in ctrl[0]["name"]


# ============================================================================
# Test: Tool registration and execution
# ============================================================================

class TestBOMTool:
    """Test GenBOMTool registration and execution."""

    def test_registered(self):
        registry = ToolRegistry()
        register_bom_tools(registry)
        assert "gen_bom" in registry.list_tools()

    def test_execute_basic(self):
        tool = GenBOMTool()
        result = tool.execute()
        assert "BOM Generated" in result
        assert "3-DOF Robotic Arm" in result

    def test_execute_with_actuators(self):
        tool = GenBOMTool()
        result = tool.execute(
            actuator_ids=["MG996R", "SG90"],
            sensor_ids=["AS5600"],
        )
        assert "MG996R" in result or "Tower Pro" in result
        assert "AS5600" in result

    def test_execute_unknown_assembly(self):
        tool = GenBOMTool()
        result = tool.execute(assembly_name="nonexistent")
        assert "错误" in result

    def test_execute_json_included(self):
        tool = GenBOMTool()
        result = tool.execute()
        assert "--- JSON ---" in result
        assert '"custom_parts"' in result

    def test_execute_markdown_included(self):
        tool = GenBOMTool()
        result = tool.execute()
        assert "## 自定义零件" in result or "## 标准件" in result
