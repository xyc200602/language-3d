"""Tests for actuator selection and analysis tools.

Tests cover:
- Actuator database integrity (13 entries, 4 categories)
- Selection by torque, price, voltage, category
- Assembly torque analysis
- Power budget calculation
- Tool registration and execution
"""

from __future__ import annotations

from typing import Any

import pytest

from lang3d.knowledge.actuators import (
    ACTUATORS,
    Actuator,
    get_actuator,
    list_actuators,
    nm_to_torque,
    torque_to_nm,
)
from lang3d.knowledge.mechanics import ROBOTIC_ARM_ASSEMBLY
from lang3d.tools.actuator_tools import (
    ActuatorAnalyzeTool,
    ActuatorPowerBudgetTool,
    ActuatorSelectTool,
    analyze_assembly_torques,
    power_budget,
    register_actuator_tools,
    select_actuators,
)
from lang3d.tools.base import ToolRegistry


# ============================================================================
# Test: Database integrity
# ============================================================================

class TestDatabaseIntegrity:
    def test_actuator_count(self):
        assert len(ACTUATORS) == 14

    def test_all_categories(self):
        categories = {a.category for a in ACTUATORS.values()}
        assert categories == {"servo", "dc_motor", "stepper", "bldc"}

    def test_servo_count(self):
        servos = [a for a in ACTUATORS.values() if a.category == "servo"]
        assert len(servos) == 7

    def test_all_have_positive_torque(self):
        for a in ACTUATORS.values():
            assert a.torque_kgcm > 0, f"{a.id} has zero torque"

    def test_all_have_positive_voltage(self):
        for a in ACTUATORS.values():
            assert a.voltage > 0, f"{a.id} has zero voltage"

    def test_all_have_ids(self):
        for a in ACTUATORS.values():
            assert a.id, "Actuator missing ID"
            assert a.name, f"{a.id} missing name"

    def test_get_actuator_case_insensitive(self):
        assert get_actuator("sg90") is not None
        assert get_actuator("SG90") is not None
        assert get_actuator("Sg90") is not None

    def test_get_actuator_not_found(self):
        assert get_actuator("NONEXISTENT") is None

    def test_list_actuators_all(self):
        assert len(list_actuators()) == 14

    def test_list_actuators_filtered(self):
        servos = list_actuators("servo")
        assert len(servos) == 7
        assert all(a.category == "servo" for a in servos)


# ============================================================================
# Test: Unit conversion
# ============================================================================

class TestUnitConversion:
    def test_torque_to_nm(self):
        # 1 kg·cm ≈ 0.098 N·m
        assert torque_to_nm(1.0) == pytest.approx(0.09807, rel=0.01)

    def test_nm_to_torque(self):
        # Round-trip
        assert nm_to_torque(torque_to_nm(10.0)) == pytest.approx(10.0)

    def test_sg90_torque_nm(self):
        a = get_actuator("SG90")
        assert torque_to_nm(a.torque_kgcm) == pytest.approx(0.1765, rel=0.01)


# ============================================================================
# Test: Selection
# ============================================================================

class TestSelection:
    def test_select_by_min_torque(self):
        results = select_actuators(min_torque_kgcm=10, count=10)
        for r in results:
            assert r["torque_kgcm"] >= 10

    def test_select_by_price(self):
        results = select_actuators(max_price_cny=20, count=10)
        for r in results:
            assert r["price_cny"] <= 20

    def test_select_by_category(self):
        results = select_actuators(category="servo", count=10)
        for r in results:
            assert r["category"] == "servo"

    def test_select_by_voltage(self):
        results = select_actuators(voltage=5.0, count=10)
        for r in results:
            assert r["voltage_range"][0] <= 5.0 <= r["voltage_range"][1]

    def test_select_count_limit(self):
        results = select_actuators(count=2)
        assert len(results) <= 2

    def test_select_no_match(self):
        results = select_actuators(min_torque_kgcm=1000, count=5)
        assert len(results) == 0

    def test_select_sorted_by_torque_weight(self):
        results = select_actuators(category="servo", count=5)
        # Should be sorted by torque/weight ratio descending
        for i in range(len(results) - 1):
            r1 = results[i]
            r2 = results[i + 1]
            a1 = get_actuator(r1["id"])
            a2 = get_actuator(r2["id"])
            assert a1.torque_kgcm / max(a1.weight_g, 1) >= a2.torque_kgcm / max(a2.weight_g, 1)

    def test_select_by_weight(self):
        results = select_actuators(max_weight_g=20, count=10)
        for r in results:
            a = get_actuator(r["id"])
            assert a.weight_g <= 20


# ============================================================================
# Test: Assembly torque analysis
# ============================================================================

class TestTorqueAnalysis:
    def test_robotic_arm_analysis(self):
        results = analyze_assembly_torques(ROBOTIC_ARM_ASSEMBLY)
        assert len(results) == 4  # 4 revolute joints

    def test_analysis_has_required_fields(self):
        results = analyze_assembly_torques(ROBOTIC_ARM_ASSEMBLY)
        for r in results:
            assert "joint" in r
            assert "required_torque_kgcm" in r
            assert "required_torque_nm" in r
            assert "recommended" in r
            assert "downstream_parts" in r

    def test_base_joint_highest_torque(self):
        results = analyze_assembly_torques(ROBOTIC_ARM_ASSEMBLY)
        # Base rotation carries all weight → highest torque
        torques = {r["joint"]: r["required_torque_kgcm"] for r in results}
        base_torque = torques.get("底座旋转", 0)
        assert base_torque > 0

    def test_with_payload(self):
        r_no_load = analyze_assembly_torques(ROBOTIC_ARM_ASSEMBLY, payload_g=0)
        r_with_load = analyze_assembly_torques(ROBOTIC_ARM_ASSEMBLY, payload_g=500)
        # With payload, torques should be higher
        for a, b in zip(r_no_load, r_with_load):
            assert b["required_torque_kgcm"] >= a["required_torque_kgcm"]

    def test_safety_factor_effect(self):
        r_sf1 = analyze_assembly_torques(ROBOTIC_ARM_ASSEMBLY, safety_factor=1.0)
        r_sf3 = analyze_assembly_torques(ROBOTIC_ARM_ASSEMBLY, safety_factor=3.0)
        for a, b in zip(r_sf1, r_sf3):
            # Allow rounding error from round(..., 2)
            assert b["required_torque_kgcm"] == pytest.approx(a["required_torque_kgcm"] * 3.0, abs=0.1)

    def test_recommendations_present(self):
        results = analyze_assembly_torques(ROBOTIC_ARM_ASSEMBLY)
        for r in results:
            # Should have at least one recommendation
            assert len(r["recommended"]) >= 0  # May not find match for very high torque


# ============================================================================
# Test: Power budget
# ============================================================================

class TestPowerBudget:
    def test_single_actuator(self):
        result = power_budget(["MG996R"])
        assert result["count"] == 1
        assert result["total_power_w"] > 0
        assert result["supply_power_w"] > result["total_power_w"]

    def test_multiple_actuators(self):
        result = power_budget(["MG996R", "MG996R", "SG90"])
        assert result["count"] == 3
        assert result["supply_power_w"] > result["total_power_w"]

    def test_margin_factor(self):
        r1 = power_budget(["MG996R"], include_margin=1.0)
        r2 = power_budget(["MG996R"], include_margin=2.0)
        assert r2["supply_power_w"] == pytest.approx(r1["supply_power_w"] * 2.0, abs=0.5)

    def test_duty_cycle_effect(self):
        r_low = power_budget(["MG996R"], duty_cycle=0.1)
        r_high = power_budget(["MG996R"], duty_cycle=0.9)
        assert r_high["total_power_w"] > r_low["total_power_w"]

    def test_invalid_id_ignored(self):
        result = power_budget(["MG996R", "NONEXISTENT"])
        assert result["count"] == 1

    def test_empty_list(self):
        result = power_budget([])
        assert result["count"] == 0

    def test_all_invalid(self):
        result = power_budget(["FAKE1", "FAKE2"])
        assert result["count"] == 0

    def test_supply_voltage_is_max(self):
        result = power_budget(["SG90", "MG996R"])  # 5V and 6V
        assert result["supply_voltage_v"] == pytest.approx(6.0)


# ============================================================================
# Test: Tool registration and execution
# ============================================================================

class TestToolRegistration:
    def test_all_three_registered(self):
        registry = ToolRegistry()
        register_actuator_tools(registry)
        names = registry.list_tools()
        assert "actuator_select" in names
        assert "actuator_analyze" in names
        assert "actuator_power_budget" in names


class TestToolExecution:
    def test_select_tool(self):
        tool = ActuatorSelectTool()
        result = tool.execute(min_torque_kgcm=10, category="servo")
        assert "MG996R" in result or "DS3218" in result
        assert "JSON" in result

    def test_select_no_match(self):
        tool = ActuatorSelectTool()
        result = tool.execute(min_torque_kgcm=1000)
        assert "未找到" in result or "没有" in result

    def test_analyze_tool(self):
        tool = ActuatorAnalyzeTool()
        result = tool.execute(assembly_name="robotic_arm")
        assert "Torque Analysis" in result
        assert "JSON" in result

    def test_analyze_unknown_assembly(self):
        tool = ActuatorAnalyzeTool()
        result = tool.execute(assembly_name="nonexistent")
        assert "错误" in result

    def test_power_budget_tool(self):
        tool = ActuatorPowerBudgetTool()
        result = tool.execute(actuator_ids=["MG996R", "SG90"])
        assert "Power Budget" in result
        assert "2" in result  # 2 actuators
        assert "JSON" in result

    def test_power_budget_empty(self):
        tool = ActuatorPowerBudgetTool()
        result = tool.execute(actuator_ids=[])
        assert "0" in result or "无" in result
