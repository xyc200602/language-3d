"""Tests for power budget calculation and battery selection (Task 51).

Covers:
  - PowerConsumer (effective power calculations)
  - BatterySpec (energy, discharge, density)
  - BATTERY_CATALOG integrity
  - PowerBudgetCalculator (peak/avg/runtime/recommend)
  - auto_populate_from_actuators
  - PowerBudgetTool
  - Registration
"""

import math

import pytest

from lang3d.tools.power_budget import (
    BATTERY_CATALOG,
    BatterySpec,
    PowerBudgetCalculator,
    PowerBudgetTool,
    PowerConsumer,
    auto_populate_from_actuators,
    get_battery,
    list_batteries,
    register_power_budget_tools,
)
from lang3d.tools.base import ToolRegistry


# ============================================================================
# PowerConsumer
# ============================================================================


class TestPowerConsumer:
    def test_effective_avg(self):
        c = PowerConsumer(name="Motor", category="motor",
                          peak_power_w=20.0, avg_power_w=10.0,
                          duty_cycle=0.5, quantity=2)
        assert c.effective_avg_w == pytest.approx(10.0)  # 10 * 0.5 * 2

    def test_effective_peak(self):
        c = PowerConsumer(name="Motor", category="motor",
                          peak_power_w=20.0, avg_power_w=10.0, quantity=3)
        assert c.effective_peak_w == 60.0

    def test_default_quantity(self):
        c = PowerConsumer(name="LED", category="other",
                          peak_power_w=0.5, avg_power_w=0.3)
        assert c.quantity == 1
        assert c.effective_avg_w == pytest.approx(0.3)

    def test_zero_duty_cycle(self):
        c = PowerConsumer(name="Standby", category="other",
                          peak_power_w=5.0, avg_power_w=3.0, duty_cycle=0.0)
        assert c.effective_avg_w == 0.0


# ============================================================================
# BatterySpec
# ============================================================================


class TestBatterySpec:
    def test_energy_wh(self):
        b = BatterySpec(id="T", name="Test", chemistry="lipo",
                        voltage=11.1, capacity_ah=2.2,
                        c_rate=30.0, weight_g=175, price_cny=90)
        assert b.energy_wh == pytest.approx(24.42, rel=0.01)

    def test_max_discharge_w(self):
        b = BatterySpec(id="T", name="Test", chemistry="lipo",
                        voltage=11.1, capacity_ah=2.2,
                        c_rate=30.0, weight_g=175, price_cny=90)
        assert b.max_discharge_w == pytest.approx(30.0 * 24.42, rel=0.01)

    def test_max_discharge_a(self):
        b = BatterySpec(id="T", name="Test", chemistry="lipo",
                        voltage=11.1, capacity_ah=2.2,
                        c_rate=30.0, weight_g=175, price_cny=90)
        assert b.max_discharge_a == pytest.approx(66.0)

    def test_energy_density_auto_computed(self):
        b = BatterySpec(id="T", name="Test", chemistry="lipo",
                        voltage=11.1, capacity_ah=2.2,
                        c_rate=30.0, weight_g=175, price_cny=90)
        # 24.42 Wh / 0.175 kg ≈ 139.5 Wh/kg
        assert b.energy_density_wh_kg > 100


class TestBatteryCatalog:
    def test_all_have_valid_energy(self):
        for bid, b in BATTERY_CATALOG.items():
            assert b.energy_wh > 0, f"{bid} has zero energy"
            assert b.voltage > 0, f"{bid} has zero voltage"
            assert b.capacity_ah > 0, f"{bid} has zero capacity"

    def test_get_battery(self):
        assert get_battery("LIPO_3S_2200") is not None
        assert get_battery("lipo_3s_2200") is not None
        assert get_battery("NONEXISTENT") is None

    def test_list_batteries_filter(self):
        lipos = list_batteries("lipo")
        assert len(lipos) >= 2
        assert all(b.chemistry == "lipo" for b in lipos)

    def test_list_all(self):
        all_bat = list_batteries()
        assert len(all_bat) == len(BATTERY_CATALOG)


# ============================================================================
# PowerBudgetCalculator
# ============================================================================


class TestPowerBudgetCalculator:
    def test_empty_system(self):
        calc = PowerBudgetCalculator("Empty")
        assert calc.compute_total_peak() == 0.0
        assert calc.compute_total_avg() == 0.0

    def test_add_consumer(self):
        calc = PowerBudgetCalculator("TestBot")
        calc.add_consumer(PowerConsumer(
            name="Motor", category="motor",
            peak_power_w=20.0, avg_power_w=10.0, duty_cycle=0.6,
        ))
        assert calc.compute_total_peak() == 20.0
        assert calc.compute_total_avg() == pytest.approx(6.0)

    def test_multiple_consumers(self):
        calc = PowerBudgetCalculator("TestBot")
        calc.add_consumer(PowerConsumer(
            name="Motor L", category="motor",
            peak_power_w=15.0, avg_power_w=8.0, duty_cycle=0.6,
        ))
        calc.add_consumer(PowerConsumer(
            name="Motor R", category="motor",
            peak_power_w=15.0, avg_power_w=8.0, duty_cycle=0.6,
        ))
        calc.add_consumer(PowerConsumer(
            name="SBC", category="controller",
            peak_power_w=10.0, avg_power_w=7.0, duty_cycle=1.0,
        ))
        assert calc.compute_total_peak() == 40.0
        assert calc.compute_total_avg() == pytest.approx(16.6)

    def test_add_motor(self):
        calc = PowerBudgetCalculator("TestBot")
        calc.add_motor("Drive Motor", "TT_MOTOR", duty_cycle=0.5, quantity=2)
        # add_motor creates one consumer with quantity=2
        assert len(calc.consumers) == 1
        assert calc.consumers[0].quantity == 2
        total_peak = calc.compute_total_peak()
        assert total_peak > 0

    def test_add_controller(self):
        calc = PowerBudgetCalculator("TestBot")
        calc.add_controller("Raspberry Pi", tdp_w=7.5)
        assert len(calc.consumers) == 1
        c = calc.consumers[0]
        assert c.peak_power_w == 7.5
        assert c.avg_power_w == pytest.approx(5.25)  # 70% of TDP

    def test_add_servo(self):
        calc = PowerBudgetCalculator("TestBot")
        calc.add_servo("Arm Joint", "MG996R", duty_cycle=0.3)
        assert len(calc.consumers) == 1
        assert calc.consumers[0].category == "servo"

    def test_estimate_runtime(self):
        calc = PowerBudgetCalculator("TestBot")
        calc.add_consumer(PowerConsumer(
            name="Load", category="other",
            peak_power_w=10.0, avg_power_w=10.0, duty_cycle=1.0,
        ))
        # 2.2Ah * 11.1V * 0.8 / 10W ≈ 1.95h
        runtime = calc.estimate_runtime(2.2, 11.1)
        assert 1.5 < runtime < 2.5

    def test_recommend_battery(self):
        calc = PowerBudgetCalculator("TestBot")
        calc.add_consumer(PowerConsumer(
            name="Motor", category="motor",
            peak_power_w=15.0, avg_power_w=8.0, duty_cycle=0.6,
        ))
        recs = calc.recommend_battery(runtime_target_h=0.5)
        assert len(recs) > 0
        for r in recs:
            assert r["runtime_h"] >= 0.5

    def test_generate_report(self):
        calc = PowerBudgetCalculator("TestBot")
        calc.add_motor("Drive", "TT_MOTOR", quantity=2)
        calc.add_controller("ESP32", tdp_w=2.0)
        report = calc.generate_report()
        assert "Power Budget Report" in report
        assert "TestBot" in report
        assert "Peak power" in report
        assert "Recommended Batteries" in report

    def test_to_dict(self):
        calc = PowerBudgetCalculator("TestBot")
        calc.add_consumer(PowerConsumer(
            name="Load", category="motor",
            peak_power_w=10.0, avg_power_w=5.0, duty_cycle=0.5,
        ))
        d = calc.to_dict()
        assert d["system_name"] == "TestBot"
        assert len(d["consumers"]) == 1
        assert d["total_peak_w"] == 10.0


class TestAutoPopulate:
    def test_from_actuators(self):
        calc = auto_populate_from_actuators(
            ["TT_MOTOR", "TT_MOTOR", "MG996R"],
            compute_controller_w=5.0,
            system_name="DiffBot",
        )
        assert calc.system_name == "DiffBot"
        assert len(calc.consumers) >= 3  # 2 motors + 1 servo + controller
        assert calc.compute_total_peak() > 0

    def test_unknown_actuator(self):
        calc = auto_populate_from_actuators(["UNKNOWN_MOTOR"])
        # Should still work with fallback
        assert calc.compute_total_peak() >= 0


# ============================================================================
# PowerBudgetTool
# ============================================================================


class TestPowerBudgetTool:
    def test_report_mode(self):
        tool = PowerBudgetTool()
        result = tool.execute(
            actuator_ids=["TT_MOTOR", "TT_MOTOR"],
            system_name="DiffBot",
        )
        assert "Power Budget Report" in result
        assert "DiffBot" in result

    def test_json_mode(self):
        import json
        tool = PowerBudgetTool()
        result = tool.execute(
            actuator_ids=["TT_MOTOR"],
            mode="json",
        )
        data = json.loads(result)
        assert "total_peak_w" in data
        assert "consumers" in data

    def test_missing_actuators(self):
        tool = PowerBudgetTool()
        result = tool.execute()
        assert "错误" in result


# ============================================================================
# Registration
# ============================================================================


class TestRegistration:
    def test_power_budget_registered(self):
        registry = ToolRegistry()
        register_power_budget_tools(registry)
        names = [t.name for t in registry._tools.values()]
        assert "power_budget" in names
