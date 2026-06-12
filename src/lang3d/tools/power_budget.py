"""Power budget calculation and battery selection for robotic systems.

Provides:
  - PowerConsumer: individual power load definition
  - PowerBudgetCalculator: system-level power analysis
  - BATTERY_CATALOG: common battery specifications
  - PowerBudgetTool: Agent tool for power budget reports
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from ..knowledge.actuators import get_actuator, list_actuators
from ..models.base import ToolDefinition
from .base import Tool


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PowerConsumer:
    """A single power consumer in the system."""
    name: str                    # e.g. "Left motor", "IPC", "Servo 1"
    category: str                # "motor", "servo", "controller", "sensor", "compute", "other"
    peak_power_w: float          # Peak power draw (W)
    avg_power_w: float           # Average power draw (W)
    duty_cycle: float = 1.0      # Fraction of time active (0..1)
    voltage: float = 0.0         # Operating voltage (V), 0 = unspecified
    quantity: int = 1            # Number of identical units

    @property
    def effective_avg_w(self) -> float:
        """Average power accounting for duty cycle and quantity."""
        return self.avg_power_w * self.duty_cycle * self.quantity

    @property
    def effective_peak_w(self) -> float:
        """Peak power accounting for quantity."""
        return self.peak_power_w * self.quantity


@dataclass
class BatterySpec:
    """Battery specification."""
    id: str                      # e.g. "LIPO_3S_2200"
    name: str                    # e.g. "LiPo 3S 2200mAh"
    chemistry: str               # "lipo", "lifepo4", "18650", "lead_acid", "nimh"
    voltage: float               # Nominal voltage (V)
    capacity_ah: float           # Capacity in Ah
    c_rate: float                # Max continuous discharge rate (C)
    weight_g: float              # Weight (g)
    price_cny: float             # Approximate price (CNY)
    cycle_life: int = 500        # Charge/discharge cycles
    energy_density_wh_kg: float = 0.0  # Wh/kg (auto-computed if 0)
    description: str = ""

    def __post_init__(self):
        if self.energy_density_wh_kg <= 0 and self.weight_g > 0:
            self.energy_density_wh_kg = (self.voltage * self.capacity_ah) / (self.weight_g / 1000.0)

    @property
    def energy_wh(self) -> float:
        return self.voltage * self.capacity_ah

    @property
    def max_discharge_w(self) -> float:
        return self.c_rate * self.energy_wh

    @property
    def max_discharge_a(self) -> float:
        return self.c_rate * self.capacity_ah


# ---------------------------------------------------------------------------
# Battery catalog
# ---------------------------------------------------------------------------

BATTERY_CATALOG: dict[str, BatterySpec] = {
    "LIPO_2S_1300": BatterySpec(
        id="LIPO_2S_1300", name="LiPo 2S 1300mAh 25C",
        chemistry="lipo", voltage=7.4, capacity_ah=1.3,
        c_rate=25.0, weight_g=82, price_cny=50,
        cycle_life=300,
        description="小型无人机/机器人常用，轻量级",
    ),
    "LIPO_3S_2200": BatterySpec(
        id="LIPO_3S_2200", name="LiPo 3S 2200mAh 30C",
        chemistry="lipo", voltage=11.1, capacity_ah=2.2,
        c_rate=30.0, weight_g=175, price_cny=90,
        cycle_life=300,
        description="中型机器人/AGV 底盘动力",
    ),
    "LIPO_4S_5000": BatterySpec(
        id="LIPO_4S_5000", name="LiPo 4S 5000mAh 20C",
        chemistry="lipo", voltage=14.8, capacity_ah=5.0,
        c_rate=20.0, weight_g=480, price_cny=200,
        cycle_life=300,
        description="大型底盘/双臂机器人主电源",
    ),
    "LIPO_6S_5000": BatterySpec(
        id="LIPO_6S_5000", name="LiPo 6S 5000mAh 15C",
        chemistry="lipo", voltage=22.2, capacity_ah=5.0,
        c_rate=15.0, weight_g=680, price_cny=320,
        cycle_life=300,
        description="工业级 AGV/大型机器人",
    ),
    "LIFEPO4_12V_10AH": BatterySpec(
        id="LIFEPO4_12V_10AH", name="LiFePO4 12V 10Ah",
        chemistry="lifepo4", voltage=12.8, capacity_ah=10.0,
        c_rate=2.0, weight_g=1200, price_cny=250,
        cycle_life=2000,
        description="安全长寿，适合固定/移动基站",
    ),
    "LIFEPO4_24V_20AH": BatterySpec(
        id="LIFEPO4_24V_20AH", name="LiFePO4 24V 20Ah",
        chemistry="lifepo4", voltage=25.6, capacity_ah=20.0,
        c_rate=2.0, weight_g=3200, price_cny=600,
        cycle_life=2000,
        description="大型 AGV/仓储机器人",
    ),
    "18650_3S2P_6000": BatterySpec(
        id="18650_3S2P_6000", name="18650 3S2P 6000mAh (6x 3000mAh)",
        chemistry="18650", voltage=11.1, capacity_ah=6.0,
        c_rate=5.0, weight_g=330, price_cny=150,
        cycle_life=800,
        description="18650 锂电组，成本/性能平衡",
    ),
    "18650_4S2P_6000": BatterySpec(
        id="18650_4S2P_6000", name="18650 4S2P 6000mAh (8x 3000mAh)",
        chemistry="18650", voltage=14.8, capacity_ah=6.0,
        c_rate=5.0, weight_g=440, price_cny=200,
        cycle_life=800,
        description="14.8V 18650 组，适合中型底盘",
    ),
    "LEAD_12V_7AH": BatterySpec(
        id="LEAD_12V_7AH", name="铅酸 12V 7Ah",
        chemistry="lead_acid", voltage=12.0, capacity_ah=7.0,
        c_rate=0.5, weight_g=2100, price_cny=60,
        cycle_life=200,
        description="低成本 UPS/AGV 备用电源，重",
    ),
}


def get_battery(battery_id: str) -> BatterySpec | None:
    """Look up a battery by ID."""
    return BATTERY_CATALOG.get(battery_id.upper())


def list_batteries(chemistry: str = "") -> list[BatterySpec]:
    """List batteries, optionally filtered by chemistry."""
    batteries = list(BATTERY_CATALOG.values())
    if chemistry:
        batteries = [b for b in batteries if b.chemistry == chemistry]
    return batteries


# ---------------------------------------------------------------------------
# Power budget calculator
# ---------------------------------------------------------------------------

class PowerBudgetCalculator:
    """System-level power budget analysis."""

    def __init__(self, system_name: str = "Robot"):
        self.system_name = system_name
        self.consumers: list[PowerConsumer] = []

    def add_consumer(self, consumer: PowerConsumer) -> None:
        self.consumers.append(consumer)

    def add_motor(self, name: str, motor_id: str, duty_cycle: float = 0.6,
                  quantity: int = 1) -> None:
        """Add a DC motor consumer from actuator database."""
        act = get_actuator(motor_id)
        if act is None:
            # Fallback: estimate from name
            self.add_consumer(PowerConsumer(
                name=name, category="motor",
                peak_power_w=10.0, avg_power_w=5.0,
                duty_cycle=duty_cycle, quantity=quantity,
            ))
            return

        # Power = V * I
        peak_w = act.voltage * (act.current_stall_ma / 1000.0)
        avg_w = act.voltage * (act.current_idle_ma / 1000.0 + (act.current_stall_ma - act.current_idle_ma) / 1000.0 * 0.3)

        self.add_consumer(PowerConsumer(
            name=name, category="motor",
            peak_power_w=round(peak_w, 2),
            avg_power_w=round(avg_w, 2),
            duty_cycle=duty_cycle,
            voltage=act.voltage,
            quantity=quantity,
        ))

    def add_controller(self, name: str, tdp_w: float, voltage: float = 5.0) -> None:
        """Add a compute/controller load (IPC, SBC, MCU)."""
        self.add_consumer(PowerConsumer(
            name=name, category="controller",
            peak_power_w=tdp_w, avg_power_w=tdp_w * 0.7,
            duty_cycle=1.0, voltage=voltage,
        ))

    def add_sensor_load(self, name: str, power_w: float, quantity: int = 1) -> None:
        """Add sensor power draw."""
        self.add_consumer(PowerConsumer(
            name=name, category="sensor",
            peak_power_w=power_w, avg_power_w=power_w,
            duty_cycle=1.0, quantity=quantity,
        ))

    def add_servo(self, name: str, servo_id: str, duty_cycle: float = 0.3,
                  quantity: int = 1) -> None:
        """Add a servo consumer from actuator database."""
        act = get_actuator(servo_id)
        if act is None:
            self.add_consumer(PowerConsumer(
                name=name, category="servo",
                peak_power_w=5.0, avg_power_w=2.0,
                duty_cycle=duty_cycle, quantity=quantity,
            ))
            return

        peak_w = act.voltage * (act.current_stall_ma / 1000.0)
        # Average power at ~20% load: servos draw holding current continuously,
        # with intermittent bursts during movement.  20% of (stall-idle) + idle
        # gives a realistic operating average.
        avg_w = act.voltage * (
            act.current_idle_ma / 1000.0
            + (act.current_stall_ma - act.current_idle_ma) / 1000.0 * 0.2
        )

        self.add_consumer(PowerConsumer(
            name=name, category="servo",
            peak_power_w=round(peak_w, 2),
            avg_power_w=round(avg_w, 2),
            duty_cycle=duty_cycle,
            voltage=act.voltage,
            quantity=quantity,
        ))

    def compute_total_peak(self) -> float:
        """Total peak power (W) — worst case all consumers at full load."""
        return sum(c.effective_peak_w for c in self.consumers)

    def compute_total_avg(self) -> float:
        """Total average power (W) — realistic operating estimate."""
        total = sum(c.effective_avg_w for c in self.consumers)
        # Apply a minimum floor: a robot with servos/controllers should never
        # report < 1 W average.  This prevents absurd battery life estimates
        # (e.g., 1000+ hours) when duty cycles are low.
        if total > 0 and total < 1.0 and self.consumers:
            total = 1.0
        return total

    def estimate_runtime(self, battery_ah: float, battery_voltage: float,
                         safety_factor: float = 0.8) -> float:
        """Estimate runtime in hours given battery specs.

        Args:
            battery_ah: Battery capacity in Ah.
            battery_voltage: Battery nominal voltage in V.
            safety_factor: Usable capacity fraction (0.8 = 80% DoD).
        """
        usable_wh = battery_ah * battery_voltage * safety_factor
        avg_w = self.compute_total_avg()
        if avg_w <= 0:
            return float("inf")
        return usable_wh / avg_w

    def recommend_battery(self, runtime_target_h: float = 1.0,
                          chemistry: str = "") -> list[dict[str, Any]]:
        """Recommend batteries meeting runtime target.

        Returns list of {battery, runtime_h, margin_pct} sorted by runtime.
        """
        results: list[dict[str, Any]] = []
        for bat in list_batteries(chemistry):
            runtime = self.estimate_runtime(bat.capacity_ah, bat.voltage)
            margin = (runtime / runtime_target_h - 1.0) * 100
            if runtime >= runtime_target_h:
                results.append({
                    "battery": bat,
                    "runtime_h": round(runtime, 2),
                    "margin_pct": round(margin, 1),
                })

        results.sort(key=lambda r: r["runtime_h"], reverse=True)
        return results

    def generate_report(self) -> str:
        """Generate markdown power budget report."""
        lines = [
            f"# Power Budget Report — {self.system_name}",
            "",
            "## Power Consumers",
            "",
            "| # | Component | Category | Qty | Peak (W) | Avg (W) | Duty | Eff. Avg (W) |",
            "|---|-----------|----------|-----|----------|---------|------|-------------|",
        ]

        for i, c in enumerate(self.consumers, 1):
            lines.append(
                f"| {i} | {c.name} | {c.category} | {c.quantity} | "
                f"{c.peak_power_w:.1f} | {c.avg_power_w:.1f} | "
                f"{c.duty_cycle:.0%} | {c.effective_avg_w:.1f} |"
            )

        total_peak = self.compute_total_peak()
        total_avg = self.compute_total_avg()

        lines.extend([
            f"| | **TOTAL** | | | **{total_peak:.1f}** | **{total_avg:.1f}** | | **{total_avg:.1f}** |",
            "",
            "## Summary",
            f"- **Peak power**: {total_peak:.1f} W",
            f"- **Average power**: {total_avg:.1f} W",
            f"- **Number of loads**: {len(self.consumers)}",
            "",
            "## Recommended Batteries (1h target)",
            "",
            "| Battery | Voltage | Capacity | Runtime | Margin | Weight | Price |",
            "|---------|---------|----------|---------|--------|--------|-------|",
        ])

        recs = self.recommend_battery(runtime_target_h=1.0)
        if recs:
            for r in recs[:5]:
                b = r["battery"]
                lines.append(
                    f"| {b.name} | {b.voltage:.1f}V | {b.capacity_ah:.1f}Ah | "
                    f"{r['runtime_h']:.1f}h | +{r['margin_pct']:.0f}% | "
                    f"{b.weight_g:.0f}g | ¥{b.price_cny:.0f} |"
                )
        else:
            lines.append("| No suitable battery found | | | | | | |")

        lines.extend(["", "## Power Supply Recommendations"])
        lines.append(f"- Minimum PSU rating: **{total_peak * 1.2:.0f} W** (20% headroom)")
        lines.append(f"- Recommended battery: **{total_avg:.0f} Wh minimum** for 1h operation")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export as serializable dict."""
        return {
            "system_name": self.system_name,
            "consumers": [
                {
                    "name": c.name,
                    "category": c.category,
                    "peak_power_w": c.peak_power_w,
                    "avg_power_w": c.avg_power_w,
                    "duty_cycle": c.duty_cycle,
                    "voltage": c.voltage,
                    "quantity": c.quantity,
                    "effective_avg_w": c.effective_avg_w,
                    "effective_peak_w": c.effective_peak_w,
                }
                for c in self.consumers
            ],
            "total_peak_w": round(self.compute_total_peak(), 2),
            "total_avg_w": round(self.compute_total_avg(), 2),
        }


# ---------------------------------------------------------------------------
# Auto-populate from assembly
# ---------------------------------------------------------------------------

def auto_populate_from_actuators(
    actuator_ids: list[str],
    compute_controller_w: float = 10.0,
    system_name: str = "Robot",
) -> PowerBudgetCalculator:
    """Create a power budget calculator pre-populated from actuator list.

    Args:
        actuator_ids: List of actuator IDs used in the system.
        compute_controller_w: Power draw of main controller (W).
        system_name: System name for report.
    """
    calc = PowerBudgetCalculator(system_name)

    # Group actuators by type for labeling
    counts: dict[str, int] = {}
    for aid in actuator_ids:
        act = get_actuator(aid)
        cat = act.category if act else "other"
        counts[cat] = counts.get(cat, 0) + 1

    # Add individual actuators
    motor_idx = 0
    servo_idx = 0
    for aid in actuator_ids:
        act = get_actuator(aid)
        if act is None:
            continue
        if act.category == "dc_motor":
            motor_idx += 1
            calc.add_motor(f"Motor {motor_idx} ({act.name})", aid, duty_cycle=0.6)
        elif act.category == "servo":
            servo_idx += 1
            calc.add_servo(f"Servo {servo_idx} ({act.name})", aid, duty_cycle=0.3)
        elif act.category in ("stepper", "bldc"):
            motor_idx += 1
            calc.add_motor(f"Motor {motor_idx} ({act.name})", aid, duty_cycle=0.5)
        else:
            peak_w = act.voltage * (act.current_stall_ma / 1000.0)
            calc.add_consumer(PowerConsumer(
                name=f"{act.name}", category="other",
                peak_power_w=round(peak_w, 2),
                avg_power_w=round(peak_w * 0.3, 2),
                duty_cycle=0.5,
            ))

    if compute_controller_w > 0:
        calc.add_controller("Main Controller", compute_controller_w)

    return calc


# ---------------------------------------------------------------------------
# Tool: power_budget
# ---------------------------------------------------------------------------


class PowerBudgetTool(Tool):
    name = "power_budget"
    description = (
        "功率预算分析：计算系统总功耗（峰值/平均），推荐电池选型，"
        "估算运行时间，生成功率预算报告。支持电机/舵机/控制器/传感器负载。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "actuator_ids": {
                    "type": "array", "items": {"type": "string"},
                    "description": "执行器 ID 列表",
                },
                "compute_controller_w": {
                    "type": "number",
                    "description": "主控功耗 W（默认 10）",
                },
                "system_name": {
                    "type": "string",
                    "description": "系统名称",
                },
                "mode": {
                    "type": "string",
                    "enum": ["report", "json"],
                    "description": "输出模式：report(Markdown) 或 json（默认 report）",
                },
            }, "required": ["actuator_ids"]},
        )

    def execute(self, *, actuator_ids: list[str] | None = None,
                compute_controller_w: float = 10.0,
                system_name: str = "Robot",
                mode: str = "report",
                **kwargs: Any) -> str:
        if not actuator_ids:
            return "错误：未指定执行器 ID 列表"

        calc = auto_populate_from_actuators(
            actuator_ids, compute_controller_w, system_name
        )

        if mode == "json":
            return json.dumps(calc.to_dict(), ensure_ascii=False, indent=2)

        return calc.generate_report()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_power_budget_tools(registry: Any) -> None:
    """Register power budget tools."""
    registry.register(PowerBudgetTool())
