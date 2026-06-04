"""Actuator selection and analysis tools.

Tools:
  actuator_select   - Select actuators based on torque/speed/price requirements
  actuator_analyze  - Analyze an assembly's torque requirements and recommend actuators
  actuator_power_budget - Calculate total power consumption and recommend power supply
"""

from __future__ import annotations

import json
from typing import Any

from ..knowledge.actuators import (
    ACTUATORS,
    Actuator,
    get_actuator,
    list_actuators,
    torque_to_nm,
)
from ..knowledge.mechanics import Assembly
from ..knowledge.sensors import get_sensor
from ..models.base import ToolDefinition
from ..tools.assembly_solver import AssemblySolver
from .base import Tool


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------

def select_actuators(
    *,
    min_torque_kgcm: float = 0,
    max_torque_kgcm: float = float("inf"),
    min_rpm: float = 0,
    max_rpm: float = float("inf"),
    max_price_cny: float = float("inf"),
    max_weight_g: float = float("inf"),
    voltage: float = 0,
    category: str = "",
    count: int = 3,
) -> list[dict[str, Any]]:
    """Select actuators matching criteria, sorted by best fit."""
    candidates: list[Actuator] = []
    for a in list_actuators(category):
        if a.torque_kgcm < min_torque_kgcm:
            continue
        if a.torque_kgcm > max_torque_kgcm:
            continue
        if a.rpm < min_rpm and a.category != "servo":
            continue
        if a.rpm > max_rpm and a.rpm > 0:
            continue
        if a.price_cny > max_price_cny:
            continue
        if a.weight_g > max_weight_g and max_weight_g < float("inf"):
            continue
        if voltage > 0 and not (a.voltage_range[0] <= voltage <= a.voltage_range[1]):
            continue
        candidates.append(a)

    # Sort: best torque/weight ratio first
    candidates.sort(key=lambda a: a.torque_kgcm / max(a.weight_g, 1), reverse=True)
    return [_actuator_to_dict(a) for a in candidates[:count]]


# ---------------------------------------------------------------------------
# Assembly torque analysis
# ---------------------------------------------------------------------------

def analyze_assembly_torques(
    assembly: Assembly,
    safety_factor: float = 2.0,
    payload_g: float = 0,
) -> list[dict[str, Any]]:
    """Analyze torque requirements for each revolute joint in an assembly.

    For each joint, estimates the required torque based on the weight
    of all downstream parts + payload, multiplied by the distance to
    the furthest part center.

    Args:
        assembly: The assembly to analyze.
        safety_factor: Torque safety margin (default 2x).
        payload_g: Additional payload at the end-effector (grams).
    """
    solver = AssemblySolver(assembly)
    placements = solver.solve()

    # Build child tree
    children_of: dict[str, list[str]] = {}
    joint_map: dict[str, Any] = {}
    for j in assembly.joints:
        children_of.setdefault(j.parent, []).append(j.child)
        joint_map[j.child] = j

    # Part weights (estimate from material density * volume)
    part_weights: dict[str, float] = {}
    for p in assembly.parts:
        # Rough estimate: PLA density ~1.25 g/cm³
        dims = p.dimensions
        vol_mm3 = 1.0
        for v in dims.values():
            vol_mm3 *= v
        vol_mm3 = max(vol_mm3, 100)  # minimum 100mm³
        part_weights[p.name] = vol_mm3 * 0.00125  # g/mm³ → grams

    results: list[dict[str, Any]] = []

    for j in assembly.joints:
        if j.type != "revolute":
            continue

        # Find joint position
        joint_pos = placements.get(j.parent, {}).get("position", [0, 0, 0])

        # Find all downstream parts (BFS from j.child)
        downstream: set[str] = set()
        stack = [j.child]
        while stack:
            curr = stack.pop()
            if curr in downstream:
                continue
            downstream.add(curr)
            for child in children_of.get(curr, []):
                stack.append(child)

        # Calculate total weight and max lever arm
        total_weight_g = sum(part_weights.get(p, 10) for p in downstream) + payload_g
        max_lever_mm = 0
        for pname in downstream:
            p_pos = placements.get(pname, {}).get("position", joint_pos)
            dx = p_pos[0] - joint_pos[0]
            dy = p_pos[1] - joint_pos[1]
            dz = p_pos[2] - joint_pos[2]
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            max_lever_mm = max(max_lever_mm, dist)

        # Torque = weight * lever arm (converted to kg·cm)
        lever_cm = max_lever_mm / 10.0
        weight_kg = total_weight_g / 1000.0
        required_torque = weight_kg * lever_cm * safety_factor

        # Find recommended actuator
        recs = select_actuators(
            min_torque_kgcm=required_torque,
            max_price_cny=200,
            count=2,
        )

        results.append({
            "joint": j.description or j.child,
            "parent": j.parent,
            "child": j.child,
            "downstream_parts": list(downstream),
            "total_weight_g": round(total_weight_g, 1),
            "max_lever_mm": round(max_lever_mm, 1),
            "required_torque_kgcm": round(required_torque, 2),
            "required_torque_nm": round(torque_to_nm(required_torque), 3),
            "safety_factor": safety_factor,
            "recommended": recs[:2],
        })

    return results


# ---------------------------------------------------------------------------
# Power budget
# ---------------------------------------------------------------------------

def power_budget(
    actuator_ids: list[str],
    duty_cycle: float = 0.3,
    include_margin: float = 1.3,
    sensor_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Calculate power budget for a set of actuators and sensors.

    Args:
        actuator_ids: List of actuator IDs.
        duty_cycle: Fraction of time actuators are under load (0-1).
        include_margin: Power supply overhead factor (default 1.3x).
        sensor_ids: Optional list of sensor IDs to include in power budget.
    """
    actuators: list[Actuator] = []
    for aid in actuator_ids:
        a = get_actuator(aid)
        if a:
            actuators.append(a)

    # Resolve sensors
    resolved_sensors = []
    for sid in (sensor_ids or []):
        s = get_sensor(sid)
        if s:
            resolved_sensors.append(s)

    if not actuators and not resolved_sensors:
        return {"actuators": [], "sensors": [], "count": 0, "total_power_w": 0,
                "supply_power_w": 0, "supply_voltage_v": 0, "supply_current_a": 0,
                "margin_factor": include_margin, "duty_cycle": duty_cycle,
                "recommendation": "无执行器和传感器"}

    # Per-actuator power estimate:
    # Idle power + (stall - idle) * duty_cycle, all at nominal voltage
    total_power_w = 0.0
    max_voltage = 0.0
    actuator_details = []

    for a in actuators:
        avg_current_a = (a.current_idle_ma + (a.current_stall_ma - a.current_idle_ma) * duty_cycle) / 1000.0
        power_w = avg_current_a * a.voltage
        total_power_w += power_w
        max_voltage = max(max_voltage, a.voltage)

        actuator_details.append({
            "id": a.id,
            "name": a.name,
            "voltage_v": a.voltage,
            "avg_current_ma": round(avg_current_a * 1000, 0),
            "power_w": round(power_w, 1),
        })

    # Sensor power (continuous, always on)
    sensor_details = []
    for s in resolved_sensors:
        power_w = (s.current_ma / 1000.0) * s.voltage
        total_power_w += power_w
        max_voltage = max(max_voltage, s.voltage)
        sensor_details.append({
            "id": s.id,
            "name": s.name,
            "voltage_v": s.voltage,
            "current_ma": s.current_ma,
            "power_w": round(power_w, 2),
        })

    supply_power_w = total_power_w * include_margin
    supply_current_a = supply_power_w / max_voltage if max_voltage > 0 else 0

    return {
        "actuators": actuator_details,
        "sensors": sensor_details,
        "count": len(actuators),
        "total_power_w": round(total_power_w, 3),
        "duty_cycle": duty_cycle,
        "supply_power_w": round(supply_power_w, 3),
        "supply_voltage_v": round(max_voltage, 1),
        "supply_current_a": round(supply_current_a, 3),
        "margin_factor": include_margin,
        "recommendation": f"推荐电源: {round(supply_power_w, 1)}W / {round(max_voltage, 1)}V / {round(supply_current_a, 2)}A",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _actuator_to_dict(a: Actuator) -> dict[str, Any]:
    return {
        "id": a.id,
        "name": a.name,
        "category": a.category,
        "torque_kgcm": a.torque_kgcm,
        "torque_nm": round(torque_to_nm(a.torque_kgcm), 3),
        "rpm": a.rpm,
        "voltage": a.voltage,
        "voltage_range": list(a.voltage_range),
        "current_stall_ma": a.current_stall_ma,
        "weight_g": a.weight_g,
        "price_cny": a.price_cny,
        "rotation_range": list(a.rotation_range),
        "interface": a.interface,
        "description": a.description,
    }


# ---------------------------------------------------------------------------
# Tools: actuator_select
# ---------------------------------------------------------------------------

class ActuatorSelectTool(Tool):
    name = "actuator_select"
    description = (
        "根据力矩/速度/价格约束筛选执行器（舵机/电机/步进）。"
        "返回最匹配的执行器列表。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "min_torque_kgcm": {"type": "number", "description": "最小力矩 (kg·cm)"},
                "max_price_cny": {"type": "number", "description": "最高价格 (元)"},
                "max_weight_g": {"type": "number", "description": "最大重量 (g)"},
                "voltage": {"type": "number", "description": "工作电压 (V)"},
                "category": {"type": "string", "description": "类型: servo/dc_motor/stepper/bldc"},
                "count": {"type": "integer", "description": "返回数量 (默认 3)"},
            }, "required": []},
        )

    def execute(self, *, min_torque_kgcm: float = 0, max_price_cny: float = 9999,
                max_weight_g: float = 99999, voltage: float = 0,
                category: str = "", count: int = 3, **kwargs: Any) -> str:
        results = select_actuators(
            min_torque_kgcm=min_torque_kgcm, max_price_cny=max_price_cny,
            max_weight_g=max_weight_g, voltage=voltage,
            category=category, count=count,
        )
        if not results:
            return "未找到匹配的执行器。尝试放宽约束条件。"
        lines = [f"[Actuator Select] 找到 {len(results)} 个匹配:"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"  {i}. {r['name']} ({r['id']}): "
                f"{r['torque_kgcm']}kg·cm, {r['voltage']}V, "
                f"{r['weight_g']}g, ¥{r['price_cny']}"
            )
        lines.append("\n--- JSON ---")
        lines.append(json.dumps(results, ensure_ascii=False, indent=2))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools: actuator_analyze
# ---------------------------------------------------------------------------

class ActuatorAnalyzeTool(Tool):
    name = "actuator_analyze"
    description = (
        "分析装配体的力矩需求，为每个关节推荐执行器。"
        "输入装配体名称，输出每个旋转关节的力矩分析和执行器推荐。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "assembly_name": {"type": "string", "description": "装配体名称（如 robotic_arm）"},
                "safety_factor": {"type": "number", "description": "安全系数（默认 2.0）"},
                "payload_g": {"type": "number", "description": "末端负载（克，默认 0）"},
            }, "required": []},
        )

    def execute(self, *, assembly_name: str = "robotic_arm",
                safety_factor: float = 2.0, payload_g: float = 0,
                **kwargs: Any) -> str:
        from .assembly_solver import _resolve_assembly
        assembly = _resolve_assembly(assembly_name, "")
        if assembly is None:
            return f"错误：未找到装配体 '{assembly_name}'"

        results = analyze_assembly_torques(assembly, safety_factor, payload_g)
        if not results:
            return "该装配体没有旋转关节，无需力矩分析。"

        lines = [f"[Torque Analysis] {assembly.name}", f"安全系数: {safety_factor}x, 负载: {payload_g}g", ""]
        for r in results:
            lines.append(f"  关节: {r['joint']}")
            lines.append(f"    下游零件: {', '.join(r['downstream_parts'])}")
            lines.append(f"    总重量: {r['total_weight_g']}g, 最大力臂: {r['max_lever_mm']}mm")
            lines.append(f"    需求力矩: {r['required_torque_kgcm']} kg·cm ({r['required_torque_nm']} N·m)")
            if r["recommended"]:
                rec = r["recommended"][0]
                lines.append(f"    推荐: {rec['name']} ({rec['torque_kgcm']}kg·cm, ¥{rec['price_cny']})")
            lines.append("")

        lines.append("--- JSON ---")
        lines.append(json.dumps(results, ensure_ascii=False, indent=2))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools: actuator_power_budget
# ---------------------------------------------------------------------------

class ActuatorPowerBudgetTool(Tool):
    name = "actuator_power_budget"
    description = (
        "计算执行器组的总功耗，推荐电源规格。"
        "输入执行器 ID 列表，输出功耗分析和电源推荐。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "actuator_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "执行器 ID 列表（如 ['MG996R', 'MG996R', 'SG90']）",
                },
                "duty_cycle": {"type": "number", "description": "占空比 (0-1, 默认 0.3)"},
                "sensor_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "传感器 ID 列表（如 ['AS5600', 'MPU6050']）",
                },
            }, "required": ["actuator_ids"]},
        )

    def execute(self, *, actuator_ids: list[str], duty_cycle: float = 0.3,
                sensor_ids: list[str] | None = None,
                **kwargs: Any) -> str:
        result = power_budget(actuator_ids, duty_cycle, sensor_ids=sensor_ids)
        if "error" in result:
            return result["error"]

        lines = [
            f"[Power Budget] {result['count']} 个执行器" +
            (f" + {len(result.get('sensors', []))} 个传感器" if result.get('sensors') else ""),
            f"占空比: {duty_cycle * 100:.0f}%, 裕量: {result['margin_factor']}x",
            "",
            "--- 执行器功耗 ---",
        ]
        for a in result["actuators"]:
            lines.append(f"  {a['name']}: {a['voltage_v']}V / {a['avg_current_ma']:.0f}mA = {a['power_w']}W")

        if result.get("sensors"):
            lines.append("")
            lines.append("--- 传感器功耗 ---")
            for s in result["sensors"]:
                lines.append(f"  {s['name']}: {s['voltage_v']}V / {s['current_ma']:.0f}mA = {s['power_w']}W")

        lines.extend([
            "",
            f"总功耗: {result['total_power_w']}W",
            f"推荐电源: {result['supply_power_w']}W / {result['supply_voltage_v']}V / {result['supply_current_a']}A",
            "",
            "--- JSON ---",
            json.dumps(result, ensure_ascii=False, indent=2),
        ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_actuator_tools(registry: Any) -> None:
    """Register actuator selection and analysis tools."""
    registry.register(ActuatorSelectTool())
    registry.register(ActuatorAnalyzeTool())
    registry.register(ActuatorPowerBudgetTool())
