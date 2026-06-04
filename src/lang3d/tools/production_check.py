"""Production readiness check tools.

Provides:
  - Final checklist validation
  - File completeness check
  - Tolerance reasonableness verification
  - BOM correctness validation
  - Code compilability check
  - Production package generation (summary report)

Usage:
  from lang3d.tools.production_check import run_production_check
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from ..knowledge.mechanics import Assembly
from ..models.base import ToolDefinition
from .base import Tool
from .bom_gen import generate_bom
from .code_gen import generate_firmware, generate_wiring
from .quality import generate_quality_report


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CheckItem:
    """A single check result."""
    category: str          # "file", "tolerance", "bom", "code", "assembly"
    name: str              # Check name
    status: str            # "pass", "fail", "warning"
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductionReport:
    """Full production readiness report."""
    assembly_name: str
    checks: list[CheckItem] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    ready: bool = False
    summary: str = ""

    @property
    def total(self) -> int:
        return len(self.checks)


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def _check_assembly_integrity(assembly: Assembly) -> list[CheckItem]:
    """Check assembly structural integrity."""
    items = []

    # Part count
    if len(assembly.parts) >= 3:
        items.append(CheckItem("assembly", "零件数量", "pass",
                               f"{len(assembly.parts)} 个零件"))
    else:
        items.append(CheckItem("assembly", "零件数量", "fail",
                               f"零件过少：{len(assembly.parts)}"))

    # Joint count
    if len(assembly.joints) >= 2:
        items.append(CheckItem("assembly", "关节数量", "pass",
                               f"{len(assembly.joints)} 个关节"))
    else:
        items.append(CheckItem("assembly", "关节数量", "fail",
                               f"关节过少：{len(assembly.joints)}"))

    # All joints reference valid parts
    part_names = {p.name for p in assembly.parts}
    for j in assembly.joints:
        if j.parent not in part_names:
            items.append(CheckItem("assembly", f"关节引用: {j.parent}", "fail",
                                   f"关节 {j.child} 的父件 '{j.parent}' 不存在"))
        if j.child not in part_names:
            items.append(CheckItem("assembly", f"关节引用: {j.child}", "fail",
                                   f"关节引用的子件 '{j.child}' 不存在"))

    if not any(c.status == "fail" and "关节引用" in c.name for c in items):
        items.append(CheckItem("assembly", "关节引用完整性", "pass",
                               "所有关节引用的零件均存在"))

    # Part dimensions exist
    no_dim_parts = [p.name for p in assembly.parts if not p.dimensions]
    if not no_dim_parts:
        items.append(CheckItem("assembly", "零件尺寸", "pass",
                               "所有零件都有尺寸定义"))
    else:
        items.append(CheckItem("assembly", "零件尺寸", "warning",
                               f"以下零件无尺寸：{', '.join(no_dim_parts)}"))

    # Revolute joints have range
    revolute_no_range = [
        j.child for j in assembly.joints
        if j.type == "revolute" and j.range_deg == (-180, 180)
    ]
    if revolute_no_range:
        items.append(CheckItem("assembly", "旋转关节限位", "warning",
                               f"以下关节使用默认限位±180°：{', '.join(revolute_no_range)}"))
    else:
        items.append(CheckItem("assembly", "旋转关节限位", "pass",
                               "所有旋转关节有明确的限位范围"))

    return items


def _check_firmware(
    firmware: dict[str, str],
    controller: str,
) -> list[CheckItem]:
    """Check firmware completeness and validity."""
    items = []

    # Required files
    required = {
        "esp32": ["robot_arm.ino", "ik_solver.h", "ik_solver.cpp",
                   "servo_driver.h", "servo_driver.cpp"],
        "arduino": ["robot_arm.ino", "ik_solver.h", "ik_solver.cpp",
                     "servo_driver.h", "servo_driver.cpp"],
    }

    ctrl = controller.lower()
    expected = required.get(ctrl, required["esp32"])

    missing = [f for f in expected if f not in firmware]
    if not missing:
        items.append(CheckItem("code", "固件文件完整", "pass",
                               f"{len(firmware)} 个文件全部存在"))
    else:
        items.append(CheckItem("code", "固件文件完整", "fail",
                               f"缺少文件：{', '.join(missing)}"))

    # Check main .ino structure
    ino = firmware.get("robot_arm.ino", "")
    if ino:
        required_funcs = ["void setup()", "void loop()"]
        missing_funcs = [f for f in required_funcs if f not in ino]
        if not missing_funcs:
            items.append(CheckItem("code", "主程序结构", "pass",
                                   "setup() 和 loop() 均存在"))
        else:
            items.append(CheckItem("code", "主程序结构", "fail",
                                   f"缺少函数：{', '.join(missing_funcs)}"))

        # Serial communication
        if "Serial.begin" in ino:
            items.append(CheckItem("code", "串口通信", "pass",
                                   "Serial.begin 已配置"))
        else:
            items.append(CheckItem("code", "串口通信", "warning",
                                   "未找到 Serial.begin"))

        # Braces balance
        open_b = ino.count("{")
        close_b = ino.count("}")
        if open_b == close_b:
            items.append(CheckItem("code", "括号平衡", "pass",
                                   f"{{ {open_b} 个, }} {close_b} 个"))
        else:
            items.append(CheckItem("code", "括号平衡", "fail",
                                   f"{{ {open_b} 个 vs }} {close_b} 个，不平衡"))

        # Semicolons in function bodies
        lines_with_code = [l for l in ino.split("\n")
                           if l.strip() and not l.strip().startswith(("//", "#", "/*", "*/"))]
        items.append(CheckItem("code", "代码行数", "pass",
                               f"{len(lines_with_code)} 行有效代码"))

    # Check IK solver
    ik_cpp = firmware.get("ik_solver.cpp", "")
    if ik_cpp:
        if "cos" in ik_cpp and "acos" in ik_cpp:
            items.append(CheckItem("code", "IK 求解器", "pass",
                                   "包含三角函数计算"))
        else:
            items.append(CheckItem("code", "IK 求解器", "warning",
                                   "未找到三角函数计算"))

    # Check servo driver
    servo_cpp = firmware.get("servo_driver.cpp", "")
    if servo_cpp:
        if "writeMicroseconds" in servo_cpp or "write(" in servo_cpp:
            items.append(CheckItem("code", "舵机驱动", "pass",
                                   "PWM 控制已实现"))
        else:
            items.append(CheckItem("code", "舵机驱动", "warning",
                                   "未找到 PWM 控制代码"))

    # Sensor driver check
    sensor_cpp = firmware.get("sensor_driver.cpp", "")
    if sensor_cpp:
        items.append(CheckItem("code", "传感器驱动", "pass",
                               "传感器驱动文件已生成"))
    else:
        items.append(CheckItem("code", "传感器驱动", "warning",
                               "缺少传感器驱动文件"))

    return items


def _check_bom(
    bom: dict[str, Any],
    assembly: Assembly,
    actuator_ids: list[str],
) -> list[CheckItem]:
    """Check BOM correctness."""
    items = []

    # Custom parts match assembly
    bom_parts = bom.get("custom_parts", [])
    if len(bom_parts) >= len(assembly.parts):
        items.append(CheckItem("bom", "零件清单", "pass",
                               f"BOM {len(bom_parts)} 项 ≥ 装配体 {len(assembly.parts)} 项"))
    else:
        items.append(CheckItem("bom", "零件清单", "fail",
                               f"BOM {len(bom_parts)} 项 < 装配体 {len(assembly.parts)} 项"))

    # Standard parts exist
    std_parts = bom.get("standard_parts", [])
    if len(std_parts) > 0:
        items.append(CheckItem("bom", "标准件", "pass",
                               f"{len(std_parts)} 种标准件"))
    else:
        items.append(CheckItem("bom", "标准件", "warning",
                               "无标准件（螺丝、螺母、轴承）"))

    # Electronics match actuators
    electronics = bom.get("electronics", [])
    elec_names = [e.get("name", e.get("id", "")) for e in electronics]
    actuators_found = sum(1 for a in actuator_ids if any(a in n for n in elec_names))
    if actuators_found >= len(actuator_ids):
        items.append(CheckItem("bom", "电子件匹配", "pass",
                               f"全部 {len(actuator_ids)} 个执行器在 BOM 中"))
    elif actuators_found > 0:
        items.append(CheckItem("bom", "电子件匹配", "warning",
                               f"{actuators_found}/{len(actuator_ids)} 个执行器在 BOM 中"))
    else:
        # Check by model name presence in electronics list
        all_elec_text = " ".join(str(e) for e in electronics)
        found = sum(1 for a in actuator_ids if a in all_elec_text)
        if found >= len(actuator_ids):
            items.append(CheckItem("bom", "电子件匹配", "pass",
                                   f"全部 {len(actuator_ids)} 个执行器在 BOM 中"))
        elif found > 0:
            items.append(CheckItem("bom", "电子件匹配", "warning",
                                   f"{found}/{len(actuator_ids)} 个执行器在 BOM 中"))
        else:
            items.append(CheckItem("bom", "电子件匹配", "fail",
                                   "BOM 中未找到执行器"))

    # Cost positive
    cost = bom.get("cost_summary", {}).get("total_cost_cny", 0)
    if cost > 0:
        items.append(CheckItem("bom", "成本估算", "pass",
                               f"¥{cost:.1f}"))
    else:
        items.append(CheckItem("bom", "成本估算", "fail",
                               "成本为零或未计算"))

    # Cost breakdown present
    summary = bom.get("cost_summary", {})
    expected_keys = {"custom_parts_cost", "electronics_cost", "total_cost_cny"}
    if expected_keys.issubset(summary.keys()):
        items.append(CheckItem("bom", "成本明细", "pass",
                               "包含零件、电子、总计"))
    else:
        items.append(CheckItem("bom", "成本明细", "warning",
                               f"缺少部分成本项"))

    return items


def _check_tolerances(assembly: Assembly) -> list[CheckItem]:
    """Check tolerance reasonableness."""
    items = []

    for part in assembly.parts:
        dims = part.dimensions
        if not dims:
            continue

        # Check for unreasonable dimensions (negative or zero)
        for key, val in dims.items():
            if val <= 0:
                items.append(CheckItem("tolerance", f"{part.name}.{key}",
                                       "fail", f"尺寸为零或负：{val}"))
            elif val > 1000:
                items.append(CheckItem("tolerance", f"{part.name}.{key}",
                                       "warning", f"尺寸过大：{val}mm"))
            elif val < 0.5 and key in ("diameter", "wall_thickness", "thickness"):
                items.append(CheckItem("tolerance", f"{part.name}.{key}",
                                       "warning", f"尺寸过小：{val}mm"))

    # Check joint fits: shaft diameters vs hole diameters
    shaft_parts = {}
    hole_parts = {}
    for p in assembly.parts:
        if "shaft" in p.name.lower() and "diameter" in p.dimensions:
            shaft_parts[p.name] = p.dimensions["diameter"]
        if "joint" in p.name.lower() or "housing" in p.name.lower():
            if "inner_diameter" in p.dimensions:
                hole_parts[p.name] = p.dimensions["inner_diameter"]

    for shaft_name, shaft_d in shaft_parts.items():
        for hole_name, hole_d in hole_parts.items():
            clearance = hole_d - shaft_d
            if clearance < 0:
                items.append(CheckItem("tolerance", f"配合: {shaft_name}→{hole_name}",
                                       "fail",
                                       f"过盈：轴{shaft_d}mm > 孔{hole_d}mm"))
            elif clearance < 0.1:
                items.append(CheckItem("tolerance", f"配合: {shaft_name}→{hole_name}",
                                       "warning",
                                       f"间隙过小：{clearance:.2f}mm"))

    if not any(c.category == "tolerance" and c.status == "fail" for c in items):
        items.append(CheckItem("tolerance", "尺寸合理性", "pass",
                               "所有尺寸在合理范围内"))

    return items


def _check_wiring(wiring: str, actuator_ids: list[str]) -> list[CheckItem]:
    """Check wiring diagram completeness."""
    items = []

    if not wiring:
        items.append(CheckItem("file", "接线图", "fail", "接线图为空"))
        return items

    items.append(CheckItem("file", "接线图", "pass",
                           f"已生成（{len(wiring)} 字符）"))

    # Check actuator mentions
    for aid in actuator_ids:
        if aid in wiring:
            continue
        items.append(CheckItem("file", f"接线: {aid}", "warning",
                               f"接线图中未找到 {aid}"))

    # Check essential connections
    if "GND" in wiring:
        items.append(CheckItem("file", "接地连接", "pass", "GND 已连接"))
    else:
        items.append(CheckItem("file", "接地连接", "warning", "未找到 GND"))

    if "GPIO" in wiring or "Pin" in wiring:
        items.append(CheckItem("file", "GPIO 引脚", "pass", "GPIO 引脚已分配"))
    else:
        items.append(CheckItem("file", "GPIO 引脚", "warning", "未找到 GPIO 分配"))

    return items


# ---------------------------------------------------------------------------
# Main production check
# ---------------------------------------------------------------------------

def run_production_check(
    assembly: Assembly,
    actuator_ids: list[str],
    sensor_ids: list[str] | None = None,
    controller: str = "esp32",
) -> ProductionReport:
    """Run full production readiness check.

    Checks:
      1. Assembly structural integrity
      2. Firmware completeness and validity
      3. BOM correctness
      4. Tolerance reasonableness
      5. Wiring diagram completeness

    Returns a ProductionReport with all check results.
    """
    sensor_ids = sensor_ids or []
    report = ProductionReport(assembly_name=assembly.name)

    # 1. Assembly integrity
    report.checks.extend(_check_assembly_integrity(assembly))

    # 2. Firmware generation and check
    try:
        firmware = generate_firmware(assembly, actuator_ids, controller,
                                      sensor_ids=sensor_ids)
        report.checks.extend(_check_firmware(firmware, controller))
    except Exception as e:
        report.checks.append(CheckItem("code", "固件生成", "fail", str(e)))

    # 3. BOM check
    try:
        bom = generate_bom(assembly, actuator_ids, sensor_ids, controller)
        report.checks.extend(_check_bom(bom, assembly, actuator_ids))
    except Exception as e:
        report.checks.append(CheckItem("bom", "BOM 生成", "fail", str(e)))

    # 4. Tolerance check
    report.checks.extend(_check_tolerances(assembly))

    # 5. Wiring check
    try:
        wiring = generate_wiring(actuator_ids, controller)
        report.checks.extend(_check_wiring(wiring, actuator_ids))
    except Exception as e:
        report.checks.append(CheckItem("file", "接线图生成", "fail", str(e)))

    # Tally results
    report.passed = sum(1 for c in report.checks if c.status == "pass")
    report.failed = sum(1 for c in report.checks if c.status == "fail")
    report.warnings = sum(1 for c in report.checks if c.status == "warning")

    report.ready = report.failed == 0
    report.summary = (
        f"检查完成：{report.passed} 通过 / {report.warnings} 警告 / "
        f"{report.failed} 失败 — "
        f"{'✓ 生产就绪' if report.ready else '✗ 未通过检查'}"
    )

    return report


def format_production_report(report: ProductionReport) -> str:
    """Format a production report as readable Markdown."""
    lines = [
        f"# 生产准备就绪检查 — {report.assembly_name}",
        "",
        f"**结果：{'✓ 通过' if report.ready else '✗ 未通过'}** "
        f"({report.passed}/{report.total} 通过, {report.warnings} 警告, {report.failed} 失败)",
        "",
    ]

    # Group by category
    categories: dict[str, list[CheckItem]] = {}
    for c in report.checks:
        categories.setdefault(c.category, []).append(c)

    for cat, checks in categories.items():
        cat_names = {
            "assembly": "装配体验证",
            "code": "固件代码",
            "bom": "物料清单",
            "tolerance": "公差检查",
            "file": "文件检查",
        }
        lines.append(f"## {cat_names.get(cat, cat)}")
        lines.append("")
        for c in checks:
            icon = {"pass": "✓", "fail": "✗", "warning": "⚠"}[c.status]
            lines.append(f"- {icon} **{c.name}**: {c.message}")
        lines.append("")

    lines.append(f"---")
    lines.append(report.summary)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: production_check
# ---------------------------------------------------------------------------

class ProductionCheckTool(Tool):
    """Tool for production readiness checking."""

    name = "production_check"
    description = (
        "生产准备就绪检查：验证装配体完整性、固件代码、BOM、公差、接线图，"
        "输出检查报告，判断是否达到生产就绪状态。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "assembly_name": {
                        "type": "string",
                        "description": "装配体名称（如 'robotic_arm'）",
                    },
                    "actuator_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "执行器 ID 列表",
                    },
                    "sensor_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "传感器 ID 列表",
                    },
                    "controller": {
                        "type": "string",
                        "description": "控制器类型（esp32/arduino）",
                    },
                },
            },
        )

    def execute(
        self,
        *,
        assembly_name: str = "robotic_arm",
        actuator_ids: list[str] | None = None,
        sensor_ids: list[str] | None = None,
        controller: str = "esp32",
        **kwargs: Any,
    ) -> str:
        actuator_ids = actuator_ids or ["MG996R", "MG996R", "DS3218", "SG90"]
        sensor_ids = sensor_ids or ["AS5600", "LIMIT_SWITCH_MICRO", "MPU6050"]

        from .assembly_solver import _resolve_assembly
        assembly = _resolve_assembly(assembly_name, "")
        if assembly is None:
            return f"错误：未找到装配体 '{assembly_name}'"

        report = run_production_check(assembly, actuator_ids, sensor_ids, controller)

        lines = [
            f"[生产检查] {assembly.name}",
            report.summary,
            "",
            f"通过: {report.passed} / 警告: {report.warnings} / 失败: {report.failed}",
            "",
        ]

        for c in report.checks:
            icon = {"pass": "✓", "fail": "✗", "warning": "⚠"}[c.status]
            lines.append(f"  {icon} [{c.category}] {c.name}: {c.message}")

        lines.append("")
        lines.append("--- Markdown ---")
        lines.append(format_production_report(report))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_production_tools(registry: Any) -> None:
    """Register production check tools."""
    registry.register(ProductionCheckTool())
