"""Assembly guide generation for robotic assemblies.

Generates step-by-step assembly instructions including:
  - Parts and tools checklist
  - Assembly steps (from joint chain order)
  - Wiring instructions (pin mapping table)
  - Calibration procedure
  - Troubleshooting guide

Tools:
  gen_assembly_guide - Generate assembly guide document
"""

from __future__ import annotations

import json
from typing import Any

from ..knowledge.actuators import get_actuator
from ..knowledge.mechanics import Assembly
from ..knowledge.sensors import get_sensor
from ..models.base import ToolDefinition
from .assembly_solver import _resolve_assembly
from .base import Tool
from .code_gen import _assign_pins


# ---------------------------------------------------------------------------
# Assembly guide generation
# ---------------------------------------------------------------------------

def generate_assembly_guide(
    assembly: Assembly,
    actuator_ids: list[str] | None = None,
    sensor_ids: list[str] | None = None,
    controller: str = "esp32",
) -> str:
    """Generate an assembly guide as Markdown.

    Args:
        assembly: The mechanical assembly.
        actuator_ids: Actuator IDs (one per revolute joint).
        sensor_ids: Sensor IDs.
        controller: Controller type.

    Returns a Markdown string.
    """
    lines: list[str] = []

    # Title
    lines.append(f"# 装配指导书 — {assembly.name}")
    lines.append("")
    lines.append(f"> {assembly.description}")
    lines.append("")

    # Section 1: Parts checklist
    lines.append("## 1. 零件清单")
    lines.append("")
    lines.append("### 自定义零件（3D 打印）")
    lines.append("")
    for i, part in enumerate(assembly.parts, 1):
        dims_str = "×".join(f"{v}" for v in part.dimensions.values())
        lines.append(f"{i}. **{part.name}** — {part.description}")
        lines.append(f"   - 材料: {part.material}, 尺寸: {dims_str}")
        if part.notes:
            lines.append(f"   - 备注: {part.notes}")
    lines.append("")

    # Standard parts
    lines.append("### 标准件")
    lines.append("")
    revolute_joints = [j for j in assembly.joints if j.type == "revolute"]
    fixed_joints = [j for j in assembly.joints if j.type == "fixed"]
    joint_count = len(revolute_joints) + len(fixed_joints)
    lines.append(f"- M3×10 螺丝 × {joint_count * 4}")
    lines.append(f"- M3 螺母 × {joint_count * 4}")
    if revolute_joints:
        lines.append(f"- MR105ZZ 轴承 × {len(revolute_joints) * 2}")
    lines.append("")

    # Tools needed
    lines.append("## 2. 所需工具")
    lines.append("")
    tools = [
        ("十字螺丝刀", "M3 螺丝拧紧"),
        ("内六角扳手套装", "M6 螺丝（如需要）"),
        ("烙铁 + 焊锡", "杜邦线焊接（可选）"),
        ("万用表", "电路检查"),
        ("USB 数据线", f"连接 {controller.upper()}"),
    ]
    for name, usage in tools:
        lines.append(f"- {name}（{usage}）")
    lines.append("")

    # Section 3: Assembly steps
    lines.append("## 3. 装配步骤")
    lines.append("")

    step_num = 1
    for j in assembly.joints:
        lines.append(f"### 步骤 {step_num}: 安装 {j.child} → {j.parent}")
        lines.append(f"")
        lines.append(f"- 关节类型: {'旋转关节' if j.type == 'revolute' else '固定连接' if j.type == 'fixed' else '滑动关节'}")
        if j.type == "revolute":
            lines.append(f"- 旋转范围: {j.range_deg[0]}° ~ {j.range_deg[1]}°")
        lines.append(f"- 父件: **{j.parent}**（{j.parent_anchor}面）")
        lines.append(f"- 子件: **{j.child}**（{j.child_anchor}面）")
        lines.append(f"- 说明: {j.description or '将子件安装到父件上'}")
        lines.append(f"")
        lines.append(f"操作:")
        lines.append(f"1. 将 **{j.child}** 的 {j.child_anchor} 面对准 **{j.parent}** 的 {j.parent_anchor} 面")
        lines.append(f"2. 对准安装孔位")
        if j.type == "revolute":
            lines.append(f"3. 安装轴承（MR105ZZ × 2）")
            lines.append(f"4. 穿入轴销，两端用 M3×10 螺丝 + 螺母固定")
            if actuator_ids and step_num - 1 < len(actuator_ids):
                a = get_actuator(actuator_ids[step_num - 1])
                if a:
                    lines.append(f"5. 安装 {a.name} 舵机到关节位置")
        else:
            lines.append(f"3. 用 M3×10 螺丝 + 螺母固定（4 个）")
        lines.append(f"")
        step_num += 1

    # Section 4: Wiring
    if actuator_ids:
        lines.append("## 4. 接线说明")
        lines.append("")
        lines.append(f"控制器: **{controller.upper()}**")
        lines.append("")

        pins = _assign_pins(len(actuator_ids), controller)
        lines.append("| 舵机 | 信号引脚 | VCC | GND |")
        lines.append("|------|---------|-----|-----|")
        for i, aid in enumerate(actuator_ids):
            a = get_actuator(aid)
            name = a.name if a else aid
            pin = pins[i] if i < len(pins) else "?"
            lines.append(f"| {name} | GPIO{pin} | {a.voltage if a else '?'}V | GND |")
        lines.append("")
        lines.append("注意事项:")
        lines.append("- 所有舵机 GND 连接到控制器 GND")
        lines.append("- 舵机使用独立电源，**不要**从 USB 供电")
        lines.append("- 在舵机电源线上并联 100μF 电容")
        lines.append("")

        # Sensor wiring
        if sensor_ids:
            lines.append("### 传感器接线")
            lines.append("")
            for sid in sensor_ids:
                s = get_sensor(sid)
                if s:
                    lines.append(f"**{s.name}**（{s.interface}）")
                    for pin_desc in s.pins:
                        lines.append(f"- {pin_desc}")
                    lines.append("")
            lines.append("")

    # Section 5: Calibration
    lines.append("## 5. 校准步骤")
    lines.append("")
    lines.append("1. 上电前检查所有螺丝是否拧紧")
    lines.append("2. 连接 USB 到电脑，打开串口监视器（波特率 115200）")
    lines.append("3. 观察启动信息：应显示 'Robot Arm Ready'")
    lines.append("4. 发送 `H` 命令归零，观察各关节是否回到零位")
    lines.append("5. 如果零位不对，调整舵机与关节的安装角度")
    lines.append("6. 发送 `P` 命令打印当前角度，确认零位读数")
    if sensor_ids:
        lines.append("7. 发送 `S` 命令检查传感器状态")
    lines.append("")

    # Section 6: Test
    lines.append("## 6. 测试")
    lines.append("")
    lines.append("逐关节测试（参考测试序列）：")
    lines.append("- [ ] 底座旋转正常")
    lines.append("- [ ] 肩部俯仰正常")
    lines.append("- [ ] 肘部弯曲正常")
    lines.append("- [ ] 腕部旋转正常")
    lines.append("- [ ] 联动运动平滑")
    lines.append("- [ ] 限位开关触发正确")
    lines.append("- [ ] 无异常噪音")
    lines.append("- [ ] 无异常发热")
    lines.append("")

    # Section 7: Troubleshooting
    lines.append("## 7. 常见问题排查")
    lines.append("")
    problems = [
        ("舵机不动", "检查接线（信号线/VCC/GND）、检查电源（独立供电）、检查串口命令格式"),
        ("舵机抖动", "降低供电电压或增加电容（470μF）、检查负载是否超力矩"),
        ("运动不平滑", "增加插值时间（T 命令）、检查关节是否有摩擦"),
        ("串口无响应", "检查波特率（115200）、检查 USB 线（需要数据线）、检查驱动安装"),
        ("角度不对", "校准零位、检查舵机安装方向、调整 joint offset"),
    ]
    for problem, solution in problems:
        lines.append(f"**{problem}**: {solution}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class GenAssemblyGuideTool(Tool):
    name = "gen_assembly_guide"
    description = (
        "生成装配指导书：零件清单 + 装配步骤 + 接线说明 + 校准 + 测试 + 问题排查。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "assembly_name": {"type": "string", "description": "装配体名称"},
                "actuator_ids": {
                    "type": "array", "items": {"type": "string"},
                    "description": "执行器 ID 列表",
                },
                "sensor_ids": {
                    "type": "array", "items": {"type": "string"},
                    "description": "传感器 ID 列表",
                },
                "controller": {
                    "type": "string", "enum": ["esp32", "arduino"],
                    "description": "控制器类型",
                },
            }, "required": []},
        )

    def execute(self, *, assembly_name: str = "robotic_arm",
                actuator_ids: list[str] | None = None,
                sensor_ids: list[str] | None = None,
                controller: str = "esp32",
                **kwargs: Any) -> str:
        assembly = _resolve_assembly(assembly_name, "")
        if assembly is None:
            return f"错误：未找到装配体 '{assembly_name}'"

        guide = generate_assembly_guide(assembly, actuator_ids, sensor_ids, controller)
        return f"[Assembly Guide Generated] {assembly.name}\n\n{guide}"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_assembly_doc_tools(registry: Any) -> None:
    """Register assembly documentation tools."""
    registry.register(GenAssemblyGuideTool())
