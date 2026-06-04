"""Quality control for robotic assemblies.

Provides:
  - Dimensional inspection checklist (critical fit dimensions)
  - Post-assembly test procedure
  - Long-term maintenance guide

Tools:
  quality_check - Generate quality control checklist for assembly
"""

from __future__ import annotations

import json
from typing import Any

from ..knowledge.mechanics import Assembly, Part, PRINT_TOLERANCES
from ..models.base import ToolDefinition
from .assembly_solver import _resolve_assembly
from .base import Tool


# ---------------------------------------------------------------------------
# Dimensional inspection
# ---------------------------------------------------------------------------

def generate_inspection_checklist(
    assembly: Assembly,
    tolerance_mm: float = 0.2,
) -> dict[str, Any]:
    """Generate a dimensional inspection checklist for assembly parts.

    Args:
        assembly: The assembly to inspect.
        tolerance_mm: Default tolerance for critical dimensions.

    Returns structured inspection checklist.
    """
    checklist: dict[str, Any] = {
        "assembly_name": assembly.name,
        "default_tolerance_mm": tolerance_mm,
        "parts": [],
    }

    for part in assembly.parts:
        dims = part.dimensions
        notes = part.notes.lower()
        checks: list[dict[str, Any]] = []

        for dim_name, nominal_value in dims.items():
            # Determine criticality
            critical = False
            dim_tolerance = tolerance_mm

            if any(kw in dim_name.lower() for kw in ["diameter", "shaft", "hole", "bore"]):
                critical = True
                dim_tolerance = PRINT_TOLERANCES.get("bearing_fit", 0.1)
            elif any(kw in dim_name.lower() for kw in ["thickness", "width", "height"]):
                dim_tolerance = tolerance_mm
            elif any(kw in dim_name.lower() for kw in ["length"]):
                dim_tolerance = tolerance_mm * 2  # Length less critical

            # Shaft/hole dimensions are most critical
            if "shaft" in dim_name or "inner" in dim_name:
                dim_tolerance = PRINT_TOLERANCES.get("bearing_fit", 0.1)
                critical = True

            checks.append({
                "dimension": dim_name,
                "nominal_mm": nominal_value,
                "tolerance_mm": round(dim_tolerance, 2),
                "min_mm": round(nominal_value - dim_tolerance, 2),
                "max_mm": round(nominal_value + dim_tolerance, 2),
                "critical": critical,
            })

        checklist["parts"].append({
            "part_name": part.name,
            "category": part.category,
            "checks": checks,
            "notes": part.notes,
        })

    # Summary
    total_checks = sum(len(p["checks"]) for p in checklist["parts"])
    critical_checks = sum(
        1 for p in checklist["parts"] for c in p["checks"] if c["critical"]
    )
    checklist["summary"] = {
        "total_dimensions": total_checks,
        "critical_dimensions": critical_checks,
        "non_critical_dimensions": total_checks - critical_checks,
    }

    return checklist


# ---------------------------------------------------------------------------
# Post-assembly test procedure
# ---------------------------------------------------------------------------

def generate_test_procedure(assembly: Assembly) -> dict[str, Any]:
    """Generate post-assembly test procedure.

    Returns structured test procedure with phases.
    """
    revolute_joints = [j for j in assembly.joints if j.type == "revolute"]
    fixed_joints = [j for j in assembly.joints if j.type == "fixed"]

    procedure: dict[str, Any] = {
        "assembly_name": assembly.name,
        "phases": [],
    }

    # Phase 1: Visual inspection
    procedure["phases"].append({
        "name": "外观检查",
        "order": 1,
        "steps": [
            {"action": "检查所有零件是否有打印缺陷（翘曲、层裂、气泡）", "pass_fail": True},
            {"action": "检查螺丝是否全部拧紧", "pass_fail": True},
            {"action": "检查关节是否有异常间隙", "pass_fail": True},
            {"action": "检查线缆走线是否正确、无干涉", "pass_fail": True},
        ],
    })

    # Phase 2: Mechanical range test
    mech_steps = []
    for j in revolute_joints:
        mech_steps.append({
            "action": f"手动旋转 {j.description or j.child}（范围 {j.range_deg[0]}° ~ {j.range_deg[1]}°）",
            "expected": f"平滑旋转，无卡顿，角度覆盖完整",
            "pass_fail": True,
        })
    for j in fixed_joints:
        mech_steps.append({
            "action": f"检查 {j.child} 与 {j.parent} 的固定连接",
            "expected": "无松动，无间隙",
            "pass_fail": True,
        })

    procedure["phases"].append({
        "name": "机械范围测试",
        "order": 2,
        "steps": mech_steps,
    })

    # Phase 3: Electrical test
    procedure["phases"].append({
        "name": "电气测试",
        "order": 3,
        "steps": [
            {"action": "上电前用万用表检查电源对地是否短路", "pass_fail": True},
            {"action": "连接电源，检查各舵机供电电压", "expected": "电压在规格范围内", "pass_fail": True},
            {"action": "连接 USB 串口，检查通信", "expected": "显示 'Robot Arm Ready'", "pass_fail": True},
            {"action": "发送归零命令 H，检查各关节回零", "expected": "各关节回到零位", "pass_fail": True},
        ],
    })

    # Phase 4: Functional test
    func_steps = [
        {"action": "逐关节发送运动命令，验证响应", "expected": "每个关节独立运动正常", "pass_fail": True},
        {"action": "测试联动运动（G 命令）", "expected": "联动平滑，无抖动", "pass_fail": True},
        {"action": "测试极限位置（最大角度）", "expected": "不超出机械限位", "pass_fail": True},
    ]
    procedure["phases"].append({
        "name": "功能测试",
        "order": 4,
        "steps": func_steps,
    })

    # Phase 5: Endurance test
    procedure["phases"].append({
        "name": "耐久测试",
        "order": 5,
        "steps": [
            {"action": "连续运行 30 分钟", "expected": "无异常发热（舵机 < 60°C）", "pass_fail": True},
            {"action": "重复归零-运动循环 50 次", "expected": "精度不降低", "pass_fail": True},
            {"action": "检查螺丝是否松动", "pass_fail": True},
        ],
    })

    return procedure


# ---------------------------------------------------------------------------
# Maintenance guide
# ---------------------------------------------------------------------------

def generate_maintenance_guide(assembly: Assembly) -> dict[str, Any]:
    """Generate long-term maintenance guide.

    Returns structured maintenance schedule.
    """
    guide: dict[str, Any] = {
        "assembly_name": assembly.name,
        "schedules": [],
    }

    # Weekly
    guide["schedules"].append({
        "interval": "每周",
        "tasks": [
            {"task": "检查螺丝紧固状态", "action": "用手感受各关节螺丝，如有松动用螺丝刀拧紧"},
            {"task": "检查关节运动平滑性", "action": "手动旋转各关节，感受是否有异常阻力"},
            {"task": "清洁关节表面", "action": "用干布擦拭关节和导轨表面灰尘"},
        ],
    })

    # Monthly
    guide["schedules"].append({
        "interval": "每月",
        "tasks": [
            {"task": "润滑关节轴承", "action": "在各轴承处滴入少量润滑油（PTFE 或锂基脂）"},
            {"task": "检查线缆磨损", "action": "检查关节处线缆是否有弯折或破损"},
            {"task": "校准零位", "action": "发送 H 命令归零，检查零位是否偏移"},
        ],
    })

    # Quarterly
    guide["schedules"].append({
        "interval": "每季度",
        "tasks": [
            {"task": "全面校准", "action": "使用串口命令 P 检查各关节角度读数，与实际对比"},
            {"task": "更换磨损零件", "action": "检查轴承和关节销是否有明显磨损"},
            {"task": "检查电气连接", "action": "检查杜邦线接触是否良好，焊点是否脱焊"},
        ],
    })

    # Yearly
    guide["schedules"].append({
        "interval": "每年",
        "tasks": [
            {"task": "深度清洁和检查", "action": "拆解、清洁、检查所有零件，更换磨损件"},
            {"task": "更换舵机", "action": "检查舵机齿轮和电位器磨损，必要时更换"},
            {"task": "更新固件", "action": "检查是否有新版本固件，必要时更新"},
        ],
    })

    # Common issues
    guide["common_issues"] = [
        {
            "symptom": "关节运动有异响",
            "cause": "轴承缺油或螺丝过紧",
            "fix": "润滑轴承或适当松开螺丝",
        },
        {
            "symptom": "精度逐渐下降",
            "cause": "舵机齿轮磨损或螺丝松动",
            "fix": "更换舵机或重新紧固螺丝",
        },
        {
            "symptom": "舵机过热",
            "cause": "负载过重或运动过于频繁",
            "fix": "降低运动频率或更换更大扭矩舵机",
        },
        {
            "symptom": "串口通信不稳定",
            "cause": "杜邦线接触不良或线缆过长",
            "fix": "更换杜邦线或缩短线缆长度",
        },
    ]

    return guide


# ---------------------------------------------------------------------------
# Full quality report
# ---------------------------------------------------------------------------

def generate_quality_report(assembly: Assembly) -> dict[str, Any]:
    """Generate complete quality control report."""
    return {
        "inspection": generate_inspection_checklist(assembly),
        "test_procedure": generate_test_procedure(assembly),
        "maintenance": generate_maintenance_guide(assembly),
    }


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class QualityCheckTool(Tool):
    name = "quality_check"
    description = (
        "生成质量控制报告：尺寸检查清单 + 装配后测试流程 + 长期维护指南。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "assembly_name": {"type": "string", "description": "装配体名称"},
                "tolerance_mm": {"type": "number", "description": "默认公差（mm，默认 0.2）"},
                "section": {
                    "type": "string",
                    "enum": ["all", "inspection", "test", "maintenance"],
                    "description": "输出内容：all/inspection/test/maintenance",
                },
            }, "required": []},
        )

    def execute(self, *, assembly_name: str = "robotic_arm",
                tolerance_mm: float = 0.2, section: str = "all",
                **kwargs: Any) -> str:
        assembly = _resolve_assembly(assembly_name, "")
        if assembly is None:
            return f"错误：未找到装配体 '{assembly_name}'"

        report = generate_quality_report(assembly)

        lines = [f"[Quality Report] {assembly.name}", ""]

        if section in ("all", "inspection"):
            insp = report["inspection"]
            lines.append("## 尺寸检查清单")
            lines.append("")
            for p in insp["parts"]:
                lines.append(f"### {p['part_name']} ({p['category']})")
                for c in p["checks"]:
                    crit = "⚠️" if c["critical"] else " "
                    lines.append(
                        f"  {crit} {c['dimension']}: {c['nominal_mm']}mm "
                        f"±{c['tolerance_mm']}mm [{c['min_mm']}, {c['max_mm']}]"
                    )
                lines.append("")

        if section in ("all", "test"):
            test = report["test_procedure"]
            lines.append("## 装配后测试流程")
            lines.append("")
            for phase in test["phases"]:
                lines.append(f"### Phase {phase['order']}: {phase['name']}")
                for i, step in enumerate(phase["steps"], 1):
                    lines.append(f"  {i}. [ ] {step['action']}")
                lines.append("")

        if section in ("all", "maintenance"):
            maint = report["maintenance"]
            lines.append("## 维护指南")
            lines.append("")
            for sched in maint["schedules"]:
                lines.append(f"### {sched['interval']}")
                for task in sched["tasks"]:
                    lines.append(f"  - {task['task']}: {task['action']}")
                lines.append("")
            lines.append("### 常见问题")
            for issue in maint["common_issues"]:
                lines.append(f"  **{issue['symptom']}**: {issue['fix']}")

        lines.append("")
        lines.append("--- JSON ---")
        if section == "all":
            lines.append(json.dumps(report, ensure_ascii=False, indent=2))
        elif section == "inspection":
            lines.append(json.dumps(report["inspection"], ensure_ascii=False, indent=2))
        elif section == "test":
            lines.append(json.dumps(report["test_procedure"], ensure_ascii=False, indent=2))
        elif section == "maintenance":
            lines.append(json.dumps(report["maintenance"], ensure_ascii=False, indent=2))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_quality_tools(registry: Any) -> None:
    """Register quality control tools."""
    registry.register(QualityCheckTool())
