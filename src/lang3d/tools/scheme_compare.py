"""Multi-robot scheme comparison tool.

Generates 2-3 robot configurations for given requirements, compares them
across workspace, precision, cost, and complexity dimensions, and
recommends the best scheme.

Supported configurations:
  - Serial arm (串联臂) — articulated, 3-6 DOF
  - SCARA — selective compliance, fast horizontal movement
  - Delta / Parallel (并联臂) — high speed, high precision

Usage:
  from lang3d.tools.scheme_compare import compare_schemes, recommend_scheme
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..knowledge.actuators import ACTUATORS
from ..knowledge.mechanics import Assembly, Joint, Part
from ..models.base import ToolDefinition
from .base import Tool
from .bom_gen import generate_bom


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RobotRequirement:
    """User requirements for a robot design."""
    reach_mm: float = 300          # Working radius
    payload_g: float = 200         # Max payload
    precision_mm: float = 1.0      # Required precision (lower = better)
    budget_cny: float = 1000       # Max budget
    prefer_speed: bool = False     # Prefer speed over precision
    prefer_precision: bool = False # Prefer precision over speed
    prefer_cost: bool = False      # Minimize cost


@dataclass
class SchemeMetrics:
    """Quantitative metrics for a robot scheme."""
    workspace_radius_mm: float     # Max horizontal reach
    workspace_height_mm: float     # Vertical range
    workspace_volume_dm3: float    # Approximate workspace volume
    precision_mm: float            # Estimated positioning accuracy
    max_speed_mm_s: float          # Estimated max tip speed
    payload_capacity_g: float      # Max recommended payload
    part_count: int                # Number of custom parts
    joint_count: int               # Number of actuated joints
    estimated_cost_cny: float      # Total estimated cost
    complexity_score: float        # 1-10 (10 = most complex)
    assembly_time_min: float       # Estimated assembly time
    weight_g: float                # Total estimated weight


@dataclass
class Scheme:
    """A complete robot scheme with assembly and metrics."""
    name: str
    description: str
    config_type: str               # "serial", "scara", "delta"
    assembly: Assembly
    actuator_ids: list[str]
    sensor_ids: list[str]
    controller: str
    metrics: SchemeMetrics
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    score: float = 0.0             # Overall score (computed)


# ---------------------------------------------------------------------------
# Scheme generators
# ---------------------------------------------------------------------------

def _generate_serial_arm(req: RobotRequirement) -> Scheme:
    """Generate a 3-DOF serial articulated arm scheme."""
    reach = req.reach_mm
    # Split reach into 3 links (base_height + shoulder + forearm)
    base_h = round(reach * 0.15, 1)
    L1 = round(reach * 0.35, 1)    # Shoulder link
    L2 = round(reach * 0.35, 1)    # Forearm
    wrist_len = round(reach * 0.15, 1)

    parts = [
        Part("base_plate", "structural", "底座圆盘",
             dimensions={"diameter": base_h * 2.5, "thickness": 5}),
        Part("base_joint_housing", "structural", "基座关节壳体",
             dimensions={"diameter": base_h, "height": base_h}),
        Part("shoulder_link", "link", "肩部连杆",
             dimensions={"length": L1, "width": 15, "height": 12}),
        Part("elbow_joint", "structural", "肘部关节",
             dimensions={"diameter": 12, "height": 14}),
        Part("forearm_link", "link", "前臂连杆",
             dimensions={"length": L2, "width": 12, "height": 10}),
        Part("wrist_joint", "structural", "腕部关节",
             dimensions={"diameter": 10, "height": 10}),
        Part("end_effector_mount", "structural", "末端安装座",
             dimensions={"length": wrist_len, "width": 10, "height": 8}),
        Part("servo_holder", "structural", "舵机支架",
             dimensions={"length": 20, "width": 15, "height": 10}),
    ]

    joints = [
        Joint("revolute", "base_plate", "base_joint_housing",
              range_deg=(-180, 180), description="基座旋转",
              parent_anchor="top", child_anchor="bottom", axis="z"),
        Joint("revolute", "base_joint_housing", "shoulder_link",
              range_deg=(-90, 90), description="肩部俯仰",
              parent_anchor="top", child_anchor="bottom", axis="y"),
        Joint("revolute", "shoulder_link", "elbow_joint",
              range_deg=(-135, 135), description="肘部弯曲",
              parent_anchor="top", child_anchor="bottom", axis="y"),
        Joint("fixed", "elbow_joint", "forearm_link",
              parent_anchor="top", child_anchor="bottom"),
        Joint("revolute", "forearm_link", "wrist_joint",
              range_deg=(-180, 180), description="腕部旋转",
              parent_anchor="top", child_anchor="bottom", axis="z"),
        Joint("fixed", "wrist_joint", "end_effector_mount",
              parent_anchor="top", child_anchor="bottom"),
    ]

    assembly = Assembly(
        name=f"Serial-{int(reach)}mm",
        parts=parts,
        joints=joints,
        description=f"3-DOF 串联机械臂，工作半径 {int(reach)}mm",
    )

    # Select actuators based on payload
    if req.payload_g > 500:
        actuator_ids = ["DS3218", "DS3218", "DS3218", "MG996R"]
    elif req.payload_g > 200:
        actuator_ids = ["MG996R", "MG996R", "DS3218", "SG90"]
    else:
        actuator_ids = ["MG90S", "MG90S", "MG996R", "SG90"]

    sensor_ids = ["AS5600", "LIMIT_SWITCH_MICRO", "MPU6050"]

    # Compute metrics
    ws_radius = round(L1 + L2, 1)
    ws_height = round(base_h + L1 + L2 + wrist_len, 1)
    ws_volume = round(math.pi * ws_radius**2 * ws_height / 1e6, 2)  # dm³

    # Cost estimation
    actuator_cost = sum(ACTUATORS.get(a, ACTUATORS["SG90"]).price_cny for a in actuator_ids)
    part_cost = len(parts) * 3.0  # ~3 CNY per 3D printed part
    electronics = 35 + 15 + 10  # Controller + sensors + wires
    total_cost = round(actuator_cost + part_cost + electronics, 1)

    total_weight = sum(ACTUATORS.get(a, ACTUATORS["SG90"]).weight_g for a in actuator_ids)
    total_weight += len(parts) * 8  # ~8g per PLA part

    metrics = SchemeMetrics(
        workspace_radius_mm=ws_radius,
        workspace_height_mm=ws_height,
        workspace_volume_dm3=ws_volume,
        precision_mm=0.5,
        max_speed_mm_s=200,
        payload_capacity_g=req.payload_g,
        part_count=len(parts),
        joint_count=4,
        estimated_cost_cny=total_cost,
        complexity_score=4.0,
        assembly_time_min=90,
        weight_g=total_weight,
    )

    return Scheme(
        name="串联机械臂",
        description=f"3-DOF 串联关节臂，工作半径 {ws_radius:.0f}mm",
        config_type="serial",
        assembly=assembly,
        actuator_ids=actuator_ids,
        sensor_ids=sensor_ids,
        controller="esp32",
        metrics=metrics,
        pros=[
            "工作空间大，覆盖球形体",
            "结构简单，易于理解",
            "零件少，成本低",
            "逆向运动学成熟",
        ],
        cons=[
            "刚度较低，精度受限",
            "末端误差累积",
            "高速运动时抖动",
        ],
    )


def _generate_scara(req: RobotRequirement) -> Scheme:
    """Generate a SCARA robot scheme."""
    reach = req.reach_mm
    base_h = round(reach * 0.2, 1)
    L1 = round(reach * 0.4, 1)     # First horizontal link
    L2 = round(reach * 0.35, 1)    # Second horizontal link
    z_travel = round(reach * 0.3, 1)  # Vertical travel

    parts = [
        Part("base_column", "structural", "立柱",
             dimensions={"length": 30, "width": 30, "height": base_h}),
        Part("shoulder_joint", "structural", "肩部关节",
             dimensions={"diameter": 20, "height": 15}),
        Part("upper_arm", "link", "上臂",
             dimensions={"length": L1, "width": 20, "height": 12}),
        Part("elbow_joint", "structural", "肘部关节",
             dimensions={"diameter": 16, "height": 12}),
        Part("forearm", "link", "前臂",
             dimensions={"length": L2, "width": 16, "height": 10}),
        Part("z_axis_carriage", "structural", "Z轴滑台",
             dimensions={"length": 20, "width": 20, "height": z_travel}),
        Part("end_effector", "structural", "末端执行器",
             dimensions={"diameter": 15, "height": 20}),
        Part("base_plate", "structural", "底座",
             dimensions={"length": 40, "width": 40, "thickness": 5}),
    ]

    joints = [
        Joint("revolute", "base_plate", "base_column",
              range_deg=(-180, 180), description="基座旋转",
              parent_anchor="top", child_anchor="bottom", axis="z"),
        Joint("revolute", "base_column", "upper_arm",
              range_deg=(-135, 135), description="肩部水平旋转",
              parent_anchor="top", child_anchor="bottom", axis="z"),
        Joint("revolute", "upper_arm", "forearm",
              range_deg=(-135, 135), description="肘部水平旋转",
              parent_anchor="top", child_anchor="bottom", axis="z"),
        Joint("prismatic", "forearm", "z_axis_carriage",
              range_deg=(0, z_travel), description="Z轴直线运动",
              parent_anchor="top", child_anchor="bottom", axis="z"),
        Joint("revolute", "z_axis_carriage", "end_effector",
              range_deg=(-180, 180), description="末端旋转",
              parent_anchor="top", child_anchor="bottom", axis="z"),
    ]

    assembly = Assembly(
        name=f"SCARA-{int(reach)}mm",
        parts=parts,
        joints=joints,
        description=f"SCARA 机器人，水平工作半径 {int(reach)}mm",
    )

    actuator_ids = ["MG996R", "MG996R", "MG996R", "SG90"]
    sensor_ids = ["AS5600", "LIMIT_SWITCH_MICRO"]

    ws_radius = round(L1 + L2, 1)
    ws_height = round(z_travel, 1)
    ws_volume = round(math.pi * ws_radius**2 * ws_height / 1e6, 2)

    actuator_cost = sum(ACTUATORS.get(a, ACTUATORS["SG90"]).price_cny for a in actuator_ids)
    part_cost = len(parts) * 3.5
    electronics = 35 + 10 + 8
    total_cost = round(actuator_cost + part_cost + electronics, 1)

    total_weight = sum(ACTUATORS.get(a, ACTUATORS["SG90"]).weight_g for a in actuator_ids)
    total_weight += len(parts) * 10

    metrics = SchemeMetrics(
        workspace_radius_mm=ws_radius,
        workspace_height_mm=ws_height,
        workspace_volume_dm3=ws_volume,
        precision_mm=0.2,
        max_speed_mm_s=500,
        payload_capacity_g=req.payload_g * 0.8,
        part_count=len(parts),
        joint_count=4,
        estimated_cost_cny=total_cost,
        complexity_score=5.0,
        assembly_time_min=120,
        weight_g=total_weight,
    )

    return Scheme(
        name="SCARA 机器人",
        description=f"SCARA 机器人，水平半径 {ws_radius:.0f}mm",
        config_type="scara",
        assembly=assembly,
        actuator_ids=actuator_ids,
        sensor_ids=sensor_ids,
        controller="esp32",
        metrics=metrics,
        pros=[
            "水平方向速度快",
            "垂直方向刚度高",
            "重复定位精度高",
            "适合取放操作",
        ],
        cons=[
            "工作空间仅限于圆柱体",
            "Z轴行程有限",
            "不适用于多角度操作",
            "结构比串联臂复杂",
        ],
    )


def _generate_delta(req: RobotRequirement) -> Scheme:
    """Generate a Delta/parallel robot scheme."""
    reach = req.reach_mm
    base_radius = round(reach * 0.3, 1)
    arm_length = round(reach * 0.45, 1)
    rod_length = round(reach * 0.45, 1)
    platform_radius = round(reach * 0.08, 1)

    parts = [
        Part("top_platform", "structural", "上平台（固定）",
             dimensions={"diameter": base_radius * 2, "thickness": 5}),
        Part("arm_1", "link", "主动臂 1",
             dimensions={"length": arm_length, "width": 12, "height": 8}),
        Part("arm_2", "link", "主动臂 2",
             dimensions={"length": arm_length, "width": 12, "height": 8}),
        Part("arm_3", "link", "主动臂 3",
             dimensions={"length": arm_length, "width": 12, "height": 8}),
        Part("rod_1a", "link", "从动杆 1a",
             dimensions={"length": rod_length, "diameter": 3}),
        Part("rod_1b", "link", "从动杆 1b",
             dimensions={"length": rod_length, "diameter": 3}),
        Part("rod_2a", "link", "从动杆 2a",
             dimensions={"length": rod_length, "diameter": 3}),
        Part("rod_2b", "link", "从动杆 2b",
             dimensions={"length": rod_length, "diameter": 3}),
        Part("rod_3a", "link", "从动杆 3a",
             dimensions={"length": rod_length, "diameter": 3}),
        Part("rod_3b", "link", "从动杆 3b",
             dimensions={"length": rod_length, "diameter": 3}),
        Part("moving_platform", "structural", "运动平台",
             dimensions={"diameter": platform_radius * 2, "thickness": 4}),
        Part("base_frame", "structural", "底座框架",
             dimensions={"length": base_radius * 3, "width": base_radius * 3, "height": 8}),
    ]

    joints = [
        Joint("revolute", "top_platform", "arm_1",
              range_deg=(-60, 60), description="电机 1",
              parent_anchor="top", child_anchor="bottom", axis="y"),
        Joint("revolute", "top_platform", "arm_2",
              range_deg=(-60, 60), description="电机 2",
              parent_anchor="top", child_anchor="bottom", axis="y"),
        Joint("revolute", "top_platform", "arm_3",
              range_deg=(-60, 60), description="电机 3",
              parent_anchor="top", child_anchor="bottom", axis="y"),
        Joint("fixed", "arm_1", "rod_1a",
              parent_anchor="top", child_anchor="bottom"),
        Joint("fixed", "arm_1", "rod_1b",
              parent_anchor="top", child_anchor="bottom"),
        Joint("fixed", "arm_2", "rod_2a",
              parent_anchor="top", child_anchor="bottom"),
        Joint("fixed", "arm_2", "rod_2b",
              parent_anchor="top", child_anchor="bottom"),
        Joint("fixed", "arm_3", "rod_3a",
              parent_anchor="top", child_anchor="bottom"),
        Joint("fixed", "arm_3", "rod_3b",
              parent_anchor="top", child_anchor="bottom"),
    ]

    assembly = Assembly(
        name=f"Delta-{int(reach)}mm",
        parts=parts,
        joints=joints,
        description=f"Delta 并联机器人，工作直径 {int(reach)}mm",
    )

    actuator_ids = ["MG996R", "MG996R", "MG996R"]
    sensor_ids = ["AS5600", "LIMIT_SWITCH_MICRO"]

    # Delta workspace is roughly a cylinder
    ws_radius = round(reach * 0.4, 1)
    ws_height = round(reach * 0.3, 1)
    ws_volume = round(math.pi * ws_radius**2 * ws_height / 1e6, 2)

    actuator_cost = sum(ACTUATORS.get(a, ACTUATORS["SG90"]).price_cny for a in actuator_ids)
    part_cost = len(parts) * 3.0
    electronics = 35 + 10 + 10
    total_cost = round(actuator_cost + part_cost + electronics, 1)

    total_weight = sum(ACTUATORS.get(a, ACTUATORS["SG90"]).weight_g for a in actuator_ids)
    total_weight += len(parts) * 6

    metrics = SchemeMetrics(
        workspace_radius_mm=ws_radius,
        workspace_height_mm=ws_height,
        workspace_volume_dm3=ws_volume,
        precision_mm=0.1,
        max_speed_mm_s=1000,
        payload_capacity_g=req.payload_g * 0.5,
        part_count=len(parts),
        joint_count=3,
        estimated_cost_cny=total_cost,
        complexity_score=7.5,
        assembly_time_min=180,
        weight_g=total_weight,
    )

    return Scheme(
        name="Delta 并联机器人",
        description=f"Delta 并联机器人，工作直径 {ws_radius * 2:.0f}mm",
        config_type="delta",
        assembly=assembly,
        actuator_ids=actuator_ids,
        sensor_ids=sensor_ids,
        controller="esp32",
        metrics=metrics,
        pros=[
            "运动速度极快",
            "定位精度最高",
            "结构刚度高",
            "运动学解耦",
        ],
        cons=[
            "工作空间相对较小",
            "负载能力较低",
            "结构复杂，零件多",
            "装配调校难度大",
        ],
    )


# ---------------------------------------------------------------------------
# Scheme generators registry
# ---------------------------------------------------------------------------

SCHEME_GENERATORS = {
    "serial": _generate_serial_arm,
    "scara": _generate_scara,
    "delta": _generate_delta,
}


# ---------------------------------------------------------------------------
# Comparison and scoring
# ---------------------------------------------------------------------------

def generate_schemes(
    req: RobotRequirement,
    types: list[str] | None = None,
) -> list[Scheme]:
    """Generate robot schemes for comparison.

    Args:
        req: Robot requirements.
        types: List of scheme types to generate. Default: all three.

    Returns:
        List of Scheme objects with metrics.
    """
    if types is None:
        types = ["serial", "scara", "delta"]

    schemes = []
    for t in types:
        gen = SCHEME_GENERATORS.get(t)
        if gen:
            schemes.append(gen(req))

    return schemes


def score_schemes(
    schemes: list[Scheme],
    req: RobotRequirement,
) -> list[Scheme]:
    """Score and rank schemes based on requirements.

    Scoring criteria (each 0-10):
      - Workspace fit: how well workspace meets reach requirement
      - Precision: meets precision requirement
      - Speed: max tip speed
      - Payload: meets payload requirement
      - Cost: within budget
      - Complexity: lower is better

    Returns schemes sorted by score (best first).
    """
    for scheme in schemes:
        m = scheme.metrics
        scores = {}

        # Workspace fit (0-10): how much of required reach is covered
        ratio = m.workspace_radius_mm / max(req.reach_mm, 1)
        scores["workspace"] = min(10, round(ratio * 10, 1))

        # Precision (0-10): lower precision_mm is better
        if m.precision_mm <= req.precision_mm:
            scores["precision"] = 10
        else:
            scores["precision"] = max(0, round(10 - (m.precision_mm - req.precision_mm) * 5, 1))

        # Speed (0-10)
        scores["speed"] = min(10, round(m.max_speed_mm_s / 100, 1))

        # Payload (0-10)
        if m.payload_capacity_g >= req.payload_g:
            scores["payload"] = 10
        else:
            ratio_p = m.payload_capacity_g / max(req.payload_g, 1)
            scores["payload"] = round(ratio_p * 10, 1)

        # Cost (0-10): within budget is good
        if m.estimated_cost_cny <= req.budget_cny:
            scores["cost"] = 10
        else:
            over = (m.estimated_cost_cny - req.budget_cny) / max(req.budget_cny, 1)
            scores["cost"] = max(0, round(10 - over * 10, 1))

        # Complexity (0-10): lower is better
        scores["complexity"] = max(0, round(10 - m.complexity_score, 1))

        # Weighted total
        weights = {
            "workspace": 2.0,
            "precision": 1.5,
            "speed": 1.0,
            "payload": 2.0,
            "cost": 1.5,
            "complexity": 1.0,
        }

        # Adjust weights based on preferences
        if req.prefer_speed:
            weights["speed"] = 4.0
        if req.prefer_precision:
            weights["precision"] = 5.0
        if req.prefer_cost:
            weights["cost"] = 4.0

        total_weight = sum(weights.values())
        weighted_sum = sum(scores[k] * weights[k] for k in scores)
        scheme.score = round(weighted_sum / total_weight, 2)

    # Sort by score descending
    schemes.sort(key=lambda s: s.score, reverse=True)
    return schemes


def compare_schemes(
    req: RobotRequirement,
    types: list[str] | None = None,
) -> list[Scheme]:
    """Generate and compare robot schemes.

    Returns schemes sorted by score (best first).
    """
    schemes = generate_schemes(req, types)
    return score_schemes(schemes, req)


def recommend_scheme(
    req: RobotRequirement,
    types: list[str] | None = None,
) -> Scheme:
    """Get the recommended scheme for given requirements."""
    schemes = compare_schemes(req, types)
    if not schemes:
        raise ValueError("No schemes generated")
    return schemes[0]


def format_comparison_table(schemes: list[Scheme]) -> str:
    """Format schemes as a Markdown comparison table."""
    lines = [
        "# 机器人方案对比",
        "",
        "| 指标 | " + " | ".join(s.name for s in schemes) + " |",
        "|------|" + "|".join(["------"] * len(schemes)) + "|",
    ]

    # Metrics rows
    rows = [
        ("工作半径 (mm)", lambda s: f"{s.metrics.workspace_radius_mm:.0f}"),
        ("工作高度 (mm)", lambda s: f"{s.metrics.workspace_height_mm:.0f}"),
        ("工作空间 (dm³)", lambda s: f"{s.metrics.workspace_volume_dm3:.1f}"),
        ("定位精度 (mm)", lambda s: f"{s.metrics.precision_mm:.1f}"),
        ("最大速度 (mm/s)", lambda s: f"{s.metrics.max_speed_mm_s:.0f}"),
        ("负载能力 (g)", lambda s: f"{s.metrics.payload_capacity_g:.0f}"),
        ("零件数量", lambda s: f"{s.metrics.part_count}"),
        ("关节数量", lambda s: f"{s.metrics.joint_count}"),
        ("预估成本 (¥)", lambda s: f"{s.metrics.estimated_cost_cny:.0f}"),
        ("复杂度 (1-10)", lambda s: f"{s.metrics.complexity_score:.1f}"),
        ("装配时间 (min)", lambda s: f"{s.metrics.assembly_time_min:.0f}"),
        ("重量 (g)", lambda s: f"{s.metrics.weight_g:.0f}"),
        ("**综合评分**", lambda s: f"**{s.score:.1f}**"),
    ]

    for label, getter in rows:
        values = " | ".join(getter(s) for s in schemes)
        lines.append(f"| {label} | {values} |")

    # Pros/cons for each
    for s in schemes:
        lines.append("")
        lines.append(f"## {s.name}（评分 {s.score:.1f}）")
        lines.append(f"{s.description}")
        lines.append("")
        lines.append("**优势：**")
        for p in s.pros:
            lines.append(f"- {p}")
        lines.append("")
        lines.append("**劣势：**")
        for c in s.cons:
            lines.append(f"- {c}")

    # Recommendation
    if schemes:
        lines.append("")
        lines.append(f"## 推荐方案：{schemes[0].name}")
        lines.append(f"综合评分最高 {schemes[0].score:.1f} 分")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: scheme_compare
# ---------------------------------------------------------------------------

class SchemeCompareTool(Tool):
    """Tool for comparing robot design schemes."""

    name = "scheme_compare"
    description = (
        "多机器人方案对比：给定需求，生成串联臂/SCARA/Delta 三种方案，"
        "对比工作空间、精度、速度、成本、复杂度，推荐最优方案。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "reach_mm": {
                        "type": "number",
                        "description": "工作半径（毫米），默认 300",
                    },
                    "payload_g": {
                        "type": "number",
                        "description": "负载重量（克），默认 200",
                    },
                    "precision_mm": {
                        "type": "number",
                        "description": "精度要求（毫米），默认 1.0",
                    },
                    "budget_cny": {
                        "type": "number",
                        "description": "预算（人民币），默认 1000",
                    },
                    "prefer_speed": {
                        "type": "boolean",
                        "description": "优先考虑速度",
                    },
                    "prefer_precision": {
                        "type": "boolean",
                        "description": "优先考虑精度",
                    },
                    "prefer_cost": {
                        "type": "boolean",
                        "description": "优先考虑成本",
                    },
                },
            },
        )

    def execute(
        self,
        *,
        reach_mm: float = 300,
        payload_g: float = 200,
        precision_mm: float = 1.0,
        budget_cny: float = 1000,
        prefer_speed: bool = False,
        prefer_precision: bool = False,
        prefer_cost: bool = False,
        **kwargs: Any,
    ) -> str:
        req = RobotRequirement(
            reach_mm=reach_mm,
            payload_g=payload_g,
            precision_mm=precision_mm,
            budget_cny=budget_cny,
            prefer_speed=prefer_speed,
            prefer_precision=prefer_precision,
            prefer_cost=prefer_cost,
        )

        schemes = compare_schemes(req)

        lines = [
            f"[方案对比] 需求：半径 {reach_mm}mm，负载 {payload_g}g，精度 {precision_mm}mm",
            "",
        ]

        for i, s in enumerate(schemes):
            marker = "★ 推荐" if i == 0 else f"  方案 {i + 1}"
            lines.append(f"{marker}: {s.name}（评分 {s.score:.1f}）")
            lines.append(f"  工作半径: {s.metrics.workspace_radius_mm:.0f}mm")
            lines.append(f"  精度: {s.metrics.precision_mm:.1f}mm")
            lines.append(f"  速度: {s.metrics.max_speed_mm_s:.0f}mm/s")
            lines.append(f"  负载: {s.metrics.payload_capacity_g:.0f}g")
            lines.append(f"  成本: ¥{s.metrics.estimated_cost_cny:.0f}")
            lines.append(f"  零件: {s.metrics.part_count}个")
            lines.append("")

        lines.append("--- Markdown ---")
        lines.append(format_comparison_table(schemes))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_scheme_tools(registry: Any) -> None:
    """Register scheme comparison tools."""
    registry.register(SchemeCompareTool())
