"""Mechanical design knowledge base."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Part:
    """A mechanical part definition."""

    name: str
    category: str
    description: str
    material: str = "PLA"
    dimensions: dict[str, float] = field(default_factory=dict)  # name -> mm
    notes: str = ""


@dataclass
class Joint:
    """A joint connecting two parts."""

    type: str  # "revolute", "prismatic", "fixed"
    parent: str
    child: str
    range_deg: tuple[float, float] = (-180, 180)
    description: str = ""


@dataclass
class Assembly:
    """A complete assembly of parts and joints."""

    name: str
    parts: list[Part] = field(default_factory=list)
    joints: list[Joint] = field(default_factory=list)
    description: str = ""


# ============================================================================
# Standard Parts Library
# ============================================================================

ROBOTIC_ARM_PARTS: list[Part] = [
    Part(
        name="base_plate",
        category="structural",
        description="底座板，固定整个机械臂",
        dimensions={"diameter": 120, "thickness": 8},
        notes="需要 4 个 M6 安装孔",
    ),
    Part(
        name="base_joint_housing",
        category="joint",
        description="底座旋转关节外壳",
        dimensions={"outer_diameter": 80, "height": 40, "wall_thickness": 5},
        notes="内含轴承座",
    ),
    Part(
        name="shoulder_link",
        category="structural",
        description="肩部连杆，连接底座和肘关节",
        dimensions={"length": 150, "width": 40, "height": 30},
        notes="内部走线通道",
    ),
    Part(
        name="elbow_joint",
        category="joint",
        description="肘部旋转关节",
        dimensions={"outer_diameter": 60, "height": 35, "shaft_diameter": 12},
        notes="需要紧定螺钉孔",
    ),
    Part(
        name="forearm_link",
        category="structural",
        description="前臂连杆",
        dimensions={"length": 120, "width": 35, "height": 25},
        notes="轻量化设计，内部中空",
    ),
    Part(
        name="wrist_joint",
        category="joint",
        description="腕部旋转关节",
        dimensions={"outer_diameter": 40, "height": 25, "shaft_diameter": 8},
        notes="紧凑设计",
    ),
    Part(
        name="end_effector_mount",
        category="structural",
        description="末端执行器安装座",
        dimensions={"diameter": 35, "height": 15},
        notes="标准法兰接口",
    ),
    Part(
        name="servo_holder",
        category="actuator",
        description="SG90 舵机安装座",
        dimensions={"width": 24, "length": 30, "height": 12},
        notes="适配 SG90 尺寸",
    ),
]

ROBOTIC_ARM_ASSEMBLY = Assembly(
    name="3-DOF Robotic Arm",
    description="3 自由度桌面机械臂，适用于教育和实验",
    parts=ROBOTIC_ARM_PARTS,
    joints=[
        Joint("revolute", "base_plate", "base_joint_housing", (-180, 180), "底座旋转"),
        Joint("revolute", "base_joint_housing", "shoulder_link", (-90, 90), "肩部俯仰"),
        Joint("revolute", "shoulder_link", "elbow_joint", (-135, 135), "肘部弯曲"),
        Joint("fixed", "elbow_joint", "forearm_link", description="固定连接"),
        Joint("revolute", "forearm_link", "wrist_joint", (-180, 180), "腕部旋转"),
        Joint("fixed", "wrist_joint", "end_effector_mount", description="固定连接"),
    ],
)

# Standard tolerances for 3D printing
PRINT_TOLERANCES = {
    "tight_fit": 0.15,  # mm - for press fits
    "sliding_fit": 0.3,  # mm - for sliding parts
    "loose_fit": 0.5,  # mm - for easy assembly
    "bearing_fit": 0.1,  # mm - for bearing seats
}

# Standard screw sizes for 3D printed parts
STANDARD_SCREWS = {
    "M3": {"tap_hole": 2.5, "clearance_hole": 3.4, "head_diameter": 5.5},
    "M4": {"tap_hole": 3.3, "clearance_hole": 4.5, "head_diameter": 7.0},
    "M5": {"tap_hole": 4.2, "clearance_hole": 5.5, "head_diameter": 8.5},
    "M6": {"tap_hole": 5.0, "clearance_hole": 6.6, "head_diameter": 10.0},
}


def get_part(name: str) -> Part | None:
    """Get a part by name."""
    for part in ROBOTIC_ARM_PARTS:
        if part.name == name:
            return part
    return None


def get_all_categories() -> list[str]:
    """Get all part categories."""
    return sorted(set(p.category for p in ROBOTIC_ARM_PARTS))


def get_parts_by_category(category: str) -> list[Part]:
    """Get all parts in a category."""
    return [p for p in ROBOTIC_ARM_PARTS if p.category == category]
