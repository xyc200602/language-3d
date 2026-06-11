"""Assembly template knowledge base — structured templates for common robot configurations.

Provides:
  - AssemblyTemplate: A reusable blueprint that maps a robot concept (e.g. "6-DOF arm")
    to a parts list, joint chain, and connection methods.
  - 6 pre-built templates derived from ROBOT_PROFILES and built-in assemblies.
  - Weighted search across keywords, robot type, and DOF.
  - Conversion to runtime Assembly objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .mechanics import Assembly, ConnectionMethod, Joint, Part

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TemplatePartSpec:
    """A part specification within an assembly template."""

    name_pattern: str  # e.g. "shoulder_link"
    category: str  # "structural" | "actuator" | "bearing" | "joint"
    description_cn: str
    material: str = "PLA"
    dimensions: dict[str, float] = field(default_factory=dict)
    optional: bool = False
    count: int = 1
    part_catalog_id: str = ""  # link to PART_CATALOG entry


@dataclass
class TemplateJointSpec:
    """A joint specification within an assembly template."""

    type: str  # "revolute" | "fixed" | "prismatic"
    parent: str
    child: str
    range_deg: tuple[float, float] = (-180, 180)
    parent_anchor: str = "top"
    child_anchor: str = "bottom"
    axis: str = "auto"
    connection_method: str = "bolted"
    offset: tuple[float, float, float] | None = None  # positional offset (x,y,z) in mm


@dataclass
class AssemblyTemplate:
    """A reusable assembly blueprint."""

    id: str
    name_en: str
    name_cn: str
    description: str
    dof: int
    robot_type: str  # "arm" | "mobile_base" | "scara"
    keywords: list[str]  # ["3dof", "arm", "机械臂"]
    parts: list[TemplatePartSpec]
    joints: list[TemplateJointSpec]
    default_angles: dict[str, float] = field(default_factory=dict)
    parameter_overrides: dict[str, dict[str, float]] = field(default_factory=dict)
    connection_defaults: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Template definitions (6 templates)
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, AssemblyTemplate] = {}


def _register(t: AssemblyTemplate) -> None:
    TEMPLATES[t.id] = t


# ── Template 1: 3-DOF Desktop Arm ──────────────────────────────────────
# Derived from ROBOTIC_ARM_ASSEMBLY in mechanics.py

_register(AssemblyTemplate(
    id="3dof_arm",
    name_en="3-DOF Desktop Arm",
    name_cn="3自由度桌面臂",
    description="基础3自由度桌面机械臂，SG90舵机驱动，适用于教育和实验",
    dof=3,
    robot_type="arm",
    keywords=["3dof", "arm", "桌面臂", "机械臂", "3自由度", "robotic arm", "desktop",
              "舵机", "servo", "教育", "education"],
    parts=[
        TemplatePartSpec("base_plate", "structural", "底座板", "PLA",
                         {"diameter": 120, "thickness": 8}),
        TemplatePartSpec("base_joint_housing", "joint", "底座旋转关节外壳", "PLA",
                         {"outer_diameter": 80, "height": 40, "wall_thickness": 5},
                         part_catalog_id="servo_sg90"),
        TemplatePartSpec("shoulder_link", "structural", "肩部连杆", "PLA",
                         {"length": 150, "width": 40, "height": 30}),
        TemplatePartSpec("elbow_joint", "joint", "肘部旋转关节", "PLA",
                         {"outer_diameter": 60, "height": 35, "shaft_diameter": 12},
                         part_catalog_id="servo_sg90"),
        TemplatePartSpec("forearm_link", "structural", "前臂连杆", "PLA",
                         {"length": 120, "width": 35, "height": 25}),
        TemplatePartSpec("wrist_joint", "joint", "腕部旋转关节", "PLA",
                         {"outer_diameter": 40, "height": 25, "shaft_diameter": 8},
                         part_catalog_id="servo_sg90"),
        TemplatePartSpec("end_effector_mount", "structural", "末端执行器安装座", "PLA",
                         {"diameter": 35, "height": 15}),
        TemplatePartSpec("servo_holder", "actuator", "SG90舵机安装座", "PLA",
                         {"width": 24, "length": 30, "height": 12},
                         part_catalog_id="servo_sg90"),
    ],
    joints=[
        TemplateJointSpec("revolute", "base_plate", "base_joint_housing",
                          (-180, 180), axis="z"),
        TemplateJointSpec("revolute", "base_joint_housing", "shoulder_link",
                          (-90, 90), axis="y"),
        TemplateJointSpec("revolute", "shoulder_link", "elbow_joint",
                          (-135, 135),
                          parent_anchor="front", child_anchor="back", axis="x"),
        TemplateJointSpec("fixed", "elbow_joint", "forearm_link",
                          parent_anchor="front", child_anchor="back"),
        TemplateJointSpec("revolute", "forearm_link", "wrist_joint",
                          (-180, 180),
                          parent_anchor="front", child_anchor="back", axis="z"),
        TemplateJointSpec("fixed", "wrist_joint", "end_effector_mount",
                          parent_anchor="front", child_anchor="back"),
    ],
    default_angles={
        "base_joint_housing": 0.0,
        "shoulder_link": -45.0,
        "elbow_joint": -30.0,
        "forearm_link": 0.0,
        "wrist_joint": 0.0,
    },
    parameter_overrides={
        "shoulder_link": {"length": 150},
        "forearm_link": {"length": 120},
    },
))


# ── Template 2: 4-DOF Standalone Arm ──────────────────────────────────
# Inspired by EXAMPLE_ARM_STANDALONE pattern with added wrist DOF

_register(AssemblyTemplate(
    id="4dof_arm",
    name_en="4-DOF Standalone Arm",
    name_cn="4自由度独立臂",
    description="4自由度独立机械臂，MG996R舵机驱动，含肩/肘/腕/夹爪",
    dof=4,
    robot_type="arm",
    keywords=["4dof", "arm", "机械臂", "4自由度", "standalone", "独立臂", "mg996r",
              "舵机", "夹爪", "gripper"],
    parts=[
        TemplatePartSpec("base_plate", "structural", "底座板", "PLA",
                         {"diameter": 120, "thickness": 8}),
        TemplatePartSpec("shoulder_joint", "actuator", "肩部舵机", "ABS",
                         {"diameter": 45, "height": 50},
                         part_catalog_id="servo_mg996r"),
        TemplatePartSpec("upper_arm_link", "structural", "上臂连杆", "PLA",
                         {"length": 140, "width": 35, "height": 25}),
        TemplatePartSpec("elbow_joint", "actuator", "肘部舵机", "ABS",
                         {"diameter": 45, "height": 40},
                         part_catalog_id="servo_mg996r"),
        TemplatePartSpec("forearm_link", "structural", "前臂连杆", "PLA",
                         {"length": 120, "width": 30, "height": 20}),
        TemplatePartSpec("wrist_joint", "actuator", "腕部舵机", "ABS",
                         {"diameter": 35, "height": 30},
                         part_catalog_id="servo_mg996r"),
        TemplatePartSpec("gripper_base", "structural", "夹爪基座", "PLA",
                         {"length": 40, "width": 35, "height": 15}),
        TemplatePartSpec("gripper_finger_left", "structural", "夹爪左手指", "PLA",
                         {"length": 35, "width": 6, "height": 15}),
        TemplatePartSpec("gripper_finger_right", "structural", "夹爪右手指", "PLA",
                         {"length": 35, "width": 6, "height": 15}),
    ],
    joints=[
        # joint1: base yaw — servo sits on base, rotates around Z
        TemplateJointSpec("revolute", "base_plate", "shoulder_joint",
                          (-180, 180), axis="z"),
        # joint2: shoulder pitch — link tilts up/down around Y
        TemplateJointSpec("revolute", "shoulder_joint", "upper_arm_link",
                          (-90, 90), axis="y"),
        # joint3: elbow pitch — forearm tilts around X (perpendicular to arm)
        TemplateJointSpec("revolute", "upper_arm_link", "elbow_joint",
                          (-135, 135),
                          parent_anchor="front", child_anchor="back", axis="x"),
        # elbow housing to forearm (fixed mount)
        TemplateJointSpec("fixed", "elbow_joint", "forearm_link",
                          parent_anchor="front", child_anchor="back"),
        # joint4: wrist pitch — gripper tilts around X
        TemplateJointSpec("revolute", "forearm_link", "wrist_joint",
                          (-90, 90),
                          parent_anchor="front", child_anchor="back", axis="x"),
        # wrist to gripper base (fixed mount)
        TemplateJointSpec("fixed", "wrist_joint", "gripper_base",
                          parent_anchor="front", child_anchor="back"),
        # gripper fingers: prismatic (slide open/close), offset to sides
        TemplateJointSpec("prismatic", "gripper_base", "gripper_finger_left",
                          (-8, 12),
                          parent_anchor="front", child_anchor="back",
                          axis="x", offset=(-8, 0, 0)),
        TemplateJointSpec("prismatic", "gripper_base", "gripper_finger_right",
                          (-8, 12),
                          parent_anchor="front", child_anchor="back",
                          axis="x", offset=(8, 0, 0)),
    ],
    default_angles={
        "shoulder_joint": 0.0,
        "upper_arm_link": -45.0,
        "elbow_joint": -30.0,
        "wrist_joint": 15.0,
        "gripper_finger_left": 0.0,
        "gripper_finger_right": 0.0,
    },
))


# ── Template 3: 6-DOF Belt-Driven Arm ────────────────────────────────
# Derived from PAROL6 / Thor profiles — NEMA17 + GT2 belt transmission

_register(AssemblyTemplate(
    id="6dof_belt_arm",
    name_en="6-DOF Belt-Driven Arm",
    name_cn="6自由度皮带传动臂",
    description="6自由度桌面机械臂，NEMA17步进电机+GT2同步带传动，参考PAROL6设计",
    dof=6,
    robot_type="arm",
    keywords=["6dof", "arm", "机械臂", "6自由度", "belt", "皮带", "nema17",
              "步进", "stepper", "gt2", "parol6", "工业"],
    parts=[
        # Base
        TemplatePartSpec("base_plate", "structural", "底座板", "PLA",
                         {"diameter": 140, "thickness": 10}),
        TemplatePartSpec("base_housing", "structural", "底座关节壳体", "PLA",
                         {"outer_diameter": 90, "height": 50, "wall_thickness": 5},
                         part_catalog_id="nema17_stepper"),
        # Joint 1 (base rotation)
        TemplatePartSpec("shoulder_bracket", "structural", "肩部支架", "PLA",
                         {"length": 70, "width": 60, "height": 50}),
        TemplatePartSpec("shoulder_motor", "actuator", "肩部驱动电机", "steel",
                         {"body_size": 42.3, "body_length": 40},
                         part_catalog_id="nema17_stepper"),
        # Joint 2 (shoulder pitch) — belt driven
        TemplatePartSpec("upper_arm_link", "structural", "上臂连杆", "PLA",
                         {"length": 200, "width": 45, "height": 35}),
        TemplatePartSpec("shoulder_pulley", "actuator", "肩部同步轮", "aluminum",
                         {"teeth": 20, "width": 6},
                         part_catalog_id="gt2_pulley"),
        TemplatePartSpec("shoulder_belt", "actuator", "肩部同步带", "rubber",
                         {}, part_catalog_id="gt2_belt", optional=True),
        # Joint 3 (elbow pitch) — belt driven
        TemplatePartSpec("elbow_motor_bracket", "structural", "肘部电机支架", "PLA",
                         {"length": 50, "width": 45, "height": 40},
                         part_catalog_id="nema17_stepper"),
        TemplatePartSpec("elbow_pulley", "actuator", "肘部同步轮", "aluminum",
                         {"teeth": 20, "width": 6},
                         part_catalog_id="gt2_pulley"),
        TemplatePartSpec("elbow_belt", "actuator", "肘部同步带", "rubber",
                         {}, part_catalog_id="gt2_belt", optional=True),
        TemplatePartSpec("forearm_link", "structural", "前臂连杆", "PLA",
                         {"length": 160, "width": 40, "height": 30}),
        # Joint 4 (wrist pitch) — belt driven
        TemplatePartSpec("wrist_pitch_motor", "actuator", "腕部俯仰电机", "steel",
                         {"body_size": 42.3, "body_length": 34},
                         part_catalog_id="nema17_stepper"),
        TemplatePartSpec("wrist_pitch_pulley", "actuator", "腕部俯仰同步轮", "aluminum",
                         {"teeth": 16, "width": 6},
                         part_catalog_id="gt2_pulley"),
        # Joint 5 (wrist yaw)
        TemplatePartSpec("wrist_yaw_motor", "actuator", "腕部偏航电机", "steel",
                         {"body_size": 42.3, "body_length": 28},
                         part_catalog_id="nema17_stepper"),
        TemplatePartSpec("wrist_yaw_housing", "structural", "腕部偏航壳体", "PLA",
                         {"outer_diameter": 45, "height": 30}),
        # Joint 6 (wrist roll)
        TemplatePartSpec("wrist_roll_motor", "actuator", "腕部旋转电机", "steel",
                         {"body_size": 42.3, "body_length": 28},
                         part_catalog_id="nema17_stepper"),
        TemplatePartSpec("end_effector_flange", "structural", "末端法兰", "PLA",
                         {"diameter": 40, "height": 12}),
        # Bearings
        TemplatePartSpec("bearing_joint_1", "bearing", "关节1轴承", "steel",
                         {"inner_diameter": 8, "outer_diameter": 22, "width": 7},
                         part_catalog_id="bearing_608"),
        TemplatePartSpec("bearing_joint_2", "bearing", "关节2轴承", "steel",
                         {"inner_diameter": 8, "outer_diameter": 22, "width": 7},
                         part_catalog_id="bearing_608"),
    ],
    joints=[
        TemplateJointSpec("revolute", "base_plate", "base_housing",
                          (-180, 180), axis="z"),
        TemplateJointSpec("revolute", "base_housing", "shoulder_bracket",
                          (-90, 90), axis="y"),
        TemplateJointSpec("fixed", "shoulder_bracket", "shoulder_motor"),
        TemplateJointSpec("revolute", "shoulder_bracket", "upper_arm_link",
                          (-120, 120),
                          parent_anchor="front", child_anchor="back", axis="x"),
        TemplateJointSpec("fixed", "shoulder_motor", "shoulder_pulley",
                          parent_anchor="top", child_anchor="center"),
        TemplateJointSpec("fixed", "upper_arm_link", "elbow_motor_bracket",
                          parent_anchor="front", child_anchor="back"),
        TemplateJointSpec("revolute", "elbow_motor_bracket", "forearm_link",
                          (-135, 135),
                          parent_anchor="front", child_anchor="back", axis="x"),
        TemplateJointSpec("fixed", "elbow_motor_bracket", "elbow_pulley",
                          parent_anchor="top", child_anchor="center"),
        TemplateJointSpec("fixed", "wrist_pitch_motor", "wrist_pitch_pulley",
                          parent_anchor="top", child_anchor="center"),
        TemplateJointSpec("fixed", "base_housing", "bearing_joint_1",
                          parent_anchor="top", child_anchor="center"),
        TemplateJointSpec("fixed", "elbow_motor_bracket", "bearing_joint_2",
                          parent_anchor="bottom", child_anchor="center"),
        TemplateJointSpec("revolute", "forearm_link", "wrist_yaw_housing",
                          (-120, 120),
                          parent_anchor="front", child_anchor="back", axis="x"),
        TemplateJointSpec("fixed", "wrist_yaw_housing", "wrist_pitch_motor"),
        TemplateJointSpec("revolute", "wrist_yaw_housing", "wrist_roll_motor",
                          (-180, 180),
                          parent_anchor="front", child_anchor="back", axis="z"),
        TemplateJointSpec("fixed", "wrist_roll_motor", "end_effector_flange",
                          parent_anchor="front", child_anchor="back"),
    ],
    default_angles={
        "base_housing": 0.0,
        "shoulder_bracket": 0.0,
        "upper_arm_link": -45.0,
        "forearm_link": -30.0,
        "wrist_yaw_housing": 15.0,
        "wrist_roll_motor": 0.0,
    },
    parameter_overrides={
        "upper_arm_link": {"length": 200},
        "forearm_link": {"length": 160},
    },
))


# ── Template 4: 4-Wheel Differential Drive ──────────────────────────
# Derived from EXAMPLE_4W_ROBOT / Leo Rover profile

_register(AssemblyTemplate(
    id="diff_drive_4w",
    name_en="4-Wheel Differential Drive",
    name_cn="4轮差速底盘",
    description="4轮差速驱动底盘，直流减速电机+编码器，适用于室内导航",
    dof=2,
    robot_type="mobile_base",
    keywords=["diff", "differential", "4w", "4轮", "差速", "底盘", "mobile",
              "base", "轮式", "wheeled", "rover", "导航", "navigation"],
    parts=[
        TemplatePartSpec("chassis_plate", "structural", "底盘板", "Aluminum",
                         {"length": 200, "width": 160, "thickness": 3}),
        TemplatePartSpec("top_plate", "structural", "顶板", "Aluminum",
                         {"length": 180, "width": 140, "thickness": 2},
                         optional=True),
        # Wheels + motors
        TemplatePartSpec("wheel_fl", "structural", "左前轮", "rubber",
                         {"diameter": 65, "width": 26}, count=1),
        TemplatePartSpec("wheel_fr", "structural", "右前轮", "rubber",
                         {"diameter": 65, "width": 26}, count=1),
        TemplatePartSpec("wheel_rl", "structural", "左后轮", "rubber",
                         {"diameter": 65, "width": 26}, count=1),
        TemplatePartSpec("wheel_rr", "structural", "右后轮", "rubber",
                         {"diameter": 65, "width": 26}, count=1),
        TemplatePartSpec("motor_fl", "actuator", "左前电机", "steel",
                         {"diameter": 25, "length": 54},
                         part_catalog_id="dc_motor"),
        TemplatePartSpec("motor_fr", "actuator", "右前电机", "steel",
                         {"diameter": 25, "length": 54},
                         part_catalog_id="dc_motor"),
        TemplatePartSpec("motor_rl", "actuator", "左后电机", "steel",
                         {"diameter": 25, "length": 54},
                         part_catalog_id="dc_motor"),
        TemplatePartSpec("motor_rr", "actuator", "右后电机", "steel",
                         {"diameter": 25, "length": 54},
                         part_catalog_id="dc_motor"),
        # Electronics
        TemplatePartSpec("motor_driver", "actuator", "电机驱动板", "electronics",
                         {"length": 55, "width": 40, "height": 15},
                         optional=True),
        TemplatePartSpec("controller_board", "actuator", "主控板", "electronics",
                         {"length": 70, "width": 55, "height": 15},
                         optional=True),
        TemplatePartSpec("battery_holder", "structural", "电池盒", "PLA",
                         {"length": 80, "width": 55, "height": 30},
                         optional=True),
        TemplatePartSpec("caster_wheel", "structural", "万向轮", "rubber",
                         {"diameter": 30}, optional=True),
    ],
    joints=[
        TemplateJointSpec("fixed", "chassis_plate", "motor_fl",
                          connection_method="bolted"),
        TemplateJointSpec("fixed", "chassis_plate", "motor_fr",
                          connection_method="bolted"),
        TemplateJointSpec("fixed", "chassis_plate", "motor_rl",
                          connection_method="bolted"),
        TemplateJointSpec("fixed", "chassis_plate", "motor_rr",
                          connection_method="bolted"),
        TemplateJointSpec("revolute", "motor_fl", "wheel_fl", (-180, 180),
                          parent_anchor="left", child_anchor="center", axis="y",
                          connection_method="press_fit"),
        TemplateJointSpec("revolute", "motor_fr", "wheel_fr", (-180, 180),
                          parent_anchor="right", child_anchor="center", axis="y",
                          connection_method="press_fit"),
        TemplateJointSpec("revolute", "motor_rl", "wheel_rl", (-180, 180),
                          parent_anchor="left", child_anchor="center", axis="y",
                          connection_method="press_fit"),
        TemplateJointSpec("revolute", "motor_rr", "wheel_rr", (-180, 180),
                          parent_anchor="right", child_anchor="center", axis="y",
                          connection_method="press_fit"),
        TemplateJointSpec("fixed", "chassis_plate", "motor_driver"),
        TemplateJointSpec("fixed", "chassis_plate", "controller_board"),
        TemplateJointSpec("fixed", "chassis_plate", "battery_holder"),
        TemplateJointSpec("fixed", "chassis_plate", "caster_wheel"),
        TemplateJointSpec("fixed", "chassis_plate", "top_plate"),
    ],
    default_angles={
        "wheel_fl": 0.0,
        "wheel_fr": 0.0,
        "wheel_rl": 0.0,
        "wheel_rr": 0.0,
    },
))


# ── Template 5: 4-Wheel Mecanum Chassis ─────────────────────────────
# Based on Leo Rover profile adapted for mecanum wheels

_register(AssemblyTemplate(
    id="mecanum_4w",
    name_en="4-Wheel Mecanum Chassis",
    name_cn="4轮麦克纳姆底盘",
    description="4轮麦克纳姆轮底盘，全向移动，适用于精密定位场景",
    dof=2,
    robot_type="mobile_base",
    keywords=["mecanum", "麦克纳姆", "全向轮", "omnidirectional", "4w", "4轮",
              "底盘", "mobile", "base", "holonomic"],
    parts=[
        TemplatePartSpec("chassis_frame", "structural", "底盘框架", "Aluminum",
                         {"length": 220, "width": 180, "thickness": 3}),
        TemplatePartSpec("top_plate", "structural", "顶板", "Aluminum",
                         {"length": 200, "width": 160, "thickness": 2},
                         optional=True),
        # Mecanum wheels
        TemplatePartSpec("mecanum_fl", "structural", "左前麦克纳姆轮", "rubber",
                         {"diameter": 60, "width": 30}),
        TemplatePartSpec("mecanum_fr", "structural", "右前麦克纳姆轮", "rubber",
                         {"diameter": 60, "width": 30}),
        TemplatePartSpec("mecanum_rl", "structural", "左后麦克纳姆轮", "rubber",
                         {"diameter": 60, "width": 30}),
        TemplatePartSpec("mecanum_rr", "structural", "右后麦克纳姆轮", "rubber",
                         {"diameter": 60, "width": 30}),
        # Motors
        TemplatePartSpec("motor_fl", "actuator", "左前电机", "steel",
                         {"diameter": 25, "length": 54},
                         part_catalog_id="dc_motor"),
        TemplatePartSpec("motor_fr", "actuator", "右前电机", "steel",
                         {"diameter": 25, "length": 54},
                         part_catalog_id="dc_motor"),
        TemplatePartSpec("motor_rl", "actuator", "左后电机", "steel",
                         {"diameter": 25, "length": 54},
                         part_catalog_id="dc_motor"),
        TemplatePartSpec("motor_rr", "actuator", "右后电机", "steel",
                         {"diameter": 25, "length": 54},
                         part_catalog_id="dc_motor"),
        # Electronics
        TemplatePartSpec("motor_driver", "actuator", "电机驱动板", "electronics",
                         {"length": 55, "width": 40, "height": 15},
                         optional=True),
        TemplatePartSpec("controller_board", "actuator", "主控板", "electronics",
                         {"length": 70, "width": 55, "height": 15},
                         optional=True),
        TemplatePartSpec("battery_holder", "structural", "电池盒", "PLA",
                         {"length": 80, "width": 55, "height": 30},
                         optional=True),
        TemplatePartSpec("side_plate_left", "structural", "左侧板", "Aluminum",
                         {"length": 220, "width": 60, "thickness": 2}),
        TemplatePartSpec("side_plate_right", "structural", "右侧板", "Aluminum",
                         {"length": 220, "width": 60, "thickness": 2}),
    ],
    joints=[
        TemplateJointSpec("fixed", "chassis_frame", "motor_fl",
                          connection_method="bolted"),
        TemplateJointSpec("fixed", "chassis_frame", "motor_fr",
                          connection_method="bolted"),
        TemplateJointSpec("fixed", "chassis_frame", "motor_rl",
                          connection_method="bolted"),
        TemplateJointSpec("fixed", "chassis_frame", "motor_rr",
                          connection_method="bolted"),
        TemplateJointSpec("revolute", "motor_fl", "mecanum_fl", (-180, 180),
                          parent_anchor="left", child_anchor="center", axis="y",
                          connection_method="press_fit"),
        TemplateJointSpec("revolute", "motor_fr", "mecanum_fr", (-180, 180),
                          parent_anchor="right", child_anchor="center", axis="y",
                          connection_method="press_fit"),
        TemplateJointSpec("revolute", "motor_rl", "mecanum_rl", (-180, 180),
                          parent_anchor="left", child_anchor="center", axis="y",
                          connection_method="press_fit"),
        TemplateJointSpec("revolute", "motor_rr", "mecanum_rr", (-180, 180),
                          parent_anchor="right", child_anchor="center", axis="y",
                          connection_method="press_fit"),
        TemplateJointSpec("fixed", "chassis_frame", "motor_driver"),
        TemplateJointSpec("fixed", "chassis_frame", "controller_board"),
        TemplateJointSpec("fixed", "chassis_frame", "battery_holder"),
        TemplateJointSpec("fixed", "chassis_frame", "side_plate_left"),
        TemplateJointSpec("fixed", "chassis_frame", "side_plate_right"),
        TemplateJointSpec("fixed", "chassis_frame", "top_plate"),
    ],
    default_angles={
        "mecanum_fl": 0.0,
        "mecanum_fr": 0.0,
        "mecanum_rl": 0.0,
        "mecanum_rr": 0.0,
    },
))


# ── Template 6: SCARA Arm ───────────────────────────────────────────

_register(AssemblyTemplate(
    id="scara_arm",
    name_en="SCARA Arm",
    name_cn="SCARA型臂",
    description="SCARA型机械臂，水平旋转关节+垂直升降，适用于装配和拾放",
    dof=4,
    robot_type="scara",
    keywords=["scara", "水平关节", "装配", "拾放", "pick", "place", "assembly",
              "4dof", "4自由度", "prismatic"],
    parts=[
        TemplatePartSpec("base_column", "structural", "底座立柱", "Aluminum",
                         {"diameter": 100, "height": 80, "wall_thickness": 5}),
        TemplatePartSpec("shoulder_arm", "structural", "肩部水平臂", "PLA",
                         {"length": 200, "width": 50, "height": 30}),
        TemplatePartSpec("elbow_arm", "structural", "肘部水平臂", "PLA",
                         {"length": 160, "width": 45, "height": 25}),
        TemplatePartSpec("vertical_carriage", "structural", "垂直滑台", "PLA",
                         {"length": 50, "width": 50, "height": 80}),
        TemplatePartSpec("end_effector", "structural", "末端执行器", "PLA",
                         {"length": 40, "width": 30, "height": 20}),
        TemplatePartSpec("shoulder_motor", "actuator", "肩部电机", "steel",
                         {"body_size": 42.3, "body_length": 40},
                         part_catalog_id="nema17_stepper"),
        TemplatePartSpec("elbow_motor", "actuator", "肘部电机", "steel",
                         {"body_size": 42.3, "body_length": 34},
                         part_catalog_id="nema17_stepper"),
        TemplatePartSpec("lift_motor", "actuator", "升降电机", "steel",
                         {"body_size": 42.3, "body_length": 40},
                         part_catalog_id="nema17_stepper"),
        TemplatePartSpec("linear_shaft", "bearing", "直线轴", "steel",
                         {"diameter": 8, "length": 200},
                         part_catalog_id="linear_shaft"),
    ],
    joints=[
        TemplateJointSpec("revolute", "base_column", "shoulder_arm",
                          (-180, 180), axis="z"),
        TemplateJointSpec("revolute", "shoulder_arm", "elbow_arm",
                          (-150, 150),
                          parent_anchor="front", child_anchor="back", axis="z"),
        TemplateJointSpec("prismatic", "elbow_arm", "vertical_carriage",
                          (0, 100), axis="z"),
        TemplateJointSpec("revolute", "vertical_carriage", "end_effector",
                          (-180, 180),
                          parent_anchor="front", child_anchor="back", axis="z"),
        # Motors (fixed mounts)
        TemplateJointSpec("fixed", "base_column", "shoulder_motor"),
        TemplateJointSpec("fixed", "shoulder_arm", "elbow_motor"),
        TemplateJointSpec("fixed", "elbow_arm", "lift_motor"),
        TemplateJointSpec("fixed", "elbow_arm", "linear_shaft"),
    ],
    default_angles={
        "shoulder_arm": 0.0,
        "elbow_arm": -30.0,
        "vertical_carriage": 0.0,
        "end_effector": 0.0,
    },
    parameter_overrides={
        "shoulder_arm": {"length": 200},
        "elbow_arm": {"length": 160},
    },
))


# ---------------------------------------------------------------------------
# Search & conversion
# ---------------------------------------------------------------------------


def search_assembly_templates(
    query: str = "",
    robot_type: str = "",
    min_dof: int = 0,
    max_dof: int = 99,
) -> list[AssemblyTemplate]:
    """Weighted search over assembly templates.

    Scoring: keyword match 3x, robot_type 5x, DOF proximity 2x.
    Returns templates sorted by descending score.
    """
    results: list[tuple[float, AssemblyTemplate]] = []

    q_lower = query.lower().strip()

    for tpl in TEMPLATES.values():
        score = 0.0

        # DOF filter (hard constraint)
        if tpl.dof < min_dof or tpl.dof > max_dof:
            continue

        # Robot type filter (hard constraint if specified)
        if robot_type and tpl.robot_type != robot_type.lower():
            continue

        # Keyword scoring (3x weight)
        if q_lower:
            for kw in tpl.keywords:
                if q_lower in kw.lower() or kw.lower() in q_lower:
                    score += 3.0
                    break

            # Also match against name_cn and name_en
            if q_lower in tpl.name_cn.lower() or q_lower in tpl.name_en.lower():
                score += 3.0

            # Partial match on individual query tokens
            tokens = q_lower.replace("-", " ").replace("_", " ").split()
            for token in tokens:
                for kw in tpl.keywords:
                    if token in kw.lower():
                        score += 1.5
                        break

        # Robot type match (5x weight)
        if robot_type and tpl.robot_type == robot_type.lower():
            score += 5.0

        # DOF proximity scoring (2x weight)
        if q_lower:
            # Extract DOF number from query if present
            for token in q_lower.replace("-", " ").replace("_", " ").split():
                if token.endswith("dof") or token.endswith("自由度"):
                    try:
                        prefix = token.replace("dof", "").replace("自由度", "")
                        q_dof = int(prefix)
                        if q_dof == tpl.dof:
                            score += 2.0
                        elif abs(q_dof - tpl.dof) <= 1:
                            score += 1.0
                    except (ValueError, IndexError):
                        pass

        # If no query specified, include all with base score
        if not q_lower and not robot_type:
            score = 1.0

        if score > 0:
            results.append((score, tpl))

    results.sort(key=lambda x: x[0], reverse=True)
    return [tpl for _, tpl in results]


def template_to_assembly(
    template: AssemblyTemplate,
    overrides: dict[str, dict[str, float]] | None = None,
) -> Assembly:
    """Convert a template to a runtime Assembly object.

    Args:
        template: The template to convert.
        overrides: Optional per-part dimension overrides.
            Key = part name_pattern, value = dimension dict to merge.

    Returns:
        An Assembly ready for use with AssemblySolver.
    """
    overrides = overrides or {}

    # Build parts
    parts: list[Part] = []
    for ps in template.parts:
        dims = dict(ps.dimensions)
        if ps.name_pattern in overrides:
            dims.update(overrides[ps.name_pattern])
        parts.append(Part(
            name=ps.name_pattern,
            category=ps.category,
            description=ps.description_cn,
            material=ps.material,
            dimensions=dims,
        ))

    # Build joints
    joints: list[Joint] = []
    for js in template.joints:
        conn = None
        if js.connection_method:
            conn = ConnectionMethod(type=js.connection_method)
        joints.append(Joint(
            type=js.type,
            parent=js.parent,
            child=js.child,
            range_deg=js.range_deg,
            parent_anchor=js.parent_anchor,
            child_anchor=js.child_anchor,
            axis=js.axis,
            connection=conn,
            offset=js.offset,
        ))

    # Merge default angles with template overrides
    angles = dict(template.default_angles)

    return Assembly(
        name=template.name_en,
        parts=parts,
        joints=joints,
        description=template.description,
        default_angles=angles,
    )


def list_assembly_templates() -> list[dict[str, Any]]:
    """Return a summary list of all available templates."""
    summaries = []
    for tpl in TEMPLATES.values():
        req_parts = sum(1 for p in tpl.parts if not p.optional)
        opt_parts = sum(1 for p in tpl.parts if p.optional)
        summaries.append({
            "id": tpl.id,
            "name_en": tpl.name_en,
            "name_cn": tpl.name_cn,
            "dof": tpl.dof,
            "robot_type": tpl.robot_type,
            "parts_required": req_parts,
            "parts_optional": opt_parts,
            "joints": len(tpl.joints),
            "description": tpl.description,
        })
    return summaries
