"""Mechanical design knowledge base."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ============================================================================
# Material Density Constants (kg/m³)
# ============================================================================

class MaterialDensity:
    """Common material densities in kg/m³."""
    PLA = 1240
    ABS = 1040
    PETG = 1270
    Nylon = 1140
    TPU = 1200
    Aluminum = 2700
    Steel = 7850
    StainlessSteel = 8000
    Copper = 8960
    Brass = 8500
    Titanium = 4430
    CarbonFiber = 1600

    @classmethod
    def get(cls, material: str) -> float:
        """Get density by material name (case-insensitive, partial match)."""
        name = material.lower().replace(" ", "").replace("-", "").replace("_", "")
        mapping = {
            "pla": cls.PLA,
            "abs": cls.ABS,
            "petg": cls.PETG,
            "nylon": cls.Nylon,
            "tpu": cls.TPU,
            "aluminum": cls.Aluminum,
            "aluminium": cls.Aluminum,
            "alu": cls.Aluminum,
            "steel": cls.Steel,
            "carbonsteel": cls.Steel,
            "stainlesssteel": cls.StainlessSteel,
            "stainless": cls.StainlessSteel,
            "copper": cls.Copper,
            "brass": cls.Brass,
            "titanium": cls.Titanium,
            "carbonfiber": cls.CarbonFiber,
            "cf": cls.CarbonFiber,
        }
        return mapping.get(name, cls.PLA)  # default to PLA if unknown

    @classmethod
    def all_materials(cls) -> dict[str, float]:
        """Return all material name -> density pairs."""
        return {
            "PLA": cls.PLA, "ABS": cls.ABS, "PETG": cls.PETG,
            "Nylon": cls.Nylon, "TPU": cls.TPU,
            "Aluminum": cls.Aluminum, "Steel": cls.Steel,
            "StainlessSteel": cls.StainlessSteel,
            "Copper": cls.Copper, "Brass": cls.Brass,
            "Titanium": cls.Titanium, "CarbonFiber": cls.CarbonFiber,
        }


@dataclass
class Part:
    """A mechanical part definition."""

    name: str
    category: str
    description: str
    material: str = "PLA"
    dimensions: dict[str, float] = field(default_factory=dict)  # name -> mm
    notes: str = ""
    # Mass properties (Task 45)
    mass: float = 0.0  # kg, 0 = not yet computed
    density: float = 0.0  # kg/m³, 0 = use material default
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0)  # mm from origin
    inertia_tensor: list[list[float]] = field(default_factory=lambda: [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])  # 3x3 kg·mm²

    def effective_density(self) -> float:
        """Return density: explicit value if set, otherwise material default."""
        if self.density > 0:
            return self.density
        return MaterialDensity.get(self.material)

    def compute_volume_mm3(self) -> float:
        """Estimate volume in mm³ from dimensions.

        Uses the correct formula based on detected shape type:
        - Box: length * width * height
        - Cylinder: pi * radius^2 * height
        - Sphere: 4/3 * pi * radius^3
        - Tube: pi * (outer_radius^2 - inner_radius^2) * height

        Shape detection order:
        1. Explicit "shape" key in dimensions
        2. Category name match
        3. Heuristic from available dimension keys (e.g. radius/diameter → cylinder)
        """
        if not self.dimensions:
            return 0.0

        shape = self.dimensions.get("shape", self.category).lower()
        has_radius = (
            "radius" in self.dimensions
            or "diameter" in self.dimensions
            or "outer_diameter" in self.dimensions
        )
        has_height = "height" in self.dimensions or "length" in self.dimensions

        # Cylinder: pi * r^2 * h
        if "cylinder" in shape or "圆" in shape or "shaft" in shape or "axle" in shape:
            r = self.dimensions.get("radius", 0)
            if r == 0 and "diameter" in self.dimensions:
                r = self.dimensions["diameter"] / 2.0
            h = self.dimensions.get("height", self.dimensions.get("length", 0))
            if r > 0 and h > 0:
                return math.pi * r * r * h

        # Sphere: 4/3 * pi * r^3
        if "sphere" in shape or "球" in shape:
            r = self.dimensions.get("radius", 0)
            if r == 0 and "diameter" in self.dimensions:
                r = self.dimensions["diameter"] / 2.0
            if r > 0:
                return (4.0 / 3.0) * math.pi * r * r * r

        # Tube / hollow cylinder: pi * (r_outer^2 - r_inner^2) * h
        if "tube" in shape or "管" in shape or "hollow" in shape:
            outer_r = self.dimensions.get("outer_radius", self.dimensions.get("radius", 0))
            inner_r = self.dimensions.get("inner_radius", 0)
            h = self.dimensions.get("height", self.dimensions.get("length", 0))
            if outer_r > 0 and h > 0:
                return math.pi * (outer_r * outer_r - inner_r * inner_r) * h

        # Heuristic: if we have radius/diameter but no width, treat as cylinder
        if has_radius and "width" not in self.dimensions and has_height:
            r = self.dimensions.get("radius", 0)
            if r == 0 and "diameter" in self.dimensions:
                r = self.dimensions["diameter"] / 2.0
            if r == 0 and "outer_diameter" in self.dimensions:
                r = self.dimensions["outer_diameter"] / 2.0
            h = self.dimensions.get("height", self.dimensions.get("length", 0))
            if r > 0 and h > 0:
                return math.pi * r * r * h

        # Box / default: l * w * h
        l = self.dimensions.get("length", self.dimensions.get("diameter", 0))
        w = self.dimensions.get("width", 0)
        h = self.dimensions.get("height", self.dimensions.get("thickness", 0))
        if l > 0 and w > 0 and h > 0:
            return l * w * h

        # Fallback: detect circular vs box dimensions
        d = self.dimensions
        diameter = d.get("diameter", d.get("outer_diameter", 0))
        if diameter and diameter > 0:
            r = diameter / 2.0
            h = d.get("height", d.get("length", d.get("width", 10)))
            if h > 0:
                return math.pi * r * r * h

        # Box fallback: length × width × height
        l = d.get("length", 0)
        w = d.get("width", 0)
        h = d.get("height", d.get("thickness", 0))
        if l > 0 and w > 0 and h > 0:
            return l * w * h

        # Last resort: first 3 numeric values
        vals = [v for v in d.values() if isinstance(v, (int, float))]
        if len(vals) >= 3:
            return vals[0] * vals[1] * vals[2]
        elif len(vals) >= 2 and has_radius:
            r = self.dimensions.get("radius", 0)
            if r == 0 and "diameter" in self.dimensions:
                r = self.dimensions["diameter"] / 2.0
            if r == 0 and "outer_diameter" in self.dimensions:
                r = self.dimensions["outer_diameter"] / 2.0
            h = vals[0] if not ("height" in self.dimensions) else self.dimensions["height"]
            return math.pi * r * r * h if r > 0 and h > 0 else 0.0
        return 0.0

    def compute_estimated_mass(self) -> float:
        """Estimate mass (kg) from volume and density."""
        if self.mass > 0:
            return self.mass
        vol_mm3 = self.compute_volume_mm3()
        vol_m3 = vol_mm3 * 1e-9
        return vol_m3 * self.effective_density()


@dataclass
class AttachmentPoint:
    """A precise 3D attachment point with position and normal vector."""

    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    name: str = ""


@dataclass
class BoltHole:
    """A single bolt hole definition for bolted connections."""

    position: tuple[float, float, float]  # 3D position on the face (mm)
    diameter: float = 3.0                 # Bolt hole diameter (mm), e.g. M3=3.4, M4=4.5
    depth: float = 0.0                    # Hole depth (mm), 0 = through hole
    bolt_size: str = "M3"                 # Bolt specification


@dataclass
class ConnectionMethod:
    """Physical connection method between two parts.

    Describes HOW parts are physically joined, as opposed to the
    kinematic Joint type which describes the resulting DOF.
    """

    type: str  # "bolted" | "press_fit" | "snap_fit" | "adhesive" | "welded" | "magnetic"

    # --- Bolted connection ---
    bolt_size: str = "M3"                  # e.g. "M3", "M4", "M5", "M6"
    bolt_count: int = 0                    # Number of bolts
    bolt_holes: list[BoltHole] = field(default_factory=list)  # Hole positions
    torque_nm: float = 0.0                 # Recommended tightening torque (N·m)

    # --- Press-fit connection ---
    interference_mm: float = 0.0           # Interference amount (mm), e.g. 0.05-0.15

    # --- Snap-fit connection ---
    snap_count: int = 0                    # Number of snap features
    snap_force_n: float = 0.0              # Insertion/removal force (N)

    # --- Adhesive connection ---
    adhesive_type: str = ""                # "epoxy" | "cyanoacrylate" | "structural_acrylic" | "hot_melt"
    bond_area_mm2: float = 0.0            # Bonding surface area (mm²)

    # --- Welded connection ---
    weld_type: str = ""                    # "butt" | "fillet" | "spot" | "tig" | "mig"

    # --- Bolted connection ---
    hole_type: str = "through_hole"  # "through_hole" | "threaded_hole" | "nut_pocket" | "thread_insert"

    # --- Set screw connection ---
    set_screw_size: str = "M3"  # Set screw / grub screw specification

    # --- Feature generation tracking ---
    features_generated: bool = False       # True after ConnectionFeatureEngine processes this

    # Required geometric constraints for each connection type
    @property
    def required_constraints(self) -> list[str]:
        """Return the geometric constraints required for this connection type."""
        _CONSTRAINT_MAP = {
            "bolted": ["coincident", "concentric"],    # Face flush + holes aligned
            "press_fit": ["concentric", "distance"],    # Shaft-hole concentric + axial position
            "snap_fit": ["coincident"],                 # Face flush, snap geometry aligned
            "adhesive": ["coincident"],                  # Faces in contact
            "welded": ["coincident"],                    # Weld seam faces in contact
            "magnetic": ["coincident"],                  # Faces in contact
        }
        return _CONSTRAINT_MAP.get(self.type, ["coincident"])

    @property
    def required_parts(self) -> list[str]:
        """Return additional parts needed for this connection type."""
        _PARTS_MAP = {
            "bolted": ["bolt", "nut", "washer"],       # bolt + nut + optional washer
            "press_fit": [],                             # No additional parts
            "snap_fit": [],                              # Integrated into part geometry
            "adhesive": ["adhesive"],                    # Glue/epoxy
            "welded": [],                                # No additional parts
            "magnetic": ["magnet"],                      # Magnet inserts
        }
        return _PARTS_MAP.get(self.type, [])

    def describe(self) -> str:
        """Human-readable description of the connection method."""
        descriptions = {
            "bolted": f"螺栓连接({self.bolt_size}×{self.bolt_count}, 扭矩{self.torque_nm}N·m)",
            "press_fit": f"压入配合(过盈{self.interference_mm}mm)",
            "snap_fit": f"卡扣连接({self.snap_count}处, 插入力{self.snap_force_n}N)",
            "adhesive": f"黏结({self.adhesive_type}, 面积{self.bond_area_mm2}mm²)",
            "welded": f"焊接({self.weld_type})",
            "magnetic": "磁吸连接",
        }
        return descriptions.get(self.type, self.type)


@dataclass
class Joint:
    """A joint connecting two parts."""

    type: str  # "revolute", "prismatic", "fixed"
    parent: str
    child: str
    range_deg: tuple[float, float] = (-180, 180)
    description: str = ""
    # Assembly constraints for auto-positioning
    parent_anchor: str = "top"       # Parent connection face: top/bottom/left/right/front/back
    child_anchor: str = "bottom"     # Child connection face
    offset: tuple[float, float, float] = (0, 0, 0)  # Fine adjustment offset (mm)
    # Rotation axis for revolute joints (overrides parent_anchor inference)
    # "auto" = infer from parent_anchor, "x"/"y"/"z" = explicit axis
    axis: str = "auto"
    # If True, skip sibling auto-distribution for this joint (child placed at anchor center)
    no_distribute: bool = False
    # Group tag for sibling distribution: joints with same (parent, parent_anchor, distribution_group)
    # are distributed together. Empty string = legacy behavior (group with all same-anchor siblings).
    distribution_group: str = ""
    # --- Task 63: 约束模型字段 ---
    parent_attachment: tuple | None = None   # 精确 3D 附着点（覆盖 parent_anchor）
    child_attachment: tuple | None = None    # 精确 3D 附着点（覆盖 child_anchor）
    parent_normal: tuple | None = None      # 附着点法向量
    child_normal: tuple | None = None       # 附着点法向量
    constraint_type: str = ""               # "coincident"/"concentric"/"distance"/"angle"/"parallel"
    constraint_distance: float = 0.0        # distance 约束参数（mm）
    constraint_angle_deg: float = 0.0       # angle 约束参数（度）
    # --- Task 69: 物理连接方式 ---
    connection: ConnectionMethod | None = None  # How parts are physically fastened
    # --- Mimic joint: this joint follows another joint (e.g. gripper finger coupling)
    mimic_joint: str = ""           # Name of the joint to mimic (by child part name)
    mimic_multiplier: float = 1.0   # Multiplier applied to the mimicked joint's angle
    mimic_offset: float = 0.0       # Offset added after multiplication (degrees)


@dataclass
class Assembly:
    """A complete assembly of parts and joints."""

    name: str
    parts: list[Part] = field(default_factory=list)
    joints: list[Joint] = field(default_factory=list)
    description: str = ""
    # Default joint angles for initial pose (child_part_name -> angle_degrees)
    default_angles: dict[str, float] = field(default_factory=dict)
    # Assembly mass properties (Task 45)
    total_mass: float = 0.0  # kg, 0 = not yet computed
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0)  # mm from origin
    inertia_tensor: list[list[float]] = field(default_factory=lambda: [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])


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
        Joint("revolute", "base_plate", "base_joint_housing", (-180, 180), "底座旋转",
              parent_anchor="top", child_anchor="bottom", axis="z"),
        Joint("revolute", "base_joint_housing", "shoulder_link", (-90, 90), "肩部俯仰",
              parent_anchor="top", child_anchor="bottom", axis="y"),
        Joint("revolute", "shoulder_link", "elbow_joint", (-135, 135), "肘部弯曲",
              parent_anchor="top", child_anchor="bottom", axis="y"),
        Joint("fixed", "elbow_joint", "forearm_link", description="固定连接",
              parent_anchor="top", child_anchor="bottom"),
        Joint("revolute", "forearm_link", "wrist_joint", (-180, 180), "腕部旋转",
              parent_anchor="top", child_anchor="bottom", axis="z"),
        Joint("fixed", "wrist_joint", "end_effector_mount", description="固定连接",
              parent_anchor="top", child_anchor="bottom"),
    ],
)


# ============================================================================
# OpenMANIPULATOR-X (ROBOTIS) — Reverse-engineered from URDF + BOM
# ============================================================================
#
# Source: ROBOTIS OpenMANIPULATOR-X
#   URDF:  github.com/ROBOTIS-GIT/open_manipulator (main branch)
#   Manual: emanual.robotis.com/docs/en/platform/openmanipulator_x/
#
# Specs:
#   Actuators:  5 × DYNAMIXEL XM430-W350-T
#   DOF:        5  (4 revolute arm joints + 1 gripper prismatic DOF)
#   Payload:    500 g
#   Reach:      380 mm
#   Mass:       ~711 g
#   Repeatability: < 0.2 mm
#
# Kinematic chain (URDF joint origin offsets, converted to mm):
#   world → link1 (fixed)
#   link1 → link2  joint1  revolute Z  ±180°   offset (12, 0, 0)
#   link2 → link3  joint2  revolute Y  ±86°    offset (0, 0, 59.5)
#   link3 → link4  joint3  revolute Y  -86/+80°  offset (24, 0, 128)
#   link4 → link5  joint4  revolute Y  -97/+113° offset (124, 0, 0)
#   link5 → end_effector        fixed            offset (126, 0, 0)
#   link5 → gripper_left   prismatic Y ±11/+20   offset (81.7, 21, 0)
#   link5 → gripper_right  prismatic -Y (mimic)  offset (81.7, -21, 0)
#
# BOM highlights:
#   5 × XM430-W350-T
#   FR12-H101-K × 2, FR12-H104-K × 1, FR12-S101-K × 1, FR12-S102-K × 2
#   LINK FRAME (LONG/SHORT), RAIL BRACKET (LEFT/RIGHT), PALM GRIPPER × 2
#   Fasteners: M2 × 38, M2.5 × 52 (various lengths), M3 × 4
#   Spacers: X-SP × 24
#   Idler pulleys: DC12-IDLER × 3

OPEN_MANIPULATOR_X_PARTS: list[Part] = [
    # --- link1: Base mounting plate + XM430 joint1 ---
    Part(
        name="omx_link1",
        category="structural",
        description="底座安装板，含 XM430-W350 底部关节",
        material="Aluminum",
        dimensions={"diameter": 60, "height": 30},
        mass=0.07912,
        density=MaterialDensity.Aluminum,
        notes="XM430-W350-T 内置，4×M3 安装孔",
    ),
    # --- link2: Shoulder sub-assembly (servo + FR12-H101 frame) ---
    Part(
        name="omx_link2",
        category="structural",
        description="肩部子组件（XM430 + FR12-H101 框架），关节1输出",
        material="Aluminum",
        dimensions={"length": 50, "width": 50, "height": 78},
        mass=0.09841,
        density=MaterialDensity.Aluminum,
        notes="视觉 Z 偏移 19mm，关节2 位于顶部",
    ),
    # --- link3: Upper arm (FR12-S102 + link rod long) ---
    Part(
        name="omx_link3",
        category="structural",
        description="上臂连杆（FR12-S102 + 长连杆），肩到肘",
        material="Aluminum",
        dimensions={"length": 36, "width": 36, "height": 128},
        mass=0.13851,
        density=MaterialDensity.Aluminum,
        notes="含 XM430-W350 关节3",
    ),
    # --- link4: Forearm (FR12-S101 + link rod short) ---
    Part(
        name="omx_link4",
        category="structural",
        description="前臂连杆（FR12-S101 + 短连杆），肘到腕",
        material="Aluminum",
        dimensions={"length": 124, "width": 36, "height": 36},
        mass=0.13275,
        density=MaterialDensity.Aluminum,
        notes="含 XM430-W350 关节4，水平延伸",
    ),
    # --- link5: Wrist / gripper base (FR12-H104 + FR12-H101) ---
    Part(
        name="omx_link5",
        category="structural",
        description="腕部子组件（FR12-H104 + FR12-H101），夹爪基座",
        material="Aluminum",
        dimensions={"length": 90, "width": 40, "height": 30},
        mass=0.14328,
        density=MaterialDensity.Aluminum,
        notes="含 XM430-W350 关节5（夹爪驱动），安装导轨和夹爪",
    ),
    # --- Gripper finger (left) ---
    Part(
        name="omx_gripper_left",
        category="structural",
        description="夹爪左手指（PALM GRIPPER）",
        material="Aluminum",
        dimensions={"length": 30, "width": 8, "height": 15},
        mass=0.001,
        notes="导轨滑块驱动，橡胶垫附着面",
    ),
    # --- Gripper finger (right) ---
    Part(
        name="omx_gripper_right",
        category="structural",
        description="夹爪右手指（PALM GRIPPER）",
        material="Aluminum",
        dimensions={"length": 30, "width": 8, "height": 15},
        mass=0.001,
        notes="导轨滑块驱动，与左手指镜像",
    ),
    # --- End-effector reference point (virtual) ---
    Part(
        name="omx_end_effector",
        category="structural",
        description="末端执行器参考点（虚拟标记）",
        dimensions={"length": 10, "width": 10, "height": 10},
        notes="URDF 虚拟 link，仅用于 TCP 定位",
    ),
]

OPEN_MANIPULATOR_X_ASSEMBLY = Assembly(
    name="OpenMANIPULATOR-X",
    description="ROBOTIS OpenMANIPULATOR-X 5-DOF 机械臂，5×XM430-W350-T 驱动，380mm 臂展",
    parts=OPEN_MANIPULATOR_X_PARTS,
    joints=[
        # --- joint1: Base rotation (Z-axis, ±180°) ---
        # URDF origin: (12, 0, 0) mm — small forward offset
        Joint("revolute", "omx_link1", "omx_link2",
              range_deg=(-180, 180), description="底座旋转 (joint1)",
              parent_anchor="top", child_anchor="bottom",
              offset=(12, 0, 0), axis="z"),

        # --- joint2: Shoulder pitch (Y-axis, ±86°) ---
        # URDF origin: (0, 0, 59.5) mm — vertical rise
        Joint("revolute", "omx_link2", "omx_link3",
              range_deg=(-86, 86), description="肩部俯仰 (joint2)",
              parent_anchor="top", child_anchor="bottom",
              offset=(0, 0, 59.5), axis="y"),

        # --- joint3: Elbow pitch (Y-axis, -86° to +80°) ---
        # URDF origin: (24, 0, 128) mm — upper arm length
        Joint("revolute", "omx_link3", "omx_link4",
              range_deg=(-86, 80), description="肘部俯仰 (joint3)",
              parent_anchor="top", child_anchor="bottom",
              offset=(24, 0, 128), axis="y"),

        # --- joint4: Wrist pitch (Y-axis, -97° to +113°) ---
        # URDF origin: (124, 0, 0) mm — forearm extends horizontally
        Joint("revolute", "omx_link4", "omx_link5",
              range_deg=(-97, 113), description="腕部俯仰 (joint4)",
              parent_anchor="right", child_anchor="left",
              offset=(0, 0, 0), axis="y"),

        # --- end_effector: Fixed TCP marker ---
        # URDF origin: (126, 0, 0) mm from link5
        Joint("fixed", "omx_link5", "omx_end_effector",
              description="末端执行器参考点",
              parent_anchor="right", child_anchor="left",
              offset=(36, 0, 0)),

        # --- gripper_left: Prismatic finger (Y-axis, -11 to +20 mm) ---
        # URDF origin: (81.7, 21, 0) mm
        Joint("prismatic", "omx_link5", "omx_gripper_left",
              range_deg=(-11, 20), description="夹爪左手指",
              parent_anchor="right", child_anchor="left",
              offset=(42, 21, 0), axis="y"),

        # --- gripper_right: Prismatic finger (-Y-axis, mimic left) ---
        # URDF origin: (81.7, -21, 0) mm
        Joint("prismatic", "omx_link5", "omx_gripper_right",
              range_deg=(-11, 20), description="夹爪右手指（镜像）",
              parent_anchor="right", child_anchor="left",
              offset=(42, -21, 0), axis="y"),
    ],
    default_angles={
        "omx_link2": 0.0,   # joint1: home
        "omx_link3": 0.0,   # joint2: home
        "omx_link4": 0.0,   # joint3: home
        "omx_link5": 0.0,   # joint4: home
    },
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
    """Get a part by name across all built-in assemblies."""
    for part in ROBOTIC_ARM_PARTS + OPEN_MANIPULATOR_X_PARTS:
        if part.name == name:
            return part
    return None


def get_all_categories() -> list[str]:
    """Get all part categories across all built-in assemblies."""
    all_parts = ROBOTIC_ARM_PARTS + OPEN_MANIPULATOR_X_PARTS
    return sorted(set(p.category for p in all_parts))


def get_parts_by_category(category: str) -> list[Part]:
    """Get all parts in a category across all built-in assemblies."""
    all_parts = ROBOTIC_ARM_PARTS + OPEN_MANIPULATOR_X_PARTS
    return [p for p in all_parts if p.category == category]


# ============================================================================
# Assembly Mass Computation (Task 45)
# ============================================================================


def compute_assembly_mass(
    assembly: Assembly,
    positions: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Compute total mass, center of mass, and inertia tensor for an assembly.

    Uses weighted average for center of mass and the parallel axis theorem
    for inertia tensor. If a part has mass=0, estimates from volume/density.

    Args:
        assembly: The assembly to analyze.
        positions: Optional solver output ``{part_name: {"position": [x,y,z]}}``.
                   When provided, each part's global position is added to its
                   local center-of-mass so the assembly COM reflects real layout.

    Returns:
        dict with total_mass (kg), center_of_mass (mm), inertia_tensor (kg·mm²),
        and per-part breakdown.
    """
    parts_data: list[dict[str, Any]] = []
    total_mass = 0.0

    for part in assembly.parts:
        m = part.compute_estimated_mass()
        cx, cy, cz = part.center_of_mass
        # Add global position from solver output if available
        if positions and part.name in positions:
            pos = positions[part.name]["position"]
            cx += pos[0]
            cy += pos[1]
            cz += pos[2]
        parts_data.append({
            "name": part.name,
            "mass_kg": m,
            "material": part.material,
            "density_kg_m3": part.effective_density(),
            "com_mm": [cx, cy, cz],
        })
        total_mass += m

    # Center of mass: mass-weighted average
    com_x, com_y, com_z = 0.0, 0.0, 0.0
    if total_mass > 0:
        for pd in parts_data:
            m = pd["mass_kg"]
            cx, cy, cz = pd["com_mm"]
            com_x += m * cx
            com_y += m * cy
            com_z += m * cz
        com_x /= total_mass
        com_y /= total_mass
        com_z /= total_mass

    # Inertia tensor via parallel axis theorem
    # I_total = sum(I_local_part + m * (d² * E - d ⊗ d))
    # where d = com_part - com_assembly
    inertia = [[0.0] * 3 for _ in range(3)]

    for i, pd in enumerate(parts_data):
        m = pd["mass_kg"]
        dx = pd["com_mm"][0] - com_x
        dy = pd["com_mm"][1] - com_y
        dz = pd["com_mm"][2] - com_z

        # Add local inertia tensor from the part (about its own COM)
        local_I = assembly.parts[i].inertia_tensor
        inertia[0][0] += local_I[0][0]
        inertia[1][1] += local_I[1][1]
        inertia[2][2] += local_I[2][2]
        inertia[0][1] += local_I[0][1]
        inertia[0][2] += local_I[0][2]
        inertia[1][2] += local_I[1][2]

        # Parallel axis theorem contribution (shift from part COM to assembly COM)
        inertia[0][0] += m * (dy * dy + dz * dz)
        inertia[1][1] += m * (dx * dx + dz * dz)
        inertia[2][2] += m * (dx * dx + dy * dy)
        inertia[0][1] -= m * dx * dy
        inertia[0][2] -= m * dx * dz
        inertia[1][2] -= m * dy * dz

    # Symmetric
    inertia[1][0] = inertia[0][1]
    inertia[2][0] = inertia[0][2]
    inertia[2][1] = inertia[1][2]

    # Update assembly fields
    assembly.total_mass = total_mass
    assembly.center_of_mass = (com_x, com_y, com_z)
    assembly.inertia_tensor = inertia

    return {
        "total_mass_kg": round(total_mass, 6),
        "center_of_mass_mm": [round(com_x, 2), round(com_y, 2), round(com_z, 2)],
        "inertia_tensor_kg_mm2": [
            [round(v, 6) for v in row]
            for row in inertia
        ],
        "parts": parts_data,
        "num_parts": len(parts_data),
    }
