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
    Titanium = 4500
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
        """Estimate volume in mm³ from dimensions (rough bounding box)."""
        if not self.dimensions:
            return 0.0
        # Try common dimension names
        l = self.dimensions.get("length", self.dimensions.get("diameter", 0))
        w = self.dimensions.get("width", 0)
        h = self.dimensions.get("height", self.dimensions.get("thickness", 0))
        if l > 0 and w > 0 and h > 0:
            return l * w * h
        # Multiply all dimension values as a fallback
        vals = list(self.dimensions.values())
        result = vals[0]
        for v in vals[1:]:
            result *= v
        return result

    def compute_estimated_mass(self) -> float:
        """Estimate mass (kg) from volume and density."""
        if self.mass > 0:
            return self.mass
        vol_mm3 = self.compute_volume_mm3()
        vol_m3 = vol_mm3 * 1e-9
        return vol_m3 * self.effective_density()


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


@dataclass
class Assembly:
    """A complete assembly of parts and joints."""

    name: str
    parts: list[Part] = field(default_factory=list)
    joints: list[Joint] = field(default_factory=list)
    description: str = ""
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


# ============================================================================
# Assembly Mass Computation (Task 45)
# ============================================================================


def compute_assembly_mass(assembly: Assembly) -> dict[str, Any]:
    """Compute total mass, center of mass, and inertia tensor for an assembly.

    Uses weighted average for center of mass and the parallel axis theorem
    for inertia tensor. If a part has mass=0, estimates from volume/density.

    Returns:
        dict with total_mass (kg), center_of_mass (mm), inertia_tensor (kg·mm²),
        and per-part breakdown.
    """
    parts_data: list[dict[str, Any]] = []
    total_mass = 0.0

    for part in assembly.parts:
        m = part.compute_estimated_mass()
        cx, cy, cz = part.center_of_mass
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
