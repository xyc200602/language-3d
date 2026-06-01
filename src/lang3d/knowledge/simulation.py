"""Simulation knowledge base - FEA patterns and mesh recommendations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FEAPattern:
    """Predefined FEA analysis pattern."""

    name: str
    description: str
    constraints_template: str
    mesh_recommendation: str


# ============================================================================
# Standard FEA Analysis Patterns
# ============================================================================

FEA_PATTERNS: dict[str, FEAPattern] = {
    "cantilever_bend": FEAPattern(
        name="Cantilever Bend",
        description="One end fixed, load applied at free end. Simulates beam bending.",
        constraints_template="fixed_bottom, force_top_face",
        mesh_recommendation="medium",
    ),
    "compression_test": FEAPattern(
        name="Compression Test",
        description="Bottom face fixed, uniform pressure on top face. Simulates compressive loading.",
        constraints_template="fixed_bottom, pressure_top",
        mesh_recommendation="coarse",
    ),
    "tensile_test": FEAPattern(
        name="Tensile Test",
        description="One end fixed, pulling force on opposite end. Simulates tensile loading.",
        constraints_template="fixed_bottom, force_top_upward",
        mesh_recommendation="medium",
    ),
    "gravitational_load": FEAPattern(
        name="Gravitational Load",
        description="Bottom fixed, self-weight loading via gravity. Simulates weight support.",
        constraints_template="fixed_bottom, gravity",
        mesh_recommendation="coarse",
    ),
}

# ============================================================================
# Mesh Size Recommendations
# ============================================================================

MESH_SIZES: dict[str, dict] = {
    "coarse": {
        "max_element_size_factor": 0.2,
        "description": "Coarse mesh for quick estimates (fastest)",
        "min_elements_per_edge": 3,
    },
    "medium": {
        "max_element_size_factor": 0.1,
        "description": "Medium mesh for general analysis (balanced)",
        "min_elements_per_edge": 6,
    },
    "fine": {
        "max_element_size_factor": 0.05,
        "description": "Fine mesh for accurate results (slower)",
        "min_elements_per_edge": 10,
    },
    "very_fine": {
        "max_element_size_factor": 0.02,
        "description": "Very fine mesh for high-precision analysis (slowest)",
        "min_elements_per_edge": 20,
    },
}


# ============================================================================
# Joint Types for Motion Simulation
# ============================================================================

@dataclass
class JointType:
    """Definition of a kinematic joint type."""

    name: str
    dof: int
    motion_type: str  # "rotational", "translational", "fixed", "spherical"
    default_range: list[float]  # [min, max] in degrees or mm


JOINT_TYPES: dict[str, JointType] = {
    "revolute": JointType(
        name="Revolute",
        dof=1,
        motion_type="rotational",
        default_range=[-180.0, 180.0],
    ),
    "prismatic": JointType(
        name="Prismatic",
        dof=1,
        motion_type="translational",
        default_range=[0.0, 100.0],
    ),
    "fixed": JointType(
        name="Fixed",
        dof=0,
        motion_type="fixed",
        default_range=[0.0, 0.0],
    ),
    "spherical": JointType(
        name="Spherical",
        dof=3,
        motion_type="rotational",
        default_range=[-90.0, 90.0],
    ),
}


# ============================================================================
# CFD Fluid Properties
# ============================================================================

@dataclass
class FluidProperties:
    """Properties of a fluid for CFD simulation."""

    name: str
    density: float  # kg/m^3
    kinematic_viscosity: float  # m^2/s


FLUID_PRESETS: dict[str, FluidProperties] = {
    "air": FluidProperties(
        name="Air (20°C, 1 atm)",
        density=1.204,
        kinematic_viscosity=1.516e-5,
    ),
    "water": FluidProperties(
        name="Water (20°C, 1 atm)",
        density=998.2,
        kinematic_viscosity=1.004e-6,
    ),
}


def get_fluid(name: str) -> FluidProperties | None:
    """Get fluid properties by name (case-insensitive)."""
    return FLUID_PRESETS.get(name.lower())


@dataclass
class CFDPattern:
    """Predefined CFD analysis pattern."""

    name: str
    solver: str
    turbulence_model: str


CFD_PATTERNS: dict[str, CFDPattern] = {
    "pipe_flow": CFDPattern(
        name="Internal Pipe Flow",
        solver="simpleFoam",
        turbulence_model="kOmegaSST",
    ),
    "external_flow": CFDPattern(
        name="External Aerodynamics",
        solver="simpleFoam",
        turbulence_model="kEpsilon",
    ),
    "heat_exchanger": CFDPattern(
        name="Heat Exchanger",
        solver="buoyantSimpleFoam",
        turbulence_model="kOmegaSST",
    ),
}

CFD_MESH_SIZES: dict[str, dict] = {
    "coarse": {
        "max_cell_size_factor": 0.2,
        "boundary_layers": 2,
        "description": "Coarse mesh for quick estimates",
    },
    "medium": {
        "max_cell_size_factor": 0.1,
        "boundary_layers": 4,
        "description": "Medium mesh for general analysis",
    },
    "fine": {
        "max_cell_size_factor": 0.05,
        "boundary_layers": 6,
        "description": "Fine mesh for accurate results",
    },
}


def recommend_mesh_size(bounding_box_max_dim: float) -> str:
    """Recommend a mesh size based on the model's bounding box max dimension.

    Args:
        bounding_box_max_dim: Maximum dimension of the model bounding box in mm.

    Returns:
        Mesh size key: 'coarse', 'medium', 'fine', or 'very_fine'.
    """
    if bounding_box_max_dim <= 20:
        return "fine"
    elif bounding_box_max_dim <= 100:
        return "medium"
    elif bounding_box_max_dim <= 500:
        return "medium"
    else:
        return "coarse"
