"""Mass properties tools — compute part mass, center of mass, inertia, assembly properties.

Task 45: Provides tools for computing and verifying physical mass properties
of parts and assemblies. These are essential for stability analysis, motor
torque calculation, and URDF export.
"""

from __future__ import annotations

import json
from typing import Any

from ..knowledge.mechanics import (
    Assembly,
    MaterialDensity,
    Part,
    compute_assembly_mass,
)
from ..models.base import ToolDefinition
from .base import Tool


class ComputePartMassTool(Tool):
    """Compute mass for a single part from its dimensions and material."""

    name = "compute_part_mass"
    description = (
        "Compute the mass of a part from its dimensions (mm) and material. "
        "Returns mass (kg), volume (mm³), and density (kg/m³)."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "dimensions": {
                        "type": "object",
                        "description": "Part dimensions in mm, e.g. {\"length\": 100, \"width\": 50, \"height\": 10}",
                    },
                    "material": {
                        "type": "string",
                        "description": "Material name (PLA, ABS, PETG, Aluminum, Steel, etc.)",
                    },
                    "density": {
                        "type": "number",
                        "description": "Custom density in kg/m³ (overrides material default)",
                    },
                },
                "required": ["dimensions"],
            },
        )

    def execute(
        self,
        *,
        dimensions: dict[str, float],
        material: str = "PLA",
        density: float = 0,
        **kwargs: Any,
    ) -> str:
        part = Part(
            name="query",
            category="query",
            description="mass query",
            material=material,
            dimensions=dimensions,
            density=density,
        )
        vol_mm3 = part.compute_volume_mm3()
        eff_density = part.effective_density()
        mass_kg = part.compute_estimated_mass()

        result = {
            "volume_mm3": round(vol_mm3, 2),
            "density_kg_m3": eff_density,
            "mass_kg": round(mass_kg, 6),
            "mass_g": round(mass_kg * 1000, 4),
            "material": material,
            "dimensions": dimensions,
        }
        return json.dumps(result, indent=2, ensure_ascii=False)


class ComputeCOMTool(Tool):
    """Compute the center of mass for a set of parts."""

    name = "compute_com"
    description = (
        "Compute the center of mass (COM) for multiple parts. "
        "Each part needs mass (kg) and position (x,y,z in mm). "
        "Returns weighted-average COM coordinates."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "parts": {
                        "type": "array",
                        "description": "List of parts with mass_kg and position_mm [x,y,z]",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "mass_kg": {"type": "number"},
                                "position_mm": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                },
                            },
                            "required": ["mass_kg", "position_mm"],
                        },
                    },
                },
                "required": ["parts"],
            },
        )

    def execute(self, *, parts: list[dict], **kwargs: Any) -> str:
        total_mass = 0.0
        com_x, com_y, com_z = 0.0, 0.0, 0.0

        for p in parts:
            m = p["mass_kg"]
            pos = p["position_mm"]
            com_x += m * pos[0]
            com_y += m * pos[1]
            com_z += m * pos[2]
            total_mass += m

        if total_mass > 0:
            com_x /= total_mass
            com_y /= total_mass
            com_z /= total_mass

        result = {
            "center_of_mass_mm": [
                round(com_x, 2),
                round(com_y, 2),
                round(com_z, 2),
            ],
            "total_mass_kg": round(total_mass, 6),
            "num_parts": len(parts),
        }
        return json.dumps(result, indent=2, ensure_ascii=False)


class ComputeInertiaTool(Tool):
    """Compute moment of inertia for a simple shape."""

    name = "compute_inertia"
    description = (
        "Compute the moment of inertia tensor (kg·mm²) for a simple shape "
        "(box, cylinder, or sphere) given dimensions and material. "
        "Returns the 3x3 inertia tensor about the center of mass."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "shape": {
                        "type": "string",
                        "enum": ["box", "cylinder", "sphere"],
                        "description": "Shape type",
                    },
                    "dimensions": {
                        "type": "object",
                        "description": (
                            "Shape dimensions in mm. "
                            "Box: {length, width, height}. "
                            "Cylinder: {radius, height}. "
                            "Sphere: {radius}."
                        ),
                    },
                    "material": {
                        "type": "string",
                        "description": "Material name",
                    },
                    "density": {
                        "type": "number",
                        "description": "Custom density in kg/m³",
                    },
                },
                "required": ["shape", "dimensions"],
            },
        )

    def execute(
        self,
        *,
        shape: str,
        dimensions: dict[str, float],
        material: str = "PLA",
        density: float = 0,
        **kwargs: Any,
    ) -> str:
        eff_density = density if density > 0 else MaterialDensity.get(material)

        if shape == "box":
            a = dimensions.get("length", 0) / 1000  # m
            b = dimensions.get("width", 0) / 1000
            c = dimensions.get("height", 0) / 1000
            vol = a * b * c
            m = vol * eff_density
            # Ixx = m/12 * (b² + c²), etc.
            ixx = m / 12 * (b**2 + c**2) * 1e6  # convert to kg·mm²
            iyy = m / 12 * (a**2 + c**2) * 1e6
            izz = m / 12 * (a**2 + b**2) * 1e6

        elif shape == "cylinder":
            r = dimensions.get("radius", 0) / 1000  # m
            h = dimensions.get("height", 0) / 1000
            vol = 3.14159265 * r * r * h
            m = vol * eff_density
            # Ixx = Iyy = m/12 * (3r² + h²), Izz = m/2 * r²
            ixx = m / 12 * (3 * r**2 + h**2) * 1e6
            iyy = ixx
            izz = m / 2 * r**2 * 1e6

        elif shape == "sphere":
            r = dimensions.get("radius", 0) / 1000
            vol = 4 / 3 * 3.14159265 * r**3
            m = vol * eff_density
            # Ixx = Iyy = Izz = 2/5 * m * r²
            ixx = 2 / 5 * m * r**2 * 1e6
            iyy = ixx
            izz = ixx

        else:
            return json.dumps({"error": f"Unknown shape: {shape}"})

        inertia = [
            [round(ixx, 6), 0.0, 0.0],
            [0.0, round(iyy, 6), 0.0],
            [0.0, 0.0, round(izz, 6)],
        ]

        result = {
            "shape": shape,
            "dimensions": dimensions,
            "material": material,
            "density_kg_m3": eff_density,
            "volume_m3": round(vol, 10),
            "mass_kg": round(m, 6),
            "inertia_tensor_kg_mm2": inertia,
        }
        return json.dumps(result, indent=2, ensure_ascii=False)


class ComputeAssemblyPropertiesTool(Tool):
    """Compute complete mass properties for an Assembly."""

    name = "compute_assembly_properties"
    description = (
        "Compute total mass, center of mass, and inertia tensor for an assembly "
        "from a list of parts with their masses and positions. Uses weighted average "
        "for COM and parallel axis theorem for inertia."
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
                        "description": "Name of the assembly",
                    },
                    "parts": {
                        "type": "array",
                        "description": (
                            "List of parts with name, mass_kg, com_mm [x,y,z], "
                            "and optional inertia_tensor (3x3 kg·mm²)"
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "mass_kg": {"type": "number"},
                                "com_mm": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                },
                                "inertia_tensor": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                    },
                                },
                            },
                            "required": ["name", "mass_kg", "com_mm"],
                        },
                    },
                },
                "required": ["assembly_name", "parts"],
            },
        )

    def execute(
        self,
        *,
        assembly_name: str,
        parts: list[dict],
        **kwargs: Any,
    ) -> str:
        assembly = Assembly(name=assembly_name)

        for p in parts:
            part = Part(
                name=p["name"],
                category="assembly_part",
                description="",
                mass=p["mass_kg"],
                center_of_mass=tuple(p["com_mm"]),
            )
            if "inertia_tensor" in p:
                part.inertia_tensor = p["inertia_tensor"]
            assembly.parts.append(part)

        result = compute_assembly_mass(assembly)
        result["assembly_name"] = assembly_name
        return json.dumps(result, indent=2, ensure_ascii=False)


def register_mass_properties_tools(registry: Any) -> None:
    """Register all mass property tools."""
    tools = [
        ComputePartMassTool(),
        ComputeCOMTool(),
        ComputeInertiaTool(),
        ComputeAssemblyPropertiesTool(),
    ]
    for tool in tools:
        registry.register(tool)
