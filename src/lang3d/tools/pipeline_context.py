"""Shared pipeline context for export_engineering_package.

Avoids redundant computation across the 12-step export pipeline by computing
assembly positions, mass, and subsystem decomposition once and sharing them
with all downstream modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..knowledge.mechanics import Assembly, compute_assembly_mass


@dataclass
class AssemblyContext:
    """Shared intermediate results for the export pipeline.

    Created once at the start of export_engineering_package() and passed
    to all downstream modules so they don't need to recompute positions,
    mass, subsystems, etc.
    """

    assembly: Assembly
    positions: dict[str, Any] = field(default_factory=dict)
    mass_result: dict[str, Any] = field(default_factory=dict)
    subsystems: dict[str, list[str]] = field(default_factory=dict)
    solved: bool = False
    mass_computed: bool = False
    subsystems_built: bool = False

    def ensure_positions(self) -> dict[str, Any]:
        """Compute assembly positions if not already done."""
        if not self.solved:
            from .assembly_solver import AssemblySolver

            # Ensure default_angles exist for arm-like assemblies.
            # If the LLM didn't provide them, inject reasonable bend
            # angles so the arm isn't a straight line.
            self._ensure_default_angles()

            solver = AssemblySolver(self.assembly)
            self.positions = solver.solve()
            # Adjust ground contact: find lowest part considering geometry
            self._adjust_ground_contact()
            self.solved = True
        return self.positions

    def _ensure_default_angles(self) -> None:
        """If default_angles are missing, inject reasonable arm bend angles."""
        da = self.assembly.default_angles
        # If already set with non-zero values, keep them
        if da and any(v != 0 for v in da.values()):
            return

        # Check if this looks like an arm (has revolute joints)
        revolute_children = [
            j.child for j in self.assembly.joints if j.type == "revolute"
        ]
        if not revolute_children:
            return

        # Generate progressive bend angles: first joint -45°, then decreasing
        n = len(revolute_children)
        for i, child in enumerate(revolute_children):
            if child not in da:
                # Progressive angles: -45, -30, -15, 10 ...
                angle = -45 + i * 20
                if i == n - 1 and n > 2:
                    angle = 15  # wrist angles up slightly
                da[child] = angle

        self.assembly.default_angles = da

    def _adjust_ground_contact(self) -> None:
        """Shift all positions up so the lowest point sits at Z=0."""
        min_z = float("inf")
        for name, placement in self.positions.items():
            pos = placement["position"]
            z_center = pos[2]
            # Estimate the lowest extent of the part
            part = next(
                (p for p in self.assembly.parts if p.name == name), None
            )
            if part is None:
                continue
            dims = part.dimensions
            # For cylindrical parts, the lowest point after rotation
            # depends on the radius (diameter/2)
            rot = placement.get("rotation", [0, 0, 1, 0])
            angle_deg = abs(rot[3]) if len(rot) > 3 else 0
            if "diameter" in dims or "outer_diameter" in dims:
                d = dims.get("outer_diameter", dims.get("diameter", 10))
                radius = d / 2
                # If rotated significantly, the cylinder axis is not vertical
                # and the lowest point is at z_center - radius
                if angle_deg > 10:
                    low = z_center - radius
                else:
                    h = dims.get("height", dims.get("length", 10))
                    low = z_center - h / 2
            else:
                # Box part: half-height in Z
                h = dims.get("height", dims.get("thickness", 5))
                # If rotated, the bounding box changes, but approximate
                low = z_center - h / 2
            if low < min_z:
                min_z = low

        if min_z < 0:
            shift = -min_z
            for placement in self.positions.values():
                placement["position"][2] = round(
                    placement["position"][2] + shift, 4
                )

    def ensure_mass(self) -> dict[str, Any]:
        """Compute mass properties if not already done."""
        if not self.mass_computed:
            self.mass_result = compute_assembly_mass(self.assembly)
            self.mass_computed = True
        return self.mass_result

    def ensure_subsystems(self) -> dict[str, list[str]]:
        """Build subsystem decomposition if not already done."""
        if not self.subsystems_built:
            from .export_package import _build_subsystems
            self.subsystems = _build_subsystems(self.assembly, self.positions)
            self.subsystems_built = True
        return self.subsystems

    def get_com(self) -> list[float]:
        """Get center of mass, computing mass if needed."""
        mass = self.ensure_mass()
        return list(mass["center_of_mass_mm"])
