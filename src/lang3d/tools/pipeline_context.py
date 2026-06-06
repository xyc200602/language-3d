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
            solver = AssemblySolver(self.assembly)
            self.positions = solver.solve()
            self.solved = True
        return self.positions

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
