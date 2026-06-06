"""Public tool interface for assembly VLM visual verification."""

from __future__ import annotations

import json
from typing import Any

from ..agent.assembly_visual_verifier import verify_assembly_visual
from ..knowledge.mechanics import Assembly
from ..models.base import ToolDefinition
from .assembly_solver import AssemblySolver
from .base import Tool


class AssemblyVLMSolveTool(Tool):
    """Solve assembly with VLM visual verification closed loop.

    Runs iterative verification: solve → render → VLM check → correct → re-solve.
    """

    name = "assembly_vlm_solve"
    description = (
        "Solve assembly with visual verification closed loop. "
        "Renders the assembly, uses VLM to detect layout problems "
        "(collision, floating parts, wrong orientation), and iteratively "
        "corrects constraints up to max_iterations rounds."
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
                        "description": "Name of the assembly to verify (default: 'complex_robot')",
                    },
                    "expected_layout": {
                        "type": "string",
                        "description": "Text description of expected layout for VLM comparison",
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Maximum verification iterations (default: 3)",
                    },
                    "detail_level": {
                        "type": "string",
                        "enum": ["fast", "standard", "detailed", "maximum"],
                        "description": "VLM analysis detail level (default: 'detailed')",
                    },
                },
                "required": [],
            },
        )

    def execute(
        self,
        *,
        assembly_name: str = "complex_robot",
        expected_layout: str = "",
        max_iterations: int = 3,
        detail_level: str = "detailed",
        **kwargs: Any,
    ) -> str:
        # Build assembly
        if assembly_name == "complex_robot":
            from .export_package import build_complex_robot
            assembly = build_complex_robot()
        else:
            return f"Error: Unknown assembly '{assembly_name}'. Use 'complex_robot'."

        # Initial solve
        solver = AssemblySolver(assembly)
        positions = solver.solve()

        # Get model backend
        model_backend = None
        try:
            from ..models.router import ModelRouter, TaskType
            router = ModelRouter()
            model_backend = router.get_backend(TaskType.VISION)
        except Exception:
            pass  # Will use heuristic verification

        # Run visual verification
        result = verify_assembly_visual(
            assembly=assembly,
            positions=positions,
            model_backend=model_backend,
            expected_layout=expected_layout,
            max_iterations=max_iterations,
            detail_level=detail_level,
        )

        # Format result
        lines = [
            f"Assembly Visual Verification: {'PASSED' if result.passed else 'NEEDS ATTENTION'}",
            f"Rounds: {result.round_number}/{max_iterations}",
        ]

        if result.problems:
            lines.append(f"Problems detected: {len(result.problems)}")
            for p in result.problems:
                lines.append(
                    f"  [{p.severity.value}] {p.problem_type.value}: {p.description}"
                )
                if p.affected_parts:
                    lines.append(f"    Parts: {', '.join(p.affected_parts)}")
                if p.suggestion:
                    lines.append(f"    Fix: {p.suggestion}")

        if result.corrections_applied:
            lines.append(f"Corrections applied: {len(result.corrections_applied)}")
            for c in result.corrections_applied:
                lines.append(f"  {c.get('correction_type', '?')}: {c.get('reason', '')}")

        return "\n".join(lines)
