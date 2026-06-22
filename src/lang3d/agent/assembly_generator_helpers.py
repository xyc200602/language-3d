"""Thin re-export layer for assembly_generator functions.

The AssemblyPipeline (pipeline.py) imports from this module rather than
directly from ``tools.assembly_generator`` to:

1. Decouple the pipeline from the 3500-line assembly_generator.py —
   if functions are later refactored into separate modules, only this
   file needs updating.
2. Provide clear, minimal signatures for each pipeline stage — the
   pipeline doesn't need to know about the 50+ internal functions in
   assembly_generator.

Added 2026-06-22 as part of the multi-agent Step 2 (pipeline split).
"""

from __future__ import annotations

# Stage 1: Architect
from ..tools.assembly_generator import (
    generate_assembly_from_nl,
    _normalize_gripper_fingers as normalize_gripper_fingers,
    _ensure_arm_default_angles as ensure_arm_default_angles,
    _raise_on_wheel_in_arm as raise_on_wheel_in_arm,
    _validate_proportions as validate_proportions,
    _validate_assembly as validate_assembly,
)

# Stage 2: Solver
from ..tools.pipeline_context import AssemblyContext
from ..tools.mesh_collision import MeshCollisionChecker
from ..tools.collision_resolver import CollisionResolver


def run_collision_check_and_resolve(assembly, positions):
    """Check for mesh collisions and auto-resolve if found.

    Returns (possibly_modified_assembly, possibly_modified_positions).
    """
    try:
        checker = MeshCollisionChecker()
        result = checker.check_assembly_collisions(
            assembly, positions, skip_adjacent=True,
        )
        severe = [
            p for p in result.pairs
            if p.is_collision and p.penetration_depth_mm > 1.0
        ]
        if severe:
            resolver = CollisionResolver(max_rounds=2)
            resolution = resolver.resolve(assembly, positions)
            if resolution.modified_assembly is not None:
                return resolution.modified_assembly, resolution.modified_positions
    except ImportError:
        pass  # trimesh/python-fcl not installed
    except Exception:
        pass  # Non-fatal — collision check is advisory
    return assembly, positions


# Stage 3: CAD Engineer
from ..tools.export_package import generate_part_stls


# Stage 4: Verifier
from ..tools.assembly_generator import (
    _vlm_check_assembly as vlm_check_assembly,
)


# Stage 5: Fixer
from .modifier import apply_targeted_fix_from_vlm


# Stage 6: Export
from ..tools.export_package import export_engineering_package
from ..tools.vtk_renderer import render_assembly_from_positions
