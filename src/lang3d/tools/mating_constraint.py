"""[DEPRECATED — NOT IN PRODUCTION PIPELINE] Mating constraint solver.

.. warning::
    This module is **not imported** by the production pipeline. The active
    solver is :mod:`lang3d.tools.assembly_solver`. This file is retained
    only for its tests and as a reference for future CAD-grade constraint
    work. Unlike :mod:`constraint_solver`, its revolute handling is correct
    (applies a real rotation matrix), but it is still unused in production.

Mating constraint solver — real CAD-style constraint resolution.

Upgrades assembly positioning from the anchor-face (6-basic-face) system
to proper mating constraints:

- **coincident**: two faces aligned with zero gap (face mating)
- **concentric**: two cylindrical features share the same axis (shaft-in-hole)
- **distance**: two faces/points maintain a specified offset
- **angle**: two axes/faces maintain a specified angle
- **parallel**: two faces/axes are parallel
- **perpendicular**: two faces/axes are perpendicular

The solver:
1. Builds a constraint dependency graph from Joint definitions
2. Converts anchor-face joints to mating constraints automatically (backward compat)
3. Sequentially solves constraints from root outward (BFS order)

Pure-function module: no FreeCAD imports, no I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..knowledge.mechanics import Assembly, Joint, Part
from .assembly_solver import (
    ANCHOR_DIRECTIONS,
    _anchor_offset_for_part,
    _half_extent,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MatingConstraint:
    """A single geometric constraint between two parts.

    reference_entity describes what geometry on each part participates:
    - ("face", anchor_name): a named face (top/bottom/left/right/front/back)
    - ("point", (x, y, z)): a specific 3D point in part-local coordinates
    - ("axis", (x, y, z)): a direction vector (e.g. shaft axis)
    - ("cylinder", (x, y, z), radius): a cylindrical feature center + radius
    """

    constraint_type: str  # "coincident" | "concentric" | "distance" | "angle" | "parallel" | "perpendicular"
    parent_part: str
    child_part: str
    parent_entity: tuple  # entity descriptor for parent
    child_entity: tuple   # entity descriptor for child
    parameters: dict[str, float] = field(default_factory=dict)
    # "distance": distance_mm, "angle": angle_deg


@dataclass
class SolvedPosition:
    """Result for a single part after constraint solving."""

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    rotation_angle_deg: float = 0.0
    rotation_matrix: list[list[float]] = field(default_factory=lambda: [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
    ])


# ---------------------------------------------------------------------------
# Vector / matrix helpers
# ---------------------------------------------------------------------------

def _vec_add(a: tuple, b: tuple) -> tuple:
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def _vec_sub(a: tuple, b: tuple) -> tuple:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def _vec_scale(a: tuple, s: float) -> tuple:
    return (a[0]*s, a[1]*s, a[2]*s)

def _vec_dot(a: tuple, b: tuple) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def _vec_cross(a: tuple, b: tuple) -> tuple:
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )

def _vec_len(a: tuple) -> float:
    return math.sqrt(a[0]**2 + a[1]**2 + a[2]**2)

def _vec_normalize(a: tuple) -> tuple:
    l = _vec_len(a)
    if l < 1e-12:
        return (0.0, 0.0, 0.0)
    return (a[0]/l, a[1]/l, a[2]/l)

def _mat_identity() -> list[list[float]]:
    return [[1,0,0],[0,1,0],[0,0,1]]

def _mat_mul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    return [
        [sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]

def _mat_vec(M: list[list[float]], v: tuple) -> tuple:
    return (
        M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
        M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
        M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2],
    )

def _rotation_matrix(axis: tuple, angle_rad: float) -> list[list[float]]:
    """Rodrigues' rotation formula."""
    ax = _vec_normalize(axis)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    t = 1 - c
    x, y, z = ax
    return [
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c],
    ]

def _rot_mat_to_axis_angle(M: list[list[float]]) -> tuple[tuple[float,float,float], float]:
    """Convert rotation matrix to axis-angle. Returns (axis, angle_deg)."""
    angle = math.acos(max(-1, min(1, (M[0][0]+M[1][1]+M[2][2]-1)/2)))
    if abs(angle) < 1e-10:
        return ((0, 0, 1), 0.0)
    if abs(angle - math.pi) < 1e-10:
        # 180 degree rotation
        for i in range(3):
            ax = [0.0, 0.0, 0.0]
            ax[i] = math.sqrt(max(0, (M[i][i]+1)/2))
            if ax[i] > 1e-10:
                return ((ax[0], ax[1], ax[2]), 180.0)
        return ((1, 0, 0), 180.0)
    x = M[2][1] - M[1][2]
    y = M[0][2] - M[2][0]
    z = M[1][0] - M[0][1]
    axis = _vec_normalize((x, y, z))
    return (axis, math.degrees(angle))


# ---------------------------------------------------------------------------
# Anchor-to-constraint conversion (backward compatibility)
# ---------------------------------------------------------------------------

def anchor_to_constraints(joint: Joint) -> list[MatingConstraint]:
    """Convert a legacy anchor-face Joint to MatingConstraints.

    A typical anchor-face joint (parent top → child bottom) converts to:
    1. A coincident constraint (face mating)
    2. A concentric constraint if parent_attachment/child_attachment are set
    """
    constraints: list[MatingConstraint] = []

    # Use explicit constraint_type if available
    ctype = joint.constraint_type
    if ctype:
        pe = ("point", joint.parent_attachment) if joint.parent_attachment else ("face", joint.parent_anchor)
        ce = ("point", joint.child_attachment) if joint.child_attachment else ("face", joint.child_anchor)
        params: dict[str, float] = {}
        if ctype == "distance":
            params["distance_mm"] = joint.constraint_distance
        elif ctype == "angle":
            params["angle_deg"] = joint.constraint_angle_deg
        constraints.append(MatingConstraint(
            constraint_type=ctype,
            parent_part=joint.parent,
            child_part=joint.child,
            parent_entity=pe,
            child_entity=ce,
            parameters=params,
        ))
    else:
        # Default: coincident (face mating)
        constraints.append(MatingConstraint(
            constraint_type="coincident",
            parent_part=joint.parent,
            child_part=joint.child,
            parent_entity=("face", joint.parent_anchor),
            child_entity=("face", joint.child_anchor),
        ))

    return constraints


# ---------------------------------------------------------------------------
# Constraint solver
# ---------------------------------------------------------------------------

class ConstraintSolver:
    """Solve mating constraints to position parts in an assembly.

    Usage::

        solver = ConstraintSolver(assembly)
        positions = solver.solve()
        # positions["bracket"] = SolvedPosition(position=(0,0,10), rotation_angle_deg=0)
    """

    def __init__(self, assembly: Assembly) -> None:
        self.assembly = assembly
        self._parts: dict[str, Part] = {p.name: p for p in assembly.parts}
        self._constraints: list[MatingConstraint] = []
        self._positions: dict[str, SolvedPosition] = {}

    def solve(
        self,
        constraints: list[MatingConstraint] | None = None,
        base_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        joint_angles: dict[str, float] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Solve all constraints and return part placements.

        If constraints are not provided, they are derived from the Assembly's
        joints using anchor_to_constraints().

        Returns dict mapping part_name -> {"position": [x,y,z], "rotation": [ax,ay,az,angle_deg]}.
        """
        angles = joint_angles or {}

        # Build constraints from joints if not provided
        if constraints is not None:
            self._constraints = constraints
        else:
            self._constraints = []
            for j in self.assembly.joints:
                self._constraints.extend(anchor_to_constraints(j))

        # Find root parts (never appear as child)
        child_names = {c.child_part for c in self._constraints}
        all_names = {c.parent_part for c in self._constraints} | child_names
        roots = [n for n in all_names if n not in child_names]

        # If no roots found, use first part
        if not roots:
            roots = [list(self._parts.keys())[0]] if self._parts else []

        # Initialize root positions
        for root_name in roots:
            self._positions[root_name] = SolvedPosition(position=base_position)

        # Build adjacency: parent -> [(constraint, child)]
        adj: dict[str, list[tuple[MatingConstraint, str]]] = {}
        for c in self._constraints:
            adj.setdefault(c.parent_part, []).append((c, c.child_part))

        # BFS solve from roots
        visited = set(roots)
        queue = list(roots)
        while queue:
            parent_name = queue.pop(0)
            for constraint, child_name in adj.get(parent_name, []):
                if child_name in visited:
                    continue
                visited.add(child_name)
                self._apply_constraint(parent_name, child_name, constraint, angles)
                queue.append(child_name)

        # Place unconnected parts at base
        for pname in self._parts:
            if pname not in self._positions:
                self._positions[pname] = SolvedPosition(position=base_position)

        # Convert to output format
        result: dict[str, dict[str, Any]] = {}
        for name, sp in self._positions.items():
            axis, angle = _rot_mat_to_axis_angle(sp.rotation_matrix)
            result[name] = {
                "position": list(sp.position),
                "rotation": [axis[0], axis[1], axis[2], angle],
            }
        return result

    def _apply_constraint(
        self,
        parent_name: str,
        child_name: str,
        constraint: MatingConstraint,
        joint_angles: dict[str, float],
    ) -> None:
        """Apply a single constraint to position child relative to parent."""
        parent_pos = self._positions.get(parent_name)
        if parent_pos is None:
            parent_pos = SolvedPosition()
        child_part = self._parts.get(child_name)

        if child_part is None:
            self._positions[child_name] = SolvedPosition()
            return

        # Find corresponding joint for angle info
        joint = self._find_joint(parent_name, child_name)
        angle_deg = 0.0
        if joint:
            angle_deg = angles_value(joint_angles, joint, child_name)

        dispatch = {
            "coincident": self._solve_coincident,
            "concentric": self._solve_concentric,
            "distance": self._solve_distance,
            "angle": self._solve_angle,
            "parallel": self._solve_parallel,
            "perpendicular": self._solve_perpendicular,
        }
        handler = dispatch.get(constraint.constraint_type, self._solve_coincident)
        pos = handler(parent_name, parent_pos, child_name, child_part,
                       constraint, joint, angle_deg)

        self._positions[child_name] = pos

    def _find_joint(self, parent: str, child: str) -> Joint | None:
        for j in self.assembly.joints:
            if j.parent == parent and j.child == child:
                return j
        return None

    # ------------------------------------------------------------------
    # Constraint solvers
    # ------------------------------------------------------------------

    def _solve_coincident(
        self,
        parent_name: str,
        parent_pos: SolvedPosition,
        child_name: str,
        child_part: Part,
        constraint: MatingConstraint,
        joint: Joint | None,
        angle_deg: float,
    ) -> SolvedPosition:
        """Face mating: child face touches parent face, normals opposed."""
        # Get parent face point in world frame
        parent_anchor = _entity_anchor(constraint.parent_entity, parent_name, self._parts)
        child_anchor = _entity_anchor(constraint.child_entity, child_name, self._parts)

        parent_face_normal = _entity_normal(constraint.parent_entity)
        child_face_normal = _entity_normal(constraint.child_entity)

        # Parent anchor point in world frame
        p_local = _anchor_offset_for_part(
            self._parts[parent_name], parent_anchor
        ) if parent_anchor else (0, 0, 0)
        p_world = _vec_add(parent_pos.position, _mat_vec(parent_pos.rotation_matrix, p_local))

        # We want child face normal to oppose parent face normal
        parent_normal_world = _mat_vec(parent_pos.rotation_matrix, parent_face_normal)

        # Compute rotation to align child normal anti-parallel to parent normal
        child_normal_local = _vec_scale(child_face_normal, -1)  # oppose parent
        align_rot = _alignment_rotation(child_normal_local, parent_normal_world)

        # Apply joint angle rotation
        joint_rot = _mat_identity()
        if joint and joint.type == "revolute" and abs(angle_deg) > 1e-6:
            # Use explicit joint axis if specified, otherwise fall back to parent normal
            axis_map = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
            rot_axis = axis_map.get(
                getattr(joint, "axis", "auto"),
                parent_normal_world,
            )
            joint_rot = _rotation_matrix(rot_axis, math.radians(angle_deg))

        # Child anchor offset in local frame
        c_local = _anchor_offset_for_part(child_part, child_anchor) if child_anchor else (0, 0, 0)

        # Combined rotation
        combined_rot = _mat_mul(joint_rot, align_rot)

        # Child anchor in world frame (rotated)
        c_world_offset = _mat_vec(combined_rot, c_local)

        # Child center = parent anchor point - child anchor world offset
        child_center = _vec_sub(p_world, c_world_offset)

        # Apply explicit offset from joint
        if joint and joint.offset != (0, 0, 0):
            child_center = _vec_add(child_center, joint.offset)

        return SolvedPosition(
            position=child_center,
            rotation_matrix=combined_rot,
        )

    def _solve_concentric(
        self,
        parent_name: str,
        parent_pos: SolvedPosition,
        child_name: str,
        child_part: Part,
        constraint: MatingConstraint,
        joint: Joint | None,
        angle_deg: float,
    ) -> SolvedPosition:
        """Concentric: child axis aligns with parent axis."""
        # If we have point entities, position child center at parent point
        pe = constraint.parent_entity
        ce = constraint.child_entity

        if pe[0] == "point" and ce[0] == "point":
            parent_pt = pe[1]
            parent_world = _vec_add(parent_pos.position, _mat_vec(parent_pos.rotation_matrix, parent_pt))

            child_pt_local = ce[1]
            # Align child axis with parent rotation
            joint_rot = _mat_identity()
            if joint and joint.type == "revolute" and abs(angle_deg) > 1e-6:
                axis = _vec_normalize(parent_pt) if _vec_len(parent_pt) > 1e-6 else (0, 0, 1)
                joint_rot = _rotation_matrix(axis, math.radians(angle_deg))

            combined_rot = _mat_mul(joint_rot, parent_pos.rotation_matrix)
            child_world_offset = _mat_vec(combined_rot, child_pt_local)
            child_center = _vec_sub(parent_world, child_world_offset)

            if joint and joint.offset != (0, 0, 0):
                child_center = _vec_add(child_center, joint.offset)

            return SolvedPosition(position=child_center, rotation_matrix=combined_rot)

        # Fall back to coincident if no point entities
        return self._solve_coincident(
            parent_name, parent_pos, child_name, child_part,
            constraint, joint, angle_deg,
        )

    def _solve_distance(
        self,
        parent_name: str,
        parent_pos: SolvedPosition,
        child_name: str,
        child_part: Part,
        constraint: MatingConstraint,
        joint: Joint | None,
        angle_deg: float,
    ) -> SolvedPosition:
        """Distance: child is positioned at a specified distance from parent face."""
        dist = constraint.parameters.get("distance_mm", 0.0)
        # Start with coincident solution
        result = self._solve_coincident(
            parent_name, parent_pos, child_name, child_part,
            constraint, joint, angle_deg,
        )
        # Offset along parent normal
        parent_normal = _entity_normal(constraint.parent_entity)
        parent_normal_world = _mat_vec(parent_pos.rotation_matrix, parent_normal)
        result.position = _vec_add(result.position, _vec_scale(parent_normal_world, dist))
        return result

    def _solve_angle(
        self,
        parent_name: str,
        parent_pos: SolvedPosition,
        child_name: str,
        child_part: Part,
        constraint: MatingConstraint,
        joint: Joint | None,
        angle_deg: float,
    ) -> SolvedPosition:
        """Angle: child is rotated by a specified angle relative to parent."""
        target_angle = constraint.parameters.get("angle_deg", 0.0)
        # Start with coincident positioning
        result = self._solve_coincident(
            parent_name, parent_pos, child_name, child_part,
            constraint, joint, angle_deg,
        )
        # Apply additional angle rotation
        parent_normal = _entity_normal(constraint.parent_entity)
        parent_normal_world = _mat_vec(parent_pos.rotation_matrix, parent_normal)
        angle_rot = _rotation_matrix(parent_normal_world, math.radians(target_angle))
        result.rotation_matrix = _mat_mul(angle_rot, result.rotation_matrix)
        return result

    def _solve_parallel(
        self,
        parent_name: str,
        parent_pos: SolvedPosition,
        child_name: str,
        child_part: Part,
        constraint: MatingConstraint,
        joint: Joint | None,
        angle_deg: float,
    ) -> SolvedPosition:
        """Parallel: child face/axis parallel to parent face/axis.

        Uses coincident for positioning (correct location). The full
        orientation guarantee (ensuring parallel rotation) requires
        computing rotation from parent face normal to child face
        direction, which is non-trivial for general cases. The current
        behavior is correct for positioning but incomplete for rotation.
        """
        # TODO: Add explicit rotation enforcement after coincident positioning.
        # The full fix requires computing rotation from parent face normal to
        # child face direction, which is non-trivial for general constraint
        # topologies. The current behavior positions correctly but may not
        # enforce strict parallelism of orientation.
        return self._solve_coincident(
            parent_name, parent_pos, child_name, child_part,
            constraint, joint, angle_deg,
        )

    def _solve_perpendicular(
        self,
        parent_name: str,
        parent_pos: SolvedPosition,
        child_name: str,
        child_part: Part,
        constraint: MatingConstraint,
        joint: Joint | None,
        angle_deg: float,
    ) -> SolvedPosition:
        """Perpendicular: child axis is 90° to parent axis."""
        # Start with coincident, then rotate child 90° around parent normal
        result = self._solve_coincident(
            parent_name, parent_pos, child_name, child_part,
            constraint, joint, angle_deg,
        )
        parent_normal = _entity_normal(constraint.parent_entity)
        parent_normal_world = _mat_vec(parent_pos.rotation_matrix, parent_normal)
        perp_rot = _rotation_matrix(parent_normal_world, math.pi / 2)
        result.rotation_matrix = _mat_mul(perp_rot, result.rotation_matrix)
        return result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _entity_anchor(entity: tuple, part_name: str, parts: dict[str, Part]) -> str | None:
    """Extract anchor name from entity descriptor."""
    if entity[0] == "face":
        return entity[1]
    return None


def _entity_normal(entity: tuple) -> tuple[float, float, float]:
    """Get the normal vector for an entity descriptor."""
    if entity[0] == "face":
        return ANCHOR_DIRECTIONS.get(entity[1], (0, 0, 1))
    if entity[0] == "axis":
        return entity[1]
    if entity[0] == "point":
        # Points have no inherent normal; default to +Z.
        # TODO: Infer from constraint partner entity for more accuracy.
        return (0, 0, 1)
    return (0, 0, 1)


def _alignment_rotation(local_dir: tuple, target_dir: tuple) -> list[list[float]]:
    """Compute rotation matrix to align local_dir with target_dir.

    Uses the axis-angle formulation: rotate around the cross product
    of the two vectors by the angle between them.
    """
    a = _vec_normalize(local_dir)
    b = _vec_normalize(target_dir)
    dot = _vec_dot(a, b)

    if dot > 0.9999:
        return _mat_identity()  # Already aligned
    if dot < -0.9999:
        # Anti-parallel: 180° rotation around any perpendicular axis
        perp = _vec_cross(a, (1, 0, 0))
        if _vec_len(perp) < 1e-6:
            perp = _vec_cross(a, (0, 1, 0))
        return _rotation_matrix(_vec_normalize(perp), math.pi)

    axis = _vec_cross(a, b)
    axis = _vec_normalize(axis)
    angle = math.acos(max(-1, min(1, dot)))
    return _rotation_matrix(axis, angle)


def angles_value(angles: dict[str, float], joint: Joint, child_name: str) -> float:
    """Look up angle for a joint from the angles dict."""
    # Try by child name
    if child_name in angles:
        return angles[child_name]
    # Try by joint description
    if joint.description and joint.description in angles:
        return angles[joint.description]
    # Try default angles from assembly
    return 0.0


def constraint_solve_tool_factory() -> tuple[Any, Any]:
    """Create the constraint_solve tool and its definition."""
    from ..models.base import ToolDefinition

    definition = ToolDefinition(
        name="constraint_solve",
        description="Solve mating constraints for an assembly and return part placements",
        parameters={
            "assembly_name": {"type": "string", "description": "Name of the assembly to solve"},
            "constraints": {"type": "array", "description": "Optional list of MatingConstraint dicts"},
            "base_position": {"type": "array", "description": "Base position [x,y,z] for root part"},
            "joint_angles": {"type": "object", "description": "Joint angles {child_name: angle_deg}"},
        },
    )

    class _ConstraintSolveTool:
        name = "constraint_solve"

        def execute(self, *, assembly_name: str = "", constraints: list | None = None,
                    base_position: list | None = None, joint_angles: dict | None = None,
                    **kwargs) -> str:
            from ..knowledge.mechanics import find_assembly
            asm = find_assembly(assembly_name)
            if asm is None:
                return f"Assembly '{assembly_name}' not found"
            bp = tuple(base_position) if base_position else (0, 0, 0)
            solver = ConstraintSolver(asm)
            mc_list = None
            if constraints:
                mc_list = [
                    MatingConstraint(
                        constraint_type=c.get("type", "coincident"),
                        parent_part=c["parent"],
                        child_part=c["child"],
                        parent_entity=tuple(c.get("parent_entity", ("face", "top"))),
                        child_entity=tuple(c.get("child_entity", ("face", "bottom"))),
                        parameters=c.get("parameters", {}),
                    )
                    for c in constraints
                ]
            result = solver.solve(constraints=mc_list, base_position=bp, joint_angles=joint_angles)
            lines = [f"Assembly: {asm.name}", f"Parts: {len(result)}", ""]
            for name, pos in sorted(result.items()):
                lines.append(f"  {name}: pos={[round(v,1) for v in pos['position']]} rot={[round(v,2) for v in pos['rotation']]}")
            return "\n".join(lines)

        def get_definition(self) -> ToolDefinition:
            return definition

    return definition, _ConstraintSolveTool
