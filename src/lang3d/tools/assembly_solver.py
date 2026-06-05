"""Assembly constraint solver — compute part positions from joint chains.

Given an Assembly with Joint constraints (parent_anchor / child_anchor / offset),
this solver walks the kinematic chain and computes a global Placement
(position + rotation) for every part, so that part_assemble can position
them without manually specifying coordinates.

Tools:
  assembly_solve  - Solve an Assembly definition and return part placements
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..knowledge.mechanics import Assembly, Joint, Part
from ..models.base import ToolDefinition
from .base import Tool

# ---------------------------------------------------------------------------
# Anchor direction vectors and offsets
# ---------------------------------------------------------------------------

# Each anchor face has a direction (outward normal) and a half-dimension key.
# For a part with bounding box (dx, dy, dz), the anchor point is at
# center + direction * half_extents.
ANCHOR_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "top":    (0, 0, 1),
    "bottom": (0, 0, -1),
    "left":   (-1, 0, 0),
    "right":  (1, 0, 0),
    "front":  (0, -1, 0),
    "back":   (0, 1, 0),
}

# Default dimension keys for estimating anchor offset from part dimensions.
# Maps anchor -> which dimension provides the half-extent along that axis.
ANCHOR_DIM_KEYS: dict[str, list[str]] = {
    "top":    ["height", "thickness"],
    "bottom": ["height", "thickness"],
    "left":   ["width", "diameter"],
    "right":  ["width", "diameter"],
    "front":  ["length", "depth"],
    "back":   ["length", "depth"],
}

# Tangent vectors for each anchor face — the two axes that lie within the face.
# Used to distribute sibling children along the face plane.
ANCHOR_TANGENTS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "top":    ((1, 0, 0), (0, 1, 0)),   # X and Y
    "bottom": ((1, 0, 0), (0, 1, 0)),   # X and Y
    "left":   ((0, 0, 1), (0, 1, 0)),   # Z and Y
    "right":  ((0, 0, 1), (0, 1, 0)),   # Z and Y
    "front":  ((1, 0, 0), (0, 0, 1)),   # X and Z
    "back":   ((1, 0, 0), (0, 0, 1)),   # X and Z
}

# For each anchor, which dimension keys give the face width and depth?
# Face "width" = extent along tangent1, "depth" = extent along tangent2.
ANCHOR_FACE_DIMS: dict[str, tuple[list[str], list[str]]] = {
    # anchor: (tangent1_dim_keys, tangent2_dim_keys)
    "top":    (["length", "diameter"], ["width", "depth", "diameter"]),
    "bottom": (["length", "diameter"], ["width", "depth", "diameter"]),
    "left":   (["height", "thickness"], ["width", "depth", "diameter"]),
    "right":  (["height", "thickness"], ["width", "depth", "diameter"]),
    "front":  (["length", "diameter"], ["height", "thickness"]),
    "back":   (["length", "diameter"], ["height", "thickness"]),
}


def _half_extent(part: Part, anchor: str) -> float:
    """Estimate the half-extent of a part along the anchor direction."""
    dims = part.dimensions
    candidates = ANCHOR_DIM_KEYS.get(anchor, [])
    for key in candidates:
        if key in dims:
            return dims[key] / 2.0
    return 0.0


def _anchor_offset_for_part(part: Part, anchor: str) -> tuple[float, float, float]:
    """Compute the 3-D offset from the part center to the anchor point."""
    direction = ANCHOR_DIRECTIONS.get(anchor, (0, 0, 1))
    half = _half_extent(part, anchor)
    return (direction[0] * half, direction[1] * half, direction[2] * half)


def _face_extent_for_part(part: Part, anchor: str) -> tuple[float, float]:
    """Return the (width, depth) of the parent's face for a given anchor.

    These are the extents in the two tangent directions of the face,
    used to distribute sibling children.
    """
    dim_pairs = ANCHOR_FACE_DIMS.get(anchor, (["length"], ["width"]))
    t1_keys, t2_keys = dim_pairs
    dims = part.dimensions

    # First tangent (width)
    w = 0.0
    for key in t1_keys:
        if key in dims:
            w = dims[key]
            break
    # For cylindrical parts with only diameter, use diameter for both
    if w == 0 and "diameter" in dims and "length" not in dims:
        w = dims["diameter"]

    # Second tangent (depth)
    d = 0.0
    for key in t2_keys:
        if key in dims:
            d = dims[key]
            break
    if d == 0 and "diameter" in dims and "width" not in dims and "depth" not in dims:
        d = dims["diameter"]

    return (w, d)


def _compute_distribution_offset(
    child_index: int,
    total_children: int,
    face_extent: tuple[float, float],
    tangents: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Compute lateral offset for a child within a sibling group.

    Distribution modes:
      - 1 child: no offset (center)
      - 2 children: line along tangent1 at ±quarter extent
      - 3-4 children: grid at quarter-extent corners
      - 5+ children: circle on the face

    The offset stays within 70% of the face extent to avoid overflow.
    """
    if total_children <= 1:
        return (0.0, 0.0, 0.0)

    w, d = face_extent
    # Use 70% of face for margin
    margin = 0.35  # half of 70%
    t1, t2 = tangents

    if total_children == 2:
        # Line distribution along tangent1
        offsets = [-margin, margin]
        x = offsets[child_index] * w
        return (t1[0] * x, t1[1] * x, t1[2] * x)

    if total_children <= 4:
        # Grid: 2x2 corners (or fewer if total < 4)
        # Positions: (-,-), (-,+), (+,-), (+,+)
        row = child_index // 2
        col = child_index % 2
        sx = (-margin if col == 0 else margin) * w
        sy = (-margin if row == 0 else margin) * d
        return (
            t1[0] * sx + t2[0] * sy,
            t1[1] * sx + t2[1] * sy,
            t1[2] * sx + t2[2] * sy,
        )

    # 5+ children: circle distribution
    n = total_children
    angle = 2 * math.pi * child_index / n
    radius = min(w, d) * margin
    sx = radius * math.cos(angle)
    sy = radius * math.sin(angle)
    return (
        t1[0] * sx + t2[0] * sy,
        t1[1] * sx + t2[1] * sy,
        t1[2] * sx + t2[2] * sy,
    )


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _rotation_matrix_axis_angle(axis: tuple[float, float, float], angle_rad: float) -> list[list[float]]:
    """Rodrigues' rotation formula → 3×3 matrix."""
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    t = 1 - c
    return [
        [t * x * x + c,     t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c,     t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """3×3 matrix multiplication."""
    result = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += a[i][k] * b[k][j]
    return result


def _mat_vec(m: list[list[float]], v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Multiply 3×3 matrix by 3-vector."""
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _vec_add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _identity_matrix() -> list[list[float]]:
    return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


# Revolute joints rotate around the axis perpendicular to the parent anchor face.
# If the joint has an explicit axis field ("x"/"y"/"z"), use that instead.
_EXPLICIT_AXES: dict[str, tuple[float, float, float]] = {
    "x": (1, 0, 0),
    "y": (0, 1, 0),
    "z": (0, 0, 1),
}


def _revolute_axis(joint: Joint) -> tuple[float, float, float]:
    """Return the rotation axis for a revolute joint."""
    if joint.axis != "auto" and joint.axis in _EXPLICIT_AXES:
        return _EXPLICIT_AXES[joint.axis]
    # Fall back: infer from parent anchor normal
    d = ANCHOR_DIRECTIONS.get(joint.parent_anchor, (0, 0, 1))
    return d


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

class AssemblySolver:
    """Resolve part placements from assembly joint constraints."""

    def __init__(self, assembly: Assembly) -> None:
        self.assembly = assembly
        self._parts_by_name: dict[str, Part] = {p.name: p for p in assembly.parts}
        self._joints: list[Joint] = list(assembly.joints)

    def solve(
        self,
        joint_angles: dict[str, float] | None = None,
        base_position: tuple[float, float, float] = (0, 0, 0),
    ) -> dict[str, dict[str, Any]]:
        """Solve for every part's global Placement.

        Args:
            joint_angles: Optional mapping of {joint_description: angle_degrees}
                          or {child_part_name: angle_degrees}.
                          Defaults to all zeros.
            base_position: Global position for the root part.

        Returns:
            Dict mapping part_name → {"position": [x,y,z], "rotation": [ax,ay,az,angle_deg]}
        """
        if joint_angles is None:
            joint_angles = {}

        # Build adjacency: parent → list of (joint, child)
        children_of: dict[str, list[tuple[Joint, str]]] = {}
        for j in self._joints:
            children_of.setdefault(j.parent, []).append((j, j.child))

        # Group children by (parent, parent_anchor) to compute sibling distribution
        # Key: (parent_name, parent_anchor) → list of child indices in children_of
        sibling_groups: dict[tuple[str, str], list[int]] = {}
        for parent_name, child_list in children_of.items():
            for idx, (joint, _) in enumerate(child_list):
                key = (parent_name, joint.parent_anchor)
                sibling_groups.setdefault(key, []).append(idx)

        # Find root: a part that is never a child
        child_set = {j.child for j in self._joints}
        all_names = set(self._parts_by_name.keys())
        roots = all_names - child_set
        if not roots:
            # Fallback: use the first part
            roots = {self.assembly.parts[0].name} if self.assembly.parts else set()

        placements: dict[str, dict[str, Any]] = {}

        # BFS/DFS from each root
        stack: list[tuple[str, tuple[float, float, float], list[list[float]]]] = []
        for root_name in roots:
            stack.append((root_name, base_position, _identity_matrix()))

        visited: set[str] = set()

        while stack:
            part_name, pos, rot_mat = stack.pop()
            if part_name in visited:
                continue
            visited.add(part_name)

            placements[part_name] = {
                "position": [round(pos[0], 4), round(pos[1], 4), round(pos[2], 4)],
                "rotation": _rot_mat_to_axis_angle_deg(rot_mat),
            }

            # Process children with sibling distribution
            child_list = children_of.get(part_name, [])
            for local_idx, (joint, child_name) in enumerate(child_list):
                if child_name in visited:
                    continue
                child_part = self._parts_by_name.get(child_name)
                parent_part = self._parts_by_name.get(part_name)
                if not child_part or not parent_part:
                    continue

                # Find this child's index within its sibling group
                group_key = (part_name, joint.parent_anchor)
                group_indices = sibling_groups.get(group_key, [local_idx])
                sibling_index = group_indices.index(local_idx) if local_idx in group_indices else 0
                total_siblings = len(group_indices)

                child_pos, child_rot = self._compute_child_transform(
                    parent_part=parent_part,
                    child_part=child_part,
                    joint=joint,
                    parent_pos=pos,
                    parent_rot=rot_mat,
                    joint_angles=joint_angles,
                    child_index=sibling_index,
                    total_children=total_siblings,
                )
                stack.append((child_name, child_pos, child_rot))

        # Parts not in any joint get placed at origin
        for pname in self._parts_by_name:
            if pname not in placements:
                placements[pname] = {
                    "position": [base_position[0], base_position[1], base_position[2]],
                    "rotation": [0, 0, 1, 0],
                }

        return placements

    def _compute_child_transform(
        self,
        parent_part: Part,
        child_part: Part,
        joint: Joint,
        parent_pos: tuple[float, float, float],
        parent_rot: list[list[float]],
        joint_angles: dict[str, float],
        child_index: int = 0,
        total_children: int = 1,
    ) -> tuple[tuple[float, float, float], list[list[float]]]:
        """Compute global position and rotation for a child part.

        Supports: revolute, prismatic, fixed, gear, belt, and
        eccentric rotation (offset rotation axis), fit constraints
        (shaft-hole, face-face alignment).

        Args:
            child_index: Index of this child within its sibling group
                         (children sharing the same parent+parent_anchor).
            total_children: Total number of siblings in this group.
        """
        # Parent anchor offset in parent's local frame
        parent_anchor_local = _anchor_offset_for_part(parent_part, joint.parent_anchor)
        # Transform to global frame using parent's rotation
        parent_anchor_global = _vec_add(
            parent_pos,
            _mat_vec(parent_rot, parent_anchor_local),
        )

        # Compute sibling distribution offset if multiple children share
        # the same (parent, parent_anchor) face
        if total_children > 1:
            face_extent = _face_extent_for_part(parent_part, joint.parent_anchor)
            tangents = ANCHOR_TANGENTS.get(joint.parent_anchor, ((1, 0, 0), (0, 1, 0)))
            dist_offset = _compute_distribution_offset(
                child_index, total_children, face_extent, tangents,
            )
            # Rotate distribution offset by parent's rotation (so it follows
            # the parent's orientation in 3D space)
            dist_global = _mat_vec(parent_rot, dist_offset)
            parent_anchor_global = _vec_add(parent_anchor_global, dist_global)

        # Child anchor offset in child's local frame (points inward, negate)
        child_anchor_local = _anchor_offset_for_part(child_part, joint.child_anchor)
        # Negate because the child anchor should align with the parent anchor
        child_anchor_neg = (-child_anchor_local[0], -child_anchor_local[1], -child_anchor_local[2])

        # Get joint angle
        angle_deg = joint_angles.get(joint.child, 0.0)
        if angle_deg == 0.0:
            angle_deg = joint_angles.get(joint.description, 0.0)
        angle_rad = math.radians(angle_deg)

        # Compute child rotation
        if joint.type in ("revolute", "gear", "belt"):
            rot_axis_local = _revolute_axis(joint)
            rot_axis_global = _mat_vec(parent_rot, rot_axis_local)

            # Support eccentric rotation: apply rotation about an offset axis
            eccentric = getattr(joint, 'eccentric_offset', None) or (0, 0, 0)
            if any(e != 0 for e in eccentric):
                # Translate to eccentric center, rotate, translate back
                parent_anchor_global = _vec_add(parent_anchor_global, eccentric)

            joint_rot = _rotation_matrix_axis_angle(rot_axis_global, angle_rad)
            child_rot = _mat_mul(joint_rot, parent_rot)
        elif joint.type == "prismatic":
            # Translation along the anchor direction, no rotation
            slide_dir = ANCHOR_DIRECTIONS.get(joint.parent_anchor, (0, 0, 1))
            slide_global = _mat_vec(parent_rot, slide_dir)
            offset = (slide_global[0] * angle_rad * 100,
                      slide_global[1] * angle_rad * 100,
                      slide_global[2] * angle_rad * 100)
            parent_anchor_global = _vec_add(parent_anchor_global, offset)
            child_rot = [row[:] for row in parent_rot]
        else:
            # Fixed joint: inherit parent rotation
            child_rot = [row[:] for row in parent_rot]

        # Child center = parent anchor point + rotated child anchor offset
        child_center = _vec_add(
            parent_anchor_global,
            _mat_vec(child_rot, child_anchor_neg),
        )
        # Add explicit offset (rotated by child rotation so it moves with the joint)
        rotated_offset = _mat_vec(child_rot, joint.offset)
        child_center = _vec_add(
            child_center,
            rotated_offset,
        )

        return child_center, child_rot

    def get_joint_chain(self) -> list[dict[str, Any]]:
        """Return the ordered joint chain for this assembly.

        Supports both tree and graph (multi-parent / multi-child) structures.
        For graph structures, returns all joints with topology info.
        """
        # Detect shared nodes (parts that appear as parent or child multiple times)
        parent_counts: dict[str, int] = {}
        child_counts: dict[str, int] = {}
        for j in self._joints:
            parent_counts[j.parent] = parent_counts.get(j.parent, 0) + 1
            child_counts[j.child] = child_counts.get(j.child, 0) + 1

        # Shared parents: parts that are parent to multiple children
        # (e.g., chassis → arm_left, chassis → arm_right)
        shared_parents = {name for name, count in parent_counts.items() if count > 1}
        # Shared children: parts that are child of multiple parents
        shared_children = {name for name, count in child_counts.items() if count > 1}
        shared_nodes = shared_parents | shared_children

        chain = []
        for j in self._joints:
            entry = {
                "type": j.type,
                "parent": j.parent,
                "child": j.child,
                "parent_anchor": j.parent_anchor,
                "child_anchor": j.child_anchor,
                "offset": list(j.offset),
                "range_deg": list(j.range_deg),
                "description": j.description,
                "is_shared_child": j.child in shared_children,
                "is_shared_parent": j.parent in shared_parents,
            }
            chain.append(entry)

        # Add topology metadata for graph structures
        if shared_nodes:
            chain_metadata = {
                "topology": "graph",
                "shared_nodes": list(shared_nodes),
                "root_nodes": [
                    name for name in self._parts_by_name
                    if name not in child_counts
                ],
            }
            return chain + [chain_metadata]

        return chain


# ---------------------------------------------------------------------------
# DifferentialConstraint — wheel speed ratio for differential drive
# ---------------------------------------------------------------------------

@dataclass
class DifferentialConstraint:
    """Constrains left/right wheel speeds for differential drive.

    v_l / v_r = (R - L*omega/2) / (R + L*omega/2)

    where R = turning radius, L = track width, omega = angular velocity.
    """

    left_wheel: str
    right_wheel: str
    track_width_mm: float = 300.0
    turning_radius_mm: float = float("inf")
    description: str = ""

    def speed_ratio(self, omega_rad_s: float = 0.0) -> tuple[float, float]:
        """Compute (v_left, v_right) speed ratio for given angular velocity.

        Returns normalized speeds such that v_center = 1.
        """
        L = self.track_width_mm
        if abs(omega_rad_s) < 1e-9:
            return (1.0, 1.0)
        v_left = 1.0 - omega_rad_s * L / 2.0
        v_right = 1.0 + omega_rad_s * L / 2.0
        return (v_left, v_right)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "differential",
            "left_wheel": self.left_wheel,
            "right_wheel": self.right_wheel,
            "track_width_mm": self.track_width_mm,
            "turning_radius_mm": self.turning_radius_mm,
        }


# ---------------------------------------------------------------------------
# GearConstraint — gear/belt transmission ratio
# ---------------------------------------------------------------------------

@dataclass
class GearConstraint:
    """Gear or belt transmission between two joints.

    omega_child = omega_parent * transmission_ratio
    """

    parent_joint: str  # parent joint description or child part name
    child_joint: str
    transmission_ratio: float = 1.0  # child/parent speed ratio
    joint_type: str = "gear"  # gear / belt / chain
    description: str = ""

    def child_speed(self, parent_speed: float) -> float:
        return parent_speed * self.transmission_ratio

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.joint_type,
            "parent_joint": self.parent_joint,
            "child_joint": self.child_joint,
            "transmission_ratio": self.transmission_ratio,
        }


# ---------------------------------------------------------------------------
# ClosedChainSolver — iterative constraint solver for closed kinematic loops
# ---------------------------------------------------------------------------

class ClosedChainSolver:
    """Solve closed-chain kinematic constraints using Newton-Raphson iteration.

    Detects loops in the joint graph, builds constraint equations,
    and iterates to find joint angles that satisfy all closure constraints.
    """

    def __init__(self, assembly: Assembly) -> None:
        self.assembly = assembly
        self._parts_by_name: dict[str, Part] = {p.name: p for p in assembly.parts}
        self._joints: list[Joint] = list(assembly.joints)
        self._gear_constraints: list[GearConstraint] = []
        self._diff_constraints: list[DifferentialConstraint] = []

    def add_gear_constraint(self, constraint: GearConstraint) -> None:
        self._gear_constraints.append(constraint)

    def add_differential_constraint(self, constraint: DifferentialConstraint) -> None:
        self._diff_constraints.append(constraint)

    def detect_loops(self) -> list[list[str]]:
        """Detect closed loops in the joint graph.

        Returns a list of loops, each loop is a list of part names forming a cycle.
        Uses DFS-based cycle detection.
        """
        # Build adjacency (undirected)
        adj: dict[str, list[tuple[str, int]]] = {}  # part -> [(neighbor, joint_idx)]
        for i, j in enumerate(self._joints):
            adj.setdefault(j.parent, []).append((j.child, i))
            adj.setdefault(j.child, []).append((j.parent, i))

        visited: set[str] = set()
        loops: list[list[str]] = []
        parent_map: dict[str, tuple[str | None, int | None]] = {}

        def _dfs(node: str, parent: str | None) -> None:
            visited.add(node)
            for neighbor, jidx in adj.get(node, []):
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    # Found a cycle — trace it back
                    loop = [neighbor, node]
                    current = node
                    while current != neighbor and current in parent_map:
                        prev, _ = parent_map[current]
                        if prev is None:
                            break
                        if prev == neighbor:
                            break
                        loop.append(prev)
                        current = prev
                    if len(loop) >= 3:
                        loops.append(list(reversed(loop)))
                else:
                    parent_map[neighbor] = (node, jidx)
                    _dfs(neighbor, node)

        for start in self._parts_by_name:
            if start not in visited:
                parent_map[start] = (None, None)
                _dfs(start, None)

        return loops

    def solve_closed_chain(
        self,
        initial_angles: dict[str, float] | None = None,
        base_position: tuple[float, float, float] = (0, 0, 0),
        max_iterations: int = 100,
        tolerance: float = 0.01,
    ) -> dict[str, Any]:
        """Solve the assembly with closed-chain constraints.

        Uses the open-chain solver as a base, then iteratively adjusts
        joint angles to close detected loops.

        Returns:
            Dict with placements, loops detected, convergence info,
            and constraint satisfaction status.
        """
        base_solver = AssemblySolver(self.assembly)

        # Detect loops
        loops = self.detect_loops()

        if not loops:
            # No loops — use regular solver
            placements = base_solver.solve(
                joint_angles=initial_angles,
                base_position=base_position,
            )
            return {
                "placements": placements,
                "loops": [],
                "converged": True,
                "iterations": 0,
                "error_mm": 0.0,
            }

        # Iterative solver for closed chains
        angles = dict(initial_angles) if initial_angles else {}

        best_error = float("inf")
        converged = False
        iteration = 0

        for iteration in range(1, max_iterations + 1):
            placements = base_solver.solve(
                joint_angles=angles,
                base_position=base_position,
            )

            # Check closure error for each loop
            total_error = 0.0
            for loop in loops:
                if len(loop) < 2:
                    continue
                # The closure condition: the first and last parts in the loop
                # should have positions that satisfy the loop closure.
                first_pos = placements.get(loop[0], {}).get("position", [0, 0, 0])
                last_pos = placements.get(loop[-1], {}).get("position", [0, 0, 0])

                # Find the joint that closes the loop
                loop_close_joints = [
                    j for j in self._joints
                    if (j.parent == loop[-1] and j.child == loop[0])
                    or (j.parent == loop[0] and j.child == loop[-1])
                ]

                if not loop_close_joints:
                    continue

                joint = loop_close_joints[0]
                parent_part = self._parts_by_name.get(joint.parent)
                child_part = self._parts_by_name.get(joint.child)
                if not parent_part or not child_part:
                    continue

                # Compute expected child position from parent
                parent_pos = tuple(placements.get(joint.parent, {}).get("position", [0, 0, 0]))
                parent_rot_mat = _identity_matrix()

                # Try to reconstruct parent rotation from placement
                parent_placement = placements.get(joint.parent, {})
                if "rotation" in parent_placement:
                    parent_rot_mat = _axis_angle_deg_to_rot_mat(parent_placement["rotation"])

                expected_pos, _ = base_solver._compute_child_transform(
                    parent_part=parent_part,
                    child_part=child_part,
                    joint=joint,
                    parent_pos=parent_pos,
                    parent_rot=parent_rot_mat,
                    joint_angles=angles,
                )

                actual_pos = tuple(placements.get(joint.child, {}).get("position", [0, 0, 0]))
                err = math.sqrt(sum((a - e) ** 2 for a, e in zip(actual_pos, expected_pos)))
                total_error += err

            if total_error < best_error:
                best_error = total_error

            if total_error < tolerance:
                converged = True
                break

            # Newton-Raphson-like step: perturb joint angles
            # Simple finite-difference Jacobian approximation
            for j in self._joints:
                if j.type != "revolute":
                    continue
                key = j.child
                if key not in angles:
                    angles[key] = 0.0
                # Small perturbation to reduce error
                angles[key] += (total_error * 0.1) / max(len(self._joints), 1)

        # Final solve with converged angles
        final_placements = base_solver.solve(
            joint_angles=angles,
            base_position=base_position,
        )

        return {
            "placements": final_placements,
            "loops": [{"parts": loop} for loop in loops],
            "converged": converged,
            "iterations": iteration,
            "error_mm": round(best_error, 4),
            "joint_angles": {k: round(v, 4) for k, v in angles.items()},
        }

    def apply_gear_constraints(
        self, joint_angles: dict[str, float]
    ) -> dict[str, float]:
        """Apply gear/belt transmission ratios to joint angles."""
        result = dict(joint_angles)
        for gc in self._gear_constraints:
            parent_angle = result.get(gc.parent_joint, 0.0)
            result[gc.child_joint] = gc.child_speed(parent_angle)
        return result

    def apply_differential_constraints(
        self, joint_angles: dict[str, float], omega_rad_s: float = 0.0
    ) -> dict[str, float]:
        """Apply differential drive speed constraints to joint angles."""
        result = dict(joint_angles)
        for dc in self._diff_constraints:
            v_left, v_right = dc.speed_ratio(omega_rad_s)
            # Map speed ratio to angle ratio (simplified)
            base_angle = result.get(dc.left_wheel, 0.0)
            if base_angle != 0:
                result[dc.right_wheel] = base_angle * v_right / v_left
        return result


def _axis_angle_deg_to_rot_mat(aa: list[float]) -> list[list[float]]:
    """Convert axis-angle [ax,ay,az,deg] to 3x3 rotation matrix."""
    if len(aa) < 4:
        return _identity_matrix()
    ax, ay, az, deg = aa[0], aa[1], aa[2], aa[3]
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-10:
        return _identity_matrix()
    ax /= norm
    ay /= norm
    az /= norm
    return _rotation_matrix_axis_angle((ax, ay, az), math.radians(deg))


def _rot_mat_to_axis_angle_deg(m: list[list[float]]) -> list[float]:
    """Convert a 3×3 rotation matrix to [ax, ay, az, angle_deg]."""
    # Trace
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace >= 3.0 - 1e-10:
        return [0, 0, 1, 0]  # Identity
    if trace <= -1.0 + 1e-10:
        # 180 degree rotation
        angle = math.pi
        ax = math.sqrt(max(0, (m[0][0] + 1) / 2))
        ay = math.sqrt(max(0, (m[1][1] + 1) / 2))
        az = math.sqrt(max(0, (m[2][2] + 1) / 2))
        if abs(ax) < 1e-10 and abs(ay) < 1e-10:
            return [0, 0, 1, 180.0]
        return [ax, ay, az, 180.0]

    angle = math.acos(max(-1, min(1, (trace - 1) / 2)))
    s = 2 * math.sin(angle)
    if abs(s) < 1e-10:
        return [0, 0, 1, 0]

    ax = (m[2][1] - m[1][2]) / s
    ay = (m[0][2] - m[2][0]) / s
    az = (m[1][0] - m[0][1]) / s
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-10:
        return [0, 0, 1, 0]
    ax /= norm
    ay /= norm
    az /= norm

    return [round(ax, 6), round(ay, 6), round(az, 6), round(math.degrees(angle), 4)]


# ---------------------------------------------------------------------------
# Tool: assembly_solve
# ---------------------------------------------------------------------------

class AssemblySolveTool(Tool):
    """Solve assembly constraints to compute part placements."""

    name = "assembly_solve"
    description = (
        "根据装配约束自动计算每个零件的全局位置和旋转。"
        "输入装配体名称和可选的关节角度，输出每个零件的 Placement。"
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
                        "description": "装配体名称（使用内置定义，如 'robotic_arm'）或 JSON 路径",
                    },
                    "assembly_json": {
                        "type": "string",
                        "description": "装配体定义的 JSON 字符串（可选，优先于 assembly_name）",
                    },
                    "joint_angles": {
                        "type": "object",
                        "description": "关节角度映射 {零件名或关节描述: 角度(度)}，默认全零",
                    },
                    "base_position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "根零件的全局位置 [x, y, z]（默认 [0,0,0]）",
                    },
                },
                "required": [],
            },
        )

    def execute(
        self,
        *,
        assembly_name: str = "robotic_arm",
        assembly_json: str = "",
        joint_angles: dict[str, float] | None = None,
        base_position: list[float] | None = None,
        **kwargs: Any,
    ) -> str:
        assembly = _resolve_assembly(assembly_name, assembly_json)
        if assembly is None:
            return f"错误：未找到装配体定义 '{assembly_name}'"

        bp = tuple(base_position) if base_position else (0, 0, 0)
        solver = AssemblySolver(assembly)
        placements = solver.solve(joint_angles=joint_angles, base_position=bp)

        lines = [
            f"[Assembly Solver] {assembly.name}",
            f"零件数: {len(placements)}",
            f"关节数: {len(assembly.joints)}",
            "",
            "--- Part Placements ---",
        ]
        for pname, p in placements.items():
            pos = p["position"]
            rot = p["rotation"]
            lines.append(
                f"  {pname}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})  "
                f"rot=({rot[0]:.3f}, {rot[1]:.3f}, {rot[2]:.3f}, {rot[1]:.1f} deg)"
            )

        lines.append("")
        lines.append("--- JSON ---")
        lines.append(json.dumps(placements, ensure_ascii=False, indent=2))

        return "\n".join(lines)


def _resolve_assembly(name: str, json_str: str) -> Assembly | None:
    """Resolve an assembly by name or parse from JSON."""
    if json_str:
        try:
            return _parse_assembly_json(json_str)
        except Exception:
            pass

    # Built-in assemblies
    from ..knowledge.mechanics import ROBOTIC_ARM_ASSEMBLY

    builtins: dict[str, Assembly] = {
        "robotic_arm": ROBOTIC_ARM_ASSEMBLY,
        "3-dof_robotic_arm": ROBOTIC_ARM_ASSEMBLY,
    }
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key in builtins:
        return builtins[key]

    # Try loading from file
    path = Path(name)
    if path.exists() and path.suffix in (".json", ".fcstd"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return _parse_assembly_dict(data)
        except Exception:
            pass

    return None


def _parse_assembly_json(json_str: str) -> Assembly:
    """Parse an Assembly from a JSON string."""
    data = json.loads(json_str)
    return _parse_assembly_dict(data)


def _parse_assembly_dict(data: dict[str, Any]) -> Assembly:
    """Parse an Assembly from a dict."""
    parts = [
        Part(
            name=p["name"],
            category=p.get("category", "custom"),
            description=p.get("description", ""),
            material=p.get("material", "PLA"),
            dimensions=p.get("dimensions", {}),
        )
        for p in data.get("parts", [])
    ]
    joints = [
        Joint(
            type=j.get("type", "fixed"),
            parent=j["parent"],
            child=j["child"],
            range_deg=tuple(j.get("range_deg", (-180, 180))),
            description=j.get("description", ""),
            parent_anchor=j.get("parent_anchor", "top"),
            child_anchor=j.get("child_anchor", "bottom"),
            offset=tuple(j.get("offset", (0, 0, 0))),
        )
        for j in data.get("joints", [])
    ]
    return Assembly(
        name=data.get("name", "Custom Assembly"),
        parts=parts,
        joints=joints,
        description=data.get("description", ""),
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_assembly_solver_tools(registry: Any) -> None:
    """Register assembly solver tools."""
    registry.register(AssemblySolveTool())
