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
import logging
import math
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..knowledge.mechanics import Assembly, Joint, Part
from ..models.base import ToolDefinition
from .base import Tool

logger = logging.getLogger(__name__)

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
    "center": (0, 0, 0),
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
    "center": [],
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
    "center": ((1, 0, 0), (0, 1, 0)),   # X and Y
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
    # Fallback: for front/back/left/right, use diameter/outer_diameter if present
    if anchor in ("front", "back", "left", "right"):
        for dkey in ("diameter", "outer_diameter"):
            if dkey in dims:
                return dims[dkey] / 2.0
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


def _max_dimension(part: Part) -> float:
    """Return the largest dimension of a part."""
    if not part.dimensions:
        return 0.0
    return max(part.dimensions.values())


def _clamp_child_offset(
    parent_pos: tuple[float, float, float],
    child_pos: tuple[float, float, float],
    parent_part: Part,
    child_part: Part,
    max_factor: float = 3.0,
) -> tuple[float, float, float]:
    """Reject child-parent offsets that exceed a sane engineering bound.

    Per project CLAUDE.md: "LLM 给出离谱尺寸/位置时，应该报错让 LLM 重试，
    而不是悄悄修正".  The previous implementation silently scaled extreme
    offsets down to ``max_factor × (parent + child)`` and logged a warning,
    which let the rest of the pipeline run on corrupted geometry and made
    downstream failures (collisions, bad URDF origins) hard to attribute.

    Now the function raises ``ValueError`` so the assembly generator's VLM
    feedback loop can react by regenerating the assembly with explicit
    guidance about the offending joint.

    The bound is intentionally generous (3× the sum of largest dimensions)
    so legitimately long reach arms are not falsely rejected; only truly
    broken offsets (e.g. LLM hallucinating a -1500mm coordinate) raise.
    """
    dx = child_pos[0] - parent_pos[0]
    dy = child_pos[1] - parent_pos[1]
    dz = child_pos[2] - parent_pos[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)

    max_offset = max_factor * (_max_dimension(parent_part) + _max_dimension(child_part))
    if max_offset < 1.0:
        max_offset = 500.0  # fallback for parts with no dimensions

    if dist > max_offset and dist > 1e-6:
        raise ValueError(
            f"Extreme joint offset for '{parent_part.name}'→"
            f"'{child_part.name}': {dist:.1f}mm exceeds {max_factor:.1f}× "
            f"({max_factor:.1f} × ({_max_dimension(parent_part):.0f} + "
            f"{_max_dimension(child_part):.0f}) = {max_offset:.1f}mm). "
            f"The LLM-generated offset/dimensions are inconsistent; "
            f"regenerate the assembly with corrected values."
        )

    return child_pos


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
# Anchor face alignment rotation
# ---------------------------------------------------------------------------

def _anchor_alignment_rotation(parent_anchor: str, child_anchor: str) -> list[list[float]]:
    """Compute rotation to align child anchor face with parent anchor face.

    Rotates the child so its child_anchor outward normal points in the
    direction opposite to the parent_anchor outward normal, ensuring the
    two faces are oriented toward each other.

    For example, parent_anchor="left" + child_anchor="bottom":
      parent left normal = (-1,0,0), child bottom normal = (0,0,-1)
      child bottom should point in +X (opposite of parent left = +1,0,0)
      → rotate child so (0,0,-1) maps to (1,0,0) → R_y(-90°)
    """
    d_parent = ANCHOR_DIRECTIONS.get(parent_anchor, (0, 0, 1))
    d_child = ANCHOR_DIRECTIONS.get(child_anchor, (0, 0, 1))

    # Target: child anchor normal should face the parent (opposite of parent normal)
    target = (-d_parent[0], -d_parent[1], -d_parent[2])

    dot = d_child[0] * target[0] + d_child[1] * target[1] + d_child[2] * target[2]

    # Already aligned (e.g., top→bottom: child (0,0,-1) → target (0,0,-1))
    if dot > 1.0 - 1e-10:
        return _identity_matrix()

    # Anti-parallel: 180° rotation (e.g., top→top: child (0,0,1) → target (0,0,-1))
    if dot < -1.0 + 1e-10:
        if abs(d_child[0]) < 0.9:
            perp = (1.0, 0.0, 0.0)
        else:
            perp = (0.0, 1.0, 0.0)
        cross = (
            d_child[1] * perp[2] - d_child[2] * perp[1],
            d_child[2] * perp[0] - d_child[0] * perp[2],
            d_child[0] * perp[1] - d_child[1] * perp[0],
        )
        norm = math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)
        if norm < 1e-10:
            return _identity_matrix()
        return _rotation_matrix_axis_angle(
            (cross[0] / norm, cross[1] / norm, cross[2] / norm), math.pi
        )

    # General case: rotate around cross(d_child, target)
    cross = (
        d_child[1] * target[2] - d_child[2] * target[1],
        d_child[2] * target[0] - d_child[0] * target[2],
        d_child[0] * target[1] - d_child[1] * target[0],
    )
    norm = math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)
    if norm < 1e-10:
        return _identity_matrix()
    angle = math.acos(max(-1.0, min(1.0, dot)))
    return _rotation_matrix_axis_angle(
        (cross[0] / norm, cross[1] / norm, cross[2] / norm), angle
    )


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _is_cylindrical_part(part: Part) -> bool:
    """Check if a part has cylindrical dimensions."""
    dims = part.dimensions
    return "diameter" in dims or "outer_diameter" in dims


def _cylinder_base_orientation(axis: tuple[float, float, float]) -> list[list[float]]:
    """Rotation to align a cylinder's default Z axis with the given joint axis.

    VTK add_cylinder creates Y-aligned, then RotateX(90) → Z-aligned.
    For a wheel on axis=y: need to rotate Z→Y = R_x(90deg).
    """
    ax, ay, az = axis
    norm = math.sqrt(ax*ax + ay*ay + az*az)
    if norm < 1e-10:
        return _identity_matrix()
    ax, ay, az = ax/norm, ay/norm, az/norm

    if abs(az - 1.0) < 1e-10 or abs(az + 1.0) < 1e-10:
        return _identity_matrix()  # Z axis, already aligned
    elif abs(ay - 1.0) < 1e-10 or abs(ay + 1.0) < 1e-10:
        return _rotation_matrix_axis_angle((1, 0, 0), -math.pi / 2 * (1 if ay > 0 else -1))
    elif abs(ax - 1.0) < 1e-10 or abs(ax + 1.0) < 1e-10:
        return _rotation_matrix_axis_angle((0, 1, 0), math.pi / 2 * (1 if ax > 0 else -1))
    else:
        # General: rotate Z to target via cross product
        z = (0, 0, 1)
        cross = (z[1]*az - z[2]*ay, z[2]*ax - z[0]*az, z[0]*ay - z[1]*ax)
        # Normalize the cross product (rotation axis)
        cross_norm = math.sqrt(sum(c**2 for c in cross))
        if cross_norm > 1e-10:
            cross = tuple(c / cross_norm for c in cross)
        dot = az  # z · axis_normalized
        return _rotation_matrix_axis_angle(cross, math.acos(max(-1, min(1, dot))))


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


def _quaternion_from_matrix(m: list[list[float]]) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a quaternion (w, x, y, z)."""
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2][1] - m[1][2]) * s
        y = (m[0][2] - m[2][0]) * s
        z = (m[1][0] - m[0][1]) * s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-10:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / norm, x / norm, y / norm, z / norm)


def _matrix_from_quaternion(q: tuple[float, float, float, float]) -> list[list[float]]:
    """Convert a quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = q
    return [
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ]


def _matrix_transpose(m: list[list[float]]) -> list[list[float]]:
    """Transpose a 3x3 matrix. For rotation matrices, this equals inverse."""
    return [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]


def _average_rotations(matrices: list[list[list[float]]]) -> list[list[float]]:
    """Average multiple rotation matrices via quaternion averaging.

    Converts each matrix to a quaternion, averages the quaternion
    components, normalizes, and converts back.
    """
    if not matrices:
        return _identity_matrix()
    if len(matrices) == 1:
        return matrices[0]

    # Accumulate quaternion components
    aw, ax, ay, az = 0.0, 0.0, 0.0, 0.0
    for m in matrices:
        qw, qx, qy, qz = _quaternion_from_matrix(m)
        # Ensure consistent hemisphere (dot with accumulator)
        if qw * aw + qx * ax + qy * ay + qz * az < 0:
            qw, qx, qy, qz = -qw, -qx, -qy, -qz
        aw += qw
        ax += qx
        ay += qy
        az += qz

    norm = math.sqrt(aw * aw + ax * ax + ay * ay + az * az)
    if norm < 1e-10:
        return _identity_matrix()
    avg_q = (aw / norm, ax / norm, ay / norm, az / norm)
    return _matrix_from_quaternion(avg_q)


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
# Connection method → constraint derivation
# ---------------------------------------------------------------------------

def connection_to_constraints(joint: Joint) -> list[dict[str, Any]]:
    """Derive geometric constraint dicts from a joint's connection method.

    Maps ConnectionMethod types to constraint dicts that describe the
    geometric requirements for the joint:

    - bolted → coincident (face contact) + optional concentric (hole alignment)
    - press_fit → concentric (shaft-hole) + distance(0)
    - welded/snap_fit/adhesive/magnetic → coincident (face contact)
    - No connection → legacy anchor-based coincident

    Returns:
        List of constraint dicts with keys:
            type: "coincident"/"concentric"/"distance"/"angle"/"parallel"/"perpendicular"
            parent_anchor: str (anchor face name)
            child_anchor: str (anchor face name)
            distance: float (for distance constraints)
            angle_deg: float (for angle constraints)
    """
    constraints: list[dict[str, Any]] = []

    # Priority 1: explicit constraint_type on the joint
    if joint.constraint_type:
        c: dict[str, Any] = {
            "type": joint.constraint_type,
            "parent_anchor": joint.parent_anchor,
            "child_anchor": joint.child_anchor,
        }
        if joint.parent_attachment:
            c["parent_attachment"] = joint.parent_attachment
        if joint.child_attachment:
            c["child_attachment"] = joint.child_attachment
        if joint.constraint_type == "distance":
            c["distance"] = joint.constraint_distance
        elif joint.constraint_type == "angle":
            c["angle_deg"] = joint.constraint_angle_deg
        constraints.append(c)
        return constraints

    # Priority 2: connection method
    if joint.connection is not None:
        conn_type = joint.connection.type
        if conn_type == "bolted":
            # Face-to-face contact
            constraints.append({
                "type": "coincident",
                "parent_anchor": joint.parent_anchor,
                "child_anchor": joint.child_anchor,
            })
            # Bolt hole alignment (if attachment points specified)
            if joint.parent_attachment and joint.child_attachment:
                constraints.append({
                    "type": "concentric",
                    "parent_attachment": joint.parent_attachment,
                    "child_attachment": joint.child_attachment,
                })
        elif conn_type == "press_fit":
            # Shaft-in-hole alignment
            constraints.append({
                "type": "concentric",
                "parent_anchor": joint.parent_anchor,
                "child_anchor": joint.child_anchor,
            })
            constraints.append({
                "type": "distance",
                "parent_anchor": joint.parent_anchor,
                "child_anchor": joint.child_anchor,
                "distance": 0,
            })
        elif conn_type in ("welded", "snap_fit", "adhesive", "magnetic"):
            constraints.append({
                "type": "coincident",
                "parent_anchor": joint.parent_anchor,
                "child_anchor": joint.child_anchor,
            })
        else:
            constraints.append({
                "type": "coincident",
                "parent_anchor": joint.parent_anchor,
                "child_anchor": joint.child_anchor,
            })
        return constraints

    # Priority 3: fallback to anchor-based (legacy)
    constraints.append({
        "type": "coincident",
        "parent_anchor": joint.parent_anchor,
        "child_anchor": joint.child_anchor,
    })
    return constraints


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
        resolve_collisions: bool = False,
        collision_max_rounds: int = 2,
    ) -> dict[str, dict[str, Any]]:
        """Solve for every part's global Placement.

        Uses constraint-based positioning with multi-parent topology support.
        Parts that are children of multiple parents (e.g., a plate bolted to
        four standoffs) get their positions averaged from all parent
        suggestions.

        Args:
            joint_angles: Optional mapping of {joint_description: angle_degrees}
                          or {child_part_name: angle_degrees}.
                          Defaults to all zeros.
            base_position: Global position for the root part.

        Returns:
            Dict mapping part_name -> {"position": [x,y,z], "rotation": [ax,ay,az,angle_deg]}
        """
        if joint_angles is None:
            joint_angles = dict(self.assembly.default_angles)

        # Resolve mimic joints: if a joint mimics another, compute its angle
        # from the mimicked joint's angle.  This ensures the solver places
        # coupled parts (e.g. gripper fingers) at the correct position.
        # F13: two-pass to remove ordering dependency — a mimic source may
        # itself be resolved in the same loop, so we iterate twice.
        for _pass in range(2):
            for j in self._joints:
                if j.mimic_joint:
                    source_angle = joint_angles.get(j.mimic_joint, 0.0)
                    resolved = source_angle * j.mimic_multiplier + j.mimic_offset
                    joint_angles[j.child] = resolved

        # Build adjacency: parent -> list of (joint, child)
        children_of: dict[str, list[tuple[Joint, str]]] = {}
        parent_of: dict[str, list[tuple[Joint, str]]] = {}
        for j in self._joints:
            children_of.setdefault(j.parent, []).append((j, j.child))
            parent_of.setdefault(j.child, []).append((j, j.parent))

        # Multi-parent parts: children referenced by more than one parent
        multi_parent_parts = {
            child for child, parents in parent_of.items()
            if len(parents) > 1
        }

        # Group children by (parent, parent_anchor, distribution_group) to compute
        # sibling distribution.  Joints sharing the same key are spread across the
        # parent face; different distribution_groups are placed independently at the
        # anchor centre.  no_distribute=True is excluded from auto-distribution.
        sibling_groups: dict[tuple[str, str, str], list[int]] = {}
        for parent_name, child_list in children_of.items():
            for idx, (joint, _) in enumerate(child_list):
                if not joint.no_distribute:
                    key = (parent_name, joint.parent_anchor, joint.distribution_group)
                    sibling_groups.setdefault(key, []).append(idx)

        # Find root: a part that is never a child
        child_set = {j.child for j in self._joints}
        all_names = set(self._parts_by_name.keys())
        roots = all_names - child_set
        if not roots:
            # Fallback: use the first part
            roots = {self.assembly.parts[0].name} if self.assembly.parts else set()

        # Phase 1: DFS from roots (uses stack.pop(); F14 corrected comment).

        positions: dict[str, tuple[float, float, float]] = {}
        rot_mats: dict[str, list[list[float]]] = {}

        stack: list[tuple[str, tuple[float, float, float], list[list[float]]]] = []
        for root_name in roots:
            stack.append((root_name, base_position, _identity_matrix()))

        visited: set[str] = set()

        while stack:
            part_name, pos, rot_mat = stack.pop()
            if part_name in visited:
                continue
            visited.add(part_name)

            positions[part_name] = pos
            rot_mats[part_name] = rot_mat

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
                if joint.no_distribute:
                    sibling_index, total_siblings = 0, 1
                else:
                    group_key = (part_name, joint.parent_anchor, joint.distribution_group)
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

        # Phase 2: Multi-parent consistency check.
        #
        # P1-2: previously this phase AVERAGED positions from all parents,
        # which is physically impossible — a rigid plate bolted to 4
        # standoffs at different heights cannot sit at their average height.
        # Now we use the FIRST parent's suggestion (tree semantics) and
        # warn when other parents disagree by more than a tolerance.
        # Closed-chain kinematic constraints should be handled by
        # ClosedChainSolver, not by averaging.
        for child_name in multi_parent_parts:
            if child_name not in positions:
                continue

            pos_suggestions: list[tuple[float, float, float]] = []
            rot_suggestions: list[list[list[float]]] = []
            for joint, parent_name in parent_of[child_name]:
                if parent_name not in positions:
                    continue
                parent_part = self._parts_by_name.get(parent_name)
                child_part = self._parts_by_name.get(child_name)
                if not parent_part or not child_part:
                    continue

                suggested_pos, suggested_rot = self._compute_child_transform(
                    parent_part=parent_part,
                    child_part=child_part,
                    joint=joint,
                    parent_pos=positions[parent_name],
                    parent_rot=rot_mats[parent_name],
                    joint_angles=joint_angles,
                )
                pos_suggestions.append(suggested_pos)
                rot_suggestions.append(suggested_rot)

            if len(pos_suggestions) > 1:
                # Use the FIRST parent's suggestion (tree semantics).
                first_pos = pos_suggestions[0]
                first_rot = rot_suggestions[0]

                # Warn when other parents disagree significantly.
                for idx in range(1, len(pos_suggestions)):
                    disp = math.sqrt(
                        (pos_suggestions[idx][0] - first_pos[0]) ** 2
                        + (pos_suggestions[idx][1] - first_pos[1]) ** 2
                        + (pos_suggestions[idx][2] - first_pos[2]) ** 2
                    )
                    if disp > 5.0:
                        logger.warning(
                            "Multi-parent part '%s': parent #%d suggests "
                            "position %.1fmm from parent #0's suggestion "
                            "(%.1fmm disagreement). Using first parent. "
                            "For closed-chain constraints use "
                            "ClosedChainSolver.",
                            child_name, idx, disp, disp,
                        )

                # Apply delta from current position to first-parent suggestion
                old_pos = positions[child_name]
                delta = (
                    first_pos[0] - old_pos[0],
                    first_pos[1] - old_pos[1],
                    first_pos[2] - old_pos[2],
                )
                self._apply_delta(child_name, delta, positions, children_of)

                # Use first parent's rotation
                old_rot = rot_mats[child_name]
                rot_mats[child_name] = first_rot
                old_rot_inv = _matrix_transpose(old_rot)
                delta_rot = _mat_mul(first_rot, old_rot_inv)
                self._apply_rotation_delta(
                    child_name, delta_rot, rot_mats, children_of,
                    positions=positions,
                )

        # Phase 3: Convert to output format.
        # Apply visual cylinder orientation as a post-processing step so
        # cylindrical parts (servo motors) render aligned with their rotation
        # axis without contaminating the kinematic chain (CYL-FIX).
        placements: dict[str, dict[str, Any]] = {}
        for pname in self._parts_by_name:
            if pname in positions:
                p = positions[pname]
                # NaN/Inf check — invalid positions must not silently
                # propagate to renderers and URDF exporters.
                if any(not math.isfinite(v) for v in p):
                    logger.warning(
                        "Part %s has non-finite position %s — "
                        "falling back to base",
                        pname, p,
                    )
                    placements[pname] = {
                        "position": [base_position[0], base_position[1], base_position[2]],
                        "rotation": [0, 0, 1, 0],
                        "kinematic_rotation": [0, 0, 1, 0],
                    }
                    continue
                r = rot_mats.get(pname, _identity_matrix())
                # kinematic_rotation = pure joint-chain rotation, used by
                # URDF/IK consumers that must NOT be contaminated by the
                # visual cylinder_orient offset (P0-2: previously the visual
                # rotation leaked into URDF joint origin RPY).
                r_kin = r
                r = self._visual_rotation_for_part(pname, r)
                placements[pname] = {
                    "position": [round(p[0], 4), round(p[1], 4), round(p[2], 4)],
                    "rotation": _rot_mat_to_axis_angle_deg(r),
                    "kinematic_rotation": _rot_mat_to_axis_angle_deg(r_kin),
                }
            else:
                logger.warning(
                    "Part %s has no solved position (orphan or "
                    "disconnected) — placed at base %s",
                    pname, base_position,
                )
                placements[pname] = {
                    "position": [base_position[0], base_position[1], base_position[2]],
                    "rotation": [0, 0, 1, 0],
                    "kinematic_rotation": [0, 0, 1, 0],
                }

        # Phase 4: Collision resolution (opt-in).
        # Runs a detect→fix→re-solve loop on the solved placements.
        # Default off so existing callers see no behaviour change.
        if resolve_collisions:
            try:
                from .collision_resolver import CollisionResolver
                resolver = CollisionResolver(max_rounds=collision_max_rounds)
                resolution = resolver.resolve(
                    self.assembly, placements, joint_angles,
                )
                if resolution.resolved:
                    logger.info(
                        "Collision resolution succeeded in %d round(s)",
                        resolution.rounds_used,
                    )
                    placements = resolution.modified_positions
                elif resolution.modified_positions:
                    logger.warning(
                        "Collision resolution incomplete: %d remaining "
                        "(history: %s)",
                        resolution.remaining_count,
                        resolution.collision_history,
                    )
                    placements = resolution.modified_positions
            except Exception as exc:
                logger.warning(
                    "Collision resolution skipped (%s); returning base solve",
                    exc,
                )

        return placements

    @staticmethod
    def _apply_delta(
        root_name: str,
        delta: tuple[float, float, float],
        positions: dict[str, tuple[float, float, float]],
        children_of: dict[str, list[tuple[Joint, str]]],
    ) -> None:
        """Apply a position delta to a part and all its BFS descendants."""
        queue: deque[str] = deque([root_name])
        visited: set[str] = set()
        while queue:
            name = queue.popleft()
            if name in visited:
                continue
            visited.add(name)
            if name in positions:
                old = positions[name]
                positions[name] = (
                    old[0] + delta[0],
                    old[1] + delta[1],
                    old[2] + delta[2],
                )
            for _, child in children_of.get(name, []):
                if child not in visited:
                    queue.append(child)

    @staticmethod
    def _apply_rotation_delta(
        root_name: str,
        delta_rot: list[list[float]],
        rot_mats: dict[str, list[list[float]]],
        children_of: dict[str, list[tuple[Joint, str]]],
        positions: dict[str, tuple[float, float, float]] | None = None,
    ) -> None:
        """Apply a rotation delta to a part and all its BFS descendants.

        F12: previously only ``rot_mats`` was updated, leaving descendant
        positions stale — so rotating a multi-parent part did not move its
        children, causing visible misalignment in render.  Now when
        ``positions`` is provided we rotate each descendant's position
        around the root part's position (the pivot).
        """
        pivot = positions.get(root_name) if positions else None
        queue: deque[str] = deque([root_name])
        visited: set[str] = set()
        while queue:
            name = queue.popleft()
            if name in visited:
                continue
            visited.add(name)
            if name in rot_mats:
                rot_mats[name] = _mat_mul(delta_rot, rot_mats[name])
            # Rotate the position around the pivot so children follow the
            # rotation delta applied to the root.
            if positions is not None and pivot is not None and name in positions:
                pos = positions[name]
                rel = (pos[0] - pivot[0], pos[1] - pivot[1], pos[2] - pivot[2])
                rotated = _mat_vec(delta_rot, rel)
                positions[name] = (
                    pivot[0] + rotated[0],
                    pivot[1] + rotated[1],
                    pivot[2] + rotated[2],
                )
            for _, child in children_of.get(name, []):
                if child not in visited:
                    queue.append(child)

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

        # F10: clamp the joint value to its declared range so solver input
        # (or a bad default_angles entry) can't drive a part beyond its
        # mechanical limits.  For prismatic joints range_deg holds mm, not
        # degrees, but the clamp still applies.
        if joint.range_deg:
            lo, hi = joint.range_deg[0], joint.range_deg[1]
            if lo <= hi:
                angle_deg = max(lo, min(hi, angle_deg))
        angle_rad = math.radians(angle_deg)

        # Anchor alignment: rotate child so its anchor faces the parent's anchor
        align_rot = _anchor_alignment_rotation(joint.parent_anchor, joint.child_anchor)

        # Compute child rotation (kinematic only — NO visual cylinder_orient).
        #
        # CYL-FIX: cylinder_orient (which orients cylindrical parts along
        # their rotation axis for rendering) is applied as a POST-PROCESSING
        # step at output time (see _visual_rotation_for_part).  Including it
        # here would contaminate the kinematic chain because rot_mats[name]
        # is used as parent_rot for children — the visual rotation would
        # leak into child position computations, breaking IK and FK.
        #
        # F15: the rotation order ``joint_rot @ parent_rot @ align`` is
        # correct because ``joint_rot`` is a rotation about a GLOBAL axis
        # (rot_axis_global transforms the local axis into the parent frame),
        # so it must LEFT-multiply parent_rot.
        if joint.type in ("revolute", "gear", "belt"):
            rot_axis_local = _revolute_axis(joint)
            rot_axis_global = _mat_vec(parent_rot, rot_axis_local)
            joint_rot = _rotation_matrix_axis_angle(rot_axis_global, angle_rad)
            anchor_rot = _mat_mul(joint_rot, _mat_mul(parent_rot, align_rot))
        elif joint.type == "prismatic":
            # Translation along the joint axis (e.g. axis="x" for gripper
            # fingers that slide left/right).  Fall back to anchor direction
            # for legacy assemblies without an explicit axis.
            #
            # F9: the joint value for a prismatic joint is a displacement in
            # mm (consistent with urdf_export, which treats range_deg as mm
            # and converts to metres).  Previously the value was converted to
            # radians and multiplied by 100 — a unit mismatch that inflated
            # real offsets (a 12mm finger stroke became ~21mm, and larger
            # LLM-generated values blew up to hundreds of mm, the 467mm
            # symptom).  Use the raw mm value directly.
            if joint.axis in _EXPLICIT_AXES:
                slide_dir = _EXPLICIT_AXES[joint.axis]
            else:
                slide_dir = ANCHOR_DIRECTIONS.get(joint.parent_anchor, (0, 0, 1))
            slide_global = _mat_vec(parent_rot, slide_dir)
            displacement_mm = angle_deg  # mm, NOT angle_rad * 100
            offset = (slide_global[0] * displacement_mm,
                      slide_global[1] * displacement_mm,
                      slide_global[2] * displacement_mm)
            parent_anchor_global = _vec_add(parent_anchor_global, offset)
            anchor_rot = _mat_mul(parent_rot, align_rot)
        else:
            # Fixed joint: inherit parent rotation + anchor alignment
            anchor_rot = _mat_mul(parent_rot, align_rot)

        # Child center = parent anchor point + rotated child anchor offset
        child_center = _vec_add(
            parent_anchor_global,
            _mat_vec(anchor_rot, child_anchor_neg),
        )

        # --- Center-anchor half-extent offset ---
        # When child_anchor="center" but parent_anchor is a face anchor
        # (top/bottom/front/back/left/right), child_anchor_neg is (0,0,0)
        # so the child CENTER sits at the parent's face.  But the child has
        # dimensions — half of it extends INTO the parent, causing visual
        # intersection.  Push the child outward along the parent anchor
        # normal by the child's half-extent so the child's near face
        # aligns with the parent's face instead.
        # center/center pairs (e.g. bearing press-fit) are excluded — those
        # are intentionally concentric.
        if joint.child_anchor == "center" and joint.parent_anchor != "center":
            child_half = _half_extent(child_part, joint.parent_anchor)
            if child_half > 0:
                _center_normal = ANCHOR_DIRECTIONS.get(
                    joint.parent_anchor, (0, 0, 1),
                )
                _center_normal_global = _mat_vec(parent_rot, _center_normal)
                child_center = _vec_add(child_center, (
                    _center_normal_global[0] * child_half,
                    _center_normal_global[1] * child_half,
                    _center_normal_global[2] * child_half,
                ))

        # --- Revolute joint clearance ---
        # Movable joints (revolute/gear/belt) need a visible physical gap
        # between the parent face and the child face — this represents the
        # bearing/shaft interface that allows rotation.  Without it, parts
        # sit flush against each other and look "fused" in renders.
        # The clearance is proportional to the parent's height so it
        # scales naturally across large bases and small servos.
        if joint.type in ("revolute", "gear", "belt"):
            parent_h = parent_part.dimensions.get(
                "height", parent_part.dimensions.get("length", 20.0),
            )
            _clearance_mm = max(3.0, min(8.0, parent_h * 0.12))
            _anchor_normal_local = ANCHOR_DIRECTIONS.get(
                joint.parent_anchor, (0, 0, 1),
            )
            _anchor_normal_global = _mat_vec(parent_rot, _anchor_normal_local)
            child_center = _vec_add(child_center, (
                _anchor_normal_global[0] * _clearance_mm,
                _anchor_normal_global[1] * _clearance_mm,
                _anchor_normal_global[2] * _clearance_mm,
            ))

        # Add explicit offset. IMPORTANT: the offset defines the joint's
        # MOUNT POINT relative to the parent (e.g. a wheel mounts at
        # track/2, wheelbase/2 from chassis center) — it is a fixed
        # connection point and must NOT rotate with the joint's own motion.
        # Previously this used ``anchor_rot`` (which includes ``joint_rot``
        # for revolute joints), so spinning a wheel 90° rotated its
        # [-100,90,-49] offset to [-49,90,100], catapulting the wheel from
        # Z=47 to Z=196. We now rotate the offset only by the PARENT's
        # orientation + the anchor alignment (parent_rot @ align_rot),
        # which is the mounting frame WITHOUT the joint's own rotation.
        joint_offset = joint.offset or (0, 0, 0)
        offset_frame = _mat_mul(parent_rot, align_rot)
        rotated_offset = _mat_vec(offset_frame, joint_offset)
        child_center = _vec_add(
            child_center,
            rotated_offset,
        )

        # Apply constraint refinements from connection_to_constraints()
        child_center = self._apply_constraint_refinement(
            joint, child_center, parent_rot,
        )

        # Sanity clamp: if the child is displaced too far from the parent,
        # clamp to a reasonable maximum.  This prevents extreme positions
        # caused by incorrect LLM dimensions or offsets (e.g. -1500mm).
        child_center = _clamp_child_offset(
            parent_pos, child_center, parent_part, child_part,
        )

        return child_center, anchor_rot

    def _find_driving_axis(
        self, part_name: str,
    ) -> tuple[float, float, float] | None:
        """Find the rotation axis a cylindrical part drives as a parent.

        Searches for revolute/gear/belt joints where *part_name* is the
        PARENT.  Returns the first matching joint's local rotation axis,
        or ``None`` if the part doesn't drive any rotary joint.

        This lets us orient cylinders connected via *fixed* joints along
        their functional rotation axis (e.g. a wrist motor bolted to the
        elbow link still needs its cylinder axis aligned with the wrist
        rotation axis).
        """
        for j in self._joints:
            if j.parent == part_name and j.type in ("revolute", "gear", "belt"):
                return _revolute_axis(j)
        return None

    def _visual_rotation_for_part(
        self, part_name: str, kinematic_rot: list[list[float]],
    ) -> list[list[float]]:
        """Apply visual cylinder orientation to a part's kinematic rotation.

        HISTORICALLY this was a POST-PROCESSING step that re-oriented
        cylindrical parts (servos, wheels) for rendering, because the STL
        was always generated with the cylinder along Z (FreeCAD
        makeCylinder default) regardless of the joint axis.  A wheel on
        axis="y" got R_x(90°) here to point Z→Y, a servo on axis="z" got
        identity.

        THAT COMPENSATION IS NOW REMOVED.  ``part_feature_engine`` sets
        ``orient_axis`` on every cylinder op: wheels (axis="y") → STL
        cylinder built along Y; servos (axis="z") → STL along Z.  The
        correct orientation is baked into the STL geometry itself, so
        applying a second orientation here would DOUBLE-ROTATE the wheel
        (correct Y + R_x(90°) → back to Z = 磨盘).  Returning the pure
        kinematic rotation trusts the STL's baked orientation.  This stays
        a no-op for the kinematic chain (``rot_mats`` is unchanged); it
        only affects the output placements used by the renderer.

        Kept as a method (not deleted) so callers and tests that reference
        ``_apply_cylinder_orientation`` still resolve; the body is now a
        documented pass-through.
        """
        return kinematic_rot

    def _apply_constraint_refinement(
        self,
        joint: Joint,
        child_pos: tuple[float, float, float],
        parent_rot: list[list[float]],
    ) -> tuple[float, float, float]:
        """Adjust child position based on connection-derived constraints.

        Applies positional refinements from ``connection_to_constraints()``.
        Iterates **all** constraints (a bolted joint produces both coincident
        and concentric).

        - bolted with attachment points: concentric alignment offset
        - distance constraint: add gap/offset along anchor direction
        """
        constraints = connection_to_constraints(joint)
        if not constraints:
            return child_pos

        for c in constraints:
            ctype = c.get("type", "")

            if ctype == "distance" and "distance" in c:
                dist = c["distance"]
                # Apply offset along the anchor direction (from parent toward child)
                parent_dir = ANCHOR_DIRECTIONS.get(joint.parent_anchor, (0, 0, 1))
                global_dir = _mat_vec(parent_rot, parent_dir)
                child_pos = _vec_add(
                    child_pos,
                    (global_dir[0] * dist, global_dir[1] * dist, global_dir[2] * dist),
                )

            elif ctype == "concentric":
                # For bolted connections with attachment points: align bolt holes
                pa = c.get("parent_attachment")
                ca = c.get("child_attachment")
                if pa and ca:
                    align_offset = (
                        (pa[0] - ca[0]) * 0.5,
                        (pa[1] - ca[1]) * 0.5,
                        (pa[2] - ca[2]) * 0.5,
                    )
                    global_align = _mat_vec(parent_rot, align_offset)
                    child_pos = _vec_add(child_pos, global_align)

        return child_pos

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

    def _compute_loop_error(
        self,
        placements: dict[str, dict],
        loops: list[list[str]],
        base_solver: "AssemblySolver",
        angles: dict[str, float],
    ) -> float:
        """Compute total closure error (mm) for all loops."""
        total_error = 0.0
        for loop in loops:
            if len(loop) < 2:
                continue
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
            parent_pos = tuple(placements.get(joint.parent, {}).get("position", [0, 0, 0]))
            parent_rot_mat = _identity_matrix()
            parent_placement = placements.get(joint.parent, {})
            # P0-2: use kinematic_rotation for loop-closure math (visual
            # rotation must not contaminate the FK recomputation).
            parent_rot_aa = (
                parent_placement.get("kinematic_rotation")
                or parent_placement.get("rotation")
            )
            if parent_rot_aa:
                parent_rot_mat = _axis_angle_deg_to_rot_mat(parent_rot_aa)
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
        return total_error

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

        # Identify revolute joints for gradient computation
        revolute_keys = [j.child for j in self._joints if j.type == "revolute"]
        for key in revolute_keys:
            if key not in angles:
                angles[key] = 0.0

        for iteration in range(1, max_iterations + 1):
            placements = base_solver.solve(
                joint_angles=angles,
                base_position=base_position,
            )

            total_error = self._compute_loop_error(
                placements, loops, base_solver, angles)

            if total_error < best_error:
                best_error = total_error

            if total_error < tolerance:
                converged = True
                break

            # Finite-difference Jacobian: perturb each revolute joint
            # independently to estimate gradient, then take a step.
            delta = 0.1  # degrees perturbation
            step_size = 0.5  # damping factor
            for key in revolute_keys:
                old_angle = angles[key]
                angles[key] = old_angle + delta
                perturbed = base_solver.solve(
                    joint_angles=angles,
                    base_position=base_position,
                )
                perturbed_error = self._compute_loop_error(
                    perturbed, loops, base_solver, angles)
                gradient = (perturbed_error - total_error) / delta
                angles[key] = old_angle  # restore
                if abs(gradient) > 1e-6:
                    angles[key] = old_angle - step_size * total_error / gradient

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
        "Automatically compute the global position and rotation of each part "
        "based on assembly constraints. Takes an assembly name and optional "
        "joint angles, returns the Placement for each part."
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
                        "description": "Assembly name (using built-in definitions such as 'robotic_arm') or JSON path",
                    },
                    "assembly_json": {
                        "type": "string",
                        "description": "JSON string of assembly definition (optional, takes priority over assembly_name)",
                    },
                    "joint_angles": {
                        "type": "object",
                        "description": "Joint angle mapping {part name or joint description: angle(degrees)}, default all zeros",
                    },
                    "base_position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Global position of the root part [x, y, z] (default [0,0,0])",
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
                f"rot=({rot[0]:.3f}, {rot[1]:.3f}, {rot[2]:.3f}, {rot[3]:.1f} deg)"
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
        except Exception as _e:
            pass  # TODO: parse_assembly_json from cache failed (no logger available)

    # Built-in assemblies
    from ..knowledge.mechanics import (
        OPEN_MANIPULATOR_X_ASSEMBLY,
        ROBOTIC_ARM_ASSEMBLY,
    )

    builtins: dict[str, Assembly] = {
        "robotic_arm": ROBOTIC_ARM_ASSEMBLY,
        "3-dof_robotic_arm": ROBOTIC_ARM_ASSEMBLY,
        "open_manipulator_x": OPEN_MANIPULATOR_X_ASSEMBLY,
        "openmanipulator_x": OPEN_MANIPULATOR_X_ASSEMBLY,
        "openmanipulatorx": OPEN_MANIPULATOR_X_ASSEMBLY,
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
        except Exception as _e:
            pass  # TODO: parse_assembly from file failed (no logger available)

    # Try assembly templates (exact ID match only)
    try:
        from ..knowledge.assembly_templates import TEMPLATES, template_to_assembly
        tpl_key = name.lower().replace(" ", "_").replace("-", "_")
        if tpl_key in TEMPLATES:
            return template_to_assembly(TEMPLATES[tpl_key])
    except Exception as _e:
        pass  # TODO: template lookup failed (no logger available)

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
    joints = []
    for j in data.get("joints", []):
        # Parse connection method if present
        connection = None
        cm_type = j.get("connection_method", "")
        if cm_type:
            cd = j.get("connection_detail", {}) or {}
            from ..knowledge.mechanics import ConnectionMethod
            connection = ConnectionMethod(
                type=cm_type,
                bolt_size=cd.get("bolt_size", "M3"),
                bolt_count=cd.get("bolt_count", 0),
                torque_nm=cd.get("torque_nm", 0.0),
                interference_mm=cd.get("interference_mm", 0.0),
                snap_count=cd.get("snap_count", 0),
                snap_force_n=cd.get("snap_force_n", 0.0),
                adhesive_type=cd.get("adhesive_type", ""),
                bond_area_mm2=cd.get("bond_area_mm2", 0.0),
                weld_type=cd.get("weld_type", ""),
            )

        # Parse attachment points
        pa = j.get("parent_attachment")
        ca = j.get("child_attachment")
        pn = j.get("parent_normal")
        cn = j.get("child_normal")

        joints.append(Joint(
            type=j.get("type", "fixed"),
            parent=j["parent"],
            child=j["child"],
            range_deg=tuple(j.get("range_deg", (-180, 180))),
            description=j.get("description", ""),
            parent_anchor=j.get("parent_anchor", "top"),
            child_anchor=j.get("child_anchor", "bottom"),
            offset=tuple(j.get("offset") or (0, 0, 0)),
            axis=j.get("axis", "auto"),
            no_distribute=j.get("no_distribute", False),
            distribution_group=j.get("distribution_group", ""),
            parent_attachment=tuple(pa) if pa else None,
            child_attachment=tuple(ca) if ca else None,
            parent_normal=tuple(pn) if pn else None,
            child_normal=tuple(cn) if cn else None,
            constraint_type=j.get("constraint_type", ""),
            constraint_distance=j.get("constraint_distance", 0.0),
            constraint_angle_deg=j.get("constraint_angle_deg", 0.0),
            connection=connection,
        ))

    return Assembly(
        name=data.get("name", "Custom Assembly"),
        parts=parts,
        joints=joints,
        description=data.get("description", ""),
        default_angles=data.get("default_angles", {}),
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_assembly_solver_tools(registry: Any) -> None:
    """Register assembly solver tools."""
    registry.register(AssemblySolveTool())
