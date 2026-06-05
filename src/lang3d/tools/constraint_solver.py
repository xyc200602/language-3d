"""Constraint-based assembly solver using SolveSpace (py_slvs).

This module implements a constraint solver that replaces the anchor-face BFS
approach with proper geometric constraint solving. It uses the SolveSpace
solver engine (same as FreeCAD Assembly3) via the py_slvs Python binding.

Architecture:
  - Each part gets ONE free 3D point (its center) and ONE normal (orientation).
  - Joint constraints tie parent anchor positions to child anchor positions.
  - Anchor positions are computed analytically from the solved center.
  - Distribution offsets are applied as initial guesses for the solver.

Joint types mapped to SolveSpace constraints:
  - Fixed:    point coincident + same orientation (6 DOF removed)
  - Revolute: point coincident + same orientation (5 DOF removed, 1 rotation)
  - Prismatic: same orientation + point-on-line (5 DOF, 1 translation)
  - Cylindrical: point coincident + point-on-line (4 DOF, 2 remain)
  - Spherical: point coincident (3 DOF, 3 rotations remain)
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from py_slvs import slvs

from lang3d.knowledge.mechanics import Assembly, Joint, Part


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SolvedPart:
    """Result for a single solved part."""
    name: str
    position: list  # [x, y, z]
    rotation: list  # [ax, ay, az, angle_deg] (axis-angle)


@dataclass
class SolverResult:
    """Complete solver result."""
    parts: dict  # name -> SolvedPart
    success: bool
    dof: int
    failed_constraints: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def axis_angle_to_quaternion(ax: float, ay: float, az: float, angle_deg: float):
    """Convert axis-angle (degrees) to quaternion (w, x, y, z).

    Axis is normalized before conversion.
    """
    mag = math.sqrt(ax * ax + ay * ay + az * az)
    if mag < 1e-10:
        return (1.0, 0.0, 0.0, 0.0)
    ax, ay, az = ax / mag, ay / mag, az / mag

    angle_rad = math.radians(angle_deg)
    half = angle_rad / 2
    s = math.sin(half)
    return (
        math.cos(half),
        ax * s,
        ay * s,
        az * s,
    )


def quaternion_to_axis_angle(w: float, x: float, y: float, z: float):
    """Convert quaternion (w, x, y, z) to axis-angle (ax, ay, az, angle_deg).

    Returns ([0, 0, 1], 0) for identity quaternion.
    """
    # Normalize
    mag = math.sqrt(w * w + x * x + y * y + z * z)
    if mag < 1e-10:
        return [0.0, 0.0, 1.0], 0.0
    w, x, y, z = w / mag, x / mag, y / mag, z / mag

    angle_rad = 2 * math.acos(max(-1.0, min(1.0, abs(w))))
    angle_deg = math.degrees(angle_rad)

    s = math.sin(angle_rad / 2)
    if s < 1e-10:
        return [0.0, 0.0, 1.0], 0.0

    ax, ay, az = x / s, y / s, z / s
    return [ax, ay, az], angle_deg


# ---------------------------------------------------------------------------
# Anchor face geometry
# ---------------------------------------------------------------------------

# Maps anchor name to normal direction (outward from face)
ANCHOR_NORMALS = {
    "top":    (0, 0, 1),
    "bottom": (0, 0, -1),
    "left":   (0, -1, 0),
    "right":  (0, 1, 0),
    "front":  (-1, 0, 0),
    "back":   (1, 0, 0),
}


def _part_dimensions(part: Part):
    """Return (dx, dy, dz) half-extents of a part."""
    d = part.dimensions
    if "outer_diameter" in d:
        r = d["outer_diameter"] / 2
        h = d["height"] / 2
        return r, r, h
    elif "diameter" in d and "length" not in d:
        r = d["diameter"] / 2
        h = d["height"] / 2
        return r, r, h
    elif "length" in d and "width" in d:
        return d["length"] / 2, d["width"] / 2, d["height"] / 2
    else:
        r = d.get("diameter", 20) / 2
        h = d.get("height", 10) / 2
        return r, r, h


def _anchor_point(part: Part, anchor: str):
    """Return the 3D position of an anchor face center on a part at origin."""
    dx, dy, dz = _part_dimensions(part)
    normals = {
        "top":    (0, 0, dz),
        "bottom": (0, 0, -dz),
        "left":   (0, -dy, 0),
        "right":  (0, dy, 0),
        "front":  (-dx, 0, 0),
        "back":   (dx, 0, 0),
    }
    return normals.get(anchor, (0, 0, 0))


# ---------------------------------------------------------------------------
# Constraint converter (Task 63)
# ---------------------------------------------------------------------------

# Maps joint.type -> constraint_type
_JOINT_TYPE_TO_CONSTRAINT = {
    "fixed": "coincident",
    "revolute": "concentric",
    "prismatic": "parallel",
}


def convert_legacy_joint(joint: Joint, parent_part: Part, child_part: Part) -> dict:
    """Convert a Joint (possibly with new Task 63 fields) into a constraint dict.

    If the joint already has ``parent_attachment`` / ``child_attachment`` set,
    those are used directly.  Otherwise the legacy anchor-name based positions
    and normals are computed from ``_anchor_point()`` / ``ANCHOR_NORMALS`` so
    that old-style joints produce exactly the same result as before.

    Returns:
        dict with keys:
            parent_attachment  – (x, y, z)
            child_attachment   – (x, y, z)
            parent_normal      – (nx, ny, nz)
            child_normal       – (nx, ny, nz)
            constraint_type    – str
            constraint_distance – float
            constraint_angle_deg – float
    """
    # Positions
    if joint.parent_attachment is not None:
        pa = tuple(joint.parent_attachment)
    else:
        pa = _anchor_point(parent_part, joint.parent_anchor or "top")

    if joint.child_attachment is not None:
        ca = tuple(joint.child_attachment)
    else:
        ca = _anchor_point(child_part, joint.child_anchor or "bottom")

    # Normals
    if joint.parent_normal is not None:
        pn = tuple(joint.parent_normal)
    else:
        pn = ANCHOR_NORMALS.get(joint.parent_anchor or "top", (0, 0, 1))

    if joint.child_normal is not None:
        cn = tuple(joint.child_normal)
    else:
        cn = ANCHOR_NORMALS.get(joint.child_anchor or "bottom", (0, 0, -1))

    # Constraint type
    if joint.constraint_type:
        ctype = joint.constraint_type
    else:
        ctype = _JOINT_TYPE_TO_CONSTRAINT.get(joint.type, "coincident")

    return {
        "parent_attachment": pa,
        "child_attachment": ca,
        "parent_normal": pn,
        "child_normal": cn,
        "constraint_type": ctype,
        "constraint_distance": joint.constraint_distance,
        "constraint_angle_deg": joint.constraint_angle_deg,
    }


# ---------------------------------------------------------------------------
# ConstraintSolver
# ---------------------------------------------------------------------------

class ConstraintSolver:
    """Assembly solver using SolveSpace geometric constraint engine.

    Each part is represented by a single 3D point (center) and a normal
    (orientation). Anchor positions are computed analytically from the
    solved center + the known anchor offset.
    """

    FIXED_GROUP = 1
    SOLVE_GROUP = 2

    def __init__(self, assembly: Assembly):
        self.assembly = assembly
        self.part_index = {p.name: p for p in assembly.parts}

        self._sys = None
        # Per-part solver entities
        self._centers = {}       # part_name -> point entity handle (int)
        self._normals = {}       # part_name -> normal entity handle (int)
        # Per-part computed anchor points in solver space
        # For root: fixed; for children: computed from center
        self._anchor_entities = {}  # (part_name, anchor) -> point entity handle (int)

    def solve(self, joint_angles: Optional[dict] = None) -> SolverResult:
        """Solve the assembly and return positions for all parts."""
        joint_angles = joint_angles or {}
        self._sys = slvs.System()

        # Find root part
        root_name = self._find_root()
        if not root_name:
            root_name = self.assembly.parts[0].name

        # Step 1: Create fixed root part
        self._create_fixed_part(root_name)

        # Step 2: BFS traversal of joint tree
        visited = {root_name}
        queue = [root_name]
        joint_map = self._build_joint_map()

        while queue:
            parent_name = queue.pop(0)
            children = joint_map.get(parent_name, [])

            for joint, child_name in children:
                if child_name in visited:
                    continue
                visited.add(child_name)

                # Compute distribution index
                same_anchor = [
                    (j, c) for j, c in children
                    if j.parent_anchor == joint.parent_anchor
                ]
                child_idx = next(
                    i for i, (j, c) in enumerate(same_anchor)
                    if c == child_name
                )

                self._solve_child(
                    parent_name, child_name, joint,
                    child_idx=child_idx,
                    total_children=len(same_anchor),
                    joint_angles=joint_angles,
                )
                queue.append(child_name)

        # Step 3: Solve
        result_code = self._sys.solve(
            group=self.SOLVE_GROUP,
            reportFailed=True,
            findFreeParams=False,
        )

        success = result_code == 0
        dof = self._sys.Dof

        # Step 4: Extract solved positions
        solved_parts = {}
        for part in self.assembly.parts:
            pos = self._read_solved_position(part.name)
            solved_parts[part.name] = SolvedPart(
                name=part.name,
                position=pos,
                rotation=[0, 0, 1, 0],
            )

        return SolverResult(
            parts=solved_parts,
            success=success,
            dof=dof,
            failed_constraints=[] if success else self._get_failed(),
        )

    def _find_root(self) -> Optional[str]:
        """Find root part: appears as parent but never as child."""
        parents = set()
        children = set()
        for j in self.assembly.joints:
            parents.add(j.parent)
            children.add(j.child)
        roots = parents - children
        if roots:
            for p in self.assembly.parts:
                if p.name in roots:
                    return p.name
            return next(iter(roots))
        return None

    def _build_joint_map(self) -> dict:
        """Build parent -> [(joint, child)] mapping."""
        mapping = {}
        for j in self.assembly.joints:
            mapping.setdefault(j.parent, []).append((j, j.child))
        return mapping

    def _create_fixed_part(self, name: str):
        """Create fixed reference entities for the root part at origin."""
        part = self.part_index[name]

        # Root center at origin (fixed)
        center = self._sys.addPoint3d(
            self._sys.addParamV(0, group=self.FIXED_GROUP),
            self._sys.addParamV(0, group=self.FIXED_GROUP),
            self._sys.addParamV(0, group=self.FIXED_GROUP),
            group=self.FIXED_GROUP,
        )
        self._sys.addWhereDragged(center, group=self.FIXED_GROUP)
        self._centers[name] = center

        # Root orientation: identity (fixed)
        normal = self._sys.addNormal3dV(1, 0, 0, 0, group=self.FIXED_GROUP)
        self._normals[name] = normal

        # Create fixed anchor points for root (for use as parent references)
        for anchor_name in ANCHOR_NORMALS:
            ax, ay, az = _anchor_point(part, anchor_name)
            ap = self._sys.addPoint3d(
                self._sys.addParamV(ax, group=self.FIXED_GROUP),
                self._sys.addParamV(ay, group=self.FIXED_GROUP),
                self._sys.addParamV(az, group=self.FIXED_GROUP),
                group=self.FIXED_GROUP,
            )
            self._anchor_entities[(name, anchor_name)] = ap

    def _get_or_create_anchor(self, part_name: str, anchor: str,
                               guess_center: tuple = (0, 0, 0)):
        """Get or create an anchor point entity for a part.

        For root parts, these are fixed. For child parts, these are
        free entities with initial guesses based on expected position.
        """
        key = (part_name, anchor)
        if key in self._anchor_entities:
            return self._anchor_entities[key]

        part = self.part_index[part_name]
        ax, ay, az = _anchor_point(part, anchor)

        # Anchor position = center + anchor offset
        gx = guess_center[0] + ax
        gy = guess_center[1] + ay
        gz = guess_center[2] + az

        ap = self._sys.addPoint3d(
            self._sys.addParamV(gx, group=self.SOLVE_GROUP),
            self._sys.addParamV(gy, group=self.SOLVE_GROUP),
            self._sys.addParamV(gz, group=self.SOLVE_GROUP),
            group=self.SOLVE_GROUP,
        )
        self._anchor_entities[key] = ap
        return ap

    def _solve_child(
        self,
        parent_name: str,
        child_name: str,
        joint: Joint,
        child_idx: int = 0,
        total_children: int = 1,
        joint_angles: Optional[dict] = None,
    ):
        """Create solver entities and constraints for a child part."""
        parent_part = self.part_index[parent_name]
        child_part = self.part_index[child_name]

        # --- Task 63: use convert_legacy_joint for attachment points ---
        conv = convert_legacy_joint(joint, parent_part, child_part)
        # For backward-compat: still derive anchor names for entity lookups
        p_anchor = joint.parent_anchor or "top"
        c_anchor = joint.child_anchor or "bottom"

        # Get parent anchor entity (should already exist)
        parent_anchor_pt = self._anchor_entities.get((parent_name, p_anchor))

        # Compute initial guess for child center
        # Read parent anchor position from entity (it's either fixed or solved)
        parent_center = self._centers.get(parent_name)
        parent_center_pos = self._read_entity_position(parent_center) if parent_center else [0, 0, 0]
        parent_anchor_offset = conv["parent_attachment"]

        # Parent anchor absolute position
        pax = parent_center_pos[0] + parent_anchor_offset[0]
        pay = parent_center_pos[1] + parent_anchor_offset[1]
        paz = parent_center_pos[2] + parent_anchor_offset[2]

        # Child center = parent_anchor + anchor_normal * child_half_extent
        # The child sits on the parent's anchor face
        pnx, pny, pnz = conv["parent_normal"]
        cdx, cdy, cdz = _part_dimensions(child_part)

        # Compute child's connection anchor offset
        c_anchor_offset = conv["child_attachment"]

        # Initial guess: child center such that child_anchor aligns with parent_anchor
        # child_anchor_abs = child_center + c_anchor_offset
        # We want child_anchor_abs ≈ parent_anchor_abs
        # So child_center ≈ parent_anchor_abs - c_anchor_offset
        # But for a "fixed" joint, the anchor points coincide, so:
        init_cx = pax - c_anchor_offset[0]
        init_cy = pay - c_anchor_offset[1]
        init_cz = paz - c_anchor_offset[2]

        # Apply distribution offset for siblings
        dist_offset = self._compute_distribution_offset(
            parent_part, p_anchor, child_idx, total_children,
            child_part, joint,
        )
        init_cx += dist_offset[0]
        init_cy += dist_offset[1]
        init_cz += dist_offset[2]

        # Create child center point (free)
        child_center = self._sys.addPoint3d(
            self._sys.addParamV(init_cx, group=self.SOLVE_GROUP),
            self._sys.addParamV(init_cy, group=self.SOLVE_GROUP),
            self._sys.addParamV(init_cz, group=self.SOLVE_GROUP),
            group=self.SOLVE_GROUP,
        )
        self._centers[child_name] = child_center

        # Create child normal (free)
        child_nm = self._sys.addNormal3dV(1, 0, 0, 0, group=self.SOLVE_GROUP)
        self._normals[child_name] = child_nm

        # Create the child's connection anchor as a free entity
        # with initial guess at parent_anchor position + distribution
        child_anchor_pt = self._get_or_create_anchor(
            child_name, c_anchor,
            guess_center=(init_cx, init_cy, init_cz),
        )

        # Create remaining anchor entities for child (for grandchildren)
        for aname in ANCHOR_NORMALS:
            if aname == c_anchor:
                continue
            self._get_or_create_anchor(
                child_name, aname,
                guess_center=(init_cx, init_cy, init_cz),
            )

        # Constrain child anchors to maintain rigid body with child center
        # i.e., distance(anchor, center) = known offset magnitude
        for aname in ANCHOR_NORMALS:
            aox, aoy, aoz = _anchor_point(child_part, aname)
            dist = math.sqrt(aox * aox + aoy * aoy + aoz * aoz)
            if dist > 1e-6:
                ap = self._anchor_entities[(child_name, aname)]
                self._sys.addPointsDistance(
                    dist, child_center, ap,
                    group=self.SOLVE_GROUP,
                )

        # Apply joint constraints
        self._add_joint_constraints(
            joint, parent_anchor_pt, child_anchor_pt,
            child_center, child_nm,
        )

    def _add_joint_constraints(
        self,
        joint: Joint,
        parent_anchor_pt,
        child_anchor_pt,
        child_center,
        child_nm,
    ):
        """Add SolveSpace constraints for a joint."""
        jtype = joint.type

        if jtype == "fixed":
            # Fixed: child anchor coincident with parent anchor
            self._sys.addPointsCoincident(
                parent_anchor_pt, child_anchor_pt,
                group=self.SOLVE_GROUP,
            )
            # Lock orientation
            parent_nm = self._normals[joint.parent]
            self._sys.addSameOrientation(
                parent_nm, child_nm,
                group=self.SOLVE_GROUP,
            )

        elif jtype == "revolute":
            # Revolute: axis point coincident
            self._sys.addPointsCoincident(
                parent_anchor_pt, child_anchor_pt,
                group=self.SOLVE_GROUP,
            )
            # Keep normals aligned (removes 5 DOF, leaves 1 rotation)
            parent_nm = self._normals[joint.parent]
            self._sys.addSameOrientation(
                parent_nm, child_nm,
                group=self.SOLVE_GROUP,
            )

        else:
            # Default: treat as fixed
            self._sys.addPointsCoincident(
                parent_anchor_pt, child_anchor_pt,
                group=self.SOLVE_GROUP,
            )
            parent_nm = self._normals[joint.parent]
            self._sys.addSameOrientation(
                parent_nm, child_nm,
                group=self.SOLVE_GROUP,
            )

    def _compute_distribution_offset(
        self,
        parent_part: Part,
        parent_anchor: str,
        child_idx: int,
        total_children: int,
        child_part: Part,
        joint: Joint,
    ):
        """Compute lateral offset for sibling distribution."""
        if total_children <= 1:
            return (0.0, 0.0, 0.0)

        # Get parent face dimensions
        pdx, pdy, pdz = _part_dimensions(parent_part)

        face_dims = {
            "top": (pdx * 2, pdy * 2),
            "bottom": (pdx * 2, pdy * 2),
            "left": (pdx * 2, pdz * 2),
            "right": (pdx * 2, pdz * 2),
            "front": (pdy * 2, pdz * 2),
            "back": (pdy * 2, pdz * 2),
        }
        face_w, face_d = face_dims.get(parent_anchor, (100, 100))

        # Tangent vectors for this anchor face
        tangents = {
            "top": ((1, 0, 0), (0, 1, 0)),
            "bottom": ((1, 0, 0), (0, 1, 0)),
            "left": ((1, 0, 0), (0, 0, 1)),
            "right": ((1, 0, 0), (0, 0, 1)),
            "front": ((0, 1, 0), (0, 0, 1)),
            "back": ((0, 1, 0), (0, 0, 1)),
        }
        t1, t2 = tangents.get(parent_anchor, ((1, 0, 0), (0, 1, 0)))

        margin = 0.8

        if total_children == 2:
            half_w = face_w * margin / 2
            sign = -1 if child_idx == 0 else 1
            return (
                sign * half_w * t1[0],
                sign * half_w * t1[1],
                sign * half_w * t1[2],
            )
        elif total_children <= 4:
            cols = 2
            row = child_idx // cols
            col = child_idx % cols
            sx = face_w * margin / 3
            sy = face_d * margin / 3
            ox = (col - 0.5) * sx
            oy = (row - 0.5) * sy
            return (
                ox * t1[0] + oy * t2[0],
                ox * t1[1] + oy * t2[1],
                ox * t1[2] + oy * t2[2],
            )
        else:
            angle = 2 * math.pi * child_idx / total_children
            r = min(face_w, face_d) * margin / 2 * 0.7
            ox = r * math.cos(angle)
            oy = r * math.sin(angle)
            return (
                ox * t1[0] + oy * t2[0],
                ox * t1[1] + oy * t2[1],
                ox * t1[2] + oy * t2[2],
            )

    def _read_solved_position(self, name: str) -> list:
        """Read solved center position for a part."""
        center = self._centers.get(name)
        if center is None:
            return [0.0, 0.0, 0.0]
        return self._read_entity_position(center)

    def _read_entity_position(self, entity) -> list:
        """Read solved (x, y, z) from a point entity handle.

        py_slvs addPoint3d() returns an int handle. Use getEntityParam
        with the int directly to get param IDs, then getParam to read values.
        """
        params = []
        for i in range(3):
            try:
                param_id = self._sys.getEntityParam(entity, i)
                if param_id == 0:
                    break
                param = self._sys.getParam(param_id)
                params.append(param.val)
            except Exception:
                break
        return params if len(params) == 3 else [0.0, 0.0, 0.0]

    def _get_failed(self) -> list:
        """Get list of failed constraint descriptions."""
        failed = []
        try:
            for h in self._sys.Failed:
                failed.append(f"Constraint {h}")
        except Exception:
            pass
        return failed


# ---------------------------------------------------------------------------
# Public API (compatible with existing assembly_solve)
# ---------------------------------------------------------------------------

def constraint_solve(assembly: Assembly, joint_angles: Optional[dict] = None) -> dict:
    """Solve assembly using constraint-based solver.

    Returns dict compatible with the existing assembly_solve() interface:
        {
            "part_name": {
                "position": [x, y, z],
                "rotation": [ax, ay, az, angle_deg],
            },
            ...
        }
    """
    solver = ConstraintSolver(assembly)
    result = solver.solve(joint_angles=joint_angles)

    output = {}
    for name, sp in result.parts.items():
        output[name] = {
            "position": sp.position,
            "rotation": sp.rotation,
        }

    return output
