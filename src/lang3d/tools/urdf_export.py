"""URDF export — convert Assembly to ROS2 URDF/Xacro + Gazebo simulation config.

Public API:
  - AssemblyToURDF  : core converter (Assembly → URDF XML string)
  - ROS2PackageBuilder : generate complete ROS2 package directory structure
  - URDFExportTool  : Agent tool wrapping the converter
  - register_urdf_tools : registration helper
"""

from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..knowledge.mechanics import Assembly, Joint, Part
from ..models.base import ToolDefinition
from .base import Tool

# ============================================================================
# Constants
# ============================================================================

# Standard colors for URDF materials
_DEFAULT_MATERIALS: dict[str, str] = {
    "gray": "0.7 0.7 0.7 1.0",
    "blue": "0.0 0.0 0.8 1.0",
    "green": "0.0 0.8 0.0 1.0",
    "red": "0.8 0.0 0.0 1.0",
    "orange": "1.0 0.5 0.0 1.0",
    "yellow": "1.0 1.0 0.0 1.0",
    "black": "0.1 0.1 0.1 1.0",
    "white": "1.0 1.0 1.0 1.0",
}

# Maps joint.type to URDF joint type
_JOINT_TYPE_MAP: dict[str, str] = {
    "revolute": "revolute",
    "continuous": "continuous",
    "prismatic": "prismatic",
    "fixed": "fixed",
}

# Maps joint.axis string to URDF xyz vector
_AXIS_MAP: dict[str, list[float]] = {
    "x": [1.0, 0.0, 0.0],
    "y": [0.0, 1.0, 0.0],
    "z": [0.0, 0.0, 1.0],
}


# ============================================================================
# Helper functions
# ============================================================================


def _sanitize_name(name: str) -> str:
    """Make a name URDF-safe (lowercase, underscores, no special chars)."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "link"


def _mm_to_m(v: float) -> float:
    """Convert millimeters to meters."""
    return v / 1000.0


def _mm2_to_m2(v: float) -> float:
    """Convert mm² to m²."""
    return v / 1_000_000.0


def _infer_collision_primitive(part: Part) -> tuple[str, tuple[float, ...]] | None:
    """Infer a simple collision primitive for a part based on its dimensions.

    Returns:
        ("box", (length_m, width_m, height_m)) for box-like parts
        ("cylinder", (radius_m, length_m)) for cylindrical parts
        None if no simple primitive fits (use mesh fallback)

    Standard practice in robotics simulation: use simplified collision
    geometry (box/cylinder) for contact physics, keep detailed mesh only
    for visual rendering.  This gives:
      - Faster collision detection (O(1) vs O(triangles))
      - Predictable contacts (flat faces vs noisy mesh surface)
      - Better grasp behavior (parallel clamping surfaces)
    """
    d = part.dimensions
    has_box_dims = (
        d.get("length", 0) > 0
        and d.get("width", 0) > 0
        and d.get("height", 0) > 0
    )
    has_cyl_dims = (
        (d.get("diameter", 0) > 0 or d.get("outer_diameter", 0) > 0)
        and d.get("height", 0) > 0
        and "length" not in d
        and "width" not in d
    )

    if has_cyl_dims:
        dia = d.get("diameter", d.get("outer_diameter", 0))
        h = d["height"]
        return ("cylinder", (_mm_to_m(dia / 2.0), _mm_to_m(h)))

    if has_box_dims:
        # URDF <box size="X Y Z">.  Solver convention: X=width, Y=length,
        # Z=height (matches freecad.py makeBox(width, length, height) and
        # _infer_inertia/_infer_local_com which use lx=width, ly=length).
        # Previously this emitted (length, width, height) — X/Y swapped
        # relative to the mesh, mis-sizing the collision box.
        return ("box", (
            _mm_to_m(d["width"]),
            _mm_to_m(d["length"]),
            _mm_to_m(d["height"]),
        ))

    return None


def _resolve_axis(joint: Joint) -> list[float]:
    # ``joint.axis`` may be None when the LLM omits it (or for fixed joints
    # that have no rotational axis). Guard the .lower() so export doesn't
    # crash on a None axis — fall through to the anchor-based inference,
    # which always returns a valid vector. Same class of None-guard the
    # pipeline broadened for Joint string fields (commit on export safety).
    axis = joint.axis or ""
    if axis != "auto" and axis.lower() in _AXIS_MAP:
        return _AXIS_MAP[axis.lower()]
    # Infer from parent_anchor
    anchor = (joint.parent_anchor or "").lower()
    if anchor in ("top", "bottom"):
        return [0.0, 0.0, 1.0]
    elif anchor in ("left", "right"):
        return [1.0, 0.0, 0.0]
    elif anchor in ("front", "back"):
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]  # default z


def _infer_inertia(part: Part) -> dict[str, float]:
    """Infer simplified inertia tensor for a part.

    Returns dict with ixx, ixy, ixz, iyy, iyz, izz (kg·m²).
    Uses bounding-box or cylinder approximation.
    """
    mass = part.compute_estimated_mass()
    if mass <= 0:
        mass = 0.01  # fallback 10g

    dims = part.dimensions
    # Try cylinder
    if "diameter" in dims or "outer_diameter" in dims:
        d = dims.get("diameter", dims.get("outer_diameter", 10.0))
        h = dims.get("height", dims.get("thickness", dims.get("length", 10.0)))
        r = _mm_to_m(d / 2.0)
        h_m = _mm_to_m(h)
        ixx = mass * (3 * r * r + h_m * h_m) / 12.0
        izz = mass * r * r / 2.0
        return {"ixx": round(ixx, 8), "ixy": 0.0, "ixz": 0.0,
                "iyy": round(ixx, 8), "iyz": 0.0, "izz": round(izz, 8)}

    # Try box — matches freecad.py makeBox(width, length, height):
    # STL X = width, Y = length, Z = height (see _infer_local_com docstring).
    lx = _mm_to_m(dims.get("width", 10.0))
    ly = _mm_to_m(dims.get("length", dims.get("width", 10.0)))
    lz = _mm_to_m(dims.get("height", dims.get("thickness", 10.0)))
    if lx <= 0:
        lx = 0.01
    if ly <= 0:
        ly = 0.01
    if lz <= 0:
        lz = 0.01

    ixx = mass * (ly * ly + lz * lz) / 12.0
    iyy = mass * (lx * lx + lz * lz) / 12.0
    izz = mass * (lx * lx + ly * ly) / 12.0
    return {"ixx": round(ixx, 8), "ixy": 0.0, "ixz": 0.0,
            "iyy": round(iyy, 8), "iyz": 0.0, "izz": round(izz, 8)}


def _infer_local_com(part: Part) -> tuple[float, float, float]:
    """Infer a part's centre of mass in its local (link) frame, in mm.

    Returns the geometric centre of the part's bounding shape.  Used as the
    URDF ``<inertial><origin xyz>`` when ``part.center_of_mass`` is unset
    (the common case — the field defaults to ``(0,0,0)`` and the pipeline
    never populates it).

    This MUST be consistent with ``_infer_inertia``: the inertia tensor
    produced there is the uniform-body tensor *about the geometric centre*,
    so the inertial origin must point at that same centre.  An origin of
    ``(0,0,0)`` while the tensor is centre-relative is what made MuJoCo
    PD-hold diverge (gravity torque arm computed against the wrong point).

    Coordinate convention follows the ACTUAL FreeCAD STL frame.  The
    FreeCAD exporter (``freecad.py:1491``) calls ``Part.makeBox(width,
    length, height)``, so the STL vertex frame has:

      * **box** X axis = ``width``, Y axis = ``length``, Z = ``height``
        → geometric centre is ``(w/2, l/2, h/2)``
      * **cylinder** along +Z, centred in XY → ``(0, 0, h/2)``

    For non-uniform parts (holes, fillets) this is an approximation, on par
    with the ``_infer_inertia`` uniform-body approximation — using both
    together keeps the tensor and origin self-consistent.

    History: prior to 2026-06-22 this returned ``(l/2, w/2, h/2)`` assuming
    ``makeBox(length, width, height)``.  But the exporter actually swaps to
    ``makeBox(width, length, height)``, so the X/Y were swapped relative
    to the real STL — this caused MuJoCo finger prismatic joints to slide
    6.9mm under PD-hold (the COM was offset 23mm from the slide axis).
    """
    dims = part.dimensions

    # Cylinder (mirrors _infer_inertia's branch order and key lookup)
    if "diameter" in dims or "outer_diameter" in dims:
        h = dims.get("height", dims.get("thickness", dims.get("length", 10.0)))
        return (0.0, 0.0, h / 2.0)

    # Box — matches freecad.py makeBox(width, length, height):
    # STL X = width, Y = length, Z = height.  Centre = (W/2, L/2, H/2).
    lx = dims.get("width", 10.0)
    ly = dims.get("length", dims.get("width", 10.0))
    lz = dims.get("height", dims.get("thickness", 10.0))
    return (lx / 2.0, ly / 2.0, lz / 2.0)


def _pick_material_color(part: Part) -> str:
    """Pick a default material color based on part category."""
    cat = part.category.lower()
    if "joint" in cat:
        return "orange"
    if "actuator" in cat or "servo" in cat:
        return "blue"
    if "structural" in cat:
        return "gray"
    return "gray"


def _axis_angle_to_rpy(aa: list[float]) -> tuple[float, float, float]:
    """Convert axis-angle [ax, ay, az, angle_deg] to roll-pitch-yaw (radians).

    Uses Rodrigues' rotation formula to build a rotation matrix,
    then extracts ZYX Euler angles (roll, pitch, yaw).
    """
    if len(aa) < 4 or abs(aa[3]) < 1e-6:
        return (0.0, 0.0, 0.0)

    ax, ay, az = aa[0], aa[1], aa[2]
    deg = aa[3]
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-10:
        return (0.0, 0.0, 0.0)
    ax /= norm
    ay /= norm
    az /= norm

    angle_rad = math.radians(deg)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    t = 1 - c

    # Rodrigues' rotation matrix
    r00 = t * ax * ax + c
    r01 = t * ax * ay - s * az
    r02 = t * ax * az + s * ay
    r10 = t * ax * ay + s * az
    r11 = t * ay * ay + c
    r12 = t * ay * az - s * ax
    r20 = t * ax * az - s * ay
    r21 = t * ay * az + s * ax
    r22 = t * az * az + c

    # Extract ZYX Euler angles from rotation matrix
    # pitch (Y rotation) from -r20
    sin_pitch = -r20
    sin_pitch = max(-1.0, min(1.0, sin_pitch))

    if abs(cos_pitch := math.cos(math.asin(sin_pitch))) > 1e-6:
        roll = math.atan2(r21, r22)
        pitch = math.asin(sin_pitch)
        yaw = math.atan2(r10, r00)
    else:
        # Gimbal lock: set yaw = 0
        roll = math.atan2(-r12, r11)
        pitch = math.asin(sin_pitch)
        yaw = 0.0

    return (roll, pitch, yaw)


def _axis_angle_to_rot_matrix(aa: list[float]) -> list[list[float]]:
    """Convert axis-angle [ax, ay, az, deg] to 3x3 rotation matrix (Rodrigues)."""
    if len(aa) < 4 or abs(aa[3]) < 1e-6:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ax, ay, az = aa[0], aa[1], aa[2]
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-10:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ax /= norm
    ay /= norm
    az /= norm
    angle_rad = math.radians(aa[3])
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    t = 1 - c
    return [
        [t * ax * ax + c,     t * ax * ay - s * az, t * ax * az + s * ay],
        [t * ax * ay + s * az, t * ay * ay + c,     t * ay * az - s * ax],
        [t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c],
    ]


def _mat3_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """3x3 matrix multiply."""
    r = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r[i][j] += a[i][k] * b[k][j]
    return r


def _mat3_transpose(m: list[list[float]]) -> list[list[float]]:
    """Transpose a 3x3 matrix."""
    return [[m[j][i] for j in range(3)] for i in range(3)]


def _rot_matrix_to_rpy(m: list[list[float]]) -> tuple[float, float, float]:
    """Extract ZYX Euler angles (roll, pitch, yaw) from 3x3 rotation matrix."""
    sin_pitch = -m[2][0]
    sin_pitch = max(-1.0, min(1.0, sin_pitch))
    if abs(cos_pitch := math.cos(math.asin(sin_pitch))) > 1e-6:
        roll = math.atan2(m[2][1], m[2][2])
        pitch = math.asin(sin_pitch)
        yaw = math.atan2(m[1][0], m[0][0])
    else:
        roll = math.atan2(-m[1][2], m[1][1])
        pitch = math.asin(sin_pitch)
        yaw = 0.0
    return (roll, pitch, yaw)


def _relative_axis_angle_to_rpy(
    parent_aa: list[float], child_aa: list[float]
) -> tuple[float, float, float]:
    """Compute relative rotation R_child * R_parent^T and return as RPY.

    This gives the orientation of the child frame expressed in the parent
    frame — exactly what URDF joint origin rpy needs.
    """
    r_child = _axis_angle_to_rot_matrix(child_aa)
    r_parent = _axis_angle_to_rot_matrix(parent_aa)
    r_rel = _mat3_mul(r_child, _mat3_transpose(r_parent))
    return _rot_matrix_to_rpy(r_rel)


# ============================================================================
# AssemblyToURDF — core converter
# ============================================================================


@dataclass
class URDFLink:
    """Intermediate representation of a URDF link."""
    name: str
    mass: float = 0.0  # kg
    com: tuple[float, float, float] = (0.0, 0.0, 0.0)  # meters
    inertia: dict[str, float] = field(default_factory=dict)
    visual_mesh: str = ""  # relative STL path
    collision_mesh: str = ""  # relative STL path (may be same as visual)
    material_color: str = "gray"
    # Optional collision primitive for simplified contact geometry.
    # When set, the URDF <collision> element uses this primitive instead
    # of the full STL mesh.  This is standard practice in robotics:
    # visual geometry is detailed (mesh), collision geometry is simple
    # (box/cylinder) for fast and predictable contacts.
    # Tuple format: ("box", (length_m, width_m, height_m)) or
    #               ("cylinder", (radius_m, length_m))
    collision_primitive: tuple[str, tuple[float, ...]] | None = None


@dataclass
class URDFJoint:
    """Intermediate representation of a URDF joint."""
    name: str
    type: str = "fixed"  # revolute, prismatic, fixed, continuous
    parent: str = ""
    child: str = ""
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)  # meters
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)  # radians
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    lower: float = 0.0  # radians
    upper: float = 0.0  # radians
    effort: float = 5.0  # N·m — typical hobby servo (was 100.0, 20× too high)
    velocity: float = 5.2  # rad/s — ~0.1s/60° (was 3.14)
    mimic_joint: str = ""           # If set, this joint mimics another
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0


@dataclass
class GazeboPlugin:
    """Gazebo simulation plugin configuration."""
    name: str
    filename: str
    params: dict[str, Any] = field(default_factory=dict)


class AssemblyToURDF:
    """Convert an Assembly to URDF XML.

    Usage::

        converter = AssemblyToURDF(assembly, meshes_dir="meshes")
        xml_string = converter.convert()
    """

    def __init__(
        self,
        assembly: Assembly,
        meshes_dir: str = "meshes",
        package_name: str = "",
        positions: dict[str, dict] | None = None,
    ) -> None:
        self.assembly = assembly
        self.meshes_dir = meshes_dir
        self.package_name = package_name or _sanitize_name(assembly.name)
        self.positions = positions or {}

        self._links: list[URDFLink] = []
        self._joints: list[URDFJoint] = []
        self._gazebo_plugins: list[GazeboPlugin] = []
        self._part_index: dict[str, Part] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(self) -> str:
        """Build the intermediate model and return URDF XML string."""
        self._part_index = {p.name: p for p in self.assembly.parts}
        self._build_links()
        self._build_joints()
        self._auto_add_gazebo_plugins()
        return self._render_xml()

    def get_links(self) -> list[URDFLink]:
        """Return intermediate links (call after convert)."""
        return self._links

    def get_joints(self) -> list[URDFJoint]:
        """Return intermediate joints (call after convert)."""
        return self._joints

    def get_gazebo_plugins(self) -> list[GazeboPlugin]:
        """Return gazebo plugins (call after convert)."""
        return self._gazebo_plugins

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_links(self) -> None:
        for part in self.assembly.parts:
            link_name = _sanitize_name(part.name)
            mass = part.compute_estimated_mass()
            if mass <= 0:
                mass = 0.01

            com_m = (
                _mm_to_m(part.center_of_mass[0]),
                _mm_to_m(part.center_of_mass[1]),
                _mm_to_m(part.center_of_mass[2]),
            )
            # part.center_of_mass defaults to (0,0,0) and the pipeline never
            # populates it; a zero origin paired with a centre-relative
            # inertia tensor makes MuJoCo PD-hold diverge (wrong gravity
            # torque arm).  Fall back to the geometric centre so the tensor
            # and its reference point stay self-consistent.  See AGENTS.md §3.2.
            if part.center_of_mass == (0.0, 0.0, 0.0):
                com = _infer_local_com(part)
                com_m = (_mm_to_m(com[0]), _mm_to_m(com[1]), _mm_to_m(com[2]))

            inertia = _infer_inertia(part)

            # If user provided an inertia tensor with non-zero diagonal, use it
            it = part.inertia_tensor
            if any(it[i][i] > 0 for i in range(3)):
                inertia = {
                    "ixx": round(_mm2_to_m2(it[0][0]), 8),
                    "ixy": round(_mm2_to_m2(it[0][1]), 8),
                    "ixz": round(_mm2_to_m2(it[0][2]), 8),
                    "iyy": round(_mm2_to_m2(it[1][1]), 8),
                    "iyz": round(_mm2_to_m2(it[1][2]), 8),
                    "izz": round(_mm2_to_m2(it[2][2]), 8),
                }

            mesh_file = f"{self.meshes_dir}/{link_name}.stl"
            self._links.append(URDFLink(
                name=link_name,
                mass=round(mass, 6),
                com=com_m,
                inertia=inertia,
                visual_mesh=mesh_file,
                collision_mesh=mesh_file,
                material_color=_pick_material_color(part),
                collision_primitive=_infer_collision_primitive(part),
            ))

    def _build_joints(self) -> None:
        # Build child→joint-name lookup for mimic joint resolution
        child_to_joint_name: dict[str, str] = {}
        for j in self.assembly.joints:
            p = _sanitize_name(j.parent)
            c = _sanitize_name(j.child)
            child_to_joint_name[c] = f"{p}_to_{c}"

        for i, joint in enumerate(self.assembly.joints):
            jtype = _JOINT_TYPE_MAP.get(joint.type, "fixed")
            parent_name = _sanitize_name(joint.parent)
            child_name = _sanitize_name(joint.child)

            # Try solver positions first for computing joint origin
            origin_xyz, origin_rpy = self._compute_joint_origin(joint)

            # Fallback: use joint offset + parent part dimensions
            if origin_xyz is None:
                joint_offset = joint.offset or (0, 0, 0)
                origin_xyz = (
                    _mm_to_m(joint_offset[0]),
                    _mm_to_m(joint_offset[1]),
                    _mm_to_m(joint_offset[2]),
                )

                # For non-root children, estimate origin from parent part dimensions
                parent_part = self._part_index.get(joint.parent)
                if parent_part and origin_xyz == (0.0, 0.0, 0.0):
                    anchor = joint.parent_anchor.lower()
                    dims = parent_part.dimensions
                    origin_m: tuple[float, float, float] | None = None
                    if anchor in ("top", "bottom"):
                        h = dims.get("height", dims.get("thickness", 0.0))
                        if h > 0:
                            origin_m = (0.0, 0.0, _mm_to_m(h))
                    elif anchor in ("front", "back"):
                        # front/back faces have outward normals along ±Y, so
                        # the child sits offset along the part's length axis.
                        ln = dims.get("length", dims.get("depth", 0.0))
                        if ln > 0:
                            origin_m = (0.0, _mm_to_m(ln), 0.0)
                    elif anchor in ("left", "right"):
                        w = dims.get("width", dims.get("diameter", 0.0))
                        if w > 0:
                            origin_m = (_mm_to_m(w), 0.0, 0.0)
                    if origin_m is not None:
                        origin_xyz = origin_m

                origin_rpy = (0.0, 0.0, 0.0)

            axis_vec = _resolve_axis(joint)
            if joint.type == "prismatic":
                # prismatic range_deg semantically holds millimeters; URDF limit is in meters
                lower_rad = _mm_to_m(joint.range_deg[0])
                upper_rad = _mm_to_m(joint.range_deg[1])
            else:
                lower_rad = math.radians(joint.range_deg[0])
                upper_rad = math.radians(joint.range_deg[1])

            # Resolve mimic joint: convert child part name to URDF joint name
            mimic_joint_name = ""
            if joint.mimic_joint:
                mimic_child = _sanitize_name(joint.mimic_joint)
                mimic_joint_name = child_to_joint_name.get(mimic_child, "")

            # Infer effort/velocity from actuator catalog reference.
            # Falls back to dataclass defaults (5.0 N·m, 5.2 rad/s) when no
            # catalog info is available — much more realistic than the old
            # 100.0 N·m which was 500× too high for hobby servos.
            _effort = 5.0
            _velocity = 5.2
            _child_part = self._part_index.get(joint.child)
            if _child_part and _child_part.notes and "catalog:" in _child_part.notes:
                try:
                    from ..knowledge.actuators import get_actuator, torque_to_nm
                    _cat = _child_part.notes.split("catalog:")[1].split()[0].split(",")[0]
                    _act_id = _cat.split("_")[-1].upper()
                    _act = get_actuator(_act_id)
                    if _act and _act.torque_kgcm > 0:
                        _effort = round(torque_to_nm(_act.torque_kgcm) * 2.0, 2)
                        _velocity = round(math.pi / (3.0 * max(_act.speed_s_per_60deg, 0.01)), 2)
                except Exception:
                    pass

            self._joints.append(URDFJoint(
                name=f"{parent_name}_to_{child_name}",
                type=jtype,
                parent=parent_name,
                child=child_name,
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                axis=tuple(axis_vec),  # type: ignore[arg-type]
                lower=round(lower_rad, 4),
                upper=round(upper_rad, 4),
                effort=_effort,
                velocity=_velocity,
                mimic_joint=mimic_joint_name,
                mimic_multiplier=joint.mimic_multiplier,
                mimic_offset=joint.mimic_offset,
            ))

    def _compute_joint_origin(self, joint) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
        """Compute joint origin (xyz, rpy) from solver positions.

        Returns (None, None) if solver positions are unavailable for either
        the parent or child part, signalling the caller to use fallback logic.
        """
        parent_pos = self.positions.get(joint.parent, {}).get("position")
        child_pos = self.positions.get(joint.child, {}).get("position")

        if not parent_pos or not child_pos:
            return None, None

        # Relative displacement: child - parent (mm → m)
        rel_xyz = tuple(_mm_to_m(child_pos[i] - parent_pos[i]) for i in range(3))

        # Relative rotation: R_child * R_parent^(-1) → RPY
        # This gives the orientation of the child frame relative to the parent frame.
        #
        # P0-2: use ``kinematic_rotation`` (pure joint-chain rotation) instead
        # of ``rotation`` (which includes the visual cylinder_orient offset).
        # The visual offset contaminates joint origin RPY and breaks IK/FK
        # round-trips for cylindrical parts (servos, bearings).  Fall back to
        # ``rotation`` for older solver outputs that don't have the split yet.
        def _kin_rot(name: str) -> list[float]:
            p = self.positions.get(name, {})
            return p.get("kinematic_rotation") or p.get("rotation", [0, 0, 1, 0])

        child_rot = _kin_rot(joint.child)
        parent_rot = _kin_rot(joint.parent)
        rpy = _relative_axis_angle_to_rpy(parent_rot, child_rot)

        return rel_xyz, rpy

    def _auto_add_gazebo_plugins(self) -> None:
        """Auto-detect drive train type and add Gazebo plugins."""
        revolute_count = sum(1 for j in self.assembly.joints if j.type == "revolute")

        # Check if this looks like a differential drive (2 revolute joints on z axis)
        z_revolute = sum(
            1 for j in self.assembly.joints
            if j.type == "revolute" and _resolve_axis(j) == [0.0, 0.0, 1.0]
        )

        if z_revolute >= 2:
            # Differential drive
            wheel_joints = [
                j for j in self._joints if j.type == "revolute"
                and j.axis == (0.0, 0.0, 1.0)
            ]
            if len(wheel_joints) >= 2:
                # Derive wheel separation from positions (distance between
                # left/right wheel X coords) or fall back to default.
                wheel_separation = 0.15
                if self.positions:
                    left_name = wheel_joints[0].name.split("_to_")[0]
                    right_name = wheel_joints[1].name.split("_to_")[0]
                    lp = self.positions.get(left_name, {}).get("position")
                    rp = self.positions.get(right_name, {}).get("position")
                    if lp and rp:
                        wheel_separation = abs(
                            _mm_to_m(lp[0] - rp[0])
                        ) or 0.15

                # Derive wheel diameter from part dimensions or fall back.
                wheel_diameter = 0.065
                child_name_0 = wheel_joints[0].name.split("_to_")[-1]
                child_part = self._part_index.get(child_name_0)
                if child_part:
                    d = child_part.dimensions.get(
                        "diameter",
                        child_part.dimensions.get("outer_diameter", 0),
                    )
                    if d > 0:
                        wheel_diameter = _mm_to_m(d)

                self._gazebo_plugins.append(GazeboPlugin(
                    name="diff_drive",
                    filename="libgazebo_ros_diff_drive.so",
                    params={
                        "leftJoint": wheel_joints[0].name,
                        "rightJoint": wheel_joints[1].name,
                        "wheelSeparation": round(wheel_separation, 4),
                        "wheelDiameter": round(wheel_diameter, 4),
                        "commandTopic": "cmd_vel",
                        "odometryTopic": "odom",
                        "odometryFrame": "odom",
                    },
                ))

        if revolute_count > 0:
            # Joint state publisher for arms
            self._gazebo_plugins.append(GazeboPlugin(
                name="joint_state_publisher",
                filename="libgazebo_ros_joint_state_publisher.so",
                params={
                    "jointNames": " ".join(
                        j.name for j in self._joints if j.type in ("revolute", "prismatic")
                    ),
                    "updateRate": 50,
                },
            ))

    def _render_xml(self) -> str:
        root = ET.Element("robot")
        root.set("name", _sanitize_name(self.assembly.name))

        # Links
        for link in self._links:
            link_el = ET.SubElement(root, "link", name=link.name)

            # Inertial
            inertial = ET.SubElement(link_el, "inertial")
            origin = ET.SubElement(inertial, "origin")
            origin.set("xyz", " ".join(f"{v:.6f}" for v in link.com))
            origin.set("rpy", "0 0 0")
            mass_el = ET.SubElement(inertial, "mass")
            mass_el.set("value", f"{link.mass:.6f}")
            inertia_el = ET.SubElement(inertial, "inertia")
            for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
                inertia_el.set(key, f"{link.inertia.get(key, 0.0):.8f}")

            # Visual
            visual = ET.SubElement(link_el, "visual")
            vis_origin = ET.SubElement(visual, "origin")
            vis_origin.set("xyz", "0 0 0")
            vis_origin.set("rpy", "0 0 0")
            vis_geom = ET.SubElement(visual, "geometry")
            vis_mesh = ET.SubElement(vis_geom, "mesh")
            vis_mesh.set("filename", link.visual_mesh)
            # STL files are exported by FreeCAD in millimetres, but URDF
            # convention treats mesh units as metres.  Without an explicit
            # scale, MuJoCo/PyBullet/Gazebo all interpret a 200mm STL as
            # a 200-metre robot — producing massive mesh interpenetration
            # and immediate physics blow-up.  Scale 0.001 converts mm→m.
            vis_mesh.set("scale", "0.001 0.001 0.001")
            mat_el = ET.SubElement(visual, "material")
            mat_el.set("name", link.material_color)
            color_el = ET.SubElement(mat_el, "color")
            color_el.set("rgba", _DEFAULT_MATERIALS.get(link.material_color, "0.7 0.7 0.7 1.0"))

            # Collision
            # Use simplified primitive geometry when available — standard
            # robotics practice that gives faster + more predictable
            # contacts than concave STL meshes.  Visual stays as full mesh.
            collision = ET.SubElement(link_el, "collision")
            col_origin = ET.SubElement(collision, "origin")
            col_origin.set("xyz", "0 0 0")
            col_origin.set("rpy", "0 0 0")
            col_geom = ET.SubElement(collision, "geometry")
            if link.collision_primitive is not None:
                prim_type, prim_dims = link.collision_primitive
                if prim_type == "box":
                    box_el = ET.SubElement(col_geom, "box")
                    # URDF box size is the FULL edge length
                    box_el.set(
                        "size",
                        f"{prim_dims[0]:.6f} {prim_dims[1]:.6f} {prim_dims[2]:.6f}",
                    )
                elif prim_type == "cylinder":
                    cyl_el = ET.SubElement(col_geom, "cylinder")
                    cyl_el.set("radius", f"{prim_dims[0]:.6f}")
                    cyl_el.set("length", f"{prim_dims[1]:.6f}")
            else:
                # Fallback: full STL mesh (same as visual)
                col_mesh = ET.SubElement(col_geom, "mesh")
                col_mesh.set("filename", link.collision_mesh)
                col_mesh.set("scale", "0.001 0.001 0.001")

        # Joints
        for joint in self._joints:
            joint_el = ET.SubElement(root, "joint", name=joint.name, type=joint.type)
            ET.SubElement(joint_el, "parent", link=joint.parent)
            ET.SubElement(joint_el, "child", link=joint.child)
            origin_el = ET.SubElement(joint_el, "origin")
            origin_el.set("xyz", " ".join(f"{v:.6f}" for v in joint.origin_xyz))
            origin_el.set("rpy", " ".join(f"{v:.6f}" for v in joint.origin_rpy))
            axis_el = ET.SubElement(joint_el, "axis")
            axis_el.set("xyz", " ".join(f"{v:.1f}" for v in joint.axis))

            if joint.type in ("revolute", "prismatic"):
                # URDF spec order: dynamics before limit. Typical hobby-servo
                # joint values; downstream can override via actuator catalog.
                dynamics_el = ET.SubElement(joint_el, "dynamics")
                dynamics_el.set("damping", "0.1")
                dynamics_el.set("friction", "0.05")
                limit_el = ET.SubElement(joint_el, "limit")
                limit_el.set("lower", f"{joint.lower:.4f}")
                limit_el.set("upper", f"{joint.upper:.4f}")
                limit_el.set("effort", f"{joint.effort:.1f}")
                limit_el.set("velocity", f"{joint.velocity:.2f}")

            # Mimic joint coupling (e.g. gripper fingers that move symmetrically)
            if joint.mimic_joint:
                mimic_el = ET.SubElement(joint_el, "mimic")
                mimic_el.set("joint", joint.mimic_joint)
                mimic_el.set("multiplier", f"{joint.mimic_multiplier:.1f}")
                mimic_el.set("offset", f"{joint.mimic_offset:.4f}")

        # Transmissions for actuated joints (ros2_control / Gazebo).
        # Each revolute/prismatic joint gets a SimpleTransmission so
        # gazebo_ros2_control can apply effort commands.
        for joint in self._joints:
            if joint.type not in ("revolute", "prismatic"):
                continue
            trans_el = ET.SubElement(root, "transmission")
            trans_el.set("name", f"{joint.name}_trans")
            type_el = ET.SubElement(trans_el, "type")
            type_el.text = "transmission_interface/SimpleTransmission"
            joint_ref = ET.SubElement(trans_el, "joint", name=joint.name)
            hw_el = ET.SubElement(joint_ref, "hardwareInterface")
            hw_el.text = "hardware_interface/EffortJointInterface"
            actuator_el = ET.SubElement(
                trans_el, "actuator", name=f"{joint.name}_motor",
            )
            act_hw_el = ET.SubElement(actuator_el, "hardwareInterface")
            act_hw_el.text = "hardware_interface/EffortJointInterface"
            mech_el = ET.SubElement(actuator_el, "mechanicalReduction")
            mech_el.text = "1"

        # Gazebo ros2_control plugin for simulation actuation
        gazebo_ctrl = ET.SubElement(root, "gazebo")
        ctrl_plugin = ET.SubElement(
            gazebo_ctrl, "plugin",
            name="gazebo_ros2_control",
            filename="libgazebo_ros2_control.so",
        )
        params_el = ET.SubElement(ctrl_plugin, "parameters")
        params_el.text = f"{self.package_name}/config/ros2_control.yaml"

        # Gazebo plugins
        for plugin in self._gazebo_plugins:
            gazebo = ET.SubElement(root, "gazebo")
            plugin_el = ET.SubElement(gazebo, "plugin", name=plugin.name, filename=plugin.filename)
            if plugin.name == "diff_drive":
                for k, v in plugin.params.items():
                    child = ET.SubElement(plugin_el, k)
                    child.text = str(v)
            elif plugin.name == "joint_state_publisher":
                child = ET.SubElement(plugin_el, "jointNames")
                child.text = str(plugin.params.get("jointNames", ""))
                child2 = ET.SubElement(plugin_el, "updateRate")
                child2.text = str(plugin.params.get("updateRate", 50))

        # Pretty print
        _indent_xml(root)
        return ET.tostring(root, encoding="unicode", xml_declaration=False)


# ============================================================================
# ROS2 Package Builder
# ============================================================================


class ROS2PackageBuilder:
    """Generate a complete ROS2 package directory from a URDF string.

    Directory structure::

        <package_name>/
        ├── package.xml
        ├── CMakeLists.txt
        ├── urdf/
        │   └── <robot>.urdf
        ├── meshes/
        │   └── (STL files go here)
        ├── launch/
        │   └── display.launch.py
        └── config/
            └── joint_names.yaml
    """

    def __init__(self, package_name: str, urdf_xml: str) -> None:
        self.package_name = _sanitize_name(package_name)
        self.urdf_xml = urdf_xml

    def write(self, output_dir: str | Path) -> str:
        """Write the complete ROS2 package to disk. Returns output path."""
        base = Path(output_dir) / self.package_name
        (base / "urdf").mkdir(parents=True, exist_ok=True)
        (base / "meshes").mkdir(parents=True, exist_ok=True)
        (base / "launch").mkdir(parents=True, exist_ok=True)
        (base / "config").mkdir(parents=True, exist_ok=True)

        # URDF file — rewrite mesh paths to package:// URIs so the URDF
        # resolves correctly from its urdf/ subdir.  Without this rewrite,
        # URDFs reference ``meshes/X.stl`` which is interpreted relative
        # to the URDF location (``urdf/meshes/X.stl``) and the meshes
        # are not found by ROS2 rviz2/Gazebo, MuJoCo, or PyBullet.
        urdf_xml_fixed = self._rewrite_mesh_paths_to_package_uri(self.urdf_xml)
        urdf_path = base / "urdf" / f"{self.package_name}.urdf"
        urdf_path.write_text(urdf_xml_fixed, encoding="utf-8")

        # Also write a "flat" URDF inside urdf/ alongside the main URDF.
        # Uses ../meshes/X.stl relative paths so non-ROS2 consumers (MuJoCo,
        # PyBullet) that don't understand package:// URIs can load it
        # directly.  Placing it in urdf/ (not package root) makes ../meshes/
        # resolve correctly to <package>/meshes/.
        flat_urdf = self._rewrite_mesh_paths_relative(self.urdf_xml, "../meshes")
        flat_path = base / "urdf" / f"{self.package_name}_flat.urdf"
        flat_path.write_text(flat_urdf, encoding="utf-8")

        # package.xml
        (base / "package.xml").write_text(self._package_xml(), encoding="utf-8")

        # CMakeLists.txt
        (base / "CMakeLists.txt").write_text(self._cmakeLists(), encoding="utf-8")

        # launch file
        (base / "launch" / "display.launch.py").write_text(
            self._launch_file(), encoding="utf-8"
        )

        # config
        (base / "config" / "joint_names.yaml").write_text(
            self._joint_names_yaml(), encoding="utf-8"
        )
        (base / "config" / "ros2_control.yaml").write_text(
            self._ros2_control_yaml(), encoding="utf-8"
        )
        (base / "config" / "rviz.rviz").write_text(
            self._rviz_config(), encoding="utf-8"
        )

        # Gazebo launch file
        (base / "launch" / "gazebo.launch.py").write_text(
            self._gazebo_launch_file(), encoding="utf-8"
        )

        return str(base)

    def _rewrite_mesh_paths_to_package_uri(self, urdf_xml: str) -> str:
        """Rewrite ``meshes/X.stl`` paths to ``package://<pkg>/meshes/X.stl``.

        URDFs generated by AssemblyToURDF use relative mesh paths
        (``meshes/X.stl``) which don't resolve correctly when the URDF
        lives in a ``urdf/`` subdirectory of a ROS2 package.  This helper
        converts every mesh reference to the ROS2-canonical ``package://``
        URI so the URDF can be loaded from any directory by rviz2, Gazebo,
        or other ROS2 tools.
        """
        import re

        pattern = re.compile(
            r'(<mesh\s+filename=")(meshes/[^"]+)(")',
            re.IGNORECASE,
        )

        def _replace(match: re.Match[str]) -> str:
            prefix, mesh_path, suffix = match.group(1), match.group(2), match.group(3)
            return f'{prefix}package://{self.package_name}/{mesh_path}{suffix}'

        return pattern.sub(_replace, urdf_xml)

    def _rewrite_mesh_paths_relative(self, urdf_xml: str, prefix: str) -> str:
        """Rewrite ``meshes/X.stl`` paths to ``<prefix>/meshes/X.stl``.

        Produces a non-ROS2 URDF for direct loading by MuJoCo / PyBullet
        without needing package:// resolution.  ``prefix="../meshes"``
        makes the URDF work from a ``urdf/`` subdir.
        """
        import re

        pattern = re.compile(
            r'(<mesh\s+filename=")(meshes/[^"]+)(")',
            re.IGNORECASE,
        )

        def _replace(match: re.Match[str]) -> str:
            head, mesh_path, tail = match.group(1), match.group(2), match.group(3)
            return f'{head}{prefix}/{mesh_path.split("/", 1)[1]}{tail}'

        return pattern.sub(_replace, urdf_xml)

    def _package_xml(self) -> str:
        return f"""\
<?xml version="1.0"?>
<package format="3">
  <name>{self.package_name}</name>
  <version>0.0.1</version>
  <description>Auto-generated by Language-3D Agent</description>
  <maintainer email="lang3d@example.com">lang3d</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <exec_depend>robot_state_publisher</exec_depend>
  <exec_depend>joint_state_publisher_gui</exec_depend>
  <exec_depend>rviz2</exec_depend>
  <exec_depend>xacro</exec_depend>
  <exec_depend>gazebo_ros</exec_depend>
  <exec_depend>gazebo_ros2_control</exec_depend>
  <exec_depend>ros2_control</exec_depend>
  <exec_depend>ros2_controllers</exec_depend>
  <exec_depend>joint_trajectory_controller</exec_depend>
  <exec_depend>joint_state_broadcaster</exec_depend>
  <exec_depend>controller_manager</exec_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
"""

    def _cmakeLists(self) -> str:
        return f"""\
cmake_minimum_required(VERSION 3.8)
project({self.package_name})

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf meshes launch config
  DESTINATION share/${{PROJECT_NAME}})

ament_package()
"""

    def _launch_file(self) -> str:
        return f"""\
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('{self.package_name}')
    urdf_file = os.path.join(pkg_dir, 'urdf', '{self.package_name}.urdf')

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{{'robot_description': robot_description}}],
            output='screen',
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', os.path.join(pkg_dir, 'config', 'rviz.rviz')],
            output='screen',
        ),
    ])
"""

    def _joint_names_yaml(self) -> str:
        # Parse joint names from URDF
        try:
            root = ET.fromstring(self.urdf_xml)
            joints = root.findall(".//joint[@type='revolute']")
            joints += root.findall(".//joint[@type='prismatic']")
            names = [j.get("name", "") for j in joints if j.get("name")]
        except ET.ParseError:
            names = []
        joint_list = "\n".join(f"  - \"{n}\"" for n in names)
        return f"joint_names:\n{joint_list}\n"

    def _ros2_control_yaml(self) -> str:
        """Generate ros2_control YAML config for Gazebo simulation."""
        try:
            root = ET.fromstring(self.urdf_xml)
            joints = root.findall(".//joint[@type='revolute']")
            joints += root.findall(".//joint[@type='prismatic']")
            names = [j.get("name", "") for j in joints if j.get("name")]
        except ET.ParseError:
            names = []

        joint_list = "\n".join(f"      - {n}" for n in names)
        return f"""\
controller_manager:
  ros__parameters:
    update_rate: 100
    use_sim_time: true

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    joint_trajectory_controller:
      type: joint_trajectory_controller/JointTrajectoryController

joint_trajectory_controller:
  ros__parameters:
    joints:
{joint_list}
    command_interfaces: [position]
    state_interfaces: [position, velocity]
"""

    def _rviz_config(self) -> str:
        """Minimal RViz2 config: RobotModel + TF + grid."""
        return f"""\
Panels:
  - Class: rviz_common/Displays
    Name: Displays
Visualization Manager:
  Displays:
    - Class: rviz_default_plugins/RobotModel
      Name: RobotModel
      Description Topic:
        Value: /robot_description
      Visual Enabled: true
      Collision Enabled: false
    - Class: rviz_default_plugins/TF
      Name: TF
      Show Axes: true
      Show Names: false
    - Class: rviz_default_plugins/Grid
      Name: Grid
      Cell Size: 0.5
      Plane Cell Count: 20
  Global Options:
    Fixed Frame: base_plate
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 1.5
      Focal Point:
        X: 0.0
        Y: 0.0
        Z: 0.3
"""

    def _gazebo_launch_file(self) -> str:
        """Launch file: start Gazebo, spawn the robot, publish state."""
        return f"""\
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('{self.package_name}')
    urdf_file = os.path.join(pkg_dir, 'urdf', '{self.package_name}.urdf')

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # Start Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'),
                         'launch', 'gazebo.launch.py')
        ),
    )

    # Spawn the robot in Gazebo
    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', '{self.package_name}',
                   '-topic', '/robot_description'],
        output='screen',
    )

    # Publish robot state
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{{'robot_description': robot_description}}],
        output='screen',
    )

    return LaunchDescription([gazebo, spawn, robot_state_publisher])
"""


# ============================================================================
# XML pretty-print helper
# ============================================================================


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    """Add indentation to XML tree (in-place)."""
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent
    if level == 0:
        elem.tail = "\n"


# ============================================================================
# Agent Tool
# ============================================================================


class URDFExportTool(Tool):
    """Export an Assembly to URDF format for ROS2 simulation."""

    name = "urdf_export"
    description = "Export an Assembly to ROS2 URDF format (robot description + meshes)"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "assembly_name": {
                        "type": "string",
                        "description": "Name of the assembly to export (must exist in workspace)",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for the ROS2 package (default: workspace)",
                    },
                    "package_name": {
                        "type": "string",
                        "description": "ROS2 package name (default: sanitized assembly name)",
                    },
                    "mode": {
                        "type": "string",
                        "description": "Output mode: 'xml' returns URDF XML string, 'package' writes full ROS2 package (default: package)",
                    },
                },
                "required": ["assembly_name"],
            },
        )

    def execute(
        self,
        *,
        assembly_name: str,
        output_dir: str = "",
        package_name: str = "",
        mode: str = "package",
        **kwargs: Any,
    ) -> str:
        # Try to load assembly from the global assemblies dict
        assembly = _find_assembly(assembly_name)
        if assembly is None:
            return f"错误：未找到装配体 '{assembly_name}'"

        try:
            from .assembly_solver import AssemblySolver
            solver = AssemblySolver(assembly)
            # Solve with joint_angles={} so URDF joint origins describe the
            # home pose (all joints at zero), not the default pose.  See
            # export_package.py step 4 for the rationale.
            positions = solver.solve(joint_angles={})
            converter = AssemblyToURDF(
                assembly,
                meshes_dir="meshes",
                package_name=package_name,
                positions=positions,
            )
            urdf_xml = converter.convert()
        except Exception as e:
            return f"URDF 转换失败：{e}"

        if mode == "xml":
            return urdf_xml

        # Full ROS2 package
        out = output_dir or os.getcwd()
        builder = ROS2PackageBuilder(
            package_name=package_name or assembly.name,
            urdf_xml=urdf_xml,
        )
        try:
            pkg_path = builder.write(out)
            links = converter.get_links()
            joints = converter.get_joints()
            plugins = converter.get_gazebo_plugins()
            return (
                f"ROS2 包已生成：{pkg_path}\n"
                f"  Links: {len(links)}\n"
                f"  Joints: {len(joints)}\n"
                f"  Gazebo plugins: {len(plugins)}\n"
                f"  结构: urdf/ meshes/ launch/ config/"
            )
        except Exception as e:
            return f"ROS2 包生成失败：{e}"


# ============================================================================
# Assembly lookup helper
# ============================================================================


def _find_assembly(name: str) -> Assembly | None:
    """Try to find an assembly by name from the standard library."""
    from ..knowledge.mechanics import ROBOTIC_ARM_ASSEMBLY, OPEN_MANIPULATOR_X_ASSEMBLY
    candidates = [ROBOTIC_ARM_ASSEMBLY, OPEN_MANIPULATOR_X_ASSEMBLY]
    for a in candidates:
        if a.name == name or _sanitize_name(a.name) == _sanitize_name(name):
            return a
    return None


# ============================================================================
# Registration
# ============================================================================


def register_urdf_tools(registry: Any) -> None:
    """Register URDF export tools."""
    registry.register(URDFExportTool())
