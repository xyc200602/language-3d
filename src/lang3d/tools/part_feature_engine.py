"""Part feature engine — infers engineering features from part metadata.

Pure-function module: inputs a ``Part`` (name / category / dimensions),
outputs a list of FreeCAD operation dicts.  No FreeCAD import, no I/O.

Key design: ``generate_ops`` tracks the **current body name** (``body``)
through every boolean / shell / pocket operation that creates a new document
object.  Each helper receives the current body name and returns the new one.
Old intermediate objects are deleted so the FreeCAD document stays clean.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..knowledge.fastener_catalog import get_clearance_hole
from ..knowledge.mechanics import Part

logger = logging.getLogger(__name__)

# M3 clearance hole radius (ISO 273 normal fit) — used in raw_script templates.
_M3_CLEARANCE_R = get_clearance_hole("M3") / 2  # 1.7mm = Ø3.4mm

if TYPE_CHECKING:
    from ..knowledge.mechanics import Joint


# ============================================================================
# Data model
# ============================================================================


@dataclass
class FeatureConfig:
    """Engineering features inferred for a single part."""

    mounting_holes: list[dict] = field(default_factory=list)
    bore: dict | None = None
    bearing_seats: list[dict] = field(default_factory=list)
    shell: dict | None = None
    fillets: list[dict] = field(default_factory=list)
    chamfers: list[dict] = field(default_factory=list)
    cable_channels: list[dict] = field(default_factory=list)


# ============================================================================
# Feature inference — name-based dispatch
# ============================================================================

# Bearing specs keyed by a prefix that matches the joint name.
# Each entry: (bore_diameter, shoulder_diameter, shell_thickness)
_JOINT_BEARING_MAP: dict[str, tuple[float, float, float]] = {
    "base": (22, 30, 3.0),
    "shoulder": (15, 22, 2.5),
    "elbow": (10, 16, 2.0),
    "wrist": (8, 12, 2.0),
}


def _classify(name: str, part: Part | None = None) -> str:
    """Return a broad part family from the part name and/or category.

    Uses part.category first (when available) for robust classification,
    then falls back to name-based patterns for LLM-generated assemblies.
    """
    # Category-first classification (more robust than name matching)
    if part is not None:
        cat = (getattr(part, "category", "") or "").lower()
        d = getattr(part, "dimensions", {}) or {}
        if cat == "actuator" and "length" in d and "width" in d:
            # Box-type actuator (NEMA-style) → motor mounting holes
            return "motor"
        elif cat == "sensor":
            return "sensor_mount"
        elif cat in ("controller", "electronics"):
            return "pcb"
        elif cat == "bearing":
            return "bearing_seat"

    n = name.lower()

    # --- Exact / prefix matches (original) ---
    if n in ("base_plate", "top_plate"):
        return "plate"
    if n.startswith("standoff_"):
        return "standoff"
    if n.startswith("wheel_"):
        return "wheel"
    if n.startswith("motor_"):
        return "motor"
    if "finger" in n and ("gripper" in n or "effector" in n):
        return "gripper_finger"
    if "_gripper" in n or "gripper_base" in n:
        return "gripper"
    if n.startswith("battery_box"):
        return "battery_box"
    if n.startswith("encoder_"):
        return "encoder"
    if n == "motor_driver_board":
        return "pcb"
    if n == "ipc_body":
        return "pcb"
    if n in ("ipc_bracket", "camera_bracket"):
        return "bracket"
    if n == "ipc_fan":
        return "fan"
    if n == "sensor_tower_post":
        return "sensor_tower_post"
    if n in ("imu_mount", "lidar_mount"):
        return "sensor_mount"

    # --- Arm joints: arm_[lr]_joint OR specific joint patterns ---
    if re.match(r"arm_[lr]_(base|shoulder|elbow|wrist)", n):
        return "arm_joint"
    # Specific joint name patterns (shoulder_joint, elbow_joint, *_rot_joint, etc.)
    # but NOT generic words like "joint_housing" which should use default geometry
    if re.search(r"_(joint|rot_joint|revolute_joint)$", n):
        return "arm_joint"
    if n.endswith("_joint") and not n.startswith("arm_link"):
        return "arm_joint"

    # --- Arm links: arm_[lr]_(upper_link|forearm) OR generic link patterns ---
    if re.match(r"arm_[lr]_(upper_link|forearm)", n):
        return "arm_link"
    # Generic link name patterns (arm_link_*, *_link_*, link_*)
    if re.search(r"(arm_link|_link$|link_)", n):
        return "arm_link"

    # --- End effector / gripper ---
    if "end_effector" in n or "effector" in n:
        return "gripper"

    # --- Broad keyword fallbacks ---
    if "plate" in n:
        return "plate"
    if "bracket" in n:
        return "bracket"
    if "post" in n and ("sensor" in n or "tower" in n):
        return "sensor_tower_post"
    if "mount" in n and ("sensor" in n or "imu" in n or "lidar" in n or "camera" in n):
        return "sensor_mount"

    return "unknown"


def infer_features(part: Part) -> FeatureConfig:
    """Analyse *part* and return the engineering features it should have."""

    name = part.name
    d = part.dimensions
    family = _classify(name, part)
    cfg = FeatureConfig()

    if family == "plate":
        l, w = d["length"], d["width"]
        margin = min(l, w) * 0.08
        hole_r = (get_clearance_hole("M4") if name == "base_plate"
                  else get_clearance_hole("M3")) / 2  # M4 vs M3 clearance
        cfg.mounting_holes = [
            {
                "diameter_mm": hole_r * 2,
                "pattern": "grid",
                "count_x": 2,
                "count_y": 2,
                "margin": margin,
            }
        ]
        cfg.chamfers = [{"size_mm": 0.5}]

    elif family == "standoff":
        cfg.bore = {"diameter_mm": get_clearance_hole("M3"), "through": True}  # M3 clearance

    elif family == "wheel":
        cfg.bore = {
            "diameter_mm": 6.0,
            "through": True,
            "keyway": True,
            "keyway_width": 2.0,
        }

    elif family == "motor":
        l, w, h = d["length"], d["width"], d["height"]
        # 4-corner M3 mounting holes
        margin_motor = min(l, w) * 0.1
        cfg.mounting_holes = [
            {
                "diameter_mm": get_clearance_hole("M3"),
                "pattern": "grid",
                "count_x": 2,
                "count_y": 2,
                "margin": margin_motor,
            }
        ]
        # Front shaft bore
        cfg.bore = {"diameter_mm": 3.0, "through": True}
        cfg.chamfers = [{"size_mm": 0.5}]

    elif family == "arm_joint":
        m = re.match(r"arm_[lr]_(base|shoulder|elbow|wrist)", name.lower())
        joint_type = m.group(1) if m else "wrist"
        spec = _JOINT_BEARING_MAP[joint_type]
        bore_d, shoulder_d, shell_t = spec
        od = d.get("outer_diameter", 40)
        h = d["height"]

        cfg.bore = {"diameter_mm": bore_d, "through": True}
        cfg.bearing_seats = [
            {
                "bore_diameter": bore_d,
                "shoulder_diameter": shoulder_d,
                "depth": h * 0.15,
            }
        ]
        cfg.shell = {"thickness_mm": shell_t, "faces_to_remove": []}
        cfg.fillets = [{"radius_mm": 2.0}]

        # Flange mounting holes — 4× polar pattern, M3 for small, M4 for base
        hole_d = get_clearance_hole("M4") if joint_type == "base" else get_clearance_hole("M3")
        cfg.mounting_holes = [
            {
                "diameter_mm": hole_d,
                "pattern": "polar",
                "count": 4,
                "pitch_radius": od / 2 * 0.75,
            }
        ]

    elif family == "arm_link":
        l, w, h = d["length"], d["width"], d["height"]
        # Two end M4 holes
        cfg.mounting_holes = [
            {
                "diameter_mm": get_clearance_hole("M4"),
                "pattern": "two_ends",
            }
        ]
        # Cable channel
        cfg.cable_channels = [
            {
                "width": 8,
                "height": 5,
                "start_offset": l * 0.15,
                "end_offset": l * 0.85,
            }
        ]
        cfg.fillets = [{"radius_mm": 2.0}]

    elif family == "gripper":
        cfg.mounting_holes = [
            {
                "diameter_mm": get_clearance_hole("M3"),
                "pattern": "bottom_center",
            }
        ]
        # U-slot
        cfg.cable_channels = [
            {
                "width": 20,
                "height": 15,
                "start_offset": 0,
                "end_offset": d["length"],
            }
        ]
        cfg.chamfers = [{"size_mm": 0.5}]

    elif family == "battery_box":
        l, w, h = d["length"], d["width"], d["height"]
        margin_bat = min(l, w) * 0.08
        cfg.mounting_holes = [
            {
                "diameter_mm": get_clearance_hole("M4"),
                "pattern": "grid",
                "count_x": 2,
                "count_y": 2,
                "margin": margin_bat,
            }
        ]
        cfg.chamfers = []  # no edge finishing on complex hollow geometry

    elif family == "encoder":
        cfg.bore = {"diameter_mm": 6.0, "through": True}

    elif family == "pcb":
        l, w = d["length"], d["width"]
        h = d.get("height", d.get("thickness", 5))
        margin_pcb = min(l, w) * 0.08
        cfg.mounting_holes = [
            {
                "diameter_mm": get_clearance_hole("M3"),
                "pattern": "grid",
                "count_x": 2,
                "count_y": 2,
                "margin": margin_pcb,
            }
        ]
        cfg.chamfers = [{"size_mm": 0.5}]
        # Add ventilation slots for enclosure-type PCBs (ipc_body, cpu_box)
        if "body" in name.lower() or "box" in name.lower() or "cpu" in name.lower():
            cfg.cable_channels = [
                {
                    "width": min(l * 0.3, 40),
                    "height": h * 0.4,
                    "start_offset": l * 0.1,
                    "end_offset": l * 0.3,
                },
                {
                    "width": min(l * 0.3, 40),
                    "height": h * 0.4,
                    "start_offset": l * 0.7,
                    "end_offset": l * 0.9,
                },
            ]
            cfg.fillets = [{"radius_mm": 1.5}]

    elif family == "bracket":
        l, w, h = d["length"], d["width"], d["height"]
        margin_bkt = min(l, w) * 0.1
        cfg.mounting_holes = [
            {
                "diameter_mm": get_clearance_hole("M3"),
                "pattern": "two_faces",
                "margin": margin_bkt,
            }
        ]
        cfg.fillets = [{"radius_mm": 3.0}]

    elif family == "fan":
        od = d["diameter"]
        cfg.bore = {"diameter_mm": 20.0, "through": True}
        cfg.mounting_holes = [
            {
                "diameter_mm": get_clearance_hole("M3"),
                "pattern": "polar",
                "count": 4,
                "pitch_radius": od / 2 * 0.8,
            }
        ]

    elif family == "sensor_tower_post":
        cfg.bore = {"diameter_mm": 5.0, "through": True}
        # Add mounting holes at base — use polar for cylindrical, grid for box
        if "length" in d and "width" in d:
            # Box-shaped post
            cfg.mounting_holes = [
                {
                    "diameter_mm": get_clearance_hole("M3"),
                    "pattern": "grid",
                    "count_x": 2,
                    "count_y": 2,
                    "margin": min(d["length"], d["width"]) * 0.15,
                }
            ]
            cfg.cable_channels = [
                {
                    "width": 6,
                    "height": 4,
                    "start_offset": d["length"] * 0.1,
                    "end_offset": d["length"] * 0.9,
                }
            ]
        elif "diameter" in d or "outer_diameter" in d:
            # Cylindrical post — polar mounting holes
            od = d.get("outer_diameter", d.get("diameter", 15))
            cfg.mounting_holes = [
                {
                    "diameter_mm": get_clearance_hole("M3"),
                    "pattern": "polar",
                    "count": 4,
                    "pitch_radius": od / 2 * 0.7,
                }
            ]
        cfg.chamfers = [{"size_mm": 0.5}]

    elif family == "sensor_mount":
        # imu_mount is box-shaped, lidar_mount is cylindrical
        if "length" in d:
            l, w = d["length"], d["width"]
            margin_sm = min(l, w) * 0.15
            cfg.mounting_holes = [
                {
                    "diameter_mm": get_clearance_hole("M2.5"),
                    "pattern": "grid",
                    "count_x": 2,
                    "count_y": 2,
                    "margin": margin_sm,
                }
            ]
        elif "diameter" in d:
            od = d["diameter"]
            cfg.bore = {"diameter_mm": 6.0, "through": True}
            cfg.mounting_holes = [
                {
                    "diameter_mm": get_clearance_hole("M2.5"),
                    "pattern": "polar",
                    "count": 4,
                    "pitch_radius": od / 2 * 0.7,
                }
            ]

    # "unknown" → infer from category + shape
    if family == "unknown":
        # Category-based fallback: give structural parts at least fillets
        cat = getattr(part, "category", "")
        if cat == "actuator":
            # Actuator-shaped box → treat like motor
            if "length" in d and "width" in d:
                margin_unk = min(d["length"], d["width"]) * 0.1
                cfg.mounting_holes = [{
                    "diameter_mm": get_clearance_hole("M3"),
                    "pattern": "grid",
                    "count_x": 2,
                    "count_y": 2,
                    "margin": margin_unk,
                }]
                cfg.bore = {"diameter_mm": 3.0, "through": True}
            cfg.chamfers = [{"size_mm": 0.5}]
        elif cat in ("structural", "mechanical"):
            # Structural link → fillets and mounting holes
            l = d.get("length", 0)
            w = d.get("width", 0)
            h = d.get("height", d.get("thickness", 0))
            if l > 0 and w > 0:
                cfg.mounting_holes = [{
                    "diameter_mm": get_clearance_hole("M4"),
                    "pattern": "two_ends",
                }]
                cfg.fillets = [{"radius_mm": min(l, w, h) * 0.1 if h > 0 else 2.0}]
        elif cat == "sensor":
            cfg.mounting_holes = [{
                "diameter_mm": get_clearance_hole("M3"),
                "pattern": "grid",
                "count_x": 2,
                "count_y": 2,
                "margin": min(d.get("length", 20), d.get("width", 20)) * 0.1,
            }]

    return cfg


# ============================================================================
# Operation generation
# ============================================================================

# Valid op types that _build_script() supports.
_VALID_OPS = {
    "new_doc", "make_box", "make_cylinder", "make_sphere", "make_cone",
    "boolean", "cylinder_with_hole", "plate_with_holes",
    "move", "rotate",
    "fillet", "chamfer",
    "sweep", "loft",
    "polar_pattern", "linear_pattern", "mirror",
    "shell", "draft",
    "create_sketch", "extrude_sketch", "revolve_sketch", "pocket",
    "save", "export_stl", "export_step",
    "status", "object_info", "volume_check", "compute_mass",
    "delete_object", "raw_script",
}


def _has_cylindrical_dims(d: dict) -> bool:
    """True when the part is fundamentally cylindrical.

    Standoffs have both ``length`` and ``diameter`` but are cylinders,
    so we treat any part with ``diameter`` (or ``outer_diameter``) and
    without ``width`` as cylindrical.
    """
    if "outer_diameter" in d:
        return True
    if "diameter" in d and "width" not in d:
        return True
    return False


def _cyl_radius(d: dict) -> float:
    if "outer_diameter" in d:
        return d["outer_diameter"] / 2
    return d["diameter"] / 2


class _StepNamer:
    """Generate unique body / tool names for sequential boolean ops."""

    def __init__(self, base: str) -> None:
        self._base = base
        self._counter = 0

    def body(self) -> str:
        self._counter += 1
        return f"{self._base}_body{self._counter}"

    def tool(self, tag: str) -> str:
        self._counter += 1
        return f"{self._base}_{tag}{self._counter}"


_CREATE_OP_TYPES = frozenset(
    {"make_box", "make_cylinder", "cylinder_with_hole", "polar_pattern", "linear_pattern"}
)


def _suffix_feature_names(ops: list[dict], joint_idx: int) -> None:
    """Append ``_j{joint_idx}`` to all feature object names *in-place*.

    Ensures feature names are globally unique across multiple joints that
    involve the same part (e.g. a bracket bolted to both a base and a link).
    """
    suffix = f"_j{joint_idx}"
    rename: dict[str, str] = {}
    for op in ops:
        if op.get("type") in _CREATE_OP_TYPES and op.get("name"):
            old = op["name"]
            new = old + suffix
            op["name"] = new
            rename[old] = new
    for op in ops:
        if op.get("type") == "move" and op.get("object") in rename:
            op["object"] = rename[op["object"]]


def _generate_standard_body(
    part: Part,
    d: dict,
    name: str,
    family: str,
    config: FeatureConfig,
    sn: _StepNamer,
    ops: list[dict],
) -> str:
    """Run Steps 1-9 for a standard (non-raw_script) part.

    Appends base-primitive, shell, holes, bore, keyway, cable-channel,
    fillet, and chamfer operations to *ops* and returns the final body
    name.  Extracted from ``generate_ops`` so that raw_script families
    (arm_link / arm_joint / gripper / gripper_finger) can skip these
    steps while still receiving Step 10 (connection_features) and
    Step 11 (export_stl) downstream.
    """
    body = name

    # ------------------------------------------------------------------
    # 1. Base primitive (smart selection)
    # ------------------------------------------------------------------
    use_plate_with_holes = (
        family in ("plate", "pcb", "sensor_mount")
        and "length" in d
        and config.mounting_holes
        and config.mounting_holes[0].get("pattern") == "grid"
    )

    use_cyl_with_hole = (
        _has_cylindrical_dims(d)
        and config.bore is not None
    )

    if use_plate_with_holes:
        hole = config.mounting_holes[0]
        ops.append({
            "type": "plate_with_holes",
            "length": d["length"],
            "width": d["width"],
            "thickness": d["height"],
            "hole_radius": hole["diameter_mm"] / 2,
            "hole_count_x": hole.get("count_x", 2),
            "hole_count_y": hole.get("count_y", 2),
            "margin": hole.get("margin", min(d["length"], d["width"]) * 0.1),
            "name": body,
        })
    elif use_cyl_with_hole:
        ops.append({
            "type": "cylinder_with_hole",
            "outer_radius": _cyl_radius(d),
            "inner_radius": config.bore["diameter_mm"] / 2,
            "height": d["height"],
            # Wheels (revolute axis="y") must have their cylinder re-pointed
            # along Y so the tyre + bore align with the axle — see the
            # orient_axis handling in freecad.py.  Without this the wheel
            # STL is a flat disc (磨盘) lying in the XY plane, not a wheel.
            # Convention matches vtk_renderer._add_dimension_approximation
            # (name.startswith("wheel_") → orient "y").
            "orient_axis": "x" if name.lower().startswith("wheel_") else "z",
            "name": body,
        })
    elif _has_cylindrical_dims(d):
        ops.append({
            "type": "make_cylinder",
            "radius": _cyl_radius(d),
            "height": d.get("height", d.get("length", 10)),
            "orient_axis": "x" if name.lower().startswith("wheel_") else "z",
            "name": body,
        })
    elif "length" in d and "width" in d:
        ops.append({
            "type": "make_box",
            "length": d["length"],
            "width": d["width"],
            "height": d.get("height", d.get("thickness", 5)),
            "name": body,
        })
    else:
        ops.append({
            "type": "make_cylinder",
            "radius": 10,
            "height": d.get("height", 10),
            "name": body,
        })

    # ------------------------------------------------------------------
    # 2. Shell (hollow) — do BEFORE holes when base is simple box
    #    (shell on simple box is reliable; shell on pierced plate fails)
    # ------------------------------------------------------------------
    if config.shell:
        shell_name = sn.body()
        ops.append({
            "type": "shell",
            "object": body,
            "thickness": config.shell["thickness_mm"],
            "faces_to_remove": config.shell.get("faces_to_remove", []),
            "result_name": shell_name,
        })
        ops.append({"type": "delete_object", "object": body})
        body = shell_name

    # ------------------------------------------------------------------
    # 2b. Battery box hollow (inner box boolean cut — more reliable than shell)
    # ------------------------------------------------------------------
    if family == "battery_box" and "length" in d:
        t = 2.0  # wall thickness mm
        il = d["length"] - 2 * t
        iw = d["width"] - 2 * t
        ih = d["height"] - t  # open top
        inner_name = sn.tool("inner")
        ops.append({
            "type": "make_box",
            "length": il,
            "width": iw,
            "height": ih,
            "name": inner_name,
        })
        ops.append({
            "type": "move",
            "object": inner_name,
            "dx": t,
            "dy": t,
            "dz": 0,  # bottom aligned, open at top
        })
        body = _bool_cut(ops, body, inner_name, sn)

    # ------------------------------------------------------------------
    # 3. Extra mounting holes (boolean cut or polar_pattern)
    # ------------------------------------------------------------------
    for hole in config.mounting_holes:
        pattern = hole.get("pattern", "")
        # Skip if already handled by plate_with_holes
        if pattern == "grid" and use_plate_with_holes:
            continue

        if pattern == "polar":
            body = _add_polar_mounting_holes(ops, body, name, hole, sn)
        elif pattern == "grid" and not use_plate_with_holes:
            body = _add_grid_boolean_cuts(ops, body, name, d, hole, sn)
        elif pattern in ("two_ends", "bottom_center", "two_faces"):
            body = _add_simple_holes(ops, body, name, d, hole, sn)

    # ------------------------------------------------------------------
    # 4. Motor shaft bore (box-based motors)
    # ------------------------------------------------------------------
    if family == "motor" and config.bore and not use_cyl_with_hole:
        body = _add_motor_shaft_bore(ops, body, name, d, config.bore, sn)

    # ------------------------------------------------------------------
    # 5. Bearing seats (stepped cylinder boolean cut)
    # ------------------------------------------------------------------
    for seat in config.bearing_seats:
        body = _add_bearing_seat(ops, body, name, d, seat, sn)

    # ------------------------------------------------------------------
    # 6. Keyway (sketch + extrude + boolean cut)
    # ------------------------------------------------------------------
    if config.bore and config.bore.get("keyway"):
        body = _add_keyway(ops, body, name, d, config.bore, sn)

    # ------------------------------------------------------------------
    # 7. Cable channels (sketch + pocket) — creates new object
    # ------------------------------------------------------------------
    for ch in config.cable_channels:
        body = _add_cable_channel(ops, body, name, d, ch, sn)

    # ------------------------------------------------------------------
    # 8. Fillets — modify object IN-PLACE (no name change)
    # ------------------------------------------------------------------
    for f in config.fillets:
        ops.append({
            "type": "fillet",
            "object": body,
            "radius": f["radius_mm"],
        })

    # ------------------------------------------------------------------
    # 9. Chamfers — modify object IN-PLACE (no name change)
    # ------------------------------------------------------------------
    for c in config.chamfers:
        ops.append({
            "type": "chamfer",
            "object": body,
            "size": c["size_mm"],
        })

    return body


def generate_ops(
    part: Part,
    config: FeatureConfig | None = None,
    joints: list[Joint] | None = None,
    all_parts: list[Part] | None = None,
) -> list[dict]:
    """Return a list of FreeCAD operation dicts for *part*.

    If *config* is ``None`` it is inferred via :func:`infer_features`.

    If *joints* is provided, connection features (bolt holes, bearing seats,
    etc.) are generated from ``Joint.connection`` metadata and merged into
    the ops list before the final export.

    If *all_parts* is provided, bolted-connection holes are derived from a
    SHARED pattern computed once per joint (from BOTH mating parts' faces)
    so the holes ALIGN across the interface — the fix for "连不上".
    Without it each part lays out holes from its own dimensions.

    The function tracks the **current body name** through every operation
    that creates a new document object (``boolean``, ``shell``, ``pocket``).
    Old bodies are deleted so FreeCAD's document stays consistent and no
    name clashes occur.
    """
    if config is None:
        config = infer_features(part)

    d = part.dimensions
    name = part.name
    ops: list[dict] = [{"type": "new_doc", "name": name}]
    family = _classify(name, part)
    sn = _StepNamer(name)

    # --- Specialized geometry for arm parts (C-channel links,
    #     servo-mount joints, parallel-jaw gripper) ---
    # raw_script families produce the full body via inline FreeCAD Python
    # and expose it as ``{name}_final``.  They then FALL THROUGH to
    # Step 10 (connection_features) and Step 11 (export_stl) so that
    # bolt holes, bearing seats, and other connection geometry are
    # generated on top of the raw_script body — closing the integration
    # gap where arm parts previously had no connection features at all.
    _RAW_SCRIPT_FAMILIES = ("arm_link", "arm_joint", "gripper_finger", "gripper")
    is_raw_script = family in _RAW_SCRIPT_FAMILIES
    if is_raw_script:
        if family == "arm_link":
            ops.extend(_arm_link_ops(name, d))
        elif family == "arm_joint":
            ops.extend(_arm_joint_ops(name, d))
        elif family == "gripper_finger":
            ops.extend(_gripper_finger_ops(name, d))
        else:  # gripper
            ops.extend(_gripper_ops(name, d))
        body = f"{name}_final"
    else:
        # Standard path: Steps 1-9 (base primitive → fillets/chamfers).
        body = _generate_standard_body(part, d, name, family, config, sn, ops)

    # ------------------------------------------------------------------
    # 10. Connection features — from Joint.connection metadata
    # ------------------------------------------------------------------
    if joints:
        from .connection_features import (
            ConnectionFeatureEngine,
            merge_connection_ops,
        )
        from ..knowledge.mechanics import Joint as JointType

        engine = ConnectionFeatureEngine()
        # Index the sibling parts (if provided) by name so we can resolve
        # the MATE of each joint's current part and compute ONE shared
        # bolt-hole pattern for the interface.  Without this, each part
        # ran _auto_layout_bolts from its own face → mismatched holes
        # ("连不上").  See ConnectionFeatureEngine.compute_shared_bolt_pattern.
        _by_name = {p.name: p for p in (all_parts or [])}
        for ji, joint in enumerate(joints):
            if joint.connection is None:
                continue
            # Check if this joint involves the current part
            if joint.parent != name and joint.child != name:
                continue
            anchor = joint.parent_anchor if joint.parent == name else joint.child_anchor

            # Compute a shared bolt pattern for this joint ONCE, so BOTH
            # mating parts drill holes at matching (u,v) fractions.  We
            # recompute per part (cheap) but deterministically — the result
            # depends only on (bolt_count, bolt_size, the two face sizes),
            # so both parts get the identical list.
            shared_pattern = None
            if all_parts is not None and joint.connection.type == "bolted":
                mate_name = joint.child if joint.parent == name else joint.parent
                mate = _by_name.get(mate_name)
                if mate is not None:
                    mate_anchor = (
                        joint.child_anchor if joint.parent == name
                        else joint.parent_anchor
                    )
                    shared_pattern = ConnectionFeatureEngine.compute_shared_bolt_pattern(
                        part_a=part, part_b=mate,
                        anchor_a=anchor, anchor_b=mate_anchor,
                        conn=joint.connection,
                    )

            result = engine.generate_features(
                structural_part=part,
                connection=joint.connection,
                anchor=anchor,
                bolt_pattern=shared_pattern,
            )

            # For revolute joints with bolted connections, add a central
            # bearing bore so the shaft can pass through.  Bolts hold the
            # housing; the bearing bore allows rotation.
            if joint.type == "revolute" and joint.connection.type == "bolted":
                d = part.dimensions
                thickness = ConnectionFeatureEngine._infer_thickness(d, anchor)
                face_l = ConnectionFeatureEngine._face_length(anchor, d)
                face_w = ConnectionFeatureEngine._face_width(anchor, d)
                min_face = min(face_l, face_w)

                # Skip bearing bore if face is too small — the 10mm bore
                # would overlap with corner bolt holes on faces < 30mm.
                if min_face >= 30.0:
                    bore_d = 10.0  # MR105ZZ bearing OD = 10mm
                    bore_name = f"{name}_bearing_bore_{ji}"
                    bx, by, bz = ConnectionFeatureEngine._anchor_center(anchor, d, thickness)
                    result.ops.append({
                        "type": "make_cylinder",
                        "radius": bore_d / 2,
                        "height": thickness + 4,
                        "name": bore_name,
                    })
                    result.ops.append({
                        "type": "move",
                        "object": bore_name,
                        "dx": bx, "dy": by, "dz": bz - 2,
                    })
                    result.features_generated.append(
                        f"Bearing bore O{bore_d:.1f}mm (shaft passage for revolute)"
                    )
                else:
                    logger.info(
                        "Skipping bearing bore on %s: face %.0fx%.0fmm "
                        "too small (minimum 30mm for bearing housing)",
                        name, face_l, face_w,
                    )

            if result.ops:
                # Add joint-index suffix for cross-joint name uniqueness
                if ji > 0:
                    _suffix_feature_names(result.ops, ji)
                ops = merge_connection_ops(ops, result.ops, body)
                # Track body name changes from merge
                for op in reversed(ops):
                    if op.get("type") == "boolean" and op.get("operation") == "cut":
                        body = op.get("result_name", body)
                        break
            # NOTE: result.fastener_ops (bolt/nut/washer 3D models) are
            # intentionally NOT merged into the part STL.  Fasteners span
            # the joint interface between TWO parts, so positioning them
            # in part-local coordinates is geometrically incorrect.
            # Adding them as separate assembly-level parts is the right
            # architecture (tracked as a separate P0 task).
            joint.connection.features_generated = True

    # ------------------------------------------------------------------
    # 11. Export STL — only the final body
    # ------------------------------------------------------------------
    ops.append({
        "type": "export_stl",
        "object": body,
        "name": name,
        "path": f"{{WORKSPACE}}/{name}.stl",
    })

    return ops


# ============================================================================
# Helper generators — each returns the NEW body name
# ============================================================================


def _bool_cut(
    ops: list[dict],
    body: str,
    tool_name: str,
    sn: _StepNamer,
    *,
    extra_cleanup: list[str] | None = None,
) -> str:
    """Boolean-cut *tool_name* from *body*, delete old body + tool, return new."""
    new_body = sn.body()
    ops.append({
        "type": "boolean",
        "operation": "cut",
        "object1": body,
        "object2": tool_name,
        "result_name": new_body,
    })
    ops.append({"type": "delete_object", "object": body})
    ops.append({"type": "delete_object", "object": tool_name})
    if extra_cleanup:
        for name in extra_cleanup:
            ops.append({"type": "delete_object", "object": name})
    return new_body


def _add_polar_mounting_holes(
    ops: list[dict], body: str, name: str, hole: dict, sn: _StepNamer,
) -> str:
    """Add a polar-patterned mounting hole array + boolean cut.

    Returns the new body name.
    """
    hole_r = hole["diameter_mm"] / 2
    count = hole.get("count", 4)
    pitch_r = hole.get("pitch_radius", 10)

    # Create single hole cylinder at pitch radius
    cyl_name = sn.tool("mhole_cyl")
    ops.append({
        "type": "make_cylinder",
        "radius": hole_r,
        "height": 100,  # will be cut through
        "name": cyl_name,
    })
    # Move to pitch radius on X axis
    ops.append({
        "type": "move",
        "object": cyl_name,
        "dx": pitch_r,
        "dy": 0,
        "dz": 0,
    })
    # Polar pattern the hole
    pattern_name = sn.tool("mpat")
    ops.append({
        "type": "polar_pattern",
        "object": cyl_name,
        "count": count,
        "angle": 360.0,
        "axis": [0, 0, 1],
        "center": [0, 0, 0],
        "result_name": pattern_name,
    })
    # Boolean cut from main body, cleanup the source cylinder
    return _bool_cut(ops, body, pattern_name, sn, extra_cleanup=[cyl_name])


def _add_grid_boolean_cuts(
    ops: list[dict], body: str, name: str, d: dict, hole: dict, sn: _StepNamer,
) -> str:
    """Add grid mounting holes via individual cylinders + boolean cuts.

    To avoid N sequential boolean cuts, we fuse all hole cylinders into
    one compound first, then do a single cut.  Returns new body name.
    """
    hole_r = hole["diameter_mm"] / 2
    nx = hole.get("count_x", 2)
    ny = hole.get("count_y", 2)
    margin = hole.get("margin", 10)
    l = d.get("length") or d.get("diameter", 20)
    w = d.get("width") or d.get("diameter", 20)
    h = d["height"]

    sx = (l - 2 * margin) / max(nx - 1, 1) if nx > 1 else 0
    sy = (w - 2 * margin) / max(ny - 1, 1) if ny > 1 else 0

    # Create all hole cylinders
    hole_names: list[str] = []
    for ix in range(nx):
        for iy in range(ny):
            hname = sn.tool("ghole")
            hole_names.append(hname)
            ops.append({
                "type": "make_cylinder",
                "radius": hole_r,
                "height": h + 2,  # slight overshoot
                "name": hname,
            })
            x = margin + ix * sx
            y = margin + iy * sy
            ops.append({
                "type": "move",
                "object": hname,
                "dx": x,
                "dy": y,
                "dz": -1,
            })

    # Fuse all holes into one compound
    if len(hole_names) == 1:
        compound = hole_names[0]
    else:
        compound = sn.tool("ghole_compound")
        # Fuse progressively: first pair, then accumulate
        prev = hole_names[0]
        for i in range(1, len(hole_names)):
            fused = sn.tool("ghole_fused")
            ops.append({
                "type": "boolean",
                "operation": "union",
                "object1": prev,
                "object2": hole_names[i],
                "result_name": fused,
            })
            # Delete intermediate fuse results (not the first pair inputs,
            # which are individual hole cylinders still needed for reference)
            if i > 1:
                ops.append({"type": "delete_object", "object": prev})
            prev = fused
        compound = prev

    # Single boolean cut — cleanup all individual hole cylinders
    return _bool_cut(ops, body, compound, sn, extra_cleanup=list(hole_names))


def _add_simple_holes(
    ops: list[dict], body: str, name: str, d: dict, hole: dict, sn: _StepNamer,
) -> str:
    """Add simple mounting holes for brackets, grippers, etc.

    Returns the new body name.
    """
    hole_r = hole["diameter_mm"] / 2
    pattern = hole.get("pattern", "")
    l = d["length"]
    w = d["width"]
    h = d["height"]

    if pattern == "two_ends":
        for x_pos in [l * 0.1, l * 0.9]:
            hname = sn.tool("endhole")
            ops.append({
                "type": "make_cylinder",
                "radius": hole_r,
                "height": h + 2,
                "name": hname,
            })
            ops.append({
                "type": "move",
                "object": hname,
                "dx": x_pos,
                "dy": w / 2,
                "dz": -1,
            })
            body = _bool_cut(ops, body, hname, sn)

    elif pattern == "bottom_center":
        hname = sn.tool("bothole")
        ops.append({
            "type": "make_cylinder",
            "radius": hole_r,
            "height": h * 0.5,
            "name": hname,
        })
        ops.append({
            "type": "move",
            "object": hname,
            "dx": l / 2,
            "dy": w / 2,
            "dz": 0,
        })
        body = _bool_cut(ops, body, hname, sn)

    elif pattern == "two_faces":
        margin = hole.get("margin", 10)
        for x_pos, y_pos, z_pos in [
            (l / 2, margin, h / 2),
            (margin, w / 2, h / 2),
        ]:
            hname = sn.tool("facehole")
            ops.append({
                "type": "make_cylinder",
                "radius": hole_r,
                "height": 15,
                "name": hname,
            })
            ops.append({
                "type": "move",
                "object": hname,
                "dx": x_pos,
                "dy": y_pos,
                "dz": z_pos,
            })
            body = _bool_cut(ops, body, hname, sn)

    return body


def _add_motor_shaft_bore(
    ops: list[dict], body: str, name: str, d: dict, bore: dict, sn: _StepNamer,
) -> str:
    """Add front shaft bore for box-based motors.  Returns new body name."""
    bore_r = bore["diameter_mm"] / 2
    l = d["length"]
    w = d["width"]
    h = d["height"]
    hname = sn.tool("shaft")
    ops.append({
        "type": "make_cylinder",
        "radius": bore_r,
        "height": h,
        "name": hname,
    })
    ops.append({
        "type": "move",
        "object": hname,
        "dx": l / 2,
        "dy": w / 2,
        "dz": 0,
    })
    return _bool_cut(ops, body, hname, sn)


def _add_bearing_seat(
    ops: list[dict], body: str, name: str, d: dict, seat: dict, sn: _StepNamer,
) -> str:
    """Add bearing seat via stepped cylinder boolean cut.  Returns new body name."""
    bore_d = seat["bore_diameter"]
    shoulder_d = seat["shoulder_diameter"]
    depth = seat["depth"]
    h = d["height"]

    # Deep bore (through)
    deep_name = sn.tool("bdeep")
    ops.append({
        "type": "make_cylinder",
        "radius": bore_d / 2,
        "height": h,
        "name": deep_name,
    })

    # Shallow shoulder
    shoulder_name = sn.tool("bshld")
    ops.append({
        "type": "make_cylinder",
        "radius": shoulder_d / 2,
        "height": depth,
        "name": shoulder_name,
    })

    # Fuse into compound tool
    compound_name = sn.tool("bcomp")
    ops.append({
        "type": "boolean",
        "operation": "union",
        "object1": deep_name,
        "object2": shoulder_name,
        "result_name": compound_name,
    })

    # Cut compound from main body, cleanup intermediate tools
    return _bool_cut(ops, body, compound_name, sn,
                     extra_cleanup=[deep_name, shoulder_name])


def _add_keyway(
    ops: list[dict], body: str, name: str, d: dict, bore: dict, sn: _StepNamer,
) -> str:
    """Add a keyway slot (make_box + move + boolean cut).  Returns new body name."""
    bore_r = bore["diameter_mm"] / 2
    kw = bore.get("keyway_width", 2.0)
    h = d["height"]

    # Use make_box for the keyway tool — simpler and more reliable than sketch
    tool_name = sn.tool("kwbox")
    ops.append({
        "type": "make_box",
        "length": kw,
        "width": kw * 2,
        "height": h,
        "name": tool_name,
    })
    # Position: box sits at x=bore_r (edge of bore), centered on Y axis
    ops.append({
        "type": "move",
        "object": tool_name,
        "dx": bore_r,
        "dy": -kw,
        "dz": 0,
    })
    return _bool_cut(ops, body, tool_name, sn)


def _add_cable_channel(
    ops: list[dict], body: str, name: str, d: dict, ch: dict, sn: _StepNamer,
) -> str:
    """Add a cable routing channel (make_box + boolean cut).

    Uses make_box positioned at the channel location + boolean cut instead of
    sketch + pocket, because FreeCAD's pocket operation fails on complex
    boolean-cut geometry.
    """
    cw = ch["width"]
    ch_h = ch["height"]
    l = d["length"]
    w = d["width"]
    h = d["height"]
    start = ch["start_offset"]
    end = ch["end_offset"]

    # Create a box for the channel volume
    tool_name = sn.tool("chbox")
    ops.append({
        "type": "make_box",
        "length": end - start,
        "width": cw,
        "height": ch_h,
        "name": tool_name,
    })
    # Position: centered on Y, cut into top surface
    ops.append({
        "type": "move",
        "object": tool_name,
        "dx": start,
        "dy": (w - cw) / 2,
        "dz": h - ch_h,  # top-aligned
    })
    return _bool_cut(ops, body, tool_name, sn)


# ============================================================================
# Specialized arm part generators — complex geometry via raw_script
# ============================================================================


def _arm_link_ops(name: str, d: dict) -> list[dict]:
    """C-channel link with U-profile, servo-mount flanges at both ends.

    Generates a realistic arm link that looks like a C-beam:
    - Main body: C-channel cross-section extruded along length
    - Both ends: servo mounting ears with M3 holes
    - Lightening pocket in the center web
    """
    length = d.get("length", 100)   # solver: Y extent (front/back)
    width = d.get("width", 25)      # solver: X extent (left/right)
    height = d.get("height", 15)    # solver: Z extent

    # IMPORTANT: FreeCAD makeBox(X, Y, Z).  Solver convention: X=width,
    # Y=length, Z=height (same as make_box/plate_with_holes ops — see
    # freecad.py).  All makeBox calls here use (W, L, H) so the STL bbox
    # matches what the solver expects.  Without this the STL has length
    # on X (FreeCAD default) and the mesh appears rotated 90° vs the
    # solved position → interpenetration with adjacent parts.
    # Mounting tabs are removed (they exceeded the height budget).

    script = f'''
import FreeCAD, Part

L, W, H = {length}, {width}, {height}
web_t = max(3.0, W * 0.15)

# --- Build a structural C-channel spanning the FULL length L ---
# A real C-channel (like aluminium extrusion / bent sheet) is one continuous
# profile extruded along its length. Cross-section in X-Z is a "[":
#   - top flange + bottom flange (horizontal, full width W)
#   - web (vertical, full height H) on one side connecting them.
# All three span the FULL length L (the extrusion direction, solver Y).
#
# BUG HISTORY: the previous version built flanges of length (L-web_t)/2 and a
# short central web, producing a part whose total Y-extent was only ~L/2.
# The solver positioned joints assuming length=L, so adjacent joints sat
# ~L/2 apart in empty space — every link looked disconnected from its
# neighbour with a 20-50mm gap (data/runs/4dof_arm STL bbox analysis showed
# shoulder_link_1 solver L=120 vs actual STL L=62). Fix: flanges + web all
# span the full length L so the STL bbox matches the solver's assumption.
# makeBox(X, Y, Z) = (width, length, height) per solver convention.

# Top flange: full width + length, thin, at the top.
top = Part.makeBox(W, L, web_t)
top.translate(FreeCAD.Vector(0, 0, H - web_t))
# Bottom flange: full width + length, thin, at the bottom.
bot = Part.makeBox(W, L, web_t)
# Web: vertical plate connecting top+bottom along the full length, on the
# -X side (forms the "[" opening toward +X so the joint housings on +X
# have clearance). Full height H, full length L.
web = Part.makeBox(web_t, L, H)
body = top.fuse(bot).fuse(web)

# --- Lightening pocket (cut a window in the +X-facing web gap to reduce
# weight while keeping the structural flanges + web intact) ---
pocket_l = L * 0.55
pocket_h = max(H - 2 * web_t - 4, 2)
pocket = Part.makeBox(W, pocket_l, pocket_h)
pocket.translate(FreeCAD.Vector(0, (L - pocket_l) / 2, web_t + 2))
body = body.cut(pocket)

# --- M3 mounting holes at both ends ---
for y_pos in [L * 0.1, L * 0.9]:
    for x_off in [W * 0.5]:
        h_cyl = Part.makeCylinder({_M3_CLEARANCE_R}, H + 2)
        h_cyl.translate(FreeCAD.Vector(x_off, y_pos, -1))
        body = body.cut(h_cyl)

# --- Fillet small edges (cosmetic; guarded so a non-manifold edge never
# aborts the whole STL export — see _gripper_finger_ops for the same pattern) ---
try:
    body = body.makeFillet(0.8, [_e for _e in body.Edges if _e.Length < W * 0.6][:40])
except Exception:
    pass

obj = doc.addObject("Part::Feature", "{name}_final")
obj.Shape = body
doc.recompute()
'''
    return [{"type": "raw_script", "script": script}]


def _arm_joint_ops(name: str, d: dict) -> list[dict]:
    """Servo motor joint with visible output shaft, bearing collar, and flange.

    Generates a realistic servo that LOOKS like it can rotate:
    - Main body: cylinder housing (fixed part, mounts to parent)
    - Bottom mounting flange: wider disc with 4x M3 holes
    - Output shaft: protruding cylinder from TOP (rotating part, mounts to child)
    - Bearing collar: ring at shaft base creates visible rotation interface
    - D-cut flat on shaft for securing link/horn
    - Wire exit slot on side
    - Groove between flange and body for visual separation
    """
    od = d.get("diameter", d.get("outer_diameter", 40))
    h = d.get("height", 30)
    shaft_r = max(3.0, od * 0.08)

    script = f'''
import FreeCAD, Part, math

od, h = {od}, {h}
shaft_r = {shaft_r}
shaft_h = max(8.0, h * 0.25)  # output shaft protrusion height
flange_r = od / 2 + od * 0.25
flange_h = max(4.0, h * 0.12)
collar_r = shaft_r + 3.0  # bearing collar radius

# --- Main cylinder body (servo housing, fixed part) ---
body = Part.makeCylinder(od/2, h)

# --- Bottom mounting flange (attaches to parent link) ---
flange = Part.makeCylinder(flange_r, flange_h)
body = body.fuse(flange)

# --- Output shaft (protrudes from TOP — this is the ROTATING part) ---
shaft = Part.makeCylinder(shaft_r, shaft_h)
shaft.translate(FreeCAD.Vector(0, 0, h))
body = body.fuse(shaft)

# --- Bearing collar (ring at shaft base, visible rotation interface) ---
collar_outer = Part.makeCylinder(collar_r, 4.0)
collar_outer.translate(FreeCAD.Vector(0, 0, h - 2))
collar_bore = Part.makeCylinder(shaft_r + 0.5, 6.0)
collar_bore.translate(FreeCAD.Vector(0, 0, h - 3))
collar = collar_outer.cut(collar_bore)
body = body.fuse(collar)

# --- D-cut flat on output shaft (for securing horn/link) ---
d_flat = Part.makeBox(shaft_r * 1.5, shaft_r * 3, shaft_h + 4)
d_flat.translate(FreeCAD.Vector(-shaft_r * 0.3, -shaft_r * 1.5, h - 2))
body = body.cut(d_flat)

# --- 4x M3 mounting holes on flange ---
for i in range(4):
    angle = math.radians(45 + i * 90)
    x = flange_r * 0.75 * math.cos(angle)
    y = flange_r * 0.75 * math.sin(angle)
    h_cyl = Part.makeCylinder({_M3_CLEARANCE_R}, flange_h + 2)
    h_cyl.translate(FreeCAD.Vector(x, y, -1))
    body = body.cut(h_cyl)

# --- Wire exit slot (side notch) ---
slot = Part.makeBox(6, od/2 + 2, 4)
slot.translate(FreeCAD.Vector(-3, -1, h * 0.3))
body = body.cut(slot)

# --- Decorative groove between flange and body ---
groove_outer = Part.makeCylinder(od/2 + 0.5, 1.5)
groove_outer.translate(FreeCAD.Vector(0, 0, flange_h - 0.5))
groove_inner = Part.makeCylinder(od/2 - 1.5, 3.0)
groove_inner.translate(FreeCAD.Vector(0, 0, flange_h - 1))
groove = groove_outer.cut(groove_inner)
body = body.cut(groove)

# --- Fillet ---
try:
    body = body.makeFillet(0.8, [_e for _e in body.Edges if _e.Length < od * 0.3][:30])
except Exception:
    pass

obj = doc.addObject("Part::Feature", "{name}_final")
obj.Shape = body
doc.recompute()
'''
    return [{"type": "raw_script", "script": script}]


def _gripper_ops(name: str, d: dict) -> list[dict]:
    """Gripper housing — either base-only or complete gripper with fingers.

    When the part name contains "base", generates ONLY the base block
    (fingers are separate parts in the assembly).

    For all other gripper names (end_effector, gripper, arm_l_gripper, etc.),
    generates a COMPLETE gripper with base + two visible L-shaped fingers,
    since the gripper is a single part in the assembly.
    """
    length = d.get("length", 40)
    width = d.get("width", 35)
    height = d.get("height", 15)

    # If name contains "base", generate base-only (fingers are separate parts)
    is_base_only = "base" in name.lower()

    if is_base_only:
        return _gripper_base_ops(name, d)
    else:
        return _gripper_complete_ops(name, d)


def _gripper_base_ops(name: str, d: dict) -> list[dict]:
    """Gripper base block with servo cavity and linear guide rails.

    Generates ONLY the base (fingers are separate parts):
    - Main housing with internal servo cavity
    - Front face has two parallel linear rail grooves (for finger tabs)
    - Top mounting holes (to attach to wrist)
    - Side servo wire exit
    """
    length = d.get("length", 40)    # solver: Y extent
    width = d.get("width", 35)      # solver: X extent
    height = d.get("height", 15)    # solver: Z extent

    # makeBox(X=width, Y=length, Z=height) per solver convention.
    script = f'''
import FreeCAD, Part, math

L, W, H = {length}, {width}, {height}
rail_depth = max(2.0, H * 0.15)
rail_w = max(5.0, W * 0.15)

# --- Main housing block (X=width, Y=length, Z=height) ---
body = Part.makeBox(W, L, H)

# --- Servo cavity (hollowed center) ---
cavity_l = L * 0.55
cavity_w = W * 0.45
cavity = Part.makeBox(cavity_w, cavity_l, H - 4)
cavity.translate(FreeCAD.Vector((W - cavity_w) / 2, L * 0.1, 2))
body = body.cut(cavity)

# --- Two parallel linear rail grooves on FRONT face (Y = L) ---
for x_center in [W * 0.25, W * 0.75]:
    rail = Part.makeBox(rail_w, rail_depth + 1, H - 4)
    rail.translate(FreeCAD.Vector(x_center - rail_w / 2, L - rail_depth, 2))
    body = body.cut(rail)

# --- Top mounting holes (2x M3 to attach to wrist joint) ---
for x_off in [W * 0.25, W * 0.75]:
    for y_off in [L * 0.2, L * 0.8]:
        h_cyl = Part.makeCylinder({_M3_CLEARANCE_R}, H + 2)
        h_cyl.translate(FreeCAD.Vector(x_off, y_off, -1))
        body = body.cut(h_cyl)

# --- Servo wire exit slot (back face Y=0) ---
wire = Part.makeBox(8, 4, 5)
wire.translate(FreeCAD.Vector(W * 0.4, -1, H * 0.2))
body = body.cut(wire)

# --- Decorative fillet on front edges ---
try:
    body = body.makeFillet(0.8, [_e for _e in body.Edges
                                  if _e.Length < W * 0.3][:25])
except Exception:
    pass

obj = doc.addObject("Part::Feature", "{name}_final")
obj.Shape = body
doc.recompute()
'''
    return [{"type": "raw_script", "script": script}]


def _gripper_complete_ops(name: str, d: dict) -> list[dict]:
    """Complete gripper with base housing and two L-shaped fingers.

    Used when the gripper is a single part in the assembly (e.g. end_effector).
    - Base block with servo cavity and mounting holes
    - Two parallel L-shaped fingers extending forward
    - Visible gap between fingers (looks like they can slide)
    - Wider grip tips with ribbed inner surfaces
    """
    length = d.get("length", 50)
    width = d.get("width", 30)
    height = d.get("height", 15)

    script = f'''
import FreeCAD, Part, math

L, W, H = {length}, {width}, {height}
base_l = L * 0.4
finger_l = L * 0.6
finger_w = max(4.0, W * 0.12)
tip_extra = finger_w * 0.6
gap = max(6.0, W * 0.2)

# --- Base housing block ---
body = Part.makeBox(base_l, W, H)

# --- Servo cavity in base ---
cavity = Part.makeBox(base_l - 4, W * 0.5, H - 4)
cavity.translate(FreeCAD.Vector(2, W * 0.25, 2))
body = body.cut(cavity)

# --- Left finger (L-shaped, extends forward in +X) ---
lf_x = base_l
lf_y = (W - gap) / 2 - finger_w
lf = Part.makeBox(finger_l, finger_w, H)
lf.translate(FreeCAD.Vector(lf_x, lf_y, 0))
# Left finger tip hooks inward (toward center, +Y direction)
ltip = Part.makeBox(finger_l * 0.2, finger_w + tip_extra, H)
ltip.translate(FreeCAD.Vector(lf_x + finger_l * 0.8,
              lf_y - tip_extra / 2, 0))
lf = lf.fuse(ltip)

# --- Right finger (L-shaped, extends forward in +X) ---
rf_x = base_l
rf_y = (W + gap) / 2
rf = Part.makeBox(finger_l, finger_w, H)
rf.translate(FreeCAD.Vector(rf_x, rf_y, 0))
# Right finger tip hooks inward (toward center, -Y direction)
rtip = Part.makeBox(finger_l * 0.2, finger_w + tip_extra, H)
rtip.translate(FreeCAD.Vector(rf_x + finger_l * 0.8,
              rf_y + finger_w - tip_extra / 2, 0))
rf = rf.fuse(rtip)

# --- Grip ribs on left finger inner face ---
for i in range(3):
    rib_y = lf_y + finger_w * (0.3 + i * 0.2)
    rib = Part.makeBox(finger_l * 0.15, 0.8, 1.0)
    rib.translate(FreeCAD.Vector(lf_x + finger_l * 0.82, rib_y - 0.4, -0.5))
    lf = lf.cut(rib)

# --- Grip ribs on right finger inner face ---
for i in range(3):
    rib_y = rf_y + finger_w * (0.3 + i * 0.2)
    rib = Part.makeBox(finger_l * 0.15, 0.8, 1.0)
    rib.translate(FreeCAD.Vector(rf_x + finger_l * 0.82, rib_y - 0.4, -0.5))
    rf = rf.cut(rib)

# --- Visible slot between fingers and base (sliding interface) ---
slot_l = 2.0
slot = Part.makeBox(slot_l, gap + finger_w * 2, H)
slot.translate(FreeCAD.Vector(base_l - slot_l / 2,
              (W - gap) / 2 - finger_w, 0))

# --- Assemble: base + fingers ---
body = body.fuse(lf).fuse(rf)

# --- M3 mounting holes on base ---
for x_off in [base_l * 0.3, base_l * 0.7]:
    for y_off in [W * 0.15, W * 0.85]:
        h_cyl = Part.makeCylinder({_M3_CLEARANCE_R}, H + 2)
        h_cyl.translate(FreeCAD.Vector(x_off, y_off, -1))
        body = body.cut(h_cyl)

# --- Chamfer finger tips for grip ---
try:
    body = body.makeChamfer(0.5, [_e for _e in body.Edges
                                   if _e.Length < finger_w * 2][:30])
except Exception:
    pass

obj = doc.addObject("Part::Feature", "{name}_final")
obj.Shape = body
doc.recompute()
'''
    return [{"type": "raw_script", "script": script}]


def _gripper_finger_ops(name: str, d: dict) -> list[dict]:
    """Individual gripper finger — L-shaped with rail tab.

    Generates a single finger that slides on the gripper base's rail:
    - Main bar: extends forward (length direction)
    - L-shaped tip: hooks inward at the end to grip objects
    - Rail tab: protrusion on back that slides in the base rail groove
    - Grip pad: ribbed surface on the inner face

    Left vs right is detected from the name to mirror the tip direction.
    """
    length = d.get("length", 35)    # solver: Y extent
    width = d.get("width", 12)      # solver: X extent
    height = d.get("height", 15)    # solver: Z extent

    # Detect left/right from name to mirror the L-tip direction
    n_lower = name.lower()
    is_left = "left" in n_lower
    tip_dir = 1.0 if is_left else -1.0

    # Solver convention: makeBox(X=width, Y=length, Z=height).
    # The finger's main bar extends along -Y (FRONT, the reach direction —
    # the arm chain reaches toward -Y/front, so the gripper fingers must
    # also point -Y to grip objects in front of the robot). The L-tip hooks
    # sideways along X (width direction) at the front (-Y) end.
    # NOTE: a previous version built the bar at +Y (0..+L) with the tip at
    # the +Y end, which pointed the fingers BACKWARD (+Y = back anchor) —
    # the "夹爪前后反了" defect. Flip to -Y so the tip leads into the
    # forward workspace.
    script = f'''
import FreeCAD, Part, math

L, W, H = {length}, {width}, {height}
tip_dir = {tip_dir}

# --- Main bar extends toward -Y (front). makeBox then mirror about Y=0. ---
bar = Part.makeBox(W, L, H)
# Shift so the bar spans Y = -L .. 0 (tip end at -L, rail end at 0).
bar.translate(FreeCAD.Vector(0, -L, 0))

# --- Rail tab on back (fits into base rail groove), at the +Y (rear) end. ---
rail_tab_l = max(3.0, L * 0.1)
rail_tab_w = max(3.0, W * 0.8)
rail_tab = Part.makeBox(rail_tab_w, rail_tab_l, H * 0.6)
rail_tab.translate(FreeCAD.Vector((W - rail_tab_w) / 2,
                    -rail_tab_l * 0.5, H * 0.2))
bar = bar.fuse(rail_tab)

# --- L-shaped tip at front end (Y = -L .. -L+tip_l), hooks inward along X. ---
tip_l = L * 0.25
tip_w = max(4.0, W * 0.4)
if tip_dir > 0:
    tip_x = W  # extend toward +X
else:
    tip_x = -tip_w  # extend toward -X
tip = Part.makeBox(tip_w, tip_l, H)
tip.translate(FreeCAD.Vector(tip_x, -L, 0))
bar = bar.fuse(tip)

# --- Grip pad ribs (grooves on inner face of tip) ---
# These are shallow horizontal grooves for grip texture.  Previously
# the rib makeBox started at Z=-0.5 (poking through the finger's
# bottom face at Z=0), which created an open hole → non-watertight
# STL (euler=1, is_watertight=False) that VTK rendered as a thin
# shell instead of a solid prong.  Now the grooves sit fully INSIDE
# the tip volume: they start at Z=1.0 with height 1.5, leaving a
# 1mm floor so the boolean cut never breaks the outer surface.
rib_count = 3
rib_spacing = tip_w / (rib_count + 1)
for i in range(rib_count):
    rib_x = tip_x + rib_spacing * (i + 1) if tip_dir > 0 else tip_x + tip_w - rib_spacing * (i + 1)
    rib = Part.makeBox(1.0, tip_l * 0.8, 1.5)
    rib.translate(FreeCAD.Vector(rib_x - 0.5, -L + tip_l * 0.1, 1.0))
    bar = bar.cut(rib)

# --- NO chamfer on the finger body ---
# A previous version applied ``makeChamfer(0.5, ...)`` here, swallowed
# inside ``try/except: pass`` (AGENTS.md §1.1 violation).  Verified by
# experiment (scripts/verify_finger_watertight.py): the chamfer was NOT
# cosmetic — it broke water-tightness.  The finger is a fused multi-box
# shape (bar + rail_tab + tip); the fuse produces non-manifold edges
# along the shared tip/bar seam.  OCC's makeChamfer partially applies to
# those edges and leaves open triangles in the tip's top face → STL with
# euler=7 and 3 boundary holes (trimesh), rendered by VTK as a thin
# broken shell.  Removing the chamfer yields a watertight solid
# (watertight=True, euler=8).  A finger is a functional grip element;
# a decorative edge break is not worth a non-printable mesh.

obj = doc.addObject("Part::Feature", "{name}_final")
obj.Shape = bar
doc.recompute()
'''
    return [{"type": "raw_script", "script": script}]
