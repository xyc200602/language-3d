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
from dataclasses import dataclass, field

from ..knowledge.mechanics import Part


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


def _classify(name: str) -> str:
    """Return a broad part family from the part name."""
    n = name.lower()
    if n in ("base_plate", "top_plate"):
        return "plate"
    if n.startswith("standoff_"):
        return "standoff"
    if n.startswith("wheel_"):
        return "wheel"
    if n.startswith("motor_"):
        return "motor"
    if "_gripper" in n:
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
    # Arm joints
    if re.match(r"arm_[lr]_(base|shoulder|elbow|wrist)", n):
        return "arm_joint"
    # Arm links
    if re.match(r"arm_[lr]_(upper_link|forearm)", n):
        return "arm_link"
    return "unknown"


def infer_features(part: Part) -> FeatureConfig:
    """Analyse *part* and return the engineering features it should have."""

    name = part.name
    d = part.dimensions
    family = _classify(name)
    cfg = FeatureConfig()

    if family == "plate":
        l, w = d["length"], d["width"]
        margin = min(l, w) * 0.08
        hole_r = 2.0 if name == "base_plate" else 1.5  # M4 vs M3
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
        cfg.bore = {"diameter_mm": 2.6, "through": True}  # M3 clearance

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
                "diameter_mm": 3.0,
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
        hole_d = 4.0 if joint_type == "base" else 3.0
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
                "diameter_mm": 4.0,
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
                "diameter_mm": 3.0,
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
                "diameter_mm": 4.0,
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
        margin_pcb = min(l, w) * 0.08
        cfg.mounting_holes = [
            {
                "diameter_mm": 3.0,
                "pattern": "grid",
                "count_x": 2,
                "count_y": 2,
                "margin": margin_pcb,
            }
        ]
        cfg.chamfers = [{"size_mm": 0.5}]

    elif family == "bracket":
        l, w, h = d["length"], d["width"], d["height"]
        margin_bkt = min(l, w) * 0.1
        cfg.mounting_holes = [
            {
                "diameter_mm": 3.0,
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
                "diameter_mm": 3.0,
                "pattern": "polar",
                "count": 4,
                "pitch_radius": od / 2 * 0.8,
            }
        ]

    elif family == "sensor_tower_post":
        cfg.bore = {"diameter_mm": 5.0, "through": True}

    elif family == "sensor_mount":
        # imu_mount is box-shaped, lidar_mount is cylindrical
        if "length" in d:
            l, w = d["length"], d["width"]
            margin_sm = min(l, w) * 0.15
            cfg.mounting_holes = [
                {
                    "diameter_mm": 2.5,
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
                    "diameter_mm": 2.5,
                    "pattern": "polar",
                    "count": 4,
                    "pitch_radius": od / 2 * 0.7,
                }
            ]

    # "unknown" → empty config → fallback primitives

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


def generate_ops(part: Part, config: FeatureConfig | None = None) -> list[dict]:
    """Return a list of FreeCAD operation dicts for *part*.

    If *config* is ``None`` it is inferred via :func:`infer_features`.

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
    family = _classify(name)
    sn = _StepNamer(name)

    # ``body`` tracks the internal name of the current main solid.
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
            "name": body,
        })
    elif _has_cylindrical_dims(d):
        ops.append({
            "type": "make_cylinder",
            "radius": _cyl_radius(d),
            "height": d["height"],
            "name": body,
        })
    elif "length" in d and "width" in d:
        ops.append({
            "type": "make_box",
            "length": d["length"],
            "width": d["width"],
            "height": d["height"],
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

    # ------------------------------------------------------------------
    # 10. Export STL — only the final body
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
    l = d["length"]
    w = d["width"]
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
    """Add a cable routing channel (sketch + pocket).  Returns new body name."""
    cw = ch["width"]
    ch_h = ch["height"]
    l = d["length"]
    w = d["width"]
    h = d["height"]
    start = ch["start_offset"]
    end = ch["end_offset"]

    sketch_name = sn.tool("chsk")
    pocket_name = sn.tool("pocket")

    # Create sketch on XY plane offset to top surface
    ops.append({
        "type": "create_sketch",
        "name": sketch_name,
        "plane": "XY",
        "offset": h,  # top surface
        "elements": [
            {
                "type": "rectangle",
                "x": start,
                "y": (w - cw) / 2,
                "width": end - start,
                "height": cw,
            }
        ],
    })
    # Pocket cut downward — creates new document object
    ops.append({
        "type": "pocket",
        "sketch": sketch_name,
        "target": body,
        "depth": ch_h,
        "reverse": True,
        "name": pocket_name,
    })
    ops.append({"type": "delete_object", "object": body})
    ops.append({"type": "delete_object", "object": sketch_name})
    return pocket_name
