"""FreeCAD Python API control tool.

Provides parametric 3D modeling capabilities through FreeCAD's Python API.
FreeCAD is free, open-source, and supports headless (no GUI) operation.

Since FreeCAD bundles its own Python (3.11), this module uses subprocess
to call FreeCAD's Python interpreter for all operations.

Installation:
  1. winget install FreeCAD
  2. Or download from https://www.freecad.org/downloads.php
  3. Set FREECAD_PATH env var if not in default location
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from ..models.base import ToolDefinition
from .base import Tool
from .process_manager import _process_manager


# --- Security helpers ---

import re as _re

def _safe_name(name: str) -> str:
    """Sanitize a FreeCAD object/document name to prevent script injection.

    Only allows alphanumeric, underscore, hyphen, and CJK characters.
    """
    if not name:
        return "Unnamed"
    sanitized = _re.sub(r"[^A-Za-z0-9_\-]", "_", name)
    if not sanitized or sanitized.startswith("_"):
        sanitized = "obj_" + sanitized
    return sanitized[:64]


def _safe_path(path: str) -> str:
    """Sanitize a file path to prevent script injection in f-string contexts.

    Rejects paths containing quotes or other dangerous characters.
    NOTE: Paths should be passed via json.dumps() in generated scripts
    rather than f-string interpolation to prevent injection.
    """
    p = str(path)
    # Block characters that could escape string contexts in generated scripts
    if any(c in p for c in ('"', "'", "\n", "\r", ";", "`", "$", "\x00")):
        raise ValueError(f"Path contains dangerous characters: {path!r}")
    return p


def _validate_raw_script(script: str) -> None:
    """Validate a raw_script operation to block dangerous FreeCAD API calls."""
    blocked_patterns = [
        r"\bos\.system\b",
        r"\bos\.popen\b",
        r"\bsubprocess\b",
        r"\bexec\s*\(",
        r"\beval\s*\(",
        r"\b__import__\b",
        r"\bopen\s*\(",
        r"\bshutil\b",
        r"\bctypes\b",
        r"\bimportlib\b",
        r"\bPath\s*\([^)]*\)\s*\.write_text\b",
        r"\bPath\s*\([^)]*\)\s*\.write_bytes\b",
        r"\b__builtins__\b",
        r"\bcompile\s*\(",
        r"\bglobals\s*\(\s*\)\s*\[",
        r"\bgetattr\s*\([^)]*__builtins__",
        r"\bsocket\b",
        r"\bhttp\b",
        r"\burllib\b",
    ]
    for pat in blocked_patterns:
        if _re.search(pat, script):
            raise ValueError(f"Raw script contains blocked pattern: {pat}")


# --- FreeCAD subprocess bridge ---

def _find_freecad_python() -> str | None:
    """Find FreeCAD's bundled Python executable."""
    # Check env var
    fc_path = os.environ.get("FREECAD_PATH")
    if fc_path:
        python = str(Path(fc_path) / "python.exe")
        if Path(python).exists():
            return python

    # Common installation paths
    common_paths = [
        os.path.expanduser(r"~\AppData\Local\Programs\FreeCAD 1.1\bin"),
        os.path.expanduser(r"~\AppData\Local\Programs\FreeCAD 1.0\bin"),
        r"C:\Program Files\FreeCAD 1.1\bin",
        r"C:\Program Files\FreeCAD 1.0\bin",
        r"C:\Program Files\FreeCAD\bin",
    ]
    for p in common_paths:
        python = str(Path(p) / "python.exe")
        if Path(python).exists():
            return python

    return None


def _run_freecad_script(script: str, timeout: int = 300) -> str:
    """Execute a Python script using FreeCAD's bundled Python.

    Returns the stdout output.
    Raises RuntimeError on failure.
    """
    fc_python = _find_freecad_python()
    if not fc_python:
        raise RuntimeError(
            "FreeCAD not found. Install with: winget install FreeCAD\n"
            "Or set FREECAD_PATH to FreeCAD's bin directory."
        )

    # Write script to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [fc_python, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            raise RuntimeError(f"FreeCAD script error:\n{error_msg}")
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        raise RuntimeError("FreeCAD script timed out")
    finally:
        Path(script_path).unlink(missing_ok=True)


def _xml_set_param(root: "ET.Element", group_name: str, param_name: str, value: str, tag: str = "FCBool") -> None:
    """Set a parameter in FreeCAD's user.cfg XML tree.

    Searches all FCParamGroup elements for one matching group_name,
    then sets or creates a child element with the given param_name and value.
    tag: XML element type (FCBool, FCString, FCInt, FCFloat).
    """
    import xml.etree.ElementTree as ET

    for group in root.iter("FCParamGroup"):
        if group.get("Name") == group_name:
            for child in group:
                if child.get("Name") == param_name:
                    child.set("Value", value)
                    break
            else:
                ET.SubElement(group, tag, {"Name": param_name, "Value": value})
            break



# ---------------------------------------------------------------------------
# Script builders extracted to freecad_script_builder.py (P1-1 split).
# Re-exported here for backward compatibility.
# ---------------------------------------------------------------------------
from .freecad_script_builder import (  # noqa: F401
    _build_script, _build_batch_script,
    _FASTENER_DIMS, _FREECAD_ANCHOR_NORMALS,
    SUBSYSTEM_COLORS, _mesh_export_lines,
)

def _z_to_normal_rotation(
    normal: tuple[float, float, float],
) -> tuple[float, float, float, float] | None:
    """Axis-angle rotation mapping +Z to *normal*.

    FreeCAD cylinders from ``Part.makeCylinder`` align along +Z.  This
    returns ``(ax, ay, az, angle_deg)`` for ``Shape.rotate()`` so the
    cylinder aligns with the anchor face normal.  Returns None if no
    rotation is needed.
    """
    import math as _math
    nx, ny, nz = normal
    dot = nz
    if dot > 0.9999:
        return None
    if dot < -0.9999:
        return (1.0, 0.0, 0.0, 180.0)
    ax_raw, ay_raw = -ny, nx
    al = _math.sqrt(ax_raw * ax_raw + ay_raw * ay_raw)
    angle = _math.degrees(_math.acos(max(-1.0, min(1.0, dot))))
    return (ax_raw / al, ay_raw / al, 0.0, angle)


def _compute_bolt_hole_world_positions(
    joint,
    positions: dict[str, dict],
    part_dims: dict[str, dict],
) -> list[tuple[tuple[float, float, float], tuple[float, float, float], float]]:
    """Compute world-space positions of bolt holes for a single bolted joint.

    Reuses :class:`ConnectionFeatureEngine` to get the same bolt layout that
    was used to drill the holes in the STL, then transforms each hole from
    FreeCAD part-local (corner-origin, X=length/Y=width/Z=height) to world
    coordinates using the same transform chain the VTK renderer applies:
    center → R_z(-90°) swap → solver rotation → translate.

    Args:
        joint: Joint object with ``connection.type == "bolted"``.
        positions: Solver output ``{part_name: {"position": [...], "rotation": [...]}}``.
        part_dims: ``{part_name: dimensions_dict}`` lookup.

    Returns:
        List of ``(world_pos, normal_world, thickness)`` tuples where:
        - ``world_pos`` is ``(x, y, z)`` of the bolt hole centre on the face.
        - ``normal_world`` is the unit outward normal of the face in world
          space (bolt head sits on the +normal side, nut on the −normal side).
        - ``thickness`` is the part thickness at this face (for bolt length).
        Empty list if the joint is not bolted or cannot be resolved.
    """
    import math as _math

    conn = getattr(joint, "connection", None)
    if conn is None or getattr(conn, "type", "") != "bolted":
        return []
    if getattr(joint, "type", "") == "prismatic":
        return []

    parent_name = getattr(joint, "parent", "")
    child_name = getattr(joint, "child", "")

    # Determine which part is structural (gets drilled) → its anchor positions the holes.
    # Deferred imports to avoid circular dependency (connection_features ← freecad is safe,
    # but connection_features imports knowledge.mechanics which other tools import).
    from .connection_features import ConnectionFeatureEngine, _pick_structural_part
    from .assembly_solver import _rotation_matrix_axis_angle, _mat_vec
    from ..knowledge.mechanics import Part

    parent_dims = part_dims.get(parent_name, {})
    child_dims = part_dims.get(child_name, {})
    parent_part = Part(name=parent_name, category="structural",
                       description="", dimensions=parent_dims)
    child_part = Part(name=child_name, category="structural",
                      description="", dimensions=child_dims)
    structural = _pick_structural_part(parent_part, child_part)
    structural_name = structural.name

    # Fallback: if the picked structural part is cylindrical (only
    # diameter/height, no length/width face for bolt layout), try the
    # other part.  This happens because _pick_structural_part is called
    # with both categories set to "structural" (we don't have the real
    # categories here), so name heuristics may pick a cylindrical joint
    # over a box link.  The cylindrical part cannot receive a bolt layout.
    _sd = part_dims.get(structural_name, {})
    if "length" not in _sd and "width" not in _sd:
        _other = child_part if structural is parent_part else parent_part
        _od = part_dims.get(_other.name, {})
        if "length" in _od or "width" in _od:
            structural = _other
            structural_name = structural.name

    is_structural_child = structural_name == child_name
    anchor = joint.child_anchor if is_structural_child else joint.parent_anchor

    d = part_dims.get(structural_name, {})
    if not d:
        return []

    # Skip if still cylindrical after fallback (both parts cylindrical)
    if "length" not in d and "width" not in d:
        return []

    bolt_size = getattr(conn, "bolt_size", "M3") or "M3"
    count = getattr(conn, "bolt_count", 4) or 4
    if count <= 0:
        return []

    # Hole diameter (try catalog, fall back to heuristic)
    try:
        from ..knowledge.fastener_catalog import get_clearance_hole_with_tolerance
        hole_d, _, _ = get_clearance_hole_with_tolerance(bolt_size)
    except Exception:
        try:
            hole_d = float(bolt_size.lstrip("M")) + 0.4
        except Exception:
            hole_d = 3.4

    engine = ConnectionFeatureEngine()
    bolt_layout = engine._auto_layout_bolts(count, d, anchor, hole_d)
    if not bolt_layout:
        return []

    thickness = ConnectionFeatureEngine._infer_thickness(d, anchor)

    # Solver world transform for the structural part
    pos_data = positions.get(structural_name, {})
    if not pos_data:
        return []
    part_pos = pos_data.get("position", [0, 0, 0])
    part_rot = pos_data.get("rotation", [0, 0, 1, 0])

    # Build solver rotation matrix from axis-angle degrees
    ax_r, ay_r, az_r, angle_deg = part_rot
    axis_len = _math.sqrt(ax_r * ax_r + ay_r * ay_r + az_r * az_r)
    if axis_len < 1e-10 or abs(angle_deg) < 1e-6:
        R_solver = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
    else:
        R_solver = _rotation_matrix_axis_angle(
            (ax_r / axis_len, ay_r / axis_len, az_r / axis_len),
            _math.radians(angle_deg),
        )

    # Part dimensions in FreeCAD-local convention
    L = d.get("length", 20)
    W = d.get("width", 20)
    H = d.get("height", d.get("thickness", 10))

    # Anchor normal in FreeCAD-local
    fc_normal = _FREECAD_ANCHOR_NORMALS.get(anchor, (0, 0, 1))

    # Apply R_z(-90°) to the normal: (x,y,z) → (y, -x, z)
    # This is the renderer's swap_xy transform.
    swapped_normal = (fc_normal[1], -fc_normal[0], fc_normal[2])

    # Apply solver rotation to the swapped normal → world-space normal
    normal_world = _mat_vec(R_solver, swapped_normal)
    nl = _math.sqrt(normal_world[0]**2 + normal_world[1]**2 + normal_world[2]**2)
    if nl > 1e-10:
        normal_world = (normal_world[0]/nl, normal_world[1]/nl, normal_world[2]/nl)

    results = []
    for uv_pos, _hole_dia in bolt_layout:
        # Bolt hole position in FreeCAD-local (corner-origin, X=length/Y=width/Z=height)
        fc_pos = engine._position_on_face(uv_pos, anchor, d, thickness)
        fc_x, fc_y, fc_z = fc_pos

        # Step 1: Subtract part centre (corner → centre origin)
        cx = fc_x - L / 2.0
        cy = fc_y - W / 2.0
        cz = fc_z - H / 2.0

        # Step 2: Apply R_z(-90°): (x,y,z) → (y, -x, z)
        # This mirrors the VTK renderer's swap_xy transform.
        sx = cy        # swapped X = original Y (width)
        sy = -cx       # swapped Y = -original X (length)
        sz = cz        # Z unchanged

        # Step 3: Apply solver rotation
        rx, ry, rz = _mat_vec(R_solver, (sx, sy, sz))

        # Step 4: Translate to world
        world_x = part_pos[0] + rx
        world_y = part_pos[1] + ry
        world_z = part_pos[2] + rz

        results.append(((world_x, world_y, world_z), normal_world, thickness))

    return results


def _fastener_script_lines(
    joints: list,
    positions: dict[str, dict],
    part_dims: dict[str, dict],
) -> list[str]:
    """Generate FreeCAD script lines for bolts, washers, and nuts.

    Uses :func:`_compute_bolt_hole_world_positions` to place each fastener at
    the actual bolt hole location on the structural part's anchor face,
    oriented along the face normal.  Each fastener consists of a socket-head
    cap screw (head + shank), a flat washer, and a hex nut.

    Prismatic joints (sliding gripper fingers) are skipped — they don't
    use bolts.
    """
    import math as _math

    lines: list[str] = []
    fidx = 0

    for joint in joints:
        conn = getattr(joint, "connection", None)
        if conn is None or getattr(conn, "type", "") != "bolted":
            continue
        if getattr(joint, "type", "") == "prismatic":
            continue

        bolt_holes = _compute_bolt_hole_world_positions(joint, positions, part_dims)
        if not bolt_holes:
            continue

        bolt_size = getattr(conn, "bolt_size", "M3") or "M3"
        dims = _FASTENER_DIMS.get(bolt_size, _FASTENER_DIMS["M3"])
        head_r, head_h, shank_r, washer_r, washer_h, nut_r, nut_h = dims

        parent_name = getattr(joint, "parent", "")
        child_name = getattr(joint, "child", "")
        lines.append(
            f"# --- Bolts {bolt_size}: "
            f"{parent_name} -> {child_name} ---"
        )

        for world_pos, normal, thickness in bolt_holes:
            shank_length = max(thickness + washer_h + nut_h + 1.0, 8.0)

            # Rotation to align +Z cylinder with the anchor normal.
            rot = _z_to_normal_rotation(normal)
            rot_ax, rot_ay, rot_az, rot_ang = rot if rot else (0, 0, 1, 0.0)

            nx, ny, nz = normal
            px, py, pz = world_pos

            def _emit_cylinder(
                radius: float, height: float,
                cx: float, cy: float, cz: float,
                name: str, color: tuple,
            ) -> None:
                """Emit FreeCAD lines for a cylinder centred at (cx,cy,cz)."""
                # makeCylinder base at z=0, top at z=height.
                # Centre it: translate base by -height/2 in Z.
                lines.append(f"_f = Part.makeCylinder({radius}, {height})")
                lines.append(
                    f"_f.translate(FreeCAD.Vector(0, 0, {-height / 2.0:.2f}))"
                )
                if abs(rot_ang) > 0.01:
                    lines.append(
                        f"_f.rotate(FreeCAD.Vector(0,0,0), "
                        f"FreeCAD.Vector({rot_ax:.2f},{rot_ay:.2f},{rot_az:.2f}), "
                        f"{rot_ang:.1f})"
                    )
                lines.append(
                    f"_f.translate(FreeCAD.Vector({cx:.1f}, {cy:.1f}, {cz:.1f}))"
                )
                lines.append(
                    f'_o = doc.addObject("Part::Feature", "{name}_{fidx}")'
                )
                lines.append("_o.Shape = _f")
                lines.append(f"if _o.ViewObject is not None: _o.ViewObject.ShapeColor = {color}")
                # No per-fastener recompute — see the per-part note above.
                # A dual-arm assembly emits ~400 fasteners (×4 cylinders each);
                # a recompute after each would be O(n²) and overflowed FreeCAD's
                # stack.  The single final recompute (before save) resolves all.

            # world_pos is ON the +normal face.  Bolt goes THROUGH the part:
            # head flush on +normal face, shank through body, washer + nut
            # on the −normal face.

            # Bolt head: flush on +normal face, extending outward
            hd = head_h / 2.0
            _emit_cylinder(
                head_r, head_h,
                px + nx * hd, py + ny * hd, pz + nz * hd,
                f"bolt_{bolt_size}_head", (0.80, 0.80, 0.82),
            )

            # Bolt shank: from +normal face through part to −normal protrusion
            sd = shank_length / 2.0
            _emit_cylinder(
                shank_r, shank_length,
                px - nx * sd, py - ny * sd, pz - nz * sd,
                f"bolt_{bolt_size}_shank", (0.80, 0.80, 0.82),
            )

            # Washer: flush on −normal face
            wd = thickness + washer_h / 2.0
            _emit_cylinder(
                washer_r, washer_h,
                px - nx * wd, py - ny * wd, pz - nz * wd,
                f"washer_{bolt_size}", (0.70, 0.70, 0.72),
            )

            # Nut: beyond washer on −normal side
            nd = thickness + washer_h + nut_h / 2.0
            _emit_cylinder(
                nut_r, nut_h,
                px - nx * nd, py - ny * nd, pz - nz * nd,
                f"nut_{bolt_size}", (0.65, 0.65, 0.68),
            )

            fidx += 1

        lines.append("")

    if lines:
        lines.insert(
            0,
            "# =========================================================",
        )
        lines.insert(
            1,
            "# FASTENERS — bolts, washers, nuts at bolted connections",
        )
        lines.insert(
            2,
            "# =========================================================",
        )
        lines.insert(3, "")

    return lines


def build_assembly_script(
    assembly_parts: list[dict],
    positions: dict[str, dict],
    output_path: str = "",
    exploded: bool = False,
    explode_factor: float = 1.5,
    joints: list | None = None,
) -> str:
    """Build a FreeCAD script that renders a full assembly with positioned parts.

    Args:
        assembly_parts: List of dicts with keys: name, shape_type, dimensions, subsystem
        positions: Dict mapping part_name → {"position": [x,y,z], "rotation": [rx,ry,rz]}
        output_path: If set, save FCStd and export STL/STEP here
        exploded: If True, offset parts along their position vector for exploded view
        explode_factor: Distance multiplier for exploded view
        joints: Optional list of Joint objects.  When provided, bolt/nut/washer
            geometry is generated at every bolted connection.

    Returns:
        FreeCAD Python script text.
    """
    lines = [
        "import FreeCAD",
        "import Part",
        "import Mesh",
        "import math",
        "",
        'doc = FreeCAD.newDocument("Assembly")',
        "",
    ]

    for i, part_info in enumerate(assembly_parts):
        name = part_info["name"]
        shape_type = part_info.get("shape_type", "box")
        dims = part_info.get("dimensions", {})
        subsystem = part_info.get("subsystem", "")

        pos_data = positions.get(name, {})
        pos = pos_data.get("position", [0, 0, 0])
        rot = pos_data.get("rotation", [0, 0, 1, 0])

        if exploded:
            # Offset along position vector from center
            dist = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
            if dist > 0:
                scale = 1.0 + (explode_factor - 1.0) * min(dist / 200.0, 1.0)
                pos = [pos[0] * scale, pos[1] * scale, pos[2] * scale]

        # Create shape — prefer real STL geometry, fall back to primitive
        stl_path = part_info.get("stl_path")
        if stl_path:
            lines.append("import os as _os")
            lines.append(f"_stl_path = {json.dumps(str(stl_path))}")
            lines.append("if _os.path.exists(_stl_path):")
            lines.append("    _mesh = Mesh.read(_stl_path)")
            # FreeCAD 1.1 exposes the topology via ``.Topology`` (capital);
            # older builds used ``.topology``.  Try both so the assembly
            # render works across versions — a NameError here silently
            # killed the WHOLE assembly export (assembly.stl/FCStd never
            # generated) because the caller wrapped it in except: pass.
            lines.append("    _topo = getattr(_mesh, 'Topology', None)")
            lines.append("    if _topo is None: _topo = getattr(_mesh, 'topology')")
            lines.append("    _shape = Part.Shape(_topo)")
            lines.append("else:")
            _indent = "    "
        else:
            _indent = ""

        if shape_type == "cylinder" or "diameter" in dims or "outer_diameter" in dims:
            r = dims.get("diameter", dims.get("outer_diameter", 10)) / 2
            h = dims.get("height", 10)
            lines.append(f"{_indent}_shape = Part.makeCylinder({r}, {h})")
        elif shape_type == "box" or ("length" in dims and "width" in dims):
            l = dims.get("length", 10)
            w = dims.get("width", 10)
            h = dims.get("height", 10)
            # Solver convention: X=width (left/right), Y=length (front/back), Z=height.
            # FreeCAD makeBox(X, Y, Z), so pass (width, length, height) to match solver.
            lines.append(f"{_indent}_shape = Part.makeBox({w}, {l}, {h})")
        else:
            r = dims.get("diameter", 10) / 2
            h = dims.get("height", 10)
            lines.append(f"{_indent}_shape = Part.makeCylinder({r}, {h})")

        # Apply rotation around origin, then translate to position
        if rot and len(rot) == 4 and abs(rot[3]) > 1e-6:
            ax, ay, az, angle = rot
            lines.append(f"_shape.rotate(FreeCAD.Vector(0,0,0), FreeCAD.Vector({ax},{ay},{az}), {angle:.4f})")

        lines.append(f"_shape.translate(FreeCAD.Vector({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}))")

        # Add to document
        lines.append(f'_obj = doc.addObject("Part::Feature", "{name}")')
        lines.append("_obj.Shape = _shape")

        # Apply subsystem color (guard ViewObject — None in headless FreeCADCmd)
        color = SUBSYSTEM_COLORS.get(subsystem, (0.7, 0.7, 0.7))
        lines.append(f"if _obj.ViewObject is not None: _obj.ViewObject.ShapeColor = ({color[0]}, {color[1]}, {color[2]})")
        # NOTE: no per-object doc.recompute() here.  The old code called
        # recompute after EVERY object, which is O(n²): recompute #438
        # processes all 438 objects.  On a dual-arm assembly (38 parts + 400
        # fasteners = 438 objects) this triggered a FreeCAD stack-buffer-
        # overrun (exit 0xC0000409) — the assembly.stl was never produced.
        # Object creation + Shape assignment does not need an immediate
        # recompute; a single recompute before save/export resolves all
        # objects at once (O(n), not O(n²)).
        lines.append("")

    # --- Fasteners (bolts, washers, nuts) at bolted connections ---
    if joints and not exploded:
        part_dims_lookup = {
            p["name"]: p.get("dimensions", {}) for p in assembly_parts
        }
        fastener_lines = _fastener_script_lines(
            joints, positions, part_dims_lookup
        )
        lines.extend(fastener_lines)

    # Save and export.  A single recompute here resolves ALL objects at once
    # (O(n)) — replacing the per-object recomputes that made the build O(n²)
    # and overflowed FreeCAD's stack on the 438-object dual-arm assembly.
    lines.append("doc.recompute()")
    if output_path:
        lines.append(f'doc.saveAs({json.dumps(str(output_path) + ".FCStd")})')
        lines.append(f'_all_objs = [o for o in doc.Objects if hasattr(o, "Shape")]')
        lines.extend(_mesh_export_lines("_all_objs", str(output_path) + ".stl"))
        lines.append("print(f'Assembly saved and exported')")

    lines.append("print(f'Assembly: {len(doc.Objects)} objects created')")

    return "\n".join(lines)


def build_assembly_stl_trimesh(
    assembly_parts: list[dict],
    positions: dict[str, dict],
    output_path: str,
    joints: list | None = None,
) -> str:
    """Export an assembly STL via trimesh (no FreeCAD subprocess).

    FreeCAD's single-script assembly render overflows its stack on large
    assemblies (the dual-arm robot = 38 parts + 400 fasteners = 438 objects
    crashes with STATUS_STACK_BUFFER_OVERRUN).  This pure-Python path uses
    trimesh to load each part's STL, apply the solver transform, add
    simple cylindrical fasteners at every bolted connection, and merge —
    producing the same ``assembly.stl`` deliverable without FreeCAD's
    process-size limit.

    The geometry is approximate: parts use their real per-part STL
    (already exported by the pipeline) and fasteners are parametric
    cylinders (bolt head + shank + washer + nut) sized from the real ISO
    catalog, oriented along the joint face normal — the same data the
    FreeCAD path uses, just rendered by trimesh instead.

    Args:
        assembly_parts: list of dicts with name, stl_path, dimensions.
        positions: solver output {name: {"position":[x,y,z], "rotation":[...]}}.
        output_path: where to write assembly.stl (the .stl suffix is added).
        joints: optional Joint list — bolted connections get fasteners.

    Returns:
        The path to the written assembly.stl.
    """
    import trimesh
    import numpy as np
    from .connection_features import (
        ConnectionFeatureEngine,
        get_bolt_head_dims, get_nut_dims, get_washer_dims,
    )

    meshes: list[trimesh.Trimesh] = []
    # --- Parts: load each STL, apply solver transform ---
    for part in assembly_parts:
        name = part["name"]
        stl_path = part.get("stl_path")
        if not stl_path or not Path(stl_path).exists():
            continue
        try:
            m = trimesh.load(stl_path, process=False)
        except Exception:
            continue
        if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
            continue
        pose = positions.get(name, {})
        pos = pose.get("position", [0, 0, 0])
        rot = pose.get("rotation")
        # Centre the STL (FreeCAD writes corner-at-origin) then transform.
        # The renderer's swap_xy convention: for arm-chain parts the STL has
        # length on X and is swapped; for chassis boxes the STL already
        # matches solver axes.  We mirror the renderer's _swap decision.
        n_lower = name.lower()
        _is_wheel = n_lower.startswith("wheel_")
        _is_chassis_box = n_lower in (
            "base_plate", "chassis_body", "battery_box", "top_plate",
        ) or n_lower.startswith("standoff_")
        if not _is_wheel and not _is_chassis_box:
            # swap_xy: rotate 90° about Z (X↔Y) to match solver convention.
            m.apply_transform(trimesh.transformations.rotation_matrix(
                np.radians(-90), [0, 0, 1],
            ))
        # Apply solver rotation (axis-angle) then translation.
        if rot and len(rot) == 4 and abs(rot[3]) > 1e-6:
            ax, ay, az, ang = rot
            m.apply_transform(trimesh.transformations.rotation_matrix(
                np.radians(ang), [ax, ay, az],
            ))
        m.apply_translation([pos[0], pos[1], pos[2]])
        meshes.append(m)

    # --- Fasteners at bolted joints ---
    if joints:
        part_dims_lookup = {p["name"]: p.get("dimensions", {}) for p in assembly_parts}
        for joint in joints:
            conn = getattr(joint, "connection", None)
            if conn is None or getattr(conn, "type", "") != "bolted":
                continue
            if getattr(joint, "type", "") == "prismatic":
                continue
            try:
                holes = _compute_bolt_hole_world_positions(
                    joint, positions, part_dims_lookup,
                )
            except Exception:
                continue
            bolt_size = getattr(conn, "bolt_size", "M3") or "M3"
            head_d, head_h = get_bolt_head_dims(bolt_size)
            try:
                nut_w, nut_h = get_nut_dims(bolt_size)
            except Exception:
                nut_w, nut_h = head_d * 1.6, head_d * 0.5
            try:
                washer_od, washer_id, washer_h = get_washer_dims(bolt_size)
            except Exception:
                washer_od, washer_id, washer_h = head_d * 1.4, head_d * 0.4, 1.0
            for (wx, wy, wz), normal, thickness in holes:
                n = np.array(normal, dtype=float)
                norm = np.linalg.norm(n)
                if norm < 1e-9:
                    continue
                n = n / norm
                # Bolt head (cylinder, axis along normal, on +normal face).
                head_r = head_d / 2.0
                head_cyl = trimesh.creation.cylinder(head_r, height=head_h)
                head_cyl.apply_transform(_align_z_to(n))
                head_cyl.apply_translation(
                    [wx + n[0] * head_h / 2, wy + n[1] * head_h / 2,
                     wz + n[2] * head_h / 2],
                )
                meshes.append(head_cyl)
                # Shank (through the part).
                shank_r = float(bolt_size[1:]) / 2.0 if bolt_size[1:].isdigit() else head_r * 0.4
                shank_len = max(thickness + head_h, 5.0)
                shank = trimesh.creation.cylinder(shank_r, height=shank_len)
                shank.apply_transform(_align_z_to(n))
                shank.apply_translation([wx, wy, wz])
                meshes.append(shank)
                # Nut (cylinder approx) on -normal face.
                nut_r = nut_w / 2.0 * 0.9  # across-corners approx → radius
                nut = trimesh.creation.cylinder(nut_r, height=nut_h)
                nut.apply_transform(_align_z_to(n))
                nut.apply_translation(
                    [wx - n[0] * (thickness + nut_h / 2),
                     wy - n[1] * (thickness + nut_h / 2),
                     wz - n[2] * (thickness + nut_h / 2)],
                )
                meshes.append(nut)

    if not meshes:
        Path(output_path + ".stl").write_bytes(b"")
        return output_path + ".stl"
    combined = trimesh.util.concatenate(meshes)
    out = output_path + ".stl"
    combined.export(out)
    return out


def _align_z_to(direction) -> np.ndarray:
    """4×4 transform rotating +Z onto ``direction`` (for cylinder orientation)."""
    import numpy as np
    d = np.array(direction, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    # Rotation axis = z × d, angle = arccos(z·d).
    axis = np.cross(z, d)
    dot = float(np.clip(np.dot(z, d), -1.0, 1.0))
    import trimesh
    if np.linalg.norm(axis) < 1e-9:
        # Already aligned (or anti-aligned).
        if dot < 0:
            return trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        return np.eye(4)
    angle = np.arccos(dot)
    return trimesh.transformations.rotation_matrix(angle, axis)


def _shape_type_for_part(part: "Part") -> str:
    """Determine shape type from part dimensions."""
    dims = part.dimensions
    if "diameter" in dims or "outer_diameter" in dims:
        return "cylinder"
    if "length" in dims and "width" in dims:
        return "box"
    return "cylinder"


def _subsystem_for_part(part_name: str) -> str:
    """Infer subsystem from part name."""
    if part_name.startswith("arm_l_"):
        return "arm_left"
    if part_name.startswith("arm_r_"):
        return "arm_right"
    if "ipc" in part_name:
        return "ipc"
    if part_name.startswith(("sensor_", "imu_", "lidar_", "camera_")):
        return "sensor_tower"
    return "chassis"


def _execute_operations(operations: list[dict]) -> str:
    """Execute a sequence of FreeCAD operations and return the output."""
    script = _build_script(operations)
    try:
        output = _run_freecad_script(script)
        # Format volume_check results nicely
        has_vc = any(op.get("type") == "volume_check" for op in operations)
        if has_vc and output:
            for line in output.splitlines():
                if line.strip().startswith("VOLUME_CHECK_JSON:"):
                    json_str = line.strip()[len("VOLUME_CHECK_JSON:"):]
                    try:
                        data = json.loads(json_str)
                        passed = data.get("pass", False)
                        objects = data.get("objects", [])
                        result_lines = [
                            f"[Volume Check] {'PASS' if passed else 'FAIL'}",
                            f"Objects checked: {len(objects)}",
                        ]
                        for obj in objects:
                            dims = obj.get("dims", [0, 0, 0])
                            result_lines.append(
                                f"  {obj.get('label', obj.get('name', '?'))}: "
                                f"volume={obj.get('volume', 0):.1f}mm³ "
                                f"dims={dims[0]:.1f}x{dims[1]:.1f}x{dims[2]:.1f}mm"
                            )
                            if "volume_warning" in obj:
                                result_lines.append(f"    ⚠ {obj['volume_warning']}")
                            if "dim_warning" in obj:
                                result_lines.append(f"    ⚠ {obj['dim_warning']}")
                        result_lines.append(f"\n--- JSON ---\n{json_str}")
                        return "\n".join(result_lines)
                    except json.JSONDecodeError:
                        pass
        # Format compute_mass results
        has_cm = any(op.get("type") == "compute_mass" for op in operations)
        if has_cm and output:
            for line in output.splitlines():
                if line.strip().startswith("MASS_CHECK_JSON:"):
                    json_str = line.strip()[len("MASS_CHECK_JSON:"):]
                    try:
                        data = json.loads(json_str)
                        objects = data.get("objects", [])
                        total_mass = data.get("total_mass_kg", 0)
                        result_lines = [
                            "[Mass Check]",
                            f"Objects: {len(objects)}, Total mass: {total_mass:.6f} kg",
                        ]
                        for obj in objects:
                            result_lines.append(
                                f"  {obj.get('label', obj.get('name', '?'))}: "
                                f"volume={obj['volume_mm3']:.2f}mm³ "
                                f"mass={obj['mass_kg']:.6f}kg "
                                f"com={obj['center_mm']}"
                            )
                        result_lines.append(f"\n--- JSON ---\n{json_str}")
                        return "\n".join(result_lines)
                    except json.JSONDecodeError:
                        pass
        return output if output else "OK"
    except RuntimeError as e:
        return f"Error: {e}"


def _is_freecad_available() -> bool:
    """Check if FreeCAD is available."""
    return _find_freecad_python() is not None


# --- FreeCAD GUI support ---

VIEW_METHODS: dict[str, str] = {
    "isometric": "viewIsometric",
    "front": "viewFront",
    "top": "viewTop",
    "right": "viewRight",
    "left": "viewLeft",
    "back": "viewBack",
    "bottom": "viewBottom",
}


def _find_freecad_exe() -> str | None:
    """Find FreeCAD executable (GUI)."""
    fc_path = os.environ.get("FREECAD_PATH")
    if fc_path:
        exe = str(Path(fc_path) / "FreeCAD.exe")
        if Path(exe).exists():
            return exe

    common_paths = [
        os.path.expanduser(r"~\AppData\Local\Programs\FreeCAD 1.1\bin"),
        os.path.expanduser(r"~\AppData\Local\Programs\FreeCAD 1.0\bin"),
        r"C:\Program Files\FreeCAD 1.1\bin",
        r"C:\Program Files\FreeCAD 1.0\bin",
        r"C:\Program Files\FreeCAD\bin",
    ]
    for p in common_paths:
        exe = str(Path(p) / "FreeCAD.exe")
        if Path(exe).exists():
            return exe

    return None


# --- Tool definitions ---

class FCNewDocTool(Tool):
    name = "fc_new_doc"
    description = "Create a new FreeCAD document for 3D modeling"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "name": {"type": "string", "description": "Document name (optional)"},
            }, "required": []},
        )

    def execute(self, *, name: str = "Unnamed", **kwargs: Any) -> str:
        return _execute_operations([{"type": "new_doc", "name": name}])


class FCMakeBoxTool(Tool):
    name = "fc_make_box"
    description = "Create a 3D box with given dimensions (length, width, height in mm)"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "length": {"type": "number", "description": "Length in mm (X axis)"},
                "width": {"type": "number", "description": "Width in mm (Y axis)"},
                "height": {"type": "number", "description": "Height in mm (Z axis)"},
                "name": {"type": "string", "description": "Object name (optional)"},
            }, "required": ["length", "width", "height"]},
        )

    def execute(self, *, length: float, width: float, height: float, name: str = "Box", **kwargs: Any) -> str:
        return _execute_operations([
            {"type": "new_doc"},
            {"type": "make_box", "length": length, "width": width, "height": height, "name": name},
        ])


class FCMakeCylinderTool(Tool):
    name = "fc_make_cylinder"
    description = "Create a 3D cylinder with given radius and height (in mm)"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "radius": {"type": "number", "description": "Radius in mm"},
                "height": {"type": "number", "description": "Height in mm (Z axis)"},
                "name": {"type": "string", "description": "Object name (optional)"},
            }, "required": ["radius", "height"]},
        )

    def execute(self, *, radius: float, height: float, name: str = "Cylinder", **kwargs: Any) -> str:
        return _execute_operations([
            {"type": "new_doc"},
            {"type": "make_cylinder", "radius": radius, "height": height, "name": name},
        ])


class FCMakeSphereTool(Tool):
    name = "fc_make_sphere"
    description = "Create a 3D sphere with given radius (in mm)"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "radius": {"type": "number", "description": "Radius in mm"},
                "name": {"type": "string", "description": "Object name (optional)"},
            }, "required": ["radius"]},
        )

    def execute(self, *, radius: float, name: str = "Sphere", **kwargs: Any) -> str:
        return _execute_operations([
            {"type": "new_doc"},
            {"type": "make_sphere", "radius": radius, "name": name},
        ])


class FCMakeConeTool(Tool):
    name = "fc_make_cone"
    description = "Create a 3D cone with given radius, top radius, and height (in mm)"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "radius1": {"type": "number", "description": "Bottom radius in mm"},
                "radius2": {"type": "number", "description": "Top radius in mm (0 for pointed cone)"},
                "height": {"type": "number", "description": "Height in mm"},
                "name": {"type": "string", "description": "Object name (optional)"},
            }, "required": ["radius1", "height"]},
        )

    def execute(self, *, radius1: float, radius2: float = 0, height: float = 10, name: str = "Cone", **kwargs: Any) -> str:
        return _execute_operations([
            {"type": "new_doc"},
            {"type": "make_cone", "radius1": radius1, "radius2": radius2, "height": height, "name": name},
        ])


class FCBooleanTool(Tool):
    name = "fc_boolean"
    description = "Perform boolean operation: union (fuse), cut (difference), or intersection (common) of two objects"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "operation": {"type": "string", "enum": ["union", "cut", "intersection"], "description": "Boolean operation type"},
                "object1": {"type": "string", "description": "Name of first object"},
                "object2": {"type": "string", "description": "Name of second object"},
                "result_name": {"type": "string", "description": "Name for result (optional)"},
            }, "required": ["operation", "object1", "object2"]},
        )

    def execute(self, *, operation: str, object1: str, object2: str, result_name: str = "Result", **kwargs: Any) -> str:
        return _execute_operations([
            {"type": "boolean", "operation": operation, "object1": object1, "object2": object2, "result_name": result_name},
        ])


class FCMoveTool(Tool):
    name = "fc_move"
    description = "Move an object by a translation vector (dx, dy, dz in mm)"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "object": {"type": "string", "description": "Name of object to move"},
                "dx": {"type": "number", "description": "Translation along X (mm)"},
                "dy": {"type": "number", "description": "Translation along Y (mm)"},
                "dz": {"type": "number", "description": "Translation along Z (mm)"},
            }, "required": ["object", "dx", "dy", "dz"]},
        )

    def execute(self, *, object: str, dx: float, dy: float, dz: float, **kwargs: Any) -> str:
        return _execute_operations([{"type": "move", "object": object, "dx": dx, "dy": dy, "dz": dz}])


class FCRotateTool(Tool):
    name = "fc_rotate"
    description = "Rotate an object around an axis by a given angle (degrees)"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "object": {"type": "string", "description": "Name of object to rotate"},
                "axis": {"type": "string", "enum": ["x", "y", "z"], "description": "Rotation axis"},
                "angle": {"type": "number", "description": "Rotation angle in degrees"},
            }, "required": ["object", "axis", "angle"]},
        )

    def execute(self, *, object: str, axis: str, angle: float, **kwargs: Any) -> str:
        return _execute_operations([{"type": "rotate", "object": object, "axis": axis, "angle": angle}])


class FCCylinderWithHoleTool(Tool):
    name = "fc_cylinder_with_hole"
    description = "Create a cylinder with a concentric through-hole (like a bushing or ring)"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "outer_radius": {"type": "number", "description": "Outer radius in mm"},
                "inner_radius": {"type": "number", "description": "Inner hole radius in mm"},
                "height": {"type": "number", "description": "Height in mm"},
                "name": {"type": "string", "description": "Object name (optional)"},
            }, "required": ["outer_radius", "inner_radius", "height"]},
        )

    def execute(self, *, outer_radius: float, inner_radius: float, height: float, name: str = "CylinderWithHole", **kwargs: Any) -> str:
        return _execute_operations([
            {"type": "new_doc"},
            {"type": "cylinder_with_hole", "outer_radius": outer_radius, "inner_radius": inner_radius, "height": height, "name": name},
        ])


class FCPlateWithHolesTool(Tool):
    name = "fc_plate_with_holes"
    description = "Create a rectangular plate with evenly spaced mounting holes"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "length": {"type": "number", "description": "Plate length in mm"},
                "width": {"type": "number", "description": "Plate width in mm"},
                "thickness": {"type": "number", "description": "Plate thickness in mm"},
                "hole_radius": {"type": "number", "description": "Hole radius in mm"},
                "hole_count_x": {"type": "integer", "description": "Number of holes along X"},
                "hole_count_y": {"type": "integer", "description": "Number of holes along Y"},
                "margin": {"type": "number", "description": "Margin from edge in mm (optional)"},
                "name": {"type": "string", "description": "Object name (optional)"},
            }, "required": ["length", "width", "thickness", "hole_radius"]},
        )

    def execute(self, *, length: float, width: float, thickness: float, hole_radius: float,
                hole_count_x: int = 2, hole_count_y: int = 2, margin: float = 0, name: str = "PlateWithHoles",
                **kwargs: Any) -> str:
        return _execute_operations([
            {"type": "new_doc"},
            {"type": "plate_with_holes", "length": length, "width": width, "thickness": thickness,
             "hole_radius": hole_radius, "hole_count_x": hole_count_x, "hole_count_y": hole_count_y,
             "margin": margin, "name": name},
        ])


class FCFilletTool(Tool):
    name = "fc_fillet"
    description = "Add fillets (rounded edges) to all edges of an object"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "object": {"type": "string", "description": "Name of object to fillet"},
                "radius": {"type": "number", "description": "Fillet radius in mm"},
            }, "required": ["object", "radius"]},
        )

    def execute(self, *, object: str, radius: float, **kwargs: Any) -> str:
        return _execute_operations([{"type": "fillet", "object": object, "radius": radius}])


class FCChamferTool(Tool):
    name = "fc_chamfer"
    description = "Add chamfers to all edges of an object"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "object": {"type": "string", "description": "Name of object to chamfer"},
                "size": {"type": "number", "description": "Chamfer size in mm"},
            }, "required": ["object", "size"]},
        )

    def execute(self, *, object: str, size: float, **kwargs: Any) -> str:
        return _execute_operations([{"type": "chamfer", "object": object, "size": size}])


class FCSaveTool(Tool):
    name = "fc_save"
    description = "Save the active FreeCAD document (.FCStd format)"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "path": {"type": "string", "description": "Save file path (.FCStd)"},
            }, "required": ["path"]},
        )

    def execute(self, *, path: str, **kwargs: Any) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return _execute_operations([{"type": "save", "path": path}])


class FCExportSTLTool(Tool):
    name = "fc_export_stl"
    description = "Export objects in the active FreeCAD document as STL file"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "path": {"type": "string", "description": "Output STL file path"},
                "object": {"type": "string", "description": "Object name (optional, exports all if omitted)"},
                "tolerance": {"type": "number", "description": "Meshing tolerance mm (default: 0.1, lower = finer)"},
            }, "required": ["path"]},
        )

    def execute(self, *, path: str, object: str = "", tolerance: float = 0.1, **kwargs: Any) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return _execute_operations([{"type": "export_stl", "path": path, "object": object, "tolerance": tolerance}])


class FCExportSTEPTool(Tool):
    name = "fc_export_step"
    description = "Export objects as STEP file (high-precision CAD format)"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "path": {"type": "string", "description": "Output STEP file path"},
                "object": {"type": "string", "description": "Object name (optional)"},
            }, "required": ["path"]},
        )

    def execute(self, *, path: str, object: str = "", **kwargs: Any) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return _execute_operations([{"type": "export_step", "path": path, "object": object}])


class FCStatusTool(Tool):
    name = "fc_status"
    description = "Check FreeCAD connection status"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def execute(self, **kwargs: Any) -> str:
        fc_python = _find_freecad_python()
        if not fc_python:
            return "FreeCAD not found. Install with: winget install FreeCAD"
        return f"FreeCAD available: {fc_python}"


class FCGetObjectInfoTool(Tool):
    name = "fc_object_info"
    description = "Get detailed information about a FreeCAD object (dimensions, volume, area)"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "object": {"type": "string", "description": "Name of the object"},
            }, "required": ["object"]},
        )

    def execute(self, *, object: str, **kwargs: Any) -> str:
        return _execute_operations([{"type": "object_info", "object": object}])


class FCDeleteObjectTool(Tool):
    name = "fc_delete_object"
    description = "Delete an object from the active document"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "object": {"type": "string", "description": "Name of object to delete"},
            }, "required": ["object"]},
        )

    def execute(self, *, object: str, **kwargs: Any) -> str:
        return _execute_operations([{"type": "delete_object", "object": object}])


class FCScriptTool(Tool):
    name = "fc_script"
    description = "Execute a FreeCAD Python script for advanced/custom operations"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "script": {"type": "string", "description": "Python script to execute in FreeCAD context"},
            }, "required": ["script"]},
        )

    def execute(self, *, script: str, **kwargs: Any) -> str:
        return _execute_operations([{"type": "raw_script", "script": script}])


class FCBatchTool(Tool):
    """Execute multiple FreeCAD operations in a single subprocess call.

    This is the recommended way to perform multi-step modeling since each
    individual tool call creates a separate FreeCAD process with no state
    persistence. Using fc_batch, all operations run in one process.
    """

    name = "fc_batch"
    description = (
        "Execute multiple FreeCAD operations in one call. "
        "Use this for multi-step modeling tasks (create objects, boolean ops, export). "
        "Each operation is a JSON object with 'type' and parameters. "
        "Available types: new_doc, make_box, make_cylinder, make_sphere, make_cone, "
        "boolean, move, rotate, cylinder_with_hole, plate_with_holes, fillet, chamfer, "
        "save, export_stl, export_step, object_info, delete_object, volume_check, "
        "sweep (profile along path: springs, threads, pipe bends), "
        "loft (transition between profiles: cones, brackets, nozzles), "
        "polar_pattern (circular array: bolt hole circles, fan blades), "
        "linear_pattern (linear array: heat sink fins, gratings), "
        "mirror (symmetric features: right arm = mirror of left), "
        "shell (hollow solid by removing faces: boxes, containers), "
        "draft (taper faces for injection molding / 3D printing)."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "operations": {
                    "type": "array",
                    "description": "List of operations to execute sequentially in one FreeCAD session",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "description": "Operation type"},
                        },
                        "required": ["type"],
                    },
                },
            }, "required": ["operations"]},
        )

    def execute(self, *, operations: list[dict], **kwargs: Any) -> str:
        # Ensure paths in export/save operations have parent dirs
        for op in operations:
            if op.get("type") in ("export_stl", "export_step", "save") and "path" in op:
                Path(op["path"]).parent.mkdir(parents=True, exist_ok=True)
        return _execute_operations(operations)


class FCOpenGUITool(Tool):
    """Launch FreeCAD GUI to visualize a 3D model with optional camera preset.

    Uses a startup macro to set the camera view. The macro opens the document,
    applies the camera preset, and fits the view.
    """

    name = "fc_open_gui"
    description = (
        "Launch FreeCAD GUI to visualize a 3D model. "
        "Opens a .FCStd file and sets camera view. "
        "Use after fc_batch to save a model, then open it in GUI for visual verification."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to .FCStd file to open (optional, opens empty FreeCAD if omitted)",
                    },
                    "view": {
                        "type": "string",
                        "enum": list(VIEW_METHODS.keys()),
                        "description": "Camera view preset: isometric (default), front, top, right, left, back, bottom",
                    },
                    "fit_all": {
                        "type": "boolean",
                        "description": "Fit all objects in view (default: true)",
                    },
                    "wait_seconds": {
                        "type": "integer",
                        "description": "Seconds to wait for FreeCAD window to appear (default: 5)",
                    },
                },
                "required": [],
            },
        )

    def execute(
        self,
        *,
        file_path: str = "",
        view: str = "isometric",
        fit_all: bool = True,
        wait_seconds: int = 5,
        **kwargs: Any,
    ) -> str:
        fc_exe = _find_freecad_exe()
        if not fc_exe:
            return "Error: FreeCAD.exe not found. Install with: winget install FreeCAD"

        # Use process manager for graceful shutdown of existing FreeCAD
        _process_manager.kill_existing()
        time.sleep(1)

        # Clean up FreeCAD recovery cache to prevent Document Recovery dialog
        try:
            import shutil
            cache_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "Temp" / "FreeCAD"
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
        except Exception as _e:
            logger.debug("FreeCAD cache cleanup failed: %s", _e)

        # Disable welcome screen before launching GUI (headless config set)
        try:
            fc_python = _find_freecad_python()
            if fc_python:
                subprocess.run(
                    [fc_python, "-c",
                     "import FreeCAD;"
                     "p=FreeCAD.ParamGet('User parameter:BaseApp/Preferences/General');"
                     "p.SetBool('FirstRun',0);"
                     "p.SetBool('ShowWelcome',0);"
                     "p.SetString('AutoloadModule','Part');"
                     "s=FreeCAD.ParamGet('User parameter:BaseApp/Preferences/Mod/Start');"
                     "s.SetBool('ShowOnStartup',0)"],
                    capture_output=True, timeout=15,
                    encoding="utf-8", errors="replace",
                )
        except Exception as _e:
            logger.debug("FreeCAD version probe failed: %s", _e)

        # Also directly patch the user.cfg to ensure settings persist
        try:
            import xml.etree.ElementTree as ET
            cfg_path = Path(os.environ.get("APPDATA", "")) / "FreeCAD" / "v1-1" / "user.cfg"
            if cfg_path.exists():
                tree = ET.parse(str(cfg_path))
                root = tree.getroot()
                # Set FirstRun=0 in General group (suppresses First Run wizard)
                _xml_set_param(root, "General", "FirstRun", "0")
                # Set AutoloadModule to Part (skip Start page)
                _xml_set_param(root, "General", "AutoloadModule", "Part", tag="FCString")
                # Set ShowOnStartup=0 in Start group (suppresses Start page)
                _xml_set_param(root, "Start", "ShowOnStartup", "0")
                tree.write(str(cfg_path), encoding="UTF-8", xml_declaration=True)
        except Exception as _e:
            logger.debug("FreeCAD config tweak failed: %s", _e)

        view_method = VIEW_METHODS.get(view, "viewIsometric")
        cmd = [fc_exe]

        if file_path:
            if not Path(file_path).exists():
                return f"Error: File not found: {file_path}"

            # Build startup macro: open doc + set camera + dismiss dialogs
            macro_lines = [
                "import FreeCAD",
                "import FreeCADGui",
                "import time",
                "",
                "def _dismiss_dialogs():",
                "    \"\"\"Dismiss Welcome, Setup, Recovery, and Unsaved dialogs.\"\"\"",
                "    try:",
                "        from PySide2 import QtWidgets, QtCore",
                "        app = QtWidgets.QApplication.instance()",
                "        if app:",
                "            for w in app.topLevelWidgets():",
                "                try:",
                "                    wclass = w.metaObject().className() if w.metaObject() else ''",
                "                    title = w.windowTitle().lower() if w.windowTitle() else ''",
                "                    # Close QDialog-based popups (setup wizard, recovery)",
                "                    if 'Dialog' in wclass:",
                "                        print(f'Dismissing dialog: {w.windowTitle()}')",
                "                        # For 'Unsaved Document' dialog, click Discard button",
                "                        if 'unsaved' in title or 'save' in title:",
                "                            for btn in w.findChildren(QtWidgets.QPushButton):",
                "                                btxt = btn.text().lower()",
                "                                if 'discard' in btxt or 'don\\'t save' in btxt or 'no' in btxt:",
                "                                    btn.click()",
                "                                    break",
                "                            else:",
                "                                w.close()",
                "                        else:",
                "                            w.close()",
                "                        continue",
                "                except Exception:",
                "                    pass",
                "            # Also close the Start page tab if present",
                "            try:",
                "                mw = app.activeWindow()",
                "                if mw:",
                "                    for tb in mw.findChildren(QtWidgets.QTabBar):",
                "                        for i in range(tb.count()):",
                "                            if 'start' in tb.tabText(i).lower():",
                "                                tb.removeTab(i)",
                "                                break",
                "            except Exception:",
                "                pass",
                "    except Exception:",
                "        pass",
                "",
                "def _set_camera():",
                "    try:",
                "        _dismiss_dialogs()",
                "        v = FreeCADGui.activeDocument().activeView()",
                f"        v.{view_method}()",
                "    except Exception:",
                "        pass",
                "",
                "# Disable welcome screen permanently",
                "try:",
                "    params = FreeCAD.ParamGet('User parameter:BaseApp/Preferences/General')",
                "    params.SetBool('FirstRun', False)",
                "    params.SetBool('ShowWelcome', False)",
                "    start_params = FreeCAD.ParamGet('User parameter:BaseApp/Preferences/Mod/Start')",
                "    start_params.SetBool('ShowOnStartup', False)",
                "except Exception:",
                "    pass",
                "",
                "# Force switch to Part workbench to close Start page",
                "try:",
                "    FreeCADGui.activateWorkbench('Part')",
                "except Exception:",
                "    pass",
                "",
                f'doc = FreeCAD.openDocument({json.dumps(str(file_path))})',
                "time.sleep(0.5)",
                "",
            ]
            macro_lines.extend([
                "def _ensure_visible():",
                "    \"\"\"Make all objects visible with proper display mode.\"\"\"",
                "    try:",
                "        _doc = FreeCAD.ActiveDocument",
                "        if not _doc:",
                "            return",
                "        for _o in _doc.Objects:",
                "            try:",
                "                if hasattr(_o, 'ViewObject') and _o.ViewObject is not None:",
                "                    _o.ViewObject.Visibility = True",
                "                    if hasattr(_o.ViewObject, 'DisplayMode'):",
                "                        _o.ViewObject.DisplayMode = 'Flat Lines'",
                "            except Exception:",
                "                pass",
                "        _doc.recompute()",
                "        FreeCADGui.SendMsgToActiveView('ViewFit')",
                "    except Exception:",
                "        pass",
                "",
                "# Make all objects visible immediately",
                "for _o in doc.Objects:",
                "    try:",
                "        if hasattr(_o, 'ViewObject') and _o.ViewObject is not None:",
                "            _o.ViewObject.Visibility = True",
                "    except Exception:",
                "        pass",
                "doc.recompute()",
                "",
            ])
            if fit_all:
                macro_lines.extend([
                    "try:",
                    "    FreeCADGui.SendMsgToActiveView('ViewFit')",
                    "except Exception:",
                    "    pass",
                    "",
                ])
            macro_lines.extend([
                "print(f'Opened: {doc.Name}')",
                "",
                "# Delayed calls: dismiss dialogs early, then ensure visibility, then set camera",
                "from PySide2 import QtCore",
                "QtCore.QTimer.singleShot(200, _dismiss_dialogs)",
                "QtCore.QTimer.singleShot(800, _dismiss_dialogs)",
                "QtCore.QTimer.singleShot(1000, _ensure_visible)",
                "QtCore.QTimer.singleShot(2500, _set_camera)",
            ])

            macro_content = "\n".join(macro_lines)
            macro_dir = Path(tempfile.gettempdir()) / "lang3d_macros"
            macro_dir.mkdir(exist_ok=True)
            macro_path = macro_dir / f"open_gui_{int(time.time())}.py"
            macro_path.write_text(macro_content, encoding="utf-8")

            cmd.append(str(macro_path))

        proc = _process_manager.launch_gui(cmd)

        # Wait for window to appear
        time.sleep(wait_seconds)

        # Verify FreeCAD window is visible
        from ..tools.screen import _find_windows_by_title

        windows = _find_windows_by_title("FreeCAD")
        info_parts = [f"FreeCAD GUI launched (PID: {proc.pid})"]

        if windows:
            hwnd, title = windows[0]
            info_parts.append(f"Window: '{title}' (hwnd: {hwnd})")
        else:
            info_parts.append(
                "Window not yet detected. Use list_windows or window_capture to check."
            )

        if file_path:
            info_parts.append(f"Camera: {view}" + (" (fit all)" if fit_all else ""))

        return "\n".join(info_parts)


class FCCloseGUITool(Tool):
    """Close the FreeCAD GUI application."""

    name = "fc_close_gui"
    description = "Close the FreeCAD GUI application gracefully"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def execute(self, **kwargs: Any) -> str:
        from ..tools.screen import _find_windows_by_title

        windows = _find_windows_by_title("FreeCAD")
        if not windows:
            return "No FreeCAD window found (already closed or not running)"

        count = len(windows)

        # Use process manager for graceful shutdown
        _process_manager.kill_existing()

        time.sleep(1)
        remaining = _find_windows_by_title("FreeCAD")
        if remaining:
            return f"Closed {count} FreeCAD window(s) (some still visible)"
        return f"FreeCAD closed: {count} window(s)"


class FCSetCameraTool(Tool):
    """Change camera view in FreeCAD by reopening with a new startup macro.

    Since we cannot send commands to a running FreeCAD instance from outside,
    this tool closes the current GUI and relaunches with the desired camera preset.
    Requires the file_path of the document to reopen.
    """

    name = "fc_set_camera"
    description = (
        "Change the camera view in FreeCAD. "
        "Closes and reopens FreeCAD with the new view preset. "
        "You must provide the file_path of the .FCStd document."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to .FCStd file to reopen with new camera",
                    },
                    "view": {
                        "type": "string",
                        "enum": list(VIEW_METHODS.keys()),
                        "description": "Camera view preset",
                    },
                    "fit_all": {
                        "type": "boolean",
                        "description": "Fit all objects in view (default: true)",
                    },
                },
                "required": ["file_path", "view"],
            },
        )

    def execute(
        self, *, file_path: str, view: str = "isometric", fit_all: bool = True, **kwargs: Any
    ) -> str:
        if not Path(file_path).exists():
            return f"Error: File not found: {file_path}"

        if view not in VIEW_METHODS:
            return f"Error: Unknown view '{view}'. Available: {', '.join(VIEW_METHODS.keys())}"

        # Close existing FreeCAD
        close_tool = FCCloseGUITool()
        close_tool.execute()

        time.sleep(2)

        # Reopen with new camera
        open_tool = FCOpenGUITool()
        result = open_tool.execute(file_path=file_path, view=view, fit_all=fit_all)
        return f"[Camera changed to '{view}']\n{result}"


class RenderAssemblyTool(Tool):
    """Render a full assembly in FreeCAD with subsystem-colored parts.

    Generates a single FCStd file + STL with all parts positioned according
    to solved assembly positions. Supports exploded view and multi-view output.
    """

    name = "render_assembly"
    description = (
        "Render a full assembly in FreeCAD with all parts positioned and colored by subsystem. "
        "Generates FCStd + STL files. Supports exploded view. "
        "Requires assembly_json with parts and positions."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "assembly_json": {
                        "type": "string",
                        "description": "JSON string with 'parts' (list of part dicts with name, dimensions) and 'positions' (dict of name→position)",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output file path prefix (without extension). Generates .FCStd and .stl",
                    },
                    "exploded": {
                        "type": "boolean",
                        "description": "Generate exploded view (parts spread out from center)",
                    },
                    "explode_factor": {
                        "type": "number",
                        "description": "Explode distance multiplier (default: 1.5)",
                    },
                },
                "required": ["assembly_json", "output_path"],
            },
        )

    def execute(self, *, assembly_json: str, output_path: str, exploded: bool = False,
                explode_factor: float = 1.5, **kwargs: Any) -> str:
        try:
            data = json.loads(assembly_json)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON: {e}"

        parts = data.get("parts", [])
        positions = data.get("positions", {})

        if not parts or not positions:
            return "Error: assembly_json must contain 'parts' and 'positions'"

        # Prepare part info list
        assembly_parts = []
        for p in parts:
            pname = p.get("name", f"part_{len(assembly_parts)}")
            dims = p.get("dimensions", {})
            assembly_parts.append({
                "name": pname,
                "shape_type": "cylinder" if ("diameter" in dims or "outer_diameter" in dims) else "box",
                "dimensions": dims,
                "subsystem": _subsystem_for_part(pname),
            })

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        script = build_assembly_script(
            assembly_parts=assembly_parts,
            positions=positions,
            output_path=output_path,
            exploded=exploded,
            explode_factor=explode_factor,
        )

        try:
            result = _run_freecad_script(script, timeout=300)
            fcstd_path = Path(f"{output_path}.FCStd")
            stl_path = Path(f"{output_path}.stl")
            info_lines = [
                "Assembly rendered successfully",
                f"Parts: {len(assembly_parts)}",
                f"FCStd: {fcstd_path} ({'exists' if fcstd_path.exists() else 'not found'})",
                f"STL: {stl_path} ({'exists' if stl_path.exists() else 'not found'})",
            ]
            if exploded:
                info_lines.append(f"Exploded view (factor={explode_factor})")
            return "\n".join(info_lines)
        except RuntimeError as e:
            return f"Error rendering assembly: {e}"


def register_freecad_tools(registry: Any) -> None:
    """Register all FreeCAD tools."""
    tools = [
        FCNewDocTool(),
        FCMakeBoxTool(),
        FCMakeCylinderTool(),
        FCMakeSphereTool(),
        FCMakeConeTool(),
        FCBooleanTool(),
        FCMoveTool(),
        FCRotateTool(),
        FCCylinderWithHoleTool(),
        FCPlateWithHolesTool(),
        FCFilletTool(),
        FCChamferTool(),
        FCSaveTool(),
        FCExportSTLTool(),
        FCExportSTEPTool(),
        FCStatusTool(),
        FCGetObjectInfoTool(),
        FCDeleteObjectTool(),
        FCScriptTool(),
        FCBatchTool(),
        FCOpenGUITool(),
        FCCloseGUITool(),
        FCSetCameraTool(),
        RenderAssemblyTool(),
    ]
    for tool in tools:
        registry.register(tool)
