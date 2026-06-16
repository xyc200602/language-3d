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
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

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


def _build_script(operations: list[dict]) -> str:
    """Build a FreeCAD Python script from a list of operations.

    Each operation is a dict with 'type' and parameters.
    """
    lines = [
        "import FreeCAD",
        "import FreeCAD as App",
        "import Part",
        "import Mesh",
        "import json",
        "import sys",
        "",
    ]

    for op in operations:
        op_type = op["type"]
        lines.append(f"# Operation: {op_type}")

        if op_type == "new_doc":
            name = _safe_name(op.get("name", "Unnamed"))
            lines.append(f'doc = FreeCAD.newDocument("{name}")')

        elif op_type == "make_box":
            l = float(op.get("length", 0))
            w = float(op.get("width", 0))
            h = float(op.get("height", 0))
            name = _safe_name(op.get("name", "Box"))
            lines.append(f'box = Part.makeBox({l}, {w}, {h})')
            lines.append(f'obj = doc.addObject("Part::Feature", "{name}")')
            lines.append("obj.Shape = box")
            lines.append("doc.recompute()")

        elif op_type == "make_cylinder":
            r = float(op.get("radius", 0))
            h = float(op.get("height", 0))
            name = _safe_name(op.get("name", "Cylinder"))
            lines.append(f'cyl = Part.makeCylinder({r}, {h})')
            lines.append(f'obj = doc.addObject("Part::Feature", "{name}")')
            lines.append("obj.Shape = cyl")
            lines.append("doc.recompute()")

        elif op_type == "make_sphere":
            r = float(op["radius"])
            name = _safe_name(op.get("name", "Sphere"))
            lines.append(f'sphere = Part.makeSphere({r})')
            lines.append(f'obj = doc.addObject("Part::Feature", "{name}")')
            lines.append("obj.Shape = sphere")
            lines.append("doc.recompute()")

        elif op_type == "make_cone":
            r1 = float(op["radius1"])
            r2 = float(op.get("radius2", 0))
            h = float(op["height"])
            name = _safe_name(op.get("name", "Cone"))
            lines.append(f'cone = Part.makeCone({r1}, {r2}, {h})')
            lines.append(f'obj = doc.addObject("Part::Feature", "{name}")')
            lines.append("obj.Shape = cone")
            lines.append("doc.recompute()")

        elif op_type == "boolean":
            operation = op.get("operation", "union")  # union, cut, intersection
            obj1_name = _safe_name(op.get("object1", "Box"))
            obj2_name = _safe_name(op.get("object2", "Box"))
            result_name = _safe_name(op.get("result_name", "Result"))
            bool_map = {"union": "fuse", "cut": "cut", "intersection": "common"}
            method = bool_map.get(operation, "fuse")
            lines.append(f'o1 = doc.getObject("{obj1_name}")')
            lines.append(f'o2 = doc.getObject("{obj2_name}")')
            lines.append(f'result_shape = o1.Shape.{method}(o2.Shape)')
            lines.append(f'result = doc.addObject("Part::Feature", "{result_name}")')
            lines.append("result.Shape = result_shape")
            lines.append("doc.recompute()")

        elif op_type == "move":
            obj_name = _safe_name(op.get("object", "Box"))
            dx = float(op.get("dx", 0))
            dy = float(op.get("dy", 0))
            dz = float(op.get("dz", 0))
            lines.append(f'_obj = doc.getObject("{obj_name}")')
            lines.append("_moved = _obj.Shape.copy()")
            lines.append(f"_moved.translate(FreeCAD.Vector({dx}, {dy}, {dz}))")
            lines.append("_obj.Shape = _moved")
            lines.append("doc.recompute()")

        elif op_type == "cylinder_with_hole":
            orad = float(op["outer_radius"])
            irad = float(op["inner_radius"])
            h = float(op["height"])
            name = _safe_name(op.get("name", "CylinderWithHole"))
            lines.append(f'_outer = Part.makeCylinder({orad}, {h})')
            lines.append(f'_inner = Part.makeCylinder({irad}, {h})')
            lines.append("_result = _outer.cut(_inner)")
            lines.append(f'obj = doc.addObject("Part::Feature", "{name}")')
            lines.append("obj.Shape = _result")
            lines.append("doc.recompute()")

        elif op_type == "plate_with_holes":
            length = float(op["length"])
            width = float(op["width"])
            thickness = float(op["thickness"])
            hole_radius = float(op["hole_radius"])
            nx = int(float(op.get("hole_count_x", 2)))
            ny = int(float(op.get("hole_count_y", 2)))
            margin = float(op.get("margin", 0))
            name = _safe_name(op.get("name", "PlateWithHoles"))
            if margin == 0:
                margin = min(length, width) * 0.1
            sx = (length - 2 * margin) / max(nx - 1, 1) if nx > 1 else 0
            sy = (width - 2 * margin) / max(ny - 1, 1) if ny > 1 else 0
            lines.append(f'_plate = Part.makeBox({length}, {width}, {thickness})')
            lines.append(f'_hole = Part.makeCylinder({hole_radius}, {thickness})')
            lines.append(f"for ix in range({nx}):")
            lines.append(f"    for iy in range({ny}):")
            lines.append(f"        x = {margin} + ix * {sx}")
            lines.append(f"        y = {margin} + iy * {sy}")
            lines.append("        h = _hole.copy()")
            lines.append("        h.translate(FreeCAD.Vector(x, y, 0))")
            lines.append("        _plate = _plate.cut(h)")
            lines.append(f'obj = doc.addObject("Part::Feature", "{name}")')
            lines.append("obj.Shape = _plate")
            lines.append("doc.recompute()")

        elif op_type == "fillet":
            obj_name = _safe_name(op["object"])
            radius = float(op["radius"])
            lines.append(f'_obj = doc.getObject("{obj_name}")')
            lines.append(f"_fillet = _obj.Shape.makeFillet({radius}, _obj.Shape.Edges)")
            lines.append("_obj.Shape = _fillet")
            lines.append("doc.recompute()")

        elif op_type == "chamfer":
            obj_name = _safe_name(op["object"])
            size = float(op["size"])
            lines.append(f'_obj = doc.getObject("{obj_name}")')
            lines.append(f"_chamfer = _obj.Shape.makeChamfer({size}, _obj.Shape.Edges)")
            lines.append("_obj.Shape = _chamfer")
            lines.append("doc.recompute()")

        elif op_type == "save":
            path = _safe_path(op["path"])
            # Make all objects visible before saving so they show in GUI
            lines.append("for _o in doc.Objects:")
            lines.append("    try:")
            lines.append("        if hasattr(_o, 'ViewObject') and _o.ViewObject is not None:")
            lines.append("            _o.ViewObject.Visibility = True")
            lines.append("    except Exception:")
            lines.append("        pass")
            lines.append("doc.recompute()")
            lines.append(f'doc.saveAs({json.dumps(str(path))})')

        elif op_type == "export_stl":
            path = _safe_path(op["path"])
            obj_name = _safe_name(op["object"]) if op.get("object") else ""
            tolerance = float(op.get("tolerance", 0.1))
            # Use Mesh.export which is more robust than Mesh.Mesh(shape.tessellate())
            if obj_name:
                lines.append(f'_export_list = [doc.getObject("{obj_name}")]')
            else:
                lines.append("_export_list = [o for o in doc.Objects if hasattr(o, 'Shape')]")
            lines.append(f'Mesh.export(_export_list, {json.dumps(str(path))})')
            lines.append("import os")
            lines.append(f'_stl_path = {json.dumps(str(path))}')
            lines.append('print(f"STL exported: {os.path.getsize(_stl_path):,} bytes")')

        elif op_type == "export_step":
            path = _safe_path(op["path"])
            obj_name = _safe_name(op["object"]) if op.get("object") else ""
            if obj_name:
                lines.append(f'_obj = doc.getObject("{obj_name}")')
                lines.append(f'_obj.Shape.exportStep({json.dumps(str(path))})')
            else:
                lines.append("_shapes = [o.Shape for o in doc.Objects if hasattr(o, 'Shape')]")
                lines.append("if not _shapes:")
                lines.append("    raise RuntimeError('No shapes found to export')")
                lines.append("_compound = _shapes[0]")
                lines.append("for s in _shapes[1:]:")
                lines.append("    _compound = _compound.fuse(s)")
                lines.append(f'_compound.exportStep({json.dumps(str(path))})')

        elif op_type == "status":
            lines.append("if FreeCAD.ActiveDocument:")
            lines.append("    print(f'Document: {FreeCAD.ActiveDocument.Name}')")
            lines.append("    for o in FreeCAD.ActiveDocument.Objects:")
            lines.append("        if hasattr(o, 'Shape'):")
            lines.append("            bb = o.Shape.BoundBox")
            lines.append("            print(f'  {o.Name}: {bb.XLength:.1f}x{bb.YLength:.1f}x{bb.ZLength:.1f}mm')")
            lines.append("        else:")
            lines.append(f"            print(f'  {{o.Name}} ({{o.TypeId}})')")
            lines.append("else:")
            lines.append("    print('No active document')")

        elif op_type == "object_info":
            obj_name = _safe_name(op["object"])
            lines.append(f'_obj = doc.getObject("{obj_name}")')
            lines.append("if _obj and hasattr(_obj, 'Shape'):")
            lines.append("    s = _obj.Shape")
            lines.append("    bb = s.BoundBox")
            lines.append("    print(f'Volume: {s.Volume:.2f} mm3')")
            lines.append("    print(f'Area: {s.Area:.2f} mm2')")
            lines.append("    print(f'Dims: {bb.XLength:.2f}x{bb.YLength:.2f}x{bb.ZLength:.2f}mm')")
            lines.append("    print(f'Edges: {len(s.Edges)}, Faces: {len(s.Faces)}, Vertices: {len(s.Vertexes)}')")

        elif op_type == "delete_object":
            obj_name = _safe_name(op["object"])
            lines.append(f'doc.removeObject("{obj_name}")')
            lines.append("doc.recompute()")

        elif op_type == "volume_check":
            # Lightweight verification: load file and check all object
            # dimensions without opening GUI or using VLM.
            path = _safe_path(op.get("path", "")) if op.get("path", "") else ""
            checks = op.get("checks", {})
            # Expected dims, volume range, etc.
            expected_dims = checks.get("dimensions", {})
            min_volume = float(checks.get("min_volume", 0))
            max_volume = float(checks.get("max_volume", float("inf")))
            if path:
                lines.append(f'_vc_doc = FreeCAD.openDocument({json.dumps(str(path))})')
                lines.append("if not _vc_doc:")
                lines.append('    print("VOLUME_CHECK:error: Failed to open document")')
                lines.append("else:")
            else:
                lines.append("_vc_doc = doc")
            lines.append("_vc_results = []")
            lines.append("_vc_pass = True")
            lines.append("for _vc_o in _vc_doc.Objects:")
            lines.append("    if hasattr(_vc_o, 'Shape') and _vc_o.Shape is not None:")
            lines.append("        try:")
            lines.append("            _vc_s = _vc_o.Shape")
            lines.append("            _vc_bb = _vc_s.BoundBox")
            lines.append("            _vc_info = {")
            lines.append('                "name": _vc_o.Name,')
            lines.append('                "label": _vc_o.Label,')
            lines.append('                "volume": round(_vc_s.Volume, 2),')
            lines.append('                "area": round(_vc_s.Area, 2),')
            lines.append('                "dims": [round(_vc_bb.XLength, 2), round(_vc_bb.YLength, 2), round(_vc_bb.ZLength, 2)],')
            lines.append('                "center": [round(_vc_bb.Center.x, 2), round(_vc_bb.Center.y, 2), round(_vc_bb.Center.z, 2)],')
            lines.append('                "edges": len(_vc_s.Edges), "faces": len(_vc_s.Faces),')
            lines.append("            }")
            if min_volume > 0:
                lines.append(f"            if _vc_s.Volume < {min_volume}:")
                lines.append(f'                _vc_info["volume_warning"] = "below minimum {min_volume}"')
                lines.append("                _vc_pass = False")
            if max_volume < float("inf"):
                lines.append(f"            if _vc_s.Volume > {max_volume}:")
                lines.append(f'                _vc_info["volume_warning"] = "above maximum {max_volume}"')
                lines.append("                _vc_pass = False")
            if expected_dims:
                for dim_name, dim_val in expected_dims.items():
                    dim_val = float(dim_val)
                    dim_map = {"length": "XLength", "width": "YLength", "height": "ZLength",
                               "x": "XLength", "y": "YLength", "z": "ZLength"}
                    attr = dim_map.get(dim_name.lower(), dim_name)
                    tol = float(checks.get("tolerance_mm", 1.0))
                    lines.append(f'            _vc_actual = getattr(_vc_bb, "{attr}", 0)')
                    lines.append(f"            if abs(_vc_actual - {dim_val}) > {tol}:")
                    lines.append(f'                _vc_info["dim_warning"] = "{dim_name}: expected {dim_val}, got " + str(round(_vc_actual, 2))')
                    lines.append("                _vc_pass = False")
            lines.append("            _vc_results.append(_vc_info)")
            lines.append("        except Exception:")
            lines.append("            pass")
            lines.append('import json as _vc_json')
            lines.append("_vc_output = {'pass': _vc_pass, 'objects': _vc_results, 'total_objects': len(_vc_results)}")
            lines.append('print("VOLUME_CHECK_JSON:" + _vc_json.dumps(_vc_output))')
            if path:
                lines.append("FreeCAD.closeDocument(_vc_doc.Name)")

        elif op_type == "rotate":
            obj_name = _safe_name(op.get("object", "Box"))
            axis = op.get("axis", "z")  # x, y, z
            angle = float(op.get("angle", 0))
            lines.append("import math")
            lines.append(f'_obj = doc.getObject("{obj_name}")')
            lines.append(f'_axis_map = {{"x": FreeCAD.Vector(1,0,0), "y": FreeCAD.Vector(0,1,0), "z": FreeCAD.Vector(0,0,1)}}')
            lines.append(f'_rot_axis = _axis_map["{axis}"]')
            lines.append("_rotated = _obj.Shape.copy()")
            lines.append("_center = _rotated.BoundBox.Center")
            lines.append("_moved = _rotated.copy()")
            lines.append(f"_moved.rotate(_center, _rot_axis, {angle})")
            lines.append("_obj.Shape = _moved")
            lines.append("doc.recompute()")

        elif op_type == "raw_script":
            _validate_raw_script(op["script"])
            lines.append(op["script"])

        elif op_type == "compute_mass":
            # Compute mass properties for all solid objects in the document
            obj_name = _safe_name(op["object"]) if op.get("object") else ""
            density = float(op.get("density", 1240))  # kg/m³, default PLA
            path = _safe_path(op.get("path", "")) if op.get("path", "") else ""
            if path:
                lines.append(f'_cm_doc = FreeCAD.openDocument({json.dumps(str(path))})')
            else:
                lines.append("_cm_doc = doc")
            lines.append("_cm_results = []")
            if obj_name:
                lines.append(f'_cm_objs = [doc.getObject("{obj_name}")]')
            else:
                lines.append("_cm_objs = [o for o in _cm_doc.Objects if hasattr(o, 'Shape') and o.Shape is not None and o.Shape.Solids]")
            lines.append("for _cm_o in _cm_objs:")
            lines.append("    try:")
            lines.append("        _cm_s = _cm_o.Shape")
            lines.append("        _cm_vol_mm3 = _cm_s.Volume  # mm³")
            lines.append("        _cm_vol_m3 = _cm_vol_mm3 * 1e-9")
            lines.append(f"        _cm_mass = _cm_vol_m3 * {density}  # kg")
            lines.append("        _cm_bb = _cm_s.BoundBox")
            lines.append("        _cm_cx = round(_cm_bb.Center.x, 2)")
            lines.append("        _cm_cy = round(_cm_bb.Center.y, 2)")
            lines.append("        _cm_cz = round(_cm_bb.Center.z, 2)")
            lines.append("        _cm_info = {")
            lines.append('            "name": _cm_o.Name,')
            lines.append('            "label": _cm_o.Label,')
            lines.append('            "volume_mm3": round(_cm_vol_mm3, 2),')
            lines.append('            "mass_kg": round(_cm_mass, 6),')
            lines.append(f'            "density_kg_m3": {density},')
            lines.append('            "center_mm": [_cm_cx, _cm_cy, _cm_cz],')
            lines.append('            "dims_mm": [round(_cm_bb.XLength, 2), round(_cm_bb.YLength, 2), round(_cm_bb.ZLength, 2)],')
            lines.append("        }")
            lines.append("        _cm_results.append(_cm_info)")
            lines.append("    except Exception:")
            lines.append("        pass")
            lines.append("import json as _cm_json")
            lines.append("_cm_output = {'objects': _cm_results, 'total_objects': len(_cm_results)}")
            lines.append("_cm_total_mass = sum(r['mass_kg'] for r in _cm_results)")
            lines.append("_cm_output['total_mass_kg'] = round(_cm_total_mass, 6)")
            lines.append('print("MASS_CHECK_JSON:" + _cm_json.dumps(_cm_output))')
            if path:
                lines.append("FreeCAD.closeDocument(_cm_doc.Name)")

        elif op_type == "sweep":
            # Sweep a profile along a path (e.g. spring, thread, pipe bend)
            name = _safe_name(op.get("name", "Sweep"))
            profile_type = op.get("profile", "circle")  # circle, rectangle, custom
            profile_radius = float(op.get("profile_radius", 2.0))
            profile_width = float(op.get("profile_width", 4.0))
            profile_height = float(op.get("profile_height", 4.0))
            path_type = op.get("path_type", "helix")  # helix, circle, line, custom
            # Helix params
            pitch = float(op.get("pitch", 5.0))
            height = float(op.get("height", 20.0))
            helix_radius = float(op.get("helix_radius", 10.0))
            turns = float(op.get("turns", 0))
            # Circle path params
            path_radius = float(op.get("path_radius", 15.0))
            # Line path params
            line_length = float(op.get("line_length", 30.0))
            line_dir = op.get("line_direction", [1, 0, 0])
            solid = op.get("solid", True)  # True = solid, False = hollow pipe
            frenet = op.get("frenet", True)
            # Build profile wire
            lines.append("import Part")
            lines.append("import FreeCAD")
            if profile_type == "circle":
                lines.append(f"_prof = Part.Wire(Part.makeCircle({profile_radius}))")
            elif profile_type == "rectangle":
                lines.append(f"_prof_hw = {profile_width} / 2")
                lines.append(f"_prof_hh = {profile_height} / 2")
                lines.append("_prof = Part.Wire(Part.makePolygon([")
                lines.append("    FreeCAD.Vector(-_prof_hw, -_prof_hh, 0),")
                lines.append("    FreeCAD.Vector(_prof_hw, -_prof_hh, 0),")
                lines.append("    FreeCAD.Vector(_prof_hw, _prof_hh, 0),")
                lines.append("    FreeCAD.Vector(-_prof_hw, _prof_hh, 0),")
                lines.append("    FreeCAD.Vector(-_prof_hw, -_prof_hh, 0),")
                lines.append("]))")
            else:
                # custom: user provides raw script to create _prof wire
                custom_script = op.get("custom_profile_script", "_prof = Part.Wire(Part.makeCircle(2.0))")
                _validate_raw_script(custom_script)
                lines.append(custom_script)
            # Build path wire
            if path_type == "helix":
                if turns > 0:
                    lines.append(f"_path = Part.Wire(Part.makeHelix({pitch}, {height}, {helix_radius}, 0, {turns}))")
                else:
                    lines.append(f"_path = Part.Wire(Part.makeHelix({pitch}, {height}, {helix_radius}))")
                # Move profile to start of helix path
                lines.append(f"_prof.translate(FreeCAD.Vector({helix_radius}, 0, 0))")
            elif path_type == "circle":
                lines.append(f"_path = Part.Wire(Part.makeCircle({path_radius}))")
                lines.append(f"_prof.translate(FreeCAD.Vector({path_radius}, 0, 0))")
            elif path_type == "line":
                dx, dy, dz = float(line_dir[0]), float(line_dir[1]), float(line_dir[2])
                lines.append(f"_path = Part.Wire(Part.makeLine(FreeCAD.Vector(0,0,0), FreeCAD.Vector({dx * line_length}, {dy * line_length}, {dz * line_length})))")
            else:
                custom_path = op.get("custom_path_script", "_path = Part.Wire(Part.makeLine(FreeCAD.Vector(0,0,0), FreeCAD.Vector(30,0,0)))")
                _validate_raw_script(custom_path)
                lines.append(custom_path)
            # Perform sweep
            lines.append(f"_sweep = _prof.makePipeShell([_path], {solid}, {frenet})")
            lines.append(f'obj = doc.addObject("Part::Feature", "{name}")')
            lines.append("obj.Shape = _sweep")
            lines.append("doc.recompute()")

        elif op_type == "loft":
            # Loft between multiple profiles (e.g. transitions, brackets)
            name = _safe_name(op.get("name", "Loft"))
            profiles = op.get("profiles", [])
            solid = op.get("solid", True)
            ruled = op.get("ruled", False)
            if profiles:
                # Build profiles from specifications
                # Each profile: {type: "circle"|"rectangle", radius, center:[x,y,z], ...}
                for i, prof in enumerate(profiles):
                    ptype = prof.get("type", "circle")
                    _c = prof.get("center", [0, 0, 0])
                    cx, cy, cz = float(_c[0]), float(_c[1]), float(_c[2])
                    if ptype == "circle":
                        r = float(prof.get("radius", 5.0))
                        lines.append(f"_loft_p{i} = Part.Wire(Part.makeCircle({r}, FreeCAD.Vector({cx},{cy},{cz})))")
                    elif ptype == "rectangle":
                        w = float(prof.get("width", 10.0))
                        h = float(prof.get("height", 10.0))
                        hw, hh = w / 2, h / 2
                        lines.append(f"_loft_p{i} = Part.Wire(Part.makePolygon([")
                        lines.append(f"    FreeCAD.Vector({cx - hw},{cy - hh},{cz}),")
                        lines.append(f"    FreeCAD.Vector({cx + hw},{cy - hh},{cz}),")
                        lines.append(f"    FreeCAD.Vector({cx + hw},{cy + hh},{cz}),")
                        lines.append(f"    FreeCAD.Vector({cx - hw},{cy + hh},{cz}),")
                        lines.append(f"    FreeCAD.Vector({cx - hw},{cy - hh},{cz}),")
                        lines.append("]))")
                    elif ptype == "polygon":
                        r = float(prof.get("radius", 5.0))
                        sides = int(float(prof.get("sides", 6)))
                        import math as _math
                        pts = []
                        for s in range(sides + 1):
                            ang = 2 * 3.141592653589793 * s / sides
                            pts.append(f"FreeCAD.Vector({cx + r * round(_math.cos(ang), 6)},{cy + r * round(_math.sin(ang), 6)},{cz})")
                        lines.append(f"_loft_p{i} = Part.Wire(Part.makePolygon([{', '.join(pts)}]))")
                    else:
                        # ellipse
                        r1 = float(prof.get("radius1", 10.0))
                        r2 = float(prof.get("radius2", 5.0))
                        lines.append(f"_loft_p{i} = Part.Wire(Part.makeEllipse({r1}, {r2}, FreeCAD.Vector({cx},{cy},{cz})))")
                profile_list = ", ".join(f"_loft_p{i}" for i in range(len(profiles)))
                lines.append(f"_loft_shapes = [{profile_list}]")
            else:
                # Two simple circles at z=0 and z=height
                r1 = float(op.get("radius1", 10.0))
                r2 = float(op.get("radius2", 5.0))
                h = float(op.get("height", 20.0))
                lines.append(f"_loft_p0 = Part.Wire(Part.makeCircle({r1}, FreeCAD.Vector(0,0,0)))")
                lines.append(f"_loft_p1 = Part.Wire(Part.makeCircle({r2}, FreeCAD.Vector(0,0,{h})))")
                lines.append("_loft_shapes = [_loft_p0, _loft_p1]")
            lines.append(f"_loft = Part.makeLoft(_loft_shapes, {solid}, {ruled}, False)")
            lines.append(f'obj = doc.addObject("Part::Feature", "{name}")')
            lines.append("obj.Shape = _loft")
            lines.append("doc.recompute()")

        elif op_type == "polar_pattern":
            # Circular array: replicate a feature around an axis
            obj_name = _safe_name(op["object"]) if op.get("object") else ""
            count = int(op.get("count", 6))
            angle = float(op.get("angle", 360.0))
            axis_vec = op.get("axis", [0, 0, 1])
            center = op.get("center", [0, 0, 0])
            result_name = _safe_name(op.get("result_name", "PolarPattern"))
            if obj_name:
                lines.append(f"_src = doc.getObject('{obj_name}').Shape")
            else:
                lines.append("_src = [o for o in doc.Objects if hasattr(o, 'Shape') and o.Shape.Solids][-1].Shape")
            lines.append(f"_pattern_shapes = []")
            ax_x, ax_y, ax_z = float(axis_vec[0]), float(axis_vec[1]), float(axis_vec[2])
            cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
            lines.append(f"for _i in range({count}):")
            lines.append(f"    _a = {angle} * _i / {count}")
            lines.append("    _copy = _src.copy()")
            lines.append(f"    _copy.rotate(FreeCAD.Vector({cx},{cy},{cz}), FreeCAD.Vector({ax_x},{ax_y},{ax_z}), _a)")
            lines.append("    _pattern_shapes.append(_copy)")
            lines.append("_pattern_comp = _pattern_shapes[0]")
            lines.append("for _ps in _pattern_shapes[1:]:")
            lines.append("    _pattern_comp = _pattern_comp.fuse(_ps)")
            lines.append(f'obj = doc.addObject("Part::Feature", "{result_name}")')
            lines.append("obj.Shape = _pattern_comp")
            lines.append("doc.recompute()")

        elif op_type == "linear_pattern":
            # Linear array: replicate a feature along a direction
            obj_name = _safe_name(op["object"]) if op.get("object") else ""
            count = int(op.get("count", 4))
            spacing = float(op.get("spacing", 10.0))
            direction = op.get("direction", [1, 0, 0])
            result_name = _safe_name(op.get("result_name", "LinearPattern"))
            if obj_name:
                lines.append(f"_src = doc.getObject('{obj_name}').Shape")
            else:
                lines.append("_src = [o for o in doc.Objects if hasattr(o, 'Shape') and o.Shape.Solids][-1].Shape")
            dx, dy, dz = float(direction[0]), float(direction[1]), float(direction[2])
            lines.append(f"_lin_shapes = []")
            lines.append(f"for _i in range({count}):")
            lines.append("    _copy = _src.copy()")
            lines.append(f"    _copy.translate(FreeCAD.Vector({dx * spacing} * _i, {dy * spacing} * _i, {dz * spacing} * _i))")
            lines.append("    _lin_shapes.append(_copy)")
            lines.append("_lin_comp = _lin_shapes[0]")
            lines.append("for _ls in _lin_shapes[1:]:")
            lines.append("    _lin_comp = _lin_comp.fuse(_ls)")
            lines.append(f'obj = doc.addObject("Part::Feature", "{result_name}")')
            lines.append("obj.Shape = _lin_comp")
            lines.append("doc.recompute()")

        elif op_type == "mirror":
            # Mirror a feature across a plane
            obj_name = _safe_name(op["object"]) if op.get("object") else ""
            plane = op.get("plane", "YZ")  # XY, YZ, XZ
            result_name = _safe_name(op.get("result_name", "Mirror"))
            if obj_name:
                lines.append(f"_mir_src = doc.getObject('{obj_name}').Shape")
            else:
                lines.append("_mir_src = [o for o in doc.Objects if hasattr(o, 'Shape') and o.Shape.Solids][-1].Shape")
            # Build mirror transform
            if plane == "YZ":
                lines.append("_mir = _mir_src.mirror(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0))")
            elif plane == "XZ":
                lines.append("_mir = _mir_src.mirror(FreeCAD.Vector(0,0,0), FreeCAD.Vector(0,1,0))")
            elif plane == "XY":
                lines.append("_mir = _mir_src.mirror(FreeCAD.Vector(0,0,0), FreeCAD.Vector(0,0,1))")
            else:
                # Custom plane normal
                nx, ny, nz = op.get("plane_normal", [1, 0, 0])
                nx, ny, nz = float(nx), float(ny), float(nz)
                lines.append(f"_mir = _mir_src.mirror(FreeCAD.Vector(0,0,0), FreeCAD.Vector({nx},{ny},{nz}))")
            lines.append(f'obj = doc.addObject("Part::Feature", "{result_name}")')
            lines.append("obj.Shape = _mir")
            lines.append("doc.recompute()")

        elif op_type == "shell":
            # Hollow out a solid by removing faces
            obj_name = _safe_name(op["object"]) if op.get("object") else ""
            thickness = float(op.get("thickness", 2.0))
            # faces_to_remove: list of face indices to open
            faces_to_remove = op.get("faces_to_remove", [])
            result_name = _safe_name(op.get("result_name", "Shell"))
            if obj_name:
                lines.append(f"_shell_obj = doc.getObject('{obj_name}')")
            else:
                lines.append("_shell_obj = [o for o in doc.Objects if hasattr(o, 'Shape') and o.Shape.Solids][-1]")
            if faces_to_remove:
                face_indices = faces_to_remove
                lines.append(f"_shell_faces = [_shell_obj.Shape.Faces[_fi] for _fi in {face_indices!r}]")
            else:
                # Remove top face (last face, typically Z-max)
                lines.append("_shell_faces = [_shell_obj.Shape.Faces[-1]]")
            lines.append(f"_shell = _shell_obj.Shape.makeThickness(_shell_faces, {thickness}, 0.01)")
            lines.append(f'obj = doc.addObject("Part::Feature", "{result_name}")')
            lines.append("obj.Shape = _shell")
            lines.append("doc.recompute()")

        elif op_type == "draft":
            # Apply draft angle to faces (taper for injection molding)
            obj_name = _safe_name(op["object"]) if op.get("object") else ""
            angle = float(op.get("angle", 2.0))
            # direction: the pull direction
            direction = op.get("direction", [0, 0, 1])
            face_indices = op.get("faces", [])
            neutral_plane_origin = op.get("neutral_plane", [0, 0, 0])
            result_name = _safe_name(op.get("result_name", "Draft"))
            if obj_name:
                lines.append(f"_draft_obj = doc.getObject('{obj_name}')")
            else:
                lines.append("_draft_obj = [o for o in doc.Objects if hasattr(o, 'Shape') and o.Shape.Solids][-1]")
            if face_indices:
                lines.append(f"_draft_faces = [_draft_obj.Shape.Faces[_fi] for _fi in {face_indices!r}]")
            else:
                # Apply to all side faces (skip top and bottom)
                lines.append("_draft_faces = [f for f in _draft_obj.Shape.Faces]")
            dx, dy, dz = float(direction[0]), float(direction[1]), float(direction[2])
            no_x, no_y, no_z = float(neutral_plane_origin[0]), float(neutral_plane_origin[1]), float(neutral_plane_origin[2])
            lines.append(f"_draft_angle = {angle}")
            lines.append(f"_draft_dir = FreeCAD.Vector({dx},{dy},{dz})")
            lines.append("# Apply draft using makeDraftShape (FreeCAD Part API)")
            lines.append("_draft_result = _draft_obj.Shape")
            lines.append("_drafted = False")
            lines.append("for _df in _draft_faces:")
            lines.append("    try:")
            lines.append(f"        _ds = Part.makeDraftShape(_df, _draft_dir, _draft_angle, FreeCAD.Vector({no_x},{no_y},{no_z}))")
            lines.append("        _draft_result = _draft_result.fuse(_ds)")
            lines.append("        _drafted = True")
            lines.append("    except Exception:")
            lines.append("        pass")
            lines.append("if not _drafted:")
            lines.append("    # Fallback: use raw transform approach (scale + translate)")
            lines.append("    _draft_result = _draft_obj.Shape")
            lines.append(f'obj = doc.addObject("Part::Feature", "{result_name}")')
            lines.append("obj.Shape = _draft_result")
            lines.append("doc.recompute()")

        # ---------------------------------------------------------------
        # Sketch-based modeling operations
        # ---------------------------------------------------------------

        elif op_type == "create_sketch":
            # Create a 2D sketch from points/lines/arcs/circles
            name = _safe_name(op.get("name", "Sketch"))
            elements = op.get("elements", [])
            plane = op.get("plane", "XY")  # XY, XZ, YZ
            offset = float(op.get("offset", 0.0))
            lines.append("import Sketcher")
            lines.append(f'_sketch_obj = doc.addObject("Sketcher::SketchObject", "{name}")')
            # Set sketch plane — AttachmentSupport expects [(obj, (sub,))]
            if plane == "XZ":
                lines.append("_sketch_obj.AttachmentSupport = [(doc.getObject('Origin'), ('XZ_Plane',))]")
            elif plane == "YZ":
                lines.append("_sketch_obj.AttachmentSupport = [(doc.getObject('Origin'), ('YZ_Plane',))]")
            else:
                lines.append("_sketch_obj.AttachmentSupport = [(doc.getObject('Origin'), ('XY_Plane',))]")
            if offset != 0.0:
                lines.append(f"_sketch_obj.AttachmentOffset = App.Placement(App.Vector(0,0,{offset}), App.Vector(0,0,1), 0)")
            # Add sketch elements
            for idx_el, el in enumerate(elements):
                etype = el.get("type", "")
                if etype == "point":
                    px, py = float(el.get("x", 0)), float(el.get("y", 0))
                    lines.append(f"_sketch_obj.addGeometry(Part.Point(App.Vector({px},{py},0)), False)")
                elif etype == "line":
                    x1, y1 = float(el.get("x1", 0)), float(el.get("y1", 0))
                    x2, y2 = float(el.get("x2", 0)), float(el.get("y2", 0))
                    lines.append(f"_sketch_obj.addGeometry(Part.LineSegment(App.Vector({x1},{y1},0), App.Vector({x2},{y2},0)), False)")
                elif etype == "circle":
                    cx, cy = float(el.get("cx", 0)), float(el.get("cy", 0))
                    r = float(el.get("radius", 5))
                    lines.append(f"_sketch_obj.addGeometry(Part.Circle(App.Vector({cx},{cy},0), App.Vector(0,0,1), {r}), False)")
                elif etype == "arc":
                    cx, cy = float(el.get("cx", 0)), float(el.get("cy", 0))
                    r = float(el.get("radius", 5))
                    a1 = float(el.get("start_angle", 0))
                    a2 = float(el.get("end_angle", 360))
                    lines.append(f"_sketch_obj.addGeometry(Part.ArcOfCircle(Part.Circle(App.Vector({cx},{cy},0), App.Vector(0,0,1), {r}), {a1}, {a2}), False)")
                elif etype == "rectangle":
                    rx, ry = float(el.get("x", 0)), float(el.get("y", 0))
                    rw, rh = float(el.get("width", 10)), float(el.get("height", 10))
                    lines.append(f"# Rectangle at ({rx},{ry}) {rw}x{rh}")
                    lines.append(f"_sketch_obj.addGeometry(Part.LineSegment(App.Vector({rx},{ry},0), App.Vector({rx + rw},{ry},0)), False)")
                    lines.append(f"_sketch_obj.addGeometry(Part.LineSegment(App.Vector({rx + rw},{ry},0), App.Vector({rx + rw},{ry + rh},0)), False)")
                    lines.append(f"_sketch_obj.addGeometry(Part.LineSegment(App.Vector({rx + rw},{ry + rh},0), App.Vector({rx},{ry + rh},0)), False)")
                    lines.append(f"_sketch_obj.addGeometry(Part.LineSegment(App.Vector({rx},{ry + rh},0), App.Vector({rx},{ry},0)), False)")
                elif etype == "polygon":
                    cx, cy = float(el.get("cx", 0)), float(el.get("cy", 0))
                    r = float(el.get("radius", 10))
                    n = int(float(el.get("sides", 6)))
                    lines.append(f"import math")
                    lines.append(f"for _pi in range({n}):")
                    lines.append(f"    _a1 = 2 * math.pi * _pi / {n}")
                    lines.append(f"    _a2 = 2 * math.pi * (_pi + 1) / {n}")
                    lines.append(f"    _p1 = App.Vector({cx} + {r} * math.cos(_a1), {cy} + {r} * math.sin(_a1), 0)")
                    lines.append(f"    _p2 = App.Vector({cx} + {r} * math.cos(_a2), {cy} + {r} * math.sin(_a2), 0)")
                    lines.append(f"    _sketch_obj.addGeometry(Part.LineSegment(_p1, _p2), False)")
            lines.append("doc.recompute()")
            lines.append(f'print("Sketch created: {name}")')

        elif op_type == "extrude_sketch":
            # Extrude a sketch into a 3D solid
            sketch_name = _safe_name(op.get("sketch", "Sketch"))
            height = float(op.get("height", 10.0))
            result_name = _safe_name(op.get("name", "Extrusion"))
            direction = op.get("direction", "z")  # x, y, z
            midplane = op.get("midplane", False)
            reverse = op.get("reverse", False)
            # Direction vector
            if direction == "x":
                dx, dy, dz = 1, 0, 0
            elif direction == "y":
                dx, dy, dz = 0, 1, 0
            else:
                dx, dy, dz = 0, 0, 1
            if reverse:
                dx, dy, dz = -dx, -dy, -dz
            lines.append(f"_sketch_ref = doc.getObject('{sketch_name}')")
            lines.append("if _sketch_ref is None:")
            lines.append(f"    raise RuntimeError('Sketch {sketch_name} not found')")
            lines.append("_sketch_shape = _sketch_ref.Shape")
            lines.append("_sketch_wires = _sketch_shape.Wires")
            lines.append("if len(_sketch_wires) == 0:")
            lines.append("    # Try to get Face from sketch")
            lines.append("    _sketch_faces = _sketch_shape.Faces")
            lines.append("    if len(_sketch_faces) > 0:")
            lines.append("        _sketch_wires = [_sketch_faces[0].OuterWire]")
            lines.append(f"_extrude_dir = App.Vector({dx}, {dy}, {dz})")
            lines.append(f"_extrude_h = {height}")
            if midplane:
                lines.append(f"_extrude_h = _extrude_h / 2.0")
                lines.append("_extrude_shape = _sketch_shape.extrude(_extrude_dir * _extrude_h)")
                lines.append("_extrude_shape2 = _sketch_shape.extrude(_extrude_dir * -_extrude_h)")
                lines.append("_extrude_result = _extrude_shape.fuse(_extrude_shape2)")
            else:
                lines.append("_extrude_result = _sketch_shape.extrude(_extrude_dir * _extrude_h)")
            lines.append(f'obj = doc.addObject("Part::Feature", "{result_name}")')
            lines.append("obj.Shape = _extrude_result")
            lines.append("doc.recompute()")
            lines.append(f'print("Extruded sketch {sketch_name} -> {result_name}, height={height}")')

        elif op_type == "revolve_sketch":
            # Revolve a sketch around an axis to create a solid of revolution
            sketch_name = _safe_name(op.get("sketch", "Sketch"))
            result_name = _safe_name(op.get("name", "Revolution"))
            axis = op.get("axis", "z")  # x, y, z
            angle = float(op.get("angle", 360.0))  # degrees
            base_pt = op.get("base", [0, 0, 0])
            base_pt = [float(base_pt[0]), float(base_pt[1]), float(base_pt[2])]
            if axis == "x":
                ax_vec = "App.Vector(1, 0, 0)"
            elif axis == "y":
                ax_vec = "App.Vector(0, 1, 0)"
            else:
                ax_vec = "App.Vector(0, 0, 1)"
            lines.append(f"_sketch_ref = doc.getObject('{sketch_name}')")
            lines.append("if _sketch_ref is None:")
            lines.append(f"    raise RuntimeError('Sketch {sketch_name} not found')")
            lines.append("_sketch_shape = _sketch_ref.Shape")
            lines.append(f"_rev_base = App.Vector({base_pt[0]}, {base_pt[1]}, {base_pt[2]})")
            lines.append(f"_rev_axis = {ax_vec}")
            lines.append(f"_rev_angle = {angle}")
            lines.append("_rev_result = _sketch_shape.revolve(_rev_base, _rev_axis, _rev_angle)")
            lines.append(f'obj = doc.addObject("Part::Feature", "{result_name}")')
            lines.append("obj.Shape = _rev_result")
            lines.append("doc.recompute()")
            lines.append(f'print(f"Revolved sketch → {result_name}, angle={angle}")')

        elif op_type == "pocket":
            # Cut a pocket (pocket) from a solid using a sketch profile
            sketch_name = _safe_name(op.get("sketch", "Sketch"))
            raw_target = op.get("target", "")
            target = _safe_name(raw_target) if raw_target else ""
            depth = float(op.get("depth", 5.0))
            through_all = op.get("through_all", False)
            result_name = _safe_name(op.get("name", "PocketResult"))
            direction = op.get("direction", "z")
            reverse = op.get("reverse", False)
            if direction == "x":
                dx, dy, dz = 1, 0, 0
            elif direction == "y":
                dx, dy, dz = 0, 1, 0
            else:
                dx, dy, dz = 0, 0, 1
            if reverse:
                dx, dy, dz = -dx, -dy, -dz
            lines.append(f"_sketch_ref = doc.getObject('{sketch_name}')")
            lines.append("if _sketch_ref is None:")
            lines.append(f"    raise RuntimeError('Sketch {sketch_name} not found')")
            lines.append("_pocket_profile = _sketch_ref.Shape")
            lines.append(f"_pocket_dir = App.Vector({dx}, {dy}, {dz})")
            if through_all:
                lines.append("_pocket_depth = 10000  # large value for through-all")
            else:
                lines.append(f"_pocket_depth = {depth}")
            lines.append("_pocket_tool = _pocket_profile.extrude(_pocket_dir * _pocket_depth)")
            if target:
                lines.append(f"_target_ref = doc.getObject('{target}')")
                lines.append("if _target_ref is None:")
                lines.append(f"    raise RuntimeError('Target {target} not found')")
                lines.append("_target_shape = _target_ref.Shape")
            else:
                lines.append("# Use last created solid as target")
                lines.append("_solids = [o for o in doc.Objects if hasattr(o, 'Shape') and o.Shape.Solids]")
                lines.append("if not _solids:")
                lines.append("    raise RuntimeError('No solid found to cut pocket into')")
                lines.append("_target_shape = _solids[-1].Shape")
            lines.append("_pocket_result = _target_shape.cut(_pocket_tool)")
            lines.append(f'obj = doc.addObject("Part::Feature", "{result_name}")')
            lines.append("obj.Shape = _pocket_result")
            lines.append("doc.recompute()")
            lines.append(f'print("Pocket cut: depth={depth}, through_all={through_all}")')

        lines.append("")

    return "\n".join(lines)


def _build_batch_script(all_ops: list[list[dict]]) -> str:
    """Build a single FreeCAD script that processes multiple parts.

    Each element of *all_ops* is a list of operations for one part.
    All parts are created in the same FreeCAD session, which avoids
    the overhead of spawning 41 separate subprocesses.

    Returns the combined script text.
    """
    header = [
        "import FreeCAD",
        "import FreeCAD as App",
        "import Part",
        "import Mesh",
        "import json",
        "import sys",
        "import os",
        "",
    ]
    parts_lines: list[str] = []
    for part_ops in all_ops:
        part_lines: list[str] = []
        for op in part_ops:
            op_type = op["type"]
            part_lines.append(f"# Operation: {op_type}")

            if op_type == "new_doc":
                name = _safe_name(op.get("name", "Unnamed"))
                part_lines.append(f'doc = FreeCAD.newDocument("{name}")')

            elif op_type == "make_box":
                l, w, h = float(op["length"]), float(op["width"]), float(op["height"])
                name = _safe_name(op.get("name", "Box"))
                part_lines.append(f'box = Part.makeBox({l}, {w}, {h})')
                part_lines.append(f'obj = doc.addObject("Part::Feature", "{name}")')
                part_lines.append("obj.Shape = box")
                part_lines.append("doc.recompute()")

            elif op_type == "make_cylinder":
                r, h = float(op["radius"]), float(op["height"])
                name = _safe_name(op.get("name", "Cylinder"))
                part_lines.append(f'cyl = Part.makeCylinder({r}, {h})')
                part_lines.append(f'obj = doc.addObject("Part::Feature", "{name}")')
                part_lines.append("obj.Shape = cyl")
                part_lines.append("doc.recompute()")

            elif op_type == "make_sphere":
                r = float(op["radius"])
                name = _safe_name(op.get("name", "Sphere"))
                part_lines.append(f'sphere = Part.makeSphere({r})')
                part_lines.append(f'obj = doc.addObject("Part::Feature", "{name}")')
                part_lines.append("obj.Shape = sphere")
                part_lines.append("doc.recompute()")

            elif op_type == "export_stl":
                path = _safe_path(op["path"])
                name = _safe_name(op.get("name", ""))
                part_lines.append(f"_export_list = [o for o in doc.Objects if hasattr(o, 'Shape')]")
                part_lines.append(f'Mesh.export(_export_list, {json.dumps(str(path))})')
                part_lines.append(f'print({json.dumps(f"Exported: {path}")})')

            # For other op types, fall back to _build_script
            else:
                # Generate via _build_script and extract just the body lines
                single_script = _build_script([op])
                # Skip the header imports
                body_lines = []
                for line in single_script.split("\n"):
                    stripped = line.strip()
                    if stripped.startswith("import "):
                        continue
                    if stripped:
                        body_lines.append(line)
                part_lines.extend(body_lines)

            part_lines.append("")

        # Close document after each part to free memory
        part_lines.append("FreeCAD.closeDocument(FreeCAD.ActiveDocument.Name)")
        part_lines.append("")

        parts_lines.extend(part_lines)

    return "\n".join(header + parts_lines)


# --- Assembly rendering ---


# Subsystem color map (name → (r, g, b) floats 0-1)
SUBSYSTEM_COLORS: dict[str, tuple[float, float, float]] = {
    "chassis": (0.2, 0.4, 0.8),       # blue
    "arm_left": (0.8, 0.3, 0.2),      # red-orange
    "arm_right": (0.2, 0.7, 0.3),     # green
    "ipc": (0.7, 0.5, 0.1),           # gold
    "sensor_tower": (0.6, 0.2, 0.7),  # purple
}


# Simplified fastener dimensions: (head_r, head_h, shank_r, washer_r,
# washer_h, nut_r, nut_h) in mm.  Derived from DIN 912 / 934 / 125 specs.
_FASTENER_DIMS: dict[str, tuple[float, ...]] = {
    "M2":   (1.9, 2.0, 1.0,  2.5, 0.3, 2.3, 1.6),
    "M2.5": (2.25, 2.0, 1.25, 3.0, 0.5, 2.9, 2.0),
    "M3":   (2.75, 2.5, 1.5,  3.5, 0.5, 3.2, 2.4),
    "M4":   (3.5, 3.0, 2.0,  4.5, 0.8, 4.05, 3.2),
    "M5":   (4.25, 4.0, 2.5, 5.0, 1.0, 4.6, 4.7),
    "M6":   (5.0, 5.0, 3.0,  6.0, 1.6, 5.75, 5.2),
}


# Anchor face outward-normal directions in FreeCAD-local coordinates.
# FreeCAD make_box creates parts at corner-origin with X=length, Y=width,
# Z=height, so: top=+Z, bottom=-Z, front=-Y(width), back=+Y(width),
# left=-X(length), right=+X(length).
_FREECAD_ANCHOR_NORMALS: dict[str, tuple[float, float, float]] = {
    "top":    (0, 0, 1),
    "bottom": (0, 0, -1),
    "front":  (0, -1, 0),
    "back":   (0, 1, 0),
    "left":   (-1, 0, 0),
    "right":  (1, 0, 0),
}


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
                lines.append(f"_o.ViewObject.ShapeColor = {color}")
                lines.append("doc.recompute()")

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
            lines.append("    _shape = Part.Shape(_mesh.topology)")
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

        # Apply subsystem color
        color = SUBSYSTEM_COLORS.get(subsystem, (0.7, 0.7, 0.7))
        lines.append(f"_obj.ViewObject.ShapeColor = ({color[0]}, {color[1]}, {color[2]})")
        lines.append("doc.recompute()")
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

    # Save and export
    if output_path:
        lines.append(f'doc.saveAs({json.dumps(str(output_path) + ".FCStd")})')
        lines.append(f'_all_objs = [o for o in doc.Objects if hasattr(o, "Shape")]')
        lines.append(f'Mesh.export(_all_objs, {json.dumps(str(output_path) + ".stl")})')
        lines.append("print(f'Assembly saved and exported')")

    lines.append("print(f'Assembly: {len(doc.Objects)} objects created')")

    return "\n".join(lines)


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
        except Exception:
            pass

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
        except Exception:
            pass

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
        except Exception:
            pass

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
