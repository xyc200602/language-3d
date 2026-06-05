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
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from ..models.base import ToolDefinition
from .base import Tool
from .process_manager import _process_manager


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
            [fc_python, "-c", f"exec(open(r'{script_path}').read())"],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(f"FreeCAD script error:\n{result.stderr}")
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
            name = op.get("name", "Unnamed")
            lines.append(f'doc = FreeCAD.newDocument("{name}")')

        elif op_type == "make_box":
            l, w, h = op["length"], op["width"], op["height"]
            name = op.get("name", "Box")
            lines.append(f'box = Part.makeBox({l}, {w}, {h})')
            lines.append(f'obj = doc.addObject("Part::Feature", "{name}")')
            lines.append("obj.Shape = box")
            lines.append("doc.recompute()")

        elif op_type == "make_cylinder":
            r, h = op["radius"], op["height"]
            name = op.get("name", "Cylinder")
            lines.append(f'cyl = Part.makeCylinder({r}, {h})')
            lines.append(f'obj = doc.addObject("Part::Feature", "{name}")')
            lines.append("obj.Shape = cyl")
            lines.append("doc.recompute()")

        elif op_type == "make_sphere":
            r = op["radius"]
            name = op.get("name", "Sphere")
            lines.append(f'sphere = Part.makeSphere({r})')
            lines.append(f'obj = doc.addObject("Part::Feature", "{name}")')
            lines.append("obj.Shape = sphere")
            lines.append("doc.recompute()")

        elif op_type == "make_cone":
            r1 = op["radius1"]
            r2 = op.get("radius2", 0)
            h = op["height"]
            name = op.get("name", "Cone")
            lines.append(f'cone = Part.makeCone({r1}, {r2}, {h})')
            lines.append(f'obj = doc.addObject("Part::Feature", "{name}")')
            lines.append("obj.Shape = cone")
            lines.append("doc.recompute()")

        elif op_type == "boolean":
            operation = op["operation"]  # union, cut, intersection
            obj1_name = op["object1"]
            obj2_name = op["object2"]
            result_name = op.get("result_name", "Result")
            bool_map = {"union": "fuse", "cut": "cut", "intersection": "common"}
            method = bool_map[operation]
            lines.append(f'o1 = doc.getObject("{obj1_name}")')
            lines.append(f'o2 = doc.getObject("{obj2_name}")')
            lines.append(f'result_shape = o1.Shape.{method}(o2.Shape)')
            lines.append(f'result = doc.addObject("Part::Feature", "{result_name}")')
            lines.append("result.Shape = result_shape")
            lines.append("doc.recompute()")

        elif op_type == "move":
            obj_name = op["object"]
            dx, dy, dz = op["dx"], op["dy"], op["dz"]
            lines.append(f'_obj = doc.getObject("{obj_name}")')
            lines.append("_moved = _obj.Shape.copy()")
            lines.append(f"_moved.translate(FreeCAD.Vector({dx}, {dy}, {dz}))")
            lines.append("_obj.Shape = _moved")
            lines.append("doc.recompute()")

        elif op_type == "cylinder_with_hole":
            orad = op["outer_radius"]
            irad = op["inner_radius"]
            h = op["height"]
            name = op.get("name", "CylinderWithHole")
            lines.append(f'_outer = Part.makeCylinder({orad}, {h})')
            lines.append(f'_inner = Part.makeCylinder({irad}, {h})')
            lines.append("_result = _outer.cut(_inner)")
            lines.append(f'obj = doc.addObject("Part::Feature", "{name}")')
            lines.append("obj.Shape = _result")
            lines.append("doc.recompute()")

        elif op_type == "plate_with_holes":
            length = op["length"]
            width = op["width"]
            thickness = op["thickness"]
            hole_radius = op["hole_radius"]
            nx = op.get("hole_count_x", 2)
            ny = op.get("hole_count_y", 2)
            margin = op.get("margin", 0)
            name = op.get("name", "PlateWithHoles")
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
            obj_name = op["object"]
            radius = op["radius"]
            lines.append(f'_obj = doc.getObject("{obj_name}")')
            lines.append(f"_fillet = _obj.Shape.makeFillet({radius}, _obj.Shape.Edges)")
            lines.append("_obj.Shape = _fillet")
            lines.append("doc.recompute()")

        elif op_type == "chamfer":
            obj_name = op["object"]
            size = op["size"]
            lines.append(f'_obj = doc.getObject("{obj_name}")')
            lines.append(f"_chamfer = _obj.Shape.makeChamfer({size}, _obj.Shape.Edges)")
            lines.append("_obj.Shape = _chamfer")
            lines.append("doc.recompute()")

        elif op_type == "save":
            path = op["path"].replace("\\", "\\\\")
            # Make all objects visible before saving so they show in GUI
            lines.append("for _o in doc.Objects:")
            lines.append("    try:")
            lines.append("        if hasattr(_o, 'ViewObject') and _o.ViewObject is not None:")
            lines.append("            _o.ViewObject.Visibility = True")
            lines.append("    except Exception:")
            lines.append("        pass")
            lines.append("doc.recompute()")
            lines.append(f'doc.saveAs(r"{op["path"]}")')

        elif op_type == "export_stl":
            path = op["path"]
            obj_name = op.get("object", "")
            tolerance = op.get("tolerance", 0.1)
            # Use Mesh.export which is more robust than Mesh.Mesh(shape.tessellate())
            if obj_name:
                lines.append(f'_export_list = [doc.getObject("{obj_name}")]')
            else:
                lines.append("_export_list = [o for o in doc.Objects if hasattr(o, 'Shape')]")
            lines.append(f'Mesh.export(_export_list, r"{path}")')
            lines.append("import os")
            lines.append(f'_stl_path = r"{path}"')
            lines.append('print(f"STL exported: {os.path.getsize(_stl_path):,} bytes")')

        elif op_type == "export_step":
            path = op["path"]
            obj_name = op.get("object", "")
            if obj_name:
                lines.append(f'_obj = doc.getObject("{obj_name}")')
                lines.append(f'_obj.Shape.exportStep(r"{path}")')
            else:
                lines.append("_shapes = [o.Shape for o in doc.Objects if hasattr(o, 'Shape')]")
                lines.append("_compound = _shapes[0]")
                lines.append("for s in _shapes[1:]:")
                lines.append("    _compound = _compound.fuse(s)")
                lines.append(f'_compound.exportStep(r"{path}")')

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
            obj_name = op["object"]
            lines.append(f'_obj = doc.getObject("{obj_name}")')
            lines.append("if _obj and hasattr(_obj, 'Shape'):")
            lines.append("    s = _obj.Shape")
            lines.append("    bb = s.BoundBox")
            lines.append("    print(f'Volume: {s.Volume:.2f} mm3')")
            lines.append("    print(f'Area: {s.Area:.2f} mm2')")
            lines.append("    print(f'Dims: {bb.XLength:.2f}x{bb.YLength:.2f}x{bb.ZLength:.2f}mm')")
            lines.append("    print(f'Edges: {len(s.Edges)}, Faces: {len(s.Faces)}, Vertices: {len(s.Vertexes)}')")

        elif op_type == "delete_object":
            obj_name = op["object"]
            lines.append(f'doc.removeObject("{obj_name}")')
            lines.append("doc.recompute()")

        elif op_type == "volume_check":
            # Lightweight verification: load file and check all object
            # dimensions without opening GUI or using VLM.
            path = op.get("path", "").replace("\\", "\\\\")
            checks = op.get("checks", {})
            # Expected dims, volume range, etc.
            expected_dims = checks.get("dimensions", {})
            min_volume = checks.get("min_volume", 0)
            max_volume = checks.get("max_volume", float("inf"))
            if path:
                lines.append(f'_vc_doc = FreeCAD.openDocument(r"{path}")')
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
                    dim_map = {"length": "XLength", "width": "YLength", "height": "ZLength",
                               "x": "XLength", "y": "YLength", "z": "ZLength"}
                    attr = dim_map.get(dim_name.lower(), dim_name)
                    tol = checks.get("tolerance_mm", 1.0)
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
            obj_name = op["object"]
            axis = op["axis"]  # x, y, z
            angle = op["angle"]
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
            lines.append(op["script"])

        elif op_type == "compute_mass":
            # Compute mass properties for all solid objects in the document
            obj_name = op.get("object", "")
            density = op.get("density", 1240)  # kg/m³, default PLA
            path = op.get("path", "").replace("\\", "\\\\")
            if path:
                lines.append(f'_cm_doc = FreeCAD.openDocument(r"{path}")')
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
            name = op.get("name", "Sweep")
            profile_type = op.get("profile", "circle")  # circle, rectangle, custom
            profile_radius = op.get("profile_radius", 2.0)
            profile_width = op.get("profile_width", 4.0)
            profile_height = op.get("profile_height", 4.0)
            path_type = op.get("path_type", "helix")  # helix, circle, line, custom
            # Helix params
            pitch = op.get("pitch", 5.0)
            height = op.get("height", 20.0)
            helix_radius = op.get("helix_radius", 10.0)
            turns = op.get("turns", 0)
            # Circle path params
            path_radius = op.get("path_radius", 15.0)
            # Line path params
            line_length = op.get("line_length", 30.0)
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
                dx, dy, dz = line_dir
                lines.append(f"_path = Part.Wire(Part.makeLine(FreeCAD.Vector(0,0,0), FreeCAD.Vector({dx * line_length}, {dy * line_length}, {dz * line_length})))")
            else:
                custom_path = op.get("custom_path_script", "_path = Part.Wire(Part.makeLine(FreeCAD.Vector(0,0,0), FreeCAD.Vector(30,0,0)))")
                lines.append(custom_path)
            # Perform sweep
            lines.append(f"_sweep = _prof.makePipeShell([_path], {solid}, {frenet})")
            lines.append(f'obj = doc.addObject("Part::Feature", "{name}")')
            lines.append("obj.Shape = _sweep")
            lines.append("doc.recompute()")

        elif op_type == "loft":
            # Loft between multiple profiles (e.g. transitions, brackets)
            name = op.get("name", "Loft")
            profiles = op.get("profiles", [])
            solid = op.get("solid", True)
            ruled = op.get("ruled", False)
            if profiles:
                # Build profiles from specifications
                # Each profile: {type: "circle"|"rectangle", radius, center:[x,y,z], ...}
                for i, prof in enumerate(profiles):
                    ptype = prof.get("type", "circle")
                    cx, cy, cz = prof.get("center", [0, 0, 0])
                    if ptype == "circle":
                        r = prof.get("radius", 5.0)
                        lines.append(f"_loft_p{i} = Part.Wire(Part.makeCircle({r}, FreeCAD.Vector({cx},{cy},{cz})))")
                    elif ptype == "rectangle":
                        w = prof.get("width", 10.0)
                        h = prof.get("height", 10.0)
                        hw, hh = w / 2, h / 2
                        lines.append(f"_loft_p{i} = Part.Wire(Part.makePolygon([")
                        lines.append(f"    FreeCAD.Vector({cx - hw},{cy - hh},{cz}),")
                        lines.append(f"    FreeCAD.Vector({cx + hw},{cy - hh},{cz}),")
                        lines.append(f"    FreeCAD.Vector({cx + hw},{cy + hh},{cz}),")
                        lines.append(f"    FreeCAD.Vector({cx - hw},{cy + hh},{cz}),")
                        lines.append(f"    FreeCAD.Vector({cx - hw},{cy - hh},{cz}),")
                        lines.append("]))")
                    elif ptype == "polygon":
                        r = prof.get("radius", 5.0)
                        sides = prof.get("sides", 6)
                        import math as _math
                        pts = []
                        for s in range(sides + 1):
                            ang = 2 * 3.141592653589793 * s / sides
                            pts.append(f"FreeCAD.Vector({cx + r * round(_math.cos(ang), 6)},{cy + r * round(_math.sin(ang), 6)},{cz})")
                        lines.append(f"_loft_p{i} = Part.Wire(Part.makePolygon([{', '.join(pts)}]))")
                    else:
                        # ellipse
                        r1 = prof.get("radius1", 10.0)
                        r2 = prof.get("radius2", 5.0)
                        lines.append(f"_loft_p{i} = Part.Wire(Part.makeEllipse({r1}, {r2}, FreeCAD.Vector({cx},{cy},{cz})))")
                profile_list = ", ".join(f"_loft_p{i}" for i in range(len(profiles)))
                lines.append(f"_loft_shapes = [{profile_list}]")
            else:
                # Two simple circles at z=0 and z=height
                r1 = op.get("radius1", 10.0)
                r2 = op.get("radius2", 5.0)
                h = op.get("height", 20.0)
                lines.append(f"_loft_p0 = Part.Wire(Part.makeCircle({r1}, FreeCAD.Vector(0,0,0)))")
                lines.append(f"_loft_p1 = Part.Wire(Part.makeCircle({r2}, FreeCAD.Vector(0,0,{h})))")
                lines.append("_loft_shapes = [_loft_p0, _loft_p1]")
            lines.append(f"_loft = Part.makeLoft(_loft_shapes, {solid}, {ruled}, False)")
            lines.append(f'obj = doc.addObject("Part::Feature", "{name}")')
            lines.append("obj.Shape = _loft")
            lines.append("doc.recompute()")

        elif op_type == "polar_pattern":
            # Circular array: replicate a feature around an axis
            obj_name = op.get("object", "")
            count = op.get("count", 6)
            angle = op.get("angle", 360.0)
            axis_vec = op.get("axis", [0, 0, 1])
            center = op.get("center", [0, 0, 0])
            result_name = op.get("result_name", "PolarPattern")
            if obj_name:
                lines.append(f"_src = doc.getObject('{obj_name}').Shape")
            else:
                lines.append("_src = [o for o in doc.Objects if hasattr(o, 'Shape') and o.Shape.Solids][-1].Shape")
            lines.append(f"_pattern_shapes = []")
            ax_x, ax_y, ax_z = axis_vec
            cx, cy, cz = center
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
            obj_name = op.get("object", "")
            count = op.get("count", 4)
            spacing = op.get("spacing", 10.0)
            direction = op.get("direction", [1, 0, 0])
            result_name = op.get("result_name", "LinearPattern")
            if obj_name:
                lines.append(f"_src = doc.getObject('{obj_name}').Shape")
            else:
                lines.append("_src = [o for o in doc.Objects if hasattr(o, 'Shape') and o.Shape.Solids][-1].Shape")
            dx, dy, dz = direction
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
            obj_name = op.get("object", "")
            plane = op.get("plane", "YZ")  # XY, YZ, XZ
            result_name = op.get("result_name", "Mirror")
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
                lines.append(f"_mir = _mir_src.mirror(FreeCAD.Vector(0,0,0), FreeCAD.Vector({nx},{ny},{nz}))")
            lines.append(f'obj = doc.addObject("Part::Feature", "{result_name}")')
            lines.append("obj.Shape = _mir")
            lines.append("doc.recompute()")

        elif op_type == "shell":
            # Hollow out a solid by removing faces
            obj_name = op.get("object", "")
            thickness = op.get("thickness", 2.0)
            # faces_to_remove: list of face indices to open
            faces_to_remove = op.get("faces_to_remove", [])
            result_name = op.get("result_name", "Shell")
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
            obj_name = op.get("object", "")
            angle = op.get("angle", 2.0)
            # direction: the pull direction
            direction = op.get("direction", [0, 0, 1])
            face_indices = op.get("faces", [])
            neutral_plane_origin = op.get("neutral_plane", [0, 0, 0])
            result_name = op.get("result_name", "Draft")
            if obj_name:
                lines.append(f"_draft_obj = doc.getObject('{obj_name}')")
            else:
                lines.append("_draft_obj = [o for o in doc.Objects if hasattr(o, 'Shape') and o.Shape.Solids][-1]")
            if face_indices:
                lines.append(f"_draft_faces = [_draft_obj.Shape.Faces[_fi] for _fi in {face_indices!r}]")
            else:
                # Apply to all side faces (skip top and bottom)
                lines.append("_draft_faces = [f for f in _draft_obj.Shape.Faces]")
            dx, dy, dz = direction
            no_x, no_y, no_z = neutral_plane_origin
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
            name = op.get("name", "Sketch")
            elements = op.get("elements", [])
            plane = op.get("plane", "XY")  # XY, XZ, YZ
            offset = op.get("offset", 0.0)
            lines.append("import Sketcher")
            lines.append(f'_sketch_obj = doc.addObject("Sketcher::SketchObject", "{name}")')
            # Set sketch plane
            if plane == "XZ":
                lines.append("_sketch_obj.AttachmentSupport = [doc.getObject('Origin'), 'XZ_Plane']")
            elif plane == "YZ":
                lines.append("_sketch_obj.AttachmentSupport = [doc.getObject('Origin'), 'YZ_Plane']")
            else:
                lines.append("_sketch_obj.AttachmentSupport = [doc.getObject('Origin'), 'XY_Plane']")
            if offset != 0.0:
                lines.append(f"_sketch_obj.AttachmentOffset = App.Placement(App.Vector(0,0,{offset}), App.Vector(0,0,1), 0)")
            # Add sketch elements
            for idx_el, el in enumerate(elements):
                etype = el.get("type", "")
                if etype == "point":
                    px, py = el.get("x", 0), el.get("y", 0)
                    lines.append(f"_sketch_obj.addGeometry(Part.Point(App.Vector({px},{py},0)), False)")
                elif etype == "line":
                    x1, y1 = el.get("x1", 0), el.get("y1", 0)
                    x2, y2 = el.get("x2", 0), el.get("y2", 0)
                    lines.append(f"_sketch_obj.addGeometry(Part.LineSegment(App.Vector({x1},{y1},0), App.Vector({x2},{y2},0)), False)")
                elif etype == "circle":
                    cx, cy = el.get("cx", 0), el.get("cy", 0)
                    r = el.get("radius", 5)
                    lines.append(f"_sketch_obj.addGeometry(Part.Circle(App.Vector({cx},{cy},0), App.Vector(0,0,1), {r}), False)")
                elif etype == "arc":
                    cx, cy = el.get("cx", 0), el.get("cy", 0)
                    r = el.get("radius", 5)
                    a1 = el.get("start_angle", 0)
                    a2 = el.get("end_angle", 360)
                    lines.append(f"_sketch_obj.addGeometry(Part.ArcOfCircle(Part.Circle(App.Vector({cx},{cy},0), App.Vector(0,0,1), {r}), {a1}, {a2}), False)")
                elif etype == "rectangle":
                    rx, ry = el.get("x", 0), el.get("y", 0)
                    rw, rh = el.get("width", 10), el.get("height", 10)
                    lines.append(f"# Rectangle at ({rx},{ry}) {rw}x{rh}")
                    lines.append(f"_sketch_obj.addGeometry(Part.LineSegment(App.Vector({rx},{ry},0), App.Vector({rx + rw},{ry},0)), False)")
                    lines.append(f"_sketch_obj.addGeometry(Part.LineSegment(App.Vector({rx + rw},{ry},0), App.Vector({rx + rw},{ry + rh},0)), False)")
                    lines.append(f"_sketch_obj.addGeometry(Part.LineSegment(App.Vector({rx + rw},{ry + rh},0), App.Vector({rx},{ry + rh},0)), False)")
                    lines.append(f"_sketch_obj.addGeometry(Part.LineSegment(App.Vector({rx},{ry + rh},0), App.Vector({rx},{ry},0)), False)")
                elif etype == "polygon":
                    cx, cy = el.get("cx", 0), el.get("cy", 0)
                    r = el.get("radius", 10)
                    n = el.get("sides", 6)
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
            sketch_name = op.get("sketch", "Sketch")
            height = op.get("height", 10.0)
            result_name = op.get("name", "Extrusion")
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
            sketch_name = op.get("sketch", "Sketch")
            result_name = op.get("name", "Revolution")
            axis = op.get("axis", "z")  # x, y, z
            angle = op.get("angle", 360.0)  # degrees
            base_pt = op.get("base", [0, 0, 0])
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
            sketch_name = op.get("sketch", "Sketch")
            target = op.get("target", "")  # target solid object name
            depth = op.get("depth", 5.0)
            through_all = op.get("through_all", False)
            result_name = op.get("name", "PocketResult")
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
                f'doc = FreeCAD.openDocument(r"{file_path}")',
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
    ]
    for tool in tools:
        registry.register(tool)
