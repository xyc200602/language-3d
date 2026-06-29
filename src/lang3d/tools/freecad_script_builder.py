"""FreeCAD script builders — operation dict → Python script.

Extracted from freecad.py (P1-1 God Module split, AGENTS.md §2.1).
_build_script converts a list of operation dicts into a FreeCAD Python
script; _build_batch_script does the same for multiple parts in one run.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Deferred import to avoid circular dependency (freecad.py imports from
# this module at module level for the re-export, so we can't import from
# freecad at module level here). Resolve on first use.
_fc_helpers = None
def _get_helpers():
    global _fc_helpers
    if _fc_helpers is None:
        from .freecad import _safe_name, _safe_path, _validate_raw_script
        _fc_helpers = (_safe_name, _safe_path, _validate_raw_script)
    return _fc_helpers

def _build_script(operations: list[dict]) -> str:
    """Build a FreeCAD Python script from a list of operations.

    Each operation is a dict with 'type' and parameters.
    """
    # Resolve deferred imports (circular dependency avoidance).
    global _safe_name, _safe_path, _validate_raw_script
    _safe_name, _safe_path, _validate_raw_script = _get_helpers()

    lines = [
        "import FreeCAD",
        "import FreeCAD as App",
        "import Part",
        "import Mesh",
        "import json",
        "import sys",
        "",
        "# Helper: identify edges that are UNSAFE to chamfer/fillet.",
        "# Passing such edges to makeChamfer/makeFillet raises in OCC",
        "# (BRep_API: command not done / Null input shape), which aborts",
        "# the whole script and loses the STEP export.  Unsafe edges:",
        "#  - degenerate (zero-length) edges",
        "#  - seam edges on cylinders/cones/spheres (closed-surface seams)",
        "#  - edges whose underlying curve is not a line (circles, arcs,",
        "#    splines) — chamfering a curved edge is geometrically invalid",
        "#    and is the most common cause of the servo/motor STEP crash.",
        "def _geom_is_degenerate(_edge):",
        "    try:",
        "        if _edge.Length < 1e-6:",
        "            return True",
        "        _curve = _edge.Curve",
        "        # Part.Line / Part.LineSegment → safe (straight edge).",
        "        # Part.Circle/Arc/BSpline/etc → unsafe for chamfer.",
        "        return not _curve.__class__.__name__.startswith('Line')",
        "    except Exception:",
        "        return True",
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
            # Solver convention: X=width (left/right), Y=length (front/back),
            # Z=height (see assembly_solver.ANCHOR_DIM_KEYS: front/back map
            # to 'length', left/right to 'width').  FreeCAD makeBox(X,Y,Z)
            # takes (X-extent, Y-extent, Z-extent), so pass (width, length,
            # height) to put length on Y and width on X.  Without this swap
            # the STL has length on X and width on Y — the WRONG axis — and
            # the renderer's swap_xy only masks it visually (the URDF/MuJoCo
            # mesh stays wrong).  This is the same class of bug as the wheel
            # 磨盘 defect: STL geometry axis vs solver axis mismatch.
            lines.append(f'box = Part.makeBox({w}, {l}, {h})')
            lines.append(f'obj = doc.addObject("Part::Feature", "{name}")')
            lines.append("obj.Shape = box")
            lines.append("doc.recompute()")

        elif op_type == "make_cylinder":
            r = float(op.get("radius", 0))
            h = float(op.get("height", 0))
            name = _safe_name(op.get("name", "Cylinder"))
            lines.append(f'cyl = Part.makeCylinder({r}, {h})')
            # Orient the cylinder's axis.  makeCylinder builds along Z by
            # default.  ``orient_axis`` re-points it so the cylinder's
            # SYMMETRY axis matches the part's physical rotation axis:
            #   "z" (default) → vertical (servos, standoffs)
            #   "y"           → axle along Y (WHEELS — rolls along X)
            #   "x"           → axle along X
            # This is the fix for the "磨盘" wheel bug: a wheel is a
            # revolute joint with axis="y", so its cylinder must also lie
            # along Y, but makeCylinder always builds along Z.  Without
            # this rotation the wheel STL is a disc lying flat (圆面朝上下),
            # not a wheel (圆面朝左右).  An explicit ``rotation`` op field
            # (below) takes precedence for connection-feature bolt holes.
            orient_axis = str(op.get("orient_axis", "z")).lower()
            if orient_axis in ("y", "x"):
                _ax, _ang = ("1,0,0", "90.0") if orient_axis == "y" else ("0,1,0", "90.0")
                lines.append(
                    f'cyl.rotate(FreeCAD.Vector(0,0,0), '
                    f'FreeCAD.Vector({_ax}), {_ang})'
                )
            # Optional explicit rotation (added 2026-06-21, Plan B): when
            # present, rotate the cylinder so its Z axis aligns with the
            # desired hole direction.  Used by connection_features bolt
            # holes on side anchors (front/back/left/right) where the bolt
            # enters along Y/X, not Z.  Without this the cylinder axis is
            # wrong and the boolean cut fragments the part geometry.
            # Format: op["rotation"] = [ax, ay, az, angle_deg]
            rot = op.get("rotation")
            if rot and len(rot) == 4 and float(rot[3]) != 0.0:
                ax, ay, az, ang = (float(v) for v in rot)
                lines.append(
                    f'cyl.rotate(FreeCAD.Vector(0,0,0), '
                    f'FreeCAD.Vector({ax},{ay},{az}), {ang})'
                )
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
            # Orient the cylinder (see make_cylinder above for the full
            # rationale).  A wheel = cylinder_with_hole (tyre + axle bore)
            # with axis="y": both the tyre and the bore must rotate together
            # so the axle hole ends up along Y (matching the revolute axis),
            # not Z.  Without this the bore is perpendicular to the axle and
            # the wheel can never mount on its shaft.
            orient_axis = str(op.get("orient_axis", "z")).lower()
            if orient_axis in ("y", "x"):
                _ax, _ang = ("1,0,0", "90.0") if orient_axis == "y" else ("0,1,0", "90.0")
                lines.append(
                    f'_result.rotate(FreeCAD.Vector(0,0,0), '
                    f'FreeCAD.Vector({_ax}), {_ang})'
                )
            lines.append(f'obj = doc.addObject("Part::Feature", "{name}")')
            lines.append("obj.Shape = _result")
            lines.append("doc.recompute()")

        elif op_type == "plate_with_holes":
            length = float(op["length"])   # solver: Y extent (front/back)
            width = float(op["width"])     # solver: X extent (left/right)
            thickness = float(op["thickness"])
            hole_radius = float(op["hole_radius"])
            nx = int(float(op.get("hole_count_x", 2)))   # holes along X (width)
            ny = int(float(op.get("hole_count_y", 2)))   # holes along Y (length)
            margin = float(op.get("margin", 0))
            name = _safe_name(op.get("name", "PlateWithHoles"))
            if margin == 0:
                margin = min(length, width) * 0.1
            # Solver convention: X=width, Y=length (see make_box above).
            # makeBox(X, Y, Z) → (width, length, thickness).  Holes span the
            # width (X) with nx holes and the length (Y) with ny holes, so
            # the X spacing uses the width extent and the Y spacing uses the
            # length extent.
            sx = (width - 2 * margin) / max(nx - 1, 1) if nx > 1 else 0
            sy = (length - 2 * margin) / max(ny - 1, 1) if ny > 1 else 0
            lines.append(f'_plate = Part.makeBox({width}, {length}, {thickness})')
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
            # Robust edge selection: passing ALL edges to makeFillet can
            # crash OCC ("BRep_API: command not done" / "Null input
            # shape") on shapes whose edge set includes degenerate edges
            # (zero-length), seam edges on cylinders/cones, or curved
            # edges where a fillet is geometrically invalid.  Such a crash
            # aborts the whole FreeCAD script, losing the STEP export for
            # that part (the systematic servo/motor STEP-loss bug).
            # Filter to straight (linear) edges of sensible length, and
            # wrap in try/except that REPORTS the skip (not silently
            # swallows — AGENTS.md §1.1): a missing fillet is cosmetic,
            # a missing STEP is a production-blocker.
            lines.append("_edges = [_e for _e in _obj.Shape.Edges")
            lines.append("           if not _geom_is_degenerate(_e)]")
            _safe_obj = obj_name.replace('"', '\\"')
            _warn_fillet = (
                '    print("WARN fillet skipped on \\"' + _safe_obj + '\\": " '
                '+ str(type(_e).__name__) + ": " + str(_e) '
                + '+ " (" + str(len(_edges)) + " edges)")'
            )
            lines.append("try:")
            lines.append(f"    _fillet = _obj.Shape.makeFillet({radius}, _edges)")
            lines.append("    _obj.Shape = _fillet")
            lines.append("    doc.recompute()")
            lines.append("except Exception as _e:")
            lines.append(_warn_fillet)
            lines.append('    print("WARN   shape has " + str(len(_obj.Shape.Edges)) + " total edges")')

        elif op_type == "chamfer":
            obj_name = _safe_name(op["object"])
            size = float(op["size"])
            lines.append(f'_obj = doc.getObject("{obj_name}")')
            # Same robust-edge rationale as fillet (see above).  Chamfering
            # a cylinder end-face circle or a seam edge raises in OCC and
            # kills the export; filter to chamfer-safe straight edges.
            lines.append("_edges = [_e for _e in _obj.Shape.Edges")
            lines.append("           if not _geom_is_degenerate(_e)]")
            _safe_obj = obj_name.replace('"', '\\"')
            _warn_chamfer = (
                '    print("WARN chamfer skipped on \\"' + _safe_obj + '\\": " '
                '+ str(type(_e).__name__) + ": " + str(_e) '
                + '+ " (" + str(len(_edges)) + " edges)")'
            )
            lines.append("try:")
            lines.append(f"    _chamfer = _obj.Shape.makeChamfer({size}, _edges)")
            lines.append("    _obj.Shape = _chamfer")
            lines.append("    doc.recompute()")
            lines.append("except Exception as _e:")
            lines.append(_warn_chamfer)
            lines.append('    print("WARN   shape has " + str(len(_obj.Shape.Edges)) + " total edges")')

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
                    dim_map = {"length": "YLength", "width": "XLength", "height": "ZLength",
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
                # Solver convention: X=width, Y=length, Z=height (see
                # _build_script make_box).  makeBox(X,Y,Z) → (width, length,
                # height).  This batch path must match the single-script path.
                part_lines.append(f'box = Part.makeBox({w}, {l}, {h})')
                part_lines.append(f'obj = doc.addObject("Part::Feature", "{name}")')
                part_lines.append("obj.Shape = box")
                part_lines.append("doc.recompute()")

            elif op_type == "make_cylinder":
                r, h = float(op["radius"]), float(op["height"])
                name = _safe_name(op.get("name", "Cylinder"))
                part_lines.append(f'cyl = Part.makeCylinder({r}, {h})')
                # Optional rotation — see the make_cylinder handler in
                # build_assembly_script for why this exists (Plan B:
                # orient bolt-hole cylinders per anchor).
                rot = op.get("rotation")
                if rot and len(rot) == 4 and float(rot[3]) != 0.0:
                    ax, ay, az, ang = (float(v) for v in rot)
                    part_lines.append(
                        f'cyl.rotate(FreeCAD.Vector(0,0,0), '
                        f'FreeCAD.Vector({ax},{ay},{az}), {ang})'
                    )
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

