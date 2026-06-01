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


def _run_freecad_script(script: str, timeout: int = 60) -> str:
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
        )
        if result.returncode != 0:
            raise RuntimeError(f"FreeCAD script error:\n{result.stderr}")
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        raise RuntimeError("FreeCAD script timed out")
    finally:
        Path(script_path).unlink(missing_ok=True)


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

        elif op_type == "rotate":
            obj_name = op["object"]
            axis = op["axis"]  # x, y, z
            angle = op["angle"]
            lines.append("import math")
            lines.append(f'_obj = doc.getObject("{obj_name}")')
            lines.append(f'_axis_map = {{"x": FreeCAD.Vector(1,0,0), "y": FreeCAD.Vector(0,1,0), "z": FreeCAD.Vector(0,0,1)}}')
            lines.append(f'_rot_axis = _axis_map["{axis}"]')
            lines.append(f'_rad = math.radians({angle})')
            lines.append("_rotated = _obj.Shape.copy()")
            lines.append("_center = _rotated.BoundBox.Center")
            lines.append(f"_rotation = FreeCAD.Rotation(_rot_axis, _rad)")
            lines.append("_moved = _rotated.copy()")
            lines.append("_moved.translate(FreeCAD.Vector(-_center.x, -_center.y, -_center.z))")
            lines.append("_moved.rotate(_rotation)")
            lines.append("_moved.translate(_center)")
            lines.append("_obj.Shape = _moved")
            lines.append("doc.recompute()")

        elif op_type == "raw_script":
            lines.append(op["script"])

        lines.append("")

    return "\n".join(lines)


def _execute_operations(operations: list[dict]) -> str:
    """Execute a sequence of FreeCAD operations and return the output."""
    script = _build_script(operations)
    try:
        output = _run_freecad_script(script)
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
        "save, export_stl, export_step, object_info, delete_object."
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

        view_method = VIEW_METHODS.get(view, "viewIsometric")
        cmd = [fc_exe]

        if file_path:
            if not Path(file_path).exists():
                return f"Error: File not found: {file_path}"

            # Build startup macro: open doc + set camera
            macro_lines = [
                "import FreeCAD",
                "import FreeCADGui",
                "import time",
                "",
                f'doc = FreeCAD.openDocument(r"{file_path}")',
                "time.sleep(0.5)",
                "",
                "def _set_camera():",
                "    try:",
                "        v = FreeCADGui.activeDocument().activeView()",
                f"        v.{view_method}()",
                "    except Exception:",
                "        pass",
                "",
            ]
            if fit_all:
                macro_lines.extend([
                    "    try:",
                    '        FreeCADGui.SendMsgToActiveView("ViewFit")',
                    "    except Exception:",
                    "        pass",
                    "",
                ])
            macro_lines.extend([
                "    print(f'Opened: {doc.Name}')",
                "",
                "# Delay camera to let GUI finish rendering",
                "from PySide2 import QtCore",
                "QtCore.QTimer.singleShot(1500, _set_camera)",
            ])

            macro_content = "\n".join(macro_lines)
            macro_dir = Path(tempfile.gettempdir()) / "lang3d_macros"
            macro_dir.mkdir(exist_ok=True)
            macro_path = macro_dir / f"open_gui_{int(time.time())}.py"
            macro_path.write_text(macro_content, encoding="utf-8")

            cmd.append(str(macro_path))

        proc = subprocess.Popen(cmd)

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
        import ctypes

        from ..tools.screen import _find_windows_by_title

        windows = _find_windows_by_title("FreeCAD")
        if not windows:
            return "No FreeCAD window found (already closed or not running)"

        hwnd, title = windows[0]
        user32 = ctypes.windll.user32
        # Send WM_CLOSE (0x0010) for graceful shutdown
        user32.PostMessageW(hwnd, 0x0010, 0, 0)

        # Wait and verify window closed
        time.sleep(2)
        remaining = _find_windows_by_title("FreeCAD")
        if remaining:
            return f"Sent close signal to FreeCAD: '{title}' (window still visible)"
        return f"FreeCAD closed: '{title}'"


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
