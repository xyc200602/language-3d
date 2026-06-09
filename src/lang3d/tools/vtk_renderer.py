"""VTK offscreen renderer — deterministic, headless 3D rendering.

Replaces the screenshot-based approach (SetForegroundWindow + ImageGrab.grab)
with VTK's SetOffScreenRendering(1). No display, no window management,
no timing assumptions, no Windows-specific APIs.

Renders STL meshes or dimension-based approximations (box/cylinder) from
multiple camera angles and saves PNG files for VLM analysis.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Camera presets (Z-up coordinate system matching FreeCAD)
# ---------------------------------------------------------------------------
VIEW_PRESETS: dict[str, dict[str, tuple[float, float, float]]] = {
    "isometric": {"position": (350, -350, 250), "focal": (0, 0, 50), "up": (0, 0, 1)},
    "front": {"position": (0, -450, 50), "focal": (0, 0, 50), "up": (0, 0, 1)},
    "top": {"position": (0, 0, 500), "focal": (0, 0, 0), "up": (0, -1, 0)},
    "right": {"position": (450, 0, 50), "focal": (0, 0, 50), "up": (0, 0, 1)},
}

# Default subsystem colors (RGB, 0-1)
SUBSYSTEM_COLORS: dict[str, tuple[float, float, float]] = {
    "chassis": (0.20, 0.40, 0.80),
    "arm_left": (0.85, 0.30, 0.20),
    "arm_right": (0.20, 0.75, 0.30),
    "ipc": (0.75, 0.55, 0.10),
    "sensor_tower": (0.60, 0.20, 0.75),
    "default": (0.65, 0.65, 0.65),
}

# Category-based colors for better visual distinction of part types
CATEGORY_COLORS: dict[str, tuple[float, float, float]] = {
    "structural": (0.20, 0.45, 0.80),   # blue
    "actuator": (0.85, 0.25, 0.20),     # red
    "mechanical": (0.15, 0.70, 0.25),   # green
    "electronics": (0.85, 0.65, 0.10),  # amber
    "sensor": (0.60, 0.20, 0.75),       # purple
}

# Category-based material properties (specular, specular_power, ambient, diffuse)
CATEGORY_MATERIALS: dict[str, dict[str, float]] = {
    "structural": {"specular": 0.4, "specular_power": 30, "ambient": 0.15, "diffuse": 0.85},
    "actuator": {"specular": 0.3, "specular_power": 20, "ambient": 0.10, "diffuse": 0.80},
    "mechanical": {"specular": 0.5, "specular_power": 40, "ambient": 0.15, "diffuse": 0.85},
    "electronics": {"specular": 0.1, "specular_power": 5, "ambient": 0.20, "diffuse": 0.80},
    "sensor": {"specular": 0.2, "specular_power": 15, "ambient": 0.15, "diffuse": 0.80},
}


def _default_color(index: int) -> tuple[float, float, float]:
    """Generate a distinct color by index using golden-ratio hue spacing."""
    hue = (index * 0.618033988749895) % 1.0
    # HSV to RGB with S=0.6, V=0.8
    s, v = 0.6, 0.8
    c = v * s
    x = c * (1 - abs((hue * 6) % 2 - 1))
    m = v - c
    if hue < 1 / 6:
        r, g, b = c, x, 0
    elif hue < 2 / 6:
        r, g, b = x, c, 0
    elif hue < 3 / 6:
        r, g, b = 0, c, x
    elif hue < 4 / 6:
        r, g, b = 0, x, c
    elif hue < 5 / 6:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (r + m, g + m, b + m)


def _infer_subsystem(part_name: str) -> str:
    """Infer subsystem from part name for color mapping."""
    n = part_name.lower()
    if n.startswith("arm_l_"):
        return "arm_left"
    if n.startswith("arm_r_"):
        return "arm_right"
    if "ipc" in n:
        return "ipc"
    if "sensor" in n or "imu" in n or "lidar" in n or "camera" in n:
        return "sensor_tower"
    # Everything else is chassis
    return "chassis"


# ---------------------------------------------------------------------------
# VTKOffscreenRenderer
# ---------------------------------------------------------------------------


class VTKOffscreenRenderer:
    """Headless 3D renderer using VTK's offscreen render window.

    Usage::

        renderer = VTKOffscreenRenderer()
        renderer.load_stl("part.stl", color=(0.2, 0.5, 0.85))
        png_paths = renderer.render_all_views("/tmp/renders")
        # -> ["/tmp/renders/isometric.png", "/tmp/renders/front.png", ...]
    """

    def __init__(self, width: int = 1200, height: int = 900) -> None:
        self.width = width
        self.height = height
        self._actors: list[Any] = []
        self._content_actors: list[Any] = []  # assembly parts only (excludes grid/axes)
        self._has_content = False

    # ------------------------------------------------------------------
    # Adding geometry
    # ------------------------------------------------------------------

    def _apply_transform(
        self,
        actor: Any,
        position: tuple[float, float, float] = (0, 0, 0),
        rotation: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Apply rotation then translation to a VTK actor via UserTransform.

        VTK's default is PostMultiply, where each new operation is
        post-multiplied: M = M_old * Op.  To get the effective matrix
        T(pos) * R (rotate around local origin, then translate), we must
        call Translate *first* and RotateWXYZ *second*:

            M = I * T(pos)       →  M = T(pos)
            M = T(pos) * R       →  M = T(pos) * R

        Applied to vertex v: T(pos) * R * v  (rotate first, translate second).
        """
        import vtk

        if rotation is None or (rotation[3] == 0.0):
            actor.SetPosition(*position)
            return

        ax, ay, az, angle_deg = rotation
        transform = vtk.vtkTransform()
        # PostMultiply (VTK default): each new op is post-multiplied.
        # To get T(pos) * R, call Translate first, then RotateWXYZ.
        transform.Translate(*position)
        transform.RotateWXYZ(angle_deg, ax, ay, az)
        actor.SetUserTransform(transform)

    def _apply_material(self, actor: Any, category: str = "") -> None:
        """Apply category-based material properties to an actor."""
        mat = CATEGORY_MATERIALS.get(category)
        if mat:
            actor.GetProperty().SetSpecular(mat["specular"])
            actor.GetProperty().SetSpecularPower(mat["specular_power"])
            actor.GetProperty().SetAmbient(mat["ambient"])
            actor.GetProperty().SetDiffuse(mat["diffuse"])
        else:
            actor.GetProperty().SetSpecular(0.2)
            actor.GetProperty().SetSpecularPower(10)
            actor.GetProperty().SetAmbient(0.15)
            actor.GetProperty().SetDiffuse(0.85)

    def load_stl(
        self,
        stl_path: str,
        color: tuple[float, float, float] = (0.65, 0.65, 0.65),
        opacity: float = 1.0,
        position: tuple[float, float, float] | None = None,
        rotation: tuple[float, float, float, float] | None = None,
        category: str = "",
    ) -> None:
        """Load an STL file and add it to the scene."""
        import vtk  # lazy import — only needed when rendering

        if not os.path.isfile(stl_path):
            raise FileNotFoundError(f"STL file not found: {stl_path}")

        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_path)
        reader.Update()

        # Center the mesh so its bounding box center is at the origin.
        # FreeCAD generates STL with corner at (0,0,0), but the solver
        # and VTK's box/cylinder approximations assume centered geometry.
        center_filter = vtk.vtkCenterOfMass()
        center_filter.SetInputConnection(reader.GetOutputPort())
        center_filter.Update()
        cx, cy, cz = center_filter.GetCenter()

        shift = vtk.vtkTransform()
        shift.Translate(-cx, -cy, -cz)
        shift_filter = vtk.vtkTransformPolyDataFilter()
        shift_filter.SetInputConnection(reader.GetOutputPort())
        shift_filter.SetTransform(shift)
        shift_filter.Update()

        # Smooth normals for nice shading
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(shift_filter.GetOutputPort())
        normals.ComputePointNormalsOn()
        normals.SplittingOff()
        normals.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        self._apply_material(actor, category)
        actor.GetProperty().SetInterpolationToPhong()

        self._apply_transform(actor, position or (0, 0, 0), rotation)

        self._actors.append(actor)
        self._content_actors.append(actor)
        self._has_content = True

    def add_box(
        self,
        length: float,
        width: float,
        height: float,
        color: tuple[float, float, float] = (0.65, 0.65, 0.65),
        opacity: float = 1.0,
        position: tuple[float, float, float] = (0, 0, 0),
        rotation: tuple[float, float, float, float] | None = None,
        category: str = "",
    ) -> None:
        """Add a box (approximation from dimensions)."""
        import vtk

        source = vtk.vtkCubeSource()
        source.SetXLength(length)
        source.SetYLength(width)
        source.SetZLength(height)
        source.SetCenter(0, 0, 0)
        source.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        self._apply_material(actor, category)
        actor.GetProperty().SetInterpolationToPhong()
        self._apply_transform(actor, position, rotation)

        self._actors.append(actor)
        self._content_actors.append(actor)
        self._has_content = True

    def add_cylinder(
        self,
        radius: float,
        height: float,
        resolution: int = 32,
        color: tuple[float, float, float] = (0.65, 0.65, 0.65),
        opacity: float = 1.0,
        position: tuple[float, float, float] = (0, 0, 0),
        rotation: tuple[float, float, float, float] | None = None,
        category: str = "",
    ) -> None:
        """Add a cylinder along the Z axis (approximation from dimensions)."""
        import vtk

        source = vtk.vtkCylinderSource()
        source.SetRadius(radius)
        source.SetHeight(height)
        source.SetResolution(resolution)
        # vtkCylinderSource aligns along Y by default; rotate to Z
        source.Update()

        transform = vtk.vtkTransform()
        transform.RotateX(90)  # Y -> Z

        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(source.GetOutputPort())
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(transform_filter.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        self._apply_material(actor, category)
        actor.GetProperty().SetInterpolationToPhong()
        self._apply_transform(actor, position, rotation)

        self._actors.append(actor)
        self._content_actors.append(actor)
        self._has_content = True

    def add_axes(self, length: float = 30, label_offset: float = 5) -> None:
        """Add XYZ axis indicator in the corner."""
        import vtk

        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(length, length, length)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.02)

        self._actors.append(axes)
        # Don't set _has_content for axes

    def add_floor_grid(
        self, size: float = 400, spacing: float = 50, z_position: float = 0
    ) -> None:
        """Add a ground-plane grid (XY plane) at z=z_position."""
        import vtk

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        half = size / 2
        n_lines = int(size / spacing) + 1
        idx = 0
        # Lines parallel to Y axis (at various X positions)
        for i in range(n_lines):
            x = -half + i * spacing
            points.InsertNextPoint(x, -half, z_position)
            points.InsertNextPoint(x, half, z_position)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, idx)
            line.GetPointIds().SetId(1, idx + 1)
            lines.InsertNextCell(line)
            idx += 2

        # Lines parallel to X axis (at various Y positions)
        for i in range(n_lines):
            y = -half + i * spacing
            points.InsertNextPoint(-half, y, z_position)
            points.InsertNextPoint(half, y, z_position)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, idx)
            line.GetPointIds().SetId(1, idx + 1)
            lines.InsertNextCell(line)
            idx += 2

        grid = vtk.vtkPolyData()
        grid.SetPoints(points)
        grid.SetLines(lines)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(grid)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.55, 0.55, 0.55)
        actor.GetProperty().SetLineWidth(1.0)
        actor.GetProperty().SetOpacity(0.5)

        self._actors.append(actor)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _build_renderer(self) -> Any:
        """Build a vtkRenderer with all actors and lighting."""
        import vtk

        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.95, 0.95, 0.97)    # top
        renderer.SetBackground2(0.85, 0.85, 0.88)    # bottom
        renderer.GradientBackgroundOn()
        renderer.SetAmbient(0.25, 0.25, 0.25)

        # Key light
        light1 = vtk.vtkLight()
        light1.SetPosition(80, -60, 120)
        light1.SetFocalPoint(0, 0, 50)
        light1.SetIntensity(0.9)
        light1.SetLightTypeToSceneLight()
        renderer.AddLight(light1)

        # Fill light (softer, opposite side)
        light2 = vtk.vtkLight()
        light2.SetPosition(-60, 80, 80)
        light2.SetFocalPoint(0, 0, 50)
        light2.SetIntensity(0.4)
        light2.SetLightTypeToSceneLight()
        renderer.AddLight(light2)

        # Rim light (back-top, for edge definition)
        light3 = vtk.vtkLight()
        light3.SetPosition(0, 0, 200)
        light3.SetFocalPoint(0, 0, 50)
        light3.SetIntensity(0.3)
        light3.SetLightTypeToSceneLight()
        renderer.AddLight(light3)

        for actor in self._actors:
            renderer.AddActor(actor)

        return renderer

    def _compute_bounds(self, renderer: Any) -> tuple[float, float, float, float, float, float]:
        """Compute bounding box of content actors only (excludes floor grid/axes).

        This ensures the auto-framing camera positions itself to show the
        assembly, not the reference grid.
        """
        import vtk

        # Build a temporary renderer with only content actors for bounds
        temp = vtk.vtkRenderer()
        for actor in self._content_actors:
            temp.AddActor(actor)
        temp.ResetCamera()
        return temp.ComputeVisiblePropBounds()

    def render_to_file(self, view_name: str, output_path: str) -> str:
        """Render a single view to a PNG file. Returns the output path.

        Uses auto-framing: the camera focal point is set to the center of the
        scene bounding box, and the camera distance is computed from the scene
        size so the entire assembly is always visible.  The viewing *direction*
        and up vector come from VIEW_PRESETS.
        """
        import vtk

        if not self._has_content:
            raise RuntimeError("No geometry added to renderer")

        preset = VIEW_PRESETS.get(view_name, VIEW_PRESETS["isometric"])

        renderer = self._build_renderer()

        # --- Auto-frame based on actual actor bounds ---
        xmin, xmax, ymin, ymax, zmin, zmax = self._compute_bounds(renderer)

        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        cz = (zmin + zmax) / 2

        # Scene extent
        sx = xmax - xmin
        sy = ymax - ymin
        sz = zmax - zmin
        max_extent = max(sx, sy, sz)

        # View direction from preset (unit vector)
        px, py, pz = preset["position"]
        fx, fy, fz = preset["focal"]
        dx, dy, dz = px - fx, py - fy, pz - fz
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        dir_x, dir_y, dir_z = dx / norm, dy / norm, dz / norm

        # Camera distance: enough to fit the full scene with padding.
        # Factor accounts for perspective FOV and ensures padding.
        distance = max_extent * 1.8

        camera = renderer.GetActiveCamera()
        camera.SetPosition(cx + dir_x * distance, cy + dir_y * distance, cz + dir_z * distance)
        camera.SetFocalPoint(cx, cy, cz)
        camera.SetViewUp(*preset["up"])

        # Use parallel (orthographic) projection for engineering renders.
        # This avoids perspective distortion where distant parts appear tiny.
        # ParallelScale = half the visible world height.
        # For 1200x900 image (aspect 4:3), visible height = 2*scale,
        # visible width = 2*scale * 4/3.  With 20% padding:
        camera.SetParallelProjection(1)
        camera.SetParallelScale(max(max_extent * 0.6, 50))  # at least 50mm visible

        rw = vtk.vtkRenderWindow()
        rw.SetOffScreenRendering(1)
        rw.SetSize(self.width, self.height)
        rw.SetMultiSamples(4)  # MSAA 4x
        rw.AddRenderer(renderer)
        rw.Render()

        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(rw)
        w2i.Update()

        png = vtk.vtkPNGWriter()
        png.SetFileName(output_path)
        png.SetInputConnection(w2i.GetOutputPort())
        png.Write()

        return output_path

    def render_all_views(
        self,
        output_dir: str,
        views: list[str] | None = None,
        prefix: str = "",
    ) -> list[str]:
        """Render multiple views. Returns list of PNG file paths.

        Args:
            output_dir: Directory to save PNG files.
            views: List of view names from VIEW_PRESETS. Default: all 4.
            prefix: Optional filename prefix (e.g. part name).

        Returns:
            List of absolute paths to rendered PNG files.
        """
        if views is None:
            views = list(VIEW_PRESETS.keys())

        os.makedirs(output_dir, exist_ok=True)
        paths: list[str] = []

        for view_name in views:
            fname = f"{prefix}_{view_name}.png" if prefix else f"{view_name}.png"
            output_path = os.path.join(output_dir, fname)
            self.render_to_file(view_name, output_path)
            paths.append(output_path)

        return paths

    def clear(self) -> None:
        """Remove all actors, reset for reuse."""
        self._actors.clear()
        self._content_actors.clear()
        self._has_content = False


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def render_stl_multi_angle(
    stl_path: str,
    output_dir: str,
    views: list[str] | None = None,
    color: tuple[float, float, float] = (0.65, 0.65, 0.65),
    width: int = 1200,
    height: int = 900,
) -> list[str]:
    """One-shot: load one STL, render multiple views.

    Returns list of PNG file paths.
    """
    r = VTKOffscreenRenderer(width=width, height=height)
    r.load_stl(stl_path, color=color)
    r.add_axes()
    return r.render_all_views(output_dir, views=views)


def render_assembly_from_positions(
    parts: list[dict[str, Any]],
    positions: dict[str, dict[str, Any]],
    output_dir: str,
    views: list[str] | None = None,
    stl_dir: str | None = None,
    width: int = 1200,
    height: int = 900,
) -> list[str]:
    """Render an assembly from part definitions and solved positions.

    Args:
        parts: List of part dicts with 'name', 'dimensions', optionally 'category'.
        positions: Dict of part_name -> {'position': [x,y,z], 'rotation': [ax,ay,az,angle]}.
        output_dir: Where to save PNGs.
        stl_dir: Optional directory to look for STL files. If found, uses real mesh.
        views: Camera angle names.
        width/height: Image resolution.

    Returns:
        List of rendered PNG file paths.
    """
    r = VTKOffscreenRenderer(width=width, height=height)

    for idx, part in enumerate(parts):
        name = part.get("name", f"part_{idx}")
        dims = part.get("dimensions", {})
        category = part.get("category", "")
        # Use category-based color for better visual distinction,
        # fall back to subsystem inference then index-based color
        if category in CATEGORY_COLORS:
            color = CATEGORY_COLORS[category]
        else:
            subsystem = _infer_subsystem(name)
            color = SUBSYSTEM_COLORS.get(subsystem, _default_color(idx))
        pos_data = positions.get(name, {})
        pos = pos_data.get("position", [0, 0, 0])
        rot = pos_data.get("rotation", None)
        # Convert rotation list to tuple if present
        rot_tuple = tuple(rot) if rot and rot[3] != 0.0 else None

        # Try STL first (real geometry)
        stl_loaded = False
        if stl_dir:
            stl_path = os.path.join(stl_dir, f"{name}.stl")
            if os.path.isfile(stl_path):
                r.load_stl(stl_path, color=color, position=tuple(pos), rotation=rot_tuple, category=category)
                stl_loaded = True

        # Fallback to dimension-based approximation
        if not stl_loaded:
            _add_dimension_approximation(r, dims, color, pos, rot_tuple, category=category)

    r.add_axes(length=25)
    r.add_floor_grid(size=400, spacing=50, z_position=0)
    return r.render_all_views(output_dir, views=views)


def _add_dimension_approximation(
    renderer: VTKOffscreenRenderer,
    dims: dict[str, float],
    color: tuple[float, float, float],
    position: list[float],
    rotation: tuple[float, float, float, float] | None = None,
    category: str = "",
) -> None:
    """Add a box or cylinder approximation based on dimension dict.

    For box parts with a significant Y-axis rotation (typical of arm joints),
    the length and height dimensions are swapped so the long axis ends up
    horizontal after the actor rotation is applied.
    """
    if "outer_diameter" in dims or "diameter" in dims:
        d = dims.get("outer_diameter", dims.get("diameter", 10))
        h = dims.get("height", dims.get("length", 10))
        renderer.add_cylinder(
            radius=d / 2, height=h, color=color, position=tuple(position),
            rotation=rotation, category=category,
        )
    elif "length" in dims and "width" in dims:
        l = dims["length"]
        w = dims["width"]
        h = dims.get("height", dims.get("thickness", 5))
        # Detect Y-axis rotation (arm joints rotate around Y).
        # R_Y(±90°) swaps X and Z, so swap length↔height so the
        # long axis (length) ends up horizontal after rotation.
        swap_axes = False
        if rotation is not None and abs(rotation[1]) > 0.5 and abs(rotation[3]) > 30:
            swap_axes = True
        if swap_axes:
            l, h = h, l
        renderer.add_box(
            length=l, width=w, height=h, color=color, position=tuple(position),
            rotation=rotation, category=category,
        )
    else:
        # Minimal fallback
        renderer.add_box(10, 10, 10, color=color, position=tuple(position),
                         rotation=rotation, category=category)
