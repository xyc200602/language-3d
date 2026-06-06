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
    "isometric": {"position": (130, 130, 100), "focal": (0, 0, 50), "up": (0, 0, 1)},
    "front": {"position": (0, -200, 50), "focal": (0, 0, 50), "up": (0, 0, 1)},
    "top": {"position": (0, 1, 250), "focal": (0, 0, 0), "up": (0, 1, 0)},
    "right": {"position": (200, 0, 50), "focal": (0, 0, 50), "up": (0, 0, 1)},
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
        self._has_content = False

    # ------------------------------------------------------------------
    # Adding geometry
    # ------------------------------------------------------------------

    def load_stl(
        self,
        stl_path: str,
        color: tuple[float, float, float] = (0.65, 0.65, 0.65),
        opacity: float = 1.0,
        position: tuple[float, float, float] | None = None,
    ) -> None:
        """Load an STL file and add it to the scene."""
        import vtk  # lazy import — only needed when rendering

        if not os.path.isfile(stl_path):
            raise FileNotFoundError(f"STL file not found: {stl_path}")

        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_path)
        reader.Update()

        # Smooth normals for nice shading
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(reader.GetOutputPort())
        normals.ComputePointNormalsOn()
        normals.SplittingOff()
        normals.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetSpecular(0.3)
        actor.GetProperty().SetSpecularPower(20)
        actor.GetProperty().SetInterpolationToPhong()

        if position is not None:
            actor.SetPosition(*position)

        self._actors.append(actor)
        self._has_content = True

    def add_box(
        self,
        length: float,
        width: float,
        height: float,
        color: tuple[float, float, float] = (0.65, 0.65, 0.65),
        opacity: float = 1.0,
        position: tuple[float, float, float] = (0, 0, 0),
    ) -> None:
        """Add an axis-aligned box (approximation from dimensions)."""
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
        actor.GetProperty().SetSpecular(0.2)
        actor.GetProperty().SetSpecularPower(10)
        actor.GetProperty().SetInterpolationToPhong()
        actor.SetPosition(*position)

        self._actors.append(actor)
        self._has_content = True

    def add_cylinder(
        self,
        radius: float,
        height: float,
        resolution: int = 32,
        color: tuple[float, float, float] = (0.65, 0.65, 0.65),
        opacity: float = 1.0,
        position: tuple[float, float, float] = (0, 0, 0),
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
        actor.GetProperty().SetSpecular(0.2)
        actor.GetProperty().SetSpecularPower(10)
        actor.GetProperty().SetInterpolationToPhong()
        actor.SetPosition(*position)

        self._actors.append(actor)
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
        self, size: float = 400, spacing: float = 50, y_position: float = 0
    ) -> None:
        """Add a ground-plane grid at y=y_position."""
        import vtk

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        half = size / 2
        n_lines = int(size / spacing) + 1
        idx = 0
        for i in range(n_lines):
            x = -half + i * spacing
            points.InsertNextPoint(x, y_position, -half)
            points.InsertNextPoint(x, y_position, half)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, idx)
            line.GetPointIds().SetId(1, idx + 1)
            lines.InsertNextCell(line)
            idx += 2

        for i in range(n_lines):
            z = -half + i * spacing
            points.InsertNextPoint(-half, y_position, z)
            points.InsertNextPoint(half, y_position, z)
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
        actor.GetProperty().SetColor(0.75, 0.75, 0.75)
        actor.GetProperty().SetLineWidth(0.5)
        actor.GetProperty().SetOpacity(0.4)

        self._actors.append(actor)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _build_renderer(self) -> Any:
        """Build a vtkRenderer with all actors and lighting."""
        import vtk

        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.96, 0.96, 0.96)
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

        for actor in self._actors:
            renderer.AddActor(actor)

        return renderer

    def render_to_file(self, view_name: str, output_path: str) -> str:
        """Render a single view to a PNG file. Returns the output path."""
        import vtk

        if not self._has_content:
            raise RuntimeError("No geometry added to renderer")

        preset = VIEW_PRESETS.get(view_name, VIEW_PRESETS["isometric"])

        renderer = self._build_renderer()

        camera = renderer.GetActiveCamera()
        camera.SetPosition(*preset["position"])
        camera.SetFocalPoint(*preset["focal"])
        camera.SetViewUp(*preset["up"])
        camera.SetParallelProjection(0)  # perspective
        renderer.ResetCamera()

        rw = vtk.vtkRenderWindow()
        rw.SetOffScreenRendering(1)
        rw.SetSize(self.width, self.height)
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
        subsystem = part.get("category", "")
        pos_data = positions.get(name, {})
        pos = pos_data.get("position", [0, 0, 0])
        color = SUBSYSTEM_COLORS.get(subsystem, _default_color(idx))

        # Try STL first (real geometry)
        stl_loaded = False
        if stl_dir:
            stl_path = os.path.join(stl_dir, f"{name}.stl")
            if os.path.isfile(stl_path):
                r.load_stl(stl_path, color=color, position=tuple(pos))
                stl_loaded = True

        # Fallback to dimension-based approximation
        if not stl_loaded:
            _add_dimension_approximation(r, dims, color, pos)

    r.add_axes(length=25)
    return r.render_all_views(output_dir, views=views)


def _add_dimension_approximation(
    renderer: VTKOffscreenRenderer,
    dims: dict[str, float],
    color: tuple[float, float, float],
    position: list[float],
) -> None:
    """Add a box or cylinder approximation based on dimension dict."""
    if "outer_diameter" in dims or "diameter" in dims:
        d = dims.get("outer_diameter", dims.get("diameter", 10))
        h = dims.get("height", dims.get("length", 10))
        renderer.add_cylinder(
            radius=d / 2, height=h, color=color, position=tuple(position)
        )
    elif "length" in dims and "width" in dims:
        l = dims["length"]
        w = dims["width"]
        h = dims.get("height", dims.get("thickness", 5))
        renderer.add_box(
            length=l, width=w, height=h, color=color, position=tuple(position)
        )
    else:
        # Minimal fallback
        renderer.add_box(10, 10, 10, color=color, position=tuple(position))
