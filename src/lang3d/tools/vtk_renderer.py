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

# Category-based colors — distinct industrial palette for VLM readability.
# Key principle: each category must be visually distinguishable so the VLM
# (and humans) can tell joints from links from gripper at a glance.
CATEGORY_COLORS: dict[str, tuple[float, float, float]] = {
    "structural": (0.72, 0.74, 0.78),   # bright aluminum silver (links, base)
    "actuator": (0.80, 0.50, 0.12),     # industrial orange (servos/joints — like Dynamixel)
    "mechanical": (0.78, 0.22, 0.18),   # red (gripper/end effector — stands out)
    "electronics": (0.15, 0.50, 0.22),  # PCB green
    "sensor": (0.22, 0.52, 0.82),       # sensor blue
    "bearing": (0.85, 0.86, 0.90),      # chrome silver
}

# Category-based material properties (specular, specular_power, ambient, diffuse)
CATEGORY_MATERIALS: dict[str, dict[str, float]] = {
    "structural": {"specular": 0.55, "specular_power": 55, "ambient": 0.18, "diffuse": 0.82},
    "actuator": {"specular": 0.45, "specular_power": 35, "ambient": 0.15, "diffuse": 0.85},
    "mechanical": {"specular": 0.35, "specular_power": 25, "ambient": 0.16, "diffuse": 0.84},
    "electronics": {"specular": 0.20, "specular_power": 10, "ambient": 0.20, "diffuse": 0.80},
    "sensor": {"specular": 0.30, "specular_power": 25, "ambient": 0.16, "diffuse": 0.84},
    "bearing": {"specular": 0.75, "specular_power": 90, "ambient": 0.14, "diffuse": 0.78},
}

# Edge color per category — dark version of base color for visible geometry edges
CATEGORY_EDGE_COLORS: dict[str, tuple[float, float, float]] = {
    "structural": (0.32, 0.34, 0.38),
    "actuator": (0.35, 0.20, 0.05),
    "mechanical": (0.32, 0.08, 0.06),
    "electronics": (0.06, 0.22, 0.10),
    "sensor": (0.08, 0.22, 0.35),
    "bearing": (0.40, 0.41, 0.44),
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

    def _apply_material(self, actor: Any, category: str = "",
                        show_edges: bool = False) -> None:
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
        # Only show edges for simple primitives (box/cylinder),
        # NOT for STL meshes which have dense triangulation
        if show_edges:
            edge_color = CATEGORY_EDGE_COLORS.get(category, (0.25, 0.25, 0.25))
            actor.GetProperty().EdgeVisibilityOn()
            actor.GetProperty().SetEdgeColor(*edge_color)
            actor.GetProperty().SetLineWidth(1.0)

    def load_stl(
        self,
        stl_path: str,
        color: tuple[float, float, float] = (0.65, 0.65, 0.65),
        opacity: float = 1.0,
        position: tuple[float, float, float] | None = None,
        rotation: tuple[float, float, float, float] | None = None,
        category: str = "",
        swap_xy: bool = False,
    ) -> None:
        """Load an STL file and add it to the scene.

        Args:
            swap_xy: If True, apply R_z(90°) to swap X↔Y axes.
                     Needed for FreeCAD-generated STL files where
                     makeBox puts length along X, but the solver's
                     convention puts length along Y (front/back).
        """
        import vtk  # lazy import — only needed when rendering

        if not os.path.isfile(stl_path):
            raise FileNotFoundError(f"STL file not found: {stl_path}")

        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_path)
        reader.Update()

        # Center the mesh so its bounding box center is at the origin.
        # FreeCAD generates STL with corner at (0,0,0), but the solver
        # computes positions based on bounding box half-extents (not
        # centroid), so we must center on the bounding box center to
        # ensure STL parts align exactly where the solver expects them.
        raw = reader.GetOutput()
        bounds = raw.GetBounds()  # [xmin,xmax,ymin,ymax,zmin,zmax]
        cx = (bounds[0] + bounds[1]) / 2.0
        cy = (bounds[2] + bounds[3]) / 2.0
        cz = (bounds[4] + bounds[5]) / 2.0

        shift = vtk.vtkTransform()
        shift.Translate(-cx, -cy, -cz)
        shift_filter = vtk.vtkTransformPolyDataFilter()
        shift_filter.SetInputConnection(reader.GetOutputPort())
        shift_filter.SetTransform(shift)
        shift_filter.Update()

        # Swap X↔Y axes for FreeCAD-generated STL meshes.
        # FreeCAD convention: makeBox(length,width,height) → X=length, Y=width
        # Solver convention: front/back=Y (length), left/right=X (width)
        # R_z(-90°) maps FreeCAD +X → solver -Y (front), so parts extend
        # in the correct direction (away from parent, toward child).
        if swap_xy:
            swap = vtk.vtkTransform()
            swap.RotateZ(-90)
            swap_filter = vtk.vtkTransformPolyDataFilter()
            swap_filter.SetInputConnection(shift_filter.GetOutputPort())
            swap_filter.SetTransform(swap)
            swap_filter.Update()
            last_output = swap_filter.GetOutputPort()
        else:
            last_output = shift_filter.GetOutputPort()

        # Smooth normals for nice shading
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(last_output)
        normals.ComputePointNormalsOn()
        normals.SplittingOff()
        normals.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        self._apply_material(actor, category, show_edges=False)
        actor.GetProperty().SetInterpolationToPhong()

        self._apply_transform(actor, position or (0, 0, 0), rotation)

        self._actors.append(actor)
        self._content_actors.append(actor)
        self._has_content = True

        # Feature edges: extract sharp geometric edges (>55°) and render as
        # thin dark lines. This makes part boundaries and mechanical features
        # (bearing collars, output shafts, finger outlines) visible to the VLM
        # without the visual noise of showing every STL triangle edge.
        try:
            edges = vtk.vtkFeatureEdges()
            edges.SetInputConnection(normals.GetOutputPort())
            edges.BoundaryEdgesOff()
            edges.ManifoldEdgesOff()
            edges.NonManifoldEdgesOff()
            edges.FeatureEdgesOn()
            edges.SetFeatureAngle(55.0)
            edges.Update()

            if edges.GetOutput().GetNumberOfCells() > 0:
                edge_mapper = vtk.vtkPolyDataMapper()
                edge_mapper.SetInputConnection(edges.GetOutputPort())

                edge_actor = vtk.vtkActor()
                edge_actor.SetMapper(edge_mapper)
                edge_color = CATEGORY_EDGE_COLORS.get(category, (0.2, 0.2, 0.2))
                edge_actor.GetProperty().SetColor(*edge_color)
                edge_actor.GetProperty().SetLineWidth(2.0)
                edge_actor.GetProperty().SetOpacity(opacity)

                self._apply_transform(edge_actor, position or (0, 0, 0), rotation)

                self._actors.append(edge_actor)
                self._content_actors.append(edge_actor)
        except Exception:
            pass  # Feature edges are cosmetic — never block rendering

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

        # Solver convention: X=left/right(width), Y=front/back(length), Z=top/bottom(height)
        # This matches ANCHOR_DIRECTIONS: front=(0,-1,0), left=(-1,0,0), top=(0,0,1)
        source = vtk.vtkCubeSource()
        source.SetXLength(width)
        source.SetYLength(length)
        source.SetZLength(height)
        source.SetCenter(0, 0, 0)
        source.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        self._apply_material(actor, category, show_edges=True)
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
        self._apply_material(actor, category, show_edges=True)
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

    def _build_renderer(self, centroid: tuple[float, float, float] = (0, 0, 50)) -> Any:
        """Build a vtkRenderer with all actors and lighting.

        Lights are positioned relative to *centroid* so the entire assembly
        is well-lit regardless of where it sits in world coordinates.
        """
        import vtk

        cx, cy, cz = centroid

        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.95, 0.95, 0.97)    # top
        renderer.SetBackground2(0.85, 0.85, 0.88)    # bottom
        renderer.GradientBackgroundOn()
        renderer.SetAmbient(0.28, 0.28, 0.28)

        # Key light — warm, upper-right-front (relative to scene centroid)
        light1 = vtk.vtkLight()
        light1.SetPosition(cx + 100, cy - 80, cz + 150)
        light1.SetFocalPoint(cx, cy, cz)
        light1.SetIntensity(1.0)
        light1.SetColor(1.0, 0.98, 0.95)
        light1.SetLightTypeToSceneLight()
        renderer.AddLight(light1)

        # Fill light — cool, upper-left-back
        light2 = vtk.vtkLight()
        light2.SetPosition(cx - 80, cy + 100, cz + 100)
        light2.SetFocalPoint(cx, cy, cz)
        light2.SetIntensity(0.5)
        light2.SetColor(0.92, 0.95, 1.0)
        light2.SetLightTypeToSceneLight()
        renderer.AddLight(light2)

        # Rim/back light — from behind for silhouette edge definition
        light3 = vtk.vtkLight()
        light3.SetPosition(cx - 30, cy - 30, cz + 200)
        light3.SetFocalPoint(cx, cy, cz)
        light3.SetIntensity(0.4)
        light3.SetColor(1.0, 1.0, 1.0)
        light3.SetLightTypeToSceneLight()
        renderer.AddLight(light3)

        for actor in self._actors:
            renderer.AddActor(actor)

        return renderer

    def _compute_bounds_raw(self) -> tuple[float, float, float, float, float, float]:
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

    # Alias for backwards compatibility
    _compute_bounds = _compute_bounds_raw

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

        # First compute bounds to determine scene centroid for lighting.
        xmin, xmax, ymin, ymax, zmin, zmax = self._compute_bounds_raw()

        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        cz = (zmin + zmax) / 2

        # Scene extent
        sx = xmax - xmin
        sy = ymax - ymin
        sz = zmax - zmin
        max_extent = max(sx, sy, sz)

        # Build renderer with lights positioned at the scene centroid.
        renderer = self._build_renderer(centroid=(cx, cy, cz))

        # View direction from preset (unit vector)
        px, py, pz = preset["position"]
        fx, fy, fz = preset["focal"]
        dx, dy, dz = px - fx, py - fy, pz - fz
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        dir_x, dir_y, dir_z = dx / norm, dy / norm, dz / norm

        # Camera distance: enough to fit the full scene with padding.
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


def _add_fasteners_for_joints(
    renderer: VTKOffscreenRenderer,
    joints: list,
    positions: dict[str, dict],
    parts: list[dict],
) -> None:
    """Add bolt/washer/nut cylinders at REAL bolt hole positions for VTK rendering.

    Uses :func:`_compute_bolt_hole_world_positions` to place each fastener at
    the actual hole location on the structural part's anchor face, oriented
    along the face normal.  This replaces the previous midpoint + circular
    approximation which placed fasteners between parent and child centres
    rather than at the drilled holes.
    """
    from .freecad import _FASTENER_DIMS, _compute_bolt_hole_world_positions

    # Build part dimensions lookup: name -> dimensions dict
    part_dims: dict[str, dict] = {}
    for p in parts:
        name = p.get("name")
        if name:
            part_dims[name] = p.get("dimensions", {}) or {}

    for joint in joints or []:
        conn = getattr(joint, "connection", None)
        if conn is None or getattr(conn, "type", "") != "bolted":
            continue
        if getattr(joint, "type", "") == "prismatic":
            continue

        # Compute world-space bolt hole positions from the structural part's
        # connection feature layout (same holes that were drilled in the STL).
        bolt_holes = _compute_bolt_hole_world_positions(joint, positions, part_dims)
        if not bolt_holes:
            continue

        bolt_size = getattr(conn, "bolt_size", "M3") or "M3"
        dims = _FASTENER_DIMS.get(bolt_size, _FASTENER_DIMS["M3"])
        head_r, head_h, shank_r, washer_r, washer_h, nut_r, nut_h = dims

        # _FASTENER_DIMS values already match ISO 4762 (socket head cap
        # screws) exactly — M3 head = 5.5 mm diameter, M5 = 8.5 mm, etc.
        # At 1920×1080 render resolution with a ~500 mm arm span that
        # gives ~4 px/mm, so a real M3 head occupies ~22 px and is
        # clearly visible without artificial enlargement.  Keep the
        # nominal dimensions for engineering accuracy.

        for world_pos, normal, thickness in bolt_holes:
            # Bolt length: grip thickness + nut engagement + small thread
            # protrusion.  Washer height is NOT included — the washer is
            # rendered separately below the nut.  Minimum 6 mm so even the
            # thinnest plates get enough shank for the nut to grip.
            length = max(thickness + nut_h + 1.0, 6.0)

            # Rotation to align default-Z cylinder with the anchor normal.
            rot = _rotation_from_z_to(normal)

            nx, ny, nz = normal
            px, py, pz = world_pos
            half_len = length / 2.0

            # Bolt head: sits on the exterior (+normal side) of the face.
            head_d = half_len + head_h / 2.0
            renderer.add_cylinder(
                radius=head_r, height=head_h,
                color=(0.80, 0.80, 0.82),
                position=(px + nx * head_d, py + ny * head_d, pz + nz * head_d),
                rotation=rot,
                category="bearing",
            )
            # Bolt shank: centred on the hole, running along the normal.
            renderer.add_cylinder(
                radius=shank_r, height=length,
                color=(0.75, 0.75, 0.78),
                position=(px, py, pz),
                rotation=rot,
                category="bearing",
            )
            # Washer: on the interior (−normal side) of the face.
            washer_d = half_len + washer_h / 2.0
            renderer.add_cylinder(
                radius=washer_r, height=washer_h,
                color=(0.70, 0.70, 0.72),
                position=(px - nx * washer_d, py - ny * washer_d, pz - nz * washer_d),
                rotation=rot,
                category="bearing",
            )
            # Nut: beyond the washer on the interior side.
            nut_d = half_len + washer_h + nut_h / 2.0
            renderer.add_cylinder(
                radius=nut_r, height=nut_h,
                color=(0.65, 0.65, 0.68),
                position=(px - nx * nut_d, py - ny * nut_d, pz - nz * nut_d),
                rotation=rot,
                category="bearing",
            )


def _rotation_from_z_to(
    normal: tuple[float, float, float],
) -> tuple[float, float, float, float] | None:
    """Axis-angle rotation that maps the +Z direction to *normal*.

    VTK cylinders are created along Z (after the internal RotateX(90)).
    This returns the (ax, ay, az, angle_deg) tuple to pass to
    ``add_cylinder(rotation=...)`` so the cylinder aligns with *normal*.
    Returns None when no rotation is needed (normal already ≈ +Z).
    """
    nx, ny, nz = normal
    dot = nz  # cos(angle) since |(0,0,1)| = 1 and |normal| = 1
    if dot > 0.9999:
        return None
    if dot < -0.9999:
        # 180° rotation — any horizontal axis works; use X.
        return (1.0, 0.0, 0.0, 180.0)
    # Rotation axis = (0,0,1) × normal = (-ny, nx, 0)
    ax_raw, ay_raw = -ny, nx
    al = math.sqrt(ax_raw * ax_raw + ay_raw * ay_raw)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
    return (ax_raw / al, ay_raw / al, 0.0, angle)


def _symmetrize_gripper_fingers(
    parts: list[dict[str, Any]],
    positions: dict[str, dict[str, Any]],
) -> None:
    """Ensure gripper fingers are symmetric, visible, and extend beyond the base.

    The solver applies cumulative rotations along the kinematic chain.  By the
    time the chain reaches the gripper, the rotation can be a complex 3-D
    quaternion.  Two problems arise:

    1. **Asymmetry** — the two fingers end up at different X/Z positions
       because the rotation mixes the Y-offset into other axes differently
       for +Y vs −Y offsets.
    2. **Occlusion** — the solver's local offsets ``[0, ±gap, z_lift]`` may
       place finger *centres* inside the gripper_base bounding box after
       rotation, hiding the fingers inside the base mesh.

    This render-only fixup recomputes finger positions so that:

    * Both fingers share the same X/Z (the midpoint of the solver output).
    * The gap is **at least** ``base_width + 25`` mm so fingers are clearly
      outside the base mesh on both sides.
    * Fingers are pushed slightly further from the base centre along the
      arm's forward direction so they extend beyond the base.
    * The gap direction is the horizontal lateral (perpendicular to the
      arm forward vector), not raw world-Y, so it works for any arm pose.
    """
    import math

    # --- Locate fingers and gripper base ---
    left_name: str | None = None
    right_name: str | None = None
    base_name: str | None = None
    base_width = 50.0
    base_length = 30.0
    for p in parts:
        nl = p.get("name", "").lower()
        if "finger" in nl and "left" in nl:
            left_name = p["name"]
        elif "finger" in nl and "right" in nl:
            right_name = p["name"]
        elif "gripper" in nl and "base" in nl:
            base_name = p["name"]
            dims = p.get("dimensions", {})
            if isinstance(dims, dict):
                base_width = float(dims.get("width", 50.0))
                base_length = float(dims.get("length", 30.0))

    if not left_name or not right_name:
        return

    lp = positions.get(left_name)
    rp = positions.get(right_name)
    if not lp or not rp:
        return

    lpos = lp.get("position", [0, 0, 0])
    rpos = rp.get("position", [0, 0, 0])

    # Midpoint of the two fingers (solver output, before fixup)
    mid_x = (lpos[0] + rpos[0]) / 2.0
    mid_y = (lpos[1] + rpos[1]) / 2.0
    mid_z = (lpos[2] + rpos[2]) / 2.0

    # --- Compute arm forward direction ---
    # Use the vector from the arm root (first part, e.g. base_plate) through
    # the gripper base to approximate the reach direction.  If gripper base
    # position is unavailable, fall back to the midpoint-to-origin vector.
    base_pos: list[float] | None = None
    if base_name:
        bp = positions.get(base_name)
        if bp:
            base_pos = list(bp.get("position", [0, 0, 0]))

    root_pos: list[float] | None = None
    if parts:
        first = parts[0].get("name")
        if first:
            fp = positions.get(first)
            if fp:
                root_pos = list(fp.get("position", [0, 0, 0]))

    fwd = (-1.0, 0.0, 0.0)  # default forward
    if base_pos and root_pos:
        dx = base_pos[0] - root_pos[0]
        dy = base_pos[1] - root_pos[1]
        dz = base_pos[2] - root_pos[2]
        dlen = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dlen > 1.0:
            fwd = (dx / dlen, dy / dlen, dz / dlen)

    # Lateral = forward × world-up (horizontal perpendicular to arm)
    up = (0.0, 0.0, 1.0)
    lat = (
        fwd[1] * up[2] - fwd[2] * up[1],
        fwd[2] * up[0] - fwd[0] * up[2],
        fwd[0] * up[1] - fwd[1] * up[0],
    )
    lat_len = math.sqrt(lat[0] ** 2 + lat[1] ** 2 + lat[2] ** 2)
    if lat_len > 0.01:
        lat = (lat[0] / lat_len, lat[1] / lat_len, lat[2] / lat_len)
    else:
        lat = (0.0, 1.0, 0.0)

    # --- Push fingers further from the base centre ---
    # Extend the base→midpoint vector so fingers are clearly beyond the base
    # mesh.  The fingers are rendered as 50 mm-wide boxes (±25 mm in X from
    # centre), so the push plus fwd_offset places finger centres far enough
    # forward that the back of each finger clears the base.
    if base_pos:
        vx = mid_x - base_pos[0]
        vy = mid_y - base_pos[1]
        vz = mid_z - base_pos[2]
        push = 1.5
        mid_x = base_pos[0] + vx * push
        mid_y = base_pos[1] + vy * push
        mid_z = base_pos[2] + vz * push

    # --- Gap: wide enough that finger centres are outside the base ---
    raw_gap = abs(lpos[1] - rpos[1])
    gap = max(raw_gap, base_width + 30.0, 65.0)

    # Horizontal forward direction (project fwd onto XY plane) for placing
    # fingers so they extend forward without changing their Z height.
    fwd_h_len = math.sqrt(fwd[0] ** 2 + fwd[1] ** 2)
    if fwd_h_len > 0.01:
        fhx = fwd[0] / fwd_h_len
        fhy = fwd[1] / fwd_h_len
    else:
        fhx, fhy = -1.0, 0.0

    # Place fingers symmetrically along the lateral direction.
    # Push finger centres forward horizontally so the finger boxes
    # (rendered as oversized world-axis-aligned prongs) extend beyond
    # the arm tip without overlapping the servo or base.
    fwd_offset = 30.0
    lp["position"] = [
        mid_x + fhx * fwd_offset + lat[0] * (-gap / 2.0),
        mid_y + fhy * fwd_offset + lat[1] * (-gap / 2.0),
        mid_z,
    ]
    rp["position"] = [
        mid_x + fhx * fwd_offset + lat[0] * (gap / 2.0),
        mid_y + fhy * fwd_offset + lat[1] * (gap / 2.0),
        mid_z,
    ]

    # No rotation — fingers are rendered as world-axis-aligned boxes
    # (long along X=forward, narrow along Y=gap, tall along Z).
    lp["rotation"] = None
    rp["rotation"] = None


def render_assembly_from_positions(
    parts: list[dict[str, Any]],
    positions: dict[str, dict[str, Any]],
    output_dir: str,
    views: list[str] | None = None,
    stl_dir: str | None = None,
    width: int = 1200,
    height: int = 900,
    joints: list | None = None,
) -> list[str]:
    """Render an assembly from part definitions and solved positions.

    Args:
        parts: List of part dicts with 'name', 'dimensions', optionally 'category'.
        positions: Dict of part_name -> {'position': [x,y,z], 'rotation': [ax,ay,az,angle]}.
        output_dir: Where to save PNGs.
        stl_dir: Optional directory to look for STL files. If found, uses real mesh.
        views: Camera angle names.
        width/height: Image resolution.
        joints: Optional list of Joint objects.  When provided, bolt/washer/nut
            cylinders are rendered at every bolted connection.

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
        # Finger parts get a high-contrast yellow override so they remain
        # visible against the larger red gripper_base in renders.
        if "finger" in name.lower():
            color = (1.0, 0.85, 0.1)
        pos_data = positions.get(name, {})
        pos = pos_data.get("position", [0, 0, 0])
        rot = pos_data.get("rotation", None)
        # Convert rotation list to tuple if present
        rot_tuple = tuple(rot) if rot and rot[3] != 0.0 else None

        # All parts — including gripper fingers — use swap_xy=True.  FreeCAD
        # generates STLs with length on X; swap_xy (R_z(-90°)) maps FreeCAD
        # +X → solver -Y (front), so the long axis ends up pointing forward
        # along the arm.  For fingers, the L-shaped tips (FreeCAD ±Y) map to
        # solver ∓X, so the left finger tip (at solver -X) curves toward +X
        # and the right finger tip (at solver +X) curves toward -X — the
        # grip surfaces face each other correctly.

        # Try STL first (real geometry)
        stl_loaded = False
        if stl_dir:
            stl_path = os.path.join(stl_dir, f"{name}.stl")
            if os.path.isfile(stl_path):
                r.load_stl(stl_path, color=color, position=tuple(pos), rotation=rot_tuple, category=category, swap_xy=True)
                stl_loaded = True

        # Fallback to dimension-based approximation
        if not stl_loaded:
            _add_dimension_approximation(r, dims, color, pos, rot_tuple, category=category)

    # Fasteners (bolts, washers, nuts) at bolted joints
    if joints:
        _add_fasteners_for_joints(r, joints, positions, parts)

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
        renderer.add_box(
            length=l, width=w, height=h, color=color, position=tuple(position),
            rotation=rotation, category=category,
        )
    else:
        # Minimal fallback
        renderer.add_box(10, 10, 10, color=color, position=tuple(position),
                         rotation=rotation, category=category)
