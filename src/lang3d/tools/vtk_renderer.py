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
    # Gripper close-up: same iso-style direction vector, but the focal
    # point and parallel scale are overridden at render time (via
    # render_assembly_from_positions' gripper_closeup flag) to zoom in on
    # the gripper so 32mm-spaced fingers are clearly visible to the VLM.
    "gripper_closeup": {"position": (350, -350, 250), "focal": (0, 0, 50), "up": (0, 0, 1)},
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

# Arm-link tints — distinct cool blue-grey shades per link position so the
# joint-fold region (where shoulder/elbow/wrist links bend) stays visually
# readable.  Without this, all structural links share CATEGORY_COLORS[
# "structural"] and fuse into one silver blob at the elbow, which VLM and
# humans misread as a collision.
_ARM_LINK_TINTS: dict[str, tuple[float, float, float]] = {
    "shoulder": (0.72, 0.74, 0.78),   # base silver (proximal)
    "upper":    (0.60, 0.66, 0.73),   # slightly darker blue-grey
    "elbow":    (0.52, 0.59, 0.68),   # mid blue-grey
    "forearm":  (0.44, 0.53, 0.64),   # cooler blue-grey
    "wrist":    (0.36, 0.48, 0.62),   # coolest blue-grey (distal)
}


def _link_position_tint(name: str) -> tuple[float, float, float] | None:
    """Return a position-based tint for an arm link, or None if not a link.

    Matches names like ``shoulder_link``, ``upper_arm``, ``elbow_link``,
    ``forearm``, ``wrist_link``.  Requires BOTH a position keyword and a
    link/arm keyword so parts like ``shoulder_bolt`` or ``wrist_joint`` are
    not tinted (joints keep their actuator orange).
    """
    n = name.lower()
    has_link = ("link" in n) or ("arm" in n) or ("fore" in n)
    if not has_link:
        return None
    if "shoulder" in n or "upper" in n:
        return _ARM_LINK_TINTS["shoulder"] if "shoulder" in n else _ARM_LINK_TINTS["upper"]
    if "elbow" in n:
        return _ARM_LINK_TINTS["elbow"]
    if "fore" in n or "lower" in n:
        return _ARM_LINK_TINTS["forearm"]
    if "wrist" in n:
        return _ARM_LINK_TINTS["wrist"]
    return None


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

        # --- Geometry repair (added 2026-06-21) ---
        # FreeCAD boolean cuts (bolt holes, counterbores, cavities) routinely
        # produce non-manifold / fragmented meshes.  Non-watertight meshes
        # render as see-through shells in VTK (the gripper finger problem),
        # and inverted faces render as holes.  trimesh.repair fixes winding
        # and fills holes so VTK shows closed surfaces.
        #
        # Triangle-count decimation is handled further down the VTK pipeline
        # (after swap_xy) via vtkQuadricDecimation — trimesh's
        # simplify_quadric_decimation needs the optional ``fast_simplification``
        # package which is not a project dependency, but VTK ships its own.
        #
        # Safe degradation: if trimesh or the repair fails for any reason,
        # fall through to the raw STL so rendering still works (just ugly).
        stl_path = self._repair_stl_for_render(stl_path)

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

        # NOTE (2026-06-21): triangle decimation was attempted here for
        # the 90k-face gripper_base mesh but REMOVED — both
        # vtkQuadricDecimation and vtkDecimatePro crash with an access
        # violation when chained into vtkPolyDataNormals on these
        # non-manifold boolean-cut meshes.  The root fix is upstream:
        # connection_features.py now generates bolt holes with the
        # correct cylinder axis per anchor (see _anchor_rotation), so
        # the boolean cuts no longer fragment the geometry into 90k
        # triangles in the first place.  Decimation is therefore not
        # needed for clean source meshes.  If a fragmented mesh still
        # reaches the renderer, trimesh.repair (above) softens it and
        # the render shows the part with slightly noisy normals rather
        # than crashing.

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

    # ------------------------------------------------------------------
    # STL geometry repair — called by load_stl before VTK ingestion
    # ------------------------------------------------------------------
    @staticmethod
    def _repair_stl_for_render(stl_path: str) -> str:
        """Repair an STL's topology and return a path to the cleaned mesh.

        Returns ``stl_path`` unchanged on any failure (safe degradation —
        rendering continues with the raw, possibly-fragmented mesh rather
        than crashing).  On success, returns a temp-file path holding the
        repaired mesh; the caller's VTK reader consumes it transparently.

        Operations:
        1. ``trimesh.load(process=True)`` — merges duplicate vertices,
           fixes winding on individual faces.
        2. ``trimesh.repair.fill_holes``  — caps open holes so the mesh
           is watertight.  Non-watertight meshes render as see-through
           shells in VTK, which is why the gripper fingers (euler=1)
           looked like thin slivers instead of solid prongs.
        3. ``trimesh.repair.fix_winding`` — ensures outward normals
           (inverted faces also cause shell-like rendering).

        NOTE: triangle-count decimation is intentionally NOT done here.
        trimesh's ``simplify_quadric_decimation`` needs the optional
        ``fast_simplification`` package; VTK's ``vtkQuadricDecimation``
        and ``vtkDecimatePro`` both crash with an access violation on
        the non-manifold boolean-cut meshes when chained into
        ``vtkPolyDataNormals``; and trimesh's voxelize+marching-cubes
        path needs ``scikit-image``.  None of these optional deps are
        project requirements.  The real fix is upstream —
        ``connection_features._anchor_rotation`` now orients bolt-hole
        cylinders correctly per anchor so the boolean cuts no longer
        fragment geometry into 90k triangles in the first place.
        """
        try:
            import tempfile

            import trimesh

            mesh = trimesh.load(stl_path, process=True)
            # Some loads return a Scene or multi-body result; coerce to mesh.
            if not isinstance(mesh, trimesh.Trimesh):
                if hasattr(mesh, "geometry") and len(mesh.geometry) > 0:
                    mesh = list(mesh.geometry.values())[0]
                else:
                    return stl_path

            trimesh.repair.fix_winding(mesh)
            trimesh.repair.fill_holes(mesh)

            # Only write a temp file if repair produced a watertight mesh
            # that differs from the (broken) input — otherwise skip the
            # I/O and let VTK read the original directly.
            if not mesh.is_watertight:
                return stl_path  # repair couldn't help; don't lie about it

            tmp = tempfile.NamedTemporaryFile(
                suffix="_repaired.stl", delete=False, prefix="lang3d_"
            )
            tmp.close()
            mesh.export(tmp.name)
            return tmp.name
        except Exception:
            # Never let repair break rendering — fall back to raw STL.
            return stl_path

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
        orient_axis: str = "z",
    ) -> None:
        """Add a cylinder along the requested axis (approximation from dims).

        ``vtkCylinderSource`` aligns along Y by default. We rotate to the
        requested axis so the cylinder's HEIGHT extends the right way:

        * ``orient_axis="z"`` (default) — height along Z. Used for servos,
          standoffs, vertical shafts (RotateX(90): Y→Z).
        * ``orient_axis="y"`` — height along Y. Used for WHEELS whose axle is
          along Y (axis="y" joint): the cylinder lies on its side so its
          circular faces point ±Y and it rolls along X. No rotation needed
          (Y is the source default).
        * ``orient_axis="x"`` — height along X (RotateZ(90): Y→X).

        Before this parameter existed, ALL cylinders got RotateX(90) (height
        on Z), which stood wheels vertically — a wheel rendered as a tall
        disc instead of lying on its side. That distorted the chassis's
        rendered aspect ratio (wheels read as circles in top view instead
        of narrow rectangles) and is the root cause of "长宽对不上".
        """
        import vtk

        source = vtk.vtkCylinderSource()
        source.SetRadius(radius)
        source.SetHeight(height)
        source.SetResolution(resolution)
        source.Update()

        transform = vtk.vtkTransform()
        if orient_axis == "z":
            transform.RotateX(90)  # Y -> Z (servos, vertical shafts)
        elif orient_axis == "x":
            transform.RotateZ(90)  # Y -> X
        # orient_axis == "y": no rotation (source default = along Y = wheels)

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

    def render_to_file(
        self,
        view_name: str,
        output_path: str,
        focus_point: tuple[float, float, float] | None = None,
        parallel_scale: float | None = None,
        direction: tuple[float, float, float] | None = None,
    ) -> str:
        """Render a single view to a PNG file. Returns the output path.

        Uses auto-framing: the camera focal point is set to the center of the
        scene bounding box, and the camera distance is computed from the scene
        size so the entire assembly is always visible.  The viewing *direction*
        and up vector come from VIEW_PRESETS.

        Overrides (for the gripper close-up view):
          * focus_point — world-space point the camera aims at (default:
            scene bounding-box centre).
          * parallel_scale — half-height of the visible orthographic volume
            in mm (default: ``max(scene_extent * 0.6, 50)``).  A small value
            (e.g. 60) zooms in on the gripper so fingers are VLM-visible.
          * direction — override the camera viewing direction (unit vector
            from camera toward focal point). Default comes from VIEW_PRESETS.
            Used for the gripper close-up so the camera looks PERPENDICULAR
            to the finger-gap axis (two fingers side-by-side, gap visible),
            not INTO the gap (which makes them overlap into a solid block).
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
        if focus_point is not None:
            # Close-up: aim the camera at the gripper, not the scene centre.
            cx, cy, cz = focus_point

        # Scene extent
        sx = xmax - xmin
        sy = ymax - ymin
        sz = zmax - zmin
        max_extent = max(sx, sy, sz)

        # Build renderer with lights positioned at the scene centroid.
        renderer = self._build_renderer(centroid=(cx, cy, cz))

        # View direction from preset (unit vector) — overridable via the
        # ``direction`` kwarg (gripper close-up uses a perpendicular-to-gap
        # direction so two fingers render side-by-side, not overlapped).
        if direction is not None:
            dir_x, dir_y, dir_z = direction
            dnorm = math.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
            if dnorm > 1e-9:
                dir_x, dir_y, dir_z = dir_x / dnorm, dir_y / dnorm, dir_z / dnorm
        else:
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
        # visible width = 2*scale * 4/3.
        #
        # PER-VIEW framing (2026-06-22): each view's camera looks along a
        # specific axis, so the two WORLD axes that project onto the screen
        # are different per view.  Scaling on the global max_extent wastes
        # frame space: a top view of a 150mm-wide × 520mm-long arm used
        # scale=312 (from the 520mm Y span), shrinking the 150mm X width
        # to 7% of the frame — the arm read as a thin vertical line in
        # 93% empty space.  Instead, compute the scale from the TWO axes
        # this view actually displays, so the assembly fills the frame in
        # BOTH dimensions without clipping.
        #
        # Screen-axis mapping (derived from VIEW_PRESETS directions):
        #   top   : width=world X, height=world Y  (looking down -Z)
        #   front : width=world X, height=world Z  (looking +Y, up=+Z)
        #   right : width=world Y, height=world Z  (looking -X, up=+Z)
        #   iso   : width≈X+Y diagonal, height≈Z   (compound; use max of
        #           the three to be safe)
        camera.SetParallelProjection(1)
        if parallel_scale is not None:
            camera.SetParallelScale(parallel_scale)
        else:
            # Compute the assembly's projected extent onto this view's
            # screen axes (screen-right and screen-up) by projecting the
            # world AABB corners through the camera basis.  This handles
            # every view correctly — including iso, where the diagonal
            # camera direction makes screen-right = (0.71, 0.71, 0),
            # so a 150mm-wide × 520mm-long arm projects to 474mm of
            # screen width (not the 150mm a naive X-axis fit would give).
            cam_dir = (dir_x, dir_y, dir_z)
            upv = preset["up"]
            # screen-right = view_dir × up (normalised)
            srx = cam_dir[1]*upv[2] - cam_dir[2]*upv[1]
            sry = cam_dir[2]*upv[0] - cam_dir[0]*upv[2]
            srz = cam_dir[0]*upv[1] - cam_dir[1]*upv[0]
            srn = math.sqrt(srx*srx + sry*sry + srz*srz) or 1.0
            screen_right = (srx/srn, sry/srn, srz/srn)
            # AABB projected half-extents: sum |axis_component| × half-span
            # (the projection of a box onto a unit vector is the sum of
            # absolute component products — standard AABB projection).
            hx, hy, hz = sx/2.0, sy/2.0, sz/2.0
            proj_w = (abs(screen_right[0])*hx + abs(screen_right[1])*hy
                      + abs(screen_right[2])*hz) * 2.0
            proj_h = (abs(upv[0])*hx + abs(upv[1])*hy + abs(upv[2])*hz) * 2.0
            aspect = self.width / self.height
            scale_for_height = proj_h / 2.0
            scale_for_width = proj_w / (2.0 * aspect)
            view_scale = max(scale_for_height, scale_for_width, 50.0)
            # 15% padding so parts don't touch the frame edge.
            view_scale *= 1.15
            camera.SetParallelScale(view_scale)

        # Explicitly set clipping range to span the full scene.  VTK's
        # default auto-computed clipping range can clip thin parts (like
        # the 8mm base_plate) when the camera is far away and the scene
        # is elongated — the base_plate vanished from top-view renders
        # because its Z=[-4,4] sat outside the tight default near/far
        # planes.  Setting [1, distance*3] guarantees every part within
        # the camera's view volume is rendered.
        camera.SetClippingRange(1.0, distance * 3.0)

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

        # Post-render crop: trim empty margins so elongated assemblies
        # (e.g. a 150×546mm arm) fill the frame instead of leaving 30-60%
        # empty space on the short axis.  The VTK orthographic camera
        # must fit the long axis, which inflates the scale and wastes the
        # short axis.  Cropping the rendered PNG to the content bbox
        # (with 5% padding) recovers that space for every view without
        # changing the camera math.  This makes the arm visually larger
        # for the VLM without any clipping of actual parts.
        self._crop_to_content(output_path)

        return output_path

    def _crop_to_content(self, image_path: str) -> None:
        """Crop a rendered PNG to its content bounding box + 5% padding.

        Uses PIL to find non-background pixels (the assembly) and trims
        the uniform-background margins.  The result is resized back to
        the original dimensions so downstream consumers see a consistent
        image size.  No-op if the image is >90% content (already tight)
        or PIL is unavailable.
        """
        try:
            from PIL import Image
            import numpy as np

            img = Image.open(image_path)
            arr = np.array(img)
            # Background detection: the renderer uses a light gradient
            # background (>180 on all RGB channels).  Content is darker.
            if len(arr.shape) == 3:
                gray = arr.mean(axis=2)
            else:
                gray = arr
            content_mask = gray < 180
            if content_mask.sum() < 10:
                return  # effectively empty — leave as-is
            rows = np.any(content_mask, axis=1)
            cols = np.any(content_mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            h, w = gray.shape
            # If content already fills >90% of the frame, skip crop.
            content_h_pct = (rmax - rmin) / h
            content_w_pct = (cmax - cmin) / w
            if content_h_pct > 0.9 and content_w_pct > 0.9:
                return
            # Add 5% padding on each side (relative to content size, not
            # full image, so small content gets visible breathing room).
            pad_h = max(20, int((rmax - rmin) * 0.05))
            pad_w = max(20, int((cmax - cmin) * 0.05))
            rmin = max(0, rmin - pad_h)
            rmax = min(h, rmax + pad_h)
            cmin = max(0, cmin - pad_w)
            cmax = min(w, cmax + pad_w)
            cropped = img.crop((cmin, rmin, cmax, rmax))
            # Resize back to original dimensions while PRESERVING ASPECT RATIO.
            # The previous code did cropped.resize((w, h)) which non-uniformly
            # stretched the content: a tall box (455×781) became 1200×900,
            # stretching X by 2.64× but Y by only 1.15× → the box read as wide
            # instead of tall. This was the root cause of "长宽对不上" (every
            # view's proportions didn't match). We now fit the cropped content
            # into the target size with uniform scaling and pad the remainder
            # with the background color so the saved PNG keeps its original
            # pixel dimensions without distorting the geometry.
            cw, ch = cropped.size
            scale = min(w / cw, h / ch)
            new_cw, new_ch = int(cw * scale), int(ch * scale)
            cropped = cropped.resize((new_cw, new_ch), Image.LANCZOS)
            canvas = Image.new(img.mode, (w, h), (245, 245, 250))
            canvas.paste(cropped, ((w - new_cw) // 2, (h - new_ch) // 2))
            canvas.save(image_path)
        except Exception:
            # Cropping is a visual enhancement only — never block render.
            pass

    def render_all_views(
        self,
        output_dir: str,
        views: list[str] | None = None,
        prefix: str = "",
        view_overrides: dict[str, dict] | None = None,
    ) -> list[str]:
        """Render multiple views. Returns list of PNG file paths.

        Args:
            output_dir: Directory to save PNG files.
            views: List of view names from VIEW_PRESETS. Default: the four
                standard engineering views (isometric/front/top/right); the
                ``gripper_closeup`` view is opt-in only.
            prefix: Optional filename prefix (e.g. part name).
            view_overrides: Optional per-view camera overrides, e.g.
                ``{"gripper_closeup": {"focus_point": (0,-170,330),
                "parallel_scale": 60.0}}``.  Passed through to render_to_file.

        Returns:
            List of absolute paths to rendered PNG files.
        """
        if views is None:
            # Default: the four standard views only (gripper_closeup is opt-in).
            views = ["isometric", "front", "top", "right"]

        os.makedirs(output_dir, exist_ok=True)
        paths: list[str] = []

        for view_name in views:
            fname = f"{prefix}_{view_name}.png" if prefix else f"{view_name}.png"
            output_path = os.path.join(output_dir, fname)
            overrides = (view_overrides or {}).get(view_name, {})
            self.render_to_file(
                view_name,
                output_path,
                focus_point=overrides.get("focus_point"),
                parallel_scale=overrides.get("parallel_scale"),
                direction=overrides.get("direction"),
            )
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
            # Bolt shank length: grip thickness + washer + nut + small
            # protrusion past the nut.  Minimum 8 mm for nut engagement on
            # very thin plates.
            shank_length = max(thickness + washer_h + nut_h + 1.0, 8.0)

            # Rotation to align default-Z cylinder with the anchor normal.
            rot = _rotation_from_z_to(normal)

            nx, ny, nz = normal
            px, py, pz = world_pos

            # world_pos is ON the +normal face.  The bolt goes THROUGH the
            # part: head flush on +normal face, shank through the body,
            # washer + nut on the −normal face.
            hd = head_h / 2.0
            renderer.add_cylinder(
                radius=head_r, height=head_h,
                color=(0.80, 0.80, 0.82),
                position=(px + nx * hd, py + ny * hd, pz + nz * hd),
                rotation=rot,
                category="bearing",
            )
            # Shank: from +normal face through part to −normal protrusion.
            sd = shank_length / 2.0
            renderer.add_cylinder(
                radius=shank_r, height=shank_length,
                color=(0.75, 0.75, 0.78),
                position=(px - nx * sd, py - ny * sd, pz - nz * sd),
                rotation=rot,
                category="bearing",
            )
            # Washer: flush on the −normal face.
            wd = thickness + washer_h / 2.0
            renderer.add_cylinder(
                radius=washer_r, height=washer_h,
                color=(0.70, 0.70, 0.72),
                position=(px - nx * wd, py - ny * wd, pz - nz * wd),
                rotation=rot,
                category="bearing",
            )
            # Nut: beyond the washer on the −normal side.
            nd = thickness + washer_h + nut_h / 2.0
            renderer.add_cylinder(
                radius=nut_r, height=nut_h,
                color=(0.65, 0.65, 0.68),
                position=(px - nx * nd, py - ny * nd, pz - nz * nd),
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
    # Project finger separation onto the lateral direction (not just Y).
    sep_x = rpos[0] - lpos[0]
    sep_y = rpos[1] - lpos[1]
    sep_z = rpos[2] - lpos[2]
    raw_gap = abs(sep_x * lat[0] + sep_y * lat[1] + sep_z * lat[2])
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
    gripper_closeup: bool = False,
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
        # Arm-link position tint takes priority over generic category color
        # so the elbow fold region stays readable (distinct shades per link
        # instead of one fused silver blob).  Falls through to category for
        # non-link structural parts (base_plate, struts, ...).
        link_tint = _link_position_tint(name)
        if link_tint is not None:
            color = link_tint
        elif category in CATEGORY_COLORS:
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

        # STL axis-swap policy.  FreeCAD's makeBox puts the FIRST arg on X.
        # - Arm-chain parts (raw_script families: arm_link / arm_joint /
        #   gripper / gripper_finger) generate length on X, so they need
        #   swap_xy (R_z(-90°): +X → -Y) so the link extends along the arm.
        # - Chassis BOX parts (base_plate, chassis_body, battery_box, top_plate)
        #   now generate with X=width, Y=length (the make_box/plate_with_holes
        #   ops swap to match the solver convention).  swap_xy would
        #   DOUBLE-rotate them → the long edge ends up on the wrong axis
        #   (the "车底盘方向不对" defect).  These must NOT swap.
        # - Wheels (orient_axis="y") are radially symmetric about Y; swapping
        #   turns them into 磨盘.  No swap.
        n_lower = name.lower()
        _is_wheel = n_lower.startswith("wheel_")
        _is_chassis_box = n_lower in (
            "base_plate", "chassis_body", "battery_box", "top_plate",
        ) or n_lower.startswith("standoff_")
        _swap = not _is_wheel and not _is_chassis_box

        # Try STL first (real geometry)
        stl_loaded = False
        if stl_dir:
            stl_path = os.path.join(stl_dir, f"{name}.stl")
            if os.path.isfile(stl_path):
                r.load_stl(stl_path, color=color, position=tuple(pos), rotation=rot_tuple, category=category, swap_xy=_swap)
                stl_loaded = True

        # Fallback to dimension-based approximation
        if not stl_loaded:
            _add_dimension_approximation(r, dims, color, pos, rot_tuple, category=category, name=name)

    # Fasteners (bolts, washers, nuts) at bolted joints
    if joints:
        _add_fasteners_for_joints(r, joints, positions, parts)

    r.add_axes(length=25)
    r.add_floor_grid(size=400, spacing=50, z_position=0)

    # Gripper close-up override: if requested AND the close-up view is in
    # the render list, aim the camera at the finger/gripper centroid with a
    # tight parallel scale so the ~32mm finger gap is clearly visible to
    # the VLM (which otherwise sees fingers as 1-2px slivers at the edge of
    # a full-arm render).  Safe degradation: if no gripper/finger parts are
    # present, the override is skipped and the close-up falls back to the
    # default scene-centre framing.
    view_overrides: dict[str, dict] = {}
    if gripper_closeup and views and "gripper_closeup" in views:
        finger_names = [n for n in positions if "finger" in n.lower()]
        gripper_names = [n for n in positions if "gripper" in n.lower()]
        target_names = finger_names or gripper_names
        if target_names:
            pts = [positions[n]["position"] for n in target_names]
            gx = sum(p[0] for p in pts) / len(pts)
            gy = sum(p[1] for p in pts) / len(pts)
            gz = sum(p[2] for p in pts) / len(pts)

            # Camera DIRECTION: look PERPENDICULAR to the finger-gap axis so
            # the two fingers render side-by-side with the gap between them.
            # Pre-fix the camera used the iso direction (350,-350,250), which
            # looks INTO the X-axis gap → the two fingers overlapped into one
            # solid block in the render, and the VLM reported "no two prongs".
            # The gap axis = direction separating the two fingers (largest
            # component-wise spread of ALL finger centres — for a dual-arm
            # robot that's the X axis separating the two grippers, not the
            # tiny intra-gripper gap). The camera must look along a DIFFERENT
            # horizontal axis so the fingers are in the frame.
            direction = (0.0, -1.0, 0.3)  # safe default: look along -Y
            if len(finger_names) >= 2:
                fpts = [positions[n]["position"] for n in finger_names]
                spread = [max(p[i] for p in fpts) - min(p[i] for p in fpts)
                          for i in range(3)]
                gap_axis = spread.index(max(spread))
                # Look perpendicular to the gap axis, horizontally.
                if gap_axis == 0:      # gap along X → look along Y
                    direction = (0.0, -1.0, 0.3)
                elif gap_axis == 1:    # gap along Y → look along X
                    direction = (1.0, 0.0, 0.3)
                else:                  # gap along Z (rare) → look along Y
                    direction = (0.0, -1.0, 0.3)

            # Adaptive zoom: fit ALL gripper/finger parts in frame + margin.
            # A single-arm gripper has fingers ~32-50mm apart → scale 60
            # (120mm view) shows the gap crisply.  But a dual-arm robot with
            # collision-aware splay has its two grippers far apart (X span
            # 400-600mm) — a fixed 60 scale leaves them off-frame, so the
            # VLM sees "plain background, no gripper" and the fix-loop
            # deadlocks.  Scale to 1.4× the view-plane half-extent (the
            # larger X/Y spread, since ``direction`` is horizontal) with a
            # 60mm floor so a single gripper still fills the frame.
            view_half = 0.0
            if len(pts) >= 2:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                view_half = max((max(xs) - min(xs)) / 2.0,
                                (max(ys) - min(ys)) / 2.0)
            parallel_scale = max(60.0, view_half * 1.4)

            view_overrides["gripper_closeup"] = {
                "focus_point": (gx, gy, gz),
                "parallel_scale": parallel_scale,
                "direction": direction,
            }

    return r.render_all_views(output_dir, views=views, view_overrides=view_overrides)


def _add_dimension_approximation(
    renderer: VTKOffscreenRenderer,
    dims: dict[str, float],
    color: tuple[float, float, float],
    position: list[float],
    rotation: tuple[float, float, float, float] | None = None,
    category: str = "",
    name: str = "",
) -> None:
    """Add a box or cylinder approximation based on dimension dict.

    For box parts with a significant Y-axis rotation (typical of arm joints),
    the length and height dimensions are swapped so the long axis ends up
    horizontal after the actor rotation is applied.

    Cylinder orientation is inferred from the part NAME: wheels (``wheel_*``)
    have their axle along Y (axis="y" joint), so the cylinder height (wheel
    width) extends along Y — the cylinder lies on its side and rolls along X.
    All other cylinders (servos, standoffs) keep height on Z (vertical).
    """
    if "outer_diameter" in dims or "diameter" in dims:
        d = dims.get("outer_diameter", dims.get("diameter", 10))
        h = dims.get("height", dims.get("length", 10))
        # Wheels have axle along Y (axis="y"); everything else is vertical (Z).
        orient = "y" if name.lower().startswith("wheel_") else "z"
        renderer.add_cylinder(
            radius=d / 2, height=h, color=color, position=tuple(position),
            rotation=rotation, category=category, orient_axis=orient,
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
