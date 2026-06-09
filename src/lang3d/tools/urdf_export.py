"""URDF export — convert Assembly to ROS2 URDF/Xacro + Gazebo simulation config.

Public API:
  - AssemblyToURDF  : core converter (Assembly → URDF XML string)
  - ROS2PackageBuilder : generate complete ROS2 package directory structure
  - URDFExportTool  : Agent tool wrapping the converter
  - register_urdf_tools : registration helper
"""

from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..knowledge.mechanics import Assembly, Joint, Part
from ..models.base import ToolDefinition
from .base import Tool

# ============================================================================
# Constants
# ============================================================================

# Standard colors for URDF materials
_DEFAULT_MATERIALS: dict[str, str] = {
    "gray": "0.7 0.7 0.7 1.0",
    "blue": "0.0 0.0 0.8 1.0",
    "green": "0.0 0.8 0.0 1.0",
    "red": "0.8 0.0 0.0 1.0",
    "orange": "1.0 0.5 0.0 1.0",
    "yellow": "1.0 1.0 0.0 1.0",
    "black": "0.1 0.1 0.1 1.0",
    "white": "1.0 1.0 1.0 1.0",
}

# Maps joint.type to URDF joint type
_JOINT_TYPE_MAP: dict[str, str] = {
    "revolute": "revolute",
    "continuous": "continuous",
    "prismatic": "prismatic",
    "fixed": "fixed",
}

# Maps joint.axis string to URDF xyz vector
_AXIS_MAP: dict[str, list[float]] = {
    "x": [1.0, 0.0, 0.0],
    "y": [0.0, 1.0, 0.0],
    "z": [0.0, 0.0, 1.0],
}


# ============================================================================
# Helper functions
# ============================================================================


def _sanitize_name(name: str) -> str:
    """Make a name URDF-safe (lowercase, underscores, no special chars)."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "link"


def _mm_to_m(v: float) -> float:
    """Convert millimeters to meters."""
    return v / 1000.0


def _mm2_to_m2(v: float) -> float:
    """Convert mm² to m²."""
    return v / 1_000_000.0


def _resolve_axis(joint: Joint) -> list[float]:
    """Resolve joint axis from explicit setting or anchor inference."""
    if joint.axis != "auto" and joint.axis.lower() in _AXIS_MAP:
        return _AXIS_MAP[joint.axis.lower()]
    # Infer from parent_anchor
    anchor = joint.parent_anchor.lower()
    if anchor in ("top", "bottom"):
        return [0.0, 0.0, 1.0]
    elif anchor in ("left", "right"):
        return [1.0, 0.0, 0.0]
    elif anchor in ("front", "back"):
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]  # default z


def _infer_inertia(part: Part) -> dict[str, float]:
    """Infer simplified inertia tensor for a part.

    Returns dict with ixx, ixy, ixz, iyy, iyz, izz (kg·m²).
    Uses bounding-box or cylinder approximation.
    """
    mass = part.compute_estimated_mass()
    if mass <= 0:
        mass = 0.01  # fallback 10g

    dims = part.dimensions
    # Try cylinder
    if "diameter" in dims or "outer_diameter" in dims:
        d = dims.get("diameter", dims.get("outer_diameter", 10.0))
        h = dims.get("height", dims.get("thickness", dims.get("length", 10.0)))
        r = _mm_to_m(d / 2.0)
        h_m = _mm_to_m(h)
        ixx = mass * (3 * r * r + h_m * h_m) / 12.0
        izz = mass * r * r / 2.0
        return {"ixx": round(ixx, 8), "ixy": 0.0, "ixz": 0.0,
                "iyy": round(ixx, 8), "iyz": 0.0, "izz": round(izz, 8)}

    # Try box
    lx = _mm_to_m(dims.get("length", dims.get("width", 10.0)))
    ly = _mm_to_m(dims.get("width", 10.0))
    lz = _mm_to_m(dims.get("height", dims.get("thickness", 10.0)))
    if lx <= 0:
        lx = 0.01
    if ly <= 0:
        ly = 0.01
    if lz <= 0:
        lz = 0.01

    ixx = mass * (ly * ly + lz * lz) / 12.0
    iyy = mass * (lx * lx + lz * lz) / 12.0
    izz = mass * (lx * lx + ly * ly) / 12.0
    return {"ixx": round(ixx, 8), "ixy": 0.0, "ixz": 0.0,
            "iyy": round(iyy, 8), "iyz": 0.0, "izz": round(izz, 8)}


def _pick_material_color(part: Part) -> str:
    """Pick a default material color based on part category."""
    cat = part.category.lower()
    if "joint" in cat:
        return "orange"
    if "actuator" in cat or "servo" in cat:
        return "blue"
    if "structural" in cat:
        return "gray"
    return "gray"


# ============================================================================
# AssemblyToURDF — core converter
# ============================================================================


@dataclass
class URDFLink:
    """Intermediate representation of a URDF link."""
    name: str
    mass: float = 0.0  # kg
    com: tuple[float, float, float] = (0.0, 0.0, 0.0)  # meters
    inertia: dict[str, float] = field(default_factory=dict)
    visual_mesh: str = ""  # relative STL path
    collision_mesh: str = ""  # relative STL path (may be same as visual)
    material_color: str = "gray"


@dataclass
class URDFJoint:
    """Intermediate representation of a URDF joint."""
    name: str
    type: str = "fixed"  # revolute, prismatic, fixed, continuous
    parent: str = ""
    child: str = ""
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)  # meters
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)  # radians
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    lower: float = 0.0  # radians
    upper: float = 0.0  # radians
    effort: float = 100.0  # N or Nm
    velocity: float = 3.14  # m/s or rad/s


@dataclass
class GazeboPlugin:
    """Gazebo simulation plugin configuration."""
    name: str
    filename: str
    params: dict[str, Any] = field(default_factory=dict)


class AssemblyToURDF:
    """Convert an Assembly to URDF XML.

    Usage::

        converter = AssemblyToURDF(assembly, meshes_dir="meshes")
        xml_string = converter.convert()
    """

    def __init__(
        self,
        assembly: Assembly,
        meshes_dir: str = "meshes",
        package_name: str = "",
    ) -> None:
        self.assembly = assembly
        self.meshes_dir = meshes_dir
        self.package_name = package_name or _sanitize_name(assembly.name)

        self._links: list[URDFLink] = []
        self._joints: list[URDFJoint] = []
        self._gazebo_plugins: list[GazeboPlugin] = []
        self._part_index: dict[str, Part] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(self) -> str:
        """Build the intermediate model and return URDF XML string."""
        self._part_index = {p.name: p for p in self.assembly.parts}
        self._build_links()
        self._build_joints()
        self._auto_add_gazebo_plugins()
        return self._render_xml()

    def get_links(self) -> list[URDFLink]:
        """Return intermediate links (call after convert)."""
        return self._links

    def get_joints(self) -> list[URDFJoint]:
        """Return intermediate joints (call after convert)."""
        return self._joints

    def get_gazebo_plugins(self) -> list[GazeboPlugin]:
        """Return gazebo plugins (call after convert)."""
        return self._gazebo_plugins

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_links(self) -> None:
        for part in self.assembly.parts:
            link_name = _sanitize_name(part.name)
            mass = part.compute_estimated_mass()
            if mass <= 0:
                mass = 0.01

            com_m = (
                _mm_to_m(part.center_of_mass[0]),
                _mm_to_m(part.center_of_mass[1]),
                _mm_to_m(part.center_of_mass[2]),
            )

            inertia = _infer_inertia(part)

            # If user provided an inertia tensor with non-zero diagonal, use it
            it = part.inertia_tensor
            if any(it[i][i] > 0 for i in range(3)):
                inertia = {
                    "ixx": round(_mm2_to_m2(it[0][0]), 8),
                    "ixy": round(_mm2_to_m2(it[0][1]), 8),
                    "ixz": round(_mm2_to_m2(it[0][2]), 8),
                    "iyy": round(_mm2_to_m2(it[1][1]), 8),
                    "iyz": round(_mm2_to_m2(it[1][2]), 8),
                    "izz": round(_mm2_to_m2(it[2][2]), 8),
                }

            mesh_file = f"{self.meshes_dir}/{link_name}.stl"
            self._links.append(URDFLink(
                name=link_name,
                mass=round(mass, 6),
                com=com_m,
                inertia=inertia,
                visual_mesh=mesh_file,
                collision_mesh=mesh_file,
                material_color=_pick_material_color(part),
            ))

    def _build_joints(self) -> None:
        for i, joint in enumerate(self.assembly.joints):
            jtype = _JOINT_TYPE_MAP.get(joint.type, "fixed")
            parent_name = _sanitize_name(joint.parent)
            child_name = _sanitize_name(joint.child)

            # Origin: use joint offset (converted mm→m)
            joint_offset = joint.offset or (0, 0, 0)
            origin_xyz = (
                _mm_to_m(joint_offset[0]),
                _mm_to_m(joint_offset[1]),
                _mm_to_m(joint_offset[2]),
            )

            # For non-root children, estimate origin from parent part dimensions
            parent_part = self._part_index.get(joint.parent)
            if parent_part and origin_xyz == (0.0, 0.0, 0.0):
                h = parent_part.dimensions.get(
                    "height",
                    parent_part.dimensions.get("thickness", 0.0),
                )
                if h > 0:
                    anchor = joint.parent_anchor.lower()
                    if anchor == "top":
                        origin_xyz = (0.0, 0.0, _mm_to_m(h))

            axis_vec = _resolve_axis(joint)
            lower_rad = math.radians(joint.range_deg[0])
            upper_rad = math.radians(joint.range_deg[1])

            self._joints.append(URDFJoint(
                name=f"{parent_name}_to_{child_name}",
                type=jtype,
                parent=parent_name,
                child=child_name,
                origin_xyz=origin_xyz,
                origin_rpy=(0.0, 0.0, 0.0),
                axis=tuple(axis_vec),  # type: ignore[arg-type]
                lower=round(lower_rad, 4),
                upper=round(upper_rad, 4),
            ))

    def _auto_add_gazebo_plugins(self) -> None:
        """Auto-detect drive train type and add Gazebo plugins."""
        revolute_count = sum(1 for j in self.assembly.joints if j.type == "revolute")

        # Check if this looks like a differential drive (2 revolute joints on z axis)
        z_revolute = sum(
            1 for j in self.assembly.joints
            if j.type == "revolute" and _resolve_axis(j) == [0.0, 0.0, 1.0]
        )

        if z_revolute >= 2:
            # Differential drive
            wheel_joints = [
                j for j in self._joints if j.type == "revolute"
                and j.axis == (0.0, 0.0, 1.0)
            ]
            if len(wheel_joints) >= 2:
                self._gazebo_plugins.append(GazeboPlugin(
                    name="diff_drive",
                    filename="libgazebo_ros_diff_drive.so",
                    params={
                        "leftJoint": wheel_joints[0].name,
                        "rightJoint": wheel_joints[1].name,
                        "wheelSeparation": 0.15,
                        "wheelDiameter": 0.065,
                        "commandTopic": "cmd_vel",
                        "odometryTopic": "odom",
                        "odometryFrame": "odom",
                    },
                ))

        if revolute_count > 0:
            # Joint state publisher for arms
            self._gazebo_plugins.append(GazeboPlugin(
                name="joint_state_publisher",
                filename="libgazebo_ros_joint_state_publisher.so",
                params={
                    "jointNames": " ".join(
                        j.name for j in self._joints if j.type in ("revolute", "prismatic")
                    ),
                    "updateRate": 50,
                },
            ))

    def _render_xml(self) -> str:
        root = ET.Element("robot")
        root.set("name", _sanitize_name(self.assembly.name))

        # Links
        for link in self._links:
            link_el = ET.SubElement(root, "link", name=link.name)

            # Inertial
            inertial = ET.SubElement(link_el, "inertial")
            origin = ET.SubElement(inertial, "origin")
            origin.set("xyz", " ".join(f"{v:.6f}" for v in link.com))
            origin.set("rpy", "0 0 0")
            mass_el = ET.SubElement(inertial, "mass")
            mass_el.set("value", f"{link.mass:.6f}")
            inertia_el = ET.SubElement(inertial, "inertia")
            for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
                inertia_el.set(key, f"{link.inertia.get(key, 0.0):.8f}")

            # Visual
            visual = ET.SubElement(link_el, "visual")
            vis_origin = ET.SubElement(visual, "origin")
            vis_origin.set("xyz", "0 0 0")
            vis_origin.set("rpy", "0 0 0")
            vis_geom = ET.SubElement(visual, "geometry")
            vis_mesh = ET.SubElement(vis_geom, "mesh")
            vis_mesh.set("filename", link.visual_mesh)
            mat_el = ET.SubElement(visual, "material")
            mat_el.set("name", link.material_color)
            color_el = ET.SubElement(mat_el, "color")
            color_el.set("rgba", _DEFAULT_MATERIALS.get(link.material_color, "0.7 0.7 0.7 1.0"))

            # Collision
            collision = ET.SubElement(link_el, "collision")
            col_origin = ET.SubElement(collision, "origin")
            col_origin.set("xyz", "0 0 0")
            col_origin.set("rpy", "0 0 0")
            col_geom = ET.SubElement(collision, "geometry")
            col_mesh = ET.SubElement(col_geom, "mesh")
            col_mesh.set("filename", link.collision_mesh)

        # Joints
        for joint in self._joints:
            joint_el = ET.SubElement(root, "joint", name=joint.name, type=joint.type)
            ET.SubElement(joint_el, "parent", link=joint.parent)
            ET.SubElement(joint_el, "child", link=joint.child)
            origin_el = ET.SubElement(joint_el, "origin")
            origin_el.set("xyz", " ".join(f"{v:.6f}" for v in joint.origin_xyz))
            origin_el.set("rpy", " ".join(f"{v:.6f}" for v in joint.origin_rpy))
            axis_el = ET.SubElement(joint_el, "axis")
            axis_el.set("xyz", " ".join(f"{v:.1f}" for v in joint.axis))

            if joint.type in ("revolute", "prismatic"):
                limit_el = ET.SubElement(joint_el, "limit")
                limit_el.set("lower", f"{joint.lower:.4f}")
                limit_el.set("upper", f"{joint.upper:.4f}")
                limit_el.set("effort", f"{joint.effort:.1f}")
                limit_el.set("velocity", f"{joint.velocity:.2f}")

        # Gazebo plugins
        for plugin in self._gazebo_plugins:
            gazebo = ET.SubElement(root, "gazebo")
            plugin_el = ET.SubElement(gazebo, "plugin", name=plugin.name, filename=plugin.filename)
            if plugin.name == "diff_drive":
                for k, v in plugin.params.items():
                    child = ET.SubElement(plugin_el, k)
                    child.text = str(v)
            elif plugin.name == "joint_state_publisher":
                child = ET.SubElement(plugin_el, "jointNames")
                child.text = str(plugin.params.get("jointNames", ""))
                child2 = ET.SubElement(plugin_el, "updateRate")
                child2.text = str(plugin.params.get("updateRate", 50))

        # Pretty print
        _indent_xml(root)
        return ET.tostring(root, encoding="unicode", xml_declaration=False)


# ============================================================================
# ROS2 Package Builder
# ============================================================================


class ROS2PackageBuilder:
    """Generate a complete ROS2 package directory from a URDF string.

    Directory structure::

        <package_name>/
        ├── package.xml
        ├── CMakeLists.txt
        ├── urdf/
        │   └── <robot>.urdf
        ├── meshes/
        │   └── (STL files go here)
        ├── launch/
        │   └── display.launch.py
        └── config/
            └── joint_names.yaml
    """

    def __init__(self, package_name: str, urdf_xml: str) -> None:
        self.package_name = _sanitize_name(package_name)
        self.urdf_xml = urdf_xml

    def write(self, output_dir: str | Path) -> str:
        """Write the complete ROS2 package to disk. Returns output path."""
        base = Path(output_dir) / self.package_name
        (base / "urdf").mkdir(parents=True, exist_ok=True)
        (base / "meshes").mkdir(parents=True, exist_ok=True)
        (base / "launch").mkdir(parents=True, exist_ok=True)
        (base / "config").mkdir(parents=True, exist_ok=True)

        # URDF file
        urdf_path = base / "urdf" / f"{self.package_name}.urdf"
        urdf_path.write_text(self.urdf_xml, encoding="utf-8")

        # package.xml
        (base / "package.xml").write_text(self._package_xml(), encoding="utf-8")

        # CMakeLists.txt
        (base / "CMakeLists.txt").write_text(self._cmakeLists(), encoding="utf-8")

        # launch file
        (base / "launch" / "display.launch.py").write_text(
            self._launch_file(), encoding="utf-8"
        )

        # config
        (base / "config" / "joint_names.yaml").write_text(
            self._joint_names_yaml(), encoding="utf-8"
        )

        return str(base)

    def _package_xml(self) -> str:
        return f"""\
<?xml version="1.0"?>
<package format="3">
  <name>{self.package_name}</name>
  <version>0.0.1</version>
  <description>Auto-generated by Language-3D Agent</description>
  <maintainer email="lang3d@example.com">lang3d</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <exec_depend>robot_state_publisher</exec_depend>
  <exec_depend>joint_state_publisher_gui</exec_depend>
  <exec_depend>rviz2</exec_depend>
  <exec_depend>xacro</exec_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
"""

    def _cmakeLists(self) -> str:
        return f"""\
cmake_minimum_required(VERSION 3.8)
project({self.package_name})

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf meshes launch config
  DESTINATION share/${{PROJECT_NAME}})

ament_package()
"""

    def _launch_file(self) -> str:
        return f"""\
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('{self.package_name}')
    urdf_file = os.path.join(pkg_dir, 'urdf', '{self.package_name}.urdf')

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{{'robot_description': robot_description}}],
            output='screen',
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', os.path.join(pkg_dir, 'config', 'rviz.rviz')],
            output='screen',
        ),
    ])
"""

    def _joint_names_yaml(self) -> str:
        # Parse joint names from URDF
        try:
            root = ET.fromstring(self.urdf_xml)
            joints = root.findall(".//joint[@type='revolute']")
            joints += root.findall(".//joint[@type='prismatic']")
            names = [j.get("name", "") for j in joints if j.get("name")]
        except ET.ParseError:
            names = []
        joint_list = "\n".join(f"  - \"{n}\"" for n in names)
        return f"joint_names:\n{joint_list}\n"


# ============================================================================
# XML pretty-print helper
# ============================================================================


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    """Add indentation to XML tree (in-place)."""
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent
    if level == 0:
        elem.tail = "\n"


# ============================================================================
# Agent Tool
# ============================================================================


class URDFExportTool(Tool):
    """Export an Assembly to URDF format for ROS2 simulation."""

    name = "urdf_export"
    description = "Export an Assembly to ROS2 URDF format (robot description + meshes)"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "assembly_name": {
                        "type": "string",
                        "description": "Name of the assembly to export (must exist in workspace)",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for the ROS2 package (default: workspace)",
                    },
                    "package_name": {
                        "type": "string",
                        "description": "ROS2 package name (default: sanitized assembly name)",
                    },
                    "mode": {
                        "type": "string",
                        "description": "Output mode: 'xml' returns URDF XML string, 'package' writes full ROS2 package (default: package)",
                    },
                },
                "required": ["assembly_name"],
            },
        )

    def execute(
        self,
        *,
        assembly_name: str,
        output_dir: str = "",
        package_name: str = "",
        mode: str = "package",
        **kwargs: Any,
    ) -> str:
        # Try to load assembly from the global assemblies dict
        assembly = _find_assembly(assembly_name)
        if assembly is None:
            return f"错误：未找到装配体 '{assembly_name}'"

        try:
            converter = AssemblyToURDF(
                assembly,
                meshes_dir="meshes",
                package_name=package_name,
            )
            urdf_xml = converter.convert()
        except Exception as e:
            return f"URDF 转换失败：{e}"

        if mode == "xml":
            return urdf_xml

        # Full ROS2 package
        out = output_dir or os.getcwd()
        builder = ROS2PackageBuilder(
            package_name=package_name or assembly.name,
            urdf_xml=urdf_xml,
        )
        try:
            pkg_path = builder.write(out)
            links = converter.get_links()
            joints = converter.get_joints()
            plugins = converter.get_gazebo_plugins()
            return (
                f"ROS2 包已生成：{pkg_path}\n"
                f"  Links: {len(links)}\n"
                f"  Joints: {len(joints)}\n"
                f"  Gazebo plugins: {len(plugins)}\n"
                f"  结构: urdf/ meshes/ launch/ config/"
            )
        except Exception as e:
            return f"ROS2 包生成失败：{e}"


# ============================================================================
# Assembly lookup helper
# ============================================================================


def _find_assembly(name: str) -> Assembly | None:
    """Try to find an assembly by name from the standard library."""
    from ..knowledge.mechanics import ROBOTIC_ARM_ASSEMBLY
    candidates = [ROBOTIC_ARM_ASSEMBLY]
    for a in candidates:
        if a.name == name or _sanitize_name(a.name) == _sanitize_name(name):
            return a
    return None


# ============================================================================
# Registration
# ============================================================================


def register_urdf_tools(registry: Any) -> None:
    """Register URDF export tools."""
    registry.register(URDFExportTool())
