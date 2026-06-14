"""Tests for URDF export tool (Task 54).

Covers:
  - Helper functions: _sanitize_name, _resolve_axis, _infer_inertia
  - AssemblyToURDF converter: links, joints, inertia, Gazebo plugins
  - ROS2PackageBuilder: directory structure, file contents
  - URDFExportTool: execution, registration
  - Integration: robotic arm assembly → URDF
"""

import math
import os
import tempfile
from pathlib import Path

import pytest

from lang3d.knowledge.mechanics import (
    Assembly,
    Joint,
    Part,
    ROBOTIC_ARM_ASSEMBLY,
)
from lang3d.tools.urdf_export import (
    AssemblyToURDF,
    GazeboPlugin,
    ROS2PackageBuilder,
    URDFExportTool,
    _infer_inertia,
    _mm_to_m,
    _resolve_axis,
    _sanitize_name,
    register_urdf_tools,
)
from lang3d.tools.base import ToolRegistry


# ============================================================================
# Helper functions
# ============================================================================


class TestSanitizeName:
    def test_lowercase(self):
        assert _sanitize_name("Base Plate") == "base_plate"

    def test_special_chars(self):
        assert _sanitize_name("L-Bracket (v2)") == "l_bracket_v2"

    def test_underscores_collapsed(self):
        assert _sanitize_name("a___b") == "a_b"

    def test_empty(self):
        assert _sanitize_name("") == "link"

    def test_chinese_fallback(self):
        # Chinese chars get replaced with underscores
        result = _sanitize_name("底座")
        assert result  # non-empty


class TestMmToM:
    def test_zero(self):
        assert _mm_to_m(0) == 0.0

    def test_1000mm(self):
        assert _mm_to_m(1000) == 1.0

    def test_150mm(self):
        assert _mm_to_m(150) == pytest.approx(0.15)


class TestResolveAxis:
    def test_explicit_x(self):
        j = Joint("revolute", "a", "b", axis="x")
        assert _resolve_axis(j) == [1.0, 0.0, 0.0]

    def test_explicit_y(self):
        j = Joint("revolute", "a", "b", axis="y")
        assert _resolve_axis(j) == [0.0, 1.0, 0.0]

    def test_explicit_z(self):
        j = Joint("revolute", "a", "b", axis="z")
        assert _resolve_axis(j) == [0.0, 0.0, 1.0]

    def test_auto_top(self):
        j = Joint("revolute", "a", "b", parent_anchor="top")
        assert _resolve_axis(j) == [0.0, 0.0, 1.0]

    def test_auto_left(self):
        j = Joint("revolute", "a", "b", parent_anchor="left")
        assert _resolve_axis(j) == [1.0, 0.0, 0.0]

    def test_auto_front(self):
        j = Joint("revolute", "a", "b", parent_anchor="front")
        assert _resolve_axis(j) == [0.0, 1.0, 0.0]


class TestInferInertia:
    def test_box_inertia(self):
        p = Part(name="box", category="structural", description="box",
                 dimensions=dict(length=100, width=50, height=30))
        inertia = _infer_inertia(p)
        assert inertia["ixx"] > 0
        assert inertia["iyy"] > 0
        assert inertia["izz"] > 0
        assert inertia["ixy"] == 0.0
        assert inertia["ixz"] == 0.0
        assert inertia["iyz"] == 0.0

    def test_cylinder_inertia(self):
        p = Part(name="cyl", category="structural", description="cyl",
                 dimensions=dict(diameter=60, height=40))
        inertia = _infer_inertia(p)
        assert inertia["ixx"] > 0
        assert inertia["izz"] > 0
        # For a cylinder about z: izz = m*r^2/2, ixx = m*(3r^2+h^2)/12
        assert inertia["izz"] > 0

    def test_zero_mass_fallback(self):
        p = Part(name="empty", category="structural", description="empty")
        inertia = _infer_inertia(p)
        assert inertia["ixx"] >= 0  # should not crash


# ============================================================================
# AssemblyToURDF converter
# ============================================================================


def _make_simple_assembly():
    """Create a simple 3-part assembly for testing."""
    return Assembly(
        name="Test Robot",
        parts=[
            Part(name="base", category="structural", description="Base plate",
                 dimensions=dict(length=100, width=100, height=10),
                 mass=0.5),
            Part(name="arm", category="structural", description="Arm link",
                 dimensions=dict(length=150, width=40, height=30),
                 mass=0.2),
            Part(name="gripper", category="joint", description="Gripper",
                 dimensions=dict(diameter=30, height=20),
                 mass=0.05),
        ],
        joints=[
            Joint("revolute", "base", "arm", (-90, 90),
                  axis="z", parent_anchor="top", child_anchor="bottom"),
            Joint("fixed", "arm", "gripper",
                  parent_anchor="top", child_anchor="bottom"),
        ],
    )


class TestAssemblyToURDF:
    def test_convert_returns_xml(self):
        asm = _make_simple_assembly()
        converter = AssemblyToURDF(asm)
        xml = converter.convert()
        assert "<robot" in xml
        assert "</robot>" in xml

    def test_links_generated(self):
        asm = _make_simple_assembly()
        converter = AssemblyToURDF(asm)
        converter.convert()
        links = converter.get_links()
        assert len(links) == 3
        names = {l.name for l in links}
        assert "base" in names
        assert "arm" in names
        assert "gripper" in names

    def test_joints_generated(self):
        asm = _make_simple_assembly()
        converter = AssemblyToURDF(asm)
        converter.convert()
        joints = converter.get_joints()
        assert len(joints) == 2
        types = {j.type for j in joints}
        assert "revolute" in types
        assert "fixed" in types

    def test_link_has_inertia(self):
        asm = _make_simple_assembly()
        converter = AssemblyToURDF(asm)
        xml = converter.convert()
        assert "inertia" in xml
        assert "ixx" in xml
        assert "iyy" in xml
        assert "izz" in xml

    def test_link_has_mass(self):
        asm = _make_simple_assembly()
        converter = AssemblyToURDF(asm)
        xml = converter.convert()
        assert "mass" in xml
        assert "value" in xml

    def test_joint_has_axis(self):
        asm = _make_simple_assembly()
        converter = AssemblyToURDF(asm)
        xml = converter.convert()
        assert "axis" in xml
        assert "xyz" in xml

    def test_revolute_has_limits(self):
        asm = _make_simple_assembly()
        converter = AssemblyToURDF(asm)
        xml = converter.convert()
        assert "limit" in xml
        assert "lower" in xml
        assert "upper" in xml
        assert "effort" in xml

    def test_visual_has_mesh(self):
        asm = _make_simple_assembly()
        converter = AssemblyToURDF(asm)
        xml = converter.convert()
        assert "mesh" in xml
        assert "filename" in xml

    def test_collision_has_mesh(self):
        asm = _make_simple_assembly()
        converter = AssemblyToURDF(asm)
        xml = converter.convert()
        # collision section should exist
        assert xml.count("collision") >= 3  # one per link

    def test_material_color(self):
        asm = _make_simple_assembly()
        converter = AssemblyToURDF(asm)
        xml = converter.convert()
        assert "material" in xml
        assert "rgba" in xml


class TestGazeboPlugins:
    def test_arm_gets_joint_state_publisher(self):
        asm = _make_simple_assembly()
        converter = AssemblyToURDF(asm)
        converter.convert()
        plugins = converter.get_gazebo_plugins()
        names = [p.name for p in plugins]
        assert "joint_state_publisher" in names

    def test_diff_drive_detected(self):
        """Assembly with 2 z-axis revolute joints gets diff drive plugin."""
        asm = Assembly(
            name="DiffBot",
            parts=[
                Part(name="chassis", category="structural", description="chassis",
                     dimensions=dict(length=150, width=100, height=10)),
                Part(name="wheel_l", category="structural", description="left wheel",
                     dimensions=dict(diameter=65, height=26)),
                Part(name="wheel_r", category="structural", description="right wheel",
                     dimensions=dict(diameter=65, height=26)),
            ],
            joints=[
                Joint("revolute", "chassis", "wheel_l", axis="z",
                      parent_anchor="left", child_anchor="bottom"),
                Joint("revolute", "chassis", "wheel_r", axis="z",
                      parent_anchor="right", child_anchor="bottom"),
            ],
        )
        converter = AssemblyToURDF(asm)
        converter.convert()
        plugins = converter.get_gazebo_plugins()
        names = [p.name for p in plugins]
        assert "diff_drive" in names

    def test_gazebo_xml_output(self):
        asm = _make_simple_assembly()
        converter = AssemblyToURDF(asm)
        xml = converter.convert()
        assert "gazebo" in xml
        assert "plugin" in xml


class TestUserProvidedInertia:
    def test_custom_inertia_used(self):
        p = Part(
            name="heavy", category="structural", description="heavy",
            dimensions=dict(length=100, width=50, height=30),
            mass=1.0,
            inertia_tensor=[
                [0.5, 0.0, 0.0],
                [0.0, 0.3, 0.0],
                [0.0, 0.0, 0.2],
            ],
        )
        asm = Assembly(name="test", parts=[p], joints=[])
        converter = AssemblyToURDF(asm)
        links = converter._links
        # Need to call convert first
        converter.convert()
        links = converter.get_links()
        assert len(links) == 1
        # Check that custom inertia values (converted to m²) are used
        ixx = links[0].inertia["ixx"]
        assert ixx > 0
        # 0.5 kg*mm² = 0.5e-6 kg*m²
        assert ixx == pytest.approx(5e-7, rel=0.01)


# ============================================================================
# ROS2PackageBuilder
# ============================================================================


class TestROS2PackageBuilder:
    def test_creates_directory_structure(self):
        asm = _make_simple_assembly()
        converter = AssemblyToURDF(asm)
        xml = converter.convert()

        with tempfile.TemporaryDirectory() as tmp:
            builder = ROS2PackageBuilder("test_robot", xml)
            path = builder.write(tmp)
            assert Path(path).exists()
            assert (Path(path) / "urdf").is_dir()
            assert (Path(path) / "meshes").is_dir()
            assert (Path(path) / "launch").is_dir()
            assert (Path(path) / "config").is_dir()

    def test_urdf_file_written(self):
        asm = _make_simple_assembly()
        xml = AssemblyToURDF(asm).convert()
        with tempfile.TemporaryDirectory() as tmp:
            builder = ROS2PackageBuilder("test_robot", xml)
            path = builder.write(tmp)
            urdf_file = Path(path) / "urdf" / "test_robot.urdf"
            assert urdf_file.exists()
            content = urdf_file.read_text()
            assert "<robot" in content

    def test_package_xml(self):
        asm = _make_simple_assembly()
        xml = AssemblyToURDF(asm).convert()
        with tempfile.TemporaryDirectory() as tmp:
            builder = ROS2PackageBuilder("test_robot", xml)
            path = builder.write(tmp)
            pkg = Path(path) / "package.xml"
            assert pkg.exists()
            content = pkg.read_text()
            assert "ament_cmake" in content
            assert "test_robot" in content

    def test_cmakelists(self):
        asm = _make_simple_assembly()
        xml = AssemblyToURDF(asm).convert()
        with tempfile.TemporaryDirectory() as tmp:
            builder = ROS2PackageBuilder("test_robot", xml)
            path = builder.write(tmp)
            cmake = Path(path) / "CMakeLists.txt"
            assert cmake.exists()
            content = cmake.read_text()
            assert "ament_package" in content

    def test_launch_file(self):
        asm = _make_simple_assembly()
        xml = AssemblyToURDF(asm).convert()
        with tempfile.TemporaryDirectory() as tmp:
            builder = ROS2PackageBuilder("test_robot", xml)
            path = builder.write(tmp)
            launch = Path(path) / "launch" / "display.launch.py"
            assert launch.exists()
            content = launch.read_text()
            assert "robot_state_publisher" in content
            assert "generate_launch_description" in content

    def test_joint_names_yaml(self):
        asm = _make_simple_assembly()
        xml = AssemblyToURDF(asm).convert()
        with tempfile.TemporaryDirectory() as tmp:
            builder = ROS2PackageBuilder("test_robot", xml)
            path = builder.write(tmp)
            yaml_file = Path(path) / "config" / "joint_names.yaml"
            assert yaml_file.exists()
            content = yaml_file.read_text()
            assert "joint_names" in content


# ============================================================================
# URDFExportTool
# ============================================================================


class TestURDFExportTool:
    def test_xml_mode(self):
        tool = URDFExportTool()
        result = tool.execute(
            assembly_name="3-DOF Robotic Arm",
            mode="xml",
        )
        assert "<robot" in result
        assert "</robot>" in result

    def test_xml_mode_robotic_arm(self):
        tool = URDFExportTool()
        result = tool.execute(
            assembly_name="3-DOF Robotic Arm",
            mode="xml",
        )
        # Should have 8 links, 6 joints (robotic arm)
        assert "link" in result
        assert "joint" in result

    def test_package_mode(self):
        tool = URDFExportTool()
        with tempfile.TemporaryDirectory() as tmp:
            result = tool.execute(
                assembly_name="3-DOF Robotic Arm",
                mode="package",
                output_dir=tmp,
            )
            assert "ROS2 包已生成" in result
            assert "Links:" in result
            assert "Joints:" in result

    def test_unknown_assembly(self):
        tool = URDFExportTool()
        result = tool.execute(assembly_name="nonexistent_robot")
        assert "错误" in result

    def test_definition(self):
        tool = URDFExportTool()
        defn = tool.get_definition()
        assert defn.name == "urdf_export"
        assert "assembly_name" in defn.parameters["properties"]


class TestRegistration:
    def test_urdf_registered(self):
        registry = ToolRegistry()
        register_urdf_tools(registry)
        names = [t.name for t in registry._tools.values()]
        assert "urdf_export" in names

    def test_tool_in_category(self):
        registry = ToolRegistry()
        register_urdf_tools(registry)
        assert "urdf_export" in registry._tools


# ============================================================================
# Integration: full robotic arm → URDF
# ============================================================================


class TestRoboticArmIntegration:
    def test_full_convert(self):
        converter = AssemblyToURDF(ROBOTIC_ARM_ASSEMBLY)
        xml = converter.convert()
        assert "<robot" in xml
        # 8 parts = 8 links
        links = converter.get_links()
        assert len(links) == 8
        # 6 joints
        joints = converter.get_joints()
        assert len(joints) == 6

    def test_revolute_joints_have_limits(self):
        converter = AssemblyToURDF(ROBOTIC_ARM_ASSEMBLY)
        converter.convert()
        revolute_joints = [j for j in converter.get_joints() if j.type == "revolute"]
        assert len(revolute_joints) >= 3
        for j in revolute_joints:
            assert j.lower < j.upper

    def test_all_links_have_mass(self):
        converter = AssemblyToURDF(ROBOTIC_ARM_ASSEMBLY)
        converter.convert()
        for link in converter.get_links():
            assert link.mass > 0

    def test_all_links_have_mesh(self):
        converter = AssemblyToURDF(ROBOTIC_ARM_ASSEMBLY)
        converter.convert()
        for link in converter.get_links():
            assert link.visual_mesh.endswith(".stl")
            assert link.collision_mesh.endswith(".stl")

    def test_package_roundtrip(self):
        converter = AssemblyToURDF(ROBOTIC_ARM_ASSEMBLY)
        xml = converter.convert()
        with tempfile.TemporaryDirectory() as tmp:
            builder = ROS2PackageBuilder("robotic_arm", xml)
            path = builder.write(tmp)
            # Read back URDF
            urdf = (Path(path) / "urdf" / "robotic_arm.urdf").read_text()
            assert "<robot" in urdf
            assert "base_plate" in urdf or "base" in urdf


# ============================================================================
# Prismatic vs revolute limit unit handling
# ============================================================================

class TestPrismaticLimitUnits:
    """Verify prismatic joint limits export in meters while revolute stays in radians."""

    def _make_assembly(self, joint):
        parts = [
            Part("base", "structural", "base",
                 dimensions={"length": 50, "width": 30, "height": 10}),
            Part("child", "mechanical", "child",
                 dimensions={"length": 20, "width": 10, "height": 10}),
        ]
        return Assembly(
            name="limit_unit_test",
            description="unit test for joint limit conversion",
            parts=parts,
            joints=[joint],
        )

    def test_prismatic_limits_in_meters(self):
        joint = Joint(
            "prismatic", "base", "child",
            range_deg=(-8, 12),
            description="finger slide",
            parent_anchor="front", child_anchor="back",
            axis="x",
        )
        assembly = self._make_assembly(joint)
        converter = AssemblyToURDF(assembly)
        converter.convert()
        joints = converter.get_joints()
        assert len(joints) == 1
        j = joints[0]
        # range_deg [-8, 12] (millimeters) -> meters [-0.008, 0.012]
        assert j.lower == round(_mm_to_m(-8), 4)
        assert j.upper == round(_mm_to_m(12), 4)

    def test_revolute_limits_still_in_radians(self):
        joint = Joint(
            "revolute", "base", "child",
            range_deg=(-90, 90),
            description="shoulder pitch",
            parent_anchor="front", child_anchor="back",
            axis="x",
        )
        assembly = self._make_assembly(joint)
        converter = AssemblyToURDF(assembly)
        converter.convert()
        joints = converter.get_joints()
        assert len(joints) == 1
        j = joints[0]
        # range_deg [-90, 90] -> radians [-pi/2, pi/2], rounded to 4 decimals
        assert j.lower == round(math.radians(-90), 4)
        assert j.upper == round(math.radians(90), 4)

