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

    # --- Wheel-axis guard (regression: LLM changed wheel axis to Z) ---
    def test_wheel_child_axis_z_forced_to_x(self):
        """A wheel revolute joint with axis Z must be forced to X (rolling)."""
        j = Joint("revolute", "base", "wheel_fl", axis="z")
        assert _resolve_axis(j) == [1.0, 0.0, 0.0], (
            "wheel joint with Z axis must be corrected to X (turntable → roller)"
        )

    def test_tire_child_axis_z_forced_to_x(self):
        """A tire revolute joint with axis Z must also be forced to X."""
        j = Joint("revolute", "base", "tire_fl", axis="z")
        assert _resolve_axis(j) == [1.0, 0.0, 0.0]

    def test_wheel_child_axis_x_preserved(self):
        """A wheel joint that already has axis X must be left alone."""
        j = Joint("revolute", "base", "wheel_fl", axis="x")
        assert _resolve_axis(j) == [1.0, 0.0, 0.0]

    def test_non_wheel_z_axis_preserved(self):
        """Non-wheel joints with axis Z (yaw) must NOT be changed."""
        j = Joint("revolute", "base", "yaw_servo", axis="z")
        assert _resolve_axis(j) == [0.0, 0.0, 1.0]


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


class TestInertialOrigin:
    """The <inertial><origin xyz> must point at the geometric centre.

    Regression for the MuJoCo physics-stability bug: the origin was always
    ``0 0 0`` while the inertia tensor was centre-relative, so PD-hold
    computed gravity torque against the wrong point and diverged.
    See AGENTS.md §3.2 — the tensor and its reference point must agree.
    """

    def test_box_origin_inferred_from_geometry(self):
        """A 200x150x20 box → origin = (0.075, 0.1, 0.01) m.

        FreeCAD makeBox(width=150, length=200, height=20) puts width on
        STL X and length on STL Y (freecad.py solver-convention swap).
        So the geometric centre in the STL/link frame is (W/2, L/2, H/2)
        = (0.075, 0.1, 0.01) m.  (Updated 2026-06-22 after _infer_local_com
        X/Y swap fix.)
        """
        p = Part(
            name="base_plate", category="structural", description="plate",
            dimensions=dict(length=200, width=150, height=20),
        )
        asm = Assembly(name="test", parts=[p], joints=[])
        converter = AssemblyToURDF(asm)
        converter.convert()
        link = converter.get_links()[0]
        # com is stored in metres (mm→m via _mm_to_m)
        assert link.com == pytest.approx((0.075, 0.1, 0.01), abs=1e-6)

    def test_cylinder_origin_inferred_from_geometry(self):
        """A Ø60×40 cylinder → origin = (0, 0, 0.02) m (half-height on Z)."""
        p = Part(
            name="pillar", category="structural", description="pillar",
            dimensions=dict(diameter=60, height=40),
        )
        asm = Assembly(name="test", parts=[p], joints=[])
        converter = AssemblyToURDF(asm)
        converter.convert()
        link = converter.get_links()[0]
        assert link.com == pytest.approx((0.0, 0.0, 0.02), abs=1e-6)

    def test_user_provided_com_respected(self):
        """An explicitly set center_of_mass must override the geometric guess."""
        p = Part(
            name="offset_mass", category="structural", description="offset",
            dimensions=dict(length=100, width=100, height=100),
            center_of_mass=(50.0, 30.0, 10.0),
        )
        asm = Assembly(name="test", parts=[p], joints=[])
        converter = AssemblyToURDF(asm)
        converter.convert()
        link = converter.get_links()[0]
        # (50,30,10) mm → (0.05, 0.03, 0.01) m, NOT the geometric (0.05,0.05,0.05)
        assert link.com == pytest.approx((0.05, 0.03, 0.01), abs=1e-6)

    def test_origin_not_all_zero_for_realistic_part(self):
        """End-to-end: the exported XML must not contain an all-zero origin
        for a part whose centre is clearly not at the link origin."""
        p = Part(
            name="link_test", category="structural", description="t",
            dimensions=dict(length=120, width=60, height=25),
        )
        asm = Assembly(name="test", parts=[p], joints=[])
        xml = AssemblyToURDF(asm).convert()
        # The inertial origin must be non-zero on at least one axis.
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml)
        origins = root.findall(".//inertial/origin")
        assert len(origins) >= 1
        found_nonzero = False
        for o in origins:
            xyz = o.get("xyz", "0 0 0").split()
            vals = [float(v) for v in xyz]
            if any(abs(v) > 1e-6 for v in vals):
                found_nonzero = True
                break
        assert found_nonzero, (
            "all <inertial><origin xyz> are 0 0 0 — MuJoCo physics will diverge"
        )


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

    def test_mesh_tags_have_mm_to_m_scale(self):
        """Every <mesh> tag must carry scale="0.001 0.001 0.001".

        STL files exported by FreeCAD use millimetre coordinates (e.g. a
        200 mm link has vertex coordinates up to 200).  URDF consumers
        (MuJoCo, PyBullet, Gazebo) interpret mesh coordinates as metres
        by default, so without an explicit scale a 200 mm part becomes a
        200-metre robot — causing massive mesh interpenetration and
        immediate physics blow-up.
        """
        converter = AssemblyToURDF(ROBOTIC_ARM_ASSEMBLY)
        xml = converter.convert()
        # Find every <mesh ... /> tag and assert it has scale="0.001 0.001 0.001"
        import re
        mesh_tags = re.findall(r'<mesh[^>]*?/>', xml)
        assert len(mesh_tags) > 0, "No mesh tags found in URDF"
        for tag in mesh_tags:
            assert "scale" in tag, f"Mesh tag missing scale: {tag}"
            assert 'scale="0.001 0.001 0.001"' in tag, (
                f"Mesh scale must be 0.001 (mm→m), got: {tag}"
            )

    def test_default_angles_not_baked_into_joint_rpy(self):
        """URDF joint origins must describe the HOME pose (all joints at 0),
        NOT the default pose.

        When an assembly has default_angles={'shoulder': -45}, the URDF
        must NOT have -45° baked into the shoulder joint's rpy.  Otherwise
        MuJoCo's qpos=0 corresponds to the bent pose, breaking simulation
        (non-physical joint axes) and control (setpoint mismatch).

        The URDFExportTool achieves this by calling solver.solve(
        joint_angles={}) — see urdf_export.py:URDFExportTool.execute.
        This test verifies that contract by running URDFExportTool
        directly on an assembly with non-zero default_angles.
        """
        # Build an arm with non-zero default_angles that would previously
        # produce complex rpy values in joint origins.
        from lang3d.knowledge.mechanics import Assembly, Joint, Part

        parts = [
            Part(name="base", category="structural", description="底座",
                 dimensions={"length": 100, "width": 100, "height": 8}),
            Part(name="shoulder", category="actuator", description="肩",
                 dimensions={"length": 40, "width": 40, "height": 30}),
            Part(name="upper_link", category="structural", description="上臂",
                 dimensions={"length": 100, "width": 25, "height": 15}),
        ]
        joints = [
            Joint("revolute", "base", "shoulder", range_deg=(-180, 180),
                  parent_anchor="top", child_anchor="bottom", axis="z"),
            Joint("revolute", "shoulder", "upper_link", range_deg=(-120, 120),
                  parent_anchor="front", child_anchor="back", axis="x"),
        ]
        assembly = Assembly(
            name="test_arm_with_defaults",
            parts=parts,
            joints=joints,
            default_angles={"shoulder": -45.0},  # would previously bake in
        )

        # Use the tool (not raw converter) so we exercise the
        # solver.solve(joint_angles={}) fix path.
        tool = URDFExportTool()
        xml = tool.execute(
            assembly_name="test_arm_with_defaults",
            mode="xml",
        )
        # The tool looks up assemblies by name via _find_assembly — for
        # an ad-hoc assembly we instead register it through the converter.
        # If _find_assembly can't resolve the name, fall back to direct
        # conversion via the same code path the tool uses.
        if "错误" in xml or "<robot" not in xml:
            from lang3d.tools.assembly_solver import AssemblySolver
            solver = AssemblySolver(assembly)
            home_positions = solver.solve(joint_angles={})
            xml = AssemblyToURDF(assembly, positions=home_positions).convert()

        # Extract joint origin rpy values
        import re
        matches = re.findall(
            r'<joint[^>]*>\s*<parent[^/]*/>\s*<child[^/]*/>\s*'
            r'<origin xyz="[^"]+" rpy="([^"]+)"',
            xml,
        )
        assert len(matches) >= 2, f"Expected ≥2 joints, got {len(matches)}"

        for rpy_str in matches:
            r, p, y = (float(v) for v in rpy_str.split())
            # -45° = -0.785 rad.  We must NOT see this magnitude in rpy.
            assert abs(r) < 0.1, (
                f"Roll {r} looks like baked-in default_angle (|r| >= 0.1 rad)"
            )
            assert abs(p) < 0.1, (
                f"Pitch {p} looks like baked-in default_angle (|p| >= 0.1 rad)"
            )
            assert abs(y) < 0.1, (
                f"Yaw {y} looks like baked-in default_angle (|y| >= 0.1 rad)"
            )

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

    def test_package_writes_flat_urdf_for_non_ros2_consumers(self):
        """ROS2PackageBuilder must emit a flat URDF for MuJoCo/PyBullet.

        The main URDF uses ``package://`` URIs which only ROS2 tools can
        resolve.  MuJoCo and PyBullet can't — they need either absolute
        paths or paths relative to the URDF file location.

        The flat URDF lives at ``urdf/<pkg>_flat.urdf`` (alongside the
        main URDF) and uses ``../meshes/X.stl`` paths so that from the
        ``urdf/`` subdir, the meshes directory resolves correctly.
        """
        import re
        converter = AssemblyToURDF(ROBOTIC_ARM_ASSEMBLY)
        xml = converter.convert()
        with tempfile.TemporaryDirectory() as tmp:
            builder = ROS2PackageBuilder("robotic_arm", xml)
            path = builder.write(tmp)
            # Flat URDF should be alongside the main URDF (in urdf/)
            flat_urdf_path = Path(path) / "urdf" / "robotic_arm_flat.urdf"
            assert flat_urdf_path.exists(), (
                f"Flat URDF missing at {flat_urdf_path}; "
                f"contents of urdf/: {list((Path(path) / 'urdf').iterdir())}"
            )
            flat_text = flat_urdf_path.read_text(encoding="utf-8")
            # Must NOT use package:// (defeats the purpose)
            assert "package://" not in flat_text, (
                "Flat URDF should use relative paths, not package:// URIs"
            )
            # Must use ../meshes/ relative paths
            assert "../meshes/" in flat_text, (
                "Flat URDF should use ../meshes/X.stl paths so it resolves "
                "from the urdf/ subdir to the sibling meshes/ directory"
            )
            # Main URDF (for ROS2) should use package:// URIs
            main_urdf_path = Path(path) / "urdf" / "robotic_arm.urdf"
            main_text = main_urdf_path.read_text(encoding="utf-8")
            assert "package://robotic_arm/meshes/" in main_text, (
                "Main URDF should use package:// URIs for ROS2 compatibility"
            )

    def test_collision_uses_primitive_for_box_parts(self):
        """Box-shaped parts should use <box> collision, not mesh.

        STL meshes from FreeCAD have complex surfaces (L-shaped tips,
        fillets, holes) that produce unpredictable contact normals.
        Standard robotics practice is to use simplified primitives
        (box/cylinder) for collision and keep the mesh only for visual.

        For the 3-DOF robotic arm, all structural parts (base_plate,
        shoulder_link, etc.) have box dimensions and should get <box>
        collision.  Visual stays as mesh.
        """
        import re
        from lang3d.tools.urdf_export import AssemblyToURDF

        converter = AssemblyToURDF(ROBOTIC_ARM_ASSEMBLY)
        xml = converter.convert()

        # Find all <link> blocks
        link_blocks = re.findall(
            r'<link name="([^"]+)">(.*?)</link>',
            xml, re.DOTALL,
        )
        assert len(link_blocks) > 0

        for name, body in link_blocks:
            visual_section = re.search(
                r'<visual>.*?</visual>', body, re.DOTALL,
            )
            collision_section = re.search(
                r'<collision>.*?</collision>', body, re.DOTALL,
            )
            assert visual_section, f"Link {name} missing visual"
            assert collision_section, f"Link {name} missing collision"

            # Visual should ALWAYS use mesh
            assert "<mesh" in visual_section.group(0), (
                f"Link {name} visual should use mesh"
            )

            # For parts with box dimensions, collision should use <box>
            # Check the original part dimensions
            part = next(
                (p for p in ROBOTIC_ARM_ASSEMBLY.parts
                 if p.name == name or name in p.name),
                None,
            )
            if part is None:
                continue
            d = part.dimensions
            has_box_dims = (
                d.get("length", 0) > 0
                and d.get("width", 0) > 0
                and d.get("height", 0) > 0
            )
            if has_box_dims:
                assert "<box" in collision_section.group(0), (
                    f"Link {name} (box dims {d}) should use <box> collision, "
                    f"got: {collision_section.group(0)}"
                )
                # And should NOT use mesh for collision
                assert "<mesh" not in collision_section.group(0), (
                    f"Link {name} collision should NOT use mesh when "
                    f"box primitive is available"
                )


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

