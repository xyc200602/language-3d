"""Tests for export_package tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lang3d.knowledge.mechanics import Assembly, Joint, Part
from lang3d.tools.base import ToolRegistry
from lang3d.tools.export_package import (
    ExportPackageTool,
    _build_subsystems,
    _freecad_ops_for_part,
    build_complex_robot,
    export_engineering_package,
    register_export_package_tools,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def complex_robot() -> Assembly:
    return build_complex_robot()


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "test_export"


# ============================================================================
# 1. build_complex_robot
# ============================================================================


def test_build_complex_robot(complex_robot: Assembly) -> None:
    assert complex_robot.name == "4-Wheel Mobile Robot with Dual Arms"
    assert len(complex_robot.parts) == 41
    assert len(complex_robot.joints) > 0


# ============================================================================
# 2. _freecad_ops_for_part
# ============================================================================


def test_freecad_ops_for_part() -> None:
    # Cylindrical part with outer_diameter
    p1 = Part("joint_housing", "joint", "test",
              dimensions=dict(outer_diameter=60, height=35))
    ops1 = _freecad_ops_for_part(p1)
    assert ops1[0]["type"] == "new_doc"
    assert ops1[1]["type"] == "make_cylinder"
    assert ops1[1]["radius"] == 30.0

    # Box part (plain structural part not matching any special family)
    p2 = Part("plate", "structural", "test",
              dimensions=dict(length=300, width=200, height=5))
    ops2 = _freecad_ops_for_part(p2)
    assert ops2[1]["type"] == "plate_with_holes"
    assert ops2[1]["length"] == 300

    # Cylinder with diameter (no length)
    p3 = Part("wheel", "structural", "test",
              dimensions=dict(diameter=65, height=26))
    ops3 = _freecad_ops_for_part(p3)
    assert ops3[1]["type"] == "make_cylinder"
    assert ops3[1]["radius"] == 32.5

    # Fallback
    p4 = Part("misc", "structural", "test",
              dimensions=dict(height=10))
    ops4 = _freecad_ops_for_part(p4)
    assert ops4[1]["type"] == "make_cylinder"

    # All ops should end with export_stl
    for ops in [ops1, ops2, ops3, ops4]:
        assert ops[-1]["type"] == "export_stl"


# ============================================================================
# 3. export_engineering_package structure
# ============================================================================


def test_export_engineering_package_structure(
    complex_robot: Assembly, output_dir: Path
) -> None:
    result = export_engineering_package(complex_robot, output_dir)

    # Check directory structure
    assert (output_dir / "freecad_scripts").is_dir()
    assert (output_dir / "firmware").is_dir()
    assert (output_dir / "ros2_package").is_dir()
    assert (output_dir / "subsystems").is_dir()

    # Check result keys
    assert "generated_files" in result
    assert "total_parts" in result
    assert "output_dir" in result


# ============================================================================
# 4. design_report.json
# ============================================================================


def test_design_report_json(
    complex_robot: Assembly, output_dir: Path
) -> None:
    export_engineering_package(complex_robot, output_dir)

    report_path = output_dir / "design_report.json"
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["total_parts"] >= 41
    assert "urdf_links" in report
    assert report["urdf_links"] >= 41
    assert "subsystems" in report
    assert "peak_power_w" in report


# ============================================================================
# 5. FreeCAD scripts exist
# ============================================================================


def test_freecad_scripts_exist(
    complex_robot: Assembly, output_dir: Path
) -> None:
    export_engineering_package(complex_robot, output_dir)

    fc_dir = output_dir / "freecad_scripts"
    py_files = list(fc_dir.glob("*.py"))
    assert len(py_files) == len(complex_robot.parts)

    for script in py_files:
        content = script.read_text(encoding="utf-8")
        assert "import FreeCAD" in content


# ============================================================================
# 6. Firmware files exist
# ============================================================================


def test_firmware_files_exist(
    complex_robot: Assembly, output_dir: Path
) -> None:
    export_engineering_package(complex_robot, output_dir)

    fw_dir = output_dir / "firmware"
    expected_files = [
        "robot_arm.ino",
        "ik_solver.h",
        "servo_driver.h",
        "dc_motor_driver.h",
        "dc_motor_driver.cpp",
        "odometry.cpp",
    ]
    for fname in expected_files:
        assert (fw_dir / fname).exists(), f"Missing firmware file: {fname}"


# ============================================================================
# 7. ROS2 package structure
# ============================================================================


def test_ros2_package_structure(
    complex_robot: Assembly, output_dir: Path
) -> None:
    export_engineering_package(complex_robot, output_dir)

    ros2_root = output_dir / "ros2_package"
    assert ros2_root.is_dir()

    # ROS2PackageBuilder creates a subdirectory named after the package
    pkg_dirs = [d for d in ros2_root.iterdir() if d.is_dir()]
    assert len(pkg_dirs) >= 1
    pkg_dir = pkg_dirs[0]

    assert (pkg_dir / "package.xml").exists()
    assert (pkg_dir / "CMakeLists.txt").exists()
    assert (pkg_dir / "urdf").is_dir()
    assert (pkg_dir / "launch").is_dir()
    assert (pkg_dir / "config").is_dir()


# ============================================================================
# 8. Markdown reports non-empty
# ============================================================================


def test_markdown_reports_nonempty(
    complex_robot: Assembly, output_dir: Path
) -> None:
    export_engineering_package(complex_robot, output_dir)

    md_reports = [
        "bom.md",
        "assembly_guide.md",
        "stability_report.md",
        "power_report.md",
        "wiring_diagram.md",
        "cable_routing_report.md",
    ]
    for fname in md_reports:
        path = output_dir / fname
        assert path.exists(), f"Missing report: {fname}"
        content = path.read_text(encoding="utf-8")
        assert len(content) > 0, f"Empty report: {fname}"


# ============================================================================
# 9. Subsystem JSON files
# ============================================================================


def test_subsystem_json_files(
    complex_robot: Assembly, output_dir: Path
) -> None:
    export_engineering_package(complex_robot, output_dir)

    ss_dir = output_dir / "subsystems"
    expected_subsystems = [
        "chassis.json",
        "arm_left.json",
        "arm_right.json",
        "electronics.json",
        "sensor_tower.json",
    ]
    for fname in expected_subsystems:
        path = ss_dir / fname
        assert path.exists(), f"Missing subsystem file: {fname}"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "name" in data
        assert "parts" in data
        assert "part_count" in data


# ============================================================================
# 10. Tool registration
# ============================================================================


def test_tool_registration() -> None:
    registry = ToolRegistry()
    register_export_package_tools(registry)

    tool = registry.get("export_package")
    assert tool is not None
    assert tool.name == "export_package"


# ============================================================================
# 11. Tool definition
# ============================================================================


def test_tool_definition() -> None:
    tool = ExportPackageTool()
    defn = tool.get_definition()

    assert defn.name == "export_package"
    params = defn.parameters
    assert "properties" in params
    prop_names = set(params["properties"].keys())
    assert "assembly_name" in prop_names
    assert "assembly_json" in prop_names
    assert "output_dir" in prop_names
    assert "actuator_ids" in prop_names
    assert "controller" in prop_names
    assert "components" in prop_names


# ============================================================================
# 12. Tool execute
# ============================================================================


def test_tool_execute(tmp_path: Path) -> None:
    tool = ExportPackageTool()
    result = tool.execute(
        assembly_name="complex_robot",
        output_dir=str(tmp_path / "exec_test"),
    )

    assert "Engineering Package Export Complete" in result
    assert "complex_robot" in result.lower() or "4-wheel" in result.lower()
    assert json.loads(result.split("```json\n")[1].split("\n```")[0])
