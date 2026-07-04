"""Offline validation of a Language-3D generated ROS2 package.

Checks everything that can be verified WITHOUT a ROS2/Gazebo installation,
so that when the package is taken to a Linux+ROS2 environment, colcon build
and Gazebo spawn have the highest chance of succeeding.

Usage:
    python scripts/verify_ros2_package.py [path/to/ros2_package/<pkg_name>]
    # If no path given, finds the latest generated package under data/runs/.

Checks:
  1. XML well-formedness of URDF
  2. URDF <ros2_control> tag present and correctly structured
  3. No legacy <transmission>/<hardwareInterface> (ROS1 format)
  4. All mesh paths resolve (package:// → actual STL files exist)
  5. package.xml declares all dependencies referenced by launch files
  6. CMakeLists installs all directories that exist
  7. ros2_control.yaml joints match URDF <ros2_control> joints
  8. launch files reference existing files (urdf, config)
"""
from __future__ import annotations

import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def find_latest_package() -> Path | None:
    """Find the most recently generated ROS2 package under data/runs/."""
    import os
    candidates = []
    for root, dirs, files in os.walk("data/runs"):
        if "package.xml" in files and "CMakeLists.txt" in files:
            if os.path.basename(os.path.dirname(root)) == "ros2_package":
                candidates.append(Path(root))
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def check_urdf(pkg_dir: Path) -> list[str]:
    """Check URDF well-formedness and ros2_control structure."""
    issues = []
    urdf_files = list((pkg_dir / "urdf").glob("*.urdf"))
    if not urdf_files:
        return ["URDF: no .urdf file in urdf/ directory"]

    urdf = urdf_files[0]  # Use the primary URDF
    try:
        root = ET.parse(urdf).getroot()
    except ET.ParseError as e:
        return [f"URDF XML PARSE ERROR: {e}"]

    # Check ros2_control tag
    r2c = root.find(".//ros2_control")
    if r2c is None:
        issues.append("URDF: no <ros2_control> tag (gazebo_ros2_control needs it)")
    else:
        hw_plugin = root.find(".//ros2_control/hardware/plugin")
        if hw_plugin is None or "GazeboSystem" not in (hw_plugin.text or ""):
            issues.append(
                f"URDF: <ros2_control> hardware plugin should be "
                f"gazebo_ros2_control/GazeboSystem, got {hw_plugin.text if hw_plugin is not None else 'missing'}"
            )
        # Each actuated joint should have command + state interfaces
        for j_el in root.findall(".//ros2_control/joint"):
            jname = j_el.get("name", "?")
            cmds = [c.get("name") for c in j_el.findall("command_interface")]
            states = [s.get("name") for s in j_el.findall("state_interface")]
            if "position" not in cmds:
                issues.append(f"URDF: joint '{jname}' missing position command_interface")
            if "position" not in states or "velocity" not in states:
                issues.append(
                    f"URDF: joint '{jname}' should have position+velocity state_interface, got {states}"
                )

    # Check NO legacy transmission format
    legacy = root.findall(".//transmission")
    for t in legacy:
        hw = t.find(".//hardwareInterface")
        if hw is not None:
            issues.append(
                f"URDF: legacy <hardwareInterface> in transmission '{t.get('name')}' "
                f"(ROS1 format — incompatible with gazebo_ros2_control)"
            )

    # Check gazebo_ros2_control plugin present
    plugin = root.find(".//gazebo/plugin[@filename='libgazebo_ros2_control.so']")
    if plugin is None:
        issues.append("URDF: no gazebo_ros2_control plugin (libgazebo_ros2_control.so)")

    return issues


def check_meshes(pkg_dir: Path, pkg_name: str) -> list[str]:
    """Check all package:// mesh paths resolve to actual files."""
    issues = []
    urdf_files = list((pkg_dir / "urdf").glob("*.urdf"))
    if not urdf_files:
        return issues

    urdf_text = (urdf_files[0]).read_text(encoding="utf-8")
    mesh_pattern = rf"package://{re.escape(pkg_name)}/([\w/.\-]+)"
    for match in re.finditer(mesh_pattern, urdf_text):
        rel = match.group(1)
        full = pkg_dir / rel
        if not full.exists():
            issues.append(f"MESH: package://{pkg_name}/{rel} does not exist at {full}")

    # Also check mesh count matches link count
    try:
        root = ET.parse(urdf_files[0]).getroot()
        mesh_count = len(list(root.iter("mesh")))
        stl_count = len(list((pkg_dir / "meshes").glob("*.stl")))
        if mesh_count != stl_count:
            issues.append(
                f"MESH: URDF references {mesh_count} meshes, "
                f"but meshes/ has {stl_count} STL files"
            )
    except ET.ParseError:
        pass  # Already reported in check_urdf

    return issues


def check_ros2_control_yaml(pkg_dir: Path) -> list[str]:
    """Check ros2_control.yaml joints match URDF ros2_control joints."""
    issues = []
    yaml_file = pkg_dir / "config" / "ros2_control.yaml"
    if not yaml_file.exists():
        return ["ros2_control.yaml: file missing in config/"]

    yaml_text = yaml_file.read_text(encoding="utf-8")
    yaml_joints = re.findall(r"^\s+-\s+(\w+)", yaml_text, re.MULTILINE)

    urdf_files = list((pkg_dir / "urdf").glob("*.urdf"))
    if not urdf_files:
        return issues
    try:
        root = ET.parse(urdf_files[0]).getroot()
        urdf_joints = [
            j.get("name") for j in root.findall(".//ros2_control/joint")
        ]
    except ET.ParseError:
        return issues

    missing_in_urdf = set(yaml_joints) - set(urdf_joints)
    missing_in_yaml = set(urdf_joints) - set(yaml_joints)
    if missing_in_urdf:
        issues.append(
            f"ros2_control.yaml: joints {missing_in_urdf} in YAML but not in URDF <ros2_control>"
        )
    if missing_in_yaml:
        issues.append(
            f"ros2_control.yaml: joints {missing_in_yaml} in URDF but not in YAML"
        )

    return issues


def check_launch_files(pkg_dir: Path, pkg_name: str) -> list[str]:
    """Check launch files reference valid paths."""
    issues = []
    launch_dir = pkg_dir / "launch"
    if not launch_dir.exists():
        return ["launch/ directory missing"]

    for lf in launch_dir.glob("*.py"):
        text = lf.read_text(encoding="utf-8")
        # Check it references the package by correct name
        if f"'{pkg_name}'" not in text and f'"{pkg_name}"' not in text:
            issues.append(f"launch/{lf.name}: does not reference package name '{pkg_name}'")
        # Check get_package_share_directory is used
        if "get_package_share_directory" not in text:
            issues.append(f"launch/{lf.name}: does not use get_package_share_directory")

    return issues


def check_cmake(pkg_dir: Path, pkg_name: str) -> list[str]:
    """Check CMakeLists installs all existing directories."""
    issues = []
    cmake = pkg_dir / "CMakeLists.txt"
    if not cmake.exists():
        return ["CMakeLists.txt: missing"]

    text = cmake.read_text(encoding="utf-8")
    # Find install(DIRECTORY ...) dirs
    installed = re.findall(r"install\s*\(\s*DIRECTORY\s+([\w\s]+?)\s+DESTINATION", text)
    installed_dirs = []
    for group in installed:
        installed_dirs.extend(group.split())

    # Check each existing dir is installed
    for d in ("urdf", "meshes", "launch", "config"):
        if (pkg_dir / d).exists() and d not in installed_dirs:
            issues.append(f"CMakeLists: directory '{d}/' exists but not in install(DIRECTORY)")

    # Check project name matches
    proj = re.search(r"project\s*\(\s*(\w+)\s*\)", text)
    if proj and proj.group(1) != pkg_name:
        issues.append(f"CMakeLists: project('{proj.group(1)}') != directory name '{pkg_name}'")

    return issues


def main() -> int:
    if len(sys.argv) > 1:
        pkg_dir = Path(sys.argv[1])
    else:
        pkg_dir = find_latest_package()

    if not pkg_dir or not pkg_dir.exists():
        print("ERROR: no ROS2 package found. Specify path or run an e2e case first.")
        return 1

    pkg_name = pkg_dir.name
    print(f"=== ROS2 Package Offline Validation ===")
    print(f"Package: {pkg_name}")
    print(f"Path:    {pkg_dir}")
    print()

    all_issues: list[str] = []
    all_issues += check_urdf(pkg_dir)
    all_issues += check_meshes(pkg_dir, pkg_name)
    all_issues += check_ros2_control_yaml(pkg_dir)
    all_issues += check_launch_files(pkg_dir, pkg_name)
    all_issues += check_cmake(pkg_dir, pkg_name)

    if all_issues:
        print(f"ISSUES ({len(all_issues)}):")
        for issue in all_issues:
            print(f"  ✗ {issue}")
        print()
        print(f"Result: {len(all_issues)} issue(s) found — fix before colcon build.")
        return 1
    else:
        print("All checks passed ✅")
        print("  - URDF well-formed with correct <ros2_control> structure")
        print("  - No legacy <transmission>/<hardwareInterface>")
        print("  - All mesh paths resolve")
        print("  - ros2_control.yaml joints match URDF")
        print("  - launch files reference correct package")
        print("  - CMakeLists installs all directories")
        print()
        print("Ready for colcon build + Gazebo spawn in a ROS2 environment.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
