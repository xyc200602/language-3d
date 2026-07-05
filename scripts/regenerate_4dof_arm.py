"""DIAGNOSTIC ONLY — DO NOT CITE AS BENCHMARK EVIDENCE.

This script regenerates the 4dof_arm example using a hand-written assembly
+ pre-existing STLs, bypassing the LLM generation and VLM verification loop
(verification_status="PASSED" is injected). It exists to verify that the
export/URDF/MuJoCo pipeline fixes work together on known-good geometry.

It is NOT a benchmark run — it does not exercise the NL→assembly→VLM loop
that the e2e production test validates. Benchmark scores in the paper come
exclusively from tests/test_e2e_production.py, which runs the full pipeline.

Usage:
    python scripts/regenerate_4dof_arm.py

Verifies that all three pipeline fixes work together:
  1. URDF mesh tags have scale="0.001 0.001 0.001" (mm→m)
  2. URDF joint origins use home pose (no default_angles baked in)
  3. ROS2 package URDF uses package:// URIs + flat URDF for MuJoCo/PyBullet

Usage:
    python scripts/regenerate_4dof_arm.py

Loads the existing 4dof_arm STLs (from data/e2e_results/...) to skip the
slow FreeCAD subprocess, runs export_engineering_package with the fixed
code, then loads the resulting URDF in MuJoCo via sim_mujoco to verify
physics is stable.
"""

from __future__ import annotations

import io
import sys
import os
from pathlib import Path

# Make src importable when running this script directly
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

# Force UTF-8 stdout so Chinese diagnostic text doesn't crash on Windows GBK
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def main() -> None:
    from lang3d.knowledge.mechanics import Assembly, Joint, Part
    from lang3d.tools.export_package import export_engineering_package

    # ---- Reconstruct the 4dof_arm assembly (matches the original example) ----
    parts = [
        Part(name="base_plate", category="structural",
             description="底座安装板",
             material="Aluminum",
             dimensions={"length": 200, "width": 150, "height": 8}),
        Part(name="shoulder_joint", category="actuator",
             description="肩部旋转舵机",
             material="Steel",
             dimensions={"diameter": 40, "height": 35}),
        Part(name="shoulder_link", category="structural",
             description="肩部连杆",
             material="Aluminum",
             dimensions={"length": 120, "width": 25, "height": 15}),
        Part(name="elbow_joint", category="actuator",
             description="肘部舵机",
             material="Steel",
             dimensions={"diameter": 36, "height": 30}),
        Part(name="elbow_link", category="structural",
             description="肘部连杆",
             material="Aluminum",
             dimensions={"length": 100, "width": 25, "height": 15}),
        Part(name="wrist_joint", category="actuator",
             description="腕部舵机",
             material="Steel",
             dimensions={"diameter": 28, "height": 28}),
        Part(name="wrist_link", category="structural",
             description="腕部连杆",
             material="Aluminum",
             dimensions={"length": 60, "width": 20, "height": 12}),
        Part(name="gripper_base", category="mechanical",
             description="夹爪基座",
             material="PLA",
             dimensions={"length": 28, "width": 50, "height": 32}),
        Part(name="gripper_servo", category="actuator",
             description="夹爪驱动舵机SG90",
             material="Steel",
             dimensions={"length": 23, "width": 12, "height": 22}),
        Part(name="gripper_finger_left", category="mechanical",
             description="夹爪左手指",
             material="PLA",
             dimensions={"length": 60, "width": 14, "height": 28}),
        Part(name="gripper_finger_right", category="mechanical",
             description="夹爪右手指",
             material="PLA",
             dimensions={"length": 60, "width": 14, "height": 28}),
    ]
    joints = [
        Joint("revolute", "base_plate", "shoulder_joint",
              range_deg=(-180, 180), description="底座旋转",
              parent_anchor="top", child_anchor="bottom", axis="z"),
        Joint("revolute", "shoulder_joint", "shoulder_link",
              range_deg=(-120, 120), description="肩部俯仰",
              parent_anchor="front", child_anchor="back", axis="x"),
        Joint("revolute", "shoulder_link", "elbow_joint",
              range_deg=(-150, 150), description="肘部俯仰",
              parent_anchor="front", child_anchor="back", axis="x"),
        Joint("fixed", "elbow_joint", "elbow_link",
              description="固定", parent_anchor="front", child_anchor="back"),
        Joint("fixed", "elbow_link", "wrist_joint",
              description="固定", parent_anchor="front", child_anchor="back"),
        Joint("revolute", "wrist_joint", "wrist_link",
              range_deg=(-180, 180), description="腕部旋转",
              parent_anchor="front", child_anchor="back", axis="y"),
        Joint("fixed", "wrist_link", "gripper_base",
              description="固定", parent_anchor="front", child_anchor="back"),
        Joint("fixed", "gripper_base", "gripper_servo",
              description="SG90 安装", parent_anchor="top", child_anchor="bottom"),
        Joint("prismatic", "gripper_base", "gripper_finger_left",
              range_deg=(-8, 12), description="夹爪左手指",
              parent_anchor="front", child_anchor="back",
              offset=(-16, 0, 0), axis="x"),
        Joint("prismatic", "gripper_base", "gripper_finger_right",
              range_deg=(-8, 12), description="夹爪右手指",
              parent_anchor="front", child_anchor="back",
              offset=(16, 0, 0), axis="x"),
    ]
    assembly = Assembly(
        name="4dof_robot_arm",
        parts=parts,
        joints=joints,
        description="4自由度单机械臂（端到端验证用）",
        default_angles={
            "shoulder_joint": 0,
            "shoulder_link": -45,
            "elbow_joint": -30,
            "wrist_link": 15,
        },
    )

    # ---- Locate existing STLs to skip FreeCAD subprocess ----
    existing_stls = (
        Path(__file__).resolve().parent.parent
        / "data" / "e2e_results" / "4dof_arm_20260615_021554"
        / "engineering_package" / "stl_parts"
    )
    if not existing_stls.exists():
        print(f"ERROR: Existing STLs not found at {existing_stls}")
        sys.exit(1)

    # ---- Output to a fresh directory ----
    out_dir = Path(__file__).resolve().parent.parent / "data" / "e2e_results" / "4dof_arm_regenerated"
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print(f"=== Running export_engineering_package ===")
    print(f"  assembly: {assembly.name} ({len(parts)} parts, {len(joints)} joints)")
    print(f"  existing_stl_dir: {existing_stls}")
    print(f"  output_dir: {out_dir}")
    print()

    # Disable VLM verification (no API key needed for this test)
    os.environ.pop("GLM_API_KEY", None)

    result = export_engineering_package(
        assembly=assembly,
        output_dir=out_dir,
        existing_stl_dir=existing_stls,
        verification_status="PASSED",  # skip VLM loop
    )

    print(f"=== Export finished ===")
    print(f"  generated_files: {len(result.get('generated_files', []))}")
    print(f"  urdf_links: {result.get('urdf_links', '?')}")
    print(f"  urdf_joints: {result.get('urdf_joints', '?')}")
    print(f"  total_mass_kg: {result.get('total_mass_kg', '?')}")
    print()

    # ---- Inspect the generated URDF ----
    urdf_main = out_dir / "ros2_package" / "4dof_robot_arm" / "urdf" / "4dof_robot_arm.urdf"
    urdf_flat = out_dir / "ros2_package" / "4dof_robot_arm" / "urdf" / "4dof_robot_arm_flat.urdf"
    urdf_root = out_dir / "urdf.xml"

    print(f"=== URDF file checks ===")
    print(f"  {urdf_main.name} exists: {urdf_main.exists()}")
    print(f"  {urdf_flat.name} exists: {urdf_flat.exists()}")
    print(f"  urdf.xml at root exists: {urdf_root.exists()}")

    import re
    if urdf_main.exists():
        text = urdf_main.read_text(encoding="utf-8")
        # Fix 1: mesh scale
        scales = re.findall(r'scale="([^"]+)"', text)
        print(f"\n  Fix 1 (mesh scale):")
        print(f"    scale attributes: {len(scales)}")
        if scales:
            print(f"    sample: {scales[0]}")
            all_correct = all(s == "0.001 0.001 0.001" for s in scales)
            print(f"    all 0.001: {all_correct}")

        # Fix 2: no default_angles in rpy
        joint_origins = re.findall(
            r'<joint name="([^"]+)"[^>]*>.*?<origin xyz="[^"]+" rpy="([^"]+)"',
            text, re.DOTALL
        )
        print(f"\n  Fix 2 (no default_angles in rpy):")
        print(f"    default_angles = {assembly.default_angles}")
        max_rpy_deg = 0
        for name, rpy_str in joint_origins:
            rpy_rad = [float(x) for x in rpy_str.split()]
            rpy_deg = [round(x * 57.2958, 1) for x in rpy_rad]
            max_abs = max(abs(x) for x in rpy_deg)
            max_rpy_deg = max(max_rpy_deg, max_abs)
            print(f"      {name:45s} rpy_deg=({rpy_deg[0]:+6.1f},{rpy_deg[1]:+6.1f},{rpy_deg[2]:+6.1f})")
        # Default angles contain -45, -30, 15. None should appear in rpy.
        print(f"    max |rpy|: {max_rpy_deg}° (structural rotations like ±90° are OK)")

        # Fix 3: package:// URIs
        print(f"\n  Fix 3 (mesh paths):")
        pkg_uris = len(re.findall(r'package://', text))
        print(f"    package:// URIs in main URDF: {pkg_uris}")

    if urdf_flat.exists():
        flat_text = urdf_flat.read_text(encoding="utf-8")
        relative_paths = len(re.findall(r'\.\./meshes/', flat_text))
        print(f"    ../meshes/ paths in flat URDF: {relative_paths}")

    # ---- Load in MuJoCo via sim_mujoco tool ----
    print(f"\n=== MuJoCo validation (sim_mujoco) ===")
    from lang3d.tools.sim_mujoco import SimMujocoTool

    # Try loading the flat URDF (works without ROS2 package:// resolution)
    flat_urdf_path = str(urdf_flat)
    sim_result = SimMujocoTool().execute(
        urdf_path=flat_urdf_path,
        mode="validate",
        duration_sec=1.0,
    )
    # Extract key lines only
    for line in sim_result.split("\n"):
        if any(k in line for k in ("加载结果", "MuJoCo body", "Mesh 路径警告",
                                    "Rewrote", "结构验证", "物理稳定",
                                    "Mesh 自动修复", "能动关节数")):
            print(f"  {line.strip()}")

    print(f"\n=== Done. Output dir: {out_dir} ===")


if __name__ == "__main__":
    main()
