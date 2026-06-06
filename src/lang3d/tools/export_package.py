"""Engineering package export tool.

Exports a complete engineering package for a robotic assembly including:
  - FreeCAD modeling scripts
  - URDF + ROS2 package
  - BOM (Bill of Materials)
  - Assembly guide
  - Firmware code
  - Wiring diagram
  - Cable routing report
  - Power budget report
  - Stability analysis report
  - Design report JSON + README
  - Subsystem decomposition

Public API:
  build_complex_robot      : Build the 41-part complex robot assembly
  _freecad_ops_for_part    : Generate FreeCAD operations for a part
  _build_subsystems        : Decompose assembly into subsystems
  export_engineering_package : Full 12-step export pipeline
  ExportPackageTool         : Agent tool
  register_export_package_tools : registration helper
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from ..knowledge.mechanics import (
    Assembly,
    Joint,
    Part,
    compute_assembly_mass,
)
from ..models.base import ToolDefinition
from .assembly_solver import AssemblySolver
from .base import Tool
from .pipeline_context import AssemblyContext

logger = logging.getLogger(__name__)


# ============================================================================
# Build complex robot assembly (41 parts)
# ============================================================================


def build_complex_robot() -> Assembly:
    """Build the 41-part 4-wheel mobile robot with dual arms."""
    parts: list[Part] = []
    joints: list[Joint] = []

    # ---- Chassis ----
    chassis_parts = [
        Part("base_plate", "structural", "主底盘板",
             dimensions=dict(length=300, width=200, height=5), material="Aluminum"),
        Part("top_plate", "structural", "顶板",
             dimensions=dict(length=280, width=180, height=3), material="Aluminum"),
        Part("standoff_fl", "structural", "前左铜柱",
             dimensions=dict(length=8, diameter=6, height=50)),
        Part("standoff_fr", "structural", "前右铜柱",
             dimensions=dict(length=8, diameter=6, height=50)),
        Part("standoff_rl", "structural", "后左铜柱",
             dimensions=dict(length=8, diameter=6, height=50)),
        Part("standoff_rr", "structural", "后右铜柱",
             dimensions=dict(length=8, diameter=6, height=50)),
        Part("motor_fl", "actuator", "前左驱动电机",
             dimensions=dict(length=40, width=30, height=25)),
        Part("motor_fr", "actuator", "前右驱动电机",
             dimensions=dict(length=40, width=30, height=25)),
        Part("motor_rl", "actuator", "后左驱动电机",
             dimensions=dict(length=40, width=30, height=25)),
        Part("motor_rr", "actuator", "后右驱动电机",
             dimensions=dict(length=40, width=30, height=25)),
        Part("wheel_fl", "structural", "前左轮",
             dimensions=dict(diameter=65, height=26)),
        Part("wheel_fr", "structural", "前右轮",
             dimensions=dict(diameter=65, height=26)),
        Part("wheel_rl", "structural", "后左轮",
             dimensions=dict(diameter=65, height=26)),
        Part("wheel_rr", "structural", "后右轮",
             dimensions=dict(diameter=65, height=26)),
        Part("battery_box", "battery", "锂电池组",
             dimensions=dict(length=150, width=60, height=40)),
        Part("motor_driver_board", "controller", "电机驱动板",
             dimensions=dict(length=70, width=50, height=10)),
        Part("encoder_fl", "sensor", "前左编码器",
             dimensions=dict(diameter=12, height=5)),
        Part("encoder_fr", "sensor", "前右编码器",
             dimensions=dict(diameter=12, height=5)),
        Part("encoder_rl", "sensor", "后左编码器",
             dimensions=dict(diameter=12, height=5)),
        Part("encoder_rr", "sensor", "后右编码器",
             dimensions=dict(diameter=12, height=5)),
    ]
    parts.extend(chassis_parts)

    # Standoffs from base_plate top face, distributed to 4 corners
    for s in ["fl", "fr", "rl", "rr"]:
        joints.append(Joint("fixed", "base_plate", f"standoff_{s}",
                            parent_anchor="top", child_anchor="bottom"))

    # top_plate is placed directly above base_plate center, on top of standoffs
    # Height = base_plate(5) + standoff(50) = 55mm above base center
    joints.append(Joint("fixed", "base_plate", "top_plate",
                        parent_anchor="top", child_anchor="bottom",
                        offset=(0, 0, 50), no_distribute=True))

    for s in ["fl", "fr", "rl", "rr"]:
        joints.append(Joint("fixed", "base_plate", f"motor_{s}",
                            parent_anchor="bottom", child_anchor="top"))
        joints.append(Joint("revolute", f"motor_{s}", f"wheel_{s}", axis="z",
                            range_deg=(-360, 360),
                            parent_anchor="left", child_anchor="bottom"))
        joints.append(Joint("fixed", f"motor_{s}", f"encoder_{s}",
                            parent_anchor="right", child_anchor="bottom"))
    joints.append(Joint("fixed", "base_plate", "battery_box",
                        parent_anchor="top", child_anchor="bottom",
                        offset=(0, 0, 5), no_distribute=True))

    # ---- IPC ----
    ipc_parts = [
        Part("ipc_bracket", "structural", "工控机支架",
             dimensions=dict(length=120, width=80, height=40)),
        Part("ipc_body", "controller", "工控机主体",
             dimensions=dict(length=110, width=75, height=30)),
        Part("ipc_fan", "structural", "散热风扇",
             dimensions=dict(diameter=40, height=10)),
    ]
    parts.extend(ipc_parts)
    # IPC centered on top_plate (own distribution group, no auto-distribute with arms)
    joints.append(Joint("fixed", "top_plate", "ipc_bracket",
                        parent_anchor="top", child_anchor="bottom",
                        offset=(0, 0, 0), distribution_group="ipc"))
    joints.append(Joint("fixed", "ipc_bracket", "ipc_body",
                        parent_anchor="top", child_anchor="bottom"))
    joints.append(Joint("fixed", "ipc_body", "ipc_fan",
                        parent_anchor="top", child_anchor="bottom"))

    # Motor driver board on top_plate, offset forward
    joints.append(Joint("fixed", "top_plate", "motor_driver_board",
                        parent_anchor="top", child_anchor="bottom",
                        offset=(0, 60, 0), distribution_group="driver"))

    # ---- Left arm & Right arm ----
    # Arms use distribution_group="arms" so they form their own 2-element sibling
    # group on top_plate's top face.  The solver's line-distribution spreads them
    # along X (tangent1 of the top face).  We keep explicit offsets to shift them
    # symmetrically along Y instead (left arm at Y=-70, right arm at Y=+70).
    arm_offsets = {"arm_l": (0, -70, 0), "arm_r": (0, 70, 0)}
    for side, prefix in [("左", "arm_l"), ("右", "arm_r")]:
        arm_parts = [
            Part(f"{prefix}_base", "joint", f"{side}臂底座旋转关节",
                 dimensions=dict(outer_diameter=80, height=40)),
            Part(f"{prefix}_shoulder", "joint", f"{side}臂肩关节",
                 dimensions=dict(outer_diameter=60, height=35)),
            Part(f"{prefix}_upper_link", "structural", f"{side}臂上臂",
                 dimensions=dict(length=150, width=40, height=30)),
            Part(f"{prefix}_elbow", "joint", f"{side}臂肘关节",
                 dimensions=dict(outer_diameter=50, height=30)),
            Part(f"{prefix}_forearm", "structural", f"{side}臂前臂",
                 dimensions=dict(length=120, width=35, height=25)),
            Part(f"{prefix}_wrist", "joint", f"{side}臂腕关节",
                 dimensions=dict(outer_diameter=40, height=25)),
            Part(f"{prefix}_gripper", "structural", f"{side}臂末端执行器",
                 dimensions=dict(length=60, width=30, height=20)),
        ]
        parts.extend(arm_parts)
        arm_offset = arm_offsets[prefix]
        joints.extend([
            Joint("revolute", "top_plate", f"{prefix}_base", (-180, 180),
                  f"{side}臂旋转", axis="z",
                  parent_anchor="top", child_anchor="bottom",
                  offset=arm_offset, distribution_group="arms"),
            Joint("revolute", f"{prefix}_base", f"{prefix}_shoulder", (-90, 90),
                  f"{side}肩俯仰", axis="y", parent_anchor="top", child_anchor="bottom"),
            Joint("fixed", f"{prefix}_shoulder", f"{prefix}_upper_link",
                  parent_anchor="top", child_anchor="bottom"),
            Joint("revolute", f"{prefix}_upper_link", f"{prefix}_elbow", (-135, 135),
                  f"{side}肘弯曲", axis="y", parent_anchor="top", child_anchor="bottom"),
            Joint("fixed", f"{prefix}_elbow", f"{prefix}_forearm",
                  parent_anchor="top", child_anchor="bottom"),
            Joint("revolute", f"{prefix}_forearm", f"{prefix}_wrist", (-180, 180),
                  f"{side}腕旋转", axis="z", parent_anchor="top", child_anchor="bottom"),
            Joint("fixed", f"{prefix}_wrist", f"{prefix}_gripper",
                  parent_anchor="top", child_anchor="bottom"),
        ])

    # ---- Sensor tower ----
    # Placed at rear center of top_plate
    sensor_parts = [
        Part("sensor_tower_post", "structural", "传感器塔立柱",
             dimensions=dict(diameter=20, height=120)),
        Part("imu_mount", "sensor", "IMU 安装座",
             dimensions=dict(length=25, width=15, height=5)),
        Part("lidar_mount", "sensor", "LiDAR 安装座",
             dimensions=dict(diameter=80, height=40)),
        Part("camera_bracket", "structural", "摄像头支架",
             dimensions=dict(length=30, width=20, height=15)),
    ]
    parts.extend(sensor_parts)
    joints.extend([
        Joint("fixed", "top_plate", "sensor_tower_post",
              parent_anchor="top", child_anchor="bottom",
              offset=(-100, 0, 0), distribution_group="sensor_tower"),
        Joint("fixed", "sensor_tower_post", "imu_mount",
              parent_anchor="top", child_anchor="bottom"),
        Joint("fixed", "sensor_tower_post", "lidar_mount",
              parent_anchor="top", child_anchor="bottom", offset=(0, 0, 40)),
        Joint("fixed", "sensor_tower_post", "camera_bracket",
              parent_anchor="top", child_anchor="bottom", offset=(0, 0, 80)),
    ])

    return Assembly(
        name="4-Wheel Mobile Robot with Dual Arms",
        description="4轮差速底盘移动机器人 + 工控机 + 双 3-DOF 机械臂",
        parts=parts,
        joints=joints,
    )


# ============================================================================
# FreeCAD operations for a part
# ============================================================================


def _freecad_ops_for_part(part: Part) -> list[dict]:
    """Generate FreeCAD operation list for a single part with engineering features."""
    from .part_feature_engine import generate_ops
    return generate_ops(part)


# ============================================================================
# Subsystem decomposition
# ============================================================================


def _build_subsystems(
    assembly: Assembly,
    positions: dict[str, Any],
) -> dict[str, list[str]]:
    """Decompose assembly into subsystems.

    Returns a dict mapping subsystem name -> list of part names.
    Default decomposition works for the complex_robot assembly pattern.
    """
    subsystems: dict[str, list[str]] = {
        "chassis": [
            p.name for p in assembly.parts
            if p.name in ["base_plate", "top_plate"]
            or p.name.startswith(("standoff_", "motor_", "wheel_", "encoder_"))
            or p.name in ("battery_box", "motor_driver_board")
        ],
        "arm_left": [
            p.name for p in assembly.parts if p.name.startswith("arm_l_")
        ],
        "arm_right": [
            p.name for p in assembly.parts if p.name.startswith("arm_r_")
        ],
        "ipc": [
            p.name for p in assembly.parts if "ipc" in p.name
        ],
        "sensor_tower": [
            p.name for p in assembly.parts
            if p.category == "sensor" or "sensor" in p.name
            or "lidar" in p.name or "camera" in p.name
        ],
    }
    return subsystems


# ============================================================================
# 12-step export pipeline
# ============================================================================


def export_engineering_package(
    assembly: Assembly,
    output_dir: Path,
    actuator_ids: list[str] | None = None,
    controller: str = "esp32",
    components: list[str] | None = None,
) -> dict[str, Any]:
    """Export a complete engineering package for the assembly.

    Args:
        assembly: The mechanical assembly to export.
        output_dir: Output directory path.
        actuator_ids: Actuator IDs (default: TT_MOTOR x4 + MG996R x6).
        controller: Controller type (default: esp32).
        components: Optional component filter (None = all).

    Returns:
        Dict with generated file list and key metrics.
    """
    if actuator_ids is None:
        actuator_ids = ["TT_MOTOR"] * 4 + ["MG996R"] * 6

    output_dir = Path(output_dir)
    generated_files: list[str] = []

    # ---- Create shared context ----
    ctx = AssemblyContext(assembly=assembly)

    # ---- Step 1: Solve assembly positions ----
    positions = ctx.ensure_positions()

    # ---- Step 1b: VLM visual verification ----
    # Try to run closed-loop VLM verification: render → VLM check → fix → re-solve
    vlm_result = None
    try:
        from ..models.glm import GLMBackend
        import os as _os

        api_key = _os.environ.get("GLM_API_KEY", "")
        base_url = _os.environ.get(
            "GLM_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4"
        )
        vision_model = _os.environ.get("VISION_MODEL", "GLM-4V-Plus")

        if api_key:
            vlm_backend = GLMBackend(
                api_key=api_key,
                base_url=base_url,
                vision_model=vision_model,
            )
            from ..agent.assembly_visual_verifier import verify_assembly_visual

            vlm_result = verify_assembly_visual(
                assembly=assembly,
                positions=positions,
                model_backend=vlm_backend,
                expected_layout=(
                    "4-wheeled differential mobile robot with: "
                    "rectangular chassis at origin, 4 wheels at corners, "
                    "dual 3-DOF arms symmetrically on left (Y=-70) and right (Y=+70), "
                    "IPC centered on top, sensor tower at rear"
                ),
                max_iterations=3,
                detail_level="detailed",
            )
            # If corrections were applied, re-solve positions
            if vlm_result.corrections_applied and not vlm_result.passed:
                from ..agent.assembly_visual_verifier import apply_corrections
                corrected_assembly = apply_corrections(assembly, vlm_result.corrections_applied)
                solver = AssemblySolver(corrected_assembly)
                positions = solver.solve()
                assembly = corrected_assembly
                ctx = AssemblyContext(assembly=assembly)
                positions = ctx.ensure_positions()
    except Exception as e:
        logger.warning("VLM verification skipped: %s", e)

    # ---- Step 2: Compute mass properties ----
    mass_result = ctx.ensure_mass()

    # ---- Step 3: Generate FreeCAD scripts ----
    from .freecad import _build_script, _build_batch_script
    fc_dir = output_dir / "freecad_scripts"
    fc_dir.mkdir(parents=True, exist_ok=True)

    # Build all part operation lists
    all_part_ops = []
    for part in assembly.parts:
        ops = _freecad_ops_for_part(part)
        # Replace workspace placeholder with actual path
        for op in ops:
            if op.get("type") == "export_stl" and "{WORKSPACE}" in op.get("path", ""):
                op["path"] = op["path"].replace("{WORKSPACE}", str(fc_dir.parent))
        all_part_ops.append(ops)

    # Write individual scripts (for user reference and debugging)
    for i, part in enumerate(assembly.parts):
        script = _build_script(all_part_ops[i])
        script_path = fc_dir / f"{part.name}.py"
        script_path.write_text(script, encoding="utf-8")
        generated_files.append(str(script_path))

    # ---- Step 3b: Render assembly ----
    from .freecad import (
        build_assembly_script,
        _shape_type_for_part,
        _subsystem_for_part,
        _run_freecad_script,
    )
    assembly_parts_info = []
    for p in assembly.parts:
        assembly_parts_info.append({
            "name": p.name,
            "shape_type": _shape_type_for_part(p),
            "dimensions": p.dimensions,
            "subsystem": _subsystem_for_part(p.name),
        })
    render_path = output_dir / "assembly"
    assembly_script = build_assembly_script(
        assembly_parts=assembly_parts_info,
        positions=positions,
        output_path=str(render_path),
    )
    (output_dir / "assembly_render_script.py").write_text(assembly_script, encoding="utf-8")
    generated_files.append(str(output_dir / "assembly_render_script.py"))
    # Execute the render if FreeCAD is available
    try:
        from .freecad import _find_freecad_python
        if _find_freecad_python():
            _run_freecad_script(assembly_script, timeout=300)
            if render_path.with_suffix(".FCStd").exists():
                generated_files.append(str(render_path.with_suffix(".FCStd")))
            if render_path.with_suffix(".stl").exists():
                generated_files.append(str(render_path.with_suffix(".stl")))
    except Exception:
        pass  # FreeCAD rendering is optional; skip if not available

    # Exploded view
    exploded_script = build_assembly_script(
        assembly_parts=assembly_parts_info,
        positions=positions,
        output_path=str(output_dir / "assembly_exploded"),
        exploded=True,
    )
    (output_dir / "assembly_exploded_script.py").write_text(exploded_script, encoding="utf-8")
    generated_files.append(str(output_dir / "assembly_exploded_script.py"))

    # ---- Step 4: Generate URDF + ROS2 package ----
    from .urdf_export import AssemblyToURDF, ROS2PackageBuilder
    converter = AssemblyToURDF(assembly)
    urdf_xml = converter.convert()
    (output_dir / "urdf.xml").write_text(urdf_xml, encoding="utf-8")
    generated_files.append(str(output_dir / "urdf.xml"))

    builder = ROS2PackageBuilder("mobile_robot_dual_arm", urdf_xml)
    ros2_dir = output_dir / "ros2_package"
    builder.write(str(ros2_dir))
    generated_files.append(str(ros2_dir))

    # ---- Step 5: Generate BOM ----
    from .bom_gen import generate_bom, format_bom_markdown
    bom = generate_bom(
        assembly,
        actuator_ids=actuator_ids,
        controller=controller,
    )
    bom_md = format_bom_markdown(bom)
    (output_dir / "bom.md").write_text(bom_md, encoding="utf-8")
    generated_files.append(str(output_dir / "bom.md"))

    # ---- Step 6: Generate assembly guide ----
    from .assembly_doc import generate_assembly_guide
    guide = generate_assembly_guide(
        assembly,
        actuator_ids=actuator_ids,
        controller=controller,
    )
    (output_dir / "assembly_guide.md").write_text(guide, encoding="utf-8")
    generated_files.append(str(output_dir / "assembly_guide.md"))

    # ---- Step 7: Generate firmware ----
    from .code_gen import (
        generate_firmware,
        gen_motor_driver_code,
        gen_odometry_code,
        generate_wiring,
    )
    fw_dir = output_dir / "firmware"
    fw_dir.mkdir(parents=True, exist_ok=True)

    arm_parts = [p for p in assembly.parts if p.name.startswith("arm_l_")]
    arm_joints = [j for j in assembly.joints
                  if j.parent.startswith("arm_l_") and j.child.startswith("arm_l_")]
    arm_assembly = Assembly(name="Left Arm", parts=arm_parts, joints=arm_joints)

    firmware = generate_firmware(
        arm_assembly,
        actuator_ids=["MG996R", "MG996R", "MG996R"],
        controller=controller,
    )
    for fname, content in firmware.items():
        (fw_dir / fname).write_text(content, encoding="utf-8")
        generated_files.append(str(fw_dir / fname))

    motors = [
        dict(motor_id="TT_MOTOR", encoder_id="HALL_TT_7PPR",
             pwm_pin=5 + i * 2, dir_pin1=6 + i * 2, dir_pin2=7 + i * 2,
             enc_a_pin=18 + i * 2, enc_b_pin=19 + i * 2)
        for i in range(4)
    ]
    motor_code = gen_motor_driver_code(motors)
    for fname, content in motor_code.items():
        (fw_dir / fname).write_text(content, encoding="utf-8")
        generated_files.append(str(fw_dir / fname))

    odo_code = gen_odometry_code(
        wheel_radius_mm=32.5,
        wheel_base_mm=200.0,
        encoder_ppr=7,
        gear_ratio=48.0,
    )
    (fw_dir / "odometry.cpp").write_text(odo_code, encoding="utf-8")
    generated_files.append(str(fw_dir / "odometry.cpp"))

    # ---- Step 8: Generate wiring diagram ----
    wiring = generate_wiring(
        actuator_ids=actuator_ids,
        controller=controller,
    )
    (output_dir / "wiring_diagram.md").write_text(wiring, encoding="utf-8")
    generated_files.append(str(output_dir / "wiring_diagram.md"))

    # ---- Step 9: Generate cable routing report ----
    from .cable_routing import (
        auto_detect_connections,
        build_3d_grid,
        find_cable_path,
        generate_cable_report,
    )
    cables = auto_detect_connections(assembly)
    grid = build_3d_grid(positions, assembly.parts)
    cable_paths = []
    for spec in cables:
        start_pos = positions.get(spec.start_connector, {}).get("position", [0, 0, 0])
        end_pos = positions.get(spec.end_connector, {}).get("position", [0, 0, 0])
        cp = find_cable_path(
            grid,
            start=(start_pos[0], start_pos[1], start_pos[2]),
            end=(end_pos[0], end_pos[1], end_pos[2]),
            spec=spec,
        )
        cable_paths.append(cp)
    cable_report = generate_cable_report(cable_paths, assembly.name)
    (output_dir / "cable_routing_report.md").write_text(cable_report, encoding="utf-8")
    generated_files.append(str(output_dir / "cable_routing_report.md"))

    # ---- Step 10: Generate power budget report ----
    from .power_budget import PowerBudgetCalculator
    calc = PowerBudgetCalculator("MobileRobotDualArm")
    calc.add_motor("Drive Motor", "TT_MOTOR", duty_cycle=0.5, quantity=4)
    calc.add_servo("Arm Servos", "MG996R", duty_cycle=0.3, quantity=6)
    calc.add_controller("IPC", tdp_w=15.0)
    calc.add_sensor_load("Sensors", power_w=2.0, quantity=3)
    power_report = calc.generate_report()
    (output_dir / "power_report.md").write_text(power_report, encoding="utf-8")
    generated_files.append(str(output_dir / "power_report.md"))

    peak_w = calc.compute_total_peak()
    avg_w = calc.compute_total_avg()
    batt_recs = calc.recommend_battery(runtime_target_h=0.5)

    # ---- Step 11: Generate stability report ----
    from .stability import (
        compute_support_polygon,
        compute_static_stability,
        check_tip_over_risk,
    )
    contacts = []
    for s in ["fl", "fr", "rl", "rr"]:
        pos = positions[f"wheel_{s}"]["position"]
        contacts.append([pos[0], pos[1], pos[2]])

    com = list(mass_result["center_of_mass_mm"])
    polygon = compute_support_polygon(contacts)
    poly_2d = [[p[0], p[1]] for p in (polygon if len(polygon) >= 3 else contacts)]
    static_stab = compute_static_stability(com, poly_2d)
    tip_risk = check_tip_over_risk(
        com=com, contact_points=contacts,
        mass_kg=mass_result["total_mass_kg"],
    )

    stab_lines = [
        "# Stability Analysis Report\n",
        f"## Robot: {assembly.name}\n",
        f"- Total mass: {mass_result['total_mass_kg']:.3f} kg",
        f"- Center of mass: ({com[0]:.1f}, {com[1]:.1f}, {com[2]:.1f}) mm\n",
        "## Support Polygon",
        f"- Contact points: {len(contacts)}",
        f"- Convex hull vertices: {len(polygon)}\n",
        "## Static Stability",
        f"- Stable: {static_stab.get('stable', 'N/A')}",
        f"- Margin: {static_stab.get('margin_mm', 'N/A'):.1f} mm\n",
        "## Tip-Over Risk Assessment",
        f"- Risk level: {tip_risk['risk_level']}",
        f"- Min stability margin: {tip_risk.get('min_stability_margin_mm', 'N/A')}",
        "",
    ]
    (output_dir / "stability_report.md").write_text(
        "\n".join(stab_lines), encoding="utf-8")
    generated_files.append(str(output_dir / "stability_report.md"))

    # ---- Step 12: Design report JSON + README + subsystem JSONs ----
    subsystems = ctx.ensure_subsystems()

    # Subsystem JSON files
    ss_dir = output_dir / "subsystems"
    ss_dir.mkdir(parents=True, exist_ok=True)
    for ss_name, ss_parts in subsystems.items():
        ss_data = {
            "name": ss_name,
            "parts": ss_parts,
            "part_count": len(ss_parts),
            "positions": {k: v for k, v in positions.items() if k in ss_parts},
        }
        (ss_dir / f"{ss_name}.json").write_text(
            json.dumps(ss_data, indent=2, ensure_ascii=False), encoding="utf-8")
        generated_files.append(str(ss_dir / f"{ss_name}.json"))

    # Design report JSON
    report = {
        "requirement": "4-wheel differential mobile robot + IPC + dual 3-DOF arms",
        "total_parts": len(assembly.parts),
        "total_joints": len(assembly.joints),
        "total_mass_kg": round(mass_result["total_mass_kg"], 3),
        "center_of_mass_mm": [round(v, 1) for v in com],
        "subsystems": {k: len(v) for k, v in subsystems.items()},
        "assembly_solved": len(positions) == len(assembly.parts),
        "urdf_links": len(converter.get_links()),
        "urdf_joints": len(converter.get_joints()),
        "cable_count": len(cable_paths),
        "peak_power_w": round(peak_w, 1),
        "avg_power_w": round(avg_w, 1),
        "stability": {
            "risk_level": tip_risk["risk_level"],
            "margin_mm": round(static_stab.get("margin_mm", 0), 1),
        },
        "battery_recommendations": [
            {
                "name": r.get("battery").name if hasattr(r.get("battery"), "name") else str(r.get("battery")),
                "runtime_h": r.get("runtime_h"),
                "margin_pct": r.get("margin_pct"),
                "capacity_ah": r.get("battery").capacity_ah if hasattr(r.get("battery"), "capacity_ah") else None,
                "voltage": r.get("battery").voltage if hasattr(r.get("battery"), "voltage") else None,
                "price_cny": r.get("battery").price_cny if hasattr(r.get("battery"), "price_cny") else None,
            }
            for r in batt_recs[:3]
        ],
        "categories": sorted({p.category for p in assembly.parts}),
    }
    (output_dir / "design_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    generated_files.append(str(output_dir / "design_report.json"))

    # README
    readme = f"""# {assembly.name}

## 概述

{assembly.description}

- **零件总数**: {len(assembly.parts)}
- **关节总数**: {len(assembly.joints)}
- **总质量**: {mass_result['total_mass_kg']:.2f} kg
- **子系统**: {', '.join(subsystems.keys())}

## 子系统

| 子系统 | 零件数 | 说明 |
|--------|--------|------|
| 底盘 (chassis) | {len(subsystems['chassis'])} | 底板 + 顶板 + 4 铜柱 + 4 电机 + 4 轮 + 4 编码器 + 电池 + 驱动板 |
| 左臂 (arm_left) | {len(subsystems['arm_left'])} | 底座旋转 → 肩 → 上臂 → 肘 → 前臂 → 腕 → 末端执行器 |
| 右臂 (arm_right) | {len(subsystems['arm_right'])} | 左臂镜像 |
| 工控机 (ipc) | {len(subsystems['ipc'])} | 支架 + 主体 + 散热风扇 |
| 传感器塔 (sensor_tower) | {len(subsystems['sensor_tower'])} | 立柱 + IMU + LiDAR + 摄像头 |

## 工程包内容

```
{output_dir.name}/
├── README.md                     # 本文件
├── design_report.json            # 设计指标摘要
├── bom.md                        # BOM 物料清单（含价格）
├── assembly_guide.md             # 装配指导书
├── stability_report.md           # 稳定性分析报告
├── power_report.md               # 功率预算报告
├── wiring_diagram.md             # 接线图
├── cable_routing_report.md       # 电缆走线报告
├── urdf.xml                      # URDF 机器人描述文件
├── freecad_scripts/              # FreeCAD 建模脚本 ({len(assembly.parts)} 个零件)
├── firmware/                     # 固件代码
│   ├── robot_arm.ino             # 机械臂主程序
│   ├── ik_solver.h               # 逆运动学求解器
│   ├── servo_driver.h            # 舵机驱动
│   ├── dc_motor_driver.h/.cpp    # 直流电机 PID 驱动
│   └── odometry.cpp              # 差速里程计
├── ros2_package/                 # ROS2 功能包
│   ├── urdf/                     # URDF + meshes
│   ├── launch/                   # 启动文件
│   ├── config/                   # 参数配置
│   ├── package.xml
│   └── CMakeLists.txt
└── subsystems/                   # 子系统分解
    ├── chassis.json
    ├── arm_left.json
    ├── arm_right.json
    ├── ipc.json
    └── sensor_tower.json
```

## 关键指标

| 指标 | 值 |
|------|-----|
| 总质量 | {mass_result['total_mass_kg']:.2f} kg |
| 质心位置 | ({com[0]:.1f}, {com[1]:.1f}, {com[2]:.1f}) mm |
| 峰值功耗 | {peak_w:.1f} W |
| 平均功耗 | {avg_w:.1f} W |
| 稳定性风险 | {tip_risk['risk_level']} |
| URDF 链接数 | {len(converter.get_links())} |
| URDF 关节数 | {len(converter.get_joints())} |
| 电缆数量 | {len(cable_paths)} |

## 使用说明

### FreeCAD 建模
```bash
freecadcmd freecad_scripts/base_plate.py
```

### ROS2 仿真
```bash
cp -r ros2_package/ ~/ros2_ws/src/
cd ~/ros2_ws && colcon build
ros2 launch mobile_robot_dual_arm display.launch.py
```

### 固件烧录
将 `firmware/` 目录中的代码烧录至 ESP32 控制器。

---
*Generated by Language-3D Agent on {date.today().isoformat()}*
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    generated_files.append(str(output_dir / "README.md"))

    return {
        "output_dir": str(output_dir),
        "generated_files": generated_files,
        "total_parts": len(assembly.parts),
        "total_joints": len(assembly.joints),
        "total_mass_kg": round(mass_result["total_mass_kg"], 3),
        "urdf_links": len(converter.get_links()),
        "urdf_joints": len(converter.get_joints()),
        "peak_power_w": round(peak_w, 1),
        "avg_power_w": round(avg_w, 1),
        "stability_risk": tip_risk["risk_level"],
        "cable_count": len(cable_paths),
        "subsystems": {k: len(v) for k, v in subsystems.items()},
    }


# ============================================================================
# Built-in assembly registry
# ============================================================================

_BUILTIN_ASSEMBLIES: dict[str, Assembly] = {}


def _get_builtin_assembly(name: str) -> Assembly | None:
    """Get a built-in assembly by name."""
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key in ("complex_robot", "4w_dual_arm", "4_wheel_mobile_robot_with_dual_arms"):
        return build_complex_robot()

    # Check lazy-loaded builtins
    if key in _BUILTIN_ASSEMBLIES:
        return _BUILTIN_ASSEMBLIES[key]

    return None


def _resolve_assembly_input(
    assembly_name: str | None,
    assembly_json: str | None,
) -> Assembly | None:
    """Resolve assembly from name or JSON string."""
    if assembly_json:
        try:
            data = json.loads(assembly_json)
            parts = []
            for pd in data.get("parts", []):
                parts.append(Part(
                    name=pd["name"],
                    category=pd.get("category", "structural"),
                    description=pd.get("description", ""),
                    material=pd.get("material", "PLA"),
                    dimensions=pd.get("dimensions", {}),
                ))
            joints = []
            for jd in data.get("joints", []):
                joints.append(Joint(
                    joint_type=jd.get("type", jd.get("joint_type", "fixed")),
                    parent=jd["parent"],
                    child=jd["child"],
                    range_deg=jd.get("range_deg"),
                    description=jd.get("description", ""),
                    axis=jd.get("axis"),
                    parent_anchor=jd.get("parent_anchor"),
                    child_anchor=jd.get("child_anchor"),
                    offset=tuple(jd["offset"]) if "offset" in jd else None,
                ))
            return Assembly(
                name=data.get("name", "Custom Assembly"),
                parts=parts,
                joints=joints,
                description=data.get("description", ""),
            )
        except Exception:
            pass

    if assembly_name:
        builtin = _get_builtin_assembly(assembly_name)
        if builtin:
            return builtin

    # Default to complex_robot
    return build_complex_robot()


# ============================================================================
# ExportPackageTool
# ============================================================================


class ExportPackageTool(Tool):
    """Export a complete engineering package for a robotic assembly."""

    name = "export_package"
    description = "Export a complete engineering package (URDF, BOM, firmware, FreeCAD scripts, reports) for a robotic assembly"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "assembly_name": {
                        "type": "string",
                        "description": "Name of the assembly to export (default: 'complex_robot'). Options: complex_robot, 4w_dual_arm",
                    },
                    "assembly_json": {
                        "type": "string",
                        "description": "JSON string defining the assembly (parts + joints). If provided, overrides assembly_name.",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory path (default: ./<assembly_name>_export)",
                    },
                    "actuator_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of actuator IDs (default: TT_MOTOR x4 + MG996R x6)",
                    },
                    "controller": {
                        "type": "string",
                        "description": "Controller type (default: esp32)",
                    },
                    "components": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of components to include (None = all)",
                    },
                },
                "required": [],
            },
        )

    def execute(self, **kwargs: Any) -> str:
        assembly_name = kwargs.get("assembly_name")
        assembly_json = kwargs.get("assembly_json")
        output_dir = kwargs.get("output_dir")
        actuator_ids = kwargs.get("actuator_ids")
        controller = kwargs.get("controller", "esp32")
        components = kwargs.get("components")

        # Resolve assembly
        assembly = _resolve_assembly_input(assembly_name, assembly_json)
        if assembly is None:
            return "Error: Could not resolve assembly. Provide assembly_name or assembly_json."

        # Resolve output directory
        if output_dir:
            out_path = Path(output_dir)
        else:
            name = (assembly_name or assembly.name).lower().replace(" ", "_")
            out_path = Path(".") / f"{name}_export"

        # Parse actuator_ids if passed as JSON string
        if isinstance(actuator_ids, str):
            try:
                actuator_ids = json.loads(actuator_ids)
            except Exception:
                actuator_ids = None

        # Parse components if passed as JSON string
        if isinstance(components, str):
            try:
                components = json.loads(components)
            except Exception:
                components = None

        result = export_engineering_package(
            assembly=assembly,
            output_dir=out_path,
            actuator_ids=actuator_ids,
            controller=controller,
            components=components,
        )

        # Format Markdown summary
        summary = f"""# Engineering Package Export Complete

## Assembly: {assembly.name}
- **Parts**: {result['total_parts']}
- **Joints**: {result['total_joints']}
- **Total mass**: {result['total_mass_kg']:.2f} kg

## Output: `{result['output_dir']}`

### Generated Files ({len(result['generated_files'])} files)

| Category | Contents |
|----------|----------|
| FreeCAD Scripts | {result['total_parts']} part modeling scripts |
| URDF | {result['urdf_links']} links, {result['urdf_joints']} joints |
| ROS2 Package | urdf/ + launch/ + config/ + package.xml |
| Firmware | robot_arm.ino, ik_solver.h, servo_driver.h, dc_motor_driver.h/.cpp, odometry.cpp |
| Reports | BOM, assembly guide, stability, power budget, wiring, cable routing |
| Subsystems | {', '.join(f'{k}({v})' for k, v in result['subsystems'].items())} |

### Key Metrics
- Peak power: {result['peak_power_w']:.1f} W
- Average power: {result['avg_power_w']:.1f} W
- Stability risk: {result['stability_risk']}
- Cable count: {result['cable_count']}
"""

        # Include JSON summary
        summary += f"\n```json\n{json.dumps(result, indent=2, ensure_ascii=False)}\n```"

        return summary


# ============================================================================
# Registration
# ============================================================================


def register_export_package_tools(registry: Any) -> None:
    """Register export package tools."""
    registry.register(ExportPackageTool())
