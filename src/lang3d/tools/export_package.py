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
import os
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
        # Left-side wheels extend outward via motor's front (-Y) face
        # Right-side wheels extend outward via motor's back (+Y) face
        # This produces R(X, ±90°) alignment → wheel axle along Y (upright)
        wheel_anchor = "front" if s.endswith("l") else "back"
        joints.append(Joint("revolute", f"motor_{s}", f"wheel_{s}", axis="y",
                            range_deg=(-360, 360),
                            parent_anchor=wheel_anchor, child_anchor="bottom"))
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
    # Default arm angles: shoulders pitch 90° so arms extend horizontally
    # outward. Left arm needs -90° (extends toward -X), right arm needs +90° (extends toward +X).
    # Elbows: left elbow -30° (cumulative -120° = 30° below horizontal),
    # right elbow +30° (cumulative +120° = 30° below horizontal, symmetric).
    default_angles = {
        "arm_l_shoulder": -90.0,
        "arm_l_elbow": -30.0,
        "arm_r_shoulder": 90.0,
        "arm_r_elbow": 30.0,
    }
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

    # Apply default ConnectionMethod to any joint that lacks one so the
    # connection feature engine (bolt holes, bearing seats) fires on
    # hand-authored assemblies the same way it does on LLM-generated ones.
    # Without this, build_complex_robot()'s 41 parts would ship with STLs
    # that have no mounting holes even though the BOM lists 100+ fasteners.
    from .assembly_generator import apply_default_connection_methods
    apply_default_connection_methods(joints, parts=parts)

    return Assembly(
        name="4-Wheel Mobile Robot with Dual Arms",
        description="4轮差速底盘移动机器人 + 工控机 + 双 3-DOF 机械臂",
        parts=parts,
        joints=joints,
        default_angles=default_angles,
    )


# ============================================================================
# FreeCAD operations for a part
# ============================================================================


def _freecad_ops_for_part(
    part: Part,
    joints: list | None = None,
    all_parts: list | None = None,
) -> list[dict]:
    """Generate FreeCAD operation list for a single part with engineering features.

    If *joints* is provided, connection features (bolt holes, bearing seats,
    snap-fits, etc.) are generated from ``Joint.connection`` metadata and
    merged into the ops list before the final export.

    If *all_parts* is provided, bolted-connection holes are derived from a
    SHARED pattern per joint so mating holes ALIGN ("对接好"); otherwise each
    part lays out holes independently (the "连不上" defect).
    """
    from .part_feature_engine import generate_ops
    return generate_ops(part, joints=joints, all_parts=all_parts)


# ============================================================================
# Real FreeCAD STL generation (reusable)
# ============================================================================


def generate_part_stls(
    assembly: Assembly,
    stl_dir: Path | str,
    fc_dir: Path | str | None = None,
    timeout: int = 60,
) -> tuple[str, dict[str, Any]]:
    """Generate real FreeCAD STLs for each part in the assembly.

    Extracted from ``export_engineering_package`` so the same STL generation
    can run *before* the VLM verification loop — giving the VLM (and the
    user) real C-channel / servo-housing / gripper-finger geometry instead
    of trimesh box approximations.

    Args:
        assembly: The mechanical assembly.
        stl_dir: Directory where per-part ``{name}.stl`` files are written.
        fc_dir: Directory for human-readable FreeCAD scripts.  When
            ``None``, defaults to ``{stl_dir parent}/freecad_scripts``.
        timeout: FreeCAD subprocess timeout per part (seconds).

    Returns:
        ``(stl_dir_path, validation_report_dict)``.  When FreeCAD is not
        available the report dict contains ``{"skipped": True}`` and no
        STLs are generated — the caller should fall back to trimesh
        previews.
    """
    from .freecad import _build_script, _find_freecad_python

    stl_dir_p = Path(stl_dir)
    stl_dir_p.mkdir(parents=True, exist_ok=True)

    if fc_dir is None:
        fc_dir = stl_dir_p.parent / "freecad_scripts"
    fc_dir = Path(fc_dir)
    fc_dir.mkdir(parents=True, exist_ok=True)

    # Build per-part joint lookup so connection features (bolt holes, bearing
    # seats, snap-fits, etc.) are generated for each part.
    joints_by_part: dict[str, list] = {}
    for joint in assembly.joints:
        for pn in (joint.parent, joint.child):
            joints_by_part.setdefault(pn, []).append(joint)

    # Build all part operation lists and write standalone scripts
    all_part_ops: list[list[dict]] = []
    for part in assembly.parts:
        part_joints = joints_by_part.get(part.name, [])
        # Pass the full parts list so bolted connections compute a SHARED
        # hole pattern per joint → mating holes align ("对接好").
        ops = _freecad_ops_for_part(
            part, joints=part_joints, all_parts=assembly.parts,
        )
        for op in ops:
            if op.get("type") == "export_stl" and "{WORKSPACE}" in op.get("path", ""):
                op["path"] = op["path"].replace("{WORKSPACE}", str(fc_dir.parent))
        all_part_ops.append(ops)

    for i, part in enumerate(assembly.parts):
        script = _build_script(all_part_ops[i])
        script_path = fc_dir / f"{part.name}.py"
        script_path.write_text(script, encoding="utf-8")

    # Check FreeCAD availability — bail out early if not installed so the
    # caller can fall back to trimesh preview STLs.
    if not _find_freecad_python():
        logger.info("FreeCAD not available — skipping real STL generation")
        return (str(stl_dir_p), {"skipped": True})

    # Run FreeCAD to generate the actual STLs
    try:
        from .part_validator import validate_all_parts, _simplify_config
        validation_report = validate_all_parts(
            assembly.parts, str(stl_dir_p), timeout=timeout,
            joints_by_part=joints_by_part,
            all_parts=assembly.parts,
        )

        # Rewrite standalone scripts for parts that were simplified so the
        # saved scripts match the geometry that was actually exported.
        from .part_feature_engine import infer_features, generate_ops as _gen
        for r in validation_report.results:
            if r.simplification_level > 0 and r.passed:
                part_obj = next(p for p in assembly.parts if p.name == r.part_name)
                cfg = infer_features(part_obj)
                cfg_simple = _simplify_config(cfg, r.simplification_level)
                simplified_ops = _gen(part_obj, config=cfg_simple, all_parts=assembly.parts)
                for op in simplified_ops:
                    if op.get("type") == "export_stl" and "{WORKSPACE}" in op.get("path", ""):
                        op["path"] = op["path"].replace("{WORKSPACE}", str(fc_dir.parent))
                script = _build_script(simplified_ops)
                script_path = fc_dir / f"{r.part_name}.py"
                script_path.write_text(script, encoding="utf-8")
                logger.info(
                    "Part '%s' simplified to level %d (%s)",
                    r.part_name, r.simplification_level, r.simplification_note,
                )
        logger.info(
            "Part STL generation: %d/%d passed (%.0f%%)",
            validation_report.passed, validation_report.total_parts,
            validation_report.pass_rate * 100,
        )
        return (str(stl_dir_p), validation_report.to_dict())
    except Exception as e:
        logger.warning("FreeCAD STL generation failed: %s", e)
        return (str(stl_dir_p), {"skipped": True, "error": str(e)})


# ============================================================================
# Subsystem decomposition
# ============================================================================


def _build_subsystems(
    assembly: Assembly,
    positions: dict[str, Any],
) -> dict[str, list[str]]:
    """Decompose assembly into subsystems dynamically.

    Uses naming conventions (arm_l_, arm_r_) when available, then falls
    back to part category for the rest.  Works for any assembly type:
    mobile robots, robotic arms, legged robots, etc.
    """
    subsystems: dict[str, list[str]] = {}

    # 1. Detect named arms (arm_l_*, arm_r_* or left_*, right_*)
    for prefix, label in [("arm_l_", "arm_left"), ("arm_r_", "arm_right"),
                           ("left_", "arm_left"), ("right_", "arm_right")]:
        parts = [p.name for p in assembly.parts if p.name.startswith(prefix)]
        if parts:
            subsystems.setdefault(label, []).extend(parts)

    # 2. Detect sensors
    sensor_parts = [
        p.name for p in assembly.parts
        if p.category == "sensor"
        or any(kw in p.name.lower() for kw in ("sensor", "lidar", "camera", "imu"))
    ]
    if sensor_parts:
        subsystems["sensor_tower"] = sensor_parts

    # 3. Detect electronics / controllers
    electronics = [
        p.name for p in assembly.parts
        if p.category in ("electronics", "controller", "battery")
        or any(kw in p.name.lower() for kw in ("ipc", "battery", "driver", "controller", "pcb"))
    ]
    if electronics:
        subsystems["electronics"] = electronics

    # 4. Everything else → chassis / main structure
    assigned = set()
    for parts_list in subsystems.values():
        assigned.update(parts_list)
    remaining = [p.name for p in assembly.parts if p.name not in assigned]
    if remaining:
        subsystems["chassis"] = remaining

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
    verification_status: str = "UNKNOWN",
    verification_warnings: list[str] | None = None,
    existing_stl_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Export a complete engineering package for the assembly.

    Args:
        assembly: The mechanical assembly to export.
        output_dir: Output directory path.
        actuator_ids: Actuator IDs (default: TT_MOTOR x4 + MG996R x6).
        controller: Controller type (default: esp32).
        components: Optional component filter (None = all).
        verification_status: VLM loop status — "PASSED", "FAILED_MAX_ROUNDS",
            or "UNKNOWN" (default). Stamped into design_report.json so
            downstream consumers can detect unverified packages.
        verification_warnings: VLM problems from the last round. Stored
            alongside the status in design_report.json.
        existing_stl_dir: When provided, copy the already-generated STLs
            from this directory into ``{output_dir}/stl_parts`` and skip
            the FreeCAD validation subprocess (the STLs were produced
            during the VLM loop).  Avoids a redundant ~2 min FreeCAD run.

    Returns:
        Dict with generated file list and key metrics.
    """
    if actuator_ids is None:
        # Derive from assembly: count revolute joints as servos, motors by naming
        n_motors = sum(1 for p in assembly.parts
                       if "motor" in p.name.lower() or "wheel" in p.name.lower())
        n_servos = sum(1 for j in assembly.joints if j.type == "revolute")
        actuator_ids = ["TT_MOTOR"] * n_motors + ["MG996R"] * n_servos
        if not actuator_ids:
            actuator_ids = ["MG996R"] * len(assembly.joints)

    output_dir = Path(output_dir)
    generated_files: list[str] = []

    # ---- Create shared context ----
    ctx = AssemblyContext(assembly=assembly)

    # ---- Step 1: Solve assembly positions ----
    positions = ctx.ensure_positions()

    # ---- Step 1b: VLM visual verification ----
    # Try to run closed-loop VLM verification: render → VLM check → fix → re-solve.
    # Skip when the generate phase already PASSED — re-verifying risks the
    # independent VLM call disagreeing and apply_corrections perturbing a
    # known-good assembly (root cause of the earlier 330mm finger-origin bug).
    vlm_result = None
    vlm_backend = None
    if verification_status == "PASSED":
        logger.info("VLM visual verification skipped (already PASSED in generate phase)")
    try:
        from ..models.glm import GLMBackend
        import os as _os

        api_key = _os.environ.get("GLM_API_KEY", "")
        base_url = _os.environ.get(
            "GLM_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4"
        )
        vision_model = _os.environ.get("VISION_MODEL", "GLM-4V-Plus")

        if api_key and verification_status != "PASSED":
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
                    assembly.description
                    or f"{assembly.name}: {len(assembly.parts)}-part assembly "
                    f"with {sum(1 for j in assembly.joints if j.type == 'revolute')} "
                    f"revolute joints"
                ),
                max_iterations=3,
                detail_level="detailed",
            )
            # If corrections were applied, re-solve positions
            if vlm_result.corrections_applied and not vlm_result.passed:
                from ..agent.assembly_visual_verifier import apply_corrections
                corrected_assembly = apply_corrections(assembly, vlm_result.corrections_applied)
                # Re-normalise gripper fingers: apply_corrections can perturb
                # finger offsets; restore the symmetric centre-anchored geometry
                # so URDF origins stay sane.
                from .assembly_generator import _normalize_gripper_fingers
                corrected_assembly = _normalize_gripper_fingers(corrected_assembly)
                assembly = corrected_assembly
                ctx = AssemblyContext(assembly=assembly)
                positions = ctx.ensure_positions()
    except Exception as e:
        logger.warning("VLM verification skipped: %s", e)

    # ---- Step 2: Compute mass properties ----
    mass_result = ctx.ensure_mass()

    # ---- Step 3: Generate FreeCAD scripts ----
    from .freecad import _build_script, _build_batch_script, _run_freecad_script
    fc_dir = output_dir / "freecad_scripts"
    fc_dir.mkdir(parents=True, exist_ok=True)

    # Build per-part joint lookup so connection features (bolt holes, bearing
    # seats, snap-fits, adhesive grooves, etc.) are generated for each part
    # that participates in a joint.  This activates the previously-dead
    # connection_features module in the main export pipeline.
    joints_by_part: dict[str, list] = {}
    for joint in assembly.joints:
        for pn in (joint.parent, joint.child):
            joints_by_part.setdefault(pn, []).append(joint)

    # Build all part operation lists
    all_part_ops = []
    for part in assembly.parts:
        part_joints = joints_by_part.get(part.name, [])
        ops = _freecad_ops_for_part(part, joints=part_joints)
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

    # ---- Step 3a: Validate parts via FreeCAD execution ----
    stl_dir = output_dir / "stl_parts"
    stl_dir.mkdir(parents=True, exist_ok=True)
    validation_report_path = output_dir / "part_validation_report.json"

    # When the VLM loop already generated real STLs, copy them across and
    # skip the redundant FreeCAD subprocess run.
    if existing_stl_dir is not None:
        import shutil as _shutil
        _src = Path(existing_stl_dir)
        _copied = 0
        if _src.is_dir():
            for _stl in _src.glob("*.stl"):
                _shutil.copy2(_stl, stl_dir / _stl.name)
                _copied += 1
        logger.info(
            "Reused %d existing STLs from %s (skipping FreeCAD validation)",
            _copied, _src,
        )
    else:
        try:
            from .freecad import _find_freecad_python
            if _find_freecad_python():
                from .part_validator import validate_all_parts
                logger.info("Running FreeCAD validation for %d parts (vlm_backend=%s)",
                            len(assembly.parts), "available" if vlm_backend else "None")
                validation_report = validate_all_parts(
                    assembly.parts, str(stl_dir), timeout=60,
                    vlm_router=vlm_backend,
                    vlm_sample=len(assembly.parts),
                    joints_by_part=joints_by_part,
                )
                # Save validation report
                validation_report_path.write_text(
                    json.dumps(validation_report.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                generated_files.append(str(validation_report_path))

                # Replace full-feature ops with simplified ops for parts that needed it
                simplified_count = 0
                for r in validation_report.results:
                    if r.simplification_level > 0 and r.passed:
                        part_obj = next(p for p in assembly.parts if p.name == r.part_name)
                        from .part_validator import _simplify_config
                        from .part_feature_engine import infer_features, generate_ops as _gen
                        cfg = infer_features(part_obj)
                        cfg_simple = _simplify_config(cfg, r.simplification_level)
                        simplified_ops = _gen(part_obj, config=cfg_simple, all_parts=assembly.parts)
                        # Fix workspace placeholder
                        for op in simplified_ops:
                            if op.get("type") == "export_stl" and "{WORKSPACE}" in op.get("path", ""):
                                op["path"] = op["path"].replace("{WORKSPACE}", str(fc_dir.parent))
                        # Replace in all_part_ops
                        idx = next(i for i, p in enumerate(assembly.parts) if p.name == r.part_name)
                        all_part_ops[idx] = simplified_ops
                        # Rewrite the script with simplified ops
                        script = _build_script(simplified_ops)
                        script_path = fc_dir / f"{r.part_name}.py"
                        script_path.write_text(script, encoding="utf-8")
                        simplified_count += 1
                        logger.info(
                            "Part '%s' simplified to level %d (%s)",
                            r.part_name, r.simplification_level, r.simplification_note,
                        )
                if simplified_count > 0:
                    logger.info("%d parts were simplified for FreeCAD compatibility", simplified_count)
                logger.info(
                    "Part validation: %d/%d passed (%.0f%%)",
                    validation_report.passed, validation_report.total_parts,
                    validation_report.pass_rate * 100,
                )
        except Exception as e:
            logger.warning("FreeCAD part validation skipped: %s", e)

    # ---- Step 4b: Export per-part STEP files ----
    # STEP (B-rep) exports give downstream CAD tools (SolidWorks, Fusion 360,
    # KiCad MCAD, FreeCAD itself) precise geometry instead of tessellated
    # STL meshes.  Reuses the existing ``export_step`` op handler in
    # ``_build_script`` — appends one op per part to the same geometry ops
    # already produced above (with simplifications applied during Step 3a).
    step_dir = output_dir / "step_parts"
    step_dir.mkdir(parents=True, exist_ok=True)
    try:
        from .freecad import _find_freecad_python
        if _find_freecad_python():
            step_ok, step_total = 0, len(assembly.parts)
            for i, part in enumerate(assembly.parts):
                # Strip the STL export op so the script only writes STEP —
                # avoids regenerating STLs we already have.
                step_ops = [op for op in all_part_ops[i]
                            if op.get("type") != "export_stl"]
                step_path = step_dir / f"{part.name}.step"
                step_ops.append({"type": "export_step", "path": str(step_path)})
                step_script = _build_script(step_ops).replace(
                    "{WORKSPACE}", str(fc_dir.parent).replace("\\", "/"))
                try:
                    _run_freecad_script(step_script, timeout=120)
                    if step_path.exists():
                        step_ok += 1
                        generated_files.append(str(step_path))
                except Exception as e:
                    logger.warning(
                        "STEP export failed for '%s': %s", part.name, e,
                    )
            logger.info(
                "STEP export: %d/%d parts written to %s",
                step_ok, step_total, step_dir,
            )
    except Exception as e:
        logger.warning("STEP export skipped: %s", e)

    # ---- Step 3b: Render assembly ----
    from .freecad import (
        build_assembly_script,
        _shape_type_for_part,
        _subsystem_for_part,
        _run_freecad_script,
    )
    assembly_parts_info = []
    for p in assembly.parts:
        _stl = stl_dir / f"{p.name}.stl"
        assembly_parts_info.append({
            "name": p.name,
            "shape_type": _shape_type_for_part(p),
            "dimensions": p.dimensions,
            "subsystem": _subsystem_for_part(p.name),
            "stl_path": str(_stl) if _stl.exists() else None,
        })
    render_path = output_dir / "assembly"
    assembly_script = build_assembly_script(
        assembly_parts=assembly_parts_info,
        positions=positions,
        output_path=str(render_path),
        joints=assembly.joints,
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
            else:
                logger.warning(
                    "assembly render script ran but produced no assembly.stl"
                )
    except Exception as e:
        logger.warning("Assembly render failed (FreeCAD path): %s", e)
    # Fallback: build the assembly STL via trimesh when FreeCAD failed (or the
    # STL is missing).  FreeCAD overflows its stack on large assemblies (the
    # 438-object dual-arm crashes); trimesh has no such limit and produces the
    # same deliverable — positioned parts + fasteners at bolted joints.
    if not render_path.with_suffix(".stl").exists():
        try:
            from .freecad import build_assembly_stl_trimesh
            stl_out = build_assembly_stl_trimesh(
                assembly_parts=assembly_parts_info,
                positions=positions,
                output_path=str(render_path),
                joints=assembly.joints,
            )
            if render_path.with_suffix(".stl").exists():
                generated_files.append(str(render_path.with_suffix(".stl")))
                logger.info("assembly.stl built via trimesh fallback (%d bytes)",
                            render_path.with_suffix(".stl").stat().st_size)
        except Exception as e:
            logger.warning("trimesh assembly fallback failed: %s", e)

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
    # URDF joint origins must describe the home pose (all joint angles = 0),
    # NOT the default pose.  Using default_angles-baked positions here would
    # put non-zero rpy into joint origins, making MuJoCo's qpos=0 correspond
    # to the bent pose instead of the straight/home pose.  This breaks
    # simulation (joint axes get rotated to non-physical orientations) and
    # breaks control (PID setpoints become inconsistent with kinematics).
    home_positions = ctx.ensure_home_positions()
    converter = AssemblyToURDF(assembly, positions=home_positions)
    urdf_xml = converter.convert()
    (output_dir / "urdf.xml").write_text(urdf_xml, encoding="utf-8")
    generated_files.append(str(output_dir / "urdf.xml"))

    builder = ROS2PackageBuilder(assembly.name, urdf_xml)
    ros2_dir = output_dir / "ros2_package"
    builder.write(str(ros2_dir))
    generated_files.append(str(ros2_dir))

    # Copy STL meshes into ROS2 package meshes/ directory
    import shutil
    from .urdf_export import _sanitize_name
    ros2_meshes = ros2_dir / _sanitize_name(assembly.name) / "meshes"
    ros2_meshes.mkdir(parents=True, exist_ok=True)
    for part in assembly.parts:
        stl_src = stl_dir / f"{part.name}.stl"
        if stl_src.exists():
            shutil.copy2(str(stl_src), str(ros2_meshes / f"{_sanitize_name(part.name)}.stl"))
        else:
            logger.warning("STL not found for ROS2 mesh copy: %s", stl_src)

    # ---- Step 4c: Optional physics + grasp validation (MuJoCo) ----
    # Triggered by LANG3D_RUN_SIM_VALIDATE=1 so normal exports stay fast.
    # Runs sim_mujoco (structure + physics) and, if a gripper is detected,
    # sim_grasp (cube pickup test).  Results are stamped into
    # design_report.json under "simulation_validation".  Failures are
    # informational only — they don't abort the export.
    sim_validation = _run_post_export_sim_validation(
        ros2_dir / _sanitize_name(assembly.name),
        assembly_name=_sanitize_name(assembly.name),
    )

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

    # Collect all actuator joints (revolute) for firmware
    revolute_joints = [j for j in assembly.joints if j.type == "revolute"]
    actuator_names = [j.child for j in revolute_joints]
    # Also include parts explicitly categorized as actuator
    actuator_parts = [p for p in assembly.parts if p.category == "actuator"]
    actuator_names_set = set(actuator_names) | {p.name for p in actuator_parts}
    servo_ids = ["MG996R"] * len(actuator_names_set) if actuator_names_set else ["MG996R"]

    # Build the firmware sub-assembly. Previously this kept only the revolute
    # joints, which dropped every fixed connector between a servo and its link.
    # The kinematic chain then broke and `_extract_chain` collapsed every
    # segment length to 0. We now take the closure of the arm chain — seed
    # with the actuator parts and keep absorbing any joint that touches the
    # growing set — so all fixed connectors survive and the chain stays
    # connected. `generate_firmware` still selects servos via `j.type ==
    # "revolute"`, so the firmware output is unchanged.
    chain_names = set(actuator_names_set)
    # Seed the closure with the actuator parents too, so a servo that sits
    # between two links pulls in both neighbouring structural parts.
    for j in revolute_joints:
        chain_names.add(j.parent)

    changed = True
    while changed:
        changed = False
        for j in assembly.joints:
            if (j.parent in chain_names) != (j.child in chain_names):
                chain_names.add(j.parent)
                chain_names.add(j.child)
                changed = True

    arm_sub_assembly = Assembly(
        name=assembly.name + "_actuators",
        parts=[p for p in assembly.parts if p.name in chain_names],
        joints=[
            j for j in assembly.joints
            if j.parent in chain_names and j.child in chain_names
        ],
    )

    firmware = generate_firmware(
        arm_sub_assembly,
        actuator_ids=servo_ids,
        controller=controller,
    )
    for fname, content in firmware.items():
        (fw_dir / fname).write_text(content, encoding="utf-8")
        generated_files.append(str(fw_dir / fname))

    # Motor driver code: only generate if there are wheel/drive motors
    drive_motor_parts = [
        p for p in assembly.parts
        if "motor" in p.name.lower() and "servo" not in p.name.lower()
    ]
    n_drive_motors = len(drive_motor_parts)
    if n_drive_motors > 0:
        motors = [
            dict(motor_id="TT_MOTOR", encoder_id="HALL_TT_7PPR",
                 pwm_pin=5 + i * 2, dir_pin1=6 + i * 2, dir_pin2=7 + i * 2,
                 enc_a_pin=18 + i * 2, enc_b_pin=19 + i * 2)
            for i in range(n_drive_motors)
        ]
        motor_code = gen_motor_driver_code(motors)
        for fname, content in motor_code.items():
            (fw_dir / fname).write_text(content, encoding="utf-8")
            generated_files.append(str(fw_dir / fname))

        # Odometry only makes sense for wheeled robots
        if any("wheel" in p.name.lower() for p in assembly.parts):
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
    calc = PowerBudgetCalculator(assembly.name)
    if n_drive_motors > 0:
        calc.add_motor("Drive Motor", "TT_MOTOR", duty_cycle=0.5, quantity=n_drive_motors)
    n_servos = len(revolute_joints)
    if n_servos > 0:
        calc.add_servo("Servos", "MG996R", duty_cycle=0.6, quantity=n_servos)
    # Add controller if electronics present
    if any(p.category in ("electronics", "controller") for p in assembly.parts):
        calc.add_controller("Main Controller", tdp_w=15.0)
    # Add sensor load if sensors present
    n_sensors = sum(1 for p in assembly.parts if p.category == "sensor")
    if n_sensors > 0:
        calc.add_sensor_load("Sensors", power_w=2.0, quantity=n_sensors)
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
    # Support any assembly type: prefer wheel contacts for mobile robots,
    # fall back to lowest-Z parts (e.g., base_plate for arms)
    wheel_suffixes = ["fl", "fr", "rl", "rr"]
    has_wheels = all(f"wheel_{s}" in positions for s in wheel_suffixes)
    if has_wheels:
        for s in wheel_suffixes:
            pos = positions[f"wheel_{s}"]["position"]
            contacts.append([pos[0], pos[1], pos[2]])
    else:
        # Expand ground-contact parts into footprint corners using their
        # dimensions.  Solver convention: length→Y (forward), width→X
        # (lateral) — same as assembly_verifier.py's COM check.  Using
        # only the centre point (the old behaviour) produces a degenerate
        # polygon for single-base robots, always reporting "unstable".
        parts_by_name = {p.name: p for p in assembly.parts}
        # Ground-contact detection: a part is "on the ground" when its BOTTOM
        # face (centre Z minus half-height) is within 5mm of the BASE PLATE's
        # bottom. The ground reference MUST be the base, not the global Z
        # minimum — on a long 7-DOF arm the gripper fingers droop below the
        # base (Z<0), and using them as z_min excludes the actual base from
        # the "ground" set, leaving only the fingers' far-away footprint to
        # define the support polygon (margin -577mm on a 558mm base that
        # should pass). The earlier centre-Z + 10%-range test had the same
        # flaw in the other direction (sweeping fingers IN). Both are wrong;
        # the base plate is the ground truth for "what touches the floor".
        base_part = next(
            (p for p in assembly.parts
             if "base" in p.name.lower() and "plate" in p.name.lower()),
            None,
        )
        if base_part and base_part.name in positions:
            base_h = base_part.dimensions.get(
                "height", base_part.dimensions.get("thickness", 8))
            z_min = positions[base_part.name]["position"][2] - base_h / 2.0
        else:
            z_min = min(p["position"][2] for p in positions.values())
        for pname, pdata in positions.items():
            z = pdata["position"][2]
            part = parts_by_name.get(pname)
            dims = part.dimensions if part and part.dimensions else {}
            h = dims.get("height", dims.get("thickness",
                    dims.get("length", dims.get("diameter", 10))))
            bottom_z = z - h / 2.0
            if bottom_z > z_min + 5.0:
                continue  # not actually touching the ground
            pos = pdata["position"]
            part = parts_by_name.get(pname)
            dims = part.dimensions if part and part.dimensions else {}
            cx, cy, cz = pos[0], pos[1], pos[2]
            if "length" in dims and "width" in dims:
                # Box footprint: 4 corners (width→X, length→Y).
                hx = dims["width"] / 2.0
                hy = dims["length"] / 2.0
                for dx, dy in [(-hx, -hy), (hx, -hy), (-hx, hy), (hx, hy)]:
                    contacts.append([cx + dx, cy + dy, cz])
            elif "diameter" in dims:
                # Circular footprint: 8-point polygon.
                r = dims["diameter"] / 2.0
                import math as _m
                for i in range(8):
                    a = i * _m.pi / 4
                    contacts.append([cx + r * _m.cos(a), cy + r * _m.sin(a), cz])
            else:
                contacts.append([cx, cy, cz])
        # If still empty, use the first part (root)
        if not contacts and positions:
            first_pos = next(iter(positions.values()))["position"]
            contacts.append([first_pos[0], first_pos[1], first_pos[2]])

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
        f"- Margin: {static_stab.get('margin_mm', 'N/A') if static_stab.get('margin_mm') is not None else 'N/A'} mm\n",
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
    kinematic = ctx.ensure_kinematic_analysis()

    design_warnings: list[str] = []
    if tip_risk.get("risk_level") == "CRITICAL":
        design_warnings.append(
            "Stability risk is CRITICAL — robot may tip over. "
            "Consider widening the base or lowering the center of mass."
        )
    if all(abs(v) < 0.1 for v in com):
        design_warnings.append(
            "Center of mass is at origin — positions may not have been "
            "applied to mass calculation."
        )
    if peak_w > 0 and avg_w > 0 and peak_w / avg_w > 20:
        design_warnings.append(
            f"Peak/avg power ratio is {peak_w/avg_w:.0f}x — "
            "average power estimate may be unrealistic."
        )
    if verification_status != "PASSED":
        design_warnings.append(
            f"VLM verification status is {verification_status} — "
            "package may not match design intent. Review renders before "
            "manufacturing."
        )
    if kinematic.get("loop_count", 0) > 0 and not kinematic.get("converged", True):
        design_warnings.append(
            f"Closed-chain solver did not converge: "
            f"{kinematic.get('loop_count')} loops, "
            f"residual error {kinematic.get('error_mm', 0):.2f}mm."
        )

    report = {
        "requirement": getattr(assembly, "description", assembly.name),
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
        "verification_status": verification_status,
        "verification_warnings": list(verification_warnings or []),
        "kinematic_analysis": kinematic,
        "simulation_validation": sim_validation,
        "warnings": design_warnings,
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
{chr(10).join(f"| {name} | {len(parts)} | {', '.join(parts[:5])}{'...' if len(parts) > 5 else ''} |" for name, parts in subsystems.items())}

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
├── stl_parts/                    # STL 网格 (3D打印用)
├── step_parts/                   # STEP B-rep 模型 (CAD 互操作)
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


def _run_post_export_sim_validation(
    pkg_dir: Path,
    assembly_name: str,
) -> dict[str, Any]:
    """Run optional MuJoCo physics + grasp validation on the exported URDF.

    Triggered by environment variable ``LANG3D_RUN_SIM_VALIDATE=1``.
    When disabled (default), returns ``{"enabled": False}`` immediately
    so normal exports stay fast.

    When enabled:
      1. Runs ``sim_mujoco`` on the flat URDF — validates structure
         (mesh paths resolve, masses non-zero, joints movable) and
         physics stability (PD-hold under gravity).
      2. If structure passes and a gripper is detected (≥2 SLIDE joints),
         runs ``sim_grasp`` — spawns a cube between the fingers and
         tests whether the gripper can statically hold it.

    Failures are informational only — they're recorded in the returned
    dict (and stamped into design_report.json) but never abort the export.

    Args:
        pkg_dir: Path to the ROS2 package directory (containing urdf/
                 and meshes/ subdirs).
        assembly_name: Sanitized package name (used to locate URDF files).

    Returns:
        Dict with ``enabled``, ``mujoco``, and ``grasp`` sub-dicts.
    """
    if os.environ.get("LANG3D_RUN_SIM_VALIDATE", "").lower() not in (
        "1", "true", "yes", "on",
    ):
        return {"enabled": False, "reason": "LANG3D_RUN_SIM_VALIDATE not set"}

    flat_urdf = pkg_dir / "urdf" / f"{assembly_name}_flat.urdf"
    if not flat_urdf.exists():
        return {
            "enabled": True,
            "ran": False,
            "reason": f"flat URDF not found at {flat_urdf}",
        }

    result: dict[str, Any] = {
        "enabled": True,
        "ran": True,
        "flat_urdf": str(flat_urdf),
        "mujoco": None,
        "grasp": None,
    }

    # ---- sim_mujoco: structure + physics validation ----
    try:
        from .sim_mujoco import SimMujocoTool
        tool = SimMujocoTool()
        report = tool.execute(
            urdf_path=str(flat_urdf),
            mode="validate",
            duration_sec=1.0,
        )
        # Parse the JSON tail of the report
        json_start = report.find("--- JSON ---")
        mujoco_summary: dict[str, Any] = {"raw_report": report[:500]}
        if json_start >= 0:
            import json as _json
            try:
                mujoco_summary = _json.loads(
                    report[json_start + len("--- JSON ---"):].strip()
                )
            except Exception as _e:
                pass  # TODO: mujoco summary JSON parse failed (no logger available)
        # Extract the key signals
        result["mujoco"] = {
            "verdict_ok": mujoco_summary.get("verdict_ok"),
            "load_ok": mujoco_summary.get("load_ok"),
            "n_bodies": mujoco_summary.get("n_bodies"),
            "n_joints": mujoco_summary.get("n_joints"),
            "physics_stable": (
                mujoco_summary.get("physics", {}).get("stabilized")
                if isinstance(mujoco_summary.get("physics"), dict)
                else None
            ),
            "warnings": mujoco_summary.get("warnings", []),
        }
        logger.info(
            "sim_mujoco validation: verdict=%s, physics_stable=%s",
            result["mujoco"]["verdict_ok"],
            result["mujoco"]["physics_stable"],
        )
    except ImportError:
        result["mujoco"] = {
            "ran": False,
            "reason": "mujoco package not installed (pip install mujoco)",
        }
    except Exception as exc:
        result["mujoco"] = {"ran": False, "reason": f"error: {exc}"}
        logger.warning("sim_mujoco validation failed: %s", exc)

    # ---- sim_grasp: gripper pickup test ----
    # Only run if structure validation passed (no point testing grasp on
    # a URDF that can't even load).
    mujoco_ok = (
        result.get("mujoco", {}).get("verdict_ok") is True
        and result.get("mujoco", {}).get("load_ok") is True
    )
    if not mujoco_ok:
        result["grasp"] = {
            "ran": False,
            "reason": "skipped (sim_mujoco structure validation failed)",
        }
        return result

    try:
        from .sim_mujoco import SimGraspTool
        tool = SimGraspTool()
        report = tool.execute(
            urdf_path=str(flat_urdf),
            cube_size_mm=20.0,
            cube_mass_g=20.0,
            grasp_force_n=20.0,
            duration_sec=2.0,
        )
        json_start = report.find("--- JSON ---")
        grasp_summary: dict[str, Any] = {}
        if json_start >= 0:
            import json as _json
            try:
                grasp_summary = _json.loads(
                    report[json_start + len("--- JSON ---"):].strip()
                )
            except Exception as _e:
                pass  # TODO: grasp summary JSON parse failed (no logger available)
        result["grasp"] = {
            "ran": True,
            "grasp_ok": grasp_summary.get("grasp_ok"),
            "lifted": grasp_summary.get("lifted"),
            "geometry_ok": grasp_summary.get("geometry_ok"),
            "held_against_gravity": grasp_summary.get("held_against_gravity"),
            "phase_a_contacts_max": grasp_summary.get("phase_a_contacts_max"),
            "slip_b_mm": (
                grasp_summary.get("slip_b_m", 0) * 1000
                if grasp_summary.get("slip_b_m") is not None
                else None
            ),
            "lift_c_mm": (
                grasp_summary.get("lift_c_m", 0) * 1000
                if grasp_summary.get("lift_c_m") is not None
                else None
            ),
            "note": grasp_summary.get("note"),
        }
        if "NO GRIPPER" in report:
            result["grasp"] = {
                "ran": False,
                "reason": "no gripper detected (need 2 SLIDE joints)",
            }
        logger.info(
            "sim_grasp validation: grasp_ok=%s, lifted=%s",
            result["grasp"].get("grasp_ok"),
            result["grasp"].get("lifted"),
        )
    except ImportError:
        result["grasp"] = {
            "ran": False,
            "reason": "mujoco package not installed",
        }
    except Exception as exc:
        result["grasp"] = {"ran": False, "reason": f"error: {exc}"}
        logger.warning("sim_grasp validation failed: %s", exc)

    return result


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
                    type=jd.get("type", jd.get("joint_type", "fixed")),
                    parent=jd["parent"],
                    child=jd["child"],
                    range_deg=tuple(jd["range_deg"]) if "range_deg" in jd else (-180, 180),
                    description=jd.get("description", ""),
                    axis=jd.get("axis", "auto"),
                    parent_anchor=jd.get("parent_anchor", "top"),
                    child_anchor=jd.get("child_anchor", "bottom"),
                    offset=tuple(jd["offset"]) if "offset" in jd else None,
                    no_distribute=jd.get("no_distribute", False),
                    distribution_group=jd.get("distribution_group", ""),
                ))
            assembly = Assembly(
                name=data.get("name", "Custom Assembly"),
                parts=parts,
                joints=joints,
                description=data.get("description", ""),
            )
            if "default_angles" in data:
                assembly.default_angles = data["default_angles"]
            return assembly
        except Exception as _e:
            pass  # TODO: assembly.json re-parse for web failed (no logger available)

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
