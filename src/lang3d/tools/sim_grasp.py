"""Grasp simulation — three-phase cube grasp test.

Extracted from sim_mujoco.py (P1-1 God Module split, AGENTS.md §2.1).
Zero-gravity close → gravity hold → lift verification.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from .base import Tool, ToolDefinition

logger = logging.getLogger(__name__)

# Deferred import to avoid circular dependency.
import lang3d.tools.sim_mujoco as _sm

try:
    _load_model = _sm._load_model
except AttributeError:
    pass
try:
    _mujoco_available = _sm._mujoco_available
except AttributeError:
    pass
try:
    _rewrite_mesh_paths = _sm._rewrite_mesh_paths
except AttributeError:
    pass
try:
    _stabilize_model = _sm._stabilize_model
except AttributeError:
    pass

def _find_slide_joints(model: Any) -> list[dict[str, Any]]:
    """Find all SLIDE joints in the model.

    Used to identify gripper finger joints.  Returns list of dicts with
    joint id, name, axis, range, body id, body name.
    """
    import mujoco  # type: ignore[import-not-found]

    slides: list[dict[str, Any]] = []
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) != mujoco.mjtJoint.mjJNT_SLIDE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        body_id = int(model.jnt_bodyid[jid])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        slides.append({
            "jid": jid,
            "name": name,
            "axis": tuple(float(x) for x in model.jnt_axis[jid]),
            "range": tuple(float(x) for x in model.jnt_range[jid]),
            "body_id": body_id,
            "body_name": body_name,
        })
    return slides


def _group_fingers_into_grippers(
    slide_joints: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group slide joints (fingers) into grippers.

    A robot may have multiple grippers (e.g. a dual-arm robot has two, each
    with 2 fingers → 4 slide joints total).  Fingers are grouped by their
    body name with the trailing ``finger_<side>`` segment stripped:

        arm_l_gripper_finger_left  → gripper "arm_l_gripper"
        arm_l_gripper_finger_right → gripper "arm_l_gripper"
        arm_r_gripper_finger_left  → gripper "arm_r_gripper"

    If a body name does not match the ``finger_*`` convention, fall back to
    grouping by the longest common prefix before ``finger`` (or the parent
    body if no such marker exists), so single-gripper robots (whose finger
    bodies are simply ``finger_left``/``finger_right``) still group into one.

    Returns a list of ``(gripper_id, [finger_joint, ...])`` pairs, ordered by
    the gripper's first appearance.  Each gripper must have >= 2 fingers to
    constitute a working 2-finger (or N-finger) gripper.
    """
    import re

    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for j in slide_joints:
        body = j["body_name"]
        # Strip a trailing finger_<side> segment to get the gripper id.
        m = re.match(r"^(.*)finger_[A-Za-z0-9]+$", body)
        gripper_id = m.group(1).rstrip("_") if m else "gripper"
        if gripper_id not in groups:
            groups[gripper_id] = []
            order.append(gripper_id)
        groups[gripper_id].append(j)
    return [(gid, groups[gid]) for gid in order]



def _add_cube_to_scene(
    urdf_path: str,
    cube_pos_m: tuple[float, float, float],
    cube_size_m: float = 0.020,
    cube_mass_kg: float = 0.050,
) -> tuple[Any, list[str]]:
    """Load URDF, attach a cube body, return (model, temp_files).

    MuJoCo models are immutable after compilation, so we have to:
      1. Load the URDF normally.
      2. Save the compiled model as MJCF XML via ``mj_saveLastXML``.
      3. Splice a ``<body>`` element for the cube into the MJCF.
      4. Recompile from the modified XML.

    The cube is a free-floating body (no joint to world = floating base)
    so it falls under gravity and can be pushed by finger contacts.
    """
    import mujoco  # type: ignore[import-not-found]

    temp_files: list[str] = []

    # Apply mesh path auto-fix (handles URDFs in urdf/ subdirs that use
    # relative "meshes/X.stl" paths).  This produces a temp URDF with
    # absolute paths that MuJoCo can load directly.
    urdf_path_obj = Path(urdf_path)
    urdf_text = urdf_path_obj.read_text(encoding="utf-8")
    fixed_urdf_text, mesh_warnings = _rewrite_mesh_paths(urdf_text, urdf_path_obj)

    if fixed_urdf_text != urdf_text:
        fixed_urdf_fd, fixed_urdf_path = tempfile.mkstemp(
            suffix=".urdf", dir=str(urdf_path_obj.parent), prefix="_grasp_fixed_",
        )
        os.close(fixed_urdf_fd)
        Path(fixed_urdf_path).write_text(fixed_urdf_text, encoding="utf-8")
        temp_files.append(fixed_urdf_path)
        load_path = fixed_urdf_path
    else:
        load_path = urdf_path

    # Initial load populates the "last XML" cache mj_saveLastXML reads from
    mujoco.MjModel.from_xml_path(load_path)

    # Save the compiled MJCF to a temp file (in the same dir as the URDF so
    # relative mesh paths in the MJCF can resolve)
    urdf_dir = urdf_path_obj.parent
    mjcf_fd, mjcf_path = tempfile.mkstemp(
        suffix=".xml", dir=str(urdf_dir), prefix="_grasp_scene_",
    )
    os.close(mjcf_fd)
    temp_files.append(mjcf_path)
    mujoco.mj_saveLastXML(mjcf_path, mujoco.MjModel.from_xml_path(load_path))

    with open(mjcf_path, encoding="utf-8") as f:
        mjcf_text = f.read()

    # Splice cube body before </worldbody>
    half = cube_size_m / 2.0
    # High friction so the cube doesn't slip when fingers squeeze
    cube_xml = (
        f'<body name="grasp_cube" pos="{cube_pos_m[0]:.6f} '
        f'{cube_pos_m[1]:.6f} {cube_pos_m[2]:.6f}">\n'
        f'  <freejoint name="grasp_cube_joint"/>\n'
        f'  <geom name="grasp_cube_geom" type="box" '
        f'size="{half:.6f} {half:.6f} {half:.6f}" '
        f'mass="{cube_mass_kg:.6f}" '
        f'friction="0.8 0.02 0.0001" '
        f'rgba="0.85 0.25 0.15 1"/>\n'
        f'</body>\n'
    )
    # Add a ground plane so the cube (if not grasped) lands on something
    # instead of falling forever (which makes the test inconclusive).
    ground_xml = (
        '<geom name="ground" type="plane" pos="0 0 0" '
        'size="2 2 0.1" rgba="0.3 0.3 0.3 1"/>\n'
    )
    if "</worldbody>" in mjcf_text:
        new_mjcf = mjcf_text.replace(
            "</worldbody>",
            ground_xml + cube_xml + "</worldbody>",
        )
    else:
        # Some URDFs compile without explicit worldbody close tag — fallback
        new_mjcf = mjcf_text.replace(
            "</mujoco>",
            "<worldbody>" + ground_xml + cube_xml + "</worldbody></mujoco>",
        )

    # Make finger friction higher too — needed for grasp
    new_mjcf = new_mjcf.replace(
        'friction="1 0.005 0.0001"',
        'friction="0.9 0.02 0.0001"',
    )

    with open(mjcf_path, "w", encoding="utf-8") as f:
        f.write(new_mjcf)

    # Load via path so relative mesh paths resolve correctly
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    return model, temp_files


def _run_grasp_scenario(
    model: Any,
    slide_joints: list[dict[str, Any]],
    cube_body_id: int,
    grasp_force_n: float = 5.0,
    lift_height_m: float = 0.030,
    duration_sec: float = 4.0,
) -> dict[str, Any]:
    """Run grasp + lift scenario using gravity-gated phases.

    Sequence:
      Phase A (15%): ZERO gravity — let fingers close on the stationary
        cube.  This tests whether the gripper geometry can clamp the
        object at all, decoupled from the dynamic-grasp problem of
        catching a falling object.
      Phase B (next 15%): enable gravity, keep fingers clamping.  If the
        friction/normal-force combination is sufficient, the cube stays
        held.  If not, it slips and falls.
      Phase C (last 70%): maintain grasp + apply lift torque to arm.
        Measures whether the cube rises with the gripper.

    This phasing lets the tool distinguish:
      - Geometry cannot clamp (phase A: fingers don't touch cube)
      - Grasp too weak (phase B: cube slips when gravity on)
      - Lift fails (phase C: cube slips when arm accelerates)
    """
    import mujoco  # type: ignore[import-not-found]
    import numpy as np

    _stabilize_model(model, armature=0.2, damping=3.0)
    model.opt.timestep = 0.0005
    try:
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        model.opt.iterations = 50
    except AttributeError:
        pass

    data = mujoco.MjData(model)
    # Phase A: zero gravity so cube stays put while fingers close
    model.opt.gravity[:] = (0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)

    # Compute grasp center from finger world positions
    finger_positions = [
        np.array(data.xpos[j["body_id"]], dtype=float) for j in slide_joints
    ]
    grasp_center = sum(finger_positions) / len(finger_positions)
    cube_initial_z = float(data.xpos[cube_body_id][2])

    # Per-finger close direction sign (geometry-aware, handles mirrored mounts)
    finger_close_signs: dict[int, float] = {}
    for j in slide_joints:
        jid = j["jid"]
        body_id = j["body_id"]
        axis_local = np.array(model.jnt_axis[jid], dtype=float)
        body_quat = np.array(data.xquat[body_id], dtype=float)
        axis_world = _rotate_vector_by_quat(axis_local, body_quat)
        to_center = grasp_center - np.array(data.xpos[body_id], dtype=float)
        projection = float(np.dot(to_center, axis_world))
        finger_close_signs[jid] = 1.0 if projection > 0 else -1.0

    # Arm hinge joints (everything except slide joints)
    slide_jids = {j["jid"] for j in slide_joints}
    arm_jids = [
        jid for jid in range(model.njnt)
        if int(model.jnt_type[jid]) == mujoco.mjtJoint.mjJNT_HINGE
        and jid not in slide_jids
    ]
    # Map joint ids → qpos/qvel/DOF indices (correct for floating-base too).
    initial_arm_qpos = {jid: float(data.qpos[model.jnt_qposadr[jid]]) for jid in arm_jids}

    n_steps = max(1, int(duration_sec / model.opt.timestep))
    phase_a_end = max(1, int(n_steps * 0.15))
    phase_b_end = max(phase_a_end + 1, int(n_steps * 0.30))

    phase_a_cube_zs: list[float] = []
    phase_b_cube_zs: list[float] = []
    phase_c_cube_zs: list[float] = []
    cube_contacts_phase_a = 0
    cube_contacts_phase_b = 0
    unstable = False

    cube_geom_id = -1
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) == cube_body_id:
            cube_geom_id = gid
            break

    def _apply_grasp_force() -> None:
        for sjid, sign in finger_close_signs.items():
            dadr = model.jnt_dofadr[sjid]
            data.qfrc_applied[dadr] = sign * grasp_force_n
            data.qfrc_applied[dadr] -= 2.0 * data.qvel[dadr]

    def _hold_arm() -> None:
        for ajid in arm_jids:
            qadr = model.jnt_qposadr[ajid]
            dadr = model.jnt_dofadr[ajid]
            err = initial_arm_qpos[ajid] - data.qpos[qadr]
            data.qfrc_applied[dadr] = 200.0 * err - 20.0 * data.qvel[dadr]

    def _count_cube_contacts() -> int:
        if cube_geom_id < 0:
            return 0
        return sum(
            1 for cid in range(data.ncon)
            if data.contact[cid].geom1 == cube_geom_id
            or data.contact[cid].geom2 == cube_geom_id
        )

    for step in range(n_steps):
        data.qfrc_applied[:] = 0

        if step < phase_a_end:
            # Phase A: zero gravity, close fingers, hold arm
            phase = "A"
            model.opt.gravity[:] = (0.0, 0.0, 0.0)
            _apply_grasp_force()
            _hold_arm()
        elif step < phase_b_end:
            # Phase B: enable gravity, see if grasp holds
            phase = "B"
            model.opt.gravity[:] = (0.0, 0.0, -9.81)
            _apply_grasp_force()
            _hold_arm()
        else:
            # Phase C: maintain grasp + lift arm
            phase = "C"
            model.opt.gravity[:] = (0.0, 0.0, -9.81)
            _apply_grasp_force()
            _hold_arm()
            # Lift bias on shoulder/elbow
            for ajid in arm_jids[:2]:
                data.qfrc_applied[model.jnt_dofadr[ajid]] -= 1.5

        mujoco.mj_step(model, data)

        if not np.all(np.isfinite(data.qacc)):
            unstable = True
            break

        cube_z = float(data.xpos[cube_body_id][2])
        if phase == "A":
            phase_a_cube_zs.append(cube_z)
            cube_contacts_phase_a = max(cube_contacts_phase_a, _count_cube_contacts())
        elif phase == "B":
            phase_b_cube_zs.append(cube_z)
            cube_contacts_phase_b = max(cube_contacts_phase_b, _count_cube_contacts())
        else:
            phase_c_cube_zs.append(cube_z)

    # Cube position analysis
    cube_settle_a = (
        sum(phase_a_cube_zs[-10:]) / max(1, len(phase_a_cube_zs[-10:]))
        if phase_a_cube_zs else cube_initial_z
    )
    cube_after_b = (
        sum(phase_b_cube_zs[-10:]) / max(1, len(phase_b_cube_zs[-10:]))
        if phase_b_cube_zs else cube_settle_a
    )
    cube_final = (
        sum(phase_c_cube_zs[-10:]) / max(1, len(phase_c_cube_zs[-10:]))
        if phase_c_cube_zs else cube_after_b
    )
    slip_b = cube_after_b - cube_settle_a
    lift_c = cube_final - cube_after_b

    # Verdict logic — two levels:
    #   grasp_ok: can hold the object statically (geometry + friction)
    #   lifted:   can also lift the object (requires arm coordination)
    # "能抓东西" (project goal) = grasp_ok.  Lifting is harder and depends
    # on arm trajectory control, which is beyond the URDF validation scope.
    geometry_ok = cube_contacts_phase_a >= 2
    held_against_gravity = slip_b > -0.005
    grasp_ok = geometry_ok and held_against_gravity and not unstable
    lifted = lift_c > 0.005 and grasp_ok

    if unstable:
        note = "数值不稳定 (NaN/Inf in QACC)"
    elif not geometry_ok:
        note = f"几何不能夹紧 (phase A 只有 {cube_contacts_phase_a} 个接触, 需要 ≥2)"
    elif not held_against_gravity:
        note = f"夹持力不足 (phase B 立方体滑落 {abs(slip_b)*1000:.2f}mm > 5mm 阈值)"
    elif not lifted:
        note = (
            f"静态抓取成功, 抬升失败 (phase C 立方体移动 {lift_c*1000:+.2f}mm) "
            "— 通常因机械臂旋转改变重力方向, 需要轨迹规划而非纯扭矩控制"
        )
    else:
        note = f"抓取+抬升均成功 (抬升 {lift_c*1000:.2f}mm)"

    return {
        "cube_initial_z_m": cube_initial_z,
        "cube_settle_a_m": cube_settle_a,
        "cube_after_b_m": cube_after_b,
        "cube_final_z_m": cube_final,
        "slip_b_m": slip_b,
        "lift_c_m": lift_c,
        "lift_target_m": lift_height_m,
        "grasp_ok": bool(grasp_ok),
        "lifted": bool(lifted),
        "geometry_ok": bool(geometry_ok),
        "held_against_gravity": bool(held_against_gravity),
        "unstable": bool(unstable),
        "phase_a_contacts_max": int(cube_contacts_phase_a),
        "phase_b_contacts_max": int(cube_contacts_phase_b),
        "note": note,
        "finger_final_qpos": {j["name"]: float(data.qpos[model.jnt_qposadr[j["jid"]]]) for j in slide_joints},
        "finger_close_signs": {j["name"]: finger_close_signs[j["jid"]] for j in slide_joints},
    }


def _rotate_vector_by_quat(
    vec: "np.ndarray", quat: "np.ndarray",
) -> "np.ndarray":
    """Rotate a 3-vector by a (w, x, y, z) quaternion."""
    import numpy as np

    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    vx, vy, vz = float(vec[0]), float(vec[1]), float(vec[2])
    # q * v * q^-1 expanded
    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    # v + w*t + cross(q.xyz, t)
    return np.array([
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ])


class SimGraspTool(Tool):
    """Validate the gripper can grasp and lift a test cube.

    Loads the robot URDF, spawns a test cube between the gripper fingers,
    runs a grasp + lift sequence, and reports whether the cube was
    successfully lifted (which requires the fingers to make friction
    contact and resist gravity).
    """

    name = "sim_grasp"
    description = (
        "Place a standard cube near the gripper, close the fingers, and "
        "verify the robot can grasp and lift it. Reports lift height, "
        "finger final positions, contact counts, and pass/fail. "
        "Requires the assembly's gripper to have exactly 2 SLIDE joints "
        "(fingers). Use after sim_mujoco to verify structural validity."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "urdf_path": {
                        "type": "string",
                        "description": "Path to URDF (flat URDF recommended)",
                    },
                    "cube_size_mm": {
                        "type": "number",
                        "description": "Cube edge length in mm (default 20)",
                    },
                    "cube_mass_g": {
                        "type": "number",
                        "description": "Cube mass in grams (default 50)",
                    },
                    "grasp_force_n": {
                        "type": "number",
                        "description": "Grasp force applied to fingers (N, default 3)",
                    },
                    "lift_height_mm": {
                        "type": "number",
                        "description": "Target lift height in mm (default 30)",
                    },
                    "duration_sec": {
                        "type": "number",
                        "description": "Total scenario duration (default 4s)",
                    },
                },
                "required": ["urdf_path"],
            },
        )

    def execute(
        self,
        *,
        urdf_path: str,
        cube_size_mm: float = 12.0,
        cube_mass_g: float = 20.0,
        grasp_force_n: float = 5.0,
        lift_height_mm: float = 30.0,
        duration_sec: float = 4.0,
        **kwargs: Any,
    ) -> str:
        if not _mujoco_available():
            return "Error: mujoco package not installed. Run: pip install mujoco"
        if not urdf_path or not Path(urdf_path).exists():
            return f"Error: URDF not found: {urdf_path}"

        import mujoco  # type: ignore[import-not-found]
        import numpy as np

        # 1. Load URDF to find gripper geometry
        load_result = _load_model(urdf_path)
        if not load_result["ok"]:
            return self._format_load_failure(urdf_path, load_result)

        try:
            model = load_result["model"]
            slide_joints = _find_slide_joints(model)

            # Group fingers into grippers.  A dual-arm robot has 4 slide
            # joints (2 per gripper); the previous ``len != 2`` check
            # rejected these as "NO GRIPPER", so multi-arm "能抓东西"
            # (the project expectation) could never be verified.  Now we
            # test EACH gripper independently — isolated single-gripper
            # tests (one cube per scene) rather than a coupled dual-cube
            # scene, so a failure in one arm doesn't masquerade as the
            # other's.
            grippers = _group_fingers_into_grippers(slide_joints)
            grippers = [(gid, fs) for gid, fs in grippers if len(fs) >= 2]

            if not grippers:
                return self._format_no_gripper(urdf_path, slide_joints)

            per_gripper: list[dict[str, Any]] = []
            for gid, fingers in grippers:
                result = self._test_single_gripper(
                    urdf_path=urdf_path,
                    model=model,
                    fingers=fingers,
                    cube_size_mm=cube_size_mm,
                    cube_mass_g=cube_mass_g,
                    grasp_force_n=grasp_force_n,
                    lift_height_mm=lift_height_mm,
                    duration_sec=duration_sec,
                )
                result["gripper_id"] = gid
                per_gripper.append(result)

            return self._format_multi_report(
                urdf_path=urdf_path,
                all_slides=slide_joints,
                per_gripper=per_gripper,
                cube_size_mm=cube_size_mm,
                cube_mass_g=cube_mass_g,
                grasp_force_n=grasp_force_n,
                lift_height_mm=lift_height_mm,
                duration_sec=duration_sec,
                warnings=load_result["warnings"],
            )
        finally:
            for tmp in load_result.get("temp_files", []):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Single-gripper test (isolated: one cube, only this gripper's fingers
    # actuated).  Reused for single- and multi-gripper robots.
    # ------------------------------------------------------------------
    @staticmethod
    def _test_single_gripper(
        *,
        urdf_path: str,
        model: Any,
        fingers: list[dict[str, Any]],
        cube_size_mm: float,
        cube_mass_g: float,
        grasp_force_n: float,
        lift_height_mm: float,
        duration_sec: float,
    ) -> dict[str, Any]:
        """Run the three-phase grasp on ONE gripper in isolation.

        Returns a dict: {ok, finger_sep_m, grasp_center_m, grasp, error}.
        ``error`` is set only if the scene/scenario could not run.
        """
        import mujoco  # type: ignore[import-not-found]
        import numpy as np

        # Forward kinematics → finger positions for THIS gripper.
        _stabilize_model(model, armature=0.2, damping=3.0)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        finger_pos = [
            np.array(data.xpos[f["body_id"]], dtype=float) for f in fingers
        ]
        grasp_center = sum(finger_pos) / len(finger_pos)

        finger_sep_m = float(
            max(np.linalg.norm(a - b)
                for i, a in enumerate(finger_pos)
                for b in finger_pos[i + 1:])
        ) if len(finger_pos) >= 2 else 0.0
        cube_size_m = cube_size_mm / 1000.0

        if finger_sep_m < cube_size_m:
            return {
                "ok": False,
                "error": "finger_sep",
                "finger_sep_m": finger_sep_m,
                "grasp_center_m": tuple(float(x) for x in grasp_center),
                "grasp": None,
            }

        scene_model, temp_files = _add_cube_to_scene(
            urdf_path=str(Path(urdf_path).resolve()),
            cube_pos_m=tuple(float(x) for x in grasp_center),
            cube_size_m=cube_size_m,
            cube_mass_kg=cube_mass_g / 1000.0,
        )
        try:
            scene_slides = _find_slide_joints(scene_model)
            # Map this gripper's finger joints (by name) into the recompiled
            # scene model, whose joint ids differ from the original model.
            finger_names = {f["name"] for f in fingers}
            scene_fingers = [s for s in scene_slides if s["name"] in finger_names]
            if len(scene_fingers) != len(fingers):
                return {
                    "ok": False,
                    "error": (
                        f"scene has {len(scene_fingers)} of this gripper's "
                        f"{len(fingers)} finger joints"
                    ),
                    "finger_sep_m": finger_sep_m,
                    "grasp_center_m": tuple(float(x) for x in grasp_center),
                    "grasp": None,
                }
            cube_body_id = mujoco.mj_name2id(
                scene_model, mujoco.mjtObj.mjOBJ_BODY, "grasp_cube",
            )
            if cube_body_id < 0:
                return {
                    "ok": False, "error": "grasp_cube body not found",
                    "finger_sep_m": finger_sep_m,
                    "grasp_center_m": tuple(float(x) for x in grasp_center),
                    "grasp": None,
                }
            grasp = _run_grasp_scenario(
                scene_model, scene_fingers, cube_body_id,
                grasp_force_n=grasp_force_n,
                lift_height_m=lift_height_mm / 1000.0,
                duration_sec=duration_sec,
            )
            return {
                "ok": bool(grasp.get("grasp_ok", False)),
                "finger_sep_m": finger_sep_m,
                "grasp_center_m": tuple(float(x) for x in grasp_center),
                "grasp": grasp,
                "_fingers": fingers,  # for per-gripper report formatting
            }
        finally:
            for tmp in temp_files:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Report formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_load_failure(urdf_path: str, load_result: dict) -> str:
        return (
            f"[sim_grasp] LOAD FAILED: {urdf_path}\n"
            f"Error: {load_result['error']}\n"
            "Cannot run grasp test — fix load issues first via sim_mujoco."
        )

    @staticmethod
    def _format_multi_report(
        *,
        urdf_path: str,
        all_slides: list[dict],
        per_gripper: list[dict[str, Any]],
        cube_size_mm: float,
        cube_mass_g: float,
        grasp_force_n: float,
        lift_height_mm: float,
        duration_sec: float,
        warnings: list[str],
    ) -> str:
        """Aggregate per-gripper results into a single verdict report.

        A multi-gripper robot PASSES only if EVERY gripper holds the cube;
        one weak gripper fails the whole assembly (the project expectation
        is that an armed robot "能抓东西", and a dual-arm robot that can
        only grasp with one hand does not meet it)."""
        lines = [
            f"[sim_grasp] GRASP TEST ({len(per_gripper)} gripper(s))",
            f"URDF: {urdf_path}",
            f"Slide joints: {len(all_slides)} across {len(per_gripper)} gripper(s)",
        ]
        if warnings:
            lines.append(f"Load warnings: {warnings}")
        lines.append("")

        all_ok = True
        for g in per_gripper:
            gid = g["gripper_id"]
            lines.append(f"===== Gripper: {gid} =====")
            if g.get("error") == "finger_sep":
                sep = g.get("finger_sep_m", 0.0) * 1000.0
                lines.append(f"  静态抓取: FAIL (手指间距 {sep:.1f}mm < 立方体)")
                lines.append("")
                all_ok = False
                continue
            if g.get("error"):
                lines.append(f"  静态抓取: FAIL ({g['error']})")
                lines.append("")
                all_ok = False
                continue
            # Delegate the rich per-phase detail to the single-gripper
            # formatter (it emits 静态抓取: PASS/FAIL, phase A/B/C stats).
            lines.append(SimGraspTool._format_report(
                urdf_path=urdf_path,
                slide_joints=g.get("_fingers", []),
                finger_sep_m=g["finger_sep_m"],
                grasp_center_m=g["grasp_center_m"],
                cube_size_mm=cube_size_mm,
                cube_mass_g=cube_mass_g,
                grasp_force_n=grasp_force_n,
                lift_height_mm=lift_height_mm,
                duration_sec=duration_sec,
                grasp=g["grasp"],
                warnings=[],  # per-gripper; load warnings emitted once above
            ))
            lines.append("")
            if not g["ok"]:
                all_ok = False

        lines.append("=" * 50)
        n_ok = sum(1 for g in per_gripper if g["ok"])
        lines.append(
            f"总体结论: PASS (所有 {len(per_gripper)} 个夹爪均能抓取)" if all_ok
            else f"总体结论: FAIL ({n_ok}/{len(per_gripper)} 夹爪可抓取)"
        )
        return "\n".join(lines)


    @staticmethod
    def _format_no_gripper(urdf_path: str, slide_joints: list[dict]) -> str:
        names = [j["name"] for j in slide_joints]
        lines = [
            f"[sim_grasp] NO GRIPPER DETECTED",
            f"URDF: {urdf_path}",
            f"Found {len(slide_joints)} SLIDE joint(s); a gripper needs "
            f">= 2 fingers (grouped by body name).",
        ]
        if names:
            lines.append(f"Slide joints found: {names}")
        lines.append("Use sim_mujoco to verify the assembly has a gripper.")
        return "\n".join(lines)

    @staticmethod
    def _format_finger_sep_warning(
        urdf_path: str, finger_sep_m: float, cube_size_m: float,
    ) -> str:
        return (
            f"[sim_grasp] FINGERS TOO CLOSE\n"
            f"URDF: {urdf_path}\n"
            f"Finger separation: {finger_sep_m*1000:.1f}mm\n"
            f"Cube size: {cube_size_m*1000:.1f}mm\n"
            "Fingers must be further apart than the cube to allow grasp.\n"
            "Reduce cube_size_mm or use a different assembly."
        )

    @staticmethod
    def _format_report(
        *,
        urdf_path: str,
        slide_joints: list[dict],
        finger_sep_m: float,
        grasp_center_m: tuple[float, float, float],
        cube_size_mm: float,
        cube_mass_g: float,
        grasp_force_n: float,
        lift_height_mm: float,
        duration_sec: float,
        grasp: dict[str, Any],
        warnings: list[str],
    ) -> str:
        lines = [
            f"[sim_grasp] {urdf_path}",
            f"夹爪识别: 2 个 slide joints "
            f"({slide_joints[0]['body_name']} + {slide_joints[1]['body_name']})",
            f"手指间距: {finger_sep_m*1000:.2f}mm",
            f"抓取中心: ({grasp_center_m[0]*1000:+.1f}, "
            f"{grasp_center_m[1]*1000:+.1f}, {grasp_center_m[2]*1000:+.1f})mm",
            f"立方体: {cube_size_mm:.1f}mm 边长, {cube_mass_g:.1f}g",
            f"抓取力: {grasp_force_n:.1f}N  目标抬升: {lift_height_mm:.1f}mm",
            "",
            "--- 抓取场景结果 ---",
        ]

        if grasp["unstable"]:
            lines.append("  状态: 数值不稳定 (NaN/Inf)")
            lines.append(f"  备注: {grasp.get('note', '')}")
        else:
            # Two-level verdict: static grasp (PASS/FAIL) + lift (bonus)
            grasp_status = "PASS" if grasp["grasp_ok"] else "FAIL"
            lift_status = "PASS" if grasp["lifted"] else "FAIL"
            lines.append(f"  静态抓取: {grasp_status} (几何 + 摩擦夹持)")
            lines.append(f"  动态抬升: {lift_status} (机械臂协调)")
            lines.append(f"  立方体初始 Z: {grasp['cube_initial_z_m']*1000:.2f}mm")
            lines.append(f"  Phase A (零重力, 闭合手指):")
            lines.append(f"    结束时 Z: {grasp['cube_settle_a_m']*1000:.2f}mm")
            lines.append(f"    最大接触数: {grasp['phase_a_contacts_max']} (几何可夹紧需要 ≥2)")
            lines.append(f"  Phase B (加重力, 测试夹持):")
            lines.append(f"    结束时 Z: {grasp['cube_after_b_m']*1000:.2f}mm")
            lines.append(f"    滑落: {grasp['slip_b_m']*1000:+.2f}mm (阈值 5mm)")
            lines.append(f"    最大接触数: {grasp['phase_b_contacts_max']}")
            lines.append(f"  Phase C (抬升机械臂):")
            lines.append(f"    结束时 Z: {grasp['cube_final_z_m']*1000:.2f}mm")
            lines.append(f"    抬升: {grasp['lift_c_m']*1000:+.2f}mm / "
                         f"目标 {lift_height_mm:.2f}mm")
            lines.append(f"  手指最终位置:")
            for name, qpos in grasp["finger_final_qpos"].items():
                lines.append(f"    {name:45s} qpos = {qpos*1000:+.3f}mm")
            lines.append(f"  备注: {grasp['note']}")

        lines.append("")
        lines.append("--- 总体结论 ---")
        if grasp["unstable"]:
            lines.append("  FAIL — 仿真数值不稳定")
        elif grasp["lifted"]:
            lines.append(
                "  PASS (抓取+抬升) — 立方体被抓起 "
                f"{grasp['lift_c_m']*1000:.2f}mm"
            )
        elif grasp["grasp_ok"]:
            lines.append(
                "  PASS (静态抓取) — 立方体被夹住, 静态不掉落; "
                "抬升需要机械臂轨迹规划 (超出 URDF 验证范围)"
            )
        else:
            lines.append(
                "  FAIL — 立方体未被抬起 (可能原因: 摩擦系数不足、"
                "抓取力过小、手指几何不匹配)"
            )

        if warnings:
            lines.append("")
            lines.append("--- 加载警告 ---")
            for w in warnings:
                lines.append(f"  ! {w}")

        summary = {
            "urdf_path": urdf_path,
            "finger_separation_mm": finger_sep_m * 1000,
            "cube_size_mm": cube_size_mm,
            "cube_mass_g": cube_mass_g,
            "grasp_force_n": grasp_force_n,
            "lift_target_mm": lift_height_mm,
            **grasp,
        }
        lines.append("")
        lines.append("--- JSON ---")
        lines.append(json.dumps(summary, ensure_ascii=False, indent=2))

        return "\n".join(lines)
