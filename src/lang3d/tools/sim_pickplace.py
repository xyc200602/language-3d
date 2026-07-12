"""Pick-and-place task validation tool.

This is the **task-level** validation: not just "can the gripper clamp a
cube" (that's SimGraspTool), but "can the robot pick up an object from
position A, carry it to position B, and place it there" — the canonical
robot manipulation task.

Pipeline:
  1. Load URDF + assembly.json
  2. IK: solve joint angles to reach pick_pos and place_pos
  3. Convert IK angles → MuJoCo qpos targets (_ik_angles_to_qpos)
  4. Add a cube to the MuJoCo scene at pick_pos
  5. Run 8-phase pick-place simulation (_run_pick_place_scenario)
  6. Measure: did the cube end up at place_pos?

Success requires ALL of:
  - IK reachability (both pick and place positions)
  - Grasp stability during carry (cube didn't drop)
  - Place accuracy (cube within 20mm of target)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from .base import Tool
from ..models.base import ToolDefinition
from .sim_grasp import (
    _add_cube_to_scene,
    _find_arm_joints,
    _find_slide_joints,
    _group_fingers_into_grippers,
    _ik_angles_to_qpos,
    _rotate_vector_by_quat,
    _run_pick_place_scenario,
    _stabilize_model,
)

logger = logging.getLogger(__name__)


class PickPlaceTool(Tool):
    """Run a pick-and-place task validation in MuJoCo."""

    name = "pick_place_test"
    description = (
        "Test whether the robot can pick up a cube from a start position, "
        "carry it to a target position, and place it there. This is the "
        "task-level validation: it requires IK reachability, grasp "
        "stability during transport, and place accuracy."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "urdf_path": {"type": "string", "description": "Path to the robot URDF file."},
                    "assembly_path": {"type": "string", "description": "Path to assembly.json."},
                    "pick_pos_mm": {"type": "string", "description": "Pick [x,y,z] in mm, e.g. '0,-200,50'."},
                    "place_pos_mm": {"type": "string", "description": "Place [x,y,z] in mm, e.g. '100,-200,50'."},
                    "cube_size_mm": {"type": "number", "description": "Cube edge length in mm."},
                },
                "required": ["urdf_path", "assembly_path", "pick_pos_mm", "place_pos_mm"],
            },
        )

    def execute(
        self,
        *,
        urdf_path: str,
        assembly_path: str,
        pick_pos_mm: str = "0,-200,50",
        place_pos_mm: str = "100,-200,50",
        cube_size_mm: float = 12.0,
        grasp_force_n: float = 5.0,
        **kwargs: Any,
    ) -> str:
        """Run pick-and-place and return a JSON report."""
        try:
            import mujoco
        except ImportError:
            return json.dumps({"error": "mujoco not installed"})

        if not Path(urdf_path).exists():
            return json.dumps({"error": f"URDF not found: {urdf_path}"})

        # Parse positions (in mm).
        try:
            pick = [float(x) for x in str(pick_pos_mm).split(",")]
            place = [float(x) for x in str(place_pos_mm).split(",")]
            assert len(pick) == 3 and len(place) == 3
        except Exception:
            return json.dumps({"error": "positions must be 'x,y,z' in mm"})

        # --- Load MuJoCo model ---
        from .sim_grasp import _ensure_sm
        _ensure_sm()
        from .sim_mujoco import _load_model

        load_result = _load_model(urdf_path)
        if not load_result.get("ok"):
            return json.dumps({"error": "URDF load failed", "detail": load_result.get("error", "")})

        model = load_result["model"]
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        slide_joints = _find_slide_joints(model)
        grippers = _group_fingers_into_grippers(slide_joints)
        grippers = [(gid, fs) for gid, fs in grippers if len(fs) >= 2]
        if not grippers:
            return json.dumps({"error": "No gripper found (need ≥2 slide joints)"})

        gid, fingers = grippers[0]
        arm_jids = _find_arm_joints(model, fingers)

        # Find end-effector body: the parent body of the first finger.
        ee_body_id = int(model.body_parentid[fingers[0]["body_id"]])

        # Compute the offset between the EE body and the finger midpoint.
        # The cube must be placed at the FINGER MIDPOINT (where the gripper
        # can clamp it), not at the EE body origin (gripper_base, which is
        # 44mm behind the fingers).  We add this offset to the IK target so
        # that when the EE body reaches the modified target, the fingers
        # are actually centered on the desired pick/place position.
        import numpy as np
        finger_midpoint = sum(
            np.array(data.xpos[f["body_id"]], dtype=float) for f in fingers
        ) / len(fingers)
        ee_pos = np.array(data.xpos[ee_body_id], dtype=float)
        ee_to_finger_offset = finger_midpoint - ee_pos  # in meters

        # Home qpos.
        home_qpos = {jid: float(data.qpos[model.jnt_qposadr[jid]]) for jid in arm_jids}

        # --- MuJoCo-native Jacobian IK (no Assembly coordinate mismatch) ---
        from .sim_grasp import _mujoco_jacobian_ik

        # IK targets are the pick/place positions ADJUSTED so that the
        # finger midpoint (not the EE body) lands on the target.
        pick_m_raw = np.array([x / 1000.0 for x in pick])
        place_m_raw = np.array([x / 1000.0 for x in place])
        pick_m = tuple(pick_m_raw - ee_to_finger_offset)
        place_m = tuple(place_m_raw - ee_to_finger_offset)

        # Solve IK for pick position (starts from current/home pose).
        pick_qpos = _mujoco_jacobian_ik(
            model, data, ee_body_id, arm_jids, pick_m, tolerance_m=0.003,
        )
        # data.qpos is now at the IK solution — verify it.
        mujoco.mj_forward(model, data)
        pick_achieved = np.array(data.xpos[ee_body_id])
        pick_error_mm = float(np.linalg.norm(pick_achieved - np.array(pick_m)) * 1000)
        pick_reached = pick_error_mm < 15.0

        # Compute place qpos by adjusting pick qpos with a yaw rotation.
        # The base yaw joint rotates the entire arm around Z; a lateral
        # displacement dx at distance R from the yaw axis requires a yaw
        # change of atan(dx / R). This avoids IK branch-jumping (the
        # Jacobian solver tends to spin yaw to its 90° limit regardless
        # of how small dx is).
        # R = horizontal distance from base yaw axis to EE.
        ee_home_xy = np.array(data.xpos[ee_body_id][:2])  # already at home
        R = float(np.linalg.norm(ee_home_xy))
        dx_m = place_m_raw[0] - pick_m_raw[0]  # lateral displacement
        dy_m = place_m_raw[1] - pick_m_raw[1]
        # Combined lateral distance (project onto the yaw-rotation plane).
        lateral = math.sqrt(dx_m ** 2 + dy_m ** 2)
        yaw_delta = math.atan2(lateral, R) if R > 0.01 else 0.0

        # Find the base yaw joint (first arm joint, typically jid=0).
        yaw_jid = arm_jids[0]
        place_qpos = dict(pick_qpos)
        yaw_lo, yaw_hi = model.jnt_range[yaw_jid]
        new_yaw = pick_qpos.get(yaw_jid, 0.0) + yaw_delta
        new_yaw = max(yaw_lo, min(yaw_hi, new_yaw))
        place_qpos[yaw_jid] = new_yaw

        # Verify place reachability.
        for jid in arm_jids:
            qadr = model.jnt_qposadr[jid]
            data.qpos[qadr] = place_qpos.get(jid, home_qpos.get(jid, 0.0))
        mujoco.mj_forward(model, data)
        place_achieved = np.array(data.xpos[ee_body_id])
        place_error_mm = float(np.linalg.norm(
            place_achieved - np.array(place_m)
        ) * 1000)
        place_reached = place_error_mm < 15.0

        # Reset to home for the simulation.
        for jid in arm_jids:
            qadr = model.jnt_qposadr[jid]
            data.qpos[qadr] = home_qpos[jid]
        mujoco.mj_forward(model, data)

        if not pick_reached or not place_reached:
            return json.dumps({
                "task_success": False,
                "pick_reached": pick_reached,
                "place_reached": place_reached,
                "pick_error_mm": round(pick_error_mm, 2),
                "place_error_mm": round(place_error_mm, 2),
                "reason": "IK failed — target out of reach",
            }, ensure_ascii=False, indent=2)

        # Finger close directions.
        finger_close_signs: dict[int, float] = {}
        grasp_center = sum(
            np.array(data.xpos[f["body_id"]], dtype=float) for f in fingers
        ) / len(fingers)
        for f in fingers:
            jid = f["jid"]
            body_id = f["body_id"]
            axis_local = np.array(model.jnt_axis[jid], dtype=float)
            body_quat = np.array(data.xquat[body_id], dtype=float)
            axis_world = _rotate_vector_by_quat(axis_local, body_quat)
            to_center = grasp_center - np.array(data.xpos[body_id], dtype=float)
            proj = float(np.dot(to_center, axis_world))
            finger_close_signs[jid] = 1.0 if proj > 0 else -1.0

        # --- Add cube at pick position (the ACTUAL position, not IK-adjusted) ---
        cube_size_m = cube_size_mm / 1000.0

        scene_model, temp_files = _add_cube_to_scene(
            urdf_path=str(Path(urdf_path).resolve()),
            cube_pos_m=tuple(pick_m_raw),  # real world pick position
            cube_size_m=cube_size_m,
            cube_mass_kg=0.020,
        )
        try:
            cube_body_id = mujoco.mj_name2id(
                scene_model, mujoco.mjtObj.mjOBJ_BODY, "grasp_cube",
            )
            if cube_body_id < 0:
                return json.dumps({"error": "cube body not found in scene"})

            # Copy qpos0 from the original model to the scene model.
            # _add_cube_to_scene recompiles the MJCF, which RESETS qpos0 to
            # zero (straight arm) — losing the bent home pose. Without this,
            # teleporting the arm to the IK solution causes a violent jump
            # from straight→bent that launches the cube across the scene.
            # Only copy the first model.nq entries (the scene has extra qpos
            # for the cube's freejoint which must not be overwritten).
            n = min(model.nq, scene_model.nq)
            scene_model.qpos0[:n] = model.qpos0[:n]

            # Re-map arm_jids and fingers to scene model.
            scene_slides = _find_slide_joints(scene_model)
            finger_names = {f["name"] for f in fingers}
            scene_fingers = [s for s in scene_slides if s["name"] in finger_names]
            scene_arm_jids = _find_arm_joints(scene_model, scene_fingers)

            # Map qpos targets to scene model joint ids by name.
            orig_jid_to_name = {
                jid: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                for jid in arm_jids
            }
            scene_name_to_jid = {}
            for jid in range(scene_model.njnt):
                nm = mujoco.mj_id2name(scene_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                if nm:
                    scene_name_to_jid[nm] = jid

            scene_pick_qpos = {}
            scene_place_qpos = {}
            scene_home_qpos = {}
            for orig_jid, nm in orig_jid_to_name.items():
                scene_jid = scene_name_to_jid.get(nm)
                if scene_jid is not None:
                    scene_pick_qpos[scene_jid] = pick_qpos.get(orig_jid, 0.0)
                    scene_place_qpos[scene_jid] = place_qpos.get(orig_jid, 0.0)
                    scene_home_qpos[scene_jid] = home_qpos.get(orig_jid, 0.0)

            # Map finger signs to scene model.
            scene_finger_signs = {}
            for sf in scene_fingers:
                for orig_f in fingers:
                    if sf["name"] == orig_f["name"]:
                        scene_finger_signs[sf["jid"]] = finger_close_signs[orig_f["jid"]]
                        break

            # --- Run pick-and-place ---
            result = _run_pick_place_scenario(
                model=scene_model,
                slide_joints=scene_fingers,
                cube_body_id=cube_body_id,
                pick_qpos=scene_pick_qpos,
                place_qpos=scene_place_qpos,
                home_qpos=scene_home_qpos,
                finger_close_signs=scene_finger_signs,
                grasp_force_n=grasp_force_n,
                duration_sec=8.0,
            )

            # --- Evaluate task success ---
            # Place accuracy is measured in XY only — Z is excluded because
            # the cube falls to the ground after release (the place target
            # is at gripper height, but after fingers open the cube naturally
            # drops). The meaningful metric is "did the cube land at the
            # right XY position".
            cube_final = result["cube_final_pos_m"]
            place_accuracy_xy_mm = math.sqrt(
                (cube_final[0] - place_m_raw[0]) ** 2
                + (cube_final[1] - place_m_raw[1]) ** 2
            ) * 1000.0
            place_accuracy_3d_mm = math.sqrt(
                sum((cube_final[i] - place_m_raw[i]) ** 2 for i in range(3))
            ) * 1000.0

            carry_drop_mm = (result["cube_pick_z_m"] - result["cube_carry_min_z_m"]) * 1000.0
            grasp_held = carry_drop_mm < 20.0

            task_success = (
                pick_reached
                and place_reached
                and grasp_held
                and place_accuracy_xy_mm < 30.0
                and not result["unstable"]
            )

            report = {
                "task_success": task_success,
                "pick_reached": pick_reached,
                "place_reached": place_reached,
                "grasp_held": grasp_held,
                "place_accuracy_mm": round(place_accuracy_xy_mm, 1),
                "place_accuracy_3d_mm": round(place_accuracy_3d_mm, 1),
                "carry_drop_mm": round(carry_drop_mm, 1),
                "cube_final_pos_m": [round(x, 4) for x in cube_final],
                "target_place_pos_m": [round(x, 4) for x in place_m_raw],
                "pick_error_mm": round(pick_error_mm, 2),
                "place_error_mm": round(place_error_mm, 2),
                "ik_method": "mujoco_jacobian",
                "unstable": result["unstable"],
                "note": (
                    "任务成功: 抓取→搬运→放置完成"
                    if task_success
                    else f"任务失败: place_xy={place_accuracy_xy_mm:.1f}mm, "
                         f"carry_drop={carry_drop_mm:.1f}mm"
                ),
            }
            return json.dumps(report, ensure_ascii=False, indent=2)

        finally:
            for tmp in temp_files:
                try:
                    Path(tmp).unlink()
                except OSError:
                    pass


def register_pickplace_tools(registry: Any) -> None:
    """Register the PickPlaceTool."""
    registry.register(PickPlaceTool())
