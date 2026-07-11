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
        if not Path(assembly_path).exists():
            return json.dumps({"error": f"assembly.json not found: {assembly_path}"})

        # Parse positions.
        try:
            pick = [float(x) for x in str(pick_pos_mm).split(",")]
            place = [float(x) for x in str(place_pos_mm).split(",")]
            assert len(pick) == 3 and len(place) == 3
        except Exception:
            return json.dumps({"error": "positions must be 'x,y,z' in mm"})

        # Load assembly for IK.
        from ..knowledge.mechanics import Assembly, Joint, Part

        raw = json.loads(Path(assembly_path).read_text("utf-8"))
        parts = [Part(**p) for p in raw.get("parts", [])]
        joints = [Joint(**j) for j in raw.get("joints", [])]
        assembly = Assembly(
            name=raw.get("name", ""),
            parts=parts,
            joints=joints,
            description=raw.get("description", ""),
            default_angles=raw.get("default_angles", {}),
        )

        # --- IK: solve for pick and place ---
        from .ik_solver import solve_ik

        ik_pick = solve_ik(assembly, target=tuple(pick), approach="auto", tolerance_mm=10.0)
        ik_place = solve_ik(assembly, target=tuple(place), approach="auto", tolerance_mm=10.0,
                            initial_angles=ik_pick.joint_angles)

        pick_reached = ik_pick.reachable
        place_reached = ik_place.reachable

        if not pick_reached or not place_reached:
            return json.dumps({
                "task_success": False,
                "pick_reached": pick_reached,
                "place_reached": place_reached,
                "pick_error_mm": round(ik_pick.error_mm, 2),
                "place_error_mm": round(ik_place.error_mm, 2),
                "reason": "IK failed — target out of reach",
            }, ensure_ascii=False, indent=2)

        # --- Load MuJoCo model ---
        from .sim_grasp import _ensure_sm
        _ensure_sm()
        from .sim_mujoco import _load_model

        load_result = _load_model(urdf_path)
        if not load_result.get("ok"):
            return json.dumps({"error": "URDF load failed", "detail": load_result.get("error", "")})

        model = load_result["model"]
        slide_joints = _find_slide_joints(model)
        grippers = _group_fingers_into_grippers(slide_joints)
        grippers = [(gid, fs) for gid, fs in grippers if len(fs) >= 2]
        if not grippers:
            return json.dumps({"error": "No gripper found (need ≥2 slide joints)"})

        gid, fingers = grippers[0]
        arm_jids = _find_arm_joints(model, fingers)

        # --- Convert IK angles to MuJoCo qpos ---
        pick_qpos = _ik_angles_to_qpos(ik_pick.joint_angles, model, arm_jids)
        place_qpos = _ik_angles_to_qpos(ik_place.joint_angles, model, arm_jids)

        # Home qpos from current model state.
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        home_qpos = {jid: float(data.qpos[model.jnt_qposadr[jid]]) for jid in arm_jids}

        # Finger close directions.
        finger_close_signs: dict[int, float] = {}
        import numpy as np
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

        # --- Add cube at pick position ---
        cube_size_m = cube_size_mm / 1000.0
        pick_m = tuple(x / 1000.0 for x in pick)
        place_m = tuple(x / 1000.0 for x in place)

        scene_model, temp_files = _add_cube_to_scene(
            urdf_path=str(Path(urdf_path).resolve()),
            cube_pos_m=pick_m,
            cube_size_m=cube_size_m,
            cube_mass_kg=0.020,
        )
        try:
            cube_body_id = mujoco.mj_name2id(
                scene_model, mujoco.mjtObj.mjOBJ_BODY, "grasp_cube",
            )
            if cube_body_id < 0:
                return json.dumps({"error": "cube body not found in scene"})

            # Re-map arm_jids and fingers to scene model.
            scene_slides = _find_slide_joints(scene_model)
            finger_names = {f["name"] for f in fingers}
            scene_fingers = [s for s in scene_slides if s["name"] in finger_names]
            scene_arm_jids = _find_arm_joints(scene_model, scene_fingers)

            # Re-map qpos targets to scene model joint ids.
            scene_pick_qpos = {}
            scene_place_qpos = {}
            scene_home_qpos = {}
            scene_finger_signs = {}
            for sj in scene_slides:
                scene_finger_signs[sj["jid"]] = finger_close_signs.get(fingers[0]["jid"], 1.0)
                if len(fingers) > 1 and sj["name"] == fingers[1]["name"]:
                    scene_finger_signs[sj["jid"]] = finger_close_signs.get(fingers[1]["jid"], -1.0)

            # Map by joint name.
            orig_jid_to_name = {
                jid: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                for jid in arm_jids
            }
            scene_name_to_jid = {}
            for jid in range(scene_model.njnt):
                nm = mujoco.mj_id2name(scene_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                if nm:
                    scene_name_to_jid[nm] = jid

            for orig_jid, nm in orig_jid_to_name.items():
                scene_jid = scene_name_to_jid.get(nm)
                if scene_jid is not None:
                    scene_pick_qpos[scene_jid] = pick_qpos.get(orig_jid, 0.0)
                    scene_place_qpos[scene_jid] = place_qpos.get(orig_jid, 0.0)
                    scene_home_qpos[scene_jid] = home_qpos.get(orig_jid, 0.0)

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
            cube_final = result["cube_final_pos_m"]
            place_accuracy_mm = math.sqrt(
                sum((cube_final[i] - place_m[i]) ** 2 for i in range(3))
            ) * 1000.0

            carry_drop_mm = (result["cube_pick_z_m"] - result["cube_carry_min_z_m"]) * 1000.0
            grasp_held = carry_drop_mm < 20.0  # cube didn't drop >20mm during carry

            task_success = (
                pick_reached
                and place_reached
                and grasp_held
                and place_accuracy_mm < 30.0
                and not result["unstable"]
            )

            report = {
                "task_success": task_success,
                "pick_reached": pick_reached,
                "place_reached": place_reached,
                "grasp_held": grasp_held,
                "place_accuracy_mm": round(place_accuracy_mm, 1),
                "carry_drop_mm": round(carry_drop_mm, 1),
                "cube_final_pos_m": [round(x, 4) for x in cube_final],
                "target_place_pos_m": [round(x, 4) for x in place_m],
                "pick_ik_method": ik_pick.method,
                "place_ik_method": ik_place.method,
                "pick_ik_error_mm": round(ik_pick.error_mm, 2),
                "place_ik_error_mm": round(ik_place.error_mm, 2),
                "unstable": result["unstable"],
                "note": (
                    "任务成功: 抓取→搬运→放置完成"
                    if task_success
                    else f"任务失败: place_accuracy={place_accuracy_mm:.1f}mm, "
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
