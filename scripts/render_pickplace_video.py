#!/usr/bin/env python
"""Render the 4dof_arm pick-and-place task as an MP4 video.

Usage:
    python scripts/render_pickplace_video.py
Output:
    data/screenshots/pickplace_4dof.mp4
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

import mujoco
from PIL import Image, ImageDraw

from lang3d.tools.sim_grasp import (
    _ensure_sm,
    _add_cube_to_scene,
    _find_slide_joints,
    _group_fingers_into_grippers,
    _find_arm_joints,
    _rotate_vector_by_quat,
    _mujoco_jacobian_ik,
)
from lang3d.tools.sim_mujoco import _load_model, _stabilize_model

_ensure_sm()


def main():
    bm = open(
        PROJECT / "data/runs/4dof_arm/BENCHMARK", encoding="utf-8"
    ).read().strip().split("\n")[0]
    urdf = str(PROJECT / f"data/runs/4dof_arm/{bm}/engineering_package/urdf.xml")
    output_mp4 = str(PROJECT / "data/screenshots/pickplace_4dof.mp4")

    model = _load_model(urdf)["model"]
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    slides = _find_slide_joints(model)
    grippers = _group_fingers_into_grippers(slides)
    gid, fingers = grippers[0]
    arm_jids = _find_arm_joints(model, fingers)
    ee_bid = int(model.body_parentid[fingers[0]["body_id"]])

    fm = sum(np.array(data.xpos[f["body_id"]]) for f in fingers) / len(fingers)
    ee_pos = np.array(data.xpos[ee_bid])
    offset = fm - ee_pos
    home_qpos = {
        jid: float(data.qpos[model.jnt_qposadr[jid]]) for jid in arm_jids
    }

    # IK pick (target = finger midpoint, adjusted for EE offset)
    pick_m = fm - offset
    pick_qpos = _mujoco_jacobian_ik(
        model, data, ee_bid, arm_jids, tuple(pick_m), tolerance_m=0.003
    )
    for jid in arm_jids:
        data.qpos[model.jnt_qposadr[jid]] = home_qpos[jid]
    mujoco.mj_forward(model, data)

    # Place = pick + 30mm X (via yaw rotation)
    R = float(np.linalg.norm(data.xpos[ee_bid][:2]))
    yaw_delta = math.atan2(0.030, R)
    place_qpos = dict(pick_qpos)
    place_qpos[arm_jids[0]] += yaw_delta

    # Finger signs
    finger_signs = {}
    grasp_center = sum(
        np.array(data.xpos[f["body_id"]]) for f in fingers
    ) / len(fingers)
    for f in fingers:
        axis_local = np.array(model.jnt_axis[f["jid"]])
        body_quat = np.array(data.xquat[f["body_id"]])
        axis_world = _rotate_vector_by_quat(axis_local, body_quat)
        to_center = grasp_center - np.array(data.xpos[f["body_id"]])
        finger_signs[f["jid"]] = 1.0 if np.dot(to_center, axis_world) > 0 else -1.0

    # Build scene
    scene_model, tmp = _add_cube_to_scene(
        str(Path(urdf).resolve()), tuple(fm), 0.012, 0.020
    )
    n = min(model.nq, scene_model.nq)
    scene_model.qpos0[:n] = model.qpos0[:n]

    cube_bid = mujoco.mj_name2id(
        scene_model, mujoco.mjtObj.mjOBJ_BODY, "grasp_cube"
    )
    scene_slides = _find_slide_joints(scene_model)
    finger_names = {f["name"] for f in fingers}
    scene_fingers = [
        s for s in scene_slides if s["name"] in finger_names
    ]
    scene_arm = _find_arm_joints(scene_model, scene_fingers)

    # Map qpos by name
    scene_pick, scene_place, scene_home = {}, {}, {}
    for jid in arm_jids:
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        for sjid in range(scene_model.njnt):
            snm = mujoco.mj_id2name(
                scene_model, mujoco.mjtObj.mjOBJ_JOINT, sjid
            )
            if snm == nm:
                scene_pick[sjid] = pick_qpos.get(jid, 0)
                scene_place[sjid] = place_qpos.get(jid, 0)
                scene_home[sjid] = home_qpos.get(jid, 0)
                break

    scene_signs = {}
    for sf in scene_fingers:
        for of in fingers:
            if sf["name"] == of["name"]:
                scene_signs[sf["jid"]] = finger_signs[of["jid"]]
                break

    # Setup
    _stabilize_model(scene_model, armature=0.2, damping=3.0)
    scene_model.opt.timestep = 0.0005
    sdata = mujoco.MjData(scene_model)

    renderer = mujoco.Renderer(scene_model, height=480, width=640)
    # Create a custom camera view.
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.fixedcamid = -1
    cam.lookat[:] = [0, -0.3, 0.1]
    cam.distance = 0.8
    cam.azimuth = -60
    cam.elevation = -25

    def _ss(t):
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    n_steps = int(8.0 / scene_model.opt.timestep)
    phases = {
        "approach": (0.00, 0.06),
        "grasp":    (0.06, 0.20),
        "hold":     (0.20, 0.26),
        "lift":     (0.26, 0.30),
        "carry":    (0.30, 0.65),
        "descend":  (0.65, 0.72),
        "release":  (0.72, 0.80),
        "retreat":  (0.80, 0.92),
    }
    pb = {k: (int(n_steps * v[0]), int(n_steps * v[1])) for k, v in phases.items()}
    end_step = pb["retreat"][1]

    # Teleport arm to pick
    for ajid in scene_arm:
        qadr = scene_model.jnt_qposadr[ajid]
        sdata.qpos[qadr] = scene_pick.get(ajid, scene_home.get(ajid, 0.0))
    mujoco.mj_forward(scene_model, sdata)

    frame_every = 40
    frames = []
    phase_labels = {
        "approach": "1. Approach Pick",
        "grasp":    "2. Grasp Close (zero-G)",
        "hold":     "3. Grasp Hold (gravity ON)",
        "lift":     "4. Lift",
        "carry":    "5. Carry (+30mm X)",
        "descend":  "6. Descend to Place",
        "release":  "7. Release",
        "retreat":  "8. Retreat",
    }
    cur_phase = "approach"

    for step in range(end_step):
        sdata.qfrc_applied[:] = 0

        if step < pb["approach"][1]:
            cur_phase = "approach"
            gf = 0.0
            tgt = scene_pick
            g = (0, 0, 0)
        elif step < pb["grasp"][1]:
            cur_phase = "grasp"
            gf = 5.0
            tgt = scene_pick
            g = (0, 0, 0)
        elif step < pb["hold"][1]:
            cur_phase = "hold"
            gf = 5.0
            tgt = scene_pick
            g = (0, 0, -9.81)
        elif step < pb["lift"][1]:
            cur_phase = "lift"
            gf = 5.0
            tgt = scene_pick
            g = (0, 0, -9.81)
        elif step < pb["carry"][1]:
            cur_phase = "carry"
            gf = 5.0
            t = _ss((step - pb["carry"][0]) / max(pb["carry"][1] - pb["carry"][0], 1))
            tgt = {}
            for jid in set(scene_pick) | set(scene_place):
                h = scene_pick.get(jid, 0)
                p = scene_place.get(jid, h)
                tgt[jid] = h + (p - h) * t
            g = (0, 0, -9.81)
        elif step < pb["descend"][1]:
            cur_phase = "descend"
            gf = 5.0
            tgt = scene_place
            g = (0, 0, -9.81)
        elif step < pb["release"][1]:
            cur_phase = "release"
            gf = 0.0
            tgt = scene_place
            g = (0, 0, -9.81)
        else:
            cur_phase = "retreat"
            gf = 0.0
            tgt = {jid: v + 0.3 for jid, v in scene_place.items()}
            g = (0, 0, -9.81)

        scene_model.opt.gravity[:] = g
        for ajid in scene_arm:
            qadr = scene_model.jnt_qposadr[ajid]
            dadr = scene_model.jnt_dofadr[ajid]
            sdata.qfrc_applied[dadr] = (
                200.0 * (tgt.get(ajid, 0) - sdata.qpos[qadr])
                - 20.0 * sdata.qvel[dadr]
            )
        for sjid, sign in scene_signs.items():
            dadr = scene_model.jnt_dofadr[sjid]
            sdata.qfrc_applied[dadr] = sign * gf - 2.0 * sdata.qvel[dadr]

        mujoco.mj_step(scene_model, sdata)

        if step % frame_every == 0:
            renderer.update_scene(sdata, camera=cam)
            pix = renderer.render().copy()
            frames.append((pix, cur_phase, step))

    print(f"Rendered {len(frames)} frames")

    # Save frames with labels
    frames_dir = PROJECT / "data/screenshots/pickplace_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    for i, (pix, phase, step) in enumerate(frames):
        img = Image.fromarray(pix)
        draw = ImageDraw.Draw(img)
        label = f"{phase_labels.get(phase, phase)} [{step}/{end_step}]"
        draw.text((10, 10), label, fill=(255, 255, 255))
        cube_x = sdata.xpos[cube_bid][0] * 1000
        draw.text(
            (10, 460),
            f"Cube X: {cube_x:.0f}mm (target: +30mm)",
            fill=(0, 255, 0),
        )
        img.save(frames_dir / f"frame_{i:04d}.png")

    # Encode MP4
    os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-framerate", "20",
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20",
        output_mp4,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        sz = Path(output_mp4).stat().st_size / 1024
        print(f"Video saved: {output_mp4} ({sz:.0f} KB)")
    else:
        print(f"ffmpeg error: {result.stderr[-300:]}")

    for t in tmp:
        try:
            Path(t).unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
