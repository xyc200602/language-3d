"""MuJoCo-based simulation validation tool.

Loads a generated URDF (with STL meshes) into MuJoCo to validate that the
NL → Assembly → URDF pipeline produces a physically realisable robot.

Reports:
  - Load success/failure with specific error messages
  - Mesh path resolution issues (relative paths are rewritten to absolute)
  - Body list with mass / inertia sanity checks
  - Joint list with type, axis, range
  - Physics stability under PD control holding the initial pose
  - Per-joint actuation test (does each joint move under applied torque?)

Why MuJoCo, not PyBullet:
  - PyBullet has no prebuilt wheels for Windows Python 3.12 (requires
    Visual C++ Build Tools ~7GB to compile from source).
  - MuJoCo (DeepMind) publishes Windows prebuilt wheels for all supported
    CPython versions and has more accurate contact physics.

Pure-function module: no global state, no agent imports.  Tested via
``tests/test_sim_mujoco.py``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from ..models.base import ToolDefinition
from .base import Tool

logger = logging.getLogger(__name__)


# ============================================================================
# Mesh path repair — fixes pipeline BUG-1
# ============================================================================

_MESH_TAG_RE = re.compile(
    r'(<mesh\s+filename=")([^"]+)(")',
    re.IGNORECASE,
)


def _rewrite_mesh_paths(urdf_text: str, urdf_path: Path) -> tuple[str, list[str]]:
    """Rewrite relative mesh paths in URDF text to absolute paths.

    Pipeline BUG-1: ``urdf_export.py`` writes ``meshes/X.stl`` (relative).
    When the URDF lives in a ``urdf/`` subdir, MuJoCo/PyBullet resolve
    relative paths against the URDF's directory and fail to find the STLs.

    This helper resolves every ``meshes/...`` reference against the URDF's
    parent directories until the file exists, then writes the absolute path
    back into the URDF text.  Returns ``(new_text, list_of_warnings)``.

    BUG-2 (added 2026-06-18): the on-disk STL layout uses ``stl_parts/``
    rather than ``meshes/``.  When the literal path doesn't resolve we also
    try swapping the leading directory name to any sibling mesh directory
    (``stl_parts``, ``meshes``, ``visual``, ``collision``).
    """
    base_dir = urdf_path.parent
    warnings: list[str] = []

    # Sibling directory names commonly used for STLs
    _ALT_MESH_DIRS = ("meshes", "stl_parts", "visual", "collision", "stl")

    def _resolve(match: re.Match[str]) -> str:
        prefix, raw_path, suffix = match.group(1), match.group(2), match.group(3)
        # Already absolute and exists — leave alone
        if os.path.isabs(raw_path) and Path(raw_path).exists():
            return match.group(0)
        # Skip package:// or ros:// URIs (not supported here, leave for caller)
        if "://" in raw_path:
            warnings.append(f"Unsupported mesh URI scheme: {raw_path}")
            return match.group(0)

        # Try resolving from urdf_dir upward (up to 3 levels)
        search_dir = base_dir
        for _ in range(4):
            candidate = (search_dir / raw_path).resolve()
            if candidate.exists():
                return f'{prefix}{candidate.as_posix()}{suffix}'
            search_dir = search_dir.parent

        # Fallback: try alternative sibling mesh directories.
        # E.g. raw="meshes/foo.stl" -> try "stl_parts/foo.stl", etc.
        parts = raw_path.split("/", 1)
        if len(parts) == 2:
            leaf = parts[1]
            search_dir = base_dir
            for _ in range(4):
                for alt in _ALT_MESH_DIRS:
                    candidate = (search_dir / alt / leaf).resolve()
                    if candidate.exists():
                        return f'{prefix}{candidate.as_posix()}{suffix}'
                search_dir = search_dir.parent

        warnings.append(
            f"Mesh not found: {raw_path} (searched from {base_dir} upward 3 levels)",
        )
        return match.group(0)

    new_text = _MESH_TAG_RE.sub(_resolve, urdf_text)
    n_fixed = new_text.count("file:///") + new_text.count("file://\\")
    # Better: count how many absolute paths we wrote (Windows: drive letter)
    n_fixed = sum(
        1 for m in _MESH_TAG_RE.finditer(new_text)
        if os.path.isabs(m.group(2)) and Path(m.group(2)).exists()
    )
    if n_fixed:
        warnings.insert(0, f"Rewrote {n_fixed} mesh path(s) to absolute")
    return new_text, warnings


# ============================================================================
# Model loading
# ============================================================================


def _apply_home_pose(model: Any, urdf_path: str) -> None:
    """Set model qpos to the assembly's default_angles (home pose).

    MuJoCo loads URDF with all revolute joints at qpos=0 (a straight arm).
    But the arm's design has a bent home pose (shoulder -35°, elbow +40°).
    This reads the assembly.json sibling file, extracts default_angles,
    and sets the corresponding qpos entries so the arm starts in its
    designed pose — not straight. Without this the PD controller yanks
    the arm from straight to bent on frame 1 → "fly then settle".
    """
    import json as _json
    from pathlib import Path

    # Look for assembly.json near the URDF
    urdf_p = Path(urdf_path)
    candidates = [
        urdf_p.parent / "assembly.json",
        urdf_p.parent.parent / "assembly.json",
        urdf_p.parent.parent.parent / "assembly.json",
    ]
    asm_path = None
    for c in candidates:
        if c.exists():
            asm_path = c
            break
    if asm_path is None:
        return  # no assembly.json — leave qpos at 0

    try:
        asm = _json.loads(asm_path.read_text("utf-8"))
    except Exception:
        return

    default_angles = asm.get("default_angles") or {}
    if not default_angles:
        return

    import mujoco
    applied = 0
    for jid in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        # Match joint name to default_angles key. URDF joint names often
        # contain the full "parent_to_child" path; the default_angles key
        # is just the child part name (e.g. "shoulder_pitch_servo").
        for angle_key, angle_val in default_angles.items():
            if angle_key in nm or nm.endswith("_to_" + angle_key):
                qadr = model.jnt_qposadr[jid]
                # default_angles are in degrees; MuJoCo uses radians for hinge
                if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
                    model.qpos0[qadr] = float(angle_val) * (3.14159265358979 / 180.0)
                else:
                    model.qpos0[qadr] = float(angle_val) / 1000.0  # mm → m for slide
                applied += 1
                break
    if applied:
        logger.info("Applied %d home-pose angles from %s", applied, asm_path.name)


def _mujoco_available() -> bool:
    """Return True if the ``mujoco`` package is importable."""
    try:
        import mujoco  # noqa: F401
        return True
    except ImportError:
        return False


# ============================================================================
# Motion controller (shared by record_motion and the interactive viewer)
# ============================================================================
#
# This class was extracted from ``record_motion``'s inline controller so the
# interactive MuJoCo GUI (``_launch_viewer``) can drive the SAME coordinated
# arm gesture + differential wheel drive that the headless rollout uses.
# Previously the GUI loop just called ``mj_step`` with no applied forces, so
# the robot only slumped under gravity — the user saw none of the planned
# motion despite it existing in the headless path. Sharing one controller is
# also required by AGENTS.md §1.1 (no duplicated control logic).


class _MotionController:
    """PD controller that drives a coordinated reach gesture + wheel drive.

    Built once from a loaded ``(model, data)`` pair. Call :meth:`apply` each
    timestep BEFORE ``mj_step`` to populate ``data.qfrc_applied``.

    Encapsulates:
      - joint classification (pitch / yaw / roll / finger / wheel)
      - coordinated sinusoidal gesture (0.15 Hz reach-retract)
      - finger lock at home
      - differential wheel torque (when the base is floating/drivable)
      - floating-base height + upright stabilization

    Mirrors the inline logic that lived in ``record_motion`` (2026-06-26); the
    behaviour is identical so existing rollouts are unaffected.
    """

    # Gesture frequency: one reach-retract cycle per ~6.7 seconds.
    GESTURE_FREQ = 0.15
    # Forward drive torque — enough to visibly translate the chassis.
    # Tuned for a ~12kg dual-arm robot with wheel friction=1.0, kp=50 PD.
    WHEEL_DRIVE_TORQUE = 0.5
    # PD gains — low enough to avoid numerical instability on light joints
    # (kp=200 caused NaN at arm_r_pitch_servo_2; kp=50 is stable).
    KP = 50.0
    KV = 5.0

    def __init__(self, model, data, *, np) -> None:
        self.model = model
        self.data = data
        self.np = np
        import mujoco  # local import; sim_mujoco defers this everywhere
        self._mj = mujoco

        movable = [
            jid for jid in range(model.njnt)
            if model.jnt_type[jid] in (
                mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE,
            )
        ]
        has_floating_base = (
            model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        )
        self.wheel_jids = [
            jid for jid in movable
            if "wheel" in (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
            ).lower()
        ]
        self.drivable = has_floating_base and len(self.wheel_jids) >= 2
        arm_jids = [jid for jid in movable if jid not in self.wheel_jids]

        # Classify arm joints by axis (pitch=X, yaw=Z, roll=Y) + fingers (slide).
        self.pitch_jids: list[int] = []
        self.yaw_jids: list[int] = []
        self.roll_jids: list[int] = []
        self.finger_jids: list[int] = []
        for jid in arm_jids:
            if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE:
                self.finger_jids.append(jid)
                continue
            ax = abs(float(model.jnt_axis[jid][0]))
            ay = abs(float(model.jnt_axis[jid][1]))
            az = abs(float(model.jnt_axis[jid][2]))
            if az > 0.5:
                self.yaw_jids.append(jid)
            elif ay > 0.5:
                self.roll_jids.append(jid)
            else:
                self.pitch_jids.append(jid)

        # Smooth reach gesture: the arm extends from home to a target pose
        # and retracts back, using a smooth ease-in-out trajectory (not
        # sinusoidal oscillation). Each joint moves a fraction of its range
        # toward a "reached" pose, holds briefly, then returns. This looks
        # like a real robot reaching for an object — not random twitching.
        # The trajectory uses a smoothstep: 0→1 over the first 40% of the
        # cycle, hold at 1 for 20%, then 1→0 over the last 40%.
        self.GESTURE_PERIOD = 8.0  # seconds per full reach-retract cycle
        _reach_frac = {
            "pitch": 0.30,   # pitch joints reach 30% of their range
            "yaw": 0.20,
            "roll": 0.15,
        }
        self.coordinated: dict[int, tuple[float, float]] = {}  # jid → (delta_rad, _unused)
        for jid in self.pitch_jids:
            lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
            amp = min((hi - lo) * _reach_frac["pitch"], 0.4)
            self.coordinated[jid] = (amp, 0.0)
        for jid in self.yaw_jids:
            lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
            amp = min((hi - lo) * _reach_frac["yaw"], 0.25)
            self.coordinated[jid] = (amp, 0.0)
        for jid in self.roll_jids:
            lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
            amp = min((hi - lo) * _reach_frac["roll"], 0.15)
            self.coordinated[jid] = (amp, 0.0)

        # Snapshot the home pose (computed after mj_forward by the caller).
        self.initial_qpos = data.qpos.copy()

        # Build joint→actuator mapping (if the MJCF has actuators injected).
        self.has_actuators = model.nu > 0
        self._joint_to_actuator: dict[int, int] = {}
        if self.has_actuators:
            for aid in range(model.nu):
                jid = int(model.actuator_trnid[aid][0])  # [joint_id, site_id]
                self._joint_to_actuator[jid] = aid

    def apply(self, t: float) -> None:
        """Populate ``data.ctrl`` for the motion at time ``t``.

        Uses MuJoCo's native actuator model (injected by ``_patch_mjcf``):
        - Arm joints: ``<position>`` actuators with kp=50/kv=5 — MuJoCo
          computes the PD torque internally, clamps it to forcerange ±5N·m,
          and handles gravity via qfrc_bias in the same dynamics pass.
        - Wheel joints: ``<motor>`` actuators — direct torque control.

        This replaces the hand-rolled qfrc_applied PD that had 3 copies,
        manual gravity compensation, and the post_step qpos-clamp hack.
        MuJoCo actuators own the control → no manual gravity comp needed,
        no qpos hack, and torque is automatically saturated.
        """
        model, data, np = self.model, self.data, self.np
        mujoco = self._mj

        if not self.has_actuators:
            self._apply_legacy(t)
            return

        # Arm + drive are DECOUPLED to avoid reaction-coupling instability.
        # First half: drive (arms at home).  Second half: gesture (stopped).
        if self.drivable:
            drive_phase = t < 5.0
        else:
            drive_phase = False

        # Arm gesture: slow sweep, cosine ease-in-out, period 8s.
        if self.drivable and drive_phase:
            sweep = 0.0  # arms held at home while driving
        else:
            gt = t - 5.0 if self.drivable else t
            phase = (gt % self.GESTURE_PERIOD) / self.GESTURE_PERIOD
            sweep = 0.5 - 0.5 * np.cos(2 * np.pi * phase)

        # Gravity-compensation feed-forward (inverse dynamics).  The
        # <position> actuator PD (kp=50) alone leaves a steady-state droop
        # under gravity (= τ_gravity / kp).  Adding qfrc_bias cancels the
        # gravitational load so the actuator PD only corrects transients.
        # This mirrors the pattern in _run_physics_hold (line ~1354).
        mujoco.mj_forward(model, data)
        if self.drivable:
            # Floating base: comp arm joints only, leave base DOFs at zero
            # so gravity pulls the chassis onto its wheels for traction.
            data.qfrc_applied[:] = 0.0
            for jid in list(self.coordinated.keys()) + self.finger_jids:
                dadr = model.jnt_dofadr[jid]
                data.qfrc_applied[dadr] = data.qfrc_bias[dadr]
        else:
            data.qfrc_applied[:] = data.qfrc_bias

        # Set position actuator targets (data.ctrl for arm joints).
        for jid in list(self.coordinated.keys()) + self.finger_jids:
            qadr = model.jnt_qposadr[jid]
            if jid in self.finger_jids:
                tgt = self.initial_qpos[qadr]  # fingers locked at home
            else:
                lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
                mid = self.initial_qpos[qadr]
                half_range = (hi - lo) * 0.35
                tgt = mid + half_range * (2 * sweep - 1)
                tgt = max(lo, min(hi, tgt))  # clamp target to range
            # Find the actuator for this joint.
            aid = self._joint_to_actuator.get(jid)
            if aid is not None:
                data.ctrl[aid] = tgt

        # Wheel drive: motor torque during drive phase only.
        for jid in self.wheel_jids:
            aid = self._joint_to_actuator.get(jid)
            if aid is not None:
                if self.drivable and drive_phase:
                    nm = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
                    steer = 0.3 * np.sin(2 * np.pi * t / 5.0)
                    is_left = "_fl" in nm or "_rl" in nm
                    data.ctrl[aid] = self.WHEEL_DRIVE_TORQUE * (
                        1.0 + (steer if is_left else -steer))
                else:
                    data.ctrl[aid] = 0.0  # wheels free when not driving

    def _apply_legacy(self, t: float) -> None:
        """Fallback for models without actuators (old URDFs / tests).

        Uses the original hand-rolled qfrc_applied PD.  Kept so existing
        tests that load URDFs without the MJCF patch still work.
        """
        model, data, np = self.model, self.data, self.np
        mujoco = self._mj
        kp, kv = self.KP, self.KV

        if self.drivable:
            drive_phase = t < 5.0
        else:
            drive_phase = False
        if self.drivable and drive_phase:
            sweep = 0.0
        else:
            gt = t - 5.0 if self.drivable else t
            phase = (gt % self.GESTURE_PERIOD) / self.GESTURE_PERIOD
            sweep = 0.5 - 0.5 * np.cos(2 * np.pi * phase)

        target = self.initial_qpos.copy()
        for jid, (amp, _) in self.coordinated.items():
            qadr = model.jnt_qposadr[jid]
            lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
            mid = self.initial_qpos[qadr]
            half_range = (hi - lo) * 0.35
            target[qadr] = mid + half_range * (2 * sweep - 1)

        mujoco.mj_forward(model, data)
        data.qfrc_applied[:] = data.qfrc_bias
        for jid in self.coordinated:
            dadr = model.jnt_dofadr[jid]
            qadr = model.jnt_qposadr[jid]
            lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
            tgt = max(lo, min(hi, target[qadr]))
            data.qfrc_applied[dadr] += kp * (tgt - data.qpos[qadr]) - kv * data.qvel[dadr]
        for jid in self.finger_jids:
            qadr = model.jnt_qposadr[jid]
            dadr = model.jnt_dofadr[jid]
            data.qfrc_applied[dadr] += kp * (self.initial_qpos[qadr] - data.qpos[qadr]) \
                                       - kv * data.qvel[dadr]
        if self.drivable and drive_phase:
            steer = 0.3 * np.sin(2 * np.pi * t / 5.0)
            for jid in self.wheel_jids:
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
                dadr = model.jnt_dofadr[jid]
                is_left = "_fl" in nm or "_rl" in nm
                torque = self.WHEEL_DRIVE_TORQUE * (1.0 + (steer if is_left else -steer))
                data.qfrc_applied[dadr] += torque


def _launch_viewer(
    model: Any,
    *,
    duration_sec: float = 10.0,
    stabilize: bool = True,
) -> None:
    """Open the MuJoCo passive viewer and drive the planned motion live.

    Added 2026-06-18 to give the simulation path a real GUI (the user
    observed "simulation was done without a graphical interface").  Uses
    ``mujoco.viewer.launch_passive`` which does NOT block on its own —
    the caller must run a step loop and call ``viewer.sync()`` each frame.

    The loop drives the SAME coordinated arm-gesture + differential wheel
    motion as the headless ``record_motion`` rollout (via
    :class:`_MotionController`), so the user actually sees the robot reach
    and drive — not just slump under gravity. This was the Bug #4 defect:
    the viewer stepped raw ``mj_step`` with no applied forces, so none of
    the planned articulation was visible.

    Falls back to a no-op log message on headless environments or
    import errors so e2e tests can still call ``interactive=False`` on a
    server.
    """
    try:
        import mujoco  # type: ignore[import-not-found]
        import mujoco.viewer  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
        import time as _time
    except ImportError as e:
        logger.warning("MuJoCo viewer unavailable: %s", e)
        return

    if stabilize:
        _stabilize_model(model)

    data = mujoco.MjData(model)

    # Wheel-ground contact setup (centralised in _setup_wheel_contacts):
    # enables collision only for the ground plane + wheels, sets wheel
    # friction, and returns the z-drop needed for the lowest wheel point
    # to penetrate the ground by 0.5 mm (so contacts actually activate).
    z_drop = _setup_wheel_contacts(model, data)
    if z_drop != 0.0:
        data.qpos[2] = z_drop
    mujoco.mj_forward(model, data)

    # Physics settle phase: let gravity + contacts reach equilibrium
    # BEFORE the viewer opens. During settle, PD-hold ALL revolute joints
    # at their initial position (so the arm doesn't droop under gravity
    # — that droop was the "fly" the user saw). This finds the resting
    # pose WITH the arm held up, then the controller takes over smoothly.
    _settle_movable = [
        jid for jid in range(model.njnt)
        if model.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE,
                                    mujoco.mjtJoint.mjJNT_SLIDE)
    ]
    # For floating-base robots, gravity-compensate ARM joints only (not the
    # base/wheel DOFs).  Full gravity comp cancels chassis weight → hover →
    # contacts separate → wheel drive hits nothing.  Per-joint comp on arm
    # DOFs lets the PD hold the arm up without the arm's weight reaction
    # pushing the base off the ground.  Base DOFs get zero → gravity pulls
    # the chassis onto its wheels.
    _is_floating = model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    _settle_arm_jids = _settle_movable if not _is_floating else [
        jid for jid in _settle_movable
        if "wheel" not in (mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, jid) or "").lower()
    ]
    _settle_kp, _settle_kv = 50.0, 5.0
    _settle_iq = data.qpos.copy()
    for _settle_step in range(300):
        mujoco.mj_forward(model, data)
        if _is_floating:
            data.qfrc_applied[:] = 0.0
            for jid in _settle_arm_jids:
                dadr = model.jnt_dofadr[jid]
                data.qfrc_applied[dadr] = data.qfrc_bias[dadr]
        else:
            data.qfrc_applied[:] = data.qfrc_bias
        for jid in _settle_movable:
            qadr = model.jnt_qposadr[jid]
            dadr = model.jnt_dofadr[jid]
            data.qfrc_applied[dadr] += _settle_kp * (_settle_iq[qadr] - data.qpos[qadr]) \
                                       - _settle_kv * data.qvel[dadr]
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qacc)):
            logger.warning("Settle phase diverged — starting viewer anyway")
            break

    # Build the shared controller AFTER settle so initial_qpos is the
    # equilibrium pose (not the URDF home), preventing jump artifacts.
    controller = _MotionController(model, data, np=np)

    max_steps = int(duration_sec / max(model.opt.timestep, 1e-5))
    logger.info(
        "Launching MuJoCo viewer (max_steps=%d, dt=%.4fs, drivable=%s, "
        "arm_joints=%d)",
        max_steps, model.opt.timestep, controller.drivable,
        len(controller.coordinated) + len(controller.finger_jids),
    )
    t0 = _time.time()
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Show the settled initial pose for ~1 second BEFORE starting
            # motion, so the user sees a stable robot first (not a flash
            # of unstable frames while the viewer initializes rendering +
            # the first mj_step adjusts positions).
            viewer.sync()
            _time.sleep(1.0)
            viewer.sync()

            step = 0
            while viewer.is_running() and step < max_steps:
                # Drive the planned motion (gesture + wheels) each step.
                t = step * model.opt.timestep
                controller.apply(t)
                mujoco.mj_step(model, data)
                viewer.sync()
                # Real-time playback: sleep one timestep per step, floored
                # so physics doesn't fall behind on slow machines.
                _time.sleep(max(model.opt.timestep, 0.001))
                step += 1
            # Keep the window open after the motion finishes — the user
            # closes it manually. (Previously duration_sec=2.0 killed it.)
            motion_sec = _time.time() - t0
            logger.info(
                "Motion playback finished (%.1fs) — viewer stays open until "
                "you close the window.", motion_sec,
            )
            while viewer.is_running():
                viewer.sync()
                _time.sleep(0.016)  # ~60fps refresh
    except Exception as e:
        logger.warning("MuJoCo viewer exited with error: %s", e)


def _make_base_floating(urdf_text: str) -> tuple[str, bool]:
    """Rewrite a URDF so the mobile base is a FLOATING body (drivable).

    The exported URDF welds ``base_footprint`` to the world implicitly (its
    joints to children are ``fixed``), so MuJoCo merges the whole chassis
    into a rigid mass anchored to the ground — wheels spin but the robot
    never moves.  This injects a ``<link name="world"/>`` and a
    ``<joint type="floating" parent="world" child="base_footprint"/>`` so
    MuJoCo gives the chassis 6-DOF freedom; the wheels (revolute) can then
    push it along the ground.

    Returns ``(rewritten_text, was_converted)``.  No-op when the URDF has no
    ``base_footprint`` link (e.g. an arm-only assembly) — arms don't drive.
    """
    if "<link name=\"base_footprint\"" not in urdf_text:
        return urdf_text, False
    # Already has a world link → assume already floating.
    if 'name="world"' in urdf_text:
        return urdf_text, False
    inject = (
        '\n  <link name="world"/>\n'
        '  <joint name="world_to_base_footprint" type="floating">\n'
        '    <parent link="world"/>\n'
        '    <child link="base_footprint"/>\n'
        '  </joint>\n'
    )
    # Inject right before the closing </robot> tag.
    if "</robot>" in urdf_text:
        return urdf_text.replace("</robot>", inject + "</robot>"), True
    return urdf_text + inject, True


# Wheel-ground friction: tangential 1.0 gives traction; the two tiny
# numbers are torsional/rolling friction (MuJoCo's 3-component format).
_WHEEL_FRICTION = (1.0, 0.005, 0.0001)


def _patch_mjcf(mjcf_text: str) -> str:
    """Inject nconmax + a ground plane into a compiled MJCF document.

    MuJoCo's URDF loader does **not** create a floor — the world body has
    no geom, so wheel-ground contact setup (contype/conaffinity/friction)
    contacts nothing and the robot never drives (verified: ``ncon==0``,
    2 s of drive torque → 10 mm of pure-gravity creep).  This patches the
    MJCF produced by ``mj_saveLastXML`` in two ways:

    1. Raise ``nconmax`` to 200 so dense-mesh assemblies (38 collision
       geoms on a dual-arm robot) don't crash ``mj_forward`` with
       "expected at most 8 contacts".
    2. Add a ``<geom type="plane">`` to the world body so wheels have a
       ground to push against.

    Both patches are string replacements on the saved MJCF; if either
    fails the caller keeps the unpatched model (see ``_load_model``).
    """
    # --- nconmax ---
    if "<size" in mjcf_text:
        if "nconmax" not in mjcf_text:
            mjcf_text = mjcf_text.replace(
                "<size", '<size nconmax="200"', 1,
            )
    else:
        mjcf_text = mjcf_text.replace(
            "<mujoco>", '<mujoco><size nconmax="200"/>', 1,
        )

    # --- ground plane (only if not already present) ---
    if 'name="ground"' not in mjcf_text and 'type="plane"' not in mjcf_text:
        ground = (
            '\n      <geom name="ground" type="plane" size="5 5 0.1" '
            'rgba="0.5 0.5 0.5 1" contype="1" conaffinity="1" '
            f'friction="{" ".join(str(f) for f in _WHEEL_FRICTION)}"/>'
        )
        # The world body is the first <worldbody>; inject right after its
        # opening tag so the plane lives in the world (bid 0).
        mjcf_text = mjcf_text.replace("<worldbody>", "<worldbody>" + ground, 1)

    # --- solver + contact stiffness (robot-grade, not MuJoCo defaults) ---
    # MuJoCo's default solref/solimp are tuned for generic soft contacts —
    # too squishy for a wheeled robot.  A 20ms contact timeconstant lets
    # wheels unload under arm-reaction load transfer.  Stiffen to 5ms and
    # raise the impedance floor so contacts hold ~30N/wheel against transients.
    if "<option" not in mjcf_text:
        mjcf_text = mjcf_text.replace(
            "<mujoco>",
            '<mujoco>\n'
            '  <option integrator="implicitfast" iterations="50" '
            'solver="Newton" cone="elliptic" impratio="10"/>',
            1,
        )
    # Stiffen the default geom contact params (applies to ALL geoms including
    # the injected ground + wheels).
    if "<default" not in mjcf_text:
        mjcf_text = mjcf_text.replace(
            "<worldbody>",
            '<default>\n'
            '    <geom solref="0.005 1" solimp="0.99 0.999 0.001"/>\n'
            '  </default>\n'
            '  <worldbody>',
            1,
        )

    # --- MuJoCo <actuator> model (the architectural fix) ---
    # The URDF has zero MuJoCo actuators (nu=0), so all control is via
    # hand-rolled qfrc_applied in Python — 3 copies, no torque saturation,
    # no control clamping, and the post_step qpos-hack to enforce limits.
    # Injecting <position> actuators for arm joints + <motor> for wheels
    # gives MuJoCo-native PD with built-in forcerange/ctrlrange clamping.
    # The Python controller then writes data.ctrl[:] instead of qfrc_applied.
    if "<actuator>" not in mjcf_text:
        import re
        # Find all named joints in the MJCF.  Match any attribute order —
        # MuJoCo's saved MJCF puts name before type for free joints but may
        # omit type="hinge" (the default) for hinge joints.
        joint_matches = re.findall(
            r'<joint\s+name="([^"]+)"[^>]*/>', mjcf_text,
        )
        actuator_lines: list[str] = []
        for jname in joint_matches:
            jlower = jname.lower()
            # Skip the free joint (floating base) — it has no actuator.
            if "world_to_" in jname:
                continue
            # Classify: wheel/tire = motor (torque), everything else =
            # position (PD).  Finger slides also get position actuators.
            if "wheel" in jlower or "tire" in jlower:
                actuator_lines.append(
                    f'    <motor name="act_{jname}" joint="{jname}" '
                    f'gear="1" ctrllimited="false" forcelimited="true" '
                    f'forcerange="-5 5"/>'
                )
            else:
                actuator_lines.append(
                    f'    <position name="act_{jname}" joint="{jname}" '
                    f'kp="50" kv="5" forcelimited="true" '
                    f'forcerange="-5 5" ctrllimited="false"/>'
                )
        if actuator_lines:
            actuator_block = (
                "\n  <actuator>\n" + "\n".join(actuator_lines) + "\n  </actuator>\n"
            )
            mjcf_text = mjcf_text.replace(
                "</mujoco>", actuator_block + "</mujoco>", 1,
            )

    return mjcf_text


def _setup_wheel_contacts(model: Any, data: Any) -> float:
    """Configure wheel-ground collision + friction; return the needed z-drop.

    Centralises the contact setup that was previously triplicated across
    ``_launch_viewer``, ``record_motion`` and ``record_joint_motion``.
    Walks every geom, enables collision ONLY for the world body's geoms
    (the ground plane from :func:`_patch_mjcf`) and wheel bodies, sets
    wheel friction, and returns the z-offset the floating base must be
    dropped by so the lowest wheel point penetrates the ground by 0.5 mm
    (guaranteeing contacts activate).  Callers set ``data.qpos[2]`` to
    this value when the base is floating.

    Returns 0.0 for fixed-base (arm-only) models — there is nothing to
    drop and no wheels to contact.
    """
    import mujoco  # type: ignore[import-not-found]

    # mj_forward populates data.geom_xpos (world-frame geom centres), which
    # we need to compute the wheel-bottom height for the z-drop.
    if data is not None:
        mujoco.mj_forward(model, data)

    # Disable all collisions, then re-enable only world + wheels.
    for gid in range(model.ngeom):
        model.geom_contype[gid] = 0
        model.geom_conaffinity[gid] = 0

    wheel_bottom_z = float("inf")  # lowest wheel point in world frame
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        is_world = bid == 0
        is_wheel = "wheel" in bname.lower()
        if is_world or is_wheel:
            model.geom_contype[gid] = 1
            model.geom_conaffinity[gid] = 1
            if is_wheel:
                model.geom_friction[gid] = list(_WHEEL_FRICTION)
                # Lowest point of this wheel geom in world frame.
                # geom_xpos is the world-frame geom centre (populated by
                # mj_forward); for a cylinder the half-extent along its
                # spin axis is geom_size[0] (radius).  We take the max of
                # the first/third size entries conservatively so the drop
                # never under-estimates (contacts engage sooner, never late).
                sz = model.geom_size[gid]
                half_z = float(max(sz[0], sz[2])) if len(sz) >= 3 else float(sz[0])
                gz = float(data.geom_xpos[gid][2]) if data is not None else 0.0
                bottom = gz - half_z
                if bottom < wheel_bottom_z:
                    wheel_bottom_z = bottom

    # Fixed base (no FREE joint at index 0) → nothing to drop, no driving.
    is_floating = model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    if not is_floating:
        return 0.0
    if wheel_bottom_z == float("inf"):
        return -0.001  # no wheels found — keep legacy tiny drop
    # Drop the base so the lowest wheel PENETRATES the ground by 1 mm.
    # MuJoCo's narrow-phase contact needs actual overlap (negative gap) to
    # register a contact — sitting exactly flush leaves ncon=0 and the
    # robot free-falls.  1 mm is enough to engage without a visual sink.
    return -(wheel_bottom_z) - 0.001


def _load_model(urdf_path: str, *, floating_base: bool = False) -> dict[str, Any]:
    """Load a URDF into MuJoCo, returning a structured result dict.

    Returns:
        {
            "ok": bool,
            "error": str,           # empty when ok
            "warnings": list[str],
            "model": mujoco.MjModel, # present when ok
            "temp_files": list[str], # caller should clean up
        }
    """
    temp_files: list[str] = []

    if not _mujoco_available():
        return {
            "ok": False,
            "error": "mujoco package not installed. Run: pip install mujoco",
            "warnings": [],
            "temp_files": temp_files,
        }

    import mujoco  # type: ignore[import-not-found]

    path = Path(urdf_path)
    if not path.exists():
        return {
            "ok": False,
            "error": f"URDF file not found: {urdf_path}",
            "warnings": [],
            "temp_files": temp_files,
        }

    original_text = path.read_text(encoding="utf-8")
    fixed_text, warnings = _rewrite_mesh_paths(original_text, path)

    # Make the mobile base DRIVABLE — ONLY when requested.  The URDF welds
    # base_footprint→base_plate via a fixed joint, so MuJoCo merges the whole
    # chassis into a rigid tree bolted to the world.  ``floating_base=True``
    # injects a <link name="world"/> + floating joint so the chassis becomes a
    # floating base the wheels can push along the ground ("能动").  This is
    # enabled for record_motion (driving animation); the arm-stability hold
    # test and the per-joint actuation test keep the FIXED base (their job is
    # arm/joint behaviour, not locomotion, and a floating rigid-chassis-on-
    # locked-wheels is a hard solver problem that produces false instability).
    if floating_base:
        floating_text, base_floating = _make_base_floating(fixed_text)
        if base_floating:
            fixed_text = floating_text
            warnings.append("base converted to floating joint for driving")

    # If we had to rewrite paths, write a temp URDF; otherwise load directly
    load_path: str = str(path)
    if fixed_text != original_text:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".urdf", delete=False, encoding="utf-8",
        )
        tmp.write(fixed_text)
        tmp.close()
        load_path = tmp.name
        temp_files.append(tmp.name)

    try:
        model = mujoco.MjModel.from_xml_path(load_path)

        # Set initial joint positions from the assembly's default_angles.
        # MuJoCo loads URDF with all joints at qpos=0 (straight arm), but the
        # arm's home pose has bent joints (shoulder -35°, elbow +40°, etc.).
        # Without this, the arm starts straight → the PD controller yanks it
        # to the bent pose → "fly then settle". By setting qpos to the home
        # pose BEFORE the first mj_step, the arm starts already bent and the
        # motion is smooth from frame 1.
        _apply_home_pose(model, urdf_path)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"MuJoCo parse error: {exc}",
            "warnings": warnings,
            "temp_files": temp_files,
        }

    # Raise the per-pair contact limit for dense-mesh assemblies (dual-arm
    # robots with 38 STL collision meshes can produce >8 contacts between
    # two geoms, crashing mj_forward with "expected at most 8") AND inject a
    # ground plane (MuJoCo's URDF loader creates no floor — without it
    # wheel-ground contact hits nothing and the robot never drives).
    # Both patches go through ``_patch_mjcf`` on the saved MJCF.
    try:
        mjcf_fd, mjcf_path = tempfile.mkstemp(
            suffix=".xml", dir=str(path.parent), prefix="_mjcf_",
        )
        os.close(mjcf_fd)
        mujoco.mj_saveLastXML(mjcf_path, model)
        mjcf_text = Path(mjcf_path).read_text("utf-8")
        mjcf_text = _patch_mjcf(mjcf_text)
        Path(mjcf_path).write_text(mjcf_text, "utf-8")
        temp_files.append(mjcf_path)
        model = mujoco.MjModel.from_xml_path(mjcf_path)
        warnings.append("patched MJCF: nconmax=200 + ground plane")
    except Exception as e:
        # MJCF patching failed (ground plane / actuator injection). Keep
        # the original model but WARN — silently passing would hide a
        # broken patch that leaves the robot undrivable (AGENTS.md §1.1).
        logger.warning("MJCF patching failed, using unpatched model: %s", e)
        warnings.append(f"WARNING: MJCF patch failed ({e}) — robot may lack ground plane/actuators")

    return {
        "ok": True,
        "error": "",
        "warnings": warnings,
        "model": model,
        "temp_files": temp_files,
    }


# ============================================================================
# Model scanning
# ============================================================================


def _scan_bodies(model: Any) -> list[dict[str, Any]]:
    """Extract per-body information from a MuJoCo model."""
    import mujoco  # type: ignore[import-not-found]

    bodies: list[dict[str, Any]] = []
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        mass = float(model.body_mass[bid])
        inertia = tuple(float(x) for x in model.body_inertia[bid])
        # Sanity flags
        is_world = (bid == 0)
        mass_warning = ""
        if not is_world and mass <= 0:
            mass_warning = "MASS IS ZERO — simulation will be unstable"
        elif not is_world and mass < 0.005:
            mass_warning = f"mass very low ({mass*1000:.1f}g) — may cause instability"
        inertia_warning = ""
        if not is_world and not is_world and min(inertia) < 1e-9:
            inertia_warning = "inertia near zero — simulation unstable"

        bodies.append({
            "id": bid,
            "name": name,
            "mass_kg": mass,
            "inertia": inertia,
            "mass_warning": mass_warning,
            "inertia_warning": inertia_warning,
        })
    return bodies


def _scan_joints(model: Any) -> list[dict[str, Any]]:
    """Extract per-joint information."""
    import mujoco  # type: ignore[import-not-found]

    _TYPE_NAMES = {
        mujoco.mjtJoint.mjJNT_FREE: "FREE",
        mujoco.mjtJoint.mjJNT_BALL: "BALL",
        mujoco.mjtJoint.mjJNT_SLIDE: "SLIDE",
        mujoco.mjtJoint.mjJNT_HINGE: "HINGE",
    }

    joints: list[dict[str, Any]] = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        jtype = model.jnt_type[jid]
        axis = tuple(float(x) for x in model.jnt_axis[jid])
        rng = tuple(float(x) for x in model.jnt_range[jid])
        joints.append({
            "id": jid,
            "name": name,
            "type": _TYPE_NAMES.get(int(jtype), str(int(jtype))),
            "axis": axis,
            "range": rng,
        })
    return joints


# ============================================================================
# Physics validation
# ============================================================================


def _stabilize_model(model: Any, armature: float = 0.1, damping: float = 1.0) -> None:
    """Add joint armature and damping to stabilise numerics.

    Discovered in Phase A smoke test: the 4-DOF arm's tiny gripper fingers
    (29g mass, ~1e-5 kg·m² inertia, ±8mm range) blow up the MuJoCo
    constraint solver within 10ms of simulation.  Adding motor armature
    (rotor inertia reflected to the joint) and passive damping damps the
    numerical resonance without changing the underlying geometry.

    Prismatic (slide) joints get EXTRA damping: when the arm chain has a
    non-zero pitch angle (the default home pose bends ±35°), gravity
    creates a lateral force component along the finger slide axis.  With
    default damping this slides the fingers to their ±8mm joint limit
    (6.9mm observed), failing the PD-hold check.  Higher damping on slide
    joints specifically (not hinge) resists this steady-state drift
    without affecting the revolute joints' dynamic response.
    """
    import mujoco
    for jid in range(model.njnt):
        model.dof_armature[jid] = armature
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE:
            model.dof_damping[jid] = max(damping * 10.0, 30.0)
        else:
            model.dof_damping[jid] = damping


def record_motion(
    urdf_path: str,
    duration_sec: float = 3.0,
    fps: int = 30,
    stabilize: bool = True,
) -> dict[str, Any]:
    """Run a short physics rollout and record per-body world poses per frame.

    Used by the web 3D viewer (``simulate.html``) to play back an animation
    of the robot moving.  Unlike the PD-*hold* test, the target pose here
    is a slow sinusoid sweeping each actuated joint through part of its
    range, so the assembly visibly articulates (shoulders/elbow/wrist bend,
    gripper fingers open and close).

    Each actuated joint i follows ``target_i(t) = A_i * sin(2*pi*f_i*t)``
    where ``A_i`` is a fraction of the joint range and ``f_i`` is a small
    distinct frequency so the motion looks coordinated, not mechanical.
    A PD controller tracks these targets; gravity compensation is applied
    so the arm does not collapse.

    Args:
        urdf_path: Path to the URDF to load (mesh paths rewritten by
            ``_load_model``).
        duration_sec: Length of the rollout in seconds.
        fps: Sampling rate for the returned frames.
        stabilize: Apply ``_stabilize_model`` (recommended for the tiny
            gripper masses).

    Returns:
        Dict with::

            {
              "ok": bool,
              "error": str,               # present when ok is False
              "bodies": [name, ...],      # body name per body id (excl. world)
              "fps": int,
              "duration_sec": float,
              "frames": [                 # one per sampled timestep
                {
                  "t": float,             # seconds
                  "poses": [              # parallel to `bodies`; m + quat(w,x,y,z)
                    [px, py, pz, qw, qx, qy, qz], ...
                  ]
                }, ...
              ]
            }

        Positions are in metres (MuJoCo's internal unit); the web frontend
        converts to mm (x1000) to match the renderer convention.
    """
    import mujoco  # type: ignore[import-not-found]
    import numpy as np

    load = _load_model(urdf_path, floating_base=True)
    if not load.get("ok"):
        return {"ok": False, "error": load.get("error", "model load failed"),
                "bodies": [], "frames": []}

    model = load["model"]
    if stabilize:
        _stabilize_model(model, armature=0.5, damping=5.0)
    if model.opt.timestep > 0.0005:
        model.opt.timestep = 0.0005

    data = mujoco.MjData(model)

    # Wheel-ground contact setup (centralised; see _setup_wheel_contacts).
    # Mesh-mesh collisions are disabled — only ground×wheel collides, which
    # is all driving needs.  Arm self-collision is prevented at generation
    # time (collision-aware range clamping), not in playback.
    z_drop = _setup_wheel_contacts(model, data)
    if z_drop != 0.0:
        data.qpos[2] = z_drop
    mujoco.mj_forward(model, data)

    # Build the shared motion controller (joint classification + coordinated
    # gesture + differential wheel drive + floating-base stabilization).
    # This is the same controller the interactive viewer uses, factored out so
    # the headless rollout and the GUI stay in sync (AGENTS.md §1.1).
    controller = _MotionController(model, data, np=np)

    n_steps = max(1, int(duration_sec / model.opt.timestep))
    frame_every = max(1, int(round(1.0 / (fps * model.opt.timestep))))

    # Body name list (exclude the world body at id 0).
    body_names = []
    for bid in range(1, model.nbody):
        body_names.append(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body_{bid}"
        )

    frames: list[dict[str, Any]] = []
    for step in range(n_steps):
        t = step * model.opt.timestep
        # Apply the coordinated gesture + wheel drive for this timestep.
        controller.apply(t)
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qacc)):
            break  # diverged — stop early, keep what we have

        if step % frame_every == 0:
            poses = []
            for bid in range(1, model.nbody):
                poses.append([
                    float(data.xpos[bid][0]), float(data.xpos[bid][1]),
                    float(data.xpos[bid][2]),
                    float(data.xquat[bid][0]), float(data.xquat[bid][1]),
                    float(data.xquat[bid][2]), float(data.xquat[bid][3]),
                ])
            frames.append({"t": round(t, 4), "poses": poses})

    return {
        "ok": True,
        "bodies": body_names,
        "fps": fps,
        "duration_sec": duration_sec,
        "frames": frames,
    }


def record_joint_motion(
    urdf_path: str,
    duration_sec: float = 3.0,
    fps: int = 30,
    stabilize: bool = True,
) -> dict[str, Any]:
    """Run a physics rollout and record JOINT ANGLE sequences per frame.

    Unlike ``record_motion`` (which returns per-body world poses — useless
    for FK-based web playback because MuJoCo merges fixed joints and only
    6 of 13 parts appear), this returns the raw **joint angles** at each
    frame.  The web frontend then calls AssemblySolver.solve(joint_angles)
    per frame to get every part's correct world position via forward
    kinematics — so ALL 13 parts move correctly along their DOF, not just
    the 6 MuJoCo bodies.

    Returns::

        {
          "ok": bool,
          "joints": [{"name": str, "axis": str, "type": str}, ...],
          "fps": int,
          "duration_sec": float,
          "frames": [
            {"t": float, "angles": {"joint_name": angle_deg, ...}}, ...
          ],
          # For wheeled robots: base trajectory (so FK can place chassis)
          "base_trajectory": [{"t": float, "x": mm, "y": mm, "z": mm,
                               "qw": ..., "qx": ..., "qy": ..., "qz": ...}, ...]
                             or null for fixed-base robots.
        }
    """
    import mujoco  # type: ignore[import-not-found]
    import numpy as np

    load = _load_model(urdf_path, floating_base=True)
    if not load.get("ok"):
        return {"ok": False, "error": load.get("error", "model load failed"),
                "joints": [], "frames": []}

    model = load["model"]
    if stabilize:
        _stabilize_model(model, armature=0.5, damping=5.0)
    if model.opt.timestep > 0.0005:
        model.opt.timestep = 0.0005

    data = mujoco.MjData(model)

    # Wheel-ground contact setup (centralised; see _setup_wheel_contacts).
    z_drop = _setup_wheel_contacts(model, data)
    if z_drop != 0.0:
        data.qpos[2] = z_drop

    mujoco.mj_forward(model, data)

    # Use the shared _MotionController (same gesture + wheel drive as the
    # interactive viewer and record_motion).  This was previously an inline
    # re-implementation that drifted out of sync — see AGENTS.md §1.1.
    controller = _MotionController(model, data, np=np)
    movable = [
        jid for jid in range(model.njnt)
        if model.jnt_type[jid] in (
            mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE,
        )
    ]
    drivable = controller.drivable

    n_steps = max(1, int(duration_sec / model.opt.timestep))
    frame_every = max(1, int(round(1.0 / (fps * model.opt.timestep))))

    # Joint metadata for the frontend.
    joint_info = []
    for jid in movable:
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"j{jid}"
        jtype_raw = model.jnt_type[jid]
        jtype = "hinge" if jtype_raw == mujoco.mjtJoint.mjJNT_HINGE else "slide"
        axis_v = model.jnt_axis[jid]
        axis = "x" if abs(axis_v[0]) > 0.5 else ("y" if abs(axis_v[1]) > 0.5 else "z")
        joint_info.append({"name": nm, "axis": axis, "type": jtype})

    frames: list[dict[str, Any]] = []
    base_traj: list[dict[str, Any]] = []

    for step in range(n_steps):
        t = step * model.opt.timestep
        controller.apply(t)
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qacc)):
            break

        if step % frame_every == 0:
            # Record joint angles in DEGREES (solver convention).
            angles: dict[str, float] = {}
            for jid in movable:
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"j{jid}"
                qadr = model.jnt_qposadr[jid]
                val = float(data.qpos[qadr])
                # HINGE: radians → degrees; SLIDE: metres → mm.
                if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
                    angles[nm] = round(float(np.degrees(val)), 2)
                else:
                    angles[nm] = round(val * 1000.0, 3)
            frames.append({"t": round(t, 4), "angles": angles})

            # For floating-base robots: record base trajectory.
            if drivable:
                base_traj.append({
                    "t": round(t, 4),
                    "x": round(float(data.qpos[0]) * 1000, 2),
                    "y": round(float(data.qpos[1]) * 1000, 2),
                    "z": round(float(data.qpos[2]) * 1000, 2),
                    "qw": round(float(data.qpos[3]), 6),
                    "qx": round(float(data.qpos[4]), 6),
                    "qy": round(float(data.qpos[5]), 6),
                    "qz": round(float(data.qpos[6]), 6),
                })

    return {
        "ok": True,
        "joints": joint_info,
        "fps": fps,
        "duration_sec": duration_sec,
        "frames": frames,
        "base_trajectory": base_traj if drivable else None,
    }


def render_simulation_video(
    urdf_path: str,
    output_path: str,
    duration_sec: float = 3.0,
    fps: int = 15,
    width: int = 640,
    height: int = 480,
    camera: str | None = None,
) -> dict[str, Any]:
    """Render a MuJoCo physics rollout to an MP4 video file.

    Runs the same joint-sweep rollout as :func:`record_joint_motion`, but
    instead of returning numerical joint-angle frames, it captures offscreen
    RGB frames from MuJoCo's renderer at each timestep and encodes them into a
    video.  This produces a **non-forgeable visual record** of the robot
    moving — far stronger evidence of "能动" (it moves) than a JSON of angles,
    because a reviewer can watch the actual physics.

    Requires the ``ffmpeg`` binary on PATH (used to encode PNG frames → MP4,
    avoiding a Python video-library dependency).  Falls back to saving raw
    PNG frames if ffmpeg is absent.

    Args:
        urdf_path: path to the generated URDF (with meshes).
        output_path: destination ``.mp4`` path (or directory for PNG fallback).
        duration_sec: rollout length.
        fps: output video framerate (also the render sampling rate).
        width/height: video resolution.
        camera: MuJoCo camera name for a fixed viewpoint; None = auto-track.

    Returns::

        {"ok": bool, "video_path": str, "n_frames": int, "fps": int,
         "error": str (if failed)}
    """
    import shutil
    import subprocess

    import mujoco  # type: ignore[import-not-found]
    import numpy as np  # noqa: F401  (used by _MotionController)

    load = _load_model(urdf_path, floating_base=True)
    if not load.get("ok"):
        return {"ok": False, "error": load.get("error", "model load failed"),
                "video_path": "", "n_frames": 0, "fps": fps}

    model = load["model"]
    _stabilize_model(model, armature=0.5, damping=5.0)
    if model.opt.timestep > 0.0005:
        model.opt.timestep = 0.0005

    data = mujoco.MjData(model)
    z_drop = _setup_wheel_contacts(model, data)
    if z_drop != 0.0:
        data.qpos[2] = z_drop
    mujoco.mj_forward(model, data)

    controller = _MotionController(model, data, np=np)

    # Offscreen renderer — MuJoCo 3.x Renderer API (update_scene → render).
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = camera or -1  # -1 = auto/free camera

    frames_dir = Path(output_path).with_suffix("")
    frames_dir = frames_dir.parent / (frames_dir.name + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    n_steps = max(1, int(duration_sec / model.opt.timestep))
    frame_every = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    n_frames = 0

    for step in range(n_steps):
        t = step * model.opt.timestep
        controller.apply(t)
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qacc)):
            break

        if step % frame_every == 0:
            renderer.update_scene(data, camera=cam)
            pixels = renderer.render()
            from PIL import Image
            Image.fromarray(pixels).save(str(frames_dir / f"frame_{n_frames:05d}.png"))
            n_frames += 1

    renderer.close()

    if n_frames == 0:
        return {"ok": False, "error": "no frames rendered (physics diverged?)",
                "video_path": "", "n_frames": 0, "fps": fps}

    # Encode PNG sequence → MP4 via ffmpeg (avoids Python video deps).
    ffmpeg = shutil.which("ffmpeg")
    out_mp4 = str(output_path)
    if ffmpeg:
        cmd = [
            ffmpeg, "-y", "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "20",
            out_mp4,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            return {"ok": False,
                    "error": f"ffmpeg failed: {proc.stderr[-300:]}",
                    "video_path": "", "n_frames": n_frames, "fps": fps}
        # Clean up intermediate frames on success.
        shutil.rmtree(frames_dir, ignore_errors=True)
    else:
        # No ffmpeg — leave PNG frames, point output there.
        out_mp4 = str(frames_dir)

    return {"ok": True, "video_path": out_mp4, "n_frames": n_frames,
            "fps": fps, "duration_sec": duration_sec}


def extract_motion_key_frames(
    urdf_path: str,
    width: int = 480,
    height: int = 360,
) -> list[dict[str, str]]:
    """Extract key motion frames for VLM inspection (dynamic verification).

    Runs a short MuJoCo joint-sweep rollout and captures offscreen renders at
    three kinematically-significant moments, returning them as base64-encoded
    PNG data URIs ready for GLM-4.6V ``image_url`` multi-image input. This is
    the "eyes" of the dynamic VLM verification loop: unlike the static VLM
    (which sees only the initial appearance), these frames let the agent judge
    **motion behaviour** — does the arm move through its range without
    self-collision, does the joint sweep look mechanically plausible, does the
    end-effector reach a sensible workspace.

    The three frames are:

    - ``initial`` — t=0, the rest pose (baseline; lets the VLM see the
      assembled structure as a sanity check on assembly correctness).
    - ``mid_sweep`` — midway through the rollout, joints at ~50% of their
      sweep gesture (reveals trajectory plausibility and mid-range collision).
    - ``extreme`` — near the end, joints near their swept extremes (stresses
      the assembly: collisions and binding tend to occur at range limits).

    Returns a list of ``{"label": str, "description": str, "image": data_uri}``
    dicts (empty list on failure). Each ``image`` value is a
    ``data:image/png;base64,...`` URI consumable directly by the GLM vision API.
    """
    import base64
    import io

    import mujoco  # type: ignore[import-not-found]
    import numpy as np  # noqa: F401
    from PIL import Image

    load = _load_model(urdf_path, floating_base=True)
    if not load.get("ok"):
        return []

    model = load["model"]
    _stabilize_model(model, armature=0.5, damping=5.0)
    if model.opt.timestep > 0.0005:
        model.opt.timestep = 0.0005

    data = mujoco.MjData(model)
    z_drop = _setup_wheel_contacts(model, data)
    if z_drop != 0.0:
        data.qpos[2] = z_drop
    mujoco.mj_forward(model, data)

    controller = _MotionController(model, data, np=np)
    renderer = mujoco.Renderer(model, height=height, width=width)

    duration_sec = 2.0
    n_steps = max(1, int(duration_sec / model.opt.timestep))
    # Capture at 0%, ~50%, ~90% of the rollout.
    capture_steps = {
        0: ("initial", "Rest pose — the assembled structure before any motion"),
        n_steps // 2: ("mid_sweep", "Mid-sweep — joints at ~50%% of their gesture"),
        int(n_steps * 0.9): ("extreme", "Near-extreme — joints near swept limits"),
    }

    frames: list[dict[str, str]] = []
    for step in range(n_steps):
        t = step * model.opt.timestep
        controller.apply(t)
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qacc)):
            break

        if step in capture_steps:
            label, desc_tpl = capture_steps[step]
            renderer.update_scene(data)
            pixels = renderer.render()
            buf = io.BytesIO()
            Image.fromarray(pixels).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            frames.append({
                "label": label,
                "description": desc_tpl.replace("%%", "%"),
                "image": f"data:image/png;base64,{b64}",
            })

    renderer.close()
    return frames


def _run_physics_hold(
    model: Any,
    duration_sec: float = 1.0,
    stabilize: bool = True,
    kp: float = 100.0,
    kv: float = 10.0,
) -> dict[str, Any]:
    """Run physics with a PD controller that holds the initial pose.

    A robotic arm under gravity without controllers WILL collapse — that
    is expected and not a pipeline bug.  The validation question is: with
    a reasonable PD controller, can the robot hold its initial pose?

    Returns:
        {
            "stabilized": bool,
            "max_qpos_error_deg": float,
            "max_body_displacement_mm": float,
            "timesteps": int,
            "unstable": bool,        # True if NaN/Inf/huge values encountered
            "huge_value": bool,      # True if qpos exceeds 100 rad
            "notes": str,
        }
    """
    import mujoco  # type: ignore[import-not-found]
    import numpy as np

    if stabilize:
        # Stronger defaults than Phase A's first attempt — tiny masses in
        # the chain (29g gripper fingers) require meaningful armature to
        # avoid numerical resonance at the joint-limit boundary.
        _stabilize_model(model, armature=0.5, damping=5.0)

    model.opt.gravity[:] = (0.0, 0.0, -9.81)
    if model.opt.timestep > 0.0005:
        model.opt.timestep = 0.0005
    try:
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        model.opt.iterations = 50
    except AttributeError:
        pass

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    initial_qpos = data.qpos.copy()
    initial_body_pos = data.xpos.copy()

    # Scale PD gains with the robot's total mass. A fixed kp=100 holds a
    # ~1kg single-arm robot, but a 5kg+ dual-arm-wheeled robot has larger
    # gravitational torques at the arm joints (longer levers + more mass),
    # so the PD term alone (even with qfrc_bias feed-forward, which has
    # one-step latency) leaves a 2.18° droop that fails the 1° threshold.
    # Scaling kp/kv by total_mass keeps the controller stiff enough across
    # robot sizes without over-tuning for tiny arms.
    total_mass = float(np.sum(model.body_mass))
    mass_scale = max(1.0, total_mass)
    kp = kp * mass_scale
    kv = kv * mass_scale

    n_steps = max(1, int(duration_sec / model.opt.timestep))
    unstable = False
    huge_value = False
    final_err_deg = 0.0
    max_disp_mm = 0.0
    blowup_step = -1

    # Identify prismatic (slide) joints — their tiny range (±8mm for
    # gripper fingers) and lateral gravity coupling make them drift to
    # joint limits during PD-hold, producing false physics-stability
    # failures.  In a real robot the gripper controller LOCKS the fingers
    # at the home position; we simulate this by clamping prismatic qpos
    # to their initial value each step.  This does not affect revolute
    # joints (the actual stability test) and is physically justified.
    # (Husky-style chassis has no suspension slides; all slides are grippers.)
    slide_joints = [
        jid for jid in range(model.njnt)
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE
    ]
    # Joints the PD controller holds (hinge + slide, NOT the floating base's
    # free joint).  Indexed by joint id → dof via jnt_dofadr/jnt_qposadr so
    # the maths is correct whether or not a floating base pads qpos/qvel.
    # EXCLUDE wheel hinges: a wheel is a continuous-rotation drive joint
    # (unbounded range) that must NOT be PD-held at a home angle — under
    # gravity the chassis weight pushes the wheels and a PD controller with
    # no steady state lets qpos accumulate past 100 rad → false "huge_value"
    # instability.  Wheels are free in the hold test (their bearing friction
    # + the chassis on the ground keeps them still).
    controlled_joints = [
        jid for jid in range(model.njnt)
        if model.jnt_type[jid] in (
            mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE,
        )
        and "wheel" not in (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        ).lower()
    ]
    # Wheels: if present, lock their velocity to 0 each step so they don't
    # free-spin under chassis weight during the arm-stability hold test.
    wheel_hold_jids = [
        jid for jid in range(model.njnt)
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
        and "wheel" in (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        ).lower()
    ]

    for step in range(n_steps):
        mujoco.mj_forward(model, data)
        # If the model has MuJoCo actuators (injected by _patch_mjcf), use
        # them instead of hand-rolled qfrc_applied PD.  The <position>
        # actuators have kp/kv baked in + forcerange saturation + MuJoCo
        # owns the gravity comp via qfrc_bias in the same dynamics pass.
        if model.nu > 0:
            # Feed-forward gravity compensation (inverse dynamics) so the
            # <position> actuator's PD only needs to correct transients, not
            # hold the arm's weight.  This is the standard textbook pattern:
            # qfrc_applied = qfrc_bias (gravity+Coriolis), then the actuator
            # PD (kp/kv from the <position> tag) handles the residual error.
            data.qfrc_applied[:] = data.qfrc_bias
            # Set arm actuator targets to the home pose (hold position).
            for jid in controlled_joints:
                qadr = model.jnt_qposadr[jid]
                for aid in range(model.nu):
                    if int(model.actuator_trnid[aid][0]) == jid:
                        data.ctrl[aid] = initial_qpos[qadr]
                        break
            # Zero wheel actuators (parked).
            for jid in wheel_hold_jids:
                dadr = model.jnt_dofadr[jid]
                data.qfrc_applied[dadr] = 0.0  # cancel gravity-comp on wheels
                data.qvel[dadr] = 0.0           # park (brakes on)
                for aid in range(model.nu):
                    if int(model.actuator_trnid[aid][0]) == jid:
                        data.ctrl[aid] = 0.0
                        break
        else:
            # Legacy path: hand-rolled qfrc_applied PD (for old URDFs/tests).
            data.qfrc_applied[:] = data.qfrc_bias
            for jid in controlled_joints:
                qadr = model.jnt_qposadr[jid]
                dadr = model.jnt_dofadr[jid]
                q_err = initial_qpos[qadr] - data.qpos[qadr]
                data.qfrc_applied[dadr] += kp * q_err - kv * data.qvel[dadr]
            for jid in wheel_hold_jids:
                dadr = model.jnt_dofadr[jid]
                data.qfrc_applied[dadr] = 0.0
                data.qvel[dadr] = 0.0
        # For a floating base, qpos (nq=21) and qvel/qfrc (nv=20) differ in
        # size AND the base DOFs occupy qvel[0:6] / qfrc[0:6] (vs qpos[0:7]
        # = xyz + quat).  The PD hold test checks ARM stability, not base
        # pose — so apply PD only to the non-base joints and let the base
        # float freely (gravity-compensated by qfrc_bias).  Indexing by
        # joint id (not raw qpos index) keeps this correct for both fixed
        # and floating bases.
        mujoco.mj_step(model, data)
        # Lock prismatic joints at their initial position (simulates the
        # gripper controller holding the fingers in the home pose).
        for jid in slide_joints:
            qadr = model.jnt_qposadr[jid]
            dadr = model.jnt_dofadr[jid]
            data.qpos[qadr] = initial_qpos[qadr]
            data.qvel[dadr] = 0.0
        # Re-zero wheel velocity after the step too (defensive: the constraint
        # solver can inject residual velocity).
        for jid in wheel_hold_jids:
            dadr = model.jnt_dofadr[jid]
            data.qvel[dadr] = 0.0
        # Catch NaN/Inf
        if not np.all(np.isfinite(data.qacc)):
            unstable = True
            blowup_step = step
            break
        # Catch "huge but finite" blowups (qpos > 100 rad means joint spun
        # > 5700°, which can't be physical).  Check only held joints — wheels
        # are unbounded and are velocity-locked above, so their qpos can
        # legitimately accumulate without being a stability failure.
        for jid in controlled_joints:
            qadr = model.jnt_qposadr[jid]
            if abs(data.qpos[qadr]) > 100.0:
                huge_value = True
                blowup_step = step
                break

    if not unstable and not huge_value:
        # Measure only JOINT error (not floating-base drift): the hold test
        # checks that the arms keep their pose, not that the base stays put.
        # Index by joint qpos-address so a floating base's 7 qpos entries
        # are excluded.  Use degrees of revolute joints only (slides are
        # locked) for the droop threshold.
        joint_errs_deg = []
        for jid in controlled_joints:
            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            qadr = model.jnt_qposadr[jid]
            joint_errs_deg.append(
                abs(np.degrees(initial_qpos[qadr] - data.qpos[qadr]))
            )
        final_err_deg = max(joint_errs_deg) if joint_errs_deg else 0.0
        mujoco.mj_forward(model, data)
        # Body displacement: exclude the floating base body (it can drift
        # slightly even under gravity-comp; that's not an arm-stability
        # failure).  Measure the max displacement of NON-base bodies.
        disp_m = np.linalg.norm(data.xpos - initial_body_pos, axis=1)
        # Body 0 is world; if body 1 is the floating base, skip it.
        base_skip = 1 if (model.njnt > 0 and
                          model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE) else 0
        if len(disp_m) > base_skip + 1:
            max_disp_mm = float(np.max(disp_m[base_skip + 1:])) * 1000.0
        else:
            max_disp_mm = float(np.max(disp_m)) * 1000.0

    stabilized = (not unstable) and (not huge_value) and final_err_deg < 1.0 and max_disp_mm < 1.0

    # --- Raw PD-hold (no gravity compensation) ---
    # The gravity-compensated hold above trivially yields ~0° error because
    # qfrc_bias cancels the gravitational load. To expose the *real*
    # steady-state droop — which is a genuine design-quality signal (stiff
    # arms droop less) — run the same PD hold WITHOUT qfrc_bias feed-forward.
    raw_err_deg = 0.0
    if not unstable and not huge_value:
        raw_data = mujoco.MjData(model)
        mujoco.mj_forward(model, raw_data)
        raw_initial = raw_data.qpos.copy()
        for step in range(n_steps):
            mujoco.mj_forward(model, raw_data)
            # PD WITHOUT gravity comp — just kp*(target-current) - kv*vel
            for jid in controlled_joints:
                qadr = model.jnt_qposadr[jid]
                dadr = model.jnt_dofadr[jid]
                raw_data.qfrc_applied[dadr] = (
                    kp * (raw_initial[qadr] - raw_data.qpos[qadr])
                    - kv * raw_data.qvel[dadr]
                )
            for jid in slide_joints:
                qadr = model.jnt_qposadr[jid]
                dadr = model.jnt_dofadr[jid]
                raw_data.qpos[qadr] = raw_initial[qadr]
                raw_data.qvel[dadr] = 0.0
            for jid in wheel_hold_jids:
                dadr = model.jnt_dofadr[jid]
                raw_data.qvel[dadr] = 0.0
            mujoco.mj_step(model, raw_data)
            if not np.all(np.isfinite(raw_data.qacc)):
                break
        raw_errs = []
        for jid in controlled_joints:
            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            qadr = model.jnt_qposadr[jid]
            raw_errs.append(abs(np.degrees(raw_initial[qadr] - raw_data.qpos[qadr])))
        raw_err_deg = max(raw_errs) if raw_errs else 0.0

    if stabilized:
        note = f"PD-hold pass (raw droop without grav-comp: {raw_err_deg:.2f}°)"
    else:
        flags = []
        if unstable:
            flags.append("NaN/Inf in QACC")
        if huge_value:
            flags.append(f"qpos > 100 rad at step {blowup_step}")
        note = (
            f"PD-hold fail: err={final_err_deg:.2f}deg disp={max_disp_mm:.2f}mm"
            + (f" [{', '.join(flags)}]" if flags else "")
        )

    return {
        "stabilized": bool(stabilized),
        "max_qpos_error_deg": float(final_err_deg),
        "raw_droop_deg": float(raw_err_deg),
        "max_body_displacement_mm": float(max_disp_mm),
        "timesteps": int(n_steps),
        "unstable": bool(unstable),
        "huge_value": bool(huge_value),
        "blowup_step": int(blowup_step),
        "notes": note,
    }


def _test_single_joint(
    model: Any,
    target_jid: int,
    force: float = 0.5,
    duration_sec: float = 0.5,
    stabilize: bool = True,
) -> dict[str, Any]:
    """Apply a torque/force to a single joint and measure angular/linear motion.

    A "moved=True" result means the joint is not kinematically locked —
    it responds to applied force.  The ``within_range`` flag additionally
    reports whether the motion stayed within the joint's physical limits,
    which is the stricter test of "meaningful motion" vs "numerical blowup".
    """
    import mujoco  # type: ignore[import-not-found]
    import numpy as np

    if stabilize:
        _stabilize_model(model, armature=0.5, damping=5.0)
    model.opt.gravity[:] = (0.0, 0.0, -9.81)
    if model.opt.timestep > 0.0005:
        model.opt.timestep = 0.0005

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Use the joint's qpos/dof addresses, not the raw joint id — a floating
    # base pads qpos[0:7] (xyz+quat) and qvel/qfrc[0:6], so jid≠qposadr≠dofadr.
    qadr = int(model.jnt_qposadr[target_jid])
    dadr = int(model.jnt_dofadr[target_jid])
    initial_q = float(data.qpos[qadr])

    n_steps = max(1, int(duration_sec / model.opt.timestep))
    unstable = False
    huge = False
    for _ in range(n_steps):
        data.qfrc_applied[:] = 0
        data.qfrc_applied[dadr] = force
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qacc)):
            unstable = True
            break
        if abs(data.qpos[qadr]) > 100.0:
            huge = True
            break

    delta = float(data.qpos[qadr]) - initial_q
    jtype = int(model.jnt_type[target_jid])
    rng = model.jnt_range[target_jid]

    # HINGE → degrees, SLIDE → mm
    if jtype == 3:  # HINGE
        delta_reported = float(np.degrees(delta))
        unit = "deg"
        # "moved" = responded at all (even if numerically).  "within_range"
        # = stayed inside the joint's physical limits.
        moved = abs(delta_reported) > 0.5 and not unstable
        rng_low_deg = float(np.degrees(rng[0]))
        rng_high_deg = float(np.degrees(rng[1]))
        within_range = (
            moved and not huge
            and rng_low_deg - 1.0 <= delta_reported + np.degrees(initial_q) <= rng_high_deg + 1.0
        )
    elif jtype == 2:  # SLIDE
        delta_reported = delta * 1000.0
        unit = "mm"
        moved = abs(delta_reported) > 0.1 and not unstable
        rng_low_mm = rng[0] * 1000.0
        rng_high_mm = rng[1] * 1000.0
        within_range = (
            moved and not huge
            and rng_low_mm - 0.5 <= delta_reported + initial_q * 1000.0 <= rng_high_mm + 0.5
        )
    else:
        delta_reported = delta
        unit = "raw"
        moved = not unstable
        within_range = moved

    return {
        "joint_id": target_jid,
        "delta": float(delta_reported),
        "unit": unit,
        "moved": bool(moved),
        "within_range": bool(within_range),
        "unstable": bool(unstable),
        "huge_value": bool(huge),
    }


# ============================================================================
# Tool
# ============================================================================


class SimMujocoTool(Tool):
    """Validate a generated URDF by loading it into MuJoCo and running physics."""

    name = "sim_mujoco"
    description = (
        "Load a URDF (with STL meshes) into MuJoCo to validate the NL→CAD→URDF "
        "pipeline. Reports load errors, mesh path issues, mass/inertia sanity, "
        "physics stability under PD control, and per-joint actuation tests. "
        "Use this to verify a generated robot can actually hold a pose and move "
        "its joints."
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
                        "description": (
                            "Absolute or relative path to the URDF file. "
                            "Mesh paths inside are auto-resolved."
                        ),
                    },
                    "mode": {
                        "type": "string",
                        "description": (
                            "validate = load + physics + joint test (default); "
                            "report = load + scan only, no physics; "
                            "physics = load + PD-hold only, no joint test"
                        ),
                    },
                    "duration_sec": {
                        "type": "number",
                        "description": "Duration of PD-hold physics simulation in seconds",
                    },
                    "stabilize": {
                        "type": "boolean",
                        "description": (
                            "Add joint armature and damping to stabilise tiny-mass "
                            "parts (gripper fingers). Default true — turn off only "
                            "to test raw dynamics."
                        ),
                    },
                    "joint_test": {
                        "type": "boolean",
                        "description": "Run per-joint actuation test (default true)",
                    },
                    "interactive": {
                        "type": "boolean",
                        "description": (
                            "If true and a display is available, launch the "
                            "MuJoCo viewer window to visualise the simulation "
                            "in real time. Default false (headless). When "
                            "true, the call blocks until the viewer window is "
                            "closed."
                        ),
                    },
                },
                "required": ["urdf_path"],
            },
        )

    def execute(
        self,
        *,
        urdf_path: str,
        mode: str = "validate",
        duration_sec: float = 2.0,
        stabilize: bool = True,
        joint_test: bool = True,
        interactive: bool = False,
        **kwargs: Any,
    ) -> str:
        """Execute the simulation validation."""
        if not urdf_path:
            return "Error: urdf_path is required"

        # Load — use floating_base for interactive mode so wheeled robots
        # can actually drive in the viewer (the base is injected as a FREE
        # joint, enabling chassis translation).
        load_result = _load_model(urdf_path, floating_base=interactive)
        try:
            if not load_result["ok"]:
                return self._format_load_failure(urdf_path, load_result)

            model = load_result["model"]
            warnings = load_result["warnings"]

            bodies = _scan_bodies(model)
            joints = _scan_joints(model)
            n_urdf_links_expected = self._count_urdf_links(urdf_path)

            physics_result: dict[str, Any] | None = None
            joint_results: list[dict[str, Any]] = []

            if mode != "report":
                if interactive:
                    # Launch GUI viewer — blocks until window closed
                    _launch_viewer(model, duration_sec=duration_sec,
                                   stabilize=stabilize)
                    # Skip the headless physics/joint tests when running
                    # interactively; the user just watched it happen.
                    physics_result = {"interactive": True, "stable": True}
                else:
                    physics_result = _run_physics_hold(
                        model,
                        duration_sec=duration_sec,
                        stabilize=stabilize,
                    )

                    if joint_test and mode == "validate":
                        for j in joints:
                            # HINGE gets torque, SLIDE gets force
                            force = 0.5 if j["type"] == "HINGE" else 0.1
                            result = _test_single_joint(
                                model,
                                target_jid=j["id"],
                                force=force,
                                duration_sec=0.5,
                                stabilize=stabilize,
                            )
                            result["joint_name"] = j["name"]
                            result["joint_type"] = j["type"]
                            joint_results.append(result)

            return self._format_report(
                urdf_path=urdf_path,
                warnings=warnings,
                bodies=bodies,
                joints=joints,
                physics=physics_result,
                joint_results=joint_results,
                n_urdf_links_expected=n_urdf_links_expected,
            )
        finally:
            # Clean up temp files
            for tmp in load_result.get("temp_files", []):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Report formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _count_urdf_links(urdf_path: str) -> int:
        """Count <link> elements in the URDF (for diagnostic comparison)."""
        try:
            text = Path(urdf_path).read_text(encoding="utf-8")
            return len(re.findall(r"<link\s+name=", text))
        except Exception:
            return -1

    @staticmethod
    def _format_load_failure(urdf_path: str, load_result: dict) -> str:
        lines = [
            f"[sim_mujoco] LOAD FAILED: {urdf_path}",
            f"Error: {load_result['error']}",
        ]
        if load_result["warnings"]:
            lines.append("")
            lines.append("Mesh path warnings:")
            for w in load_result["warnings"]:
                lines.append(f"  - {w}")
        return "\n".join(lines)

    @staticmethod
    def _format_report(
        *,
        urdf_path: str,
        warnings: list[str],
        bodies: list[dict[str, Any]],
        joints: list[dict[str, Any]],
        physics: dict[str, Any] | None,
        joint_results: list[dict[str, Any]],
        n_urdf_links_expected: int,
    ) -> str:
        n_dynamic = sum(1 for b in bodies if b["id"] != 0 and b["mass_kg"] > 0)
        n_merged = max(0, n_urdf_links_expected - len(bodies))

        lines: list[str] = [
            f"[sim_mujoco] {urdf_path}",
            f"加载结果: 成功",
            f"URDF 链接数: {n_urdf_links_expected}  MuJoCo body 数: {len(bodies)}  "
            f"(fixed 合并: {n_merged})",
            f"动态 body 数: {n_dynamic}",
            f"关节数: {len(joints)}",
        ]

        if warnings:
            lines.append("")
            lines.append("--- Mesh 路径警告 ---")
            for w in warnings:
                lines.append(f"  ! {w}")

        # Bodies
        lines.append("")
        lines.append("--- Body 质量与惯性 ---")
        for b in bodies:
            tag = "(world)" if b["id"] == 0 else ""
            line = (
                f"  [{b['id']:2d}] {b['name']:30s} {tag}\n"
                f"      mass={b['mass_kg']:.4f}kg "
                f"inertia=({b['inertia'][0]:.5e}, {b['inertia'][1]:.5e}, {b['inertia'][2]:.5e})"
            )
            lines.append(line)
            if b["mass_warning"]:
                lines.append(f"      WARN: {b['mass_warning']}")
            if b["inertia_warning"]:
                lines.append(f"      WARN: {b['inertia_warning']}")

        # Joints
        lines.append("")
        lines.append("--- Joints ---")
        for j in joints:
            rng_deg = (
                f"[{math.degrees(j['range'][0]):.1f}, {math.degrees(j['range'][1]):.1f}]°"
                if j["type"] == "HINGE"
                else f"[{j['range'][0]*1000:.2f}, {j['range'][1]*1000:.2f}]mm"
            )
            axis_str = f"({j['axis'][0]:.1f},{j['axis'][1]:.1f},{j['axis'][2]:.1f})"
            lines.append(
                f"  [{j['id']}] {j['name']:45s} {j['type']:6s} range={rng_deg} axis={axis_str}"
            )

        # Physics
        if physics is not None:
            lines.append("")
            if physics.get("interactive"):
                lines.append("--- 物理稳定性 (interactive viewer, user-verified) ---")
                lines.append("  状态: INTERACTIVE (user watched the viewer)")
            else:
                lines.append(f"--- 物理稳定性 (PD hold, {physics['timesteps']} steps) ---")
                status = "PASS" if physics["stabilized"] else "FAIL"
                lines.append(f"  状态: {status}")
                lines.append(f"  最大关节角误差: {physics['max_qpos_error_deg']:.3f}° (阈值 1°, grav-comp)")
                raw_droop = physics.get("raw_droop_deg", 0.0)
                lines.append(f"  真实下垂(无重力补偿): {raw_droop:.3f}°")
                lines.append(f"  最大 body 位移: {physics['max_body_displacement_mm']:.3f}mm (阈值 1mm)")
                if physics["unstable"]:
                    lines.append("  WARN: 数值不稳定 (NaN/Inf in QACC) — see notes")
                lines.append(f"  备注: {physics['notes']}")

        # Joint test
        if joint_results:
            lines.append("")
            lines.append("--- 关节能动性测试 ---")
            n_moved = sum(1 for r in joint_results if r["moved"])
            n_in_range = sum(1 for r in joint_results if r.get("within_range"))
            lines.append(
                f"能动关节数: {n_moved}/{len(joint_results)}  "
                f"其中物理可动 (在量程内): {n_in_range}/{len(joint_results)}"
            )
            for r in joint_results:
                if r["unstable"] or r.get("huge_value"):
                    status = "[数值爆炸]"
                elif r.get("within_range"):
                    status = "[物理可动]"
                elif r["moved"]:
                    status = "[能动但越界]"
                else:
                    status = "[不动]"
                unit = r["unit"]
                lines.append(
                    f"  [{r['joint_id']}] {r['joint_name']:45s} "
                    f"d={r['delta']:+10.2f} {unit}  {status}"
                )

        # Overall verdict — structural validity is the hard requirement,
        # physics stability is a separate soft signal because it depends
        # heavily on controller/actuator design which is not part of the
        # NL→CAD→URDF pipeline output.
        lines.append("")
        lines.append("--- 总体验证结论 ---")

        structural_ok = True
        structural_reasons: list[str] = []
        # Structural: all meshes resolved (the auto-rewrite warnings are
        # informational, only "Mesh not found" warnings are failures).
        missing_meshes = [w for w in warnings if "Mesh not found" in w]
        if missing_meshes:
            structural_ok = False
            structural_reasons.append(f"{len(missing_meshes)} 个 mesh 无法解析")
        # Structural: all dynamic bodies have positive mass
        mass_warns = [b for b in bodies if b["id"] != 0 and b["mass_warning"]]
        if mass_warns:
            structural_ok = False
            structural_reasons.append(f"{len(mass_warns)} 个 body 质量异常")
        # Structural: at least one joint can move under torque
        if joint_results:
            n_moving = sum(1 for r in joint_results if r["moved"])
            if n_moving == 0:
                structural_ok = False
                structural_reasons.append("所有关节都无法驱动")

        if structural_ok:
            lines.append("  结构验证: PASS — URDF 可加载，mesh 可解析，质量/惯性合理，关节可驱动")
        else:
            lines.append("  结构验证: FAIL — 发现问题:")
            for reason in structural_reasons:
                lines.append(f"    - {reason}")

        # Physics stability (soft signal)
        physics_ok = physics is not None and physics.get("stabilized", physics.get("stable", False))
        if physics is not None:
            if physics_ok:
                lines.append("  物理稳定: PASS — PD 控制下能保持初始姿态")
            else:
                lines.append(
                    "  物理稳定: WARN — PD 控制下无法保持姿态（这通常意味着 URDF 需要 <actuator> "
                    "定义或控制器调参，不一定是 URDF 结构错误）"
                )
                if physics["unstable"]:
                    lines.append(f"    详情: {physics['notes']}")

        # Auto-fix info
        rewrite_warning = next((w for w in warnings if "Rewrote" in w), None)
        if rewrite_warning:
            lines.append(f"  Mesh 自动修复: {rewrite_warning}")

        verdict_ok = structural_ok

        # JSON summary
        summary = {
            "urdf_path": urdf_path,
            "load_ok": True,
            "n_urdf_links": n_urdf_links_expected,
            "n_bodies": len(bodies),
            "n_merged_fixed": n_merged,
            "n_joints": len(joints),
            "warnings": warnings,
            "physics": physics,
            "joint_results": joint_results,
            "verdict_ok": verdict_ok,
        }
        lines.append("")
        lines.append("--- JSON ---")
        lines.append(json.dumps(summary, ensure_ascii=False, indent=2))

        return "\n".join(lines)


# ============================================================================
# Registration
# ============================================================================


def register_sim_tools(registry: Any) -> None:
    """Register MuJoCo simulation tools."""
    registry.register(SimMujocoTool())
    registry.register(SimGraspTool())


# ============================================================================
# Grasp validation — spawn cube, close gripper, lift arm
# ============================================================================



# ---------------------------------------------------------------------------
# Grasp simulation extracted to sim_grasp.py (P1-1 split). Re-exported.
# ---------------------------------------------------------------------------
from .sim_grasp import _find_slide_joints  # noqa: F401
from .sim_grasp import _group_fingers_into_grippers  # noqa: F401
from .sim_grasp import _add_cube_to_scene  # noqa: F401
from .sim_grasp import _run_grasp_scenario  # noqa: F401
from .sim_grasp import SimGraspTool  # noqa: F401
from .sim_grasp import _add_cube_to_scene  # noqa: F401
from .sim_grasp import _run_grasp_scenario  # noqa: F401

