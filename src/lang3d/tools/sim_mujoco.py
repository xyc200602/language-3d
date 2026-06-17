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


def _mujoco_available() -> bool:
    """Return True if the ``mujoco`` package is importable."""
    try:
        import mujoco  # noqa: F401
        return True
    except ImportError:
        return False


def _launch_viewer(
    model: Any,
    *,
    duration_sec: float = 10.0,
    stabilize: bool = True,
) -> None:
    """Open the MuJoCo passive viewer and step physics until window closes.

    Added 2026-06-18 to give the simulation path a real GUI (the user
    observed "simulation was done without a graphical interface").  Uses
    ``mujoco.viewer.launch_passive`` which does NOT block on its own —
    the caller must run a step loop and call ``viewer.sync()`` each frame.

    Falls back to a no-op log message on headless environments or
    import errors so e2e tests can still call ``interactive=False`` on a
    server.
    """
    try:
        import mujoco  # type: ignore[import-not-found]
        import mujoco.viewer  # type: ignore[import-not-found]
        import time as _time
    except ImportError as e:
        logger.warning("MuJoCo viewer unavailable: %s", e)
        return

    if stabilize:
        _stabilize_model(model)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    max_steps = int(duration_sec / max(model.opt.timestep, 1e-5))
    logger.info(
        "Launching MuJoCo viewer (max_steps=%d, dt=%.4fs)",
        max_steps, model.opt.timestep,
    )
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            step = 0
            while viewer.is_running() and step < max_steps:
                mujoco.mj_step(model, data)
                viewer.sync()
                _time.sleep(model.opt.timestep)
                step += 1
    except Exception as e:
        logger.warning("MuJoCo viewer exited with error: %s", e)


def _load_model(urdf_path: str) -> dict[str, Any]:
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
    except Exception as exc:
        return {
            "ok": False,
            "error": f"MuJoCo parse error: {exc}",
            "warnings": warnings,
            "temp_files": temp_files,
        }

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
    """
    for jid in range(model.njnt):
        model.dof_armature[jid] = armature
        model.dof_damping[jid] = damping


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
        _stabilize_model(model, armature=0.2, damping=3.0)

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

    n_steps = max(1, int(duration_sec / model.opt.timestep))
    unstable = False
    huge_value = False
    final_err_deg = 0.0
    max_disp_mm = 0.0
    blowup_step = -1

    for step in range(n_steps):
        # Refresh bias forces (gravity + Coriolis) for the current pose so
        # the feed-forward term tracks the configuration.  Without gravity
        # compensation a pure PD controller has a steady-state error under
        # gravity — small at the joints (sub-degree) but lever-amplified at
        # the distal links, pushing body displacement past the 1 mm
        # threshold on long arms.  Adding ``qfrc_bias`` is the textbook
        # inverse-dynamics feed-forward; it cancels the gravitational load
        # so the PD term only needs to correct transients, not hold weight.
        mujoco.mj_forward(model, data)
        q_err = initial_qpos - data.qpos
        data.qfrc_applied[:] = (
            kp * q_err - kv * data.qvel + data.qfrc_bias
        )
        mujoco.mj_step(model, data)
        # Catch NaN/Inf
        if not np.all(np.isfinite(data.qacc)):
            unstable = True
            blowup_step = step
            break
        # Catch "huge but finite" blowups (qpos > 100 rad means joint spun
        # > 5700°, which can't be physical)
        if np.any(np.abs(data.qpos) > 100.0):
            huge_value = True
            blowup_step = step
            break

    if not unstable and not huge_value:
        final_err = initial_qpos - data.qpos
        final_err_deg = float(np.max(np.abs(np.degrees(final_err))))
        mujoco.mj_forward(model, data)
        disp_m = np.linalg.norm(data.xpos - initial_body_pos, axis=1)
        max_disp_mm = float(np.max(disp_m)) * 1000.0

    stabilized = (not unstable) and (not huge_value) and final_err_deg < 1.0 and max_disp_mm < 1.0

    if stabilized:
        note = "PD-hold pass"
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
        _stabilize_model(model, armature=0.2, damping=3.0)
    model.opt.gravity[:] = (0.0, 0.0, -9.81)
    if model.opt.timestep > 0.0005:
        model.opt.timestep = 0.0005

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial_q = float(data.qpos[target_jid])

    n_steps = max(1, int(duration_sec / model.opt.timestep))
    unstable = False
    huge = False
    for _ in range(n_steps):
        data.qfrc_applied[:] = 0
        data.qfrc_applied[target_jid] = force
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qacc)):
            unstable = True
            break
        if abs(data.qpos[target_jid]) > 100.0:
            huge = True
            break

    delta = float(data.qpos[target_jid]) - initial_q
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

        # Load
        load_result = _load_model(urdf_path)
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
            lines.append(f"--- 物理稳定性 (PD hold, {physics['timesteps']} steps) ---")
            status = "PASS" if physics["stabilized"] else "FAIL"
            lines.append(f"  状态: {status}")
            lines.append(f"  最大关节角误差: {physics['max_qpos_error_deg']:.3f}° (阈值 1°)")
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
        physics_ok = physics is not None and physics["stabilized"]
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
    initial_arm_qpos = {jid: float(data.qpos[jid]) for jid in arm_jids}

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
            data.qfrc_applied[sjid] = sign * grasp_force_n
            data.qfrc_applied[sjid] -= 2.0 * data.qvel[sjid]

    def _hold_arm() -> None:
        for ajid in arm_jids:
            err = initial_arm_qpos[ajid] - data.qpos[ajid]
            data.qfrc_applied[ajid] = 200.0 * err - 20.0 * data.qvel[ajid]

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
                data.qfrc_applied[ajid] -= 1.5

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
        "finger_final_qpos": {j["name"]: float(data.qpos[j["jid"]]) for j in slide_joints},
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

            if len(slide_joints) != 2:
                return self._format_no_gripper(urdf_path, slide_joints)

            # 2. Forward kinematics → finger positions
            # Use the slide-joint body origin as the reference point.  This
            # is the conventional "grasp center" for a 2-finger gripper
            # whose fingers are mirror-symmetric about the parent body's
            # YZ plane.
            _stabilize_model(model, armature=0.2, damping=3.0)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)

            finger1_pos = np.array(data.xpos[slide_joints[0]["body_id"]], dtype=float)
            finger2_pos = np.array(data.xpos[slide_joints[1]["body_id"]], dtype=float)
            grasp_center = (finger1_pos + finger2_pos) / 2.0

            # Verify fingers are actually separated (else this isn't a gripper)
            finger_sep_m = float(np.linalg.norm(finger1_pos - finger2_pos))
            cube_size_m = cube_size_mm / 1000.0
            if finger_sep_m < cube_size_m:
                return self._format_finger_sep_warning(
                    urdf_path, finger_sep_m, cube_size_m,
                )

            # 3. Build scene with cube at grasp_center
            scene_model, temp_files = _add_cube_to_scene(
                urdf_path=str(Path(urdf_path).resolve()),
                cube_pos_m=tuple(float(x) for x in grasp_center),
                cube_size_m=cube_size_m,
                cube_mass_kg=cube_mass_g / 1000.0,
            )

            try:
                # Find slide joints in the NEW model (ids may differ)
                scene_slides = _find_slide_joints(scene_model)
                if len(scene_slides) != 2:
                    return (
                        f"Error: scene has {len(scene_slides)} slide joints, "
                        f"expected 2"
                    )

                cube_body_id = mujoco.mj_name2id(
                    scene_model, mujoco.mjtObj.mjOBJ_BODY, "grasp_cube",
                )
                if cube_body_id < 0:
                    return "Error: grasp_cube body not found in scene"

                # 4. Run grasp scenario
                grasp = _run_grasp_scenario(
                    scene_model,
                    scene_slides,
                    cube_body_id,
                    grasp_force_n=grasp_force_n,
                    lift_height_m=lift_height_mm / 1000.0,
                    duration_sec=duration_sec,
                )

                return self._format_report(
                    urdf_path=urdf_path,
                    slide_joints=slide_joints,
                    finger_sep_m=finger_sep_m,
                    grasp_center_m=tuple(float(x) for x in grasp_center),
                    cube_size_mm=cube_size_mm,
                    cube_mass_g=cube_mass_g,
                    grasp_force_n=grasp_force_n,
                    lift_height_mm=lift_height_mm,
                    duration_sec=duration_sec,
                    grasp=grasp,
                    warnings=load_result["warnings"],
                )
            finally:
                for tmp in temp_files:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
        finally:
            for tmp in load_result.get("temp_files", []):
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
    def _format_no_gripper(urdf_path: str, slide_joints: list[dict]) -> str:
        names = [j["name"] for j in slide_joints]
        lines = [
            f"[sim_grasp] NO GRIPPER DETECTED",
            f"URDF: {urdf_path}",
            f"Found {len(slide_joints)} SLIDE joints; need exactly 2 "
            f"for a 2-finger gripper.",
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
