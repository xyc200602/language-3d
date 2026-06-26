"""Standardized E2E Production Test for Language-3D.

Validates the core value proposition: natural language → production-grade robot
assembly folder.  Each test case goes through the full
``generate_assembly_with_vlm_loop()`` pipeline and validates output across 6
phases.  Only Phase 1 (NL→Assembly) and Phase 4 (engineering package files)
fail the test; all other phases produce warnings.

Usage:
    # Pytest (CI-friendly)
    python -m pytest tests/test_e2e_production.py -v
    python -m pytest tests/test_e2e_production.py -v -k "4wheel_dual_arm"
    python -m pytest tests/ -v -m "not e2e"  # skip E2E, run unit tests only

    # Standalone script
    python tests/test_e2e_production.py                    # all cases
    python tests/test_e2e_production.py --case 4dof_arm    # single case
    python tests/test_e2e_production.py --list              # list cases
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Ensure project src is importable when running as a standalone script
# ---------------------------------------------------------------------------
_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT / "src"))

# Load .env for API keys
try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

ROBOT_TEST_CASES: list[dict[str, Any]] = [
    {
        "id": "4wheel_dual_arm",
        "description": (
            "设计一个4轮双臂机器人，四个差速驱动轮分布在底盘四角，"
            "底盘上方左右各安装一个3自由度机械臂"
        ),
        "min_parts": 15,
        "min_joints": 4,
        "expect_wheels": True,
        "expect_arms": True,
    },
    {
        "id": "humanoid_2leg_2arm",
        "description": (
            "设计一个2腿2臂人型机器人，两条腿各有3个关节，"
            "两只手臂各有3个自由度，顶部有摄像头"
        ),
        "min_parts": 12,
        "min_joints": 6,
        "expect_wheels": False,
        "expect_arms": True,
    },
    {
        "id": "4dof_arm",
        "description": (
            "设计一个4自由度机械臂，底座固定，包含肩部旋转、"
            "肩部俯仰、肘部弯曲和腕部旋转关节"
        ),
        "min_parts": 6,
        "min_joints": 4,
        "expect_wheels": False,
        "expect_arms": True,
    },
    {
        "id": "2dof_arm",
        "description": "设计一个2自由度机械臂，底座旋转加肩部俯仰，带夹爪",
        "min_parts": 6,
        "min_joints": 3,
        "expect_wheels": False,
        "expect_arms": True,
    },
    {
        "id": "3dof_arm",
        "description": "设计一个3自由度机械臂，底座旋转、肩部俯仰、肘部弯曲，带夹爪",
        "min_parts": 7,
        "min_joints": 4,
        "expect_wheels": False,
        "expect_arms": True,
    },
    {
        "id": "5dof_arm",
        "description": "设计一个5自由度机械臂，底座旋转、肩部俯仰、肘部弯曲、腕部俯仰、腕部滚转，带夹爪",
        "min_parts": 9,
        "min_joints": 6,
        "expect_wheels": False,
        "expect_arms": True,
    },
    {
        "id": "6dof_arm",
        "description": "设计一个6自由度机械臂，工业级球腕结构，带夹爪",
        "min_parts": 11,
        "min_joints": 7,
        "expect_wheels": False,
        "expect_arms": True,
    },
    {
        "id": "7dof_arm",
        "description": "设计一个7自由度冗余机械臂，带夹爪",
        "min_parts": 13,
        "min_joints": 8,
        "expect_wheels": False,
        "expect_arms": True,
    },
]

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------
PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
WARN = "WARN"

# Set by _main() when --pipeline flag is used (Step 2 architecture test).
args_pipeline_mode = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check(
    checks: list[dict],
    phase: str,
    step: str,
    condition: bool,
    detail: str = "",
    *,
    critical: bool = False,
) -> bool:
    """Record a check result and return *condition*."""
    status = PASS if condition else FAIL
    checks.append(
        {
            "phase": phase,
            "step": step,
            "status": status,
            "detail": detail,
            "critical": critical,
        }
    )
    icon = "PASS" if status == PASS else "FAIL"
    tag = " (CRITICAL)" if critical else ""
    print(f"  [{icon}{tag}] {step}: {detail}")
    return condition


def _warn(checks: list[dict], phase: str, step: str, detail: str) -> None:
    checks.append({"phase": phase, "step": step, "status": WARN, "detail": detail})
    print(f"  [WARN] {step}: {detail}")


def _skip(checks: list[dict], phase: str, step: str, reason: str) -> None:
    checks.append({"phase": phase, "step": step, "status": SKIP, "detail": reason})
    print(f"  [SKIP] {step}: {reason}")


def _joints_form_tree(assembly: Any) -> bool:
    """Return True if joints connect all parts into a single component (BFS)."""
    if not assembly.parts:
        return False
    part_names = {p.name for p in assembly.parts}
    if not assembly.joints:
        return len(part_names) <= 1

    adj: dict[str, set[str]] = {n: set() for n in part_names}
    for j in assembly.joints:
        if j.parent in adj and j.child in adj:
            adj[j.parent].add(j.child)
            adj[j.child].add(j.parent)

    visited: set[str] = set()
    queue = deque([next(iter(part_names))])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        queue.extend(adj.get(node, set()) - visited)

    return visited == part_names


def _has_category_parts(assembly: Any, keywords: list[str]) -> bool:
    """Check if any part name or category matches one of *keywords*."""
    for p in assembly.parts:
        name_lower = p.name.lower()
        cat_lower = (p.category or "").lower()
        for kw in keywords:
            if kw in name_lower or kw in cat_lower:
                return True
    return False


def _compute_score(checks: list[dict]) -> float:
    if not checks:
        return 0.0
    passed = sum(1 for c in checks if c["status"] == PASS)
    return round(passed / len(checks) * 100, 1)


def _save_report(
    output_dir: str,
    test_id: str,
    description: str,
    checks: list[dict],
    score: float,
) -> str:
    report = {
        "test_id": test_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": description,
        "score": score,
        "checks": checks,
    }
    report_path = os.path.join(output_dir, "e2e_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report_path


# ---------------------------------------------------------------------------
# 6 Validation Phases
# ---------------------------------------------------------------------------


def _phase1_nl_to_assembly(
    checks: list[dict],
    case: dict,
    output_dir: str,
) -> dict | None:
    """NL -> Assembly via VLM loop.  CRITICAL phase."""
    phase = "phase1"
    description = case["description"]

    from lang3d.tools.assembly_generator import generate_assembly_with_vlm_loop

    t0 = time.time()
    try:
        result = generate_assembly_with_vlm_loop(
            description=description,
            output_dir=output_dir,
            max_rounds=3,
        )
    except Exception as exc:
        _check(
            checks, phase, "vlm_loop_completed", False,
            f"Exception: {exc}", critical=True,
        )
        return None
    dt = time.time() - t0

    assembly = result.get("assembly")
    passed = result.get("passed", False)
    rounds = result.get("rounds", 0)

    _check(
        checks, phase, "vlm_loop_completed", assembly is not None,
        f"VLM loop done ({dt:.1f}s), {rounds} rounds, passed={passed}",
        critical=True,
    )

    if assembly is None:
        return None

    _check(
        checks, phase, "part_count",
        len(assembly.parts) >= case["min_parts"],
        f"Parts: {len(assembly.parts)} (min {case['min_parts']})",
        critical=True,
    )

    _check(
        checks, phase, "joint_count",
        len(assembly.joints) >= case["min_joints"],
        f"Joints: {len(assembly.joints)} (min {case['min_joints']})",
        critical=True,
    )

    _check(
        checks, phase, "connected_tree",
        _joints_form_tree(assembly),
        "Joints form connected tree",
        critical=True,
    )

    if case.get("expect_wheels"):
        _check(
            checks, phase, "has_wheels",
            _has_category_parts(assembly, ["wheel", "轮"]),
            "Wheel-like parts present",
        )

    if case.get("expect_arms"):
        _check(
            checks, phase, "has_arms",
            _has_category_parts(assembly, ["arm", "臂", "gripper", "夹爪", "shoulder", "elbow", "wrist", "effector"]),
            "Arm-like parts present",
        )

    return result


def _phase1_nl_to_assembly_pipeline(
    checks: list[dict],
    case: dict,
    output_dir: str,
) -> dict | None:
    """NL -> Assembly via the multi-agent AssemblyPipeline (Step 2).

    This is the pipeline-based alternative to ``_phase1_nl_to_assembly``.
    It uses ``AssemblyPipeline.run()`` instead of the monolithic
    ``generate_assembly_with_vlm_loop``.  The return dict has the same
    keys so downstream phases (2-7) work unchanged.
    """
    phase = "phase1"
    description = case["description"]

    from lang3d.agent.pipeline import AssemblyPipeline, PipelineContext

    t0 = time.time()
    try:
        ctx = PipelineContext(
            description=description,
            output_dir=output_dir,
            max_rounds=3,
        )
        pipeline = AssemblyPipeline(ctx)
        result = pipeline.run()
    except Exception as exc:
        _check(
            checks, phase, "vlm_loop_completed", False,
            f"Pipeline exception: {exc}", critical=True,
        )
        return None
    dt = time.time() - t0

    assembly = result.get("assembly")
    passed = result.get("passed", False)
    rounds = result.get("rounds", 0)

    _check(
        checks, phase, "vlm_loop_completed", assembly is not None,
        f"Pipeline done ({dt:.1f}s), {rounds} rounds, passed={passed}",
        critical=True,
    )

    if assembly is None:
        return None

    _check(
        checks, phase, "part_count",
        len(assembly.parts) >= case["min_parts"],
        f"Parts: {len(assembly.parts)} (min {case['min_parts']})",
        critical=True,
    )

    _check(
        checks, phase, "joint_count",
        len(assembly.joints) >= case["min_joints"],
        f"Joints: {len(assembly.joints)} (min {case['min_joints']})",
        critical=True,
    )

    _check(
        checks, phase, "connected_tree",
        _joints_form_tree(assembly),
        "Joints form connected tree",
        critical=True,
    )

    if case.get("expect_arms"):
        _check(
            checks, phase, "has_arms",
            _has_category_parts(assembly, ["arm", "臂", "gripper", "夹爪", "shoulder", "elbow", "wrist", "effector"]),
            "Arm-like parts present",
        )

    return result


def _phase2_position_solving(
    checks: list[dict],
    assembly: Any,
    positions: dict,
) -> None:
    """Validate position data quality.  Non-critical phase."""
    phase = "phase2"

    if not positions:
        _warn(checks, phase, "positions", "No positions data")
        return

    _check(
        checks, phase, "all_parts_positioned",
        len(positions) == len(assembly.parts),
        f"Positions: {len(positions)}/{len(assembly.parts)}",
    )

    nan_count = 0
    has_rotation = 0
    for _pname, pdata in positions.items():
        pos = pdata.get("position", [0, 0, 0])
        if any(isinstance(v, float) and (math.isnan(v) or math.isinf(v)) for v in pos):
            nan_count += 1
        if "rotation" in pdata:
            has_rotation += 1

    # NaN/Inf positions mean the solver failed catastrophically — output is garbage.
    _check(
        checks, phase, "no_nan_positions",
        nan_count == 0,
        f"NaN/Inf positions: {nan_count}",
        critical=True,
    )
    _warn(checks, phase, "rotation_data", f"Parts with rotation: {has_rotation}/{len(positions)}")

    # Outlier detection
    if positions:
        xs = [p.get("position", [0, 0, 0])[0] for p in positions.values()]
        ys = [p.get("position", [0, 0, 0])[1] for p in positions.values()]
        zs = [p.get("position", [0, 0, 0])[2] for p in positions.values()]
        cx, cy, cz = sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)
        outliers = 0
        for _pname, pdata in positions.items():
            pos = pdata.get("position", [0, 0, 0])
            dist = math.sqrt(
                (pos[0] - cx) ** 2 + (pos[1] - cy) ** 2 + (pos[2] - cz) ** 2
            )
            if dist > 2000:
                outliers += 1
        _warn(checks, phase, "outliers", f"Parts >2000mm from centroid: {outliers}")


def _phase3_render_quality(
    checks: list[dict],
    result: dict,
) -> None:
    """Validate render images exist and are non-trivial.  Non-critical phase."""
    phase = "phase3"

    production_render_dir = result.get("production_render_dir", "")
    renders: list[Path] = []

    if production_render_dir and os.path.isdir(production_render_dir):
        renders = list(Path(production_render_dir).glob("*.png"))
    else:
        # Fallback to VLM loop renders
        render_dir = result.get("render_dir", "")
        if render_dir and os.path.isdir(render_dir):
            round_dirs = sorted(
                [
                    d
                    for d in Path(render_dir).iterdir()
                    if d.is_dir() and d.name.startswith("round_")
                ],
                key=lambda d: d.name,
            )
            if round_dirs:
                renders = list(round_dirs[-1].glob("*.png"))

    if not renders:
        _skip(checks, phase, "render_check", "No render images found")
        return

    total_size = sum(f.stat().st_size for f in renders)
    avg_size = total_size / len(renders) if renders else 0

    _check(
        checks, phase, "render_count",
        len(renders) >= 3,
        f"Render views: {len(renders)} (min 3)",
    )

    _check(
        checks, phase, "render_quality",
        avg_size > 10_000,
        f"Average size: {avg_size / 1024:.1f}KB (min 10KB)",
    )


def _phase4_engineering_package(
    checks: list[dict],
    export_dir: str | None,
) -> None:
    """Validate required files and directories.  CRITICAL phase."""
    phase = "phase4"

    if not export_dir or not os.path.isdir(export_dir):
        _check(
            checks, phase, "pkg_exists", False,
            "No export directory", critical=True,
        )
        return

    _check(
        checks, phase, "pkg_exists", True,
        f"Export dir: {export_dir}", critical=True,
    )

    required_files = [
        "design_report.json",
        "bom.md",
        "assembly_guide.md",
        "urdf.xml",
        "README.md",
    ]
    for fname in required_files:
        fpath = Path(export_dir) / fname
        exists = fpath.exists()
        size = fpath.stat().st_size if exists else 0
        _check(
            checks, phase, f"pkg_{fname}",
            exists and size > 50,
            f"{fname}: {'exists' if exists else 'MISSING'} ({size} bytes)",
            critical=True,
        )

    required_dirs = ["freecad_scripts", "firmware", "stl_parts", "subsystems"]
    for dname in required_dirs:
        dpath = Path(export_dir) / dname
        exists = dpath.is_dir()
        n_files = len(list(dpath.iterdir())) if exists else 0
        _check(
            checks, phase, f"pkg_dir_{dname}",
            exists and n_files > 0,
            f"{dname}/: {n_files} files",
            critical=True,
        )


def _phase5_content_validation(
    checks: list[dict],
    assembly: Any,
    export_dir: str | None,
) -> None:
    """Validate content of key files.  Non-critical phase."""
    phase = "phase5"

    if not export_dir or not os.path.isdir(export_dir):
        _skip(checks, phase, "content_check", "No export directory")
        return

    # design_report.json
    report_path = Path(export_dir) / "design_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            _check(
                checks, phase, "report_mass",
                report.get("total_mass_kg", 0) > 0,
                f"Total mass: {report.get('total_mass_kg', 'N/A')} kg",
            )
            _check(
                checks, phase, "report_parts_match",
                report.get("total_parts", 0) == len(assembly.parts),
                f"Report parts: {report.get('total_parts', 'N/A')} vs assembly: {len(assembly.parts)}",
            )
            # VLM verification gate — stamped by generate_assembly_with_vlm_loop.
            # PASSED = VLM visual check passed at least once;
            # FAILED_MAX_ROUNDS = all rounds failed (package still exported for debugging);
            # UNKNOWN = older export without status reporting.
            verif_status = report.get("verification_status", "UNKNOWN")
            _check(
                checks, phase, "verification_status",
                verif_status == "PASSED",
                f"VLM verification status: {verif_status}",
                critical=True,
            )
            # Kinematic analysis — closed-chain loop detection + differential
            # drive inference. Must be present and converged (or have no loops).
            kin = report.get("kinematic_analysis") or {}
            if kin:
                _check(
                    checks, phase, "kinematic_analysis_present",
                    True,
                    f"Loops: {kin.get('loop_count', 0)}, "
                    f"converged: {kin.get('converged')}",
                )
                if kin.get("loop_count", 0) > 0:
                    _check(
                        checks, phase, "kinematic_loops_converged",
                        bool(kin.get("converged")),
                        f"Closed-chain error: {kin.get('error_mm', '?')}mm "
                        f"after {kin.get('iterations', '?')} iterations",
                    )
                if "differential_constraint" in kin:
                    dc = kin["differential_constraint"]
                    _check(
                        checks, phase, "differential_constraint_detected",
                        True,
                        f"Differential pair: {dc.get('left_wheel')}/"
                        f"{dc.get('right_wheel')} "
                        f"track={dc.get('track_width_mm')}mm",
                    )
        except json.JSONDecodeError:
            _warn(checks, phase, "report_json", "design_report.json is not valid JSON")

    # urdf.xml
    urdf_path = Path(export_dir) / "urdf.xml"
    if urdf_path.exists():
        content = urdf_path.read_text(encoding="utf-8")
        has_links = "<link" in content
        has_joints = "<joint" in content
        _check(
            checks, phase, "urdf_structure",
            has_links and has_joints,
            f"URDF has <link>: {has_links}, <joint>: {has_joints}",
        )

        # URDF joint origin sanity — catch parts placed hundreds of mm from
        # their parent (root cause of the 4dof_arm gripper_finger_left 322mm
        # offset that Phase 5 previously let through).  Threshold scales with
        # the parent part's largest dimension so a tiny servo can't justify a
        # 0.3m joint origin, while a large chassis still can.
        try:
            import xml.etree.ElementTree as ET

            from lang3d.tools.urdf_export import _sanitize_name

            root_elem = ET.fromstring(content)
            # Key by the sanitized name so it matches the ``link`` attribute
            # emitted in the URDF (names are lower-cased / de-punctuated).
            part_max_dim_m = {}
            for p in assembly.parts:
                d = p.dimensions or {}
                md = max(d.values()) if d else 0
                part_max_dim_m[_sanitize_name(p.name)] = md / 1000.0

            absurd = []
            for je in root_elem.findall(".//joint"):
                if je.get("type", "") not in ("revolute", "prismatic", "continuous"):
                    continue
                oe = je.find("origin")
                if oe is None:
                    continue
                xyz = oe.get("xyz", "0 0 0").split()
                if len(xyz) < 3:
                    continue
                try:
                    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
                except ValueError:
                    continue
                mag = math.sqrt(x * x + y * y + z * z)
                parent_el = je.find("parent")
                parent_link = (
                    parent_el.get("link", "") if parent_el is not None else ""
                )
                threshold = max(0.2, 2.0 * part_max_dim_m.get(parent_link, 0.1))
                if mag > threshold:
                    absurd.append({
                        "joint": je.get("name", "?"),
                        "mag_m": round(mag, 3),
                        "thresh_m": round(threshold, 3),
                    })
            _check(
                checks, phase, "urdf_origins_sane",
                len(absurd) == 0,
                f"Absurd movable-joint origins: {len(absurd)} "
                f"(threshold max(0.2m, 2x parent_dim))",
                critical=True,
            )
        except ET.ParseError as exc:
            _warn(checks, phase, "urdf_parse", f"URDF XML parse failed: {exc}")

    # FreeCAD scripts: check engineering features
    fc_dir = Path(export_dir) / "freecad_scripts"
    if fc_dir.is_dir():
        scripts = list(fc_dir.glob("*.py"))
        engineering_kw = ["Boolean", "cut", "fuse", "fillet", "chamfer", "hole", "shell"]
        complex_count = 0
        for script in scripts:
            try:
                content = script.read_text(encoding="utf-8")
                if any(kw in content for kw in engineering_kw):
                    complex_count += 1
            except Exception:
                pass
        ratio = complex_count / len(scripts) if scripts else 0
        _check(
            checks, phase, "script_complexity",
            ratio >= 0.5,
            f"Scripts with engineering features: {complex_count}/{len(scripts)} ({ratio:.0%})",
        )

    # Optional: trimesh STL quality
    stl_dir = Path(export_dir) / "stl_parts"
    if stl_dir.is_dir():
        stl_files = list(stl_dir.glob("*.stl"))
        if stl_files:
            try:
                import trimesh

                total_tri = 0
                watertight = 0
                for stl_file in stl_files:
                    mesh = trimesh.load(str(stl_file))
                    if hasattr(mesh, "faces"):
                        total_tri += len(mesh.faces)
                    if hasattr(mesh, "is_watertight") and mesh.is_watertight:
                        watertight += 1
                _check(
                    checks, phase, "stl_triangle_count",
                    total_tri > 10_000,
                    f"Total triangles: {total_tri:,}",
                )
                _check(
                    checks, phase, "stl_watertight_ratio",
                    watertight >= len(stl_files) * 0.6,
                    f"Watertight: {watertight}/{len(stl_files)}",
                )
                # Gripper fingers are functional parts — a non-watertight
                # finger STL means a broken mesh (open holes in the L-hook
                # tip) that renders as a thin shell and cannot physically
                # clamp an object.  The 60% overall ratio hides this: a run
                # with 4 broken fingers still passes at 89%.  Fingers must
                # ALL be watertight — no exceptions, no ratio.  AGENTS.md §5.1
                # (带夹爪的装配体必须 sim_grasp) presupposes intact geometry.
                finger_files = [f for f in stl_files if "finger" in f.stem.lower()]
                if finger_files:
                    finger_wt = sum(
                        1 for f in finger_files
                        if hasattr((m := trimesh.load(str(f))), "is_watertight")
                        and m.is_watertight
                    )
                    _check(
                        checks, phase, "gripper_finger_watertight",
                        finger_wt == len(finger_files),
                        f"Gripper fingers watertight: {finger_wt}/{len(finger_files)} "
                        "(ALL required — broken fingers cannot grasp)",
                        critical=True,
                    )
            except ImportError:
                _skip(checks, phase, "mesh_quality", "trimesh not installed")

    # STEP completeness: every exported STL must have a matching STEP.
    # STEP is the "production-level" deliverable (project expectation:
    # 生产级 3D 模型 STL/STEP).  A missing STEP means the FreeCAD script
    # raised during export (chamfer/fillet BRep_API failure) — a real bug,
    # not a cosmetic gap.  Previously Phase 4 only checked "step_parts/
    # exists", so 5/13 missing STEP files scored as PASS.  Now: the STEP
    # count must EQUAL the STL count.  No ratio, no floor — if a part's
    # CAD op fails, fix the op, don't relax the check.
    step_dir = Path(export_dir) / "step_parts"
    stl_dir = Path(export_dir) / "stl_parts"
    if stl_dir.is_dir() and step_dir.is_dir():
        stl_stems = {f.stem for f in stl_dir.glob("*.stl")}
        step_stems = {f.stem for f in step_dir.glob("*.step")}
        missing = sorted(stl_stems - step_stems)
        _check(
            checks, phase, "step_completeness",
            len(missing) == 0,
            f"STEP files: {len(step_stems)}/{len(stl_stems)}"
            + (f" (missing: {', '.join(missing[:5])})" if missing else ""),
            critical=True,
        )

    # VLM visual match check
    val_report_path = Path(export_dir) / "part_validation_report.json"
    if val_report_path.exists():
        try:
            val_data = json.loads(val_report_path.read_text(encoding="utf-8"))
            validation_results = val_data.get("results", [])
            if validation_results:
                vlm_verified_count = sum(
                    1 for r in validation_results if r.get("vlm_match") is not None
                )
                _check(
                    checks, phase, "vlm_match_executed",
                    vlm_verified_count > 0,
                    f"Parts with VLM verification: {vlm_verified_count}/{len(validation_results)}",
                    critical=False,
                )
        except json.JSONDecodeError:
            _warn(checks, phase, "vlm_report_json", "part_validation_report.json is not valid JSON")


def _phase6_physical_sanity(
    checks: list[dict],
    assembly: Any,
    positions: dict,
) -> None:
    """Collision detection, motion sweep, COM stability, and reachability.

    The static and tree-reachability checks are advisory; the motion-collision
    sweep, COM stability, and workspace-volume checks are ``critical=True`` so
    that a self-colliding, tipping, or kinematically degenerate robot fails
    validation rather than silently passing.
    """
    phase = "phase6"

    # Collision detection (optional)
    try:
        from lang3d.tools.mesh_collision import MeshCollisionChecker

        checker = MeshCollisionChecker()
        t0 = time.time()
        collision_result = checker.check_assembly_collisions(
            assembly=assembly,
            placements=positions,
            skip_adjacent=True,
        )
        dt = time.time() - t0

        collisions = collision_result.pairs if collision_result else []
        severe = [
            c for c in collisions if c.is_collision and c.penetration_depth_mm > 5.0
        ]
        _warn(
            checks, phase, "no_severe_collisions",
            f"Severe(>5mm): {len(severe)}, checked: {collision_result.pairs_checked} ({dt:.2f}s)",
        )
    except (ImportError, RuntimeError):
        _skip(checks, phase, "collision_detection", "MeshCollisionChecker not available")
    except Exception as exc:
        _warn(checks, phase, "collision_detection", f"Error: {exc}")

    # Motion sweep: sample each revolute joint across its range_deg and FCL
    # collide at every sample.  Catches self-collisions that only appear mid
    # motion (a static check at home pose cannot see them).  Requires
    # python-fcl; degrades to _warn with an install hint otherwise.
    #
    # Pass criterion: ZERO collisions across the joint motion sweep.
    # A collision during articulation is an interpenetration defect
    # (穿模) — two parts occupying the same space.  The previous criterion
    # ("each joint retains >=35% collision-free arc") let 6 colliding
    # joints score as CRITICAL PASS, hiding structural defects like arm
    # servos piercing the chassis body.  If a real design legitimately
    # self-collides at a workspace extreme, that is a design constraint to
    # fix (limit the joint range), not a reason to relax the test.  No
    # ratio, no usable-arc carve-out: collision_count == 0 or FAIL.
    try:
        from lang3d.tools.motion_collision import MotionCollisionChecker

        mc = MotionCollisionChecker(num_samples=5)
        motion_result = mc.check_motion_collisions(
            assembly=assembly, skip_adjacent=True,
        )
        colliding = [
            jr.joint_name for jr in motion_result.joint_results if jr.has_collision
        ]

        _check(
            checks, phase, "motion_collision_sweep",
            len(colliding) == 0,
            f"Motion sweep: {motion_result.joints_checked} joints, "
            f"{len(colliding)} with collisions"
            + (f" ({', '.join(colliding)})" if colliding else "")
            + (" — collision-free" if not colliding else " — 穿模 detected"),
            critical=True,
        )
    except (ImportError, RuntimeError) as exc:
        _warn(
            checks, phase, "motion_collision_sweep",
            f"Skipped (needs python-fcl trimesh): {exc}",
        )
    except Exception as exc:
        _warn(checks, phase, "motion_collision_sweep", f"Error: {exc}")

    # COM stability: assembly center of mass must project inside the support
    # polygon formed by the ground-contact parts, otherwise the robot tips
    # over.  Catches top-heavy / narrow-base assemblies a static collision
    # check cannot detect.
    try:
        from lang3d.agent.assembly_verifier import AssemblyVerifier

        verifier = AssemblyVerifier()
        com_result = verifier.check_center_of_mass_stability(assembly, positions)
        com_ok = bool(com_result.verified and com_result.inside_support_polygon)
        _check(
            checks, phase, "com_stability",
            com_ok,
            f"COM within support polygon: {com_ok} "
            f"({com_result.notes or 'n/a'})",
            critical=True,
        )
    except Exception as exc:
        _warn(checks, phase, "com_stability", f"Check error: {exc}")

    # Parts reachable from root
    if assembly.joints and positions:
        part_names = {p.name for p in assembly.parts}
        adj: dict[str, set[str]] = {n: set() for n in part_names}
        for j in assembly.joints:
            if j.parent in adj and j.child in adj:
                adj[j.parent].add(j.child)
                adj[j.child].add(j.parent)

        visited: set[str] = set()
        queue = deque([assembly.parts[0].name])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            queue.extend(adj.get(node, set()) - visited)

        reachable_ratio = len(visited) / len(part_names) if part_names else 0
        _check(
            checks, phase, "parts_reachable",
            reachable_ratio == 1.0,
            f"Reachable: {len(visited)}/{len(part_names)} ({reachable_ratio:.0%})",
        )

    # Workspace sampling: verify the arm can reach a non-trivial volume.
    # Drives each revolute joint to its range midpoint (others at home) and
    # measures the end-effector bounding-box max edge.  A degenerate assembly
    # whose joints don't actually move the end-effector collapses to a point
    # and fails here.  Pure FK — does not need FCL.
    try:
        from lang3d.tools.assembly_solver import AssemblySolver

        ee_candidates = ("gripper_base", "end_effector", "gripper")
        ee_name = next(
            (p.name for p in assembly.parts
             if any(k in p.name.lower() for k in ee_candidates)),
            assembly.parts[-1].name if assembly.parts else "",
        )

        part_names_set = {p.name for p in assembly.parts}
        solver = AssemblySolver(assembly)
        home = solver.solve()
        home_ee = home.get(ee_name, {}).get("position")

        samples = []
        if home_ee:
            samples.append(tuple(home_ee))

        rev_joints = [j for j in assembly.joints if j.type == "revolute"]
        for j in rev_joints:
            if not j.range_deg or j.child not in part_names_set:
                continue
            lo, hi = j.range_deg[0], j.range_deg[1]
            # Sample at both endpoints (and the midpoint when non-zero).  End
            # points are essential because many real arms declare symmetric
            # ranges (e.g. (-180, 180)) whose midpoint is 0 == home, which
            # would otherwise collapse the bounding box to a single point.
            for target in (lo, hi, (lo + hi) / 2.0):
                angles = dict(assembly.default_angles or {})
                angles[j.child] = target
                try:
                    placements = solver.solve(joint_angles=angles)
                    ee = placements.get(ee_name, {}).get("position")
                    if ee:
                        samples.append(tuple(ee))
                except Exception:
                    pass

        if len(samples) >= 2:
            xs = [s[0] for s in samples]
            ys = [s[1] for s in samples]
            zs = [s[2] for s in samples]
            bbox = max(
                max(xs) - min(xs),
                max(ys) - min(ys),
                max(zs) - min(zs),
            )
            largest_dim = max(
                (max((p.dimensions or {}).values(), default=0)
                 for p in assembly.parts),
                default=0,
            )
            threshold_mm = max(50.0, largest_dim * 0.5)
            _check(
                checks, phase, "workspace_nontrivial",
                bbox > threshold_mm,
                f"Workspace bbox max edge: {bbox:.0f}mm "
                f"(threshold {threshold_mm:.0f}mm)",
                critical=True,
            )
        else:
            _warn(
                checks, phase, "workspace_nontrivial",
                "Insufficient samples for workspace analysis",
            )
    except Exception as exc:
        _warn(checks, phase, "workspace_nontrivial", f"Error: {exc}")


# ---------------------------------------------------------------------------
# Phase 7: MuJoCo Simulation (added 2026-06-18)
# ---------------------------------------------------------------------------


def _phase7_mujoco_simulation(
    checks: list[dict],
    export_dir: str,
    min_joints: int,
    assembly: Any = None,
) -> None:
    """Load the URDF into MuJoCo and verify physics + joint actuation.

    The e2e pipeline is *headless* by design — interactive viewer is
    available via the CLI's ``/sim`` command on a generated run.
    """
    phase = "phase7"

    if not export_dir:
        _skip(checks, phase, "mujoco_loads", "No export directory")
        _skip(checks, phase, "mujoco_physics_stable", "No export directory")
        _skip(checks, phase, "mujoco_joints_actuate", "No export directory")
        return

    urdf_path = os.path.join(export_dir, "urdf.xml")
    if not os.path.isfile(urdf_path):
        _check(
            checks, phase, "mujoco_loads", False,
            f"URDF not found: {urdf_path}", critical=True,
        )
        return

    try:
        import mujoco  # type: ignore[import-not-found]
    except ImportError:
        _skip(checks, phase, "mujoco_loads",
              "mujoco not installed (pip install mujoco)")
        _skip(checks, phase, "mujoco_physics_stable", "mujoco not installed")
        _skip(checks, phase, "mujoco_joints_actuate", "mujoco not installed")
        return

    # Use the project's SimMujocoTool to avoid duplicating mesh-path logic
    try:
        from lang3d.tools.sim_mujoco import SimMujocoTool
        tool = SimMujocoTool()
        # interactive=False always — e2e must be non-blocking
        report_text = tool.execute(
            urdf_path=urdf_path,
            mode="validate",
            duration_sec=1.5,
            interactive=False,
        )
        # The tool returns a text report; the JSON summary is the last
        # fenced block.  Parse loosely: look for "JOINT_TEST" or "PHYSICS".
        report_lower = report_text.lower()

        loads_ok = "load failed" not in report_lower
        _check(
            checks, phase, "mujoco_loads", loads_ok,
            "URDF loaded into MuJoCo" if loads_ok else
            f"Load failed: {report_text[:200]}", critical=True,
        )
        if not loads_ok:
            return

        # Physics stable: prefer the structured JSON field
        # ``physics.stabilized`` over fragile string matching.  The old
        # check ('"unstable" not in report') false-failed because the
        # report legitimately contains ``"unstable": false`` for every
        # joint (meaning "this joint is NOT unstable"), which the
        # substring search counted as a failure signal.
        import json as _json
        import re as _re
        physics_stable: bool | None = None
        physics_detail = ""
        # Extract the last JSON block from the report
        _json_blocks = _re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", report_text)
        for _blk in reversed(_json_blocks):
            try:
                _doc = _json.loads(_blk)
                if isinstance(_doc, dict) and "physics" in _doc:
                    _ph = _doc["physics"]
                    if isinstance(_ph, dict) and "stabilized" in _ph:
                        physics_stable = bool(_ph["stabilized"])
                        physics_detail = (
                            f"stabilized={physics_stable}, "
                            f"err={_ph.get('max_qpos_error_deg', '?')}deg, "
                            f"disp={_ph.get('max_body_displacement_mm', '?')}mm"
                        )
                        break
            except (_json.JSONDecodeError, ValueError):
                continue

        if physics_stable is None:
            # Fallback: structured field not found — use a strict match that
            # only catches an explicit failure, not the "unstable": false
            # success records.  Look for "pd-hold fail" or a genuine
            # "unstable": true.
            _has_fail = "pd-hold fail" in report_lower
            _has_true_unstable = bool(_re.search(
                r'"unstable"\s*:\s*true', report_lower
            ))
            _has_nan_inf = bool(_re.search(
                r'\b(nan|inf)\b', report_lower
            ))
            physics_ok = not (_has_fail or _has_true_unstable or _has_nan_inf)
            physics_detail = "fallback string match"
        else:
            physics_ok = physics_stable
        _check(
            checks, phase, "mujoco_physics_stable", physics_ok,
            f"PD-hold physics stable ({physics_detail})" if physics_ok else
            f"Physics unstable ({physics_detail}): {report_text[:200]}",
            critical=True,
        )

        # Count actuated joints
        actuated = report_lower.count("joint_name") or report_lower.count("joint:")
        # Fallback: count "actuated: yes" patterns
        if not actuated:
            import re as _re
            actuated = len(_re.findall(r"\bactuated\b.*?\byes\b", report_lower))
        _check(
            checks, phase, "mujoco_joints_actuate",
            actuated >= min_joints,
            f"Actuated joints: {actuated} (min {min_joints})",
            critical=True,
        )

        # sim_grasp: if the assembly has a gripper (2 SLIDE finger joints),
        # run the three-phase grasp test (zero-g close → gravity hold →
        # lift) and require grasp_ok.  AGENTS.md §5.1: "带夹爪的装配体必须
        # sim_grasp, 不做不许标完成".  Previously this requirement had ZERO
        # e2e coverage — a robot with broken gripper STLs still scored
        # 92% PASS because no check ever exercised the grasp.  Now the
        # "能抓东西" project expectation is actually verified.
        has_gripper = any(
            getattr(j, "type", "") == "prismatic"
            and any("finger" in c.lower() for c in (j.child, j.parent))
            for j in assembly.joints
        )
        if not has_gripper:
            _skip(
                checks, phase, "sim_grasp",
                "No gripper (no finger prismatic joints) — grasp test N/A",
            )
        else:
            try:
                from lang3d.tools.sim_mujoco import SimGraspTool
                grasp_tool = SimGraspTool()
                grasp_report = grasp_tool.execute(
                    urdf_path=str(Path(urdf_path).resolve()),
                )
                # The tool emits an authoritative aggregate verdict:
                # "总体结论: PASS (所有 N 个夹爪均能抓取)" iff EVERY gripper
                # holds the cube; any failing gripper ⇒ "总体结论: FAIL".
                # Match that line case-insensitively — NOT per-gripper
                # "静态抓取: pass" (which would PASS if only one of several
                # grippers succeeded).  Single-gripper robots also emit
                # "总体结论: PASS", so this covers both cases.
                report_lower = grasp_report.lower()
                grasp_ok = "总体结论: pass" in report_lower
                # Extract the verdict line for the detail message.
                verdict_line = next(
                    (ln for ln in grasp_report.splitlines()
                     if "静态抓取" in ln or "总体结论" in ln),
                    grasp_report[:120],
                )
                _check(
                    checks, phase, "sim_grasp", grasp_ok,
                    f"Grasp test: {'cube held against gravity' if grasp_ok else 'FAILED'}"
                    f" — {verdict_line.strip()[:100]}",
                    critical=True,
                )
            except Exception as exc:
                _check(
                    checks, phase, "sim_grasp", False,
                    f"Grasp test error: {exc}", critical=True,
                )
    except Exception as exc:
        _check(
            checks, phase, "mujoco_loads", False,
            f"Exception during sim: {exc}", critical=True,
        )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_e2e_case(case: dict) -> dict:
    """Execute a single E2E test case and return the report dict."""
    test_id = case["id"]
    description = case["description"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    # Canonical layout: data/runs/<case_id>/<timestamp>/
    # (replaces the legacy data/e2e_results/<case>_<ts>/ split)
    output_dir = os.path.join("data", "runs", test_id, ts)
    os.makedirs(output_dir, exist_ok=True)

    checks: list[dict] = []

    print(f"\n{'=' * 70}")
    print(f"  E2E Test: {test_id}")
    print(f"  Input: {description}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 70}")

    # Phase 1: NL -> Assembly
    print(f"\n--- Phase 1: NL → Assembly (CRITICAL) ---")
    if args_pipeline_mode:
        result = _phase1_nl_to_assembly_pipeline(checks, case, output_dir)
    else:
        result = _phase1_nl_to_assembly(checks, case, output_dir)

    if result is None:
        score = _compute_score(checks)
        _save_report(output_dir, test_id, description, checks, score)
        return {
            "test_id": test_id,
            "score": score,
            "checks": checks,
            "output_dir": output_dir,
            "critical_fail": True,
        }

    assembly = result.get("assembly")
    positions = result.get("positions", {})
    export_dir = result.get("export_dir")

    # Phase 2: Position Solving
    print(f"\n--- Phase 2: Position Solving ---")
    _phase2_position_solving(checks, assembly, positions)

    # Phase 3: Render Quality
    print(f"\n--- Phase 3: Render Quality ---")
    _phase3_render_quality(checks, result)

    # Phase 4: Engineering Package (CRITICAL)
    print(f"\n--- Phase 4: Engineering Package (CRITICAL) ---")
    _phase4_engineering_package(checks, export_dir)

    # Phase 5: Content Validation
    print(f"\n--- Phase 5: Content Validation ---")
    _phase5_content_validation(checks, assembly, export_dir)

    # Phase 6: Physical Sanity
    print(f"\n--- Phase 6: Physical Sanity ---")
    _phase6_physical_sanity(checks, assembly, positions)

    # Phase 7: MuJoCo Simulation (added 2026-06-18)
    print(f"\n--- Phase 7: MuJoCo Simulation ---")
    _phase7_mujoco_simulation(checks, export_dir or "", case["min_joints"], assembly)

    # Compute score and save report
    score = _compute_score(checks)
    report_path = _save_report(output_dir, test_id, description, checks, score)

    # Determine if any critical check failed
    critical_fails = [
        c for c in checks if c["status"] == FAIL and c.get("critical")
    ]

    # Summary
    passed_n = sum(1 for c in checks if c["status"] == PASS)
    failed_n = sum(1 for c in checks if c["status"] == FAIL)
    warn_n = sum(1 for c in checks if c["status"] == WARN)
    skip_n = sum(1 for c in checks if c["status"] == SKIP)

    print(f"\n{'=' * 70}")
    print(f"  Result: {test_id}")
    print(f"  Score: {score:.1f}%")
    print(f"  Checks: {len(checks)} total, {passed_n} pass, {failed_n} fail, "
          f"{warn_n} warn, {skip_n} skip")
    if critical_fails:
        print(f"  CRITICAL FAILURES: {len(critical_fails)}")
        for cf in critical_fails:
            print(f"    FAIL {cf['step']}: {cf['detail']}")
    print(f"  Report: {report_path}")
    print(f"{'=' * 70}")

    return {
        "test_id": test_id,
        "score": score,
        "checks": checks,
        "output_dir": output_dir,
        "critical_fail": len(critical_fails) > 0,
    }


# ---------------------------------------------------------------------------
# Pytest integration
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.parametrize(
    "case",
    ROBOT_TEST_CASES,
    ids=[c["id"] for c in ROBOT_TEST_CASES],
)
def test_e2e_production(case: dict) -> None:
    """Standardized E2E production test — parameterized over robot types."""
    api_key = os.environ.get("GLM_API_KEY", "")
    if not api_key:
        pytest.skip("GLM_API_KEY not set — skipping E2E test")

    report = run_e2e_case(case)

    # Phase 1 & 4 failures cause test FAIL
    critical_fails = [c for c in report["checks"] if c["status"] == FAIL and c.get("critical")]
    assert not critical_fails, (
        f"{len(critical_fails)} critical check(s) failed for '{case['id']}':\n"
        + "\n".join(f"  - {c['step']}: {c['detail']}" for c in critical_fails)
    )


# ---------------------------------------------------------------------------
# Standalone script entry point
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(description="Language-3D E2E Production Test")
    parser.add_argument("--case", type=str, default=None, help="Run a single test case by ID")
    parser.add_argument("--list", action="store_true", help="List available test cases")
    parser.add_argument(
        "--pipeline", action="store_true",
        help="Use the multi-agent AssemblyPipeline instead of the legacy "
             "generate_assembly_with_vlm_loop (Step 2 architecture).",
    )
    args = parser.parse_args()

    global args_pipeline_mode
    args_pipeline_mode = args.pipeline

    if args.list:
        print("Available E2E test cases:")
        for c in ROBOT_TEST_CASES:
            print(f"  {c['id']}: {c['description']}")
        return

    api_key = os.environ.get("GLM_API_KEY", "")
    if not api_key:
        print("ERROR: GLM_API_KEY not set.  Export it or add to .env file.")
        sys.exit(2)

    if args.case:
        cases = [c for c in ROBOT_TEST_CASES if c["id"] == args.case]
        if not cases:
            print(f"ERROR: Unknown case '{args.case}'.  Use --list to see available cases.")
            sys.exit(2)
    else:
        cases = ROBOT_TEST_CASES

    all_results = []
    has_critical_fail = False

    for case in cases:
        report = run_e2e_case(case)
        all_results.append(report)
        if report["critical_fail"]:
            has_critical_fail = True

    # Final summary
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    for r in all_results:
        status = "FAIL" if r["critical_fail"] else "PASS"
        print(f"  [{status}] {r['test_id']}: score={r['score']:.1f}%")
    print(f"{'=' * 70}")

    sys.exit(1 if has_critical_fail else 0)


if __name__ == "__main__":
    _main()
