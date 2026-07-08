"""Extract continuous quantitative metrics from all e2e run reports.

These metrics are ALREADY computed by the pipeline but collapsed into
binary PASS/FAIL. This script extracts the raw numbers so we can report
them as a proper quantitative evaluation table instead of a self-score.
"""
from __future__ import annotations
import json, glob, re
from pathlib import Path
from collections import defaultdict

def extract_detail(checks, step_name):
    """Find a check by step name and return its detail string."""
    for c in checks:
        if c.get("step") == step_name:
            return c.get("detail", "")
    return ""

def parse_com_margin(detail):
    """Parse COM stability margin_mm from detail string."""
    # e.g. "COM=(0.0,-119.0,45.4)mm, 在支撑多边形内 (裕量 60.0mm)"
    m = re.search(r"裕量\s*([\d.]+)\s*mm", detail)
    if m:
        return float(m.group(1))
    m = re.search(r"margin\s*[:=]?\s*([\d.]+)\s*mm", detail, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None

def parse_workspace(detail):
    """Parse workspace bbox max edge from detail string."""
    m = re.search(r"max edge:\s*([\d.]+)\s*mm", detail)
    if m:
        return float(m.group(1))
    return None

def parse_collision(detail):
    """Parse FCL collision info: (severe_count, total_checked)."""
    m = re.search(r"Severe\(>[^)]+\):\s*(\d+),\s*checked:\s*(\d+)", detail)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def parse_motion_sweep(detail):
    """Parse motion collision sweep: (joints, collisions)."""
    m = re.search(r"(\d+)\s+joints?,\s*(\d+)\s+with collisions", detail)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def parse_pd_hold(detail):
    """Parse PD-hold error and displacement."""
    m = re.search(r"err=([\d.]+)deg,\s*disp=([\d.]+)mm", detail)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None

def parse_actuated(detail):
    """Parse actuated DOF count."""
    m = re.search(r"Actuated joints:\s*(\d+)", detail)
    if m:
        return int(m.group(1))
    return None

def parse_watertight(detail):
    """Parse watertight ratio: (watertight, total)."""
    m = re.search(r"Watertight:\s*(\d+)/(\d+)", detail)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def parse_triangles(detail):
    """Parse total triangle count."""
    m = re.search(r"Total triangles:\s*([\d,]+)", detail)
    if m:
        return int(m.group(1).replace(",", ""))
    return None

def parse_reachable(detail):
    """Parse reachable parts: (reachable, total)."""
    m = re.search(r"Reachable:\s*(\d+)/(\d+)", detail)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

# Collect latest run per case
cases = defaultdict(list)
for report_path in sorted(Path("data/runs").rglob("e2e_report.json")):
    case = report_path.parent.parent.name
    try:
        d = json.load(open(report_path, encoding="utf-8"))
        cases[case].append((d.get("timestamp", ""), d))
    except Exception:
        pass

print("=" * 90)
print("QUANTITATIVE METRICS EXTRACTION (continuous values behind the self-score)")
print("=" * 90)

for case in sorted(cases):
    # Use the median-scoring run (the typical outcome), matching the paper's
    # Table I caption ("median-scoring run per case"). Prefer runs that carry
    # a structured metrics block; fall back to all non-zero runs.
    runs = sorted(cases[case], key=lambda x: x[0])
    nonzero = [(ts, d) for ts, d in runs if d.get("score", 0) > 0]
    if not nonzero:
        continue
    metrics_runs = [(ts, d) for ts, d in nonzero
                    if any("metrics" in c for c in d.get("checks", []))]
    pool = metrics_runs if metrics_runs else nonzero
    pool.sort(key=lambda pair: pair[1].get("score", 0))
    best = pool[len(pool) // 2][1]  # median-scoring run
    checks = best.get("checks", [])
    score = best.get("score", 0)

    com_margin = parse_com_margin(extract_detail(checks, "com_stability"))
    workspace = parse_workspace(extract_detail(checks, "workspace_nontrivial"))
    severe, total_checked = parse_collision(extract_detail(checks, "no_severe_collisions"))
    m_joints, m_collisions = parse_motion_sweep(extract_detail(checks, "motion_collision_sweep"))
    pd_err, pd_disp = parse_pd_hold(extract_detail(checks, "mujoco_physics_stable"))
    actuated = parse_actuated(extract_detail(checks, "mujoco_joints_actuate"))
    wt_n, wt_total = parse_watertight(extract_detail(checks, "stl_watertight_ratio"))
    triangles = parse_triangles(extract_detail(checks, "stl_triangle_count"))
    reach_n, reach_total = parse_reachable(extract_detail(checks, "parts_reachable"))

    # grasp detail
    grasp_detail = extract_detail(checks, "sim_grasp")
    grasp_is_static = "静态抓取" in grasp_detail or "held against gravity" in grasp_detail
    grasp_is_lift = "抬升" in grasp_detail and "PASS" in grasp_detail and "失败" not in grasp_detail.split("抬升")[0]

    print(f"\n--- {case} (score={score}%) ---")
    print(f"  COM stability margin:     {com_margin} mm" if com_margin else "  COM stability margin:     N/A")
    print(f"  Workspace extent:         {workspace} mm" if workspace else "  Workspace extent:         N/A")
    print(f"  FCL collisions:           {severe} severe / {total_checked} pairs" if severe is not None else "  FCL collisions:           N/A")
    print(f"  Motion sweep:             {m_collisions}/{m_joints} joints collide" if m_joints else "  Motion sweep:             N/A")
    print(f"  PD-hold tracking error:   {pd_err}° / {pd_disp}mm" if pd_err is not None else "  PD-hold tracking error:   N/A")
    print(f"  Actuated DOFs:            {actuated}" if actuated else "  Actuated DOFs:            N/A")
    print(f"  STL watertight:           {wt_n}/{wt_total}" if wt_n else "  STL watertight:           N/A")
    print(f"  Triangle count:           {triangles:,}" if triangles else "  Triangle count:           N/A")
    print(f"  Parts reachable:          {reach_n}/{reach_total}" if reach_n else "  Parts reachable:          N/A")
    print(f"  Grasp:                    {'static-hold PASS (lift NOT gated)' if grasp_is_static else 'FAIL' if 'FAIL' in grasp_detail else 'N/A'}")
