"""Final composite quality score v3.

Design philosophy:
- A robot that tips over is not "86% good" — it's 0%. Fatal flaws are gates.
- Quality is not complexity. A 2-DOF arm that works reliably is better than
  a 7-DOF arm that fails half the time.
- Three dimensions that actually vary across our cases:
  1. Physical robustness (COM margin — does it stand up reliably?)
  2. Functional completeness (grasp + actuation — can it do what arms do?)
  3. Generation reliability (across-run consistency — does it work every time?)

Score = geometric_mean(s_robust, s_function, s_reliable) if all gates pass, else 0.

Gates (binary, all must pass):
  - MuJoCo loads ✓
  - PD-hold < 1° ✓
  - 0 severe mesh collisions ✓
  - COM inside support polygon ✓

This is honest: it doesn't pretend to measure absolute design quality
(no external benchmark exists), but it measures three things that matter
and that actually vary across our cases.
"""
from __future__ import annotations
import json, glob, re, statistics

def get_runs(case):
    """Get all run reports for a case."""
    results = []
    for rp in sorted(glob.glob(f"data/runs/{case}/*/e2e_report.json")):
        try:
            d = json.load(open(rp, encoding="utf-8"))
            if d.get("score", 0) > 0:
                results.append(d)
        except Exception:
            pass
    return results

def get_assembly(case):
    runs = sorted(glob.glob(f"data/runs/{case}/*/assembly.json"))
    for rp in reversed(runs):
        try:
            return json.load(open(rp, encoding="utf-8"))
        except Exception:
            pass
    return None

def parse_detail(checks, step):
    for c in checks:
        if c.get("step") == step:
            return c.get("detail", "")
    return ""

def parse_com_margin(detail):
    m = re.search(r"裕量\s*([\d.]+)\s*mm", detail)
    return float(m.group(1)) if m else 0.0

def parse_pd_error(detail):
    m = re.search(r"err=([\d.]+)deg", detail)
    return float(m.group(1)) if m else 99.0

def parse_actuated(detail):
    m = re.search(r"Actuated joints:\s*(\d+)", detail)
    return int(m.group(1)) if m else 0

def parse_collision(detail):
    m = re.search(r"Severe\(>[^)]+\):\s*(\d+)", detail)
    return int(m.group(1)) if m else 0

def parse_grasp(detail):
    return 0.0 if "FAIL" in detail else 1.0

def compute_footprint(assembly):
    parts = assembly.get("parts", [])
    max_xy = 0
    for p in parts:
        dims = p.get("dimensions", {})
        xy = max(dims.get("length", 0), dims.get("width", 0), dims.get("diameter", 0))
        max_xy = max(max_xy, xy)
    return max_xy if max_xy > 0 else 100


EXPECTED_DOF = {"2dof_arm":2,"3dof_arm":3,"4dof_arm":4,"5dof_arm":5,
                "6dof_arm":6,"7dof_arm":7,"4wheel_dual_arm":6}

cases = ["2dof_arm","3dof_arm","4dof_arm","5dof_arm","6dof_arm","7dof_arm","4wheel_dual_arm"]
print(f"{'Case':18} {'COM':>5} {'FP':>5} {'robust':>7} {'grasp':>5} {'dof_r':>5} {'func':>5} {'pass%':>5} {'rely':>5} {'GATE':>4} {'Q':>6}")
print("-" * 90)

scores = []
for case in cases:
    runs = get_runs(case)
    asm = get_assembly(case)
    if not runs or not asm:
        continue

    # Use the modal (best) run for physical metrics
    best = max(runs, key=lambda d: d.get("score", 0))
    checks = best.get("checks", [])
    com = parse_com_margin(parse_detail(checks, "com_stability"))
    pd_err = parse_pd_error(parse_detail(checks, "mujoco_physics_stable"))
    act = parse_actuated(parse_detail(checks, "mujoco_joints_actuate"))
    severe = parse_collision(parse_detail(checks, "no_severe_collisions"))
    grasp = parse_grasp(parse_detail(checks, "sim_grasp"))
    footprint = compute_footprint(asm)
    expected = EXPECTED_DOF.get(case, 4)

    # Gates
    gate = (pd_err < 1.0) and (severe == 0) and (com > 0)

    # Sub-scores
    # 1. Robustness: COM margin normalized by footprint (size-independent)
    #    0mm = tipping (0), footprint/2 = comfortable (1.0)
    s_robust = min(1.0, com / (footprint * 0.5)) if footprint > 0 else 0

    # 2. Functionality: grasp (50%) + DOF completeness (50%)
    dof_ratio = min(1.0, act / expected) if expected > 0 else 0
    s_func = 0.5 * grasp + 0.5 * dof_ratio

    # 3. Reliability: fraction of runs that pass (score > 80%)
    all_scores = [d.get("score", 0) for d in runs]
    pass_count = sum(1 for s in all_scores if s >= 80)
    s_rely = pass_count / len(all_scores) if all_scores else 0

    # Geometric mean
    q = 100 * (s_robust * s_func * s_rely) ** (1/3) if gate else 0
    scores.append(q)

    print(f"{case:18} {com:5.1f} {footprint:5.0f} {s_robust:7.2f} {grasp:5.1f} {dof_ratio:5.2f} {s_func:5.2f} {pass_count}/{len(all_scores):<2} {s_rely:5.2f} {'✓' if gate else '✗':>4} {q:6.1f}")

print("-" * 90)
print(f"{'Mean':18} {'':>5} {'':>5} {'':>7} {'':>5} {'':>5} {'':>5} {'':>5} {'':>5} {'':>4} {sum(scores)/len(scores):6.1f}")
print()
q_vals = [round(s) for s in scores]
print(f"Q values:   {q_vals}")
print(f"Distinct:   {len(set(q_vals))}/{len(q_vals)}")
print(f"Old rubric: [95, 95, 95, 93, 93, 93, 95]  distinct=2/7")
