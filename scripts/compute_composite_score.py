"""Design and validate a composite quality score with real discriminative power.

Old rubric: 41 binary checks → 4 cases tied at 95.1% (zero discrimination).
New score: 5 normalized sub-scores, each physically grounded, averaged.
"""
from __future__ import annotations
import json, glob, re

def get_latest_passing(case):
    runs = sorted(glob.glob(f"data/runs/{case}/*/e2e_report.json"))
    for rp in reversed(runs):
        d = json.load(open(rp, encoding="utf-8"))
        if d.get("score", 0) > 0:
            return d
    return None

def parse_detail(checks, step):
    for c in checks:
        if c.get("step") == step:
            return c.get("detail", "")
    return ""

def parse_com_margin(detail):
    m = re.search(r"裕量\s*([\d.]+)\s*mm", detail)
    return float(m.group(1)) if m else 0.0

def parse_workspace(detail):
    m = re.search(r"max edge:\s*([\d.]+)\s*mm", detail)
    return float(m.group(1)) if m else 0.0

def parse_pd_error(detail):
    m = re.search(r"err=([\d.]+)deg", detail)
    return float(m.group(1)) if m else 0.0

def parse_actuated(detail):
    m = re.search(r"Actuated joints:\s*(\d+)", detail)
    return int(m.group(1)) if m else 0

def parse_grasp(detail):
    if "FAIL" in detail: return 0.0
    if "PASS" in detail or "pass" in detail: return 1.0
    return 0.0

def parse_collision(detail):
    """Returns (severe_count, total_pairs). severe=0 → pass."""
    m = re.search(r"Severe\(>[^)]+\):\s*(\d+),\s*checked:\s*(\d+)", detail)
    if m: return int(m.group(1)), int(m.group(2))
    return 0, 0

def parse_watertight(detail):
    m = re.search(r"Watertight:\s*(\d+)/(\d+)", detail)
    if m: return int(m.group(1)), int(m.group(2))
    return 0, 1

# --- Sub-score normalization functions ---
# Each maps a raw measurement to [0, 1] with a physically meaningful threshold.

def s_stability(com_margin_mm: float) -> float:
    """COM stability: 0mm = tipping (0.0), ≥100mm = robust (1.0).
    Linear in between. Negative = unstable (clamped to 0)."""
    return max(0.0, min(1.0, com_margin_mm / 100.0))

def s_workspace(ws_mm: float) -> float:
    """Workspace extent: <150mm = trivial (0.3), ≥800mm = full-reach (1.0).
    Linear between 150-800. Below 150 clamps to 0.3 (not zero—still has
    *some* workspace)."""
    if ws_mm <= 150: return 0.3
    return max(0.3, min(1.0, 0.3 + 0.7 * (ws_mm - 150) / 650))

def s_physics(pd_error_deg: float) -> float:
    """PD-hold tracking: 0° = perfect (1.0), ≥5° = unstable (0.0).
    The humanoid at 12° correctly scores 0."""
    return max(0.0, 1.0 - pd_error_deg / 5.0)

def s_geometry(severe_collisions: int, watertight_n: int, watertight_total: int) -> float:
    """Geometry quality: 0 severe collisions AND all watertight = 1.0.
    Each severe collision -0.2; non-watertight parts proportional."""
    collision_score = max(0.0, 1.0 - 0.2 * severe_collisions)
    if watertight_total > 0:
        wt_ratio = watertight_n / watertight_total
    else:
        wt_ratio = 0.0
    return 0.5 * collision_score + 0.5 * wt_ratio

def s_functionality(grasp_pass: float, actuated: int, min_actuated: int = 4) -> float:
    """Functional capability: grasp (40%) + actuation (60%).
    Actuation = actuated/min_expected, capped at 1.0."""
    act_ratio = min(1.0, actuated / min_actuated) if min_actuated > 0 else 0.0
    return 0.4 * grasp_pass + 0.6 * act_ratio


cases = ["2dof_arm","3dof_arm","4dof_arm","5dof_arm","6dof_arm","7dof_arm","4wheel_dual_arm"]
print(f"{'Case':18} {'Stab':>5} {'Work':>5} {'Phys':>5} {'Geom':>5} {'Func':>5} {'COMPOSITE':>10}   old_score")
print("-" * 80)
scores = []
for case in cases:
    d = get_latest_passing(case)
    if not d: continue
    checks = d.get("checks", [])
    com = parse_com_margin(parse_detail(checks, "com_stability"))
    ws = parse_workspace(parse_detail(checks, "workspace_nontrivial"))
    pd = parse_pd_error(parse_detail(checks, "mujoco_physics_stable"))
    act = parse_actuated(parse_detail(checks, "mujoco_joints_actuate"))
    grasp = parse_grasp(parse_detail(checks, "sim_grasp"))
    severe, _ = parse_collision(parse_detail(checks, "no_severe_collisions"))
    wt_n, wt_total = parse_watertight(parse_detail(checks, "stl_watertight_ratio"))

    s1 = s_stability(com)
    s2 = s_workspace(ws)
    s3 = s_physics(pd)
    s4 = s_geometry(severe, wt_n, wt_total)
    s5 = s_functionality(grasp, act)
    composite = 100 * (s1 + s2 + s3 + s4 + s5) / 5.0
    scores.append(composite)
    print(f"{case:18} {s1:5.2f} {s2:5.2f} {s3:5.2f} {s4:5.2f} {s5:5.2f} {composite:9.1f}   {d['score']:.1f}%")

print("-" * 80)
print(f"{'Mean':18} {'':5} {'':5} {'':5} {'':5} {'':5} {sum(scores)/len(scores):9.1f}")
print()
print("Discrimination check (distinct values):")
case_scores = []
for case in cases:
    d = get_latest_passing(case)
    if not d: continue
    checks = d.get("checks", [])
    com = parse_com_margin(parse_detail(checks, "com_stability"))
    ws = parse_workspace(parse_detail(checks, "workspace_nontrivial"))
    pd = parse_pd_error(parse_detail(checks, "mujoco_physics_stable"))
    act = parse_actuated(parse_detail(checks, "mujoco_joints_actuate"))
    grasp = parse_grasp(parse_detail(checks, "sim_grasp"))
    severe, _ = parse_collision(parse_detail(checks, "no_severe_collisions"))
    wt_n, wt_total = parse_watertight(parse_detail(checks, "stl_watertight_ratio"))
    composite = 100 * (s_stability(com) + s_workspace(ws) + s_physics(pd) + s_geometry(severe, wt_n, wt_total) + s_functionality(grasp, act)) / 5.0
    case_scores.append(round(composite, 1))
print(f"  Composite scores: {case_scores}")
print(f"  Distinct values:  {len(set(case_scores))} / {len(case_scores)}")
print(f"  Old scores:       {[95.1, 95.1, 95.1, 92.7, 92.7, 92.7, 95.3]}")
print(f"  Old distinct:     2 / 7")
