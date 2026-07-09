"""Backfill correct actuated_joints + arm_dof into BENCHMARK e2e reports.

The prior string-match count double-counted the gripper's mimic finger,
reporting actuated=6 for a 4-DOF arm (visibly wrong in Table I). Recompute
from the assembly: arm_dof = revolute count; actuated_joints = revolute +
non-mimic prismatic. Keeps a .bak backup.
"""
import json
import shutil
from pathlib import Path

CASES = ["2dof_arm", "3dof_arm", "4dof_arm", "5dof_arm", "6dof_arm", "7dof_arm", "4wheel_dual_arm"]
for case in CASES:
    ts = Path(f"data/runs/{case}/BENCHMARK").read_text(encoding="utf-8").splitlines()[0].strip()
    base = Path(f"data/runs/{case}/{ts}")
    rp = base / "e2e_report.json"
    if not rp.exists():
        print(f"{case}: no report")
        continue
    asm = json.load(open(base / "assembly.json", encoding="utf-8"))
    _WHEEL = ("wheel", "轮")
    # arm revolute (excluding wheels), wheel revolute, driven prismatic
    arm_rev = 0
    n_wheels = 0
    for j in asm["joints"]:
        if j["type"] in ("revolute", "continuous"):
            if any(kw in (j.get("child") or "").lower() for kw in _WHEEL):
                n_wheels += 1
            else:
                arm_rev += 1
    driven_p = sum(1 for j in asm["joints"] if j["type"] == "prismatic" and not j.get("mimic_joint"))
    total = arm_rev + n_wheels + driven_p

    doc = json.loads(rp.read_text(encoding="utf-8"))
    chk = next((c for c in doc.get("checks", []) if c.get("step") == "mujoco_joints_actuate"), None)
    if chk is None:
        print(f"{case}: no mujoco_joints_actuate check")
        continue
    old_arm = chk.get("metrics", {}).get("arm_dof", "?")
    bak = rp.with_suffix(".json.bak2")
    if not bak.exists():
        shutil.copy2(rp, bak)
    chk.setdefault("metrics", {})
    chk["metrics"]["actuated_joints"] = total
    chk["metrics"]["arm_dof"] = arm_rev
    extra = f", {n_wheels} wheels" if n_wheels else ""
    chk["detail"] = f"Actuated joints: {total} (arm DOF {arm_rev}{extra}; min {chk['metrics'].get('min_expected', '?')})"
    rp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{case:18}: arm_dof {old_arm} -> {arm_rev} (wheels={n_wheels}, total_actuated={total})")
