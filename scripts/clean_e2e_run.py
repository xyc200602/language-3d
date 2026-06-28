"""Clean single-case e2e: run one case to FULL completion, assert all
outputs exist (asm/pos/stl/render), report honestly."""
from __future__ import annotations
import os, sys, time, json, re
from pathlib import Path
for line in Path(".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k,_,v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from lang3d.tools.assembly_generator import generate_assembly_with_vlm_loop

CASE = sys.argv[1] if len(sys.argv) > 1 else "4dof_arm"
DESCS = {
    "4dof_arm": "设计一个4自由度机械臂，带一个平行夹爪",
    "4wheel_dual_arm": "设计一个4轮双臂机器人，四个差速驱动轮分布在底盘四角，底盘上方左右各安装一个3自由度机械臂",
}
desc = DESCS[CASE]
print(f"=== clean e2e: {CASE} ===", flush=True)
t0 = time.time()
r = generate_assembly_with_vlm_loop(description=desc, max_rounds=3)
dt = time.time() - t0
out = r.get("production_render_dir") or r.get("export_dir") or r.get("render_dir") or ""
# find the actual run dir (parent of render_dir or export_dir)
run_dir = out
# walk up to the timestamp dir (YYYYMMDD_HHMMSS)
while run_dir and not re.match(r"^[0-9]{8}_[0-9]{6}$", os.path.basename(run_dir)):
    run_dir = os.path.dirname(run_dir)
    if run_dir == os.path.dirname(run_dir): break
print(f"\n=== DONE {dt:.0f}s ===", flush=True)
print(f"passed={r.get('passed')} status={r.get('final_status')} rounds={r.get('rounds')}", flush=True)
print(f"run_dir={run_dir}", flush=True)
# completeness check
checks = {
    "assembly.json": os.path.exists(os.path.join(run_dir,"assembly.json")),
    "positions.json": os.path.exists(os.path.join(run_dir,"positions.json")),
    "stl_parts/": os.path.isdir(os.path.join(run_dir,"engineering_package","stl_parts")),
    "production_renders/": os.path.exists(os.path.join(run_dir,"production_renders","isometric.png")),
    "vlm_loop_summary.json": os.path.exists(os.path.join(run_dir,"vlm_loop_summary.json")),
}
print("outputs:", flush=True)
for k,v in checks.items():
    print(f"  {'OK ' if v else 'MISSING'} {k}", flush=True)
problems = r.get("problems_history",[])
for i,p in enumerate(problems,1):
    print(f"  round {i} ({len(p)} problems): {[x[:60] for x in p[:4]]}", flush=True)
all_ok = all(checks.values()) and r.get("passed")
print(f"\nVERDICT: {'SUCCESS' if all_ok else 'INCOMPLETE'}", flush=True)
sys.exit(0 if all_ok else 1)
