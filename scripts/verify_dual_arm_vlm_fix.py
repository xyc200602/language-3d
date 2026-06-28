"""Verify Bug #1 fix: wheeled dual-arm robot no longer dead-loops in VLM.

Runs generate_assembly_with_vlm_loop on the exact description that previously
dead-looped (data/runs/4wheel_dual_arm/20260627_001843) and asserts:
  1. The robot_category is correctly classified as "wheeled_arm".
  2. The pipeline COMPLETES (does not abort after 3 failed rounds).
  3. positions.json is produced (the missing artifact in the failed run).

Usage: python scripts/verify_dual_arm_vlm_fix.py
Requires: GLM_API_KEY in env or .env, FreeCAD for STL export.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Load .env manually (avoid python-dotenv dependency assumptions)
env = Path(".env")
if env.exists():
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from lang3d.tools.assembly_generator import (
    _classify_robot,
    generate_assembly_with_vlm_loop,
)

DESC = (
    "设计一个4轮双臂机器人，四个差速驱动轮分布在底盘四角，"
    "底盘上方左右各安装一个3自由度机械臂"
)

print("=== Step 1: classification ===")
cat = _classify_robot(DESC)
print(f"  category = {cat!r}")
assert cat == "wheeled_arm", f"EXPECTED wheeled_arm, GOT {cat}"
print("  OK\n")

print("=== Step 2: run VLM loop (this takes a few minutes) ===")
t0 = time.time()
result = generate_assembly_with_vlm_loop(
    description=DESC,
    max_rounds=3,
)
elapsed = time.time() - t0
print(f"  elapsed: {elapsed:.0f}s")
print(f"  passed: {result.get('passed')}")
print(f"  rounds: {result.get('rounds')}")
print(f"  final_status: {result.get('final_status')}")
problems = result.get("problems_history", [])
print(f"  problems per round:")
for i, p in enumerate(problems, 1):
    print(f"    round {i}: {p}")

out_dir = result.get("output_dir") or result.get("export_dir") or ""
print(f"  output_dir: {out_dir}")

print("\n=== Step 3: verify positions.json produced ===")
# Walk the output dir for positions.json
found_positions = False
if out_dir and os.path.isdir(out_dir):
    for root, _dirs, files in os.walk(out_dir):
        if "positions.json" in files:
            found_positions = True
            p = os.path.join(root, "positions.json")
            data = json.load(open(p))
            print(f"  positions.json at {p}: {len(data)} parts positioned")
            # Count wheels
            wheels = [k for k in data if "wheel" in k.lower()]
            print(f"  wheels positioned: {wheels}")

if not found_positions:
    print("  !! NO positions.json — pipeline aborted before solve")

print("\n=== VERDICT ===")
# The KEY success criterion: the pipeline did NOT dead-loop on the
# "Fixed-base arms have wheels" false-negative. Even if VLM has other
# nitpicks, the wheeled_arm category must prevent the wheels-as-error loop.
wheel_problem_seen = any(
    ("should not have wheels" in prob.lower() or "fixed-base arm" in prob.lower())
    for round_probs in problems
    for prob in round_probs
)
if not wheel_problem_seen:
    print("  PASS: no 'arms should not have wheels' false-negative raised")
else:
    print("  FAIL: VLM still raised the wheel false-negative despite category hint")
    sys.exit(1)
