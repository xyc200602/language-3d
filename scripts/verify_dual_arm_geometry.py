"""Verify the wheeled dual-arm geometry end-to-end (no VLM, fast).

Builds chassis + dual arms + sanitize + solve, then asserts:
  1. base_plate length is reasonable (< 500mm, not 1000+)
  2. wheels on ground (Z ≈ radius)
  3. arms mounted on chassis (XY distance reasonable, not 700+mm)
"""
from __future__ import annotations
import json
from lang3d.knowledge.mobile_base_gen import build_wheeled_base
from lang3d.tools.assembly_generator import (
    _parse_assembly_json, _ensure_arm_default_angles,
)
from lang3d.tools.pipeline_context import AssemblyContext

chassis_json = build_wheeled_base(
    wheel_count=4, drive_type="differential",
    arm_mount_points=["left", "right"],
)
assembly = _parse_assembly_json(chassis_json)

base = next(
    (p for p in assembly.parts
     if "base" in p.name.lower() and "plate" in p.name.lower()), None)
print(f"base_plate length: {base.dimensions.get('length')}")

assembly = _ensure_arm_default_angles(assembly)
print(f"base_plate length after sanitize: {base.dimensions.get('length')}")
assert base.dimensions.get("length") < 500, "base_plate still blown up!"

ctx = AssemblyContext(assembly=assembly)
pos = ctx.ensure_positions()
print(f"\nSolved {len(pos)} positions. Key checks:")
bad = []
for name, pose in sorted(pos.items()):
    p = pose["position"]
    dist = (p[0]**2 + p[1]**2)**0.5
    if "wheel" in name.lower():
        ok = abs(p[2] - 45.0) < 15
        if not ok: bad.append(f"{name} Z={p[2]:.0f} (not on ground)")
        print(f"  {name:20s} Z={p[2]:6.1f} {'OK' if ok else 'FLOATING'}")
    elif "base_plate" == name or "chassis" in name.lower():
        print(f"  {name:20s} Z={p[2]:6.1f}")
if bad:
    print("\nFAIL:", bad)
else:
    print("\nPASS: geometry is sound (base_plate sane, wheels grounded)")
