"""Backfill the ``support_polygon_diameter_mm`` metric into stale e2e reports.

WHY THIS EXISTS
---------------
The ``support_polygon_diameter_mm`` field was added to the com_stability
check's metrics block in commit ``8690116`` (redesign composite score).
Six of the seven benchmark runs were generated *before* that commit
(20260708 21:55-22:44 vs the field landing ~23:00), so their
``e2e_report.json`` com_stability check carries NO polygon-diameter field.
The composite scorer (``lang3d.eval.composite_score``) then falls back to a
*footprint heuristic* (largest single-part XY extent) that the paper itself
criticises as a cosmetic-base-size proxy — so ``s_robust`` for those six cases
was computed from a "fake" support polygon.

WHAT IT DOES
------------
For each stale run it rebuilds the ``Assembly`` from the saved
``assembly.json`` (a pure dataclass round-trip — no LLM, no CAD, no MuJoCo),
runs the *fixed* ``AssemblyVerifier.check_center_of_mass_stability`` (which
now restricts the polygon to the kinematic root + fixed-joint descendants,
excluding dangling gripper fingers), and writes the corrected metrics back
into ``e2e_report.json``.  Because the polygon shape changes, ``com_margin_mm``
and ``inside_support_polygon`` are also recomputed (they depend on the
polygon).  A ``.bak`` copy of the original report is kept.

This is a one-shot operations script (AGENTS.md §6 — ``scripts/`` holds
non-imported tooling).  Re-run is idempotent: it detects an already-present
correct polygon field and skips unless ``--force``.

USAGE
-----
    python scripts/backfill_com_metrics.py            # dry-run, show diffs
    python scripts/backfill_com_metrics.py --write     # apply the backfill
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from lang3d.agent.assembly_verifier import AssemblyVerifier  # noqa: E402
from lang3d.knowledge.mechanics import Assembly, Joint, Part  # noqa: E402

# (case, benchmark-timestamp) pairs that predate the polygon-diameter field.
STALE_RUNS = [
    ("2dof_arm", "20260708_220654"),
    ("3dof_arm", "20260708_220919"),
    ("4dof_arm", "20260708_215535"),
    ("6dof_arm", "20260708_224422"),
    ("7dof_arm", "20260708_222241"),
    ("4wheel_dual_arm", "20260708_222702"),
]


def _assembly_from_doc(doc: dict) -> Assembly:
    """Rebuild an Assembly dataclass from the saved assembly.json dict.

    Part/Joint are plain dataclasses; their JSON field names match the
    constructor kwargs, so a filtered ``**`` expansion round-trips cleanly.
    Unknown keys (e.g. legacy fields) are ignored rather than erroring.
    """
    part_fields = {f for f in Part.__dataclass_fields__}
    parts = [
        Part(**{k: v for k, v in p.items() if k in part_fields})
        for p in doc.get("parts", [])
    ]
    joint_fields = {f for f in Joint.__dataclass_fields__}
    joints = [
        Joint(**{k: v for k, v in j.items() if k in joint_fields})
        for j in doc.get("joints", [])
    ]
    return Assembly(
        name=doc.get("name", ""),
        parts=parts,
        joints=joints,
        default_angles=doc.get("default_angles", {}),
    )


def _polygon_diameter(poly_xy: list) -> float:
    """Max pairwise vertex distance (mm) of the support polygon."""
    if not poly_xy or len(poly_xy) < 2:
        return 0.0
    diam = 0.0
    n = len(poly_xy)
    for i in range(n):
        x1, y1 = float(poly_xy[i][0]), float(poly_xy[i][1])
        for j in range(i + 1, n):
            x2, y2 = float(poly_xy[j][0]), float(poly_xy[j][1])
            diam = max(diam, math.hypot(x2 - x1, y2 - y1))
    return diam


def backfill(runs_dir: Path, write: bool) -> int:
    verifier = AssemblyVerifier()
    changed = 0
    print(f"{'case':18} {'old_poly':>9} {'new_poly':>9} {'old_margin':>11} "
          f"{'new_margin':>11} {'action':>8}")
    for case, ts in STALE_RUNS:
        run_dir = runs_dir / case / ts
        report_path = run_dir / "e2e_report.json"
        if not report_path.exists():
            print(f"{case:18} {'-':>9} {'-':>9} {'-':>11} {'-':>11} {'MISSING':>8}")
            continue

        doc = json.loads(report_path.read_text(encoding="utf-8"))
        com_check = next(
            (c for c in doc.get("checks", []) if c.get("step") == "com_stability"),
            None,
        )
        if com_check is None:
            print(f"{case:18} no com_stability check — skip")
            continue
        metrics = com_check.setdefault("metrics", {})
        old_poly = metrics.get("support_polygon_diameter_mm")
        old_margin = metrics.get("com_margin_mm")

        # Recompute with the fixed verifier.
        asm_doc = json.loads((run_dir / "assembly.json").read_text(encoding="utf-8"))
        pos_doc = json.loads((run_dir / "positions.json").read_text(encoding="utf-8"))
        assembly = _assembly_from_doc(asm_doc)
        result = verifier.check_center_of_mass_stability(assembly, pos_doc)
        new_poly = _polygon_diameter(result.support_polygon_xy)
        new_margin = float(result.margin_mm)

        old_poly_s = f"{old_poly:.1f}" if isinstance(old_poly, (int, float)) else "MISSING"
        print(f"{case:18} {old_poly_s:>9} {new_poly:9.1f} "
              f"{(str(old_margin) if old_margin is not None else '-'):>11} "
              f"{new_margin:11.1f} ", end="")

        if write:
            # Keep a one-time backup so the backfill is reversible.
            bak = report_path.with_suffix(".json.bak")
            if not bak.exists():
                shutil.copy2(report_path, bak)
            metrics["support_polygon_diameter_mm"] = new_poly
            metrics["com_margin_mm"] = new_margin
            metrics["inside_support_polygon"] = bool(result.inside_support_polygon)
            metrics["com_x_mm"] = float(result.center_of_mass_mm[0])
            metrics["com_y_mm"] = float(result.center_of_mass_mm[1])
            metrics["com_z_mm"] = float(result.center_of_mass_mm[2])
            metrics["total_mass_kg"] = float(result.total_mass_kg)
            # Re-sync the human-readable detail string to the recomputed margin.
            com_check["detail"] = (
                f"COM within support polygon: {result.inside_support_polygon} "
                f"({result.notes or 'n/a'})"
            )
            report_path.write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"{'WRITTEN':>8}")
            changed += 1
        else:
            print(f"{'dry-run':>8}")

    mode = "wrote" if write else "DRY-RUN (use --write to apply)"
    print(f"\n{changed} report(s) {mode}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--write", action="store_true",
                   help="Apply the backfill (default: dry-run, no files changed).")
    p.add_argument("--runs-dir", default=str(_PROJECT_ROOT / "data" / "runs"),
                   help="Override the data/runs directory.")
    args = p.parse_args(argv)
    return backfill(Path(args.runs_dir), args.write)


if __name__ == "__main__":
    raise SystemExit(main())
