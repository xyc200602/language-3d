"""Ablation study runner for Language-3D paper.

Runs the e2e pipeline under different configurations and collects scores.
Usage:
    python scripts/ablation_study.py --case 4dof_arm --runs 3
    LANG3D_ABLATION=no_geo python scripts/ablation_study.py --case 4wheel_dual_arm --runs 2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import glob
import statistics
from pathlib import Path


def load_env() -> None:
    """Load .env so GLM_API_KEY etc. are available."""
    env = Path(".env")
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def run_once(case: str) -> dict:
    """Run one e2e test and return the score dict."""
    r = subprocess.run(
        [sys.executable, "tests/test_e2e_production.py", "--case", case],
        capture_output=True, text=True, timeout=1200,
        encoding="utf-8", errors="replace",
    )
    # Find the latest report
    runs = sorted(glob.glob(f"data/runs/{case}/*/e2e_report.json"))
    if not runs:
        return {"error": "no report", "rc": r.returncode}
    rep = json.load(open(runs[-1]))
    return {
        "score": rep.get("score", 0),
        "pass": rep.get("check_counts", {}).get("pass", 0),
        "fail": rep.get("check_counts", {}).get("fail", 0),
        "warn": rep.get("check_counts", {}).get("warn", 0),
        "rc": r.returncode,
        "run_dir": os.path.dirname(runs[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study runner")
    parser.add_argument("--case", default="4dof_arm", help="Test case ID")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    args = parser.parse_args()

    load_env()
    os.environ.setdefault(
        "FREECADCMD",
        r"C:\Users\xyc\AppData\Local\Programs\FreeCAD 1.1\bin\FreeCADCmd.exe",
    )

    ablation = os.environ.get("LANG3D_ABLATION", "none")
    print(f"=== Ablation: {ablation} | Case: {args.case} | Runs: {args.runs} ===")

    results = []
    for i in range(args.runs):
        print(f"--- Run {i+1}/{args.runs} ---", flush=True)
        r = run_once(args.case)
        print(f"  score={r.get('score', '?')} P/F/W="
              f"{r.get('pass',0)}/{r.get('fail',0)}/{r.get('warn',0)}", flush=True)
        results.append(r)

    scores = [r["score"] for r in results if "score" in r]
    print(f"\n=== SUMMARY ({ablation}) ===")
    if scores:
        print(f"Scores: {[f'{s:.1f}' for s in scores]}")
        print(f"Mean: {statistics.mean(scores):.1f}%")
        if len(scores) > 1:
            print(f"StdDev: {statistics.stdev(scores):.1f}%")
        print(f"Pass rate: {sum(1 for r in results if r.get('rc')==0)}/{len(results)}")

    # Save results
    out = Path(f"data/ablation_{ablation}_{args.case}.json")
    out.write_text(json.dumps({
        "ablation": ablation, "case": args.case, "results": results,
        "mean": statistics.mean(scores) if scores else 0,
        "stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
    }, indent=2), encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
