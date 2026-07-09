"""Consistency gate: paper claims vs. real recomputed data.

Verifies that the numerical claims in ``docs/paper/main.tex`` match what the
real run archive + the live composite scorer produce. Two layers:

1. **Composite Q (Table II / abstract)** — recomputed from
   ``data/runs/<case>/`` via :mod:`lang3d.eval.composite_score`, then compared
   to the Q values stated in the paper. This is the headline metric, so it is
   cross-checked against data (not just internal table math).

2. **Rubric + structural claims** — case-count framing, citations defined,
   section anchors, run-count denominators. These are internal-consistency
   or paper-vs-archive checks.

Run: python scripts/check_paper_consistency.py
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

TEX = Path("docs/paper/main.tex").read_text(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "data" / "runs"

errors: list[str] = []
warnings: list[str] = []


def check(condition: bool, msg: str, warn: bool = False) -> None:
    if not condition:
        (warnings if warn else errors).append(msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CASES = ["2dof_arm", "3dof_arm", "4dof_arm", "5dof_arm",
         "6dof_arm", "7dof_arm", "4wheel_dual_arm"]


def _load_runs(case: str) -> list[dict]:
    reports = []
    for rp in sorted((RUNS_DIR / case).glob("*/e2e_report.json")):
        try:
            reports.append(json.loads(rp.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return reports


def _grasp_pass_count(case: str) -> tuple[int, int]:
    """Return (pass_n, total_n) for the grasp check across all runs."""
    pass_n = total = 0
    for r in _load_runs(case):
        for c in r.get("checks", []):
            if c.get("step") == "sim_grasp":
                total += 1
                if c.get("status") == "PASS":
                    pass_n += 1
                break
    return pass_n, total


# ---------------------------------------------------------------------------
# 1. Composite Q — recompute from data, compare to paper Table II
# ---------------------------------------------------------------------------

# Lazy import so the script still collects the structural checks below even
# if the eval package has an import issue.
try:
    from lang3d.eval.composite_score import CASE_ORDER, compute_composite_for_case
    _composite_ok = True
except Exception as exc:  # pragma: no cover
    _composite_ok = False
    warnings.append(f"could not import lang3d.eval.composite_score: {exc}")

if _composite_ok:
    results = []
    for case in CASE_ORDER:
        r = compute_composite_for_case(case)
        if r is not None:
            results.append(r)

    if results:
        q_vals = {r.case: round(r.q, 2) for r in results}
        mean_q = statistics.mean(r.q for r in results)
        min_q = min(r.q for r in results)
        max_q = max(r.q for r in results)
        n_distinct = len(set(q_vals.values()))

        # Abstract / Conclusion / Intro claim: mean 0.68, range 0.49-0.84, 7/7.
        # These are parsed from the actual paper text (not hardcoded) so the
        # check stays correct as the paper evolves. Q values are deterministic
        # given the frozen BENCHMARK runs, so drift indicates a stale table.
        # Tight tolerance (0.005): the scorer rounds to 2dp, so a correct paper
        # should match within half of the last decimal place; 0.015 (the old
        # value) silently masked a 0.85-vs-0.8397 discrepancy (audit 2026-07-10).
        _Q_TOL = 0.005
        _m_mean = re.search(r"mean of ([\d.]+)", TEX)
        _m_range = re.search(r"range ([\d.]+)--([\d.]+)", TEX)
        _paper_mean = float(_m_mean.group(1)) if _m_mean else 0.68
        _paper_min = float(_m_range.group(1)) if _m_range else 0.49
        _paper_max = float(_m_range.group(2)) if _m_range else 0.84
        check(abs(mean_q - _paper_mean) < _Q_TOL,
              f"composite mean Q = {mean_q:.3f}, paper says {_paper_mean}")
        check(abs(min_q - _paper_min) < _Q_TOL,
              f"composite min Q = {min_q:.3f}, paper says {_paper_min}")
        check(abs(max_q - _paper_max) < _Q_TOL,
              f"composite max Q = {max_q:.3f}, paper says {_paper_max}")
        check(n_distinct == 7,
              f"composite distinct = {n_distinct}/7, paper says 7/7")

        # Per-case Q in Table II (tab:composite). Parse by splitting each data
        # row on '&' and taking the Q column (6th), rather than a fragile
        # anchored regex that mis-counts columns. Tight tolerance (0.005).
        _comp_block = TEX.split("tab:composite")[1].split("\\end{table}")[0] if "tab:composite" in TEX else ""
        _table_q = {}
        for line in _comp_block.splitlines():
            s = line.strip()
            if not s or s.startswith("\\") or "&" not in s:
                continue
            cells = [c.strip() for c in s.split("&")]
            if len(cells) < 6:
                continue
            case_label = cells[0].replace("\\_", "_").rstrip()
            for case in CASE_ORDER:
                short = "4wheel" if case == "4wheel_dual_arm" else case
                if case_label.startswith(short):
                    try:
                        _table_q[case] = float(cells[5])  # 6th col = Q
                    except ValueError:
                        pass
                    break
        for case, real_q in q_vals.items():
            paper_q = _table_q.get(case)
            if paper_q is not None and abs(paper_q - real_q) > _Q_TOL:
                check(False,
                      f"Table II Q for {case}: paper={paper_q}, recomputed={real_q}")

        # Grasp success rate: parse the paper's claimed rates from the text
        # (e.g. "6-DOF arm grasps in only 13% of runs" and g_rate=0.12 in
        # Table III). Strict tolerance (0.02): the rate is a small-integer
        # fraction (1/8=0.125), so a 0.08 tolerance would have let the stale
        # "17% vs actual 12%" error slip through — exactly the bug this check
        # failed to catch.
        for case, label in [("6dof_arm", "6-DOF"), ("7dof_arm", "7-DOF")]:
            p, t = _grasp_pass_count(case)
            if t > 0:
                rate = p / t
                # parse "X-DOF arm grasps in only NN% of runs" from abstract/conclusion
                _claimed = None
                # Match "6-DOF arm grasps in only 13%" or the abbreviated
                # "7-DOF in 0%" (the abstract shortens the second mention).
                _cm = re.search(rf"{label} arm grasps in only (\d+)\\?%", TEX)
                if not _cm:
                    _cm = re.search(rf"{label} in (\d+)\\?%", TEX)
                if _cm:
                    _claimed = int(_cm.group(1)) / 100.0
                if _claimed is not None:
                    check(abs(rate - _claimed) <= 0.02,
                          f"{case} grasp rate = {rate:.2f} ({p}/{t}), "
                          f"paper says {_claimed:.0%}")
                else:
                    warnings.append(f"could not parse {label} grasp-% claim in paper text")


# ---------------------------------------------------------------------------
# 2. Internal consistency (structural)
# ---------------------------------------------------------------------------

check("seven robot configurations" in TEX,
      "abstract: 'seven robot configurations' missing")
check("Seven benchmark cases" in TEX,
      "contribution #4: 'Seven benchmark cases' missing")
check("eighth" in TEX, "humanoid 'eighth' case framing missing")
check("Case-Based Experience Store" in TEX,
      "no §Method subsection for experience store (renamed from Self-Evolving)")
check("Two benchmark cases" not in TEX and "two representative cases" not in TEX,
      "leftover 'two cases' reference (contradicts seven)")
check("94 part templates" in TEX and "56 real commercial" in TEX,
      "abstract: 94 templates / 56 real missing")
check("eight connection types" in TEX.lower() or "Eight connection types" in TEX,
      "eight connection types not stated")
check("deterministic reproducibility" not in TEX.lower(),
      "conclusion still says 'deterministic reproducibility' (contradicts Reproducibility)")

# No stale 0-100 Q in abstract
_abstract = TEX.split("\\end{abstract}")[0]
_abstract_has_zero = (
    "0.0$^\\circ$ tracking error" in _abstract
    or re.search(r"0\.0\s*\^?\\?circ.*tracking\s*error", _abstract) is not None
)
check(not _abstract_has_zero,
      "abstract claims '0.0° tracking error' — contradicts §sim-limits")

# References all defined
cited = set()
for m in re.findall(r"\\cite\{([^}]+)\}", TEX):
    for k in m.split(","):
        cited.add(k.strip())
bib = Path("docs/paper/references.bib").read_text(encoding="utf-8")
defined = set(re.findall(r"@\w+\{([^,\s]+)", bib))
broken = cited - defined
check(not broken, f"broken citations: {sorted(broken)}")


# ---------------------------------------------------------------------------
# 3. Run-count denominators (paper vs archive)
# ---------------------------------------------------------------------------

for case in CASES:
    runs = _load_runs(case)
    if not runs:
        continue
    n = len(runs)
    scores = [r.get("score", 0) for r in runs]
    nonzero = [s for s in scores if s > 0]
    modal = Counter(nonzero).most_common(1)[0][0] if nonzero else 0.0
    if case == "4dof_arm":
        # §Reproducibility states a run count; flag if drifted >2
        m = re.search(r"Across\s+(\d+)\s+runs", TEX)
        if m:
            paper_n = int(m.group(1))
            check(abs(paper_n - n) <= 2,
                  f"§Reproducibility says {paper_n} 4dof runs, archive has {n}",
                  warn=True)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

print("=== PAPER CONSISTENCY CHECK (paper vs. recomputed data) ===")
print()
if _composite_ok and results:
    print("Recomputed composite Q (from data/runs/ via lang3d.eval):")
    print(f"  mean={mean_q:.2f}  range={min_q:.2f}-{max_q:.2f}  distinct={n_distinct}/7")
    for r in results:
        rc = r.raw_components
        print(f"  {r.case:18} Q={r.q:.2f} g_rate={rc.get('grasp_rate',-1):.2f} "
              f"lift={rc.get('lift_quality',-1):.2f} gate={'OK' if r.gate_passed else 'X'}")
    print()

if errors:
    print(f"ERRORS ({len(errors)}):")
    for e in errors:
        print(f"  X {e}")
else:
    print("OK No consistency errors.")
if warnings:
    print(f"\nWARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"  ! {w}")
sys.exit(1 if errors else 0)
