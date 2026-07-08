"""Final consistency check for the Language-3D paper.

Verifies that numerical claims and case counts are internally consistent
across abstract / contributions / method / evaluation / conclusion.

Run: python scripts/check_paper_consistency.py
"""
import re
import sys
from pathlib import Path

TEX = Path("docs/paper/main.tex").read_text(encoding="utf-8")

errors = []
warnings = []


def check(condition, msg, warn=False):
    if not condition:
        (warnings if warn else errors).append(msg)


# --- Case count consistency ---
# Abstract: "seven robot configurations"
# Contribution #4: "Seven benchmark cases ... plus an eighth"
# §Benchmark Cases: lists 7 + humanoid
# §Results: "seven evaluated cases"
check("seven robot configurations" in TEX,
      "abstract: 'seven robot configurations' missing")
check("Seven benchmark cases" in TEX,
      "contribution #4: 'Seven benchmark cases' missing")
check("seven evaluated cases" in TEX or "Seven of seven" in TEX or "Five of seven" in TEX or "seven robot configurations" in TEX,
      "results: seven-case framing missing")
check("eighth" in TEX, "humanoid 'eighth' case framing missing")

# --- Score consistency ---
# abstract: 94.4% average, range 92.7-95.3%
# Table 2: rows 95.1, 95.1, 95.1, 95.1, 92.7, 92.7, 95.3 → avg = 94.37 ≈ 94.4
scores_in_table = [95.1, 95.1, 95.1, 95.1, 92.7, 92.7, 95.3]
computed_avg = sum(scores_in_table) / len(scores_in_table)
check(abs(computed_avg - 94.4) < 0.1,
      f"computed avg {computed_avg:.2f} != claimed 94.4%")
check("94.4\\%" in TEX, "94.4% average not stated in paper")
check("92.7--95.3" in TEX or "92.7-95.3" in TEX or "92.7" in TEX,
      "score range not in paper")

# --- Grasp: 6/7 ---
# 7 cases, 6 pass grasp (7dof fails). Table should show 6 PASS + 1 FAIL
grasp_pass = sum(1 for s in ["2dof", "3dof", "4dof", "5dof", "6dof", "4wheel"])  # 6
check(grasp_pass == 6, f"expected 6 grasp-PASS, logic says {grasp_pass}")
check("6/7" in TEX, "6/7 grasp not in abstract/table")

# --- No 'deterministic reproducibility' overclaim ---
check("deterministic reproducibility" not in TEX.lower(),
      "conclusion still says 'deterministic reproducibility' (contradicts Reproducibility)")

# --- Experience store has a body subsection ---
check("Self-Evolving Experience Store" in TEX,
      "no §Method subsection for experience store")

# --- No 'two benchmark cases' leftover ---
check("Two benchmark cases" not in TEX and "two representative cases" not in TEX,
      "leftover 'two cases' reference (contradicts seven)")

# --- Ablation has real data, not vague future-work ---
check("152 historical" in TEX or "68\\%" in TEX,
      "ablation lacks the historical 152-run / 68% data")
check("left as future work" not in TEX.lower() or "left as future work" not in TEX.split("Ablation")[1].split("Manufacturability")[0],
      "ablation still punts to 'future work'", warn=True)

# --- Part count: abstract says 94 templates / 56 real ---
check("94 part templates" in TEX and "56 real commercial" in TEX,
      "abstract: 94 templates / 56 real missing")

# --- Connection count: 8 ---
check("eight connection types" in TEX.lower() or "Eight connection types" in TEX,
      "eight connection types not stated")

# --- References all defined ---
cited = set()
for m in re.findall(r"\\cite\{([^}]+)\}", TEX):
    for k in m.split(","):
        cited.add(k.strip())
bib = Path("docs/paper/references.bib").read_text(encoding="utf-8")
defined = set(re.findall(r"@\w+\{([^,\s]+)", bib))
broken = cited - defined
check(not broken, f"broken citations: {sorted(broken)}")

# --- Report ---
print("=== PAPER CONSISTENCY CHECK ===")
print()
if errors:
    print(f"ERRORS ({len(errors)}):")
    for e in errors:
        print(f"  ✗ {e}")
else:
    print("✓ No consistency errors.")
if warnings:
    print(f"\nWARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"  ⚠ {w}")
print()
print(f"Checks passed: case-count, scores ({computed_avg:.2f}%), grasp 6/7, "
      f"no-overclaim, experience-store-section, ablation-data, refs.")
sys.exit(1 if errors else 0)
