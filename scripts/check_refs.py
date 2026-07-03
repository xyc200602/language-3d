"""Check that all \\cite{...} keys in the paper are defined in references.bib.

Run: python scripts/check_refs.py
"""
import re
import sys
from pathlib import Path

PAPER_DIR = Path("docs/paper")

cited = set()
for fname in ["main.tex", "comparison_table.tex"]:
    txt = (PAPER_DIR / fname).read_text(encoding="utf-8")
    for m in re.findall(r"\\cite\{([^}]+)\}", txt):
        for k in m.split(","):
            cited.add(k.strip())

bib = (PAPER_DIR / "references.bib").read_text(encoding="utf-8")
defined = set(re.findall(r"@\w+\{([^,\s]+)", bib))

print(f"Cited keys ({len(cited)}): {sorted(cited)}")
print(f"Defined keys ({len(defined)}): {sorted(defined)}")
print()
broken = cited - defined
unused = defined - cited
if broken:
    print(f"BROKEN (cited but not defined, {len(broken)}): {sorted(broken)}")
else:
    print("OK: all cited keys are defined.")
if unused:
    print(f"UNUSED (defined but not cited, {len(unused)}): {sorted(unused)}")
sys.exit(1 if broken else 0)
