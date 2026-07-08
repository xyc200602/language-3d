"""Check LaTeX \\ref/\\label integrity and figure existence."""
import re
import os
from pathlib import Path

PAPER_DIR = Path("docs/paper")
tex = (PAPER_DIR / "main.tex").read_text(encoding="utf-8")
ct = (PAPER_DIR / "comparison_table.tex").read_text(encoding="utf-8")
full = tex + "\n" + ct

labels = set(re.findall(r"\\label\{([^}]+)\}", full))
refs = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", full))

undefined = refs - labels
unused = labels - refs

print(f"Labels: {sorted(labels)}")
print(f"Refs:   {sorted(refs)}")
print(f"UNDEFINED refs: {sorted(undefined) if undefined else 'none'}")
print(f"UNUSED labels:  {sorted(unused) if unused else 'none'}")

# Figures
figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
print(f"\nFigures: {figs}")
for f in figs:
    p = PAPER_DIR / f
    print(f"  {f}: {'OK' if p.exists() else 'MISSING'}")

# Tables
tables = re.findall(r"\\input\{([^}]+)\}", tex)
print(f"\nIncluded files: {tables}")
for t in tables:
    # \input{foo} may resolve to foo.tex or foo (LaTeX tries foo.tex first).
    p_tex = PAPER_DIR / f"{t}.tex"
    p_bare = PAPER_DIR / t
    status = "OK" if (p_tex.exists() or p_bare.exists()) else "MISSING"
    print(f"  {t}: {status}")
