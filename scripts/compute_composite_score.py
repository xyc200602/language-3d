"""Final composite quality score — CLI wrapper.

The scoring logic now lives in :mod:`lang3d.eval.composite_score` (moved out
of ``scripts/`` per AGENTS.md §2.2 so it is importable by tests and other
consumers).  This file is a thin dispatcher kept for the existing invocation:

    python scripts/compute_composite_score.py

It reads ``data/runs/<case>/*/e2e_report.json``, computes Q per case, and
prints the table used by the paper.
"""
from __future__ import annotations

import sys

from lang3d.eval.composite_score import main

if __name__ == "__main__":
    sys.exit(main())
