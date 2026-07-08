"""Evaluation utilities for the Language-3D assembly pipeline.

This package holds the metrics that turn a raw ``e2e_report.json`` into a
quality judgement.  It is deliberately separated from the production pipeline
(``lang3d.agent`` / ``lang3d.tools``): the pipeline *produces* robots, this
package *scores* them, and the two concerns must not be tangled (a scoring
change must not alter what the pipeline generates, and vice versa).

Modules
-------
* :mod:`lang3d.eval.composite_score` — the composite quality score Q
  (geometric mean of robustness / functionality / reliability sub-scores,
  gated on physics + collision + COM).  This is the headline metric reported
  in the paper.

Design notes
------------
* **Structured reads, not regex.**  The e2e harness now persists a
  ``metrics`` block on each physics/COM/grasp check
  (see ``tests/test_e2e_production.py`` ``_check(..., metrics=...)``).
  The composite reads those numeric fields directly.  Earlier versions parsed
  Chinese substrings out of the free-text ``detail`` field (``"裕量 59.2mm"``,
  ``"raw_droop=1.2deg"``), which broke silently whenever the wording changed.
* **Honest physics gate.**  The gate keys on ``raw_droop_deg`` — the PD-hold
  error measured *without* inverse-dynamics gravity compensation — rather than
  the gravity-compensated error (which is ~0° by construction and therefore
  carries no design-quality signal; see paper §sim-limits).
* **No LLM dependency.**  Scoring is deterministic and unit-testable without
  API keys, mirroring the philosophy of :mod:`lang3d.experience`.
"""

from lang3d.eval.composite_score import (
    CompositeResult,
    compute_composite,
    compute_composite_for_case,
    load_case_runs,
)

__all__ = [
    "CompositeResult",
    "compute_composite",
    "compute_composite_for_case",
    "load_case_runs",
]
