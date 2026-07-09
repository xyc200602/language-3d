"""Composite quality score Q for a Language-3D assembly run.

A relative rank in ``[0, 1]`` that summarises how *good* a generated robot is
compared to the other benchmark cases — **not** an absolute quality
percentage (no external benchmark exists to anchor what "100% good" means for
the NL→robot-package task, so presenting it as a percentage would be false
precision). Three sub-scores capture the dimensions that actually vary:

1. **Robustness** — does it stand up?  (COM margin normalised by the real
   support-polygon diameter — built from the kinematic root and its
   fixed-joint descendants, NOT dangling appendages that merely reach low)
2. **Functionality** — can it do what arms do?  (grasp success *rate* across
   runs, not best-of-N; plus DOF completeness.  Lift quality is measured and
   exposed in the raw components for audit but is NOT in the formula because
   no benchmark case achieves positive lift — a consistently-zero term carries
   no ranking signal; see paper §sim-limits)
3. **Reliability** — does it work every time?  (fraction of non-zero runs that
   reach a usable 80% score threshold — a cross-run pass rate, NOT a
   self-referential mean/ceiling ratio which clustered in a narrow band)

``Q = (s_robust * s_func * s_rely) ** (1/3)`` if every gate passes,
else ``Q = 0``.  The geometric mean (rather than arithmetic) is chosen so a
low sub-score significantly pulls Q down — a robot that tips over (s_robust
~0) should not average out to a middling rank.

Gates (binary, all must pass or Q collapses to 0):

* **Physics stable** — ``raw_droop_deg <= RAW_DROOP_GATE_DEG``.  This is the
  PD-hold error measured *without* inverse-dynamics gravity compensation
  (``raw_data.qfrc_applied = kp*(target-current) - kv*vel`` with no
  ``qfrc_bias`` feed-forward, see ``sim_mujoco._run_physics_hold``).  The
  gravity-compensated error is ~0° by construction and carries no
  design-quality signal — gating on it let every case pass trivially
  (paper §sim-limits documents this as "misleading").
* **Zero severe collisions** — ``severe_count == 0`` (interpenetration > 5mm).
* **COM inside support polygon** — ``com_margin_mm >= 0``.

Why this lives in ``lang3d.eval`` and not ``scripts/``
-----------------------------------------------------
The composite is the paper's headline metric.  A headline metric must (a) be
importable by tests, (b) not break when a free-text wording changes, and (c)
be reusable by other consumers (the paper-consistency gate, a future web
dashboard panel).  Those are the ``src/`` criteria in AGENTS.md §2.2 — a
``scripts/`` one-off that regex-parses Chinese strings satisfies none of them.

Data contract
-------------
The composite reads the ``metrics`` block that the e2e harness now writes on
six checks (``mujoco_physics_stable``, ``com_stability``,
``mujoco_joints_actuate``, ``no_severe_collisions``,
``motion_collision_sweep``, ``sim_grasp``).  Each check record in
``e2e_report.json["checks"]`` looks like::

    {
      "step": "mujoco_physics_stable",
      "status": "PASS",
      "detail": "PD-hold physics stable (stabilized=True, err=0.0deg, ...)",
      "critical": true,
      "metrics": {
        "stabilized": true,
        "pd_err_deg": 0.0,        # gravity-compensated (informational)
        "raw_droop_deg": 1.22,    # WITHOUT grav-comp — gated on this
        "disp_mm": 0.0
      }
    }

Reports generated before this field existed (legacy run archive) lack
``metrics``; :func:`_metric` returns ``None`` for them and the composite
degrades gracefully (gate fails → Q = 0) rather than silently fabricating a
value.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum raw (uncompensated) PD-hold droop, in degrees, for the physics
#: gate.  Empirically the 7 benchmark cases droop 0.7–3.3° (mean 1.5°); a 5°
#: threshold gives engineering headroom while still rejecting a robot whose
#: arm collapses under its own weight.  The gravity-compensated error is NOT
#: gated here (it is ~0° by construction — see module docstring).
RAW_DROOP_GATE_DEG = 5.0

#: Theoretical maximum e2e rubric score (documentation only, not used in the
#: composite).  The rubric has 41 checks, of which 2 are informational and
#: always WARN (rotation_data, outliers), so the achievable ceiling is
#: 39/41 = 95.1%, NOT 100%.  Retained as a documented reference for what a
#: "perfect" rubric run scores; s_rely now uses an absolute 80% usability
#: threshold instead of dividing by this ceiling (self-referential mean/ceiling
#: clustered in a narrow band and carried little discriminative signal).
SCORE_CEILING = 95.1

#: Expected actuated DOF per case, for the s_func DOF-completeness term.
#: Keyed by case id.  The 4-wheel dual-arm has 2 × 3-DOF arms = 6.
EXPECTED_DOF: dict[str, int] = {
    "2dof_arm": 2,
    "3dof_arm": 3,
    "4dof_arm": 4,
    "5dof_arm": 5,
    "6dof_arm": 6,
    "7dof_arm": 7,
    "4wheel_dual_arm": 6,
}

#: Default number of DOF to assume when a case is not in EXPECTED_DOF.
DEFAULT_EXPECTED_DOF = 4

#: Canonical case ordering for table/report rendering.
CASE_ORDER: tuple[str, ...] = (
    "2dof_arm",
    "3dof_arm",
    "4dof_arm",
    "5dof_arm",
    "6dof_arm",
    "7dof_arm",
    "4wheel_dual_arm",
)

#: Project root — resolved from this file so the loader works regardless of
#: the caller's working directory (the e2e harness is sometimes invoked from
#: ``tests/``, the scripts from repo root).
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_RUNS_DIR = _PROJECT_ROOT / "data" / "runs"


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class CompositeResult:
    """Outcome of scoring one case.

    ``q`` is a **relative rank score in [0, 1]** — the geometric mean of the
    three sub-scores if all gates pass, else 0.  It is NOT a "quality
    percentage": there is no external benchmark defining what 1.0 means in
    absolute terms, so presenting it as "78% good" would be a false precision.
    It only ranks the 7 benchmark cases against each other.  The sub-scores
    and gate flags are exposed so a caller can render a full breakdown (the
    paper's Table IV does exactly this).  ``raw_components`` keeps the numeric
    inputs so the result is auditable rather than a black box.
    """

    case: str
    q: float
    gate_passed: bool
    s_robust: float
    s_func: float
    s_rely: float
    raw_components: dict[str, float] = field(default_factory=dict)
    gate_reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (for report embedding / web dashboard)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Report loading
# ---------------------------------------------------------------------------


def load_case_runs(case: str, runs_dir: Path | str | None = None) -> list[dict]:
    """Load all non-zero e2e reports for *case*.

    Zero-score reports (API rate-limit failures that produced no assembly at
    all) are excluded — they are infrastructure noise, not a quality signal,
    and including them in s_rely would conflate "the LLM failed to generate"
    with "the generated robot was bad".  The paper's Reproducibility section
    reports the zero-run fraction separately for exactly this reason.

    Parameters
    ----------
    case:
        Case id, e.g. ``"4dof_arm"``.
    runs_dir:
        Override the runs directory (testing).  Defaults to
        ``<project_root>/data/runs``.
    """
    base = Path(runs_dir) if runs_dir else _RUNS_DIR
    reports: list[dict] = []
    for rp in sorted((base / case).glob("*/e2e_report.json")):
        try:
            doc = json.loads(rp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("skip unreadable report %s: %s", rp, exc)
            continue
        if doc.get("score", 0) > 0:
            reports.append(doc)
    return reports


def _load_benchmark_run(
    case: str, runs_dir: Path | str | None, reports: list[dict]
) -> dict | None:
    """Return the frozen BENCHMARK run for *case*, or None if unset.

    A ``data/runs/<case>/BENCHMARK`` file (first non-comment line = timestamp)
    pins the paper's Table I/II physics numbers to a fixed run so they stop
    drifting as new e2e runs accumulate.  This is the single source of truth
    for "which run's COM/droop/collision does the paper report" — the scorer,
    the table extractor, and the consistency gate all read the same marker.

    The run must be present in *reports* (i.e. non-zero score) and carry a
    metrics block; otherwise we fall back to median selection.
    """
    base = Path(runs_dir) if runs_dir else _RUNS_DIR
    marker = base / case / "BENCHMARK"
    if not marker.exists():
        return None
    try:
        ts = marker.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError):
        return None
    if not ts or ts.startswith("#"):
        return None
    rp = base / case / ts / "e2e_report.json"
    if not rp.exists():
        logger.debug("BENCHMARK marker points to missing run %s", rp)
        return None
    try:
        doc = json.loads(rp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if doc.get("score", 0) <= 0:
        return None
    if not any("metrics" in c for c in doc.get("checks", [])):
        return None
    return doc


def _load_assembly(case: str, runs_dir: Path | str | None = None) -> dict | None:
    """Load the most recent assembly.json for *case* (for footprint sizing)."""
    base = Path(runs_dir) if runs_dir else _RUNS_DIR
    for rp in sorted((base / case).glob("*/assembly.json"), reverse=True):
        try:
            return json.loads(rp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return None


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def _checks_by_step(report: dict) -> dict[str, dict]:
    """Index a report's checks by ``step`` (last one wins on collision)."""
    return {c["step"]: c for c in report.get("checks", []) if "step" in c}


def _metric(check: dict | None, key: str) -> float | None:
    """Read a numeric field from a check's ``metrics`` block.

    Returns ``None`` when the check is missing or the field is absent (legacy
    reports pre-date the structured-metrics landing).  Callers must treat
    ``None`` as "unknown" and gate-fail rather than fabricate a value.
    """
    if check is None:
        return None
    metrics = check.get("metrics")
    if not isinstance(metrics, dict):
        return None
    val = metrics.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _compute_footprint(assembly: dict | None) -> float:
    """Largest single-part XY extent (mm) — a FALLBACK stance normaliser.

    Prefer the real support-polygon diameter (``support_polygon_diameter_mm``
    metric) when available; this heuristic is only used for legacy reports
    that predate the polygon landing.  It over-credits large cosmetic base
    parts (e.g. 7dof's skirt) and is documented as imprecise.
    """
    if not assembly:
        return 100.0
    max_xy = 0.0
    for p in assembly.get("parts", []):
        dims = p.get("dimensions", {}) or {}
        xy = max(
            dims.get("length", 0) or 0,
            dims.get("width", 0) or 0,
            dims.get("diameter", 0) or 0,
        )
        max_xy = max(max_xy, xy)
    return max_xy if max_xy > 0 else 100.0


def _median(values: list[float]) -> float:
    """Median of a list (0.0 if empty). Avoids importing statistics for one call."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _grasp_rate(all_reports: Iterable[dict]) -> float:
    """Fraction of runs whose static grasp passed.

    Replaces the prior best-of-N 0/1, which credited a flaky gripper (6dof:
    1 of 5 runs PASS) with a full grasp mark.  Uses ALL non-zero runs so the
    score reflects how reliably the gripper works, not whether it ever worked
    once.  Returns -1.0 when no run has a grasp check at all (case has no
    gripper), so the caller can exclude grasp from s_func entirely.
    """
    pass_n = 0
    total = 0
    for r in all_reports:
        for c in r.get("checks", []):
            if c.get("step") == "sim_grasp":
                total += 1
                if c.get("status") == "PASS":
                    pass_n += 1
                break
    if total == 0:
        return -1.0
    return pass_n / total


def _lift_quality_median(all_reports: Iterable[dict]) -> float:
    """Median lift_ratio (lift_mm / target_mm, clipped to [0,1]) across runs.

    Measures whether the robot can actually RAISE the grasped object, not
    merely hold it against gravity.  Uses the median (typical run), not the
    best — a robot that lifted once in five tries is not a reliable lifter.
    Returns -1.0 when no run carries lift data (pre-refactor reports).
    """
    ratios = []
    for r in all_reports:
        for c in r.get("checks", []):
            if c.get("step") == "sim_grasp":
                ratio = _metric(c, "lift_ratio")
                if ratio is not None:
                    ratios.append(min(1.0, max(0.0, ratio)))
                break
    if not ratios:
        return -1.0
    return _median(ratios)


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------


def _gate_and_subs(
    phys_report: dict,
    assembly: dict | None,
    case: str,
    all_reports: Iterable[dict],
) -> tuple[bool, list[str], float, float, float, dict[str, float]]:
    """Compute gate flags, sub-scores, and raw components for one case.

    *phys_report* is the run whose physics/COM metrics are scored.  The caller
    chooses this (median-score run for the case-level scorer, the report
    itself for the single-run scorer).  *all_reports* feeds the distribution-
    based sub-scores (grasp_rate, lift_quality, s_rely).

    Returns ``(gate_passed, gate_reasons, s_robust, s_func, s_rely, raw)``.
    """
    reports_list = list(all_reports)
    by_step = _checks_by_step(phys_report)

    physics = by_step.get("mujoco_physics_stable")
    com = by_step.get("com_stability")
    coll = by_step.get("no_severe_collisions")
    actuated_chk = by_step.get("mujoco_joints_actuate")

    raw_droop = _metric(physics, "raw_droop_deg")
    com_margin = _metric(com, "com_margin_mm")
    poly_diam = _metric(com, "support_polygon_diameter_mm")
    severe = _metric(coll, "severe_count")
    actuated = _metric(actuated_chk, "actuated_joints")

    # Stance normaliser: prefer the real support-polygon diameter; fall back
    # to the base-part footprint heuristic only for legacy reports.
    stance = poly_diam if (poly_diam is not None and poly_diam > 0) else _compute_footprint(assembly)
    stance_source = "polygon" if (poly_diam is not None and poly_diam > 0) else "footprint_heuristic"

    expected = EXPECTED_DOF.get(case, DEFAULT_EXPECTED_DOF)
    grasp_rate = _grasp_rate(reports_list)
    lift_quality = _lift_quality_median(reports_list)

    raw: dict[str, float] = {
        "raw_droop_deg": raw_droop if raw_droop is not None else -1.0,
        "com_margin_mm": com_margin if com_margin is not None else -1.0,
        "support_polygon_diameter_mm": poly_diam if poly_diam is not None else -1.0,
        "stance_source": stance_source,
        "severe_count": severe if severe is not None else -1.0,
        "actuated_joints": actuated if actuated is not None else -1.0,
        "grasp_rate": grasp_rate,
        "lift_quality": lift_quality,
    }

    # --- Gates (all binary; any failure → Q = 0) ---
    gate_reasons: list[str] = []
    if raw_droop is None:
        gate_reasons.append("raw_droop_deg missing (legacy report)")
    elif raw_droop > RAW_DROOP_GATE_DEG:
        gate_reasons.append(f"raw_droop {raw_droop:.2f}° > {RAW_DROOP_GATE_DEG}°")
    if severe is None:
        gate_reasons.append("severe_count missing (legacy report)")
    elif severe > 0:
        gate_reasons.append(f"{int(severe)} severe collision(s)")
    if com_margin is None:
        gate_reasons.append("com_margin_mm missing (legacy report)")
    elif com_margin < 0:
        gate_reasons.append("COM outside support polygon")
    gate_passed = not gate_reasons

    # --- Sub-scores ---
    # 1. Robustness: COM margin normalised by half the stance (size-
    #    independent).  Using the real support polygon (not the base part)
    #    means a robot is rewarded for actual ground-contact spread.
    if com_margin is None:
        s_robust = 0.0
    elif stance > 0:
        s_robust = min(1.0, max(0.0, com_margin) / (0.5 * stance))
    else:
        s_robust = 0.0

    # 2. Functionality: grasp success-RATE (not best-of-N) + DOF completeness.
    #    The lift term (lift_quality) is NO LONGER in the formula: across all
    #    benchmark cases no configuration achieves positive lift (the cube
    #    drops during the lift phase — paper §sim-limits), so a 0.3-weighted
    #    lift term carried no information and diluted the grasp signal.  Lift
    #    is still measured and exposed in raw_components for audit, but a
    #    consistently-zero term does not belong in a rank score.  When the case
    #    has no gripper (grasp_rate == -1), s_func collapses to DOF only.
    dof_ratio = min(1.0, (actuated or 0.0) / expected) if expected > 0 else 0.0
    if grasp_rate < 0:
        # No gripper in any run — score on DOF only.
        s_func = dof_ratio
    else:
        g = max(0.0, grasp_rate)
        s_func = 0.6 * g + 0.4 * dof_ratio

    # 3. Reliability: fraction of non-zero runs that reach a usable threshold.
    #    This replaces the prior mean/ceiling self-normalisation, which used the
    #    rubric's own score divided by its own ceiling — a self-referential
    #    quantity that clustered in a narrow 0.92-0.99 band across very
    #    different cases (a deterministic 4wheel and a high-variance 4dof
    #    scored nearly identically) and carried little discriminative signal.
    #    A pass-rate against an absolute 80% bar is a genuine cross-run
    #    stability measure: it asks "does this robot reliably produce a
    #    usable assembly", not "how close to the rubric ceiling is the mean".
    usable_threshold = 80.0
    nonzero_scores = [r.get("score", 0) for r in reports_list if r.get("score", 0) > 0]
    if nonzero_scores:
        s_rely = sum(1 for s in nonzero_scores if s >= usable_threshold) / len(nonzero_scores)
    else:
        s_rely = 0.0
    raw["mean_score"] = sum(nonzero_scores) / len(nonzero_scores) if nonzero_scores else 0.0
    raw["reliability_pass_rate"] = s_rely
    raw["total_runs"] = len(reports_list)

    return gate_passed, gate_reasons, s_robust, s_func, s_rely, raw


def compute_composite(report: dict, assembly: dict | None = None, case: str = "") -> CompositeResult:
    """Score a *single* run report (Q in [0, 1], NOT a percentage).

    For the single-run case the distribution sub-scores collapse: grasp_rate
    is 0/1 and s_rely is score/ceiling for this one run.  Use
    :func:`compute_composite_for_case` to fold in the across-run distribution.
    """
    gate_passed, reasons, s_robust, s_func, s_rely, raw = _gate_and_subs(
        report, assembly, case, [report]
    )
    q = (s_robust * s_func * s_rely) ** (1 / 3) if gate_passed else 0.0
    return CompositeResult(
        case=case,
        q=q,
        gate_passed=gate_passed,
        s_robust=s_robust,
        s_func=s_func,
        s_rely=s_rely,
        raw_components=raw,
        gate_reasons=reasons,
    )


def compute_composite_for_case(
    case: str,
    reports: list[dict] | None = None,
    assembly: dict | None = None,
    runs_dir: Path | str | None = None,
) -> CompositeResult | None:
    """Score *case* using the median run for physics, all runs for distribution.

    Physics/COM metrics (s_robust, gate) come from the **median-score** run —
    the typical outcome, not the best.  Reporting the best run's physics was a
    form of cherry-picking: a case whose median run tips over (6dof: median
    COM = -529mm) but whose best run is stable (COM = 70mm) looked fine under
    the old protocol.  The distribution sub-scores (grasp_rate, lift_quality,
    s_rely) use ALL non-zero runs.

    ``reports`` is auto-loaded from ``data/runs/<case>/`` if not supplied.
    Returns ``None`` if the case has no runs at all.
    """
    if reports is None:
        reports = load_case_runs(case, runs_dir)
    if not reports:
        return None
    if assembly is None:
        assembly = _load_assembly(case, runs_dir)

    # Select the run whose physics/COM metrics feed the gate and s_robust.
    # Preference order:
    #   1. A frozen BENCHMARK run (data/runs/<case>/BENCHMARK file names a ts)
    #      — pins the paper tables to a fixed baseline so they stop drifting
    #      as new e2e runs accumulate.  Without this, the median (below) shifts
    #      every time a new run lands, silently changing every Table I/II cell.
    #   2. Otherwise the median-scoring metrics-carrying run (typical outcome).
    base = Path(runs_dir) if runs_dir else _RUNS_DIR
    bm_run = _load_benchmark_run(case, base, reports)

    if bm_run is not None:
        phys_report = bm_run
    else:
        metrics_runs = [r for r in reports if any("metrics" in c for c in r.get("checks", []))]
        pool = metrics_runs if metrics_runs else reports
        pool_sorted = sorted(pool, key=lambda d: d.get("score", 0))
        phys_report = pool_sorted[len(pool_sorted) // 2]

    gate_passed, reasons, s_robust, s_func, s_rely, raw = _gate_and_subs(
        phys_report, assembly, case, reports
    )
    q = (s_robust * s_func * s_rely) ** (1 / 3) if gate_passed else 0.0
    return CompositeResult(
        case=case,
        q=q,
        gate_passed=gate_passed,
        s_robust=s_robust,
        s_func=s_func,
        s_rely=s_rely,
        raw_components=raw,
        gate_reasons=reasons,
    )
# ---------------------------------------------------------------------------
# CLI (thin wrapper — the heavy lifting is importable above)
# ---------------------------------------------------------------------------


def _render_table(results: list[CompositeResult]) -> str:
    """Format results as an aligned text table (Q shown in [0,1], not %)."""
    header = (
        f"{'Case':18} {'droop':>6} {'COM':>6} {'robust':>7} "
        f"{'g_rate':>6} {'lift':>5} {'func':>5} {'rely':>5} "
        f"{'GATE':>4} {'Q':>5}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        rc = r.raw_components
        lines.append(
            f"{r.case:18} {rc.get('raw_droop_deg', -1):6.2f} "
            f"{rc.get('com_margin_mm', -1):6.1f} {r.s_robust:7.2f} "
            f"{rc.get('grasp_rate', -1):6.2f} {rc.get('lift_quality', -1):5.2f} "
            f"{r.s_func:5.2f} {r.s_rely:5.2f} "
            f"{'OK' if r.gate_passed else 'X':>4} {r.q:5.2f}"
        )
    mean_q = sum(r.q for r in results) / len(results) if results else 0.0
    lines.append("-" * len(header))
    lines.append(f"{'Mean':18} Q = {mean_q:.2f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: compute Q for all benchmark cases and print a table.

    Kept as a thin wrapper over the importable API so ``scripts/`` stays a
    dispatcher (AGENTS.md §2.2 — scripts must not hold importable logic).
    """
    results: list[CompositeResult] = []
    for case in CASE_ORDER:
        r = compute_composite_for_case(case)
        if r is None:
            logger.warning("no runs for case %s — skipping", case)
            continue
        results.append(r)
    if not results:
        print("No runs found under data/runs/. Run the e2e harness first.")
        return 1
    print(_render_table(results))
    print()
    # Q is in [0,1]; round to 2dp for the distinctness count so the ranking
    # signal is visible (rounding to integers would collapse everything to 0/1).
    q_vals = [round(r.q, 2) for r in results]
    print(f"Q values:   {q_vals}")
    print(f"Distinct:   {len(set(q_vals))}/{len(q_vals)}")
    failed = [r.case for r in results if not r.gate_passed]
    if failed:
        print(f"Gate-failed cases: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
