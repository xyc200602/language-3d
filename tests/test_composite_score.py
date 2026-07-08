"""Tests for the composite quality score (lang3d.eval.composite_score).

This is the paper's headline metric, so it has full unit coverage of:
* gate behaviour (physics / collision / COM)
* sub-score math (robustness / functionality / reliability)
* the geometric-mean aggregation and Q=0 collapse on gate failure
* legacy-report graceful degradation (no metrics block → gate fail, not crash)
* the case-level loader (best run for physics, all runs for reliability)

These tests are pure-Python (no external deps), so conftest classifies them
as ``unit`` — they run in the fast local suite.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from lang3d.eval.composite_score import (
    CASE_ORDER,
    RAW_DROOP_GATE_DEG,
    CompositeResult,
    compute_composite,
    compute_composite_for_case,
    load_case_runs,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic e2e reports with structured metrics
# ---------------------------------------------------------------------------


def _check(
    step: str,
    status: str = "PASS",
    *,
    metrics: dict | None = None,
    detail: str = "",
    critical: bool = True,
) -> dict:
    """Build one check record matching the e2e harness schema."""
    rec: dict = {"phase": "phase6", "step": step, "status": status, "detail": detail, "critical": critical}
    if metrics is not None:
        rec["metrics"] = metrics
    return rec


def _make_report(
    *,
    score: float = 95.1,
    raw_droop: float = 1.2,
    com_margin: float = 40.0,
    severe: int = 0,
    actuated: int = 4,
    grasp_ok: bool = True,
    lift_ratio: float = 0.8,
    include_metrics: bool = True,
) -> dict:
    """A well-formed passing report.

    With ``include_metrics=False`` the structured block is omitted, simulating
    a legacy report from before the metrics landing — used to verify graceful
    degradation.
    """
    checks = [
        _check("mujoco_physics_stable", metrics=None if not include_metrics else {
            "stabilized": True, "pd_err_deg": 0.0,
            "raw_droop_deg": raw_droop, "disp_mm": 0.0,
        }),
        _check("com_stability", metrics=None if not include_metrics else {
            "com_margin_mm": com_margin, "inside_support_polygon": True,
            "com_x_mm": 0.0, "com_y_mm": 0.0, "com_z_mm": 40.0,
            "total_mass_kg": 2.5,
        }),
        _check("no_severe_collisions", metrics=None if not include_metrics else {
            "severe_count": severe, "pairs_checked": 52,
        }),
        _check("mujoco_joints_actuate", metrics=None if not include_metrics else {
            "actuated_joints": actuated, "min_expected": 4,
        }),
        _check("sim_grasp", status="PASS" if grasp_ok else "FAIL",
               metrics=None if not include_metrics else {
            "grasp_ok": grasp_ok, "lifted": grasp_ok and lift_ratio > 0.05,
            "lift_mm": lift_ratio * 30.0, "lift_target_mm": 30.0,
            "lift_ratio": lift_ratio, "slip_mm": 0.0,
        }),
    ]
    return {"test_id": "4dof_arm", "score": score, "checks": checks}


def _make_assembly(footprint: float = 120.0) -> dict:
    """An assembly with one base part of the given XY footprint."""
    return {"parts": [{"name": "base", "dimensions": {"length": footprint, "width": footprint}}]}


# ---------------------------------------------------------------------------
# Single-run scoring
# ---------------------------------------------------------------------------


class TestSingleRunScoring:
    def test_passing_report_yields_nonzero_q(self):
        report = _make_report(raw_droop=1.2, com_margin=40.0)
        r = compute_composite(report, assembly=_make_assembly(120.0), case="4dof_arm")
        assert r.gate_passed
        assert r.q > 0
        # s_robust = 40 / (0.5*120) = 0.667
        assert r.s_robust == pytest.approx(40.0 / 60.0, rel=1e-3)

    def test_gate_fails_when_raw_droop_exceeds_threshold(self):
        report = _make_report(raw_droop=RAW_DROOP_GATE_DEG + 0.1)
        r = compute_composite(report, assembly=_make_assembly(), case="4dof_arm")
        assert not r.gate_passed
        assert r.q == 0.0
        assert any("raw_droop" in reason for reason in r.gate_reasons)

    def test_gate_fails_on_severe_collision(self):
        report = _make_report(severe=2)
        r = compute_composite(report, assembly=_make_assembly(), case="4dof_arm")
        assert not r.gate_passed
        assert r.q == 0.0
        assert any("severe" in reason for reason in r.gate_reasons)

    def test_gate_fails_when_com_outside_polygon(self):
        report = _make_report(com_margin=-5.0)  # negative = outside
        r = compute_composite(report, assembly=_make_assembly(), case="4dof_arm")
        assert not r.gate_passed
        assert r.q == 0.0
        assert any("support polygon" in reason for reason in r.gate_reasons)

    def test_grasp_failure_lowers_functionality(self):
        """s_func weights grasp at 0.4 (success-rate) + 0.3 (lift) = 0.7 of total.

        A grasp fail zeroes both the grasp-rate and (effectively) lift terms,
        so s_func drops by ~0.7 relative to a passing gripper.  This replaces
        the old 0.5-weight binary that credited a flaky gripper identically to
        a reliable one.
        """
        passing = compute_composite(_make_report(grasp_ok=True, lift_ratio=1.0), _make_assembly(), "4dof_arm")
        failing = compute_composite(_make_report(grasp_ok=False, lift_ratio=0.0), _make_assembly(), "4dof_arm")
        # dof_ratio = 1.0 in both (4 actuated / 4 expected)
        # passing: 0.4*1 + 0.3*1 + 0.3*1 = 1.0
        assert passing.s_func == pytest.approx(1.0)
        # failing: 0.4*0 + 0.3*0 + 0.3*1 = 0.3
        assert failing.s_func == pytest.approx(0.3)

    def test_dof_completeness_lowers_functionality(self):
        """Fewer actuated joints than expected lowers s_func via the 0.3 dof term."""
        r = compute_composite(_make_report(actuated=2, grasp_ok=True, lift_ratio=1.0), _make_assembly(), "4dof_arm")
        # dof_ratio = 2/4 = 0.5; s_func = 0.4*1 + 0.3*1 + 0.3*0.5 = 0.85
        assert r.s_func == pytest.approx(0.85)

    def test_q_is_in_zero_one_not_percentage(self):
        """Q is a relative rank in [0,1], never multiplied by 100.

        Presenting Q as a percentage implied an absolute quality grade with no
        external benchmark to anchor it. The redesigned Q is a pure rank.
        """
        r = compute_composite(_make_report(), _make_assembly(), "4dof_arm")
        assert 0.0 <= r.q <= 1.0
        # A perfect-score case (all sub-scores 1.0) yields Q = 1.0 exactly.
        perfect = compute_composite(
            _make_report(com_margin=60.0, raw_droop=0.5, grasp_ok=True, lift_ratio=1.0),
            _make_assembly(120.0), "4dof_arm",
        )
        assert perfect.s_robust == pytest.approx(1.0)  # 60/(0.5*120)=1.0
        assert perfect.q == pytest.approx(1.0)

    def test_geometric_mean_not_arithmetic(self):
        """A low sub-score should pull Q well below the arithmetic mean."""
        # s_robust tiny, others maxed
        report = _make_report(com_margin=1.0, grasp_ok=True, lift_ratio=1.0)
        r = compute_composite(report, assembly=_make_assembly(120.0), case="4dof_arm")
        assert r.gate_passed
        # geometric mean of (0.0167, 1.0, 1.0) = 0.0167^(1/3) ~ 0.256
        expected = (0.0167 * 1.0 * 1.0) ** (1 / 3)
        assert r.q == pytest.approx(expected, rel=1e-2)
        # ...which is far below the arithmetic mean (0.672)
        assert r.q < 0.67


# ---------------------------------------------------------------------------
# Legacy / missing-metrics graceful degradation
# ---------------------------------------------------------------------------


class TestLegacyReports:
    def test_report_without_metrics_fails_gate_not_crash(self):
        """A pre-metrics-landing report must not crash; it gate-fails honestly."""
        report = _make_report(include_metrics=False)
        r = compute_composite(report, assembly=_make_assembly(), case="4dof_arm")
        assert not r.gate_passed
        assert r.q == 0.0
        # All three gated metrics should be reported missing
        reasons = " ".join(r.gate_reasons)
        assert "raw_droop_deg" in reasons
        assert "com_margin_mm" in reasons
        assert "severe_count" in reasons

    def test_partial_metrics_only_gates_on_missing_gated_fields(self):
        """If only the physics metrics block is present, only physics is known."""
        report = _make_report(include_metrics=False)
        # Re-attach just the physics metrics
        for c in report["checks"]:
            if c["step"] == "mujoco_physics_stable":
                c["metrics"] = {"stabilized": True, "pd_err_deg": 0.0, "raw_droop_deg": 1.0, "disp_mm": 0.0}
        r = compute_composite(report, assembly=_make_assembly(), case="4dof_arm")
        assert not r.gate_passed  # COM + collision still missing
        reasons = " ".join(r.gate_reasons)
        assert "raw_droop" not in reasons  # physics IS known and passes
        assert "com_margin_mm" in reasons


# ---------------------------------------------------------------------------
# Case-level scoring (reliability distribution)
# ---------------------------------------------------------------------------


class TestCaseLevelScoring:
    def test_reliability_uses_mean_normalized_by_ceiling(self, tmp_path):
        """s_rely = mean(nonzero_scores) / SCORE_CEILING.

        Replaces the old P(score>=80) binary that gave near-identical s_rely
        to a deterministic case and a high-variance one.  The mean captures
        the typical distance from the reachable ceiling (95.1%).
        """
        case = "4dof_arm"
        case_dir = tmp_path / case
        for i, score in enumerate([95.1, 95.1, 61.9, 88.0]):
            run_dir = case_dir / f"run{i}"
            run_dir.mkdir(parents=True)
            report = _make_report(score=score)
            (run_dir / "e2e_report.json").write_text(json.dumps(report), encoding="utf-8")
        (case_dir / "run0" / "assembly.json").write_text(json.dumps(_make_assembly()), encoding="utf-8")

        r = compute_composite_for_case(case, runs_dir=tmp_path)
        assert r is not None
        # mean = (95.1+95.1+61.9+88.0)/4 = 85.025; ceiling = 95.1
        from lang3d.eval.composite_score import SCORE_CEILING
        expected = 85.025 / SCORE_CEILING
        assert r.s_rely == pytest.approx(expected, rel=1e-3)

    def test_median_run_used_for_physics_not_best(self, tmp_path):
        """Physical metrics come from a median-score run, not the best.

        The prior protocol took the best run's physics, which cherry-picked
        the most favourable geometry — a case whose median run tips over but
        whose best run is stable looked fine.  Median reports the typical
        outcome.  Here 3 runs: low/good-droop, mid/bad-droop, high/good-droop.
        The median (middle) run is the mid one with bad droop → gate fails,
        even though the best run would have passed.
        """
        case = "4dof_arm"
        case_dir = tmp_path / case
        reports = [
            (_make_report(score=80.0, raw_droop=1.0), "run0"),
            (_make_report(score=88.0, raw_droop=10.0), "run1"),  # median, exceeds gate
            (_make_report(score=95.1, raw_droop=1.0), "run2"),
        ]
        for rep, name in reports:
            d = case_dir / name
            d.mkdir(parents=True)
            (d / "e2e_report.json").write_text(json.dumps(rep), encoding="utf-8")
        (case_dir / "run2" / "assembly.json").write_text(json.dumps(_make_assembly()), encoding="utf-8")

        r = compute_composite_for_case(case, runs_dir=tmp_path)
        assert r is not None
        # Median run (88.0) has droop 10° → gate fails, unlike the best run
        assert not r.gate_passed
        assert r.raw_components["raw_droop_deg"] == pytest.approx(10.0)

    def test_no_runs_returns_none(self, tmp_path):
        assert compute_composite_for_case("nonexistent_case", runs_dir=tmp_path) is None

    def test_zero_score_runs_excluded_from_reliability(self, tmp_path):
        """API-failure runs (score 0) are noise, not quality signal."""
        case = "4dof_arm"
        case_dir = tmp_path / case
        for i, score in enumerate([95.1, 0.0, 0.0]):
            d = case_dir / f"run{i}"
            d.mkdir(parents=True)
            (d / "e2e_report.json").write_text(json.dumps(_make_report(score=score)), encoding="utf-8")
        (case_dir / "run0" / "assembly.json").write_text(json.dumps(_make_assembly()), encoding="utf-8")

        r = compute_composite_for_case(case, runs_dir=tmp_path)
        assert r is not None
        # Only the 95.1 run counts (zeros excluded) → mean=95.1, s_rely=1.0
        assert r.s_rely == pytest.approx(1.0)

    def test_grasp_rate_uses_all_runs_not_best_of_n(self, tmp_path):
        """grasp_rate = fraction of runs that passed grasp, across ALL runs.

        Replaces the best-of-N 0/1 that credited a flaky gripper (6dof: 1 of
        5 runs PASS) with a full grasp mark.  A 1-in-5 gripper must score
        grasp_rate = 0.2, pulling s_func down to reflect real unreliability.
        """
        case = "4dof_arm"
        case_dir = tmp_path / case
        # 5 runs: 1 grasp PASS, 4 FAIL → rate = 0.2 (mirrors real 6dof)
        for i, grasp in enumerate([True, False, False, False, False]):
            d = case_dir / f"run{i}"
            d.mkdir(parents=True)
            rep = _make_report(score=92.7, grasp_ok=grasp, lift_ratio=0.0)
            (d / "e2e_report.json").write_text(json.dumps(rep), encoding="utf-8")
        (case_dir / "run0" / "assembly.json").write_text(json.dumps(_make_assembly()), encoding="utf-8")

        r = compute_composite_for_case(case, runs_dir=tmp_path)
        assert r is not None
        assert r.raw_components["grasp_rate"] == pytest.approx(0.2)
        # s_func = 0.4*0.2 + 0.3*0 + 0.3*1 = 0.38  (dof_ratio=1, lift=0)
        assert r.s_func == pytest.approx(0.38)

    def test_metrics_run_preferred_over_legacy_same_score(self, tmp_path):
        """When a legacy (no-metrics) and a new (with-metrics) run tie on
        score, the metrics run must be selected for physics/COM extraction.

        The run archive mixes eras; without this preference a legacy run with
        unreadable physics would gate-fail on 'missing' despite a same-score
        run carrying the structured data.
        """
        case = "4dof_arm"
        case_dir = tmp_path / case
        # legacy run: high score, no metrics block
        legacy = _make_report(score=95.1, include_metrics=False)
        # new run: same score, WITH metrics block, known droop
        new = _make_report(score=95.1, include_metrics=True, raw_droop=1.5)
        for name, rep in [("legacy", legacy), ("newrun", new)]:
            d = case_dir / name
            d.mkdir(parents=True)
            (d / "e2e_report.json").write_text(json.dumps(rep), encoding="utf-8")
        (case_dir / "newrun" / "assembly.json").write_text(json.dumps(_make_assembly()), encoding="utf-8")

        r = compute_composite_for_case(case, runs_dir=tmp_path)
        assert r is not None
        # The metrics run was selected → physics known, gate can pass
        assert r.gate_passed
        assert r.raw_components["raw_droop_deg"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Result model / serialisation
# ---------------------------------------------------------------------------


class TestResultModel:
    def test_as_dict_roundtrips(self):
        r = compute_composite(_make_report(), _make_assembly(), "4dof_arm")
        d = r.as_dict()
        assert d["case"] == "4dof_arm"
        assert "q" in d and "gate_passed" in d
        assert "raw_components" in d
        # JSON-serialisable
        json.dumps(d)

    def test_gate_reasons_populated_on_failure(self):
        r = compute_composite(_make_report(raw_droop=99.0), _make_assembly(), "4dof_arm")
        assert len(r.gate_reasons) >= 1
        assert isinstance(r.gate_reasons, list)


# ---------------------------------------------------------------------------
# Loader against the real run archive (integration-ish but no heavy deps)
# ---------------------------------------------------------------------------


class TestRealArchive:
    """Smoke-test against data/runs/ if it exists. Skipped otherwise.

    Note: the historical run archive predates the structured ``metrics`` block
    (block A of the scoring refactor), so most legacy reports gate-fail on
    ``raw_droop_deg missing``.  This test only asserts the loader does not
    crash and returns a well-formed result — it does NOT assert Q > 0, because
    Q > 0 requires metrics blocks that only new (post-refactor) runs carry.
    """

    def test_all_cases_load_without_crash(self):
        any_loaded = False
        for case in CASE_ORDER:
            runs = load_case_runs(case)
            if not runs:
                continue
            any_loaded = True
            r = compute_composite_for_case(case, reports=runs)
            assert r is not None
            assert r.q >= 0.0
            assert isinstance(r.gate_reasons, list)
            # Every report loaded must be non-zero (loader filters zeros)
            assert all(rep.get("score", 0) > 0 for rep in runs)
        if not any_loaded:
            pytest.skip("no runs in data/runs/ — smoke test needs at least one case")
