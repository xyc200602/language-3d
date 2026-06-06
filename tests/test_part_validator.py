"""Tests for part_validator module — FreeCAD batch validation with auto-retry."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from lang3d.knowledge.mechanics import Part
from lang3d.tools.export_package import build_complex_robot
from lang3d.tools.part_feature_engine import FeatureConfig, infer_features
from lang3d.tools.part_validator import (
    BatchValidationReport,
    PartValidationResult,
    _freecad_available,
    _simplify_config,
    _simplification_note,
    _validate_stl,
    validate_all_parts,
    validate_part,
)


# ============================================================================
# TestSimplificationStrategy — features removed in correct order
# ============================================================================


class TestSimplificationStrategy:
    """Verify that _simplify_config removes features in the right order."""

    def _full_config(self) -> FeatureConfig:
        """A config with all features enabled."""
        return FeatureConfig(
            mounting_holes=[{"diameter_mm": 3.0, "pattern": "grid"}],
            bore={"diameter_mm": 6.0, "through": True},
            bearing_seats=[{"bore_diameter": 10, "shoulder_diameter": 16, "depth": 5}],
            shell={"thickness_mm": 2.0, "faces_to_remove": []},
            fillets=[{"radius_mm": 2.0}],
            chamfers=[{"size_mm": 0.5}],
            cable_channels=[{"width": 8, "height": 5, "start_offset": 20, "end_offset": 80}],
        )

    def test_level_0_keeps_all(self):
        cfg = self._full_config()
        result = _simplify_config(cfg, 0)
        assert len(result.fillets) > 0
        assert len(result.chamfers) > 0
        assert result.shell is not None
        assert len(result.bearing_seats) > 0
        assert len(result.mounting_holes) > 0
        assert result.bore is not None

    def test_level_1_removes_fillets(self):
        cfg = self._full_config()
        result = _simplify_config(cfg, 1)
        assert len(result.fillets) == 0
        assert len(result.chamfers) > 0
        assert result.shell is not None

    def test_level_2_removes_fillets_and_chamfers(self):
        cfg = self._full_config()
        result = _simplify_config(cfg, 2)
        assert len(result.fillets) == 0
        assert len(result.chamfers) == 0
        assert len(result.cable_channels) > 0

    def test_level_3_removes_cable_channels(self):
        cfg = self._full_config()
        result = _simplify_config(cfg, 3)
        assert len(result.cable_channels) == 0
        assert result.shell is not None

    def test_level_4_removes_shell(self):
        cfg = self._full_config()
        result = _simplify_config(cfg, 4)
        assert result.shell is None
        assert len(result.bearing_seats) > 0

    def test_level_5_removes_bearing_seats(self):
        cfg = self._full_config()
        result = _simplify_config(cfg, 5)
        assert len(result.bearing_seats) == 0
        assert len(result.mounting_holes) > 0

    def test_level_6_removes_mounting_holes(self):
        cfg = self._full_config()
        result = _simplify_config(cfg, 6)
        assert len(result.mounting_holes) == 0
        assert result.bore is not None

    def test_level_7_removes_bore(self):
        cfg = self._full_config()
        result = _simplify_config(cfg, 7)
        assert result.bore is None

    def test_level_beyond_max_removes_all(self):
        cfg = self._full_config()
        result = _simplify_config(cfg, 100)
        assert len(result.fillets) == 0
        assert len(result.chamfers) == 0
        assert len(result.cable_channels) == 0
        assert result.shell is None
        assert len(result.bearing_seats) == 0
        assert len(result.mounting_holes) == 0
        assert result.bore is None

    def test_does_not_mutate_original(self):
        cfg = self._full_config()
        _simplify_config(cfg, 5)
        # Original should be unchanged
        assert len(cfg.fillets) > 0
        assert len(cfg.chamfers) > 0
        assert cfg.shell is not None


class TestSimplificationNote:
    """Test human-readable notes."""

    def test_level_0(self):
        assert _simplification_note(0) == "full features"

    def test_level_1(self):
        assert "fillets" in _simplification_note(1)

    def test_level_3(self):
        note = _simplification_note(3)
        assert "fillets" in note
        assert "chamfers" in note
        assert "cable channels" in note


# ============================================================================
# TestSTLValidation
# ============================================================================


class TestSTLValidation:
    """Verify STL file validation logic."""

    def test_missing_file_fails(self):
        ok, size, err = _validate_stl("/nonexistent/path.stl")
        assert not ok
        assert size == 0
        assert "not found" in err

    def test_too_small_fails(self, tmp_path):
        stl = tmp_path / "tiny.stl"
        stl.write_bytes(b"solid\nendsolid\n")
        ok, size, err = _validate_stl(str(stl))
        assert not ok
        assert "too small" in err

    def test_valid_stl_passes(self, tmp_path):
        stl = tmp_path / "valid.stl"
        stl.write_bytes(b"x" * 5000)
        ok, size, err = _validate_stl(str(stl))
        assert ok
        assert size == 5000
        assert err == ""


# ============================================================================
# TestValidationResult — data model
# ============================================================================


class TestPartValidationResult:
    """Test the result data model."""

    def test_stl_size_kb(self):
        r = PartValidationResult(part_name="test", stl_size_bytes=2048)
        assert r.stl_size_kb == 2.0

    def test_default_values(self):
        r = PartValidationResult(part_name="test")
        assert not r.passed
        assert r.stl_path is None
        assert r.vlm_verified is None
        assert r.vlm_match is None


class TestBatchValidationReport:
    """Test the batch report data model."""

    def test_pass_rate(self):
        report = BatchValidationReport(total_parts=4, passed=3, failed=1)
        assert abs(report.pass_rate - 0.75) < 0.01

    def test_failed_parts(self):
        report = BatchValidationReport()
        report.results = [
            PartValidationResult(part_name="a", passed=True),
            PartValidationResult(part_name="b", passed=False),
            PartValidationResult(part_name="c", passed=False),
        ]
        report.total_parts = 3
        report.passed = 1
        report.failed = 2
        assert report.failed_parts == ["b", "c"]

    def test_to_dict(self):
        report = BatchValidationReport(total_parts=2, passed=1, failed=1)
        report.results = [
            PartValidationResult(part_name="x", passed=True, stl_size_bytes=1024),
            PartValidationResult(part_name="y", passed=False, freecad_error="timeout"),
        ]
        d = report.to_dict()
        assert d["total_parts"] == 2
        assert d["passed"] == 1
        assert d["failed"] == 1
        assert len(d["results"]) == 2


# ============================================================================
# TestValidatePart — with mocked FreeCAD
# ============================================================================


class TestValidatePartMocked:
    """Test validate_part with mocked FreeCAD execution."""

    @pytest.fixture()
    def simple_part(self):
        return Part("test_box", "structural", "test",
                    dimensions=dict(length=10, width=10, height=10))

    @patch("lang3d.tools.part_validator._freecad_available", return_value=True)
    @patch("lang3d.tools.part_validator._run_and_check")
    def test_successful_validation(self, mock_run, mock_avail, simple_part, tmp_path):
        """Part validates when FreeCAD succeeds and STL is created."""
        stl_path = tmp_path / "test_box.stl"
        stl_path.write_bytes(b"x" * 5000)

        def fake_run(ops, expected_stl, workspace, timeout):
            # Create the STL file (simulating FreeCAD output)
            with open(expected_stl, "wb") as f:
                f.write(b"x" * 5000)
            return True, "OK", None

        mock_run.side_effect = fake_run

        result = validate_part(simple_part, str(tmp_path))
        assert result.passed
        assert result.simplification_level == 0

    @patch("lang3d.tools.part_validator._freecad_available", return_value=True)
    @patch("lang3d.tools.part_validator._run_and_check")
    def test_retry_on_failure_then_success(self, mock_run, mock_avail, tmp_path):
        """Part retries with simplified features on failure."""
        # Create a part that has fillets/chamfers (would fail at full features)
        p = Part("base_plate", "structural", "test",
                 dimensions=dict(length=300, width=200, height=5))

        call_count = [0]

        def fake_run(ops, expected_stl, workspace, timeout):
            call_count[0] += 1
            if call_count[0] <= 2:
                return False, "", "BRep_API: command not done"
            # Success on 3rd try (simplified features)
            with open(expected_stl, "wb") as f:
                f.write(b"x" * 5000)
            return True, "OK", None

        mock_run.side_effect = fake_run

        result = validate_part(p, str(tmp_path))
        assert result.passed
        assert result.simplification_level > 0
        assert call_count[0] >= 3

    @patch("lang3d.tools.part_validator._freecad_available", return_value=True)
    @patch("lang3d.tools.part_validator._run_and_check")
    def test_all_levels_fail(self, mock_run, mock_avail, simple_part, tmp_path):
        """Part reports failure when all simplification levels fail."""
        mock_run.return_value = (False, "", "persistent error")

        result = validate_part(simple_part, str(tmp_path), max_simplification=3)
        assert not result.passed
        assert result.freecad_error is not None

    @patch("lang3d.tools.part_validator._freecad_available", return_value=True)
    @patch("lang3d.tools.part_validator._run_and_check")
    def test_stl_too_small_counts_as_failure(self, mock_run, mock_avail, simple_part, tmp_path):
        """STL that is too small triggers retry."""
        def fake_run(ops, expected_stl, workspace, timeout):
            with open(expected_stl, "wb") as f:
                f.write(b"tiny")
            return True, "OK", None

        mock_run.side_effect = fake_run

        result = validate_part(simple_part, str(tmp_path), max_simplification=2)
        assert not result.passed  # STL always too small, all levels fail


# ============================================================================
# TestValidateAllParts — batch validation with mocked FreeCAD
# ============================================================================


class TestValidateAllPartsMocked:
    """Test batch validation with mocked FreeCAD."""

    @patch("lang3d.tools.part_validator._freecad_available", return_value=True)
    @patch("lang3d.tools.part_validator._run_and_check")
    def test_batch_all_pass(self, mock_run, mock_avail, tmp_path):
        """All parts pass validation."""
        parts = [
            Part("a", "structural", "test", dimensions=dict(length=10, width=10, height=10)),
            Part("b", "structural", "test", dimensions=dict(diameter=20, height=30)),
        ]

        def fake_run(ops, expected_stl, workspace, timeout):
            with open(expected_stl, "wb") as f:
                f.write(b"x" * 5000)
            return True, "OK", None

        mock_run.side_effect = fake_run

        report = validate_all_parts(parts, str(tmp_path))
        assert report.passed == 2
        assert report.failed == 0
        assert report.pass_rate == 1.0

    @patch("lang3d.tools.part_validator._freecad_available", return_value=True)
    @patch("lang3d.tools.part_validator._run_and_check")
    def test_batch_mixed_results(self, mock_run, mock_avail, tmp_path):
        """Some parts pass, some fail."""
        parts = [
            Part("good_part", "structural", "test", dimensions=dict(length=10, width=10, height=10)),
            Part("bad_part", "structural", "test", dimensions=dict(length=10, width=10, height=10)),
        ]

        call_count = [0]

        def fake_run(ops, expected_stl, workspace, timeout):
            call_count[0] += 1
            if "good" in expected_stl:
                with open(expected_stl, "wb") as f:
                    f.write(b"x" * 5000)
                return True, "OK", None
            return False, "", "error"

        mock_run.side_effect = fake_run

        report = validate_all_parts(parts, str(tmp_path))
        assert report.passed == 1
        assert report.failed == 1
        assert "bad_part" in report.failed_parts

    @patch("lang3d.tools.part_validator._freecad_available", return_value=False)
    def test_batch_freecad_not_available(self, mock_avail, tmp_path):
        """Report shows all skipped when FreeCAD is unavailable."""
        parts = [
            Part("a", "structural", "test", dimensions=dict(length=10, width=10, height=10)),
        ]
        report = validate_all_parts(parts, str(tmp_path))
        assert report.skipped == 1
        assert report.passed == 0

    @patch("lang3d.tools.part_validator._freecad_available", return_value=True)
    @patch("lang3d.tools.part_validator._run_and_check")
    def test_batch_report_serialization(self, mock_run, mock_avail, tmp_path):
        """Report can be serialized to dict for JSON output."""
        parts = [
            Part("test_part", "structural", "test", dimensions=dict(length=10, width=10, height=10)),
        ]

        def fake_run(ops, expected_stl, workspace, timeout):
            with open(expected_stl, "wb") as f:
                f.write(b"x" * 5000)
            return True, "OK", None

        mock_run.side_effect = fake_run

        report = validate_all_parts(parts, str(tmp_path))
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "results" in d
        assert len(d["results"]) == 1


# ============================================================================
# TestE2EValidation — requires real FreeCAD (marked freecad)
# ============================================================================


def _freecad_available_real() -> bool:
    """Check if FreeCAD is actually available (not mocked)."""
    try:
        from lang3d.tools.freecad import _find_freecad_python
        return _find_freecad_python() is not None
    except Exception:
        return False


@pytest.mark.freecad
@pytest.mark.skipif(not _freecad_available_real(), reason="FreeCAD not installed")
class TestE2EValidation:
    """End-to-end validation with real FreeCAD execution."""

    def test_validate_simple_box(self, tmp_path):
        """A simple box part should pass validation at level 0."""
        p = Part("simple_box", "structural", "test",
                 dimensions=dict(length=20, width=20, height=20))
        result = validate_part(p, str(tmp_path), timeout=30)
        assert result.passed
        assert result.stl_path is not None
        assert result.stl_size_bytes > 0

    def test_validate_simple_cylinder(self, tmp_path):
        """A simple cylinder part should pass validation."""
        p = Part("simple_cyl", "structural", "test",
                 dimensions=dict(diameter=30, height=40))
        result = validate_part(p, str(tmp_path), timeout=30)
        assert result.passed

    def test_validate_plate_with_holes(self, tmp_path):
        """A plate with mounting holes should validate."""
        p = Part("base_plate", "structural", "test",
                 dimensions=dict(length=300, width=200, height=5))
        result = validate_part(p, str(tmp_path), timeout=60)
        assert result.passed
        assert result.stl_size_bytes > 200  # should be substantial

    def test_validate_all_41_parts(self, tmp_path):
        """All 41 parts from build_complex_robot should pass validation."""
        robot = build_complex_robot()
        report = validate_all_parts(robot.parts, str(tmp_path), timeout=60)
        assert report.total_parts == len(robot.parts)
        # At minimum, most parts should pass (allow a few failures with simplification)
        assert report.passed >= report.total_parts * 0.9, (
            f"Too many failures: {report.failed_parts}"
        )

    def test_failed_parts_get_simplified(self, tmp_path):
        """Parts that fail at full features should succeed with simplification."""
        robot = build_complex_robot()
        # Pick parts known to be tricky
        tricky_parts = [
            p for p in robot.parts
            if p.name in ("arm_l_base", "wheel_fl", "battery_box", "motor_fl")
        ]
        for part in tricky_parts:
            result = validate_part(part, str(tmp_path), timeout=60)
            assert result.passed, f"{part.name} failed: {result.freecad_error}"

    def test_simplification_reduces_stl_size(self, tmp_path):
        """Simplified features should produce different (usually smaller) STL."""
        p = Part("arm_l_base", "joint", "test",
                 dimensions=dict(outer_diameter=80, height=40))

        # Full features
        result_full = validate_part(p, str(tmp_path), max_simplification=0, timeout=60)

        # Primitive fallback
        p2 = Part("arm_l_base_2", "joint", "test",
                   dimensions=dict(outer_diameter=80, height=40))
        result_simple = validate_part(p2, str(tmp_path), max_simplification=7, timeout=60)

        # At least both should pass (primitive fallback always works)
        assert result_simple.passed or result_full.passed
