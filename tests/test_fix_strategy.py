"""Tests for failure classification and fix strategies."""

from __future__ import annotations

from lang3d.agent.fix_strategy import (
    FailureType,
    FixContext,
    check_convergence,
    classify_failure,
    generate_fix_hint,
)


class TestClassifyFailure:
    """Failure type classification tests."""

    def test_missing_feature(self):
        result = (
            "MATCH: False\n"
            "OBSERVED: A solid block\n"
            "DIFFERENCES: Missing center hole\n"
            "SUGGESTION: Add a cylindrical cut"
        )
        ctx = classify_failure(result)
        assert ctx.failure_type == FailureType.MISSING_FEATURE

    def test_wrong_dimension(self):
        result = (
            "MATCH: False\n"
            "OBSERVED: A 40mm cube\n"
            "DIFFERENCES: Dimension incorrect, expected 30mm\n"
            "SUGGESTION: Change size to 30mm"
        )
        ctx = classify_failure(result)
        assert ctx.failure_type == FailureType.WRONG_DIMENSION

    def test_wrong_position(self):
        result = (
            "MATCH: False\n"
            "OBSERVED: Hole at edge\n"
            "DIFFERENCES: Position offset, hole should be centered\n"
            "SUGGESTION: Move hole to center"
        )
        ctx = classify_failure(result)
        assert ctx.failure_type == FailureType.WRONG_POSITION

    def test_wrong_shape(self):
        result = (
            "MATCH: False\n"
            "OBSERVED: A cylinder\n"
            "DIFFERENCES: Wrong shape, expected a cube instead of cylinder\n"
            "SUGGESTION: Use box instead"
        )
        ctx = classify_failure(result)
        assert ctx.failure_type == FailureType.WRONG_SHAPE

    def test_assembly_error(self):
        result = (
            "MATCH: False\n"
            "OBSERVED: Parts separated\n"
            "DIFFERENCES: Assembly error, parts not aligned\n"
            "SUGGESTION: Fix assembly constraint"
        )
        ctx = classify_failure(result)
        assert ctx.failure_type == FailureType.ASSEMBLY_ERROR

    def test_unknown_failure(self):
        result = "MATCH: False\nOBSERVED: Something\nDIFFERENCES: Unknown issue"
        ctx = classify_failure(result)
        # Should not crash, and classify to something
        assert isinstance(ctx.failure_type, FailureType)

    def test_extracts_target_feature(self):
        result = "MATCH: False\nOBSERVED: Block\nDIFFERENCES: Missing 10mm diameter hole\nSUGGESTION: Add hole"
        ctx = classify_failure(result)
        assert "10mm" in ctx.target_feature or "hole" in ctx.target_feature.lower()


class TestGenerateFixHint:
    """Fix hint generation tests."""

    def test_missing_feature_hint(self):
        ctx = FixContext(
            failure_type=FailureType.MISSING_FEATURE,
            target_feature="Missing center hole",
        )
        hint = generate_fix_hint(ctx)
        assert "缺少" in hint or "添加" in hint
        assert "Missing center hole" in hint

    def test_dimension_hint_mentions_units(self):
        ctx = FixContext(
            failure_type=FailureType.WRONG_DIMENSION,
            target_feature="Size wrong",
            expected_value="30mm cube",
        )
        hint = generate_fix_hint(ctx)
        assert "尺寸" in hint
        assert "30mm cube" in hint

    def test_unknown_fallback(self):
        ctx = FixContext(failure_type=FailureType.UNKNOWN)
        hint = generate_fix_hint(ctx)
        assert "不匹配" in hint or "DIFFERENCES" in hint


class TestCheckConvergence:
    """Convergence detection tests."""

    def test_no_previous_fixes(self):
        assert check_convergence([], "some result") is False

    def test_similar_results_converged(self):
        prev = [
            "MATCH: False\nOBSERVED: Block without hole\nDIFFERENCES: Missing center hole 10mm",
        ]
        current = "MATCH: False\nOBSERVED: Block without hole\nDIFFERENCES: Missing center hole 10mm diameter"
        assert check_convergence(prev, current) is True

    def test_different_results_not_converged(self):
        prev = [
            "MATCH: False\nOBSERVED: No model\nDIFFERENCES: Empty workspace",
        ]
        current = "MATCH: False\nOBSERVED: Block\nDIFFERENCES: Missing hole"
        assert check_convergence(prev, current) is False

    def test_multiple_similar_converged(self):
        prev = [
            "MATCH: False\nDIFFERENCES: Hole size wrong, 8mm instead of 10mm",
            "MATCH: False\nDIFFERENCES: Hole size wrong, 9mm instead of 10mm",
        ]
        current = "MATCH: False\nDIFFERENCES: Hole size wrong, 8.5mm instead of 10mm"
        assert check_convergence(prev, current) is True
