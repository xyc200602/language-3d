"""Tests for VLM multi-angle verification."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lang3d.tools.vlm import (
    CADVerifyTool,
    _aggregate_angle_results,
    _normalize_verification,
    _parse_verification_json,
)


# ---------------------------------------------------------------------------
# Multi-angle parsing
# ---------------------------------------------------------------------------

class TestMultiAngleParsing:
    def test_empty_angles(self):
        """Empty angles string should result in no angle_list."""
        angles = ""
        angle_list = [a.strip() for a in angles.split(",") if a.strip()]
        assert angle_list == []

    def test_single_angle(self):
        angles = "isometric"
        angle_list = [a.strip() for a in angles.split(",") if a.strip()]
        assert angle_list == ["isometric"]

    def test_multiple_angles(self):
        angles = "isometric,front,top"
        angle_list = [a.strip() for a in angles.split(",") if a.strip()]
        assert angle_list == ["isometric", "front", "top"]

    def test_angles_with_whitespace(self):
        angles = " isometric , front , top "
        angle_list = [a.strip() for a in angles.split(",") if a.strip()]
        assert angle_list == ["isometric", "front", "top"]

    def test_angles_with_empty_entries(self):
        angles = "isometric,,front,"
        angle_list = [a.strip() for a in angles.split(",") if a.strip()]
        assert angle_list == ["isometric", "front"]


# ---------------------------------------------------------------------------
# Verification prompt
# ---------------------------------------------------------------------------

class TestVerificationPrompt:
    def test_prompt_contains_lenient_instructions(self):
        prompt = CADVerifyTool._build_verify_prompt("a box")
        assert "GENEROUS" in prompt
        assert "topological" in prompt

    def test_prompt_contains_expected_model(self):
        prompt = CADVerifyTool._build_verify_prompt("80x60x8 plate with 4 holes")
        assert "80x60x8 plate with 4 holes" in prompt

    def test_prompt_contains_match_example(self):
        prompt = CADVerifyTool._build_verify_prompt("test model")
        assert "Typical match example" in prompt

    def test_prompt_contains_confidence_field(self):
        prompt = CADVerifyTool._build_verify_prompt("test model")
        assert "confidence" in prompt


# ---------------------------------------------------------------------------
# Angle aggregation / voting
# ---------------------------------------------------------------------------

class TestAngleAggregation:
    def test_all_match(self):
        results = [
            {"match": True, "confidence": "high"},
            {"match": True, "confidence": "high"},
            {"match": True, "confidence": "high"},
        ]
        assert _aggregate_angle_results(results) is True

    def test_all_mismatch(self):
        results = [
            {"match": False, "confidence": "high"},
            {"match": False, "confidence": "high"},
            {"match": False, "confidence": "high"},
        ]
        assert _aggregate_angle_results(results) is False

    def test_majority_match(self):
        results = [
            {"match": True, "confidence": "medium"},
            {"match": True, "confidence": "medium"},
            {"match": False, "confidence": "medium"},
        ]
        # 2/3 match with equal weights -> 2/3 > 0.5 -> True
        assert _aggregate_angle_results(results) is True

    def test_minority_match(self):
        results = [
            {"match": True, "confidence": "medium"},
            {"match": False, "confidence": "medium"},
            {"match": False, "confidence": "medium"},
        ]
        # 1/3 match -> 1/3 <= 0.5 -> False
        assert _aggregate_angle_results(results) is False

    def test_high_confidence_outweighs(self):
        results = [
            {"match": True, "confidence": "high"},   # weight 2.0
            {"match": False, "confidence": "low"},    # weight 0.5
        ]
        # match_weight = 2.0, total = 2.5 -> 2.0/2.5 = 0.8 > 0.5 -> True
        assert _aggregate_angle_results(results) is True

    def test_low_confidence_doesnt_outweigh(self):
        results = [
            {"match": True, "confidence": "low"},    # weight 0.5
            {"match": False, "confidence": "high"},   # weight 2.0
        ]
        # match_weight = 0.5, total = 2.5 -> 0.5/2.5 = 0.2 <= 0.5 -> False
        assert _aggregate_angle_results(results) is False

    def test_empty_results(self):
        assert _aggregate_angle_results([]) is False

    def test_tie_breaking(self):
        results = [
            {"match": True, "confidence": "medium"},   # 1.0
            {"match": False, "confidence": "medium"},   # 1.0
        ]
        # match_weight = 1.0, total = 2.0 -> 0.5 NOT > 0.5 -> False
        assert _aggregate_angle_results(results) is False

    def test_missing_confidence_defaults_medium(self):
        results = [
            {"match": True},
            {"match": True},
            {"match": False},
        ]
        # All default to medium (1.0). 2/3 > 0.5 -> True
        assert _aggregate_angle_results(results) is True


# ---------------------------------------------------------------------------
# CADVerifyTool definition
# ---------------------------------------------------------------------------

class TestCADVerifyToolDefinition:
    def test_has_angles_parameter(self):
        router = MagicMock()
        tool = CADVerifyTool(router)
        defn = tool.get_definition()
        assert "angles" in defn.parameters["properties"]

    def test_angles_description(self):
        router = MagicMock()
        tool = CADVerifyTool(router)
        defn = tool.get_definition()
        desc = defn.parameters["properties"]["angles"]["description"]
        assert "multi-view" in desc.lower() or "angle" in desc.lower()

    def test_required_fields(self):
        router = MagicMock()
        tool = CADVerifyTool(router)
        defn = tool.get_definition()
        assert "expected" in defn.parameters["required"]
        assert "angles" not in defn.parameters["required"]


# ---------------------------------------------------------------------------
# _parse_verification_json with confidence
# ---------------------------------------------------------------------------

class TestParseVerificationWithConfidence:
    def test_extracts_confidence(self):
        raw = '{"match": true, "observed": "a box", "differences": null, "suggestion": null, "fix_commands": null, "confidence": "high"}'
        result = _parse_verification_json(raw)
        assert result["confidence"] == "high"

    def test_missing_confidence(self):
        raw = '{"match": true, "observed": "a box", "differences": null}'
        result = _parse_verification_json(raw)
        # confidence may not be present (no assertion on value)
        assert result["match"] is True

    def test_confidence_in_markdown_code_block(self):
        raw = '```json\n{"match": false, "observed": "test", "confidence": "low"}\n```'
        result = _parse_verification_json(raw)
        assert result["confidence"] == "low"
