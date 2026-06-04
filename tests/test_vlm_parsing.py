"""Tests for VLM verification JSON parsing robustness."""

from __future__ import annotations

from lang3d.tools.vlm import _parse_verification_json


class TestPlainJsonParsing:
    """Strategy 1/2: Plain JSON object parsing."""

    def test_simple_match_true(self):
        raw = '{"match": true, "observed": "A 30mm cube", "differences": null, "suggestion": null, "fix_commands": null}'
        result = _parse_verification_json(raw)
        assert result["match"] is True
        assert result["observed"] == "A 30mm cube"

    def test_simple_match_false(self):
        raw = '{"match": false, "observed": "A cube without hole", "differences": "Missing center hole", "suggestion": "Add cylinder cut", "fix_commands": null}'
        result = _parse_verification_json(raw)
        assert result["match"] is False
        assert "Missing center hole" in result["differences"]


class TestMarkdownJsonParsing:
    """Strategy 1: Markdown code block wrapped JSON."""

    def test_json_in_markdown_block(self):
        raw = 'Here is the verification:\n```json\n{"match": true, "observed": "A cube", "differences": null, "suggestion": null, "fix_commands": null}\n```\nThe model looks correct.'
        result = _parse_verification_json(raw)
        assert result["match"] is True

    def test_json_in_plain_code_block(self):
        raw = '```\n{"match": false, "observed": "Empty", "differences": "No model", "suggestion": "Create model", "fix_commands": null}\n```'
        result = _parse_verification_json(raw)
        assert result["match"] is False

    def test_markdown_with_confidence(self):
        raw = '```json\n{"match": true, "observed": "A box", "differences": null, "suggestion": null, "fix_commands": null, "confidence": "high"}\n```'
        result = _parse_verification_json(raw)
        assert result["match"] is True


class TestMultilineJsonParsing:
    """Strategy 2: Bracket depth tracking for multiline JSON."""

    def test_multiline_json(self):
        raw = """Based on my analysis:
{
    "match": false,
    "observed": "A cylindrical shape",
    "differences": "Expected a cube but got a cylinder",
    "suggestion": "Use box instead of cylinder",
    "fix_commands": null
}
Please fix the model."""
        result = _parse_verification_json(raw)
        assert result["match"] is False
        assert "cylinder" in result["observed"].lower() or "cylindrical" in result["observed"].lower()

    def test_nested_json(self):
        raw = '{"match": true, "observed": "A 50x50x50mm cube with rounded edges", "differences": null, "suggestion": null, "fix_commands": {"type": "none"}}'
        result = _parse_verification_json(raw)
        assert result["match"] is True


class TestFallbackRegexParsing:
    """Strategy 3: Field-by-field regex extraction."""

    def test_plain_text_fields(self):
        raw = (
            "match: true\n"
            "observed: A blue cube\n"
            "differences: None\n"
            "suggestion: None\n"
            "fix_commands: None"
        )
        result = _parse_verification_json(raw)
        assert result["match"] is True
        assert result["observed"] == "A blue cube"

    def test_quoted_fields(self):
        raw = 'match: "false" observed: "Empty viewport" differences: "No shape" suggestion: "Create shape" fix_commands: "None"'
        result = _parse_verification_json(raw)
        assert result["match"] is False


class TestChineseResponseParsing:
    """Chinese language VLM response parsing."""

    def test_chinese_json(self):
        raw = '{"match": true, "observed": "一个30毫米的立方体", "differences": null, "suggestion": null, "fix_commands": null}'
        result = _parse_verification_json(raw)
        assert result["match"] is True
        assert "立方体" in result["observed"]

    def test_chinese_markdown(self):
        raw = '根据分析：\n```json\n{"match": false, "observed": "空的工作区", "differences": "没有任何模型", "suggestion": "创建模型", "fix_commands": null}\n```'
        result = _parse_verification_json(raw)
        assert result["match"] is False


class TestMixedCaseMatch:
    """Mixed case and alternate boolean values."""

    def test_capital_true(self):
        raw = '{"match": True, "observed": "OK", "differences": null, "suggestion": null, "fix_commands": null}'
        result = _parse_verification_json(raw)
        assert result["match"] is True

    def test_string_true(self):
        raw = '{"match": "true", "observed": "OK", "differences": null, "suggestion": null, "fix_commands": null}'
        result = _parse_verification_json(raw)
        assert result["match"] is True

    def test_string_yes(self):
        raw = 'match: yes\nobserved: A cube\ndifferences: None'
        result = _parse_verification_json(raw)
        assert result["match"] is True

    def test_none_vs_null(self):
        raw = '{"match": true, "observed": "OK", "differences": "None", "suggestion": null, "fix_commands": "null"}'
        result = _parse_verification_json(raw)
        assert result["match"] is True
        # All should normalize to "None"
        assert result["differences"] == "None"
        assert result["suggestion"] == "None"
        assert result["fix_commands"] == "None"
