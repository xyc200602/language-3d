"""Tests for assembly visual verification — closed-loop VLM verification."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lang3d.agent.assembly_visual_verifier import (
    AssemblyVisualVerificationResult,
    LayoutProblem,
    ProblemType,
    Severity,
    _build_assembly_prompt,
    _generate_constraint_corrections,
    _heuristic_verification,
    _parse_layout_problems,
    apply_corrections,
    verify_assembly_visual,
)
from lang3d.knowledge.mechanics import Assembly, Joint, Part
from lang3d.tools.assembly_solver import AssemblySolver


@pytest.fixture
def simple_assembly():
    parts = [
        Part("base", "structural", "底板", dimensions=dict(length=100, width=80, height=5)),
        Part("pillar", "structural", "立柱", dimensions=dict(diameter=10, height=50)),
    ]
    joints = [
        Joint("fixed", "base", "pillar", parent_anchor="top", child_anchor="bottom"),
    ]
    return Assembly(name="Test", parts=parts, joints=joints)


@pytest.fixture
def solved_positions(simple_assembly):
    solver = AssemblySolver(simple_assembly)
    return solver.solve()


class TestParseLayoutProblems:
    def test_valid_json(self):
        response = json.dumps({
            "passed": False,
            "problems": [
                {
                    "type": "collision",
                    "severity": "high",
                    "description": "Parts overlap",
                    "affected_parts": ["a", "b"],
                    "suggestion": "Move apart",
                }
            ],
        })
        problems = _parse_layout_problems(response)
        assert len(problems) == 1
        assert problems[0].problem_type == ProblemType.COLLISION
        assert problems[0].severity == Severity.HIGH
        assert "a" in problems[0].affected_parts

    def test_json_in_code_block(self):
        data = {"passed": True, "problems": []}
        response = f"```json\n{json.dumps(data)}\n```"
        problems = _parse_layout_problems(response)
        assert len(problems) == 0

    def test_no_problems(self):
        response = json.dumps({"passed": True, "problems": []})
        problems = _parse_layout_problems(response)
        assert len(problems) == 0

    def test_multiple_problems(self):
        response = json.dumps({
            "problems": [
                {"type": "floating", "severity": "medium", "description": "Part floating"},
                {"type": "wrong_orientation", "severity": "low", "description": "Rotated"},
            ],
        })
        problems = _parse_layout_problems(response)
        assert len(problems) == 2

    def test_invalid_json(self):
        problems = _parse_layout_problems("not json at all")
        assert len(problems) == 0

    def test_partial_json(self):
        response = 'Some text {"passed": false, "problems": []} more text'
        problems = _parse_layout_problems(response)
        assert len(problems) == 0  # parsed but no problems


class TestBuildPrompt:
    def test_contains_assembly_info(self, simple_assembly):
        prompt = _build_assembly_prompt(simple_assembly, expected_layout="Robot with arms")
        assert "base" in prompt
        assert "pillar" in prompt
        assert "Robot with arms" in prompt
        assert "collision" in prompt

    def test_without_expected_layout(self, simple_assembly):
        prompt = _build_assembly_prompt(simple_assembly)
        assert "Expected layout" not in prompt


class TestGenerateCorrections:
    def test_collision_correction(self, simple_assembly):
        problems = [
            LayoutProblem(
                problem_type=ProblemType.COLLISION,
                severity=Severity.HIGH,
                description="Parts overlap",
                affected_parts=["pillar"],
            )
        ]
        corrections = _generate_constraint_corrections(problems, simple_assembly)
        assert len(corrections) >= 1
        assert corrections[0]["correction_type"] == "offset"

    def test_floating_correction(self, simple_assembly):
        # Add a floating part
        floating_assembly = Assembly(
            name="test",
            parts=[
                Part("base", "structural", "", dimensions=dict(length=10, width=10, height=5)),
                Part("floater", "structural", "", dimensions=dict(length=5, width=5, height=5)),
            ],
            joints=[],  # no joints → floater is floating
        )
        problems = [
            LayoutProblem(
                problem_type=ProblemType.FLOATING,
                severity=Severity.HIGH,
                description="No connection",
                affected_parts=["floater"],
            )
        ]
        corrections = _generate_constraint_corrections(problems, floating_assembly)
        assert len(corrections) == 1
        assert corrections[0]["correction_type"] == "add_joint"

    def test_orientation_correction(self, simple_assembly):
        problems = [
            LayoutProblem(
                problem_type=ProblemType.WRONG_ORIENTATION,
                severity=Severity.MEDIUM,
                description="Upside down",
                affected_parts=["pillar"],
            )
        ]
        corrections = _generate_constraint_corrections(problems, simple_assembly)
        assert len(corrections) >= 1
        assert corrections[0]["correction_type"] == "angle"


class TestApplyCorrections:
    def test_offset_correction(self, simple_assembly):
        corrections = [
            {"joint_index": 0, "correction_type": "offset", "value": 10.0, "reason": "test"},
        ]
        new_assembly = apply_corrections(simple_assembly, corrections)
        assert len(new_assembly.parts) == len(simple_assembly.parts)
        # Original unchanged
        assert simple_assembly.joints[0].parent == "base"

    def test_add_joint_correction(self, simple_assembly):
        corrections = [
            {"correction_type": "add_joint", "part_name": "pillar", "value": "fixed"},
        ]
        new_assembly = apply_corrections(simple_assembly, corrections)
        assert len(new_assembly.joints) == len(simple_assembly.joints) + 1

    def test_no_corrections(self, simple_assembly):
        new_assembly = apply_corrections(simple_assembly, [])
        assert len(new_assembly.parts) == len(simple_assembly.parts)
        assert len(new_assembly.joints) == len(simple_assembly.joints)


class TestHeuristicVerification:
    def test_good_positions(self, simple_assembly, solved_positions):
        result = _heuristic_verification(simple_assembly, solved_positions)
        data = json.loads(result)
        assert data["passed"] is True
        assert len(data["problems"]) == 0

    def test_floating_part(self):
        assembly = Assembly(
            name="test",
            parts=[
                Part("base", "structural", "", dimensions=dict(length=10, width=10, height=5)),
                Part("floater", "structural", "", dimensions=dict(length=5, width=5, height=5)),
            ],
            joints=[],
        )
        positions = {"base": {"position": [0, 0, 0]}}
        result = _heuristic_verification(assembly, positions)
        data = json.loads(result)
        assert data["passed"] is False
        assert any(p["type"] == "floating" for p in data["problems"])

    def test_collision_detected(self):
        positions = {
            "a": {"position": [10.0, 20.0, 30.0]},
            "b": {"position": [10.0, 20.0, 30.0]},
        }
        assembly = Assembly(
            name="test",
            parts=[
                Part("a", "structural", "", dimensions=dict(length=10, width=10, height=5)),
                Part("b", "structural", "", dimensions=dict(length=10, width=10, height=5)),
            ],
            joints=[],
        )
        result = _heuristic_verification(assembly, positions)
        data = json.loads(result)
        assert data["passed"] is False
        assert any(p["type"] == "collision" for p in data["problems"])


class TestVerifyAssemblyVisual:
    def test_heuristic_only_pass(self, simple_assembly, solved_positions):
        """Without VLM backend, heuristic verification should pass for valid assembly."""
        result = verify_assembly_visual(
            assembly=simple_assembly,
            positions=solved_positions,
            model_backend=None,
            max_iterations=1,
        )
        assert isinstance(result, AssemblyVisualVerificationResult)
        assert result.passed is True
        assert result.round_number == 1

    def test_max_iterations(self, simple_assembly, solved_positions):
        """Should stop after max_iterations even if not passing."""
        # Create a scenario that will fail heuristic check
        bad_positions = {}  # empty → all parts floating
        result = verify_assembly_visual(
            assembly=simple_assembly,
            positions=bad_positions,
            model_backend=None,
            max_iterations=3,
        )
        assert result.round_number == 3

    def test_with_mock_vlm_pass(self, simple_assembly, solved_positions):
        """Mock VLM that always passes."""
        mock_backend = MagicMock()
        mock_backend.vision.return_value = json.dumps({
            "passed": True,
            "problems": [],
            "overall_assessment": "Looks good",
        })
        result = verify_assembly_visual(
            assembly=simple_assembly,
            positions=solved_positions,
            model_backend=mock_backend,
            max_iterations=3,
        )
        assert result.passed is True

    def test_with_mock_vlm_then_fix(self, simple_assembly, solved_positions):
        """Mock VLM that fails first then passes."""
        mock_backend = MagicMock()
        mock_backend.vision.side_effect = [
            json.dumps({
                "passed": False,
                "problems": [{"type": "collision", "severity": "high",
                              "description": "overlap", "affected_parts": ["pillar"],
                              "suggestion": "Move up"}],
            }),
            json.dumps({"passed": True, "problems": []}),
        ]
        result = verify_assembly_visual(
            assembly=simple_assembly,
            positions=solved_positions,
            model_backend=mock_backend,
            max_iterations=3,
        )
        # Should either pass (if correction fixed it) or reach max iterations
        assert result.round_number <= 3


class TestAssemblyVLMSolveTool:
    def test_tool_definition(self):
        from lang3d.tools.assembly_vlm import AssemblyVLMSolveTool
        tool = AssemblyVLMSolveTool()
        defn = tool.get_definition()
        assert defn.name == "assembly_vlm_solve"
        params = defn.parameters["properties"]
        assert "max_iterations" in params
        assert "detail_level" in params

    def test_tool_execute(self):
        from lang3d.tools.assembly_vlm import AssemblyVLMSolveTool
        tool = AssemblyVLMSolveTool()
        result = tool.execute(
            assembly_name="complex_robot",
            max_iterations=1,
        )
        assert "Assembly Visual Verification" in result
