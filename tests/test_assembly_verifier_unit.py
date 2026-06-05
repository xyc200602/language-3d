"""Unit tests for AssemblyVerifier."""

from __future__ import annotations

import pytest
from pathlib import Path

from lang3d.agent.assembly_verifier import (
    AssemblyVerifier,
    AssemblyVerificationResult,
    CollisionCheck,
    FitCheck,
    PartCheck,
)
from lang3d.knowledge.mechanics import (
    Assembly,
    Joint,
    Part,
    PRINT_TOLERANCES,
    ROBOTIC_ARM_ASSEMBLY,
)


class TestFitCheck:
    def test_good_sliding_fit(self):
        """Shaft 11mm, hole 11.6mm -> clearance 0.6mm >= sliding_fit 0.3mm."""
        parent = Part(
            name="housing",
            category="joint",
            description="Housing",
            dimensions={"outer_diameter": 80, "wall_thickness": 5, "height": 40},
        )
        child = Part(
            name="shaft",
            category="joint",
            description="Shaft",
            dimensions={"shaft_diameter": 11, "height": 35},
        )
        assembly = Assembly(
            name="test",
            parts=[parent, child],
            joints=[Joint("revolute", "housing", "shaft", description="rotation")],
        )

        verifier = AssemblyVerifier()
        checks = verifier.check_joint_fits(assembly)
        assert len(checks) == 1

        # hole = 80 - 2*5 = 70mm, shaft = 11mm, clearance = 59mm >> 0.3mm
        assert checks[0].fits
        assert checks[0].clearance > 0

    def test_tight_fit_fail(self):
        """Fixed joint with insufficient clearance."""
        parent = Part(
            name="block",
            category="structural",
            description="Block",
            dimensions={"outer_diameter": 30, "wall_thickness": 12, "height": 20},
        )
        child = Part(
            name="peg",
            category="structural",
            description="Peg",
            dimensions={"shaft_diameter": 5.9, "height": 15},
        )
        assembly = Assembly(
            name="test",
            parts=[parent, child],
            joints=[Joint("fixed", "block", "peg", description="press fit")],
        )

        verifier = AssemblyVerifier()
        checks = verifier.check_joint_fits(assembly)
        assert len(checks) == 1
        # hole = 30 - 2*12 = 6mm, shaft = 5.9mm, clearance = 0.1mm < tight_fit 0.15mm
        assert not checks[0].fits

    def test_incomplete_dimensions(self):
        """Parts without relevant dimensions should note incompleteness."""
        parent = Part(name="a", category="x", description="a")
        child = Part(name="b", category="x", description="b")
        assembly = Assembly(
            name="test",
            parts=[parent, child],
            joints=[Joint("fixed", "a", "b")],
        )

        verifier = AssemblyVerifier()
        checks = verifier.check_joint_fits(assembly)
        assert len(checks) == 1
        assert checks[0].fits  # Default to True when can't check
        assert "不完整" in checks[0].notes


class TestPartCompleteness:
    def test_parts_in_workspace(self, tmp_path):
        # Create dummy files
        (tmp_path / "base_plate.fcstd").write_text("dummy")
        (tmp_path / "servo_holder.step").write_text("dummy")

        assembly = Assembly(
            name="test",
            parts=[
                Part(name="base_plate", category="x", description="base"),
                Part(name="servo_holder", category="x", description="servo"),
                Part(name="arm", category="x", description="arm"),
            ],
        )

        verifier = AssemblyVerifier()
        checks = verifier.check_part_completeness(assembly, tmp_path)
        assert len(checks) == 3
        assert checks[0].exists  # base_plate.fcstd found
        assert checks[1].exists  # servo_holder.step found
        assert not checks[2].exists  # arm not found

    def test_parts_from_results(self, tmp_path):
        assembly = Assembly(
            name="test",
            parts=[Part(name="base", category="x", description="base")],
        )
        parts_results = {
            "base": {"artifacts": ["/tmp/base.FCStd"]},
        }

        verifier = AssemblyVerifier()
        checks = verifier.check_part_completeness(assembly, tmp_path, parts_results)
        assert len(checks) == 1
        assert checks[0].exists


class TestScrewHoles:
    def test_no_issues_when_no_screws(self):
        parts = [Part(name="block", category="x", description="block")]
        assembly = Assembly(name="test", parts=parts)
        verifier = AssemblyVerifier()
        issues = verifier.check_screw_holes(assembly)
        assert len(issues) == 0

    def test_screw_mismatch_detected(self):
        """Part notes mention M3 but hole diameter doesn't match."""
        part = Part(
            name="bracket",
            category="x",
            description="Bracket",
            dimensions={"hole_diameter": 10.0},
            notes="需要 M3 螺钉孔",
        )
        assembly = Assembly(name="test", parts=[part])
        verifier = AssemblyVerifier()
        issues = verifier.check_screw_holes(assembly)
        # 10.0mm is far from M3 clearance (3.4) or tap (2.5)
        assert len(issues) == 1
        assert "不匹配" in issues[0]


class TestFullVerification:
    def test_robotic_arm_assembly(self, tmp_path):
        verifier = AssemblyVerifier()
        result = verifier.verify_assembly(ROBOTIC_ARM_ASSEMBLY, tmp_path)

        assert isinstance(result, AssemblyVerificationResult)
        assert result.assembly_name == "3-DOF Robotic Arm"
        assert len(result.part_checks) == len(ROBOTIC_ARM_ASSEMBLY.parts)
        assert len(result.fit_checks) == len(ROBOTIC_ARM_ASSEMBLY.joints)

    def test_report_generation(self, tmp_path):
        verifier = AssemblyVerifier()
        result = verifier.verify_assembly(ROBOTIC_ARM_ASSEMBLY, tmp_path)
        report = AssemblyVerifier.generate_assembly_report(result)

        assert "装配验证报告" in report
        assert "3-DOF Robotic Arm" in report
        assert "零件检查" in report
        assert "配合检查" in report


# --- Task 63: Collision check tests ---

class TestCollisionCheck:
    """Test collision check integration in AssemblyVerifier (Task 63)."""

    def test_verify_without_placements_no_collision(self, tmp_path):
        """verify_assembly without placements should not run collision checks."""
        verifier = AssemblyVerifier()
        result = verifier.verify_assembly(ROBOTIC_ARM_ASSEMBLY, tmp_path)
        assert result.collision_checks == []
        assert result.collision_free is True

    def test_verify_with_placements_runs_collision(self, tmp_path):
        """verify_assembly with placements should populate collision_checks."""
        from lang3d.tools.assembly_solver import AssemblySolver
        solver = AssemblySolver(ROBOTIC_ARM_ASSEMBLY)
        placements = solver.solve()

        verifier = AssemblyVerifier()
        result = verifier.verify_assembly(
            ROBOTIC_ARM_ASSEMBLY, tmp_path, placements=placements,
        )
        # Should have collision checks populated
        assert isinstance(result.collision_checks, list)
        assert len(result.collision_checks) > 0
        assert isinstance(result.collision_free, bool)
        # Report should contain collision section
        report = AssemblyVerifier.generate_assembly_report(result)
        assert "碰撞检查" in report
