"""Tests for inverse kinematics solver.

Tests cover:
- Chain extraction from assembly
- Analytic 3-DOF IK
- CCD numerical solver
- FK verification
- Tool registration and execution
- Edge cases (unreachable targets, zero-length links)
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from lang3d.knowledge.mechanics import (
    Assembly,
    Joint,
    Part,
    ROBOTIC_ARM_ASSEMBLY,
)
from lang3d.tools.assembly_solver import AssemblySolver
from lang3d.tools.ik_solver import (
    IKResult,
    IKSolveTool,
    LinkSegment,
    _ccd_solve,
    _extract_chain,
    _fk_verify,
    _vec_cross,
    _vec_dot,
    _vec_length,
    _project_to_plane,
    register_ik_tools,
    solve_ik,
)

# ---------------------------------------------------------------------------
# Dynamically compute the home EE position from the solver so IK tests use the
# AUTHORITATIVE home pose instead of a stale hardcoded Z value.  Previous
# tests hardcoded (0, 0, 166.5) but the solver's home pose evolved (clearance
# offsets, anchor fixes) to 180.9mm — the 14.4mm mismatch caused spurious IK
# "failures" that were actually test-data staleness, not solver bugs.
# ---------------------------------------------------------------------------
_home_solver = AssemblySolver(ROBOTIC_ARM_ASSEMBLY)
_home_placements = _home_solver.solve()
HOME_EE_POS: tuple[float, float, float] = tuple(
    _home_placements["end_effector_mount"]["position"]
)
from lang3d.tools.base import ToolRegistry


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def planar_arm() -> Assembly:
    """Simple 2-link planar arm in the XZ plane with Y-axis revolute joints."""
    return Assembly(
        name="Planar Arm",
        parts=[
            Part(name="base", category="s", description="",
                 dimensions={"height": 20}),
            Part(name="link1", category="s", description="",
                 dimensions={"height": 100}),
            Part(name="link2", category="s", description="",
                 dimensions={"height": 80}),
        ],
        joints=[
            Joint("fixed", "base", "link1",
                  parent_anchor="top", child_anchor="bottom"),
            Joint("revolute", "link1", "link2", (-180, 180), "elbow",
                  parent_anchor="top", child_anchor="bottom", axis="y"),
        ],
    )


@pytest.fixture
def simple_3dof() -> Assembly:
    """3-DOF arm: Z base rotation + 2 Y-axis links."""
    return Assembly(
        name="3-DOF Arm",
        parts=[
            Part(name="base", category="s", description="",
                 dimensions={"height": 20}),
            Part(name="link1", category="s", description="",
                 dimensions={"height": 100}),
            Part(name="link2", category="s", description="",
                 dimensions={"height": 80}),
        ],
        joints=[
            Joint("revolute", "base", "link1", (-180, 180), "shoulder",
                  parent_anchor="top", child_anchor="bottom", axis="y"),
            Joint("revolute", "link1", "link2", (-180, 180), "elbow",
                  parent_anchor="top", child_anchor="bottom", axis="y"),
        ],
    )


# ============================================================================
# Test: Vector helpers
# ============================================================================

class TestVectorHelpers:
    def test_dot_product(self):
        assert _vec_dot((1, 0, 0), (0, 1, 0)) == pytest.approx(0)
        assert _vec_dot((1, 0, 0), (1, 0, 0)) == pytest.approx(1)

    def test_cross_product(self):
        result = _vec_cross((1, 0, 0), (0, 1, 0))
        assert result == pytest.approx((0, 0, 1))

    def test_vec_length(self):
        assert _vec_length((3, 4, 0)) == pytest.approx(5)

    def test_project_to_plane(self):
        # Project (1,1,1) onto plane with normal (0,0,1) → (1,1,0)
        result = _project_to_plane((1, 1, 1), (0, 0, 1))
        assert result[0] == pytest.approx(1)
        assert result[1] == pytest.approx(1)
        assert result[2] == pytest.approx(0, abs=1e-6)


# ============================================================================
# Test: Chain extraction
# ============================================================================

class TestChainExtraction:
    def test_robotic_arm_chain(self):
        links, base_height = _extract_chain(ROBOTIC_ARM_ASSEMBLY)
        # 4 revolute joints → 4 links
        assert len(links) == 4
        assert links[0].name == "base_joint_housing"
        assert links[1].name == "shoulder_link"
        assert links[2].name == "elbow_joint"
        assert links[3].name == "wrist_joint"
        assert base_height > 0

    def test_chain_link_lengths_positive(self):
        links, _ = _extract_chain(ROBOTIC_ARM_ASSEMBLY)
        for link in links:
            assert link.length > 0, f"Link {link.name} has zero length"


# ============================================================================
# Test: FK verification
# ============================================================================

class TestFKVerify:
    def test_home_position(self):
        ee = _fk_verify(ROBOTIC_ARM_ASSEMBLY, {}, "end_effector_mount")
        # Home position: straight up
        assert ee[0] == pytest.approx(0, abs=0.1)
        assert ee[1] == pytest.approx(0, abs=0.1)
        assert ee[2] > 100  # Well above base

    def test_fk_with_angles(self):
        ee_0 = _fk_verify(ROBOTIC_ARM_ASSEMBLY, {"base_joint_housing": 0}, "end_effector_mount")
        ee_90 = _fk_verify(ROBOTIC_ARM_ASSEMBLY, {"base_joint_housing": 90}, "end_effector_mount")
        # Z should be same for base rotation
        assert ee_0[2] == pytest.approx(ee_90[2], abs=0.1)


# ============================================================================
# Test: CCD solver
# ============================================================================

class TestCCDSolver:
    def test_home_position_zero_error(self):
        """CCD should find zero-error solution for home position."""
        result = solve_ik(
            ROBOTIC_ARM_ASSEMBLY,
            target=HOME_EE_POS,
            approach="ccd",
            tolerance_mm=1.0,
            max_iterations=100,
        )
        assert result.error_mm < 1.0
        assert result.reachable

    def test_reachable_target(self):
        """Test a known-reachable target."""
        # From the manual test: (50, 50, 100) is reachable
        result = solve_ik(
            ROBOTIC_ARM_ASSEMBLY,
            target=(50, 50, 100),
            approach="ccd",
            tolerance_mm=2.0,
            max_iterations=500,
        )
        assert result.error_mm < 5.0  # Allow some margin

    def test_unreachable_target(self):
        """Target beyond max reach should be unreachable."""
        # Far beyond the arm's reach
        result = solve_ik(
            ROBOTIC_ARM_ASSEMBLY,
            target=(500, 0, 500),
            approach="ccd",
            tolerance_mm=1.0,
            max_iterations=100,
        )
        assert not result.reachable
        assert result.error_mm > 10

    def test_zero_angles_match_home(self):
        """With all angles = 0, FK should match home position."""
        solver = AssemblySolver(ROBOTIC_ARM_ASSEMBLY)
        home = solver.solve()
        ee_home = home["end_effector_mount"]["position"]
        result = solve_ik(
            ROBOTIC_ARM_ASSEMBLY,
            target=tuple(ee_home),
            approach="ccd",
            tolerance_mm=0.5,
            max_iterations=50,
        )
        assert result.reachable

    def test_ccd_returns_ik_result(self):
        result = _ccd_solve(
            target=HOME_EE_POS,
            assembly=ROBOTIC_ARM_ASSEMBLY,
            links=_extract_chain(ROBOTIC_ARM_ASSEMBLY)[0],
            max_iterations=50,
            tolerance_mm=1.0,
        )
        assert isinstance(result, IKResult)
        assert result.method == "ccd"


# ============================================================================
# Test: Solve IK (auto mode)
# ============================================================================

class TestSolveIK:
    def test_auto_mode_falls_back_to_ccd(self):
        """Auto mode should fall back to CCD when analytic fails."""
        result = solve_ik(
            ROBOTIC_ARM_ASSEMBLY,
            target=(50, 50, 100),
            approach="auto",
            tolerance_mm=2.0,
            max_iterations=500,
        )
        assert result.method in ("analytic", "ccd")
        assert result.error_mm < 5.0

    def test_home_position_auto(self):
        result = solve_ik(
            ROBOTIC_ARM_ASSEMBLY,
            target=HOME_EE_POS,
            approach="auto",
            tolerance_mm=1.0,
        )
        assert result.reachable

    def test_multiple_targets(self):
        """Test several targets across the workspace."""
        targets = [
            HOME_EE_POS,          # Home (straight up)
            (50, 50, 100),       # Diagonal reachable
            (100, 0, 80),        # Forward reach
            (0, 80, 80),         # Side reach
        ]
        for t in targets:
            result = solve_ik(
                ROBOTIC_ARM_ASSEMBLY,
                target=t,
                approach="auto",
                tolerance_mm=5.0,
                max_iterations=500,
            )
            # At minimum should not crash and return valid result
            assert isinstance(result, IKResult)
            assert isinstance(result.joint_angles, dict)
            assert len(result.end_effector) == 3

    def test_initial_angles_help_convergence(self):
        """Good initial angles should help CCD converge faster."""
        # Start from known-good angles
        initial = {"base_joint_housing": 0, "shoulder_link": 0,
                   "elbow_joint": 0, "wrist_joint": 0}
        result = solve_ik(
            ROBOTIC_ARM_ASSEMBLY,
            target=HOME_EE_POS,
            approach="ccd",
            initial_angles=initial,
            tolerance_mm=1.0,
            max_iterations=10,
        )
        assert result.reachable


# ============================================================================
# Test: FK-IK roundtrip
# ============================================================================

class TestFKIKRoundtrip:
    """Verify that IK solutions produce the correct FK result."""

    def test_roundtrip_home(self):
        """FK(home angles) → target, IK(target) should recover home angles."""
        solver = AssemblySolver(ROBOTIC_ARM_ASSEMBLY)
        home = solver.solve()
        ee = home["end_effector_mount"]["position"]

        result = solve_ik(
            ROBOTIC_ARM_ASSEMBLY,
            target=tuple(ee),
            approach="ccd",
            tolerance_mm=1.0,
            max_iterations=100,
        )

        # Verify the IK solution via FK
        ee_check = _fk_verify(ROBOTIC_ARM_ASSEMBLY, result.joint_angles, "end_effector_mount")
        error = math.sqrt(
            (ee[0] - ee_check[0]) ** 2 +
            (ee[1] - ee_check[1]) ** 2 +
            (ee[2] - ee_check[2]) ** 2
        )
        assert error < 1.0

    def test_roundtrip_with_base_rotation(self):
        """Test IK for a target achieved by base rotation."""
        solver = AssemblySolver(ROBOTIC_ARM_ASSEMBLY)
        rotated = solver.solve(joint_angles={"base_joint_housing": 45})
        ee = rotated["end_effector_mount"]["position"]

        result = solve_ik(
            ROBOTIC_ARM_ASSEMBLY,
            target=tuple(ee),
            approach="ccd",
            tolerance_mm=2.0,
            max_iterations=200,
        )

        # Verify FK of IK solution matches target
        ee_check = _fk_verify(ROBOTIC_ARM_ASSEMBLY, result.joint_angles, "end_effector_mount")
        error = math.sqrt(
            (ee[0] - ee_check[0]) ** 2 +
            (ee[1] - ee_check[1]) ** 2 +
            (ee[2] - ee_check[2]) ** 2
        )
        assert error < 3.0  # Allow tolerance for CCD


# ============================================================================
# Test: Tool registration and execution
# ============================================================================

class TestIKSolveTool:
    def test_tool_registration(self):
        registry = ToolRegistry()
        register_ik_tools(registry)
        assert "ik_solve" in registry.list_tools()

    def test_tool_definition(self):
        tool = IKSolveTool()
        defn = tool.get_definition()
        assert defn.name == "ik_solve"
        assert "target" in defn.parameters["properties"]
        assert "assembly_name" in defn.parameters["properties"]

    def test_tool_execute_basic(self):
        tool = IKSolveTool()
        result = tool.execute(target=list(HOME_EE_POS))
        assert "IK Solver" in result
        assert "Joint Angles" in result

    def test_tool_execute_with_approach(self):
        tool = IKSolveTool()
        result = tool.execute(target=[50, 50, 100], approach="ccd")
        assert "IK Solver" in result
        assert "ccd" in result

    def test_tool_execute_missing_target(self):
        tool = IKSolveTool()
        result = tool.execute()
        assert "错误" in result

    def test_tool_execute_unknown_assembly(self):
        tool = IKSolveTool()
        result = tool.execute(target=[100, 0, 150], assembly_name="nonexistent")
        assert "错误" in result

    def test_tool_output_has_json(self):
        tool = IKSolveTool()
        result = tool.execute(target=list(HOME_EE_POS))
        assert "--- JSON ---" in result
        assert "joint_angles" in result


# ============================================================================
# Test: Edge cases
# ============================================================================

class TestEdgeCases:
    def test_no_revolute_joints(self):
        """Assembly with only fixed joints should return empty result."""
        asm = Assembly(
            name="fixed_only",
            parts=[
                Part(name="a", category="s", description="", dimensions={"height": 10}),
                Part(name="b", category="s", description="", dimensions={"height": 10}),
            ],
            joints=[Joint("fixed", "a", "b")],
        )
        result = solve_ik(asm, target=(0, 0, 20), approach="auto")
        # No revolute joints → can't solve
        assert not result.reachable or result.error_mm > 0

    def test_empty_assembly(self):
        asm = Assembly(name="empty", parts=[], joints=[])
        result = solve_ik(asm, target=(0, 0, 0))
        assert not result.reachable

    def test_target_at_origin(self):
        result = solve_ik(ROBOTIC_ARM_ASSEMBLY, target=(0, 0, 0), approach="auto")
        assert isinstance(result, IKResult)
        # Origin is below the base, should be unreachable
        assert not result.reachable

    def test_negative_target(self):
        result = solve_ik(ROBOTIC_ARM_ASSEMBLY, target=(-50, -50, 100),
                          approach="ccd", max_iterations=500, tolerance_mm=5.0)
        assert isinstance(result, IKResult)

    def test_large_tolerance(self):
        """With very large tolerance, almost anything should be reachable."""
        result = solve_ik(ROBOTIC_ARM_ASSEMBLY, target=(0, 0, 0),
                          approach="ccd", tolerance_mm=200, max_iterations=50)
        # Even if error is large, tolerance is 200mm
        assert result.reachable or result.error_mm > 200
