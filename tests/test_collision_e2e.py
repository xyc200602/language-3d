"""E2E collision detection test — 41-part complex robot with FCL.

Tests that the mesh collision detection pipeline works end-to-end on the
full 41-part complex robot assembly:
  build assembly → solve positions → FCL collision check → verify results.

Also tests that intentionally injected collisions are detected correctly.

Requires:
  - python-fcl and trimesh installed
  - No API key needed (pure geometry, no VLM)

Usage:
  pytest tests/test_collision_e2e.py -v
  python tests/test_collision_e2e.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------

def _has_fcl() -> bool:
    try:
        import fcl  # noqa: F401
        import trimesh  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def complex_robot():
    """Build the 41-part complex robot assembly."""
    from lang3d.tools.export_package import build_complex_robot
    return build_complex_robot()


@pytest.fixture
def solved_positions(complex_robot):
    """Solve assembly positions for the complex robot."""
    from lang3d.tools.assembly_solver import AssemblySolver
    solver = AssemblySolver(complex_robot)
    return solver.solve()


@pytest.fixture
def checker():
    """Create a MeshCollisionChecker instance."""
    from lang3d.tools.mesh_collision import MeshCollisionChecker
    return MeshCollisionChecker()


# ---------------------------------------------------------------------------
# E2E Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_fcl(), reason="python-fcl + trimesh not installed")
class TestCollisionE2E:
    """E2E collision detection tests with the full 41-part complex robot."""

    def test_solver_produces_all_positions(self, complex_robot):
        """Solver should produce positions for all 41 parts."""
        from lang3d.tools.assembly_solver import AssemblySolver
        solver = AssemblySolver(complex_robot)
        positions = solver.solve()
        assert len(positions) >= 40
        for name, pos in positions.items():
            assert "position" in pos, f"Missing position for {name}"
            assert len(pos["position"]) == 3

    def test_collision_check_runs_on_full_robot(
        self, complex_robot, solved_positions, checker,
    ):
        """FCL collision check should run successfully on 41-part assembly."""
        result = checker.check_assembly_collisions(
            complex_robot, solved_positions, skip_adjacent=True,
        )
        assert result.parts_checked >= 40
        assert result.pairs_checked >= 100, (
            f"Expected >= 100 pairs, got {result.pairs_checked}"
        )
        # Result should have a non-empty summary
        assert "parts" in result.summary.lower()
        assert "pairs" in result.summary.lower()

    def test_no_severe_collisions_in_standard_assembly(
        self, complex_robot, solved_positions, checker,
    ):
        """Standard solved assembly should have no severe collisions.

        The complex robot is a well-designed assembly. Adjacent parts
        naturally touch at joints, but non-adjacent parts should not collide
        with deep penetration.

        A "severe" collision is defined as > 5mm penetration depth.
        Minor touching (< 1mm) between nearby parts is acceptable.
        """
        result = checker.check_assembly_collisions(
            complex_robot, solved_positions, skip_adjacent=True,
        )

        severe_collisions = [
            cp for cp in result.pairs
            if cp.is_collision and cp.penetration_depth_mm > 5.0
        ]

        # Save detailed report
        report = {
            "total_parts": result.parts_checked,
            "total_pairs": result.pairs_checked,
            "collision_free": result.collision_free,
            "total_collisions": sum(1 for cp in result.pairs if cp.is_collision),
            "severe_collisions": len(severe_collisions),
            "collision_details": [
                {
                    "part_a": cp.part_a,
                    "part_b": cp.part_b,
                    "penetration_mm": cp.penetration_depth_mm,
                    "contact_points": cp.contact_points,
                }
                for cp in result.pairs
                if cp.is_collision
            ],
        }
        report_path = Path(__file__).parent.parent / "data" / "collision_e2e_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

        assert len(severe_collisions) == 0, (
            f"Found {len(severe_collisions)} severe collisions (>5mm penetration):\n"
            + "\n".join(
                f"  {cp.part_a} <-> {cp.part_b}: "
                f"{cp.penetration_depth_mm:.2f}mm penetration"
                for cp in severe_collisions
            )
        )

    def test_injected_collision_detected(
        self, complex_robot, solved_positions, checker,
    ):
        """Intentionally placing two parts at the same position must be detected."""
        # Override positions to force collision
        modified_positions = dict(solved_positions)
        # Move wheel_fl and motor_fr to the same position as base_plate
        base_pos = modified_positions.get("base_plate", {}).get("position", [0, 0, 0])
        modified_positions["wheel_fl"] = {"position": list(base_pos)}
        modified_positions["motor_fr"] = {"position": list(base_pos)}

        result = checker.check_assembly_collisions(
            complex_robot, modified_positions, skip_adjacent=True,
        )

        assert not result.collision_free, (
            "Expected collision detection to find injected overlaps"
        )
        # Should detect collision involving at least one of the moved parts
        collision_parts = set()
        for cp in result.pairs:
            if cp.is_collision:
                collision_parts.add(cp.part_a)
                collision_parts.add(cp.part_b)
        assert "wheel_fl" in collision_parts or "motor_fr" in collision_parts, (
            f"Expected wheel_fl or motor_fr in collision set, got: {collision_parts}"
        )

    def test_no_collision_for_well_separated_parts(self, checker):
        """Simple assembly with well-separated parts should be collision-free."""
        from lang3d.knowledge.mechanics import Assembly, Joint, Part

        assembly = Assembly(
            name="separated_test",
            parts=[
                Part("a", "box", "", dimensions={"length": 10, "width": 10, "height": 10}),
                Part("b", "box", "", dimensions={"length": 10, "width": 10, "height": 10}),
                Part("c", "box", "", dimensions={"length": 10, "width": 10, "height": 10}),
            ],
            joints=[
                Joint("fixed", "a", "b"),
                Joint("fixed", "b", "c"),
            ],
        )
        positions = {
            "a": {"position": [0, 0, 0]},
            "b": {"position": [100, 0, 0]},
            "c": {"position": [200, 0, 0]},
        }
        result = checker.check_assembly_collisions(
            assembly, positions, skip_adjacent=True,
        )
        assert result.collision_free
        # a-b and b-c are adjacent (skipped), but a-c is not adjacent (1 pair)
        assert result.pairs_checked == 1

    def test_collision_check_without_skip_adjacent(
        self, complex_robot, solved_positions, checker,
    ):
        """Without skip_adjacent, adjacent parts at same anchor will show collision."""
        result = checker.check_assembly_collisions(
            complex_robot, solved_positions, skip_adjacent=False,
        )
        # Without skip_adjacent, there will be more pairs checked
        assert result.pairs_checked > 0
        # Adjacent parts touching at joints is expected — not a defect

    def test_mesh_collision_tool_execution(self, complex_robot):
        """MeshCollisionTool.execute() should work with assembly JSON input."""
        from lang3d.tools.mesh_collision import MeshCollisionTool
        tool = MeshCollisionTool()
        # Build JSON from the assembly since "complex_robot" isn't a registered name
        asm_json = json.dumps({
            "name": complex_robot.name,
            "parts": [
                {"name": p.name, "category": p.category, "dimensions": p.dimensions}
                for p in complex_robot.parts
            ],
            "joints": [
                {"type": j.type, "parent": j.parent, "child": j.child}
                for j in complex_robot.joints
            ],
        })
        output = tool.execute(assembly_json=asm_json)
        assert "Parts:" in output
        assert "Pairs checked:" in output
        assert "Collision-free:" in output

    def test_assembly_verifier_collision_integration(
        self, complex_robot, solved_positions,
    ):
        """AssemblyVerifier should integrate collision detection when placements provided."""
        from lang3d.agent.assembly_verifier import AssemblyVerifier

        verifier = AssemblyVerifier()
        result = verifier.verify_assembly(
            complex_robot,
            workspace=".",  # dummy workspace
            placements=solved_positions,
        )
        # Result should have collision info
        assert hasattr(result, "collision_checks")
        assert hasattr(result, "collision_free")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main():
    """Run collision E2E tests standalone."""
    if not _has_fcl():
        print("ERROR: python-fcl + trimesh required. Install with: pip install python-fcl trimesh")
        sys.exit(1)

    from lang3d.tools.export_package import build_complex_robot
    from lang3d.tools.assembly_solver import AssemblySolver
    from lang3d.tools.mesh_collision import MeshCollisionChecker

    print("=" * 70)
    print("Collision Detection E2E Test — 41-part Complex Robot")
    print("=" * 70)

    # Build and solve
    print("\n[1/3] Building complex robot assembly...")
    assembly = build_complex_robot()
    print(f"  Parts: {len(assembly.parts)}")
    print(f"  Joints: {len(assembly.joints)}")

    print("\n[2/3] Solving assembly positions...")
    solver = AssemblySolver(assembly)
    positions = solver.solve()
    print(f"  Positions: {len(positions)}")

    # Run collision check
    print("\n[3/3] Running FCL collision detection...")
    checker = MeshCollisionChecker()
    start = time.time()
    result = checker.check_assembly_collisions(assembly, positions, skip_adjacent=True)
    elapsed = time.time() - start

    # Report
    print(f"\n  Parts checked:    {result.parts_checked}")
    print(f"  Pairs checked:    {result.pairs_checked}")
    print(f"  Collision-free:   {result.collision_free}")
    print(f"  Total collisions: {sum(1 for cp in result.pairs if cp.is_collision)}")
    print(f"  Time:             {elapsed:.2f}s")

    if not result.collision_free:
        print("\n  Collision details:")
        for cp in result.pairs:
            if cp.is_collision:
                print(
                    f"    {cp.part_a} <-> {cp.part_b}: "
                    f"{cp.penetration_depth_mm:.2f}mm "
                    f"({cp.contact_points} contacts)"
                )

    # Save report
    report = {
        "total_parts": result.parts_checked,
        "total_pairs": result.pairs_checked,
        "collision_free": result.collision_free,
        "total_collisions": sum(1 for cp in result.pairs if cp.is_collision),
        "collision_details": [
            {
                "part_a": cp.part_a,
                "part_b": cp.part_b,
                "penetration_mm": cp.penetration_depth_mm,
                "contact_points": cp.contact_points,
            }
            for cp in result.pairs
            if cp.is_collision
        ],
        "elapsed_seconds": round(elapsed, 2),
    }
    report_path = Path(__file__).parent.parent / "data" / "collision_e2e_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n  Report: {report_path}")

    # Check for severe collisions
    severe = [cp for cp in result.pairs if cp.penetration_depth_mm > 5.0]
    if severe:
        print(f"\nFAIL: {len(severe)} severe collisions (>5mm)")
        sys.exit(1)
    else:
        print("\nPASS: No severe collisions detected")
        sys.exit(0)


if __name__ == "__main__":
    main()
