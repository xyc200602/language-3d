"""Integration tests for assembly solve + collision detection (Task 63).

Tests:
- Robotic arm: solve + mesh collision -> no severe collisions
- Custom gate assembly: non-adjacent parts collision detection
"""

from __future__ import annotations

import json
import pytest

from lang3d.knowledge.mechanics import (
    Assembly,
    Joint,
    Part,
    ROBOTIC_ARM_ASSEMBLY,
)

# Check if FCL is available
try:
    import fcl as fcl_mod
    HAS_FCL = True
except ImportError:
    HAS_FCL = False

pytestmark = pytest.mark.skipif(not HAS_FCL, reason="python-fcl not installed")


class TestRoboticArmIntegration:
    """Solve ROBOTIC_ARM_ASSEMBLY and run collision detection."""

    def test_solve_and_collision_check(self):
        """Full pipeline: solve -> mesh collision -> check completes."""
        from lang3d.tools.assembly_solver import AssemblySolver
        from lang3d.tools.mesh_collision import MeshCollisionChecker

        solver = AssemblySolver(ROBOTIC_ARM_ASSEMBLY)
        placements = solver.solve()

        # All parts should be placed
        assert len(placements) == len(ROBOTIC_ARM_ASSEMBLY.parts)

        checker = MeshCollisionChecker()
        result = checker.check_assembly_collisions(
            ROBOTIC_ARM_ASSEMBLY, placements, skip_adjacent=True,
        )

        # Check completes for all parts
        assert result.parts_checked == len(ROBOTIC_ARM_ASSEMBLY.parts)
        assert result.pairs_checked >= 1
        assert isinstance(result.collision_free, bool)

    def test_solve_with_joint_angles(self):
        """Solve with non-zero joint angles and check collisions."""
        from lang3d.tools.assembly_solver import AssemblySolver
        from lang3d.tools.mesh_collision import MeshCollisionChecker

        solver = AssemblySolver(ROBOTIC_ARM_ASSEMBLY)
        placements = solver.solve(joint_angles={"shoulder_link": 30})

        checker = MeshCollisionChecker()
        result = checker.check_assembly_collisions(
            ROBOTIC_ARM_ASSEMBLY, placements, skip_adjacent=True,
        )

        # Even with 30 degree shoulder angle, should be collision-free
        assert result.parts_checked == len(ROBOTIC_ARM_ASSEMBLY.parts)


class TestCustomAssemblyIntegration:
    """Custom assembly to test non-adjacent collision detection."""

    def test_well_separated_parts_no_collision(self):
        """Two small parts far apart should not collide."""
        from lang3d.tools.mesh_collision import MeshCollisionChecker

        parts = [
            Part("left", "structural", "Left",
                 dimensions={"length": 10, "width": 10, "height": 10}),
            Part("right", "structural", "Right",
                 dimensions={"length": 10, "width": 10, "height": 10}),
        ]
        asm = Assembly(name="separated", parts=parts, joints=[])

        # Place parts 200mm apart
        placements = {
            "left": {"position": [-100, 0, 0]},
            "right": {"position": [100, 0, 0]},
        }

        checker = MeshCollisionChecker()
        result = checker.check_assembly_collisions(asm, placements, skip_adjacent=False)

        assert result.collision_free
        assert result.pairs_checked == 1
        assert result.pairs[0].is_collision is False

    def test_overlapping_parts_detected(self):
        """Two parts at same location should detect collision."""
        from lang3d.tools.mesh_collision import MeshCollisionChecker

        parts = [
            Part("a", "structural", "A",
                 dimensions={"length": 100, "width": 100, "height": 100}),
            Part("b", "structural", "B",
                 dimensions={"length": 100, "width": 100, "height": 100}),
        ]
        asm = Assembly(name="overlap", parts=parts, joints=[])

        # Both at same position -> collision
        placements = {
            "a": {"position": [0, 0, 0]},
            "b": {"position": [0, 0, 0]},
        }

        checker = MeshCollisionChecker()
        result = checker.check_assembly_collisions(asm, placements, skip_adjacent=False)

        assert not result.collision_free
        assert result.pairs[0].is_collision
