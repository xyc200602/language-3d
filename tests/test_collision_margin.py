"""Tests for mesh collision margin (box shrink) and cylinder-mesh behaviour.

Verifies that:
  1. Flush-mounted parts (zero-depth face touch) are filtered by the margin.
  2. Real interferences deeper than 2×margin are still detected.
  3. Cylinder parts get cylinder meshes (not boxes), eliminating false
     positives between nearby round parts.
  4. Backward compatibility: margin=0.0 preserves legacy behaviour.
"""

from __future__ import annotations

import pytest

from lang3d.knowledge.mechanics import Assembly, Part
from lang3d.tools.mesh_collision import MeshCollisionChecker


def _box_part(name: str, l: float, w: float, h: float) -> Part:
    return Part(
        name=name, category="structural", description="",
        dimensions={"length": l, "width": w, "height": h},
        material="PLA", notes="",
    )


def _cyl_part(name: str, dia: float, h: float) -> Part:
    return Part(
        name=name, category="actuator", description="",
        dimensions={"diameter": dia, "height": h},
        material="Steel", notes="",
    )


def _two_part_assembly(a: Part, b: Part) -> Assembly:
    return Assembly(
        name="test", description="", parts=[a, b],
        joints=[], default_angles={},
    )


class TestCollisionMargin:
    """Box-shrink margin filters face-touches but preserves real collisions."""

    def test_flush_touch_filtered_with_margin(self):
        """Two boxes touching face-to-face → no collision with margin."""
        a = _box_part("a", 40, 40, 40)
        b = _box_part("b", 40, 40, 40)
        asm = _two_part_assembly(a, b)
        placements = {
            "a": {"position": [0, 0, 0], "rotation": [0, 0, 1, 0]},
            "b": {"position": [40, 0, 0], "rotation": [0, 0, 1, 0]},
        }
        checker = MeshCollisionChecker()
        result = checker.check_assembly_collisions(
            asm, placements, skip_adjacent=False, min_penetration_mm=0.5,
        )
        assert result.collision_free

    def test_flush_touch_detected_without_margin(self):
        """Backward compat: margin=0.0 still detects face touches."""
        a = _box_part("a", 40, 40, 40)
        b = _box_part("b", 40, 40, 40)
        asm = _two_part_assembly(a, b)
        placements = {
            "a": {"position": [0, 0, 0], "rotation": [0, 0, 1, 0]},
            "b": {"position": [40, 0, 0], "rotation": [0, 0, 1, 0]},
        }
        checker = MeshCollisionChecker()
        result = checker.check_assembly_collisions(
            asm, placements, skip_adjacent=False, min_penetration_mm=0.0,
        )
        assert not result.collision_free

    def test_real_overlap_detected_with_margin(self):
        """10mm overlap → still detected even with margin."""
        a = _box_part("a", 40, 40, 40)
        b = _box_part("b", 40, 40, 40)
        asm = _two_part_assembly(a, b)
        placements = {
            "a": {"position": [0, 0, 0], "rotation": [0, 0, 1, 0]},
            "b": {"position": [30, 0, 0], "rotation": [0, 0, 1, 0]},
        }
        checker = MeshCollisionChecker()
        result = checker.check_assembly_collisions(
            asm, placements, skip_adjacent=False, min_penetration_mm=0.5,
        )
        assert not result.collision_free

    def test_small_gap_no_collision(self):
        """1mm gap → no collision regardless of margin."""
        a = _box_part("a", 40, 40, 40)
        b = _box_part("b", 40, 40, 40)
        asm = _two_part_assembly(a, b)
        placements = {
            "a": {"position": [0, 0, 0], "rotation": [0, 0, 1, 0]},
            "b": {"position": [41, 0, 0], "rotation": [0, 0, 1, 0]},
        }
        checker = MeshCollisionChecker()
        result = checker.check_assembly_collisions(
            asm, placements, skip_adjacent=False, min_penetration_mm=0.5,
        )
        assert result.collision_free


class TestCylinderMesh:
    """Cylindrical parts use cylinder meshes, reducing false positives."""

    def test_nearby_cylinders_no_false_positive(self):
        """Two cylinders 3mm apart (edge-to-edge) should not collide
        with a small margin.  As boxes they would overlap ~5mm."""
        a = _cyl_part("motor", 40, 35)
        b = _cyl_part("servo", 36, 30)
        asm = _two_part_assembly(a, b)
        # Centres 41mm apart; radii 20+18=38; gap = 3mm.
        placements = {
            "motor": {"position": [0, 0, 0], "rotation": [0, 0, 1, 0]},
            "servo": {"position": [41, 0, 0], "rotation": [0, 0, 1, 0]},
        }
        checker = MeshCollisionChecker()
        # With box meshes + margin 0, these would falsely collide.
        # With cylinder meshes + margin 2, they should be clear.
        result = checker.check_assembly_collisions(
            asm, placements, skip_adjacent=False, min_penetration_mm=2.0,
        )
        assert result.collision_free

    def test_overlapping_cylinders_detected(self):
        """Two cylinders actually overlapping → collision detected."""
        a = _cyl_part("motor", 40, 35)
        b = _cyl_part("servo", 36, 30)
        asm = _two_part_assembly(a, b)
        # Centres 25mm apart; radii sum 38; overlap = 13mm.
        placements = {
            "motor": {"position": [0, 0, 0], "rotation": [0, 0, 1, 0]},
            "servo": {"position": [25, 0, 0], "rotation": [0, 0, 1, 0]},
        }
        checker = MeshCollisionChecker()
        result = checker.check_assembly_collisions(
            asm, placements, skip_adjacent=False, min_penetration_mm=2.0,
        )
        assert not result.collision_free
