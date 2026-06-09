"""Tests for FreeCAD assembly script rotation support.

Covers:
  - build_assembly_script includes rotate() when rotation is non-trivial
  - No rotate() when rotation is identity [0,0,1,0]
  - rotate() parameter format is correct
  - Exploded view preserves rotation
"""

import pytest

from lang3d.tools.freecad import build_assembly_script


def _make_parts():
    """Standard part definitions used across tests."""
    return [
        {
            "name": "base",
            "shape_type": "box",
            "dimensions": {"length": 100, "width": 80, "height": 10},
            "subsystem": "chassis",
        },
        {
            "name": "wheel_fl",
            "shape_type": "cylinder",
            "dimensions": {"diameter": 65, "height": 26},
            "subsystem": "drive",
        },
        {
            "name": "wheel_fr",
            "shape_type": "cylinder",
            "dimensions": {"diameter": 65, "height": 26},
            "subsystem": "drive",
        },
    ]


class TestAssemblyScriptRotation:
    """Rotation handling in build_assembly_script."""

    def test_includes_rotation_when_nontrivial(self):
        """Non-identity rotation should produce a _shape.rotate() call."""
        positions = {
            "base": {"position": [0, 0, 0], "rotation": [0, 1, 0, 90]},
            "wheel_fl": {"position": [-50, 40, 0], "rotation": [1, 0, 0, 90]},
            "wheel_fr": {"position": [50, 40, 0], "rotation": [1, 0, 0, -90]},
        }
        script = build_assembly_script(
            assembly_parts=_make_parts(),
            positions=positions,
        )
        # Each wheel should have a rotate call
        assert "_shape.rotate(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 90.0000)" in script
        assert "_shape.rotate(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), -90.0000)" in script
        assert "_shape.rotate(FreeCAD.Vector(0,0,0), FreeCAD.Vector(0,1,0), 90.0000)" in script

    def test_no_rotation_when_identity(self):
        """Identity rotation [0,0,1,0] (0 degrees) should NOT produce rotate()."""
        positions = {
            "base": {"position": [0, 0, 0], "rotation": [0, 0, 1, 0]},
            "wheel_fl": {"position": [-50, 40, 0], "rotation": [0, 0, 1, 0]},
        }
        script = build_assembly_script(
            assembly_parts=_make_parts()[:2],
            positions=positions,
        )
        assert "_shape.rotate" not in script

    def test_no_rotation_when_missing(self):
        """Parts without rotation key should not produce rotate() either."""
        positions = {
            "base": {"position": [0, 0, 0]},
            "wheel_fl": {"position": [-50, 40, 0]},
        }
        script = build_assembly_script(
            assembly_parts=_make_parts()[:2],
            positions=positions,
        )
        assert "_shape.rotate" not in script

    def test_rotation_format_correct(self):
        """Verify the rotate() call uses correct FreeCAD API: rotate(center, axis, angle)."""
        positions = {
            "base": {"position": [10, 20, 30], "rotation": [0, 0, 1, 45.5]},
        }
        script = build_assembly_script(
            assembly_parts=_make_parts()[:1],
            positions=positions,
        )
        # Should contain rotate with origin center, axis vector, and angle
        assert "_shape.rotate(FreeCAD.Vector(0,0,0), FreeCAD.Vector(0,0,1), 45.5000)" in script

    def test_translate_still_present(self):
        """translate() must still be emitted after rotate()."""
        positions = {
            "base": {"position": [10, 20, 30], "rotation": [0, 0, 1, 45]},
        }
        script = build_assembly_script(
            assembly_parts=_make_parts()[:1],
            positions=positions,
        )
        assert "_shape.translate(FreeCAD.Vector(10.00, 20.00, 30.00))" in script

    def test_rotate_before_translate(self):
        """rotate() must appear before translate() in the script."""
        positions = {
            "base": {"position": [10, 20, 30], "rotation": [0, 0, 1, 45]},
        }
        script = build_assembly_script(
            assembly_parts=_make_parts()[:1],
            positions=positions,
        )
        rotate_idx = script.index("_shape.rotate")
        translate_idx = script.index("_shape.translate")
        assert rotate_idx < translate_idx, "rotate must come before translate"

    def test_small_angle_no_rotation(self):
        """Very small angles (< 1e-6 degrees) should not produce rotate()."""
        positions = {
            "base": {"position": [0, 0, 0], "rotation": [0, 0, 1, 1e-7]},
        }
        script = build_assembly_script(
            assembly_parts=_make_parts()[:1],
            positions=positions,
        )
        assert "_shape.rotate" not in script


class TestExplodedViewRotation:
    """Exploded view should preserve rotation data."""

    def test_exploded_preserves_rotation(self):
        """Rotation should still be applied in exploded mode."""
        positions = {
            "base": {"position": [0, 0, 0], "rotation": [0, 1, 0, 90]},
            "wheel_fl": {"position": [-50, 40, 0], "rotation": [1, 0, 0, 90]},
        }
        script = build_assembly_script(
            assembly_parts=_make_parts()[:2],
            positions=positions,
            exploded=True,
        )
        assert "_shape.rotate" in script

    def test_exploded_still_translates(self):
        """In exploded mode, translate still happens (with scaled position)."""
        positions = {
            "base": {"position": [0, 0, 0], "rotation": [0, 1, 0, 90]},
        }
        script = build_assembly_script(
            assembly_parts=_make_parts()[:1],
            positions=positions,
            exploded=True,
        )
        assert "_shape.translate" in script
