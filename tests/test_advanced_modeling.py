"""Tests for advanced FreeCAD modeling operations (Task 52).

Covers: sweep, loft, polar_pattern, linear_pattern, mirror, shell, draft.
Tests script generation (unit) and FreeCAD execution (integration).
"""

import json
import math
import os
import tempfile
from pathlib import Path

import pytest

from lang3d.tools.freecad import (
    FCBatchTool,
    _build_script,
    _find_freecad_python,
)


# Skip integration tests if FreeCAD is not available
freecad_available = _find_freecad_python() is not None
requires_freecad = pytest.mark.skipif(
    not freecad_available,
    reason="FreeCAD not installed",
)


# ── Script Generation Tests ──────────────────────────────────────


class TestSweepScriptGeneration:
    def test_helix_sweep_generates_script(self):
        script = _build_script([
            {"type": "new_doc"},
            {"type": "sweep", "name": "Spring", "profile": "circle",
             "profile_radius": 1.0, "path_type": "helix",
             "pitch": 3.0, "height": 15.0, "helix_radius": 8.0},
        ])
        assert "makeHelix" in script
        assert "makePipeShell" in script
        assert '"Spring"' in script

    def test_circle_path_sweep(self):
        script = _build_script([
            {"type": "new_doc"},
            {"type": "sweep", "profile": "circle", "profile_radius": 2.0,
             "path_type": "circle", "path_radius": 10.0},
        ])
        assert "makeCircle(10.0)" in script
        assert "makePipeShell" in script

    def test_rectangle_profile_sweep(self):
        script = _build_script([
            {"type": "new_doc"},
            {"type": "sweep", "profile": "rectangle",
             "profile_width": 4.0, "profile_height": 3.0,
             "path_type": "line", "line_length": 20.0},
        ])
        assert "makePolygon" in script
        assert "makePipeShell" in script

    def test_sweep_with_turns(self):
        script = _build_script([
            {"type": "new_doc"},
            {"type": "sweep", "path_type": "helix", "turns": 5,
             "pitch": 2.0, "height": 10.0, "helix_radius": 5.0},
        ])
        assert "makeHelix(2.0, 10.0, 5.0, 0, 5)" in script


class TestLoftScriptGeneration:
    def test_simple_two_circle_loft(self):
        script = _build_script([
            {"type": "new_doc"},
            {"type": "loft", "radius1": 10.0, "radius2": 5.0, "height": 20.0},
        ])
        assert "makeLoft" in script
        assert "makeCircle(10.0" in script
        assert "makeCircle(5.0" in script

    def test_multi_profile_loft(self):
        script = _build_script([
            {"type": "new_doc"},
            {"type": "loft", "profiles": [
                {"type": "circle", "radius": 10, "center": [0, 0, 0]},
                {"type": "circle", "radius": 8, "center": [0, 0, 10]},
                {"type": "circle", "radius": 3, "center": [0, 0, 20]},
            ]},
        ])
        assert "makeLoft" in script
        assert "_loft_p0" in script
        assert "_loft_p1" in script
        assert "_loft_p2" in script

    def test_rectangle_to_circle_loft(self):
        script = _build_script([
            {"type": "new_doc"},
            {"type": "loft", "profiles": [
                {"type": "rectangle", "width": 20, "height": 20, "center": [0, 0, 0]},
                {"type": "circle", "radius": 5, "center": [0, 0, 15]},
            ]},
        ])
        assert "makePolygon" in script
        assert "makeCircle" in script


class TestPatternScriptGeneration:
    def test_polar_pattern_generates_loop(self):
        script = _build_script([
            {"type": "new_doc"},
            {"type": "make_cylinder", "radius": 3, "height": 10, "name": "Peg"},
            {"type": "polar_pattern", "object": "Peg", "count": 6, "angle": 360},
        ])
        assert "range(6)" in script
        assert "360" in script and "_i / 6" in script
        assert "fuse" in script

    def test_linear_pattern_generates_loop(self):
        script = _build_script([
            {"type": "new_doc"},
            {"type": "make_box", "length": 2, "width": 2, "height": 10, "name": "Fin"},
            {"type": "linear_pattern", "object": "Fin", "count": 5, "spacing": 4,
             "direction": [1, 0, 0]},
        ])
        assert "range(5)" in script
        assert "4" in script
        assert "fuse" in script


class TestMirrorScriptGeneration:
    def test_mirror_yz_plane(self):
        script = _build_script([
            {"type": "new_doc"},
            {"type": "make_box", "length": 10, "width": 20, "height": 5, "name": "Arm"},
            {"type": "mirror", "object": "Arm", "plane": "YZ"},
        ])
        assert "mirror" in script
        assert "Vector(1,0,0)" in script

    def test_mirror_xz_plane(self):
        script = _build_script([
            {"type": "new_doc"},
            {"type": "make_box", "length": 10, "width": 20, "height": 5, "name": "Arm"},
            {"type": "mirror", "object": "Arm", "plane": "XZ"},
        ])
        assert "Vector(0,1,0)" in script


class TestShellScriptGeneration:
    def test_shell_with_face_indices(self):
        script = _build_script([
            {"type": "new_doc"},
            {"type": "make_box", "length": 20, "width": 20, "height": 20, "name": "Box"},
            {"type": "shell", "object": "Box", "thickness": 2.0,
             "faces_to_remove": [0, 1]},
        ])
        assert "makeThickness" in script
        assert "2.0" in script

    def test_shell_default_face(self):
        script = _build_script([
            {"type": "new_doc"},
            {"type": "make_box", "length": 20, "width": 20, "height": 20, "name": "Box"},
            {"type": "shell", "object": "Box", "thickness": 1.5},
        ])
        assert "makeThickness" in script
        assert "Faces[-1]" in script


class TestDraftScriptGeneration:
    def test_draft_generates_script(self):
        script = _build_script([
            {"type": "new_doc"},
            {"type": "make_box", "length": 20, "width": 20, "height": 30, "name": "Block"},
            {"type": "draft", "object": "Block", "angle": 3.0, "faces": [0, 1, 2, 3]},
        ])
        assert "makeDraftShape" in script
        assert "3.0" in script


# ── Integration Tests (require FreeCAD) ──────────────────────────


@requires_freecad
class TestSweepIntegration:
    def test_helix_spring(self):
        """Create a helical spring and verify it has volume."""
        tool = FCBatchTool()
        result = tool.execute(operations=[
            {"type": "new_doc", "name": "Spring"},
            {"type": "sweep", "name": "Spring", "profile": "circle",
             "profile_radius": 1.0, "path_type": "helix",
             "pitch": 3.0, "height": 15.0, "helix_radius": 8.0},
            {"type": "object_info", "object": "Spring"},
        ])
        assert "Error" not in result
        assert "Volume:" in result

    def test_line_sweep_pipe(self):
        """Create a pipe by sweeping a circle along a line."""
        tool = FCBatchTool()
        result = tool.execute(operations=[
            {"type": "new_doc", "name": "Pipe"},
            {"type": "sweep", "name": "Pipe", "profile": "circle",
             "profile_radius": 3.0, "path_type": "line",
             "line_length": 30.0, "solid": False},
            {"type": "object_info", "object": "Pipe"},
        ])
        assert "Error" not in result
        assert "Volume:" in result


@requires_freecad
class TestLoftIntegration:
    def test_circle_to_circle_loft(self):
        """Create a cone-like shape by lofting between two circles."""
        tool = FCBatchTool()
        result = tool.execute(operations=[
            {"type": "new_doc", "name": "Nozzle"},
            {"type": "loft", "name": "Nozzle", "radius1": 10.0,
             "radius2": 4.0, "height": 20.0},
            {"type": "object_info", "object": "Nozzle"},
        ])
        assert "Error" not in result
        assert "Volume:" in result


@requires_freecad
class TestPolarPatternIntegration:
    def test_6_hole_circle(self):
        """Create a bolt hole circle with 6 holes."""
        tool = FCBatchTool()
        result = tool.execute(operations=[
            {"type": "new_doc", "name": "Flange"},
            {"type": "make_cylinder", "radius": 20, "height": 5, "name": "Disc"},
            {"type": "make_cylinder", "radius": 2, "height": 5, "name": "Hole"},
            {"type": "move", "object": "Hole", "dx": 14, "dy": 0, "dz": 0},
            {"type": "polar_pattern", "object": "Hole", "count": 6,
             "angle": 360, "result_name": "Holes"},
            {"type": "boolean", "operation": "cut", "object1": "Disc",
             "object2": "Holes", "result_name": "Flange"},
            {"type": "object_info", "object": "Flange"},
        ])
        assert "Error" not in result
        assert "Volume:" in result


@requires_freecad
class TestLinearPatternIntegration:
    def test_heat_sink_fins(self):
        """Create a heat sink with linear array of fins."""
        tool = FCBatchTool()
        result = tool.execute(operations=[
            {"type": "new_doc", "name": "HeatSink"},
            {"type": "make_box", "length": 30, "width": 20, "height": 3, "name": "Base"},
            {"type": "make_box", "length": 2, "width": 20, "height": 10, "name": "Fin"},
            {"type": "linear_pattern", "object": "Fin", "count": 6,
             "spacing": 5.0, "direction": [1, 0, 0], "result_name": "Fins"},
            {"type": "boolean", "operation": "union", "object1": "Base",
             "object2": "Fins", "result_name": "HeatSink"},
            {"type": "object_info", "object": "HeatSink"},
        ])
        assert "Error" not in result
        assert "Volume:" in result


@requires_freecad
class TestMirrorIntegration:
    def test_mirror_arm(self):
        """Create a mirror of a box to form symmetric arms."""
        tool = FCBatchTool()
        result = tool.execute(operations=[
            {"type": "new_doc", "name": "Arms"},
            {"type": "make_box", "length": 40, "width": 10, "height": 5, "name": "LeftArm"},
            {"type": "move", "object": "LeftArm", "dx": 20, "dy": 0, "dz": 0},
            {"type": "mirror", "object": "LeftArm", "plane": "YZ",
             "result_name": "RightArm"},
            {"type": "object_info", "object": "RightArm"},
        ])
        assert "Error" not in result
        assert "Volume:" in result


@requires_freecad
class TestShellIntegration:
    def test_hollow_box(self):
        """Create a hollow box by removing the top face."""
        tool = FCBatchTool()
        result = tool.execute(operations=[
            {"type": "new_doc", "name": "Box"},
            {"type": "make_box", "length": 30, "width": 20, "height": 15, "name": "SolidBox"},
            {"type": "shell", "object": "SolidBox", "thickness": 2.0,
             "faces_to_remove": [4], "result_name": "HollowBox"},
            {"type": "object_info", "object": "HollowBox"},
        ])
        assert "Error" not in result
        assert "Volume:" in result


@requires_freecad
class TestDraftIntegration:
    def test_draft_on_box(self):
        """Apply draft angle to a box."""
        tool = FCBatchTool()
        result = tool.execute(operations=[
            {"type": "new_doc", "name": "Draft"},
            {"type": "make_box", "length": 20, "width": 20, "height": 30, "name": "Block"},
            {"type": "draft", "object": "Block", "angle": 2.0,
             "faces": [0, 1, 2, 3], "result_name": "Drafted"},
            {"type": "object_info", "object": "Drafted"},
        ])
        assert "Error" not in result


@requires_freecad
class TestExportAdvanced:
    def test_spring_save_and_export(self):
        """Build a spring, save FCStd, export STL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fcstd = str(Path(tmpdir) / "spring.FCStd")
            stl = str(Path(tmpdir) / "spring.stl")

            tool = FCBatchTool()
            result = tool.execute(operations=[
                {"type": "new_doc", "name": "Spring"},
                {"type": "sweep", "name": "Spring", "profile": "circle",
                 "profile_radius": 0.8, "path_type": "helix",
                 "pitch": 2.5, "height": 12.0, "helix_radius": 6.0},
                {"type": "save", "path": fcstd},
                {"type": "export_stl", "path": stl},
            ])
            assert "Error" not in result
            assert Path(fcstd).exists()
            assert Path(stl).exists()
            assert Path(stl).stat().st_size > 0


# ── Registration Test ────────────────────────────────────────────


class TestAdvancedModelingRegistration:
    def test_batch_description_includes_advanced_ops(self):
        tool = FCBatchTool()
        desc = tool.description
        assert "sweep" in desc
        assert "loft" in desc
        assert "polar_pattern" in desc
        assert "linear_pattern" in desc
        assert "mirror" in desc
        assert "shell" in desc
        assert "draft" in desc

    def test_timeout_is_300(self):
        """Verify FreeCAD subprocess timeout is 300s."""
        from lang3d.tools.freecad import _run_freecad_script
        import inspect
        sig = inspect.signature(_run_freecad_script)
        assert sig.parameters["timeout"].default == 300
