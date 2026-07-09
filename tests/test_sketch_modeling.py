"""Tests for sketch-based modeling operations (Task 56).

Covers:
  - create_sketch: elements (point, line, circle, arc, rectangle, polygon)
  - extrude_sketch: basic, midplane, reverse, direction
  - revolve_sketch: basic, partial angle, axis selection
  - pocket: basic, through-all, with target
  - Script generation correctness (no placeholder leaks)
  - Tool registration (operations available via fc_batch)
"""

import pytest

from lang3d.tools.freecad import _build_script


# ============================================================================
# create_sketch
# ============================================================================


class TestCreateSketch:
    def test_basic_sketch_script(self):
        ops = [{"type": "new_doc"}, {"type": "create_sketch", "name": "MySketch"}]
        script = _build_script(ops)
        assert "Sketcher" in script
        assert "MySketch" in script
        assert "Sketcher::SketchObject" in script

    def test_default_plane_xy(self):
        ops = [{"type": "new_doc"}, {"type": "create_sketch"}]
        script = _build_script(ops)
        assert "XY_Plane" in script

    def test_plane_xz(self):
        ops = [{"type": "new_doc"}, {"type": "create_sketch", "plane": "XZ"}]
        script = _build_script(ops)
        assert "XZ_Plane" in script

    def test_plane_yz(self):
        ops = [{"type": "new_doc"}, {"type": "create_sketch", "plane": "YZ"}]
        script = _build_script(ops)
        assert "YZ_Plane" in script

    def test_with_offset(self):
        ops = [{"type": "new_doc"}, {"type": "create_sketch", "offset": 10.0}]
        script = _build_script(ops)
        assert "AttachmentOffset" in script
        assert "10" in script

    def test_line_element(self):
        ops = [{"type": "new_doc"}, {"type": "create_sketch",
                "elements": [{"type": "line", "x1": 0, "y1": 0, "x2": 10, "y2": 20}]}]
        script = _build_script(ops)
        assert "LineSegment" in script
        assert "10" in script
        assert "20" in script

    def test_circle_element(self):
        ops = [{"type": "new_doc"}, {"type": "create_sketch",
                "elements": [{"type": "circle", "cx": 5, "cy": 5, "radius": 10}]}]
        script = _build_script(ops)
        assert "Circle" in script
        assert "radius" in script or "10" in script

    def test_arc_element(self):
        ops = [{"type": "new_doc"}, {"type": "create_sketch",
                "elements": [{"type": "arc", "cx": 0, "cy": 0, "radius": 15,
                              "start_angle": 0, "end_angle": 180}]}]
        script = _build_script(ops)
        assert "ArcOfCircle" in script
        assert "180" in script

    def test_rectangle_element(self):
        ops = [{"type": "new_doc"}, {"type": "create_sketch",
                "elements": [{"type": "rectangle", "x": 0, "y": 0,
                              "width": 20, "height": 10}]}]
        script = _build_script(ops)
        assert "LineSegment" in script
        assert "20" in script
        assert "10" in script
        # Rectangle = 4 line segment edges. Count addGeometry calls
        # (not bare "LineSegment" which also appears in a safety comment).
        assert script.count("addGeometry(Part.LineSegment") == 4

    def test_polygon_element(self):
        ops = [{"type": "new_doc"}, {"type": "create_sketch",
                "elements": [{"type": "polygon", "cx": 0, "cy": 0,
                              "radius": 20, "sides": 6}]}]
        script = _build_script(ops)
        assert "math" in script
        assert "6" in script

    def test_point_element(self):
        ops = [{"type": "new_doc"}, {"type": "create_sketch",
                "elements": [{"type": "point", "x": 5, "y": 10}]}]
        script = _build_script(ops)
        assert "Point" in script

    def test_multiple_elements(self):
        ops = [{"type": "new_doc"}, {"type": "create_sketch",
                "elements": [
                    {"type": "circle", "cx": 0, "cy": 0, "radius": 10},
                    {"type": "rectangle", "x": -5, "y": -5, "width": 10, "height": 10},
                ]}]
        script = _build_script(ops)
        assert "Circle" in script
        assert "LineSegment" in script


# ============================================================================
# extrude_sketch
# ============================================================================


class TestExtrudeSketch:
    def test_basic_extrude(self):
        ops = [
            {"type": "new_doc"},
            {"type": "create_sketch", "name": "S1"},
            {"type": "extrude_sketch", "sketch": "S1", "height": 20, "name": "Ext1"},
        ]
        script = _build_script(ops)
        assert "extrude" in script
        assert "S1" in script
        assert "Ext1" in script
        assert "20" in script

    def test_direction_x(self):
        ops = [{"type": "extrude_sketch", "sketch": "S", "height": 10, "direction": "x"}]
        script = _build_script(ops)
        assert "1, 0, 0" in script

    def test_midplane(self):
        ops = [{"type": "extrude_sketch", "sketch": "S", "height": 20,
                "midplane": True, "name": "Mid"}]
        script = _build_script(ops)
        assert "fuse" in script  # midplane fuses two halves

    def test_reverse(self):
        ops = [{"type": "extrude_sketch", "sketch": "S", "height": 10, "reverse": True}]
        script = _build_script(ops)
        assert "0, 0, -1" in script  # reversed z direction


# ============================================================================
# revolve_sketch
# ============================================================================


class TestRevolveSketch:
    def test_basic_revolve(self):
        ops = [
            {"type": "new_doc"},
            {"type": "create_sketch", "name": "Prof"},
            {"type": "revolve_sketch", "sketch": "Prof", "axis": "z",
             "angle": 360, "name": "Shaft"},
        ]
        script = _build_script(ops)
        assert "revolve" in script
        assert "Prof" in script
        assert "Shaft" in script
        assert "360" in script

    def test_partial_angle(self):
        ops = [{"type": "revolve_sketch", "sketch": "S", "angle": 180}]
        script = _build_script(ops)
        assert "180" in script

    def test_axis_y(self):
        ops = [{"type": "revolve_sketch", "sketch": "S", "axis": "y"}]
        script = _build_script(ops)
        assert "0, 1, 0" in script

    def test_custom_base(self):
        ops = [{"type": "revolve_sketch", "sketch": "S",
                "base": [10, 20, 30]}]
        script = _build_script(ops)
        assert "10" in script
        assert "20" in script
        assert "30" in script


# ============================================================================
# pocket
# ============================================================================


class TestPocket:
    def test_basic_pocket(self):
        ops = [
            {"type": "new_doc"},
            {"type": "make_box", "length": 50, "width": 50, "height": 20, "name": "Block"},
            {"type": "create_sketch", "name": "Hole",
             "elements": [{"type": "circle", "cx": 25, "cy": 25, "radius": 5}]},
            {"type": "pocket", "sketch": "Hole", "target": "Block",
             "depth": 10, "name": "Pocketed"},
        ]
        script = _build_script(ops)
        assert "cut" in script
        assert "Hole" in script
        assert "Block" in script
        assert "10" in script

    def test_through_all(self):
        ops = [
            {"type": "new_doc"},
            {"type": "make_box", "length": 30, "width": 30, "height": 10, "name": "Plate"},
            {"type": "create_sketch", "name": "Slot"},
            {"type": "pocket", "sketch": "Slot", "target": "Plate",
             "through_all": True},
        ]
        script = _build_script(ops)
        assert "through_all" in script or "10000" in script

    def test_no_target_uses_last_solid(self):
        ops = [{"type": "pocket", "sketch": "S", "depth": 5}]
        script = _build_script(ops)
        assert "_solids" in script  # fallback to last solid

    def test_reverse_direction(self):
        ops = [{"type": "pocket", "sketch": "S", "depth": 5, "reverse": True}]
        script = _build_script(ops)
        assert "0, 0, -1" in script


# ============================================================================
# Script quality / no placeholder leaks
# ============================================================================


class TestScriptQuality:
    def test_no_unreplaced_placeholders(self):
        ops = [
            {"type": "new_doc"},
            {"type": "create_sketch", "name": "Test",
             "elements": [
                 {"type": "circle", "cx": 0, "cy": 0, "radius": 10},
                 {"type": "rectangle", "x": -3, "y": -3, "width": 6, "height": 6},
             ]},
            {"type": "extrude_sketch", "sketch": "Test", "height": 5},
        ]
        script = _build_script(ops)
        # Should not contain any {...} format placeholders
        assert "{" not in script or "f\"" in script or "f'" in script

    def test_combined_workflow(self):
        """Full sketch workflow: sketch → extrude → pocket."""
        ops = [
            {"type": "new_doc", "name": "Bracket"},
            {"type": "create_sketch", "name": "BracketProfile",
             "elements": [
                 {"type": "rectangle", "x": 0, "y": 0, "width": 40, "height": 30},
                 {"type": "circle", "cx": 20, "cy": 15, "radius": 5},
             ]},
            {"type": "extrude_sketch", "sketch": "BracketProfile",
             "height": 10, "name": "BracketBody"},
            {"type": "create_sketch", "name": "MountHole",
             "elements": [{"type": "circle", "cx": 20, "cy": 15, "radius": 3}]},
            {"type": "pocket", "sketch": "MountHole", "target": "BracketBody",
             "through_all": True, "name": "BracketFinal"},
        ]
        script = _build_script(ops)
        assert "Sketch" in script
        assert "extrude" in script
        assert "cut" in script
        assert "BracketBody" in script
