"""Tests for bolt hole world-position computation.

Verifies that :func:`_compute_bolt_hole_world_positions` correctly
transforms bolt hole locations from part-local FreeCAD coordinates
(corner-origin) to world-space (center-origin + solver rotation).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from lang3d.tools.freecad import (
    _compute_bolt_hole_world_positions,
    _z_to_normal_rotation,
)
from lang3d.tools.vtk_renderer import _rotation_from_z_to
from lang3d.knowledge.mechanics import Joint, ConnectionMethod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_bolted_joint(parent="bracket", child="mount",
                       pa="top", ca="bottom",
                       bolt_size="M3", bolt_count=4):
    return Joint(
        type="fixed", parent=parent, child=child,
        parent_anchor=pa, child_anchor=ca,
        connection=ConnectionMethod(
            type="bolted", bolt_size=bolt_size, bolt_count=bolt_count,
        ),
    )


BRACKET_60x40x5 = {"length": 60, "width": 40, "height": 5}
MOUNT_30x30x10 = {"length": 30, "width": 30, "height": 10}


# ---------------------------------------------------------------------------
# Position transform tests
# ---------------------------------------------------------------------------

class TestBoltHolePositions:
    def test_top_anchor_4_bolts_at_origin(self):
        """4 M3 bolts on a 60x40x5 bracket at origin, anchor=top."""
        joint = _make_bolted_joint()
        positions = {
            "bracket": {"position": [0, 0, 0], "rotation": [0, 0, 1, 0]},
            "mount": {"position": [0, 0, 10], "rotation": [0, 0, 1, 0]},
        }
        part_dims = {"bracket": BRACKET_60x40x5, "mount": MOUNT_30x30x10}

        results = _compute_bolt_hole_world_positions(joint, positions, part_dims)

        assert len(results) == 4
        # All bolts at z=2.5 (centre of 5mm height)
        for pos, normal, thickness in results:
            assert pos[2] == pytest.approx(2.5, abs=0.01)
            assert normal == (0, 0, 1)  # top face normal = +Z
            assert thickness == pytest.approx(5.0, abs=0.01)

        # Bolts should be symmetric around origin in X and Y
        xs = sorted(r[0][0] for r in results)
        ys = sorted(r[0][1] for r in results)
        assert xs[0] == pytest.approx(-xs[-1], abs=0.1)
        assert ys[0] == pytest.approx(-ys[-1], abs=0.1)

    def test_top_anchor_offset_position(self):
        """Bracket at non-origin position — bolts track the offset."""
        joint = _make_bolted_joint()
        positions = {
            "bracket": {"position": [100, 50, 200], "rotation": [0, 0, 1, 0]},
            "mount": {"position": [100, 50, 210], "rotation": [0, 0, 1, 0]},
        }
        part_dims = {"bracket": BRACKET_60x40x5, "mount": MOUNT_30x30x10}

        results = _compute_bolt_hole_world_positions(joint, positions, part_dims)

        for pos, _, _ in results:
            # Z should be 200 + 2.5 = 202.5
            assert pos[2] == pytest.approx(202.5, abs=0.01)
            # X and Y should be offset from 100, 50
            assert abs(pos[0] - 100) < 30
            assert abs(pos[1] - 50) < 25

    def test_rotated_parent_90_around_z(self):
        """Parent rotated 90° around Z — bolt X/Y positions swap."""
        joint = _make_bolted_joint()
        positions = {
            "bracket": {"position": [0, 0, 0], "rotation": [0, 0, 1, 90]},
            "mount": {"position": [0, 0, 10], "rotation": [0, 0, 1, 0]},
        }
        part_dims = {"bracket": BRACKET_60x40x5, "mount": MOUNT_30x30x10}

        results_no_rot = _compute_bolt_hole_world_positions(
            joint,
            {"bracket": {"position": [0, 0, 0], "rotation": [0, 0, 1, 0]},
             "mount": {"position": [0, 0, 10], "rotation": [0, 0, 1, 0]}},
            part_dims,
        )
        results_rot = _compute_bolt_hole_world_positions(
            joint, positions, part_dims,
        )

        # After 90° Z rotation, old X values become new Y values (with sign)
        old_xs = sorted(r[0][0] for r in results_no_rot)
        new_ys = sorted(r[0][1] for r in results_rot)
        # R_z(90°) maps (x,y) → (-y, x), so old_x maps to new_y
        for ox, ny in zip(old_xs, new_ys):
            assert ox == pytest.approx(ny, abs=0.5) or ox == pytest.approx(-ny, abs=0.5)

    def test_bottom_anchor_normal_points_down(self):
        """Bottom anchor → normal should be (0, 0, -1)."""
        joint = _make_bolted_joint(pa="bottom", ca="top")
        positions = {
            "bracket": {"position": [0, 0, 0], "rotation": [0, 0, 1, 0]},
            "mount": {"position": [0, 0, -10], "rotation": [0, 0, 1, 0]},
        }
        part_dims = {"bracket": BRACKET_60x40x5, "mount": MOUNT_30x30x10}

        results = _compute_bolt_hole_world_positions(joint, positions, part_dims)

        assert len(results) == 4
        for _, normal, _ in results:
            assert normal == (0, 0, -1)

    def test_different_bolt_counts(self):
        """bolt_count=2 → 2 positions, bolt_count=1 → 1 at centre."""
        positions = {
            "bracket": {"position": [0, 0, 0], "rotation": [0, 0, 1, 0]},
            "mount": {"position": [0, 0, 10], "rotation": [0, 0, 1, 0]},
        }
        part_dims = {"bracket": BRACKET_60x40x5, "mount": MOUNT_30x30x10}

        # 2 bolts
        j2 = _make_bolted_joint(bolt_count=2)
        r2 = _compute_bolt_hole_world_positions(j2, positions, part_dims)
        assert len(r2) == 2

        # 1 bolt at face centre
        j1 = _make_bolted_joint(bolt_count=1)
        r1 = _compute_bolt_hole_world_positions(j1, positions, part_dims)
        assert len(r1) == 1
        assert r1[0][0][0] == pytest.approx(0, abs=0.1)  # X at centre
        assert r1[0][0][1] == pytest.approx(0, abs=0.1)  # Y at centre


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_connection_none_returns_empty(self):
        """Joint without connection → empty list."""
        joint = Joint(type="fixed", parent="a", child="b",
                      parent_anchor="top", child_anchor="bottom")
        positions = {"a": {"position": [0,0,0], "rotation": [0,0,1,0]}}
        part_dims = {"a": BRACKET_60x40x5}

        results = _compute_bolt_hole_world_positions(joint, positions, part_dims)
        assert results == []

    def test_non_bolted_connection_returns_empty(self):
        """Press-fit connection → empty list (no bolts to place)."""
        joint = Joint(
            type="revolute", parent="a", child="b",
            parent_anchor="top", child_anchor="bottom",
            connection=ConnectionMethod(type="press_fit", interference_mm=0.01),
        )
        positions = {"a": {"position": [0,0,0], "rotation": [0,0,1,0]}}
        part_dims = {"a": BRACKET_60x40x5}

        results = _compute_bolt_hole_world_positions(joint, positions, part_dims)
        assert results == []

    def test_prismatic_joint_returns_empty(self):
        """Prismatic (sliding) joints have no bolts."""
        joint = Joint(
            type="prismatic", parent="a", child="b",
            parent_anchor="center", child_anchor="center",
            connection=ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=4),
        )
        positions = {"a": {"position": [0,0,0], "rotation": [0,0,1,0]}}
        part_dims = {"a": BRACKET_60x40x5}

        results = _compute_bolt_hole_world_positions(joint, positions, part_dims)
        assert results == []

    def test_cylindrical_part_returns_empty(self):
        """When BOTH parts are cylindrical (no length/width face), bolt
        layout cannot be placed on either → skip and return [].

        A single cylindrical part with a box partner falls back to the
        box partner (covered by test_cylindrical_parent_falls_back_to_box_child).
        """
        joint = _make_bolted_joint()
        positions = {
            "bracket": {"position": [0, 0, 0], "rotation": [0, 0, 1, 0]},
            "mount": {"position": [0, 0, 10], "rotation": [0, 0, 1, 0]},
        }
        part_dims = {
            "bracket": {"diameter": 30, "height": 10},
            "mount": {"diameter": 20, "height": 8},  # both cylindrical
        }

        results = _compute_bolt_hole_world_positions(joint, positions, part_dims)
        assert results == []

    def test_cylindrical_parent_falls_back_to_box_child(self):
        """When the heuristic-picked structural part is cylindrical but
        the other part is a box, _compute_bolt_hole_world_positions must
        fall back to the box part and place bolts there.

        Engineering rationale: a servo motor (cylinder) bolted to a
        mounting bracket (box) needs the bolts on the bracket, since
        the cylindrical housing cannot receive a bolt pattern.

        Regression guard for the fallback introduced in commit 031d82c.
        """
        joint = _make_bolted_joint()  # parent=bracket, child=mount
        positions = {
            "bracket": {"position": [0, 0, 0], "rotation": [0, 0, 1, 0]},
            "mount": {"position": [0, 0, 10], "rotation": [0, 0, 1, 0]},
        }
        part_dims = {
            "bracket": {"diameter": 30, "height": 10},   # cylindrical
            "mount": MOUNT_30x30x10,                      # box
        }
        results = _compute_bolt_hole_world_positions(joint, positions, part_dims)
        # Layout placed on the box (mount), not the cylinder (bracket)
        assert len(results) == 4, f"Expected 4 bolt holes on mount, got {len(results)}"
        # All holes should sit on the box's bottom face (child_anchor=bottom)
        for world_pos, normal, thickness in results:
            assert thickness == 10, f"Expected mount thickness 10, got {thickness}"
            assert normal == (0.0, 0.0, -1.0), (
                f"Expected bottom-face normal (0,0,-1), got {normal}"
            )

    def test_missing_part_in_positions_returns_empty(self):
        """Structural part not in positions dict → empty."""
        joint = _make_bolted_joint()
        positions = {"mount": {"position": [0,0,10], "rotation": [0,0,1,0]}}
        part_dims = {"bracket": BRACKET_60x40x5, "mount": MOUNT_30x30x10}

        results = _compute_bolt_hole_world_positions(joint, positions, part_dims)
        assert results == []


# ---------------------------------------------------------------------------
# Rotation helper tests
# ---------------------------------------------------------------------------

class TestRotationFromZ:
    def test_already_z_no_rotation(self):
        assert _z_to_normal_rotation((0, 0, 1)) is None
        assert _rotation_from_z_to((0, 0, 1)) is None

    def test_negative_z_180_around_x(self):
        rot_fc = _z_to_normal_rotation((0, 0, -1))
        rot_vtk = _rotation_from_z_to((0, 0, -1))
        assert rot_fc is not None
        assert rot_fc[3] == pytest.approx(180.0, abs=0.1)
        assert rot_vtk == rot_fc  # both should agree

    def test_x_axis_rotation(self):
        """Normal +X requires 90° rotation around Y."""
        rot = _z_to_normal_rotation((1, 0, 0))
        assert rot is not None
        assert rot[3] == pytest.approx(90.0, abs=0.1)

    def test_y_axis_rotation(self):
        """Normal +Y requires 90° rotation around X."""
        rot = _z_to_normal_rotation((0, 1, 0))
        assert rot is not None
        assert rot[3] == pytest.approx(90.0, abs=0.1)

    def test_freecad_and_vtk_agree(self):
        """Both rotation helpers must produce identical results."""
        for normal in [(0,0,1), (0,0,-1), (1,0,0), (-1,0,0),
                        (0,1,0), (0,-1,0)]:
            assert _z_to_normal_rotation(normal) == _rotation_from_z_to(normal)
