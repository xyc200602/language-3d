"""Tests for assembly solver sibling auto-distribution.

Tests cover:
- Single child: no offset (backward compatible)
- 2 children: line distribution along tangent1
- 3-4 children: grid distribution at face corners
- 5+ children: circle distribution
- Cylindrical parent parts
- Mixed anchors on same parent
- Explicit offset stacking with distribution
- Complex robot: 4 wheels on base_plate bottom
"""

from __future__ import annotations

import math

import pytest

from lang3d.knowledge.mechanics import Assembly, Joint, Part
from lang3d.tools.assembly_solver import (
    ANCHOR_TANGENTS,
    AssemblySolver,
    _anchor_alignment_rotation,
    _compute_distribution_offset,
    _face_extent_for_part,
    _identity_matrix,
    _mat_vec,
    _rotation_matrix_axis_angle,
)


# ============================================================================
# Fixtures
# ============================================================================


def _make_box(name: str, length: float, width: float, height: float) -> Part:
    """Helper to create a box-shaped part."""
    return Part(
        name=name, category="structural", description=name,
        dimensions={"length": length, "width": width, "height": height},
    )


def _make_cylinder(name: str, diameter: float, height: float) -> Part:
    """Helper to create a cylindrical part."""
    return Part(
        name=name, category="structural", description=name,
        dimensions={"diameter": diameter, "height": height},
    )


# ============================================================================
# Test: _face_extent_for_part
# ============================================================================


class TestFaceExtent:
    """Test _face_extent_for_part returns correct (width, depth) for faces."""

    def test_top_face_box(self):
        part = _make_box("plate", 300, 200, 5)
        w, d = _face_extent_for_part(part, "top")
        assert w == 300  # length along X (tangent1)
        assert d == 200  # width along Y (tangent2)

    def test_bottom_face_box(self):
        part = _make_box("plate", 300, 200, 5)
        w, d = _face_extent_for_part(part, "bottom")
        assert w == 300
        assert d == 200

    def test_left_face_box(self):
        part = _make_box("plate", 300, 200, 5)
        w, d = _face_extent_for_part(part, "left")
        assert w == 5    # height along Z (tangent1)
        assert d == 200  # width along Y (tangent2)

    def test_front_face_box(self):
        part = _make_box("plate", 300, 200, 5)
        w, d = _face_extent_for_part(part, "front")
        assert w == 300  # length along X (tangent1)
        assert d == 5    # height along Z (tangent2)

    def test_cylinder_top(self):
        part = _make_cylinder("post", 20, 120)
        w, d = _face_extent_for_part(part, "top")
        assert w == 20  # diameter
        assert d == 20  # diameter

    def test_no_dimensions(self):
        part = Part(name="empty", category="s", description="")
        w, d = _face_extent_for_part(part, "top")
        assert w == 0
        assert d == 0


# ============================================================================
# Test: _compute_distribution_offset
# ============================================================================


class TestDistributionOffset:
    """Test _compute_distribution_offset for various sibling counts."""

    def test_single_child_zero_offset(self):
        result = _compute_distribution_offset(0, 1, (100, 100), ((1, 0, 0), (0, 1, 0)))
        assert result == (0.0, 0.0, 0.0)

    def test_two_children_line_distribution(self):
        """2 children should be distributed along tangent1 (X for top face)."""
        t = ANCHOR_TANGENTS["top"]  # (1,0,0), (0,1,0)
        face = (100, 80)

        off0 = _compute_distribution_offset(0, 2, face, t)
        off1 = _compute_distribution_offset(1, 2, face, t)

        # Children should be at ±0.35*width along tangent1
        assert off0[0] < 0  # negative X
        assert off1[0] > 0  # positive X
        # Y should be 0 for line distribution
        assert off0[1] == pytest.approx(0, abs=1e-10)
        assert off1[1] == pytest.approx(0, abs=1e-10)
        # Magnitude check
        expected_mag = 0.35 * 100
        assert abs(off0[0]) == pytest.approx(expected_mag, abs=0.01)
        assert abs(off1[0]) == pytest.approx(expected_mag, abs=0.01)

    def test_four_children_grid(self):
        """4 children should be distributed in a 2x2 grid pattern."""
        t = ANCHOR_TANGENTS["top"]  # (1,0,0), (0,1,0)
        face = (200, 160)

        offsets = [
            _compute_distribution_offset(i, 4, face, t)
            for i in range(4)
        ]

        # All 4 should have distinct XY positions
        xy_set = set()
        for o in offsets:
            xy_set.add((round(o[0], 2), round(o[1], 2)))
        assert len(xy_set) == 4, "All 4 children should have distinct positions"

        # Check they're in 4 quadrants
        positive_x = [o for o in offsets if o[0] > 0]
        negative_x = [o for o in offsets if o[0] < 0]
        assert len(positive_x) == 2
        assert len(negative_x) == 2

    def test_three_children_grid(self):
        """3 children should still use grid pattern (first 3 of 4 positions)."""
        t = ANCHOR_TANGENTS["top"]
        face = (200, 160)

        offsets = [
            _compute_distribution_offset(i, 3, face, t)
            for i in range(3)
        ]

        # Should be 3 distinct positions
        xy_set = set()
        for o in offsets:
            xy_set.add((round(o[0], 2), round(o[1], 2)))
        assert len(xy_set) == 3

    def test_five_children_circle(self):
        """5+ children should be distributed in a circle."""
        t = ANCHOR_TANGENTS["top"]
        face = (200, 200)

        offsets = [
            _compute_distribution_offset(i, 5, face, t)
            for i in range(5)
        ]

        # All should have similar magnitude from center
        for o in offsets:
            dist = math.sqrt(o[0] ** 2 + o[1] ** 2)
            assert dist > 0, "Should not be at center"

        # Angular spacing should be ~72 degrees (360/5)
        xy_set = set()
        for o in offsets:
            xy_set.add((round(o[0], 2), round(o[1], 2)))
        assert len(xy_set) == 5, "All 5 children should have distinct positions"

    def test_z_zero_for_top_face(self):
        """For top/bottom faces, distribution should be in XY plane (Z=0)."""
        t = ANCHOR_TANGENTS["top"]
        for i in range(4):
            o = _compute_distribution_offset(i, 4, (100, 100), t)
            assert o[2] == pytest.approx(0, abs=1e-10)


# ============================================================================
# Test: 2-child assembly (line distribution)
# ============================================================================


class TestTwoChildLineDistribution:
    """Test that 2 children on the same anchor face are distributed linearly."""

    def test_two_motors_on_plate_bottom(self):
        """Two motors on bottom face should be separated along X."""
        plate = _make_box("plate", 200, 100, 5)
        motor_a = _make_box("motor_a", 30, 30, 25)
        motor_b = _make_box("motor_b", 30, 30, 25)

        asm = Assembly(
            name="two_motors",
            parts=[plate, motor_a, motor_b],
            joints=[
                Joint("fixed", "plate", "motor_a",
                      parent_anchor="bottom", child_anchor="top"),
                Joint("fixed", "plate", "motor_b",
                      parent_anchor="bottom", child_anchor="top"),
            ],
        )

        solver = AssemblySolver(asm)
        p = solver.solve()

        # Motors should NOT be at the same position
        pos_a = p["motor_a"]["position"]
        pos_b = p["motor_b"]["position"]
        assert pos_a != pos_b, "Two motors should not overlap"

        # Both below plate (negative Z)
        assert pos_a[2] < 0
        assert pos_b[2] < 0

        # Separated in X
        assert pos_a[0] != pytest.approx(pos_b[0], abs=0.1)


# ============================================================================
# Test: 4-child assembly (grid distribution)
# ============================================================================


class TestFourChildGridDistribution:
    """Test that 4 children on the same anchor face are distributed in a grid."""

    def test_four_wheels_on_base_bottom(self):
        """Four wheels under a base plate should be at 4 corners."""
        plate = _make_box("base_plate", 300, 200, 5)
        wheels = [_make_box(f"wheel_{i}", 65, 26, 65) for i in range(4)]

        asm = Assembly(
            name="4wd_chassis",
            parts=[plate] + wheels,
            joints=[
                Joint("fixed", "base_plate", "wheel_0",
                      parent_anchor="bottom", child_anchor="top"),
                Joint("fixed", "base_plate", "wheel_1",
                      parent_anchor="bottom", child_anchor="top"),
                Joint("fixed", "base_plate", "wheel_2",
                      parent_anchor="bottom", child_anchor="top"),
                Joint("fixed", "base_plate", "wheel_3",
                      parent_anchor="bottom", child_anchor="top"),
            ],
        )

        solver = AssemblySolver(asm)
        p = solver.solve()

        # All 4 wheels should have DISTINCT positions
        positions = [tuple(p[f"wheel_{i}"]["position"]) for i in range(4)]
        assert len(set(positions)) == 4, "All 4 wheels should be at different positions"

        # All should be below plate
        for pos in positions:
            assert pos[2] < 0, f"Wheel should be below plate, z={pos[2]}"

        # Should have both positive and negative X offsets
        xs = [pos[0] for pos in positions]
        assert max(xs) > 0, "At least one wheel should be at positive X"
        assert min(xs) < 0, "At least one wheel should be at negative X"

        # Should have both positive and negative Y offsets
        ys = [pos[1] for pos in positions]
        assert max(ys) > 0, "At least one wheel should be at positive Y"
        assert min(ys) < 0, "At least one wheel should be at negative Y"

    def test_four_standoffs_on_plate_top(self):
        """Four standoffs on top of a plate should be at 4 corners."""
        plate = _make_box("base_plate", 300, 200, 5)
        standoffs = [_make_cylinder(f"standoff_{i}", 8, 50) for i in range(4)]

        asm = Assembly(
            name="standoffs",
            parts=[plate] + standoffs,
            joints=[
                Joint("fixed", "base_plate", f"standoff_{i}",
                      parent_anchor="top", child_anchor="bottom")
                for i in range(4)
            ],
        )

        solver = AssemblySolver(asm)
        p = solver.solve()

        # All 4 standoffs should be at distinct positions
        positions = [tuple(p[f"standoff_{i}"]["position"]) for i in range(4)]
        assert len(set(positions)) == 4

        # All should be above plate
        plate_z = p["base_plate"]["position"][2]
        for pos in positions:
            assert pos[2] > plate_z, f"Standoff should be above plate"


# ============================================================================
# Test: 5+ children (circle distribution)
# ============================================================================


class TestFivePlusChildCircleDistribution:
    """Test that 5+ children are distributed in a circle."""

    def test_six_sensors_circle(self):
        """6 sensors around a plate should be evenly distributed in a circle."""
        plate = _make_box("plate", 200, 200, 5)
        sensors = [_make_cylinder(f"sensor_{i}", 10, 5) for i in range(6)]

        asm = Assembly(
            name="sensor_ring",
            parts=[plate] + sensors,
            joints=[
                Joint("fixed", "plate", f"sensor_{i}",
                      parent_anchor="top", child_anchor="bottom")
                for i in range(6)
            ],
        )

        solver = AssemblySolver(asm)
        p = solver.solve()

        # All 6 sensors should have distinct positions
        positions = [tuple(p[f"sensor_{i}"]["position"]) for i in range(6)]
        assert len(set(positions)) == 6

        # All should be above plate
        plate_z = p["plate"]["position"][2]
        for pos in positions:
            assert pos[2] > plate_z


# ============================================================================
# Test: Mixed anchors on same parent
# ============================================================================


class TestMixedAnchors:
    """Test that children on different anchor faces of the same parent don't interfere."""

    def test_top_and_bottom_children_independent(self):
        """Children on top and bottom faces should be distributed independently."""
        plate = _make_box("plate", 200, 100, 5)
        top_a = _make_box("top_a", 20, 20, 10)
        top_b = _make_box("top_b", 20, 20, 10)
        bot_a = _make_box("bot_a", 20, 20, 10)
        bot_b = _make_box("bot_b", 20, 20, 10)

        asm = Assembly(
            name="mixed",
            parts=[plate, top_a, top_b, bot_a, bot_b],
            joints=[
                Joint("fixed", "plate", "top_a", parent_anchor="top", child_anchor="bottom"),
                Joint("fixed", "plate", "top_b", parent_anchor="top", child_anchor="bottom"),
                Joint("fixed", "plate", "bot_a", parent_anchor="bottom", child_anchor="top"),
                Joint("fixed", "plate", "bot_b", parent_anchor="bottom", child_anchor="top"),
            ],
        )

        solver = AssemblySolver(asm)
        p = solver.solve()

        # Top children should be separated
        assert p["top_a"]["position"] != p["top_b"]["position"]
        # Bottom children should be separated
        assert p["bot_a"]["position"] != p["bot_b"]["position"]
        # Top children should be above plate
        assert p["top_a"]["position"][2] > 0
        assert p["top_b"]["position"][2] > 0
        # Bottom children should be below plate
        assert p["bot_a"]["position"][2] < 0
        assert p["bot_b"]["position"][2] < 0


# ============================================================================
# Test: Explicit offset stacking with distribution
# ============================================================================


class TestOffsetWithDistribution:
    """Test that explicit joint offsets stack on top of distribution offsets."""

    def test_four_wheels_with_offset(self):
        """4 wheels with an additional Z offset should still be grid-distributed."""
        plate = _make_box("base", 300, 200, 5)
        wheels = [_make_box(f"w_{i}", 65, 26, 65) for i in range(4)]

        asm = Assembly(
            name="offset_wheels",
            parts=[plate] + wheels,
            joints=[
                Joint("fixed", "base", f"w_{i}",
                      parent_anchor="bottom", child_anchor="top",
                      offset=(0, 0, -10))  # 10mm extra gap
                for i in range(4)
            ],
        )

        solver = AssemblySolver(asm)
        p = solver.solve()

        # All 4 should still be at distinct positions
        positions = [tuple(p[f"w_{i}"]["position"]) for i in range(4)]
        assert len(set(positions)) == 4

        # All should be below the no-offset case by ~10mm extra
        solver_no_off = AssemblySolver(Assembly(
            name="no_offset_wheels",
            parts=[plate] + [_make_box(f"w_{i}", 65, 26, 65) for i in range(4)],
            joints=[
                Joint("fixed", "base", f"w_{i}",
                      parent_anchor="bottom", child_anchor="top")
                for i in range(4)
            ],
        ))
        p_no = solver_no_off.solve()

        for i in range(4):
            z_diff = p[f"w_{i}"]["position"][2] - p_no[f"w_{i}"]["position"][2]
            # The offset is along -Z in local frame, so z should be lower
            assert z_diff < 0, f"Wheel {i} should be lower with negative offset"


# ============================================================================
# Test: Cylindrical parent distribution
# ============================================================================


class TestCylindricalParent:
    """Test distribution when the parent is a cylindrical part."""

    def test_cylindrical_hub_four_arms(self):
        """4 arms attached to a cylindrical hub should be at 4 positions."""
        hub = _make_cylinder("hub", 80, 40)
        arms = [_make_box(f"arm_{i}", 100, 20, 10) for i in range(4)]

        asm = Assembly(
            name="rotary_joint",
            parts=[hub] + arms,
            joints=[
                Joint("fixed", "hub", f"arm_{i}",
                      parent_anchor="top", child_anchor="bottom")
                for i in range(4)
            ],
        )

        solver = AssemblySolver(asm)
        p = solver.solve()

        # All 4 arms should have distinct positions
        positions = [tuple(p[f"arm_{i}"]["position"]) for i in range(4)]
        assert len(set(positions)) == 4


# ============================================================================
# Test: Complex robot wheel distribution (mirrors build_robot_3d.py)
# ============================================================================


class TestComplexRobotWheels:
    """Test the exact scenario from the 41-part robot: 4 motors on base_plate bottom."""

    def test_four_motors_separated(self):
        """4 motors attached to base_plate bottom should be grid-distributed."""
        base_plate = _make_box("base_plate", 300, 200, 5)
        motors = [_make_box(f"motor_{s}", 40, 30, 25) for s in ["fl", "fr", "rl", "rr"]]

        asm = Assembly(
            name="chassis",
            parts=[base_plate] + motors,
            joints=[
                Joint("fixed", "base_plate", f"motor_{s}",
                      parent_anchor="bottom", child_anchor="top")
                for s in ["fl", "fr", "rl", "rr"]
            ],
        )

        solver = AssemblySolver(asm)
        p = solver.solve()

        # All motors at distinct positions
        positions = {s: tuple(p[f"motor_{s}"]["position"]) for s in ["fl", "fr", "rl", "rr"]}
        assert len(set(positions.values())) == 4

        # FL and FR should be at different X from RL and RR (front vs rear)
        # All motors below plate
        for s, pos in positions.items():
            assert pos[2] < 0, f"Motor {s} should be below plate, z={pos[2]}"

    def test_dual_arms_on_top_plate_separated(self):
        """Two arm bases on top_plate should be line-distributed (2 children)."""
        top_plate = _make_box("top_plate", 280, 180, 3)
        arm_l_base = _make_cylinder("arm_l_base", 80, 40)
        arm_r_base = _make_cylinder("arm_r_base", 80, 40)

        asm = Assembly(
            name="dual_arms",
            parts=[top_plate, arm_l_base, arm_r_base],
            joints=[
                Joint("revolute", "top_plate", "arm_l_base",
                      (-180, 180), axis="z",
                      parent_anchor="top", child_anchor="bottom"),
                Joint("revolute", "top_plate", "arm_r_base",
                      (-180, 180), axis="z",
                      parent_anchor="top", child_anchor="bottom"),
            ],
        )

        solver = AssemblySolver(asm)
        p = solver.solve()

        # Arm bases should NOT be at the same position
        pos_l = p["arm_l_base"]["position"]
        pos_r = p["arm_r_base"]["position"]
        assert pos_l != pos_r, "Dual arm bases should be separated"

        # Both should be above top plate
        plate_z = p["top_plate"]["position"][2]
        assert pos_l[2] > plate_z
        assert pos_r[2] > plate_z


# ============================================================================
# Test: Single child unchanged (backward compatibility)
# ============================================================================


class TestBackwardCompatibility:
    """Ensure single-child assemblies produce the same results as before."""

    def test_single_child_no_distribution_offset(self):
        """A single child should not get any distribution offset."""
        plate = _make_box("plate", 300, 200, 5)
        post = _make_cylinder("post", 20, 100)

        asm = Assembly(
            name="single",
            parts=[plate, post],
            joints=[
                Joint("fixed", "plate", "post",
                      parent_anchor="top", child_anchor="bottom"),
            ],
        )

        solver = AssemblySolver(asm)
        p = solver.solve()

        # Post should be centered on plate (X=0, Y=0)
        assert p["post"]["position"][0] == pytest.approx(0, abs=0.01)
        assert p["post"]["position"][1] == pytest.approx(0, abs=0.01)

    def test_chain_unchanged(self):
        """A chain of single children should still stack vertically."""
        parts = [
            _make_box("a", 100, 100, 10),
            _make_box("b", 50, 50, 20),
            _make_box("c", 30, 30, 30),
        ]
        asm = Assembly(
            name="chain",
            parts=parts,
            joints=[
                Joint("fixed", "a", "b", parent_anchor="top", child_anchor="bottom"),
                Joint("fixed", "b", "c", parent_anchor="top", child_anchor="bottom"),
            ],
        )

        solver = AssemblySolver(asm)
        p = solver.solve()

        # All centered at X=0, Y=0
        for name in ["a", "b", "c"]:
            assert p[name]["position"][0] == pytest.approx(0, abs=0.01)
            assert p[name]["position"][1] == pytest.approx(0, abs=0.01)

        # Ascending Z
        assert p["b"]["position"][2] > p["a"]["position"][2]
        assert p["c"]["position"][2] > p["b"]["position"][2]


# ============================================================================
# Test: Anchor alignment rotation
# ============================================================================


class TestAnchorAlignmentRotation:
    """Test that _anchor_alignment_rotation produces correct rotations."""

    def test_top_to_bottom_is_identity(self):
        """top→bottom: child bottom (0,0,-1) should face parent top (0,0,1) → target (0,0,-1).
        (0,0,-1)→(0,0,-1) is identity rotation."""
        rot = _anchor_alignment_rotation("top", "bottom")
        assert rot == _identity_matrix()

    def test_bottom_to_top_is_identity(self):
        """bottom→top: similar reasoning, identity."""
        rot = _anchor_alignment_rotation("bottom", "top")
        assert rot == _identity_matrix()

    def test_left_to_right_is_identity(self):
        """left→right: child right (1,0,0) faces parent left (-1,0,0) → target (1,0,0).
        (1,0,0)→(1,0,0) is identity."""
        rot = _anchor_alignment_rotation("left", "right")
        assert rot == _identity_matrix()

    def test_left_to_bottom_is_90_around_y(self):
        """left→bottom: child bottom (0,0,-1) faces parent left (-1,0,0) → target (1,0,0).
        Should rotate (0,0,-1) to (1,0,0), which is -90° around Y."""
        rot = _anchor_alignment_rotation("left", "bottom")
        # Verify: rot @ (0,0,-1) should ≈ (1,0,0)
        result = _mat_vec(rot, (0, 0, -1))
        assert result[0] == pytest.approx(1.0, abs=0.01)
        assert result[1] == pytest.approx(0.0, abs=0.01)
        assert result[2] == pytest.approx(0.0, abs=0.01)

    def test_right_to_bottom_is_90_around_y(self):
        """right→bottom: child bottom (0,0,-1) faces parent right (1,0,0) → target (-1,0,0).
        Should rotate (0,0,-1) to (-1,0,0), which is 90° around Y."""
        rot = _anchor_alignment_rotation("right", "bottom")
        result = _mat_vec(rot, (0, 0, -1))
        assert result[0] == pytest.approx(-1.0, abs=0.01)
        assert result[1] == pytest.approx(0.0, abs=0.01)
        assert result[2] == pytest.approx(0.0, abs=0.01)

    def test_front_to_bottom(self):
        """front→bottom: child bottom (0,0,-1) faces parent front (0,-1,0) → target (0,1,0).
        Should rotate (0,0,-1) to (0,1,0)."""
        rot = _anchor_alignment_rotation("front", "bottom")
        result = _mat_vec(rot, (0, 0, -1))
        assert result[0] == pytest.approx(0.0, abs=0.01)
        assert result[1] == pytest.approx(1.0, abs=0.01)
        assert result[2] == pytest.approx(0.0, abs=0.01)


class TestLateralAnchorRotation:
    """Test that lateral anchor connections produce rotated child orientations."""

    def test_wheel_on_motor_left_gets_rotated(self):
        """A wheel attached to motor's left face should have non-zero rotation."""
        motor = _make_box("motor", 40, 30, 25)
        wheel = _make_cylinder("wheel", 32.5, 26)

        asm = Assembly(
            name="motor_wheel",
            parts=[motor, wheel],
            joints=[
                Joint("revolute", "motor", "wheel",
                      axis="z", range_deg=(-360, 360),
                      parent_anchor="left", child_anchor="bottom"),
            ],
        )

        solver = AssemblySolver(asm)
        p = solver.solve()

        # Wheel rotation should NOT be identity (0,0,1,0)
        wheel_rot = p["wheel"]["rotation"]
        # For left→bottom, the wheel should be rotated
        assert abs(wheel_rot[3]) > 0.1, "Wheel should have non-trivial rotation"

    def test_wheel_position_is_lateral(self):
        """Wheel should be to the left of the motor, not above or below."""
        motor = _make_box("motor", 40, 30, 25)
        wheel = _make_cylinder("wheel", 32.5, 26)

        asm = Assembly(
            name="motor_wheel",
            parts=[motor, wheel],
            joints=[
                Joint("fixed", "motor", "wheel",
                      parent_anchor="left", child_anchor="bottom"),
            ],
        )

        solver = AssemblySolver(asm)
        p = solver.solve()

        motor_x = p["motor"]["position"][0]
        wheel_x = p["wheel"]["position"][0]
        # Wheel should be to the LEFT of motor (more negative X)
        assert wheel_x < motor_x, f"Wheel ({wheel_x}) should be left of motor ({motor_x})"

    def test_encoder_on_motor_right_gets_rotated(self):
        """An encoder on motor's right face should be to the right of motor."""
        motor = _make_box("motor", 40, 30, 25)
        encoder = _make_cylinder("encoder", 6, 5)

        asm = Assembly(
            name="motor_encoder",
            parts=[motor, encoder],
            joints=[
                Joint("fixed", "motor", "encoder",
                      parent_anchor="right", child_anchor="bottom"),
            ],
        )

        solver = AssemblySolver(asm)
        p = solver.solve()

        motor_x = p["motor"]["position"][0]
        encoder_x = p["encoder"]["position"][0]
        # Encoder should be to the RIGHT of motor
        assert encoder_x > motor_x, f"Encoder ({encoder_x}) should be right of motor ({motor_x})"
