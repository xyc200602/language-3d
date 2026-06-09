"""Tests for URDF export rotation support (solver positions → joint origin).

Covers:
  - _axis_angle_to_rpy conversion function
  - AssemblyToURDF with solver positions uses computed joint origin
  - AssemblyToURDF without positions falls back to joint.offset
  - Roundtrip: axis-angle → RPY matches expected values
"""

import math

import pytest

from lang3d.knowledge.mechanics import Assembly, Joint, Part
from lang3d.tools.urdf_export import (
    AssemblyToURDF,
    _axis_angle_to_rpy,
    _mm_to_m,
)


# ============================================================================
# _axis_angle_to_rpy
# ============================================================================


class TestAxisAngleToRpy:
    """Unit tests for the axis-angle → RPY conversion."""

    def test_identity(self):
        """Identity rotation [0,0,1,0] → (0, 0, 0)."""
        rpy = _axis_angle_to_rpy([0, 0, 1, 0])
        assert rpy == (0.0, 0.0, 0.0)

    def test_short_list(self):
        """List shorter than 4 elements → identity."""
        rpy = _axis_angle_to_rpy([0, 0, 1])
        assert rpy == (0.0, 0.0, 0.0)

    def test_empty(self):
        """Empty list → identity."""
        rpy = _axis_angle_to_rpy([])
        assert rpy == (0.0, 0.0, 0.0)

    def test_90y(self):
        """Rotation of 90 degrees around Y axis → pitch ≈ π/2."""
        rpy = _axis_angle_to_rpy([0, 1, 0, 90])
        roll, pitch, yaw = rpy
        assert roll == pytest.approx(0.0, abs=1e-6)
        assert pitch == pytest.approx(math.pi / 2, abs=1e-4)
        assert yaw == pytest.approx(0.0, abs=1e-6)

    def test_45x(self):
        """Rotation of 45 degrees around X axis → roll ≈ π/4."""
        rpy = _axis_angle_to_rpy([1, 0, 0, 45])
        roll, pitch, yaw = rpy
        assert roll == pytest.approx(math.pi / 4, abs=1e-4)
        assert pitch == pytest.approx(0.0, abs=1e-6)
        assert yaw == pytest.approx(0.0, abs=1e-6)

    def test_90z(self):
        """Rotation of 90 degrees around Z axis → yaw ≈ π/2."""
        rpy = _axis_angle_to_rpy([0, 0, 1, 90])
        roll, pitch, yaw = rpy
        assert roll == pytest.approx(0.0, abs=1e-6)
        assert pitch == pytest.approx(0.0, abs=1e-6)
        assert yaw == pytest.approx(math.pi / 2, abs=1e-4)

    def test_180y(self):
        """Rotation of 180 degrees around Y axis.

        Decomposed as ZYX Euler: R = Rz(yaw) * Ry(pitch) * Rx(roll).
        For pure Y-180: the matrix is diag(-1, 1, -1), which decomposes
        as roll=π, pitch=0, yaw=π (equivalent to a single Y-180 rotation).
        """
        rpy = _axis_angle_to_rpy([0, 1, 0, 180])
        roll, pitch, yaw = rpy
        # roll=π, yaw=π is equivalent to pitch=π (Rz(π)*Ry(0)*Rx(π) = Ry(π))
        assert roll == pytest.approx(math.pi, abs=1e-4)
        assert yaw == pytest.approx(math.pi, abs=1e-4)

    def test_small_angle(self):
        """Very small angle < 1e-6 → identity."""
        rpy = _axis_angle_to_rpy([0, 0, 1, 1e-7])
        assert rpy == (0.0, 0.0, 0.0)

    def test_negative_angle(self):
        """Negative angle (clockwise) rotation around X."""
        rpy = _axis_angle_to_rpy([1, 0, 0, -90])
        roll, pitch, yaw = rpy
        assert roll == pytest.approx(-math.pi / 2, abs=1e-4)

    def test_unnormalized_axis(self):
        """Axis doesn't need to be unit length — function should normalize."""
        rpy = _axis_angle_to_rpy([0, 2, 0, 90])
        roll, pitch, yaw = rpy
        assert pitch == pytest.approx(math.pi / 2, abs=1e-4)


# ============================================================================
# AssemblyToURDF with positions
# ============================================================================


def _make_wheel_assembly():
    """Create a simple chassis + 2-wheel assembly."""
    return Assembly(
        name="WheeledBot",
        parts=[
            Part(name="chassis", category="structural", description="Chassis",
                 dimensions=dict(length=150, width=100, height=10), mass=0.5),
            Part(name="wheel_l", category="structural", description="Left wheel",
                 dimensions=dict(diameter=65, height=26), mass=0.1),
            Part(name="wheel_r", category="structural", description="Right wheel",
                 dimensions=dict(diameter=65, height=26), mass=0.1),
        ],
        joints=[
            Joint("revolute", "chassis", "wheel_l", (-180, 180),
                  axis="y", parent_anchor="left", child_anchor="center"),
            Joint("revolute", "chassis", "wheel_r", (-180, 180),
                  axis="y", parent_anchor="right", child_anchor="center"),
        ],
    )


class TestURDFWithPositions:
    """Test that solver positions are used for joint origin computation."""

    def test_uses_solver_xyz(self):
        """When positions are provided, joint origin xyz should be computed from solver data."""
        asm = _make_wheel_assembly()
        positions = {
            "chassis": {"position": [0, 0, 50], "rotation": [0, 0, 1, 0]},
            "wheel_l": {"position": [-75, 0, 25], "rotation": [1, 0, 0, 90]},
            "wheel_r": {"position": [75, 0, 25], "rotation": [1, 0, 0, -90]},
        }
        converter = AssemblyToURDF(asm, positions=positions)
        converter.convert()
        joints = converter.get_joints()
        assert len(joints) == 2

        # Left wheel: child - parent = [-75-0, 0-0, 25-50] = [-75, 0, -25] mm
        # In meters: [-0.075, 0.0, -0.025]
        left_joint = joints[0]
        assert left_joint.origin_xyz[0] == pytest.approx(-0.075, abs=1e-4)
        assert left_joint.origin_xyz[1] == pytest.approx(0.0, abs=1e-4)
        assert left_joint.origin_xyz[2] == pytest.approx(-0.025, abs=1e-4)

        # Right wheel: child - parent = [75-0, 0-0, 25-50] = [75, 0, -25] mm
        right_joint = joints[1]
        assert right_joint.origin_xyz[0] == pytest.approx(0.075, abs=1e-4)
        assert right_joint.origin_xyz[2] == pytest.approx(-0.025, abs=1e-4)

    def test_uses_solver_rpy(self):
        """When parent/child have different rotations, joint origin rpy should be non-zero.

        The rpy is now computed as R_child * R_parent^T (relative rotation),
        so any difference between parent and child rotations produces non-zero rpy.
        """
        asm = _make_wheel_assembly()
        positions = {
            "chassis": {"position": [0, 0, 50], "rotation": [0, 1, 0, 90]},
            "wheel_l": {"position": [-75, 0, 25], "rotation": [1, 0, 0, 90]},
            "wheel_r": {"position": [75, 0, 25], "rotation": [1, 0, 0, -90]},
        }
        converter = AssemblyToURDF(asm, positions=positions)
        converter.convert()
        joints = converter.get_joints()

        # Relative rotation should be non-zero since parent and child differ
        for j in joints:
            roll, pitch, yaw = j.origin_rpy
            has_rotation = abs(roll) > 0.01 or abs(pitch) > 0.01 or abs(yaw) > 0.01
            assert has_rotation, (
                f"Expected non-zero relative rotation, got rpy=({roll:.4f}, {pitch:.4f}, {yaw:.4f})"
            )

    def test_child_rotation_produces_rpy(self):
        """When only child has rotation (parent is identity), joint rpy should be non-zero.

        This is the critical case for wheels: parent (motor) has no rotation,
        child (wheel) has 90° rotation to stand upright.
        """
        asm = _make_wheel_assembly()
        positions = {
            "chassis": {"position": [0, 0, 50], "rotation": [0, 0, 1, 0]},
            "wheel_l": {"position": [-75, 0, 25], "rotation": [1, 0, 0, 90]},
            "wheel_r": {"position": [75, 0, 25], "rotation": [1, 0, 0, -90]},
        }
        converter = AssemblyToURDF(asm, positions=positions)
        converter.convert()
        joints = converter.get_joints()

        for j in joints:
            roll, pitch, yaw = j.origin_rpy
            has_rotation = abs(roll) > 0.01 or abs(pitch) > 0.01 or abs(yaw) > 0.01
            assert has_rotation, (
                f"Expected non-zero rpy from child rotation, got ({roll:.4f}, {pitch:.4f}, {yaw:.4f})"
            )

    def test_xml_contains_nonzero_rpy(self):
        """Generated URDF XML should have non-zero rpy values when rotations exist."""
        asm = _make_wheel_assembly()
        positions = {
            "chassis": {"position": [0, 0, 50], "rotation": [0, 1, 0, 90]},
            "wheel_l": {"position": [-75, 0, 25], "rotation": [1, 0, 0, 90]},
            "wheel_r": {"position": [75, 0, 25], "rotation": [1, 0, 0, -90]},
        }
        converter = AssemblyToURDF(asm, positions=positions)
        xml = converter.convert()
        # At least one joint should have a non-zero rpy value
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml)
        joint_origins = root.findall(".//joint/origin")
        has_nonzero_rpy = False
        for origin in joint_origins:
            rpy = origin.get("rpy", "0 0 0")
            parts = rpy.split()
            if any(abs(float(p)) > 1e-6 for p in parts):
                has_nonzero_rpy = True
                break
        assert has_nonzero_rpy, "Expected at least one joint with non-zero rpy"


class TestURDFWithoutPositions:
    """Test fallback behavior when no positions are provided."""

    def test_falls_back_to_offset(self):
        """Without positions, joint origin should fall back to joint.offset / parent height."""
        asm = _make_wheel_assembly()
        converter = AssemblyToURDF(asm)  # no positions
        converter.convert()
        joints = converter.get_joints()
        assert len(joints) == 2
        # Should have some origin (from parent height estimation or offset)
        for j in joints:
            assert j.origin_xyz is not None

    def test_fallback_rpy_is_zero(self):
        """Without positions, rpy should be (0, 0, 0)."""
        asm = _make_wheel_assembly()
        converter = AssemblyToURDF(asm)
        converter.convert()
        joints = converter.get_joints()
        for j in joints:
            assert j.origin_rpy == (0.0, 0.0, 0.0)

    def test_empty_positions_dict(self):
        """Empty positions dict should behave like no positions."""
        asm = _make_wheel_assembly()
        converter = AssemblyToURDF(asm, positions={})
        converter.convert()
        joints = converter.get_joints()
        for j in joints:
            assert j.origin_rpy == (0.0, 0.0, 0.0)

    def test_partial_positions_falls_back(self):
        """If only some parts have positions, missing ones should fall back."""
        asm = _make_wheel_assembly()
        positions = {
            "chassis": {"position": [0, 0, 50]},
            # wheel_l and wheel_r missing from positions
        }
        converter = AssemblyToURDF(asm, positions=positions)
        converter.convert()
        joints = converter.get_joints()
        # Both joints should fall back since child positions are missing
        for j in joints:
            assert j.origin_rpy == (0.0, 0.0, 0.0)
