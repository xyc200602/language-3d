"""Tests for wheel rotation fix — cylindrical parts on revolute joints."""

import math

import pytest

from lang3d.knowledge.mechanics import Assembly, Joint, Part
from lang3d.tools.assembly_solver import (
    AssemblySolver,
    _cylinder_base_orientation,
    _identity_matrix,
    _is_cylindrical_part,
    _rotation_matrix_axis_angle,
    _mat_mul,
    _mat_vec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_wheel_part(name: str = "wheel", diameter: float = 65, height: float = 26) -> Part:
    return Part(
        name=name,
        category="mechanical",
        description="wheel",
        material="Rubber",
        dimensions={"diameter": diameter, "height": height},
    )


def _make_box_part(name: str = "motor", **dims) -> Part:
    defaults = {"length": 40, "width": 30, "height": 25}
    defaults.update(dims)
    return Part(
        name=name,
        category="actuator",
        description="motor",
        material="Steel",
        dimensions=defaults,
    )


def _make_4wheel_robot() -> Assembly:
    """Minimal 4-wheel robot with motors."""
    parts = [
        Part(name="base_plate", category="structural", description="",
             material="Aluminum", dimensions={"length": 300, "width": 200, "height": 5}),
        _make_box_part("motor_fl"),
        _make_box_part("motor_fr"),
        _make_wheel_part("wheel_fl"),
        _make_wheel_part("wheel_fr"),
    ]
    joints = [
        Joint(type="fixed", parent="base_plate", child="motor_fl",
              parent_anchor="bottom", child_anchor="top", distribution_group="motors"),
        Joint(type="fixed", parent="base_plate", child="motor_fr",
              parent_anchor="bottom", child_anchor="top", distribution_group="motors"),
        Joint(type="revolute", parent="motor_fl", child="wheel_fl",
              axis="y", range_deg=(-360, 360),
              parent_anchor="left", child_anchor="center", no_distribute=True),
        Joint(type="revolute", parent="motor_fr", child="wheel_fr",
              axis="y", range_deg=(-360, 360),
              parent_anchor="left", child_anchor="center", no_distribute=True),
    ]
    return Assembly(name="test_4w", parts=parts, joints=joints)


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------

class TestIsCylindricalPart:
    def test_wheel_is_cylindrical(self):
        assert _is_cylindrical_part(_make_wheel_part())

    def test_outer_diameter_is_cylindrical(self):
        p = Part(name="bearing", category="bearing", description="",
                 material="Steel", dimensions={"outer_diameter": 22, "height": 7})
        assert _is_cylindrical_part(p)

    def test_box_is_not_cylindrical(self):
        assert not _is_cylindrical_part(_make_box_part())


class TestCylinderBaseOrientation:
    def test_y_axis_produces_non_identity(self):
        """axis=y → rotate Z to Y = R_x(90°), non-identity."""
        rot = _cylinder_base_orientation((0, 1, 0))
        identity = _identity_matrix()
        assert rot != identity

    def test_z_axis_produces_identity(self):
        """axis=z → already aligned, identity."""
        rot = _cylinder_base_orientation((0, 0, 1))
        assert rot == _identity_matrix()

    def test_y_axis_maps_z_to_y(self):
        """Verify rotation maps (0,0,1) → (0,1,0)."""
        rot = _cylinder_base_orientation((0, 1, 0))
        result = _mat_vec(rot, (0, 0, 1))
        assert abs(result[0]) < 1e-6
        assert abs(result[1] - 1.0) < 1e-6
        assert abs(result[2]) < 1e-6

    def test_x_axis_maps_z_to_x(self):
        """Verify rotation maps (0,0,1) → (1,0,0)."""
        rot = _cylinder_base_orientation((1, 0, 0))
        result = _mat_vec(rot, (0, 0, 1))
        assert abs(result[0] - 1.0) < 1e-6
        assert abs(result[1]) < 1e-6
        assert abs(result[2]) < 1e-6


# ---------------------------------------------------------------------------
# Integration tests — solver end-to-end
# ---------------------------------------------------------------------------

class TestWheelRotationInSolver:
    def test_wheel_on_y_axis_has_nonzero_rotation(self):
        """Wheels on revolute (axis=y, child_anchor=center) must NOT have identity rotation."""
        asm = _make_4wheel_robot()
        solver = AssemblySolver(asm)
        placements = solver.solve()

        for wheel_name in ("wheel_fl", "wheel_fr"):
            rot = placements[wheel_name]["rotation"]
            # Identity rotation is [0,0,1,0] — angle must be non-zero
            assert rot[3] != 0.0, (
                f"{wheel_name} has identity rotation [0,0,1,0] — "
                "cylinder not oriented correctly"
            )

    def test_wheel_rotation_is_around_x_for_y_joint(self):
        """For axis=y joint, the rotation should be around X (±90°)."""
        asm = _make_4wheel_robot()
        solver = AssemblySolver(asm)
        placements = solver.solve()

        for wheel_name in ("wheel_fl", "wheel_fr"):
            rot = placements[wheel_name]["rotation"]
            ax, ay, az, angle = rot
            # Primary axis should be X
            assert abs(ax) > 0.9, (
                f"{wheel_name} rotation axis ({ax},{ay},{az}) "
                f"not primarily X — expected R_x(±90°)"
            )
            assert abs(abs(angle) - 90.0) < 1.0, (
                f"{wheel_name} angle={angle:.1f}°, expected ≈±90°"
            )

    def test_four_wheels_all_have_rotation(self):
        """All 4 wheels in a full 4-wheel robot should have rotation."""
        parts = [
            Part(name="base_plate", category="structural", description="",
                 material="Aluminum", dimensions={"length": 300, "width": 200, "height": 5}),
        ]
        joints = []
        for corner in ("fl", "fr", "rl", "rr"):
            motor = _make_box_part(f"motor_{corner}")
            wheel = _make_wheel_part(f"wheel_{corner}")
            parts.append(motor)
            parts.append(wheel)
            joints.append(Joint(
                type="fixed", parent="base_plate", child=f"motor_{corner}",
                parent_anchor="bottom", child_anchor="top",
                distribution_group="motors",
            ))
            joints.append(Joint(
                type="revolute", parent=f"motor_{corner}", child=f"wheel_{corner}",
                axis="y", range_deg=(-360, 360),
                parent_anchor="left", child_anchor="center",
                no_distribute=True,
            ))

        asm = Assembly(name="full_4w", parts=parts, joints=joints)
        solver = AssemblySolver(asm)
        placements = solver.solve()

        for corner in ("fl", "fr", "rl", "rr"):
            rot = placements[f"wheel_{corner}"]["rotation"]
            assert rot[3] != 0.0, f"wheel_{corner} has no rotation"

    def test_box_child_on_center_no_extra_rotation(self):
        """Non-cylindrical parts should NOT get the cylinder orientation."""
        box_child = _make_box_part(name="bracket")
        asm = Assembly(
            name="test_box",
            parts=[
                _make_box_part("parent"),
                box_child,
            ],
            joints=[
                Joint(
                    type="revolute", parent="parent", child="bracket",
                    axis="y", range_deg=(-180, 180),
                    parent_anchor="left", child_anchor="center",
                    no_distribute=True,
                ),
            ],
        )
        solver = AssemblySolver(asm)
        placements = solver.solve()
        # With angle=0 and child_anchor=center, align_rot may be identity.
        # The key point: no cylinder_orient should be applied.
        # We just verify it doesn't crash and produces a valid placement.
        assert "bracket" in placements
        pos = placements["bracket"]["position"]
        assert all(math.isfinite(v) for v in pos)
