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
    def test_wheel_on_y_axis_has_identity_solver_rotation(self):
        """Wheels on revolute (axis=y) must have IDENTITY rotation from the
        solver, because the cylinder orientation is now baked into the STL
        geometry by part_feature_engine (orient_axis="x" for wheels).

        HISTORICALLY (pre-2026-06) the solver applied R_x(±90°) here as a
        post-processing visual step. That was REMOVED because part_feature_engine
        sets orient_axis, baking the correct orientation into the STL itself —
        applying a second rotation in the solver would DOUBLE-ROTATE the wheel
        (correct X + R_x(90°) → 磨盘/turntable). See
        ``_visual_rotation_for_part`` in assembly_solver.py.

        This test was updated 2026-07-03: the old assertion (rot[3] != 0,
        i.e. non-identity) was testing the removed post-processing behaviour.
        The correct current behaviour is identity solver rotation + STL-baked
        orientation, verified by ``test_wheel_stl_orient_axis`` below.
        """
        asm = _make_4wheel_robot()
        solver = AssemblySolver(asm)
        placements = solver.solve()

        for wheel_name in ("wheel_fl", "wheel_fr"):
            rot = placements[wheel_name]["rotation"]
            # Identity in axis-angle-deg format is [0,0,1,0] (any axis, 0°).
            assert rot[3] == 0.0, (
                f"{wheel_name} solver rotation {rot} is non-identity — "
                "the visual post-processing was removed; rotation is baked "
                "into the STL via orient_axis. A non-zero angle here would "
                "double-rotate the wheel."
            )

    def test_wheel_stl_orient_axis_is_baked(self):
        """part_feature_engine must bake orient_axis='x' into wheel STL ops.

        This is the CURRENT mechanism for wheel orientation (replacing the
        old solver-side post-processing). Verifies the contract documented in
        ``_visual_rotation_for_part``: the STL is built along X for wheels,
        so the solver stays at identity.
        """
        from lang3d.tools.part_feature_engine import generate_ops
        from lang3d.knowledge.mechanics import Part

        wheel = Part(
            name="wheel_fl", category="wheel", description="",
            dimensions={"diameter": 60, "height": 20},
        )
        ops = generate_ops(wheel)
        # The body op should carry orient_axis="x" for a wheel (name starts
        # with "wheel_"). The op type varies (cylinder, cylinder_with_hole,
        # etc.) — match by orient_axis presence.
        cyl_ops = [
            o for o in ops
            if isinstance(o, dict) and "orient_axis" in o
        ]
        assert cyl_ops, f"no op with orient_axis in wheel feature: {ops}"
        assert cyl_ops[0].get("orient_axis") == "x", (
            f"wheel cylinder op must set orient_axis='x' (got "
            f"{cyl_ops[0].get('orient_axis')!r}) — this is what lets the "
            "solver stay at identity rotation without double-rotating"
        )

    def test_four_wheels_all_have_identity_rotation(self):
        """All 4 wheels in a full 4-wheel robot should have identity solver
        rotation (orientation baked into STL, not applied by solver)."""
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
            # Identity (orientation baked into STL, not solver) — see
            # test_wheel_on_y_axis_has_identity_solver_rotation for rationale.
            assert rot[3] == 0.0, (
                f"wheel_{corner} solver rotation {rot} should be identity "
                "(orientation is STL-baked)"
            )

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
