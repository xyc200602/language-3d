"""Tests for ConstraintSolver prototype using py_slvs/SolveSpace."""

import math
import pytest

from lang3d.knowledge.mechanics import Assembly, Joint, Part
from lang3d.tools.constraint_solver import (
    ConstraintSolver,
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
    constraint_solve,
    convert_legacy_joint,
    _part_dimensions,
    _anchor_point,
)


# ---------------------------------------------------------------------------
# Helper to build simple assemblies
# ---------------------------------------------------------------------------

def _box(name, l, w, h, **kw):
    return Part(name, "structural", name, dimensions=dict(length=l, width=w, height=h), **kw)


def _cyl(name, d, h, **kw):
    return Part(name, "structural", name, dimensions=dict(diameter=d, height=h), **kw)


# ---------------------------------------------------------------------------
# Quaternion tests
# ---------------------------------------------------------------------------

class TestQuaternion:
    def test_identity(self):
        w, x, y, z = axis_angle_to_quaternion(0, 0, 1, 0)
        assert abs(w - 1.0) < 1e-6
        assert abs(x) < 1e-6
        assert abs(y) < 1e-6
        assert abs(z) < 1e-6

    def test_90deg_z(self):
        w, x, y, z = axis_angle_to_quaternion(0, 0, 1, 90)
        assert abs(w - math.cos(math.pi / 4)) < 1e-6
        assert abs(z - math.sin(math.pi / 4)) < 1e-6

    def test_roundtrip(self):
        for ax, ay, az, deg in [
            (0, 0, 1, 45), (1, 0, 0, 90), (0, 1, 0, 180),
            (0, 0, 1, 0), (1, 1, 0, 30),
        ]:
            w, x, y, z = axis_angle_to_quaternion(ax, ay, az, deg)
            r_ax, r_deg = quaternion_to_axis_angle(w, x, y, z)
            assert abs(r_deg - deg) < 1e-3, f"Expected {deg}, got {r_deg}"


# ---------------------------------------------------------------------------
# Dimension/anchor tests
# ---------------------------------------------------------------------------

class TestPartGeometry:
    def test_box_dimensions(self):
        p = _box("b", 100, 60, 10)
        dx, dy, dz = _part_dimensions(p)
        assert (dx, dy, dz) == (50, 30, 5)

    def test_cylinder_dimensions(self):
        p = _cyl("c", 40, 20)
        dx, dy, dz = _part_dimensions(p)
        assert (dx, dy, dz) == (20, 20, 10)

    def test_anchor_top(self):
        p = _box("b", 100, 60, 10)
        x, y, z = _anchor_point(p, "top")
        assert (x, y, z) == (0, 0, 5)

    def test_anchor_bottom(self):
        p = _box("b", 100, 60, 10)
        x, y, z = _anchor_point(p, "bottom")
        assert (x, y, z) == (0, 0, -5)

    def test_anchor_left(self):
        p = _box("b", 100, 60, 10)
        x, y, z = _anchor_point(p, "left")
        assert (x, y, z) == (0, -30, 0)


# ---------------------------------------------------------------------------
# Solver basic tests
# ---------------------------------------------------------------------------

class TestConstraintSolverBasic:
    def test_py_slvs_import(self):
        """Verify py_slvs is importable."""
        from py_slvs import slvs
        sys = slvs.System()
        assert sys is not None

    def test_two_parts_fixed(self):
        """Two parts: base_plate (fixed) + standoff on top."""
        parts = [
            _box("base_plate", 100, 60, 5),
            _cyl("standoff", 8, 30),
        ]
        joints = [
            Joint("fixed", "base_plate", "standoff",
                  parent_anchor="top", child_anchor="bottom"),
        ]
        asm = Assembly(name="test", description="test", parts=parts, joints=joints)
        result = ConstraintSolver(asm).solve()

        assert result.success, f"Solver failed: dof={result.dof}, failed={result.failed_constraints}"
        # Standoff should be above base_plate
        base_z = result.parts["base_plate"].position[2]
        standoff_z = result.parts["standoff"].position[2]
        assert standoff_z > base_z, f"Standoff z={standoff_z} should be > base z={base_z}"

    def test_three_parts_stacked(self):
        """Three parts stacked: base → standoff → top_plate."""
        parts = [
            _box("base", 100, 60, 5),
            _cyl("standoff", 8, 30),
            _box("top_plate", 90, 50, 3),
        ]
        joints = [
            Joint("fixed", "base", "standoff",
                  parent_anchor="top", child_anchor="bottom"),
            Joint("fixed", "standoff", "top_plate",
                  parent_anchor="top", child_anchor="bottom"),
        ]
        asm = Assembly(name="test", description="test", parts=parts, joints=joints)
        result = ConstraintSolver(asm).solve()

        assert result.success
        # Verify ordering: base_z < standoff_z < top_z
        z_order = {
            "base": result.parts["base"].position[2],
            "standoff": result.parts["standoff"].position[2],
            "top_plate": result.parts["top_plate"].position[2],
        }
        assert z_order["standoff"] > z_order["base"], \
            f"standoff z={z_order['standoff']} should be > base z={z_order['base']}"
        assert z_order["top_plate"] > z_order["standoff"], \
            f"top_plate z={z_order['top_plate']} should be > standoff z={z_order['standoff']}"

    def test_four_children_grid(self):
        """4 standoffs on base_plate top face (grid distribution)."""
        parts = [
            _box("base", 100, 60, 5),
            _cyl("s1", 8, 30),
            _cyl("s2", 8, 30),
            _cyl("s3", 8, 30),
            _cyl("s4", 8, 30),
        ]
        joints = [
            Joint("fixed", "base", "s1", parent_anchor="top", child_anchor="bottom"),
            Joint("fixed", "base", "s2", parent_anchor="top", child_anchor="bottom"),
            Joint("fixed", "base", "s3", parent_anchor="top", child_anchor="bottom"),
            Joint("fixed", "base", "s4", parent_anchor="top", child_anchor="bottom"),
        ]
        asm = Assembly(name="test", description="test", parts=parts, joints=joints)
        result = ConstraintSolver(asm).solve()

        assert result.success, f"Solver failed: {result.failed_constraints}"
        # All standoffs should be above base
        for name in ["s1", "s2", "s3", "s4"]:
            sz = result.parts[name].position[2]
            assert sz > 0, f"{name} z={sz} should be positive"

        # Standoffs should be distributed (not all at same x,y)
        positions = [tuple(result.parts[f"s{i}"].position[:2]) for i in range(1, 5)]
        unique_xy = set((round(x, 1), round(y, 1)) for x, y in positions)
        assert len(unique_xy) > 1, f"Standoffs should be distributed, got positions: {positions}"


# ---------------------------------------------------------------------------
# Integration test: constraint_solve API compatibility
# ---------------------------------------------------------------------------

class TestConstraintSolveAPI:
    def test_returns_compatible_format(self):
        """Verify constraint_solve returns same format as assembly_solve."""
        parts = [
            _box("base", 100, 60, 5),
            _cyl("standoff", 8, 30),
        ]
        joints = [
            Joint("fixed", "base", "standoff",
                  parent_anchor="top", child_anchor="bottom"),
        ]
        asm = Assembly(name="test", description="test", parts=parts, joints=joints)
        result = constraint_solve(asm)

        assert "base" in result
        assert "standoff" in result
        assert "position" in result["standoff"]
        assert "rotation" in result["standoff"]
        assert len(result["standoff"]["position"]) == 3
        assert len(result["standoff"]["rotation"]) == 4

    def test_with_joint_angles(self):
        """Verify solver accepts joint_angles parameter."""
        parts = [
            _box("base", 100, 60, 5),
            _cyl("arm", 10, 50),
        ]
        joints = [
            Joint("revolute", "base", "arm",
                  axis="z", range_deg=(-180, 180),
                  parent_anchor="top", child_anchor="bottom"),
        ]
        asm = Assembly(name="test", description="test", parts=parts, joints=joints)
        result = constraint_solve(asm, joint_angles={"arm": 45})

        assert "arm" in result
        assert result["arm"]["position"][2] > 0, "Arm should be above base"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# Constraint converter tests (Task 63)
# ---------------------------------------------------------------------------

class TestConstraintConverter:
    """Test convert_legacy_joint function."""

    def test_legacy_joint_uses_anchor_point(self):
        """Old-style joint converts to anchor-point-based attachment."""
        parent = _box("base", 100, 60, 10)
        child = _cyl("standoff", 8, 30)
        joint = Joint("fixed", "base", "standoff",
                      parent_anchor="top", child_anchor="bottom")

        conv = convert_legacy_joint(joint, parent, child)

        # parent_attachment should equal _anchor_point(parent, "top")
        assert conv["parent_attachment"] == _anchor_point(parent, "top")
        # child_attachment should equal _anchor_point(child, "bottom")
        assert conv["child_attachment"] == _anchor_point(child, "bottom")
        # Normals should come from ANCHOR_NORMALS
        assert conv["parent_normal"] == (0, 0, 1)   # top normal
        assert conv["child_normal"] == (0, 0, -1)    # bottom normal

    def test_constraint_type_mapping(self):
        """Joint type maps to correct constraint type."""
        for jtype, expected in [
            ("fixed", "coincident"),
            ("revolute", "concentric"),
            ("prismatic", "parallel"),
        ]:
            parent = _box("p", 10, 10, 10)
            child = _box("c", 10, 10, 10)
            joint = Joint(jtype, "p", "c")
            conv = convert_legacy_joint(joint, parent, child)
            assert conv["constraint_type"] == expected, f"{jtype} -> {conv['constraint_type']}"

    def test_new_fields_take_priority(self):
        """If joint has parent_attachment, it overrides anchor-based computation."""
        parent = _box("p", 100, 60, 10)
        child = _cyl("c", 8, 30)
        joint = Joint(
            "fixed", "p", "c",
            parent_anchor="top", child_anchor="bottom",
            parent_attachment=(50, 30, 5),
            child_attachment=(0, 0, -15),
            parent_normal=(1, 0, 0),
            child_normal=(0, 1, 0),
            constraint_type="distance",
            constraint_distance=10.0,
        )

        conv = convert_legacy_joint(joint, parent, child)

        assert conv["parent_attachment"] == (50, 30, 5)
        assert conv["child_attachment"] == (0, 0, -15)
        assert conv["parent_normal"] == (1, 0, 0)
        assert conv["child_normal"] == (0, 1, 0)
        assert conv["constraint_type"] == "distance"
        assert conv["constraint_distance"] == 10.0

    def test_constraint_distance_and_angle_defaults(self):
        """Default distance and angle should be 0."""
        parent = _box("p", 10, 10, 10)
        child = _box("c", 10, 10, 10)
        joint = Joint("fixed", "p", "c")
        conv = convert_legacy_joint(joint, parent, child)
        assert conv["constraint_distance"] == 0.0
        assert conv["constraint_angle_deg"] == 0.0
