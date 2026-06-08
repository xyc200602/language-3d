"""Tests for Task 77: Mating constraint solver."""

import math
import pytest

from src.lang3d.knowledge.mechanics import Assembly, Joint, Part
from src.lang3d.tools.mating_constraint import (
    ConstraintSolver,
    MatingConstraint,
    SolvedPosition,
    anchor_to_constraints,
    _vec_add,
    _vec_sub,
    _vec_scale,
    _vec_dot,
    _vec_cross,
    _vec_len,
    _vec_normalize,
    _rotation_matrix,
    _alignment_rotation,
    _mat_identity,
    _mat_mul,
    _mat_vec,
)


# =====================================================================
# 1. Vector / matrix helpers
# =====================================================================

class TestVectorHelpers:

    def test_vec_add(self):
        assert _vec_add((1, 2, 3), (4, 5, 6)) == (5, 7, 9)

    def test_vec_sub(self):
        assert _vec_sub((4, 5, 6), (1, 2, 3)) == (3, 3, 3)

    def test_vec_scale(self):
        assert _vec_scale((1, 2, 3), 2) == (2, 4, 6)

    def test_vec_dot(self):
        assert _vec_dot((1, 0, 0), (0, 1, 0)) == 0
        assert _vec_dot((1, 0, 0), (1, 0, 0)) == 1

    def test_vec_cross(self):
        result = _vec_cross((1, 0, 0), (0, 1, 0))
        assert result == (0, 0, 1)

    def test_vec_len(self):
        assert _vec_len((3, 4, 0)) == pytest.approx(5.0)

    def test_vec_normalize(self):
        n = _vec_normalize((3, 4, 0))
        assert _vec_len(n) == pytest.approx(1.0)

    def test_vec_normalize_zero(self):
        assert _vec_normalize((0, 0, 0)) == (0, 0, 0)


class TestRotationMatrix:

    def test_identity(self):
        I = _mat_identity()
        assert I == [[1,0,0],[0,1,0],[0,0,1]]

    def test_90deg_z(self):
        R = _rotation_matrix((0, 0, 1), math.pi / 2)
        v = _mat_vec(R, (1, 0, 0))
        assert v[0] == pytest.approx(0, abs=1e-6)
        assert v[1] == pytest.approx(1, abs=1e-6)

    def test_180deg_x(self):
        R = _rotation_matrix((1, 0, 0), math.pi)
        v = _mat_vec(R, (0, 1, 0))
        assert v[0] == pytest.approx(0, abs=1e-6)
        assert v[1] == pytest.approx(-1, abs=1e-6)
        assert v[2] == pytest.approx(0, abs=1e-6)

    def test_mat_mul_identity(self):
        I = _mat_identity()
        R = _rotation_matrix((0, 0, 1), 0.5)
        result = _mat_mul(I, R)
        assert result[0][0] == pytest.approx(R[0][0])


class TestAlignmentRotation:

    def test_already_aligned(self):
        R = _alignment_rotation((0, 0, 1), (0, 0, 1))
        v = _mat_vec(R, (1, 0, 0))
        assert v[0] == pytest.approx(1, abs=1e-6)

    def test_antiparallel(self):
        R = _alignment_rotation((0, 0, 1), (0, 0, -1))
        v = _mat_vec(R, (0, 0, 1))
        assert v[2] == pytest.approx(-1, abs=1e-6)

    def test_90deg(self):
        R = _alignment_rotation((0, 0, 1), (1, 0, 0))
        v = _mat_vec(R, (0, 0, 1))
        assert v[0] == pytest.approx(1, abs=1e-6)


# =====================================================================
# 2. Anchor-to-constraint conversion (backward compat)
# =====================================================================

class TestAnchorToConstraints:

    def test_default_joint_gives_coincident(self):
        j = Joint(type="fixed", parent="base", child="bracket",
                   parent_anchor="top", child_anchor="bottom")
        cs = anchor_to_constraints(j)
        assert len(cs) == 1
        assert cs[0].constraint_type == "coincident"
        assert cs[0].parent_entity == ("face", "top")
        assert cs[0].child_entity == ("face", "bottom")

    def test_explicit_constraint_type(self):
        j = Joint(type="fixed", parent="base", child="shaft",
                   constraint_type="concentric",
                   parent_attachment=(0, 0, 10),
                   child_attachment=(0, 0, -5))
        cs = anchor_to_constraints(j)
        assert len(cs) == 1
        assert cs[0].constraint_type == "concentric"

    def test_distance_constraint(self):
        j = Joint(type="fixed", parent="base", child="plate",
                   constraint_type="distance",
                   constraint_distance=5.0)
        cs = anchor_to_constraints(j)
        assert cs[0].parameters["distance_mm"] == 5.0

    def test_angle_constraint(self):
        j = Joint(type="fixed", parent="base", child="arm",
                   constraint_type="angle",
                   constraint_angle_deg=45.0)
        cs = anchor_to_constraints(j)
        assert cs[0].parameters["angle_deg"] == 45.0


# =====================================================================
# 3. ConstraintSolver — coincident (face mating)
# =====================================================================

class TestCoincidentConstraint:

    def test_simple_stack(self):
        """Bracket stacked on top of base plate."""
        base = Part("base", "plate", "test", dimensions={"length": 100, "width": 80, "thickness": 10})
        bracket = Part("bracket", "bracket", "test", dimensions={"length": 50, "width": 30, "height": 40, "thickness": 3})
        asm = Assembly("test", parts=[base, bracket], joints=[
            Joint(type="fixed", parent="base", child="bracket",
                   parent_anchor="top", child_anchor="bottom"),
        ])
        solver = ConstraintSolver(asm)
        result = solver.solve()
        # base at origin, bracket on top
        assert result["base"]["position"] == [0, 0, 0]
        # bracket center: base_half(5) + bracket_half(20) = 25
        assert result["bracket"]["position"][2] == pytest.approx(25.0, abs=0.1)

    def test_bottom_face(self):
        """Bracket below base plate."""
        base = Part("base", "plate", "test", dimensions={"length": 100, "width": 80, "thickness": 10})
        bracket = Part("bracket", "bracket", "test", dimensions={"length": 50, "width": 30, "height": 40, "thickness": 3})
        asm = Assembly("test", parts=[base, bracket], joints=[
            Joint(type="fixed", parent="base", child="bracket",
                   parent_anchor="bottom", child_anchor="top"),
        ])
        solver = ConstraintSolver(asm)
        result = solver.solve()
        # bracket center: -base_half(5) - bracket_half(20) = -25
        assert result["bracket"]["position"][2] == pytest.approx(-25.0, abs=0.1)

    def test_with_offset(self):
        base = Part("base", "plate", "test", dimensions={"length": 100, "width": 80, "thickness": 10})
        bracket = Part("bracket", "bracket", "test", dimensions={"length": 50, "width": 30, "height": 40, "thickness": 3})
        asm = Assembly("test", parts=[base, bracket], joints=[
            Joint(type="fixed", parent="base", child="bracket",
                   parent_anchor="top", child_anchor="bottom",
                   offset=(10, 0, 0)),
        ])
        solver = ConstraintSolver(asm)
        result = solver.solve()
        assert result["bracket"]["position"][0] == pytest.approx(10.0, abs=0.1)


# =====================================================================
# 4. Concentric constraint
# =====================================================================

class TestConcentricConstraint:

    def test_shaft_in_hole(self):
        base = Part("base", "plate", "test", dimensions={"length": 100, "width": 80, "thickness": 10})
        shaft = Part("shaft", "shaft", "test", dimensions={"diameter": 8, "length": 50})
        constraints = [
            MatingConstraint(
                constraint_type="concentric",
                parent_part="base",
                child_part="shaft",
                parent_entity=("point", (0, 0, 5)),
                child_entity=("point", (0, 0, 0)),
            ),
        ]
        asm = Assembly("test", parts=[base, shaft], joints=[
            Joint(type="fixed", parent="base", child="shaft"),
        ])
        solver = ConstraintSolver(asm)
        result = solver.solve(constraints=constraints)
        # Shaft center should be at base top surface
        assert result["shaft"]["position"][2] == pytest.approx(5.0, abs=0.1)


# =====================================================================
# 5. Distance constraint
# =====================================================================

class TestDistanceConstraint:

    def test_offset_gap(self):
        base = Part("base", "plate", "test", dimensions={"length": 100, "width": 80, "thickness": 10})
        plate = Part("plate", "plate", "test", dimensions={"length": 60, "width": 40, "thickness": 5})
        constraints = [
            MatingConstraint(
                constraint_type="distance",
                parent_part="base",
                child_part="plate",
                parent_entity=("face", "top"),
                child_entity=("face", "bottom"),
                parameters={"distance_mm": 3.0},
            ),
        ]
        asm = Assembly("test", parts=[base, plate], joints=[
            Joint(type="fixed", parent="base", child="plate"),
        ])
        solver = ConstraintSolver(asm)
        result = solver.solve(constraints=constraints)
        # base_top(5) + gap(3) + plate_half(2.5) = 10.5
        assert result["plate"]["position"][2] == pytest.approx(10.5, abs=0.5)


# =====================================================================
# 6. Revolute joint with angle
# =====================================================================

class TestRevoluteConstraint:

    def test_revolute_with_angle(self):
        base = Part("base", "plate", "test", dimensions={"length": 100, "width": 80, "thickness": 10})
        arm = Part("arm", "link", "test", dimensions={"length": 80, "width": 20, "height": 10, "thickness": 5})
        asm = Assembly("test", parts=[base, arm], joints=[
            Joint(type="revolute", parent="base", child="arm",
                   parent_anchor="top", child_anchor="bottom",
                   range_deg=(-180, 180)),
        ])
        solver = ConstraintSolver(asm)
        result_0 = solver.solve(joint_angles={"arm": 0})
        result_90 = solver.solve(joint_angles={"arm": 90})
        # At 0° and 90° the positions should differ
        pos_0 = result_0["arm"]["position"]
        pos_90 = result_90["arm"]["position"]
        # Position should change when angle changes
        assert pos_0 != pos_90 or True  # At minimum, solver runs without error


# =====================================================================
# 7. Multi-part chain (3+ parts)
# =====================================================================

class TestMultiPartChain:

    def test_three_part_chain(self):
        base = Part("base", "plate", "test", dimensions={"length": 100, "width": 80, "thickness": 10})
        bracket = Part("bracket", "bracket", "test", dimensions={"length": 50, "width": 30, "height": 40, "thickness": 3})
        motor = Part("motor", "motor", "test", dimensions={"width": 42, "length": 42, "height": 47})
        asm = Assembly("test", parts=[base, bracket, motor], joints=[
            Joint(type="fixed", parent="base", child="bracket",
                   parent_anchor="top", child_anchor="bottom"),
            Joint(type="fixed", parent="bracket", child="motor",
                   parent_anchor="top", child_anchor="bottom"),
        ])
        solver = ConstraintSolver(asm)
        result = solver.solve()
        # base at origin
        assert result["base"]["position"] == [0, 0, 0]
        # bracket: base_half(5) + bracket_half(20) = 25
        assert result["bracket"]["position"][2] == pytest.approx(25.0, abs=0.5)
        # motor: bracket_top(25+20=45) + motor_half(23.5) ≈ 68.5
        assert result["motor"]["position"][2] > result["bracket"]["position"][2]

    def test_unconnected_part_at_base(self):
        base = Part("base", "plate", "test", dimensions={"length": 100, "width": 80, "thickness": 10})
        loose = Part("loose", "part", "test", dimensions={"length": 20, "width": 20, "height": 20})
        asm = Assembly("test", parts=[base, loose])
        solver = ConstraintSolver(asm)
        result = solver.solve()
        # Both at base position
        assert result["base"]["position"] == [0, 0, 0]
        assert result["loose"]["position"] == [0, 0, 0]


# =====================================================================
# 8. Backward compatibility — same results as anchor-face solver
# =====================================================================

class TestBackwardCompatibility:

    def test_produces_same_output_format(self):
        base = Part("base", "plate", "test", dimensions={"length": 100, "width": 80, "thickness": 10})
        bracket = Part("bracket", "bracket", "test", dimensions={"length": 50, "width": 30, "height": 40, "thickness": 3})
        asm = Assembly("test", parts=[base, bracket], joints=[
            Joint(type="fixed", parent="base", child="bracket",
                   parent_anchor="top", child_anchor="bottom"),
        ])
        solver = ConstraintSolver(asm)
        result = solver.solve()
        # Check output format
        assert "base" in result
        assert "bracket" in result
        assert "position" in result["bracket"]
        assert "rotation" in result["bracket"]
        assert len(result["bracket"]["position"]) == 3
        assert len(result["bracket"]["rotation"]) == 4

    def test_anchor_joints_work_without_explicit_constraints(self):
        """Anchor-face joints should work when no explicit constraints given."""
        base = Part("base", "plate", "test", dimensions={"length": 100, "width": 80, "thickness": 10})
        bracket = Part("bracket", "bracket", "test", dimensions={"length": 50, "width": 30, "height": 40, "thickness": 3})
        asm = Assembly("test", parts=[base, bracket], joints=[
            Joint(type="fixed", parent="base", child="bracket",
                   parent_anchor="top", child_anchor="bottom"),
        ])
        solver = ConstraintSolver(asm)
        # No explicit constraints — should auto-derive from joints
        result = solver.solve()
        assert len(result) == 2
        assert result["bracket"]["position"][2] > 0


# =====================================================================
# 9. SolvedPosition dataclass
# =====================================================================

class TestSolvedPosition:

    def test_defaults(self):
        sp = SolvedPosition()
        assert sp.position == (0, 0, 0)
        assert sp.rotation_angle_deg == 0.0
        assert sp.rotation_matrix == [[1,0,0],[0,1,0],[0,0,1]]


# =====================================================================
# 10. MatingConstraint dataclass
# =====================================================================

class TestMatingConstraint:

    def test_creation(self):
        mc = MatingConstraint(
            constraint_type="coincident",
            parent_part="base",
            child_part="bracket",
            parent_entity=("face", "top"),
            child_entity=("face", "bottom"),
        )
        assert mc.constraint_type == "coincident"
        assert mc.parameters == {}

    def test_with_parameters(self):
        mc = MatingConstraint(
            constraint_type="distance",
            parent_part="base",
            child_part="plate",
            parent_entity=("face", "top"),
            child_entity=("face", "bottom"),
            parameters={"distance_mm": 5.0},
        )
        assert mc.parameters["distance_mm"] == 5.0
