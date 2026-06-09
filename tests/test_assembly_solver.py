"""Tests for assembly constraint solver.

Tests cover:
- Joint dataclass with new constraint fields
- AssemblySolver chain traversal
- Anchor offset computation
- Rotation matrix math
- Robotic arm full-chain solving
- Tool registration and execution
- Assembly JSON parsing
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from lang3d.knowledge.mechanics import (
    Assembly,
    ConnectionMethod,
    Joint,
    Part,
    OPEN_MANIPULATOR_X_ASSEMBLY,
    OPEN_MANIPULATOR_X_PARTS,
    ROBOTIC_ARM_ASSEMBLY,
    ROBOTIC_ARM_PARTS,
)
from lang3d.tools.assembly_solver import (
    ANCHOR_DIRECTIONS,
    AssemblySolver,
    AssemblySolveTool,
    _anchor_offset_for_part,
    _half_extent,
    _identity_matrix,
    _mat_mul,
    _mat_vec,
    _resolve_assembly,
    _rot_mat_to_axis_angle_deg,
    _rotation_matrix_axis_angle,
    _vec_add,
    connection_to_constraints,
    register_assembly_solver_tools,
)
from lang3d.tools.base import ToolRegistry


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_parts() -> list[Part]:
    """Two simple parts for testing."""
    return [
        Part(name="base", category="structural", description="Base block",
             dimensions={"width": 100, "depth": 100, "height": 20}),
        Part(name="pillar", category="structural", description="Vertical pillar",
             dimensions={"width": 30, "depth": 30, "height": 150}),
    ]


@pytest.fixture
def simple_assembly(simple_parts) -> Assembly:
    """Simple 2-part assembly: base + pillar on top."""
    return Assembly(
        name="Simple Stack",
        parts=simple_parts,
        joints=[
            Joint("fixed", "base", "pillar", description="Pillar on base",
                  parent_anchor="top", child_anchor="bottom"),
        ],
    )


@pytest.fixture
def three_part_assembly() -> Assembly:
    """Three-part chain: base → link1 → link2."""
    return Assembly(
        name="Three Part Chain",
        parts=[
            Part(name="part_a", category="structural", description="Base",
                 dimensions={"width": 50, "height": 20}),
            Part(name="part_b", category="structural", description="Link 1",
                 dimensions={"width": 30, "height": 100}),
            Part(name="part_c", category="structural", description="Link 2",
                 dimensions={"width": 20, "height": 80}),
        ],
        joints=[
            Joint("fixed", "part_a", "part_b",
                  parent_anchor="top", child_anchor="bottom"),
            Joint("fixed", "part_b", "part_c",
                  parent_anchor="top", child_anchor="bottom"),
        ],
    )


# ============================================================================
# Test: Joint dataclass new fields
# ============================================================================

class TestJointFields:
    """Test that Joint dataclass has new constraint fields."""

    def test_default_anchor_values(self):
        j = Joint("fixed", "a", "b")
        assert j.parent_anchor == "top"
        assert j.child_anchor == "bottom"
        assert j.offset == (0, 0, 0)

    def test_custom_anchors(self):
        j = Joint("revolute", "a", "b", parent_anchor="left", child_anchor="right")
        assert j.parent_anchor == "left"
        assert j.child_anchor == "right"

    def test_offset(self):
        j = Joint("fixed", "a", "b", offset=(1.5, -2.0, 3.0))
        assert j.offset == (1.5, -2.0, 3.0)

    def test_robotic_arm_joints_have_anchors(self):
        for j in ROBOTIC_ARM_ASSEMBLY.joints:
            assert j.parent_anchor in ANCHOR_DIRECTIONS, f"Joint '{j.description}' has invalid parent_anchor: {j.parent_anchor}"
            assert j.child_anchor in ANCHOR_DIRECTIONS, f"Joint '{j.description}' has invalid child_anchor: {j.child_anchor}"


# ============================================================================
# Test: Anchor offset computation
# ============================================================================

class TestAnchorOffsets:
    """Test anchor offset calculation from part dimensions."""

    def test_top_anchor_uses_height(self):
        part = Part(name="test", category="test", description="",
                    dimensions={"height": 100})
        offset = _anchor_offset_for_part(part, "top")
        assert offset == (0, 0, 50)  # height/2 upward

    def test_bottom_anchor_negative_z(self):
        part = Part(name="test", category="test", description="",
                    dimensions={"height": 40})
        offset = _anchor_offset_for_part(part, "bottom")
        assert offset == (0, 0, -20)

    def test_left_anchor(self):
        part = Part(name="test", category="test", description="",
                    dimensions={"width": 60})
        offset = _anchor_offset_for_part(part, "left")
        assert offset == (-30, 0, 0)

    def test_right_anchor(self):
        part = Part(name="test", category="test", description="",
                    dimensions={"width": 60})
        offset = _anchor_offset_for_part(part, "right")
        assert offset == (30, 0, 0)

    def test_no_dimensions(self):
        part = Part(name="test", category="test", description="")
        offset = _anchor_offset_for_part(part, "top")
        assert offset == (0, 0, 0)

    def test_half_extent_fallback(self):
        part = Part(name="test", category="test", description="",
                    dimensions={"length": 200})
        # "front" uses "length" key → half = 100
        h = _half_extent(part, "front")
        assert h == 100.0


# ============================================================================
# Test: Rotation math
# ============================================================================

class TestRotationMath:
    """Test rotation matrix helpers."""

    def test_identity_matrix(self):
        m = _identity_matrix()
        for i in range(3):
            for j in range(3):
                if i == j:
                    assert m[i][j] == pytest.approx(1.0)
                else:
                    assert m[i][j] == pytest.approx(0.0)

    def test_rotation_90_around_z(self):
        m = _rotation_matrix_axis_angle((0, 0, 1), math.pi / 2)
        # Should map (1,0,0) → (0,1,0)
        result = _mat_vec(m, (1, 0, 0))
        assert result[0] == pytest.approx(0, abs=1e-10)
        assert result[1] == pytest.approx(1, abs=1e-10)
        assert result[2] == pytest.approx(0, abs=1e-10)

    def test_rotation_180_around_z(self):
        m = _rotation_matrix_axis_angle((0, 0, 1), math.pi)
        result = _mat_vec(m, (1, 0, 0))
        assert result[0] == pytest.approx(-1, abs=1e-10)
        assert result[1] == pytest.approx(0, abs=1e-10)

    def test_rotation_0_is_identity(self):
        m = _rotation_matrix_axis_angle((0, 0, 1), 0)
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert m[i][j] == pytest.approx(expected, abs=1e-10)

    def test_mat_mul_identity(self):
        m = _rotation_matrix_axis_angle((0, 1, 0), 0.5)
        i = _identity_matrix()
        result = _mat_mul(m, i)
        for r in range(3):
            for c in range(3):
                assert result[r][c] == pytest.approx(m[r][c], abs=1e-10)

    def test_vec_add(self):
        assert _vec_add((1, 2, 3), (4, 5, 6)) == (5, 7, 9)

    def test_axis_angle_roundtrip(self):
        """Convert rotation matrix to axis-angle and back."""
        axis = (0, 0, 1)
        angle = math.radians(45)
        m = _rotation_matrix_axis_angle(axis, angle)
        ax, ay, az, deg = _rot_mat_to_axis_angle_deg(m)
        assert deg == pytest.approx(45, abs=0.1)


# ============================================================================
# Test: Simple solver
# ============================================================================

class TestSimpleSolver:
    """Test basic solver functionality."""

    def test_two_part_stack(self, simple_assembly):
        solver = AssemblySolver(simple_assembly)
        placements = solver.solve()

        assert "base" in placements
        assert "pillar" in placements

        # Base at origin
        base_pos = placements["base"]["position"]
        assert base_pos == [0, 0, 0]

        # Pillar should be above base (z > 0)
        pillar_pos = placements["pillar"]["position"]
        assert pillar_pos[2] > 0, f"Pillar z={pillar_pos[2]} should be > 0"

    def test_three_part_chain(self, three_part_assembly):
        solver = AssemblySolver(three_part_assembly)
        placements = solver.solve()

        # Part A at origin
        assert placements["part_a"]["position"][2] == pytest.approx(0, abs=0.01)

        # Part B above part A
        assert placements["part_b"]["position"][2] > placements["part_a"]["position"][2]

        # Part C above part B
        assert placements["part_c"]["position"][2] > placements["part_b"]["position"][2]

    def test_base_position_offset(self, simple_assembly):
        solver = AssemblySolver(simple_assembly)
        placements = solver.solve(base_position=(100, 200, 0))

        base_pos = placements["base"]["position"]
        assert base_pos[0] == pytest.approx(100, abs=0.01)
        assert base_pos[1] == pytest.approx(200, abs=0.01)

    def test_all_parts_placed(self, three_part_assembly):
        solver = AssemblySolver(three_part_assembly)
        placements = solver.solve()
        assert len(placements) == 3


# ============================================================================
# Test: Revolute joint
# ============================================================================

class TestRevoluteJoint:
    """Test revolute joint angle effects."""

    def test_zero_angle_same_as_fixed(self):
        parts = [
            Part(name="base", category="s", description="",
                 dimensions={"height": 20}),
            Part(name="arm", category="s", description="",
                 dimensions={"height": 100}),
        ]
        assembly = Assembly(
            name="test",
            parts=parts,
            joints=[
                Joint("revolute", "base", "arm",
                      parent_anchor="top", child_anchor="bottom"),
            ],
        )

        solver = AssemblySolver(assembly)
        p0 = solver.solve(joint_angles={"arm": 0})
        pf = AssemblySolver(Assembly(
            name="test",
            parts=parts,
            joints=[Joint("fixed", "base", "arm",
                          parent_anchor="top", child_anchor="bottom")],
        )).solve()

        assert p0["arm"]["position"] == pytest.approx(pf["arm"]["position"], abs=0.01)

    def test_nonzero_angle_changes_rotation(self):
        parts = [
            Part(name="base", category="s", description="",
                 dimensions={"height": 20}),
            Part(name="arm", category="s", description="",
                 dimensions={"height": 100}),
        ]
        assembly = Assembly(
            name="test",
            parts=parts,
            joints=[
                Joint("revolute", "base", "arm",
                      parent_anchor="top", child_anchor="bottom"),
            ],
        )

        solver = AssemblySolver(assembly)
        p0 = solver.solve(joint_angles={"arm": 0})
        p90 = solver.solve(joint_angles={"arm": 90})

        # Rotation should differ
        assert p0["arm"]["rotation"] != p90["arm"]["rotation"]


# ============================================================================
# Test: Offset
# ============================================================================

class TestJointOffset:
    """Test joint offset parameter."""

    def test_offset_applied(self):
        parts = [
            Part(name="a", category="s", description="", dimensions={"height": 20}),
            Part(name="b", category="s", description="", dimensions={"height": 20}),
        ]
        assembly_no_offset = Assembly(
            name="no_offset",
            parts=parts,
            joints=[Joint("fixed", "a", "b",
                          parent_anchor="top", child_anchor="bottom",
                          offset=(0, 0, 0))],
        )
        assembly_with_offset = Assembly(
            name="with_offset",
            parts=parts,
            joints=[Joint("fixed", "a", "b",
                          parent_anchor="top", child_anchor="bottom",
                          offset=(10, 0, 0))],
        )

        p_no = AssemblySolver(assembly_no_offset).solve()
        p_yes = AssemblySolver(assembly_with_offset).solve()

        # X position should differ by ~10mm
        diff_x = abs(p_yes["b"]["position"][0] - p_no["b"]["position"][0])
        assert diff_x == pytest.approx(10, abs=0.1)


# ============================================================================
# Test: Robotic arm full chain
# ============================================================================

class TestRoboticArmChain:
    """Test solver on the actual robotic arm assembly."""

    def test_all_parts_placed(self):
        solver = AssemblySolver(ROBOTIC_ARM_ASSEMBLY)
        placements = solver.solve()
        assert len(placements) == len(ROBOTIC_ARM_PARTS)

    def test_chain_is_ascending_z(self):
        """All parts in the chain should have increasing Z."""
        solver = AssemblySolver(ROBOTIC_ARM_ASSEMBLY)
        placements = solver.solve()

        chain = ["base_plate", "base_joint_housing", "shoulder_link",
                 "elbow_joint", "forearm_link", "wrist_joint",
                 "end_effector_mount"]

        prev_z = -1e9
        for name in chain:
            z = placements[name]["position"][2]
            assert z >= prev_z, f"{name} z={z} should be >= prev {prev_z}"
            prev_z = z

    def test_joint_angles_propagate_rotation(self):
        """Rotating a joint should propagate rotation to all downstream parts."""
        solver = AssemblySolver(ROBOTIC_ARM_ASSEMBLY)

        p_home = solver.solve(joint_angles={"base_joint_housing": 0})
        p_rotated = solver.solve(joint_angles={"base_joint_housing": 90})

        # base_joint_housing rotation should propagate to all children
        assert p_home["base_joint_housing"]["rotation"][3] == pytest.approx(0, abs=0.1)
        assert p_rotated["base_joint_housing"]["rotation"][3] == pytest.approx(90, abs=0.1)

        # Children should inherit the rotation
        assert p_rotated["shoulder_link"]["rotation"][3] == pytest.approx(90, abs=0.1)
        assert p_rotated["end_effector_mount"]["rotation"][3] == pytest.approx(90, abs=0.1)

        # Home should have zero rotation
        assert p_home["shoulder_link"]["rotation"][3] == pytest.approx(0, abs=0.1)

    def test_offset_creates_lateral_movement_on_rotation(self):
        """With a lateral offset, rotation around Z should move parts."""
        parts = [
            Part(name="base", category="s", description="",
                 dimensions={"height": 20}),
            Part(name="arm", category="s", description="",
                 dimensions={"height": 100}),
        ]
        assembly = Assembly(
            name="test_lateral",
            parts=parts,
            joints=[
                Joint("revolute", "base", "arm",
                      parent_anchor="top", child_anchor="bottom",
                      offset=(50, 0, 0)),  # 50mm lateral offset
            ],
        )
        solver = AssemblySolver(assembly)

        p0 = solver.solve(joint_angles={"arm": 0})
        p90 = solver.solve(joint_angles={"arm": 90})

        # With 50mm offset and 90deg rotation, the arm should move in XY plane
        pos_0 = p0["arm"]["position"]
        pos_90 = p90["arm"]["position"]

        # At 0 degrees, arm should be offset in X by ~50mm
        assert pos_0[0] == pytest.approx(50, abs=1)

        # At 90 degrees, that offset should move to Y
        assert pos_90[1] != pytest.approx(0, abs=1)

    def test_get_joint_chain(self):
        solver = AssemblySolver(ROBOTIC_ARM_ASSEMBLY)
        chain = solver.get_joint_chain()
        assert len(chain) == len(ROBOTIC_ARM_ASSEMBLY.joints)
        assert chain[0]["parent"] == "base_plate"
        assert chain[-1]["child"] == "end_effector_mount"


# ============================================================================
# Test: Assembly JSON parsing
# ============================================================================

class TestAssemblyParsing:
    """Test parsing assembly from JSON."""

    def test_parse_minimal_json(self):
        json_str = json.dumps({
            "name": "Test",
            "parts": [
                {"name": "a", "dimensions": {"height": 10}},
                {"name": "b", "dimensions": {"height": 20}},
            ],
            "joints": [
                {"parent": "a", "child": "b", "type": "fixed"},
            ],
        })
        from lang3d.tools.assembly_solver import _parse_assembly_json
        asm = _parse_assembly_json(json_str)
        assert asm.name == "Test"
        assert len(asm.parts) == 2
        assert len(asm.joints) == 1
        assert asm.joints[0].parent_anchor == "top"  # default
        assert asm.joints[0].child_anchor == "bottom"  # default

    def test_parse_with_anchors(self):
        json_str = json.dumps({
            "name": "Test",
            "parts": [
                {"name": "a", "dimensions": {"height": 10}},
                {"name": "b", "dimensions": {"height": 20}},
            ],
            "joints": [
                {
                    "parent": "a", "child": "b", "type": "revolute",
                    "parent_anchor": "left", "child_anchor": "right",
                    "offset": [5, 0, 0],
                },
            ],
        })
        from lang3d.tools.assembly_solver import _parse_assembly_json
        asm = _parse_assembly_json(json_str)
        j = asm.joints[0]
        assert j.parent_anchor == "left"
        assert j.child_anchor == "right"
        assert j.offset == (5, 0, 0)

    def test_resolve_builtin(self):
        asm = _resolve_assembly("robotic_arm", "")
        assert asm is not None
        assert asm.name == "3-DOF Robotic Arm"

    def test_resolve_unknown_returns_none(self):
        asm = _resolve_assembly("nonexistent_robot_xyz", "")
        assert asm is None


# ============================================================================
# Test: Tool registration and execution
# ============================================================================

class TestAssemblySolveTool:
    """Test the assembly_solve tool."""

    def test_tool_registration(self):
        registry = ToolRegistry()
        register_assembly_solver_tools(registry)
        assert "assembly_solve" in registry.list_tools()

    def test_tool_definition(self):
        tool = AssemblySolveTool()
        defn = tool.get_definition()
        assert defn.name == "assembly_solve"
        assert "assembly_name" in defn.parameters["properties"]
        assert "joint_angles" in defn.parameters["properties"]

    def test_tool_execute_builtin(self):
        tool = AssemblySolveTool()
        result = tool.execute(assembly_name="robotic_arm")
        assert "Assembly Solver" in result
        assert "base_plate" in result
        assert "end_effector_mount" in result

    def test_tool_execute_with_angles(self):
        tool = AssemblySolveTool()
        result = tool.execute(
            assembly_name="robotic_arm",
            joint_angles={"shoulder_link": 45},
        )
        assert "Assembly Solver" in result

    def test_tool_execute_unknown_assembly(self):
        tool = AssemblySolveTool()
        result = tool.execute(assembly_name="nonexistent")
        assert "错误" in result

    def test_tool_execute_custom_json(self):
        json_str = json.dumps({
            "name": "Custom",
            "parts": [
                {"name": "a", "dimensions": {"height": 50}},
                {"name": "b", "dimensions": {"height": 30}},
            ],
            "joints": [
                {"parent": "a", "child": "b", "type": "fixed",
                 "parent_anchor": "top", "child_anchor": "bottom"},
            ],
        })
        tool = AssemblySolveTool()
        result = tool.execute(assembly_json=json_str)
        assert "Custom" in result


# ============================================================================
# Test: Part_assemble integration with solver
# ============================================================================

class TestPartAssembleIntegration:
    """Test that part_assemble can use assembly_definition for auto-positioning."""

    def test_apply_solver_positions(self):
        """Test _apply_solver_positions fills in positions."""
        from lang3d.tools.part_library import PartAssembleTool
        tool = PartAssembleTool()

        parts = [
            {"file": "fake_a.stl", "name": "base_plate"},
            {"file": "fake_b.stl", "name": "base_joint_housing"},
        ]

        result = tool._apply_solver_positions(parts, "robotic_arm", None)

        # base_plate should have position set
        assert result[0].get("position") is not None
        assert len(result[0]["position"]) == 3

        # base_joint_housing should have position set and z > 0
        assert result[1].get("position") is not None
        assert result[1]["position"][2] >= result[0]["position"][2]


# ============================================================================
# Test: Edge cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_assembly(self):
        asm = Assembly(name="empty", parts=[], joints=[])
        solver = AssemblySolver(asm)
        placements = solver.solve()
        assert placements == {}

    def test_single_part_no_joints(self):
        asm = Assembly(
            name="single",
            parts=[Part(name="only", category="s", description="",
                        dimensions={"height": 10})],
            joints=[],
        )
        solver = AssemblySolver(asm)
        placements = solver.solve()
        assert "only" in placements
        assert placements["only"]["position"] == [0, 0, 0]

    def test_circular_joint_handled(self):
        """If joints form a cycle, visited set prevents infinite loop."""
        asm = Assembly(
            name="cycle",
            parts=[
                Part(name="a", category="s", description="", dimensions={"height": 10}),
                Part(name="b", category="s", description="", dimensions={"height": 10}),
            ],
            joints=[
                Joint("fixed", "a", "b"),
                Joint("fixed", "b", "a"),
            ],
        )
        solver = AssemblySolver(asm)
        placements = solver.solve()
        assert len(placements) == 2  # No infinite loop

    def test_unknown_anchor_defaults_to_top(self):
        """Unknown anchor name should still produce a result."""
        part = Part(name="test", category="s", description="",
                    dimensions={"height": 20})
        offset = _anchor_offset_for_part(part, "custom_unknown")
        # Falls through to (0,0,0) since "custom_unknown" not in ANCHOR_DIM_KEYS
        assert offset == (0, 0, 0)


# ============================================================================
# Test: Joint new fields (Task 63)
# ============================================================================

class TestJointNewFields:
    """Test Joint dataclass Task 63 constraint model fields."""

    def test_default_new_fields_are_none_or_empty(self):
        j = Joint("fixed", "a", "b")
        assert j.parent_attachment is None
        assert j.child_attachment is None
        assert j.parent_normal is None
        assert j.child_normal is None
        assert j.constraint_type == ""
        assert j.constraint_distance == 0.0
        assert j.constraint_angle_deg == 0.0

    def test_custom_constraint_fields(self):
        j = Joint(
            "fixed", "a", "b",
            parent_attachment=(10, 20, 30),
            child_attachment=(5, 0, -10),
            parent_normal=(0, 0, 1),
            child_normal=(0, 0, -1),
            constraint_type="distance",
            constraint_distance=5.0,
            constraint_angle_deg=45.0,
        )
        assert j.parent_attachment == (10, 20, 30)
        assert j.child_attachment == (5, 0, -10)
        assert j.parent_normal == (0, 0, 1)
        assert j.child_normal == (0, 0, -1)
        assert j.constraint_type == "distance"
        assert j.constraint_distance == 5.0
        assert j.constraint_angle_deg == 45.0

    def test_backward_compatible_with_existing_joints(self):
        """Old-style Joint construction still works identically."""
        j1 = Joint("revolute", "base", "arm", (-90, 90), "elbow",
                   parent_anchor="top", child_anchor="bottom", axis="z")
        assert j1.parent_anchor == "top"
        assert j1.child_anchor == "bottom"
        assert j1.axis == "z"
        # New fields have defaults
        assert j1.parent_attachment is None
        assert j1.constraint_type == ""


# ============================================================================
# Test: Connection method → constraint derivation
# ============================================================================

class TestConnectionToConstraints:
    """Test connection method to constraint derivation."""

    def test_bolted_produces_face_contact_and_alignment(self):
        """Bolted connection → coincident constraint."""
        j = Joint("fixed", "a", "b",
                  parent_anchor="top", child_anchor="bottom",
                  connection=ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=4))
        constraints = connection_to_constraints(j)
        assert len(constraints) >= 1
        assert constraints[0]["type"] == "coincident"
        assert constraints[0]["parent_anchor"] == "top"
        assert constraints[0]["child_anchor"] == "bottom"

    def test_bolted_with_attachment_produces_concentric(self):
        """Bolted with attachment points → coincident + concentric."""
        j = Joint("fixed", "a", "b",
                  parent_anchor="top", child_anchor="bottom",
                  parent_attachment=(10, 0, 0),
                  child_attachment=(5, 0, 0),
                  connection=ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=4))
        constraints = connection_to_constraints(j)
        assert len(constraints) == 2
        assert constraints[0]["type"] == "coincident"
        assert constraints[1]["type"] == "concentric"

    def test_press_fit_produces_concentric(self):
        """Press-fit → concentric + distance(0)."""
        j = Joint("fixed", "a", "b",
                  parent_anchor="top", child_anchor="bottom",
                  connection=ConnectionMethod(type="press_fit", interference_mm=0.05))
        constraints = connection_to_constraints(j)
        assert len(constraints) == 2
        types = [c["type"] for c in constraints]
        assert "concentric" in types
        assert "distance" in types

    def test_welded_produces_coincident(self):
        """Welded → coincident only."""
        j = Joint("fixed", "a", "b",
                  parent_anchor="top", child_anchor="bottom",
                  connection=ConnectionMethod(type="welded", weld_type="fillet"))
        constraints = connection_to_constraints(j)
        assert len(constraints) == 1
        assert constraints[0]["type"] == "coincident"

    def test_no_connection_uses_anchor_fallback(self):
        """No connection → legacy coincident."""
        j = Joint("fixed", "a", "b",
                  parent_anchor="left", child_anchor="right")
        constraints = connection_to_constraints(j)
        assert len(constraints) == 1
        assert constraints[0]["type"] == "coincident"
        assert constraints[0]["parent_anchor"] == "left"
        assert constraints[0]["child_anchor"] == "right"

    def test_explicit_constraint_type_overrides_connection(self):
        """Explicit constraint_type takes priority over connection."""
        j = Joint("fixed", "a", "b",
                  parent_anchor="top", child_anchor="bottom",
                  constraint_type="distance", constraint_distance=5.0,
                  connection=ConnectionMethod(type="bolted", bolt_size="M3", bolt_count=4))
        constraints = connection_to_constraints(j)
        assert len(constraints) == 1
        assert constraints[0]["type"] == "distance"
        assert constraints[0]["distance"] == 5.0

    def test_snap_fit_produces_coincident(self):
        """Snap-fit → coincident."""
        j = Joint("fixed", "a", "b",
                  parent_anchor="top", child_anchor="bottom",
                  connection=ConnectionMethod(type="snap_fit", snap_count=2))
        constraints = connection_to_constraints(j)
        assert len(constraints) == 1
        assert constraints[0]["type"] == "coincident"


# ============================================================================
# Test: Multi-parent topology handling
# ============================================================================

class TestMultiParentTopology:
    """Test multi-parent part handling in AssemblySolver."""

    def test_part_with_two_parents_gets_averaged_position(self):
        """Part with two parents should get averaged position."""
        parts = [
            Part(name="root", category="s", description="",
                 dimensions={"length": 100, "width": 100, "height": 10}),
            Part(name="support_l", category="s", description="",
                 dimensions={"length": 10, "width": 10, "height": 50}),
            Part(name="support_r", category="s", description="",
                 dimensions={"length": 10, "width": 10, "height": 50}),
            Part(name="plate", category="s", description="",
                 dimensions={"length": 80, "width": 80, "height": 5}),
        ]
        assembly = Assembly(
            name="Multi-Parent Test",
            parts=parts,
            joints=[
                Joint("fixed", "root", "support_l",
                      parent_anchor="top", child_anchor="bottom",
                      offset=(-40, 0, 0)),
                Joint("fixed", "root", "support_r",
                      parent_anchor="top", child_anchor="bottom",
                      offset=(40, 0, 0)),
                # Plate connects to both supports (multi-parent)
                Joint("fixed", "support_l", "plate",
                      parent_anchor="top", child_anchor="bottom",
                      no_distribute=True),
                Joint("fixed", "support_r", "plate",
                      parent_anchor="top", child_anchor="bottom",
                      no_distribute=True),
            ],
        )

        solver = AssemblySolver(assembly)
        placements = solver.solve()

        # Plate should be centered between the two supports
        plate_x = placements["plate"]["position"][0]
        # The plate should be closer to x=0 than to either support's position
        support_l_x = placements["support_l"]["position"][0]
        support_r_x = placements["support_r"]["position"][0]
        # Averaged x should be between the two supports
        avg_x = (support_l_x + support_r_x) / 2
        assert plate_x == pytest.approx(avg_x, abs=1.0), \
            f"plate_x={plate_x} should be near avg_x={avg_x}"

    def test_single_parent_unchanged(self):
        """Single-parent parts should work the same as before."""
        parts = [
            Part(name="base", category="s", description="",
                 dimensions={"height": 20}),
            Part(name="pillar", category="s", description="",
                 dimensions={"height": 100}),
        ]
        assembly = Assembly(
            name="Single Parent",
            parts=parts,
            joints=[
                Joint("fixed", "base", "pillar",
                      parent_anchor="top", child_anchor="bottom"),
            ],
        )

        solver = AssemblySolver(assembly)
        placements = solver.solve()

        assert "base" in placements
        assert "pillar" in placements
        # Pillar should be above base
        assert placements["pillar"]["position"][2] > placements["base"]["position"][2]

    def test_four_parent_averaging(self):
        """Part with four parents (like top_plate on standoffs) gets centered."""
        parts = [
            Part(name="root", category="s", description="",
                 dimensions={"length": 200, "width": 200, "height": 5}),
            Part(name="leg_fl", category="s", description="",
                 dimensions={"diameter": 10, "height": 50}),
            Part(name="leg_fr", category="s", description="",
                 dimensions={"diameter": 10, "height": 50}),
            Part(name="leg_bl", category="s", description="",
                 dimensions={"diameter": 10, "height": 50}),
            Part(name="leg_br", category="s", description="",
                 dimensions={"diameter": 10, "height": 50}),
            Part(name="shelf", category="s", description="",
                 dimensions={"length": 180, "width": 180, "height": 3}),
        ]
        assembly = Assembly(
            name="Four Parent Shelf",
            parts=parts,
            joints=[
                Joint("fixed", "root", "leg_fl",
                      parent_anchor="top", child_anchor="bottom",
                      distribution_group="legs", offset=(-90, 90, 0)),
                Joint("fixed", "root", "leg_fr",
                      parent_anchor="top", child_anchor="bottom",
                      distribution_group="legs", offset=(90, 90, 0)),
                Joint("fixed", "root", "leg_bl",
                      parent_anchor="top", child_anchor="bottom",
                      distribution_group="legs", offset=(-90, -90, 0)),
                Joint("fixed", "root", "leg_br",
                      parent_anchor="top", child_anchor="bottom",
                      distribution_group="legs", offset=(90, -90, 0)),
                # Shelf connects to all 4 legs
                Joint("fixed", "leg_fl", "shelf",
                      parent_anchor="top", child_anchor="bottom",
                      no_distribute=True),
                Joint("fixed", "leg_fr", "shelf",
                      parent_anchor="top", child_anchor="bottom",
                      no_distribute=True),
                Joint("fixed", "leg_bl", "shelf",
                      parent_anchor="top", child_anchor="bottom",
                      no_distribute=True),
                Joint("fixed", "leg_br", "shelf",
                      parent_anchor="top", child_anchor="bottom",
                      no_distribute=True),
            ],
        )

        solver = AssemblySolver(assembly)
        placements = solver.solve()

        # Shelf should be centered at x=0, y=0 (averaged from 4 legs)
        shelf_pos = placements["shelf"]["position"]
        # The averaged position should be near origin in x,y
        assert abs(shelf_pos[0]) < 5.0, \
            f"shelf x={shelf_pos[0]} should be near 0"
        assert abs(shelf_pos[1]) < 5.0, \
            f"shelf y={shelf_pos[1]} should be near 0"
        # Shelf should be above legs
        assert shelf_pos[2] > placements["root"]["position"][2]


# ============================================================================
# Test: Center anchor support
# ============================================================================

class TestCenterAnchor:
    """Test center anchor support."""

    def test_center_in_anchor_directions(self):
        assert "center" in ANCHOR_DIRECTIONS
        assert ANCHOR_DIRECTIONS["center"] == (0, 0, 0)

    def test_center_offset_is_zero(self):
        part = Part(name="test", category="s", description="",
                    dimensions={"height": 20, "width": 30, "length": 40})
        offset = _anchor_offset_for_part(part, "center")
        assert offset == (0, 0, 0)

    def test_joint_with_center_anchor_solves(self):
        """Joint using center anchor should still solve correctly."""
        parts = [
            Part(name="motor", category="actuator", description="",
                 dimensions={"length": 40, "width": 30, "height": 25}),
            Part(name="wheel", category="mechanical", description="",
                 dimensions={"diameter": 65, "height": 26}),
        ]
        assembly = Assembly(
            name="Wheel Test",
            parts=parts,
            joints=[
                Joint("revolute", "motor", "wheel",
                      parent_anchor="left", child_anchor="center",
                      axis="y", range_deg=(-360, 360)),
            ],
        )

        solver = AssemblySolver(assembly)
        placements = solver.solve()
        assert "wheel" in placements
        assert len(placements["wheel"]["position"]) == 3


# ============================================================================
# Test: Apply delta propagates to grandchildren
# ============================================================================

class TestApplyDeltaGrandchildren:
    """Test that _apply_delta propagates position changes to grandchildren."""

    def test_apply_delta_propagates_to_grandchildren(self):
        """Multi-parent part repositioning should propagate to grandchildren."""
        parts = [
            Part(name="root", category="s", description="",
                 dimensions={"length": 100, "width": 100, "height": 10}),
            Part(name="support_l", category="s", description="",
                 dimensions={"length": 10, "width": 10, "height": 50}),
            Part(name="support_r", category="s", description="",
                 dimensions={"length": 10, "width": 10, "height": 50}),
            Part(name="plate", category="s", description="",
                 dimensions={"length": 80, "width": 80, "height": 5}),
            Part(name="tower", category="s", description="",
                 dimensions={"length": 20, "width": 20, "height": 40}),
        ]
        assembly = Assembly(
            name="Multi-Parent with Grandchild",
            parts=parts,
            joints=[
                Joint("fixed", "root", "support_l",
                      parent_anchor="top", child_anchor="bottom",
                      offset=(-40, 0, 0)),
                Joint("fixed", "root", "support_r",
                      parent_anchor="top", child_anchor="bottom",
                      offset=(40, 0, 0)),
                # Plate connects to both supports (multi-parent)
                Joint("fixed", "support_l", "plate",
                      parent_anchor="top", child_anchor="bottom",
                      no_distribute=True),
                Joint("fixed", "support_r", "plate",
                      parent_anchor="top", child_anchor="bottom",
                      no_distribute=True),
                # Tower on top of plate (grandchild of supports)
                Joint("fixed", "plate", "tower",
                      parent_anchor="top", child_anchor="bottom"),
            ],
        )

        solver = AssemblySolver(assembly)
        placements = solver.solve()

        # Tower should be above plate
        assert placements["tower"]["position"][2] > placements["plate"]["position"][2]

        # Plate should be centered (averaged from two supports)
        plate_x = placements["plate"]["position"][0]
        assert abs(plate_x) < 5.0, f"plate_x={plate_x} should be near 0 (averaged)"

        # Tower should also be centered (inherits from plate)
        tower_x = placements["tower"]["position"][0]
        assert abs(tower_x - plate_x) < 1.0, \
            f"tower_x={tower_x} should be near plate_x={plate_x}"


# ============================================================================
# Test: Connection constraint affects position
# ============================================================================

class TestConnectionAffectsPosition:
    """Test that connection constraints affect child position."""

    def test_connection_affects_position(self):
        """Joint with constraint_type='distance' and constraint_distance=10.0
        should place child 10mm further from parent than without the constraint."""
        parts = [
            Part(name="base", category="s", description="",
                 dimensions={"height": 20}),
            Part(name="arm", category="s", description="",
                 dimensions={"height": 100}),
        ]

        # Without constraint
        assembly_no_constraint = Assembly(
            name="No Constraint",
            parts=parts,
            joints=[
                Joint("fixed", "base", "arm",
                      parent_anchor="top", child_anchor="bottom"),
            ],
        )
        solver_no = AssemblySolver(assembly_no_constraint)
        p_no = solver_no.solve()

        # With distance constraint
        assembly_with_constraint = Assembly(
            name="With Distance Constraint",
            parts=parts,
            joints=[
                Joint("fixed", "base", "arm",
                      parent_anchor="top", child_anchor="bottom",
                      constraint_type="distance", constraint_distance=10.0),
            ],
        )
        solver_yes = AssemblySolver(assembly_with_constraint)
        p_yes = solver_yes.solve()

        # The child with distance constraint should be further in Z
        # (anchor direction for "top" is +Z)
        z_no = p_no["arm"]["position"][2]
        z_yes = p_yes["arm"]["position"][2]
        diff = z_yes - z_no
        assert diff == pytest.approx(10.0, abs=0.1), \
            f"Expected ~10mm gap from constraint, got {diff:.2f}mm"


# ============================================================================
# Test: Tool output rotation format
# ============================================================================

class TestToolOutputRotationFormat:
    """Test that the tool output has correct rotation format."""

    def test_tool_output_rotation_format(self):
        """Verify the text output has the correct rotation angle in the 4th field."""
        tool = AssemblySolveTool()
        result = tool.execute(assembly_name="robotic_arm")

        # The output should contain rotation entries like:
        # rot=(0.000, 0.000, 1.000, 0.0 deg)
        # The 4th field should be the angle (rot[3]), not a duplicate of rot[1]
        import re
        # Find all rot= entries
        rot_pattern = re.compile(r'rot=\(([^)]+)\)')
        matches = rot_pattern.findall(result)
        assert len(matches) > 0, "No rotation entries found in tool output"

        for match in matches:
            parts = [p.strip() for p in match.split(',')]
            assert len(parts) == 4, f"Expected 4 fields in rot=, got {len(parts)}: {parts}"
            # The 4th field should end with 'deg' — no duplicate of field 2
            assert parts[3].endswith('deg'), f"4th field should end with 'deg': {parts[3]}"
            # Extract numeric values: fields 0-2 are axis components, field 3 is angle
            axis_vals = [float(parts[i].split()[0]) for i in range(3)]
            angle_val = float(parts[3].replace('deg', '').strip())
            # Angle should be a finite number
            assert math.isfinite(angle_val), f"Angle should be finite, got {angle_val}"
            # Verify axis is normalized (length ~1.0 or all zero for identity)
            axis_len = math.sqrt(sum(v * v for v in axis_vals))
            if axis_len > 1e-6:
                assert axis_len == pytest.approx(1.0, abs=0.01), \
                    f"Axis should be unit length, got {axis_len:.3f}"


# ============================================================================
# Test: OpenMANIPULATOR-X (reverse-engineered from ROBOTIS URDF)
# ============================================================================

class TestOpenManipulatorXDefinition:
    """Test the OpenMANIPULATOR-X assembly definition."""

    def test_has_8_parts(self):
        assert len(OPEN_MANIPULATOR_X_PARTS) == 8

    def test_has_7_joints(self):
        assert len(OPEN_MANIPULATOR_X_ASSEMBLY.joints) == 7

    def test_all_joint_refs_valid(self):
        part_names = {p.name for p in OPEN_MANIPULATOR_X_PARTS}
        for j in OPEN_MANIPULATOR_X_ASSEMBLY.joints:
            assert j.parent in part_names, f"Joint parent '{j.parent}' not in parts"
            assert j.child in part_names, f"Joint child '{j.child}' not in parts"

    def test_all_part_names_unique(self):
        names = [p.name for p in OPEN_MANIPULATOR_X_PARTS]
        assert len(names) == len(set(names))

    def test_joint_types_correct(self):
        """4 revolute + 2 prismatic + 1 fixed = 7."""
        joints = OPEN_MANIPULATOR_X_ASSEMBLY.joints
        revolute = [j for j in joints if j.type == "revolute"]
        prismatic = [j for j in joints if j.type == "prismatic"]
        fixed = [j for j in joints if j.type == "fixed"]
        assert len(revolute) == 4
        assert len(prismatic) == 2
        assert len(fixed) == 1

    def test_joint_limits_match_urdf(self):
        """Verify joint limits match the reverse-engineered URDF values."""
        joints = {j.description: j for j in OPEN_MANIPULATOR_X_ASSEMBLY.joints}

        # joint1: Z-axis, ±180°
        j1 = joints["底座旋转 (joint1)"]
        assert j1.range_deg == (-180, 180)
        assert j1.axis == "z"

        # joint2: Y-axis, ±86°
        j2 = joints["肩部俯仰 (joint2)"]
        assert j2.range_deg == (-86, 86)
        assert j2.axis == "y"

        # joint3: Y-axis, -86° to +80°
        j3 = joints["肘部俯仰 (joint3)"]
        assert j3.range_deg == (-86, 80)
        assert j3.axis == "y"

        # joint4: Y-axis, -97° to +113°
        j4 = joints["腕部俯仰 (joint4)"]
        assert j4.range_deg == (-97, 113)
        assert j4.axis == "y"

    def test_mass_properties_present(self):
        """All non-virtual parts should have mass data from URDF."""
        for p in OPEN_MANIPULATOR_X_PARTS:
            if "虚拟" not in p.description:
                assert p.mass > 0, f"Part '{p.name}' should have mass > 0"


class TestOpenManipulatorXSolving:
    """Test solving the OpenMANIPULATOR-X assembly."""

    def test_all_parts_placed(self):
        solver = AssemblySolver(OPEN_MANIPULATOR_X_ASSEMBLY)
        placements = solver.solve()
        assert len(placements) == 8

    def test_base_at_origin(self):
        solver = AssemblySolver(OPEN_MANIPULATOR_X_ASSEMBLY)
        placements = solver.solve()
        pos = placements["omx_link1"]["position"]
        assert pos[0] == pytest.approx(0, abs=0.01)
        assert pos[1] == pytest.approx(0, abs=0.01)
        assert pos[2] == pytest.approx(0, abs=0.01)

    def test_kinematic_chain_ascending(self):
        """In home position (all zeros), links should form a vertical chain
        (link1 → link2 → link3) then bend forward (link4 → link5)."""
        solver = AssemblySolver(OPEN_MANIPULATOR_X_ASSEMBLY)
        placements = solver.solve()

        # link2 is above link1 (joint1 offset is 12mm forward + anchor offsets)
        p1 = placements["omx_link1"]["position"]
        p2 = placements["omx_link2"]["position"]
        # link2 should be higher than link1
        assert p2[2] > p1[2], f"link2 z={p2[2]} should be > link1 z={p1[2]}"

        # link3 is above link2
        p3 = placements["omx_link3"]["position"]
        assert p3[2] > p2[2], f"link3 z={p3[2]} should be > link2 z={p2[2]}"

    def test_joint_angles_affect_position(self):
        """Rotating joint1 should move all downstream parts."""
        solver = AssemblySolver(OPEN_MANIPULATOR_X_ASSEMBLY)

        p_home = solver.solve()
        p_rotated = solver.solve(joint_angles={"omx_link2": 90})

        # link2 (joint1 child) rotation should change
        assert p_home["omx_link2"]["rotation"][3] == pytest.approx(0, abs=0.1)
        assert p_rotated["omx_link2"]["rotation"][3] == pytest.approx(90, abs=0.1)

        # Downstream parts should also rotate
        assert p_rotated["omx_link3"]["rotation"][3] == pytest.approx(90, abs=0.1)

    def test_resolve_as_builtin(self):
        """OpenMANIPULATOR-X should be resolvable by name."""
        asm = _resolve_assembly("open_manipulator_x", "")
        assert asm is not None
        assert asm.name == "OpenMANIPULATOR-X"

        # Also test alternate names
        assert _resolve_assembly("openmanipulator_x", "") is not None
        assert _resolve_assembly("openmanipulatorx", "") is not None

    def test_tool_execute(self):
        """AssemblySolveTool should solve OpenMANIPULATOR-X."""
        tool = AssemblySolveTool()
        result = tool.execute(assembly_name="open_manipulator_x")
        assert "OpenMANIPULATOR-X" in result
        assert "omx_link1" in result
        assert "omx_link5" in result
        assert "omx_end_effector" in result

    def test_gripper_fingers_placed(self):
        """Gripper fingers should be positioned on opposite sides of link5."""
        solver = AssemblySolver(OPEN_MANIPULATOR_X_ASSEMBLY)
        placements = solver.solve()

        left = placements["omx_gripper_left"]["position"]
        right = placements["omx_gripper_right"]["position"]

        # Both should be placed
        assert len(left) == 3
        assert len(right) == 3

        # Fingers should be on opposite sides (Y-axis) of the wrist
        # link5's Y position should be between left and right finger Y positions
        p5 = placements["omx_link5"]["position"]
        assert left[1] != right[1], "Fingers should be on opposite sides"

    def test_total_mass_reasonable(self):
        """Total mass of all parts should be close to 711g (rated)."""
        total = sum(p.mass for p in OPEN_MANIPULATOR_X_PARTS)
        # Expected ~711g, allowing for the virtual end-effector (0g) and
        # placeholder gripper fingers (1g each)
        assert 0.5 < total < 1.0, f"Total mass {total:.3f}kg seems unreasonable"
