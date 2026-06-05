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
    Joint,
    Part,
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
