"""Tests for assembly rendering — build_assembly_script and RenderAssemblyTool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lang3d.knowledge.mechanics import Assembly, Joint, Part
from lang3d.tools.assembly_solver import AssemblySolver
from lang3d.tools.freecad import (
    RenderAssemblyTool,
    SUBSYSTEM_COLORS,
    _shape_type_for_part,
    _subsystem_for_part,
    build_assembly_script,
)


@pytest.fixture
def simple_assembly() -> Assembly:
    parts = [
        Part("base_plate", "structural", "底板",
             dimensions=dict(length=100, width=80, height=5)),
        Part("pillar", "structural", "立柱",
             dimensions=dict(diameter=10, height=50)),
        Part("arm_l_gripper", "structural", "左臂末端",
             dimensions=dict(length=60, width=30, height=20)),
        Part("ipc_body", "controller", "工控机",
             dimensions=dict(length=110, width=75, height=30)),
    ]
    joints = [
        Joint("fixed", "base_plate", "pillar",
              parent_anchor="top", child_anchor="bottom"),
        Joint("fixed", "pillar", "arm_l_gripper",
              parent_anchor="top", child_anchor="bottom"),
        Joint("fixed", "base_plate", "ipc_body",
              parent_anchor="top", child_anchor="bottom"),
    ]
    return Assembly(name="Test Assembly", parts=parts, joints=joints)


@pytest.fixture
def solved_positions(simple_assembly):
    solver = AssemblySolver(simple_assembly)
    return solver.solve()


class TestShapeType:
    def test_box(self):
        p = Part("plate", "structural", "", dimensions=dict(length=10, width=5, height=2))
        assert _shape_type_for_part(p) == "box"

    def test_cylinder(self):
        p = Part("rod", "structural", "", dimensions=dict(diameter=10, height=50))
        assert _shape_type_for_part(p) == "cylinder"

    def test_outer_diameter(self):
        p = Part("joint", "joint", "", dimensions=dict(outer_diameter=60, height=35))
        assert _shape_type_for_part(p) == "cylinder"


class TestSubsystemForPart:
    def test_arm_left(self):
        assert _subsystem_for_part("arm_l_gripper") == "arm_left"

    def test_arm_right(self):
        assert _subsystem_for_part("arm_r_base") == "arm_right"

    def test_ipc(self):
        assert _subsystem_for_part("ipc_body") == "ipc"

    def test_sensor(self):
        assert _subsystem_for_part("sensor_tower_post") == "sensor_tower"

    def test_chassis(self):
        assert _subsystem_for_part("base_plate") == "chassis"

    def test_motor(self):
        assert _subsystem_for_part("motor_fl") == "chassis"


class TestBuildAssemblyScript:
    def test_generates_valid_script(self, simple_assembly, solved_positions):
        parts_info = [
            {
                "name": p.name,
                "shape_type": _shape_type_for_part(p),
                "dimensions": p.dimensions,
                "subsystem": _subsystem_for_part(p.name),
            }
            for p in simple_assembly.parts
        ]
        script = build_assembly_script(parts_info, solved_positions)
        assert "import FreeCAD" in script
        assert "import Part" in script
        assert 'FreeCAD.newDocument("Assembly")' in script
        # Check all parts are in the script
        for p in simple_assembly.parts:
            assert f'"{p.name}"' in script

    def test_with_output_path(self, simple_assembly, solved_positions, tmp_path):
        parts_info = [
            {"name": p.name, "shape_type": "box", "dimensions": p.dimensions, "subsystem": ""}
            for p in simple_assembly.parts
        ]
        script = build_assembly_script(
            parts_info, solved_positions,
            output_path=str(tmp_path / "test_assembly"),
        )
        assert "saveAs" in script
        # Assembly STL export uses MeshPart.meshFromShape + _mesh.write
        # (uniform-tessellation path, freecad.py:641 / freecad_script_builder),
        # NOT the legacy bare Mesh.export which used adaptive tessellation.
        assert "_mesh.write" in script
        assert "MeshPart" in script

    def test_exploded_view(self, simple_assembly, solved_positions):
        parts_info = [
            {"name": p.name, "shape_type": "box", "dimensions": p.dimensions, "subsystem": ""}
            for p in simple_assembly.parts
        ]
        normal = build_assembly_script(parts_info, solved_positions, exploded=False)
        exploded = build_assembly_script(parts_info, solved_positions, exploded=True, explode_factor=2.0)
        # Exploded script should have different position values for non-origin parts
        assert normal != exploded

    def test_subsystem_colors_applied(self, simple_assembly, solved_positions):
        parts_info = [
            {
                "name": "arm_l_link",
                "shape_type": "box",
                "dimensions": dict(length=10, width=5, height=3),
                "subsystem": "arm_left",
            },
        ]
        script = build_assembly_script(parts_info, solved_positions)
        assert "ViewObject.ShapeColor" in script


class TestSubsystemColors:
    def test_all_subsystems_have_colors(self):
        assert "chassis" in SUBSYSTEM_COLORS
        assert "arm_left" in SUBSYSTEM_COLORS
        assert "arm_right" in SUBSYSTEM_COLORS
        assert "ipc" in SUBSYSTEM_COLORS
        assert "sensor_tower" in SUBSYSTEM_COLORS

    def test_colors_are_valid_rgb(self):
        for name, (r, g, b) in SUBSYSTEM_COLORS.items():
            assert 0 <= r <= 1, f"{name} red out of range"
            assert 0 <= g <= 1, f"{name} green out of range"
            assert 0 <= b <= 1, f"{name} blue out of range"


class TestRenderAssemblyTool:
    def test_tool_definition(self):
        tool = RenderAssemblyTool()
        defn = tool.get_definition()
        assert defn.name == "render_assembly"
        params = defn.parameters
        assert "assembly_json" in params["properties"]
        assert "output_path" in params["properties"]

    def test_tool_execute_invalid_json(self):
        tool = RenderAssemblyTool()
        result = tool.execute(assembly_json="not json", output_path="/tmp/test")
        assert "Error" in result

    def test_tool_execute_missing_data(self, tmp_path):
        tool = RenderAssemblyTool()
        result = tool.execute(
            assembly_json=json.dumps({"parts": []}),
            output_path=str(tmp_path / "test"),
        )
        assert "Error" in result or "positions" in result
