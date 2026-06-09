"""Tests for mass properties system (Task 45).

Covers: MaterialDensity, Part mass fields, compute_assembly_mass(),
tools (compute_part_mass, compute_com, compute_inertia,
compute_assembly_properties), fc_batch compute_mass operation.
"""

import json
import pytest

from lang3d.knowledge.mechanics import (
    Assembly,
    MaterialDensity,
    Part,
    compute_assembly_mass,
    Joint,
)
from lang3d.tools.mass_properties import (
    ComputeAssemblyPropertiesTool,
    ComputeCOMTool,
    ComputeInertiaTool,
    ComputePartMassTool,
    register_mass_properties_tools,
)


# ── MaterialDensity Tests ──────────────────────────────────────


class TestMaterialDensity:
    def test_pla(self):
        assert MaterialDensity.PLA == 1240

    def test_abs(self):
        assert MaterialDensity.ABS == 1040

    def test_petg(self):
        assert MaterialDensity.PETG == 1270

    def test_aluminum(self):
        assert MaterialDensity.Aluminum == 2700

    def test_steel(self):
        assert MaterialDensity.Steel == 7850

    def test_copper(self):
        assert MaterialDensity.Copper == 8960

    def test_titanium(self):
        assert MaterialDensity.Titanium == 4430

    def test_carbon_fiber(self):
        assert MaterialDensity.CarbonFiber == 1600

    def test_get_by_name(self):
        assert MaterialDensity.get("PLA") == 1240
        assert MaterialDensity.get("aluminum") == 2700
        assert MaterialDensity.get("Aluminum") == 2700

    def test_get_partial_match(self):
        assert MaterialDensity.get("alu") == 2700
        assert MaterialDensity.get("stainless") == 8000

    def test_get_unknown_defaults_pla(self):
        assert MaterialDensity.get("unobtanium") == 1240

    def test_all_materials(self):
        all_m = MaterialDensity.all_materials()
        assert len(all_m) >= 12
        assert "PLA" in all_m
        assert "Steel" in all_m


# ── Part Mass Fields Tests ──────────────────────────────────────


class TestPartMassFields:
    def test_default_mass_zero(self):
        p = Part(name="test", category="test", description="")
        assert p.mass == 0.0
        assert p.density == 0.0
        assert p.center_of_mass == (0.0, 0.0, 0.0)

    def test_effective_density_from_material(self):
        p = Part(name="test", category="test", description="", material="Aluminum")
        assert p.effective_density() == 2700

    def test_effective_density_explicit(self):
        p = Part(name="test", category="test", description="", density=5000)
        assert p.effective_density() == 5000

    def test_effective_density_explicit_overrides_material(self):
        p = Part(name="test", category="test", description="", material="PLA", density=2000)
        assert p.effective_density() == 2000

    def test_compute_volume_box(self):
        p = Part(
            name="box", category="test", description="",
            dimensions={"length": 100, "width": 50, "height": 10},
        )
        assert p.compute_volume_mm3() == 50000

    def test_compute_volume_cylinder(self):
        p = Part(
            name="cyl", category="test", description="",
            dimensions={"diameter": 80, "height": 40},
        )
        # diameter=80 → radius=40, height=40 → cylinder volume π*r²*h = π*1600*40 ≈ 201062
        import math
        expected = math.pi * 40 * 40 * 40
        assert abs(p.compute_volume_mm3() - expected) < 1.0

    def test_compute_estimated_mass_pla(self):
        p = Part(
            name="box", category="test", description="",
            material="PLA",
            dimensions={"length": 100, "width": 50, "height": 10},
        )
        mass = p.compute_estimated_mass()
        vol_m3 = 100 * 50 * 10 * 1e-9  # 5e-4 m³
        expected = vol_m3 * 1240  # 0.62 kg
        assert abs(mass - expected) < 0.001

    def test_compute_estimated_mass_with_explicit_mass(self):
        p = Part(
            name="box", category="test", description="",
            mass=1.5,
            dimensions={"length": 100, "width": 50, "height": 10},
        )
        assert p.compute_estimated_mass() == 1.5

    def test_compute_estimated_mass_aluminum(self):
        p = Part(
            name="alu_block", category="test", description="",
            material="Aluminum",
            dimensions={"length": 100, "width": 100, "height": 100},
        )
        mass = p.compute_estimated_mass()
        vol_m3 = 100 * 100 * 100 * 1e-9  # 1e-3 m³
        expected = vol_m3 * 2700  # 2.7 kg
        assert abs(mass - expected) < 0.01


# ── compute_assembly_mass Tests ─────────────────────────────────


class TestComputeAssemblyMass:
    def _make_assembly(self) -> Assembly:
        return Assembly(
            name="test_assembly",
            parts=[
                Part(
                    name="part_a", category="structural", description="",
                    material="PLA",
                    dimensions={"length": 100, "width": 50, "height": 10},
                    center_of_mass=(0, 0, 50),
                ),
                Part(
                    name="part_b", category="structural", description="",
                    material="PLA",
                    dimensions={"length": 50, "width": 50, "height": 10},
                    center_of_mass=(0, 0, 100),
                ),
            ],
        )

    def test_total_mass(self):
        asm = self._make_assembly()
        result = compute_assembly_mass(asm)
        assert result["num_parts"] == 2
        assert result["total_mass_kg"] > 0
        # part_a = 100*50*10*1e-9*1240 = 0.062 kg
        # part_b = 50*50*10*1e-9*1240 = 0.031 kg
        # total ≈ 0.093 kg
        assert abs(result["total_mass_kg"] - 0.093) < 0.001

    def test_center_of_mass_weighted(self):
        asm = self._make_assembly()
        result = compute_assembly_mass(asm)
        com = result["center_of_mass_mm"]
        # part_a: m=0.062, com_z=50; part_b: m=0.031, com_z=100
        # weighted: (0.062*50 + 0.031*100) / 0.093 = (3.1+3.1)/0.093 = 66.67
        assert abs(com[2] - 66.67) < 1.0

    def test_inertia_tensor_shape(self):
        asm = self._make_assembly()
        result = compute_assembly_mass(asm)
        tensor = result["inertia_tensor_kg_mm2"]
        assert len(tensor) == 3
        for row in tensor:
            assert len(row) == 3
        # Symmetric
        assert tensor[0][1] == tensor[1][0]
        assert tensor[0][2] == tensor[2][0]
        assert tensor[1][2] == tensor[2][1]

    def test_assembly_fields_updated(self):
        asm = self._make_assembly()
        compute_assembly_mass(asm)
        assert asm.total_mass > 0
        assert asm.center_of_mass != (0, 0, 0)

    def test_empty_assembly(self):
        asm = Assembly(name="empty")
        result = compute_assembly_mass(asm)
        assert result["total_mass_kg"] == 0
        assert result["center_of_mass_mm"] == [0, 0, 0]
        assert result["num_parts"] == 0

    def test_single_part_assembly(self):
        asm = Assembly(
            name="single",
            parts=[
                Part(
                    name="p1", category="test", description="",
                    mass=0.5,
                    center_of_mass=(10, 20, 30),
                ),
            ],
        )
        result = compute_assembly_mass(asm)
        assert abs(result["total_mass_kg"] - 0.5) < 0.001
        assert result["center_of_mass_mm"] == [10, 20, 30]


# ── Tool Tests ──────────────────────────────────────────────────


class TestComputePartMassTool:
    def test_basic(self):
        tool = ComputePartMassTool()
        result = json.loads(tool.execute(
            dimensions={"length": 100, "width": 50, "height": 10},
            material="PLA",
        ))
        assert result["mass_kg"] > 0
        assert result["density_kg_m3"] == 1240
        assert result["volume_mm3"] == 50000

    def test_aluminum(self):
        tool = ComputePartMassTool()
        result = json.loads(tool.execute(
            dimensions={"length": 100, "width": 100, "height": 100},
            material="Aluminum",
        ))
        assert abs(result["density_kg_m3"] - 2700) < 1

    def test_custom_density(self):
        tool = ComputePartMassTool()
        result = json.loads(tool.execute(
            dimensions={"length": 100, "width": 50, "height": 10},
            density=2000,
        ))
        assert result["density_kg_m3"] == 2000


class TestComputeCOMTool:
    def test_two_parts(self):
        tool = ComputeCOMTool()
        result = json.loads(tool.execute(
            parts=[
                {"name": "a", "mass_kg": 1.0, "position_mm": [0, 0, 0]},
                {"name": "b", "mass_kg": 1.0, "position_mm": [100, 0, 0]},
            ],
        ))
        assert result["center_of_mass_mm"] == [50.0, 0.0, 0.0]
        assert abs(result["total_mass_kg"] - 2.0) < 0.001

    def test_unequal_mass(self):
        tool = ComputeCOMTool()
        result = json.loads(tool.execute(
            parts=[
                {"name": "a", "mass_kg": 3.0, "position_mm": [0, 0, 0]},
                {"name": "b", "mass_kg": 1.0, "position_mm": [100, 0, 0]},
            ],
        ))
        assert abs(result["center_of_mass_mm"][0] - 25.0) < 0.1

    def test_single_part(self):
        tool = ComputeCOMTool()
        result = json.loads(tool.execute(
            parts=[
                {"name": "a", "mass_kg": 2.0, "position_mm": [10, 20, 30]},
            ],
        ))
        assert result["center_of_mass_mm"] == [10.0, 20.0, 30.0]


class TestComputeInertiaTool:
    def test_box(self):
        tool = ComputeInertiaTool()
        result = json.loads(tool.execute(
            shape="box",
            dimensions={"length": 100, "width": 50, "height": 10},
            material="PLA",
        ))
        tensor = result["inertia_tensor_kg_mm2"]
        assert len(tensor) == 3
        # Off-diagonal should be 0 for principal axes
        assert tensor[0][1] == 0.0
        assert tensor[0][2] == 0.0
        # Diagonal should be positive
        assert tensor[0][0] > 0
        assert tensor[1][1] > 0
        assert tensor[2][2] > 0

    def test_cylinder(self):
        tool = ComputeInertiaTool()
        result = json.loads(tool.execute(
            shape="cylinder",
            dimensions={"radius": 25, "height": 50},
            material="Aluminum",
        ))
        tensor = result["inertia_tensor_kg_mm2"]
        assert tensor[0][0] == tensor[1][1]  # Ixx == Iyy for cylinder
        assert tensor[2][2] > 0  # Izz

    def test_sphere(self):
        tool = ComputeInertiaTool()
        result = json.loads(tool.execute(
            shape="sphere",
            dimensions={"radius": 50},
            material="Steel",
        ))
        tensor = result["inertia_tensor_kg_mm2"]
        # All diagonal elements equal for sphere
        assert abs(tensor[0][0] - tensor[1][1]) < 0.01
        assert abs(tensor[1][1] - tensor[2][2]) < 0.01

    def test_mass_positive(self):
        tool = ComputeInertiaTool()
        result = json.loads(tool.execute(
            shape="box",
            dimensions={"length": 100, "width": 100, "height": 100},
            material="Aluminum",
        ))
        assert result["mass_kg"] > 0


class TestComputeAssemblyPropertiesTool:
    def test_basic_assembly(self):
        tool = ComputeAssemblyPropertiesTool()
        result = json.loads(tool.execute(
            assembly_name="test_asm",
            parts=[
                {"name": "base", "mass_kg": 2.0, "com_mm": [0, 0, 0]},
                {"name": "arm", "mass_kg": 1.0, "com_mm": [0, 0, 100]},
            ],
        ))
        assert result["assembly_name"] == "test_asm"
        assert abs(result["total_mass_kg"] - 3.0) < 0.001
        # COM z = (2*0 + 1*100)/3 = 33.33
        assert abs(result["center_of_mass_mm"][2] - 33.33) < 1.0

    def test_with_inertia(self):
        tool = ComputeAssemblyPropertiesTool()
        result = json.loads(tool.execute(
            assembly_name="test_asm",
            parts=[
                {
                    "name": "base",
                    "mass_kg": 2.0,
                    "com_mm": [0, 0, 0],
                    "inertia_tensor": [[100, 0, 0], [0, 100, 0], [0, 0, 50]],
                },
                {
                    "name": "arm",
                    "mass_kg": 1.0,
                    "com_mm": [0, 0, 100],
                },
            ],
        ))
        tensor = result["inertia_tensor_kg_mm2"]
        assert len(tensor) == 3
        # Should have non-zero values from parallel axis theorem
        assert tensor[2][2] > 0


# ── Registration Test ──────────────────────────────────────────


class TestRegistration:
    def test_register(self):
        registry = type("MockRegistry", (), {"register": lambda self, t: None})()
        # Should not raise
        register_mass_properties_tools(registry)

    def test_tool_names(self):
        tools = [
            ComputePartMassTool(),
            ComputeCOMTool(),
            ComputeInertiaTool(),
            ComputeAssemblyPropertiesTool(),
        ]
        names = [t.name for t in tools]
        assert "compute_part_mass" in names
        assert "compute_com" in names
        assert "compute_inertia" in names
        assert "compute_assembly_properties" in names
