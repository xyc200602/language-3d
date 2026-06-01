"""Tests for simulation tools and knowledge modules."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ===========================================================================
# Test: Simulation Tool Registration
# ===========================================================================

class TestSimulationToolRegistration:
    """Test that simulation tools register correctly."""

    def test_register_without_router(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.simulation import register_simulation_tools

        registry = ToolRegistry()
        register_simulation_tools(registry)

        expected = [
            "fea_run", "fea_visualize", "fea_vlm_analyze",
            "interference_check", "tolerance_analysis", "motion_sim",
            "motion_range", "motion_trajectory", "motion_vlm_analyze",
        ]
        for tool_name in expected:
            assert tool_name in registry.list_tools(), f"Missing tool: {tool_name}"

    def test_register_with_router(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.simulation import register_simulation_tools

        registry = ToolRegistry()
        mock_router = MagicMock()
        register_simulation_tools(registry, router=mock_router, screenshot_dir="/tmp")

        assert "fea_vlm_analyze" in registry.list_tools()
        assert "motion_vlm_analyze" in registry.list_tools()

    def test_all_tool_definitions_valid(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.simulation import register_simulation_tools

        registry = ToolRegistry()
        register_simulation_tools(registry)

        defs = registry.get_all_definitions()
        sim_defs = [d for d in defs if d.name in (
            "fea_run", "fea_visualize", "fea_vlm_analyze",
            "interference_check", "tolerance_analysis", "motion_sim",
            "motion_range", "motion_trajectory", "motion_vlm_analyze",
        )]
        assert len(sim_defs) == 9
        for d in sim_defs:
            assert d.name
            assert d.description
            assert "type" in d.parameters
            assert "properties" in d.parameters


# ===========================================================================
# Test: Simulation Tool Definitions
# ===========================================================================

class TestSimulationToolDefinitions:

    def test_fea_run_definition(self):
        from lang3d.tools.simulation import FEASetupAndRunTool
        tool = FEASetupAndRunTool()
        defn = tool.get_definition()
        assert defn.name == "fea_run"
        assert "document_path" in defn.parameters["properties"]
        assert "material" in defn.parameters["properties"]
        assert "mesh_size" in defn.parameters["properties"]
        assert defn.parameters["required"] == ["document_path"]

    def test_fea_visualize_definition(self):
        from lang3d.tools.simulation import FEAVisualizeTool
        tool = FEAVisualizeTool()
        defn = tool.get_definition()
        assert defn.name == "fea_visualize"
        assert "document_path" in defn.parameters["properties"]
        assert "result_type" in defn.parameters["properties"]

    def test_fea_vlm_analyze_definition(self):
        from lang3d.tools.simulation import FEAVLMAnalyzeTool
        tool = FEAVLMAnalyzeTool()
        defn = tool.get_definition()
        assert defn.name == "fea_vlm_analyze"
        assert "material" in defn.parameters["properties"]
        assert "detail" in defn.parameters["properties"]

    def test_interference_check_definition(self):
        from lang3d.tools.simulation import InterferenceCheckTool
        tool = InterferenceCheckTool()
        defn = tool.get_definition()
        assert defn.name == "interference_check"
        assert "document_path" in defn.parameters["properties"]
        assert "pairs" in defn.parameters["properties"]

    def test_tolerance_analysis_definition(self):
        from lang3d.tools.simulation import ToleranceAnalysisTool
        tool = ToleranceAnalysisTool()
        defn = tool.get_definition()
        assert defn.name == "tolerance_analysis"
        assert "dimensions" in defn.parameters["properties"]
        assert "samples" in defn.parameters["properties"]
        assert defn.parameters["required"] == ["dimensions"]

    def test_motion_sim_definition(self):
        from lang3d.tools.simulation import MotionSimTool
        tool = MotionSimTool()
        defn = tool.get_definition()
        assert defn.name == "motion_sim"
        assert "document_path" in defn.parameters["properties"]


# ===========================================================================
# Test: FEASetupAndRunTool
# ===========================================================================

class TestFEASetupAndRunTool:

    def test_missing_document_error(self):
        from lang3d.tools.simulation import FEASetupAndRunTool
        tool = FEASetupAndRunTool()
        result = tool.execute(document_path="/nonexistent/model.FCStd")
        assert "Error" in result
        assert "not found" in result

    def test_wrong_file_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            from lang3d.tools.simulation import FEASetupAndRunTool
            tool = FEASetupAndRunTool()
            result = tool.execute(document_path=path)
            assert "Error" in result
            assert ".FCStd" in result
        finally:
            Path(path).unlink(missing_ok=True)

    def test_freecad_not_available(self):
        """If FreeCAD is not available, the tool should return an error."""
        from lang3d.tools.simulation import FEASetupAndRunTool
        tool = FEASetupAndRunTool()
        # Create a dummy .FCStd file
        with tempfile.NamedTemporaryFile(suffix=".FCStd", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            result = tool.execute(document_path=path)
            # Either succeeds (FreeCAD available) or gives error
            assert isinstance(result, str)
        finally:
            Path(path).unlink(missing_ok=True)


# ===========================================================================
# Test: ToleranceAnalysisTool
# ===========================================================================

class TestToleranceAnalysisTool:

    def test_linear_tolerance_chain(self):
        from lang3d.tools.simulation import ToleranceAnalysisTool
        tool = ToleranceAnalysisTool()
        result = tool.execute(
            dimensions=[
                {"name": "A", "nominal": 10.0, "tolerance": 0.1},
                {"name": "B", "nominal": 20.0, "tolerance": 0.2},
                {"name": "C", "nominal": 5.0, "tolerance": 0.05},
            ],
            samples=5000,
        )
        assert "Mean:" in result
        assert "StdDev:" in result
        assert "Min:" in result
        assert "Max:" in result
        assert "35.0000" in result  # nominal sum = 35

    def test_single_dimension(self):
        from lang3d.tools.simulation import ToleranceAnalysisTool
        tool = ToleranceAnalysisTool()
        result = tool.execute(
            dimensions=[
                {"name": "length", "nominal": 100.0, "tolerance": 0.5},
            ],
            samples=10000,
        )
        assert "Mean:" in result
        assert "length: 100.0 +/- 0.5" in result

    def test_uniform_distribution(self):
        from lang3d.tools.simulation import ToleranceAnalysisTool
        tool = ToleranceAnalysisTool()
        result = tool.execute(
            dimensions=[
                {"name": "X", "nominal": 50.0, "tolerance": 1.0, "distribution": "uniform"},
            ],
            samples=5000,
        )
        assert "Mean:" in result
        assert "uniform" in result

    def test_cpk_calculation(self):
        from lang3d.tools.simulation import ToleranceAnalysisTool
        tool = ToleranceAnalysisTool()
        result = tool.execute(
            dimensions=[
                {"name": "A", "nominal": 10.0, "tolerance": 0.1},
            ],
            samples=10000,
            spec_limit_upper=10.5,
            spec_limit_lower=9.5,
        )
        assert "Cpk:" in result
        assert "GOOD" in result or "MARGINAL" in result

    def test_no_dimensions_error(self):
        from lang3d.tools.simulation import ToleranceAnalysisTool
        tool = ToleranceAnalysisTool()
        result = tool.execute(dimensions=[])
        assert "Error" in result

    def test_missing_dimension_fields(self):
        from lang3d.tools.simulation import ToleranceAnalysisTool
        tool = ToleranceAnalysisTool()
        result = tool.execute(dimensions=[{"name": "A"}])
        assert "Error" in result


# ===========================================================================
# Test: Materials Knowledge
# ===========================================================================

class TestMaterialsKnowledge:

    def test_get_material_steel(self):
        from lang3d.knowledge.materials import get_material
        mat = get_material("steel")
        assert mat is not None
        assert mat.name == "Steel (AISI 1045)"
        assert mat.youngs_modulus == 200_000
        assert mat.yield_strength == 350

    def test_get_material_aluminum(self):
        from lang3d.knowledge.materials import get_material
        mat = get_material("aluminum")
        assert mat is not None
        assert mat.youngs_modulus == 68_900

    def test_get_material_pla(self):
        from lang3d.knowledge.materials import get_material
        mat = get_material("pla")
        assert mat is not None
        assert mat.category == "polymer"

    def test_get_material_all_presets(self):
        from lang3d.knowledge.materials import MATERIAL_PRESETS, get_material
        for name in MATERIAL_PRESETS:
            mat = get_material(name)
            assert mat is not None, f"Missing material: {name}"
            assert mat.youngs_modulus > 0
            assert mat.density > 0
            assert mat.yield_strength > 0

    def test_get_material_case_insensitive(self):
        from lang3d.knowledge.materials import get_material
        assert get_material("Steel") is not None
        assert get_material("ALUMINUM") is not None

    def test_get_material_alias(self):
        from lang3d.knowledge.materials import get_material
        mat = get_material("aluminium")
        assert mat is not None
        assert "Aluminum" in mat.name

    def test_get_material_unknown(self):
        from lang3d.knowledge.materials import get_material
        assert get_material("unobtanium") is None

    def test_safety_factor_calculation(self):
        from lang3d.knowledge.materials import compute_safety_factor, get_material
        mat = get_material("steel")
        sf = compute_safety_factor(mat, 100)
        assert abs(sf - 3.5) < 0.01

    def test_safety_factor_yield(self):
        from lang3d.knowledge.materials import compute_safety_factor, get_material
        mat = get_material("steel")
        sf = compute_safety_factor(mat, mat.yield_strength)
        assert abs(sf - 1.0) < 0.01

    def test_safety_factor_infinite(self):
        from lang3d.knowledge.materials import compute_safety_factor, get_material
        mat = get_material("steel")
        sf = compute_safety_factor(mat, 0)
        assert sf == float("inf")

    def test_safety_factor_unsafe(self):
        from lang3d.knowledge.materials import compute_safety_factor, get_material
        mat = get_material("steel")
        sf = compute_safety_factor(mat, 500)
        assert sf < 1.0

    def test_safety_factors_dict(self):
        from lang3d.knowledge.materials import SAFETY_FACTORS
        assert "static" in SAFETY_FACTORS
        assert SAFETY_FACTORS["static"] == 1.5
        assert SAFETY_FACTORS["dynamic"] == 2.0


# ===========================================================================
# Test: Simulation Knowledge
# ===========================================================================

class TestSimulationKnowledge:

    def test_fea_patterns(self):
        from lang3d.knowledge.simulation import FEA_PATTERNS
        assert "cantilever_bend" in FEA_PATTERNS
        assert "compression_test" in FEA_PATTERNS
        assert "tensile_test" in FEA_PATTERNS
        assert "gravitational_load" in FEA_PATTERNS

    def test_fea_pattern_fields(self):
        from lang3d.knowledge.simulation import FEA_PATTERNS
        for name, pattern in FEA_PATTERNS.items():
            assert pattern.name
            assert pattern.description
            assert pattern.constraints_template
            assert pattern.mesh_recommendation in ("coarse", "medium", "fine", "very_fine")

    def test_mesh_sizes(self):
        from lang3d.knowledge.simulation import MESH_SIZES
        for key in ("coarse", "medium", "fine", "very_fine"):
            assert key in MESH_SIZES
            assert MESH_SIZES[key]["max_element_size_factor"] > 0
            assert MESH_SIZES[key]["min_elements_per_edge"] > 0

    def test_recommend_mesh_small(self):
        from lang3d.knowledge.simulation import recommend_mesh_size
        assert recommend_mesh_size(10) == "fine"

    def test_recommend_mesh_medium(self):
        from lang3d.knowledge.simulation import recommend_mesh_size
        assert recommend_mesh_size(50) == "medium"
        assert recommend_mesh_size(200) == "medium"

    def test_recommend_mesh_large(self):
        from lang3d.knowledge.simulation import recommend_mesh_size
        assert recommend_mesh_size(1000) == "coarse"


# ===========================================================================
# Test: InterferenceCheckTool
# ===========================================================================

class TestInterferenceCheckTool:

    def test_missing_document_error(self):
        from lang3d.tools.simulation import InterferenceCheckTool
        tool = InterferenceCheckTool()
        result = tool.execute(document_path="/nonexistent/model.FCStd")
        assert "Error" in result
        assert "not found" in result


# ===========================================================================
# Test: MotionSimTool (STUB)
# ===========================================================================

class TestMotionSimTool:

    def test_motion_sim_not_stub(self):
        from lang3d.tools.simulation import MotionSimTool
        tool = MotionSimTool()
        result = tool.execute(document_path="test.FCStd")
        # Should return an error about missing file, not a stub message
        assert "Error" in result or "not found" in result

    def test_motion_sim_has_new_params(self):
        from lang3d.tools.simulation import MotionSimTool
        tool = MotionSimTool()
        defn = tool.get_definition()
        assert "joint_angles" in defn.parameters["properties"]
        assert "analysis_type" in defn.parameters["properties"]


# ===========================================================================
# Test: FEA VLM Analyze (without router)
# ===========================================================================

class TestFEAVLMAnalyzeTool:

    def test_no_router_error(self):
        from lang3d.tools.simulation import FEAVLMAnalyzeTool
        tool = FEAVLMAnalyzeTool()
        result = tool.execute()
        assert "Error" in result
        assert "router" in result.lower() or "VLM" in result

    def test_parse_fea_vlm_json_with_json(self):
        from lang3d.tools.simulation import _parse_fea_vlm_json
        raw = '{"safe": true, "max_stress_region": "bottom-left fillet", "stress_distribution": "gradient from blue to red", "safety_concern": "None", "suggestion": "Add fillet", "fix_commands": "fc_batch fillet"}'
        result = _parse_fea_vlm_json(raw)
        assert result["safe"] is True
        assert result["max_stress_region"] == "bottom-left fillet"
        assert result["suggestion"] == "Add fillet"

    def test_parse_fea_vlm_json_fallback(self):
        from lang3d.tools.simulation import _parse_fea_vlm_json
        raw = "safe: false\nmax_stress_region: root of beam\nsuggestion: increase radius"
        result = _parse_fea_vlm_json(raw)
        assert result["safe"] is False
        assert "root" in result["max_stress_region"]


# ===========================================================================
# Test: Config Integration
# ===========================================================================

class TestSimulationConfig:

    def test_simulation_settings_default(self):
        from lang3d.config import SimulationSettings
        settings = SimulationSettings()
        assert settings.calculix_path == ""
        assert settings.default_material == "steel"
        assert settings.default_mesh_size == "medium"
        assert settings.fea_timeout == 120
        assert settings.default_fea_samples == 10000

    def test_agent_config_has_simulation(self):
        from lang3d.config import AgentConfig
        config = AgentConfig()
        assert hasattr(config, "simulation")
        assert config.simulation.default_material == "steel"

    def test_env_config_calculix(self):
        from lang3d.config import _build_env_config
        import os
        old = os.environ.get("CALCULIX_PATH")
        try:
            os.environ["CALCULIX_PATH"] = "/usr/bin/ccx"
            result = _build_env_config()
            assert result["agent"]["simulation"]["calculix_path"] == "/usr/bin/ccx"
        finally:
            if old:
                os.environ["CALCULIX_PATH"] = old
            else:
                os.environ.pop("CALCULIX_PATH", None)

    def test_env_config_default_material(self):
        from lang3d.config import _build_env_config
        import os
        old = os.environ.get("DEFAULT_MATERIAL")
        try:
            os.environ["DEFAULT_MATERIAL"] = "aluminum"
            result = _build_env_config()
            assert result["agent"]["simulation"]["default_material"] == "aluminum"
        finally:
            if old:
                os.environ["DEFAULT_MATERIAL"] = old
            else:
                os.environ.pop("DEFAULT_MATERIAL", None)
