"""Tests for CFD tools and knowledge."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Test: CFD Tool Registration
# ===========================================================================

class TestCFDToolRegistration:

    def test_register_cfd_tools(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.cfd import register_cfd_tools

        registry = ToolRegistry()
        register_cfd_tools(registry)

        expected = ["cfd_run", "cfd_vlm_analyze"]
        for tool_name in expected:
            assert tool_name in registry.list_tools(), f"Missing tool: {tool_name}"

    def test_register_with_router(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.cfd import register_cfd_tools

        registry = ToolRegistry()
        mock_router = MagicMock()
        register_cfd_tools(registry, router=mock_router, screenshot_dir="/tmp")

        assert "cfd_vlm_analyze" in registry.list_tools()

    def test_all_cfd_tool_definitions_valid(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.cfd import register_cfd_tools

        registry = ToolRegistry()
        register_cfd_tools(registry)

        defs = registry.get_all_definitions()
        cfd_defs = [d for d in defs if d.name in ("cfd_run", "cfd_vlm_analyze")]
        assert len(cfd_defs) == 2
        for d in cfd_defs:
            assert d.name
            assert d.description
            assert "type" in d.parameters
            assert "properties" in d.parameters


# ===========================================================================
# Test: CFD Tool Definitions
# ===========================================================================

class TestCFDToolDefinitions:

    def test_cfd_run_definition(self):
        from lang3d.tools.cfd import CFDRunTool
        tool = CFDRunTool()
        defn = tool.get_definition()
        assert defn.name == "cfd_run"
        assert "document_path" in defn.parameters["properties"]
        assert "fluid" in defn.parameters["properties"]
        assert "pattern" in defn.parameters["properties"]
        assert "mesh_size" in defn.parameters["properties"]
        assert defn.parameters["required"] == ["document_path"]

    def test_cfd_vlm_analyze_definition(self):
        from lang3d.tools.cfd import CFDVLMAnalyzeTool
        tool = CFDVLMAnalyzeTool()
        defn = tool.get_definition()
        assert defn.name == "cfd_vlm_analyze"
        assert "fluid" in defn.parameters["properties"]
        assert "analysis_type" in defn.parameters["properties"]
        assert "detail" in defn.parameters["properties"]


# ===========================================================================
# Test: CFDRunTool Execution
# ===========================================================================

class TestCFDRunToolExecution:

    def test_missing_document_error(self):
        from lang3d.tools.cfd import CFDRunTool
        tool = CFDRunTool()
        result = tool.execute(document_path="/nonexistent/model.FCStd")
        assert "Error" in result
        assert "not found" in result

    def test_invalid_fluid(self):
        with tempfile.NamedTemporaryFile(suffix=".FCStd", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            from lang3d.tools.cfd import CFDRunTool
            tool = CFDRunTool()
            result = tool.execute(document_path=path, fluid="liquid_gold")
            assert "Error" in result
            assert "Unknown fluid" in result
        finally:
            Path(path).unlink(missing_ok=True)

    def test_invalid_pattern(self):
        with tempfile.NamedTemporaryFile(suffix=".FCStd", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            from lang3d.tools.cfd import CFDRunTool
            tool = CFDRunTool()
            result = tool.execute(document_path=path, pattern="supersonic")
            assert "Error" in result
            assert "Unknown pattern" in result
        finally:
            Path(path).unlink(missing_ok=True)

    def test_invalid_mesh_size(self):
        with tempfile.NamedTemporaryFile(suffix=".FCStd", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            from lang3d.tools.cfd import CFDRunTool
            tool = CFDRunTool()
            result = tool.execute(document_path=path, mesh_size="ultra_fine")
            assert "Error" in result
            assert "Unknown mesh_size" in result
        finally:
            Path(path).unlink(missing_ok=True)

    def test_openfoam_not_found_graceful(self):
        """When OpenFOAM is not found, tool should return install instructions."""
        with tempfile.NamedTemporaryFile(suffix=".FCStd", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            from lang3d.tools.cfd import CFDRunTool
            tool = CFDRunTool()
            with patch("lang3d.tools.cfd._find_openfoam", return_value=(None, "none")):
                result = tool.execute(document_path=path)
                assert "Error" in result or "OpenFOAM not found" in result
        finally:
            Path(path).unlink(missing_ok=True)


# ===========================================================================
# Test: CFDVLMAnalyzeTool (without router)
# ===========================================================================

class TestCFDVLMAnalyzeTool:

    def test_no_router_error(self):
        from lang3d.tools.cfd import CFDVLMAnalyzeTool
        tool = CFDVLMAnalyzeTool()
        result = tool.execute()
        assert "Error" in result
        assert "router" in result.lower() or "VLM" in result

    def test_parse_cfd_vlm_json_with_json(self):
        from lang3d.tools.cfd import _parse_cfd_vlm_json
        raw = (
            '{"flow_regime": "turbulent", "max_velocity": "inlet center, ~5.2 m/s", '
            '"pressure_drop": "gradual decrease from inlet to outlet", '
            '"separation_regions": "behind obstacle", '
            '"suggestion": "streamline obstacle shape"}'
        )
        result = _parse_cfd_vlm_json(raw)
        assert result["flow_regime"] == "turbulent"
        assert "inlet center" in result["max_velocity"]
        assert result["separation_regions"] == "behind obstacle"

    def test_parse_cfd_vlm_json_fallback(self):
        from lang3d.tools.cfd import _parse_cfd_vlm_json
        raw = "flow_regime: laminar\nmax_velocity: inlet\nsuggestion: increase pipe diameter"
        result = _parse_cfd_vlm_json(raw)
        assert "laminar" in result["flow_regime"]
        assert "inlet" in result["max_velocity"]


# ===========================================================================
# Test: CFD Knowledge
# ===========================================================================

class TestCFDKnowledge:

    def test_fluid_presets(self):
        from lang3d.knowledge.simulation import FLUID_PRESETS
        assert "air" in FLUID_PRESETS
        assert "water" in FLUID_PRESETS

    def test_fluid_properties_fields(self):
        from lang3d.knowledge.simulation import FLUID_PRESETS
        for name, fluid in FLUID_PRESETS.items():
            assert fluid.name
            assert fluid.density > 0
            assert fluid.kinematic_viscosity > 0

    def test_air_properties(self):
        from lang3d.knowledge.simulation import FLUID_PRESETS
        air = FLUID_PRESETS["air"]
        assert abs(air.density - 1.204) < 0.01
        assert air.kinematic_viscosity > 1e-6

    def test_water_properties(self):
        from lang3d.knowledge.simulation import FLUID_PRESETS
        water = FLUID_PRESETS["water"]
        assert abs(water.density - 998.2) < 1
        assert water.kinematic_viscosity > 1e-7

    def test_get_fluid(self):
        from lang3d.knowledge.simulation import get_fluid
        assert get_fluid("air") is not None
        assert get_fluid("water") is not None
        assert get_fluid("AIR") is not None  # case insensitive
        assert get_fluid("molten_iron") is None

    def test_cfd_patterns(self):
        from lang3d.knowledge.simulation import CFD_PATTERNS
        assert "pipe_flow" in CFD_PATTERNS
        assert "external_flow" in CFD_PATTERNS
        assert "heat_exchanger" in CFD_PATTERNS

    def test_cfd_pattern_fields(self):
        from lang3d.knowledge.simulation import CFD_PATTERNS
        for name, pattern in CFD_PATTERNS.items():
            assert pattern.name
            assert pattern.solver
            assert pattern.turbulence_model

    def test_cfd_mesh_sizes(self):
        from lang3d.knowledge.simulation import CFD_MESH_SIZES
        for key in ("coarse", "medium", "fine"):
            assert key in CFD_MESH_SIZES
            assert CFD_MESH_SIZES[key]["max_cell_size_factor"] > 0
            assert CFD_MESH_SIZES[key]["boundary_layers"] >= 1


# ===========================================================================
# Test: CFD Case Builder
# ===========================================================================

class TestCFDCaseBuilder:

    def test_build_case_creates_files(self):
        from lang3d.tools.cfd import _build_cfd_case

        with tempfile.TemporaryDirectory() as tmpdir:
            case_path = _build_cfd_case(
                case_dir=tmpdir,
                fluid_name="air",
                pattern_name="pipe_flow",
                mesh_size="medium",
                boundary_conditions=None,
                inlet_velocity=1.0,
                outlet_pressure=0.0,
            )

            # Check directories exist
            assert Path(case_path, "constant").exists()
            assert Path(case_path, "0").exists()
            assert Path(case_path, "system").exists()

            # Check key files exist
            assert Path(case_path, "system", "controlDict").exists()
            assert Path(case_path, "system", "fvSchemes").exists()
            assert Path(case_path, "system", "fvSolution").exists()
            assert Path(case_path, "constant", "transportProperties").exists()
            assert Path(case_path, "constant", "turbulenceProperties").exists()
            assert Path(case_path, "0", "U").exists()
            assert Path(case_path, "0", "p").exists()
            assert Path(case_path, "0", "k").exists()
            assert Path(case_path, "0", "omega").exists()

    def test_build_case_water_fluid(self):
        from lang3d.tools.cfd import _build_cfd_case

        with tempfile.TemporaryDirectory() as tmpdir:
            case_path = _build_cfd_case(
                case_dir=tmpdir,
                fluid_name="water",
                pattern_name="pipe_flow",
                mesh_size="fine",
                boundary_conditions=None,
                inlet_velocity=0.5,
                outlet_pressure=101325.0,
            )

            transport = Path(case_path, "constant", "transportProperties").read_text()
            # Water density ~998
            assert "998" in transport

    def test_build_case_external_flow(self):
        from lang3d.tools.cfd import _build_cfd_case

        with tempfile.TemporaryDirectory() as tmpdir:
            case_path = _build_cfd_case(
                case_dir=tmpdir,
                fluid_name="air",
                pattern_name="external_flow",
                mesh_size="coarse",
                boundary_conditions=None,
                inlet_velocity=10.0,
                outlet_pressure=0.0,
            )

            control = Path(case_path, "system", "controlDict").read_text()
            assert "simpleFoam" in control

    def test_build_case_heat_exchanger(self):
        from lang3d.tools.cfd import _build_cfd_case

        with tempfile.TemporaryDirectory() as tmpdir:
            case_path = _build_cfd_case(
                case_dir=tmpdir,
                fluid_name="water",
                pattern_name="heat_exchanger",
                mesh_size="medium",
                boundary_conditions=None,
                inlet_velocity=2.0,
                outlet_pressure=0.0,
            )

            control = Path(case_path, "system", "controlDict").read_text()
            assert "buoyantSimpleFoam" in control


# ===========================================================================
# Test: OpenFOAM Discovery
# ===========================================================================

class TestOpenFOAMDiscovery:

    def test_find_openfoam_returns_tuple(self):
        from lang3d.tools.cfd import _find_openfoam
        result = _find_openfoam()
        assert isinstance(result, tuple)
        assert len(result) == 2
        # mode should be a string
        assert isinstance(result[1], str)

    def test_win_to_wsl_path(self):
        from lang3d.tools.cfd import _win_to_wsl_path
        assert _win_to_wsl_path(r"C:\Users\test") == "/mnt/c/Users/test"
        assert _win_to_wsl_path(r"D:\data\file.txt") == "/mnt/d/data/file.txt"
        assert _win_to_wsl_path("/already/linux") == "/already/linux"


# ===========================================================================
# Test: Config Integration for CFD
# ===========================================================================

class TestCFDConfig:

    def test_simulation_settings_cfd_fields(self):
        from lang3d.config import SimulationSettings
        settings = SimulationSettings()
        assert settings.openfoam_path == ""
        assert settings.default_fluid == "air"
        assert settings.cfd_timeout == 300
        assert settings.openfoam_mode == "auto"

    def test_env_config_openfoam_path(self):
        from lang3d.config import _build_env_config
        old = os.environ.get("OPENFOAM_PATH")
        try:
            os.environ["OPENFOAM_PATH"] = "/opt/openfoam11"
            result = _build_env_config()
            assert result["agent"]["simulation"]["openfoam_path"] == "/opt/openfoam11"
        finally:
            if old:
                os.environ["OPENFOAM_PATH"] = old
            else:
                os.environ.pop("OPENFOAM_PATH", None)

    def test_env_config_openfoam_mode(self):
        from lang3d.config import _build_env_config
        old = os.environ.get("OPENFOAM_MODE")
        try:
            os.environ["OPENFOAM_MODE"] = "docker"
            result = _build_env_config()
            assert result["agent"]["simulation"]["openfoam_mode"] == "docker"
        finally:
            if old:
                os.environ["OPENFOAM_MODE"] = old
            else:
                os.environ.pop("OPENFOAM_MODE", None)
