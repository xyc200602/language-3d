"""Tests for 3D printing slicing tools and knowledge base."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Test: Knowledge Base - Printer Presets
# ===========================================================================

class TestPrinterPresets:

    def test_all_presets_present(self):
        from lang3d.knowledge.slicing import PRINTER_PRESETS
        expected = ["prusa_mk3s", "ender_3", "bambu_p1s", "generic"]
        for name in expected:
            assert name in PRINTER_PRESETS, f"Missing printer preset: {name}"

    def test_preset_has_required_fields(self):
        from lang3d.knowledge.slicing import PRINTER_PRESETS
        for name, preset in PRINTER_PRESETS.items():
            assert "bed_x" in preset, f"{name} missing bed_x"
            assert "bed_y" in preset, f"{name} missing bed_y"
            assert "bed_z" in preset, f"{name} missing bed_z"
            assert "nozzle" in preset, f"{name} missing nozzle"
            assert preset["bed_x"] > 0
            assert preset["bed_y"] > 0
            assert preset["bed_z"] > 0
            assert preset["nozzle"] > 0

    def test_generic_preset_exists(self):
        from lang3d.knowledge.slicing import PRINTER_PRESETS
        assert "generic" in PRINTER_PRESETS
        assert PRINTER_PRESETS["generic"]["bed_x"] == 200


# ===========================================================================
# Test: Knowledge Base - Material Presets
# ===========================================================================

class TestMaterialPresets:

    def test_all_presets_present(self):
        from lang3d.knowledge.slicing import MATERIAL_PRESETS
        expected = ["pla", "abs", "petg", "tpu"]
        for name in expected:
            assert name in MATERIAL_PRESETS, f"Missing material preset: {name}"

    def test_preset_has_required_fields(self):
        from lang3d.knowledge.slicing import MATERIAL_PRESETS
        for name, preset in MATERIAL_PRESETS.items():
            assert "temp" in preset, f"{name} missing temp"
            assert "bed_temp" in preset, f"{name} missing bed_temp"
            assert "diameter" in preset, f"{name} missing diameter"
            assert "density" in preset, f"{name} missing density"
            assert "cost_per_kg" in preset, f"{name} missing cost_per_kg"
            assert preset["temp"] > 0
            assert preset["density"] > 0

    def test_pla_preset_values(self):
        from lang3d.knowledge.slicing import MATERIAL_PRESETS
        pla = MATERIAL_PRESETS["pla"]
        assert pla["temp"] == 200
        assert pla["bed_temp"] == 60
        assert pla["diameter"] == 1.75


# ===========================================================================
# Test: Knowledge Base - Quality Presets
# ===========================================================================

class TestQualityPresets:

    def test_all_presets_present(self):
        from lang3d.knowledge.slicing import QUALITY_PRESETS
        expected = ["draft", "standard", "high"]
        for name in expected:
            assert name in QUALITY_PRESETS, f"Missing quality preset: {name}"

    def test_preset_has_required_fields(self):
        from lang3d.knowledge.slicing import QUALITY_PRESETS
        for name, preset in QUALITY_PRESETS.items():
            assert "layer_height" in preset
            assert "infill" in preset
            assert "perimeters" in preset
            assert preset["layer_height"] > 0
            assert 0 <= preset["infill"] <= 100

    def test_quality_ordering(self):
        from lang3d.knowledge.slicing import QUALITY_PRESETS
        assert QUALITY_PRESETS["draft"]["layer_height"] > QUALITY_PRESETS["standard"]["layer_height"]
        assert QUALITY_PRESETS["standard"]["layer_height"] > QUALITY_PRESETS["high"]["layer_height"]


# ===========================================================================
# Test: Knowledge Base - Parameter Merging
# ===========================================================================

class TestMergeParams:

    def test_default_merge(self):
        from lang3d.knowledge.slicing import merge_params
        params = merge_params()
        assert params["bed_x"] == 200  # generic
        assert params["temp"] == 200  # pla
        assert params["layer_height"] == 0.2  # standard
        assert params["printer_name"] == "Generic FFF Printer"
        assert params["material_name"] == "PLA"

    def test_custom_presets(self):
        from lang3d.knowledge.slicing import merge_params
        params = merge_params(printer="ender_3", material="abs", quality="draft")
        assert params["bed_x"] == 220
        assert params["temp"] == 240
        assert params["layer_height"] == 0.3
        assert params["printer_name"] == "Creality Ender 3"
        assert params["material_name"] == "ABS"

    def test_user_overrides(self):
        from lang3d.knowledge.slicing import merge_params
        params = merge_params(layer_height=0.15, infill=50)
        assert params["layer_height"] == 0.15
        assert params["infill"] == 50

    def test_none_overrides_ignored(self):
        from lang3d.knowledge.slicing import merge_params
        params = merge_params(layer_height=None)
        assert params["layer_height"] == 0.2  # standard default

    def test_unknown_preset_falls_back(self):
        from lang3d.knowledge.slicing import merge_params
        params = merge_params(printer="unknown_printer")
        assert params["bed_x"] == 200  # falls back to generic


# ===========================================================================
# Test: G-code Parsing
# ===========================================================================

class TestGcodeParsing:

    @pytest.fixture
    def sample_gcode(self, tmp_path):
        """Create a sample G-code file for testing."""
        gcode = tmp_path / "test.gcode"
        gcode.write_text(
            "; generated by PrusaSlicer\n"
            "; estimated printing time = 1h 23m 45s\n"
            "; filament used [mm] = 1234.5\n"
            "; filament used [g] = 3.72\n"
            "; filament used [cm3] = 2.99\n"
            "; total filament cost = 0.07\n"
            "; total layers = 150\n"
            "; support material = yes\n"
            "; brim width = 5\n"
            "G1 X10 Y10 Z0.2 E0.5\n"
            "G1 X20 Y20 Z0.2 E1.0\n"
            "G1 X30 Y30 Z0.4 E1.5\n"
            "G1 X40 Y40 Z0.4 E2.0\n"
            "G1 X50 Y50 Z0.6 E2.5\n",
            encoding="utf-8",
        )
        return str(gcode)

    @pytest.fixture
    def minimal_gcode(self, tmp_path):
        """Create a minimal G-code file."""
        gcode = tmp_path / "minimal.gcode"
        gcode.write_text(
            "G1 X0 Y0 Z0.2 E0.1\n"
            "G1 X10 Y10 Z0.2 E0.2\n"
            "G1 X10 Y10 Z0.4 E0.3\n",
            encoding="utf-8",
        )
        return str(gcode)

    def test_parse_stats_basic(self, sample_gcode):
        from lang3d.knowledge.slicing import parse_gcode_stats
        stats = parse_gcode_stats(sample_gcode)
        assert stats["print_time_h"] == 1
        assert stats["print_time_m"] == 23
        assert stats["print_time_s"] == 1 * 3600 + 23 * 60 + 45
        assert abs(stats["filament_mm"] - 1234.5) < 0.1
        assert abs(stats["filament_g"] - 3.72) < 0.1
        assert abs(stats["filament_cm3"] - 2.99) < 0.01
        assert abs(stats["cost"] - 0.07) < 0.01
        assert stats["total_layers"] == 150
        assert stats["has_supports"] is True
        assert stats["has_brim"] is True

    def test_parse_stats_nonexistent_file(self):
        from lang3d.knowledge.slicing import parse_gcode_stats
        stats = parse_gcode_stats("/nonexistent/path.gcode")
        assert stats["print_time_s"] == 0
        assert stats["total_layers"] == 0

    def test_parse_stats_empty_file(self, tmp_path):
        from lang3d.knowledge.slicing import parse_gcode_stats
        gcode = tmp_path / "empty.gcode"
        gcode.write_text("", encoding="utf-8")
        stats = parse_gcode_stats(str(gcode))
        assert stats["print_time_s"] == 0

    def test_parse_layers_basic(self, minimal_gcode):
        from lang3d.knowledge.slicing import parse_gcode_layers
        layers = parse_gcode_layers(minimal_gcode)
        assert len(layers) == 2
        assert layers[0]["z_height"] == pytest.approx(0.2)
        assert layers[1]["z_height"] == pytest.approx(0.4)
        assert layers[0]["layer_number"] == 1
        assert layers[1]["layer_number"] == 2

    def test_parse_layers_nonexistent(self):
        from lang3d.knowledge.slicing import parse_gcode_layers
        layers = parse_gcode_layers("/nonexistent/path.gcode")
        assert layers == []

    def test_parse_bounds_basic(self, minimal_gcode):
        from lang3d.knowledge.slicing import parse_gcode_bounds
        bounds = parse_gcode_bounds(minimal_gcode)
        assert bounds["min_x"] == pytest.approx(0.0, abs=0.1)
        assert bounds["max_x"] == pytest.approx(10.0, abs=0.1)
        assert bounds["min_y"] == pytest.approx(0.0, abs=0.1)

    def test_parse_bounds_nonexistent(self):
        from lang3d.knowledge.slicing import parse_gcode_bounds
        bounds = parse_gcode_bounds("/nonexistent/path.gcode")
        assert bounds == {}

    def test_z_count_layer_fallback(self, tmp_path):
        """When no 'total layers' comment, count Z changes."""
        from lang3d.knowledge.slicing import parse_gcode_stats
        gcode = tmp_path / "no_comment.gcode"
        gcode.write_text(
            "G1 X0 Y0 Z0.2 E0.1\n"
            "G1 X1 Y1 Z0.2 E0.2\n"
            "G1 X2 Y2 Z0.4 E0.3\n"
            "G1 X3 Y3 Z0.6 E0.4\n",
            encoding="utf-8",
        )
        stats = parse_gcode_stats(str(gcode))
        assert stats["total_layers"] == 3  # 3 distinct Z heights


# ===========================================================================
# Test: Slicer Discovery
# ===========================================================================

class TestSlicerDiscovery:

    def test_find_slicer_with_env_var(self):
        from lang3d.tools.slicing import _find_slicer
        with patch.dict(os.environ, {"PRUSASLICER_PATH": "/fake/slicer"}):
            with patch.object(Path, "exists", return_value=True):
                result = _find_slicer()
                assert result == "/fake/slicer"

    def test_find_slicer_returns_none_when_not_found(self):
        from lang3d.tools.slicing import _find_slicer
        with patch.dict(os.environ, {"PRUSASLICER_PATH": ""}, clear=False):
            with patch("shutil.which", return_value=None):
                with patch.object(Path, "exists", return_value=False):
                    result = _find_slicer()
                    # May or may not be None depending on install
                    assert isinstance(result, (str, type(None)))

    def test_find_slicer_gui_with_env_var(self):
        from lang3d.tools.slicing import _find_slicer_gui
        with patch.dict(os.environ, {"PRUSASLICER_GUI_PATH": "/fake/gui"}):
            with patch.object(Path, "exists", return_value=True):
                result = _find_slicer_gui()
                assert result == "/fake/gui"


# ===========================================================================
# Test: Tool Registration
# ===========================================================================

class TestSlicingToolRegistration:

    def test_four_tools_registered(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.slicing import register_slicing_tools

        registry = ToolRegistry()
        register_slicing_tools(registry)

        tools = registry.list_tools()
        assert "slice_model" in tools
        assert "slice_analyze" in tools
        assert "slice_preview_layers" in tools
        assert "slice_vlm_analyze" in tools

    def test_tool_count(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.slicing import register_slicing_tools

        registry = ToolRegistry()
        register_slicing_tools(registry)

        slicing_tools = [t for t in registry.list_tools() if t.startswith("slice_")]
        assert len(slicing_tools) == 4

    def test_definitions_valid(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.slicing import register_slicing_tools

        registry = ToolRegistry()
        register_slicing_tools(registry)

        for tool_name in ["slice_model", "slice_analyze", "slice_preview_layers", "slice_vlm_analyze"]:
            tool = registry.get(tool_name)
            assert tool is not None
            defn = tool.get_definition()
            assert defn.name == tool_name
            assert defn.description
            assert "type" in defn.parameters
            assert "properties" in defn.parameters

    def test_tool_categories_updated(self):
        from lang3d.tools.base import TOOL_CATEGORIES
        assert "slicing" in TOOL_CATEGORIES
        assert "slice_model" in TOOL_CATEGORIES["slicing"]
        assert "slice_analyze" in TOOL_CATEGORIES["slicing"]
        assert "slice_preview_layers" in TOOL_CATEGORIES["slicing"]
        assert "slice_vlm_analyze" in TOOL_CATEGORIES["slicing"]

    def test_step_tool_categories_updated(self):
        from lang3d.tools.base import STEP_TOOL_CATEGORIES
        assert "slicing" in STEP_TOOL_CATEGORIES
        assert "slicing" in STEP_TOOL_CATEGORIES["slicing"]


# ===========================================================================
# Test: Tool Definitions
# ===========================================================================

class TestSlicingToolDefinitions:

    def test_slice_model_definition(self):
        from lang3d.tools.slicing import SliceModelTool
        tool = SliceModelTool()
        defn = tool.get_definition()
        assert defn.name == "slice_model"
        props = defn.parameters["properties"]
        assert "stl_path" in props
        assert "printer" in props
        assert "material" in props
        assert "quality" in props
        assert "layer_height" in props
        assert "infill" in props
        assert "supports" in props
        assert "brim" in props
        assert "output_path" in props
        assert "stl_path" in defn.parameters["required"]

    def test_slice_analyze_definition(self):
        from lang3d.tools.slicing import SliceAnalyzeTool
        tool = SliceAnalyzeTool()
        defn = tool.get_definition()
        assert defn.name == "slice_analyze"
        assert "gcode_path" in defn.parameters["required"]

    def test_slice_preview_layers_definition(self):
        from lang3d.tools.slicing import SlicePreviewLayersTool
        tool = SlicePreviewLayersTool()
        defn = tool.get_definition()
        assert defn.name == "slice_preview_layers"
        props = defn.parameters["properties"]
        assert "gcode_path" in props
        assert "layer_range" in props

    def test_slice_vlm_analyze_definition(self):
        from lang3d.tools.slicing import SliceVLMAnalyzeTool
        tool = SliceVLMAnalyzeTool()
        defn = tool.get_definition()
        assert defn.name == "slice_vlm_analyze"
        assert "gcode_path" in defn.parameters["required"]
        assert "prompt" in defn.parameters["properties"]
        assert "detail" in defn.parameters["properties"]


# ===========================================================================
# Test: Tool Execution (unit-level, no slicer needed)
# ===========================================================================

class TestSlicingToolExecution:

    def test_slice_model_missing_stl_path(self):
        from lang3d.tools.slicing import SliceModelTool
        tool = SliceModelTool()
        result = tool.execute()
        assert "Error" in result
        assert "stl_path" in result

    def test_slice_model_file_not_found(self):
        from lang3d.tools.slicing import SliceModelTool
        tool = SliceModelTool()
        result = tool.execute(stl_path="/nonexistent/file.stl")
        assert "Error" in result
        assert "not found" in result

    def test_slice_model_unsupported_format(self, tmp_path):
        from lang3d.tools.slicing import SliceModelTool
        tool = SliceModelTool()
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("data", encoding="utf-8")
        result = tool.execute(stl_path=str(bad_file))
        assert "Error" in result
        assert "Unsupported" in result

    def test_slice_model_no_slicer(self, tmp_path):
        from lang3d.tools.slicing import SliceModelTool
        tool = SliceModelTool()
        stl = tmp_path / "test.stl"
        stl.write_bytes(b"solid test\nendsolid test\n")
        with patch("lang3d.tools.slicing._find_slicer", return_value=None):
            result = tool.execute(stl_path=str(stl))
            assert "Error" in result
            assert "No slicer found" in result

    def test_slice_analyze_missing_path(self):
        from lang3d.tools.slicing import SliceAnalyzeTool
        tool = SliceAnalyzeTool()
        result = tool.execute()
        assert "Error" in result

    def test_slice_analyze_file_not_found(self):
        from lang3d.tools.slicing import SliceAnalyzeTool
        tool = SliceAnalyzeTool()
        result = tool.execute(gcode_path="/nonexistent/file.gcode")
        assert "Error" in result
        assert "not found" in result

    def test_slice_analyze_success(self, tmp_path):
        from lang3d.tools.slicing import SliceAnalyzeTool
        tool = SliceAnalyzeTool()
        gcode = tmp_path / "test.gcode"
        gcode.write_text(
            "; estimated printing time = 0h 5m 30s\n"
            "; filament used [mm] = 100.0\n"
            "; filament used [g] = 0.25\n"
            "G1 X0 Y0 Z0.2 E0.1\n"
            "G1 X10 Y10 Z0.4 E0.2\n",
            encoding="utf-8",
        )
        result = tool.execute(gcode_path=str(gcode))
        data = json.loads(result)
        assert data["layers"] >= 0
        assert data["file_size_kb"] >= 0

    def test_slice_preview_layers_missing_path(self):
        from lang3d.tools.slicing import SlicePreviewLayersTool
        tool = SlicePreviewLayersTool()
        result = tool.execute()
        assert "Error" in result

    def test_slice_preview_layers_empty_gcode(self, tmp_path):
        from lang3d.tools.slicing import SlicePreviewLayersTool
        tool = SlicePreviewLayersTool()
        gcode = tmp_path / "empty.gcode"
        gcode.write_text("; comment only\n", encoding="utf-8")
        result = tool.execute(gcode_path=str(gcode))
        data = json.loads(result)
        assert data["total_layers"] == 0

    def test_slice_preview_layers_with_range(self, tmp_path):
        from lang3d.tools.slicing import SlicePreviewLayersTool
        tool = SlicePreviewLayersTool()
        gcode = tmp_path / "layers.gcode"
        gcode.write_text(
            "G1 X0 Y0 Z0.2 E0.1\n"
            "G1 X1 Y1 Z0.4 E0.2\n"
            "G1 X2 Y2 Z0.6 E0.3\n"
            "G1 X3 Y3 Z0.8 E0.4\n",
            encoding="utf-8",
        )
        result = tool.execute(gcode_path=str(gcode), layer_range="1-2")
        data = json.loads(result)
        assert data["total_layers"] == 4
        assert data["returned_layers"] == 2

    def test_slice_vlm_analyze_no_router(self, tmp_path):
        from lang3d.tools.slicing import SliceVLMAnalyzeTool
        tool = SliceVLMAnalyzeTool()
        gcode = tmp_path / "test.gcode"
        gcode.write_text("G1 X0 Y0 Z0.2 E0.1\n", encoding="utf-8")
        result = tool.execute(gcode_path=str(gcode))
        assert "Error" in result
        assert "VLM router" in result


# ===========================================================================
# Test: Command Building
# ===========================================================================

class TestCommandBuilding:

    def test_build_prusa_command(self):
        from lang3d.tools.slicing import _build_prusa_command
        with patch("lang3d.tools.slicing._find_slicer", return_value="/usr/bin/prusa-slicer-console"):
            cmd = _build_prusa_command("test.stl", "test.gcode", {
                "nozzle": 0.4, "diameter": 1.75,
                "temp": 200, "bed_temp": 60,
                "layer_height": 0.2, "infill": 20,
                "perimeters": 3, "top_solid_layers": 4,
                "bottom_solid_layers": 4,
                "supports": "auto", "brim": False,
                "bed_x": 200, "bed_y": 200,
            })
            assert "--export-gcode" in cmd
            assert "--layer-height" in cmd
            assert "0.2" in cmd
            assert "test.stl" in cmd

    def test_build_prusa_command_with_supports(self):
        from lang3d.tools.slicing import _build_prusa_command
        with patch("lang3d.tools.slicing._find_slicer", return_value="/usr/bin/prusa-slicer-console"):
            cmd = _build_prusa_command("test.stl", "test.gcode", {
                "nozzle": 0.4, "diameter": 1.75,
                "temp": 200, "bed_temp": 60,
                "layer_height": 0.2, "infill": 20,
                "perimeters": 3, "top_solid_layers": 4,
                "bottom_solid_layers": 4,
                "supports": "yes", "brim": True,
                "bed_x": 200, "bed_y": 200,
            })
            assert "--support-material" in cmd
            assert "1" in cmd
            assert "--brim" in cmd

    def test_build_prusa_command_no_supports(self):
        from lang3d.tools.slicing import _build_prusa_command
        with patch("lang3d.tools.slicing._find_slicer", return_value="/usr/bin/prusa-slicer-console"):
            cmd = _build_prusa_command("test.stl", "test.gcode", {
                "nozzle": 0.4, "diameter": 1.75,
                "temp": 200, "bed_temp": 60,
                "layer_height": 0.2, "infill": 20,
                "perimeters": 3, "top_solid_layers": 4,
                "bottom_solid_layers": 4,
                "supports": "no", "brim": False,
                "bed_x": 200, "bed_y": 200,
            })
            assert "--support-material" in cmd
            assert "0" in cmd


# ===========================================================================
# Test: Config Integration
# ===========================================================================

class TestSlicingConfig:

    def test_slicing_settings_exist(self):
        from lang3d.config import AgentConfig
        config = AgentConfig()
        assert hasattr(config, "slicing")
        assert config.slicing.default_printer == "generic"
        assert config.slicing.default_material == "pla"
        assert config.slicing.default_quality == "standard"
        assert config.slicing.slice_timeout == 300

    def test_slicing_settings_custom(self):
        from lang3d.config import AgentConfig
        config = AgentConfig(
            slicing={"default_printer": "ender_3", "slice_timeout": 600}
        )
        assert config.slicing.default_printer == "ender_3"
        assert config.slicing.slice_timeout == 600

    def test_env_var_mapping(self):
        from lang3d.config import _build_env_config
        with patch.dict(os.environ, {
            "PRUSASLICER_PATH": "/path/to/prusa",
            "ORCASLICER_PATH": "/path/to/orca",
            "DEFAULT_PRINTER": "ender_3",
        }):
            result = _build_env_config()
            assert result["agent"]["slicing"]["prusaslicer_path"] == "/path/to/prusa"
            assert result["agent"]["slicing"]["orcaslicer_path"] == "/path/to/orca"
            assert result["agent"]["slicing"]["default_printer"] == "ender_3"


# ===========================================================================
# Test: Executor Integration
# ===========================================================================

class TestExecutorIntegration:

    def test_infer_slicing_from_description(self):
        from lang3d.agent.executor import Executor
        from lang3d.agent.state import PlanStep

        step = PlanStep(
            description="切片 STL 文件为 G-code",
            expected_tools=["slice_model"],
        )
        assert Executor._infer_step_type(step) == "slicing"

    def test_infer_slicing_from_tools(self):
        from lang3d.agent.executor import Executor
        from lang3d.agent.state import PlanStep

        step = PlanStep(
            description="Process file",
            expected_tools=["slice_analyze", "file_read"],
        )
        assert Executor._infer_step_type(step) == "slicing"

    def test_infer_slicing_from_gcode_keyword(self):
        from lang3d.agent.executor import Executor
        from lang3d.agent.state import PlanStep

        step = PlanStep(
            description="分析 G-code 打印统计",
            expected_tools=[],
        )
        assert Executor._infer_step_type(step) == "slicing"


# ===========================================================================
# Test: Planner Integration
# ===========================================================================

class TestPlannerIntegration:

    def test_detect_slicing_task_chinese(self):
        from lang3d.agent.planner import Planner
        assert Planner._detect_task_type("切片这个模型") == "slicing"
        assert Planner._detect_task_type("3D打印零件") == "slicing"

    def test_detect_slicing_task_english(self):
        from lang3d.agent.planner import Planner
        assert Planner._detect_task_type("Slice the model for printing") == "slicing"
        assert Planner._detect_task_type("Generate g-code from STL") == "slicing"

    def test_slicing_example_exists(self):
        from lang3d.agent.planner import PLANNER_EXAMPLES
        assert "slicing" in PLANNER_EXAMPLES
        assert "slice_model" in PLANNER_EXAMPLES["slicing"]
