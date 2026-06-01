"""Tests for motion simulation tools and knowledge."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Test: Motion Tool Registration
# ===========================================================================

class TestMotionToolRegistration:
    """Test that motion tools register correctly."""

    def test_register_motion_tools(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.motion import register_motion_tools

        registry = ToolRegistry()
        register_motion_tools(registry)

        expected = ["motion_range", "motion_trajectory", "motion_vlm_analyze"]
        for tool_name in expected:
            assert tool_name in registry.list_tools(), f"Missing tool: {tool_name}"

    def test_register_motion_tools_with_router(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.motion import register_motion_tools

        registry = ToolRegistry()
        mock_router = MagicMock()
        register_motion_tools(registry, router=mock_router, screenshot_dir="/tmp")

        assert "motion_vlm_analyze" in registry.list_tools()

    def test_motion_tools_registered_via_simulation(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.simulation import register_simulation_tools

        registry = ToolRegistry()
        register_simulation_tools(registry)

        # motion tools should be registered alongside simulation tools
        for name in ["motion_range", "motion_trajectory", "motion_vlm_analyze"]:
            assert name in registry.list_tools(), f"Missing: {name}"

    def test_all_motion_tool_definitions_valid(self):
        from lang3d.tools.base import ToolRegistry
        from lang3d.tools.motion import register_motion_tools

        registry = ToolRegistry()
        register_motion_tools(registry)

        defs = registry.get_all_definitions()
        motion_defs = [d for d in defs if d.name in (
            "motion_range", "motion_trajectory", "motion_vlm_analyze",
        )]
        assert len(motion_defs) == 3
        for d in motion_defs:
            assert d.name
            assert d.description
            assert "type" in d.parameters
            assert "properties" in d.parameters


# ===========================================================================
# Test: Motion Tool Definitions
# ===========================================================================

class TestMotionToolDefinitions:

    def test_motion_range_definition(self):
        from lang3d.tools.motion import MotionRangeTool
        tool = MotionRangeTool()
        defn = tool.get_definition()
        assert defn.name == "motion_range"
        assert "document_path" in defn.parameters["properties"]
        assert "joint_name" in defn.parameters["properties"]
        assert "joint_type" in defn.parameters["properties"]
        assert defn.parameters["required"] == ["document_path", "joint_name"]

    def test_motion_trajectory_definition(self):
        from lang3d.tools.motion import MotionTrajectoryTool
        tool = MotionTrajectoryTool()
        defn = tool.get_definition()
        assert defn.name == "motion_trajectory"
        assert "start_angles" in defn.parameters["properties"]
        assert "end_angles" in defn.parameters["properties"]
        assert defn.parameters["required"] == ["document_path", "start_angles", "end_angles"]

    def test_motion_vlm_analyze_definition(self):
        from lang3d.tools.motion import MotionVLMAnalyzeTool
        tool = MotionVLMAnalyzeTool()
        defn = tool.get_definition()
        assert defn.name == "motion_vlm_analyze"
        assert "analysis_type" in defn.parameters["properties"]
        assert "detail" in defn.parameters["properties"]


# ===========================================================================
# Test: MotionSimTool (updated, not STUB)
# ===========================================================================

class TestMotionSimToolUpdated:

    def test_motion_sim_no_longer_stub(self):
        from lang3d.tools.simulation import MotionSimTool
        tool = MotionSimTool()
        assert "STUB" not in tool.description
        assert "STUB" not in tool.name
        defn = tool.get_definition()
        assert defn.name == "motion_sim"
        assert "joint_angles" in defn.parameters["properties"]
        assert "analysis_type" in defn.parameters["properties"]

    def test_motion_sim_missing_document(self):
        from lang3d.tools.simulation import MotionSimTool
        tool = MotionSimTool()
        result = tool.execute(document_path="/nonexistent/model.FCStd")
        assert "Error" in result
        assert "not found" in result

    def test_motion_sim_wrong_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            from lang3d.tools.simulation import MotionSimTool
            tool = MotionSimTool()
            result = tool.execute(document_path=path)
            assert "Error" in result
            assert ".FCStd" in result
        finally:
            Path(path).unlink(missing_ok=True)

    def test_motion_sim_forward_kinematics_no_angles(self):
        with tempfile.NamedTemporaryFile(suffix=".FCStd", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            from lang3d.tools.simulation import MotionSimTool
            tool = MotionSimTool()
            result = tool.execute(
                document_path=path,
                analysis_type="forward_kinematics",
            )
            # Should complain about missing joint_angles
            assert "Error" in result or "joint_angles" in result.lower()
        finally:
            Path(path).unlink(missing_ok=True)


# ===========================================================================
# Test: MotionRangeTool Execution
# ===========================================================================

class TestMotionRangeToolExecution:

    def test_missing_document_error(self):
        from lang3d.tools.motion import MotionRangeTool
        tool = MotionRangeTool()
        result = tool.execute(document_path="/nonexistent.FCStd", joint_name="joint1")
        assert "Error" in result

    def test_wrong_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            from lang3d.tools.motion import MotionRangeTool
            tool = MotionRangeTool()
            result = tool.execute(document_path=path, joint_name="joint1")
            assert "Error" in result
        finally:
            Path(path).unlink(missing_ok=True)

    def test_invalid_joint_type(self):
        with tempfile.NamedTemporaryFile(suffix=".FCStd", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            from lang3d.tools.motion import MotionRangeTool
            tool = MotionRangeTool()
            result = tool.execute(
                document_path=path,
                joint_name="joint1",
                joint_type="invalid_type",
            )
            assert "Error" in result
            assert "Unknown joint type" in result
        finally:
            Path(path).unlink(missing_ok=True)

    def test_invalid_angle_range(self):
        with tempfile.NamedTemporaryFile(suffix=".FCStd", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            from lang3d.tools.motion import MotionRangeTool
            tool = MotionRangeTool()
            result = tool.execute(
                document_path=path,
                joint_name="joint1",
                angle_range=[0.0],  # should be [min, max]
            )
            assert "Error" in result
        finally:
            Path(path).unlink(missing_ok=True)


# ===========================================================================
# Test: MotionTrajectoryTool Execution
# ===========================================================================

class TestMotionTrajectoryToolExecution:

    def test_missing_document_error(self):
        from lang3d.tools.motion import MotionTrajectoryTool
        tool = MotionTrajectoryTool()
        result = tool.execute(
            document_path="/nonexistent.FCStd",
            start_angles={"j1": 0},
            end_angles={"j1": 90},
        )
        assert "Error" in result

    def test_invalid_steps(self):
        with tempfile.NamedTemporaryFile(suffix=".FCStd", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            from lang3d.tools.motion import MotionTrajectoryTool
            tool = MotionTrajectoryTool()
            result = tool.execute(
                document_path=path,
                start_angles={"j1": 0},
                end_angles={"j1": 90},
                steps=0,
            )
            assert "Error" in result
        finally:
            Path(path).unlink(missing_ok=True)


# ===========================================================================
# Test: MotionVLMAnalyzeTool (without router)
# ===========================================================================

class TestMotionVLMAnalyzeTool:

    def test_no_router_error(self):
        from lang3d.tools.motion import MotionVLMAnalyzeTool
        tool = MotionVLMAnalyzeTool()
        result = tool.execute()
        assert "Error" in result
        assert "router" in result.lower() or "VLM" in result


# ===========================================================================
# Test: Motion Knowledge
# ===========================================================================

class TestMotionKnowledge:

    def test_joint_types_exist(self):
        from lang3d.knowledge.simulation import JOINT_TYPES
        assert "revolute" in JOINT_TYPES
        assert "prismatic" in JOINT_TYPES
        assert "fixed" in JOINT_TYPES
        assert "spherical" in JOINT_TYPES

    def test_joint_type_fields(self):
        from lang3d.knowledge.simulation import JOINT_TYPES
        for name, jt in JOINT_TYPES.items():
            assert jt.name
            assert isinstance(jt.dof, int)
            assert jt.motion_type in ("rotational", "translational", "fixed", "spherical")
            assert len(jt.default_range) == 2

    def test_revolute_joint(self):
        from lang3d.knowledge.simulation import JOINT_TYPES
        r = JOINT_TYPES["revolute"]
        assert r.dof == 1
        assert r.motion_type == "rotational"

    def test_prismatic_joint(self):
        from lang3d.knowledge.simulation import JOINT_TYPES
        p = JOINT_TYPES["prismatic"]
        assert p.dof == 1
        assert p.motion_type == "translational"

    def test_fixed_joint(self):
        from lang3d.knowledge.simulation import JOINT_TYPES
        f = JOINT_TYPES["fixed"]
        assert f.dof == 0

    def test_spherical_joint(self):
        from lang3d.knowledge.simulation import JOINT_TYPES
        s = JOINT_TYPES["spherical"]
        assert s.dof == 3


# ===========================================================================
# Test: Motion JSON Parser
# ===========================================================================

class TestMotionJsonParser:

    def test_parse_valid_json(self):
        from lang3d.tools.motion import _parse_motion_json
        raw = '{"objects": [{"name": "link1", "position": [1.0, 2.0, 3.0]}]}'
        result = _parse_motion_json(raw)
        assert "objects" in result
        assert result["objects"][0]["name"] == "link1"

    def test_parse_invalid_json_fallback(self):
        from lang3d.tools.motion import _parse_motion_json
        raw = "Some non-JSON output from FreeCAD"
        result = _parse_motion_json(raw)
        assert "raw" in result


# ===========================================================================
# Test: Config Integration for Motion
# ===========================================================================

class TestMotionConfig:

    def test_simulation_settings_has_cfd_fields(self):
        from lang3d.config import SimulationSettings
        settings = SimulationSettings()
        assert hasattr(settings, "openfoam_path")
        assert settings.openfoam_path == ""
        assert settings.default_fluid == "air"
        assert settings.cfd_timeout == 300
        assert settings.openfoam_mode == "auto"
