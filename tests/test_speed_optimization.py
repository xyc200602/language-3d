"""Tests for Task 29: Speed optimization — fast verification.

Tests cover:
- volume_check operation in fc_batch
- VLM default angles set to isometric,front,top
- System prompt includes fast verification guidance
- Tool descriptions include volume_check
"""

from __future__ import annotations

from typing import Any

import pytest

from lang3d.agent.core import AGENT_SYSTEM_PROMPT
from lang3d.tools.freecad import FCBatchTool, _build_script
from lang3d.tools.vlm import CADVerifyTool


# ============================================================================
# Test: volume_check operation in fc_batch
# ============================================================================

class TestVolumeCheckOperation:
    """Test the volume_check operation type."""

    def test_volume_check_in_description(self):
        """volume_check should be listed in fc_batch description."""
        tool = FCBatchTool()
        assert "volume_check" in tool.description

    def test_volume_check_script_generated(self):
        """volume_check should generate valid FreeCAD script."""
        ops = [
            {"type": "new_doc", "name": "Test"},
            {"type": "make_box", "length": 50, "width": 30, "height": 20, "name": "Box"},
            {"type": "volume_check"},
        ]
        script = _build_script(ops)
        assert "VOLUME_CHECK_JSON" in script
        assert "_vc_results" in script
        assert "_vc_pass" in script

    def test_volume_check_with_dims(self):
        """volume_check with dimension checks."""
        ops = [
            {"type": "new_doc", "name": "Test"},
            {"type": "make_box", "length": 50, "width": 30, "height": 20, "name": "Box"},
            {"type": "volume_check", "checks": {
                "dimensions": {"length": 50, "width": 30, "height": 20},
                "tolerance_mm": 1.0,
            }},
        ]
        script = _build_script(ops)
        assert "expected 50" in script
        assert "expected 30" in script
        assert "expected 20" in script
        assert "tolerance_mm" not in script  # tolerance is baked as number
        assert "1.0" in script  # tolerance value

    def test_volume_check_with_volume_limits(self):
        """volume_check with min/max volume constraints."""
        ops = [
            {"type": "new_doc", "name": "Test"},
            {"type": "volume_check", "checks": {
                "min_volume": 100,
                "max_volume": 50000,
            }},
        ]
        script = _build_script(ops)
        assert "100" in script
        assert "50000" in script
        assert "below minimum" in script
        assert "above maximum" in script

    def test_volume_check_with_path(self):
        """volume_check with external file path."""
        ops = [
            {"type": "volume_check", "path": "C:/tmp/test.FCStd"},
        ]
        script = _build_script(ops)
        assert "openDocument" in script
        assert "closeDocument" in script

    def test_volume_check_without_path_uses_current_doc(self):
        """volume_check without path uses current doc."""
        ops = [
            {"type": "new_doc", "name": "Test"},
            {"type": "volume_check"},
        ]
        script = _build_script(ops)
        assert "_vc_doc = doc" in script

    def test_volume_check_outputs_json(self):
        """volume_check should print structured JSON output."""
        ops = [{"type": "volume_check"}]
        script = _build_script(ops)
        assert "VOLUME_CHECK_JSON:" in script
        assert "json.dumps" in script.replace("_vc_json.dumps", "json.dumps")


# ============================================================================
# Test: VLM default angles
# ============================================================================

class TestVLMDefaultAngles:
    """Test that CADVerifyTool defaults to multi-angle verification."""

    def test_default_angles_is_multi(self):
        """Default angles parameter should be multi-angle, not empty."""
        import inspect
        sig = inspect.signature(CADVerifyTool.execute)
        angles_param = sig.parameters["angles"]
        assert angles_param.default == "isometric,front,top"

    def test_default_angles_not_empty(self):
        """Default should not be empty string (which disables multi-angle)."""
        import inspect
        sig = inspect.signature(CADVerifyTool.execute)
        angles_param = sig.parameters["angles"]
        assert angles_param.default != ""


# ============================================================================
# Test: System prompt
# ============================================================================

class TestSystemPrompt:
    """Test that the system prompt includes speed optimization guidance."""

    def test_prompt_mentions_volume_check(self):
        assert "volume_check" in AGENT_SYSTEM_PROMPT

    def test_prompt_mentions_level_based_verification(self):
        assert "Level 1-2" in AGENT_SYSTEM_PROMPT
        assert "Level 3+" in AGENT_SYSTEM_PROMPT

    def test_prompt_mentions_assembly_solve(self):
        assert "assembly_solve" in AGENT_SYSTEM_PROMPT

    def test_prompt_mentions_ik_solve(self):
        assert "ik_solve" in AGENT_SYSTEM_PROMPT

    def test_prompt_mentions_interference_for_assembly(self):
        """Assembly verification should prefer interference_check over VLM."""
        assert "interference_check" in AGENT_SYSTEM_PROMPT

    def test_prompt_mentions_default_angles(self):
        """Prompt should mention the default multi-angle verification."""
        assert "isometric,front,top" in AGENT_SYSTEM_PROMPT

    def test_prompt_has_workspace_placeholder(self):
        assert "{workspace}" in AGENT_SYSTEM_PROMPT


# ============================================================================
# Test: Backward compatibility
# ============================================================================

class TestBackwardCompatibility:
    """Ensure existing operations still work with the changes."""

    def test_object_info_still_works(self):
        """object_info should not be affected by volume_check addition."""
        ops = [
            {"type": "new_doc", "name": "Test"},
            {"type": "make_box", "length": 10, "width": 10, "height": 10, "name": "Box"},
            {"type": "object_info", "object": "Box"},
        ]
        script = _build_script(ops)
        assert "Volume:" in script
        assert "Dims:" in script

    def test_all_original_ops_still_generate(self):
        """All original operation types should still generate valid scripts."""
        original_types = [
            "new_doc", "make_box", "make_cylinder", "make_sphere",
            "make_cone", "boolean", "move", "save", "export_stl",
            "export_step", "object_info", "delete_object",
        ]
        for op_type in original_types:
            # Each should at least not crash _build_script
            # (some need extra params but should produce output)
            if op_type == "new_doc":
                ops = [{"type": "new_doc", "name": "Test"}]
            elif op_type == "make_box":
                ops = [{"type": "new_doc"}, {"type": "make_box", "length": 10, "width": 10, "height": 10}]
            elif op_type == "make_cylinder":
                ops = [{"type": "new_doc"}, {"type": "make_cylinder", "radius": 5, "height": 10}]
            elif op_type == "make_sphere":
                ops = [{"type": "new_doc"}, {"type": "make_sphere", "radius": 5}]
            elif op_type == "make_cone":
                ops = [{"type": "new_doc"}, {"type": "make_cone", "radius1": 5, "height": 10}]
            elif op_type == "boolean":
                ops = [
                    {"type": "new_doc"},
                    {"type": "make_box", "length": 10, "width": 10, "height": 10, "name": "A"},
                    {"type": "make_box", "length": 5, "width": 5, "height": 5, "name": "B"},
                    {"type": "boolean", "operation": "cut", "object1": "A", "object2": "B"},
                ]
            elif op_type == "move":
                ops = [
                    {"type": "new_doc"},
                    {"type": "make_box", "length": 10, "width": 10, "height": 10, "name": "Box"},
                    {"type": "move", "object": "Box", "dx": 5, "dy": 0, "dz": 0},
                ]
            elif op_type == "save":
                ops = [{"type": "new_doc"}, {"type": "save", "path": "/tmp/test.FCStd"}]
            elif op_type == "export_stl":
                ops = [{"type": "new_doc"}, {"type": "export_stl", "path": "/tmp/test.stl"}]
            elif op_type == "export_step":
                ops = [{"type": "new_doc"}, {"type": "export_step", "path": "/tmp/test.step"}]
            elif op_type == "object_info":
                ops = [
                    {"type": "new_doc"},
                    {"type": "make_box", "length": 10, "width": 10, "height": 10, "name": "Box"},
                    {"type": "object_info", "object": "Box"},
                ]
            elif op_type == "delete_object":
                ops = [
                    {"type": "new_doc"},
                    {"type": "make_box", "length": 10, "width": 10, "height": 10, "name": "Box"},
                    {"type": "delete_object", "object": "Box"},
                ]
            else:
                continue

            script = _build_script(ops)
            assert len(script) > 0, f"Script for {op_type} is empty"
            assert "FreeCAD" in script, f"Script for {op_type} missing FreeCAD import"
