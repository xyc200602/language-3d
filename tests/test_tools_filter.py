"""Tests for tool relevance filtering."""

from __future__ import annotations

from unittest.mock import MagicMock

from lang3d.tools.base import (
    STEP_TOOL_CATEGORIES,
    TOOL_CATEGORIES,
    Tool,
    ToolRegistry,
)
from lang3d.models.base import ToolDefinition


def _make_tool(name: str) -> Tool:
    """Create a mock tool with given name."""
    tool = MagicMock(spec=Tool)
    tool.name = name
    tool.get_definition.return_value = ToolDefinition(
        name=name,
        description=f"Tool {name}",
        parameters={"type": "object", "properties": {}},
    )
    return tool


def _register_all_tools(registry: ToolRegistry) -> None:
    """Register a comprehensive set of mock tools covering all categories."""
    all_tools = []
    for names in TOOL_CATEGORIES.values():
        all_tools.extend(names)
    for name in set(all_tools):
        registry.register(_make_tool(name))


class TestToolCategories:
    """Verify category and mapping constants."""

    def test_categories_exist(self):
        assert "file_ops" in TOOL_CATEGORIES
        assert "shell" in TOOL_CATEGORIES
        assert "vlm" in TOOL_CATEGORIES
        assert "freecad" in TOOL_CATEGORIES
        assert "gui" in TOOL_CATEGORIES

    def test_step_mappings_exist(self):
        assert "modeling" in STEP_TOOL_CATEGORIES
        assert "verification" in STEP_TOOL_CATEGORIES
        assert "simulation" in STEP_TOOL_CATEGORIES
        assert "general" in STEP_TOOL_CATEGORIES


class TestGetRelevantDefinitions:
    """Tool relevance filtering tests."""

    def setup_method(self):
        self.registry = ToolRegistry()
        _register_all_tools(self.registry)

    def test_modeling_step_fewer_than_all(self):
        modeling = self.registry.get_relevant_definitions("modeling")
        all_defs = self.registry.get_all_definitions()
        # Modeling should return fewer tools than all
        assert len(modeling) < len(all_defs)
        # Should include fc_batch
        names = [d.name for d in modeling]
        assert "fc_batch" in names

    def test_verification_step(self):
        defs = self.registry.get_relevant_definitions("verification")
        names = [d.name for d in defs]
        assert "cad_verify" in names

    def test_unknown_type_returns_all(self):
        defs = self.registry.get_relevant_definitions("nonexistent_type")
        all_defs = self.registry.get_all_definitions()
        assert len(defs) == len(all_defs)

    def test_extra_tools_included(self):
        defs = self.registry.get_relevant_definitions("file_ops", extra_tools=["cad_verify"])
        names = [d.name for d in defs]
        assert "cad_verify" in names

    def test_general_includes_core_tools(self):
        defs = self.registry.get_relevant_definitions("general")
        names = [d.name for d in defs]
        # General should include a broad set
        assert "bash" in names or "file_read" in names
