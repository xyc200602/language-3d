"""Tests for VLM Locate tool (vlm_locate)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from lang3d.tools.base import ToolRegistry
from lang3d.tools.vlm import VLMLocateTool, _parse_elements_json, register_vlm_tools


def test_vlm_locate_registration():
    """vlm_locate is registered with other VLM tools."""
    registry = ToolRegistry()
    mock_router = MagicMock()
    register_vlm_tools(registry, mock_router)
    assert "vlm_locate" in registry.list_tools()


def test_vlm_locate_definition():
    """vlm_locate has correct parameter definition."""
    tool = VLMLocateTool(MagicMock())
    defn = tool.get_definition()
    assert defn.name == "vlm_locate"
    assert "window_title" in defn.parameters["required"]
    assert "target" in defn.parameters["properties"]
    assert "detail" in defn.parameters["properties"]


def test_vlm_locate_no_window():
    """vlm_locate returns error when no window matches."""
    tool = VLMLocateTool(MagicMock())
    result = tool.execute(window_title="NONEXISTENT_WINDOW_12345")
    assert "Error" in result
    assert "No window found" in result


def test_parse_elements_json_array():
    """_parse_elements_json parses a valid JSON array."""
    raw = json.dumps([
        {"name": "File menu", "x": 100, "y": 15, "w": 60, "h": 20, "type": "menu"},
        {"name": "Save button", "x": 250, "y": 50, "w": 80, "h": 30, "type": "button"},
    ])
    elements = _parse_elements_json(raw)
    assert len(elements) == 2
    assert elements[0]["name"] == "File menu"
    assert elements[0]["x"] == 100
    assert elements[1]["name"] == "Save button"
    assert elements[1]["type"] == "button"


def test_parse_elements_json_wrapped():
    """_parse_elements_json extracts array from surrounding text."""
    raw = 'Here are the elements:\n[{"name": "OK", "x": 400, "y": 300, "w": 50, "h": 25, "type": "button"}]\nDone.'
    elements = _parse_elements_json(raw)
    assert len(elements) == 1
    assert elements[0]["name"] == "OK"


def test_parse_elements_json_fallback():
    """_parse_elements_json falls back to line-by-line parsing."""
    raw = "1. Menu Bar - x:10, y:5, w:200, h:25\n2. Toolbar - x:10, y:35, w:300, h:30"
    elements = _parse_elements_json(raw)
    assert len(elements) == 2
    assert elements[0]["name"] == "Menu Bar"
    assert elements[0]["x"] == 10
    assert elements[1]["y"] == 35


def test_parse_elements_json_empty():
    """_parse_elements_json returns empty list for unparseable input."""
    raw = "No elements found in this image."
    elements = _parse_elements_json(raw)
    assert elements == []


def test_vlm_tool_count_increased():
    """VLM tools now total 5 (was 4)."""
    registry = ToolRegistry()
    register_vlm_tools(registry, MagicMock())
    vlm_tools = [t for t in registry.list_tools() if t.startswith(("vlm_", "screen_analyze", "window_analyze", "cad_verify"))]
    assert len(vlm_tools) == 5
