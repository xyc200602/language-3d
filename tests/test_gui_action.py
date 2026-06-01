"""Tests for GUI action tools (gui_action.py).

Unit tests verify tool registration and definitions without actually
moving the mouse or clicking (safe for CI).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from lang3d.tools.base import ToolRegistry
from lang3d.tools.gui_action import (
    GUIClickTool,
    GUIDragTool,
    GUIHotkeyTool,
    GUIMousePosTool,
    GUIPressKeyTool,
    GUIScreenshotTool,
    GUIScrollTool,
    GUITypeTool,
    register_gui_action_tools,
)


def test_gui_action_tool_registration():
    """Test that all gui_action tools can be registered."""
    registry = ToolRegistry()
    register_gui_action_tools(registry, screenshot_dir=tempfile.gettempdir())

    expected = [
        "gui_click", "gui_type", "gui_hotkey", "gui_press_key",
        "gui_screenshot", "gui_mouse_pos", "gui_drag", "gui_scroll",
    ]
    for tool_name in expected:
        assert tool_name in registry.list_tools(), f"Missing tool: {tool_name}"


def test_gui_action_tool_count():
    """Test exact count of registered tools."""
    registry = ToolRegistry()
    register_gui_action_tools(registry)
    gui_tools = [t for t in registry.list_tools() if t.startswith("gui_")]
    assert len(gui_tools) == 8


def test_gui_action_tool_definitions():
    """Test that all gui_action tool definitions are valid."""
    registry = ToolRegistry()
    register_gui_action_tools(registry)

    defs = registry.get_all_definitions()
    gui_defs = [d for d in defs if d.name.startswith("gui_")]
    assert len(gui_defs) == 8

    for d in gui_defs:
        assert d.name
        assert d.description
        assert "type" in d.parameters
        assert "properties" in d.parameters


def test_gui_click_requires_xy():
    """Test that gui_click definition requires x and y."""
    tool = GUIClickTool()
    defn = tool.get_definition()
    assert "x" in defn.parameters["required"]
    assert "y" in defn.parameters["required"]


def test_gui_type_requires_text():
    """Test that gui_type definition requires text."""
    tool = GUITypeTool()
    defn = tool.get_definition()
    assert "text" in defn.parameters["required"]


def test_gui_hotkey_requires_keys():
    """Test that gui_hotkey definition requires keys."""
    tool = GUIHotkeyTool()
    defn = tool.get_definition()
    assert "keys" in defn.parameters["required"]


def test_gui_press_key_requires_key():
    """Test that gui_press_key definition requires key."""
    tool = GUIPressKeyTool()
    defn = tool.get_definition()
    assert "key" in defn.parameters["required"]


def test_gui_drag_requires_coordinates():
    """Test that gui_drag definition requires start and end coordinates."""
    tool = GUIDragTool()
    defn = tool.get_definition()
    for param in ("start_x", "start_y", "end_x", "end_y"):
        assert param in defn.parameters["required"]


def test_gui_scroll_requires_clicks():
    """Test that gui_scroll definition requires clicks."""
    tool = GUIScrollTool()
    defn = tool.get_definition()
    assert "clicks" in defn.parameters["required"]


def test_gui_mouse_pos_no_required_params():
    """Test that gui_mouse_pos has no required parameters."""
    tool = GUIMousePosTool()
    defn = tool.get_definition()
    assert defn.parameters["required"] == []


def test_gui_mouse_pos_executes():
    """Test that gui_mouse_pos runs and returns position info."""
    tool = GUIMousePosTool()
    result = tool.execute()
    assert "Mouse position" in result
    assert "Screen" in result


def test_gui_click_out_of_bounds():
    """Test that gui_click returns error for out-of-bounds coordinates."""
    tool = GUIClickTool()
    result = tool.execute(x=99999, y=99999)
    assert "Error" in result
    assert "out of screen bounds" in result


def test_gui_screenshot_executes():
    """Test that gui_screenshot captures fullscreen without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tool = GUIScreenshotTool(screenshot_dir=tmpdir)
        result = tool.execute()
        assert "Screenshot saved" in result or "Error" not in result


def test_gui_screenshot_invalid_region():
    """Test that gui_screenshot returns error for invalid region format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tool = GUIScreenshotTool(screenshot_dir=tmpdir)
        result = tool.execute(region="invalid")
        assert "Error" in result
