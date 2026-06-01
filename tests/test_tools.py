"""Tests for tool system."""

from __future__ import annotations

import tempfile
from pathlib import Path

from lang3d.tools.base import ToolRegistry
from lang3d.tools.bash import BashTool, PythonExecTool, register_bash_tools
from lang3d.tools.file_ops import (
    FileEditTool,
    FileGlobTool,
    FileReadTool,
    FileSearchTool,
    FileWriteTool,
    ListDirTool,
    register_file_tools,
)


def test_tool_registry_register():
    registry = ToolRegistry()
    registry.register(FileReadTool())
    assert "file_read" in registry.list_tools()


def test_tool_registry_execute():
    registry = ToolRegistry()
    registry.register(FileReadTool())

    result = registry.execute("file_read", path="/nonexistent/file.txt")
    assert "Error" in result


def test_tool_registry_unknown():
    registry = ToolRegistry()
    result = registry.execute("unknown_tool")
    assert "not found" in result


def test_file_write_read():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = str(Path(tmpdir) / "test.txt")
        registry = ToolRegistry()
        registry.register(FileWriteTool())
        registry.register(FileReadTool())

        result = registry.execute("file_write", path=filepath, content="Hello World")
        assert "Successfully" in result

        result = registry.execute("file_read", path=filepath)
        assert "Hello World" in result


def test_file_edit():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = str(Path(tmpdir) / "edit_test.txt")
        registry = ToolRegistry()
        registry.register(FileWriteTool())
        registry.register(FileReadTool())
        registry.register(FileEditTool())

        registry.execute("file_write", path=filepath, content="Hello World\nSecond Line")

        result = registry.execute("file_edit", path=filepath, old_text="Hello", new_text="Hi")
        assert "Successfully" in result

        result = registry.execute("file_read", path=filepath)
        assert "Hi World" in result


def test_list_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "file1.txt").write_text("test")
        Path(tmpdir, "subdir").mkdir()

        registry = ToolRegistry()
        registry.register(ListDirTool())

        result = registry.execute("list_dir", path=tmpdir)
        assert "file1.txt" in result
        assert "subdir" in result


def test_bash_tool():
    registry = ToolRegistry()
    registry.register(BashTool())

    result = registry.execute("bash", command="echo hello")
    assert "hello" in result


def test_python_exec():
    registry = ToolRegistry()
    registry.register(PythonExecTool())

    result = registry.execute("python_exec", code="print(2 + 3)")
    assert "5" in result


def test_register_all_file_tools():
    registry = ToolRegistry()
    register_file_tools(registry)
    assert "file_read" in registry.list_tools()
    assert "file_write" in registry.list_tools()
    assert "file_edit" in registry.list_tools()
    assert "file_search" in registry.list_tools()
    assert "file_glob" in registry.list_tools()
    assert "list_dir" in registry.list_tools()


def test_register_all_bash_tools():
    registry = ToolRegistry()
    register_bash_tools(registry)
    assert "bash" in registry.list_tools()
    assert "python_exec" in registry.list_tools()


def test_tool_definitions():
    registry = ToolRegistry()
    register_file_tools(registry)

    defs = registry.get_all_definitions()
    assert len(defs) > 0
    for d in defs:
        assert d.name
        assert d.description
        assert d.parameters


def test_freecad_tool_registration():
    """Test that FreeCAD tools can be registered (without FreeCAD installed)."""
    from lang3d.tools.freecad import register_freecad_tools

    registry = ToolRegistry()
    register_freecad_tools(registry)

    expected_tools = [
        "fc_new_doc", "fc_make_box", "fc_make_cylinder", "fc_make_sphere",
        "fc_make_cone", "fc_boolean", "fc_move", "fc_rotate",
        "fc_cylinder_with_hole", "fc_plate_with_holes",
        "fc_fillet", "fc_chamfer", "fc_save",
        "fc_export_stl", "fc_export_step", "fc_status",
        "fc_object_info", "fc_delete_object", "fc_script",
    ]
    for tool_name in expected_tools:
        assert tool_name in registry.list_tools(), f"Missing tool: {tool_name}"


def test_freecad_tool_definitions():
    """Test that FreeCAD tool definitions are valid."""
    from lang3d.tools.freecad import register_freecad_tools

    registry = ToolRegistry()
    register_freecad_tools(registry)

    defs = registry.get_all_definitions()
    fc_defs = [d for d in defs if d.name.startswith("fc_")]
    assert len(fc_defs) == 23
    for d in fc_defs:
        assert d.name.startswith("fc_")
        assert d.description
        assert "type" in d.parameters
        assert "properties" in d.parameters


def test_freecad_status_without_install():
    """Test that fc_status gives a clear error when FreeCAD is not installed."""
    from lang3d.tools.freecad import FCStatusTool

    tool = FCStatusTool()
    result = tool.execute()
    # Should return an error message, not crash
    assert isinstance(result, str)
    assert len(result) > 0


def test_screen_tool_registration():
    """Test that screen tools can be registered."""
    from lang3d.tools.screen import register_screen_tools

    registry = ToolRegistry()
    register_screen_tools(registry)

    expected = ["screen_capture", "window_capture", "list_windows"]
    for tool_name in expected:
        assert tool_name in registry.list_tools(), f"Missing tool: {tool_name}"


def test_screen_tool_definitions():
    """Test that screen tool definitions are valid."""
    from lang3d.tools.screen import register_screen_tools

    registry = ToolRegistry()
    register_screen_tools(registry)

    defs = registry.get_all_definitions()
    screen_defs = [d for d in defs if d.name in ("screen_capture", "window_capture", "list_windows")]
    assert len(screen_defs) == 3
    for d in screen_defs:
        assert d.name
        assert d.description
        assert "type" in d.parameters


def test_vlm_tool_registration():
    """Test that VLM tools can be registered."""
    from unittest.mock import MagicMock
    from lang3d.tools.vlm import register_vlm_tools

    registry = ToolRegistry()
    mock_router = MagicMock()
    register_vlm_tools(registry, mock_router)

    expected = ["vlm_analyze", "screen_analyze", "window_analyze", "cad_verify", "vlm_locate"]
    for tool_name in expected:
        assert tool_name in registry.list_tools(), f"Missing tool: {tool_name}"


def test_vlm_tool_definitions():
    """Test that VLM tool definitions are valid."""
    from unittest.mock import MagicMock
    from lang3d.tools.vlm import register_vlm_tools

    registry = ToolRegistry()
    mock_router = MagicMock()
    register_vlm_tools(registry, mock_router)

    defs = registry.get_all_definitions()
    vlm_defs = [d for d in defs if d.name in ("vlm_analyze", "screen_analyze", "window_analyze", "cad_verify", "vlm_locate")]
    assert len(vlm_defs) == 5
    for d in vlm_defs:
        assert d.name
        assert d.description
        assert "type" in d.parameters
        assert "properties" in d.parameters


def test_screen_capture_executes():
    """Test that screen_capture tool runs without error."""
    import tempfile

    registry = ToolRegistry()
    from lang3d.tools.screen import register_screen_tools
    register_screen_tools(registry, screenshot_dir=tempfile.gettempdir())

    result = registry.execute("screen_capture")
    assert "Screenshot saved to:" in result or "Error" not in result


def test_list_windows_executes():
    """Test that list_windows tool runs without error."""
    registry = ToolRegistry()
    from lang3d.tools.screen import register_screen_tools
    register_screen_tools(registry)

    result = registry.execute("list_windows")
    assert "Found" in result
    assert "visible windows:" in result
