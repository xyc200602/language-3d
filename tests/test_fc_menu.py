"""Tests for FreeCAD menu automation tools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from lang3d.tools.base import ToolRegistry
from lang3d.tools.fc_menu import (
    FCMenuClickTool,
    FCMenuWorkflowTool,
    _find_element_by_name,
    register_fc_menu_tools,
)


class TestFindElementByName:
    def test_exact_match(self):
        elements = [
            {"name": "File", "x": 50, "y": 10},
            {"name": "Edit", "x": 100, "y": 10},
            {"name": "Part", "x": 200, "y": 10},
        ]
        result = _find_element_by_name(elements, "File")
        assert result is not None
        assert result["name"] == "File"
        assert result["x"] == 50

    def test_case_insensitive(self):
        elements = [{"name": "File", "x": 50, "y": 10}]
        result = _find_element_by_name(elements, "file")
        assert result is not None
        assert result["name"] == "File"

    def test_partial_match(self):
        elements = [
            {"name": "Create Primitives", "x": 200, "y": 100},
            {"name": "Save", "x": 100, "y": 50},
        ]
        result = _find_element_by_name(elements, "primitives")
        assert result is not None
        assert result["name"] == "Create Primitives"

    def test_no_match(self):
        elements = [{"name": "File", "x": 50, "y": 10}]
        result = _find_element_by_name(elements, "nonexistent")
        assert result is None

    def test_empty_list(self):
        result = _find_element_by_name([], "File")
        assert result is None


class TestFCMenuClickTool:
    def test_registration(self):
        registry = ToolRegistry()
        registry.register(FCMenuClickTool())
        assert "fc_menu" in registry.list_tools()

    def test_definition(self):
        tool = FCMenuClickTool()
        defn = tool.get_definition()
        assert defn.name == "fc_menu"
        assert "name" in defn.parameters["properties"]
        assert defn.parameters["required"] == ["name"]

    def test_no_tools_error(self):
        tool = FCMenuClickTool()
        result = tool.execute(name="File")
        assert "Error" in result

    def test_locate_finds_nothing(self):
        mock_locate = MagicMock()
        mock_locate.execute.return_value = "No elements parsed"
        mock_click = MagicMock()

        tool = FCMenuClickTool(locate_tool=mock_locate, click_tool=mock_click)
        result = tool.execute(name="File")
        assert "No UI elements found" in result

    def test_element_not_found(self):
        mock_locate = MagicMock()
        mock_locate.execute.return_value = (
            "Found 3 UI elements:\n"
            "  1. Edit (button) @ (100, 10)\n"
            "  2. View (button) @ (200, 10)\n"
            "  3. Help (button) @ (400, 10)\n"
        )
        mock_click = MagicMock()

        tool = FCMenuClickTool(locate_tool=mock_locate, click_tool=mock_click)
        result = tool.execute(name="File")
        assert "not found" in result
        assert "Edit" in result  # Lists available elements

    def test_successful_click(self):
        mock_locate = MagicMock()
        mock_locate.execute.return_value = (
            "Found 2 UI elements:\n"
            "  1. File (menu) @ (50, 10)\n"
            "  2. Edit (menu) @ (100, 10)\n"
        )
        mock_click = MagicMock()
        mock_click.execute.return_value = "Clicked at (50, 10)"

        tool = FCMenuClickTool(locate_tool=mock_locate, click_tool=mock_click)
        result = tool.execute(name="File")
        assert "Clicked" in result
        mock_click.execute.assert_called_once_with(x=50, y=10, clicks=1, pause=1.0)

    def test_double_click(self):
        mock_locate = MagicMock()
        mock_locate.execute.return_value = (
            "Found 1 UI elements:\n"
            "  1. Box (button) @ (300, 200)\n"
        )
        mock_click = MagicMock()
        mock_click.execute.return_value = "Clicked at (300, 200)"

        tool = FCMenuClickTool(locate_tool=mock_locate, click_tool=mock_click)
        result = tool.execute(name="Box", double_click=True)
        assert "double-click" in result
        mock_click.execute.assert_called_once_with(x=300, y=200, clicks=2, pause=1.0)

    def test_element_type_filter(self):
        mock_locate = MagicMock()
        mock_locate.execute.return_value = (
            "Found 1 UI elements:\n"
            "  1. File (menu) @ (50, 10)\n"
        )
        mock_click = MagicMock()
        mock_click.execute.return_value = "Clicked at (50, 10)"

        tool = FCMenuClickTool(locate_tool=mock_locate, click_tool=mock_click)
        result = tool.execute(name="File", element_type="menu")
        assert "Clicked" in result
        # Verify vlm_locate was called with menu-specific target
        call_args = mock_locate.execute.call_args
        assert "menu bar" in call_args.kwargs.get("target", call_args[1].get("target", ""))


class TestFCMenuWorkflowTool:
    def test_registration(self):
        registry = ToolRegistry()
        registry.register(FCMenuWorkflowTool())
        assert "fc_menu_workflow" in registry.list_tools()

    def test_definition(self):
        tool = FCMenuWorkflowTool()
        defn = tool.get_definition()
        assert defn.name == "fc_menu_workflow"
        assert "workflow" in defn.parameters["properties"]
        assert defn.parameters["required"] == ["workflow"]

    def test_unknown_workflow(self):
        mock_locate = MagicMock()
        mock_click = MagicMock()
        mock_type = MagicMock()
        mock_press = MagicMock()

        tool = FCMenuWorkflowTool(
            locate_tool=mock_locate,
            click_tool=mock_click,
            type_tool=mock_type,
            press_key_tool=mock_press,
        )
        result = tool.execute(workflow="nonexistent")
        assert "Unknown workflow" in result

    def test_missing_tools_error(self):
        tool = FCMenuWorkflowTool()
        result = tool.execute(workflow="new_part")
        assert "Error" in result


class TestRegisterFcMenuTools:
    def test_register(self):
        registry = ToolRegistry()
        mock_locate = MagicMock()
        mock_click = MagicMock()
        mock_type = MagicMock()
        mock_press = MagicMock()

        register_fc_menu_tools(
            registry,
            locate_tool=mock_locate,
            click_tool=mock_click,
            type_tool=mock_type,
            press_key_tool=mock_press,
        )
        assert "fc_menu" in registry.list_tools()
        assert "fc_menu_workflow" in registry.list_tools()

    def test_definitions_valid(self):
        registry = ToolRegistry()
        register_fc_menu_tools(registry)

        defs = registry.get_all_definitions()
        menu_defs = [d for d in defs if d.name in ("fc_menu", "fc_menu_workflow")]
        assert len(menu_defs) == 2
        for d in menu_defs:
            assert d.name
            assert d.description
            assert "type" in d.parameters
            assert "properties" in d.parameters
