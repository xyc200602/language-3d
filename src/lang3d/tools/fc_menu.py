"""FreeCAD GUI menu automation tools.

Provides high-level tools that combine vlm_locate + gui_click + gui_type
to automate FreeCAD GUI operations (click menus, create sketches, extrude, etc.).

These tools are useful when fc_batch (API-based) cannot achieve the desired
operation, such as interacting with FreeCAD workbenches, task panels, or
features not exposed through the Python API.

Typical usage:
  1. fc_open_gui to launch FreeCAD
  2. fc_menu to automate GUI operations
  3. cad_verify to verify results

Comparison with fc_batch:
  - fc_batch: API-driven, precise, fast, headless. Best for parametric modeling.
  - fc_menu: GUI-driven, visual, interactive. Best for workbench-specific features,
    complex dialogs, or when the API route is unavailable.
"""

from __future__ import annotations

import time
from typing import Any

from ..models.base import ToolDefinition
from .base import Tool


# FreeCAD menu bar items (English names for menu navigation)
_MENUS = {
    "file": "File",
    "edit": "Edit",
    "view": "View",
    "insert": "Insert",
    "part": "Part",
    "mesh": "Mesh",
    "open_scad": "OpenSCAD",
    "macro": "Macro",
    "windows": "Windows",
    "help": "Help",
}


def _find_element_by_name(elements: list[dict], name: str) -> dict | None:
    """Find an element by name (case-insensitive partial match)."""
    name_lower = name.lower()
    for el in elements:
        if name_lower in el["name"].lower():
            return el
    return None


def _locate_elements(locate_tool: Any, target: str) -> list[dict]:
    """Use VLMLocateTool to find UI elements and return parsed list."""
    import json
    import re

    result = locate_tool.execute(window_title="FreeCAD", target=target, detail="standard")

    # Parse elements from the result text
    elements = []
    # Look for numbered lines like "  1. ElementName (type) @ (x, y)"
    for line in result.split("\n"):
        m = re.match(
            r'\s*\d+\.\s*(.+?)\s*\(\w+\)\s*@\s*\((\d+),\s*(\d+)\)',
            line,
        )
        if m:
            elements.append({
                "name": m.group(1).strip(),
                "x": int(m.group(2)),
                "y": int(m.group(3)),
            })
    return elements


def _click_element(click_tool: Any, x: int, y: int, pause: float = 1.0) -> str:
    """Click at coordinates and wait."""
    return click_tool.execute(x=x, y=y, pause=pause)


def _type_text(type_tool: Any, text: str, pause: float = 0.5) -> str:
    """Type text and wait."""
    return type_tool.execute(text=text, pause=pause)


def _press_key(press_key_tool: Any, key: str, pause: float = 0.5) -> str:
    """Press a key and wait."""
    return press_key_tool.execute(key=key, pause=pause)


class FCMenuClickTool(Tool):
    """Click a FreeCAD menu item by name using VLM-assisted localization.

    This tool:
    1. Uses vlm_locate to find the menu/button/element
    2. Clicks on the best match

    If the element is not found on the first attempt, this tool returns
    an error with suggestions. The Agent can then try alternative names
    or use gui_click directly with coordinates.
    """

    name = "fc_menu"
    description = (
        "Click a FreeCAD GUI element by its name. Uses VLM to locate the element on screen. "
        "Examples: 'File', 'Part', 'Create Primitives', 'Save', 'Box', 'New'. "
        "Works with menu items, toolbar buttons, sidebar items, and dialog buttons. "
        "Prefer fc_batch for parametric modeling; use this for GUI-only features."
    )

    def __init__(self, locate_tool: Any = None, click_tool: Any = None) -> None:
        self._locate_tool = locate_tool
        self._click_tool = click_tool

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Name of the GUI element to click. "
                            "Examples: 'File', 'Part', 'Box', 'Save', 'New', "
                            "'Create Primitives', 'Cylinder', 'Fillet'"
                        ),
                    },
                    "element_type": {
                        "type": "string",
                        "description": (
                            "Type of element to help narrow search: "
                            "'menu' (top menu bar), 'toolbar' (toolbar buttons), "
                            "'sidebar' (combo view panel), 'dialog' (dialog button), "
                            "'any' (search everywhere). Default: 'any'"
                        ),
                    },
                    "double_click": {
                        "type": "boolean",
                        "description": "Whether to double-click (default: false)",
                    },
                    "wait": {
                        "type": "number",
                        "description": "Seconds to wait after clicking (default: 1.0)",
                    },
                },
                "required": ["name"],
            },
        )

    def execute(
        self,
        *,
        name: str,
        element_type: str = "any",
        double_click: bool = False,
        wait: float = 1.0,
        **kwargs: Any,
    ) -> str:
        # Lazy-init tools
        if not self._locate_tool or not self._click_tool:
            return "Error: fc_menu requires vlm_locate and gui_click tools to be available"

        # Determine search target based on element_type
        target_map = {
            "menu": "menu bar items at the top (File, Edit, View, Part, etc.) and toolbar buttons",
            "toolbar": "toolbar buttons and icons at the top of the window",
            "sidebar": "combo view panel items (model tree, properties)",
            "dialog": "dialog buttons and input fields",
            "any": "all interactive elements (buttons, menus, toolbar icons, sidebar items)",
        }
        target = target_map.get(element_type, target_map["any"])

        # Step 1: Locate elements
        elements = _locate_elements(self._locate_tool, target)

        if not elements:
            return (
                f"No UI elements found for '{name}'. "
                f"Searched for: {target}\n"
                f"Try: use vlm_locate directly with a more specific target, "
                f"or use gui_click with exact coordinates."
            )

        # Step 2: Find best match
        match = _find_element_by_name(elements, name)

        if not match:
            # List available elements to help the Agent
            available = [el["name"] for el in elements[:15]]
            return (
                f"Element '{name}' not found. Available elements:\n"
                + "\n".join(f"  - {n}" for n in available)
                + f"\n\nTry a different name or use gui_click directly."
            )

        # Step 3: Click
        clicks = 2 if double_click else 1
        result = self._click_tool.execute(
            x=match["x"], y=match["y"], clicks=clicks, pause=wait
        )

        return (
            f"Clicked '{match['name']}' at ({match['x']}, {match['y']}) "
            f"({'double-click' if double_click else 'click'})\n"
            f"{result}"
        )


class FCMenuWorkflowTool(Tool):
    """Execute a named FreeCAD GUI workflow (predefined sequence of GUI operations).

    Provides common modeling workflows as single tool calls, internally using
    VLM locate + click + type to navigate FreeCAD's GUI.
    """

    name = "fc_menu_workflow"
    description = (
        "Execute a predefined FreeCAD GUI workflow by name. "
        "Available workflows:\n"
        "- 'new_part': Create a new Part design (Part workbench > Create Primitives)\n"
        "- 'add_box': Add a box via Part menu\n"
        "- 'add_cylinder': Add a cylinder via Part menu\n"
        "- 'add_sphere': Add a sphere via Part menu\n"
        "- 'save_file': Save current document via File > Save\n"
        "- 'save_as': Save as new file via File > Save As\n"
        "- 'export_mesh': Export as mesh via File > Export\n"
        "Prefer fc_batch for precise parametric modeling; use this for GUI automation testing."
    )

    def __init__(self, locate_tool: Any = None, click_tool: Any = None,
                 type_tool: Any = None, press_key_tool: Any = None) -> None:
        self._locate_tool = locate_tool
        self._click_tool = click_tool
        self._type_tool = type_tool
        self._press_key_tool = press_key_tool

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "workflow": {
                        "type": "string",
                        "description": "Workflow name: 'new_part', 'add_box', 'add_cylinder', 'add_sphere', 'save_file', 'save_as', 'export_mesh'",
                    },
                    "params": {
                        "type": "object",
                        "description": "Optional parameters for the workflow (e.g. file path for save_as)",
                    },
                },
                "required": ["workflow"],
            },
        )

    def execute(
        self,
        *,
        workflow: str,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        params = params or {}

        if not all([self._locate_tool, self._click_tool, self._type_tool, self._press_key_tool]):
            return "Error: fc_menu_workflow requires vlm_locate, gui_click, gui_type, and gui_press_key tools"

        steps_log = []

        try:
            if workflow == "new_part":
                return self._workflow_new_part(steps_log)
            elif workflow == "add_box":
                return self._workflow_add_primitive("box", params, steps_log)
            elif workflow == "add_cylinder":
                return self._workflow_add_primitive("cylinder", params, steps_log)
            elif workflow == "add_sphere":
                return self._workflow_add_primitive("sphere", params, steps_log)
            elif workflow == "save_file":
                return self._workflow_save(params, steps_log)
            elif workflow == "save_as":
                return self._workflow_save_as(params, steps_log)
            elif workflow == "export_mesh":
                return self._workflow_export(params, steps_log)
            else:
                return (
                    f"Unknown workflow: '{workflow}'. "
                    f"Available: new_part, add_box, add_cylinder, add_sphere, "
                    f"save_file, save_as, export_mesh"
                )
        except Exception as e:
            steps_log.append(f"ERROR: {e}")
            return "Workflow failed:\n" + "\n".join(f"  {s}" for s in steps_log)

    def _click_menu_item(self, menu_name: str, steps_log: list) -> bool:
        """Click a top-level menu item (File, Part, etc.)."""
        elements = _locate_elements(self._locate_tool, "top menu bar items")
        match = _find_element_by_name(elements, menu_name)
        if not match:
            steps_log.append(f"FAIL: Menu '{menu_name}' not found")
            return False
        self._click_tool.execute(x=match["x"], y=match["y"], pause=1.0)
        steps_log.append(f"OK: Clicked menu '{menu_name}' at ({match['x']}, {match['y']})")
        return True

    def _click_dropdown_item(self, item_name: str, steps_log: list) -> bool:
        """Click an item in a dropdown/submenu after menu is open."""
        # Wait for dropdown to appear
        time.sleep(0.5)
        elements = _locate_elements(self._locate_tool, "dropdown menu items and submenus")
        match = _find_element_by_name(elements, item_name)
        if not match:
            steps_log.append(f"FAIL: Dropdown item '{item_name}' not found")
            return False
        self._click_tool.execute(x=match["x"], y=match["y"], pause=1.0)
        steps_log.append(f"OK: Clicked '{item_name}' at ({match['x']}, {match['y']})")
        return True

    def _workflow_new_part(self, steps_log: list) -> str:
        """Create a new Part design."""
        # Click Part menu
        if not self._click_menu_item("Part", steps_log):
            return "Workflow 'new_part' failed:\n" + "\n".join(f"  {s}" for s in steps_log)

        # Click Create Primitives or similar
        if not self._click_dropdown_item("Primitives", steps_log):
            # Try alternative
            steps_log.append("Trying alternative: looking for 'Create Primitives'")
            if not self._click_dropdown_item("Create Primitives", steps_log):
                return "Workflow 'new_part' failed:\n" + "\n".join(f"  {s}" for s in steps_log)

        steps_log.append("OK: New Part workflow completed")
        return "Workflow 'new_part' completed:\n" + "\n".join(f"  {s}" for s in steps_log)

    def _workflow_add_primitive(self, shape: str, params: dict, steps_log: list) -> str:
        """Add a primitive shape (box, cylinder, sphere) via Part menu."""
        # Click Part menu
        if not self._click_menu_item("Part", steps_log):
            return f"Workflow 'add_{shape}' failed:\n" + "\n".join(f"  {s}" for s in steps_log)

        # Click Primitives > shape
        if not self._click_dropdown_item("Primitives", steps_log):
            if not self._click_dropdown_item("Create Primitives", steps_log):
                return f"Workflow 'add_{shape}' failed:\n" + "\n".join(f"  {s}" for s in steps_log)

        time.sleep(0.5)

        # Find and click the specific shape in the primitives dialog
        elements = _locate_elements(self._locate_tool, "buttons, list items, and tabs")
        shape_match = _find_element_by_name(elements, shape)
        if shape_match:
            self._click_tool.execute(x=shape_match["x"], y=shape_match["y"], pause=0.5)
            steps_log.append(f"OK: Selected '{shape}'")
        else:
            steps_log.append(f"WARN: '{shape}' not found in primitives dialog, attempting keyboard nav")
            # Try pressing down arrow to navigate
            self._press_key_tool.execute(key="down", pause=0.3)
            self._press_key_tool.execute(key="enter", pause=0.5)

        # Try to click Create or OK button
        time.sleep(0.5)
        elements = _locate_elements(self._locate_tool, "buttons in the current dialog")
        create_btn = _find_element_by_name(elements, "Create") or _find_element_by_name(elements, "OK")
        if create_btn:
            self._click_tool.execute(x=create_btn["x"], y=create_btn["y"], pause=1.0)
            steps_log.append("OK: Clicked Create/OK")
        else:
            steps_log.append("WARN: Create/OK button not found, trying Enter key")
            self._press_key_tool.execute(key="enter", pause=1.0)

        steps_log.append(f"OK: add_{shape} workflow completed")
        return f"Workflow 'add_{shape}' completed:\n" + "\n".join(f"  {s}" for s in steps_log)

    def _workflow_save(self, params: dict, steps_log: list) -> str:
        """Save via File > Save."""
        if not self._click_menu_item("File", steps_log):
            return "Workflow 'save_file' failed:\n" + "\n".join(f"  {s}" for s in steps_log)
        if not self._click_dropdown_item("Save", steps_log):
            # Try Ctrl+S
            steps_log.append("Trying keyboard shortcut Ctrl+S")
            from .gui_action import GUIHotkeyTool
            # Fallback: we can't easily call hotkey here, suggest Agent does it
            steps_log.append("WARN: Could not click Save menu item. Use gui_hotkey with 'ctrl+s' instead.")
            return "Workflow 'save_file' partial:\n" + "\n".join(f"  {s}" for s in steps_log)

        steps_log.append("OK: Save workflow completed")
        return "Workflow 'save_file' completed:\n" + "\n".join(f"  {s}" for s in steps_log)

    def _workflow_save_as(self, params: dict, steps_log: list) -> str:
        """Save As via File > Save As."""
        if not self._click_menu_item("File", steps_log):
            return "Workflow 'save_as' failed:\n" + "\n".join(f"  {s}" for s in steps_log)
        if not self._click_dropdown_item("Save As", steps_log):
            return "Workflow 'save_as' failed:\n" + "\n".join(f"  {s}" for s in steps_log)

        file_path = params.get("path", "")
        if file_path:
            # Wait for save dialog
            time.sleep(1.0)
            # Try to find the filename input field and type the path
            elements = _locate_elements(self._locate_tool, "input fields and text boxes")
            path_input = _find_element_by_name(elements, "File name") or _find_element_by_name(elements, "file")
            if path_input:
                self._click_tool.execute(x=path_input["x"], y=path_input["y"], pause=0.3)
                # Select all existing text and replace
                self._press_key_tool.execute(key="home", pause=0.1)
                # Type the path
                self._type_tool.execute(text=file_path, pause=0.5)
                self._press_key_tool.execute(key="enter", pause=1.0)
                steps_log.append(f"OK: Typed path '{file_path}' and pressed Enter")
            else:
                steps_log.append(f"WARN: File name input not found. Path '{file_path}' not entered.")

        steps_log.append("OK: Save As workflow completed")
        return "Workflow 'save_as' completed:\n" + "\n".join(f"  {s}" for s in steps_log)

    def _workflow_export(self, params: dict, steps_log: list) -> str:
        """Export via File > Export."""
        if not self._click_menu_item("File", steps_log):
            return "Workflow 'export_mesh' failed:\n" + "\n".join(f"  {s}" for s in steps_log)
        if not self._click_dropdown_item("Export", steps_log):
            return "Workflow 'export_mesh' failed:\n" + "\n".join(f"  {s}" for s in steps_log)

        file_path = params.get("path", "")
        if file_path:
            time.sleep(1.0)
            elements = _locate_elements(self._locate_tool, "input fields and text boxes")
            path_input = _find_element_by_name(elements, "File name") or _find_element_by_name(elements, "file")
            if path_input:
                self._click_tool.execute(x=path_input["x"], y=path_input["y"], pause=0.3)
                self._type_tool.execute(text=file_path, pause=0.5)
                self._press_key_tool.execute(key="enter", pause=1.0)
                steps_log.append(f"OK: Typed export path '{file_path}'")
            else:
                steps_log.append(f"WARN: File name input not found for export.")

        steps_log.append("OK: Export workflow completed")
        return "Workflow 'export_mesh' completed:\n" + "\n".join(f"  {s}" for s in steps_log)


def register_fc_menu_tools(registry: Any, locate_tool: Any = None,
                           click_tool: Any = None, type_tool: Any = None,
                           press_key_tool: Any = None) -> None:
    """Register FreeCAD menu automation tools."""
    registry.register(FCMenuClickTool(locate_tool=locate_tool, click_tool=click_tool))
    registry.register(FCMenuWorkflowTool(
        locate_tool=locate_tool,
        click_tool=click_tool,
        type_tool=type_tool,
        press_key_tool=press_key_tool,
    ))
