"""Tool base class and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from ..models.base import ToolDefinition

# Tool categories: map category name -> tool name prefixes / exact names
TOOL_CATEGORIES: dict[str, list[str]] = {
    "file_ops": [
        "file_read", "file_write", "file_edit", "file_search", "file_glob", "list_dir",
    ],
    "shell": ["bash", "python_exec"],
    "screen": ["screen_capture", "window_capture", "list_windows"],
    "vlm": ["vlm_analyze", "screen_analyze", "window_analyze", "cad_verify", "vlm_locate"],
    "freecad": [
        "fc_batch", "fc_open_gui", "fc_close_gui", "fc_set_camera",
        "fc_menu", "fc_menu_workflow", "fc_get_scene",
    ],
    "gui": [
        "gui_click", "gui_type", "gui_hotkey", "gui_press_key",
        "gui_screenshot", "gui_drag", "gui_scroll", "gui_mouse_pos",
    ],
    "simulation": [
        "fea_run", "fea_visualize", "fea_vlm_analyze",
        "interference_check", "tolerance_analysis",
        "motion_sim", "motion_range", "motion_trajectory", "motion_vlm_analyze",
    ],
    "motion": [
        "motion_sim", "motion_range", "motion_trajectory", "motion_vlm_analyze",
    ],
    "cfd": ["cfd_run", "cfd_vlm_analyze"],
    "solidworks": ["sw_create_part", "sw_open_gui", "sw_close_gui", "sw_export"],
    "part_library": [
        "part_search", "part_get", "part_generate", "part_list",
        "part_import", "part_save", "part_analyze_print", "part_assemble",
    ],
    "assembly": [
        "assembly_solve",
        "ik_solve",
    ],
    "slicing": [
        "slice_model", "slice_analyze", "slice_preview_layers", "slice_vlm_analyze",
    ],
    "actuator": [
        "actuator_select", "actuator_analyze", "actuator_power_budget",
    ],
    "code_gen": [
        "gen_firmware", "gen_wiring_diagram", "gen_test_sequence",
    ],
    "bom": [
        "gen_bom",
    ],
    "assembly_doc": [
        "gen_assembly_guide",
    ],
    "print_optimize": [
        "print_optimize",
    ],
    "quality": [
        "quality_check",
    ],
    "iteration": [
        "iteration_design",
    ],
}

# Map step types to the tool categories they need
STEP_TOOL_CATEGORIES: dict[str, list[str]] = {
    "modeling": ["freecad", "vlm", "gui", "file_ops", "part_library", "assembly"],
    "verification": ["vlm", "gui", "screen"],
    "simulation": ["simulation", "freecad", "vlm", "gui"],
    "cfd": ["cfd", "freecad", "vlm", "gui"],
    "motion": ["motion", "freecad", "vlm", "gui"],
    "slicing": ["slicing", "vlm", "gui"],
    "file_ops": ["file_ops", "shell"],
    "general": ["file_ops", "shell", "screen", "vlm", "freecad", "gui"],
}


class Tool(ABC):
    """Base class for all agent tools."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def get_definition(self) -> ToolDefinition:
        """Return the tool definition for the model."""
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Execute the tool with given arguments. Returns result as string."""
        ...

    def __repr__(self) -> str:
        return f"Tool({self.name})"


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def register_function(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Callable[..., str],
    ) -> None:
        """Register a simple function as a tool."""

        class FunctionTool(Tool):
            def get_definition(self) -> ToolDefinition:
                return ToolDefinition(
                    name=name,
                    description=description,
                    parameters=parameters,
                )

            def execute(self, **kwargs: Any) -> str:
                return func(**kwargs)

        ft = FunctionTool()
        ft.name = name
        ft.description = description
        self.register(ft)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all_definitions(self) -> list[ToolDefinition]:
        """Get tool definitions for all registered tools."""
        return [tool.get_definition() for tool in self._tools.values()]

    def get_relevant_definitions(
        self,
        step_type: str,
        extra_tools: list[str] | None = None,
    ) -> list[ToolDefinition]:
        """Get tool definitions relevant to a step type.

        Falls back to all definitions for unknown step types.
        """
        categories = STEP_TOOL_CATEGORIES.get(step_type)
        if categories is None:
            return self.get_all_definitions()

        # Build set of relevant tool names from categories
        relevant_names: set[str] = set()
        for cat in categories:
            for prefix in TOOL_CATEGORIES.get(cat, []):
                # Match by exact name or prefix
                for name in self._tools:
                    if name == prefix or name.startswith(prefix.split("*")[0]):
                        relevant_names.add(name)

        # Always include explicitly requested tools
        if extra_tools:
            for t in extra_tools:
                if t in self._tools:
                    relevant_names.add(t)
                else:
                    # Try prefix match
                    for name in self._tools:
                        if name.startswith(t.split("*")[0]):
                            relevant_names.add(name)

        return [self._tools[n].get_definition() for n in relevant_names if n in self._tools]

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def execute(self, tool_name: str, **kwargs: Any) -> str:
        """Execute a tool by name."""
        tool = self.get(tool_name)
        if tool is None:
            return f"Error: Tool '{tool_name}' not found. Available: {', '.join(self.list_tools())}"
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return f"Error executing {tool_name}: {e}"
