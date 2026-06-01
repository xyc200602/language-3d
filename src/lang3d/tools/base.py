"""Tool base class and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from ..models.base import ToolDefinition


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
