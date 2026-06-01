"""Thread-safe wrapper around ToolRegistry for concurrent sub-agent access."""

from __future__ import annotations

import threading
from typing import Any

from ..models.base import ToolDefinition
from ..tools.base import ToolRegistry


class SharedToolRegistry:
    """Thread-safe wrapper around ToolRegistry.

    All tool executions are serialized via a threading.Lock to prevent
    concurrent access issues. Read operations are also protected for
    consistency.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._lock = threading.Lock()

    def register(self, tool: Any) -> None:
        with self._lock:
            self._registry.register(tool)

    def get(self, name: str) -> Any:
        with self._lock:
            return self._registry.get(name)

    def get_all_definitions(self) -> list[ToolDefinition]:
        with self._lock:
            return self._registry.get_all_definitions()

    def list_tools(self) -> list[str]:
        with self._lock:
            return self._registry.list_tools()

    def execute(self, tool_name: str, **kwargs: Any) -> str:
        """Execute a tool under lock to ensure thread safety."""
        with self._lock:
            return self._registry.execute(tool_name, **kwargs)
