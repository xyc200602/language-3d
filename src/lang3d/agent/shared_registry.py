"""Thread-safe wrapper around ToolRegistry for concurrent sub-agent access."""

from __future__ import annotations

import threading
from typing import Any

from ..models.base import ToolDefinition
from ..tools.base import ToolRegistry


class SharedToolRegistry:
    """Thread-safe wrapper around ToolRegistry.

    Read operations use a shared ``_read_lock``.  Execute operations use
    per-tool locks so that different tools can execute concurrently while
    the same tool is still serialised.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._read_lock = threading.Lock()
        self._execute_locks: dict[str, threading.Lock] = {}
        self._execute_locks_lock = threading.Lock()

    def _get_execute_lock(self, tool_name: str) -> threading.Lock:
        with self._execute_locks_lock:
            if tool_name not in self._execute_locks:
                self._execute_locks[tool_name] = threading.Lock()
            return self._execute_locks[tool_name]

    def register(self, tool: Any) -> None:
        with self._read_lock:
            self._registry.register(tool)

    def get(self, name: str) -> Any:
        with self._read_lock:
            return self._registry.get(name)

    def get_all_definitions(self) -> list[ToolDefinition]:
        with self._read_lock:
            return self._registry.get_all_definitions()

    def get_relevant_definitions(
        self,
        step_type: str,
        extra_tools: list[str] | None = None,
    ) -> list[ToolDefinition]:
        """Get tool definitions relevant to a step type (thread-safe)."""
        with self._read_lock:
            return self._registry.get_relevant_definitions(step_type, extra_tools=extra_tools)

    def list_tools(self) -> list[str]:
        with self._read_lock:
            return self._registry.list_tools()

    def execute(self, tool_name: str, **kwargs: Any) -> str:
        """Execute a tool under a per-tool lock for thread safety."""
        lock = self._get_execute_lock(tool_name)
        with lock:
            return self._registry.execute(tool_name, **kwargs)
