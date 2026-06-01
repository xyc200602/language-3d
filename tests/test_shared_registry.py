"""Tests for SharedToolRegistry thread safety."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from lang3d.agent.shared_registry import SharedToolRegistry
from lang3d.tools.base import ToolRegistry


class TestSharedToolRegistry:
    def test_delegates_to_inner_registry(self):
        inner = ToolRegistry()
        shared = SharedToolRegistry(inner)

        # Register a mock tool on the inner registry
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.execute.return_value = "ok"
        shared.register(mock_tool)

        assert shared.get("test_tool") is not None
        assert "test_tool" in shared.list_tools()

    def test_execute_thread_safety(self):
        """Verify that concurrent executes are serialized."""
        inner = ToolRegistry()
        shared = SharedToolRegistry(inner)

        # Register a tool that records concurrent access
        execution_log: list[int] = []
        lock = threading.Lock()

        class TrackingTool:
            name = "track"
            description = "tracking tool"

            def get_definition(self):
                return MagicMock()

            def execute(self, **kwargs):
                tid = threading.current_thread().ident
                with lock:
                    execution_log.append(tid)
                # Simulate some work
                import time
                time.sleep(0.01)
                return f"done-{tid}"

        inner.register(TrackingTool())

        # Run many concurrent executes
        threads = []
        for _ in range(10):
            t = threading.Thread(target=lambda: shared.execute("track"))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All 10 should have executed
        assert len(execution_log) == 10

    def test_get_all_definitions(self):
        inner = ToolRegistry()
        shared = SharedToolRegistry(inner)

        mock_tool = MagicMock()
        mock_tool.name = "t1"
        mock_tool.get_definition.return_value = MagicMock()
        shared.register(mock_tool)

        defs = shared.get_all_definitions()
        assert len(defs) == 1
