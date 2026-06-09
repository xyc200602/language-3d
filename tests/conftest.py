"""Shared test fixtures for Language-3D tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def tmp_workspace():
    """Provide a temporary workspace directory."""
    with tempfile.TemporaryDirectory(prefix="lang3d_test_") as d:
        yield Path(d)


@pytest.fixture
def mock_router():
    """Provide a mock ModelRouter."""
    router = MagicMock()
    router.chat.return_value = MagicMock(
        content="mock response",
        tool_calls=[],
        usage={"input_tokens": 0, "output_tokens": 0},
    )
    return router


@pytest.fixture
def mock_tools():
    """Provide a mock ToolRegistry."""
    registry = MagicMock()
    registry.get_relevant_definitions.return_value = []
    registry.execute.return_value = "mock tool result"
    return registry
