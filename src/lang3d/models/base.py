"""Abstract base class for model backends."""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Message:
    """A chat message."""

    role: str  # "system", "user", "assistant", "tool"
    content: str | list[dict[str, Any]] = ""
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


@dataclass
class ToolCall:
    """A tool call from the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelResponse:
    """Response from a model backend."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class ToolDefinition:
    """Definition of a tool that the model can call."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


class ModelBackend(ABC):
    """Abstract base class for LLM/VLM backends."""

    def __init__(self, api_key: str = "", base_url: str = "", model: str = "") -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        *,
        max_tokens: int = 100000,
        temperature: float = 0.7,
        system: str | None = None,
    ) -> ModelResponse:
        """Send a chat completion request."""
        ...

    @abstractmethod
    def vision(
        self,
        image_path: str | Path,
        prompt: str,
        *,
        max_tokens: int = 100000,
        model: str | None = None,
    ) -> str:
        """Analyze an image with a vision model.

        Args:
            image_path: Path to the image file.
            prompt: Text prompt for analysis.
            max_tokens: Maximum tokens in response.
            model: Override vision model name. If None, use default.
        """
        ...

    def _encode_image(self, image_path: str | Path) -> str:
        """Encode an image file to base64."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return base64.b64encode(path.read_bytes()).decode("utf-8")

    def _get_media_type(self, image_path: str | Path) -> str:
        """Get MIME type from file extension."""
        suffix = Path(image_path).suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_map.get(suffix, "image/png")
