"""Ollama local model backend."""

from __future__ import annotations

import json
from typing import Any

import httpx

from .base import Message, ModelBackend, ModelResponse, ToolCall, ToolDefinition


class OllamaBackend(ModelBackend):
    """Ollama local model backend."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        vision_model: str = "llava",
    ) -> None:
        super().__init__(api_key="", base_url=base_url, model=model)
        self.vision_model = vision_model

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        *,
        max_tokens: int = 100000,
        temperature: float = 0.7,
        system: str | None = None,
    ) -> ModelResponse:
        api_messages = self._convert_messages(messages, system)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if tools:
            payload["tools"] = [self._convert_tool(t) for t in tools]

        response = httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()

        content = data.get("message", {}).get("content", "")
        tool_calls: list[ToolCall] = []

        # Ollama tool calls are in the message
        if "tool_calls" in data.get("message", {}):
            for tc in data["message"]["tool_calls"]:
                func = tc.get("function", {})
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        name=func.get("name", ""),
                        arguments=func.get("arguments", {}),
                    )
                )

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="stop",
        )

    def vision(
        self,
        image_path: str,
        prompt: str,
        *,
        max_tokens: int = 2048,
    ) -> str:
        image_data = self._encode_image(image_path)

        payload = {
            "model": self.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_data],
                }
            ],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }

        response = httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()

        return data.get("message", {}).get("content", "")

    def _convert_messages(
        self, messages: list[Message], system: str | None = None
    ) -> list[dict[str, Any]]:
        api_messages: list[dict[str, Any]] = []

        if system:
            api_messages.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role == "tool":
                api_messages.append({
                    "role": "tool",
                    "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                })
            elif msg.tool_calls:
                api_messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            }
                        }
                        for tc in msg.tool_calls
                    ],
                })
            else:
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                })

        return api_messages

    def _convert_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
