"""OpenAI model backend."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from .base import Message, ModelBackend, ModelResponse, ToolCall, ToolDefinition


class OpenAIBackend(ModelBackend):
    """OpenAI API backend."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        vision_model: str = "gpt-4o",
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, model=model)
        self.vision_model = vision_model
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

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

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
        }

        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        response = self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls: list[ToolCall] = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                import json
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "",
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        )

    def vision(
        self,
        image_path: str,
        prompt: str,
        *,
        max_tokens: int = 2048,
    ) -> str:
        image_data = self._encode_image(image_path)
        media_type = self._get_media_type(image_path)

        response = self.client.chat.completions.create(
            model=self.vision_model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        return response.choices[0].message.content or ""

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
                    "tool_call_id": msg.tool_call_id or "",
                    "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                })
            elif msg.tool_calls:
                import json
                api_messages.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
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

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]
