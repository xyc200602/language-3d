"""GLM model backend using OpenAI-compatible Coding Plan API."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .base import Message, ModelBackend, ModelResponse, ToolCall, ToolDefinition
from .cache import SemanticCache, get_cache
from .retry import RetryConfig, call_with_retry


class GLMBackend(ModelBackend):
    """GLM (Zhipu AI) backend via OpenAI-compatible Coding Plan API.

    Uses the OpenAI SDK pointed at https://open.bigmodel.cn/api/coding/paas/v4
    which provides an OpenAI-compatible interface for GLM models.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://open.bigmodel.cn/api/coding/paas/v4",
        model: str = "GLM-5.1",
        vision_model: str = "GLM-4V-Flash",
        # Alternative endpoint for models not on Coding Plan
        alt_base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        retry_config: RetryConfig | None = None,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, model=model, retry_config=retry_config)
        self.vision_model = vision_model
        self.alt_base_url = alt_base_url
        self._client: OpenAI | None = None
        self._alt_client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    @property
    def alt_client(self) -> OpenAI:
        """Client for the standard API endpoint (supports more vision models)."""
        if self._alt_client is None:
            self._alt_client = OpenAI(
                api_key=self.api_key,
                base_url=self.alt_base_url,
            )
        return self._alt_client

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        *,
        max_tokens: int = 100000,
        temperature: float = 0.7,
        system: str | None = None,
    ) -> ModelResponse:
        """Send a chat completion request via OpenAI-compatible API."""
        api_messages = self._convert_messages(messages, system)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
        }

        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        # Check cache for deterministic requests (temperature=0)
        cache = get_cache()
        cache_key = None
        if temperature < 0.01:
            cache_key = SemanticCache.make_key(
                api_messages,
                kwargs.get("tools"),
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        response = call_with_retry(
            self.client.chat.completions.create,
            **kwargs,
            retry_config=self.retry_config,
        )

        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls: list[ToolCall] = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError, AttributeError):
                    arguments = {}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        result = ModelResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "",
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        )

        # Store in cache if this was a deterministic request
        if cache_key is not None:
            cache.put(cache_key, result)

        return result

    # Vision models with their max_tokens caps and preferred endpoint.
    # 4.6V models need the standard API; 4V models work on Coding Plan.
    VISION_MODEL_CAPS: dict[str, int] = {
        "GLM-4V-Flash": 1024,
        "GLM-4V": 1024,
        "GLM-4V-Plus": 2048,
        "GLM-4.6V": 16384,
        "GLM-4.6V-Flash": 16384,
        "GLM-5V-Turbo": 16384,
    }

    def vision(
        self,
        image_path: str,
        prompt: str,
        *,
        max_tokens: int = 100000,
        model: str | None = None,
    ) -> str:
        """Analyze an image using a GLM vision model.

        Args:
            image_path: Path to the image.
            prompt: Analysis prompt.
            max_tokens: Desired max output tokens (auto-capped per model).
            model: Override vision model name. Defaults to self.vision_model.
        """
        vision_model = model or self.vision_model
        image_data = self._encode_image(image_path)
        media_type = self._get_media_type(image_path)

        # Cap max_tokens to the model's actual limit
        model_cap = self.VISION_MODEL_CAPS.get(vision_model, 1024)
        effective_max_tokens = min(max_tokens, model_cap)

        # 4.6V/5V models need the standard API endpoint
        use_alt = any(
            vision_model.startswith(p) for p in ("GLM-4.6", "GLM-5V")
        )
        api_client = self.alt_client if use_alt else self.client

        response = call_with_retry(
            api_client.chat.completions.create,
            model=vision_model,
            max_tokens=effective_max_tokens,
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
            retry_config=self.retry_config,
        )

        return response.choices[0].message.content or ""

    def _convert_messages(
        self, messages: list[Message], system: str | None = None
    ) -> list[dict[str, Any]]:
        """Convert our Message format to OpenAI API format."""
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
        """Convert ToolDefinition to OpenAI tool format."""
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
