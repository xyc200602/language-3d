"""GLM model backend using OpenAI-compatible Coding Plan API."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from .base import Message, ModelBackend, ModelResponse, ToolCall, ToolDefinition
from .cache import SemanticCache, get_cache
from .retry import RetryConfig, call_with_retry

logger = logging.getLogger(__name__)


def _salvage_answer_from_reasoning(reasoning: str) -> str:
    """Best-effort: recover a final answer from GLM-5.2 reasoning_content.

    GLM-5.2 is a reasoning model that writes its chain-of-thought to
    ``reasoning_content`` and the final answer to ``content``. In rare cases
    (or when truncated) ``content`` is empty but the model left a usable
    answer at the tail of the reasoning — typically after the last numbered
    step, often wrapped in backticks or quotes.

    This salvages that tail. It is deliberately conservative: it only fires
    when ``content`` is empty, and it strips leading reasoning markers so a
    half-finished thought doesn't get treated as JSON. Returns "" when
    nothing usable is found (caller then treats it as a genuine empty reply).
    """
    if not reasoning:
        return ""
    text = reasoning.strip()
    # Reasoning steps look like "1. ... 2. ... 3. Final: <answer>".
    # Take everything after the last step marker.
    import re
    steps = re.split(r"\n\d+\.\s", text)
    tail = steps[-1].strip() if steps else text
    # If the tail contains a JSON object/code block, extract the densest part.
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", tail, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"(\{.*\}|\[.*\])", tail, re.S)
    if m:
        return m.group(1).strip()
    # Otherwise return the tail only if it's short and answer-like (not a
    # long unfinished sentence). Long tails are likely truncated mid-thought.
    if 0 < len(tail) <= 200:
        return tail
    return ""


class GLMBackend(ModelBackend):
    """GLM (Zhipu AI) backend via OpenAI-compatible Coding Plan API.

    Uses the OpenAI SDK pointed at https://open.bigmodel.cn/api/coding/paas/v4
    which provides an OpenAI-compatible interface for GLM models.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://open.bigmodel.cn/api/coding/paas/v4",
        model: str = "GLM-5.2",
        vision_model: str = "GLM-4.6V",
        # Alternative endpoint for models not on Coding Plan
        alt_base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        retry_config: RetryConfig | None = None,
        # Per-request timeout in seconds.  The OpenAI SDK default is 600s,
        # which makes a hung upstream look like a permanently frozen e2e
        # loop.  180s is enough for the longest legitimate responses
        # (assembly JSON generation) while failing fast on a dead upstream.
        request_timeout: float = 180.0,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, model=model, retry_config=retry_config)
        self.vision_model = vision_model
        self.alt_base_url = alt_base_url
        self.request_timeout = request_timeout
        self._client: OpenAI | None = None
        self._alt_client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.request_timeout,
            )
        return self._client

    @property
    def alt_client(self) -> OpenAI:
        """Client for the standard API endpoint (supports more vision models)."""
        if self._alt_client is None:
            self._alt_client = OpenAI(
                api_key=self.api_key,
                base_url=self.alt_base_url,
                timeout=self.request_timeout,
            )
        return self._alt_client

    def _streamed_create(self, **kwargs: Any) -> Any:
        """Run a chat completion with STREAMING and reassemble a full response.

        GLM-5.2 is a reasoning model: non-streaming ``create`` blocks until
        the ENTIRE chain-of-thought finishes, which for complex prompts
        (assembly-gen, ~3k-token prompt) can take many minutes — long enough
        that the SDK's connection timeout never trips (tokens keep trickling,
        so the socket stays alive) and the e2e loop looks permanently frozen.

        Streaming delivers tokens incrementally, keeping the connection
        observably active and letting the SDK's read timeout fire if the
        upstream truly stalls. We accumulate ``content`` + ``reasoning_content``
        + ``tool_calls`` across chunks and return an object shaped like the
        non-streaming ``ChatCompletion`` so the rest of ``chat`` is unchanged.

        Falls back to the non-streaming call if the server rejects streaming.
        """
        kwargs = dict(kwargs)
        kwargs["stream"] = True
        kwargs.setdefault("stream_options", {"include_usage": True})

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: dict[str, Any] = {}
        model_name = kwargs.get("model", self.model)

        try:
            stream = self.client.chat.completions.create(**kwargs)
            for chunk in stream:
                if not getattr(chunk, "choices", None):
                    # Final usage-only chunk (include_usage).
                    _u = getattr(chunk, "usage", None)
                    if _u is not None:
                        usage = dict(_u) if isinstance(_u, dict) else {
                            k: getattr(_u, k) for k in ("prompt_tokens",
                            "completion_tokens", "total_tokens")
                            if hasattr(_u, k)
                        }
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if getattr(delta, "content", None):
                    content_parts.append(delta.content)
                _rc = getattr(delta, "reasoning_content", None)
                if _rc:
                    reasoning_parts.append(_rc)
                if getattr(delta, "tool_calls", None):
                    for tc in delta.tool_calls:
                        idx = tc.index
                        slot = tool_calls.setdefault(idx, {
                            "id": "", "type": "function",
                            "function": {"name": "", "arguments": ""},
                        })
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function and tc.function.name:
                            slot["function"]["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["function"]["arguments"] += tc.function.arguments
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
        except Exception as e:
            # If streaming is unsupported/failed, fall back to blocking call
            # so this never breaks callers that worked before.
            logger.warning(
                "Streaming create failed (%s); falling back to non-stream", e,
            )
            fb = dict(kwargs)
            fb.pop("stream", None)
            fb.pop("stream_options", None)
            return call_with_retry(
                self.client.chat.completions.create,
                retry_config=self.retry_config,
                **fb,
            )

        # Assemble a ChatCompletion-like object. We use a lightweight wrapper
        # rather than constructing the SDK's dataclass (its constructor is
        # version-sensitive) — chat() only reads .choices[0].message.content /
        # .reasoning_content / .finish_reason / .tool_calls.
        class _Msg:
            def __init__(self) -> None:
                self.content = "".join(content_parts)
                self.reasoning_content = "".join(reasoning_parts)
                self.tool_calls = [
                    tool_calls[i] for i in sorted(tool_calls)
                ] or None

        class _Choice:
            def __init__(self) -> None:
                self.message = _Msg()
                self.finish_reason = finish_reason

        class _Resp:
            def __init__(self) -> None:
                self.choices = [_Choice()]
                self.usage = usage
                self.model = model_name

        return _Resp()

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        *,
        max_tokens: int = 100000,
        temperature: float = 0.7,
        system: str | None = None,
        # GLM-5.2 is a reasoning model that emits a long chain-of-thought in
        # ``reasoning_content`` before the answer. For STRUCTURED outputs
        # (assembly JSON, fix commands) the reasoning is pure latency — the
        # model just needs to emit the JSON. Passing thinking={"type":
        # "disabled"} via extra_body skips reasoning and returns the answer
        # in seconds instead of minutes. Callers that WANT reasoning (open-
        # ended planning) leave this None. ``extra_body`` lets callers pass
        # any other vendor-specific params too.
        thinking: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
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

        # Vendor-specific params (GLM thinking control, etc.) go via extra_body.
        merged_extra: dict[str, Any] = dict(extra_body or {})
        if thinking is not None:
            merged_extra["thinking"] = thinking
        if merged_extra:
            kwargs["extra_body"] = merged_extra

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

        response = self._streamed_create(**kwargs)

        if not response.choices:
            return ModelResponse(content="", tool_calls=[], finish_reason="empty", usage={})

        choice = response.choices[0]
        content = choice.message.content or ""

        # GLM-5.2 is a reasoning model: it emits its chain-of-thought in
        # ``reasoning_content`` and only writes the final answer to
        # ``content`` once reasoning completes. When max_tokens is too small,
        # the whole budget is spent reasoning and ``content`` comes back
        # empty with ``finish_reason == "length"``. This manifested as the
        # recurring "GLM returns empty body" failures (commit b2959c8 retried
        # blindly; this fix addresses the cause: give reasoning room).
        finish_reason = choice.finish_reason or ""
        # Guard against non-string reasoning_content (e.g. MagicMock in unit
        # tests auto-creates the attribute). Only real reasoning text counts.
        _raw_reasoning = getattr(choice.message, "reasoning_content", "")
        reasoning = _raw_reasoning if isinstance(_raw_reasoning, str) else ""
        if not content and finish_reason == "length" and reasoning:
            # Reasoning was truncated before the answer. Retry with a doubled
            # token budget so the model can finish thinking AND emit content.
            bigger = min(int(max_tokens) * 2, 65536)
            logger.info(
                "GLM-5.2 reasoning truncated (finish=length, %d reasoning "
                "tokens) — retrying with max_tokens %d → %d",
                len(reasoning.split()), max_tokens, bigger,
            )
            retry_kwargs = dict(kwargs)
            retry_kwargs["max_tokens"] = bigger
            response = self._streamed_create(**retry_kwargs)
            if response.choices:
                choice = response.choices[0]
                content = choice.message.content or ""
                finish_reason = choice.finish_reason or ""
                _rr = getattr(choice.message, "reasoning_content", "")
                reasoning = _rr if isinstance(_rr, str) else ""

        # Last-resort fallback: if content is STILL empty but the model left
        # a final answer at the tail of reasoning_content (some prompts elicit
        # this), salvage it. Reasoning steps are numbered ("1. ... 2. ..."),
        # so take the text after the last step marker as the likely answer.
        if not content and reasoning:
            _salvaged = _salvage_answer_from_reasoning(reasoning)
            if _salvaged:
                logger.info(
                    "GLM-5.2 content empty — salvaged %d chars from "
                    "reasoning_content tail", len(_salvaged),
                )
                content = _salvaged

        tool_calls: list[ToolCall] = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError, AttributeError):
                    logger.warning("Failed to parse tool call arguments for %s", tc.function.name)
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
