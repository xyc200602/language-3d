"""Tests for model backend layer."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from lang3d.models.base import Message, ModelResponse, ToolCall, ToolDefinition
from lang3d.models.glm import GLMBackend
from lang3d.models.ollama import OllamaBackend
from lang3d.models.openai import OpenAIBackend
from lang3d.models.router import ModelRouter, TaskType
from lang3d.config import Config, ModelConfig, AgentConfig


def test_message_creation():
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_model_response():
    resp = ModelResponse(content="test", tool_calls=[], finish_reason="stop")
    assert resp.content == "test"
    assert resp.finish_reason == "stop"


def test_tool_call():
    tc = ToolCall(id="1", name="bash", arguments={"command": "echo"})
    assert tc.name == "bash"


def test_tool_definition():
    td = ToolDefinition(
        name="test",
        description="A test tool",
        parameters={"type": "object", "properties": {}},
    )
    assert td.name == "test"


def test_glm_backend_init():
    backend = GLMBackend(api_key="test", model="GLM-5.1")
    assert backend.api_key == "test"
    assert backend.model == "GLM-5.1"
    assert backend.base_url == "https://open.bigmodel.cn/api/coding/paas/v4"


def test_openai_backend_init():
    backend = OpenAIBackend(api_key="test", model="gpt-4o")
    assert backend.api_key == "test"


def test_ollama_backend_init():
    backend = OllamaBackend(base_url="http://localhost:11434", model="llama3")
    assert backend.model == "llama3"


def test_router_no_backends():
    config = Config()
    router = ModelRouter(config)
    assert router.available_backends == []


def test_router_with_glm():
    config = Config(
        glm=ModelConfig(api_key="test-key", base_url="https://open.bigmodel.cn/api/coding/paas/v4", model="GLM-5.1"),
    )
    router = ModelRouter(config)
    assert "glm" in router.available_backends


def test_router_get_backend():
    config = Config(
        glm=ModelConfig(api_key="test-key", base_url="https://open.bigmodel.cn/api/coding/paas/v4", model="GLM-5.1"),
        openai=ModelConfig(api_key="test-key", model="gpt-4o"),
    )
    router = ModelRouter(config)
    backend = router.get_backend(TaskType.CHAT)
    assert backend.model in ("GLM-5.1", "gpt-4o")


def test_image_encoding():
    import tempfile

    backend = GLMBackend()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        f.flush()
        encoded = backend._encode_image(f.name)
        assert len(encoded) > 0
        assert isinstance(encoded, str)


def test_media_type_detection():
    backend = GLMBackend()
    assert backend._get_media_type("test.png") == "image/png"
    assert backend._get_media_type("test.jpg") == "image/jpeg"
    assert backend._get_media_type("test.jpeg") == "image/jpeg"


class TestSalvageAnswerFromReasoning:
    """_salvage_answer_from_reasoning recovers a final answer when GLM-5.2
    (a reasoning model) leaves ``content`` empty but wrote the answer at the
    tail of ``reasoning_content``.

    This is the fallback path after the max_tokens-doubling retry; it must
    extract JSON without dragging in half-finished reasoning steps."""

    def test_extracts_json_from_final_step(self):
        from lang3d.models.glm import _salvage_answer_from_reasoning
        reasoning = (
            "1. 分析请求\n"
            "2. 构造JSON\n"
            "3. 最终输出: ```json\n{\"ok\":true}\n```"
        )
        out = _salvage_answer_from_reasoning(reasoning)
        assert "\"ok\":true" in out

    def test_extracts_bare_json_in_tail(self):
        from lang3d.models.glm import _salvage_answer_from_reasoning
        reasoning = "1. think\n2. answer\n{\"parts\":[],\"joints\":[]}"
        out = _salvage_answer_from_reasoning(reasoning)
        assert out.startswith("{")

    def test_returns_empty_for_no_reasoning(self):
        from lang3d.models.glm import _salvage_answer_from_reasoning
        assert _salvage_answer_from_reasoning("") == ""

    def test_ignores_overlong_truncated_tail(self):
        """A very long tail is likely truncated mid-thought — don't return
        a partial sentence as if it were an answer."""
        from lang3d.models.glm import _salvage_answer_from_reasoning
        long_tail = "1. start\n" + ("x" * 300)
        assert _salvage_answer_from_reasoning(long_tail) == ""


# --- Round 4: Runtime crash bug tests ---


def test_ollama_vision_accepts_model_param():
    """OllamaBackend.vision() must accept model kwarg without TypeError."""
    backend = OllamaBackend()
    import inspect
    sig = inspect.signature(backend.vision)
    assert "model" in sig.parameters

    # Also verify it can be called with model= without crashing
    # Mock _encode_image and HTTP call since no Ollama server is running
    with patch.object(backend, "_encode_image", return_value="fake_base64"):
        with patch("lang3d.models.ollama.httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"message": {"content": "test"}}
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp
            result = backend.vision("dummy.png", "describe", model="llava")
            assert result == "test"


def test_glm_malformed_tool_call_arguments():
    """GLMBackend.chat() must not crash on malformed tool_call arguments."""
    backend = GLMBackend()
    # Simulate a response with malformed arguments
    mock_tc = MagicMock()
    mock_tc.id = "tc_1"
    mock_tc.function.name = "test_tool"
    mock_tc.function.arguments = "NOT VALID JSON {{"

    mock_choice = MagicMock()
    mock_choice.message.content = ""
    mock_choice.message.tool_calls = [mock_tc]
    mock_choice.finish_reason = "stop"

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    backend._client = mock_client

    with patch("lang3d.models.glm.get_cache") as mock_get_cache:
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        result = backend.chat(
            [Message(role="user", content="test")],
            temperature=0.0,
        )
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {}  # fallback to empty dict


def test_openai_malformed_tool_call_arguments():
    """OpenAIBackend.chat() must not crash on malformed tool_call arguments."""
    backend = OpenAIBackend()
    mock_tc = MagicMock()
    mock_tc.id = "tc_1"
    mock_tc.function.name = "test_tool"
    mock_tc.function.arguments = None  # TypeError scenario

    mock_choice = MagicMock()
    mock_choice.message.content = ""
    mock_choice.message.tool_calls = [mock_tc]
    mock_choice.finish_reason = "stop"

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    backend._client = mock_client

    result = backend.chat(
        [Message(role="user", content="test")],
    )
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].arguments == {}


def test_tool_registry_raises_tool_error():
    """ToolRegistry.execute() must raise ToolError, not return error strings."""
    from lang3d.tools.base import ToolError, ToolRegistry

    registry = ToolRegistry()
    import pytest
    with pytest.raises(ToolError, match="not found"):
        registry.execute("nonexistent_tool")


def test_tool_registry_wraps_exceptions_as_tool_error():
    """ToolRegistry.execute() must wrap tool exceptions in ToolError."""
    from lang3d.tools.base import Tool, ToolError, ToolRegistry, ToolDefinition

    class FailingTool(Tool):
        name = "fail_tool"
        description = "A tool that always fails"

        def get_definition(self):
            return ToolDefinition(
                name=self.name,
                description=self.description,
                parameters={"type": "object", "properties": {}},
            )

        def execute(self, **kwargs):
            raise RuntimeError("Intentional failure")

    registry = ToolRegistry()
    registry.register(FailingTool())

    import pytest
    with pytest.raises(ToolError, match="Error executing fail_tool"):
        registry.execute("fail_tool")
