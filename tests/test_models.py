"""Tests for model backend layer."""

from __future__ import annotations

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
