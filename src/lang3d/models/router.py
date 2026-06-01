"""Model router - selects the best backend for each task type."""

from __future__ import annotations

from enum import Enum
from typing import Any

from ..config import Config
from .base import Message, ModelBackend, ModelResponse, ToolDefinition
from .glm import GLMBackend
from .ollama import OllamaBackend
from .openai import OpenAIBackend


class TaskType(str, Enum):
    """Types of tasks that may need different models."""

    CHAT = "chat"
    CODE_GENERATION = "code_generation"
    PLANNING = "planning"
    VISION = "vision"
    REASONING = "reasoning"


# Preferred backend for each task type
TASK_ROUTING: dict[TaskType, list[str]] = {
    TaskType.CHAT: ["glm", "openai", "ollama"],
    TaskType.CODE_GENERATION: ["glm", "openai", "ollama"],
    TaskType.PLANNING: ["glm", "openai", "ollama"],
    TaskType.VISION: ["glm", "openai", "ollama"],
    TaskType.REASONING: ["glm", "openai", "ollama"],
}


class VisionDetail(str, Enum):
    """Detail level for vision analysis — routes to different models.

    FAST      → GLM-4V-Flash   (free, 0.2-3s, simple tasks)
    STANDARD  → GLM-4V-Plus    (best accuracy, 3-6s, general analysis)
    DETAILED  → GLM-4.6V-Flash (verbose, 20-27s, CAD verification)
    MAXIMUM   → GLM-4.6V       (most detailed, 40-50s, complex inspection)
    """

    FAST = "fast"
    STANDARD = "standard"
    DETAILED = "detailed"
    MAXIMUM = "maximum"


# Vision detail → (model_name, default_max_tokens)
VISION_DETAIL_MODELS: dict[VisionDetail, tuple[str, int]] = {
    VisionDetail.FAST: ("GLM-4V-Flash", 1024),
    VisionDetail.STANDARD: ("GLM-4V-Plus", 2048),
    VisionDetail.DETAILED: ("GLM-4.6V-Flash", 16384),
    VisionDetail.MAXIMUM: ("GLM-4.6V", 16384),
}


class ModelRouter:
    """Routes requests to the best available model backend."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._backends: dict[str, ModelBackend] = {}
        self._init_backends()

    def _init_backends(self) -> None:
        cfg = self.config
        if cfg.glm.api_key:
            self._backends["glm"] = GLMBackend(
                api_key=cfg.glm.api_key,
                base_url=cfg.glm.base_url or "https://open.bigmodel.cn/api/coding/paas/v4",
                model=cfg.glm.model or "GLM-5.1",
                vision_model=getattr(cfg.glm, "vision_model", "") or "GLM-4V-Flash",
            )
        if cfg.openai.api_key:
            self._backends["openai"] = OpenAIBackend(
                api_key=cfg.openai.api_key,
                base_url=cfg.openai.base_url or "https://api.openai.com/v1",
                model=cfg.openai.model or "gpt-4o",
            )
        if cfg.ollama.base_url:
            self._backends["ollama"] = OllamaBackend(
                base_url=cfg.ollama.base_url,
                model=cfg.ollama.model or "llama3",
            )

    def get_backend(self, task_type: TaskType = TaskType.CHAT) -> ModelBackend:
        """Get the best backend for a given task type."""
        preferred = TASK_ROUTING.get(task_type, ["glm"])

        for name in preferred:
            if name in self._backends:
                return self._backends[name]

        if cfg_default := self.config.default_backend:
            if cfg_default in self._backends:
                return self._backends[cfg_default]

        if self._backends:
            return next(iter(self._backends.values()))

        raise RuntimeError("No model backend configured. Please set API keys in .env or config.")

    def get_default(self) -> ModelBackend:
        """Get the default backend."""
        return self.get_backend(TaskType.CHAT)

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        *,
        task_type: TaskType = TaskType.CHAT,
        max_tokens: int = 100000,
        temperature: float = 0.7,
        system: str | None = None,
    ) -> ModelResponse:
        """Chat using the best backend for the task type."""
        backend = self.get_backend(task_type)
        return backend.chat(
            messages,
            tools,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
        )

    def vision(
        self,
        image_path: str,
        prompt: str,
        *,
        max_tokens: int = 100000,
        detail: VisionDetail | None = None,
    ) -> str:
        """Analyze an image using the best vision backend.

        Args:
            image_path: Path to image file.
            prompt: Analysis prompt.
            max_tokens: Max output tokens (auto-capped per model).
            detail: Vision detail level. Routes to different models:
                FAST=GLM-4V-Flash, STANDARD=GLM-4V-Plus,
                DETAILED=GLM-4.6V-Flash, MAXIMUM=GLM-4.6V.
                If None, uses the backend's default vision_model.
        """
        backend = self.get_backend(TaskType.VISION)

        if detail is not None:
            model_name, default_mt = VISION_DETAIL_MODELS[detail]
            # Use the higher of user-specified vs model default
            effective_mt = max(max_tokens, default_mt)
            return backend.vision(
                image_path, prompt,
                max_tokens=effective_mt,
                model=model_name,
            )

        return backend.vision(image_path, prompt, max_tokens=max_tokens)

    @property
    def available_backends(self) -> list[str]:
        return list(self._backends.keys())
