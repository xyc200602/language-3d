"""Configuration management for Language-3D Agent.

Supports loading config from:
1. .env file in project root
2. Environment variables
3. Persistent config at ~/.lang3d/config.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for a single model backend."""

    api_key: str = ""
    base_url: str = ""
    model: str = ""
    vision_model: str = ""


class OrchestratorSettings(BaseModel):
    """Settings for multi-agent orchestration."""

    max_parallel_agents: int = 3
    max_retries_per_step: int = 3
    enable_parallel: bool = True


class SimulationSettings(BaseModel):
    """Settings for FEA simulation and analysis."""

    calculix_path: str = ""
    default_material: str = "steel"
    default_mesh_size: str = "medium"
    fea_timeout: int = 120
    default_fea_samples: int = 10000
    openfoam_path: str = ""
    default_fluid: str = "air"
    cfd_timeout: int = 300
    openfoam_mode: str = "auto"  # auto, wsl, native, docker


class SlicingSettings(BaseModel):
    """Settings for 3D printing slicing."""

    prusaslicer_path: str = ""
    orcaslicer_path: str = ""
    default_printer: str = "generic"
    default_material: str = "pla"
    default_quality: str = "standard"
    slice_timeout: int = 300


class RetrySettings(BaseModel):
    """Settings for LLM API call retry."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0


class AgentConfig(BaseModel):
    """Agent behavior configuration."""

    max_turns: int = 50
    workspace: str = str(Path.home() / "Desktop" / "language-3d" / "data" / "projects")
    screenshot_dir: str = str(Path.home() / "Desktop" / "language-3d" / "data" / "screenshots")
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    slicing: SlicingSettings = Field(default_factory=SlicingSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)

    # Execution tuning (unified from hardcoded values)
    max_turns_per_step: int = 10
    max_verify_retries: int = 3
    max_plan_retries: int = 3
    conversation_max_tokens: int = 8000
    tool_result_max_chars: int = 3000


class Config(BaseModel):
    """Top-level configuration."""

    glm: ModelConfig = Field(default_factory=ModelConfig)
    openai: ModelConfig = Field(default_factory=ModelConfig)
    ollama: ModelConfig = Field(default_factory=ModelConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    default_backend: str = "glm"


CONFIG_DIR = Path.home() / ".lang3d"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_dotenv() -> None:
    """Load .env file from current directory or project root."""
    from dotenv import load_dotenv as _load_dotenv

    # Try project root first
    project_env = Path(__file__).parent.parent.parent / ".env"
    if project_env.exists():
        _load_dotenv(project_env)
    else:
        _load_dotenv()


def load_config() -> Config:
    """Load configuration from env vars and persistent config file."""
    load_dotenv()

    # Start with persistent config if it exists
    config_data: dict[str, Any] = {}
    if CONFIG_FILE.exists():
        config_data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))

    # Override with environment variables
    env_overrides = _build_env_config()
    _deep_merge(config_data, env_overrides)

    return Config(**config_data)


def save_config(config: Config) -> None:
    """Persist configuration to ~/.lang3d/config.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(config.model_dump_json(indent=2), encoding="utf-8")


def _build_env_config() -> dict[str, Any]:
    """Build config dict from environment variables."""
    result: dict[str, Any] = {}

    if v := os.environ.get("GLM_API_KEY"):
        result.setdefault("glm", {})["api_key"] = v
    if v := os.environ.get("GLM_BASE_URL"):
        result.setdefault("glm", {})["base_url"] = v
    if v := os.environ.get("GLM_MODEL"):
        result.setdefault("glm", {})["model"] = v

    if v := os.environ.get("OPENAI_API_KEY"):
        result.setdefault("openai", {})["api_key"] = v
    if v := os.environ.get("OPENAI_BASE_URL"):
        result.setdefault("openai", {})["base_url"] = v
    if v := os.environ.get("OPENAI_MODEL"):
        result.setdefault("openai", {})["model"] = v

    if v := os.environ.get("OLLAMA_BASE_URL"):
        result.setdefault("ollama", {})["base_url"] = v
    if v := os.environ.get("OLLAMA_MODEL"):
        result.setdefault("ollama", {})["model"] = v

    if v := os.environ.get("VISION_MODEL"):
        # Set vision_model on the default backend
        backend = os.environ.get("LANG3D_DEFAULT_BACKEND", "glm")
        result.setdefault(backend, {})["vision_model"] = v

    if v := os.environ.get("AGENT_MAX_TURNS"):
        result.setdefault("agent", {})["max_turns"] = int(v)
    if v := os.environ.get("AGENT_WORKSPACE"):
        result.setdefault("agent", {})["workspace"] = v
    if v := os.environ.get("AGENT_MAX_TURNS_PER_STEP"):
        result.setdefault("agent", {})["max_turns_per_step"] = int(v)
    if v := os.environ.get("AGENT_MAX_VERIFY_RETRIES"):
        result.setdefault("agent", {})["max_verify_retries"] = int(v)
    if v := os.environ.get("AGENT_MAX_PLAN_RETRIES"):
        result.setdefault("agent", {})["max_plan_retries"] = int(v)
    if v := os.environ.get("AGENT_CONVERSATION_MAX_TOKENS"):
        result.setdefault("agent", {})["conversation_max_tokens"] = int(v)
    if v := os.environ.get("AGENT_TOOL_RESULT_MAX_CHARS"):
        result.setdefault("agent", {})["tool_result_max_chars"] = int(v)

    # Retry settings
    if v := os.environ.get("AGENT_RETRY_MAX_RETRIES"):
        result.setdefault("agent", {}).setdefault("retry", {})["max_retries"] = int(v)
    if v := os.environ.get("AGENT_RETRY_BASE_DELAY"):
        result.setdefault("agent", {}).setdefault("retry", {})["base_delay"] = float(v)
    if v := os.environ.get("AGENT_RETRY_MAX_DELAY"):
        result.setdefault("agent", {}).setdefault("retry", {})["max_delay"] = float(v)

    if v := os.environ.get("LANG3D_DEFAULT_BACKEND"):
        result["default_backend"] = v

    # Simulation settings
    if v := os.environ.get("CALCULIX_PATH"):
        result.setdefault("agent", {}).setdefault("simulation", {})["calculix_path"] = v
    if v := os.environ.get("DEFAULT_MATERIAL"):
        result.setdefault("agent", {}).setdefault("simulation", {})["default_material"] = v
    if v := os.environ.get("OPENFOAM_PATH"):
        result.setdefault("agent", {}).setdefault("simulation", {})["openfoam_path"] = v
    if v := os.environ.get("OPENFOAM_MODE"):
        result.setdefault("agent", {}).setdefault("simulation", {})["openfoam_mode"] = v

    # Slicing settings
    if v := os.environ.get("PRUSASLICER_PATH"):
        result.setdefault("agent", {}).setdefault("slicing", {})["prusaslicer_path"] = v
    if v := os.environ.get("ORCASLICER_PATH"):
        result.setdefault("agent", {}).setdefault("slicing", {})["orcaslicer_path"] = v
    if v := os.environ.get("DEFAULT_PRINTER"):
        result.setdefault("agent", {}).setdefault("slicing", {})["default_printer"] = v
    if v := os.environ.get("DEFAULT_MATERIAL"):
        result.setdefault("agent", {}).setdefault("slicing", {})["default_material"] = v
    if v := os.environ.get("DEFAULT_QUALITY"):
        result.setdefault("agent", {}).setdefault("slicing", {})["default_quality"] = v

    return result


def _deep_merge(base: dict, override: dict) -> None:
    """Merge override into base in-place, recursing into nested dicts."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
