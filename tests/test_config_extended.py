"""Tests for extended configuration fields."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from lang3d.config import AgentConfig


class TestAgentConfigDefaults:
    """New configuration field default values."""

    def test_max_turns_per_step_default(self):
        config = AgentConfig()
        assert config.max_turns_per_step == 25

    def test_max_verify_retries_default(self):
        config = AgentConfig()
        assert config.max_verify_retries == 3

    def test_max_plan_retries_default(self):
        config = AgentConfig()
        assert config.max_plan_retries == 3

    def test_conversation_max_tokens_default(self):
        config = AgentConfig()
        assert config.conversation_max_tokens == 8000

    def test_tool_result_max_chars_default(self):
        config = AgentConfig()
        assert config.tool_result_max_chars == 3000

    def test_custom_values(self):
        config = AgentConfig(
            max_turns_per_step=15,
            max_verify_retries=5,
            max_plan_retries=2,
            conversation_max_tokens=12000,
            tool_result_max_chars=5000,
        )
        assert config.max_turns_per_step == 15
        assert config.max_verify_retries == 5
        assert config.max_plan_retries == 2
        assert config.conversation_max_tokens == 12000
        assert config.tool_result_max_chars == 5000


class TestConfigCorruptedJson:
    def test_load_config_handles_corrupted_json(self, tmp_path):
        """Config loading should handle invalid JSON gracefully."""
        config_file = tmp_path / "config.json"
        config_file.write_text("{invalid json", encoding="utf-8")

        with patch("lang3d.config.CONFIG_FILE", config_file):
            with patch("lang3d.config.CONFIG_DIR", tmp_path):
                from lang3d.config import load_config
                config = load_config()
                # Should return default config, not crash
                assert config is not None


class TestConfigApiKeyMasking:
    def test_save_config_masks_api_keys(self, tmp_path):
        """Saved config should mask API keys."""
        from lang3d.config import save_config, Config

        config = Config(
            glm={"api_key": "sk-1234567890abcdef", "model": "GLM-5.1"},
            openai={"api_key": "sk-abcdef1234567890", "model": "gpt-4o"},
        )

        config_file = tmp_path / "config.json"
        with patch("lang3d.config.CONFIG_FILE", config_file):
            with patch("lang3d.config.CONFIG_DIR", tmp_path):
                save_config(config)

        saved = json.loads(config_file.read_text(encoding="utf-8"))
        assert saved["glm"]["api_key"] == "sk-1***"
        assert saved["openai"]["api_key"] == "sk-a***"
        assert "1234567890" not in saved["glm"]["api_key"]
