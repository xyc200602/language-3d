"""Tests for extended configuration fields."""

from __future__ import annotations

from lang3d.config import AgentConfig


class TestAgentConfigDefaults:
    """New configuration field default values."""

    def test_max_turns_per_step_default(self):
        config = AgentConfig()
        assert config.max_turns_per_step == 10

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
