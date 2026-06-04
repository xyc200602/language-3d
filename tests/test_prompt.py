"""Tests for Agent system prompt optimization.

Verifies the updated system prompt includes all required sections:
- 3D modeling workflow
- Tool usage strategy
- cad_verify verification strategy
- Modeling conventions
"""

from __future__ import annotations

from lang3d.agent.core import AGENT_SYSTEM_PROMPT


def test_prompt_contains_workflow():
    """System prompt describes the 3D modeling workflow."""
    assert "fc_batch" in AGENT_SYSTEM_PROMPT
    assert "cad_verify" in AGENT_SYSTEM_PROMPT
    assert "volume_check" in AGENT_SYSTEM_PROMPT


def test_prompt_contains_tool_prefixes():
    """System prompt lists all tool prefixes."""
    assert "fc_*" in AGENT_SYSTEM_PROMPT
    assert "sw_*" in AGENT_SYSTEM_PROMPT
    assert "gui_*" in AGENT_SYSTEM_PROMPT


def test_prompt_contains_verification_strategy():
    """System prompt includes cad_verify verification strategy."""
    assert "MATCH" in AGENT_SYSTEM_PROMPT
    assert "FIX_COMMANDS" in AGENT_SYSTEM_PROMPT
    assert "false=需修正" in AGENT_SYSTEM_PROMPT or "false" in AGENT_SYSTEM_PROMPT


def test_prompt_contains_detail_levels():
    """System prompt includes detail level descriptions."""
    assert "standard" in AGENT_SYSTEM_PROMPT
    assert "detailed" in AGENT_SYSTEM_PROMPT
    assert "detail=" in AGENT_SYSTEM_PROMPT


def test_prompt_contains_unit_convention():
    """System prompt specifies mm as the unit."""
    assert "mm" in AGENT_SYSTEM_PROMPT.lower() or "毫米" in AGENT_SYSTEM_PROMPT


def test_prompt_contains_workspace_placeholder():
    """System prompt has workspace placeholder."""
    assert "{workspace}" in AGENT_SYSTEM_PROMPT


def test_prompt_formatted():
    """System prompt can be formatted with workspace."""
    formatted = AGENT_SYSTEM_PROMPT.format(workspace="/tmp/test")
    assert "/tmp/test" in formatted
    assert "{workspace}" not in formatted


def test_prompt_has_workflow_steps():
    """System prompt lists the mandatory workflow steps."""
    assert "规划" in AGENT_SYSTEM_PROMPT or "Plan" in AGENT_SYSTEM_PROMPT
    assert "建模" in AGENT_SYSTEM_PROMPT or "Model" in AGENT_SYSTEM_PROMPT
    assert "验证" in AGENT_SYSTEM_PROMPT or "Verify" in AGENT_SYSTEM_PROMPT
    assert "修正" in AGENT_SYSTEM_PROMPT or "Fix" in AGENT_SYSTEM_PROMPT


def test_agent_registers_gui_tools():
    """Agent registers gui_action tools along with other tools."""
    from unittest.mock import patch
    from lang3d.config import Config, AgentConfig, ModelConfig

    config = Config(
        agent=AgentConfig(workspace="/tmp"),
        models=ModelConfig(),
    )
    # Patch load_config to avoid .env file dependency
    with patch("lang3d.config.load_config", return_value=config):
        from lang3d.agent.core import Agent
        agent = Agent(config=config)

    tool_names = agent.tools.list_tools()
    assert "gui_click" in tool_names
    assert "gui_screenshot" in tool_names
    assert "gui_hotkey" in tool_names
    assert "gui_drag" in tool_names
    assert "gui_scroll" in tool_names


def test_agent_has_complete_tool_set():
    """Agent has the complete expected tool set."""
    from unittest.mock import patch
    from lang3d.config import Config, AgentConfig, ModelConfig

    config = Config(
        agent=AgentConfig(workspace="/tmp"),
        models=ModelConfig(),
    )
    with patch("lang3d.config.load_config", return_value=config):
        from lang3d.agent.core import Agent
        agent = Agent(config=config)

    tool_names = agent.tools.list_tools()

    # File tools
    assert "file_read" in tool_names
    assert "file_write" in tool_names

    # Bash tools
    assert "bash" in tool_names
    assert "python_exec" in tool_names

    # Screen tools
    assert "screen_capture" in tool_names
    assert "window_capture" in tool_names
    assert "list_windows" in tool_names

    # VLM tools
    assert "vlm_analyze" in tool_names
    assert "screen_analyze" in tool_names
    assert "window_analyze" in tool_names
    assert "cad_verify" in tool_names

    # GUI automation tools
    assert "gui_click" in tool_names
    assert "gui_type" in tool_names
    assert "gui_hotkey" in tool_names
    assert "gui_screenshot" in tool_names
    assert "gui_mouse_pos" in tool_names

    # Total count should be 43+ tools
    assert len(tool_names) >= 40, f"Only {len(tool_names)} tools registered"
