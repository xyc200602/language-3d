"""Tests for context management utilities."""

from __future__ import annotations

from lang3d.agent.context import (
    estimate_messages_tokens,
    estimate_tokens,
    truncate_messages,
    truncate_tool_result,
)
from lang3d.models.base import Message


class TestEstimateTokens:
    """Token estimation tests."""

    def test_english_text(self):
        tokens = estimate_tokens("Hello world this is a test")
        assert tokens > 0

    def test_chinese_text(self):
        tokens = estimate_tokens("这是一个中文测试")
        assert tokens > 0

    def test_mixed_text(self):
        mixed = "Create a 30x30x30mm 方块，使用 fc_batch"
        tokens = estimate_tokens(mixed)
        assert tokens > 0

    def test_empty_string(self):
        assert estimate_tokens("") >= 1

    def test_chinese_denser_than_english(self):
        # Chinese chars are ~2 chars/token vs ~4 chars/token for English
        cn = estimate_tokens("测" * 100)
        en = estimate_tokens("a" * 100)
        assert cn > en


class TestEstimateMessagesTokens:
    """Message list token estimation tests."""

    def test_single_message(self):
        msgs = [Message(role="user", content="Hello")]
        assert estimate_messages_tokens(msgs) > 0

    def test_multiple_messages(self):
        msgs = [
            Message(role="user", content="Create a box"),
            Message(role="assistant", content="Done"),
        ]
        total = estimate_messages_tokens(msgs)
        assert total > 0

    def test_tool_call_messages(self):
        msgs = [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {"id": "1", "name": "bash", "arguments": {"command": "ls"}},
                ],
            ),
        ]
        total = estimate_messages_tokens(msgs)
        assert total > 0


class TestTruncateMessages:
    """Sliding window truncation tests."""

    def test_short_messages_unchanged(self):
        msgs = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        result = truncate_messages(msgs)
        assert len(result) == 2

    def test_long_messages_truncated(self):
        msgs = [Message(role="user", content=f"Message {i} " + "x" * 2000) for i in range(20)]
        result = truncate_messages(msgs, max_tokens=2000, keep_first=1, keep_last=2)
        # Should be shorter than original
        assert len(result) < len(msgs)
        # Should keep first and last
        assert result[0].content == msgs[0].content
        assert result[-1].content == msgs[-1].content
        assert result[-2].content == msgs[-2].content

    def test_within_budget_unchanged(self):
        msgs = [
            Message(role="user", content="Short"),
            Message(role="assistant", content="Also short"),
            Message(role="user", content="Tiny"),
        ]
        result = truncate_messages(msgs, max_tokens=10000)
        assert len(result) == len(msgs)


class TestTruncateToolResult:
    """Tool result truncation tests."""

    def test_short_result_unchanged(self):
        result = "This is a short result"
        assert truncate_tool_result(result) == result

    def test_long_result_truncated(self):
        result = "x" * 5000
        truncated = truncate_tool_result(result, max_chars=3000)
        assert len(truncated) < len(result)
        assert truncated.endswith("[...truncated]")

    def test_vlm_structure_preserved(self):
        result = (
            "MATCH: False\n"
            "OBSERVED: A cube\n"
            "DIFFERENCES: Missing hole\n"
            "SUGGESTION: Add cylinder cut\n"
            "FIX_COMMANDS: None\n"
            "\n--- Raw VLM output ---\n"
            + "Very long raw output " * 500
        )
        truncated = truncate_tool_result(result, max_chars=500)
        # Structured header should be preserved
        assert "MATCH:" in truncated
        assert "DIFFERENCES:" in truncated
        assert "[Raw output truncated]" in truncated


class TestTruncatePreservesToolCallPairs:
    """Verify that truncation never creates orphaned tool-result messages."""

    def test_truncate_preserves_tool_call_pairs(self):
        """After truncation, every role='tool' message must have a preceding
        assistant message with tool_calls that includes its tool_call_id."""
        # Build a message sequence with multiple tool call groups
        messages = [Message(role="user", content="System prompt")]
        for i in range(15):
            tc_id = f"tc_{i}"
            messages.append(Message(
                role="assistant",
                content=f"Assistant {i}",
                tool_calls=[{"id": tc_id, "name": "bash", "arguments": {"command": f"cmd{i}"}}],
            ))
            messages.append(Message(role="tool", content=f"Result {i} " + "x" * 300, tool_call_id=tc_id))
        messages.append(Message(role="assistant", content="Final answer"))

        result = truncate_messages(messages, max_tokens=1000, keep_first=1, keep_last=2)

        # Verify no orphaned tool messages
        for idx, msg in enumerate(result):
            if msg.role == "tool":
                assert idx > 0, "Tool message cannot be the first message"
                prev = result[idx - 1]
                assert prev.role == "assistant" and prev.tool_calls, (
                    f"Tool message at index {idx} has no preceding assistant with tool_calls"
                )
                # Verify tool_call_id is matched
                tc_ids = {tc["id"] for tc in prev.tool_calls}
                assert msg.tool_call_id in tc_ids, (
                    f"Tool message tool_call_id={msg.tool_call_id} not found in preceding assistant"
                )

    def test_truncate_never_breaks_middle_group(self):
        """A single assistant+tool group in the middle must stay together or be fully removed."""
        messages = [
            Message(role="user", content="System prompt"),
            Message(role="assistant", content="Thinking..."),
            # Group: assistant with tool_calls + 2 tool results
            Message(
                role="assistant", content="",
                tool_calls=[
                    {"id": "tc_1", "name": "bash", "arguments": {"command": "a"}},
                    {"id": "tc_2", "name": "bash", "arguments": {"command": "b"}},
                ],
            ),
            Message(role="tool", content="Result 1 " + "y" * 2000, tool_call_id="tc_1"),
            Message(role="tool", content="Result 2 " + "y" * 2000, tool_call_id="tc_2"),
            # More filler
            Message(role="assistant", content="filler " + "z" * 2000),
            Message(role="assistant", content="Final answer"),
        ]

        result = truncate_messages(messages, max_tokens=500, keep_first=1, keep_last=1)

        # Check: if any tool message appears, all must appear with their assistant
        tool_indices = [i for i, m in enumerate(result) if m.role == "tool"]
        if tool_indices:
            # Must have an assistant with tool_calls before the first tool
            first_tool_idx = tool_indices[0]
            assert first_tool_idx > 0
            prev = result[first_tool_idx - 1]
            assert prev.role == "assistant" and prev.tool_calls
