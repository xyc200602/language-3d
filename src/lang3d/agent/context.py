"""Context management utilities for token optimization.

Provides token estimation, message truncation, and tool result compression
to keep API calls within token budgets.
"""

from __future__ import annotations

from ..models.base import Message


def estimate_tokens(text: str) -> int:
    """Rough token estimation for mixed Chinese/English text.

    Chinese characters: ~2 chars/token (CJK characters are denser).
    English/other: ~4 chars/token.
    """
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, cjk // 2 + other // 4)


def estimate_messages_tokens(messages: list[Message]) -> int:
    """Estimate total tokens for a list of messages."""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.content or "")
        # Tool calls also consume tokens
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = tc.get("arguments", {})
                total += estimate_tokens(str(args))
    return total


def truncate_messages(
    messages: list[Message],
    max_tokens: int = 8000,
    keep_first: int = 1,
    keep_last: int = 2,
) -> list[Message]:
    """Apply a sliding window to keep messages within token budget.

    Always preserves the first `keep_first` messages (system/user prompt)
    and the last `keep_last` messages (recent context).

    Returns a new list (does not mutate input).
    """
    if len(messages) <= keep_first + keep_last:
        return list(messages)

    current_tokens = estimate_messages_tokens(messages)
    if current_tokens <= max_tokens:
        return list(messages)

    # Keep head and tail, drop middle
    head = messages[:keep_first]
    tail = messages[-keep_last:]

    # Try to include as many middle messages as budget allows
    remaining_budget = max_tokens - estimate_messages_tokens(head) - estimate_messages_tokens(tail)
    middle = messages[keep_first : len(messages) - keep_last]

    kept_middle: list[Message] = []
    # Take from the end of middle (most recent) backwards
    for msg in reversed(middle):
        msg_tokens = estimate_tokens(msg.content or "")
        if msg_tokens > remaining_budget:
            break
        remaining_budget -= msg_tokens
        kept_middle.insert(0, msg)

    return head + kept_middle + tail


def truncate_tool_result(result: str, max_chars: int = 3000) -> str:
    """Truncate large tool results to save tokens.

    Preserves structured VLM output header (MATCH/OBSERVED/DIFFERENCES lines)
    by keeping the first portion intact.
    """
    if len(result) <= max_chars:
        return result

    # Check if this looks like structured VLM output
    vlm_markers = ("MATCH:", "OBSERVED:", "DIFFERENCES:", "SUGGESTION:")
    has_vlm_structure = any(marker in result for marker in vlm_markers)

    if has_vlm_structure:
        # Find the "--- Raw VLM output ---" separator
        sep_idx = result.find("\n--- Raw VLM output ---")
        if sep_idx > 0:
            # Keep structured header, truncate raw output
            header = result[:sep_idx]
            if len(header) <= max_chars:
                return header + "\n[Raw output truncated]"
            return header[:max_chars] + "\n[...truncated]"

    # Generic truncation with ellipsis
    return result[:max_chars] + "\n[...truncated]"
