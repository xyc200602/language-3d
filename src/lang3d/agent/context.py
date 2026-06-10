"""Context management utilities for token optimization.

Provides token estimation, message truncation, and tool result compression
to keep API calls within token budgets.
"""

from __future__ import annotations

from ..models.base import Message


def estimate_tokens(text: str) -> int:
    """Rough token estimation for mixed Chinese/English text.

    CJK: ~1.5 token/char; English/other: ~0.25 token/char.
    """
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, int(cjk * 1.5 + other / 4))


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


def _group_messages(messages: list[Message]) -> list[list[Message]]:
    """Group messages into atomic units that must not be split.

    An assistant message with tool_calls and its subsequent role="tool"
    result messages form one inseparable group.  All other messages
    are groups of one.
    """
    groups: list[list[Message]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.role == "assistant" and msg.tool_calls:
            group = [msg]
            i += 1
            # Collect all following tool-result messages
            while i < len(messages) and messages[i].role == "tool":
                group.append(messages[i])
                i += 1
            groups.append(group)
        else:
            groups.append([msg])
            i += 1
    return groups


def _estimate_group_tokens(group: list[Message]) -> int:
    """Estimate tokens for a whole message group."""
    return estimate_messages_tokens(group)


def truncate_messages(
    messages: list[Message],
    max_tokens: int = 8000,
    keep_first: int = 1,
    keep_last: int = 2,
) -> list[Message]:
    """Apply a sliding window to keep messages within token budget.

    Always preserves the first `keep_first` messages (system/user prompt)
    and the last `keep_last` messages (recent context).

    Truncation operates on **atomic groups**: an assistant message with
    ``tool_calls`` and all its corresponding ``role="tool"`` result
    messages are always kept or dropped together, preventing orphaned
    tool-result messages that would cause LLM API errors.

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

    remaining_budget = max_tokens - estimate_messages_tokens(head) - estimate_messages_tokens(tail)

    # Group the middle portion and keep as many groups as budget allows
    middle = messages[keep_first : len(messages) - keep_last]
    groups = _group_messages(middle)

    kept_groups: list[list[Message]] = []
    # Take from the end (most recent) backwards
    for group in reversed(groups):
        group_tokens = _estimate_group_tokens(group)
        if group_tokens > remaining_budget:
            break
        remaining_budget -= group_tokens
        kept_groups.insert(0, group)

    kept_middle: list[Message] = []
    for g in kept_groups:
        kept_middle.extend(g)

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
