"""
ace_evict — message-level context eviction for multi-turn agentic inference.

Given a list of messages (OpenAI / Anthropic chat format), this module
identifies tool-result messages and compresses the oldest ones until the
total character count falls below a specified budget.

Message detection
-----------------
A message is treated as a tool result if:

- Its ``role`` is ``"tool"``, OR
- Its ``content`` (string form) starts with ``"[Tool result]:"``

The *keep_recent* most-recent tool-result messages are never touched, so
the model always has full fidelity on its most recent observations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

from .compress import ACECompressor

Message = Dict[str, Any]


def _message_content(msg: Message) -> str:
    """Extract a plain-string representation of a message's content."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Handle multi-part content blocks (Anthropic format)
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", "") or block.get("content", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _set_message_content(msg: Message, new_content: str) -> Message:
    """Return a shallow copy of *msg* with content replaced by *new_content*."""
    updated = dict(msg)
    if isinstance(msg.get("content"), list):
        # Preserve list structure: replace text in first text block
        new_blocks: List[Any] = []
        replaced = False
        for block in msg["content"]:
            if not replaced and isinstance(block, dict) and "text" in block:
                new_blocks.append({**block, "text": new_content})
                replaced = True
            else:
                new_blocks.append(block)
        if not replaced:
            new_blocks = [{"type": "text", "text": new_content}]
        updated["content"] = new_blocks
    else:
        updated["content"] = new_content
    return updated


def _is_tool_result(msg: Message) -> bool:
    """Return True if *msg* is a tool-result message."""
    if msg.get("role") == "tool":
        return True
    content = _message_content(msg)
    return content.lstrip().startswith("[Tool result]:")


def _total_chars(messages: List[Message]) -> int:
    return sum(len(_message_content(m)) for m in messages)


def ace_evict(
    messages: List[Message],
    budget_chars: int = 40_000,
    keep_recent: int = 2,
    target_ratio: float = 0.4,
    min_size: int = 200,
) -> int:
    """Compress old tool-result messages until total context fits within *budget_chars*.

    The function modifies *messages* **in-place** and returns the total number
    of characters saved.

    Parameters
    ----------
    messages:
        List of chat messages in OpenAI or Anthropic format.
    budget_chars:
        Target maximum total character count across all messages.
    keep_recent:
        Number of most-recent tool-result messages to leave untouched.
        These are the model's freshest observations and should be preserved
        at full fidelity.
    target_ratio:
        Fraction of lines to retain when compressing a message (default 0.4).
    min_size:
        Skip compression for messages whose content is shorter than this many
        characters (compressing tiny messages wastes time with no benefit).

    Returns
    -------
    int
        Total characters removed across all compressed messages.

    Examples
    --------
    >>> from ace_eviction import ace_evict
    >>> saved = ace_evict(messages, budget_chars=8_000, keep_recent=2)
    >>> print(f"ACE freed {saved:,} chars")
    """
    if _total_chars(messages) <= budget_chars:
        return 0

    compressor = ACECompressor(target_ratio=target_ratio)

    # Collect indices of tool-result messages in conversation order
    tool_indices: List[int] = [
        i for i, m in enumerate(messages) if _is_tool_result(m)
    ]

    if not tool_indices:
        return 0

    # The most-recent keep_recent tool results are off-limits
    eviction_candidates = tool_indices[:-keep_recent] if keep_recent > 0 else tool_indices

    total_saved = 0

    # Compress oldest-first until under budget
    for idx in eviction_candidates:
        if _total_chars(messages) <= budget_chars:
            break

        msg = messages[idx]
        original = _message_content(msg)

        if len(original) < min_size:
            continue

        compressed = compressor.compress(original, target_ratio=target_ratio)
        saved = len(original) - len(compressed)

        if saved <= 0:
            continue

        messages[idx] = _set_message_content(msg, compressed)
        total_saved += saved

    return total_saved
