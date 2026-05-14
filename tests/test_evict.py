"""Tests for ace_eviction.evict."""

import pytest
from ace_eviction.evict import ace_evict, _is_tool_result, _message_content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_msg(content: str, role: str = "tool") -> dict:
    return {"role": role, "content": content}


def _user_msg(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant_msg(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _bracket_tool_msg(content: str) -> dict:
    """Simulate assistant messages that embed tool results inline."""
    return {"role": "assistant", "content": f"[Tool result]: {content}"}


def _big_tool_result(n_lines: int = 100) -> str:
    """Generate a large tool result with mixed line types."""
    lines = []
    for i in range(n_lines):
        if i % 20 == 0:
            lines.append(f"Step {i}: checking progress")
        elif i % 7 == 0:
            lines.append("done")
        elif i % 5 == 0:
            lines.append(f"Output: /tmp/result_{i}.json")
        else:
            lines.append(f"Processing item {i} of {n_lines}.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# _is_tool_result detection
# ---------------------------------------------------------------------------

class TestIsToolResult:
    def test_role_tool(self):
        assert _is_tool_result({"role": "tool", "content": "anything"})

    def test_bracket_prefix(self):
        assert _is_tool_result({"role": "assistant", "content": "[Tool result]: output"})

    def test_bracket_prefix_with_leading_space(self):
        assert _is_tool_result({"role": "assistant", "content": "  [Tool result]: output"})

    def test_user_message_not_tool(self):
        assert not _is_tool_result({"role": "user", "content": "hello"})

    def test_assistant_without_prefix_not_tool(self):
        assert not _is_tool_result({"role": "assistant", "content": "Here is my plan."})


# ---------------------------------------------------------------------------
# _message_content extraction
# ---------------------------------------------------------------------------

class TestMessageContent:
    def test_string_content(self):
        msg = {"role": "tool", "content": "hello world"}
        assert _message_content(msg) == "hello world"

    def test_list_content_text_blocks(self):
        msg = {
            "role": "tool",
            "content": [
                {"type": "text", "text": "part one"},
                {"type": "text", "text": "part two"},
            ],
        }
        result = _message_content(msg)
        assert "part one" in result
        assert "part two" in result

    def test_missing_content_returns_empty(self):
        msg = {"role": "tool"}
        assert _message_content(msg) == ""


# ---------------------------------------------------------------------------
# ace_evict — no eviction needed
# ---------------------------------------------------------------------------

class TestNoEvictionNeeded:
    def test_already_under_budget(self):
        messages = [
            _user_msg("hi"),
            _tool_msg("short result"),
        ]
        saved = ace_evict(messages, budget_chars=10_000)
        assert saved == 0

    def test_no_tool_messages(self):
        messages = [
            _user_msg("hello"),
            _assistant_msg("world"),
        ]
        original = [dict(m) for m in messages]
        saved = ace_evict(messages, budget_chars=10)
        assert saved == 0
        assert messages == original


# ---------------------------------------------------------------------------
# ace_evict — basic compression
# ---------------------------------------------------------------------------

class TestBasicEviction:
    def test_reduces_total_chars(self):
        big = _big_tool_result(80)
        messages = [
            _user_msg("analyze this"),
            _tool_msg(big),
            _user_msg("now what?"),
        ]
        initial = sum(len(m["content"]) for m in messages)
        saved = ace_evict(messages, budget_chars=200, keep_recent=0, target_ratio=0.3)
        final = sum(len(m["content"]) for m in messages)
        assert saved > 0
        assert final < initial

    def test_returns_chars_saved(self):
        big = _big_tool_result(60)
        messages = [_tool_msg(big)]
        saved = ace_evict(messages, budget_chars=10, keep_recent=0, target_ratio=0.3)
        assert isinstance(saved, int)
        assert saved > 0

    def test_omit_marker_in_compressed_message(self):
        big = _big_tool_result(60)
        messages = [_tool_msg(big)]
        ace_evict(messages, budget_chars=10, keep_recent=0, target_ratio=0.3)
        assert "omitted by ACE" in messages[0]["content"]


# ---------------------------------------------------------------------------
# ace_evict — keep_recent logic
# ---------------------------------------------------------------------------

class TestKeepRecent:
    def test_most_recent_tool_not_compressed(self):
        big = _big_tool_result(60)
        messages = [
            _tool_msg("old result " + "x" * 500),
            _tool_msg(big),  # most recent — must not be touched
        ]
        original_last = messages[-1]["content"]
        ace_evict(messages, budget_chars=100, keep_recent=1, target_ratio=0.3)
        assert messages[-1]["content"] == original_last

    def test_keep_recent_zero_compresses_all(self):
        big1 = _big_tool_result(50)
        big2 = _big_tool_result(50)
        messages = [_tool_msg(big1), _tool_msg(big2)]
        saved = ace_evict(messages, budget_chars=50, keep_recent=0, target_ratio=0.3)
        assert saved > 0

    def test_keep_recent_two_only_compresses_older(self):
        msgs = [_tool_msg(_big_tool_result(50)) for _ in range(4)]
        originals = [m["content"] for m in msgs]
        ace_evict(msgs, budget_chars=200, keep_recent=2, target_ratio=0.3)
        # Last two must be untouched
        assert msgs[-1]["content"] == originals[-1]
        assert msgs[-2]["content"] == originals[-2]


# ---------------------------------------------------------------------------
# ace_evict — min_size guard
# ---------------------------------------------------------------------------

class TestMinSize:
    def test_tiny_messages_not_compressed(self):
        """Messages shorter than min_size must not be modified."""
        tiny = "done"
        messages = [_tool_msg(tiny)]
        original = messages[0]["content"]
        ace_evict(messages, budget_chars=1, keep_recent=0, min_size=200)
        assert messages[0]["content"] == original


# ---------------------------------------------------------------------------
# ace_evict — bracket-prefix tool results
# ---------------------------------------------------------------------------

class TestBracketPrefixToolResults:
    def test_bracket_prefix_messages_detected(self):
        big = "[Tool result]: " + _big_tool_result(60)
        messages = [{"role": "assistant", "content": big}]
        saved = ace_evict(messages, budget_chars=10, keep_recent=0, target_ratio=0.3)
        assert saved > 0


# ---------------------------------------------------------------------------
# ace_evict — list-content messages (Anthropic multi-part format)
# ---------------------------------------------------------------------------

class TestListContentMessages:
    def test_list_content_compressed(self):
        big = _big_tool_result(60)
        messages = [
            {
                "role": "tool",
                "content": [{"type": "text", "text": big}],
            }
        ]
        saved = ace_evict(messages, budget_chars=10, keep_recent=0, target_ratio=0.3)
        assert saved > 0
        # Content should still be a list
        assert isinstance(messages[0]["content"], list)
