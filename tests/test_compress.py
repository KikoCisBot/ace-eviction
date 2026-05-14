"""Tests for ace_eviction.compress."""

import pytest
from ace_eviction.compress import ACECompressor, compress


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lines(text: str):
    return text.splitlines()


# ---------------------------------------------------------------------------
# ACECompressor basics
# ---------------------------------------------------------------------------

class TestACECompressor:
    def test_short_content_unchanged(self):
        """Content with fewer lines than min_lines is returned unchanged."""
        c = ACECompressor(target_ratio=0.4, min_lines=4)
        short = "line1\nline2\nline3"
        assert c.compress(short) == short

    def test_first_line_always_kept(self):
        """The first non-empty line must always be in the output."""
        lines = ["First important line"] + ["done"] * 20 + ["Last line"]
        content = "\n".join(lines)
        result = ACECompressor(target_ratio=0.1).compress(content)
        assert "First important line" in result

    def test_last_line_always_kept(self):
        """The last non-empty line must always be in the output."""
        lines = ["First line"] + ["done"] * 20 + ["Last important line"]
        content = "\n".join(lines)
        result = ACECompressor(target_ratio=0.1).compress(content)
        assert "Last important line" in result

    def test_omit_marker_present(self):
        """Dropped lines must produce an omit marker."""
        lines = ["First line"] + ["done"] * 10 + ["Last line"]
        content = "\n".join(lines)
        result = ACECompressor(target_ratio=0.2).compress(content)
        assert "omitted by ACE" in result

    def test_omit_marker_counts_dropped_lines(self):
        """Omit markers must report the correct count."""
        # 12 'done' lines at score 0.30 plus first + last
        lines = ["First line"] + ["done"] * 12 + ["Last line"]
        content = "\n".join(lines)
        result = ACECompressor(target_ratio=0.2).compress(content)
        # Some lines were dropped — the marker should reference the count
        import re
        matches = re.findall(r"\[\.\.\.(\d+) line", result)
        assert matches, "Expected at least one omit marker with a count"
        assert int(matches[0]) > 0

    def test_high_score_lines_retained(self):
        """Lines matching error/path patterns must be kept over boilerplate."""
        lines = (
            ["Task context"]
            + ["done"] * 8
            + ["FileNotFoundError: /tmp/missing.py"]
            + ["done"] * 8
            + ["End of output"]
        )
        content = "\n".join(lines)
        result = ACECompressor(target_ratio=0.3).compress(content)
        assert "FileNotFoundError" in result

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError):
            ACECompressor(target_ratio=0.0)

    def test_ratio_above_one_raises(self):
        with pytest.raises(ValueError):
            ACECompressor(target_ratio=1.1)

    def test_ratio_one_keeps_all(self):
        """target_ratio=1.0 should keep every line."""
        lines = ["line " + str(i) for i in range(10)]
        content = "\n".join(lines)
        result = ACECompressor(target_ratio=1.0).compress(content)
        assert result == content

    def test_per_call_ratio_override(self):
        """target_ratio passed to compress() overrides instance default."""
        lines = ["Start"] + ["done"] * 20 + ["End"]
        content = "\n".join(lines)
        c = ACECompressor(target_ratio=0.9)
        # Override with very low ratio — should produce omit markers
        result = c.compress(content, target_ratio=0.1)
        assert "omitted by ACE" in result


# ---------------------------------------------------------------------------
# Module-level compress() convenience function
# ---------------------------------------------------------------------------

class TestCompressFunction:
    def test_returns_string(self):
        content = "\n".join(["line " + str(i) for i in range(10)])
        result = compress(content)
        assert isinstance(result, str)

    def test_result_shorter_or_equal(self):
        lines = ["Here are the results"] + ["done"] * 30 + ["Finished"]
        content = "\n".join(lines)
        result = compress(content, target_ratio=0.3)
        assert len(result) <= len(content)

    def test_no_omit_when_all_kept(self):
        lines = ["line " + str(i) for i in range(5)]
        content = "\n".join(lines)
        result = compress(content, target_ratio=1.0)
        assert "omitted" not in result


# ---------------------------------------------------------------------------
# Omit marker formatting
# ---------------------------------------------------------------------------

class TestOmitMarker:
    def test_singular_marker(self):
        """A single dropped line should use singular 'line' (not 'lines')."""
        from ace_eviction.compress import _omit_marker
        assert _omit_marker(1) == "[...1 line omitted by ACE...]"

    def test_plural_marker(self):
        from ace_eviction.compress import _omit_marker
        assert _omit_marker(5) == "[...5 lines omitted by ACE...]"
