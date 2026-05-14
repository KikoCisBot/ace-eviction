"""
ACECompressor — content-aware compression for a single text block.

Given a string (typically the body of a tool-result message), the compressor
scores every line with the ACE classifier, keeps the highest-scoring fraction,
and replaces dropped runs with a compact placeholder.

The first and last non-empty lines are always preserved to maintain framing
context regardless of their scores.
"""

from __future__ import annotations

from typing import List, Tuple

from .classifier import classify_line

_OMIT_TEMPLATE = "[...{n} line{s} omitted by ACE...]"


def _omit_marker(n: int) -> str:
    return _OMIT_TEMPLATE.format(n=n, s="" if n == 1 else "s")


class ACECompressor:
    """Content-aware compressor for a single text block.

    Parameters
    ----------
    target_ratio:
        Fraction of lines to retain (default 0.4 = keep 40 % of lines).
        The actual number of lines kept may be slightly higher because the
        first/last lines are unconditionally kept.
    min_lines:
        Minimum number of lines below which compression is skipped entirely.
    """

    def __init__(self, target_ratio: float = 0.4, min_lines: int = 4) -> None:
        if not 0.0 < target_ratio <= 1.0:
            raise ValueError("target_ratio must be in (0, 1]")
        self.target_ratio = target_ratio
        self.min_lines = min_lines

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(self, content: str, target_ratio: float | None = None) -> str:
        """Compress *content* to approximately *target_ratio* of its lines.

        Parameters
        ----------
        content:
            Raw multi-line string to compress.
        target_ratio:
            Override the instance-level ratio for this call only.

        Returns
        -------
        str
            Compressed text.  Runs of dropped lines are replaced with a
            single ``[...N lines omitted by ACE...]`` marker.
        """
        ratio = target_ratio if target_ratio is not None else self.target_ratio
        lines = content.splitlines()

        if len(lines) < self.min_lines:
            return content

        keep = self._select_lines(lines, ratio)
        return self._rebuild(lines, keep)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_lines(self, lines: List[str], ratio: float) -> List[bool]:
        """Return a bool mask of which lines to keep."""
        n = len(lines)
        n_keep = max(1, round(n * ratio))

        # Score every line
        scores: List[Tuple[float, int]] = [
            (classify_line(line), idx) for idx, line in enumerate(lines)
        ]

        # Find the first and last non-empty line indices
        first_nz = next((i for i, l in enumerate(lines) if l.strip()), 0)
        last_nz = next((i for i, l in reversed(list(enumerate(lines))) if l.strip()), n - 1)

        # Sort by score descending, then by index ascending (stable tie-break)
        ranked = sorted(scores, key=lambda x: (-x[0], x[1]))

        # Pick top n_keep, but always include first_nz and last_nz
        top_indices = {idx for _, idx in ranked[:n_keep]}
        top_indices.add(first_nz)
        top_indices.add(last_nz)

        return [i in top_indices for i in range(n)]

    def _rebuild(self, lines: List[str], keep: List[bool]) -> str:
        """Reconstruct the text, collapsing dropped runs into markers."""
        result: List[str] = []
        dropped = 0

        for line, kept in zip(lines, keep):
            if kept:
                if dropped > 0:
                    result.append(_omit_marker(dropped))
                    dropped = 0
                result.append(line)
            else:
                dropped += 1

        if dropped > 0:
            result.append(_omit_marker(dropped))

        return "\n".join(result)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def compress(content: str, target_ratio: float = 0.4) -> str:
    """Compress *content*, keeping the top *target_ratio* fraction of lines.

    This is a thin wrapper around :class:`ACECompressor` for one-off use.

    Parameters
    ----------
    content:
        Multi-line string to compress.
    target_ratio:
        Fraction of lines to retain (default 0.4).

    Returns
    -------
    str
        Compressed text with omission markers.

    Examples
    --------
    >>> from ace_eviction import compress
    >>> compressed = compress(tool_output, target_ratio=0.4)
    """
    return ACECompressor(target_ratio=target_ratio).compress(content)
