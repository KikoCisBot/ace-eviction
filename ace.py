"""
ACE — Attention-Weighted Context Eviction

Content-aware context compression for multi-turn agentic LLM inference.

Usage:
    from ace import ace_compress, ace_evict

    # Compress a single tool-result string
    compressed = ace_compress(tool_result_text, target_ratio=0.4)

    # Evict old tool results from a messages list in-place
    saved = ace_evict(messages, budget_chars=40000, keep_recent=2)
"""

import re
from typing import List, Dict, Any


def ace_classify_line(s: str) -> float:
    """
    Score a single line for importance. Returns float in [0, 1].

    Priority:
      1.0  tool call JSON
      0.95 errors / exceptions
      0.9  file / workspace paths
      0.85 task-framing keywords
      0.7  numeric data
      0.6  shell commands
      0.5  default
      0.3  success boilerplate
      0.2  very long lines (>200 chars)
      0.1  meta-commentary
      0.0  blank lines
    """
    if not s.strip():
        return 0.0

    sl = s.lower()

    # Tool call JSON
    if '"name"' in s and ('"arguments"' in s or '"parameters"' in s):
        return 1.0

    # Errors
    error_phrases = ['error', 'failed', 'not found', 'traceback', 'exception', 'exit code']
    if any(p in sl for p in error_phrases):
        return 0.95

    # File paths
    if any(p in s for p in ['/results/', '/workspace/', '/tmp/']):
        return 0.9

    # Task keywords
    if any(k in sl for k in ['task:', 'begin', 'step ', 'verify', 'inspect']):
        return 0.85

    # Numeric data
    if re.search(r'\d+\.?\d*%|\d{4,}|bytes|rows|columns', sl):
        return 0.7

    # Shell commands
    if any(s.startswith(k) for k in ['$', '>>>', 'root@', 'agent@']) or \
       any(k in sl for k in ['wget', 'curl', 'pip', 'python3', 'apt-get', 'docker']):
        return 0.6

    # Success boilerplate
    if sl.strip() in {'done', 'ok', 'installed', 'saved', 'complete'}:
        return 0.3

    # Long verbose dump
    if len(s) > 200:
        return 0.2

    # Meta-commentary
    if any(k in sl for k in ['i executed', 'here are the results', 'tool results:']):
        return 0.1

    return 0.5


def ace_compress(content: str, target_ratio: float = 0.4) -> str:
    """
    Compress a multi-line string to ~target_ratio of its original line count.

    Always keeps first/last line. Omitted runs become a single marker line.

    Args:
        content:      Text to compress.
        target_ratio: Fraction of lines to keep (default 0.4 = 40%).

    Returns:
        Compressed string.
    """
    lines = content.split('\n')
    if len(lines) <= 3:
        return content

    scored = []
    for i, line in enumerate(lines):
        score = ace_classify_line(line)
        if i == 0 or i == len(lines) - 1:
            score = max(score, 0.7)
        scored.append((i, line, score))

    n_keep = max(2, int(len(lines) * target_ratio))
    keep_set = set(
        x[0] for x in sorted(scored, key=lambda x: -x[2])[:n_keep]
    )
    keep_set.add(0)
    keep_set.add(len(lines) - 1)

    out = []
    skipped = 0
    for i, line in enumerate(lines):
        if i in keep_set:
            if skipped > 0:
                out.append(f'  [...{skipped} lines omitted by ACE...]')
            skipped = 0
            out.append(line)
        else:
            skipped += 1

    if skipped > 0:
        out.append(f'  [...{skipped} lines omitted by ACE...]')

    return '\n'.join(out)


def ace_evict(
    messages: List[Dict[str, Any]],
    budget_chars: int = 40_000,
    keep_recent: int = 2,
    target_ratio: float = 0.4,
    min_size: int = 200,
) -> int:
    """
    Compress old tool results in a messages list until total size <= budget_chars.

    Operates on messages whose content starts with "[Tool result]: ".
    Modifies messages in-place. Skips the keep_recent most-recent tool results.

    Args:
        messages:     Anthropic-style messages list (dicts with 'role'/'content').
        budget_chars: Target max total chars.
        keep_recent:  Number of recent tool results to leave untouched.
        target_ratio: ACE compression ratio (default 0.4).
        min_size:     Don't compress messages shorter than this.

    Returns:
        Total chars removed.
    """
    def total_chars(msgs):
        return sum(len(m.get('content', '')) for m in msgs if isinstance(m.get('content'), str))

    if total_chars(messages) <= budget_chars:
        return 0

    prefix = '[Tool result]: '
    tool_indices = [
        i for i, m in enumerate(messages)
        if m.get('role') == 'user'
        and isinstance(m.get('content'), str)
        and m['content'].startswith(prefix)
    ]

    compressable = tool_indices[:len(tool_indices) - keep_recent]
    total_saved = 0

    for idx in compressable:
        if total_chars(messages) <= budget_chars:
            break
        result = messages[idx]['content'][len(prefix):]
        if len(result) <= min_size:
            continue
        if 'omitted by ACE' in result:
            continue

        compressed = ace_compress(result, target_ratio)
        saved = len(result) - len(compressed)
        if saved <= 0:
            continue

        total_saved += saved
        messages[idx]['content'] = prefix + compressed

    return total_saved
