"""
Line-level importance classifier for ACE (Attention-Weighted Context Eviction).

Each line in a tool-result block is assigned a score in [0.0, 1.0] that
reflects how structurally or informationally significant it is for an
ongoing agentic task.  Higher scores are retained under budget pressure.
"""

from __future__ import annotations

import json
import re

# ---------------------------------------------------------------------------
# Patterns (ordered from most- to least-specific)
# ---------------------------------------------------------------------------

_RE_TOOL_CALL = re.compile(
    r'^\s*\{.*"name"\s*:.*"(arguments|input)"\s*:',
    re.DOTALL,
)

_RE_ERROR = re.compile(
    r"(traceback|error|exception|errno|failed|fatal|critical|abort|panic)",
    re.IGNORECASE,
)

_RE_PATH = re.compile(
    r"(/tmp/|/results?/|/workspace/|/home/|/var/|/usr/|\.py\b|\.json\b|\.sh\b|\.yaml\b|\.toml\b)"
)

_RE_TASK_KEYWORD = re.compile(
    r"\b(task|step\s*\d|verify|inspect|check|assert|require|must|should|goal|objective)\b",
    re.IGNORECASE,
)

_RE_NUMERIC = re.compile(
    r"(\d{4,}|[0-9]+\.[0-9]+\s*%|[0-9]+\s*(MB|GB|KB|ms|s\b|tokens?|chars?))",
    re.IGNORECASE,
)

_RE_SHELL = re.compile(
    r"(^\s*\$\s+|\bcurl\b|\bpython3?\b|\bdocker\b|\bpip\b|\bnpm\b|\bsudo\b|\bchmod\b|\bmkdir\b|\bgrep\b|\bawk\b|\bsed\b)"
)

_RE_SUCCESS_BOILERPLATE = re.compile(
    r"^\s*(done|ok|okay|success|installed|saved|complete|finished|updated|created|deleted|removed|copied|moved)\s*\.?\s*$",
    re.IGNORECASE,
)

_RE_META_COMMENTARY = re.compile(
    r"^\s*(here (are|is)|the (following|results?|output|above)|as (you can see|shown|requested)|"
    r"please (find|note|review)|i (have|will|can)|this (is|shows))\b",
    re.IGNORECASE,
)

_MAX_LINE_LENGTH = 200


def classify_line(line: str) -> float:
    """Return an importance score in [0.0, 1.0] for a single text line.

    Parameters
    ----------
    line:
        A single line of text from a tool-result block (newline stripped).

    Returns
    -------
    float
        Importance score:

        ======  ==============================================================
        1.00    Tool call JSON (contains ``"name"`` + ``"arguments"``/``"input"``)
        0.95    Error / exception / traceback lines
        0.90    File system paths (``/tmp/``, ``.py``, etc.)
        0.85    Task-framing keywords (``task:``, ``step``, ``verify``, …)
        0.70    Numeric data (large numbers, percentages, units)
        0.60    Shell commands (``$``, ``curl``, ``python3``, …)
        0.50    Default (informational prose)
        0.30    Success boilerplate (``done``, ``ok``, ``installed``, …)
        0.20    Very long lines (> 200 chars, likely raw data dumps)
        0.10    Meta-commentary ("here are the results", …)
        0.00    Blank lines
        ======  ==============================================================
    """
    stripped = line.strip()

    # 0.00 — blank
    if not stripped:
        return 0.0

    # 1.00 — tool-call JSON
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict) and "name" in obj and (
            "arguments" in obj or "input" in obj
        ):
            return 1.0
    except (json.JSONDecodeError, ValueError):
        pass
    if _RE_TOOL_CALL.search(line):
        return 1.0

    # 0.95 — error / exception
    if _RE_ERROR.search(stripped):
        return 0.95

    # 0.10 — meta-commentary (check before path/task so it doesn't get a
    # false upgrade on lines that happen to contain a path)
    if _RE_META_COMMENTARY.match(stripped):
        return 0.10

    # 0.90 — file/workspace paths
    if _RE_PATH.search(stripped):
        return 0.90

    # 0.85 — task-framing keywords
    if _RE_TASK_KEYWORD.search(stripped):
        return 0.85

    # 0.70 — numeric data
    if _RE_NUMERIC.search(stripped):
        return 0.70

    # 0.60 — shell commands
    if _RE_SHELL.search(stripped):
        return 0.60

    # 0.30 — success boilerplate
    if _RE_SUCCESS_BOILERPLATE.match(stripped):
        return 0.30

    # 0.20 — very long lines
    if len(stripped) > _MAX_LINE_LENGTH:
        return 0.20

    # 0.50 — default
    return 0.50
