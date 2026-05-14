"""
ace_eviction — Content-aware context compression for multi-turn agentic LLM inference.

Public API
----------
ACEClassifier
    Line-level importance classifier (see :func:`classify_line`).
ACECompressor
    Compresses a single text block to a target line fraction.
ace_evict
    Compresses old tool-result messages in a conversation until it fits a
    character budget.

Convenience re-exports
----------------------
classify_line(line) -> float
    Score a single line.
compress(content, target_ratio=0.4) -> str
    Compress a multi-line string.
"""

from .classifier import classify_line
from .compress import ACECompressor, compress
from .evict import ace_evict

# Expose classifier module as a named symbol so users can do
#   from ace_eviction import ACEClassifier; ACEClassifier.classify_line(...)
# without importing the module directly.
import ace_eviction.classifier as ACEClassifier  # noqa: E402

__all__ = [
    "ACEClassifier",
    "ACECompressor",
    "ace_evict",
    "classify_line",
    "compress",
]

__version__ = "1.0.0"
