"""Log / error-message summariser.

Long build / test error logs contain a lot of noise (progress bars, duplicate
lines, installation banners).  :func:`summarize` strips that noise so that
the sanitised error text fits within the ``max_log_tokens`` budget configured
per retrieval mode (see :data:`~rag.context_budget.BUDGETS`).

Typical usage::

    from rag.log_summarizer import summarize
    clean = summarize(raw_log, max_tokens=500)
"""

from __future__ import annotations

import re

# ANSI colour/control sequences
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Carriage-return and backspace progress-bar overwrites
_PROGRESS = re.compile(r"[\r\b].*")
# Lines that only report successful routine operations (not failures/warnings)
_SUCCESS = re.compile(r"(?i)^(installing|downloading|fetched|up to date|already satisfied)")


def summarize(log: str, max_tokens: int) -> str:
    """Strip noise from *log* and hard-cap the result at *max_tokens*.

    Processing steps:

    1. Strip ANSI escape sequences.
    2. Remove carriage-return / backspace progress overwrites.
    3. Remove blank lines.
    4. De-duplicate identical lines (keeps first occurrence).
    5. Remove lines that only report successful routine operations (installs,
       downloads, etc.) since those are rarely actionable.
    6. If more than 100 unique lines remain, keep the first 50 and last 50
       (stack frames and summary lines are usually at the bottom).
    7. Hard-cap by character count (``max_tokens * 4`` chars, using the same
       4-chars/token heuristic as :func:`~rag.context_budget.estimate_tokens`).

    Args:
        log:        The raw log / error text to clean.
        max_tokens: Maximum number of tokens allowed in the output.

    Returns:
        A shorter, deduplicated, ANSI-free version of *log*.
    """
    log = _ANSI.sub("", log)
    keep: list[str] = []
    seen: set[str] = set()

    for raw in log.splitlines():
        line = _PROGRESS.sub("", raw).strip()
        if not line:
            continue
        if _SUCCESS.search(line):
            continue
        if line in seen:
            continue
        seen.add(line)
        keep.append(line)

    # Keep first 50 and last 50 lines — errors and stack traces tend to
    # appear towards the end.
    if len(keep) > 100:
        keep = keep[:50] + ["..."] + keep[-50:]

    text = "\n".join(keep)

    # Hard cap by token-equivalent character count.
    char_limit = max_tokens * 4
    if len(text) > char_limit:
        text = text[:char_limit]

    return text

