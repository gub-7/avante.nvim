"""Span deduplication and overlap-merge utilities for the RAG retrieval pipeline.

Two public APIs are provided:

* :func:`dedupe_and_merge` — works on :class:`~models.rag.FileSpan` objects
  (used by the classic hybrid retriever pipeline).
* :func:`dedupe_search_results` — works on
  :class:`~rag.backends.base.SearchResult` objects produced by backend
  adapters (used by :class:`~rag.context_packer.ContextPacker`).
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from models.rag import FileSpan

if TYPE_CHECKING:
    from rag.backends.base import SearchResult

_WS = re.compile(r"\s+")


def _norm_hash(text: str) -> str:
    """Return a SHA-256 digest of *text* after collapsing all whitespace runs."""
    return hashlib.sha256(_WS.sub(" ", text).strip().encode()).hexdigest()


def dedupe_and_merge(spans: list[FileSpan]) -> tuple[list[FileSpan], int]:
    """Deduplicate and overlap-merge a list of :class:`~models.rag.FileSpan` objects.

    Two passes are performed:

    1. **Hash dedup** — spans whose normalised content hash is identical are
       collapsed into one; the survivor keeps the highest ``score`` and the
       union of ``retrieval_sources``.
    2. **Overlap merge** — within the same URI, adjacent or overlapping line
       ranges are merged into a single span to avoid sending redundant context
       to the LLM.

    Args:
        spans: Unsorted list of spans from one or more retrieval channels.

    Returns:
        A 2-tuple ``(deduped_spans, tokens_saved)`` where *tokens_saved* is
        the total ``token_estimate`` of spans that were dropped.
    """
    # ---- Pass 1: hash dedup -------------------------------------------------
    by_hash: dict[str, FileSpan] = {}
    saved = 0

    for s in spans:
        h = _norm_hash(s.content)
        existing = by_hash.get(h)
        if existing is None:
            by_hash[h] = s
        else:
            saved += s.token_estimate
            existing.score = max(existing.score, s.score)
            existing.retrieval_sources = sorted(
                set(existing.retrieval_sources + s.retrieval_sources)
            )
            if existing.reason != s.reason:
                existing.reason = f"{existing.reason}|{s.reason}"

    # ---- Pass 2: overlap merge per URI --------------------------------------
    by_uri: dict[str, list[FileSpan]] = {}
    for s in by_hash.values():
        by_uri.setdefault(s.uri, []).append(s)

    merged: list[FileSpan] = []
    for _uri, group in by_uri.items():
        # Sort by start_line (None sorts to the front; those spans can't merge)
        group.sort(key=lambda x: (x.start_line is None, x.start_line or 0))
        current = group[0]
        for nxt in group[1:]:
            can_merge = (
                current.start_line is not None
                and nxt.start_line is not None
                and (current.end_line or 0) >= (nxt.start_line or 0) - 1
            )
            if can_merge:
                new_end = max(current.end_line or 0, nxt.end_line or 0)
                saved += nxt.token_estimate
                current = FileSpan(
                    uri=_uri,
                    path=current.path,
                    start_line=current.start_line,
                    end_line=new_end,
                    content=current.content + "\n" + nxt.content,
                    reason=f"{current.reason}+{nxt.reason}",
                    score=max(current.score, nxt.score),
                    token_estimate=current.token_estimate + nxt.token_estimate,
                    hash=_norm_hash(current.content + nxt.content),
                    retrieval_sources=sorted(
                        set(current.retrieval_sources + nxt.retrieval_sources)
                    ),
                    chunk_kind=current.chunk_kind,
                    language=current.language,
                )
            else:
                merged.append(current)
                current = nxt
        merged.append(current)

    return merged, saved


# ---------------------------------------------------------------------------
# SearchResult deduplication (used by ContextPacker)
# ---------------------------------------------------------------------------


def dedupe_search_results(
    results: list[SearchResult],
) -> tuple[list[SearchResult], int]:
    """Deduplicate a list of :class:`~rag.backends.base.SearchResult` objects.

    Two passes are performed:

    1. **chunk_id dedup** — duplicate ``chunk_id`` values collapse to one
       entry; the survivor is the one with the **higher score**.
    2. **Overlap merge** — results that share the same ``path`` and have
       overlapping ``(start_line, end_line)`` ranges are merged into one.
       The merged result keeps the **higher score**, the union of line
       ranges, and the combined ``token_count`` is taken from the
       higher-scored entry (to avoid double-counting).

    Args:
        results: Unsorted list of ``SearchResult`` objects from one or
                 more backend calls.

    Returns:
        A 2-tuple ``(deduped_results, tokens_saved)`` where *tokens_saved*
        is the total ``token_count`` of results that were dropped.
    """
    # Lazy import to avoid circular dependency at module load time.
    from rag.backends.base import SearchResult as _SR  # noqa: F401

    saved = 0

    # ---- Pass 1: chunk_id dedup --------------------------------------------
    by_chunk_id: dict[str, SearchResult] = {}
    for r in results:
        existing = by_chunk_id.get(r.chunk_id)
        if existing is None:
            by_chunk_id[r.chunk_id] = r
        else:
            # Keep the higher-scored entry; count the loser's tokens.
            if r.score > existing.score:
                saved += existing.token_count
                by_chunk_id[r.chunk_id] = r
            else:
                saved += r.token_count

    # ---- Pass 2: overlap merge per path ------------------------------------
    # Group by path (results with path=None cannot overlap).
    by_path: dict[str | None, list[SearchResult]] = {}
    for r in by_chunk_id.values():
        by_path.setdefault(r.path, []).append(r)

    merged: list[SearchResult] = []

    for path, group in by_path.items():
        if path is None or len(group) == 1:
            merged.extend(group)
            continue

        # Sort by start_line (None sorts to the front; those can't overlap).
        group.sort(key=lambda x: (x.start_line is None, x.start_line or 0))

        current = group[0]
        for nxt in group[1:]:
            can_overlap = (
                current.start_line is not None
                and nxt.start_line is not None
                and current.end_line is not None
                and nxt.end_line is not None
                and (current.end_line) >= (nxt.start_line)
            )
            if can_overlap:
                # Determine the winner by score.
                if nxt.score >= current.score:
                    winner, loser = nxt, current
                else:
                    winner, loser = current, nxt

                saved += loser.token_count
                # Build a merged result: widest line range, winner's payload.
                from dataclasses import replace

                current = replace(
                    winner,
                    start_line=min(current.start_line, nxt.start_line),  # type: ignore[arg-type]
                    end_line=max(current.end_line, nxt.end_line),  # type: ignore[arg-type]
                )
            else:
                merged.append(current)
                current = nxt
        merged.append(current)

    return merged, saved

