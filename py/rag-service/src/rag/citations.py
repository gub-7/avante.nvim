"""
Citation helpers for the RAG pipeline.

Provides a single utility function that converts a ranked list of
:class:`~models.rag.FileSpan` objects into lightweight
:class:`~models.rag.ContextCitation` records suitable for returning to
callers without the full span content.

Phase 7 implementation.
"""

from __future__ import annotations

from models.rag import ContextCitation, FileSpan


def build_citations(spans: list[FileSpan]) -> list[ContextCitation]:
    """
    Convert *spans* to :class:`~models.rag.ContextCitation` objects.

    Each citation records the source location (URI, path, line range),
    the human-readable reason the span was retrieved, and which retrieval
    channels contributed to it.

    Args:
        spans: Ordered list of :class:`~models.rag.FileSpan` objects, typically
            the ``kept`` output from :func:`~rag.context_budget.apply_budget`.

    Returns:
        Parallel list of :class:`~models.rag.ContextCitation` objects (one per
        span, same order).

    """
    return [
        ContextCitation(
            uri=s.uri,
            path=s.path,
            start_line=s.start_line,
            end_line=s.end_line,
            reason=s.reason,
            retrieval_sources=s.retrieval_sources,
        )
        for s in spans
    ]

