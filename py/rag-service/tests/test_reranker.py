"""Tests for the multi-signal reranker."""

from __future__ import annotations


def _make_span(content: str, sources: list[str], uri: str = "file:///repo/a.py"):
    import hashlib

    from models.rag import FileSpan
    from rag.context_budget import estimate_tokens

    return FileSpan(
        uri=uri,
        path=uri.removeprefix("file://"),
        start_line=1,
        end_line=5,
        content=content,
        reason="test",
        score=1.0,
        token_estimate=estimate_tokens(content),
        hash=hashlib.sha256(content.encode()).hexdigest(),
        retrieval_sources=sources,
    )


def test_exact_outranks_semantic():
    """An exact-tagged span should score higher than a semantic-only span."""
    from models.rag import RetrievalQuery
    from rag.reranker import rerank

    exact_span = _make_span("def foo(): pass", sources=["exact"])
    semantic_span = _make_span("def foo(): pass", sources=["semantic"], uri="file:///repo/b.py")

    query = RetrievalQuery(query="foo", base_uri="file:///repo")
    ranked = rerank([semantic_span, exact_span], query)

    # The span with source "exact" must rank first (or at least not last)
    first_span = ranked[0][0]
    assert "exact" in first_span.retrieval_sources, (
        f"Expected exact-tagged span to rank first, got sources={first_span.retrieval_sources}"
    )

