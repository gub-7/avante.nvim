"""Tests for the deduplication and overlap-merge logic."""

from __future__ import annotations


def _make_span(content: str, uri: str = "file:///repo/a.py", sources: list[str] | None = None):
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
        retrieval_sources=sources or ["exact"],
    )


def test_dedupe_removes_duplicate(tmp_path):
    """Two spans with identical content should collapse to one."""
    from rag.dedupe import dedupe_and_merge

    s1 = _make_span("def foo(): pass")
    s2 = _make_span("def foo(): pass")

    deduped, tokens_saved = dedupe_and_merge([s1, s2])
    assert len(deduped) == 1
    assert tokens_saved > 0


def test_dedupe_preserves_unique_spans(tmp_path):
    """Two spans with distinct content should both survive deduplication."""
    from rag.dedupe import dedupe_and_merge

    s1 = _make_span("def foo(): pass")
    s2 = _make_span("class Bar: pass", uri="file:///repo/b.py")

    deduped, tokens_saved = dedupe_and_merge([s1, s2])
    assert len(deduped) == 2
    assert tokens_saved == 0

