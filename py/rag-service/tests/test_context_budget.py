"""Tests for context budget estimation and trimming."""

from __future__ import annotations


def _make_span(content: str, uri: str = "file:///repo/a.py"):
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
        retrieval_sources=["exact"],
    )


def test_estimate_tokens_empty():
    from rag.context_budget import estimate_tokens

    assert estimate_tokens("") == 0


def test_estimate_tokens_nonempty():
    from rag.context_budget import estimate_tokens

    # Heuristic: ~4 chars per token
    assert estimate_tokens("abcd") >= 1


def test_apply_budget_caps_tokens():
    """apply_budget must not exceed the max_total_tokens for the given mode."""
    from rag.context_budget import apply_budget, estimate_tokens

    # Build 20 spans of ~200 chars each
    big_content = "x" * 200
    spans = [_make_span(big_content, uri=f"file:///repo/{i}.py") for i in range(20)]
    # Sort by score desc (budget expects pre-sorted input)
    spans.sort(key=lambda s: s.score, reverse=True)

    kept, dropped = apply_budget(spans, mode="ask", override_total=100)
    total = sum(s.token_estimate for s in kept)
    assert total <= 100 or len(kept) == 1, (
        f"Budget not respected: total={total}, kept={len(kept)}"
    )

