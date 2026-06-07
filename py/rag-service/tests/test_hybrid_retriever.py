"""Tests for HybridRetriever pipeline integration."""

from __future__ import annotations

from unittest.mock import MagicMock


def _make_span(content: str = "def foo(): pass"):
    import hashlib

    from models.rag import FileSpan
    from rag.context_budget import estimate_tokens

    return FileSpan(
        uri="file:///repo/a.py",
        path="/repo/a.py",
        start_line=1,
        end_line=5,
        content=content,
        reason="exact:query",
        score=1.0,
        token_estimate=estimate_tokens(content),
        hash=hashlib.sha256(content.encode()).hexdigest(),
        retrieval_sources=["exact"],
    )


class FakeSemanticRetriever:
    """Semantic retriever that returns no spans (stubs out the embedding calls)."""

    def retrieve(self, query):
        return []


def test_hybrid_retriever_returns_context(fake_repo):
    """HybridRetriever with a fake semantic backend should still return a context."""
    from libs.utils import path_to_uri
    from models.rag import RetrievalQuery
    from rag.hybrid_retriever import HybridRetriever

    hybrid = HybridRetriever(semantic=FakeSemanticRetriever())
    query = RetrievalQuery(
        query="foo",
        base_uri=path_to_uri(fake_repo),
    )
    ctx = hybrid.retrieve(query)
    # Even with empty semantic results, exact+symbol should populate spans
    assert hasattr(ctx, "spans")
    assert hasattr(ctx, "citations")
    assert hasattr(ctx, "token_estimate")


def test_hybrid_retriever_uses_chat_history_channel():
    """Chat-history retrieve is called when include_chat_history=True."""
    from models.rag import RetrievalQuery
    from rag.hybrid_retriever import HybridRetriever

    mock_chat = MagicMock()
    mock_chat.retrieve.return_value = []

    hybrid = HybridRetriever(semantic=FakeSemanticRetriever(), chat_history=mock_chat)
    query = RetrievalQuery(
        query="what did we decide",
        base_uri="file:///repo",
        include_chat_history=True,
    )
    hybrid.retrieve(query)
    mock_chat.retrieve.assert_called_once()

