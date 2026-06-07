"""Tests for the ripgrep-backed exact search module."""

from __future__ import annotations

import pytest


def test_exact_search_returns_spans(fake_repo):
    """ExactSearch.retrieve should find 'def foo' in the fake repo."""
    from libs.utils import path_to_uri
    from models.rag import RetrievalQuery
    from rag.exact_search import ExactSearch

    query = RetrievalQuery(
        query="foo",
        base_uri=path_to_uri(fake_repo),
    )
    spans = ExactSearch().retrieve(query, fake_repo)
    assert len(spans) >= 1, "expected at least one span for query 'foo'"


def test_exact_search_span_has_retrieval_source(fake_repo):
    """Every span must be tagged with retrieval_sources=['exact']."""
    from libs.utils import path_to_uri
    from models.rag import RetrievalQuery
    from rag.exact_search import ExactSearch

    query = RetrievalQuery(
        query="foo",
        base_uri=path_to_uri(fake_repo),
    )
    spans = ExactSearch().retrieve(query, fake_repo)
    for s in spans:
        assert "exact" in s.retrieval_sources

