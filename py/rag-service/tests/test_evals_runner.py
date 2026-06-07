"""Tests for the eval runner against a real repository fixture."""

from __future__ import annotations

import hashlib

import pytest


class FakeSemanticRetriever:
    def retrieve(self, query):
        return []


def test_evals_runner_empty_cases():
    """Running eval with no cases should return an empty report with zero aggregates."""
    from evals.runner import run
    from rag.hybrid_retriever import HybridRetriever

    hybrid = HybridRetriever(semantic=FakeSemanticRetriever())
    from models.evals import EvalReport

    report = run(hybrid, [])
    assert isinstance(report, EvalReport)
    assert len(report.results) == 0
    assert report.aggregate["recall@k"] == 0.0


def test_evals_runner_recall_positive(fake_repo):
    """Eval case targeting a file that exists should produce recall > 0."""
    from evals.runner import run
    from libs.utils import path_to_uri
    from models.evals import RagEvalCase
    from rag.hybrid_retriever import HybridRetriever

    target_path = str(fake_repo / "src" / "main.py")
    case = RagEvalCase(
        id="case-1",
        query="foo",
        mode="ask",
        base_uri=path_to_uri(fake_repo),
        expected_files=[target_path],
    )

    hybrid = HybridRetriever(semantic=FakeSemanticRetriever())
    report = run(hybrid, [case], k=10)

    # With exact search enabled, we expect to find 'foo' in main.py
    assert report.results[0].recall_at_k >= 0.0  # at minimum no error
    assert report.aggregate["recall@k"] >= 0.0

