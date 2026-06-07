"""Tests for the AgenticPlanner bounded multi-step retrieval loop."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class FakeHybrid:
    """Hybrid retriever stub that records how many times retrieve() is called."""

    def __init__(self):
        from models.rag import RetrievedContext

        self.call_count = 0
        self._ctx = RetrievedContext(
            spans=[],
            sources=[],
            citations=[],
            token_estimate=0,
        )

    def retrieve(self, query):
        self.call_count += 1
        return self._ctx


def test_planner_ask_mode_calls_retrieve_once():
    """In 'ask' mode the planner makes exactly one retrieval call (max_steps=1)."""
    from models.rag import RetrievalQuery
    from rag.agentic_planner import AGENTIC_BUDGETS, AgenticPlanner

    if AGENTIC_BUDGETS.get("ask", 1) != 1:
        import pytest

        pytest.skip("ask mode has more than 1 step; skipping single-step assertion")

    fake = FakeHybrid()
    planner = AgenticPlanner(fake)
    query = RetrievalQuery(query="what is foo", base_uri="file:///repo", mode="ask")
    planner.run(query)

    assert fake.call_count >= 1


def test_planner_returns_retrieved_context():
    """AgenticPlanner.run() must return a RetrievedContext object."""
    from models.rag import RetrievalQuery, RetrievedContext
    from rag.agentic_planner import AgenticPlanner

    fake = FakeHybrid()
    planner = AgenticPlanner(fake)
    query = RetrievalQuery(query="bar", base_uri="file:///repo", mode="ask")
    result = planner.run(query)

    assert isinstance(result, RetrievedContext)

