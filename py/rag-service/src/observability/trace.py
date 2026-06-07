"""RAG trace context manager and dataclass.

Usage::

    with start_trace(query.query, query.mode, query.base_uri) as tr:
        # ... retrieval logic ...
        tr.retrieved_spans_count = len(spans)
        tr.inserted_spans_count = len(kept)

The trace is written to ``${DATA_DIR}/traces/rag-YYYYMMDD.jsonl`` when
the context exits, regardless of success or failure.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RagTrace:
    """Structured record of a single RAG retrieval call."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    query: str = ""
    mode: str = ""
    base_uri: str = ""
    retrieved_spans_count: int = 0
    inserted_spans_count: int = 0
    dropped_spans_count: int = 0
    retrieved_tokens: int = 0
    inserted_tokens: int = 0
    deduped_tokens_saved: int = 0
    rerank_scores: list[dict] = field(default_factory=list)
    freshness_stale_count: int = 0
    freshness_recent_count: int = 0
    context_budget_used: int = 0
    expanded_spans_count: int = 0
    hardware_profile_hash: str | None = None
    backend_recommendation: dict | None = None
    retrieval_latency_ms: float = 0.0
    stages: list[dict] = field(default_factory=list)

    def add_stage(self, name: str, **kv: Any) -> None:
        """Record a named stage with elapsed milliseconds.

        Args:
            name: Human-readable stage label (e.g. ``"exact_search"``).
            **kv:  Arbitrary key/value metadata attached to the stage entry.
        """
        self.stages.append(
            {
                "name": name,
                "t_ms": round((time.perf_counter() - self._t0) * 1000, 2),
                **kv,
            }
        )

    def to_dict(self) -> dict:
        """Return a fully serialisable dict, omitting private attributes."""
        return asdict(self)


@contextmanager
def start_trace(query: str, mode: str, base_uri: str):
    """Context manager that yields a :class:`RagTrace` and writes it on exit.

    The ``_t0`` performance counter is set on the trace object immediately
    after construction so that :meth:`~RagTrace.add_stage` has a baseline.

    Args:
        query:    The user's query string.
        mode:     Retrieval mode (``ask`` / ``search`` / ``edit-small`` /
                  ``test-fix`` / ``refactor``).
        base_uri: Resource base URI (``file://...``).

    Yields:
        A :class:`RagTrace` instance that callers may annotate.
    """
    t = RagTrace(query=query, mode=mode, base_uri=base_uri)
    t._t0 = time.perf_counter()  # type: ignore[attr-defined]  # private timing anchor
    try:
        yield t
    finally:
        t.retrieval_latency_ms = round((time.perf_counter() - t._t0) * 1000, 2)  # type: ignore[attr-defined]
        from observability.jsonl_exporter import write_trace

        write_trace(t.to_dict())

