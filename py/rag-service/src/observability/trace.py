"""RAG trace context manager and dataclass.

Usage::

    with start_trace(query.query, query.mode, query.base_uri) as tr:
        # ... retrieval logic ...
        tr.retrieved_spans_count = len(spans)
        tr.inserted_spans_count = len(kept)

The trace is written to ``${DATA_DIR}/traces/rag-YYYYMMDD.jsonl`` when
the context exits, regardless of success or failure.  If a telemetry DB
connection is available (``TelemetrySink`` wired via ``TELEMETRY_DB_PATH``),
the trace is also persisted to SQLite in a best-effort manner.
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

    # ---- Increment 7: new telemetry / routing fields ----
    request_id: str | None = None
    parent_request_id: str | None = None
    chosen_backend: str | None = None
    is_shadow: bool = False
    backend_runs: list[dict] = field(default_factory=list)

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

    On exit the trace is fanned out to:
    1. JSONL via :func:`observability.jsonl_exporter.write_trace` (existing).
    2. SQLite via :class:`observability.telemetry_db.TelemetrySink` when
       ``TELEMETRY_DB_PATH`` (or ``DATA_DIR``) resolves to a valid DB.

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
        t.retrieval_latency_ms = round(
            (time.perf_counter() - t._t0) * 1000, 2  # type: ignore[attr-defined]
        )

        d = t.to_dict()

        # 1. JSONL sink (existing, best-effort)
        from observability.jsonl_exporter import write_trace
        write_trace(d)

        # 2. SQLite telemetry sink (new, best-effort)
        _write_telemetry(t)


# ---------------------------------------------------------------------------
# Private telemetry fan-out
# ---------------------------------------------------------------------------

def _write_telemetry(t: RagTrace) -> None:
    """Write *t* to the SQLite telemetry DB in a best-effort manner.

    Silently swallows all exceptions so that a broken telemetry store never
    interrupts retrieval.
    """
    try:
        from observability.telemetry_db import TelemetrySink, init_telemetry_db

        conn = init_telemetry_db()
        sink = TelemetrySink(conn)

        req_id = t.request_id or t.trace_id  # fall back to trace_id for compat

        sink.record_request(
            request_id=req_id,
            parent_request_id=t.parent_request_id,
            query=t.query,
            mode=t.mode,
            base_uri=t.base_uri,
            chosen_backend=t.chosen_backend or "unknown",
            is_shadow=t.is_shadow,
            retrieval_latency_ms=t.retrieval_latency_ms,
        )

        # Persist any backend run records attached during retrieval
        for run in t.backend_runs:
            run_id = run.get("run_id")
            if not run_id:
                continue
            sink.record_backend_run(
                run_id=run_id,
                request_id=req_id,
                backend_name=run.get("backend_name", "unknown"),
                is_shadow=bool(run.get("is_shadow", False)),
                is_primary=bool(run.get("is_primary", True)),
                top_k=int(run.get("top_k", 0)),
                latency_ms=float(run.get("latency_ms", 0.0)),
                result_count=int(run.get("result_count", 0)),
                error=run.get("error"),
            )
            results = run.get("results", [])
            if results:
                sink.record_results(
                    request_id=req_id,
                    run_id=run_id,
                    results=results,
                )
    except Exception:
        pass

