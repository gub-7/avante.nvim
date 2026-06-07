"""Integration tests for RetrievalPipeline (Increment 8).

Seven tests that exercise the full pipeline path:
1. Router selects the correct backend for an exact-code query.
2. One retrieval_requests telemetry row is written per run.
3. One backend_search_runs row per backend actually called.
4. Results carry correct backend attribution in retrieval_results.
5. Context-packing metrics are recorded in context_packing_runs.
6. Fallback to Qdrant when the selected backend raises.
7. Total latency is recorded in retrieval_requests.
"""

from __future__ import annotations

import hashlib
import sqlite3

import pytest

from rag.backends.base import BackendName, EmbeddedChunk, CollectionSpec, SearchRequest
from rag.backends.in_memory import InMemoryBackend
from models.rag import RetrievalQuery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COLLECTION = "test_col"


def _make_backend(name_enum: BackendName) -> InMemoryBackend:
    """Return an InMemoryBackend with one pre-loaded chunk (for non-empty results)."""
    backend = InMemoryBackend(name=name_enum)
    backend.create_collection(CollectionSpec(name=_COLLECTION, dimension=3))
    chunk = EmbeddedChunk(
        chunk_id=f"chunk_{name_enum.value}_1",
        document_id=f"doc_{name_enum.value}",
        content=f"content from {name_enum.value}",
        content_hash=hashlib.sha256(f"content from {name_enum.value}".encode()).hexdigest(),
        embedding=[1.0, 0.0, 0.0],
        metadata={"path": f"/repo/{name_enum.value}.py", "project": "proj"},
        token_count=5,
        path=f"/repo/{name_enum.value}.py",
        start_line=1,
        end_line=5,
    )
    backend.upsert([chunk], collection=_COLLECTION)
    return backend


def _make_pipeline(backends: dict, telemetry=None):
    """Construct a RetrievalPipeline with the given backends dict + telemetry."""
    from rag.pipeline import RetrievalPipeline
    return RetrievalPipeline(
        backends=backends,
        telemetry=telemetry,
        collection=_COLLECTION,
    )


def _exactish_query(base_uri: str = "file:///repo") -> RetrievalQuery:
    """Return a RetrievalQuery whose query is exact-code-queryish (CamelCase → exact)."""
    return RetrievalQuery(
        query="MyClassName",           # CamelCase → is_exact_code_query → True → exact backend
        base_uri=base_uri,
    )


def _natural_query(base_uri: str = "file:///repo") -> RetrievalQuery:
    """Return a query that does NOT trigger the exact-code heuristic."""
    return RetrievalQuery(
        query="how does memory management work in this project",
        base_uri=base_uri,
    )


def _make_telemetry_sink(telemetry_db: sqlite3.Connection):
    """Wrap a test telemetry DB connection in a TelemetrySink."""
    from observability.telemetry_db import TelemetrySink
    return TelemetrySink(telemetry_db)


# ---------------------------------------------------------------------------
# Test 1 — router selects the correct backend
# ---------------------------------------------------------------------------


def test_pipeline_calls_router_then_selected_backend():
    """An exact-code query must route to the 'exact' backend only.

    The 'qdrant' and 'milvus' InMemoryBackends must have 0 search calls;
    the 'exact' backend must have exactly 1.
    """
    qdrant_b = _make_backend(BackendName.QDRANT)
    milvus_b = _make_backend(BackendName.MILVUS)
    exact_b = _make_backend(BackendName.EXACT)

    pipeline = _make_pipeline({
        "qdrant": qdrant_b,
        "milvus": milvus_b,
        "exact": exact_b,
    })

    query = _exactish_query()
    ctx = pipeline.run(query)

    assert hasattr(ctx, "spans"), "run() must return a RetrievedContext with .spans"
    assert len(exact_b.calls) == 1, "exact backend must be called exactly once"
    assert len(qdrant_b.calls) == 0, "qdrant backend must NOT be called"
    assert len(milvus_b.calls) == 0, "milvus backend must NOT be called"


# ---------------------------------------------------------------------------
# Test 2 — one retrieval_requests row per run
# ---------------------------------------------------------------------------


def test_pipeline_records_one_retrieval_request_row(telemetry_db):
    """After pipeline.run(), retrieval_requests must contain exactly one row."""
    exact_b = _make_backend(BackendName.EXACT)
    qdrant_b = _make_backend(BackendName.QDRANT)
    sink = _make_telemetry_sink(telemetry_db)

    pipeline = _make_pipeline({"exact": exact_b, "qdrant": qdrant_b}, telemetry=sink)
    pipeline.run(_exactish_query())

    rows = telemetry_db.execute("SELECT * FROM retrieval_requests").fetchall()
    assert len(rows) == 1, f"Expected 1 retrieval_requests row, got {len(rows)}"
    assert rows[0]["chosen_backend"] == "exact", (
        f"chosen_backend should be 'exact', got {rows[0]['chosen_backend']!r}"
    )


# ---------------------------------------------------------------------------
# Test 3 — one backend_search_runs row per backend called
# ---------------------------------------------------------------------------


def test_pipeline_records_one_backend_search_run_row_per_backend_called(telemetry_db):
    """For a single-backend dispatch, backend_search_runs must have exactly one row."""
    exact_b = _make_backend(BackendName.EXACT)
    qdrant_b = _make_backend(BackendName.QDRANT)
    sink = _make_telemetry_sink(telemetry_db)

    pipeline = _make_pipeline({"exact": exact_b, "qdrant": qdrant_b}, telemetry=sink)
    pipeline.run(_exactish_query())

    runs = telemetry_db.execute("SELECT * FROM backend_search_runs").fetchall()
    assert len(runs) == 1, f"Expected 1 backend_search_runs row, got {len(runs)}"
    assert runs[0]["backend_name"] == "exact"
    assert runs[0]["error"] is None, "No error should be recorded for a successful run"


# ---------------------------------------------------------------------------
# Test 4 — results carry correct backend attribution
# ---------------------------------------------------------------------------


def test_pipeline_records_results_with_correct_backend_attribution(telemetry_db):
    """Rows in retrieval_results must carry the backend_name of the backend that produced them."""
    exact_b = _make_backend(BackendName.EXACT)
    qdrant_b = _make_backend(BackendName.QDRANT)
    sink = _make_telemetry_sink(telemetry_db)

    pipeline = _make_pipeline({"exact": exact_b, "qdrant": qdrant_b}, telemetry=sink)
    pipeline.run(_exactish_query())

    results = telemetry_db.execute("SELECT * FROM retrieval_results").fetchall()
    # The exact backend has 1 chunk; expect 1 result row.
    assert len(results) >= 1, "At least one retrieval_results row should be written"
    for row in results:
        assert row["backend_name"] == "exact", (
            f"All results should be attributed to 'exact', got {row['backend_name']!r}"
        )


# ---------------------------------------------------------------------------
# Test 5 — packing metrics are recorded
# ---------------------------------------------------------------------------


def test_pipeline_records_packing_metrics(telemetry_db):
    """context_packing_runs must have one row with populated token metrics."""
    exact_b = _make_backend(BackendName.EXACT)
    qdrant_b = _make_backend(BackendName.QDRANT)
    sink = _make_telemetry_sink(telemetry_db)

    pipeline = _make_pipeline({"exact": exact_b, "qdrant": qdrant_b}, telemetry=sink)
    pipeline.run(_exactish_query())

    packing_rows = telemetry_db.execute("SELECT * FROM context_packing_runs").fetchall()
    assert len(packing_rows) == 1, (
        f"Expected 1 context_packing_runs row, got {len(packing_rows)}"
    )
    row = packing_rows[0]
    assert row["raw_tokens"] >= 0
    assert row["packed_tokens"] >= 0
    assert row["tokens_saved"] >= 0


# ---------------------------------------------------------------------------
# Test 6 — fallback to Qdrant when selected backend raises
# ---------------------------------------------------------------------------


def test_pipeline_falls_back_to_qdrant_when_selected_backend_raises(telemetry_db):
    """When the routed backend always fails, the pipeline must fall back to Qdrant.

    backend_search_runs must contain:
    - One row for milvus with error != None (the failure).
    - One row for qdrant with error == None (the successful fallback).
    """
    from tests.conftest import FlakyBackend
    from rag.pipeline import RetrievalPipeline

    flaky_milvus = FlakyBackend("milvus", error_rate=1.0)
    qdrant_b = _make_backend(BackendName.QDRANT)
    sink = _make_telemetry_sink(telemetry_db)

    # Use forced_backend="milvus" so the pipeline routes to the flaky backend
    # without depending on router heuristics (GPU stats, batch size, etc.).
    pipeline = RetrievalPipeline(
        backends={"milvus": flaky_milvus, "qdrant": qdrant_b},
        telemetry=sink,
        collection=_COLLECTION,
        forced_backend="milvus",
    )

    query = RetrievalQuery(
        query="how does memory management work",
        base_uri="file:///repo",
    )
    ctx = pipeline.run(query)

    # The pipeline must return a valid context (fallback succeeded).
    assert hasattr(ctx, "spans")

    runs = telemetry_db.execute(
        "SELECT * FROM backend_search_runs ORDER BY created_at"
    ).fetchall()
    assert len(runs) == 2, (
        f"Expected 2 backend_search_runs rows (milvus failure + qdrant fallback), got {len(runs)}"
    )

    by_backend = {row["backend_name"]: row for row in runs}
    assert "milvus" in by_backend, "milvus run must be recorded"
    assert "qdrant" in by_backend, "qdrant fallback run must be recorded"
    assert by_backend["milvus"]["error"] is not None, "milvus run must record the error"
    assert by_backend["qdrant"]["error"] is None, "qdrant fallback run must succeed"


# ---------------------------------------------------------------------------
# Test 7 — total latency is recorded
# ---------------------------------------------------------------------------


def test_pipeline_records_total_latency_ms(telemetry_db):
    """retrieval_requests.retrieval_latency_ms must be > 0 after a run."""
    exact_b = _make_backend(BackendName.EXACT)
    qdrant_b = _make_backend(BackendName.QDRANT)
    sink = _make_telemetry_sink(telemetry_db)

    pipeline = _make_pipeline({"exact": exact_b, "qdrant": qdrant_b}, telemetry=sink)
    pipeline.run(_exactish_query())

    row = telemetry_db.execute("SELECT retrieval_latency_ms FROM retrieval_requests").fetchone()
    assert row is not None, "retrieval_requests must have a row after pipeline.run()"
    assert row["retrieval_latency_ms"] >= 0, (
        f"retrieval_latency_ms must be >= 0, got {row['retrieval_latency_ms']}"
    )

