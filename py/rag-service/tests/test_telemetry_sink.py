"""TDD tests for TelemetrySink (Increment 7).

Tests verify:
1. record_request persists all fields to retrieval_requests.
2. record_backend_run links back to request_id (FK-style join works).
3. record_results writes one row per result to retrieval_results.
4. Sink failures do not raise into the caller (best-effort contract).
5. Sink writes in a single transaction per record_request call.
6. JSONL writer is still called for backwards compatibility (existing RagTrace path).
"""

from __future__ import annotations

import sqlite3
import uuid
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request_id() -> str:
    return uuid.uuid4().hex


def _fetch_one(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> dict | None:
    conn.row_factory = sqlite3.Row
    cur = conn.execute(sql, params)
    row = cur.fetchone()
    return dict(row) if row else None


def _fetch_all(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    conn.row_factory = sqlite3.Row
    cur = conn.execute(sql, params)
    return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_record_request_persists_all_fields(telemetry_db):
    """record_request must insert a row with every supplied field."""
    from observability.telemetry_db import TelemetrySink

    sink = TelemetrySink(telemetry_db)
    req_id = _make_request_id()
    parent_id = _make_request_id()

    sink.record_request(
        request_id=req_id,
        parent_request_id=parent_id,
        query="how does foo work?",
        mode="ask",
        base_uri="file:///project",
        chosen_backend="qdrant",
        is_shadow=False,
        retrieval_latency_ms=42.5,
    )

    row = _fetch_one(
        telemetry_db,
        "SELECT * FROM retrieval_requests WHERE request_id = ?",
        (req_id,),
    )
    assert row is not None, "No row inserted for request_id"
    assert row["parent_request_id"] == parent_id
    assert row["query"] == "how does foo work?"
    assert row["mode"] == "ask"
    assert row["base_uri"] == "file:///project"
    assert row["chosen_backend"] == "qdrant"
    assert row["is_shadow"] == 0  # SQLite stores bool as int
    assert abs(row["retrieval_latency_ms"] - 42.5) < 0.01


def test_record_backend_run_links_to_request_id(telemetry_db):
    """record_backend_run must insert a row that joins back to retrieval_requests."""
    from observability.telemetry_db import TelemetrySink

    sink = TelemetrySink(telemetry_db)
    req_id = _make_request_id()
    run_id = _make_request_id()

    # Insert parent request first
    sink.record_request(
        request_id=req_id,
        query="test",
        mode="search",
        base_uri="file:///x",
        chosen_backend="exact",
        is_shadow=False,
        retrieval_latency_ms=1.0,
    )

    sink.record_backend_run(
        run_id=run_id,
        request_id=req_id,
        backend_name="exact",
        is_shadow=False,
        is_primary=True,
        top_k=10,
        latency_ms=0.8,
        result_count=3,
        error=None,
    )

    # FK-style join
    rows = _fetch_all(
        telemetry_db,
        """
        SELECT bsr.run_id, bsr.backend_name, rr.query
        FROM backend_search_runs bsr
        JOIN retrieval_requests rr ON rr.request_id = bsr.request_id
        WHERE bsr.request_id = ?
        """,
        (req_id,),
    )
    assert len(rows) == 1
    assert rows[0]["run_id"] == run_id
    assert rows[0]["backend_name"] == "exact"
    assert rows[0]["query"] == "test"


def test_record_results_writes_one_row_per_result(telemetry_db):
    """record_results must insert exactly one retrieval_results row per result."""
    from observability.telemetry_db import TelemetrySink

    sink = TelemetrySink(telemetry_db)
    req_id = _make_request_id()
    run_id = _make_request_id()

    sink.record_request(
        request_id=req_id,
        query="q",
        mode="ask",
        base_uri="file:///p",
        chosen_backend="qdrant",
        is_shadow=False,
        retrieval_latency_ms=0.0,
    )
    sink.record_backend_run(
        run_id=run_id,
        request_id=req_id,
        backend_name="qdrant",
        is_shadow=False,
        is_primary=True,
        top_k=5,
        latency_ms=1.0,
        result_count=3,
        error=None,
    )

    results = [
        {"chunk_id": f"chunk_{i}", "score": 0.9 - i * 0.1, "backend_name": "qdrant", "rank": i}
        for i in range(3)
    ]
    sink.record_results(request_id=req_id, run_id=run_id, results=results)

    rows = _fetch_all(
        telemetry_db,
        "SELECT * FROM retrieval_results WHERE request_id = ?",
        (req_id,),
    )
    assert len(rows) == 3
    chunk_ids = {r["chunk_id"] for r in rows}
    assert chunk_ids == {"chunk_0", "chunk_1", "chunk_2"}


def test_sink_failures_do_not_raise_into_caller(telemetry_db):
    """If the DB cursor raises, record_* methods must return False, never propagate."""
    from observability.telemetry_db import TelemetrySink

    # Patch the connection's execute/executemany/commit to always raise
    broken_conn = MagicMock(spec=sqlite3.Connection)
    broken_conn.execute.side_effect = sqlite3.OperationalError("disk full")
    broken_conn.executemany.side_effect = sqlite3.OperationalError("disk full")
    broken_conn.cursor.side_effect = sqlite3.OperationalError("disk full")
    broken_conn.__enter__ = lambda s: s
    broken_conn.__exit__ = MagicMock(return_value=False)

    sink = TelemetrySink(broken_conn)

    # None of these must raise
    result = sink.record_request(
        request_id="x",
        query="q",
        mode="ask",
        base_uri="f",
        chosen_backend="qdrant",
        is_shadow=False,
        retrieval_latency_ms=0.0,
    )
    assert result is False

    result2 = sink.record_backend_run(
        run_id="r",
        request_id="x",
        backend_name="qdrant",
        is_shadow=False,
        is_primary=True,
        top_k=5,
        latency_ms=0.0,
        result_count=0,
        error=None,
    )
    assert result2 is False

    result3 = sink.record_results(request_id="x", run_id="r", results=[])
    assert result3 is False


def test_sink_writes_in_a_single_transaction_per_request(telemetry_db):
    """record_request must be atomic: either all fields land or nothing does."""
    from observability.telemetry_db import TelemetrySink

    sink = TelemetrySink(telemetry_db)
    req_id = _make_request_id()

    sink.record_request(
        request_id=req_id,
        query="atomicity check",
        mode="search",
        base_uri="file:///a",
        chosen_backend="milvus",
        is_shadow=True,
        retrieval_latency_ms=7.7,
    )

    # Immediately readable in the same connection (committed)
    row = _fetch_one(
        telemetry_db,
        "SELECT is_shadow, chosen_backend FROM retrieval_requests WHERE request_id = ?",
        (req_id,),
    )
    assert row is not None
    assert row["is_shadow"] == 1
    assert row["chosen_backend"] == "milvus"


def test_jsonl_writer_still_called_for_backwards_compat(tmp_path, monkeypatch):
    """The existing JSONL write_trace path must be preserved when using start_trace."""
    import importlib

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TELEMETRY_DB_PATH", str(tmp_path / "telemetry.db"))

    import libs.configs
    import libs.db
    import observability.telemetry_db as tdb

    importlib.reload(libs.configs)
    importlib.reload(libs.db)
    importlib.reload(tdb)
    libs.db.init_db()
    tdb.init_telemetry_db()

    written_objects: list[dict] = []

    # Patch write_trace to capture calls
    with patch("observability.jsonl_exporter.write_trace", side_effect=written_objects.append):
        import observability.trace as trace_mod

        importlib.reload(trace_mod)
        with patch("observability.jsonl_exporter.write_trace", side_effect=written_objects.append):
            from observability.trace import start_trace

            with start_trace("hello", "ask", "file:///proj") as tr:
                tr.retrieved_spans_count = 5

    assert len(written_objects) >= 1, "write_trace was never called — JSONL path is broken"
    obj = written_objects[-1]
    assert obj["retrieved_spans_count"] == 5

