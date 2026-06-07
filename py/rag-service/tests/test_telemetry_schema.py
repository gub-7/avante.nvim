"""TDD tests for the telemetry SQLite schema (Increment 7).

Tests verify:
1. All required tables are created by init_telemetry_db().
2. Each table's column set matches the plan's DDL exactly.
3. Calling init_telemetry_db() twice is idempotent (no error, no data loss).
"""

from __future__ import annotations

import sqlite3

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_TABLES = {
    "retrieval_requests",
    "backend_search_runs",
    "retrieval_results",
    "rerank_runs",
    "context_packing_runs",
    "answer_outcomes",
}

# Expected column names per table (matches the plan's DDL)
EXPECTED_COLUMNS: dict[str, set[str]] = {
    "retrieval_requests": {
        "request_id",
        "parent_request_id",
        "query",
        "mode",
        "base_uri",
        "chosen_backend",
        "is_shadow",
        "overlap_at_10",
        "overlap_at_50",
        "overlap_at_100",
        "retrieval_latency_ms",
        "created_at",
    },
    "backend_search_runs": {
        "run_id",
        "request_id",
        "backend_name",
        "is_shadow",
        "is_primary",
        "top_k",
        "latency_ms",
        "result_count",
        "error",
        "created_at",
    },
    "retrieval_results": {
        "id",
        "request_id",
        "run_id",
        "chunk_id",
        "score",
        "backend_name",
        "rank",
        "created_at",
    },
    "rerank_runs": {
        "id",
        "request_id",
        "input_count",
        "output_count",
        "latency_ms",
        "created_at",
    },
    "context_packing_runs": {
        "id",
        "request_id",
        "raw_tokens",
        "deduped_tokens",
        "packed_tokens",
        "tokens_saved",
        "tokens_saved_pct",
        "duplicate_rate",
        "created_at",
    },
    "answer_outcomes": {
        "id",
        "request_id",
        "created_at",
    },
}


def _get_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Return the set of column names for *table*."""
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def _get_tables(conn: sqlite3.Connection) -> set[str]:
    """Return the set of user-defined table names in the database."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    return {row[0] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_init_creates_all_five_tables(telemetry_db):
    """init_telemetry_db() must create all required telemetry tables."""
    tables = _get_tables(telemetry_db)
    # answer_outcomes is optional per the spec but we create it empty
    required_core = {
        "retrieval_requests",
        "backend_search_runs",
        "retrieval_results",
        "rerank_runs",
        "context_packing_runs",
    }
    assert required_core.issubset(tables), (
        f"Missing tables: {required_core - tables}"
    )


def test_schema_columns_match_plan(telemetry_db):
    """Every table's column set must match the plan DDL exactly."""
    for table, expected_cols in EXPECTED_COLUMNS.items():
        actual_cols = _get_columns(telemetry_db, table)
        assert expected_cols == actual_cols, (
            f"Table '{table}' column mismatch.\n"
            f"  Missing : {expected_cols - actual_cols}\n"
            f"  Extra   : {actual_cols - expected_cols}"
        )


def test_schema_migration_is_idempotent(tmp_path, monkeypatch):
    """Calling init_telemetry_db() twice must not raise and must leave schema intact."""
    import importlib

    monkeypatch.setenv("TELEMETRY_DB_PATH", str(tmp_path / "telemetry.db"))
    import observability.telemetry_db as tdb

    importlib.reload(tdb)

    conn1 = tdb.init_telemetry_db()
    conn1.close()

    # Second call — must not raise
    conn2 = tdb.init_telemetry_db()
    tables = _get_tables(conn2)
    conn2.close()

    assert "retrieval_requests" in tables
    assert "backend_search_runs" in tables

