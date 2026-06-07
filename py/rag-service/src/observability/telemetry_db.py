"""SQLite telemetry sink for RAG retrieval observability (Increment 7).

Provides:
- ``init_telemetry_db()`` — creates all telemetry tables (idempotent).
- ``TelemetrySink``      — best-effort writer; never raises into callers.

The database path is controlled by the ``TELEMETRY_DB_PATH`` env var,
defaulting to ``${DATA_DIR}/telemetry/telemetry.db``.

Usage::

    conn = init_telemetry_db()
    sink = TelemetrySink(conn)
    sink.record_request(request_id=..., ...)
    sink.record_backend_run(run_id=..., ...)
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _default_db_path() -> Path:
    """Return the default telemetry DB path, respecting DATA_DIR."""
    # Lazy import to avoid circular dependency at module load time
    base = Path(os.environ.get("DATA_DIR", "data"))
    tel_dir = base / "telemetry"
    tel_dir.mkdir(parents=True, exist_ok=True)
    return tel_dir / "telemetry.db"


def _db_path() -> Path:
    """Resolve the telemetry DB path from env or default."""
    env_val = os.environ.get("TELEMETRY_DB_PATH")
    if env_val:
        p = Path(env_val)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return _default_db_path()


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS retrieval_requests (
    request_id        TEXT PRIMARY KEY,
    parent_request_id TEXT,
    query             TEXT NOT NULL,
    mode              TEXT NOT NULL,
    base_uri          TEXT NOT NULL,
    chosen_backend    TEXT NOT NULL,
    is_shadow         INTEGER NOT NULL DEFAULT 0,
    overlap_at_10     REAL,
    overlap_at_50     REAL,
    overlap_at_100    REAL,
    retrieval_latency_ms REAL NOT NULL DEFAULT 0.0,
    created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS backend_search_runs (
    run_id        TEXT PRIMARY KEY,
    request_id    TEXT NOT NULL,
    backend_name  TEXT NOT NULL,
    is_shadow     INTEGER NOT NULL DEFAULT 0,
    is_primary    INTEGER NOT NULL DEFAULT 1,
    top_k         INTEGER NOT NULL DEFAULT 0,
    latency_ms    REAL NOT NULL DEFAULT 0.0,
    result_count  INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS retrieval_results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id    TEXT NOT NULL,
    run_id        TEXT NOT NULL,
    chunk_id      TEXT NOT NULL,
    score         REAL NOT NULL DEFAULT 0.0,
    backend_name  TEXT NOT NULL,
    rank          INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rerank_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id    TEXT NOT NULL,
    input_count   INTEGER NOT NULL DEFAULT 0,
    output_count  INTEGER NOT NULL DEFAULT 0,
    latency_ms    REAL NOT NULL DEFAULT 0.0,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS context_packing_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      TEXT NOT NULL,
    raw_tokens      INTEGER NOT NULL DEFAULT 0,
    deduped_tokens  INTEGER NOT NULL DEFAULT 0,
    packed_tokens   INTEGER NOT NULL DEFAULT 0,
    tokens_saved    INTEGER NOT NULL DEFAULT 0,
    tokens_saved_pct REAL NOT NULL DEFAULT 0.0,
    duplicate_rate  REAL NOT NULL DEFAULT 0.0,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS answer_outcomes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id    TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_telemetry_db(path: Path | None = None) -> sqlite3.Connection:
    """Create all telemetry tables (idempotent) and return an open connection.

    Args:
        path: Optional override for the DB file path.  Defaults to
              ``TELEMETRY_DB_PATH`` env var or ``${DATA_DIR}/telemetry/telemetry.db``.

    Returns:
        An open :class:`sqlite3.Connection` with ``row_factory`` set to
        :class:`sqlite3.Row`.
    """
    db_path = path if path is not None else _db_path()
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_CREATE_TABLES_SQL)
    conn.commit()
    return conn


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class TelemetrySink:
    """Best-effort writer for telemetry tables.

    All ``record_*`` methods return ``True`` on success and ``False`` if any
    database error occurs.  Exceptions are silently swallowed so that a broken
    telemetry store never interrupts retrieval — matching the existing
    ``write_trace`` best-effort contract.

    Args:
        conn: An open :class:`sqlite3.Connection` pointing at a database that
              has been initialised by :func:`init_telemetry_db`.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # ------------------------------------------------------------------
    # Public record methods
    # ------------------------------------------------------------------

    def record_request(
        self,
        *,
        request_id: str,
        query: str,
        mode: str,
        base_uri: str,
        chosen_backend: str,
        is_shadow: bool,
        retrieval_latency_ms: float,
        parent_request_id: str | None = None,
        overlap_at_10: float | None = None,
        overlap_at_50: float | None = None,
        overlap_at_100: float | None = None,
    ) -> bool:
        """Insert a row into ``retrieval_requests``.

        Returns:
            ``True`` on success, ``False`` on any DB error.
        """
        try:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO retrieval_requests
                    (request_id, parent_request_id, query, mode, base_uri,
                     chosen_backend, is_shadow, overlap_at_10, overlap_at_50,
                     overlap_at_100, retrieval_latency_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    parent_request_id,
                    query,
                    mode,
                    base_uri,
                    chosen_backend,
                    int(is_shadow),
                    overlap_at_10,
                    overlap_at_50,
                    overlap_at_100,
                    retrieval_latency_ms,
                    _now_iso(),
                ),
            )
            self._conn.commit()
            return True
        except Exception:
            return False

    def record_backend_run(
        self,
        *,
        run_id: str,
        request_id: str,
        backend_name: str,
        is_shadow: bool,
        is_primary: bool,
        top_k: int,
        latency_ms: float,
        result_count: int,
        error: str | None,
    ) -> bool:
        """Insert a row into ``backend_search_runs``.

        Returns:
            ``True`` on success, ``False`` on any DB error.
        """
        try:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO backend_search_runs
                    (run_id, request_id, backend_name, is_shadow, is_primary,
                     top_k, latency_ms, result_count, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    request_id,
                    backend_name,
                    int(is_shadow),
                    int(is_primary),
                    top_k,
                    latency_ms,
                    result_count,
                    error,
                    _now_iso(),
                ),
            )
            self._conn.commit()
            return True
        except Exception:
            return False

    def record_results(
        self,
        *,
        request_id: str,
        run_id: str,
        results: list[dict[str, Any]],
    ) -> bool:
        """Insert one row per result into ``retrieval_results``.

        Each item in *results* must have keys: ``chunk_id``, ``score``,
        ``backend_name``, ``rank``.

        Returns:
            ``True`` on success, ``False`` on any DB error.
        """
        try:
            now = _now_iso()
            rows = [
                (
                    request_id,
                    run_id,
                    r["chunk_id"],
                    r.get("score", 0.0),
                    r.get("backend_name", ""),
                    r.get("rank", 0),
                    now,
                )
                for r in results
            ]
            # Always call executemany (even with an empty list) so that a broken
            # connection is always exercised — consistent best-effort contract.
            self._conn.executemany(
                """
                INSERT INTO retrieval_results
                    (request_id, run_id, chunk_id, score, backend_name, rank, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()
            return True
        except Exception:
            return False

    def record_rerank(
        self,
        *,
        request_id: str,
        input_count: int,
        output_count: int,
        latency_ms: float,
    ) -> bool:
        """Insert a row into ``rerank_runs``.

        Returns:
            ``True`` on success, ``False`` on any DB error.
        """
        try:
            self._conn.execute(
                """
                INSERT INTO rerank_runs
                    (request_id, input_count, output_count, latency_ms, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (request_id, input_count, output_count, latency_ms, _now_iso()),
            )
            self._conn.commit()
            return True
        except Exception:
            return False

    def record_packing(
        self,
        *,
        request_id: str,
        raw_tokens: int,
        deduped_tokens: int,
        packed_tokens: int,
        tokens_saved: int,
        tokens_saved_pct: float,
        duplicate_rate: float,
    ) -> bool:
        """Insert a row into ``context_packing_runs``.

        Returns:
            ``True`` on success, ``False`` on any DB error.
        """
        try:
            self._conn.execute(
                """
                INSERT INTO context_packing_runs
                    (request_id, raw_tokens, deduped_tokens, packed_tokens,
                     tokens_saved, tokens_saved_pct, duplicate_rate, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    raw_tokens,
                    deduped_tokens,
                    packed_tokens,
                    tokens_saved,
                    tokens_saved_pct,
                    duplicate_rate,
                    _now_iso(),
                ),
            )
            self._conn.commit()
            return True
        except Exception:
            return False

