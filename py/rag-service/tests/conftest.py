"""Shared pytest fixtures for the RAG service test suite."""

from __future__ import annotations

import importlib
import random
import sqlite3
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from rag.backends.base import SearchRequest, SearchResult


# ---------------------------------------------------------------------------
# FlakyBackend — test double for fallback / reliability tests
# ---------------------------------------------------------------------------


class FlakyBackend:
    """Test double that fails randomly according to *error_rate*.

    Useful for testing the pipeline's fallback behaviour when a backend
    raises an exception.

    Args:
        name:        Backend name string (e.g. ``"milvus"``).
        error_rate:  Probability [0, 1] that each ``search()`` call raises.
                     Use ``1.0`` for a backend that always fails.
        latency_ms:  Simulated per-call latency in milliseconds.
    """

    def __init__(self, name: str, error_rate: float = 0.5, latency_ms: float = 0.0) -> None:
        self.name = name
        self.error_rate = error_rate
        self.latency_ms = latency_ms
        # Track calls for test assertions.
        self.calls: list = []

    def search(self, request: "SearchRequest") -> list["SearchResult"]:
        """Raise ``RuntimeError`` with probability *error_rate*, else return ``[]``."""
        self.calls.append(request)
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)
        if self.error_rate >= 1.0 or (self.error_rate > 0.0 and random.random() < self.error_rate):
            raise RuntimeError(f"FlakyBackend({self.name!r}) simulated failure")
        return []

    def is_available(self) -> bool:
        """Return True (so the pipeline doesn't skip it before calling search)."""
        return True

    def create_collection(self, spec) -> None:  # noqa: ANN001
        """No-op."""

    def upsert(self, chunks: list, collection: str) -> None:
        """No-op."""

    def delete_by_filter(self, collection: str, filters: list) -> int:
        """No-op."""
        return 0

    def stats(self):
        """Return minimal stats."""
        from rag.backends.base import BackendStats, BackendName
        try:
            backend_name = BackendName(self.name)
        except ValueError:
            backend_name = BackendName.IN_MEMORY
        return BackendStats(backend=backend_name, collection_count=0, vector_count=0)


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Redirect all data I/O to a fresh temp directory for each test.

    Reloads ``libs.configs`` and ``libs.db`` so that module-level path
    constants pick up the new ``DATA_DIR`` value, then initialises the
    SQLite schema from scratch.

    Yields:
        The temporary ``Path`` object used as ``DATA_DIR``.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # Re-import to pick up new BASE_DATA_DIR
    import libs.configs
    import libs.db

    importlib.reload(libs.configs)
    importlib.reload(libs.db)
    libs.db.init_db()
    yield tmp_path


@pytest.fixture
def fake_repo(tmp_path):
    """Create a minimal fake git repository for indexing tests.

    The repository has a ``.git`` directory and two Python source files
    inside ``src/``:

    - ``src/main.py`` — defines ``foo()`` and ``class Bar``.
    - ``src/test_main.py`` — a trivial pytest test.

    Returns:
        The root ``Path`` of the fake repository (same as ``tmp_path``).
    """
    (tmp_path / ".git").mkdir()
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text(
        "def foo():\n    return 1\n\nclass Bar:\n    def baz(self):\n        return 2\n"
    )
    (src / "test_main.py").write_text("def test_foo():\n    assert foo() == 1\n")
    return tmp_path


@pytest.fixture
def telemetry_db(tmp_path, monkeypatch) -> sqlite3.Connection:
    """Provide an initialised telemetry SQLite connection in a temp directory.

    Sets ``TELEMETRY_DB_PATH`` to ``tmp_path/telemetry.db``, initialises all
    telemetry tables via :func:`observability.telemetry_db.init_telemetry_db`,
    and yields the open :class:`sqlite3.Connection`.

    The fixture does NOT close the connection — each test is responsible for
    any extra cleanup it needs; the connection is automatically GC'd when the
    test ends.

    Yields:
        An open :class:`sqlite3.Connection` pointing at the fresh telemetry DB.
    """
    db_path = tmp_path / "telemetry.db"
    monkeypatch.setenv("TELEMETRY_DB_PATH", str(db_path))

    import observability.telemetry_db as tdb

    importlib.reload(tdb)
    conn = tdb.init_telemetry_db(path=db_path)
    yield conn
    conn.close()

