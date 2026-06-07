"""Tests for the ``rag bench`` CLI command (Increment 13).

These tests verify all five acceptance criteria from the TDD spec:

1. Bench file is loaded and each query is run against each backend.
2. p50 and p95 latency statistics are present in the per-backend summary.
3. Telemetry is written for every (backend, query) run.
4. A backend whose ``is_available()`` returns ``False`` is skipped with a
   logged warning; other backends continue normally.
5. The final summary is machine-readable JSON (suitable for CI diffing).

All tests use :class:`~rag.backends.in_memory.InMemoryBackend` as the backend
test-double so they run without any external services.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _write_bench_file(tmp_path: Path, queries: list[dict[str, Any]]) -> Path:
    """Write a bench JSON file to *tmp_path* and return the file path."""
    data = {"queries": queries}
    p = tmp_path / "bench.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _three_queries() -> list[dict[str, Any]]:
    """Return a list of three minimal bench query dicts."""
    return [
        {"query": "what does foo return?", "base_uri": "/tmp/proj"},
        {"query": "how does bar work?", "base_uri": "/tmp/proj"},
        {"query": "explain baz", "base_uri": "/tmp/proj"},
    ]


def _two_queries() -> list[dict[str, Any]]:
    """Return a list of two minimal bench query dicts."""
    return [
        {"query": "what does foo return?", "base_uri": "/tmp/proj"},
        {"query": "how does bar work?", "base_uri": "/tmp/proj"},
    ]


# ---------------------------------------------------------------------------
# Test 1 — loads query file and runs each query against each backend
# ---------------------------------------------------------------------------


def test_bench_loads_query_file_and_runs_each_query_against_each_backend(
    tmp_path: Path,
) -> None:
    """Every query must be executed against every available backend.

    With 3 queries and 2 backends the summary must show 2 backend entries,
    each reporting ``run_count == 3``.
    """
    from cli.bench import BenchRunner, load_bench_file
    from rag.backends.in_memory import InMemoryBackend

    bench_path = _write_bench_file(tmp_path, _three_queries())
    bench = load_bench_file(bench_path)

    backend_a = InMemoryBackend()
    backend_b = InMemoryBackend()

    # Give the two instances distinct logical names so they appear as separate
    # keys in the summary dict.
    from rag.backends.base import BackendName

    backend_a.name = BackendName.IN_MEMORY
    backend_b.name = "backend_b"  # type: ignore[assignment]

    runner = BenchRunner(backends=[backend_a, backend_b])
    summary = runner.run(bench)

    backend_entries = summary["backends"]
    assert len(backend_entries) == 2, (
        f"expected 2 backend entries in summary, got {list(backend_entries.keys())}"
    )

    for name, stats in backend_entries.items():
        assert stats["run_count"] == 3, (
            f"backend {name!r}: expected run_count=3, got {stats['run_count']}"
        )


# ---------------------------------------------------------------------------
# Test 2 — outputs p50/p95 latency per backend
# ---------------------------------------------------------------------------


def test_bench_outputs_p50_p95_latency_per_backend(tmp_path: Path) -> None:
    """The summary must include p50_ms and p95_ms float fields for each backend.

    Both values must be non-negative numbers; p95 must be >= p50.
    """
    from cli.bench import BenchRunner, load_bench_file
    from rag.backends.in_memory import InMemoryBackend

    bench_path = _write_bench_file(tmp_path, _three_queries())
    bench = load_bench_file(bench_path)

    runner = BenchRunner(backends=[InMemoryBackend()])
    summary = runner.run(bench)

    for name, stats in summary["backends"].items():
        assert "p50_ms" in stats, f"missing p50_ms for backend {name!r}"
        assert "p95_ms" in stats, f"missing p95_ms for backend {name!r}"
        assert isinstance(stats["p50_ms"], float), (
            f"p50_ms for {name!r} must be a float, got {type(stats['p50_ms'])}"
        )
        assert isinstance(stats["p95_ms"], float), (
            f"p95_ms for {name!r} must be a float, got {type(stats['p95_ms'])}"
        )
        assert stats["p50_ms"] >= 0.0, f"p50_ms for {name!r} must be >= 0"
        assert stats["p95_ms"] >= 0.0, f"p95_ms for {name!r} must be >= 0"
        assert stats["p95_ms"] >= stats["p50_ms"], (
            f"p95_ms must be >= p50_ms for backend {name!r}"
        )


# ---------------------------------------------------------------------------
# Test 3 — writes telemetry for every run
# ---------------------------------------------------------------------------


def test_bench_writes_telemetry_for_every_run(
    tmp_path: Path,
    telemetry_db: sqlite3.Connection,
) -> None:
    """One ``backend_search_runs`` row must be written per (backend, query) pair.

    With 2 queries and 1 backend, the table must have exactly 2 rows after the
    bench run.
    """
    from cli.bench import BenchRunner, load_bench_file
    from observability.telemetry_db import TelemetrySink
    from rag.backends.in_memory import InMemoryBackend

    bench_path = _write_bench_file(tmp_path, _two_queries())
    bench = load_bench_file(bench_path)

    sink = TelemetrySink(telemetry_db)
    runner = BenchRunner(backends=[InMemoryBackend()], sink=sink)
    runner.run(bench)

    rows = telemetry_db.execute(
        "SELECT COUNT(*) FROM backend_search_runs"
    ).fetchone()[0]
    assert rows == 2, (
        f"expected 2 telemetry rows (1 backend × 2 queries), got {rows}"
    )


# ---------------------------------------------------------------------------
# Test 4 — does not crash when one backend is unavailable
# ---------------------------------------------------------------------------


def test_bench_does_not_crash_when_one_backend_is_unavailable(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Removing MILVUS_URL (or any unavailable backend) must not crash the bench.

    Spec: missing ``MILVUS_URL`` → backend skipped with logged warning, others
    continue.

    The test simulates this by injecting a backend stub whose
    ``is_available()`` returns ``False``.  The bench must:
    - complete without raising,
    - exclude the unavailable backend from the summary,
    - log at least one WARNING that mentions the unavailable backend.
    """
    import logging

    from cli.bench import BenchRunner, load_bench_file
    from rag.backends.in_memory import InMemoryBackend

    class _UnavailableBackend:
        """Stub backend that is always unavailable."""

        name = "milvus"  # simulate a Milvus backend with no MILVUS_URL

        def is_available(self) -> bool:
            return False

        def search(self, request: Any) -> list:  # noqa: ANN401
            raise RuntimeError("should never be called")

    bench_path = _write_bench_file(tmp_path, _two_queries())
    bench = load_bench_file(bench_path)

    good_backend = InMemoryBackend()
    bad_backend = _UnavailableBackend()

    runner = BenchRunner(backends=[good_backend, bad_backend])

    with caplog.at_level(logging.WARNING, logger="cli.bench"):
        summary = runner.run(bench)

    # The unavailable backend must be absent from the summary.
    assert "milvus" not in summary["backends"], (
        "unavailable backend 'milvus' should be absent from summary"
    )

    # The available backend must still appear.
    assert len(summary["backends"]) == 1, (
        "the available backend must still be present in the summary"
    )

    # At least one warning must mention the unavailable backend.
    warning_texts = [
        r.message for r in caplog.records if r.levelno >= logging.WARNING
    ]
    assert any("milvus" in str(w).lower() for w in warning_texts), (
        f"expected a WARNING mentioning 'milvus', got: {warning_texts}"
    )


# ---------------------------------------------------------------------------
# Test 5 — emits machine-readable JSON summary
# ---------------------------------------------------------------------------


def test_bench_emits_machine_readable_json_summary(tmp_path: Path) -> None:
    """``BenchRunner.run_to_json()`` must return valid, parseable JSON.

    The parsed summary must contain:
    - ``"queries_count"`` (int)
    - ``"backends"`` (dict)

    Each backend entry must contain ``"p50_ms"``, ``"p95_ms"``,
    ``"run_count"``, ``"min_ms"``, and ``"max_ms"``.
    """
    from cli.bench import BenchRunner, load_bench_file
    from rag.backends.in_memory import InMemoryBackend

    bench_path = _write_bench_file(tmp_path, _three_queries())
    bench = load_bench_file(bench_path)

    runner = BenchRunner(backends=[InMemoryBackend()])
    json_output = runner.run_to_json(bench)

    # Must be valid JSON.
    try:
        parsed = json.loads(json_output)
    except json.JSONDecodeError as exc:
        pytest.fail(f"run_to_json() returned invalid JSON: {exc}\n{json_output}")

    # Top-level keys.
    assert "queries_count" in parsed, "missing 'queries_count' key in JSON summary"
    assert "backends" in parsed, "missing 'backends' key in JSON summary"
    assert isinstance(parsed["queries_count"], int), "'queries_count' must be int"
    assert isinstance(parsed["backends"], dict), "'backends' must be a dict"
    assert parsed["queries_count"] == 3

    # Per-backend keys.
    required_keys = {"run_count", "p50_ms", "p95_ms", "min_ms", "max_ms"}
    for name, stats in parsed["backends"].items():
        missing = required_keys - set(stats.keys())
        assert not missing, (
            f"backend {name!r} summary is missing keys: {missing}"
        )

