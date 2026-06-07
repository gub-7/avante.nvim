"""TDD tests for ShadowExecutor (Increment 9).

Tests verify:
1. Both primary and shadow backends are invoked in parallel; wall time < sum
   of individual latencies (concurrency proof).
2. Only primary results are returned to the caller.
3. Shadow run is recorded with is_shadow=True in telemetry.
4. A shadow failure does not affect the primary response.
5. overlap_at_k is computed and stored on the retrieval_requests row.
6. Shadow is disabled by default (shadow_backend=None → no shadow invocation).
7. Shadow can be enabled via request field AND via an explicit flag.
8. Global kill switch: RAG_SHADOW_DISABLED=1 prevents shadow execution.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import time
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _make_results(backend_name: str, chunk_ids: list[str]) -> list[dict]:
    """Build a list of fake search-result dicts for *backend_name*."""
    return [
        {
            "chunk_id": cid,
            "score": 1.0 - i * 0.05,
            "backend_name": backend_name,
            "rank": i,
        }
        for i, cid in enumerate(chunk_ids)
    ]


def _make_req(
    shadow: bool = False,
    shadow_backend: str | None = None,
) -> MagicMock:
    """Build a minimal request object."""
    req = MagicMock()
    req.query = "test query"
    req.mode = "ask"
    req.base_uri = "file:///proj"
    req.request_id = uuid.uuid4().hex
    req.shadow = shadow
    req.shadow_backend = shadow_backend
    return req


class SlowBackend:
    """Simulates a backend that takes a configurable amount of time."""

    def __init__(self, name: str, latency_s: float, results: list[dict]) -> None:
        self.name = name
        self._latency_s = latency_s
        self._results = results
        self.called = False

    async def search(self, req: Any) -> list[dict]:
        self.called = True
        await asyncio.sleep(self._latency_s)
        return list(self._results)


class FailingBackend:
    """Always raises on search()."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.called = False

    async def search(self, req: Any) -> list[dict]:
        self.called = True
        raise RuntimeError("Backend unavailable")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shadow_runs_in_parallel_with_primary(telemetry_db):
    """Both backends must be invoked; total wall time < sum of latencies."""
    from rag.shadow import ShadowExecutor

    primary_latency = 0.08
    shadow_latency = 0.08

    primary_chunks = [f"p_{i}" for i in range(10)]
    shadow_chunks = [f"s_{i}" for i in range(10)]

    primary_backend = SlowBackend("qdrant", primary_latency, _make_results("qdrant", primary_chunks))
    shadow_backend = SlowBackend("milvus", shadow_latency, _make_results("milvus", shadow_chunks))

    req = _make_req(shadow=True)
    executor = ShadowExecutor(sink=None)

    t0 = time.perf_counter()
    primary_results, shadow_record = await executor.run(
        primary_backend=primary_backend,
        shadow_backend=shadow_backend,
        req=req,
    )
    elapsed = time.perf_counter() - t0

    assert primary_backend.called
    assert shadow_backend.called
    # Parallel: wall time must be less than the sum of the two latencies
    assert elapsed < (primary_latency + shadow_latency) * 0.95, (
        f"Expected parallel execution but elapsed={elapsed:.3f}s >= sum={primary_latency + shadow_latency:.3f}s"
    )


@pytest.mark.asyncio
async def test_only_primary_results_returned_to_caller(telemetry_db):
    """ShadowExecutor.run() must return primary backend results, not shadow's."""
    from rag.shadow import ShadowExecutor

    primary_chunks = ["pri_a", "pri_b", "pri_c"]
    shadow_chunks = ["sha_x", "sha_y", "sha_z"]

    primary_backend = SlowBackend("qdrant", 0.001, _make_results("qdrant", primary_chunks))
    shadow_backend = SlowBackend("milvus", 0.001, _make_results("milvus", shadow_chunks))

    req = _make_req(shadow=True)
    executor = ShadowExecutor(sink=None)

    primary_results, _ = await executor.run(
        primary_backend=primary_backend,
        shadow_backend=shadow_backend,
        req=req,
    )

    returned_ids = {r["chunk_id"] for r in primary_results}
    assert returned_ids == set(primary_chunks), (
        "Shadow results must not be returned to the caller"
    )
    assert not returned_ids.intersection(set(shadow_chunks))


@pytest.mark.asyncio
async def test_shadow_is_recorded_with_is_shadow_true(telemetry_db):
    """Shadow backend run must be recorded with is_shadow=True in telemetry DB."""
    from observability.telemetry_db import TelemetrySink
    from rag.shadow import ShadowExecutor

    sink = TelemetrySink(telemetry_db)

    primary_chunks = [f"p_{i}" for i in range(5)]
    shadow_chunks = [f"s_{i}" for i in range(5)]

    primary_backend = SlowBackend("qdrant", 0.001, _make_results("qdrant", primary_chunks))
    shadow_backend = SlowBackend("milvus", 0.001, _make_results("milvus", shadow_chunks))

    req = _make_req(shadow=True)
    req.request_id = uuid.uuid4().hex

    executor = ShadowExecutor(sink=sink)
    await executor.run(
        primary_backend=primary_backend,
        shadow_backend=shadow_backend,
        req=req,
    )

    # Check shadow run stored with is_shadow=True, is_primary=False
    telemetry_db.row_factory = sqlite3.Row
    rows = telemetry_db.execute(
        "SELECT * FROM backend_search_runs WHERE request_id = ? AND is_shadow = 1",
        (req.request_id,),
    ).fetchall()

    assert len(rows) >= 1, "No shadow run recorded with is_shadow=True"
    shadow_row = dict(rows[0])
    assert shadow_row["backend_name"] == "milvus"
    assert shadow_row["is_primary"] == 0


@pytest.mark.asyncio
async def test_shadow_failure_does_not_affect_primary_response(telemetry_db):
    """A shadow backend error must not raise and must not corrupt primary results."""
    from rag.shadow import ShadowExecutor

    primary_chunks = ["good_a", "good_b"]
    primary_backend = SlowBackend("qdrant", 0.001, _make_results("qdrant", primary_chunks))
    shadow_backend = FailingBackend("milvus")

    req = _make_req(shadow=True)
    executor = ShadowExecutor(sink=None)

    # Must not raise
    primary_results, shadow_record = await executor.run(
        primary_backend=primary_backend,
        shadow_backend=shadow_backend,
        req=req,
    )

    assert shadow_backend.called
    returned_ids = {r["chunk_id"] for r in primary_results}
    assert returned_ids == set(primary_chunks)
    # Shadow record should carry the error
    assert shadow_record.get("error") is not None


@pytest.mark.asyncio
async def test_shadow_overlap_at_k_is_computed_and_stored(telemetry_db):
    """overlap_at_10/50/100 must be computed and stored on retrieval_requests."""
    from observability.telemetry_db import TelemetrySink
    from rag.shadow import ShadowExecutor

    sink = TelemetrySink(telemetry_db)

    # 5 chunks in common out of 10 → overlap@10 = 0.5
    shared = [f"shared_{i}" for i in range(5)]
    primary_only = [f"pri_{i}" for i in range(5)]
    shadow_only = [f"sha_{i}" for i in range(5)]

    primary_chunks = shared + primary_only
    shadow_chunks = shared + shadow_only

    primary_backend = SlowBackend("qdrant", 0.001, _make_results("qdrant", primary_chunks))
    shadow_backend = SlowBackend("milvus", 0.001, _make_results("milvus", shadow_chunks))

    req = _make_req(shadow=True)
    req.request_id = uuid.uuid4().hex

    executor = ShadowExecutor(sink=sink)
    await executor.run(
        primary_backend=primary_backend,
        shadow_backend=shadow_backend,
        req=req,
    )

    telemetry_db.row_factory = sqlite3.Row
    row = telemetry_db.execute(
        "SELECT overlap_at_10, overlap_at_50, overlap_at_100 "
        "FROM retrieval_requests WHERE request_id = ?",
        (req.request_id,),
    ).fetchone()

    assert row is not None, "No retrieval_requests row written for this request_id"
    assert row["overlap_at_10"] is not None
    # 5 shared out of 10 total = 0.5
    assert abs(row["overlap_at_10"] - 0.5) < 0.01, (
        f"Expected overlap_at_10 ≈ 0.5, got {row['overlap_at_10']}"
    )


@pytest.mark.asyncio
async def test_shadow_disabled_by_default(telemetry_db):
    """When no shadow backend is provided, shadow execution must not occur."""
    from rag.shadow import ShadowExecutor

    primary_chunks = ["a", "b"]
    primary_backend = SlowBackend("qdrant", 0.001, _make_results("qdrant", primary_chunks))

    req = _make_req(shadow=False)
    executor = ShadowExecutor(sink=None)

    # shadow_backend=None → no shadow
    primary_results, shadow_record = await executor.run(
        primary_backend=primary_backend,
        shadow_backend=None,
        req=req,
    )

    returned_ids = {r["chunk_id"] for r in primary_results}
    assert returned_ids == set(primary_chunks)
    assert shadow_record == {}  # empty record when shadow didn't run


@pytest.mark.asyncio
async def test_shadow_enabled_via_request_field_and_via_router_decision(telemetry_db):
    """Shadow runs when req.shadow=True OR when shadow_backend is explicitly passed."""
    from rag.shadow import ShadowExecutor

    primary_chunks = ["p1", "p2"]
    shadow_chunks = ["s1", "s2"]

    # Path 1: req.shadow = True
    primary_backend = SlowBackend("qdrant", 0.001, _make_results("qdrant", primary_chunks))
    shadow_backend = SlowBackend("milvus", 0.001, _make_results("milvus", shadow_chunks))

    req = _make_req(shadow=True)
    executor = ShadowExecutor(sink=None)
    _, record1 = await executor.run(
        primary_backend=primary_backend,
        shadow_backend=shadow_backend,
        req=req,
    )
    assert shadow_backend.called, "Shadow must run when req.shadow=True"

    # Path 2: shadow explicitly provided (req.shadow=False but backend given)
    primary_backend2 = SlowBackend("qdrant", 0.001, _make_results("qdrant", primary_chunks))
    shadow_backend2 = SlowBackend("milvus", 0.001, _make_results("milvus", shadow_chunks))

    req2 = _make_req(shadow=False)
    _, record2 = await executor.run(
        primary_backend=primary_backend2,
        shadow_backend=shadow_backend2,
        req=req2,
        force_shadow=True,
    )
    assert shadow_backend2.called, "Shadow must run when force_shadow=True"


@pytest.mark.asyncio
async def test_shadow_respects_a_global_kill_switch(monkeypatch, telemetry_db):
    """RAG_SHADOW_DISABLED=1 must prevent shadow execution entirely."""
    monkeypatch.setenv("RAG_SHADOW_DISABLED", "1")

    from rag.shadow import ShadowExecutor

    primary_chunks = ["x1", "x2"]
    shadow_chunks = ["y1", "y2"]

    primary_backend = SlowBackend("qdrant", 0.001, _make_results("qdrant", primary_chunks))
    shadow_backend = SlowBackend("milvus", 0.001, _make_results("milvus", shadow_chunks))

    req = _make_req(shadow=True)
    executor = ShadowExecutor(sink=None)

    primary_results, shadow_record = await executor.run(
        primary_backend=primary_backend,
        shadow_backend=shadow_backend,
        req=req,
    )

    assert not shadow_backend.called, (
        "Shadow backend must NOT be called when RAG_SHADOW_DISABLED=1"
    )
    returned_ids = {r["chunk_id"] for r in primary_results}
    assert returned_ids == set(primary_chunks)
    assert shadow_record == {}

