"""Tests for choose_backend_v1() in src/rag/router.py.

TDD Increment 5 — deterministic router with no I/O, no state.
"""

from __future__ import annotations

import time

import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _req(**kwargs):
    """Build a RouteRequest with sensible defaults for router tests."""
    from rag.router import RouteRequest

    return RouteRequest(**kwargs)


def _stats(**kwargs):
    """Build a SystemStats with sensible defaults for router tests."""
    from rag.router import SystemStats

    return SystemStats(**kwargs)


# ---------------------------------------------------------------------------
# Test 1: manual override returns the requested mode
# ---------------------------------------------------------------------------


def test_manual_override_returns_requested_mode() -> None:
    """When requested_backend='qdrant', the decision must be QDRANT regardless."""
    from rag.router import BackendName, choose_backend_v1

    decision = choose_backend_v1(_req(requested_backend="qdrant"))
    assert decision.primary == BackendName.QDRANT
    assert "manual" in decision.reason.lower() or "override" in decision.reason.lower()


def test_manual_override_milvus() -> None:
    """requested_backend='milvus' → primary=MILVUS."""
    from rag.router import BackendName, choose_backend_v1

    decision = choose_backend_v1(_req(requested_backend="milvus"))
    assert decision.primary == BackendName.MILVUS


def test_manual_override_exact() -> None:
    """requested_backend='exact' → primary=EXACT."""
    from rag.router import BackendName, choose_backend_v1

    decision = choose_backend_v1(_req(requested_backend="exact"))
    assert decision.primary == BackendName.EXACT


# ---------------------------------------------------------------------------
# Test 2: exact-ish query routes to EXACT
# ---------------------------------------------------------------------------


def test_exactish_query_routes_to_exact() -> None:
    """A query that looks like a code symbol should route to EXACT."""
    from rag.router import BackendName, choose_backend_v1

    decision = choose_backend_v1(_req(query="MyClassName"))
    assert decision.primary == BackendName.EXACT


def test_file_path_query_routes_to_exact() -> None:
    """A query with a file path should route to EXACT."""
    from rag.router import BackendName, choose_backend_v1

    decision = choose_backend_v1(_req(query="foo/bar/baz.ts"))
    assert decision.primary == BackendName.EXACT


def test_error_message_routes_to_exact() -> None:
    """A query with an error string should route to EXACT."""
    from rag.router import BackendName, choose_backend_v1

    decision = choose_backend_v1(_req(query="Error: invalid provider"))
    assert decision.primary == BackendName.EXACT


# ---------------------------------------------------------------------------
# Test 3: filter-heavy + small batch → QDRANT
# ---------------------------------------------------------------------------


def test_filter_heavy_low_batch_routes_to_qdrant() -> None:
    """3 filters + batch=1 → Qdrant for efficient scalar filtering."""
    from rag.router import BackendName, choose_backend_v1

    decision = choose_backend_v1(_req(filter_count=3, batch_size=1))
    assert decision.primary == BackendName.QDRANT
    assert "filter" in decision.reason.lower()


# ---------------------------------------------------------------------------
# Test 4: large batch + GPU + hot collection → MILVUS
# ---------------------------------------------------------------------------


def test_large_batch_with_gpu_and_hot_collection_routes_to_milvus() -> None:
    """batch≥10, GPU available, hot collection → Milvus GPU-CAGRA."""
    from rag.router import BackendName, choose_backend_v1

    stats = _stats(
        gpu_vram_free_mb=8192,
        gpu_util_pct=20,
        milvus_hot_collections=frozenset({"my_project"}),
    )
    decision = choose_backend_v1(
        _req(batch_size=20, collection="my_project"), sys=stats
    )
    assert decision.primary == BackendName.MILVUS
    assert "milvus" in decision.reason.lower()


# ---------------------------------------------------------------------------
# Test 5: large batch without GPU → QDRANT
# ---------------------------------------------------------------------------


def test_large_batch_without_gpu_routes_to_qdrant() -> None:
    """Large batch but no GPU (vram_free=0) → fall back to Qdrant."""
    from rag.router import BackendName, choose_backend_v1

    stats = _stats(gpu_vram_free_mb=0, gpu_util_pct=0)
    decision = choose_backend_v1(_req(batch_size=20, collection="my_project"), sys=stats)
    assert decision.primary == BackendName.QDRANT


# ---------------------------------------------------------------------------
# Test 6: large batch + hot collection BUT gpu_util high → QDRANT
# ---------------------------------------------------------------------------


def test_large_batch_with_hot_collection_but_gpu_busy_routes_to_qdrant() -> None:
    """GPU utilisation ≥ 80 % → treat GPU as busy → Qdrant, not Milvus."""
    from rag.router import BackendName, choose_backend_v1

    stats = _stats(
        gpu_vram_free_mb=8192,
        gpu_util_pct=80,
        milvus_hot_collections=frozenset({"my_project"}),
    )
    decision = choose_backend_v1(
        _req(batch_size=20, collection="my_project"), sys=stats
    )
    assert decision.primary == BackendName.QDRANT


# ---------------------------------------------------------------------------
# Test 7: default falls through to QDRANT
# ---------------------------------------------------------------------------


def test_default_falls_through_to_qdrant() -> None:
    """A plain natural-language query with no special signals → Qdrant."""
    from rag.router import BackendName, choose_backend_v1

    decision = choose_backend_v1(_req(query="how does the memory system work?"))
    assert decision.primary == BackendName.QDRANT


# ---------------------------------------------------------------------------
# Test 8: every decision has a non-empty reason string
# ---------------------------------------------------------------------------


_REASON_TEST_CASES = [
    # (description, req_kwargs, stats_kwargs)
    ("manual", {"requested_backend": "qdrant"}, {}),
    ("exact query", {"query": "MyClass"}, {}),
    ("filter heavy", {"filter_count": 3, "batch_size": 1}, {}),
    (
        "milvus",
        {"batch_size": 20, "collection": "proj"},
        {
            "gpu_vram_free_mb": 8192,
            "gpu_util_pct": 10,
            "milvus_hot_collections": frozenset({"proj"}),
        },
    ),
    ("default", {"query": "explain the architecture"}, {}),
]


@pytest.mark.parametrize("desc,req_kw,stats_kw", _REASON_TEST_CASES)
def test_router_records_reason_string(desc, req_kw, stats_kw) -> None:
    """Every RouteDecision must have a non-empty reason."""
    from rag.router import choose_backend_v1

    decision = choose_backend_v1(_req(**req_kw), sys=_stats(**stats_kw))
    assert decision.reason, f"[{desc}] reason was empty"
    assert len(decision.reason) > 5, f"[{desc}] reason too short: {decision.reason!r}"


# ---------------------------------------------------------------------------
# Test 9: router is pure — no side effects across many calls
# ---------------------------------------------------------------------------


def test_router_is_pure_no_side_effects() -> None:
    """Calling choose_backend_v1 1 000 times with identical inputs yields
    identical outputs and has no observable side effects (no DB, no network)."""
    from rag.router import choose_backend_v1

    req = _req(query="MyFunctionName", batch_size=1)
    stats = _stats(gpu_vram_free_mb=4096, gpu_util_pct=5)

    results = [choose_backend_v1(req, stats) for _ in range(1000)]
    primaries = {d.primary for d in results}
    reasons = {d.reason for d in results}

    # All calls must return the exact same decision.
    assert len(primaries) == 1, f"Non-deterministic primaries: {primaries}"
    assert len(reasons) == 1, f"Non-deterministic reasons: {reasons}"


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


def test_unknown_backend_falls_back_gracefully() -> None:
    """An unknown requested_backend value should not raise."""
    from rag.router import BackendName, choose_backend_v1

    decision = choose_backend_v1(_req(requested_backend="nonexistent_db"))
    # Should fall back to something reasonable without raising
    assert decision.primary in (BackendName.QDRANT,)
    assert decision.reason  # reason must explain the fallback


def test_mode_auto_is_default() -> None:
    """When search_mode is not set, mode_used should be AUTO."""
    from rag.router import SearchMode, choose_backend_v1

    decision = choose_backend_v1(_req())
    assert decision.mode_used == SearchMode.AUTO


def test_unknown_mode_coerced_to_auto() -> None:
    """An unrecognised search_mode value must be coerced to AUTO without raising."""
    from rag.router import SearchMode, choose_backend_v1

    decision = choose_backend_v1(_req(search_mode="some_future_mode"))
    assert decision.mode_used == SearchMode.AUTO


def test_none_sys_stats_does_not_raise() -> None:
    """Passing sys=None should be safe (treated as all-zero stats)."""
    from rag.router import choose_backend_v1

    decision = choose_backend_v1(_req(query="explain the architecture"), sys=None)
    assert decision.primary  # some decision made without raising


def test_shadow_field_is_none_by_default() -> None:
    """Shadow backend should be None unless the router explicitly sets it."""
    from rag.router import choose_backend_v1

    decision = choose_backend_v1(_req())
    assert decision.shadow is None


def test_router_timing_is_fast() -> None:
    """choose_backend_v1 should complete in well under 1 ms per call."""
    from rag.router import choose_backend_v1

    req = _req(query="how does memory work?")
    stats = _stats()

    start = time.perf_counter()
    for _ in range(1000):
        choose_backend_v1(req, stats)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # 1 000 calls in < 100 ms ≈ < 0.1 ms each
    assert elapsed_ms < 100, (
        f"Router too slow: {elapsed_ms:.1f} ms for 1 000 calls"
    )

