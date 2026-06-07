"""Tests for the MilvusBackend (Increment 11).

All tests that require a running Milvus instance are marked
``@pytest.mark.integration`` and skipped automatically unless the
``MILVUS_URL`` environment variable is set.

Unit tests (M2, M3) that only test index parameter selection are NOT
skipped — they mock the Milvus connection.

Contract coverage (via the standard RagBackend contract suite):
    1–10. Full contract suite — same as test_protocol_contract.py

Milvus-specific tests:
    M1. hot/cold collection naming convention
    M2. GPU_CAGRA index used when gpu_available() is True  [unit test]
    M3. CPU HNSW index used when no GPU available          [unit test]
    M4. batch search top_k=200 within latency budget (soft assertion)
    M5. error fallback — MilvusBackend raising triggers caller fallback
"""

from __future__ import annotations

import hashlib
import os
import time
import warnings
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rag.backends.base import (
    BackendName,
    BackendStats,
    CollectionSpec,
    EmbeddedChunk,
    MetadataFilter,
    SearchRequest,
    SearchResult,
)

# ---------------------------------------------------------------------------
# Skip guard (applies only to integration tests)
# ---------------------------------------------------------------------------

MILVUS_URL = os.environ.get("MILVUS_URL")
_integration = pytest.mark.skipif(
    not MILVUS_URL,
    reason="MILVUS_URL not set — skipping Milvus integration tests",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 8  # small dimension to keep tests fast


def _vec(i: int, dim: int = DIM) -> list[float]:
    """Deterministic unit vector for test chunk *i*."""
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _make_chunk(
    chunk_id: str,
    content: str = "def foo(): return 1",
    document_id: str = "doc1",
    embedding: list[float] | None = None,
    metadata: dict[str, Any] | None = None,
    idx: int = 0,
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        content_hash=_hash(content),
        embedding=embedding if embedding is not None else _vec(idx),
        metadata=metadata or {"project": "proj_a", "path": f"/src/{chunk_id}.py"},
        token_count=len(content.split()),
    )


def _spec(name: str = "test_col", dim: int = DIM) -> CollectionSpec:
    return CollectionSpec(name=name, dimension=dim)


# ---------------------------------------------------------------------------
# Integration fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def milvus_backend():
    """Return a MilvusBackend connected to the test Milvus instance.

    Requires ``MILVUS_URL`` to be set.  Skipped otherwise.
    """
    if not MILVUS_URL:
        pytest.skip("MILVUS_URL not set")

    from rag.backends.milvus import MilvusBackend

    backend = MilvusBackend(uri=MILVUS_URL, dimension=DIM)
    yield backend
    # Teardown: drop test collections to keep the server clean.
    try:
        from pymilvus import connections, utility

        connections.connect(alias="default", uri=MILVUS_URL)
        for col_name in [
            "test_col",
            "col_a",
            "col_b",
            "hot_project_gpu",
            "hot_project_cpu",
            "cold_memory_cpu",
            "latency_test_col",
            "fallback_col",
        ]:
            if utility.has_collection(col_name):
                utility.drop_collection(col_name)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Contract test 1 — create_collection is idempotent
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_create_collection_is_idempotent(milvus_backend):
    spec = _spec()
    milvus_backend.create_collection(spec)
    milvus_backend.create_collection(spec)  # must not raise


# ---------------------------------------------------------------------------
# Contract test 2 — upsert → immediately searchable
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_upsert_returns_after_chunks_are_searchable(milvus_backend):
    spec = _spec()
    milvus_backend.create_collection(spec)

    chunk = _make_chunk("c1", content="def foo(): pass", embedding=_vec(0), idx=0)
    milvus_backend.upsert([chunk], collection=spec.name)

    req = SearchRequest(
        query="foo",
        collection=spec.name,
        top_k=5,
        embedding=_vec(0),
    )
    results = milvus_backend.search(req)
    assert any(r.chunk_id == "c1" for r in results)


# ---------------------------------------------------------------------------
# Contract test 3 — results sorted by score descending
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_search_returns_results_sorted_by_score_desc(milvus_backend):
    spec = _spec()
    milvus_backend.create_collection(spec)

    chunks = [_make_chunk(f"c{i}", embedding=_vec(i), idx=i) for i in range(DIM)]
    milvus_backend.upsert(chunks, collection=spec.name)

    req = SearchRequest(
        query="test",
        collection=spec.name,
        top_k=DIM,
        embedding=_vec(0),
    )
    results = milvus_backend.search(req)
    assert len(results) >= 1
    for a, b in zip(results, results[1:]):
        assert a.score >= b.score - 1e-6
    assert results[0].chunk_id == "c0"


# ---------------------------------------------------------------------------
# Contract test 4 — search respects top_k
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_search_respects_top_k(milvus_backend):
    spec = _spec()
    milvus_backend.create_collection(spec)

    chunks = [_make_chunk(f"c{i}", embedding=_vec(i), idx=i) for i in range(DIM)]
    milvus_backend.upsert(chunks, collection=spec.name)

    for k in (1, 2, 3):
        req = SearchRequest(
            query="test",
            collection=spec.name,
            top_k=k,
            embedding=_vec(0),
        )
        results = milvus_backend.search(req)
        assert len(results) <= k


# ---------------------------------------------------------------------------
# Contract test 5 — metadata filter: project
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_search_respects_metadata_filter_project(milvus_backend):
    spec = _spec()
    milvus_backend.create_collection(spec)

    milvus_backend.upsert(
        [
            _make_chunk(
                "pa1",
                embedding=_vec(0),
                metadata={"project": "proj_a", "path": "/a.py"},
                idx=0,
            ),
            _make_chunk(
                "pa2",
                embedding=_vec(1),
                metadata={"project": "proj_a", "path": "/b.py"},
                idx=1,
            ),
            _make_chunk(
                "pb1",
                embedding=_vec(2),
                metadata={"project": "proj_b", "path": "/c.py"},
                idx=2,
            ),
        ],
        collection=spec.name,
    )

    req = SearchRequest(
        query="test",
        collection=spec.name,
        top_k=10,
        embedding=_vec(0),
        filters=[MetadataFilter(field="project", op="eq", value="proj_a")],
    )
    results = milvus_backend.search(req)
    assert len(results) == 2
    for r in results:
        assert r.metadata.get("project") == "proj_a"


# ---------------------------------------------------------------------------
# Contract test 6 — metadata filter: path_prefix
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_search_respects_metadata_filter_path_prefix(milvus_backend):
    spec = _spec()
    milvus_backend.create_collection(spec)

    milvus_backend.upsert(
        [
            _make_chunk(
                "pp1",
                embedding=_vec(0),
                metadata={"project": "p", "path": "/src/rag/foo.py"},
                idx=0,
            ),
            _make_chunk(
                "pp2",
                embedding=_vec(1),
                metadata={"project": "p", "path": "/src/rag/bar.py"},
                idx=1,
            ),
            _make_chunk(
                "pp3",
                embedding=_vec(2),
                metadata={"project": "p", "path": "/src/api/routes.py"},
                idx=2,
            ),
        ],
        collection=spec.name,
    )

    req = SearchRequest(
        query="test",
        collection=spec.name,
        top_k=10,
        embedding=_vec(0),
        filters=[MetadataFilter(field="path", op="prefix", value="/src/rag/")],
    )
    results = milvus_backend.search(req)
    assert len(results) == 2
    for r in results:
        assert r.metadata.get("path", "").startswith("/src/rag/")


# ---------------------------------------------------------------------------
# Contract test 7 — missing collection → empty list
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_search_returns_empty_list_when_collection_missing(milvus_backend):
    req = SearchRequest(
        query="anything",
        collection="nonexistent_collection_milvus_xyz",
        top_k=5,
        embedding=_vec(0),
    )
    results = milvus_backend.search(req)
    assert results == []


# ---------------------------------------------------------------------------
# Contract test 8 — delete_by_filter removes only matching chunks
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_delete_by_filter_removes_only_matching_chunks(milvus_backend):
    spec = _spec()
    milvus_backend.create_collection(spec)

    milvus_backend.upsert(
        [
            _make_chunk(
                "del1",
                document_id="doc_x",
                embedding=_vec(0),
                metadata={"project": "p", "path": "/x.py", "document_id": "doc_x"},
                idx=0,
            ),
            _make_chunk(
                "del2",
                document_id="doc_y",
                embedding=_vec(1),
                metadata={"project": "p", "path": "/y.py", "document_id": "doc_y"},
                idx=1,
            ),
        ],
        collection=spec.name,
    )

    deleted = milvus_backend.delete_by_filter(
        collection=spec.name,
        filters=[MetadataFilter(field="document_id", op="eq", value="doc_x")],
    )
    assert deleted >= 1

    req = SearchRequest(
        query="test",
        collection=spec.name,
        top_k=10,
        embedding=_vec(1),
    )
    results = milvus_backend.search(req)
    ids = [r.chunk_id for r in results]
    assert "del1" not in ids
    assert "del2" in ids


# ---------------------------------------------------------------------------
# Contract test 9 — stats
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_stats_reports_collection_count_and_vector_count(milvus_backend):
    spec1 = _spec("col_a")
    spec2 = _spec("col_b")
    milvus_backend.create_collection(spec1)
    milvus_backend.create_collection(spec2)

    milvus_backend.upsert(
        [_make_chunk("s1", embedding=_vec(0), idx=0)], collection="col_a"
    )
    milvus_backend.upsert(
        [_make_chunk("s2", embedding=_vec(1), idx=1)], collection="col_b"
    )

    stats: BackendStats = milvus_backend.stats()
    assert isinstance(stats, BackendStats)
    assert stats.collection_count >= 2
    if stats.vector_count is not None:
        assert stats.vector_count >= 2


# ---------------------------------------------------------------------------
# Contract test 10 — search result carries backend_name field
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_search_result_carries_backend_name_field(milvus_backend):
    spec = _spec()
    milvus_backend.create_collection(spec)
    milvus_backend.upsert(
        [_make_chunk("bn1", embedding=_vec(0), idx=0)], collection=spec.name
    )

    req = SearchRequest(
        query="test",
        collection=spec.name,
        top_k=1,
        embedding=_vec(0),
    )
    results = milvus_backend.search(req)
    assert len(results) == 1
    assert isinstance(results[0].backend, BackendName)
    assert results[0].backend == BackendName.MILVUS


# ---------------------------------------------------------------------------
# Milvus-specific M1 — hot/cold collection naming convention
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_milvus_collections_hot_cold_naming_convention(milvus_backend):
    """Collections following the hot_{project}_{tier} / cold_{project}_{tier}
    naming convention must be creatable without errors."""
    for name in ("hot_project_gpu", "hot_project_cpu", "cold_memory_cpu"):
        spec = CollectionSpec(name=name, dimension=DIM)
        milvus_backend.create_collection(spec)
        # Also verify they appear in stats
    stats = milvus_backend.stats()
    assert stats.collection_count >= 3


# ---------------------------------------------------------------------------
# Milvus-specific M2 — GPU_CAGRA index when gpu_available() is True [unit]
# ---------------------------------------------------------------------------


def test_milvus_gpu_cagra_index_used_when_gpu_available():
    """When gpu_available() returns True the backend must choose GPU_CAGRA.

    This is a unit test — it patches both the Milvus connection layer and
    gpu_available() so no running Milvus server is required.
    """
    from rag.backends.milvus import MilvusBackend

    # Patch _PYMILVUS_AVAILABLE so __init__ does not raise ImportError, and
    # provide a mock connections object so _connect() does not actually connect.
    with patch("rag.backends.milvus._PYMILVUS_AVAILABLE", True), \
         patch("rag.backends.milvus.connections", MagicMock(), create=True), \
         patch("rag.backends.milvus.gpu_available", return_value=True):
        backend = MilvusBackend(uri="http://localhost:19530", dimension=DIM)
        index_params = backend._build_index_params()

    assert index_params["index_type"] == "GPU_CAGRA", (
        f"Expected GPU_CAGRA index when GPU is available; got {index_params['index_type']}"
    )


# ---------------------------------------------------------------------------
# Milvus-specific M3 — HNSW index when no GPU [unit]
# ---------------------------------------------------------------------------


def test_milvus_cpu_hnsw_index_used_when_no_gpu():
    """When gpu_available() returns False the backend must choose HNSW.

    This is a unit test — it patches both the Milvus connection layer and
    gpu_available() so no running Milvus server is required.
    """
    from rag.backends.milvus import MilvusBackend

    with patch("rag.backends.milvus._PYMILVUS_AVAILABLE", True), \
         patch("rag.backends.milvus.connections", MagicMock(), create=True), \
         patch("rag.backends.milvus.gpu_available", return_value=False):
        backend = MilvusBackend(uri="http://localhost:19530", dimension=DIM)
        index_params = backend._build_index_params()

    assert index_params["index_type"] == "HNSW", (
        f"Expected HNSW index when GPU is unavailable; got {index_params['index_type']}"
    )


# ---------------------------------------------------------------------------
# Milvus-specific M4 — batch search top_k=200 latency budget (soft assert)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_milvus_batch_search_topk_200_returns_within_latency_budget(milvus_backend):
    """Batch search with top_k=200 should complete within 5 s (soft assertion).

    This is a SOFT assertion: we emit a warning rather than failing so that
    a slow CI runner does not cause spurious failures.  Actual latency is
    recorded in the warning message for later analysis.
    """
    spec = CollectionSpec(name="latency_test_col", dimension=DIM)
    milvus_backend.create_collection(spec)

    # Populate with 200 chunks
    chunks = [
        _make_chunk(f"lt{i}", embedding=_vec(i % DIM), idx=i % DIM)
        for i in range(200)
    ]
    milvus_backend.upsert(chunks, collection=spec.name)

    req = SearchRequest(
        query="latency test",
        collection=spec.name,
        top_k=200,
        embedding=_vec(0),
    )

    t0 = time.perf_counter()
    results = milvus_backend.search(req)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if elapsed_ms > 5000:
        warnings.warn(
            f"Milvus batch search top_k=200 took {elapsed_ms:.1f} ms (budget: 5000 ms)",
            stacklevel=2,
        )

    # Regardless of latency, we must get some results back
    assert len(results) > 0, "Batch search returned no results"


# ---------------------------------------------------------------------------
# Milvus-specific M5 — error fallback
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_milvus_fallback_to_qdrant_on_error(milvus_backend):
    """When MilvusBackend.search raises, the caller should be able to fall back.

    This test verifies that MilvusBackend propagates exceptions rather than
    swallowing them, so that the pipeline layer (Inc 8) can detect the failure
    and route to a fallback backend.
    """
    from rag.backends.milvus import MilvusBackend

    backend = MilvusBackend(uri=MILVUS_URL, dimension=DIM)

    # Simulate a network/server error by patching the internal search method.
    with patch.object(
        backend,
        "_do_search",
        side_effect=RuntimeError("Milvus connection refused"),
    ):
        with pytest.raises(RuntimeError, match="Milvus connection refused"):
            req = SearchRequest(
                query="test",
                collection="fallback_col",
                top_k=5,
                embedding=_vec(0),
            )
            backend.search(req)

