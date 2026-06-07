"""Qdrant backend tests.

All tests in this module are marked ``@pytest.mark.integration`` and are
skipped automatically when the ``QDRANT_URL`` environment variable is not set.
This keeps the CI baseline green while still allowing full coverage to be
exercised against a live Qdrant instance.

Run integration tests with:
    QDRANT_URL=http://localhost:6333 pytest -m integration tests/backends/test_qdrant_backend.py

The test suite consists of two parts:
1. The full 10-test contract suite (parametrized to include ``QdrantBackend``)
   — imported and re-executed from ``test_protocol_contract.py``.
2. Five Qdrant-specific tests verifying features only QdrantBackend exposes.

TDD: these tests are written before the implementation of
``src/rag/backends/qdrant.py``.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

from rag.backends.base import (
    BackendName,
    CollectionSpec,
    EmbeddedChunk,
    MetadataFilter,
    SearchRequest,
)

# ---------------------------------------------------------------------------
# Skip all tests unless QDRANT_URL is set
# ---------------------------------------------------------------------------

QDRANT_URL = os.environ.get("QDRANT_URL", "")

pytestmark = pytest.mark.integration

_skip_reason = (
    "QDRANT_URL environment variable not set — skipping Qdrant integration tests"
)


def _should_skip() -> bool:
    return not QDRANT_URL


# ---------------------------------------------------------------------------
# Helpers (mirror test_protocol_contract.py for isolation)
# ---------------------------------------------------------------------------

DIM = 4


def _vec(i: int, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


def _uid() -> str:
    """Return a unique UUID string usable as a Qdrant point ID."""
    return str(uuid.uuid4())


def _make_chunk(
    chunk_id: str | None = None,
    document_id: str = "doc1",
    content: str = "hello world",
    embedding: list[float] | None = None,
    metadata: dict[str, Any] | None = None,
    idx: int = 0,
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id or _uid(),
        document_id=document_id,
        content=content,
        embedding=embedding if embedding is not None else _vec(idx),
        metadata=metadata or {"project": "proj_a", "path": "/src/main.py"},
        token_count=len(content.split()),
    )


def _spec(name: str | None = None, dim: int = DIM) -> CollectionSpec:
    # Use a unique name so concurrent test runs don't collide.
    return CollectionSpec(name=name or f"test_{uuid.uuid4().hex[:8]}", dimension=dim)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def qdrant_backend():
    """QdrantBackend connected to QDRANT_URL.  Skipped without the env var."""
    if _should_skip():
        pytest.skip(_skip_reason)

    from rag.backends.qdrant import QdrantBackend

    backend = QdrantBackend(name=BackendName.QDRANT, url=QDRANT_URL)
    yield backend
    # Teardown: collections created during tests are left to expire naturally;
    # the unique names ensure no collisions across runs.


# ---------------------------------------------------------------------------
# 1. Full contract suite
# ---------------------------------------------------------------------------


class TestQdrantContract:
    """Re-run the full 10-test contract suite against QdrantBackend."""

    def _backend(self, qdrant_backend):
        return qdrant_backend

    def test_create_collection_is_idempotent(self, qdrant_backend):
        backend = self._backend(qdrant_backend)
        spec = _spec()
        backend.create_collection(spec)
        backend.create_collection(spec)  # must not raise

    def test_upsert_returns_after_chunks_are_searchable(self, qdrant_backend):
        backend = self._backend(qdrant_backend)
        spec = _spec()
        backend.create_collection(spec)

        chunk_id = _uid()
        chunk = _make_chunk(chunk_id, content="def foo(): pass", embedding=_vec(0))
        backend.upsert([chunk], collection=spec.name)

        req = SearchRequest(
            query="foo",
            collection=spec.name,
            top_k=5,
            embedding=_vec(0),
        )
        results = backend.search(req)
        ids = [r.chunk_id for r in results]
        assert chunk_id in ids

    def test_search_returns_results_sorted_by_score_desc(self, qdrant_backend):
        backend = self._backend(qdrant_backend)
        spec = _spec()
        backend.create_collection(spec)

        chunk_ids = [_uid() for _ in range(DIM)]
        chunks = [
            _make_chunk(chunk_ids[i], embedding=_vec(i), idx=i) for i in range(DIM)
        ]
        backend.upsert(chunks, collection=spec.name)

        req = SearchRequest(
            query="test",
            collection=spec.name,
            top_k=DIM,
            embedding=_vec(0),
        )
        results = backend.search(req)
        assert len(results) >= 1
        for a, b in zip(results, results[1:]):
            assert a.score >= b.score - 1e-6
        assert results[0].chunk_id == chunk_ids[0]

    def test_search_respects_top_k(self, qdrant_backend):
        backend = self._backend(qdrant_backend)
        spec = _spec()
        backend.create_collection(spec)

        chunks = [
            _make_chunk(_uid(), embedding=_vec(i), idx=i) for i in range(DIM)
        ]
        backend.upsert(chunks, collection=spec.name)

        for k in (1, 2, 3):
            req = SearchRequest(
                query="test",
                collection=spec.name,
                top_k=k,
                embedding=_vec(0),
            )
            results = backend.search(req)
            assert len(results) <= k

    def test_search_respects_metadata_filter_project(self, qdrant_backend):
        backend = self._backend(qdrant_backend)
        spec = _spec()
        backend.create_collection(spec)

        a1, a2, b1 = _uid(), _uid(), _uid()
        backend.upsert(
            [
                _make_chunk(a1, embedding=_vec(0), metadata={"project": "proj_a", "path": "/a.py"}),
                _make_chunk(a2, embedding=_vec(1), metadata={"project": "proj_a", "path": "/b.py"}),
                _make_chunk(b1, embedding=_vec(2), metadata={"project": "proj_b", "path": "/c.py"}),
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
        results = backend.search(req)
        assert len(results) == 2
        for r in results:
            assert r.metadata.get("project") == "proj_a"

    def test_search_respects_metadata_filter_path_prefix(self, qdrant_backend):
        backend = self._backend(qdrant_backend)
        spec = _spec()
        backend.create_collection(spec)

        p1, p2, p3 = _uid(), _uid(), _uid()
        backend.upsert(
            [
                _make_chunk(p1, embedding=_vec(0), metadata={"project": "p", "path": "/src/rag/foo.py"}),
                _make_chunk(p2, embedding=_vec(1), metadata={"project": "p", "path": "/src/rag/bar.py"}),
                _make_chunk(p3, embedding=_vec(2), metadata={"project": "p", "path": "/src/api/routes.py"}),
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
        results = backend.search(req)
        assert len(results) == 2
        for r in results:
            assert r.metadata.get("path", "").startswith("/src/rag/")

    def test_search_returns_empty_list_when_collection_missing(self, qdrant_backend):
        backend = self._backend(qdrant_backend)
        req = SearchRequest(
            query="anything",
            collection="nonexistent_qdrant_collection_xyz_999",
            top_k=5,
            embedding=_vec(0),
        )
        results = backend.search(req)
        assert results == []

    def test_delete_by_filter_removes_only_matching_chunks(self, qdrant_backend):
        backend = self._backend(qdrant_backend)
        spec = _spec()
        backend.create_collection(spec)

        d1, d2 = _uid(), _uid()
        backend.upsert(
            [
                _make_chunk(d1, document_id="doc_x", embedding=_vec(0), metadata={"project": "p", "path": "/x.py"}),
                _make_chunk(d2, document_id="doc_y", embedding=_vec(1), metadata={"project": "p", "path": "/y.py"}),
            ],
            collection=spec.name,
        )

        deleted = backend.delete_by_filter(
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
        results = backend.search(req)
        ids = [r.chunk_id for r in results]
        assert d1 not in ids
        assert d2 in ids

    def test_stats_reports_collection_count_and_vector_count(self, qdrant_backend):
        backend = self._backend(qdrant_backend)
        spec1 = _spec()
        spec2 = _spec()
        backend.create_collection(spec1)
        backend.create_collection(spec2)

        backend.upsert([_make_chunk(_uid(), embedding=_vec(0))], collection=spec1.name)
        backend.upsert([_make_chunk(_uid(), embedding=_vec(1))], collection=spec2.name)

        stats = backend.stats()
        assert stats.collection_count >= 2
        if stats.vector_count is not None:
            assert stats.vector_count >= 2

    def test_search_result_carries_backend_name_field(self, qdrant_backend):
        backend = self._backend(qdrant_backend)
        spec = _spec()
        backend.create_collection(spec)

        cid = _uid()
        backend.upsert([_make_chunk(cid, embedding=_vec(0))], collection=spec.name)

        req = SearchRequest(
            query="test",
            collection=spec.name,
            top_k=1,
            embedding=_vec(0),
        )
        results = backend.search(req)
        assert len(results) == 1
        assert results[0].backend == BackendName.QDRANT


# ---------------------------------------------------------------------------
# 2. Qdrant-specific tests
# ---------------------------------------------------------------------------


def test_qdrant_payload_indexes_created_for_required_filters(qdrant_backend):
    """Verify that all required payload indexes are created on a new collection.

    Required indexes (per the plan): project, path, language, symbol,
    chunk_type, git_commit.
    """
    from qdrant_client import QdrantClient

    spec = _spec()
    qdrant_backend.create_collection(spec)

    client: QdrantClient = qdrant_backend._client
    collection_info = client.get_collection(spec.name)

    # Get the payload schema — indexed fields are listed there.
    payload_schema = collection_info.payload_schema or {}
    indexed_fields = set(payload_schema.keys())

    required = {"project", "path", "language", "symbol", "chunk_type", "git_commit"}
    missing = required - indexed_fields
    assert not missing, f"Missing payload indexes: {missing}"


def test_qdrant_hnsw_cosine_config_applied(qdrant_backend):
    """Verify the collection uses cosine distance and HNSW index."""
    from qdrant_client.models import Distance

    spec = _spec()
    qdrant_backend.create_collection(spec)

    client = qdrant_backend._client
    collection_info = client.get_collection(spec.name)

    config = collection_info.config
    assert config.params.vectors.distance == Distance.COSINE


def test_qdrant_upsert_uses_chunk_id_as_point_id(qdrant_backend):
    """Verify that the Qdrant point ID is derived from the chunk_id.

    After upsert, the chunk must be retrievable via the same chunk_id that was
    used when inserting, with the chunk_id stored in the point payload.
    """
    spec = _spec()
    qdrant_backend.create_collection(spec)

    cid = _uid()
    chunk = _make_chunk(cid, content="unique content", embedding=_vec(0))
    qdrant_backend.upsert([chunk], collection=spec.name)

    # Retrieve via search and verify chunk_id round-trips correctly.
    req = SearchRequest(
        query="unique content",
        collection=spec.name,
        top_k=1,
        embedding=_vec(0),
    )
    results = qdrant_backend.search(req)
    assert len(results) == 1
    assert results[0].chunk_id == cid


def test_qdrant_returns_only_chunk_id_and_score(qdrant_backend):
    """Verify the canonical-store contract: content comes from ChunkStore.

    The QdrantBackend must store the chunk_id in the point payload so the
    pipeline layer can look up the full text in SQLite.  The ``content``
    field in SearchResult may be empty (or populated from payload for
    convenience), but ``chunk_id`` must always be correct.
    """
    spec = _spec()
    qdrant_backend.create_collection(spec)

    cid = _uid()
    chunk = _make_chunk(cid, content="canonical content lives in sqlite", embedding=_vec(0))
    qdrant_backend.upsert([chunk], collection=spec.name)

    req = SearchRequest(
        query="canonical",
        collection=spec.name,
        top_k=1,
        embedding=_vec(0),
    )
    results = qdrant_backend.search(req)
    assert len(results) == 1
    result = results[0]
    # chunk_id must be present and correct — this is the canonical-store key.
    assert result.chunk_id == cid
    # score must be a meaningful float in [−1, 1] for cosine similarity.
    assert -1.0 <= result.score <= 1.0 + 1e-6


def test_qdrant_upsert_is_idempotent(qdrant_backend):
    """Upserting the same chunk_id twice must not duplicate the point."""
    spec = _spec()
    qdrant_backend.create_collection(spec)

    cid = _uid()
    chunk = _make_chunk(cid, content="first version", embedding=_vec(0))
    qdrant_backend.upsert([chunk], collection=spec.name)
    # Upsert again with updated content — should overwrite, not duplicate.
    chunk2 = _make_chunk(cid, content="updated version", embedding=_vec(0))
    qdrant_backend.upsert([chunk2], collection=spec.name)

    req = SearchRequest(
        query="version",
        collection=spec.name,
        top_k=10,
        embedding=_vec(0),
    )
    results = qdrant_backend.search(req)
    matching = [r for r in results if r.chunk_id == cid]
    assert len(matching) == 1, f"Expected 1 point for chunk_id {cid}, got {len(matching)}"
