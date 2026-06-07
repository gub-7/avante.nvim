"""Parameterized contract suite that every RagBackend implementation must satisfy.

Tests in this module are run once per backend fixture. Adding a new backend
to the ``BACKEND_FIXTURES`` list is all that is required to enrol it in the
contract.

TDD: these tests are written first and drive the implementation of
``src/rag/backends/``.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from rag.backends.base import (
    BackendName,
    BackendStats,
    CollectionSpec,
    EmbeddedChunk,
    MetadataFilter,
    RagBackend,
    SearchMode,
    SearchRequest,
    SearchResult,
)
from rag.backends.chroma import ChromaBackend
from rag.backends.in_memory import InMemoryBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 4  # Small embedding dimension for tests


def _vec(i: int, dim: int = DIM) -> list[float]:
    """Return a deterministic unit vector for test chunk `i`."""
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


def _make_chunk(
    chunk_id: str,
    document_id: str = "doc1",
    content: str = "hello world",
    embedding: list[float] | None = None,
    metadata: dict[str, Any] | None = None,
    idx: int = 0,
) -> EmbeddedChunk:
    import hashlib

    return EmbeddedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
        embedding=embedding if embedding is not None else _vec(idx),
        metadata=metadata or {"project": "proj_a", "path": "/src/main.py"},
        token_count=len(content.split()),
    )


def _spec(name: str = "test_col", dim: int = DIM) -> CollectionSpec:
    return CollectionSpec(name=name, dimension=dim)


@pytest.fixture
def in_memory_backend() -> InMemoryBackend:
    return InMemoryBackend(name=BackendName.IN_MEMORY)


@pytest.fixture
def chroma_backend(tmp_path) -> ChromaBackend:
    """ChromaBackend using a persistent client scoped to the test's tmp_path.

    Each pytest test gets its own ``tmp_path``, so the Chroma data directory is
    unique and fully isolated.  We use ``PersistentClient`` rather than
    ``EphemeralClient`` because chromadb 1.x shares a global in-memory store
    across all ``EphemeralClient`` instances in the same process.
    """
    import chromadb

    chroma_dir = tmp_path / "chroma_test"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return ChromaBackend(name=BackendName.CHROMA, client=client)


# Parameterize over backend fixtures — add new ones here to enrol them.
BACKEND_FIXTURES = ["in_memory_backend", "chroma_backend"]


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend_fixture", BACKEND_FIXTURES)
class TestRagBackendContract:
    """Each method is one contract requirement.  All backends must satisfy all."""

    # ------------------------------------------------------------------
    # 1. create_collection is idempotent
    # ------------------------------------------------------------------
    def test_create_collection_is_idempotent(self, backend_fixture, request):
        backend: RagBackend = request.getfixturevalue(backend_fixture)
        spec = _spec()
        # First call must not raise
        backend.create_collection(spec)
        # Second call on the same name must also not raise
        backend.create_collection(spec)

    # ------------------------------------------------------------------
    # 2. upsert makes chunks immediately searchable
    # ------------------------------------------------------------------
    def test_upsert_returns_after_chunks_are_searchable(self, backend_fixture, request):
        backend: RagBackend = request.getfixturevalue(backend_fixture)
        spec = _spec()
        backend.create_collection(spec)

        chunk = _make_chunk("c1", content="def foo(): pass", embedding=_vec(0), idx=0)
        backend.upsert([chunk], collection=spec.name)

        req = SearchRequest(
            query="foo",
            collection=spec.name,
            top_k=5,
            embedding=_vec(0),
        )
        results = backend.search(req)
        ids = [r.chunk_id for r in results]
        assert "c1" in ids

    # ------------------------------------------------------------------
    # 3. search results are sorted by score descending
    # ------------------------------------------------------------------
    def test_search_returns_results_sorted_by_score_desc(self, backend_fixture, request):
        backend: RagBackend = request.getfixturevalue(backend_fixture)
        spec = _spec()
        backend.create_collection(spec)

        # Insert chunks with distinct, orthogonal embeddings.
        # Query with vec(0) → chunk "c0" should rank highest.
        chunks = [_make_chunk(f"c{i}", embedding=_vec(i), idx=i) for i in range(DIM)]
        backend.upsert(chunks, collection=spec.name)

        req = SearchRequest(
            query="test",
            collection=spec.name,
            top_k=DIM,
            embedding=_vec(0),  # most similar to c0
        )
        results = backend.search(req)
        assert len(results) >= 1
        # Scores must be non-increasing
        for a, b in zip(results, results[1:]):
            assert a.score >= b.score - 1e-6, (
                f"Results not sorted: {a.score} < {b.score}"
            )
        # The top result should be the chunk whose embedding matches the query
        assert results[0].chunk_id == "c0"

    # ------------------------------------------------------------------
    # 4. search respects top_k
    # ------------------------------------------------------------------
    def test_search_respects_top_k(self, backend_fixture, request):
        backend: RagBackend = request.getfixturevalue(backend_fixture)
        spec = _spec()
        backend.create_collection(spec)

        chunks = [_make_chunk(f"c{i}", embedding=_vec(i), idx=i) for i in range(DIM)]
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

    # ------------------------------------------------------------------
    # 5. search respects metadata_filter on "project"
    # ------------------------------------------------------------------
    def test_search_respects_metadata_filter_project(self, backend_fixture, request):
        backend: RagBackend = request.getfixturevalue(backend_fixture)
        spec = _spec()
        backend.create_collection(spec)

        # Two chunks in proj_a, one in proj_b
        backend.upsert(
            [
                _make_chunk(
                    "a1",
                    embedding=_vec(0),
                    metadata={"project": "proj_a", "path": "/a.py"},
                    idx=0,
                ),
                _make_chunk(
                    "a2",
                    embedding=_vec(1),
                    metadata={"project": "proj_a", "path": "/b.py"},
                    idx=1,
                ),
                _make_chunk(
                    "b1",
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
        results = backend.search(req)
        assert len(results) == 2
        for r in results:
            assert r.metadata.get("project") == "proj_a"

    # ------------------------------------------------------------------
    # 6. search respects metadata_filter path_prefix
    # ------------------------------------------------------------------
    def test_search_respects_metadata_filter_path_prefix(self, backend_fixture, request):
        backend: RagBackend = request.getfixturevalue(backend_fixture)
        spec = _spec()
        backend.create_collection(spec)

        backend.upsert(
            [
                _make_chunk(
                    "p1",
                    embedding=_vec(0),
                    metadata={"project": "p", "path": "/src/rag/foo.py"},
                    idx=0,
                ),
                _make_chunk(
                    "p2",
                    embedding=_vec(1),
                    metadata={"project": "p", "path": "/src/rag/bar.py"},
                    idx=1,
                ),
                _make_chunk(
                    "p3",
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
        results = backend.search(req)
        assert len(results) == 2
        for r in results:
            assert r.metadata.get("path", "").startswith("/src/rag/")

    # ------------------------------------------------------------------
    # 7. search on missing collection returns empty list (no exception)
    # ------------------------------------------------------------------
    def test_search_returns_empty_list_when_collection_missing(
        self, backend_fixture, request
    ):
        backend: RagBackend = request.getfixturevalue(backend_fixture)
        req = SearchRequest(
            query="anything",
            collection="nonexistent_collection_xyz",
            top_k=5,
            embedding=_vec(0),
        )
        results = backend.search(req)
        assert results == []

    # ------------------------------------------------------------------
    # 8. delete_by_filter removes only matching chunks
    # ------------------------------------------------------------------
    def test_delete_by_filter_removes_only_matching_chunks(
        self, backend_fixture, request
    ):
        backend: RagBackend = request.getfixturevalue(backend_fixture)
        spec = _spec()
        backend.create_collection(spec)

        backend.upsert(
            [
                _make_chunk(
                    "d1",
                    document_id="doc_x",
                    embedding=_vec(0),
                    metadata={"project": "p", "path": "/x.py"},
                    idx=0,
                ),
                _make_chunk(
                    "d2",
                    document_id="doc_y",
                    embedding=_vec(1),
                    metadata={"project": "p", "path": "/y.py"},
                    idx=1,
                ),
            ],
            collection=spec.name,
        )

        deleted = backend.delete_by_filter(
            collection=spec.name,
            filters=[MetadataFilter(field="document_id", op="eq", value="doc_x")],
        )
        assert deleted >= 1

        # doc_x chunk should be gone; doc_y chunk should remain
        req = SearchRequest(
            query="test",
            collection=spec.name,
            top_k=10,
            embedding=_vec(1),
        )
        results = backend.search(req)
        ids = [r.chunk_id for r in results]
        assert "d1" not in ids
        assert "d2" in ids

    # ------------------------------------------------------------------
    # 9. stats reports collection count and vector count
    # ------------------------------------------------------------------
    def test_stats_reports_collection_count_and_vector_count(
        self, backend_fixture, request
    ):
        backend: RagBackend = request.getfixturevalue(backend_fixture)
        spec1 = _spec("col_a")
        spec2 = _spec("col_b")
        backend.create_collection(spec1)
        backend.create_collection(spec2)

        backend.upsert(
            [_make_chunk("s1", embedding=_vec(0), idx=0)],
            collection="col_a",
        )
        backend.upsert(
            [_make_chunk("s2", embedding=_vec(1), idx=1)],
            collection="col_b",
        )

        stats: BackendStats = backend.stats()
        assert isinstance(stats, BackendStats)
        assert stats.collection_count >= 2
        # vector_count may be None for backends that don't support it
        if stats.vector_count is not None:
            assert stats.vector_count >= 2

    # ------------------------------------------------------------------
    # 10. search results carry the backend_name field
    # ------------------------------------------------------------------
    def test_search_result_carries_backend_name_field(self, backend_fixture, request):
        backend: RagBackend = request.getfixturevalue(backend_fixture)
        spec = _spec()
        backend.create_collection(spec)
        backend.upsert(
            [_make_chunk("bn1", embedding=_vec(0), idx=0)],
            collection=spec.name,
        )

        req = SearchRequest(
            query="test",
            collection=spec.name,
            top_k=1,
            embedding=_vec(0),
        )
        results = backend.search(req)
        assert len(results) == 1
        result: SearchResult = results[0]
        assert isinstance(result.backend, BackendName)

