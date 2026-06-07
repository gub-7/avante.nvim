"""In-memory backend — dict-backed implementation used in tests.

This backend never persists anything to disk.  It computes cosine similarity
using plain Python (no external libraries required) so the full contract suite
runs without Docker or any vector-store daemon.

It is **not** intended for production use.
"""

from __future__ import annotations

import math
from typing import Any

from rag.backends.base import (
    BackendName,
    BackendStats,
    CollectionSpec,
    EmbeddedChunk,
    MetadataFilter,
    SearchRequest,
    SearchResult,
)


def _cosine(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _effective_fields(chunk: EmbeddedChunk) -> dict[str, Any]:
    """Return a merged dict of chunk metadata plus top-level structural fields.

    Filters on ``document_id`` and ``chunk_id`` are routed through this so
    callers don't need to replicate structural fields inside the metadata dict.
    """
    return {
        **chunk.metadata,
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
    }


def _matches_filter(fields: dict[str, Any], f: MetadataFilter) -> bool:
    """Return True iff *fields* satisfies filter *f*."""
    val = fields.get(f.field)
    if f.op == "eq":
        return val == f.value
    if f.op == "ne":
        return val != f.value
    if f.op == "prefix":
        return isinstance(val, str) and val.startswith(f.value)
    if f.op == "in":
        return val in f.value
    if f.op == "gte":
        return val is not None and val >= f.value
    if f.op == "lte":
        return val is not None and val <= f.value
    # Unknown op — skip filter (permissive fallback)
    return True


class InMemoryBackend:
    """Dict-backed RagBackend for use in unit tests.

    Collections are stored as ``{collection_name: {chunk_id: EmbeddedChunk}}``.
    Similarity is cosine distance computed in pure Python.
    """

    def __init__(self, name: BackendName = BackendName.IN_MEMORY) -> None:
        self.name = name
        # collection_name -> {chunk_id -> EmbeddedChunk}
        self._collections: dict[str, dict[str, EmbeddedChunk]] = {}
        # Every search() call is appended here so tests can verify routing.
        self.calls: list[SearchRequest] = []

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Always available — no external dependencies."""
        return True

    def create_collection(self, spec: CollectionSpec) -> None:
        """Create the named collection if it does not already exist."""
        if spec.name not in self._collections:
            self._collections[spec.name] = {}

    def upsert(self, chunks: list[EmbeddedChunk], collection: str) -> None:
        """Insert or overwrite chunks in *collection*.

        Automatically creates the collection on first use.
        """
        if collection not in self._collections:
            self._collections[collection] = {}
        store = self._collections[collection]
        for chunk in chunks:
            store[chunk.chunk_id] = chunk

    def search(self, request: SearchRequest) -> list[SearchResult]:
        """Return results sorted by cosine similarity descending.

        Records every call in ``self.calls`` for test-routing assertions.
        Returns an empty list (no exception) when the collection is absent.
        """
        self.calls.append(request)
        store = self._collections.get(request.collection)
        if store is None:
            return []

        query_vec = request.embedding
        results: list[SearchResult] = []

        for chunk in store.values():
            effective = _effective_fields(chunk)
            # Apply metadata filters (AND semantics)
            if not all(_matches_filter(effective, f) for f in request.filters):
                continue

            if query_vec is not None and chunk.embedding is not None:
                score = _cosine(query_vec, chunk.embedding)
            else:
                score = 1.0  # no embedding → treat as equal match

            path = chunk.metadata.get("path") or None
            start_line = chunk.metadata.get("start_line") or None
            end_line = chunk.metadata.get("end_line") or None
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    score=score,
                    backend=self.name,
                    metadata=dict(chunk.metadata),
                    content=chunk.content,
                    text=chunk.content,
                    document_id=chunk.document_id,
                    token_count=chunk.token_count,
                    path=path,
                    start_line=start_line,
                    end_line=end_line,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[: request.top_k]

    def delete_by_filter(
        self, collection: str, filters: list[MetadataFilter]
    ) -> int:
        """Delete chunks whose metadata matches ALL filters.

        Returns the number of chunks removed.  Returns 0 for a missing
        collection without raising.
        """
        store = self._collections.get(collection)
        if store is None:
            return 0

        to_delete = [
            cid
            for cid, chunk in store.items()
            if all(_matches_filter(_effective_fields(chunk), f) for f in filters)
        ]
        for cid in to_delete:
            del store[cid]
        return len(to_delete)

    def stats(self) -> BackendStats:
        """Return aggregate statistics across all collections."""
        total_vectors = sum(len(chunks) for chunks in self._collections.values())
        return BackendStats(
            backend=self.name,
            collection_count=len(self._collections),
            vector_count=total_vectors,
        )
