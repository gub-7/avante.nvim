"""ChromaDB backend adapter.

Wraps a ``chromadb.Client`` to satisfy the ``RagBackend`` protocol.  In
production the client is a ``PersistentClient`` pointed at
``CHROMA_PERSIST_DIR``; in tests an isolated ``PersistentClient(tmp_path)``
is injected via the constructor so the contract suite runs without touching
any shared state.

The existing ``semantic_search.init_semantic_search()`` / ``SemanticRetriever``
pipeline delegates to this backend for all vector operations once the router
is wired up (Increment 8).  This class intentionally avoids LlamaIndex so it
can be used standalone without an embedding model.
"""

from __future__ import annotations

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


def _build_where(filters: list[MetadataFilter]) -> dict | None:
    """Convert a list of MetadataFilters to a Chroma ``$where`` clause.

    Only ``"eq"`` filters are pushed down to Chroma natively; all other
    operator types (``"prefix"``, ``"ne"``, etc.) are handled in Python
    after the query returns results.
    """
    if not filters:
        return None

    eq_clauses = []
    for f in filters:
        if f.op == "eq":
            eq_clauses.append({f.field: {"$eq": f.value}})

    if not eq_clauses:
        return None
    if len(eq_clauses) == 1:
        return eq_clauses[0]
    return {"$and": eq_clauses}


def _matches_filter_py(metadata: dict[str, Any], f: MetadataFilter) -> bool:
    """Python-side filter evaluation for ops Chroma doesn't support natively."""
    val = metadata.get(f.field)
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
    return True


# Ops that are handled natively by Chroma's $where clause.
_NATIVE_OPS = frozenset({"eq"})


class ChromaBackend:
    """Chromadb-backed implementation of the ``RagBackend`` protocol.

    Parameters
    ----------
    name:
        Backend identifier used in ``SearchResult.backend``.
    client:
        An existing ``chromadb.Client`` instance.  Pass
        ``chromadb.PersistentClient(path=str(tmp_path))`` in tests and
        ``chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))`` in
        production.
    """

    def __init__(self, name: BackendName = BackendName.CHROMA, client: Any = None) -> None:
        self.name = name
        if client is None:
            # Lazy import so callers that inject a client don't pay for
            # the config reload at import time.
            from libs.configs import CHROMA_PERSIST_DIR
            import chromadb

            client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        self._client = client

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the Chroma client is reachable."""
        try:
            self._client.heartbeat()
            return True
        except Exception:
            return False

    def create_collection(self, spec: CollectionSpec) -> None:
        """Create (or retrieve) the named Chroma collection.

        Idempotent: calling twice with the same name is safe.
        The ``spec.dimension`` and ``spec.distance`` are stored as collection
        metadata so they can be validated later; Chroma itself ignores them.
        """
        self._client.get_or_create_collection(
            name=spec.name,
            metadata={
                "dimension": spec.dimension,
                "distance": spec.distance,
                **spec.metadata,
            },
        )

    def upsert(self, chunks: list[EmbeddedChunk], collection: str) -> None:
        """Insert or update chunks in the named collection."""
        col = self._client.get_or_create_collection(name=collection)
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        embeddings = [c.embedding for c in chunks if c.embedding is not None]
        documents = [c.content for c in chunks]
        metadatas = [
            {**c.metadata, "document_id": c.document_id, "chunk_id": c.chunk_id}
            for c in chunks
        ]

        upsert_kwargs: dict[str, Any] = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        # Only pass embeddings if all chunks have them (Chroma requires consistent calls).
        if len(embeddings) == len(chunks):
            upsert_kwargs["embeddings"] = embeddings

        col.upsert(**upsert_kwargs)

    def search(self, request: SearchRequest) -> list[SearchResult]:
        """Query the collection by embedding vector.

        Filters with ``op="eq"`` are pushed down to Chroma; all other ops
        are applied in Python after the query.  Returns an empty list (no
        exception) when the collection does not exist.
        """
        try:
            col = self._client.get_collection(name=request.collection)
        except Exception:
            return []

        if request.embedding is None:
            # No embedding: fall back to returning all docs (filtered)
            all_docs = col.get(include=["documents", "metadatas"])
            results = []
            for chunk_id, doc, meta in zip(
                all_docs["ids"],
                all_docs["documents"],
                all_docs["metadatas"],
            ):
                if not all(_matches_filter_py(meta, f) for f in request.filters):
                    continue
                results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        content=doc,
                        text=doc,
                        score=1.0,
                        metadata={k: v for k, v in meta.items() if k not in ("document_id", "chunk_id")},
                        backend=self.name,
                        document_id=meta.get("document_id", ""),
                    )
                )
            return results[: request.top_k]

        # Push equality filters down; remainder applied in Python.
        where = _build_where(request.filters)
        non_native = [f for f in request.filters if f.op not in _NATIVE_OPS]

        # Fetch extra results to compensate for Python-side post-filtering
        fetch_k = max(request.top_k + len(non_native) * 10, request.top_k)

        try:
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [request.embedding],
                "n_results": fetch_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                query_kwargs["where"] = where

            res = col.query(**query_kwargs)
        except Exception:
            return []

        ids: list[str] = res["ids"][0]
        documents: list[str] = res["documents"][0]
        metadatas: list[dict] = res["metadatas"][0]
        distances: list[float] = res["distances"][0]

        out: list[SearchResult] = []
        for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            # Apply non-native filters in Python
            if not all(_matches_filter_py(meta, f) for f in non_native):
                continue
            # Chroma returns L2 distance; convert to similarity score.
            # For cosine collections, distance = 1 - similarity.
            score = 1.0 - dist
            out.append(
                SearchResult(
                    chunk_id=chunk_id,
                    content=doc,
                    text=doc,
                    score=score,
                    metadata={k: v for k, v in meta.items() if k not in ("document_id", "chunk_id")},
                    backend=self.name,
                    document_id=meta.get("document_id", ""),
                )
            )
            if len(out) >= request.top_k:
                break

        # Results from Chroma are already sorted by distance (ascending).
        # After score inversion (1 - distance) they are sorted descending.
        return out

    def delete_by_filter(
        self, collection: str, filters: list[MetadataFilter]
    ) -> int:
        """Delete chunks matching ALL filters from the collection.

        Only ``eq`` filters are pushed to Chroma's ``where``; the rest are
        applied in Python.  Returns 0 for a missing collection without
        raising.
        """
        try:
            col = self._client.get_collection(name=collection)
        except Exception:
            return 0

        where = _build_where(filters)
        non_native = [f for f in filters if f.op not in _NATIVE_OPS]

        try:
            get_kwargs: dict[str, Any] = {"include": ["metadatas"]}
            if where:
                get_kwargs["where"] = where
            existing = col.get(**get_kwargs)
        except Exception:
            return 0

        ids_to_delete: list[str] = []
        for chunk_id, meta in zip(existing["ids"], existing["metadatas"]):
            if all(_matches_filter_py(meta, f) for f in non_native):
                ids_to_delete.append(chunk_id)

        if not ids_to_delete:
            return 0

        col.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def stats(self) -> BackendStats:
        """Return aggregate statistics for this Chroma instance."""
        collections = self._client.list_collections()
        total_vectors = 0
        for col in collections:
            try:
                c = self._client.get_collection(name=col.name)
                total_vectors += c.count()
            except Exception:
                pass
        return BackendStats(
            backend=self.name,
            collection_count=len(collections),
            vector_count=total_vectors,
        )
