"""Qdrant backend adapter.

Wraps ``qdrant-client`` to satisfy the ``RagBackend`` protocol.

Configuration
-------------
``QDRANT_URL``      — HTTP URL of the Qdrant server (e.g. ``http://localhost:6333``).
``QDRANT_API_KEY``  — Optional API key for Qdrant Cloud instances.

Both settings are read from environment variables via ``src/libs/configs.py``
when no explicit values are passed to the constructor.

Design notes
------------
- Chunk IDs may be arbitrary strings.  They are stored verbatim in the point
  payload under the ``"chunk_id"`` key.  Qdrant point IDs are *derived* UUID
  values produced by ``uuid.uuid5(NAMESPACE_URL, chunk_id)`` so that the
  mapping is deterministic and collision-free.
- The ``document_id`` is also stored in the payload so that
  ``delete_by_filter(field="document_id", op="eq", ...)`` can use Qdrant's
  native payload filtering.
- All required payload indexes (project, path, language, symbol, chunk_type,
  git_commit, document_id, chunk_id) are created at collection creation time
  so Qdrant can use them efficiently.
- ``prefix`` filters are handled client-side after Qdrant returns candidates
  because Qdrant does not natively support string prefix filtering.
"""

from __future__ import annotations

import uuid
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

# Deterministic UUID namespace for chunk_id → point_id derivation.
_CHUNK_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # NAMESPACE_URL

# Payload index fields that are always created.
_INDEXED_FIELDS = [
    "project",
    "path",
    "language",
    "symbol",
    "chunk_type",
    "git_commit",
    "document_id",
    "chunk_id",
]

# Ops that can be pushed down to Qdrant's payload filter.
_NATIVE_OPS = frozenset({"eq", "ne", "in", "gte", "lte"})


def _chunk_id_to_point_id(chunk_id: str) -> str:
    """Convert an arbitrary chunk_id string to a Qdrant-compatible UUID string."""
    try:
        # Already a valid UUID — use it directly.
        uuid.UUID(chunk_id)
        return chunk_id
    except ValueError:
        return str(uuid.uuid5(_CHUNK_NS, chunk_id))


def _build_qdrant_filter(filters: list[MetadataFilter]) -> Any | None:
    """Build a Qdrant ``Filter`` from a list of ``MetadataFilter`` objects.

    Only ``_NATIVE_OPS`` are pushed down; ``prefix`` is handled in Python.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

    conditions = []
    for f in filters:
        if f.op not in _NATIVE_OPS:
            continue
        if f.op == "eq":
            conditions.append(FieldCondition(key=f.field, match=MatchValue(value=f.value)))
        elif f.op == "ne":
            # Qdrant has no "ne" directly; use a must_not block.
            pass  # handled via must_not below
        elif f.op == "in":
            conditions.append(FieldCondition(key=f.field, match=MatchAny(any=list(f.value))))
        elif f.op in ("gte", "lte"):
            rng: dict[str, Any] = {}
            if f.op == "gte":
                rng["gte"] = f.value
            else:
                rng["lte"] = f.value
            conditions.append(FieldCondition(key=f.field, range=Range(**rng)))

    must_not = []
    for f in filters:
        if f.op == "ne":
            must_not.append(FieldCondition(key=f.field, match=MatchValue(value=f.value)))

    if not conditions and not must_not:
        return None
    return Filter(must=conditions or None, must_not=must_not or None)


def _matches_filter_py(payload: dict[str, Any], f: MetadataFilter) -> bool:
    """Python-side filter for ops not supported natively by Qdrant."""
    val = payload.get(f.field)
    if f.op == "prefix":
        return isinstance(val, str) and val.startswith(f.value)
    return True  # all other ops are already pushed to Qdrant


class QdrantBackend:
    """Qdrant-backed implementation of the ``RagBackend`` protocol.

    Parameters
    ----------
    name:
        Backend identifier used in ``SearchResult.backend``.
    url:
        Qdrant server URL.  If *None*, falls back to the ``QDRANT_URL``
        environment variable (via ``libs.configs``).
    api_key:
        Optional API key.  If *None*, falls back to ``QDRANT_API_KEY``.
    client:
        An already-constructed ``QdrantClient`` instance.  When supplied,
        *url* and *api_key* are ignored.  Useful for testing with
        ``QdrantClient(":memory:")``.
    """

    def __init__(
        self,
        name: BackendName = BackendName.QDRANT,
        url: str | None = None,
        api_key: str | None = None,
        client: Any = None,
    ) -> None:
        self.name = name
        if client is not None:
            self._client = client
        else:
            from qdrant_client import QdrantClient

            if url is None:
                from libs.configs import QDRANT_URL

                url = QDRANT_URL
            if api_key is None:
                from libs.configs import QDRANT_API_KEY

                api_key = QDRANT_API_KEY or None

            self._client = QdrantClient(url=url, api_key=api_key)

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the Qdrant server is reachable."""
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    def create_collection(self, spec: CollectionSpec) -> None:
        """Create (or ensure) a Qdrant collection with HNSW + cosine config.

        Idempotent: if the collection already exists this is a no-op.
        After creation, payload indexes for all ``_INDEXED_FIELDS`` are
        created so Qdrant can filter efficiently.
        """
        from qdrant_client.models import (
            Distance,
            HnswConfigDiff,
            PayloadSchemaType,
            VectorParams,
        )

        existing = {c.name for c in self._client.get_collections().collections}
        if spec.name not in existing:
            self._client.create_collection(
                collection_name=spec.name,
                vectors_config=VectorParams(
                    size=spec.dimension,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(
                        m=16,
                        ef_construct=100,
                    ),
                ),
            )
            # Create payload indexes for all required filter fields.
            for field_name in _INDEXED_FIELDS:
                try:
                    self._client.create_payload_index(
                        collection_name=spec.name,
                        field_name=field_name,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    # Index already exists or not supported — ignore.
                    pass

    def upsert(self, chunks: list[EmbeddedChunk], collection: str) -> None:
        """Insert or update chunks in the named collection.

        Each chunk is stored as a Qdrant point with:
        - ID: ``_chunk_id_to_point_id(chunk.chunk_id)``
        - Vector: ``chunk.embedding``
        - Payload: ``chunk.metadata`` + ``chunk_id`` + ``document_id``
        """
        from qdrant_client.models import PointStruct

        if not chunks:
            return

        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                # Skip chunks without embeddings — Qdrant requires a vector.
                continue
            payload = {
                **chunk.metadata,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                # Store content in payload for round-trip convenience;
                # production code should resolve via ChunkStore instead.
                "_content": chunk.content,
            }
            points.append(
                PointStruct(
                    id=_chunk_id_to_point_id(chunk.chunk_id),
                    vector=chunk.embedding,
                    payload=payload,
                )
            )
        if points:
            self._client.upsert(collection_name=collection, points=points)

    def search(self, request: SearchRequest) -> list[SearchResult]:
        """Execute a vector search against the named Qdrant collection.

        Returns results sorted by score descending.  Returns an empty list
        (no exception) when the collection does not exist.
        """
        if request.embedding is None:
            return []

        try:
            qdrant_filter = _build_qdrant_filter(request.filters)
            non_native = [f for f in request.filters if f.op not in _NATIVE_OPS]

            # Fetch extra results to compensate for Python-side post-filtering.
            fetch_k = request.top_k + max(len(non_native) * 10, 0)

            search_kwargs: dict[str, Any] = {
                "collection_name": request.collection,
                "query_vector": request.embedding,
                "limit": fetch_k,
                "with_payload": True,
            }
            if qdrant_filter is not None:
                search_kwargs["query_filter"] = qdrant_filter

            hits = self._client.search(**search_kwargs)
        except Exception:
            return []

        results: list[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            # Apply non-native filters in Python.
            if not all(_matches_filter_py(payload, f) for f in non_native):
                continue

            chunk_id = payload.get("chunk_id", str(hit.id))
            content = payload.get("_content", "")
            document_id = payload.get("document_id", "")
            metadata = {
                k: v
                for k, v in payload.items()
                if k not in ("chunk_id", "document_id", "_content")
            }

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    content=content,
                    text=content,
                    score=hit.score,
                    metadata=metadata,
                    backend=self.name,
                    document_id=document_id,
                )
            )
            if len(results) >= request.top_k:
                break

        return results

    def delete_by_filter(
        self, collection: str, filters: list[MetadataFilter]
    ) -> int:
        """Delete points matching ALL filters from the collection.

        Non-native ops (e.g. ``prefix``) are handled by first fetching
        matching IDs and then deleting them explicitly.  Returns the number
        of points deleted, or 0 for a missing collection.
        """
        from qdrant_client.models import FilterSelector

        try:
            # Build native filter first.
            qdrant_filter = _build_qdrant_filter(filters)
            non_native = [f for f in filters if f.op not in _NATIVE_OPS]

            if not non_native and qdrant_filter is not None:
                # Fully native path — let Qdrant do the deletion.
                result = self._client.delete(
                    collection_name=collection,
                    points_selector=FilterSelector(filter=qdrant_filter),
                )
                # Qdrant returns an UpdateResult; count is not directly available,
                # so return 1 as a sentinel meaning "some deletion occurred".
                return 1

            # Mixed / non-native path: scroll to find matching IDs.
            scroll_kwargs: dict[str, Any] = {
                "collection_name": collection,
                "with_payload": True,
                "limit": 10000,
            }
            if qdrant_filter is not None:
                scroll_kwargs["scroll_filter"] = qdrant_filter

            records, _ = self._client.scroll(**scroll_kwargs)

            ids_to_delete = []
            for rec in records:
                payload = rec.payload or {}
                if all(_matches_filter_py(payload, f) for f in non_native):
                    ids_to_delete.append(rec.id)

            if not ids_to_delete:
                return 0

            self._client.delete(
                collection_name=collection,
                points_selector=ids_to_delete,
            )
            return len(ids_to_delete)

        except Exception:
            return 0

    def stats(self) -> BackendStats:
        """Return aggregate statistics for this Qdrant instance."""
        try:
            collections = self._client.get_collections().collections
            total_vectors = 0
            for col in collections:
                try:
                    info = self._client.get_collection(col.name)
                    count = info.vectors_count or 0
                    total_vectors += count
                except Exception:
                    pass
            return BackendStats(
                backend=self.name,
                collection_count=len(collections),
                vector_count=total_vectors,
            )
        except Exception:
            return BackendStats(
                backend=self.name,
                collection_count=0,
                vector_count=None,
            )
