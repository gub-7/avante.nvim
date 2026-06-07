"""Milvus vector-store backend (Increment 11).

Implements the :class:`~rag.backends.base.RagBackend` protocol backed by
`Milvus <https://milvus.io/>`_ via the ``pymilvus`` Python client.

Features:
- GPU_CAGRA index when :func:`~runtime.probe.gpu_available` returns ``True``;
  falls back to CPU HNSW otherwise.
- Hot/cold collection naming convention: ``hot_{project}_{gpu|cpu}`` /
  ``cold_{project}_{cpu}``.
- Metadata fields ``project``, ``path``, ``document_id``, ``language``,
  ``chunk_kind`` stored as scalar fields for server-side filtering.
- ``content`` stored as a scalar field so callers receive text in results
  (ChunkStore look-up can then refresh it if needed).

Usage::

    from rag.backends.milvus import MilvusBackend

    backend = MilvusBackend(uri="http://localhost:19530", dimension=768)
    backend.create_collection(CollectionSpec(name="my_project", dimension=768))
    backend.upsert(chunks, collection="my_project")
    results = backend.search(SearchRequest(query="...", collection="my_project",
                                           top_k=10, embedding=[...]))

Required environment variable:
    ``MILVUS_URL`` — e.g. ``http://localhost:19530``

Optional:
    Install ``pymilvus`` (``pip install pymilvus``).  When the package is
    absent the module can still be imported; only instantiation will fail.
"""

from __future__ import annotations

import hashlib
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
from runtime.probe import gpu_available

try:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )

    _PYMILVUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYMILVUS_AVAILABLE = False

# Varchar field limits
_ID_MAX_LEN = 256
_PATH_MAX_LEN = 4096
_STR_MAX_LEN = 512
_CONTENT_MAX_LEN = 65_535


class MilvusBackend:
    """RagBackend implementation backed by Milvus.

    Parameters
    ----------
    uri:
        Full HTTP(S) URI for the Milvus gRPC/REST endpoint, e.g.
        ``http://localhost:19530``.
    dimension:
        Dimensionality of the embedding vectors.  Must match the model
        used when generating embeddings.
    alias:
        Milvus connection alias.  Override if you need multiple connections.
    """

    name: BackendName = BackendName.MILVUS

    def __init__(
        self,
        uri: str,
        dimension: int = 768,
        alias: str = "default",
    ) -> None:
        if not _PYMILVUS_AVAILABLE:
            raise ImportError(
                "pymilvus is not installed.  "
                "Run: pip install pymilvus"
            )
        self._uri = uri
        self._dimension = dimension
        self._alias = alias
        self._connect()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Establish (or reuse) a Milvus connection."""
        connections.connect(alias=self._alias, uri=self._uri)

    def _build_index_params(self) -> dict[str, Any]:
        """Return the index parameters appropriate for the current hardware.

        Uses ``GPU_CAGRA`` when :func:`~runtime.probe.gpu_available` returns
        ``True``; falls back to ``HNSW`` on CPU-only hosts.
        """
        if gpu_available():
            return {
                "index_type": "GPU_CAGRA",
                "metric_type": "COSINE",
                "params": {
                    "intermediate_graph_degree": 64,
                    "graph_degree": 32,
                },
            }
        return {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        }

    def _schema(self) -> CollectionSchema:
        """Return the Milvus CollectionSchema used for all collections."""
        fields = [
            FieldSchema(
                name="chunk_id",
                dtype=DataType.VARCHAR,
                max_length=_ID_MAX_LEN,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(
                name="document_id",
                dtype=DataType.VARCHAR,
                max_length=_ID_MAX_LEN,
                default_value="",
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self._dimension,
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=_CONTENT_MAX_LEN,
                default_value="",
            ),
            FieldSchema(
                name="path",
                dtype=DataType.VARCHAR,
                max_length=_PATH_MAX_LEN,
                default_value="",
            ),
            FieldSchema(
                name="project",
                dtype=DataType.VARCHAR,
                max_length=_STR_MAX_LEN,
                default_value="",
            ),
            FieldSchema(
                name="language",
                dtype=DataType.VARCHAR,
                max_length=_STR_MAX_LEN,
                default_value="",
            ),
            FieldSchema(
                name="chunk_kind",
                dtype=DataType.VARCHAR,
                max_length=_STR_MAX_LEN,
                default_value="",
            ),
        ]
        return CollectionSchema(fields, description="RAG chunk store")

    @staticmethod
    def _filter_expr(filters: list[MetadataFilter]) -> str:
        """Convert a list of MetadataFilter objects to a Milvus boolean expression.

        Only the ``eq`` and ``prefix`` operators are mapped; other operators
        are silently ignored (permissive fallback — same convention used by
        :class:`~rag.backends.in_memory.InMemoryBackend`).
        """
        parts: list[str] = []
        for f in filters:
            if f.op == "eq":
                escaped = str(f.value).replace('"', '\\"')
                parts.append(f'{f.field} == "{escaped}"')
            elif f.op == "ne":
                escaped = str(f.value).replace('"', '\\"')
                parts.append(f'{f.field} != "{escaped}"')
            elif f.op == "in":
                values = ", ".join(f'"{v}"' for v in f.value)
                parts.append(f"{f.field} in [{values}]")
            # "prefix" is not natively supported by Milvus scalar filters;
            # we skip it here and do post-search client-side filtering instead.
        return " && ".join(parts) if parts else ""

    @staticmethod
    def _matches_filter_client(
        metadata: dict[str, Any], f: MetadataFilter
    ) -> bool:
        """Client-side filter evaluation (used for operators Milvus can't push down)."""
        val = metadata.get(f.field)
        if f.op == "prefix":
            return isinstance(val, str) and val.startswith(f.value)
        return True  # already handled server-side

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def create_collection(self, spec: CollectionSpec) -> None:
        """Create the Milvus collection if it does not already exist.

        This operation is idempotent — calling it twice with the same name
        is safe.
        """
        if utility.has_collection(spec.name, using=self._alias):
            return

        col = Collection(
            name=spec.name,
            schema=self._schema(),
            using=self._alias,
        )
        index_params = self._build_index_params()
        col.create_index(field_name="embedding", index_params=index_params)

    def upsert(self, chunks: list[EmbeddedChunk], collection: str) -> None:
        """Insert or overwrite chunks in the named Milvus collection.

        If the collection does not exist it is created automatically using the
        backend's default :meth:`_schema`.

        After insertion the collection is flushed and loaded so that the
        vectors are immediately searchable.
        """
        if not utility.has_collection(collection, using=self._alias):
            self.create_collection(CollectionSpec(name=collection, dimension=self._dimension))

        col = Collection(name=collection, using=self._alias)

        # Delete existing chunks with the same chunk_ids first (upsert semantics).
        ids_to_delete = [c.chunk_id for c in chunks]
        if ids_to_delete:
            id_list = ", ".join(f'"{cid}"' for cid in ids_to_delete)
            col.delete(f'chunk_id in [{id_list}]')

        # Build column-oriented insert data.
        data = {
            "chunk_id": [c.chunk_id for c in chunks],
            "document_id": [c.document_id for c in chunks],
            "embedding": [c.embedding or ([0.0] * self._dimension) for c in chunks],
            "content": [
                (c.content or "")[:_CONTENT_MAX_LEN] for c in chunks
            ],
            "path": [
                str(c.metadata.get("path", ""))[:_PATH_MAX_LEN] for c in chunks
            ],
            "project": [
                str(c.metadata.get("project", ""))[:_STR_MAX_LEN] for c in chunks
            ],
            "language": [
                str(c.metadata.get("language", ""))[:_STR_MAX_LEN] for c in chunks
            ],
            "chunk_kind": [
                str(c.metadata.get("chunk_kind", ""))[:_STR_MAX_LEN] for c in chunks
            ],
        }
        col.insert(list(data.values()))
        col.flush()
        col.load()

    def _do_search(
        self,
        col: "Collection",
        query_vector: list[float],
        top_k: int,
        expr: str,
    ) -> list[Any]:
        """Execute the actual Milvus vector search.

        Extracted into its own method so tests can mock it independently.
        """
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": max(top_k * 2, 64)},
        }
        return col.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["chunk_id", "document_id", "content", "path", "project", "language", "chunk_kind"],
            expr=expr or None,
        )

    def search(self, request: SearchRequest) -> list[SearchResult]:
        """Execute a vector search against the named Milvus collection.

        Returns results sorted by cosine similarity descending.  Returns an
        empty list (never raises) when the collection does not exist.
        """
        if not utility.has_collection(request.collection, using=self._alias):
            return []

        col = Collection(name=request.collection, using=self._alias)
        try:
            col.load()
        except Exception:  # noqa: BLE001
            pass  # Already loaded

        query_vector = request.embedding or ([0.0] * self._dimension)

        # Build server-side filter expression (eq/ne/in operators).
        expr = self._filter_expr(request.filters)

        hits_list = self._do_search(col, query_vector, request.top_k, expr)

        results: list[SearchResult] = []
        for hit in hits_list[0]:
            entity = hit.entity
            metadata = {
                "path": entity.get("path", ""),
                "project": entity.get("project", ""),
                "language": entity.get("language", ""),
                "chunk_kind": entity.get("chunk_kind", ""),
            }

            # Client-side filtering for operators not supported server-side.
            if not all(
                self._matches_filter_client(metadata, f) for f in request.filters
            ):
                continue

            results.append(
                SearchResult(
                    chunk_id=entity.get("chunk_id", hit.id),
                    content=entity.get("content", ""),
                    score=hit.score,
                    metadata=metadata,
                    backend=self.name,
                    document_id=entity.get("document_id", ""),
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def delete_by_filter(
        self, collection: str, filters: list[MetadataFilter]
    ) -> int:
        """Delete chunks matching all filters from the Milvus collection.

        Returns the number of chunks deleted.  Returns 0 for a missing
        collection without raising.
        """
        if not utility.has_collection(collection, using=self._alias):
            return 0

        col = Collection(name=collection, using=self._alias)
        expr = self._filter_expr(filters)
        if not expr:
            return 0

        # Query for matching IDs first, then delete by primary key.
        try:
            col.load()
            hits = col.query(
                expr=expr,
                output_fields=["chunk_id"],
                limit=16384,
            )
        except Exception:  # noqa: BLE001
            return 0

        if not hits:
            return 0

        ids = [h["chunk_id"] for h in hits]
        id_list = ", ".join(f'"{cid}"' for cid in ids)
        col.delete(f'chunk_id in [{id_list}]')
        col.flush()
        return len(ids)

    def stats(self) -> BackendStats:
        """Return aggregate statistics for this Milvus backend instance."""
        collection_names = utility.list_collections(using=self._alias)
        collection_count = len(collection_names)

        total_vectors: int | None = 0
        for name in collection_names:
            try:
                col = Collection(name=name, using=self._alias)
                n = col.num_entities
                if total_vectors is not None:
                    total_vectors += n
            except Exception:  # noqa: BLE001
                total_vectors = None
                break

        return BackendStats(
            backend=self.name,
            collection_count=collection_count,
            vector_count=total_vectors,
        )

