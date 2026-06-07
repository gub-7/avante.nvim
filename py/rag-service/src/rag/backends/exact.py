"""Exact (ripgrep-backed) backend adapter.

Wraps the existing :class:`~rag.exact_search.ExactSearch` so it satisfies
the :class:`~rag.backends.base.RagBackend` protocol.  Collections are mapped
to filesystem directories; embeddings are ignored because ripgrep works on
raw source text.

Design notes:
- ``upsert`` stores chunk metadata in-memory for post-search filtering; the
  filesystem is the source of truth for content.
- ``search`` converts the ``SearchRequest`` into a ``RetrievalQuery``, runs
  ripgrep via ``ExactSearch``, and maps ``FileSpan`` objects back to
  ``SearchResult`` — filtering by stored chunk metadata when available.
- ``delete_by_filter`` removes chunks from the in-memory metadata index; the
  underlying files are left untouched.
- ``stats`` always returns ``vector_count=None`` (not applicable for
  ripgrep-based search).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from libs.utils import path_to_uri
from models.rag import RetrievalQuery
from rag.backends.base import (
    BackendName,
    BackendStats,
    CollectionSpec,
    EmbeddedChunk,
    MetadataFilter,
    SearchRequest,
    SearchResult,
)
from rag.exact_search import ExactSearch, _extract_targets


def _effective_fields(chunk: EmbeddedChunk) -> dict[str, Any]:
    """Return chunk metadata merged with structural id fields."""
    return {
        **chunk.metadata,
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
    }


def _matches_filter(fields: dict[str, Any], f: MetadataFilter) -> bool:
    """Return True iff *fields* satisfies the filter predicate."""
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
    # Unknown operator — permissive fallback
    return True


class ExactBackend:
    """RagBackend adapter wrapping :class:`~rag.exact_search.ExactSearch`.

    Each "collection" maps to a filesystem directory.  The ``ExactSearch``
    engine runs ripgrep (or the pure-Python fallback) over that directory
    and converts the resulting :class:`~models.rag.FileSpan` objects into
    :class:`~rag.backends.base.SearchResult` objects.

    Metadata stored via :meth:`upsert` is kept in an in-memory index and
    used only for post-search filtering; it does not affect what ripgrep
    can find.
    """

    name: BackendName = BackendName.EXACT

    def __init__(self) -> None:
        self._es = ExactSearch()
        # collection_name -> {"base_path": Path, "chunks": {chunk_id: EmbeddedChunk}}
        self._collections: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def create_collection(self, spec: CollectionSpec) -> None:
        """Register a named collection pointing to a filesystem directory.

        The base path is taken from ``spec.metadata["base_path"]`` when
        present; otherwise the collection name is used as a relative path.

        This operation is idempotent — calling it twice with the same name
        is safe.
        """
        if spec.name in self._collections:
            return
        raw_path = spec.metadata.get("base_path", spec.name)
        self._collections[spec.name] = {
            "base_path": Path(raw_path),
            "chunks": {},  # chunk_id -> EmbeddedChunk
        }

    def upsert(self, chunks: list[EmbeddedChunk], collection: str) -> None:
        """Store chunk metadata for later post-search filtering.

        The filesystem (ripgrep) remains the canonical content source;
        this call only records the metadata index.

        Automatically creates the collection on first use (using
        ``chunk.metadata["base_path"]`` if available, otherwise the
        collection name).
        """
        if collection not in self._collections:
            # Auto-create: try to infer base_path from first chunk
            raw_path: str = ""
            if chunks:
                raw_path = chunks[0].metadata.get("base_path", collection)
            self._collections[collection] = {
                "base_path": Path(raw_path or collection),
                "chunks": {},
            }
        store = self._collections[collection]["chunks"]
        for chunk in chunks:
            store[chunk.chunk_id] = chunk

    def search(self, request: SearchRequest) -> list[SearchResult]:
        """Run ripgrep against the collection directory and return results.

        The ``request.embedding`` field is ignored; search is text-only.
        Results are sorted by ``score`` descending and capped at
        ``request.top_k``.  Returns an empty list (never raises) when the
        collection is not registered.
        """
        col = self._collections.get(request.collection)
        if col is None:
            return []

        base_path: Path = col["base_path"]
        stored_chunks: dict[str, EmbeddedChunk] = col["chunks"]

        # Build a RetrievalQuery so _extract_targets can derive search terms
        # and their priority-based base scores from the raw query string.
        # Pass the query as ``latest_error`` so that stack-frame patterns
        # (e.g. "src/main.py:1") and error-symbol identifiers in the query
        # receive their elevated base scores (4.5 / 4.0) from _extract_targets.
        rq = RetrievalQuery(
            query=request.query,
            base_uri=path_to_uri(base_path),
            latest_error=request.query,
        )
        spans = self._es.retrieve(rq, base_path)

        results: list[SearchResult] = []
        for span in spans:
            # Attempt to match the span back to a stored chunk by path.
            matched_chunk: EmbeddedChunk | None = None
            if span.path:
                for chunk in stored_chunks.values():
                    chunk_path = chunk.metadata.get("path", "")
                    if chunk_path and (
                        chunk_path == span.path
                        or span.path.endswith(chunk_path)
                        or chunk_path.endswith(span.path)
                    ):
                        matched_chunk = chunk
                        break

            # Apply metadata filters when we have a matched chunk; skip
            # unmatched spans when filters are active (no metadata to test).
            if matched_chunk is not None:
                effective = _effective_fields(matched_chunk)
                if not all(_matches_filter(effective, f) for f in request.filters):
                    continue
                chunk_id = matched_chunk.chunk_id
                metadata = dict(matched_chunk.metadata)
                document_id = matched_chunk.document_id
            else:
                if request.filters:
                    # Cannot evaluate filters without stored metadata — skip.
                    continue
                # Generate a deterministic id from the span URI.
                chunk_id = hashlib.sha256(span.uri.encode()).hexdigest()[:16]
                metadata = {"path": span.path or ""}
                document_id = ""

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    content=span.content,
                    score=span.score,
                    metadata=metadata,
                    backend=self.name,
                    document_id=document_id,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[: request.top_k]

    def delete_by_filter(
        self, collection: str, filters: list[MetadataFilter]
    ) -> int:
        """Remove chunks from the in-memory index that match all filters.

        Underlying files are not modified.  Returns the number of index
        entries removed.  Returns 0 for a missing collection without
        raising.
        """
        col = self._collections.get(collection)
        if col is None:
            return 0
        store: dict[str, EmbeddedChunk] = col["chunks"]
        to_delete = [
            cid
            for cid, chunk in store.items()
            if all(_matches_filter(_effective_fields(chunk), f) for f in filters)
        ]
        for cid in to_delete:
            del store[cid]
        return len(to_delete)

    def stats(self) -> BackendStats:
        """Return aggregate statistics for this backend instance.

        ``vector_count`` is always ``None`` because ripgrep-based search
        does not maintain a vector index.
        """
        return BackendStats(
            backend=self.name,
            collection_count=len(self._collections),
            vector_count=None,
        )

