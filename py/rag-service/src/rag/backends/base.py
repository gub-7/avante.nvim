"""Backend abstraction layer for the RAG service.

Defines the ``RagBackend`` protocol and all shared data types used by every
concrete backend implementation (InMemory, Chroma, Qdrant, Milvus, Exact).

Design principles:
- ``RagBackend`` is a ``typing.Protocol`` so backends don't need to inherit
  from a base class — structural sub-typing keeps things loosely coupled.
- All transfer objects (``EmbeddedChunk``, ``SearchRequest``, …) are plain
  ``dataclasses`` so they are cheap to construct and easy to mock in tests.
- Enums use ``str`` mixins so values serialise directly to JSON strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BackendName(str, Enum):
    """Identifies which backend produced a ``SearchResult``."""

    IN_MEMORY = "in_memory"
    CHROMA = "chroma"
    QDRANT = "qdrant"
    MILVUS = "milvus"
    EXACT = "exact"


class SearchMode(str, Enum):
    """How the backend should execute the search."""

    VECTOR = "vector"
    EXACT = "exact"
    HYBRID = "hybrid"
    AUTO = "auto"


class SearchPurpose(str, Enum):
    """Semantic intent of the search — used by the router for mode selection."""

    ASK = "ask"
    SEARCH = "search"
    EDIT_SMALL = "edit-small"
    TEST_FIX = "test-fix"
    REFACTOR = "refactor"


# ---------------------------------------------------------------------------
# Data-transfer objects
# ---------------------------------------------------------------------------


@dataclass
class MetadataFilter:
    """A single predicate applied to chunk metadata during a search.

    Supported ``op`` values:
        ``"eq"``     — exact equality
        ``"ne"``     — not equal
        ``"prefix"`` — string prefix match (client-side for backends that
                       don't support it natively)
        ``"in"``     — value is a member of a list
        ``"gte"``    — greater-than-or-equal (numeric)
        ``"lte"``    — less-than-or-equal (numeric)
    """

    field: str
    op: str  # "eq" | "ne" | "prefix" | "in" | "gte" | "lte"
    value: Any


@dataclass
class EmbeddedChunk:
    """A content chunk with its pre-computed embedding vector.

    This is the unit of storage for all backends.  The canonical text is
    persisted in the ``ChunkStore`` (SQLite); backends are only responsible
    for the vector and a lightweight metadata payload sufficient for
    filtering.

    Attributes:
        chunk_id:     Stable unique identifier for this chunk.
        document_id:  Identifier of the parent document.
        content:      Raw text content of the chunk.
        content_hash: SHA-256 of the (normalised) content; used by
                      :class:`~rag.chunk_store.ChunkStore` to detect
                      unchanged chunks and skip unnecessary writes.
        embedding:    Dense vector produced by the embedding model.  ``None``
                      before the embedding pipeline has run.
        metadata:     Free-form dict of extra metadata (language, kind, …).
        token_count:  Pre-computed token estimate for budget accounting.
        path:         Filesystem path of the source file.
        start_line:   First line of the chunk in the source file (1-indexed).
        end_line:     Last line of the chunk in the source file (inclusive).
    """

    chunk_id: str
    document_id: str
    content: str
    content_hash: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None


@dataclass
class CollectionSpec:
    """Parameters required to create (or idempotently ensure) a collection."""

    name: str
    dimension: int
    distance: str = "cosine"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchRequest:
    """All parameters needed to execute a single backend search."""

    query: str
    collection: str
    top_k: int = 10
    embedding: list[float] | None = None
    filters: list[MetadataFilter] = field(default_factory=list)
    mode: SearchMode = SearchMode.VECTOR
    purpose: SearchPurpose = SearchPurpose.ASK


@dataclass
class SearchResult:
    """A single ranked result returned from a backend search.

    Fields
    ------
    chunk_id:    Unique identifier of the chunk in the canonical store.
    score:       Similarity/relevance score, higher is better.
    metadata:    Arbitrary per-chunk metadata dict.
    backend:     Which backend produced this result.
    content:     Raw text content (legacy alias for ``text``; prefer ``text``).
    text:        Primary text field used by the packer, deduplication, and
                 pipeline layers.
    document_id: Parent document identifier.
    token_count: Estimated token count for the chunk (used by ContextPacker).
    path:        Source-file path for line-level overlap deduplication.
    start_line:  First line (1-based, inclusive) of this chunk in *path*.
    end_line:    Last line (1-based, inclusive) of this chunk in *path*.
    """

    # --- required (no default) ---
    chunk_id: str
    score: float
    backend: BackendName

    # --- optional ---
    metadata: dict[str, Any] = field(default_factory=dict)
    content: str = ""           # legacy field; new code should populate text
    text: str = ""              # primary text field
    document_id: str = ""
    token_count: int = 0
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None


@dataclass
class BackendStats:
    """Aggregate statistics reported by a backend."""

    backend: BackendName
    collection_count: int
    vector_count: int | None  # None when the backend cannot report this cheaply


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RagBackend(Protocol):
    """Structural interface that every vector-store backend must satisfy.

    Implementations are duck-typed — no inheritance required.  The
    ``@runtime_checkable`` decorator allows ``isinstance(obj, RagBackend)``
    checks in tests and the pipeline layer.
    """

    #: Name used for logging, telemetry, and ``SearchResult.backend``.
    name: BackendName

    def create_collection(self, spec: CollectionSpec) -> None:
        """Create a collection (index) if it does not already exist.

        Must be idempotent: calling it twice with the same ``spec.name``
        must not raise.
        """
        ...

    def upsert(self, chunks: list[EmbeddedChunk], collection: str) -> None:
        """Insert or update chunks in the named collection.

        After this call returns the chunks must be immediately searchable.
        """
        ...

    def search(self, request: SearchRequest) -> list[SearchResult]:
        """Execute a vector (or hybrid) search.

        Returns results sorted by ``score`` descending.  Returns an empty
        list — never raises — when the collection does not exist.
        """
        ...

    def delete_by_filter(
        self, collection: str, filters: list[MetadataFilter]
    ) -> int:
        """Delete all chunks matching *every* filter (AND semantics).

        Returns the number of chunks deleted.
        """
        ...

    def is_available(self) -> bool:
        """Return ``True`` if the backend is reachable and ready to serve queries.

        Must not raise.  Implementations should catch all connection errors and
        return ``False`` instead.  The ``BenchRunner`` (and any other caller
        that wants to skip unavailable backends gracefully) relies on this
        contract.
        """
        ...

    def stats(self) -> BackendStats:
        """Return aggregate statistics for this backend instance."""
        ...
