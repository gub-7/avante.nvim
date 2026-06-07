"""RAG backend abstractions and shared data models.

Public re-exports for the ``rag.backends`` package.  Import from here rather
than from sub-modules to keep import paths stable if files are reorganised.
"""

from rag.backends.base import (
    BackendName,
    BackendStats,
    CollectionSpec,
    EmbeddedChunk,
    MetadataFilter,
    RagBackend,
    SearchMode,
    SearchPurpose,
    SearchRequest,
    SearchResult,
)
from rag.backends.chroma import ChromaBackend
from rag.backends.exact import ExactBackend
from rag.backends.in_memory import InMemoryBackend

__all__ = [
    "BackendName",
    "BackendStats",
    "ChromaBackend",
    "CollectionSpec",
    "EmbeddedChunk",
    "ExactBackend",
    "InMemoryBackend",
    "MetadataFilter",
    "RagBackend",
    "SearchMode",
    "SearchPurpose",
    "SearchRequest",
    "SearchResult",
]
