"""Per-resource chat-history semantic index.

Each code resource gets its own ChromaDB collection named
``chat_<sha1(resource_uri)[:16]>``.  Chat messages are sanitized, embedded,
and stored so they can be retrieved as :class:`~models.rag.FileSpan` objects.

Retrieval is gated behind two heuristics:
- Query mode is ``"ask"`` (conversational), OR
- Query contains a deictic expression (e.g. "we decided", "earlier you said").
"""

from __future__ import annotations

import hashlib
import re

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore

from libs.configs import CHROMA_PERSIST_DIR
from libs.logger import logger
from models.rag import FileSpan, RetrievalQuery
from rag.context_budget import estimate_tokens
from services.chat_history import sanitize

# Regex to detect conversational back-references that warrant chat lookup
_DEICTIC = re.compile(
    r"\b(we|earlier|before|previous(?:ly)?|you said|last time|recall|remember)\b",
    re.I,
)


def _collection_for(resource_uri: str) -> str:
    """Return the Chroma collection name for the given resource URI."""
    return "chat_" + hashlib.sha1(resource_uri.encode()).hexdigest()[:16]


class ChatHistoryIndex:
    """Semantic index over per-resource chat history.

    Maintains one Chroma collection per code resource.  Provides
    :meth:`upsert`, :meth:`purge`, and :meth:`retrieve` operations.
    """

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

    def _index_for(self, resource_uri: str) -> VectorStoreIndex:
        """Return (or lazily create) the LlamaIndex index for a resource."""
        coll = self._client.get_or_create_collection(_collection_for(resource_uri))
        vs = ChromaVectorStore(chroma_collection=coll)
        sc = StorageContext.from_defaults(vector_store=vs)
        try:
            return VectorStoreIndex.from_vector_store(vs, storage_context=sc)
        except Exception:  # noqa: BLE001
            return VectorStoreIndex([], storage_context=sc)

    def upsert(self, resource_uri: str, chat_id: str, messages: list[dict]) -> None:
        """Embed and store messages for a single chat turn.

        Existing nodes for the same ``chat_id`` are replaced via
        ``refresh_ref_docs``.

        Args:
            resource_uri: Base URI of the owning code resource.
            chat_id: Unique identifier for the chat turn.
            messages: List of ``{"role": str, "content": str}`` dicts.
        """
        idx = self._index_for(resource_uri)
        docs = []
        for i, m in enumerate(messages):
            text = sanitize(m["content"])
            docs.append(
                Document(
                    text=text,
                    doc_id=f"{chat_id}#{i}",
                    metadata={
                        "resource_uri": resource_uri,
                        "chat_id": chat_id,
                        "message_idx": i,
                        "role": m["role"],
                    },
                )
            )
        try:
            idx.refresh_ref_docs(docs)
        except Exception as e:  # noqa: BLE001
            logger.warning("chat_history upsert failed: %s", e)

    def purge(self, resource_uri: str) -> None:
        """Delete the entire Chroma collection for a resource.

        Args:
            resource_uri: Base URI of the resource whose history to erase.
        """
        try:
            self._client.delete_collection(_collection_for(resource_uri))
        except Exception:  # noqa: BLE001
            pass

    def retrieve(self, query: RetrievalQuery) -> list[FileSpan]:
        """Return chat-history spans relevant to the query.

        Returns an empty list unless at least one of these conditions holds:

        1. ``query.mode == "ask"`` (conversational context is generally useful).
        2. The query contains a deictic expression referencing prior context.

        The budget allocation is ``max(1, top_k // 4)`` chat spans to avoid
        dominating the overall context window.

        Args:
            query: Retrieval parameters.

        Returns:
            List of :class:`~models.rag.FileSpan` objects tagged
            ``retrieval_sources=["chat_history"]``.
        """
        if not query.include_chat_history:
            return []

        cond_conv = query.mode == "ask"
        cond_deictic = bool(_DEICTIC.search(query.query))
        if not (cond_conv or cond_deictic):
            return []

        idx = self._index_for(query.base_uri)
        try:
            engine = idx.as_query_engine(similarity_top_k=max(query.top_k, 6))
            result = engine.query(query.query)
        except Exception as e:  # noqa: BLE001
            logger.warning("chat_history retrieve failed: %s", e)
            return []

        spans: list[FileSpan] = []
        cap = max(1, query.top_k // 4)
        for node in (result.source_nodes or [])[:cap]:
            content = str(node.node.get_content())
            md = getattr(node.node, "metadata", {}) or {}
            spans.append(
                FileSpan(
                    uri=f"chat://{md.get('chat_id', '?')}#{md.get('message_idx', 0)}",
                    path=None,
                    start_line=None,
                    end_line=None,
                    content=content,
                    reason=f"chat:{md.get('role', 'msg')}",
                    score=1.0,
                    token_estimate=estimate_tokens(content),
                    hash=hashlib.sha256(content.encode()).hexdigest(),
                    retrieval_sources=["chat_history"],
                    chunk_kind="chat",
                )
            )
        return spans

