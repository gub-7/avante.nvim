"""ChromaDB + LlamaIndex semantic search initialisation and retrieval."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
from libs.configs import BASE_DATA_DIR, CHROMA_PERSIST_DIR
from libs.logger import logger
from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from providers.factory import initialize_embed_model, initialize_llm_model

if TYPE_CHECKING:
    from llama_index.core import VectorStoreIndex as VectorStoreIndexType

_config_file: Path = BASE_DATA_DIR / "rag_config.json"


def init_semantic_search() -> tuple[VectorStoreIndexType, ChromaVectorStore, StorageContext]:  # noqa: PLR0915
    """
    Initialise Chroma + LlamaIndex and return (index, vector_store, storage_context).

    Detects provider / embed-model changes and resets the collection when they occur.
    """
    # -----------------------------------------------------------------------
    # Provider / model config
    # -----------------------------------------------------------------------
    rag_embed_provider = os.getenv("RAG_EMBED_PROVIDER", "openai")
    rag_embed_endpoint = os.getenv("RAG_EMBED_ENDPOINT", "https://api.openai.com/v1")
    rag_embed_model = os.getenv("RAG_EMBED_MODEL", "text-embedding-3-large")
    rag_embed_api_key = os.getenv("RAG_EMBED_API_KEY", None)
    rag_embed_extra = os.getenv("RAG_EMBED_EXTRA", None)

    rag_llm_provider = os.getenv("RAG_LLM_PROVIDER", "openai")
    rag_llm_endpoint = os.getenv("RAG_LLM_ENDPOINT", "https://api.openai.com/v1")
    rag_llm_model = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")
    rag_llm_api_key = os.getenv("RAG_LLM_API_KEY", None)
    rag_llm_extra = os.getenv("RAG_LLM_EXTRA", None)

    # -----------------------------------------------------------------------
    # ChromaDB client + config-change detection
    # -----------------------------------------------------------------------
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

    if _config_file.exists():
        with Path.open(_config_file, "r") as f:
            prev_config = json.load(f)
            if (
                prev_config.get("provider") != rag_embed_provider
                or prev_config.get("embed_model") != rag_embed_model
            ):
                logger.info("Detected config change, clearing existing data...")
                chroma_client.reset()

    # Save current config
    with Path.open(_config_file, "w") as f:
        json.dump({"provider": rag_embed_provider, "embed_model": rag_embed_model}, f)

    chroma_collection = chroma_client.get_or_create_collection("documents")  # pyright: ignore
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # -----------------------------------------------------------------------
    # Decode JSON extras
    # -----------------------------------------------------------------------
    try:
        embed_extra = json.loads(rag_embed_extra) if rag_embed_extra is not None else {}
    except json.JSONDecodeError:
        logger.error("Failed to decode RAG_EMBED_EXTRA, defaulting to empty dict.")
        embed_extra = {}

    try:
        llm_extra = json.loads(rag_llm_extra) if rag_llm_extra is not None else {}
    except json.JSONDecodeError:
        logger.error("Failed to decode RAG_LLM_EXTRA, defaulting to empty dict.")
        llm_extra = {}

    # -----------------------------------------------------------------------
    # Embedding model
    # -----------------------------------------------------------------------
    try:
        embed_model = initialize_embed_model(
            embed_provider=rag_embed_provider,
            embed_model=rag_embed_model,
            embed_endpoint=rag_embed_endpoint,
            embed_api_key=rag_embed_api_key,
            embed_extra=embed_extra,
        )
        logger.info("Embedding model initialized successfully.")
    except (ValueError, RuntimeError) as e:
        error_msg = f"Failed to initialize embedding model: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e

    # -----------------------------------------------------------------------
    # LLM model
    # -----------------------------------------------------------------------
    try:
        llm_model = initialize_llm_model(
            llm_provider=rag_llm_provider,
            llm_model=rag_llm_model,
            llm_endpoint=rag_llm_endpoint,
            llm_api_key=rag_llm_api_key,
            llm_extra=llm_extra,
        )
        logger.info("LLM model initialized successfully.")
    except (ValueError, RuntimeError) as e:
        error_msg = f"Failed to initialize LLM model: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e

    Settings.embed_model = embed_model
    Settings.llm = llm_model

    # -----------------------------------------------------------------------
    # Vector index
    # -----------------------------------------------------------------------
    try:
        index = load_index_from_storage(storage_context)
    except (OSError, ValueError) as e:
        logger.error("Failed to load index from storage: %s", e)
        index = VectorStoreIndex([], storage_context=storage_context)

    return index, vector_store, storage_context


# ---------------------------------------------------------------------------
# SemanticRetriever — wraps the vector index as a FileSpan producer
# ---------------------------------------------------------------------------


class SemanticRetriever:
    """Retrieves FileSpan objects via LlamaIndex vector similarity search."""

    def __init__(self, index_provider: object) -> None:
        """Initialise with a callable that returns the active VectorStoreIndex."""
        # Callable returning the current VectorStoreIndex (e.g. engine.get_index)
        self._get_index = index_provider

    def retrieve(self, query: object) -> list:  # list[FileSpan]
        """
        Run semantic vector search and return matching spans.

        Args:
            query: A :class:`~models.rag.RetrievalQuery` instance.

        Returns:
            List of :class:`~models.rag.FileSpan` objects tagged with
            ``retrieval_sources=["semantic"]``.

        """
        # Lazy import to avoid circular deps at module load time
        from libs.utils import get_node_uri  # noqa: PLC0415
        from models.rag import FileSpan  # noqa: PLC0415

        from rag.context_budget import estimate_tokens  # noqa: PLC0415

        try:
            index = self._get_index()
        except RuntimeError:
            logger.warning("SemanticRetriever: index not available, returning empty list")
            return []

        try:
            engine = index.as_query_engine(similarity_top_k=max(query.top_k * 3, 15))
            result = engine.query(query.query)
        except Exception as exc:
            logger.warning("SemanticRetriever query failed: %s", exc)
            return []

        spans: list[FileSpan] = []
        for node in result.source_nodes or []:
            uri = get_node_uri(node.node) or ""
            content = str(node.node.get_content())
            md: dict = getattr(node.node, "metadata", {}) or {}
            spans.append(
                FileSpan(
                    uri=uri,
                    path=uri.removeprefix("file://") if uri.startswith("file://") else None,
                    start_line=md.get("start_line"),
                    end_line=md.get("end_line"),
                    content=content,
                    reason="semantic:vector",
                    score=float(node.score or 0.0),
                    token_estimate=estimate_tokens(content),
                    hash=hashlib.sha256(content.encode()).hexdigest(),
                    retrieval_sources=["semantic"],
                    chunk_kind=md.get("chunk_kind"),
                    language=md.get("language"),
                ),
            )
        return spans

