"""
RAG API router.

Provides span-level retrieval endpoints.

Phase 2 exposed exact-search only.
Phase 4 added symbol search.
Phase 6 adds /retrieve and /context endpoints, and upgrades /search to use
the full HybridRetriever pipeline (exact + symbol + semantic + rerank + budget).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from libs.utils import is_local_uri, uri_to_path
from models.rag import FileSpan, RagContextResponse, RetrievalQuery, RetrievedContext
from pydantic import BaseModel
from rag.engine import get_index
from rag.hybrid_retriever import HybridRetriever
from rag.semantic_search import SemanticRetriever
from rag.symbol_index import search_symbols

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])

# ---------------------------------------------------------------------------
# Module-level singleton — constructed at import time; index is resolved lazily
# ---------------------------------------------------------------------------

_hybrid = HybridRetriever(semantic=SemanticRetriever(get_index))


# ---------------------------------------------------------------------------
# POST /api/v1/rag/search
# ---------------------------------------------------------------------------


@router.post("/search", response_model=list[FileSpan])
async def rag_search(query: RetrievalQuery) -> list[FileSpan]:
    """
    Span-only retrieval via the full hybrid pipeline (no generation).

    Returns ranked FileSpan objects from exact, symbol, and semantic channels
    without building a text context block.

    Args:
        query: Retrieval parameters including the raw query string and
            optional enrichment fields (latest_error, current_file,
            selected_text).

    Returns:
        Ordered list of FileSpan objects.

    Raises:
        HTTPException 400: base_uri is not a local file:// URI.
        HTTPException 404: The directory referenced by base_uri does
            not exist on the filesystem.

    """
    if not is_local_uri(query.base_uri):
        raise HTTPException(
            status_code=400,
            detail="search requires a local base_uri (file://...)",
        )
    base = uri_to_path(query.base_uri)
    if not base.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Directory not found: {base}",
        )
    ctx = _hybrid.retrieve(query)
    return ctx.spans


# ---------------------------------------------------------------------------
# POST /api/v1/rag/symbols
# ---------------------------------------------------------------------------


class SymbolSearchRequest(BaseModel):
    """Request body for the symbol search endpoint."""

    base_uri: str
    q: str
    kinds: list[str] | None = None
    limit: int = 30


@router.post("/symbols")
async def rag_symbols(req: SymbolSearchRequest) -> dict:
    """
    Search the symbol index for definitions matching the query string.

    Performs a LIKE match on symbol_name within the resource identified
    by base_uri.  Optionally filters by one or more symbol_kind values
    (function, class, method, interface, etc.).

    Returns:
        dict with key "results" containing a list of symbol row dicts.

    """
    return {"results": search_symbols(req.base_uri, req.q, req.kinds, req.limit)}


# ---------------------------------------------------------------------------
# POST /api/v1/rag/retrieve
# ---------------------------------------------------------------------------


@router.post("/retrieve", response_model=RetrievedContext)
async def rag_retrieve(query: RetrievalQuery) -> RetrievedContext:
    """
    Full hybrid retrieval — returns spans, citations, and compatible sources.

    Runs all available channels (exact, symbol, semantic, chat-history),
    deduplicates, applies freshness signals, reranks, and trims to the
    mode-appropriate context budget.

    Args:
        query: Retrieval parameters.

    Returns:
        A RetrievedContext containing ranked spans, lightweight citations,
        compat source objects, and a token estimate.

    """
    return _hybrid.retrieve(query)


# ---------------------------------------------------------------------------
# POST /api/v1/rag/context
# ---------------------------------------------------------------------------


@router.post("/context", response_model=RagContextResponse)
async def rag_context(query: RetrievalQuery) -> RagContextResponse:
    """
    Hybrid retrieval that also assembles a packed text context block.

    Calls the hybrid retriever and formats each span into a header + content
    block suitable for direct injection into an LLM prompt.

    Args:
        query: Retrieval parameters.

    Returns:
        A RagContextResponse with the assembled context string, raw spans,
        citations, and token estimate.

    """
    ctx = _hybrid.retrieve(query)

    blocks: list[str] = []
    for s in ctx.spans:
        head = f"--- {s.path or s.uri}"
        if s.start_line and s.end_line:
            head += f":L{s.start_line}-L{s.end_line}"
        head += f"  ({s.reason})"
        blocks.append(f"{head}\n{s.content}")

    return RagContextResponse(
        context="\n\n".join(blocks),
        spans=ctx.spans,
        citations=ctx.citations,
        token_estimate=ctx.token_estimate,
        trace_id=ctx.trace_id or "",
        runtime_plan=None,
    )
