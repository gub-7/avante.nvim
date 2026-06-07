from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, model_validator

# ---------------------------------------------------------------------------
# Enums — new in Increment 12
# ---------------------------------------------------------------------------


class SearchMode(StrEnum):
    """
    Routing / retrieval strategy for the adaptive RAG router.

    ``auto``     — router decides (default).
    ``exact``    — force exact / keyword search backend.
    ``semantic`` — force semantic / vector backend.
    ``hybrid``   — force full hybrid pipeline (exact + semantic + symbol).
    """

    auto = "auto"
    exact = "exact"
    semantic = "semantic"
    hybrid = "hybrid"


class SearchPurpose(StrEnum):
    """
    Caller intent; used by the router as a soft hint for backend selection.

    ``agentic``  — called by an autonomous agent sub-query loop.
    ``context``  — building LLM prompt context.
    ``search``   — pure search / exploration (no generation).
    """

    agentic = "agentic"
    context = "context"
    search = "search"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Legacy workflow-mode values accepted by the old ``mode`` field.
_LEGACY_WORKFLOW_MODES: frozenset[str] = frozenset(
    {"ask", "search", "edit-small", "test-fix", "refactor"},
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class FileSpan(BaseModel):
    uri: str
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    content: str
    reason: str
    score: float
    token_estimate: int
    hash: str
    retrieval_sources: list[str] = []  # exact | symbol | semantic | chat_history | expansion
    chunk_kind: str | None = None  # function | class | section | test | config | code
    language: str | None = None


class RetrievalQuery(BaseModel):
    """
    Query parameters for retrieval endpoints.

    Migration note (Increment 12)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The previous ``mode`` field (workflow intent: ask / search / edit-small …)
    has been renamed to ``workflow_mode``.  For one release the old JSON key
    ``"mode"`` is still accepted when its value is a legacy workflow-mode
    string.  The validator silently promotes it:

        {"mode": "ask"}  →  workflow_mode="ask", mode=SearchMode.auto
    """

    query: str
    base_uri: str
    top_k: int = 5

    # Renamed from ``mode`` in Increment 12.
    workflow_mode: Literal["ask", "search", "edit-small", "test-fix", "refactor"] = "ask"

    # New Increment-12 fields --------------------------------------------------
    mode: SearchMode = SearchMode.auto
    """Routing / retrieval strategy.  Defaults to ``auto`` (router decides)."""

    purpose: SearchPurpose | None = None
    """Optional caller-intent hint for the router."""

    shadow: bool = False
    """When True the router runs a shadow backend in parallel but returns only
    the primary result.  Shadow telemetry is still recorded."""

    request_id: str | None = None
    """Client-supplied idempotency / tracing key.  The route generates a
    ``uuid4`` if absent."""

    parent_request_id: str | None = None
    """For agent sub-queries: the ``request_id`` of the parent call."""

    # Existing optional fields -------------------------------------------------
    current_file: str | None = None
    selected_text: str | None = None
    latest_error: str | None = None
    changed_files: list[str] = []
    include_full_files: bool = False
    include_chat_history: bool = True
    include_stale: bool = False
    max_context_tokens: int | None = None

    # -------------------------------------------------------------------------
    # Backward-compat validator
    # -------------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _compat_legacy_mode(cls, data: object) -> object:
        """
        Promote the old ``mode`` key to ``workflow_mode`` when its value is a
        legacy workflow-mode string, so existing callers keep working.
        """
        if not isinstance(data, dict):
            return data
        raw_mode = data.get("mode")
        if raw_mode in _LEGACY_WORKFLOW_MODES:
            data = dict(data)  # shallow copy — don't mutate caller's dict
            data["workflow_mode"] = raw_mode
            del data["mode"]  # let ``mode`` fall back to SearchMode.auto default
        return data


class ContextCitation(BaseModel):
    uri: str
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    reason: str
    retrieval_sources: list[str]


class ContextSufficiency(BaseModel):
    sufficient: bool
    confidence: float
    missing: list[str] = []
    suggested_retrievals: list[str] = []


class SourceDocumentCompat(BaseModel):
    """Mirror of api/retrieve.py SourceDocument; re-exported for new endpoints."""

    uri: str
    content: str
    score: float | None = None


class RetrievedContext(BaseModel):
    response: str | None = None
    spans: list[FileSpan]
    sources: list[SourceDocumentCompat]
    citations: list[ContextCitation]
    token_estimate: int
    trace_id: str | None = None
    sufficiency: ContextSufficiency | None = None
    # New in Increment 12 — set by the route handler (uuid4 if not supplied by client).
    request_id: str | None = None


class RerankScore(BaseModel):
    final: float
    exact: float = 0
    symbol: float = 0
    semantic: float = 0
    chat_history: float = 0
    proximity: float = 0
    import_distance: float = 0
    recent_edit: float = 0
    test_relevance: float = 0
    token_penalty: float = 0
    stale_penalty: float = 0


class RagContextResponse(BaseModel):
    context: str
    spans: list[FileSpan]
    citations: list[ContextCitation]
    token_estimate: int
    trace_id: str
    runtime_plan: BackendRecommendation | None = None  # forward ref; resolve in main app


# Resolve forward reference to BackendRecommendation (defined in models.runtime)
# This is intentionally a string annotation to avoid circular imports.
# Callers should use model_rebuild() after importing both modules.

