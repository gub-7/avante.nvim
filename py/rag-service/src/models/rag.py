from typing import Literal

from pydantic import BaseModel, Field


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
    query: str
    base_uri: str
    top_k: int = 5
    mode: Literal["ask", "search", "edit-small", "test-fix", "refactor"] = "ask"
    current_file: str | None = None
    selected_text: str | None = None
    latest_error: str | None = None
    changed_files: list[str] = []
    include_full_files: bool = False
    include_chat_history: bool = True
    include_stale: bool = False
    max_context_tokens: int | None = None


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
    runtime_plan: "BackendRecommendation | None" = None  # forward ref; resolve in main app


# Resolve forward reference to BackendRecommendation (defined in models.runtime)
# This is intentionally a string annotation to avoid circular imports.
# Callers should use model_rebuild() after importing both modules.

