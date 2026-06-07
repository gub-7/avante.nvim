"""
Hybrid retriever: orchestrates exact, symbol, semantic, and chat-history channels.

Phase 6 implementation.  Combines results from all available retrieval
channels, deduplicates, reranks, and applies freshness signals and a
token-based context budget before returning the final :class:`RetrievedContext`.

Phase 8 adds JSONL tracing via :func:`~observability.trace.start_trace`.
Phase 9 adds depth-1 import expansion via :func:`~rag.expansion.expand`.
"""

from __future__ import annotations

import hashlib
import re

from libs.logger import logger
from libs.utils import is_local_uri, uri_to_path
from models.rag import (
    ContextCitation,
    FileSpan,
    RetrievalQuery,
    RetrievedContext,
    SourceDocumentCompat,
)

from rag.context_budget import apply_budget, estimate_tokens
from rag.dedupe import dedupe_and_merge
from rag.exact_search import ExactSearch
from rag.expansion import expand
from rag.freshness import compute_freshness
from rag.reranker import rerank
from rag.symbol_index import search_symbols

# ---------------------------------------------------------------------------
# SymbolRetriever
# ---------------------------------------------------------------------------


class SymbolRetriever:
    """Retrieve FileSpan objects by looking up symbol names in the symbol index."""

    def retrieve(self, query: RetrievalQuery) -> list[FileSpan]:
        """
        Run symbol-index lookup and materialise spans by reading file content.

        Extracts candidate identifier tokens from the query, selected text,
        and latest error; queries the symbol DB for each token; then reads
        the matching line ranges from disk.

        Args:
            query: The originating retrieval query.

        Returns:
            List of :class:`~models.rag.FileSpan` objects tagged with
            ``retrieval_sources=["symbol"]``.

        """
        if not is_local_uri(query.base_uri):
            return []

        # Collect candidate identifier tokens from all text fields
        terms: set[str] = set()
        for src in (
            query.query,
            query.selected_text or "",
            query.latest_error or "",
        ):
            for m in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", src):
                terms.add(m.group(0))

        spans: list[FileSpan] = []
        for term in list(terms)[:20]:
            spans.extend(self._spans_for_term(query.base_uri, term))
        return spans

    def _spans_for_term(self, base_uri: str, term: str) -> list[FileSpan]:
        """Return FileSpan objects for a single symbol *term* from the index."""
        spans: list[FileSpan] = []
        for row in search_symbols(base_uri, term, limit=8):
            file_uri = row.get("file_uri") or row.get("uri", "")
            if not file_uri:
                continue
            try:
                file_path = uri_to_path(file_uri)
            except Exception as exc:
                logger.debug("SymbolRetriever: invalid file_uri %r: %s", file_uri, exc)
                continue
            if not file_path.exists():
                continue
            try:
                lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError as exc:
                logger.debug("SymbolRetriever: cannot read %s: %s", file_path, exc)
                continue

            start_line: int | None = row.get("start_line")
            end_line: int | None = row.get("end_line")
            s = max(0, (start_line or 1) - 1)
            e = min(len(lines), (end_line or start_line or 1))
            content = "\n".join(lines[s:e])
            if not content:
                continue

            symbol_kind = row.get("symbol_kind", "function")
            symbol_name = row.get("symbol_name", term)
            language = row.get("language")

            spans.append(
                FileSpan(
                    uri=file_uri,
                    path=str(file_path),
                    start_line=start_line,
                    end_line=end_line,
                    content=content,
                    reason=f"symbol:{symbol_kind}:{symbol_name}",
                    score=3.5,
                    token_estimate=estimate_tokens(content),
                    hash=hashlib.sha256(content.encode()).hexdigest(),
                    retrieval_sources=["symbol"],
                    chunk_kind=symbol_kind,
                    language=language,
                ),
            )
        return spans


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """
    Orchestrate all retrieval channels into a single ranked context.

    Channels (in order of evaluation):
    1. Exact search (ripgrep/Python fallback)
    2. Symbol index lookup
    3. Semantic vector search
    4. Chat-history retrieval (optional; injected via constructor)

    After gathering spans the retriever:
    * Deduplicates and merges overlapping spans
    * Applies freshness signals (stale penalty / recent-edit boost)
    * Reranks using the multi-signal reranker
    * Applies the token-based context budget for the requested mode

    Args:
        semantic: A :class:`~rag.semantic_search.SemanticRetriever` instance.
        chat_history: Optional retriever that understands chat history; must
            expose a ``retrieve(query) -> list[FileSpan]`` method.

    """

    def __init__(self, semantic: object, chat_history: object = None) -> None:
        """
        Initialise retrieval channels.

        Args:
            semantic: A :class:`~rag.semantic_search.SemanticRetriever` instance.
            chat_history: Optional chat-history retriever with a
                ``retrieve(query)`` method.

        """
        self._semantic = semantic
        self._chat = chat_history
        self._exact = ExactSearch()
        self._symbol = SymbolRetriever()

    def retrieve(self, query: RetrievalQuery) -> RetrievedContext:  # noqa: C901, PLR0915
        """
        Run all channels, dedupe, rerank, expand, budget, and return context.

        Phase 8: every call emits a JSONL trace line via
        :func:`~observability.trace.start_trace`.

        Phase 9: depth-1 import expansion is performed after reranking to
        surface imported symbol definitions as additional spans.

        Args:
            query: Retrieval parameters.

        Returns:
            A :class:`~models.rag.RetrievedContext` containing spans,
            citations, compatible sources, token estimate, and trace_id.

        """
        from observability.trace import start_trace  # noqa: PLC0415

        with start_trace(query.query, query.mode, query.base_uri) as tr:
            spans: list[FileSpan] = []

            # --- Exact + symbol search (local only) ---
            if is_local_uri(query.base_uri):
                try:
                    base = uri_to_path(query.base_uri)
                    if base.exists():
                        spans.extend(self._exact.retrieve(query, base))
                        spans.extend(self._symbol.retrieve(query))
                except Exception as exc:
                    logger.warning("exact/symbol retrieval failed: %s", exc)

            # --- Semantic vector search ---
            try:
                spans.extend(self._semantic.retrieve(query))
            except Exception as exc:
                logger.warning("semantic retrieval failed: %s", exc)

            # --- Chat-history retrieval ---
            if self._chat and query.include_chat_history:
                try:
                    spans.extend(self._chat.retrieve(query))
                except Exception as exc:
                    logger.warning("chat_history retrieval failed: %s", exc)

            # --- Deduplicate and merge overlapping spans ---
            deduped, saved = dedupe_and_merge(spans)

            # --- Freshness signals ---
            stale: set[str] = set()
            recent: set[str] = set()
            if is_local_uri(query.base_uri):
                try:
                    stale, recent = compute_freshness(uri_to_path(query.base_uri))
                except Exception as exc:
                    logger.warning("freshness computation failed: %s", exc)
            if query.include_stale:
                # Caller opted in to stale results — clear the stale penalty set
                stale = set()

            # --- Rerank ---
            scored = rerank(deduped, query, stale_uris=stale, recent_uris=recent)
            ordered = [s for s, _ in scored]

            # --- Phase 9: depth-1 import expansion ---
            try:
                extra_spans = expand(ordered, query)
            except Exception as exc:
                logger.warning("expansion failed: %s", exc)
                extra_spans = []

            if extra_spans:
                # Merge expanded spans, re-sort by existing score descending
                # (expanded spans carry score=2.0, lower than ranked spans)
                combined = ordered + extra_spans
                scored = rerank(combined, query, stale_uris=stale, recent_uris=recent)
                ordered = [s for s, _ in scored]

            # --- Context budget ---
            kept, dropped = apply_budget(
                ordered,
                query.mode,
                override_total=query.max_context_tokens,
                hardware_cap=None,  # populated in Phase 12
            )

            token_estimate = sum(s.token_estimate for s in kept)

            # --- Phase 8: populate trace fields ---
            tr.retrieved_spans_count = len(deduped)
            tr.inserted_spans_count = len(kept)
            tr.dropped_spans_count = len(dropped)
            tr.retrieved_tokens = sum(s.token_estimate for s in ordered)
            tr.inserted_tokens = token_estimate
            tr.deduped_tokens_saved = saved
            tr.context_budget_used = token_estimate
            tr.freshness_stale_count = len(stale)
            tr.freshness_recent_count = len(recent)
            tr.rerank_scores = [sc.model_dump() for _, sc in scored[:20]]
            tr.expanded_spans_count = len(extra_spans)

            citations = [
                ContextCitation(
                    uri=s.uri,
                    path=s.path,
                    start_line=s.start_line,
                    end_line=s.end_line,
                    reason=s.reason,
                    retrieval_sources=s.retrieval_sources,
                )
                for s in kept
            ]

            sources_compat = [
                SourceDocumentCompat(uri=s.uri, content=s.content, score=s.score)
                for s in kept
            ]

            return RetrievedContext(
                spans=kept,
                sources=sources_compat,
                citations=citations,
                token_estimate=token_estimate,
                trace_id=tr.trace_id,
                response=None,
            )

