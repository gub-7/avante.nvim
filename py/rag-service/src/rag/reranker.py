"""Multi-signal reranker for retrieved :class:`~models.rag.FileSpan` objects."""

from __future__ import annotations

from pathlib import Path

from models.rag import FileSpan, RerankScore, RetrievalQuery

# ---------------------------------------------------------------------------
# Scoring weights (additive)
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "exact": 4.0,
    "symbol": 3.5,
    "semantic": 1.5,
    "chat_history": 1.0,
    "metadata": 1.0,
    "recent": 0.75,
    "test": 1.25,
}

# Larger spans cost more context budget — penalise proportionally
TOKEN_PENALTY_PER_1K: float = 0.8

# Extra penalty applied when a span's file is known to be stale
STALE_PENALTY: float = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _channel_score(span: FileSpan) -> tuple[float, float, float, float]:
    """Return per-channel additive scores (exact, symbol, semantic, chat_history)."""
    e = WEIGHTS["exact"] if "exact" in span.retrieval_sources else 0.0
    s = WEIGHTS["symbol"] if "symbol" in span.retrieval_sources else 0.0
    v = WEIGHTS["semantic"] if "semantic" in span.retrieval_sources else 0.0
    c = WEIGHTS["chat_history"] if "chat_history" in span.retrieval_sources else 0.0
    return e, s, v, c


def _proximity(span: FileSpan, current_file: str | None) -> float:
    """Return a [0, 1] score reflecting how close *span* is to *current_file*.

    Proximity is approximated by counting shared path components.
    """
    if not current_file or not span.path:
        return 0.0
    cur = Path(current_file).resolve()
    target = Path(span.path).resolve()
    try:
        common = len(set(cur.parts) & set(target.parts))
        return min(1.0, common * 0.1)
    except OSError:
        return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rerank(
    spans: list[FileSpan],
    query: RetrievalQuery,
    stale_uris: set[str] | None = None,
    recent_uris: set[str] | None = None,
) -> list[tuple[FileSpan, RerankScore]]:
    """Compute a composite score for each span and return them sorted best-first.

    The score is the algebraic sum of several additive bonuses and penalties:

    * Channel bonuses: ``exact`` > ``symbol`` > ``semantic`` > ``chat_history``
    * ``proximity`` — shared path components with the currently edited file
    * ``recent_edit`` — the span's file was recently modified
    * ``test_relevance`` — the span is from a test file and the query mode is
      ``test-fix``, or the span has ``chunk_kind == "test"``
    * ``token_penalty`` — larger spans are penalised to favour concise context
    * ``stale_penalty`` — the span's file has a newer version not yet indexed

    Args:
        spans: Candidate spans from one or more retrieval channels.
        query: The originating retrieval query (provides ``current_file`` and
            ``mode``).
        stale_uris: Set of file URIs that are known to be out-of-date.
        recent_uris: Set of file URIs that were recently modified/created.

    Returns:
        List of ``(FileSpan, RerankScore)`` pairs ordered by ``RerankScore.final``
        descending.  The ``span.score`` field is also updated in-place.
    """
    stale_uris = stale_uris or set()
    recent_uris = recent_uris or set()

    out: list[tuple[FileSpan, RerankScore]] = []

    for s in spans:
        e, sy, sem, ch = _channel_score(s)
        prox = _proximity(s, query.current_file)
        recent = WEIGHTS["recent"] if s.uri in recent_uris else 0.0
        test_bonus = (
            WEIGHTS["test"]
            if (s.chunk_kind == "test" or query.mode == "test-fix")
            else 0.0
        )
        token_pen = TOKEN_PENALTY_PER_1K * (s.token_estimate / 1000.0)
        stale_pen = STALE_PENALTY if s.uri in stale_uris else 0.0

        final = e + sy + sem + ch + prox + recent + test_bonus - token_pen - stale_pen

        score = RerankScore(
            final=final,
            exact=e,
            symbol=sy,
            semantic=sem,
            chat_history=ch,
            proximity=prox,
            recent_edit=recent,
            test_relevance=test_bonus,
            token_penalty=token_pen,
            stale_penalty=stale_pen,
        )
        s.score = final
        out.append((s, score))

    out.sort(key=lambda x: x[1].final, reverse=True)
    return out

