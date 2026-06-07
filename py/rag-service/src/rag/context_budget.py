"""
Context budget utilities for RAG service.

Provides token estimation, per-mode budget tables, and the
:func:`apply_budget` trimming function used by the hybrid retriever
and context-assembly endpoints.

Phase 7 extends the earlier token-estimation module with :data:`BUDGETS`
and :func:`apply_budget`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.rag import FileSpan

# ---------------------------------------------------------------------------
# Per-mode token budgets
# ---------------------------------------------------------------------------

#: Token budgets keyed by :attr:`~models.rag.RetrievalQuery.mode`.
#:
#: Keys per entry:
#:   max_total_tokens — total span-token budget passed to the LLM
#:   max_spans        — maximum number of spans to include
#:   max_doc_tokens   — per-span token cap (advisory; not enforced here)
#:   max_log_tokens   — how many tokens of latest_error to preserve
#:
#: ``max_context_tokens`` and ``top_k`` are backward-compat aliases kept
#: for callers written in earlier phases.
BUDGETS: dict[str, dict[str, int]] = {
    "ask": {
        "max_total_tokens": 6_000,
        "max_context_tokens": 6_000,  # compat alias
        "max_spans": 5,
        "top_k": 5,  # compat alias
        "max_doc_tokens": 2_000,
        "max_log_tokens": 500,
    },
    "search": {
        "max_total_tokens": 8_000,
        "max_context_tokens": 8_000,
        "max_spans": 8,
        "top_k": 8,
        "max_doc_tokens": 3_000,
        "max_log_tokens": 500,
    },
    "edit-small": {
        "max_total_tokens": 10_000,
        "max_context_tokens": 10_000,
        "max_spans": 6,
        "top_k": 6,
        "max_doc_tokens": 2_000,
        "max_log_tokens": 1_000,
    },
    "test-fix": {
        "max_total_tokens": 12_000,
        "max_context_tokens": 12_000,
        "max_spans": 8,
        "top_k": 8,
        "max_doc_tokens": 1_500,
        "max_log_tokens": 2_000,
    },
    "refactor": {
        "max_total_tokens": 20_000,
        "max_context_tokens": 12_000,
        "max_spans": 16,
        "top_k": 10,
        "max_doc_tokens": 3_000,
        "max_log_tokens": 1_000,
    },
}


def estimate_tokens(text: str) -> int:
    """
    Estimate token count using a cheap, provider-agnostic heuristic.

    Uses ~4 chars per token, which is a reasonable approximation for
    English/code mixed content without requiring a tokeniser dependency.

    Args:
        text: The text to estimate token count for.

    Returns:
        Estimated number of tokens (minimum 1 for non-empty text).

    """
    if not text:
        return 0
    # Cheap, provider-agnostic. ~4 chars per token on English/code mix.
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Budget application
# ---------------------------------------------------------------------------


def apply_budget(
    spans: list[FileSpan],
    mode: str,
    override_total: int | None = None,
    hardware_cap: object = None,
) -> tuple[list[FileSpan], list[FileSpan]]:
    """
    Trim *spans* to fit within the token budget for *mode*.

    Spans must be pre-sorted by score descending (best first) — the
    function keeps spans greedily until the budget is exhausted.

    Args:
        spans: Pre-sorted list of :class:`~models.rag.FileSpan` candidates.
        mode: Retrieval mode string (one of the keys in :data:`BUDGETS`);
            falls back to ``"ask"`` if not recognised.
        override_total: Optional caller-supplied total-token cap that
            overrides the mode default.
        hardware_cap: Optional hardware-aware budget object that applies
            additional hardware-derived constraints (populated in Phase 12;
            pass ``None`` for now).

    Returns:
        A tuple ``(kept, dropped)`` where *kept* fits within the budget and
        *dropped* contains the spans that were excluded.

    """
    cfg = BUDGETS.get(mode, BUDGETS["ask"])
    max_total = override_total or cfg["max_total_tokens"]
    max_spans = cfg["max_spans"]

    # Apply hardware cap if provided (Phase 12)
    if hardware_cap is not None:
        max_total = min(max_total, hardware_cap.max_retrieved_tokens)
        max_spans = min(max_spans, hardware_cap.max_spans)

    kept: list[FileSpan] = []
    dropped: list[FileSpan] = []
    running = 0

    for s in spans:
        if len(kept) >= max_spans or running + s.token_estimate > max_total:
            dropped.append(s)
        else:
            kept.append(s)
            running += s.token_estimate

    return kept, dropped

