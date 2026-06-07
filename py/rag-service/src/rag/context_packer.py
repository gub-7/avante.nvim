"""Context deduplication and budget-aware packing for the RAG retrieval pipeline.

``ContextPacker`` is the final stage before context is handed to the LLM:

1. Deduplicate results by ``chunk_id`` (keep highest score).
2. Deduplicate/merge results with overlapping ``(path, start_line, end_line)``
   spans (keep highest score, widen the range).
3. Sort survivors by score descending.
4. Greedily fill the token budget, preferring higher-score chunks.
5. Return a :class:`PackingResult` with rich telemetry metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.backends.base import SearchResult
from rag.dedupe import dedupe_search_results


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PackingResult:
    """Outcome of a :meth:`ContextPacker.pack` call, including telemetry fields.

    Attributes:
        chunks:          The final packed list of results that fit within the
                         token budget, ordered by score descending.
        raw_tokens:      Total ``token_count`` of all input results **before**
                         any deduplication.
        deduped_tokens:  Total ``token_count`` after deduplication, but
                         **before** the token-budget cut.
        packed_tokens:   Total ``token_count`` of the final packed output
                         (≤ *budget*).
        tokens_saved:    ``raw_tokens - packed_tokens`` — absolute saving.
        tokens_saved_pct: ``tokens_saved / raw_tokens`` as a fraction in
                          ``[0, 1]``; ``0.0`` when *raw_tokens* is zero.
        duplicate_rate:  Fraction of *input* results that were discarded by
                         deduplication (not by the budget cut); in ``[0, 1]``.
    """

    chunks: list[SearchResult] = field(default_factory=list)
    raw_tokens: int = 0
    deduped_tokens: int = 0
    packed_tokens: int = 0
    tokens_saved: int = 0
    tokens_saved_pct: float = 0.0
    duplicate_rate: float = 0.0


# ---------------------------------------------------------------------------
# ContextPacker
# ---------------------------------------------------------------------------


class ContextPacker:
    """Deduplicate and budget-pack a list of backend search results.

    Usage::

        packer = ContextPacker()
        result = packer.pack(search_results, budget=4096)
        for chunk in result.chunks:
            # chunk.backend, chunk.chunk_id, chunk.score are always present
            ...
    """

    def pack(self, results: list[SearchResult], budget: int) -> PackingResult:
        """Deduplicate *results* and greedily fill *budget* tokens.

        Args:
            results: Raw search results from one or more backend calls.
            budget:  Token budget (inclusive upper bound).  A budget of ``0``
                     returns an empty chunk list without raising.

        Returns:
            A :class:`PackingResult` instance with the packed chunks and
            telemetry metrics populated.
        """
        n_input = len(results)

        # 1. Compute raw token total before any deduplication.
        raw_tokens = sum(r.token_count for r in results)

        # 2. Deduplicate (by chunk_id then by overlapping file span).
        deduped, _tokens_dropped_by_dedup = dedupe_search_results(results)

        deduped_tokens = sum(r.token_count for r in deduped)
        n_deduped = len(deduped)

        # 3. Sort by score descending so greedy fill favours higher-quality chunks.
        deduped.sort(key=lambda r: r.score, reverse=True)

        # 4. Greedy budget fill.
        packed: list[SearchResult] = []
        packed_tokens = 0

        for chunk in deduped:
            if packed_tokens + chunk.token_count <= budget:
                packed.append(chunk)
                packed_tokens += chunk.token_count

        # 5. Compute telemetry metrics.
        tokens_saved = raw_tokens - packed_tokens
        tokens_saved_pct = tokens_saved / raw_tokens if raw_tokens > 0 else 0.0

        # duplicate_rate: fraction of *input* results removed by deduplication.
        duplicates_removed = n_input - n_deduped
        duplicate_rate = duplicates_removed / n_input if n_input > 0 else 0.0

        return PackingResult(
            chunks=packed,
            raw_tokens=raw_tokens,
            deduped_tokens=deduped_tokens,
            packed_tokens=packed_tokens,
            tokens_saved=tokens_saved,
            tokens_saved_pct=tokens_saved_pct,
            duplicate_rate=duplicate_rate,
        )

