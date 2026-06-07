"""Eval runner — executes RAG eval cases against a HybridRetriever.

No LLM generation is invoked; only the retrieval pipeline is exercised.
"""

from __future__ import annotations

from evals.metrics import mrr, precision_at_k, recall_at_k
from models.evals import EvalReport, EvalRunResult, RagEvalCase
from models.rag import RetrievalQuery


def run(hybrid: object, cases: list[RagEvalCase], k: int = 10) -> EvalReport:
    """Run all eval cases against ``hybrid`` and return an aggregated report.

    Args:
        hybrid: A :class:`~rag.hybrid_retriever.HybridRetriever` instance
            (or any object exposing a ``retrieve(RetrievalQuery) -> RetrievedContext``
            method).
        cases: List of :class:`~models.evals.RagEvalCase` objects to evaluate.
        k: Cut-off rank used for recall@k and precision@k metrics.

    Returns:
        An :class:`~models.evals.EvalReport` with per-case results and
        aggregate statistics.
    """
    results: list[EvalRunResult] = []
    trace_ids: list[str] = []

    for c in cases:
        q = RetrievalQuery(
            query=c.query,
            base_uri=c.base_uri,
            mode=c.mode,
            current_file=c.current_file,
            latest_error=c.latest_error,
            top_k=k,
            include_chat_history=False,
        )
        ctx = hybrid.retrieve(q)  # type: ignore[attr-defined]

        retrieved = [s.path or s.uri for s in ctx.spans]
        expected = set(c.expected_files)

        # Files that should NOT have been retrieved but were
        bad = set(c.must_not_retrieve) & set(retrieved)

        # Tokens consumed by irrelevant spans
        irrelevant_tokens = sum(
            s.token_estimate for s in ctx.spans if (s.path or s.uri) not in expected
        )

        # How many expected symbols appear anywhere in retrieved content
        sym_hits = sum(1 for sy in c.expected_symbols for s in ctx.spans if sy in s.content)
        sym_rate = sym_hits / max(1, len(c.expected_symbols))

        results.append(
            EvalRunResult(
                case_id=c.id,
                recall_at_k=recall_at_k(retrieved, expected, k),
                precision_at_k=precision_at_k(retrieved, expected, k),
                mrr=mrr(retrieved, expected),
                expected_symbol_hit_rate=sym_rate,
                irrelevant_context_tokens=irrelevant_tokens,
                inserted_token_count=ctx.token_estimate,
                freshness_error_rate=len(bad) / max(1, len(retrieved)),
                dedupe_savings=0,
            )
        )
        if ctx.trace_id:
            trace_ids.append(ctx.trace_id)

    agg: dict[str, float] = {
        "recall@k": sum(r.recall_at_k for r in results) / max(1, len(results)),
        "precision@k": sum(r.precision_at_k for r in results) / max(1, len(results)),
        "mrr": sum(r.mrr for r in results) / max(1, len(results)),
        "avg_tokens": sum(r.inserted_token_count for r in results) / max(1, len(results)),
    }

    return EvalReport(results=results, aggregate=agg, trace_ids=trace_ids)

