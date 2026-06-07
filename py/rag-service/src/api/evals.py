"""Evals API router.

Exposes endpoints to run RAG retrieval evaluation cases and fetch the last
report.  No LLM generation is invoked — only the retrieval pipeline is
exercised.
"""

from __future__ import annotations

from fastapi import APIRouter

from evals.rag_cases import load as load_cases
from evals.runner import run as run_eval
from libs.logger import logger
from models.evals import EvalReport

router = APIRouter(prefix="/api/v1/evals/rag", tags=["evals"])

# Cached result from the most recent run
_last_report: EvalReport | None = None


@router.post("/run")
async def evals_run() -> EvalReport | dict:
    """Run all eval cases against the live hybrid retriever.

    Loads cases from ``${DATA_DIR}/evals/rag_cases.jsonl``, executes each
    against the singleton :class:`~rag.hybrid_retriever.HybridRetriever`
    wired in ``api.rag``, and caches the resulting report.

    Returns:
        An :class:`~models.evals.EvalReport` with per-case results and
        aggregated metrics.
    """
    global _last_report
    # Import lazily to avoid circular dependency at module load time
    from api.rag import _hybrid  # noqa: PLC0415

    cases = load_cases()
    logger.info("Running %d eval cases", len(cases))
    _last_report = run_eval(_hybrid, cases)
    return _last_report


@router.get("/report")
async def evals_report() -> EvalReport | dict:
    """Return the cached result from the most recent eval run.

    Returns an empty report structure when no run has been performed yet.
    """
    if _last_report is None:
        return {"results": [], "aggregate": {}}
    return _last_report

