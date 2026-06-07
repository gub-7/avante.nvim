"""Bounded multi-step agentic retrieval planner.

The :class:`AgenticPlanner` wraps a :class:`~rag.hybrid_retriever.HybridRetriever`
and runs it iteratively until the context is deemed sufficient (as judged by
:func:`~rag.sufficiency.check`) or the per-mode step budget is exhausted.

Each additional retrieval pass is informed by the previous sufficiency
verdict — the planner adjusts the query to target specific missing pieces
(e.g. the definition of a symbol, an implementation span, etc.) rather than
repeating the original query verbatim.

Step budgets are conservative to avoid runaway latency:

+-------------+-------------+
| mode        | max_steps   |
+=============+=============+
| ask         | 1           |
+-------------+-------------+
| search      | 1           |
+-------------+-------------+
| edit-small  | 2           |
+-------------+-------------+
| test-fix    | 4           |
+-------------+-------------+
| refactor    | 6           |
+-------------+-------------+

Typical usage::

    planner = AgenticPlanner(hybrid_retriever)
    ctx = planner.run(query)
    print(ctx.sufficiency)
"""

from __future__ import annotations

from models.rag import RetrievalQuery, RetrievedContext
from rag.context_budget import BUDGETS
from rag.log_summarizer import summarize
from rag.sufficiency import check

# Maximum retrieval steps per mode.  Step 0 is always the initial call;
# additional steps (step 1, 2, …) are only made when context is insufficient.
AGENTIC_BUDGETS: dict[str, int] = {
    "ask": 1,
    "search": 1,
    "edit-small": 2,
    "test-fix": 4,
    "refactor": 6,
}


class AgenticPlanner:
    """Run :class:`~rag.hybrid_retriever.HybridRetriever` in a bounded loop.

    After the initial retrieval the planner checks
    :func:`~rag.sufficiency.check` and, if the context is not yet sufficient,
    issues a targeted follow-up query.  The loop stops as soon as either the
    context is sufficient or the per-mode step limit is reached.

    Before the first retrieval call the planner pre-processes ``latest_error``
    through :func:`~rag.log_summarizer.summarize` to keep error text within
    the ``max_log_tokens`` budget for the mode.

    Args:
        hybrid: A :class:`~rag.hybrid_retriever.HybridRetriever` instance
                (or any object with a ``retrieve(query) -> RetrievedContext``
                method).
    """

    def __init__(self, hybrid: object) -> None:
        self._hybrid = hybrid

    def run(self, query: RetrievalQuery) -> RetrievedContext:
        """Execute the agentic retrieval loop and return the best context.

        Args:
            query: The initial retrieval request.  May be mutated between
                   steps (via :meth:`~pydantic.BaseModel.model_copy`) to
                   target missing pieces.

        Returns:
            A :class:`~models.rag.RetrievedContext` with ``sufficiency``
            populated.  The ``trace_id`` reflects the *last* retrieval call.
        """
        max_steps = AGENTIC_BUDGETS.get(query.mode, 1)
        cfg = BUDGETS.get(query.mode, BUDGETS["ask"])

        # --- Pre-process latest_error to fit the log-token budget ---
        if query.latest_error:
            query = query.model_copy(
                update={
                    "latest_error": summarize(
                        query.latest_error, cfg["max_log_tokens"]
                    )
                }
            )

        # --- Initial retrieval ---
        ctx = self._hybrid.retrieve(query)  # type: ignore[attr-defined]

        # --- Iterative refinement ---
        for _step in range(1, max_steps):
            suff = check(query, ctx.spans)
            ctx.sufficiency = suff
            if suff.sufficient:
                break

            # Adjust query to target what is missing
            query = self._refine(query, suff)
            ctx = self._hybrid.retrieve(query)  # type: ignore[attr-defined]

        # Final sufficiency check (also covers the max_steps == 1 case)
        if ctx.sufficiency is None:
            ctx.sufficiency = check(query, ctx.spans)

        return ctx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _refine(query: RetrievalQuery, suff: object) -> RetrievalQuery:
        """Return a modified *query* targeting the first missing item.

        Args:
            query: The current query.
            suff:  A :class:`~models.rag.ContextSufficiency` with at least
                   one entry in ``missing``.

        Returns:
            A new :class:`~models.rag.RetrievalQuery` with one field adjusted.
        """
        from models.rag import ContextSufficiency

        if not isinstance(suff, ContextSufficiency) or not suff.missing:
            return query

        first = suff.missing[0]

        # Symbol definition lookup
        if first.startswith("definition_of:"):
            name = first.split(":", 1)[1]
            return query.model_copy(update={"query": name})

        # Need implementation span — drop chat history to reduce noise
        if first == "implementation_span":
            return query.model_copy(update={"include_chat_history": False})

        # Need test span — include stale to widen the search
        if first == "failing_test_span":
            return query.model_copy(update={"include_stale": True})

        # Need additional call-sites — refactor mode
        if first == "additional_callsites":
            return query.model_copy(update={"top_k": query.top_k + 5})

        # Default: return unchanged
        return query

