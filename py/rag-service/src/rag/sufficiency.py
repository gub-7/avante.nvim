"""Rules-based context sufficiency checker.

Determines whether the retrieved spans are sufficient to answer the query
in the requested mode.  The check is intentionally lightweight — it does
not call an LLM; it uses structural heuristics that the
:class:`~rag.agentic_planner.AgenticPlanner` uses to decide whether
additional retrieval passes are needed.

Typical usage::

    from rag.sufficiency import check
    suff = check(query, spans)
    if not suff.sufficient:
        print("Missing:", suff.missing)
"""

from __future__ import annotations

import re

from models.rag import ContextSufficiency, FileSpan, RetrievalQuery


def check(query: RetrievalQuery, spans: list[FileSpan]) -> ContextSufficiency:
    """Evaluate whether *spans* are sufficient for *query*.

    Mode-specific rules:

    **ask**
        Sufficient as long as at least one span exists.

    **test-fix**
        Requires both a test span (``chunk_kind == "test"``) and an
        implementation span (``chunk_kind`` in
        ``{"function", "method", "class"}``).  Missing kinds are reported in
        ``missing`` and corresponding retrieval hints go to
        ``suggested_retrievals``.

    **edit-small**
        If *selected_text* contains a recognised identifier, that symbol must
        appear in at least one span's content.  If not, the definition is
        flagged as missing.

    **refactor**
        Requires spans from at least two distinct files (proxy for: we have
        the definition *and* at least one call-site).

    **search / other modes**
        Falls back to: sufficient iff at least one span exists.

    Args:
        query: The original retrieval query.
        spans: The final set of kept spans from the retrieval pipeline.

    Returns:
        A :class:`~models.rag.ContextSufficiency` with a boolean verdict,
        a confidence score, and optional ``missing`` / ``suggested_retrievals``
        lists.
    """
    if query.mode == "ask":
        ok = bool(spans)
        return ContextSufficiency(
            sufficient=ok,
            confidence=0.7 if ok else 0.3,
            missing=[] if ok else ["any_relevant_doc"],
        )

    if query.mode == "test-fix":
        has_test = any(s.chunk_kind == "test" for s in spans)
        has_impl = any(
            s.chunk_kind in {"function", "method", "class"} for s in spans
        )
        missing: list[str] = []
        if not has_test:
            missing.append("failing_test_span")
        if not has_impl:
            missing.append("implementation_span")
        return ContextSufficiency(
            sufficient=not missing,
            confidence=0.8 if not missing else 0.4,
            missing=missing,
            suggested_retrievals=(
                ["retrieve_exact(latest_error)", "inspect_tests"]
                if missing
                else []
            ),
        )

    if query.mode == "edit-small" and query.selected_text:
        sym = re.search(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", query.selected_text)
        if sym:
            name = sym.group(0)
            if not any(name in s.content for s in spans):
                return ContextSufficiency(
                    sufficient=False,
                    confidence=0.4,
                    missing=[f"definition_of:{name}"],
                    suggested_retrievals=["retrieve_symbol"],
                )

    if query.mode == "refactor":
        files = {s.uri for s in spans}
        ok = len(files) >= 2
        return ContextSufficiency(
            sufficient=ok,
            confidence=0.7 if ok else 0.4,
            missing=[] if ok else ["additional_callsites"],
        )

    # Default: sufficient if any spans were retrieved
    return ContextSufficiency(
        sufficient=bool(spans),
        confidence=0.6 if spans else 0.2,
    )

