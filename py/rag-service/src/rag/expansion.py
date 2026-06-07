"""Depth-1 import / symbol expansion for retrieved spans.

After the main retrieval + rerank pipeline has produced a set of spans,
:func:`expand` inspects each span's import statements and, for recognised
languages, looks up those imported symbols in the symbol index.  Matching
symbol definitions are appended as extra ``FileSpan`` objects tagged with
``retrieval_sources=["expansion"]``.

The expansion is strictly bounded by :class:`ExpansionBudget` to avoid
blowing the context window.
"""

from __future__ import annotations

import hashlib
import re

from libs.utils import uri_to_path
from models.rag import FileSpan, RetrievalQuery
from rag.context_budget import estimate_tokens
from rag.symbol_index import search_symbols


class ExpansionBudget:
    """Hard limits that cap the total cost of one expansion pass."""

    #: Depth limit (only immediate imports are followed, not transitive ones).
    max_depth: int = 1
    #: Maximum number of extra spans to add per retrieval call.
    max_extra_spans: int = 4
    #: Maximum cumulative token budget for all extra spans combined.
    max_extra_tokens: int = 3_000


# Per-language regexes that extract the *last dotted component* of an
# import/require statement — i.e. the short name likely to appear in the
# symbol index.
IMPORT_RES: dict[str, list[re.Pattern[str]]] = {
    "python": [
        re.compile(
            r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.M
        )
    ],
    "javascript": [
        re.compile(r"""(?:import|require)\s*\(?['"]([^'"]+)['"]""")
    ],
    "typescript": [
        re.compile(r"""(?:import|require)\s*\(?['"]([^'"]+)['"]""")
    ],
    "go": [
        re.compile(r'^\s*import\s+"([^"]+)"', re.M)
    ],
    "rust": [
        re.compile(r"^\s*use\s+([\w:]+)", re.M)
    ],
    "java": [
        re.compile(r"^\s*import\s+([\w.]+);", re.M)
    ],
}


def expand(spans: list[FileSpan], query: RetrievalQuery) -> list[FileSpan]:
    """Perform one depth-1 expansion pass over *spans*.

    For each span that has a recognised language, the function extracts
    import / use / require targets, resolves each to a symbol via
    :func:`~rag.symbol_index.search_symbols`, reads the definition from
    disk, and returns it as a new :class:`~models.rag.FileSpan`.

    The result list is bounded by :attr:`ExpansionBudget.max_extra_spans`
    and :attr:`ExpansionBudget.max_extra_tokens`.  Spans whose symbol
    definitions cannot be read from disk are silently skipped.

    Args:
        spans:  The scored/reranked spans from the main pipeline.
        query:  The original retrieval query (used for ``base_uri``).

    Returns:
        A (possibly empty) list of additional :class:`~models.rag.FileSpan`
        objects.  Each has ``retrieval_sources=["expansion"]`` and
        ``score=2.0`` (lower than typical ranked spans so they sort last).
    """
    extras: list[FileSpan] = []
    used_tokens = 0

    for s in spans:
        if used_tokens >= ExpansionBudget.max_extra_tokens:
            break
        if len(extras) >= ExpansionBudget.max_extra_spans:
            break
        if not s.language:
            continue

        pats = IMPORT_RES.get(s.language, [])
        if not pats:
            continue

        symbols: set[str] = set()
        for p in pats:
            for m in p.finditer(s.content):
                for g in m.groups():
                    if g:
                        # Take only the final component so "os.path" → "path"
                        symbols.add(g.split(".")[-1])

        for sym in list(symbols)[:4]:
            for row in search_symbols(query.base_uri, sym, limit=1):
                file_uri: str = row["file_uri"]
                try:
                    file_path = uri_to_path(file_uri)
                    lines = file_path.read_text(
                        encoding="utf-8", errors="ignore"
                    ).splitlines()
                except OSError:
                    continue

                a = max(0, (row["start_line"] or 1) - 1)
                b = min(len(lines), row["end_line"] or row["start_line"] or 1)
                content = "\n".join(lines[a:b])
                if not content:
                    continue

                tok = estimate_tokens(content)
                if used_tokens + tok > ExpansionBudget.max_extra_tokens:
                    continue

                extras.append(
                    FileSpan(
                        uri=file_uri,
                        path=str(file_path),
                        start_line=row["start_line"],
                        end_line=row["end_line"],
                        content=content,
                        reason=f"expand:import:{sym}",
                        score=2.0,
                        token_estimate=tok,
                        hash=hashlib.sha256(content.encode()).hexdigest(),
                        retrieval_sources=["expansion"],
                        chunk_kind=row["symbol_kind"],
                        language=row.get("language"),
                    )
                )
                used_tokens += tok

                if len(extras) >= ExpansionBudget.max_extra_spans:
                    break

    return extras

