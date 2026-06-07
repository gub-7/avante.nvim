"""RAG backend router for the avante RAG service.

This module contains two public components:

1.  ``is_exact_code_query(query) -> bool``
    A pure heuristic that decides whether a free-text query looks like an
    *exact-code* lookup (file path, symbol name, error message, …) rather than
    a natural-language question.  No I/O, no model dependencies — pure string
    logic so it is trivially unit-testable.

2.  ``choose_backend_v1(req, sys_stats) -> RouteDecision``
    Deterministic router that picks a RAG backend for the given request and
    current system state.  No state, no DB writes, no network calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums / small value types
# ---------------------------------------------------------------------------


class BackendName(str, Enum):
    """Names of the RAG vector-search backends."""

    EXACT = "exact"
    QDRANT = "qdrant"
    MILVUS = "milvus"
    CHROMA = "chroma"
    HYBRID = "hybrid"
    AUTO = "auto"


class SearchMode(str, Enum):
    """How the router should select a backend."""

    AUTO = "auto"
    EXACT = "exact"
    QDRANT = "qdrant"
    MILVUS = "milvus"
    CHROMA = "chroma"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# is_exact_code_query
# ---------------------------------------------------------------------------

# Stopwords that are not meaningful for exact-code routing on their own.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "shall", "may", "might", "must", "can",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "us", "them", "my", "your", "his", "its", "our", "their",
        "this", "that", "these", "those",
        "what", "which", "who", "whom", "whose",
        "when", "where", "why", "how",
        "and", "but", "or", "nor", "so", "yet", "for", "as", "if",
        "in", "on", "at", "by", "from", "with", "about", "of", "to",
        "up", "out", "into", "than", "then", "than", "not", "no",
        "please", "help", "make", "use", "just", "also", "more",
        "some", "any", "all", "each", "every", "both", "few",
        "here", "there", "now", "very", "well", "still",
        "between", "difference", "explain", "describe", "summarize",
        "understand", "overall", "approach", "changes", "refactor",
        "can", "let", "want", "need", "try", "think", "feel",
    }
)

# Patterns that strongly indicate an exact-code query.
# NOTE: avoid trailing backslashes in re.VERBOSE comments — they cause
# line-continuation that silently eats the next pattern line.  All patterns
# below are written without re.VERBOSE to prevent that footgun.

# Path with at least one directory separator, or bare filename.ext
_FILE_PATH_RE = re.compile(
    r"(?:(?:[A-Za-z0-9_.+-]+[/\\])+[A-Za-z0-9_.+-]+"
    r"|[A-Za-z0-9_-]+\.[a-zA-Z]{1,6})"
)

# Symbol-like identifiers: Foo::bar, foo.bar, snake_case, PascalCase,
# camelCase, UPPER_CASE, __dunder__
_SYMBOL_RE = re.compile(
    r"(?:"
    r"[A-Za-z_][A-Za-z0-9_]*::[A-Za-z_][A-Za-z0-9_]*"          # Foo::bar
    r"|[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){1,}" # foo.bar
    r"|[A-Za-z_][A-Za-z0-9]*_[A-Za-z0-9_]+"                    # snake_case
    r"|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+"                      # PascalCase
    r"|[a-z][a-z0-9]+[A-Z][A-Za-z0-9]+"                        # camelCase
    r"|[A-Z]{2,}(?:_[A-Z0-9]+)+"                               # UPPER_CASE
    r"|__[A-Za-z_][A-Za-z0-9_]+__"                              # __dunder__
    r")"
)

_OPEN_PAREN_RE = re.compile(r"\w+\(")      # fn(
_ERROR_COLON_RE = re.compile(r"\w.*:\s+\S")  # Error: something  / TypeError: …
_STACK_FRAME_RE = re.compile(r"\w+\.\w+:\d+")  # file.ext:lineno


def is_exact_code_query(query: str) -> bool:
    """Return True if *query* looks like an exact-code / symbol lookup.

    This is a pure heuristic: no network calls, no model inference.
    It is intentionally conservative on natural-language questions.

    Args:
        query: The raw query string from the user or agent.

    Returns:
        True  — query looks like a file path, symbol name, error message, or
                code fragment that benefits from ripgrep / exact search.
        False — query looks like a natural-language question.
    """
    if not query or not query.strip():
        return False

    # Reject queries that are *only* stopwords / whitespace.
    words = query.lower().split()
    non_stop = [w.strip(".,?!;:\"'") for w in words if w.strip(".,?!;:\"'") not in _STOPWORDS]
    if not non_stop:
        return False

    # Positive signals — any hit → True.
    if _FILE_PATH_RE.search(query):
        return True
    if _SYMBOL_RE.search(query):
        return True
    if _OPEN_PAREN_RE.search(query):
        return True
    if _STACK_FRAME_RE.search(query):
        return True

    # "Error: …" / "TypeError: …" style messages
    if _ERROR_COLON_RE.search(query):
        return True

    # "undefined is not a function" and similar JS runtime errors:
    # contains a recognised JS-error fragment.
    _JS_ERROR_PHRASES = (
        "is not a function",
        "is not defined",
        "cannot read property",
        "cannot read properties",
        "null is not",
        "undefined is not",
        "unexpected token",
        "syntax error",
    )
    lower = query.lower()
    if any(phrase in lower for phrase in _JS_ERROR_PHRASES):
        return True

    return False


# ---------------------------------------------------------------------------
# Router v1 data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SystemStats:
    """Lightweight snapshot of the current system state used by the router.

    All fields are optional / defaulted so test code can construct minimal
    instances without boilerplate.
    """

    gpu_vram_free_mb: float = 0.0
    gpu_util_pct: float = 0.0
    cpu_util_pct: float = 0.0
    milvus_hot_collections: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class RouteRequest:
    """Subset of a retrieval request that the router cares about.

    Keeping this separate from the full ``RetrievalQuery`` means the router
    has zero import dependency on Pydantic / FastAPI at test time.
    """

    query: str = ""
    # Explicit backend override from the caller ("" or "auto" → let router decide).
    requested_backend: str = ""
    # ``SearchMode`` value (string) from the request — "" / "auto" → router decides.
    search_mode: str = ""
    # Number of metadata filter dimensions attached to the query.
    filter_count: int = 0
    # Batch size (number of parallel sub-queries the agent is issuing).
    batch_size: int = 1
    # Collection name the caller wants to search (for hot-collection detection).
    collection: str = ""
    # Whether the caller enabled shadow mode.
    shadow: bool = False


@dataclass(frozen=True)
class RouteDecision:
    """Output of ``choose_backend_v1``."""

    primary: BackendName
    shadow: Optional[BackendName]
    reason: str
    mode_used: SearchMode

    def __post_init__(self) -> None:  # pragma: no cover
        assert self.reason, "RouteDecision.reason must not be empty"


# ---------------------------------------------------------------------------
# Router v1 — deterministic, no side effects
# ---------------------------------------------------------------------------

# Tuning constants (kept as module-level for easy override in tests).
_LARGE_BATCH_THRESHOLD = 10
_GPU_UTIL_BUSY_PCT = 70.0       # above this → GPU is "busy"
_GPU_VRAM_HOT_MB = 512.0        # minimum free VRAM to consider Milvus hot path
_FILTER_HEAVY_THRESHOLD = 2     # ≥ this many filters → favour Qdrant scalar filtering


def choose_backend_v1(
    req: RouteRequest,
    sys: SystemStats | None = None,
) -> RouteDecision:
    """Choose a RAG backend deterministically for the given request + system state.

    Decision tree (first matching rule wins):

    1.  Manual override: caller set ``requested_backend`` to a concrete value.
    2.  Exact-code query heuristic → ``BackendName.EXACT``.
    3.  Filter-heavy + small batch → ``BackendName.QDRANT`` (good at scalar filtering).
    4.  Large batch + GPU available + hot collection → ``BackendName.MILVUS``.
    5.  Large batch + GPU busy or no VRAM → ``BackendName.QDRANT``.
    6.  Default fallback → ``BackendName.QDRANT``.

    Args:
        req:  Routing-relevant fields extracted from the retrieval request.
        sys:  Current system stats.  ``None`` is treated as all-zero stats
              (safe default for tests / environments without a probe).

    Returns:
        A :class:`RouteDecision` with ``primary``, optional ``shadow``,
        ``reason``, and ``mode_used``.  Never raises.
    """
    if sys is None:
        sys = SystemStats()

    # Normalise the requested mode — unknown values coerce to AUTO.
    raw_mode = (req.search_mode or "").strip().lower()
    try:
        mode_used = SearchMode(raw_mode) if raw_mode else SearchMode.AUTO
    except ValueError:
        mode_used = SearchMode.AUTO

    shadow: Optional[BackendName] = None

    # ------------------------------------------------------------------
    # Rule 1: explicit manual override
    # ------------------------------------------------------------------
    raw_backend = (req.requested_backend or "").strip().lower()
    if raw_backend and raw_backend not in ("auto", ""):
        try:
            primary = BackendName(raw_backend)
        except ValueError:
            primary = BackendName.QDRANT
            return RouteDecision(
                primary=primary,
                shadow=shadow,
                reason=f"unknown backend {raw_backend!r} requested; falling back to qdrant",
                mode_used=mode_used,
            )
        return RouteDecision(
            primary=primary,
            shadow=shadow,
            reason=f"manual override: caller requested {primary.value}",
            mode_used=mode_used,
        )

    # ------------------------------------------------------------------
    # Rule 2: exact-code query heuristic
    # ------------------------------------------------------------------
    if is_exact_code_query(req.query):
        return RouteDecision(
            primary=BackendName.EXACT,
            shadow=shadow,
            reason="exact-code query detected by heuristic; using ripgrep/exact search",
            mode_used=mode_used,
        )

    # ------------------------------------------------------------------
    # Rule 3: filter-heavy + small batch → Qdrant (efficient scalar filtering)
    # ------------------------------------------------------------------
    if req.filter_count >= _FILTER_HEAVY_THRESHOLD and req.batch_size < _LARGE_BATCH_THRESHOLD:
        return RouteDecision(
            primary=BackendName.QDRANT,
            shadow=shadow,
            reason=(
                f"filter-heavy query ({req.filter_count} filters, batch={req.batch_size}); "
                "Qdrant selected for efficient payload filtering"
            ),
            mode_used=mode_used,
        )

    # ------------------------------------------------------------------
    # Rule 4: large batch + GPU available + hot collection → Milvus
    # ------------------------------------------------------------------
    if req.batch_size >= _LARGE_BATCH_THRESHOLD:
        gpu_available = (
            sys.gpu_vram_free_mb >= _GPU_VRAM_HOT_MB
            and sys.gpu_util_pct < _GPU_UTIL_BUSY_PCT
        )
        collection_is_hot = (
            req.collection != ""
            and req.collection in sys.milvus_hot_collections
        )
        if gpu_available and collection_is_hot:
            return RouteDecision(
                primary=BackendName.MILVUS,
                shadow=shadow,
                reason=(
                    f"large batch (size={req.batch_size}), GPU available "
                    f"({sys.gpu_vram_free_mb:.0f} MB free, {sys.gpu_util_pct:.0f}% util), "
                    f"hot collection {req.collection!r}; Milvus GPU-CAGRA selected"
                ),
                mode_used=mode_used,
            )

        # ------------------------------------------------------------------
        # Rule 5: large batch but GPU busy / no VRAM → Qdrant
        # ------------------------------------------------------------------
        return RouteDecision(
            primary=BackendName.QDRANT,
            shadow=shadow,
            reason=(
                f"large batch (size={req.batch_size}) but GPU not suitable "
                f"(free_vram={sys.gpu_vram_free_mb:.0f} MB, util={sys.gpu_util_pct:.0f}%); "
                "falling back to Qdrant"
            ),
            mode_used=mode_used,
        )

    # ------------------------------------------------------------------
    # Rule 6: default fallback
    # ------------------------------------------------------------------
    return RouteDecision(
        primary=BackendName.QDRANT,
        shadow=shadow,
        reason="default: no specific routing rule matched; using Qdrant",
        mode_used=mode_used,
    )

