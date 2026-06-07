"""Exact (ripgrep-backed) search for the RAG retrieval pipeline.

Provides keyword/symbol/stack-frame search over local file trees.
Falls back to a pure-Python implementation when ripgrep is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from libs.logger import logger
from libs.utils import path_to_uri
from models.rag import FileSpan, RetrievalQuery
from rag.context_budget import estimate_tokens

# Path to the ripgrep binary (None if not installed).
RG = shutil.which("rg")

# Regex that matches "filename:lineno" patterns inside stack traces.
STACK_FRAME_RE = re.compile(r"([A-Za-z]:[\\/][^\s:\"']+|/[^\s:\"']+|[\w./\\-]+\.\w+):(\d+)")

# Regex that extracts symbol-like identifiers (≥3 chars, starts with letter/underscore).
SYMBOL_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")

MAX_RESULTS_PER_QUERY = 50
CONTEXT_LINES = 4


def _rg(query: str, base_path: Path, max_count: int = MAX_RESULTS_PER_QUERY) -> list[dict]:
    """Run ripgrep and return a list of hit dicts.

    Each dict contains keys: ``path``, ``line`` (1-based), ``text``.

    Falls back to :func:`_python_fallback` when ripgrep is unavailable or
    encounters an error.
    """
    if not RG:
        return _python_fallback(query, base_path, max_count)
    try:
        proc = subprocess.run(
            [
                RG,
                "--json",
                "--line-number",
                "--column",
                "--no-messages",
                "--max-count",
                str(max_count),
                "--hidden",
                "--glob",
                "!.git",
                query,
                str(base_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("ripgrep failed: %s", e)
        return _python_fallback(query, base_path, max_count)

    hits: list[dict] = []
    for line in proc.stdout.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "match":
            continue
        data = obj["data"]
        path = data["path"]["text"]
        line_no = data["line_number"]
        line_text = data["lines"]["text"].rstrip("\n")
        hits.append({"path": path, "line": line_no, "text": line_text})
    return hits


def _python_fallback(query: str, base_path: Path, max_count: int) -> list[dict]:
    """Pure-Python grep fallback used when ripgrep is unavailable."""
    out: list[dict] = []
    pat = re.compile(re.escape(query))
    for p in base_path.rglob("*"):
        if len(out) >= max_count:
            break
        if p.is_dir() or ".git" in p.parts:
            continue
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    if pat.search(line):
                        out.append({"path": str(p), "line": i, "text": line.rstrip("\n")})
                        if len(out) >= max_count:
                            break
        except OSError:
            continue
    return out


def _expand_span(path: str, line: int, ctx: int = CONTEXT_LINES) -> tuple[int, int, str]:
    """Read surrounding lines from *path* and return (start_line, end_line, content).

    Line numbers are 1-based. Returns empty string for the content on I/O error.
    """
    try:
        with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
    except OSError:
        return line, line, ""
    start = max(0, line - 1 - ctx)
    end = min(len(all_lines), line + ctx)
    return start + 1, end, "".join(all_lines[start:end])


def _extract_targets(q: RetrievalQuery) -> Iterable[tuple[str, str, float]]:
    """Yield ``(term, reason, base_score)`` tuples derived from the query.

    Priority order (highest base_score first):
    1. Stack-frame file paths and line references from ``latest_error``.
    2. Symbol identifiers from ``latest_error``.
    3. Symbols from ``selected_text``.
    4. Current file basename.
    5. The raw query string itself.
    """
    seen: set[str] = set()

    # --- Highest priority: stack traces ---
    if q.latest_error:
        for m in STACK_FRAME_RE.finditer(q.latest_error):
            term = f"{m.group(1)}:{m.group(2)}"
            if term not in seen:
                seen.add(term)
                yield term, "stack_frame", 4.5
        for m in SYMBOL_RE.finditer(q.latest_error):
            t = m.group(0)
            if t not in seen:
                seen.add(t)
                yield t, "error_symbol", 4.0

    # --- Current file basename ---
    if q.current_file:
        base = Path(q.current_file).name
        if base not in seen:
            seen.add(base)
            yield base, "current_file", 3.2

    # --- Selected text symbols ---
    if q.selected_text:
        for m in SYMBOL_RE.finditer(q.selected_text):
            t = m.group(0)
            if t not in seen:
                seen.add(t)
                yield t, "selected_symbol", 3.5

    # --- Raw query string (lowest priority) ---
    if q.query and q.query not in seen:
        yield q.query, "query_text", 2.5


class ExactSearch:
    """Ripgrep-backed exact search over a local directory tree.

    Each call to :meth:`retrieve` yields :class:`~models.rag.FileSpan`
    objects annotated with a ``reason`` tag and a base ``score``.
    """

    def retrieve(self, query: RetrievalQuery, base_path: Path) -> list[FileSpan]:
        """Return exact-match spans for all search targets derived from *query*.

        Args:
            query: The retrieval query, which may contain ``latest_error``,
                ``current_file``, ``selected_text``, and a raw ``query`` string.
            base_path: Root directory to search.

        Returns:
            List of :class:`~models.rag.FileSpan` objects ordered by discovery.
        """
        spans: list[FileSpan] = []
        for term, reason, base_score in _extract_targets(query):
            hits = _rg(term, base_path)
            for h in hits:
                start, end, content = _expand_span(h["path"], h["line"])
                if not content:
                    continue
                spans.append(
                    FileSpan(
                        uri=path_to_uri(Path(h["path"])),
                        path=h["path"],
                        start_line=start,
                        end_line=end,
                        content=content,
                        reason=f"exact:{reason}",
                        score=base_score,
                        token_estimate=estimate_tokens(content),
                        hash=hashlib.sha256(content.encode()).hexdigest(),
                        retrieval_sources=["exact"],
                    )
                )
        return spans

