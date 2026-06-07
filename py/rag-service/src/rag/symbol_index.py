"""Symbol extractor, DB writer, and search for tree-sitter-based symbol indexing."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from libs.db import get_db_connection
from libs.logger import logger
from libs.utils import path_to_uri, uri_to_path
from tree_sitter_language_pack import get_language, get_parser

QUERIES_DIR = Path(__file__).parent / "queries"

# Map from file extension to tree-sitter language name
LANG_EXT: dict[str, str] = {
    ".py": "python",
    ".lua": "lua",
    ".rs": "rust",
    ".go": "go",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".java": "java",
    ".swift": "swift",
    ".rb": "ruby",
    ".php": "php",
}

# Regex matching test file paths by name convention
TEST_FILENAME_RE = re.compile(
    r"(^test_|_test\.|_spec\.|\.spec\.|\.test\.|/tests?/)",
    re.IGNORECASE,
)

# Map tree-sitter capture names → symbol_kind values
CAPTURE_TO_KIND: dict[str, str] = {
    "function": "function",
    "method": "method",
    "class": "class",
    "interface": "interface",
    "type": "type",
    "constant": "constant",
    "variable": "variable",
    "module": "module",
}


def _load_query(lang: str) -> str | None:
    """Load a tree-sitter query file for the given language. Returns None if not found."""
    path = QUERIES_DIR / f"{lang}.scm"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _is_test_path(file_uri: str, test_patterns: list[str] | None = None) -> bool:
    """Return True when the file URI matches test-file naming conventions."""
    if TEST_FILENAME_RE.search(file_uri):
        return True
    if test_patterns:
        for p in test_patterns:
            if re.search(p, file_uri):
                return True
    return False


def extract_symbols(
    file_uri: str,
    resource_uri: str,
    content: str,
    test_patterns: list[str] | None = None,
) -> list[dict]:
    """Parse *content* with tree-sitter and return a list of symbol dicts.

    Each dict has the keys expected by the ``symbols`` DB table.
    Returns an empty list when the language is unsupported or parsing fails.
    """
    ext = Path(uri_to_path(file_uri)).suffix.lower()
    lang = LANG_EXT.get(ext)
    if not lang:
        return []

    query_text = _load_query(lang)
    if not query_text:
        return []

    try:
        parser = get_parser(lang)
        tree = parser.parse(content.encode("utf-8"))
        language = get_language(lang)
        query = language.query(query_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("symbol extract failed (%s, %s): %s", file_uri, lang, exc)
        return []

    is_test_file = _is_test_path(file_uri, test_patterns)
    text_hash = hashlib.sha256(content.encode()).hexdigest()

    rows: list[dict] = []
    for node, capture_name in query.captures(tree.root_node):
        kind = CAPTURE_TO_KIND.get(capture_name, "unknown")
        # Promote functions/methods in test files to "test" kind
        if is_test_file and kind in {"function", "method"}:
            kind = "test"

        name = node.text.decode("utf-8", errors="replace")
        start_line = node.start_point[0] + 1  # tree-sitter is 0-based
        end_line = node.end_point[0] + 1

        rows.append(
            {
                "resource_uri": resource_uri,
                "file_uri": file_uri,
                "symbol_name": name,
                "symbol_kind": kind,
                "start_line": start_line,
                "end_line": end_line,
                "language": lang,
                "text_hash": text_hash,
                "metadata": json.dumps({"is_test_file": is_test_file}),
            }
        )

    return rows


def replace_symbols_for_file(
    file_uri: str,
    resource_uri: str,
    content: str,
    test_patterns: list[str] | None = None,
) -> int:
    """Delete existing symbols for *file_uri* and insert fresh ones.

    Returns the number of symbols written.
    """
    rows = extract_symbols(file_uri, resource_uri, content, test_patterns)
    with get_db_connection() as conn:
        conn.execute("DELETE FROM symbols WHERE file_uri = ?", (file_uri,))
        if rows:
            conn.executemany(
                """
                INSERT INTO symbols (
                    resource_uri, file_uri, symbol_name, symbol_kind,
                    start_line, end_line, language, text_hash, metadata
                ) VALUES (
                    :resource_uri, :file_uri, :symbol_name, :symbol_kind,
                    :start_line, :end_line, :language, :text_hash, :metadata
                )
                """,
                rows,
            )
        conn.commit()
    return len(rows)


def search_symbols(
    base_uri: str,
    q: str,
    kinds: list[str] | None = None,
    limit: int = 30,
) -> list[dict]:
    """Return symbols matching *q* (LIKE pattern) within the given resource.

    Optionally filters by *kinds* (list of symbol_kind values).
    """
    sql = "SELECT * FROM symbols WHERE resource_uri = ? AND symbol_name LIKE ?"
    params: list = [base_uri, f"%{q}%"]

    if kinds:
        placeholders = ",".join("?" * len(kinds))
        sql += f" AND symbol_kind IN ({placeholders})"
        params.extend(kinds)

    sql += " LIMIT ?"
    params.append(limit)

    with get_db_connection() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]

