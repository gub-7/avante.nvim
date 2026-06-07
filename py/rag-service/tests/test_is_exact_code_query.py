"""Tests for the isExactCodeQuery heuristic in src/rag/router.py.

TDD Increment 4 — pure function, no I/O, no model dependencies.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Truth-table cases from the TDD spec
# ---------------------------------------------------------------------------
TRUTH_TABLE: list[tuple[str, bool]] = [
    # Positive cases — should be recognised as exact-code queries
    ("foo/bar/baz.ts", True),
    ("MyClassName", True),
    ("Error: invalid provider", True),
    ("undefined is not a function", True),
    ("Foo::bar", True),
    ("file.py", True),
    ("fn(", True),
    # Negative cases — natural-language / conceptual questions
    ("how does memory work?", False),
    ("explain the architecture", False),
    ("what is the difference between A and B", False),
]


@pytest.mark.parametrize("query,expected", TRUTH_TABLE)
def test_truth_table(query: str, expected: bool) -> None:
    """isExactCodeQuery returns the correct boolean for every spec example."""
    from rag.router import is_exact_code_query

    assert is_exact_code_query(query) is expected, (
        f"is_exact_code_query({query!r}) should be {expected}"
    )


# ---------------------------------------------------------------------------
# Property: all-stopword strings → False
# ---------------------------------------------------------------------------
STOPWORD_STRINGS: list[str] = [
    "the",
    "a an the",
    "is are was were",
    "how what why when where who",
    "the quick brown fox",
    "this that those these",
    "   ",  # only whitespace
    "",
]


@pytest.mark.parametrize("text", STOPWORD_STRINGS)
def test_stopword_only_is_false(text: str) -> None:
    """Strings consisting only of stopwords (or whitespace) must return False."""
    from rag.router import is_exact_code_query

    assert is_exact_code_query(text) is False, (
        f"Expected False for stopword-only string {text!r}"
    )


# ---------------------------------------------------------------------------
# Additional unambiguous code-like inputs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query",
    [
        "TypeError: cannot read property",
        "import foo from './bar'",
        "def my_function():",
        "class MyClass(Base):",
        "src/components/Button.tsx",
        "some_variable_name",
        "SomeClass.someMethod",
        "foo_bar_baz",
        "snake_case_identifier",
        "camelCaseSymbol",
        "PascalCaseClass",
        "__init__",
        "path/to/file.lua",
    ],
)
def test_extra_code_positives(query: str) -> None:
    """Additional code-like strings that must return True."""
    from rag.router import is_exact_code_query

    assert is_exact_code_query(query) is True, (
        f"Expected True for code-like string {query!r}"
    )


@pytest.mark.parametrize(
    "query",
    [
        "what does this do",
        "can you help me understand",
        "please refactor my code",
        "summarize the changes",
        "describe the overall approach",
    ],
)
def test_extra_natural_language_negatives(query: str) -> None:
    """Additional natural-language queries that must return False."""
    from rag.router import is_exact_code_query

    assert is_exact_code_query(query) is False, (
        f"Expected False for natural-language string {query!r}"
    )

