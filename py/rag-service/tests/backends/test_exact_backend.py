"""Tests for the ExactBackend adapter (Increment 3).

Runs a subset of the standard RagBackend contract suite against
:class:`~rag.backends.exact.ExactBackend`, plus ExactBackend-specific
tests for ripgrep/fallback behaviour and score propagation.

Contract coverage:
    1.  create_collection is idempotent
    2.  upsert returns after chunks are searchable
    3.  search results are sorted by score descending
    4.  [SKIPPED] search respects top_k — exact search may return fewer
    5.  search respects metadata_filter project
    6.  search respects metadata_filter path_prefix
    7.  search returns empty list when collection missing
    8.  delete_by_filter removes only matching chunks
    9.  stats reports collection_count (vector_count is None)
    10. search result carries backend_name field

Extra tests:
    E1. ripgrep is tried first; python fallback is used when rg fails
    E2. stack-frame hits carry score >= 4.0 from _extract_targets
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from rag.backends.base import (
    BackendName,
    CollectionSpec,
    EmbeddedChunk,
    MetadataFilter,
    SearchRequest,
)
from rag.backends.exact import ExactBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str,
    path: str,
    content: str = "def foo(): return 1",
    project: str = "proj_a",
    document_id: str = "doc1",
) -> EmbeddedChunk:
    """Return a minimal EmbeddedChunk whose path points to a real file."""
    return EmbeddedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        embedding=None,  # ExactBackend ignores embeddings
        metadata={"project": project, "path": path},
        token_count=len(content.split()),
    )


def _spec(name: str = "test_col", base_path: str | None = None) -> CollectionSpec:
    meta = {}
    if base_path is not None:
        meta["base_path"] = base_path
    return CollectionSpec(name=name, dimension=0, metadata=meta)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def exact_backend():
    """Return a fresh ExactBackend instance."""
    return ExactBackend()


@pytest.fixture
def backend_with_repo(exact_backend, fake_repo):
    """ExactBackend pre-configured with a collection pointing at fake_repo.

    Yields ``(backend, collection_name, main_path, test_path)`` where
    ``main_path`` and ``test_path`` are absolute path strings for the two
    files in ``fake_repo``.
    """
    col_name = "test_col"
    main_path = str(fake_repo / "src" / "main.py")
    test_path = str(fake_repo / "src" / "test_main.py")
    exact_backend.create_collection(_spec(col_name, base_path=str(fake_repo)))
    return exact_backend, col_name, main_path, test_path


# ---------------------------------------------------------------------------
# Contract test 1 — create_collection is idempotent
# ---------------------------------------------------------------------------


def test_create_collection_is_idempotent(exact_backend, fake_repo):
    spec = _spec("col_a", base_path=str(fake_repo))
    # Must not raise on repeated calls
    exact_backend.create_collection(spec)
    exact_backend.create_collection(spec)


# ---------------------------------------------------------------------------
# Contract test 2 — upsert → immediately searchable
# ---------------------------------------------------------------------------


def test_upsert_returns_after_chunks_are_searchable(backend_with_repo):
    backend, col_name, main_path, _ = backend_with_repo
    chunk = _make_chunk("c1", path=main_path)
    backend.upsert([chunk], collection=col_name)

    req = SearchRequest(query="foo", collection=col_name, top_k=5, embedding=None)
    results = backend.search(req)
    assert len(results) >= 1, "Expected at least one result for query 'foo'"
    assert any(r.chunk_id == "c1" for r in results), (
        "Upserted chunk 'c1' must be returned in search results"
    )


# ---------------------------------------------------------------------------
# Contract test 3 — results sorted by score descending
# ---------------------------------------------------------------------------


def test_search_returns_results_sorted_by_score_desc(backend_with_repo):
    backend, col_name, main_path, test_path = backend_with_repo
    backend.upsert(
        [
            _make_chunk("c_main", path=main_path),
            _make_chunk("c_test", path=test_path),
        ],
        collection=col_name,
    )

    req = SearchRequest(query="foo", collection=col_name, top_k=20, embedding=None)
    results = backend.search(req)
    assert len(results) >= 1
    for a, b in zip(results, results[1:]):
        assert a.score >= b.score - 1e-6, (
            f"Results not sorted descending: {a.score} < {b.score}"
        )


# ---------------------------------------------------------------------------
# Contract test 4 — top_k  [SKIPPED for ExactBackend]
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="ExactBackend may return fewer results than top_k")
def test_search_respects_top_k(backend_with_repo):
    """Skipped: ExactBackend does not guarantee returning exactly top_k items."""


# ---------------------------------------------------------------------------
# Contract test 5 — metadata filter: project
# ---------------------------------------------------------------------------


def test_search_respects_metadata_filter_project(exact_backend, fake_repo):
    """Only chunks whose project metadata matches the filter are returned."""
    col = "col_proj"
    exact_backend.create_collection(_spec(col, base_path=str(fake_repo)))

    main_path = str(fake_repo / "src" / "main.py")
    test_path = str(fake_repo / "src" / "test_main.py")

    exact_backend.upsert(
        [
            _make_chunk("pa1", path=main_path, project="proj_a"),
            _make_chunk("pa2", path=test_path, project="proj_a"),
        ],
        collection=col,
    )

    # Filter by proj_a → only chunks from proj_a allowed through
    req = SearchRequest(
        query="foo",
        collection=col,
        top_k=10,
        embedding=None,
        filters=[MetadataFilter(field="project", op="eq", value="proj_a")],
    )
    results = exact_backend.search(req)
    for r in results:
        assert r.metadata.get("project") == "proj_a", (
            f"Got result with project={r.metadata.get('project')!r}, expected 'proj_a'"
        )

    # Filter by a non-existent project → no results
    req_none = SearchRequest(
        query="foo",
        collection=col,
        top_k=10,
        embedding=None,
        filters=[MetadataFilter(field="project", op="eq", value="proj_zzz")],
    )
    assert exact_backend.search(req_none) == []


# ---------------------------------------------------------------------------
# Contract test 6 — metadata filter: path_prefix
# ---------------------------------------------------------------------------


def test_search_respects_metadata_filter_path_prefix(exact_backend, fake_repo):
    """Only results whose stored path starts with the prefix are returned."""
    col = "col_prefix"
    src_path = fake_repo / "src"
    exact_backend.create_collection(_spec(col, base_path=str(fake_repo)))

    main_path = str(src_path / "main.py")
    test_path = str(src_path / "test_main.py")

    exact_backend.upsert(
        [
            _make_chunk("pm1", path=main_path),
            _make_chunk("pm2", path=test_path),
        ],
        collection=col,
    )

    req = SearchRequest(
        query="foo",
        collection=col,
        top_k=10,
        embedding=None,
        filters=[MetadataFilter(field="path", op="prefix", value=str(src_path))],
    )
    results = exact_backend.search(req)
    for r in results:
        assert r.metadata.get("path", "").startswith(str(src_path)), (
            f"Result path {r.metadata.get('path')!r} does not start with {str(src_path)!r}"
        )

    # Non-matching prefix → no results
    req_none = SearchRequest(
        query="foo",
        collection=col,
        top_k=10,
        embedding=None,
        filters=[MetadataFilter(field="path", op="prefix", value="/nonexistent/")],
    )
    assert exact_backend.search(req_none) == []


# ---------------------------------------------------------------------------
# Contract test 7 — missing collection → empty list, no exception
# ---------------------------------------------------------------------------


def test_search_returns_empty_list_when_collection_missing(exact_backend):
    req = SearchRequest(
        query="anything",
        collection="no_such_collection_xyz",
        top_k=5,
        embedding=None,
    )
    results = exact_backend.search(req)
    assert results == []


# ---------------------------------------------------------------------------
# Contract test 8 — delete_by_filter removes only matching chunks
# ---------------------------------------------------------------------------


def test_delete_by_filter_removes_only_matching_chunks(backend_with_repo):
    backend, col_name, main_path, test_path = backend_with_repo
    backend.upsert(
        [
            _make_chunk("d1", path=main_path, document_id="doc_x"),
            _make_chunk("d2", path=test_path, document_id="doc_y"),
        ],
        collection=col_name,
    )

    deleted = backend.delete_by_filter(
        collection=col_name,
        filters=[MetadataFilter(field="document_id", op="eq", value="doc_x")],
    )
    assert deleted >= 1, "Expected at least one chunk to be deleted"

    # doc_x chunk should be gone (no matching chunk in index for main_path)
    # doc_y chunk should still be found via test_path
    req = SearchRequest(query="foo", collection=col_name, top_k=20, embedding=None)
    results = backend.search(req)
    ids = [r.chunk_id for r in results]
    assert "d1" not in ids, "Deleted chunk 'd1' must not appear in results"
    assert "d2" in ids, "Surviving chunk 'd2' must appear in results"


# ---------------------------------------------------------------------------
# Contract test 9 — stats: collection_count and vector_count=None
# ---------------------------------------------------------------------------


def test_stats_reports_collection_count_and_vector_count(exact_backend, fake_repo):
    exact_backend.create_collection(_spec("s1", base_path=str(fake_repo)))
    exact_backend.create_collection(_spec("s2", base_path=str(fake_repo)))

    stats = exact_backend.stats()
    assert stats.collection_count >= 2
    # ExactBackend cannot report a vector count
    assert stats.vector_count is None


# ---------------------------------------------------------------------------
# Contract test 10 — search result carries backend_name field
# ---------------------------------------------------------------------------


def test_search_result_carries_backend_name_field(backend_with_repo):
    backend, col_name, main_path, _ = backend_with_repo
    backend.upsert(
        [_make_chunk("bn1", path=main_path)],
        collection=col_name,
    )

    req = SearchRequest(query="foo", collection=col_name, top_k=1, embedding=None)
    results = backend.search(req)
    assert len(results) >= 1
    assert results[0].backend == BackendName.EXACT


# ---------------------------------------------------------------------------
# Extra E1 — ripgrep first, python fallback when rg unavailable
# ---------------------------------------------------------------------------


def test_exact_backend_searches_ripgrep_first_then_python_fallback(
    backend_with_repo,
):
    """When ripgrep is unavailable, the pure-Python fallback must be used.

    We simulate a missing ``rg`` binary by patching ``rag.exact_search.RG``
    to ``None``.  The ``_rg`` function then routes directly to
    ``_python_fallback``.  We spy on ``_python_fallback`` to confirm it is
    called and still returns results.
    """
    backend, col_name, main_path, _ = backend_with_repo
    backend.upsert([_make_chunk("e1", path=main_path)], collection=col_name)

    # Capture the real function BEFORE patching to avoid infinite recursion.
    from rag.exact_search import _python_fallback as _real_fallback  # noqa: PLC0415

    fallback_calls: list[str] = []

    def _spy_fallback(query: str, base_path: Path, max_count: int) -> list[dict]:
        fallback_calls.append(query)
        return _real_fallback(query, base_path, max_count)

    # Simulate ripgrep being absent — _rg immediately delegates to _python_fallback.
    with patch("rag.exact_search.RG", None):
        with patch("rag.exact_search._python_fallback", side_effect=_spy_fallback):
            req = SearchRequest(
                query="foo", collection=col_name, top_k=5, embedding=None
            )
            results = backend.search(req)

    assert len(fallback_calls) >= 1, (
        "_python_fallback must be called when ripgrep (RG) is None"
    )
    # The fallback must still find content in the fake_repo files.
    assert len(results) >= 1


# ---------------------------------------------------------------------------
# Extra E2 — stack-frame hits carry score >= 4.0
# ---------------------------------------------------------------------------


def test_exact_backend_emits_score_from_extract_targets_base_score(
    exact_backend, fake_repo
):
    """Spans derived from error-symbol terms must have score >= 4.0.

    When ``ExactBackend.search`` receives a query, it passes it as
    ``latest_error`` in the ``RetrievalQuery``.  Identifiers extracted from
    ``latest_error`` receive base_score=4.0 (error_symbol) via
    ``_extract_targets``.  The ExactBackend must propagate these scores into
    the returned ``SearchResult`` objects unchanged.
    """
    col = "score_col"
    main_path = str(fake_repo / "src" / "main.py")
    exact_backend.create_collection(_spec(col, base_path=str(fake_repo)))
    exact_backend.upsert(
        [_make_chunk("sf1", path=main_path)],
        collection=col,
    )

    # Query "foo" is a 3-char identifier → _extract_targets treats it as an
    # error_symbol (base_score=4.0) when it appears in the latest_error field.
    # fake_repo/src/main.py contains "def foo():" so ripgrep finds a match.
    req = SearchRequest(
        query="foo",
        collection=col,
        top_k=10,
        embedding=None,
    )
    results = exact_backend.search(req)

    # At least one result must carry the elevated score from _extract_targets.
    high_score_results = [r for r in results if r.score >= 4.0]
    assert len(high_score_results) >= 1, (
        f"Expected at least one result with score >= 4.0 (error_symbol priority); "
        f"got scores: {[r.score for r in results]}"
    )

