"""Tests for context deduplication + packing (Increment 6).

Verifies that:
1. Duplicate chunk_ids collapse to one; the higher-scored entry is kept.
2. Overlapping file spans (path, start_line, end_line) are merged.
3. The token budget is strictly respected.
4. Higher-scored chunks are preferred when filling the budget.
5. PackingResult carries all six telemetry metric fields.
6. A budget of 0 returns an empty result without raising.
7. chunk_id, backend, and score are preserved on each packed chunk.
"""

from __future__ import annotations

import pytest

from rag.backends.base import BackendName, SearchResult


# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------


def _make_result(
    chunk_id: str,
    score: float = 1.0,
    token_count: int = 10,
    backend: str = "chroma",
    path: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    content: str = "sample text",
) -> SearchResult:
    return SearchResult(
        backend=BackendName.CHROMA,
        chunk_id=chunk_id,
        score=score,
        content=content,
        token_count=token_count,
        metadata={},
        path=path,
        start_line=start_line,
        end_line=end_line,
    )


# ---------------------------------------------------------------------------
# Test 1 — dedupe by chunk_id
# ---------------------------------------------------------------------------


def test_dedupe_by_chunk_id():
    """Duplicate chunk_ids must collapse to one; the higher-scored entry survives."""
    from rag.dedupe import dedupe_search_results

    low = _make_result("c1", score=0.5, token_count=10)
    high = _make_result("c1", score=0.9, token_count=10)

    deduped, tokens_saved = dedupe_search_results([low, high])

    assert len(deduped) == 1, "Two entries with the same chunk_id must collapse to one"
    assert deduped[0].score == 0.9, "The higher-scored entry must be kept"
    assert tokens_saved == 10, "The lower entry's token_count must be counted as saved"


# ---------------------------------------------------------------------------
# Test 2 — dedupe by overlapping file span
# ---------------------------------------------------------------------------


def test_dedupe_by_overlapping_file_span():
    """Results with overlapping (path, start_line, end_line) must be merged."""
    from rag.dedupe import dedupe_search_results

    r1 = _make_result("c1", score=0.6, token_count=20, path="src/foo.py", start_line=1, end_line=10)
    r2 = _make_result("c2", score=0.8, token_count=20, path="src/foo.py", start_line=8, end_line=15)
    # Non-overlapping result on a different path — must survive untouched.
    r3 = _make_result("c3", score=0.7, token_count=15, path="src/bar.py", start_line=1, end_line=5)

    deduped, tokens_saved = dedupe_search_results([r1, r2, r3])

    # r1 and r2 overlap → merged into one; r3 untouched → 2 results total.
    assert len(deduped) == 2, f"Expected 2 results after overlap merge, got {len(deduped)}"
    assert tokens_saved > 0, "Merging overlapping spans must save tokens"

    # Find the merged result for src/foo.py
    foo_result = next(r for r in deduped if r.path == "src/foo.py")
    # The merged result must cover the widest span.
    assert foo_result.start_line == 1
    assert foo_result.end_line == 15
    # Must keep the higher score.
    assert foo_result.score == 0.8


# ---------------------------------------------------------------------------
# Test 3 — packer respects token budget
# ---------------------------------------------------------------------------


def test_packer_respects_token_budget():
    """Packed output must not exceed the specified token budget."""
    from rag.context_packer import ContextPacker

    packer = ContextPacker()
    # 100 chunks × 200 tokens each = 20 000 total tokens
    results = [_make_result(f"c{i}", score=float(i), token_count=200) for i in range(100)]

    budget = 8_000
    packed_result = packer.pack(results, budget=budget)

    assert packed_result.packed_tokens <= budget, (
        f"packed_tokens={packed_result.packed_tokens} exceeds budget={budget}"
    )


# ---------------------------------------------------------------------------
# Test 4 — packer prefers higher score within budget
# ---------------------------------------------------------------------------


def test_packer_prefers_higher_score_within_budget():
    """When budget is tight the highest-scored chunks must be selected first."""
    from rag.context_packer import ContextPacker

    packer = ContextPacker()

    low_score = [_make_result(f"low{i}", score=0.1, token_count=50) for i in range(10)]
    high_score = [_make_result(f"high{i}", score=0.9, token_count=50) for i in range(5)]

    # Budget fits exactly the 5 high-score chunks (5 × 50 = 250 tokens).
    results = low_score + high_score
    packed_result = packer.pack(results, budget=250)

    chunk_ids = {c.chunk_id for c in packed_result.chunks}
    expected_ids = {f"high{i}" for i in range(5)}
    assert chunk_ids == expected_ids, (
        "Only high-scored chunks should be selected when budget is tight"
    )


# ---------------------------------------------------------------------------
# Test 5 — packer reports token savings metrics
# ---------------------------------------------------------------------------


def test_packer_reports_tokens_saved_metrics():
    """PackingResult must expose all six telemetry metric fields."""
    from rag.context_packer import ContextPacker, PackingResult

    packer = ContextPacker()

    # Introduce a duplicate to generate non-zero savings.
    dup_a = _make_result("c1", score=0.5, token_count=10)
    dup_b = _make_result("c1", score=0.9, token_count=10)  # same chunk_id
    other = _make_result("c2", score=0.7, token_count=10)

    result = packer.pack([dup_a, dup_b, other], budget=100)

    # All six required fields must be present.
    assert hasattr(result, "raw_tokens"), "PackingResult must have raw_tokens"
    assert hasattr(result, "deduped_tokens"), "PackingResult must have deduped_tokens"
    assert hasattr(result, "packed_tokens"), "PackingResult must have packed_tokens"
    assert hasattr(result, "tokens_saved"), "PackingResult must have tokens_saved"
    assert hasattr(result, "tokens_saved_pct"), "PackingResult must have tokens_saved_pct"
    assert hasattr(result, "duplicate_rate"), "PackingResult must have duplicate_rate"

    # The duplicate must inflate raw_tokens above deduped_tokens.
    assert result.raw_tokens == 30  # 3 inputs × 10 tokens
    assert result.deduped_tokens == 20  # 2 unique chunks × 10 tokens
    assert result.tokens_saved > 0
    assert 0.0 <= result.tokens_saved_pct <= 1.0
    assert 0.0 <= result.duplicate_rate <= 1.0


# ---------------------------------------------------------------------------
# Test 6 — packer handles zero budget gracefully
# ---------------------------------------------------------------------------


def test_packer_handles_zero_budget_gracefully():
    """A budget of 0 must return an empty chunk list without raising."""
    from rag.context_packer import ContextPacker

    packer = ContextPacker()
    results = [_make_result(f"c{i}", token_count=10) for i in range(5)]

    packed_result = packer.pack(results, budget=0)

    assert packed_result.chunks == [], "Zero budget must yield no chunks"
    assert packed_result.packed_tokens == 0


# ---------------------------------------------------------------------------
# Test 7 — packer preserves chunk metadata for telemetry
# ---------------------------------------------------------------------------


def test_packer_preserves_chunk_metadata_for_telemetry():
    """Each packed chunk must preserve backend, chunk_id, and score."""
    from rag.context_packer import ContextPacker

    packer = ContextPacker()
    results = [
        _make_result("chunk_alpha", score=0.95, token_count=10),
        _make_result("chunk_beta", score=0.80, token_count=10),
    ]

    packed_result = packer.pack(results, budget=100)

    assert len(packed_result.chunks) == 2

    by_id = {c.chunk_id: c for c in packed_result.chunks}

    assert by_id["chunk_alpha"].backend == BackendName.CHROMA
    assert by_id["chunk_alpha"].score == 0.95

    assert by_id["chunk_beta"].backend == BackendName.CHROMA
    assert by_id["chunk_beta"].score == 0.80

