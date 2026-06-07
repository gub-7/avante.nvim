"""Tests for freshness signal computation."""

from __future__ import annotations

import pytest


def test_node_modules_is_stale(fake_repo):
    """Files inside node_modules/ must appear in stale_uris."""
    from rag.freshness import compute_freshness

    nm = fake_repo / "node_modules"
    nm.mkdir()
    (nm / "foo.js").write_text("module.exports = {};")

    stale, _recent = compute_freshness(fake_repo)
    stale_paths = [u for u in stale if "node_modules" in u]
    assert len(stale_paths) >= 1, f"Expected node_modules to be stale, got stale={stale}"


def test_regular_file_not_stale(fake_repo):
    """Regular source files should NOT appear in stale_uris."""
    from rag.freshness import compute_freshness

    stale, _recent = compute_freshness(fake_repo)
    main_uri = (fake_repo / "src" / "main.py").as_uri()
    assert main_uri not in stale

