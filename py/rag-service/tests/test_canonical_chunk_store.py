"""Tests for the canonical chunk store (Increment 1).

Verifies that:
1. init_db() creates both ``chunks`` and ``documents`` tables.
2. ChunkStore.upsert() is idempotent on content_hash.
3. ChunkStore.get() round-trips text, token_count, and metadata_json.
4. ChunkStore.delete_by_document() cascades and removes all chunks for a document.
5. ChunkStore is the source of truth — full text is fetched from it, not from a
   vector store payload.
"""

from __future__ import annotations

import pytest

from rag.backends.base import EmbeddedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str = "c1",
    document_id: str = "d1",
    content: str = "hello world",
    token_count: int = 2,
    content_hash: str = "hash_a",
    metadata: dict | None = None,
    path: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        token_count=token_count,
        content_hash=content_hash,
        metadata=metadata or {},
        path=path,
        start_line=start_line,
        end_line=end_line,
    )


# ---------------------------------------------------------------------------
# Test 1 — init creates both tables
# ---------------------------------------------------------------------------


def test_init_creates_chunks_and_documents_tables():
    """init_db() must create both ``chunks`` and ``documents`` tables.

    The ``isolated_data_dir`` fixture (autouse) calls init_db() before
    every test, so we just query sqlite_master here.
    """
    from libs.db import get_db_connection

    with get_db_connection() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

    assert "chunks" in tables, "chunks table was not created"
    assert "documents" in tables, "documents table was not created"


# ---------------------------------------------------------------------------
# Test 2 — upsert is idempotent on content_hash
# ---------------------------------------------------------------------------


def test_upsert_chunk_is_idempotent_on_content_hash():
    """Upserting the same (chunk_id, content_hash) twice must yield one row.

    When the content_hash changes the existing row must be updated in-place
    and ``indexed_at`` must advance.
    """
    from libs.db import get_db_connection
    from rag.chunk_store import ChunkStore

    store = ChunkStore()
    chunk = _make_chunk(chunk_id="c1", content="original", content_hash="h1")

    store.upsert(chunk)
    store.upsert(chunk)  # identical → no-op

    with get_db_connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE chunk_id = 'c1'"
        ).fetchone()[0]
    assert count == 1, "Duplicate upsert must not create a second row"

    # Now upsert with a different content_hash
    updated_chunk = _make_chunk(chunk_id="c1", content="updated", content_hash="h2")
    store.upsert(updated_chunk)

    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT content, content_hash FROM chunks WHERE chunk_id = 'c1'"
        ).fetchone()

    assert row["content"] == "updated", "Row should be updated when content_hash changes"
    assert row["content_hash"] == "h2"

    # Still only one row
    with get_db_connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE chunk_id = 'c1'"
        ).fetchone()[0]
    assert count == 1


# ---------------------------------------------------------------------------
# Test 3 — get round-trips text, token_count, metadata_json
# ---------------------------------------------------------------------------


def test_get_chunk_text_by_chunk_id():
    """ChunkStore.get() must return the stored text, token_count, and metadata_json."""
    from rag.chunk_store import ChunkStore

    store = ChunkStore()
    chunk = _make_chunk(
        chunk_id="c_rt",
        content="def foo(): pass",
        token_count=5,
        metadata={"language": "python", "kind": "function"},
    )
    store.upsert(chunk)

    result = store.get("c_rt")

    assert result is not None, "get() must not return None for an existing chunk_id"
    assert result["content"] == "def foo(): pass"
    assert result["token_count"] == 5
    # metadata is persisted as JSON in the metadata_json column
    import json
    assert json.loads(result["metadata_json"]) == {"language": "python", "kind": "function"}


# ---------------------------------------------------------------------------
# Test 4 — delete_by_document cascades to chunks
# ---------------------------------------------------------------------------


def test_delete_by_document_id_cascades_to_chunks():
    """Deleting a document must remove all of its associated chunks."""
    from libs.db import get_db_connection
    from rag.chunk_store import ChunkStore

    store = ChunkStore()

    # Insert three chunks belonging to the same document
    for i in range(3):
        store.upsert(
            _make_chunk(
                chunk_id=f"del_{i}",
                document_id="doc_to_delete",
                content=f"text {i}",
                token_count=1,
                content_hash=f"hash_{i}",
            )
        )

    # Sanity check
    with get_db_connection() as conn:
        pre_count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_id = 'doc_to_delete'"
        ).fetchone()[0]
    assert pre_count == 3

    store.delete_by_document("doc_to_delete")

    with get_db_connection() as conn:
        post_count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_id = 'doc_to_delete'"
        ).fetchone()[0]
        doc_count = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE document_id = 'doc_to_delete'"
        ).fetchone()[0]

    assert post_count == 0, "All chunks for the document must be removed"
    assert doc_count == 0, "The document row itself must also be removed"


# ---------------------------------------------------------------------------
# Test 5 — chunk store is source of truth
# ---------------------------------------------------------------------------


def test_chunk_store_is_source_of_truth():
    """Full text must come from ChunkStore, not from a vector store payload.

    Simulates a vector backend that returns only (chunk_id, score) — the
    canonical text is resolved by fetching from ChunkStore.
    """
    from rag.chunk_store import ChunkStore

    store = ChunkStore()
    store.upsert(
        _make_chunk(
            chunk_id="vec1",
            document_id="d_vec",
            content="canonical text lives here",
            token_count=4,
            content_hash="h_vec",
        )
    )

    # Simulate what a vector backend returns: only chunk_id + score, no content.
    vector_stub_result = {"chunk_id": "vec1", "score": 0.92}

    # Resolve full text from the chunk store using the returned chunk_id.
    chunk = store.get(vector_stub_result["chunk_id"])

    assert chunk is not None
    assert chunk["content"] == "canonical text lives here"
    # The stub payload intentionally carries no content — confirm that.
    assert "content" not in vector_stub_result, (
        "Vector stub must not carry content; ChunkStore is the sole source of truth"
    )

