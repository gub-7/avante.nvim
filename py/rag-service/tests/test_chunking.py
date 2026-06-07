"""Tests for the document chunking / splitting pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
from llama_index.core.schema import Document


@pytest.fixture
def python_doc(fake_repo) -> Document:
    """Return a LlamaIndex Document wrapping the fake_repo main.py."""
    src = fake_repo / "src" / "main.py"
    content = src.read_text()
    uri = src.as_uri()
    return Document(text=content, doc_id=uri, metadata={"uri": uri})


def test_split_documents_produces_chunks(python_doc):
    """split_documents should yield at least one chunk from a Python file."""
    from rag.chunking import split_documents

    chunks = split_documents([python_doc])
    assert len(chunks) >= 1


def test_split_documents_chunk_metadata(python_doc):
    """Each chunk produced from a code file must carry structural metadata."""
    from rag.chunking import split_documents

    chunks = split_documents([python_doc])
    for chunk in chunks:
        meta = chunk.metadata
        assert "start_line" in meta, "chunk must have start_line"
        assert "end_line" in meta, "chunk must have end_line"
        assert "chunk_kind" in meta, "chunk must have chunk_kind"
        assert "text_hash" in meta, "chunk must have text_hash"


def test_split_documents_text_hash_is_sha256(python_doc):
    """text_hash must be a 64-character hex string (SHA-256)."""
    import hashlib

    from rag.chunking import split_documents

    chunks = split_documents([python_doc])
    for chunk in chunks:
        h = chunk.metadata["text_hash"]
        assert len(h) == 64
        int(h, 16)  # must be valid hex

