"""Tests for the chat-history SQLite service and vector index."""

from __future__ import annotations

import pytest


def _make_turn(base_uri: str, chat_id: str = "chat-1"):
    from models.chat_history import ChatMessage, ChatTurnUpsert

    return ChatTurnUpsert(
        base_uri=base_uri,
        chat_id=chat_id,
        title="Test chat",
        project_root="/repo",
        messages=[
            ChatMessage(role="user", content="What is foo?", timestamp="2024-01-01T00:00:00Z"),
            ChatMessage(role="assistant", content="foo returns 1.", timestamp="2024-01-01T00:00:01Z"),
        ],
        updated_at="2024-01-01T00:00:01Z",
    )


def test_upsert_inserts_rows():
    """upsert() should insert one row per message."""
    from services.chat_history import list_recent, upsert

    turn = _make_turn("file:///repo")
    n = upsert(turn)
    assert n == 2

    rows = list_recent("file:///repo")
    assert len(rows) == 2


def test_upsert_replaces_existing():
    """Calling upsert twice should replace the existing rows, not append."""
    from services.chat_history import list_recent, upsert

    turn = _make_turn("file:///repo")
    upsert(turn)
    upsert(turn)  # second call replaces

    rows = list_recent("file:///repo")
    assert len(rows) == 2


def test_delete_removes_rows():
    """delete() should remove all rows for a given chat_id."""
    from services.chat_history import delete, list_recent, upsert

    turn = _make_turn("file:///repo")
    upsert(turn)
    n = delete("file:///repo", "chat-1")
    assert n == 2

    rows = list_recent("file:///repo")
    assert len(rows) == 0


def test_purge_removes_all():
    """purge() should remove ALL rows for a resource URI."""
    from services.chat_history import list_recent, purge, upsert

    upsert(_make_turn("file:///repo", "chat-1"))
    upsert(_make_turn("file:///repo", "chat-2"))
    n = purge("file:///repo")
    assert n == 4  # 2 messages × 2 chats

    assert list_recent("file:///repo") == []


def test_sanitize_strips_base64():
    """sanitize() should replace long base64 blobs with a placeholder."""
    from services.chat_history import sanitize

    b64 = "A" * 300 + "=="
    result = sanitize(f"Here is data: {b64} done")
    assert "<elided base64>" in result
    assert "A" * 300 not in result


def test_sanitize_truncates_long_content():
    """Content longer than PASTE_LIMIT chars should be truncated."""
    from services.chat_history import PASTE_LIMIT, sanitize

    long = "x" * (PASTE_LIMIT + 100)
    result = sanitize(long)
    assert len(result) < len(long)
    assert "<elided" in result

