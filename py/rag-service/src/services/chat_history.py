"""SQLite service layer for chat history persistence.

Manages the ``chat_history`` table — insert/replace rows per turn and
support bulk purge operations.  Chat messages are sanitized before storage
to remove tool payloads, base64 blobs, ANSI escape codes, and "Thinking..."
frames.
"""

from __future__ import annotations

import hashlib
import re

from libs.db import get_db_connection
from models.chat_history import ChatTurnUpsert
from rag.context_budget import estimate_tokens

# ---------------------------------------------------------------------------
# Sanitizer patterns
# ---------------------------------------------------------------------------

_TOOL_PAYLOAD_RE = re.compile(r"<tool_payload>.*?</tool_payload>", re.S)
_BASE64_RE = re.compile(r"\b[A-Za-z0-9+/]{200,}={0,2}\b")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_THINKING_RE = re.compile(r"^\s*Thinking\.\.\.\s*$", re.M)

PASTE_LIMIT = 4096


def sanitize(text: str) -> str:
    """Remove sensitive / noisy content from a chat message before storage.

    Strips tool payloads, large base64 blobs, ANSI escape codes, and
    "Thinking..." frames.  Truncates the result if it still exceeds
    ``PASTE_LIMIT`` bytes.

    Args:
        text: Raw message content.

    Returns:
        Cleaned message content.
    """
    t = _TOOL_PAYLOAD_RE.sub("<elided tool payload>", text)
    t = _BASE64_RE.sub("<elided base64>", t)
    t = _ANSI_RE.sub("", t)
    t = _THINKING_RE.sub("", t)
    if len(t) > PASTE_LIMIT:
        t = t[:PASTE_LIMIT] + f"\n<elided {len(t) - PASTE_LIMIT} bytes>"
    return t


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------


def upsert(turn: ChatTurnUpsert) -> int:
    """Replace all messages for a chat turn and persist them.

    Deletes any existing rows for ``(base_uri, chat_id)`` before inserting
    the new message batch.

    Args:
        turn: The complete :class:`~models.chat_history.ChatTurnUpsert`
            payload from the client.

    Returns:
        Number of rows inserted.
    """
    rows = []
    for i, m in enumerate(turn.messages):
        sanitized = sanitize(m.content)
        rows.append((
            turn.base_uri,
            turn.chat_id,
            i,
            m.role,
            sanitized,
            hashlib.sha256(sanitized.encode()).hexdigest(),
            estimate_tokens(sanitized),
            turn.title,
            m.timestamp,
        ))
    with get_db_connection() as conn:
        conn.execute(
            "DELETE FROM chat_history WHERE resource_uri = ? AND chat_id = ?",
            (turn.base_uri, turn.chat_id),
        )
        conn.executemany(
            """INSERT INTO chat_history
                   (resource_uri, chat_id, message_idx, role,
                    content_sanitized, content_hash, token_estimate, title, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
    return len(rows)


def delete(resource_uri: str, chat_id: str) -> int:
    """Delete all messages for a single chat turn.

    Args:
        resource_uri: Base URI identifying the code resource.
        chat_id: Identifier for the specific chat turn.

    Returns:
        Number of rows deleted.
    """
    with get_db_connection() as conn:
        cur = conn.execute(
            "DELETE FROM chat_history WHERE resource_uri = ? AND chat_id = ?",
            (resource_uri, chat_id),
        )
        conn.commit()
        return cur.rowcount


def purge(resource_uri: str) -> int:
    """Delete ALL chat history for a resource.

    Args:
        resource_uri: Base URI identifying the code resource.

    Returns:
        Number of rows deleted.
    """
    with get_db_connection() as conn:
        cur = conn.execute(
            "DELETE FROM chat_history WHERE resource_uri = ?",
            (resource_uri,),
        )
        conn.commit()
        return cur.rowcount


def list_recent(resource_uri: str, limit: int = 200) -> list[dict]:
    """Fetch recent chat history rows for a resource.

    Args:
        resource_uri: Base URI identifying the code resource.
        limit: Maximum number of rows to return.

    Returns:
        List of row dicts ordered by timestamp descending.
    """
    with get_db_connection() as conn:
        rows = conn.execute(
            """SELECT * FROM chat_history WHERE resource_uri = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (resource_uri, limit),
        ).fetchall()
    return [dict(r) for r in rows]

