"""Canonical chunk store — SQLite-backed source of truth for all text content.

Vector backends (Chroma, Qdrant, …) persist only ``chunk_id`` + score.
Callers retrieve the full text, token count, and metadata from here.

This module is pure CRUD — no business logic lives here.
"""

from __future__ import annotations

import json
from typing import Any

from libs.db import get_db_connection
from rag.backends.base import EmbeddedChunk


class ChunkStore:
    """CRUD interface over the ``chunks`` and ``documents`` SQLite tables.

    All methods are synchronous and operate within their own short-lived
    connection obtained from :func:`~libs.db.get_db_connection`.
    """

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert(self, chunk: EmbeddedChunk) -> None:
        """Insert or update a chunk; idempotent on ``content_hash``.

        Behaviour:
        - If ``(chunk_id, content_hash)`` already exists → no-op.
        - If ``chunk_id`` exists with a *different* ``content_hash`` → the
          row is updated and ``indexed_at`` is advanced to ``CURRENT_TIMESTAMP``.
        - If ``chunk_id`` is new → the document row is created first
          (``INSERT OR IGNORE``) and then the chunk row is inserted.

        Args:
            chunk: The :class:`~rag.backends.base.EmbeddedChunk` to persist.
        """
        metadata_json = json.dumps(chunk.metadata) if chunk.metadata else "{}"

        with get_db_connection() as conn:
            # Enable foreign key enforcement for this connection.
            conn.execute("PRAGMA foreign_keys = ON")

            # Ensure the parent document exists.
            conn.execute(
                """
                INSERT OR IGNORE INTO documents (document_id, uri, content_hash)
                VALUES (?, ?, ?)
                """,
                (chunk.document_id, chunk.document_id, chunk.content_hash),
            )

            # Check whether this exact (chunk_id, content_hash) already exists.
            existing = conn.execute(
                "SELECT content_hash FROM chunks WHERE chunk_id = ?",
                (chunk.chunk_id,),
            ).fetchone()

            if existing is not None and existing["content_hash"] == chunk.content_hash:
                # Identical content — nothing to do.
                conn.commit()
                return

            if existing is None:
                # New chunk — plain insert.
                conn.execute(
                    """
                    INSERT INTO chunks
                        (chunk_id, document_id, content_hash, content, token_count,
                         metadata_json, path, start_line, end_line)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.content_hash,
                        chunk.content,
                        chunk.token_count,
                        metadata_json,
                        chunk.path,
                        chunk.start_line,
                        chunk.end_line,
                    ),
                )
            else:
                # Same chunk_id but different content_hash — update.
                conn.execute(
                    """
                    UPDATE chunks
                    SET content_hash = ?,
                        content      = ?,
                        token_count  = ?,
                        metadata_json = ?,
                        path         = ?,
                        start_line   = ?,
                        end_line     = ?,
                        indexed_at   = CURRENT_TIMESTAMP
                    WHERE chunk_id = ?
                    """,
                    (
                        chunk.content_hash,
                        chunk.content,
                        chunk.token_count,
                        metadata_json,
                        chunk.path,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.chunk_id,
                    ),
                )

            conn.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, chunk_id: str) -> dict[str, Any] | None:
        """Return a single chunk row as a plain dict, or *None* if not found.

        Returned keys: ``chunk_id``, ``document_id``, ``content_hash``,
        ``text``, ``token_count``, ``metadata_json``, ``path``,
        ``start_line``, ``end_line``, ``indexed_at``.

        Args:
            chunk_id: The unique identifier of the chunk to retrieve.
        """
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            return dict(row) if row is not None else None

    def get_many(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """Return multiple chunk rows by their ids.

        Missing ids are silently omitted from the result.  The order of
        returned rows is not guaranteed to match the order of *chunk_ids*.

        Args:
            chunk_ids: List of chunk identifiers to retrieve.
        """
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        with get_db_connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
            return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------

    def delete_by_document(self, document_id: str) -> None:
        """Delete all chunks belonging to *document_id*, then the document.

        Because the ``chunks`` table has ``ON DELETE CASCADE`` referencing
        ``documents``, deleting the document row is sufficient to remove
        all its chunks.

        Args:
            document_id: The document whose chunks should be removed.
        """
        with get_db_connection() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (document_id,),
            )
            conn.commit()

    def delete_by_filter(
        self,
        *,
        path: str | None = None,
        document_id: str | None = None,
    ) -> None:
        """Delete chunks matching the given filter criteria.

        All supplied keyword arguments are combined with AND.  At least
        one filter must be provided.

        Args:
            path: If given, delete chunks whose ``path`` matches exactly.
            document_id: If given, delete chunks for this document only
                         (equivalent to :meth:`delete_by_document` but
                         without removing the parent document row).
        """
        conditions: list[str] = []
        params: list[Any] = []

        if path is not None:
            conditions.append("path = ?")
            params.append(path)
        if document_id is not None:
            conditions.append("document_id = ?")
            params.append(document_id)

        if not conditions:
            raise ValueError("delete_by_filter requires at least one filter argument")

        where_clause = " AND ".join(conditions)
        with get_db_connection() as conn:
            conn.execute(f"DELETE FROM chunks WHERE {where_clause}", params)
            conn.commit()

