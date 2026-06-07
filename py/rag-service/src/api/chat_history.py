"""Chat history API router.

Provides endpoints for upserting, deleting, and purging chat-history turns.
Each operation is mirrored to both the SQLite service layer and the
per-resource Chroma vector index.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from libs.logger import logger
from models.chat_history import ChatTurnUpsert
from rag.chat_history_index import ChatHistoryIndex
from services import chat_history as chat_svc

router = APIRouter(prefix="/api/v1/chat-history", tags=["chat-history"])

# Module-level singleton — shared with hybrid retriever via api.rag
_idx = ChatHistoryIndex()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class DeleteRequest(BaseModel):
    """Request body for the delete endpoint."""

    base_uri: str
    chat_id: str


class PurgeRequest(BaseModel):
    """Request body for the purge endpoint."""

    base_uri: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/upsert")
async def upsert(turn: ChatTurnUpsert) -> dict:
    """Persist a complete chat turn (all messages) and embed it.

    Replaces any existing messages stored under the same
    ``(base_uri, chat_id)`` pair.

    Args:
        turn: A :class:`~models.chat_history.ChatTurnUpsert` payload
            containing the resource URI, chat identifier, and ordered
            message list.

    Returns:
        ``{"status": "ok", "messages_indexed": <int>}``
    """
    n = chat_svc.upsert(turn)
    _idx.upsert(
        turn.base_uri,
        turn.chat_id,
        [{"role": m.role, "content": m.content} for m in turn.messages],
    )
    return {"status": "ok", "messages_indexed": n}


@router.post("/delete")
async def delete(req: DeleteRequest) -> dict:
    """Delete all messages for a single chat turn.

    Args:
        req: Identifies the resource and chat turn to remove.

    Returns:
        ``{"status": "ok", "rows_deleted": <int>}``
    """
    n = chat_svc.delete(req.base_uri, req.chat_id)
    return {"status": "ok", "rows_deleted": n}


@router.post("/purge")
async def purge(req: PurgeRequest) -> dict:
    """Delete ALL chat history for a resource (SQLite + Chroma).

    Args:
        req: Identifies the resource whose chat history to erase.

    Returns:
        ``{"status": "ok", "rows_deleted": <int>}``
    """
    n = chat_svc.purge(req.base_uri)
    _idx.purge(req.base_uri)
    return {"status": "ok", "rows_deleted": n}


# ---------------------------------------------------------------------------
# Legacy underscore-route aliases (backward compat, deprecated)
# ---------------------------------------------------------------------------


@router.post("/chat_history/upsert", include_in_schema=False)
async def _legacy_upsert(turn: ChatTurnUpsert) -> dict:
    logger.warning("deprecated underscore route hit: /chat_history/upsert")
    return await upsert(turn)


@router.post("/chat_history/delete", include_in_schema=False)
async def _legacy_delete(req: DeleteRequest) -> dict:
    logger.warning("deprecated underscore route hit: /chat_history/delete")
    return await delete(req)


@router.post("/chat_history/purge", include_in_schema=False)
async def _legacy_purge(req: PurgeRequest) -> dict:
    logger.warning("deprecated underscore route hit: /chat_history/purge")
    return await purge(req)

