"""Indexing-status endpoint: POST /api/v1/indexing-status."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from libs.logger import logger
from libs.utils import is_local_uri, uri_to_path
from pydantic import BaseModel, Field
from rag.engine import watched_resources
from services.indexing_history import indexing_history_service

if TYPE_CHECKING:
    from models.indexing_history import IndexingHistory

router = APIRouter(prefix="/api/v1", tags=["indexing"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class IndexingStatusRequest(BaseModel):
    """Request model for indexing status."""

    uri: str = Field(..., description="URI of the resource to get indexing status for")


class IndexingStatusResponse(BaseModel):
    """Model for indexing status response."""

    uri: str = Field(..., description="URI of the resource being monitored")
    is_watched: bool = Field(..., description="Whether the directory is currently being watched")
    files: list[IndexingHistory] = Field(..., description="List of files and their indexing status")
    total_files: int = Field(..., description="Total number of files processed in this directory")
    status_summary: dict[str, int] = Field(
        ...,
        description="Summary of indexing statuses (count by status)",
    )


# ---------------------------------------------------------------------------
# Shared handler logic
# ---------------------------------------------------------------------------


async def _handle_indexing_status(request: IndexingStatusRequest):  # noqa: ANN202
    """Core implementation shared by both route aliases."""
    if is_local_uri(request.uri):
        directory = uri_to_path(request.uri).resolve()
        if not directory.exists():
            raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")

    resource_files = indexing_history_service.get_indexing_status(base_uri=request.uri)

    logger.info("Found %d files in resource %s", len(resource_files), request.uri)
    for file in resource_files:
        logger.debug("File status: %s - %s", file.uri, file.status)

    status_counts: dict[str, int] = {}
    for file in resource_files:
        status_counts[file.status] = status_counts.get(file.status, 0) + 1

    return IndexingStatusResponse(
        uri=request.uri,
        is_watched=request.uri in watched_resources,
        files=resource_files,
        total_files=len(resource_files),
        status_summary=status_counts,
    )


# ---------------------------------------------------------------------------
# Routes (canonical hyphen + deprecated underscore alias)
# ---------------------------------------------------------------------------


@router.post(
    "/indexing-status",
    response_model=IndexingStatusResponse,
    summary="Get indexing status for a resource",
    description=(
        "Returns the current indexing status for all files in the specified resource, including:\n"
        "* Whether the resource is being watched\n"
        "* Status of each file in the resource"
    ),
    responses={
        200: {"description": "Successfully retrieved indexing status"},
        404: {"description": "Resource not found"},
    },
)
async def get_indexing_status_for_resource(request: IndexingStatusRequest):  # noqa: ANN201
    """Canonical hyphenated route."""
    return await _handle_indexing_status(request)


@router.post(
    "/indexing_status",
    response_model=IndexingStatusResponse,
    include_in_schema=False,  # hide from docs — deprecated alias
)
async def get_indexing_status_for_resource_underscore(request: IndexingStatusRequest):  # noqa: ANN201
    """Deprecated underscore alias — use /indexing-status instead."""
    logger.warning(
        "Deprecated endpoint /api/v1/indexing_status called; use /api/v1/indexing-status instead."
    )
    return await _handle_indexing_status(request)

