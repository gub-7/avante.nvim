"""Resource management endpoints: add, remove, list."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException
from libs.logger import logger
from libs.utils import is_local_uri, is_remote_uri, uri_to_path
from models.resource import Resource
from pydantic import BaseModel, Field
from rag.engine import index_local_resource_async, index_remote_resource_async, watched_resources
from rag.remote_fetch import is_remote_resource_exists
from rag.watcher import FileSystemHandler
from services.resource import resource_service
from watchdog.observers import Observer

router = APIRouter(prefix="/api/v1", tags=["resources"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ResourceURIRequest(BaseModel):
    """Request model for resource operations."""

    uri: str = Field(..., description="URI of the resource to watch and index")


class ResourceRequest(ResourceURIRequest):
    """Request model for adding a resource."""

    name: str = Field(..., description="Name of the resource to watch and index")


class ResourceListResponse(BaseModel):
    """Response model for listing resources."""

    resources: list[Resource] = Field(..., description="List of all resources")
    total_count: int = Field(..., description="Total number of resources")
    status_summary: dict[str, int] = Field(
        ...,
        description="Summary of resource statuses (count by status)",
    )


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.post(
    "/add_resource",
    response_model="dict[str, str]",
    summary="Add a resource for watching and indexing",
    description="Adds a resource to the watch list and starts indexing all existing documents in it asynchronously.",
    responses={
        200: {"description": "Resource successfully added and indexing started"},
        404: {"description": "Resource not found"},
        400: {"description": "Resource already being watched"},
    },
)
async def add_resource(request: ResourceRequest, background_tasks: BackgroundTasks):  # noqa: ANN201, C901
    """Add a local or remote resource and kick off indexing."""
    # If resource already active, return success immediately
    resource = resource_service.get_resource(request.uri)
    if resource and resource.status == "active":
        return {
            "status": "success",
            "message": f"Resource {request.uri} added and indexing started in background",
        }

    resource_type = "local"
    background_task = index_local_resource_async  # default; overridden below

    if is_local_uri(request.uri):
        directory = uri_to_path(request.uri)
        if not directory.exists():
            raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")

        if not directory.is_dir():
            raise HTTPException(status_code=400, detail=f"{directory} is not a directory")

        git_directory = directory / ".git"
        if not git_directory.exists() or not git_directory.is_dir():
            raise HTTPException(status_code=400, detail=f"{git_directory} ia not a git repository")

        # Start file-system watcher
        event_handler = FileSystemHandler(directory=directory)
        observer = Observer()
        observer.schedule(event_handler, str(directory), recursive=True)
        observer.start()
        watched_resources[request.uri] = observer

        background_task = index_local_resource_async

    elif is_remote_uri(request.uri):
        if not is_remote_resource_exists(request.uri):
            raise HTTPException(status_code=404, detail="web resource not found")

        resource_type = "remote"
        background_task = index_remote_resource_async

    else:
        raise HTTPException(status_code=400, detail=f"Invalid URI: {request.uri}")

    if resource:
        if resource.name != request.name:
            raise HTTPException(
                status_code=400,
                detail=f"Resource name cannot be changed: {resource.name}",
            )
        resource_service.update_resource_status(resource.uri, "active")
    else:
        exists_resource = resource_service.get_resource_by_name(request.name)
        if exists_resource:
            raise HTTPException(status_code=400, detail="Resource with same name already exists")

        resource = Resource(
            id=None,
            name=request.name,
            uri=request.uri,
            type=resource_type,
            status="active",
            indexing_status="pending",
            indexing_status_message=None,
            indexing_started_at=None,
            last_indexed_at=None,
            last_error=None,
        )
        resource_service.add_resource_to_db(resource)
        background_tasks.add_task(background_task, resource)

    return {
        "status": "success",
        "message": f"Resource {request.uri} added and indexing started in background",
    }


@router.post(
    "/remove_resource",
    response_model="dict[str, str]",
    summary="Remove a watched resource",
    description="Stops watching and indexing the specified resource.",
    responses={
        200: {"description": "Resource successfully removed from watch list"},
        404: {"description": "Resource not found in watch list"},
    },
)
async def remove_resource(request: ResourceURIRequest):  # noqa: ANN201
    """Remove a resource from the watch list."""
    resource = resource_service.get_resource(request.uri)
    if not resource or resource.status != "active":
        raise HTTPException(status_code=404, detail="Resource not being watched")

    if request.uri in watched_resources:
        observer = watched_resources[request.uri]
        observer.stop()
        observer.join()
        del watched_resources[request.uri]

    resource_service.update_resource_status(request.uri, "inactive")

    return {"status": "success", "message": f"Resource {request.uri} removed"}


@router.get(
    "/resources",
    response_model=ResourceListResponse,
    summary="List all resources",
    description="Returns a list of all resources that have been added to the system.",
    responses={
        200: {"description": "Successfully retrieved resource list"},
    },
)
async def list_resources() -> ResourceListResponse:
    """Get all resources and their current status."""
    resources = resource_service.get_all_resources()

    status_counts: dict[str, int] = {}
    for resource in resources:
        status_counts[resource.status] = status_counts.get(resource.status, 0) + 1

    return ResourceListResponse(
        resources=resources,
        total_count=len(resources),
        status_summary=status_counts,
    )

