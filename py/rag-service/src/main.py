"""RAG Service API for managing document indexing and retrieval."""  # noqa: INP001

from __future__ import annotations

import fcntl
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from libs.configs import BASE_DATA_DIR
from libs.db import init_db
from libs.logger import logger
from libs.utils import is_local_uri, is_remote_uri, uri_to_path
from rag.engine import (
    index_local_resource_async,
    index_remote_resource_async,
    init_engine,
    watched_resources,
)
from rag.remote_fetch import is_remote_resource_exists
from rag.watcher import FileSystemHandler
from services.resource import resource_service
from watchdog.observers import Observer

from api import health, indexing_status, resources, retrieve
from api import rag as rag_router
from api import runtime as runtime_router
from api import evals as evals_router
from api import chat_history as chat_history_router

LOCK_FILE = BASE_DATA_DIR / "leader.lock"


def try_acquire_leadership() -> bool:
    """Try to acquire leadership using file lock (non-blocking exclusive flock)."""
    try:
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOCK_FILE.touch(exist_ok=True)
        lock_fd = os.open(str(LOCK_FILE), os.O_RDWR)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.truncate(lock_fd, 0)
        os.write(lock_fd, str(os.getpid()).encode())
        return True
    except OSError:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
    """Initialise services on startup; clean up on shutdown."""
    is_leader = try_acquire_leadership()

    if is_leader:
        logger.info("Starting RAG service as leader (PID: %d)...", os.getpid())
        init_db()
        init_engine()

        active_resources = [r for r in resource_service.get_all_resources() if r.status == "active"]
        logger.info("Found %d active resources to sync", len(active_resources))

        for resource in active_resources:
            try:
                if is_local_uri(resource.uri):
                    directory = uri_to_path(resource.uri)
                    if not directory.exists():
                        error_msg = f"Directory not found: {directory}"
                        logger.error(error_msg)
                        resource_service.update_resource_status(resource.uri, "error", error_msg)
                        continue

                    event_handler = FileSystemHandler(directory=directory)
                    observer = Observer()
                    observer.schedule(event_handler, str(directory), recursive=True)
                    observer.start()
                    watched_resources[resource.uri] = observer
                    await index_local_resource_async(resource)

                elif is_remote_uri(resource.uri):
                    if not is_remote_resource_exists(resource.uri):
                        error_msg = "HTTPS resource not found"
                        logger.error("%s: %s", error_msg, resource.uri)
                        resource_service.update_resource_status(resource.uri, "error", error_msg)
                        continue
                    await index_remote_resource_async(resource)

                logger.debug("Successfully synced resource: %s", resource.uri)

            except (OSError, ValueError, RuntimeError) as e:
                error_msg = f"Failed to sync resource {resource.uri}: {e}"
                logger.exception(error_msg)
                resource_service.update_resource_status(resource.uri, "error", error_msg)

    yield

    if is_leader:
        for observer in watched_resources.values():
            observer.stop()
            observer.join()


app = FastAPI(
    title="RAG Service API",
    description="RAG (Retrieval-Augmented Generation) Service API for managing document indexing and retrieval.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(resources.router)
app.include_router(retrieve.router)
app.include_router(indexing_status.router)
app.include_router(rag_router.router)
app.include_router(runtime_router.router)
app.include_router(evals_router.router)
app.include_router(chat_history_router.router)

