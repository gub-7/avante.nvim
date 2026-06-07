"""Health-check endpoints."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/api/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@router.get("/api/v1/readyz")
async def readiness_probe() -> dict[str, str]:
    """Readiness probe endpoint."""
    return {"status": "ok"}

