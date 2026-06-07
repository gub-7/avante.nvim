"""FastAPI router for /api/v1/runtime — hardware profile, backend recommendation, and benchmarks."""
from fastapi import APIRouter
from pydantic import BaseModel

from models.runtime import BackendRecommendation, BenchmarkResult, HardwareProfile
from runtime.benchmark import run as run_bench
from runtime.hardware_profile import get_or_probe, save_profile
from runtime.router import recommend

router = APIRouter(prefix="/api/v1/runtime", tags=["runtime"])


# ---------------------------------------------------------------------------
# Phase 11 — Hardware profile
# ---------------------------------------------------------------------------


@router.get("/profile", response_model=HardwareProfile)
async def get_runtime_profile() -> HardwareProfile:
    """Return the cached hardware profile (or run an in-container probe if absent)."""
    return get_or_probe()


@router.post("/profile", response_model=HardwareProfile)
async def submit_runtime_profile(profile: HardwareProfile) -> HardwareProfile:
    """Accept a host-derived hardware profile and persist it as the preferred source."""
    save_profile(profile, "host")
    return profile


# ---------------------------------------------------------------------------
# Phase 12 — Backend recommendation
# ---------------------------------------------------------------------------


@router.get("/recommend", response_model=BackendRecommendation)
async def runtime_recommend(prefer_vendor: str = "auto") -> BackendRecommendation:
    """Return an advisory backend recommendation based on the current hardware profile.

    The recommendation is purely advisory — the service never spawns LLM backends.
    """
    return recommend(prefer_vendor)


# ---------------------------------------------------------------------------
# Phase 13 — Benchmark
# ---------------------------------------------------------------------------


class BenchmarkRequest(BaseModel):
    backend: str
    model: str
    endpoint: str
    context_tokens: int = 2048
    prompt: str | None = None


@router.post("/benchmark", response_model=BenchmarkResult)
async def runtime_benchmark(req: BenchmarkRequest) -> BenchmarkResult:
    """Run (or return cached) streaming benchmark against a local OpenAI-compatible endpoint."""
    return run_bench(req.backend, req.model, req.endpoint, req.context_tokens, req.prompt)

