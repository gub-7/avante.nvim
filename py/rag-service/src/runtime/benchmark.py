"""Benchmarks a local OpenAI-compatible inference endpoint and caches results."""
import json
import shutil
import subprocess
import time
from urllib.request import Request, urlopen

from libs.db import get_db_connection
from libs.logger import logger
from models.runtime import BenchmarkResult
from runtime.hardware_profile import get_or_probe, hw_hash


def _peak_vram_bytes() -> int | None:
    """Query nvidia-smi for current peak VRAM usage; returns None if unavailable."""
    rg = shutil.which("nvidia-smi")
    if not rg:
        return None
    try:
        p = subprocess.run(
            [rg, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        return max(int(x.strip()) for x in p.stdout.splitlines()) * 1024 * 1024
    except (ValueError, subprocess.SubprocessError, OSError):
        return None


def _load_cached(
    h: str, backend: str, model: str, context_tokens: int
) -> BenchmarkResult | None:
    """Return a cached BenchmarkResult if one exists for the given key."""
    with get_db_connection() as conn:
        row = conn.execute(
            """SELECT result_json FROM benchmark_cache
               WHERE hw_hash = ? AND backend = ? AND model = ? AND context_tokens = ?""",
            (h, backend, model, context_tokens),
        ).fetchone()
    if not row:
        return None
    return BenchmarkResult.model_validate_json(row["result_json"])


def run(
    backend: str,
    model: str,
    endpoint: str,
    context_tokens: int,
    prompt: str | None = None,
) -> BenchmarkResult:
    """Run a streaming chat-completion benchmark against a local endpoint.

    Results are cached in benchmark_cache keyed by (hw_hash, backend, model, context_tokens).
    Cached results are returned immediately on subsequent identical calls.

    Args:
        backend: Name of the backend being tested (e.g. "ollama", "llama.cpp").
        model: Model identifier string (e.g. "llama3:8b").
        endpoint: Base URL of the OpenAI-compatible API (e.g. "http://localhost:11434/v1").
        context_tokens: Context window size used during the run.
        prompt: Optional custom prompt; defaults to a short greeting.

    Returns:
        A BenchmarkResult (cached or freshly measured).
    """
    profile = get_or_probe()
    h = hw_hash(profile)

    cached = _load_cached(h, backend, model, context_tokens)
    if cached:
        logger.info(
            "benchmark cache hit for backend=%s model=%s context=%s",
            backend,
            model,
            context_tokens,
        )
        return cached

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt or "Say hello briefly."}],
        "stream": True,
        "max_tokens": 64,
    }).encode()

    req = Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    ttft: float | None = None
    tokens_out = 0

    try:
        with urlopen(req, timeout=60) as resp:
            for chunk in resp:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                # Heuristic: count SSE newlines as a rough token proxy
                tokens_out += chunk.count(b"\n")
    except Exception:  # noqa: BLE001
        logger.warning(
            "benchmark request failed for backend=%s model=%s", backend, model
        )
        result = BenchmarkResult(
            backend=backend,
            model=model,
            context_tokens=context_tokens,
            prompt_eval_tps=0.0,
            decode_tps=0.0,
            ttft_ms=0.0,
            **{"pass": False},
        )
        return result

    total_ms = (time.perf_counter() - t0) * 1000
    decode_tps = tokens_out / max(0.001, (total_ms - (ttft or 0)) / 1000)

    result = BenchmarkResult(
        backend=backend,
        model=model,
        context_tokens=context_tokens,
        prompt_eval_tps=0.0,
        decode_tps=decode_tps,
        ttft_ms=ttft or 0.0,
        peak_vram_bytes=_peak_vram_bytes(),
        cpu_percent=0.0,
        **{"pass": decode_tps > 0.5},
    )

    with get_db_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO benchmark_cache
               (hw_hash, backend, model, context_tokens, result_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                h,
                backend,
                model,
                context_tokens,
                result.model_dump_json(by_alias=True),
            ),
        )
        conn.commit()

    logger.info(
        "benchmark complete for backend=%s model=%s decode_tps=%.2f",
        backend,
        model,
        decode_tps,
    )
    return result

