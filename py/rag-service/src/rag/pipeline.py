"""RAG retrieval pipeline — Increment 8.

``RetrievalPipeline`` is the single coordinating entry-point for all retrieval
requests.  It replaces the direct ``HybridRetriever()`` calls in
``src/api/rag.py`` and implements the full routing + fallback + telemetry loop
described in the Phase 1 TDD.

Execution order
---------------
1. Build a ``RouteRequest`` from the incoming ``RetrievalQuery``.
2. Call ``choose_backend_v1`` to get a ``RouteDecision``.
3. Dispatch to the selected backend (from the ``backends`` dict).
   - If the backend raises → record the error, fall back to Qdrant.
   - If no backend is registered under the decided name AND a
     ``hybrid_retriever`` is provided → delegate entirely to it.
4. Convert ``SearchResult`` objects → ``FileSpan`` objects.
5. Dedupe (by chunk_id + overlapping file spans).
6. Rerank with the existing multi-signal reranker.
7. Pack into the token budget with ``ContextPacker``.
8. Write telemetry via ``TelemetrySink`` (best-effort; never raises).
9. Return ``RetrievedContext``.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from models.rag import (
    ContextCitation,
    FileSpan,
    RetrievalQuery,
    RetrievedContext,
    SourceDocumentCompat,
)
from rag.backends.base import BackendName, SearchRequest, SearchResult
from rag.context_budget import BUDGETS, estimate_tokens
from rag.context_packer import ContextPacker
from rag.dedupe import dedupe_and_merge
from rag.reranker import rerank
from rag.router import RouteRequest, SystemStats, choose_backend_v1

if TYPE_CHECKING:
    from observability.telemetry_db import TelemetrySink
    from rag.hybrid_retriever import HybridRetriever


# ---------------------------------------------------------------------------
# Live GPU stats sampling — cached with a short TTL
# ---------------------------------------------------------------------------

_STATS_TTL_S: float = 5.0  # re-sample at most every 5 seconds
_stats_cache: SystemStats | None = None
_stats_ts: float = 0.0

_NVIDIA_SMI_FREE_RE = re.compile(r"(\d+)")
_ROCM_USE_RE = re.compile(r"GPU\[(\d+)\].*?GPU\s+use\s*\(%\)\s*:\s*(\d+)", re.IGNORECASE)
_ROCM_VRAM_FREE_RE = re.compile(
    r"GPU\[(\d+)\].*?VRAM\s+Total\s+Memory\s*\(B\)\s*:\s*(\d+)", re.IGNORECASE
)
_ROCM_VRAM_USED_RE = re.compile(
    r"GPU\[(\d+)\].*?VRAM\s+Total\s+Used\s+Memory\s*\(B\)\s*:\s*(\d+)", re.IGNORECASE
)


def _run_cmd(cmd: list[str], timeout: int = 3) -> str:
    """Run a CLI command and return stdout; returns '' on any error."""
    binary = shutil.which(cmd[0])
    if not binary:
        return ""
    try:
        p = subprocess.run(
            [binary, *cmd[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return p.stdout
    except (subprocess.SubprocessError, OSError):
        return ""


def _sample_nvidia() -> tuple[float, float]:
    """Return (total_free_vram_mb, avg_util_pct) across all NVIDIA GPUs."""
    out = _run_cmd([
        "nvidia-smi",
        "--query-gpu=memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    free_total = 0.0
    utils: list[float] = []
    for row in out.splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 2:
            continue
        try:
            free_total += float(parts[0])
            utils.append(float(parts[1]))
        except ValueError:
            continue
    avg_util = sum(utils) / len(utils) if utils else 0.0
    return free_total, avg_util


def _sample_amd() -> tuple[float, float]:
    """Return (total_free_vram_mb, avg_util_pct) across all AMD GPUs via rocm-smi."""
    vram_out = _run_cmd(["rocm-smi", "--showmeminfo", "vram"])
    use_out = _run_cmd(["rocm-smi", "--showuse"])

    total: dict[int, int] = {}
    used: dict[int, int] = {}
    for line in vram_out.splitlines():
        mt = _ROCM_VRAM_FREE_RE.search(line)
        if mt:
            total[int(mt.group(1))] = int(mt.group(2))
        mu = _ROCM_VRAM_USED_RE.search(line)
        if mu:
            used[int(mu.group(1))] = int(mu.group(2))

    free_total_mb = sum(
        (total[i] - used.get(i, 0)) / (1024 * 1024)
        for i in total
    )

    utils: list[float] = []
    for line in use_out.splitlines():
        m = _ROCM_USE_RE.search(line)
        if m:
            utils.append(float(m.group(2)))
    avg_util = sum(utils) / len(utils) if utils else 0.0

    return free_total_mb, avg_util


def _sample_system_stats() -> SystemStats:
    """Return a live-sampled SystemStats, cached for _STATS_TTL_S seconds.

    Tries NVIDIA first; falls back to AMD if nvidia-smi is absent.
    Returns a zeroed SystemStats on any failure so routing is safe.
    """
    global _stats_cache, _stats_ts
    now = time.monotonic()
    if _stats_cache is not None and (now - _stats_ts) < _STATS_TTL_S:
        return _stats_cache

    free_mb, util_pct = 0.0, 0.0
    try:
        if shutil.which("nvidia-smi"):
            free_mb, util_pct = _sample_nvidia()
        elif shutil.which("rocm-smi"):
            free_mb, util_pct = _sample_amd()
    except Exception:  # noqa: BLE001 — best-effort, never interrupt routing
        pass

    _stats_cache = SystemStats(gpu_vram_free_mb=free_mb, gpu_util_pct=util_pct)
    _stats_ts = now
    return _stats_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_to_span(result: SearchResult) -> FileSpan:
    """Convert a backend ``SearchResult`` into a ``FileSpan`` for the API."""
    text = result.text or result.content or ""
    token_est = result.token_count if result.token_count > 0 else estimate_tokens(text)
    content_hash = hashlib.sha256(text.encode()).hexdigest()

    backend_val = result.backend
    backend_str = backend_val.value if isinstance(backend_val, BackendName) else str(backend_val)

    return FileSpan(
        uri=result.metadata.get("uri", f"chunk://{result.chunk_id}"),
        path=result.path or result.metadata.get("path"),
        start_line=result.start_line or result.metadata.get("start_line"),
        end_line=result.end_line or result.metadata.get("end_line"),
        content=text,
        reason=f"{backend_str}:{result.chunk_id}",
        score=result.score,
        token_estimate=token_est,
        hash=content_hash,
        retrieval_sources=[backend_str],
        chunk_kind=result.metadata.get("chunk_kind"),
        language=result.metadata.get("language"),
    )


def _query_to_route_request(req: RetrievalQuery) -> RouteRequest:
    """Extract routing-relevant fields from a ``RetrievalQuery``."""
    mode_val = req.mode
    search_mode_str = mode_val.value if hasattr(mode_val, "value") else str(mode_val)
    return RouteRequest(
        query=req.query,
        requested_backend="",
        search_mode=search_mode_str,
        filter_count=0,
        batch_size=len(req.changed_files) if req.changed_files else 1,
        collection="",
        shadow=req.shadow,
    )


# ---------------------------------------------------------------------------
# RetrievalPipeline
# ---------------------------------------------------------------------------


class RetrievalPipeline:
    """Router → backend dispatch → fallback → dedupe → rerank → pack → telemetry.

    Parameters
    ----------
    backends:
        Mapping of backend name strings (``"qdrant"``, ``"exact"``, ``"milvus"``)
        to ``RagBackend`` instances.
    telemetry:
        Optional ``TelemetrySink``; best-effort writes, never raises into callers.
    hybrid_retriever:
        Optional legacy ``HybridRetriever`` used when no registered backend
        matches the router's decision (backward-compat fallback for api/rag.py).
    collection:
        Default collection name forwarded to backend ``search()`` calls.
    token_budget:
        Hard token budget override; when ``None`` uses the mode-based budget.
    """

    def __init__(
        self,
        backends: dict[str, Any],
        telemetry: "TelemetrySink | None" = None,
        hybrid_retriever: "HybridRetriever | None" = None,
        collection: str = "default",
        token_budget: int | None = None,
        forced_backend: str | None = None,
    ) -> None:
        self._backends = backends
        self._telemetry = telemetry
        self._hybrid = hybrid_retriever
        self._collection = collection
        self._token_budget = token_budget
        self._packer = ContextPacker()
        # When set, bypasses the router — used in tests to force a specific backend.
        self._forced_backend = forced_backend

    def run(self, req: RetrievalQuery) -> RetrievedContext:
        """Execute the full retrieval pipeline and return a ``RetrievedContext``."""
        t0 = time.perf_counter()
        request_id = str(uuid4())

        # 1. Route (or use forced override for testing)
        if self._forced_backend is not None:
            chosen_name = self._forced_backend
        else:
            sys_stats = _sample_system_stats()
            decision = choose_backend_v1(_query_to_route_request(req), sys_stats)
            chosen_name = decision.primary.value

        # 2. Dispatch + fallback
        spans, backend_runs = self._dispatch(req, chosen_name)

        # 3. Dedupe
        deduped, _ = dedupe_and_merge(spans)

        # 4. Rerank
        ranked_spans = [s for s, _ in rerank(deduped, req)]

        # 5. Pack
        budget = self._resolve_budget(req)
        packing_result = self._packer.pack(
            [
                SearchResult(
                    chunk_id=s.hash,
                    score=s.score,
                    backend=BackendName.QDRANT,
                    text=s.content,
                    token_count=s.token_estimate,
                    path=s.path,
                    start_line=s.start_line,
                    end_line=s.end_line,
                )
                for s in ranked_spans
            ],
            budget=budget,
        )
        packed_ids = {c.chunk_id for c in packing_result.chunks}
        packed_spans = [s for s in ranked_spans if s.hash in packed_ids]

        # 6. Telemetry
        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        self._write_telemetry(
            request_id=request_id,
            req=req,
            chosen_name=chosen_name,
            backend_runs=backend_runs,
            spans=spans,
            packing_result=packing_result,
            total_latency_ms=total_latency_ms,
        )

        # 7. Build response
        citations = [
            ContextCitation(
                uri=s.uri, path=s.path, start_line=s.start_line, end_line=s.end_line,
                reason=s.reason, retrieval_sources=s.retrieval_sources,
            )
            for s in packed_spans
        ]
        sources = [
            SourceDocumentCompat(uri=s.uri, content=s.content, score=s.score)
            for s in packed_spans
        ]
        return RetrievedContext(
            spans=packed_spans,
            sources=sources,
            citations=citations,
            token_estimate=packing_result.packed_tokens,
            trace_id=request_id,
            request_id=request_id,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        req: RetrievalQuery,
        chosen_name: str,
    ) -> tuple[list[FileSpan], list[dict]]:
        """Dispatch to the chosen backend; fall back to qdrant on failure."""
        backend = self._backends.get(chosen_name)
        backend_runs: list[dict] = []

        if backend is None:
            if self._hybrid is not None:
                ctx = self._hybrid.retrieve(req)
                return list(ctx.spans), []
            return [], []

        search_req = SearchRequest(
            query=req.query, collection=self._collection, top_k=req.top_k
        )

        t0 = time.perf_counter()
        error: str | None = None
        results: list[SearchResult] = []
        try:
            results = backend.search(search_req)
        except Exception as exc:
            error = str(exc)
        latency = (time.perf_counter() - t0) * 1000.0

        backend_runs.append({
            "run_id": str(uuid4()),
            "backend_name": chosen_name,
            "is_shadow": False,
            "is_primary": True,
            "latency_ms": latency,
            "result_count": len(results),
            "error": error,
        })

        if error is None:
            return [_result_to_span(r) for r in results], backend_runs

        # Primary failed → fall back to qdrant
        if chosen_name != "qdrant":
            qdrant = self._backends.get("qdrant")
            if qdrant is not None:
                t2 = time.perf_counter()
                fb_error: str | None = None
                fb_results: list[SearchResult] = []
                try:
                    fb_results = qdrant.search(search_req)
                except Exception as exc2:
                    fb_error = str(exc2)
                fb_latency = (time.perf_counter() - t2) * 1000.0

                backend_runs.append({
                    "run_id": str(uuid4()),
                    "backend_name": "qdrant",
                    "is_shadow": False,
                    "is_primary": False,
                    "latency_ms": fb_latency,
                    "result_count": len(fb_results),
                    "error": fb_error,
                })

                if fb_error is None:
                    return [_result_to_span(r) for r in fb_results], backend_runs

        return [], backend_runs

    def _resolve_budget(self, req: RetrievalQuery) -> int:
        if self._token_budget is not None:
            return self._token_budget
        if req.max_context_tokens:
            return req.max_context_tokens
        wm = getattr(req, "workflow_mode", None) or "ask"
        if hasattr(wm, "value"):
            wm = wm.value
        return BUDGETS.get(str(wm), BUDGETS["ask"])["max_total_tokens"]

    def _write_telemetry(
        self, *, request_id: str, req: RetrievalQuery, chosen_name: str,
        backend_runs: list[dict], spans: list[FileSpan], packing_result: Any,
        total_latency_ms: float,
    ) -> None:
        if self._telemetry is None:
            return
        sink = self._telemetry
        wm = getattr(req, "workflow_mode", None) or "ask"
        if hasattr(wm, "value"):
            wm = wm.value
        try:
            sink.record_request(
                request_id=request_id, query=req.query, mode=str(wm),
                base_uri=req.base_uri, chosen_backend=chosen_name, is_shadow=False,
                retrieval_latency_ms=total_latency_ms,
                parent_request_id=getattr(req, "parent_request_id", None),
            )
        except Exception:
            pass
        for run in backend_runs:
            try:
                sink.record_backend_run(
                    run_id=run["run_id"], request_id=request_id,
                    backend_name=run["backend_name"],
                    is_shadow=bool(run.get("is_shadow", False)),
                    is_primary=bool(run.get("is_primary", True)),
                    top_k=req.top_k, latency_ms=run["latency_ms"],
                    result_count=run["result_count"], error=run.get("error"),
                )
            except Exception:
                pass
        if spans:
            try:
                run_id = backend_runs[0]["run_id"] if backend_runs else str(uuid4())
                sink.record_results(
                    request_id=request_id, run_id=run_id,
                    results=[
                        {
                            "chunk_id": s.hash, "score": s.score,
                            "backend_name": s.retrieval_sources[0] if s.retrieval_sources else chosen_name,
                            "rank": i,
                        }
                        for i, s in enumerate(spans)
                    ],
                )
            except Exception:
                pass
        try:
            sink.record_packing(
                request_id=request_id,
                raw_tokens=packing_result.raw_tokens,
                deduped_tokens=packing_result.deduped_tokens,
                packed_tokens=packing_result.packed_tokens,
                tokens_saved=packing_result.tokens_saved,
                tokens_saved_pct=packing_result.tokens_saved_pct,
                duplicate_rate=packing_result.duplicate_rate,
            )
        except Exception:
            pass
