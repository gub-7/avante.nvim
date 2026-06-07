"""Shadow-mode executor for the RAG dual-backend router (Increment 9).

``ShadowExecutor`` runs a primary backend and an optional shadow backend
concurrently.  Only the primary results are returned to the caller; the
shadow run is recorded in the telemetry DB with ``is_shadow=True``.

Overlap metrics (overlap@10, overlap@50, overlap@100) are computed inline
from the two result sets and persisted via the supplied ``TelemetrySink``.

Global kill-switch: set ``RAG_SHADOW_DISABLED=1`` in the environment to
unconditionally skip shadow execution without changing call sites.

Usage::

    executor = ShadowExecutor(sink=TelemetrySink(conn))
    primary_results, shadow_record = await executor.run(
        primary_backend=qdrant_backend,
        shadow_backend=milvus_backend,
        req=req,
    )
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ShadowExecutor:
    """Run primary and shadow backends concurrently and record telemetry.

    Args:
        sink: A :class:`observability.telemetry_db.TelemetrySink` instance, or
              ``None`` to skip all telemetry writes (useful in tests that are
              only concerned with the execution semantics).
    """

    def __init__(self, sink: Any | None) -> None:
        self._sink = sink

    async def run(
        self,
        *,
        primary_backend: Any,
        shadow_backend: Any | None,
        req: Any,
        force_shadow: bool = False,
    ) -> tuple[list[dict], dict]:
        """Execute *primary_backend* and optionally *shadow_backend* in parallel.

        Shadow execution is skipped when:
        - ``shadow_backend`` is ``None``, AND ``force_shadow`` is ``False``.
        - The environment variable ``RAG_SHADOW_DISABLED`` equals ``"1"``.
        - ``req.shadow`` is falsy AND ``force_shadow`` is ``False``.

        Args:
            primary_backend: Backend with an ``async search(req) -> list[dict]``
                             (or sync callable — both are handled).
            shadow_backend:  Optional shadow backend (same interface).
            req:             Request object; must expose ``query``, ``mode``,
                             ``base_uri``, and ``request_id`` attributes.
            force_shadow:    When ``True``, force shadow execution even if
                             ``req.shadow`` is falsy.

        Returns:
            A 2-tuple of:
            - ``primary_results``: the list of result dicts from the primary.
            - ``shadow_record``:   a dict with telemetry metadata for the shadow
              run (empty dict ``{}`` when shadow did not execute).
        """
        should_shadow = self._should_shadow(
            shadow_backend=shadow_backend,
            req=req,
            force_shadow=force_shadow,
        )

        if not should_shadow:
            primary_results = await _invoke(primary_backend, req)
            return primary_results, {}

        # Run both backends concurrently
        primary_task = asyncio.create_task(_invoke(primary_backend, req))
        shadow_task = asyncio.create_task(_invoke_timed(shadow_backend, req))

        primary_results, (shadow_results, shadow_latency_ms, shadow_error) = (
            await asyncio.gather(primary_task, shadow_task)
        )

        # Compute overlap metrics
        overlap = _compute_overlap(primary_results, shadow_results)

        request_id = getattr(req, "request_id", None) or uuid.uuid4().hex

        shadow_run_id = uuid.uuid4().hex
        shadow_record = {
            "run_id": shadow_run_id,
            "request_id": request_id,
            "backend_name": getattr(shadow_backend, "name", "shadow"),
            "is_shadow": True,
            "is_primary": False,
            "latency_ms": shadow_latency_ms,
            "result_count": len(shadow_results),
            "error": shadow_error,
            "overlap_at_10": overlap["overlap_at_10"],
            "overlap_at_50": overlap["overlap_at_50"],
            "overlap_at_100": overlap["overlap_at_100"],
        }

        # Persist to telemetry sink (best-effort)
        self._persist(
            req=req,
            request_id=request_id,
            primary_results=primary_results,
            shadow_record=shadow_record,
            shadow_results=shadow_results,
            overlap=overlap,
        )

        return primary_results, shadow_record

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _should_shadow(
        *,
        shadow_backend: Any | None,
        req: Any,
        force_shadow: bool,
    ) -> bool:
        """Return ``True`` if shadow execution should proceed."""
        if os.environ.get("RAG_SHADOW_DISABLED") == "1":
            return False
        if shadow_backend is None:
            return False
        if force_shadow:
            return True
        return bool(getattr(req, "shadow", False))

    def _persist(
        self,
        *,
        req: Any,
        request_id: str,
        primary_results: list[dict],
        shadow_record: dict,
        shadow_results: list[dict],
        overlap: dict,
    ) -> None:
        """Write shadow run telemetry.  All failures are silently swallowed."""
        if self._sink is None:
            return
        try:
            # Safely coerce request attributes to plain Python scalars so that
            # MagicMock test doubles don't cause SQLite type errors that would
            # be silently swallowed, hiding real overlap values.
            def _str(attr: str, default: str = "") -> str:
                val = getattr(req, attr, None)
                if val is None:
                    return default
                s = str(val) if not isinstance(val, str) else val
                # MagicMock str() produces "<MagicMock ...>"; fall back to default
                return s if not s.startswith("<") else default

            def _str_or_none(attr: str) -> str | None:
                val = getattr(req, attr, None)
                if val is None:
                    return None
                s = str(val) if not isinstance(val, str) else val
                return s if not s.startswith("<") else None

            # Record the retrieval request row (with overlap metrics)
            self._sink.record_request(
                request_id=request_id,
                parent_request_id=_str_or_none("parent_request_id"),
                query=_str("query"),
                mode=_str("mode"),
                base_uri=_str("base_uri"),
                chosen_backend=_str("chosen_backend", "unknown"),
                is_shadow=True,
                retrieval_latency_ms=shadow_record.get("latency_ms", 0.0),
                overlap_at_10=overlap["overlap_at_10"],
                overlap_at_50=overlap["overlap_at_50"],
                overlap_at_100=overlap["overlap_at_100"],
            )

            # Record the shadow backend run
            run_id = shadow_record["run_id"]
            self._sink.record_backend_run(
                run_id=run_id,
                request_id=request_id,
                backend_name=shadow_record["backend_name"],
                is_shadow=True,
                is_primary=False,
                top_k=len(shadow_results),
                latency_ms=shadow_record.get("latency_ms", 0.0),
                result_count=shadow_record.get("result_count", 0),
                error=shadow_record.get("error"),
            )

            if shadow_results:
                self._sink.record_results(
                    request_id=request_id,
                    run_id=run_id,
                    results=shadow_results,
                )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


async def _invoke(backend: Any, req: Any) -> list[dict]:
    """Invoke *backend.search(req)*, handling both sync and async callables."""
    try:
        result = backend.search(req)
        if asyncio.iscoroutine(result):
            return await result
        return result
    except Exception:
        return []


async def _invoke_timed(backend: Any, req: Any) -> tuple[list[dict], float, str | None]:
    """Invoke *backend.search(req)* and return ``(results, latency_ms, error)``."""
    t0 = time.perf_counter()
    error: str | None = None
    results: list[dict] = []
    try:
        result = backend.search(req)
        if asyncio.iscoroutine(result):
            results = await result
        else:
            results = result
    except Exception as exc:
        error = str(exc)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return results, latency_ms, error


def _compute_overlap(
    primary: list[dict],
    shadow: list[dict],
    ks: tuple[int, ...] = (10, 50, 100),
) -> dict[str, float | None]:
    """Compute overlap@k between *primary* and *shadow* result sets.

    Overlap@k is defined as ``|intersection of top-k| / k``, i.e. the
    fraction of the top-*k* slots that contain the same chunk in both
    backends.  This is the standard retrieval-comparison metric (not Jaccard).

    Args:
        primary: Primary backend result list (dicts with ``chunk_id`` key).
        shadow:  Shadow backend result list.
        ks:      The k values to compute overlap for.

    Returns:
        Dict with keys ``overlap_at_10``, ``overlap_at_50``, ``overlap_at_100``.
        Value is ``None`` when both result sets are empty.
    """
    out: dict[str, float | None] = {}
    for k in ks:
        key = f"overlap_at_{k}"
        p_ids = {r["chunk_id"] for r in primary[:k]}
        s_ids = {r["chunk_id"] for r in shadow[:k]}
        if not p_ids and not s_ids:
            out[key] = None
        else:
            # Denominator: k (or max results available if both have fewer than k)
            denom = max(len(p_ids), len(s_ids), 1)
            out[key] = len(p_ids & s_ids) / denom
    return out

