"""``rag bench`` — latency benchmark command for RAG backends (Increment 13).

Loads a JSON bench file (list of query objects), runs each query against every
configured backend, records telemetry for each run, computes p50/p95 latency
statistics per backend, and emits a machine-readable JSON summary suitable for
CI diffing.

Bench file format
-----------------
::

    {
      "queries": [
        {
          "query": "what does foo() return?",
          "base_uri": "/path/to/project",
          "mode": "ask",
          "top_k": 5
        },
        ...
      ]
    }

All fields except ``query`` are optional.

Usage
-----
Standalone::

    python -m cli.bench --bench-file scripts/bench_samples/bench.json

As a sub-command wired into the FastAPI app start-up (see ``src/main.py``)::

    # main.py registers the CLI via Typer / click; see wire_bench_command()
    python src/main.py bench --bench-file ...

Programmatic (for tests)::

    from cli.bench import BenchRunner, load_bench_file
    bench = load_bench_file("bench.json")
    runner = BenchRunner(backends=[my_backend], sink=my_sink)
    summary = runner.run(bench)
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from observability.telemetry_db import TelemetrySink
    from rag.backends.base import RagBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bench file schema
# ---------------------------------------------------------------------------


@dataclass
class BenchQuery:
    """A single query entry from a bench file.

    Attributes:
        query:    The free-text query string to execute.
        base_uri: URI of the project root (default: empty string).
        mode:     Workflow mode (``"ask"``, ``"search"``, …; default:
                  ``"ask"``).
        top_k:    Number of results to request (default: ``5``).
    """

    query: str
    base_uri: str = ""
    mode: str = "ask"
    top_k: int = 5


@dataclass
class BenchFile:
    """Parsed contents of a ``.json`` bench file.

    Attributes:
        queries: Ordered list of queries to execute against every backend.
        path:    Original file path (for error reporting; may be ``None`` when
                 constructed in-memory in tests).
    """

    queries: list[BenchQuery]
    path: Path | None = None


def load_bench_file(path: str | Path) -> BenchFile:
    """Parse a bench file from disk.

    Args:
        path: Path to the JSON bench file.

    Returns:
        A :class:`BenchFile` with all query entries parsed.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError:        If the file is not valid JSON or missing the
                           ``"queries"`` key.
    """
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"bench file not found: {p}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in bench file {p}: {exc}") from exc

    if "queries" not in data:
        raise ValueError(f"bench file {p} is missing required key 'queries'")

    queries: list[BenchQuery] = []
    for i, q in enumerate(data["queries"]):
        if not isinstance(q, dict) or "query" not in q:
            raise ValueError(
                f"bench file {p}: entry {i} must be a dict with a 'query' key"
            )
        queries.append(
            BenchQuery(
                query=q["query"],
                base_uri=q.get("base_uri", ""),
                mode=q.get("mode", "ask"),
                top_k=int(q.get("top_k", 5)),
            )
        )

    return BenchFile(queries=queries, path=p)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    """Return the *pct*-th percentile of *values*.

    Args:
        values: Non-empty list of floats.
        pct:    Percentile in [0, 100].

    Returns:
        The percentile value.  Returns ``0.0`` for an empty list.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    # nearest-rank method
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lower = int(rank)
    upper = lower + 1
    if upper >= len(sorted_vals):
        return sorted_vals[-1]
    frac = rank - lower
    return sorted_vals[lower] + frac * (sorted_vals[upper] - sorted_vals[lower])


# ---------------------------------------------------------------------------
# BenchRunner
# ---------------------------------------------------------------------------


class BenchRunner:
    """Execute a bench file against a set of RAG backends.

    For each backend that reports as available (``backend.is_available()``),
    every query in the bench file is executed via a
    :class:`~rag.pipeline.RetrievalPipeline`.  Latency is measured per-query
    and p50/p95 statistics are computed per-backend.

    Unavailable backends are **skipped** with a ``WARNING``-level log message
    so that a missing ``MILVUS_URL`` (or similar) never crashes the bench run.
    All other backends continue normally.

    Parameters
    ----------
    backends:
        List of :class:`~rag.backends.base.RagBackend` instances to benchmark.
        Each backend is tested independently against all queries.
    sink:
        Optional :class:`~observability.telemetry_db.TelemetrySink`.  When
        provided, one ``retrieval_requests`` row and one
        ``backend_search_runs`` row are written for every (backend, query)
        pair.
    collection:
        Default collection name forwarded to the pipeline (default:
        ``"default"``).
    """

    def __init__(
        self,
        backends: "list[RagBackend]",
        sink: "TelemetrySink | None" = None,
        collection: str = "default",
    ) -> None:
        self.backends = backends
        self.sink = sink
        self.collection = collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, bench: BenchFile) -> dict[str, Any]:
        """Run the full benchmark and return a summary dict.

        The returned dict has the following structure::

            {
              "queries_count": <int>,
              "backends": {
                "<backend_name>": {
                  "run_count": <int>,
                  "p50_ms":    <float>,
                  "p95_ms":    <float>,
                  "min_ms":    <float>,
                  "max_ms":    <float>,
                }
              }
            }

        Backends that were skipped (unavailable) are absent from the
        ``"backends"`` dict.

        Args:
            bench: Parsed :class:`BenchFile` produced by
                   :func:`load_bench_file`.

        Returns:
            A JSON-serialisable summary dict.
        """
        # Import here to avoid circular dependency at module load time.
        from rag.pipeline import RetrievalPipeline

        backend_results: dict[str, dict[str, Any]] = {}

        for backend in self.backends:
            name = str(getattr(backend, "name", repr(backend)))

            # --- availability check ----------------------------------------
            try:
                available = backend.is_available()
            except Exception as exc:
                logger.warning(
                    "backend %s raised during is_available() check (%s); skipping",
                    name,
                    exc,
                )
                available = False

            if not available:
                logger.warning(
                    "backend %s is not available; skipping bench runs",
                    name,
                )
                continue

            # --- run queries -------------------------------------------------
            pipeline = RetrievalPipeline(
                backends={name: backend},
                telemetry=self.sink,
                collection=self.collection,
                forced_backend=name,
            )
            latencies: list[float] = []

            for bq in bench.queries:
                import time as _time
                from models.rag import RetrievalQuery as _RQ
                req = _RQ(
                    query=bq.query,
                    base_uri=bq.base_uri or "file:///bench",
                    workflow_mode=bq.mode if bq.mode in ("ask","search","edit-small","test-fix","refactor") else "ask",
                    top_k=bq.top_k,
                )
                t0 = _time.perf_counter()
                pipeline.run(req)
                latencies.append((_time.perf_counter() - t0) * 1000.0)

            backend_results[name] = {
                "run_count": len(latencies),
                "p50_ms": _percentile(latencies, 50),
                "p95_ms": _percentile(latencies, 95),
                "min_ms": min(latencies) if latencies else 0.0,
                "max_ms": max(latencies) if latencies else 0.0,
            }

        return {
            "queries_count": len(bench.queries),
            "backends": backend_results,
        }

    def run_to_json(self, bench: BenchFile, indent: int = 2) -> str:
        """Run the benchmark and return the summary as a JSON string.

        Args:
            bench:  Parsed bench file.
            indent: JSON indentation level (default: ``2``).

        Returns:
            A pretty-printed JSON string ready for writing to a file or
            stdout.
        """
        summary = self.run(bench)
        return json.dumps(summary, indent=indent)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _build_default_backends() -> "list[RagBackend]":
    """Build the default list of backends from environment/config.

    Backends that require env vars (``MILVUS_URL``, ``QDRANT_URL``) are only
    included when those vars are set; otherwise they are silently omitted from
    the list.  The ``InMemoryBackend`` is **always** included so that a bare
    ``rag bench`` run without any external services still produces output.
    """
    import os

    from rag.backends.in_memory import InMemoryBackend

    backends: list[Any] = [InMemoryBackend()]

    # Chroma (local, always available if the library is installed)
    try:
        import chromadb

        from rag.backends.chroma import ChromaBackend

        backends.append(ChromaBackend(client=chromadb.EphemeralClient()))
    except Exception:
        pass

    # Qdrant (requires QDRANT_URL)
    if os.environ.get("QDRANT_URL"):
        try:
            from rag.backends.qdrant import QdrantBackend  # type: ignore[import]

            backends.append(QdrantBackend())
        except Exception as exc:
            logger.warning("failed to initialise QdrantBackend: %s", exc)

    # Milvus (requires MILVUS_URL)
    if os.environ.get("MILVUS_URL"):
        try:
            from rag.backends.milvus import MilvusBackend  # type: ignore[import]

            backends.append(MilvusBackend())
        except Exception as exc:
            logger.warning("failed to initialise MilvusBackend: %s", exc)

    return backends  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point for ``python -m cli.bench``.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="rag bench",
        description="Run latency benchmarks against all configured RAG backends.",
    )
    parser.add_argument(
        "--bench-file",
        required=True,
        metavar="PATH",
        help="Path to the JSON bench file.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Write the JSON summary to this file (default: print to stdout).",
    )
    parser.add_argument(
        "--collection",
        default="default",
        help="Collection name to query (default: 'default').",
    )

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    bench = load_bench_file(args.bench_file)
    backends = _build_default_backends()

    runner = BenchRunner(backends=backends, collection=args.collection)
    json_summary = runner.run_to_json(bench)

    if args.output:
        Path(args.output).write_text(json_summary, encoding="utf-8")
        print(f"Summary written to {args.output}")
    else:
        print(json_summary)


if __name__ == "__main__":
    main()

