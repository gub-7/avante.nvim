"""JSONL filesystem sink for RAG traces.

Each call to :func:`write_trace` appends a single JSON line to a
daily-rotating file under ``${DATA_DIR}/traces/``.
"""

from __future__ import annotations

import json
from datetime import datetime

from libs.configs import BASE_DATA_DIR

TRACE_DIR = BASE_DATA_DIR / "traces"
TRACE_DIR.mkdir(parents=True, exist_ok=True)


def write_trace(obj: dict) -> None:
    """Append *obj* as a single JSON line to today's trace file.

    The file name is ``rag-YYYYMMDD.jsonl``.  The directory is created
    on import; writing is a best-effort append — failures are silently
    swallowed so that a broken trace sink never interrupts retrieval.

    Args:
        obj: Serialisable dict produced by :meth:`RagTrace.to_dict`.
    """
    try:
        fname = TRACE_DIR / f"rag-{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        line = json.dumps(obj, ensure_ascii=False, default=str)
        with fname.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

