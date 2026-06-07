"""Load and persist RAG evaluation cases from a JSONL file.

Cases are stored in ``${DATA_DIR}/evals/rag_cases.jsonl`` — one JSON object
per line.  The file is created on first use.  The file path is resolved
lazily so that tests can override ``DATA_DIR`` before the first call.
"""

from __future__ import annotations

from pathlib import Path

from models.evals import RagEvalCase


def _eval_file() -> Path:
    """Return the path to the eval cases JSONL file (resolved at call time)."""
    # Import here so that DATA_DIR overrides in tests take effect
    from libs.configs import BASE_DATA_DIR  # noqa: PLC0415

    path = BASE_DATA_DIR / "evals" / "rag_cases.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load() -> list[RagEvalCase]:
    """Read all eval cases from disk.

    Returns:
        Ordered list of :class:`~models.evals.RagEvalCase` objects,
        or an empty list when the file does not exist.
    """
    path = _eval_file()
    if not path.exists():
        return []
    out: list[RagEvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(RagEvalCase.model_validate_json(line))
    return out


def append(case: RagEvalCase) -> None:
    """Append a single eval case to the JSONL file.

    Args:
        case: The :class:`~models.evals.RagEvalCase` to persist.
    """
    with _eval_file().open("a", encoding="utf-8") as f:
        f.write(case.model_dump_json() + "\n")

