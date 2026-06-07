"""Performance mode configuration; read from RAG_PERFORMANCE_MODE env var."""
import os


def load_mode() -> str:
    """Return the active performance mode (balanced | quiet | throughput)."""
    return os.getenv("RAG_PERFORMANCE_MODE", "balanced")

