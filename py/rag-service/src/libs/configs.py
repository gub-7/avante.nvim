import os
import tomllib
from pathlib import Path

# Configuration
BASE_DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
CHROMA_PERSIST_DIR = BASE_DATA_DIR / "chroma_db"
LOG_DIR = BASE_DATA_DIR / "logs"
DB_FILE = BASE_DATA_DIR / "sqlite" / "indexing_history.db"

# Configure directories
BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE.parent.mkdir(parents=True, exist_ok=True)  # Create sqlite directory
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# Optional TOML configuration file
RAG_TOML = BASE_DATA_DIR / "rag-service.toml"

# External vector-store connection URLs (optional — leave unset for local/dev use)
QDRANT_URL: str | None = os.environ.get("QDRANT_URL")
QDRANT_API_KEY: str | None = os.environ.get("QDRANT_API_KEY")
MILVUS_URL: str | None = os.environ.get("MILVUS_URL")


def load_toml() -> dict:
    """Load and return the rag-service.toml config; returns an empty dict if absent or invalid."""
    if not RAG_TOML.exists():
        return {}
    try:
        return tomllib.loads(RAG_TOML.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}

# Qdrant vector store settings
QDRANT_URL: str = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY: str = os.environ.get("QDRANT_API_KEY", "")

# Milvus vector store settings (reserved for Phase 1 Increment 11)
MILVUS_URL: str = os.environ.get("MILVUS_URL", "")
