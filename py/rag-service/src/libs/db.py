import sqlite3
from collections.abc import Generator
from contextlib import contextmanager

from libs.configs import DB_FILE

# SQLite table schemas
CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS indexing_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uri TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    error_message TEXT,
    document_id TEXT,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_uri ON indexing_history(uri);
CREATE INDEX IF NOT EXISTS idx_document_id ON indexing_history(document_id);
CREATE INDEX IF NOT EXISTS idx_content_hash ON indexing_history(content_hash);

CREATE TABLE IF NOT EXISTS resources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    uri TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL,  -- 'path' or 'https'
    status TEXT NOT NULL DEFAULT 'active',  -- 'active' or 'inactive'
    indexing_status TEXT NOT NULL DEFAULT 'pending',  -- 'pending', 'indexing', 'indexed', 'failed'
    indexing_status_message TEXT,
    indexing_started_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_indexed_at DATETIME,
    last_error TEXT
);

CREATE INDEX IF NOT EXISTS idx_resources_name ON resources(name);
CREATE INDEX IF NOT EXISTS idx_resources_uri ON resources(uri);
CREATE INDEX IF NOT EXISTS idx_resources_status ON resources(status);
CREATE INDEX IF NOT EXISTS idx_status ON indexing_history(status);

-- Symbols extracted by tree-sitter during indexing
CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_uri TEXT NOT NULL,
    file_uri TEXT NOT NULL,
    symbol_name TEXT NOT NULL,
    symbol_kind TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    language TEXT,
    text_hash TEXT,
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(symbol_name);
CREATE INDEX IF NOT EXISTS idx_symbols_file_uri ON symbols(file_uri);
CREATE INDEX IF NOT EXISTS idx_symbols_resource_uri ON symbols(resource_uri);

-- Per-project profile cache
CREATE TABLE IF NOT EXISTS project_profiles (
    resource_uri TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    profile_hash TEXT NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Chat history (denormalised; embeddings live in Chroma)
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_uri TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    message_idx INTEGER NOT NULL,
    role TEXT NOT NULL,
    content_sanitized TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    token_estimate INTEGER NOT NULL,
    title TEXT,
    timestamp TEXT,
    UNIQUE(resource_uri, chat_id, message_idx)
);
CREATE INDEX IF NOT EXISTS idx_chat_resource ON chat_history(resource_uri);
CREATE INDEX IF NOT EXISTS idx_chat_chat_id ON chat_history(resource_uri, chat_id);

-- Cached hardware profile (host-submitted preferred)
CREATE TABLE IF NOT EXISTS hardware_profile (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    profile_json TEXT NOT NULL,
    source TEXT NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Cached benchmark results keyed by (hw_hash, backend, model, context)
CREATE TABLE IF NOT EXISTS benchmark_cache (
    hw_hash TEXT NOT NULL,
    backend TEXT NOT NULL,
    model TEXT NOT NULL,
    context_tokens INTEGER NOT NULL,
    result_json TEXT NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (hw_hash, backend, model, context_tokens)
);

-- Canonical document registry (one row per ingested file/URL)
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    uri TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    metadata_json TEXT,
    indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_documents_uri ON documents(uri);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);

-- Canonical chunk store (source of truth for all text content)
-- Vector backends store only chunk_id + score; full text lives here.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    metadata_json TEXT,
    path TEXT,
    start_line INTEGER,
    end_line INTEGER,
    indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);

-- Schema version sentinel
CREATE TABLE IF NOT EXISTS _schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR REPLACE INTO _schema_meta(key, value) VALUES ('version', '3');
"""


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Get a database connection."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Initialize the SQLite database."""
    with get_db_connection() as conn:
        conn.executescript(CREATE_TABLES_SQL)
        conn.commit()
