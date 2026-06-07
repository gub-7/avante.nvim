# Implementation Guide — Hardware-Aware Agentic RAG Upgrade

> Companion to `py/rag-service/docs/RAG_UPGRADE_TDD.md`. This document is
> self-contained: a receiving agent should be able to execute each phase
> without opening any other file in the repository. All file paths, code
> snippets, line numbers, and patterns referenced by each phase are
> reproduced inline.

---

# GLOBAL CONTEXT

## G.1 Project layout (only the parts you'll touch)

```
avante3/
├── lua/avante/
│   ├── config.lua           # plugin-wide config; Config.rag_service lives at L361-L382
│   ├── path.lua             # chat history (History table); save hook at ~L179
│   ├── rag_service.lua      # RPC client to the Python sidecar; URL builders + add/remove/retrieve
│   └── utils/init.lua       # M.get_project_root() at L820 -> M.root.get()
├── tests/
│   └── rag_service_spec.lua # Busted spec for rag_service.lua
└── py/rag-service/
    ├── Dockerfile           # python:3.11-slim-bookworm; installs curl+git only; CMD uvicorn workers=3 :20250
    ├── requirements.txt     # pinned; uv pip install --system
    ├── run.sh               # nix-runner entry
    ├── shell.nix
    ├── docs/
    │   ├── RAG_UPGRADE_TDD.md
    │   └── IMPL_GUIDE.md    # <-- this file
    └── src/
        ├── main.py          # 1410 lines, monolith; refactored in Phase 0
        ├── libs/
        │   ├── configs.py   # BASE_DATA_DIR, CHROMA_PERSIST_DIR, LOG_DIR, DB_FILE
        │   ├── db.py        # CREATE_TABLES_SQL, get_db_connection(), init_db()
        │   ├── logger.py    # logger = logging.getLogger("libs.logger")
        │   └── utils.py     # uri_to_path/path_to_uri/get_node_uri/inject_uri_to_node
        ├── models/
        │   ├── resource.py
        │   └── indexing_history.py
        ├── providers/
        │   ├── factory.py   # initialize_embed_model / initialize_llm_model
        │   ├── openai.py
        │   ├── ollama.py
        │   ├── dashscope.py
        │   └── openrouter.py
        └── services/
            ├── resource.py            # resource_service singleton
            └── indexing_history.py    # indexing_history_service singleton
```

After this upgrade `src/` also contains `api/`, `rag/`, `runtime/`, `evals/`,
`observability/`, plus extended `models/`, and a new `tests/` directory.

## G.2 Key conventions (mirror these everywhere)

- **Python**: 3.11. Use modern typing (`str | None`, `list[...]`). No
  `from typing import Optional`. `__future__ annotations` import is allowed
  but optional in new files.
- **Logging**: `from libs.logger import logger` — never `logging.getLogger`.
  Format: `logger.info("msg %s", arg)`, never f-strings in log calls.
- **DB access**: always `with get_db_connection() as conn:` then
  `conn.execute(...); conn.commit()`. Row factory is already
  `sqlite3.Row` so rows are dict-like (`row["col"]`).
- **Pydantic**: v2 syntax. Use `BaseModel`, `Field(..., description=...)`.
  Top-of-file `from pydantic import BaseModel, Field`.
- **FastAPI**: routes go in `src/api/<name>.py` as
  `router = APIRouter(prefix="/api/v1", tags=["..."])`. `main.py` does
  `app.include_router(<name>.router)`.
- **URI conventions**: `file://...` for local; `http(s)://...` for remote.
  Use `is_local_uri / is_remote_uri / uri_to_path / path_to_uri` from
  `libs.utils`. Resources require `.git/` directory for local URIs (current
  validation in `add_resource`).
- **Threading**: indexing writes are protected by a module-level
  `threading.Lock()` named `index_lock` (currently in `main.py`). Move to
  `rag/engine.py` in Phase 0; reuse it from all writer paths.
- **Provider extras**: `RAG_*_EXTRA` env vars are JSON; decode with
  `json.loads(..., default={})` pattern (see `main.py` ~L335 region).
- **Lua HTTP**: `plenary.curl` for sync; `vim.system({"curl", ...}, {text=true}, cb)` for fire-and-forget. Always
  `Content-Type: application/json`, body via `vim.json.encode`. Outbound
  URIs go through `M.to_container_uri`, inbound through `M.to_local_uri`.

## G.3 Reference code snippets you will reuse in many phases

### G.3.1 `libs/configs.py` (verbatim; no changes in early phases)

```py
import os
from pathlib import Path

BASE_DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
CHROMA_PERSIST_DIR = BASE_DATA_DIR / "chroma_db"
LOG_DIR = BASE_DATA_DIR / "logs"
DB_FILE = BASE_DATA_DIR / "sqlite" / "indexing_history.db"

for d in (BASE_DATA_DIR, LOG_DIR, DB_FILE.parent, CHROMA_PERSIST_DIR):
    d.mkdir(parents=True, exist_ok=True)
```

### G.3.2 `libs/db.py` (current shape — you will append to `CREATE_TABLES_SQL`)

```py
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from libs.configs import DB_FILE

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS indexing_history ( ... );      -- §G.3.3
CREATE INDEX IF NOT EXISTS idx_uri ON indexing_history(uri);
CREATE INDEX IF NOT EXISTS idx_document_id ON indexing_history(document_id);
CREATE INDEX IF NOT EXISTS idx_content_hash ON indexing_history(content_hash);
CREATE TABLE IF NOT EXISTS resources ( ... );             -- §G.3.3
CREATE INDEX IF NOT EXISTS idx_resources_name ON resources(name);
CREATE INDEX IF NOT EXISTS idx_resources_uri ON resources(uri);
CREATE INDEX IF NOT EXISTS idx_resources_status ON resources(status);
CREATE INDEX IF NOT EXISTS idx_status ON indexing_history(status);
"""

@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db() -> None:
    with get_db_connection() as conn:
        conn.executescript(CREATE_TABLES_SQL)
        conn.commit()
```

### G.3.3 Existing DDL (you append, never modify)

```sql
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
CREATE TABLE IF NOT EXISTS resources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    uri TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL,                       -- 'path' or 'https'
    status TEXT NOT NULL DEFAULT 'active',    -- 'active' or 'inactive'
    indexing_status TEXT NOT NULL DEFAULT 'pending',
    indexing_status_message TEXT,
    indexing_started_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_indexed_at DATETIME,
    last_error TEXT
);
```

### G.3.4 `libs/utils.py` helpers (verbatim — use as-is)

```py
PATTERN_URI_PART = re.compile(r"(?P<uri>.+)__part_\d+")
METADATA_KEY_URI = "uri"

def is_local_uri(uri: str) -> bool:        return uri.startswith("file://")
def is_remote_uri(uri: str) -> bool:       return uri.startswith(("http://", "https://"))
def uri_to_path(uri: str) -> Path:         return Path(uri.removeprefix("file://"))
def path_to_uri(file_path: Path) -> str:   return file_path.as_uri()

def is_path_node(node) -> bool:
    uri = get_node_uri(node)
    return bool(uri) and is_local_uri(uri)

def get_node_uri(node) -> str | None:
    uri = node.metadata.get(METADATA_KEY_URI)
    if not uri:
        doc_id = getattr(node, "doc_id", None)
        if doc_id:
            m = PATTERN_URI_PART.match(doc_id)
            uri = m.group("uri") if m else doc_id
    if uri:
        if uri.startswith("/"):
            uri = f"file://{uri}"
        return uri
    return None

def inject_uri_to_node(node) -> None:
    if METADATA_KEY_URI in node.metadata:
        return
    uri = get_node_uri(node)
    if uri:
        node.metadata[METADATA_KEY_URI] = uri
```

### G.3.5 Existing `main.py` startup block (lines ~310-380, summarized)

```py
init_db()
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

rag_embed_provider = os.getenv("RAG_EMBED_PROVIDER", "openai")
rag_embed_endpoint = os.getenv("RAG_EMBED_ENDPOINT", "https://api.openai.com/v1")
rag_embed_model    = os.getenv("RAG_EMBED_MODEL", "text-embedding-3-large")
rag_embed_api_key  = os.getenv("RAG_EMBED_API_KEY", None)
rag_embed_extra    = os.getenv("RAG_EMBED_EXTRA", None)

rag_llm_provider   = os.getenv("RAG_LLM_PROVIDER", "openai")
rag_llm_endpoint   = os.getenv("RAG_LLM_ENDPOINT", "https://api.openai.com/v1")
rag_llm_model      = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")
rag_llm_api_key    = os.getenv("RAG_LLM_API_KEY", None)
rag_llm_extra      = os.getenv("RAG_LLM_EXTRA", None)

# Config-change detection -> chroma_client.reset()
config_file = BASE_DATA_DIR / "rag_config.json"
# ... reads prev, compares provider+embed_model, resets if changed ...

chroma_collection = chroma_client.get_or_create_collection("documents")
vector_store     = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context  = StorageContext.from_defaults(vector_store=vector_store)

embed_model = initialize_embed_model(rag_embed_provider, rag_embed_model,
                                     rag_embed_endpoint, rag_embed_api_key, embed_extra)
llm_model   = initialize_llm_model(rag_llm_provider, rag_llm_model,
                                   rag_llm_endpoint, rag_llm_api_key, llm_extra)
Settings.embed_model = embed_model
Settings.llm = llm_model

try:
    index = load_index_from_storage(storage_context)
except (OSError, ValueError):
    index = VectorStoreIndex([], storage_context=storage_context)
```

### G.3.6 Existing `code_ext_map` (verbatim from `main.py` L182-L205)

```py
code_ext_map = {
    ".py":"python", ".js":"javascript", ".ts":"typescript", ".jsx":"javascript",
    ".tsx":"typescript", ".vue":"vue", ".go":"go", ".java":"java", ".cpp":"cpp",
    ".c":"c", ".h":"cpp", ".rs":"rust", ".rb":"ruby", ".php":"php",
    ".scala":"scala", ".kt":"kotlin", ".swift":"swift", ".lua":"lua",
    ".pl":"perl", ".pm":"perl", ".t":"perl", ".pm6":"perl", ".m":"perl",
}
```

### G.3.7 Existing `required_exts` (verbatim from `main.py` L207-L256)

Used by `SimpleDirectoryReader(... required_exts=required_exts)`. Includes
`.txt .pdf .docx .xlsx .pptx .rst .json .ini .conf .toml .md .markdown .csv
.tsv .html .htm .xml .yaml .yml .css .scss .less .sass .styl .sh .bash .zsh
.fish .rb .java .go .ts .tsx .js .jsx .vue .py .php .c .cpp .h .rs .swift
.kt .lua .perl .pl .pm .t .pm6 .m`. Keep as-is in `rag/chunking.py`.

### G.3.8 Existing `process_document_batch` & `split_documents` (paraphrased)

`process_document_batch(documents)` (main.py L408-L495):
- Skip if `indexing_history_service.get_indexing_status(doc=doc)[0].status == "completed"`.
- Decode bytes if needed; reject if `is_valid_text(content)` False (printable ratio ≤ 0.95).
- Build `new_doc = Document(text=clean_text(content), doc_id=doc.doc_id, metadata=...)`,
  `inject_uri_to_node(new_doc)`.
- `indexing_history_service.update_indexing_status(doc, "indexing")` → on success
  `... "completed", metadata=doc.metadata` → on failure `... "failed", error_message=...`.
- The write to the vector store is `with index_lock: index.refresh_ref_docs(valid_documents)`.

`split_documents(documents)` (main.py L640-L695):
- For each `doc`, get `uri = get_node_uri(doc)`. If not a `file://` URI, append unchanged
  (with `metadata["orig_doc_id"] = doc.doc_id`).
- If extension in `code_ext_map`, call `CodeSplitter(language=..., chunk_lines=80,
  chunk_lines_overlap=15, max_chars=1500, parser=get_parser(lang))` and emit
  `Document(text=..., doc_id=f"{doc.doc_id}__part_{i}", metadata={..., "chunk_number": i,
  "total_chunks": len(texts), "language": ..., "orig_doc_id": doc.doc_id})`.
- Otherwise pass through.

### G.3.9 `FileSystemHandler` (main.py L431-L466)

```py
class FileSystemHandler(FileSystemEventHandler):
    def __init__(self, directory: Path) -> None: self.directory = directory
    def on_modified(self, event):  self._handle(event)
    def on_created(self, event):   self._handle(event)
    def _handle(self, event):
        if event.is_directory or str(event.src_path).endswith(".tmp"): return
        self.handle_file_change(Path(str(event.src_path)))
    def handle_file_change(self, file_path: Path) -> None:
        now = time.time()
        abs_path = file_path if file_path.is_absolute() else (self.directory / file_path)
        if abs_path in file_last_modified and now - file_last_modified[abs_path] < BATCH_PROCESSING_DELAY:
            return
        file_last_modified[abs_path] = now
        threading.Thread(target=update_index_for_file, args=(self.directory, abs_path)).start()
```

`BATCH_PROCESSING_DELAY = 1` (seconds). Globals: `file_last_modified: dict[Path, float]`,
`watched_resources: dict[str, BaseObserver]`.

### G.3.10 Existing `Config.rag_service` block (lua/avante/config.lua L361-L382)

```lua
rag_service = {
  enabled = false,
  host_mount = os.getenv("HOME"),
  runner = "docker",
  image = "quay.io/yetoneful/avante-rag-service:0.0.11",
  llm   = { provider="openai", endpoint="https://api.openai.com/v1",
            api_key="OPENAI_API_KEY", model="gpt-4o-mini", extra=nil },
  embed = { provider="openai", endpoint="https://api.openai.com/v1",
            api_key="OPENAI_API_KEY", model="text-embedding-3-large", extra=nil },
  docker_extra_args = "",
},
```

### G.3.11 Lua client RPC pattern (lua/avante/rag_service.lua)

```lua
-- Sync call:
local resp = curl.post(M.get_rag_service_url() .. "/api/v1/<route>", {
  headers = { ["Content-Type"] = "application/json" },
  body    = vim.json.encode({ ... }),
  timeout = 100000,
})
if resp.status ~= 200 then Utils.error("...", resp.body); return nil end
local jsn = vim.json.decode(resp.body)

-- Async fire-and-forget:
vim.system({ "curl", "-X", "POST", url, "-H", "Content-Type: application/json",
             "-d", vim.json.encode(body) }, { text = true }, function(out) ... end)
```

### G.3.12 Chat history save hook (lua/avante/path.lua ~L179)

```lua
function History.save(bufnr, history)
  local history_filepath = History.get_filepath(bufnr, history.filename)
  history_filepath:write(vim.json.encode(history), "w")
  History.save_latest_filename(bufnr, history.filename)
  -- INSERT chat-history-sync hook HERE  (Phase R)
end
```

`generate_project_dirname_in_storage(bufnr)` (L15) is the project key the Lua
side uses; you map it to the registered code resource URI before pushing.

## G.4 Resolved design decisions (binding)

1. **Chat history**: push from Lua → service. Stored as sub-namespace
   (separate Chroma collection `chat_<sha1(resource_uri)>`). Retention
   `last 50 chats OR 30 days` (whichever smaller). Strip tool payloads,
   base64, >4KB pastes, ANSI codes, "Thinking..." frames.
2. **Routes**: hyphens canonical. Server registers underscore aliases with
   deprecation log. Lua client updated to hyphens.
3. **Phasing**: all phases land in this delivery.
4. **Hardware**: best-effort in-container; host probe via
   `scripts/probe_hardware.py` submitted via `POST /api/v1/runtime/profile`.
   Backend recommendations are **advisory only** — service never spawns
   LLM backends.
5. **Symbol languages v1**: C, C++, C#, Python, Rust, JavaScript,
   TypeScript, Lua, Java, Go, PHP, Ruby, Swift. ripgrep + pciutils +
   vulkan-tools added to Dockerfile.

## G.5 Implementation order

```
Phase 0  Safe refactor of main.py
Phase 1  Dockerfile deps + new models + DB schema additions
Phase 2  Exact search + /api/v1/rag/search
Phase 3  Structural chunking metadata
Phase 4  Symbol index + /api/v1/rag/symbols
Phase 5  Dedupe + Reranker
Phase 6  Hybrid retriever + /api/v1/rag/retrieve + /api/v1/rag/context
Phase 7  Context budget + freshness + citations
Phase 8  Observability (JSONL trace)
Phase 9  Expansion + project profile
Phase 10 Log summarizer + sufficiency + agentic planner + /api/v1/rag/agentic-retrieve
Phase 11 Hardware probe + /api/v1/runtime/profile + scripts/probe_hardware.py
Phase 12 Hardware-aware budget + backend selector + VRAM + offload + multi-GPU + recommend
Phase 13 Benchmark + performance modes
Phase 14 Evals
Phase 15 Chat-history index + endpoints
Phase R  Lua client updates (parallel: hyphenated routes, new RPCs, chat-history sync)
Phase T  Tests
```

---

# PHASE 0 — Safe refactor of `main.py`

## Context

`main.py` is currently 1410 lines and contains: route handlers,
indexing, watchers, splitters, Chroma+LlamaIndex setup, provider init.

### Endpoint inventory to preserve (response shapes unchanged)

| Method | Path | Handler line in current main.py |
| --- | --- | --- |
| GET  | `/api/v1/readyz` | L887 |
| POST | `/api/v1/add_resource` | L894 |
| POST | `/api/v1/remove_resource` | L988 |
| POST | `/api/v1/retrieve` | L1019 |
| POST | `/api/v1/indexing-status` | L1212 |
| GET  | `/api/v1/resources` | L1289 |
| GET  | `/api/health` | L1404 |

### Pydantic models currently in `main.py`

- `ResourceURIRequest{ uri: str }` (L394)
- `ResourceRequest(ResourceURIRequest){ name: str }` (L400)
- `SourceDocument{ uri: str, content: str, score: float|None }` (L406)
- `RetrieveRequest{ query: str, base_uri: str, top_k: int|None=5 }` (L412)
- `RetrieveResponse{ response: str, sources: list[SourceDocument] }` (L425)
- `IndexingStatusRequest{ uri: str }` (L1193)
- `IndexingStatusResponse{ uri, is_watched, files, total_files, status_summary }` (L1198)
- `ResourceListResponse{ resources, total_count, status_summary }` (L1273)

### Module globals to relocate

```py
index_lock = threading.Lock()
watched_resources: dict[str, BaseObserver] = {}
file_last_modified: dict[Path, float] = {}
SIMILARITY_THRESHOLD = 0.95
MAX_SAMPLE_SIZE = 100
BATCH_PROCESSING_DELAY = 1
MAX_WORKERS = multiprocessing.cpu_count()
BATCH_SIZE = 40
```

## Files to create

| File | Why |
| --- | --- |
| `src/api/__init__.py` | empty package marker |
| `src/api/resources.py` | hosts `add_resource`, `remove_resource`, `list_resources` + their request/response models (`ResourceURIRequest`, `ResourceRequest`, `ResourceListResponse`). Exports `router = APIRouter(prefix="/api/v1")`. |
| `src/api/retrieve.py` | hosts legacy `POST /api/v1/retrieve` + `SourceDocument`, `RetrieveRequest`, `RetrieveResponse`, the directory-scoped `filter_documents` closure, and the `ResourceFilterPostProcessor` class (currently main.py L1067-L1135). |
| `src/api/indexing_status.py` | hosts `POST /api/v1/indexing-status` + models. Also register underscore alias `POST /api/v1/indexing_status` that logs deprecation and calls the same handler. |
| `src/api/health.py` | `GET /api/health` and `GET /api/v1/readyz`. |
| `src/rag/__init__.py` | empty marker. |
| `src/rag/engine.py` | owns `index`, `chroma_client`, `vector_store`, `storage_context`, `index_lock`, `watched_resources`, `file_last_modified`. Exposes `get_index()`, `process_document_batch(documents)`, `update_index_for_file(directory, abs_file_path)`, `index_local_resource_async(resource)`, `index_remote_resource_async(resource)`, `init_engine()` (called from lifespan). |
| `src/rag/chunking.py` | hosts `code_ext_map`, `required_exts`, `is_valid_text`, `clean_text`, `split_documents`, helper `get_gitignore_files`, `get_gitcrypt_files`, `get_pathspec`, `scan_directory`, `binary_extensions`. |
| `src/rag/watcher.py` | hosts `FileSystemHandler` (depends on `rag.engine.update_index_for_file`). |
| `src/rag/semantic_search.py` | Chroma + LlamaIndex setup currently in main.py L310-L380. Exposes `init_semantic_search() -> tuple[index, vector_store, storage_context]` and persists `rag_config.json` reset behavior. |
| `src/rag/remote_fetch.py` | hosts `is_remote_resource_exists`, `fetch_markdown`, `markdown_to_links`, `http_headers`. |

## Files to modify

### `src/main.py` — reduce to wiring

```py
from contextlib import asynccontextmanager
import fcntl, os
from fastapi import FastAPI
from collections.abc import AsyncGenerator

from libs.configs import BASE_DATA_DIR
from libs.db import init_db
from libs.logger import logger
from libs.utils import is_local_uri, is_remote_uri, uri_to_path
from services.resource import resource_service

from rag.engine import init_engine, index_local_resource_async, index_remote_resource_async, watched_resources
from rag.watcher import FileSystemHandler
from rag.remote_fetch import is_remote_resource_exists
from watchdog.observers import Observer

from api import resources, retrieve, indexing_status, health

LOCK_FILE = BASE_DATA_DIR / "leader.lock"

def try_acquire_leadership() -> bool: ...   # copy verbatim from current main.py L73-L91

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    is_leader = try_acquire_leadership()
    if is_leader:
        logger.info("Starting RAG service as leader (PID: %d)...", os.getpid())
        init_db()
        init_engine()                       # initializes chroma/llamaindex providers/index
        for resource in [r for r in resource_service.get_all_resources() if r.status == "active"]:
            # ... copy current main.py L96-L142 startup loop, swapping FileSystemHandler import ...
            ...
    yield
    if is_leader:
        for observer in watched_resources.values():
            observer.stop(); observer.join()

app = FastAPI(title="RAG Service API", version="1.0.0",
              docs_url="/docs", redoc_url="/redoc", lifespan=lifespan)

app.include_router(health.router)
app.include_router(resources.router)
app.include_router(retrieve.router)
app.include_router(indexing_status.router)
```

Target line count for `main.py` after refactor: **< 100 lines**.

## Acceptance

- All seven preserved endpoints return identical shapes to current behavior
  on an empty repo.
- `tests/rag_service_spec.lua` (existing Busted spec) still passes.
- Service still reaches `READY` (leader election still works).

---

# PHASE 1 — Dockerfile deps + new models + DB schema additions

## Context

Current Dockerfile (`py/rag-service/Dockerfile`):

```dockerfile
FROM python:3.11-slim-bookworm
COPY gitconfig /root/.gitconfig
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl git \
  && rm -rf /var/lib/apt/lists/* \
  && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH" PYTHONPATH=/app/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt
COPY . .
CMD ["uvicorn", "src.main:app", "--workers", "3", "--host", "0.0.0.0", "--port", "20250"]
```

Current `CREATE_TABLES_SQL` is in §G.3.3.

## Files to create

| File | Why |
| --- | --- |
| `src/models/rag.py` | data contracts for hybrid RAG (FileSpan, RetrievalQuery, RetrievedContext, ContextCitation, ContextSufficiency, RerankScore). Referenced by every later phase. |
| `src/models/runtime.py` | data contracts for runtime/hardware (HardwareProfile, GPUDevice, BackendRecommendation, ModelRuntimePlan, OffloadPlan, BenchmarkResult, HardwareAwareRagBudget). |
| `src/models/chat_history.py` | `ChatMessage`, `ChatTurnUpsert`. |
| `src/models/evals.py` | `RagEvalCase`, `EvalReport`, per-case `EvalRunResult`. |

### `src/models/rag.py` contents

```py
from typing import Literal
from pydantic import BaseModel, Field

class FileSpan(BaseModel):
    uri: str
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    content: str
    reason: str
    score: float
    token_estimate: int
    hash: str
    retrieval_sources: list[str] = []     # exact | symbol | semantic | chat_history | expansion
    chunk_kind: str | None = None         # function | class | section | test | config | code
    language: str | None = None

class RetrievalQuery(BaseModel):
    query: str
    base_uri: str
    top_k: int = 5
    mode: Literal["ask", "search", "edit-small", "test-fix", "refactor"] = "ask"
    current_file: str | None = None
    selected_text: str | None = None
    latest_error: str | None = None
    changed_files: list[str] = []
    include_full_files: bool = False
    include_chat_history: bool = True
    include_stale: bool = False
    max_context_tokens: int | None = None

class ContextCitation(BaseModel):
    uri: str
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    reason: str
    retrieval_sources: list[str]

class ContextSufficiency(BaseModel):
    sufficient: bool
    confidence: float
    missing: list[str] = []
    suggested_retrievals: list[str] = []

class SourceDocumentCompat(BaseModel):
    """Mirror of api/retrieve.py SourceDocument; re-exported for new endpoints."""
    uri: str
    content: str
    score: float | None = None

class RetrievedContext(BaseModel):
    response: str | None = None
    spans: list[FileSpan]
    sources: list[SourceDocumentCompat]
    citations: list[ContextCitation]
    token_estimate: int
    trace_id: str | None = None
    sufficiency: ContextSufficiency | None = None

class RerankScore(BaseModel):
    final: float
    exact: float = 0
    symbol: float = 0
    semantic: float = 0
    chat_history: float = 0
    proximity: float = 0
    import_distance: float = 0
    recent_edit: float = 0
    test_relevance: float = 0
    token_penalty: float = 0
    stale_penalty: float = 0

class RagContextResponse(BaseModel):
    context: str
    spans: list[FileSpan]
    citations: list[ContextCitation]
    token_estimate: int
    trace_id: str
    runtime_plan: "BackendRecommendation | None" = None  # forward ref; resolve in main app
```

### `src/models/runtime.py` contents

```py
from typing import Literal
from pydantic import BaseModel, Field

class GPUDevice(BaseModel):
    vendor: Literal["nvidia","amd","intel","apple","unknown"]
    name: str
    uuid: str | None = None
    vram_bytes: int | None = None
    free_vram_bytes: int | None = None
    driver: str | None = None
    compute_capability: str | None = None
    gfx_target: str | None = None
    supports_cuda: bool = False
    supports_rocm: bool = False
    supports_vulkan: bool = False

class HardwareProfile(BaseModel):
    os: str
    detected_in: Literal["host","container","merged","unknown"] = "container"
    cpu_model: str | None = None
    cpu_cores: int = 0
    cpu_threads: int = 0
    ram_bytes: int = 0
    gpus: list[GPUDevice] = []
    probe_warnings: list[str] = []
    captured_at: str

class BackendRecommendation(BaseModel):
    backend: Literal["ollama","llama.cpp","vllm","sglang","openai-compatible"]
    accelerator: Literal["cuda","rocm","vulkan","cpu"]
    reason: str
    env: dict[str, str] = {}
    launch_args: list[str] = []
    risk: Literal["low","medium","high"] = "low"

class ModelRuntimePlan(BaseModel):
    model_name: str
    quantization: str
    model_bytes: int
    context_tokens: int
    batch_size: int
    expected_concurrent_requests: int
    kv_cache_bytes_estimate: int
    required_vram_bytes: int
    fits_in_vram: bool
    recommendation: str

class OffloadPlan(BaseModel):
    gpu_layers: int | Literal["all"]
    main_gpu: int | None = None
    tensor_split: list[float] | None = None
    split_mode: Literal["none","layer","row"] = "none"
    context_size: int
    batch_size: int
    ubatch_size: int | None = None
    kv_cache_type: str | None = None

class BenchmarkResult(BaseModel):
    backend: str
    model: str
    context_tokens: int
    prompt_eval_tps: float
    decode_tps: float
    ttft_ms: float
    peak_vram_bytes: int | None = None
    cpu_percent: float = 0.0
    gpu_percent: float | None = None
    passed: bool = Field(alias="pass", default=True)

class HardwareAwareRagBudget(BaseModel):
    max_retrieved_tokens: int
    max_spans: int
    max_tool_log_tokens: int
    max_agentic_retrieval_steps: int
    reason: str
```

### `src/models/chat_history.py` contents

```py
from typing import Literal
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: Literal["user","assistant","tool","system"]
    content: str
    timestamp: str
    tool_name: str | None = None

class ChatTurnUpsert(BaseModel):
    base_uri: str
    chat_id: str
    title: str | None = None
    project_root: str
    messages: list[ChatMessage]
    updated_at: str
```

### `src/models/evals.py` contents

```py
from pydantic import BaseModel

class RagEvalCase(BaseModel):
    id: str
    query: str
    mode: str
    base_uri: str
    current_file: str | None = None
    latest_error: str | None = None
    expected_files: list[str]
    expected_symbols: list[str] = []
    must_not_retrieve: list[str] = []

class EvalRunResult(BaseModel):
    case_id: str
    recall_at_k: float
    precision_at_k: float
    mrr: float
    expected_symbol_hit_rate: float
    irrelevant_context_tokens: int
    inserted_token_count: int
    freshness_error_rate: float
    dedupe_savings: int

class EvalReport(BaseModel):
    results: list[EvalRunResult]
    aggregate: dict[str, float]
    trace_ids: list[str] = []
```

## Files to modify

### `py/rag-service/Dockerfile` — add deps

Append to the existing `apt-get install` line:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl git ripgrep pciutils vulkan-tools \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh
```

### `src/libs/db.py` — append to `CREATE_TABLES_SQL`

Append (do NOT modify existing DDL):

```sql
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

-- Schema version sentinel
CREATE TABLE IF NOT EXISTS _schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR REPLACE INTO _schema_meta(key, value) VALUES ('version', '2');
```

## Acceptance

- `docker build` succeeds; `which rg` inside container returns `/usr/bin/rg`.
- `init_db()` creates all six new tables idempotently.
- New Pydantic models importable: `from models.rag import FileSpan`, etc.

---

# PHASE 2 — Exact search + `/api/v1/rag/search`

## Context

ripgrep is available (Phase 1 added it). ripgrep respects `.gitignore`
already; existing `get_pathspec` covers additional patterns + git-crypt.
Token estimation: use a cheap char-based heuristic `tokens ≈ len(text)/4`
exposed as `rag.context_budget.estimate_tokens(text)`. (LlamaIndex bundles
`tiktoken`; you may optionally use it, but the heuristic is sufficient and
provider-agnostic.)

`FileSpan` from §Phase 1 will be the output.

## Files to create

| File | Why |
| --- | --- |
| `src/rag/exact_search.py` | ripgrep-backed exact search. |
| `src/rag/context_budget.py` | houses `estimate_tokens(text)` early (used by exact_search and later). |
| `src/api/rag.py` | new router `prefix="/api/v1/rag"`. First endpoint: `POST /search`. Later phases add `/retrieve`, `/agentic-retrieve`, `/symbols`, `/context`. |

### `src/rag/context_budget.py` (initial)

```py
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Cheap, provider-agnostic. ~4 chars per token on English/code mix.
    return max(1, len(text) // 4)
```

### `src/rag/exact_search.py`

```py
import json, re, shutil, subprocess, hashlib
from pathlib import Path
from typing import Iterable
from libs.logger import logger
from libs.utils import path_to_uri
from models.rag import FileSpan, RetrievalQuery
from rag.context_budget import estimate_tokens

RG = shutil.which("rg")
STACK_FRAME_RE = re.compile(r'([A-Za-z]:[\\/][^\s:"\']+|/[^\s:"\']+|[\w./\\-]+\.\w+):(\d+)')
SYMBOL_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
MAX_RESULTS_PER_QUERY = 50
CONTEXT_LINES = 4

def _rg(query: str, base_path: Path, max_count: int = MAX_RESULTS_PER_QUERY) -> list[dict]:
    if not RG:
        return _python_fallback(query, base_path, max_count)
    try:
        proc = subprocess.run(
            [RG, "--json", "--line-number", "--column", "--no-messages",
             "--max-count", str(max_count), "--hidden", "--glob", "!.git",
             query, str(base_path)],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("ripgrep failed: %s", e)
        return _python_fallback(query, base_path, max_count)
    hits = []
    for line in proc.stdout.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "match":
            continue
        data = obj["data"]
        path = data["path"]["text"]
        line_no = data["line_number"]
        line_text = data["lines"]["text"].rstrip("\n")
        hits.append({"path": path, "line": line_no, "text": line_text})
    return hits

def _python_fallback(query: str, base_path: Path, max_count: int) -> list[dict]:
    out: list[dict] = []
    pat = re.compile(re.escape(query))
    for p in base_path.rglob("*"):
        if len(out) >= max_count: break
        if p.is_dir() or ".git" in p.parts: continue
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    if pat.search(line):
                        out.append({"path": str(p), "line": i, "text": line.rstrip("\n")})
                        if len(out) >= max_count: break
        except OSError:
            continue
    return out

def _expand_span(path: str, line: int, ctx: int = CONTEXT_LINES) -> tuple[int, int, str]:
    try:
        with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
    except OSError:
        return line, line, ""
    start = max(0, line - 1 - ctx)
    end = min(len(all_lines), line + ctx)
    return start + 1, end, "".join(all_lines[start:end])

def _extract_targets(q: RetrievalQuery) -> Iterable[tuple[str, str, float]]:
    """Yield (term, reason, base_score)."""
    seen: set[str] = set()
    if q.latest_error:
        for m in STACK_FRAME_RE.finditer(q.latest_error):
            term = f"{m.group(1)}:{m.group(2)}"
            if term not in seen:
                seen.add(term); yield term, "stack_frame", 4.5
        for m in SYMBOL_RE.finditer(q.latest_error):
            t = m.group(0)
            if t not in seen:
                seen.add(t); yield t, "error_symbol", 4.0
    if q.current_file:
        base = Path(q.current_file).name
        if base not in seen:
            seen.add(base); yield base, "current_file", 3.2
    if q.selected_text:
        for m in SYMBOL_RE.finditer(q.selected_text):
            t = m.group(0)
            if t not in seen:
                seen.add(t); yield t, "selected_symbol", 3.5
    # Always include the raw query as a fuzzy term last
    if q.query and q.query not in seen:
        yield q.query, "query_text", 2.5

class ExactSearch:
    def retrieve(self, query: RetrievalQuery, base_path: Path) -> list[FileSpan]:
        spans: list[FileSpan] = []
        for term, reason, base_score in _extract_targets(query):
            hits = _rg(term, base_path)
            for h in hits:
                start, end, content = _expand_span(h["path"], h["line"])
                if not content:
                    continue
                spans.append(FileSpan(
                    uri=path_to_uri(Path(h["path"])),
                    path=h["path"],
                    start_line=start,
                    end_line=end,
                    content=content,
                    reason=f"exact:{reason}",
                    score=base_score,
                    token_estimate=estimate_tokens(content),
                    hash=hashlib.sha256(content.encode()).hexdigest(),
                    retrieval_sources=["exact"],
                ))
        return spans
```

### `src/api/rag.py` (initial)

```py
from fastapi import APIRouter, HTTPException
from pathlib import Path
from libs.utils import is_local_uri, uri_to_path
from models.rag import RetrievalQuery, FileSpan
from rag.exact_search import ExactSearch

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])

_exact = ExactSearch()

@router.post("/search", response_model=list[FileSpan])
async def rag_search(query: RetrievalQuery) -> list[FileSpan]:
    """Spans-only retrieval (no generation). Phase 2 returns exact-only;
    later phases extend this to the hybrid pipeline."""
    if not is_local_uri(query.base_uri):
        raise HTTPException(status_code=400, detail="exact search requires local base_uri")
    base = uri_to_path(query.base_uri)
    if not base.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {base}")
    return _exact.retrieve(query, base)
```

## Files to modify

- `src/main.py`: `from api import rag; app.include_router(rag.router)`.

## Acceptance

- `POST /api/v1/rag/search` with `{query, base_uri, latest_error?, current_file?, selected_text?, top_k}`
  returns `list[FileSpan]`.
- Stack-trace paths in `latest_error` produce spans tagged
  `reason="exact:stack_frame"`.
- ripgrep absent → Python fallback still returns results (cover both paths).

---

# PHASE 3 — Structural chunking metadata

## Context

`split_documents()` currently lives in `rag/chunking.py` after Phase 0.
You're augmenting it (not replacing) so existing CodeSplitter behavior
remains as a fallback. Add `start_line`, `end_line`, and `text_hash`
metadata to every chunk; emit `chunk_kind` and a placeholder `symbols`
list (populated in Phase 4).

## Files to modify

### `src/rag/chunking.py`

Replace the chunk metadata block in `split_documents()` with:

```py
import hashlib

def _line_bounds_of_chunk(full_text: str, chunk_text: str, prev_end: int) -> tuple[int, int]:
    """Find 1-indexed start/end lines of chunk_text inside full_text.
    Search starts at character offset corresponding to prev_end to handle
    overlapping CodeSplitter chunks."""
    if not chunk_text:
        return prev_end, prev_end
    full_lines = full_text.splitlines()
    chunk_lines = chunk_text.splitlines()
    if not chunk_lines:
        return prev_end, prev_end
    head = chunk_lines[0]
    start_search = max(0, prev_end - 1)
    for i in range(start_search, len(full_lines)):
        if full_lines[i] == head:
            return i + 1, min(len(full_lines), i + len(chunk_lines))
    return prev_end, prev_end + len(chunk_lines)

# ... inside the code-file branch of split_documents, after `texts = code_splitter.split_text(t)`:
prev_end = 0
for i, text in enumerate(texts):
    start_line, end_line = _line_bounds_of_chunk(t, text, prev_end)
    prev_end = end_line
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    new_doc = Document(
        text=text,
        doc_id=f"{doc.doc_id}__part_{i}",
        metadata={
            **doc.metadata,
            "chunk_number": i,
            "total_chunks": len(texts),
            "language": code_splitter.language,
            "orig_doc_id": doc.doc_id,
            "chunk_kind": "code",                    # refined in Phase 4 (function/class/test)
            "symbols": [],                           # populated in Phase 4
            "start_line": start_line,
            "end_line": end_line,
            "text_hash": text_hash,
        },
    )
    processed_documents.append(new_doc)
```

Add a section-aware path for `.md` / `.markdown` files: split on ATX
headings (`^#{1,6}\s`) and tag `chunk_kind="section"`. Keep
`chunk_lines/end_lines` computed the same way.

Add a config-entry path for `.toml/.yaml/.yml/.json/.ini`: emit
`chunk_kind="config"` with the whole file as one chunk (unless > 1500
chars, in which case fall back to byte-window splits).

## Acceptance

- Every code/markdown/config chunk in Chroma has
  `start_line`, `end_line`, `chunk_kind`, `text_hash` metadata.
- Existing semantic retrieval still works (Phase 0 path unchanged).

---

# PHASE 4 — Symbol index + `/api/v1/rag/symbols`

## Context

`tree_sitter_language_pack` is already a dependency (used by `CodeSplitter`).
`symbols` table from Phase 1 receives one row per definition. Update on
file change (delete-then-insert in single transaction).

### tree-sitter language IDs to use

Map (file extension → tree-sitter language name) used for queries:

```
.py  python      .lua  lua          .rs   rust       .go   go
.js  javascript  .jsx  javascript   .ts   typescript .tsx  tsx
.c   c           .h    c            .cpp  cpp        .hpp  cpp
.cs  c_sharp     .java java         .swift swift     .rb   ruby
.php php
```

### Symbol kinds

`function | method | class | interface | type | constant | variable | test | module | unknown`

## Files to create

| File | Why |
| --- | --- |
| `src/rag/queries/__init__.py` | empty marker |
| `src/rag/queries/<lang>.scm` | one per language (13 files), tree-sitter queries with captures `@function`, `@method`, `@class`, `@interface`, `@type`, `@constant`, `@variable`, `@module`. (Names map to symbol_kind.) Keep queries minimal — definitions only. |
| `src/rag/symbol_index.py` | symbol extractor + DB writer + search. |
| `src/api/rag.py` (extend) | add `POST /api/v1/rag/symbols`. |

### Example query file `src/rag/queries/python.scm`

```scheme
(function_definition name: (identifier) @function)
(class_definition    name: (identifier) @class)
(assignment left: (identifier) @constant
            (#match? @constant "^[A-Z_][A-Z0-9_]*$"))
```

(For each language, write a similar minimal query. JavaScript/TypeScript
captures `function_declaration`, `method_definition`, `class_declaration`;
Rust captures `function_item`, `struct_item @type`, `enum_item @type`,
`trait_item @interface`, `impl_item`'s methods; Lua captures
`function_declaration` and `local_function`; Go captures `function_declaration`,
`method_declaration`, `type_spec`; C/C++ captures `function_definition`,
`struct_specifier @class`, `class_specifier @class`; C# captures
`method_declaration`, `class_declaration`, `interface_declaration`;
Java captures `method_declaration`, `class_declaration`,
`interface_declaration`; PHP captures `function_definition`,
`method_declaration`, `class_declaration`, `interface_declaration`;
Ruby captures `method`, `class`, `module @module`; Swift captures
`function_declaration`, `class_declaration`, `protocol_declaration @interface`.)

### `src/rag/symbol_index.py`

```py
import hashlib, json, re
from pathlib import Path
from libs.db import get_db_connection
from libs.logger import logger
from libs.utils import path_to_uri, uri_to_path
from tree_sitter_language_pack import get_parser, get_language

QUERIES_DIR = Path(__file__).parent / "queries"

LANG_EXT = {
    ".py":"python", ".lua":"lua", ".rs":"rust", ".go":"go",
    ".js":"javascript", ".jsx":"javascript", ".ts":"typescript", ".tsx":"tsx",
    ".c":"c", ".h":"c", ".cpp":"cpp", ".hpp":"cpp",
    ".cs":"c_sharp", ".java":"java", ".swift":"swift",
    ".rb":"ruby", ".php":"php",
}

TEST_FILENAME_RE = re.compile(
    r"(^test_|_test\.|_spec\.|\.spec\.|\.test\.|/tests?/)", re.IGNORECASE,
)

CAPTURE_TO_KIND = {
    "function":"function", "method":"method", "class":"class",
    "interface":"interface", "type":"type", "constant":"constant",
    "variable":"variable", "module":"module",
}

def _load_query(lang: str):
    path = QUERIES_DIR / f"{lang}.scm"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")

def _is_test_path(file_uri: str, test_patterns: list[str] | None = None) -> bool:
    if TEST_FILENAME_RE.search(file_uri):
        return True
    if test_patterns:
        for p in test_patterns:
            if re.search(p, file_uri):
                return True
    return False

def extract_symbols(file_uri: str, resource_uri: str, content: str,
                    test_patterns: list[str] | None = None) -> list[dict]:
    ext = Path(uri_to_path(file_uri)).suffix.lower()
    lang = LANG_EXT.get(ext)
    if not lang:
        return []
    query_text = _load_query(lang)
    if not query_text:
        return []
    try:
        parser = get_parser(lang)
        tree = parser.parse(content.encode("utf-8"))
        language = get_language(lang)
        query = language.query(query_text)
    except Exception as e:                                     # noqa: BLE001
        logger.warning("symbol extract failed (%s, %s): %s", file_uri, lang, e)
        return []

    is_test_file = _is_test_path(file_uri, test_patterns)
    rows: list[dict] = []
    for node, capture_name in query.captures(tree.root_node):
        kind = CAPTURE_TO_KIND.get(capture_name, "unknown")
        if is_test_file and kind in {"function", "method"}:
            kind = "test"
        name = node.text.decode("utf-8", errors="replace")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        rows.append({
            "resource_uri": resource_uri,
            "file_uri": file_uri,
            "symbol_name": name,
            "symbol_kind": kind,
            "start_line": start_line,
            "end_line": end_line,
            "language": lang,
            "text_hash": hashlib.sha256(content.encode()).hexdigest(),
            "metadata": json.dumps({"is_test_file": is_test_file}),
        })
    return rows

def replace_symbols_for_file(file_uri: str, resource_uri: str, content: str,
                             test_patterns: list[str] | None = None) -> int:
    rows = extract_symbols(file_uri, resource_uri, content, test_patterns)
    with get_db_connection() as conn:
        conn.execute("DELETE FROM symbols WHERE file_uri = ?", (file_uri,))
        if rows:
            conn.executemany(
                """INSERT INTO symbols(resource_uri, file_uri, symbol_name, symbol_kind,
                                       start_line, end_line, language, text_hash, metadata)
                   VALUES (:resource_uri, :file_uri, :symbol_name, :symbol_kind,
                           :start_line, :end_line, :language, :text_hash, :metadata)""",
                rows,
            )
        conn.commit()
    return len(rows)

def search_symbols(base_uri: str, q: str, kinds: list[str] | None = None,
                   limit: int = 30) -> list[dict]:
    sql = """SELECT * FROM symbols
             WHERE resource_uri = ? AND symbol_name LIKE ?"""
    params: list = [base_uri, f"%{q}%"]
    if kinds:
        placeholders = ",".join("?" * len(kinds))
        sql += f" AND symbol_kind IN ({placeholders})"
        params.extend(kinds)
    sql += " LIMIT ?"; params.append(limit)
    with get_db_connection() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
```

### Wire symbol extraction into the indexer

In `src/rag/engine.py` `update_index_for_file()` (Phase 0 destination of
old `update_index_for_file`), after the file has been read for chunking,
call:

```py
from rag.symbol_index import replace_symbols_for_file
# ...
try:
    text = abs_file_path.read_text(encoding="utf-8", errors="ignore")
    replace_symbols_for_file(path_to_uri(abs_file_path), resource.uri, text)
except OSError:
    pass
```

Do the same inside the local resource indexing loop (one call per file).

### `src/api/rag.py` (extend)

```py
from rag.symbol_index import search_symbols
from pydantic import BaseModel

class SymbolSearchRequest(BaseModel):
    base_uri: str
    q: str
    kinds: list[str] | None = None
    limit: int = 30

@router.post("/symbols")
async def rag_symbols(req: SymbolSearchRequest):
    return {"results": search_symbols(req.base_uri, req.q, req.kinds, req.limit)}
```

## Acceptance

- Indexing a Python file produces rows in `symbols` with
  `(symbol_name, symbol_kind, start_line, end_line)`.
- Re-indexing the same file replaces (does not duplicate) its rows.
- `POST /api/v1/rag/symbols {base_uri, q:"foo"}` returns hits.

---

# PHASE 5 — Dedupe + Reranker

## Files to create

| File | Why |
| --- | --- |
| `src/rag/dedupe.py` | hash-dedup + overlap-merge of `FileSpan`s. |
| `src/rag/reranker.py` | computes `RerankScore` and `final` score per span. |

### `src/rag/dedupe.py`

```py
import hashlib, re
from models.rag import FileSpan

_WS = re.compile(r"\s+")

def _norm_hash(text: str) -> str:
    return hashlib.sha256(_WS.sub(" ", text).strip().encode()).hexdigest()

def dedupe_and_merge(spans: list[FileSpan]) -> tuple[list[FileSpan], int]:
    """Returns (deduped, tokens_saved)."""
    by_hash: dict[str, FileSpan] = {}
    saved = 0
    for s in spans:
        h = _norm_hash(s.content)
        existing = by_hash.get(h)
        if existing is None:
            by_hash[h] = s
        else:
            saved += s.token_estimate
            existing.score = max(existing.score, s.score)
            existing.retrieval_sources = sorted(set(existing.retrieval_sources + s.retrieval_sources))
            if existing.reason != s.reason:
                existing.reason = f"{existing.reason}|{s.reason}"
    # Overlap-merge spans from the same file
    by_uri: dict[str, list[FileSpan]] = {}
    for s in by_hash.values():
        by_uri.setdefault(s.uri, []).append(s)
    merged: list[FileSpan] = []
    for uri, group in by_uri.items():
        group.sort(key=lambda x: (x.start_line or 0))
        current = group[0]
        for nxt in group[1:]:
            if (current.end_line or 0) >= (nxt.start_line or 0) - 1 \
               and current.start_line is not None and nxt.start_line is not None:
                # Merge
                new_end = max(current.end_line or 0, nxt.end_line or 0)
                saved += nxt.token_estimate
                current = FileSpan(
                    uri=uri,
                    path=current.path,
                    start_line=current.start_line,
                    end_line=new_end,
                    content=current.content + "\n" + nxt.content,
                    reason=f"{current.reason}+{nxt.reason}",
                    score=max(current.score, nxt.score),
                    token_estimate=current.token_estimate + nxt.token_estimate,
                    hash=_norm_hash(current.content + nxt.content),
                    retrieval_sources=sorted(set(current.retrieval_sources + nxt.retrieval_sources)),
                    chunk_kind=current.chunk_kind,
                    language=current.language,
                )
            else:
                merged.append(current); current = nxt
        merged.append(current)
    return merged, saved
```

### `src/rag/reranker.py`

```py
from pathlib import Path
from models.rag import FileSpan, RetrievalQuery, RerankScore

WEIGHTS = {
    "exact": 4.0, "symbol": 3.5, "semantic": 1.5, "chat_history": 1.0,
    "metadata": 1.0, "recent": 0.75, "test": 1.25,
}
TOKEN_PENALTY_PER_1K = 0.8     # bigger spans hurt more
STALE_PENALTY = 1.0

def _channel_score(span: FileSpan) -> tuple[float, float, float, float]:
    e = WEIGHTS["exact"]        if "exact" in span.retrieval_sources else 0
    s = WEIGHTS["symbol"]       if "symbol" in span.retrieval_sources else 0
    v = WEIGHTS["semantic"]     if "semantic" in span.retrieval_sources else 0
    c = WEIGHTS["chat_history"] if "chat_history" in span.retrieval_sources else 0
    return e, s, v, c

def _proximity(span: FileSpan, current_file: str | None) -> float:
    if not current_file or not span.path:
        return 0.0
    cur = Path(current_file).resolve()
    target = Path(span.path).resolve()
    try:
        common = len(set(cur.parts) & set(target.parts))
        return min(1.0, common * 0.1)
    except OSError:
        return 0.0

def rerank(spans: list[FileSpan], query: RetrievalQuery,
           stale_uris: set[str] | None = None,
           recent_uris: set[str] | None = None) -> list[tuple[FileSpan, RerankScore]]:
    stale_uris = stale_uris or set()
    recent_uris = recent_uris or set()
    out: list[tuple[FileSpan, RerankScore]] = []
    for s in spans:
        e, sy, sem, ch = _channel_score(s)
        prox = _proximity(s, query.current_file)
        recent = WEIGHTS["recent"] if s.uri in recent_uris else 0
        test_bonus = WEIGHTS["test"] if s.chunk_kind == "test" or query.mode == "test-fix" else 0
        token_pen = TOKEN_PENALTY_PER_1K * (s.token_estimate / 1000.0)
        stale_pen = STALE_PENALTY if s.uri in stale_uris else 0
        final = e + sy + sem + ch + prox + recent + test_bonus - token_pen - stale_pen
        score = RerankScore(
            final=final, exact=e, symbol=sy, semantic=sem, chat_history=ch,
            proximity=prox, recent_edit=recent, test_relevance=test_bonus,
            token_penalty=token_pen, stale_penalty=stale_pen,
        )
        s.score = final
        out.append((s, score))
    out.sort(key=lambda x: x[1].final, reverse=True)
    return out
```

## Acceptance

- Identical-content spans collapse; `tokens_saved` > 0.
- Adjacent line ranges in the same file merge into one span.
- Spans with `retrieval_sources=["exact"]` outrank semantic-only spans on
  the same query.

---

# PHASE 6 — Hybrid retriever + `/api/v1/rag/retrieve` + `/api/v1/rag/context`

## Context

Existing semantic path is the LlamaIndex `index.as_query_engine(...)`
wired in `rag/semantic_search.py` after Phase 0. Wrap it in a
`SemanticRetriever.retrieve(query: RetrievalQuery) -> list[FileSpan]` that
re-uses the existing `ResourceFilterPostProcessor` for base-uri scoping
and emits `FileSpan` (tagging `retrieval_sources=["semantic"]`).

A `SymbolRetriever.retrieve(...)` wraps `search_symbols` from Phase 4 and
materializes `FileSpan` by reading file content for `[start_line, end_line]`.

## Files to create

| File | Why |
| --- | --- |
| `src/rag/semantic_search.py` (extend) | add `SemanticRetriever` class. |
| `src/rag/hybrid_retriever.py` | orchestrates channels. |
| `src/api/rag.py` (extend) | add `/retrieve` and `/context`. |

### `src/rag/semantic_search.py` — append

```py
import hashlib
from libs.utils import get_node_uri
from models.rag import FileSpan, RetrievalQuery
from rag.context_budget import estimate_tokens

class SemanticRetriever:
    def __init__(self, index_provider):
        self._get_index = index_provider     # callable returning current VectorStoreIndex

    def retrieve(self, query: RetrievalQuery) -> list[FileSpan]:
        index = self._get_index()
        engine = index.as_query_engine(similarity_top_k=max(query.top_k * 3, 15))
        result = engine.query(query.query)
        spans: list[FileSpan] = []
        for node in result.source_nodes or []:
            uri = get_node_uri(node.node) or ""
            content = str(node.node.get_content())
            md = getattr(node.node, "metadata", {}) or {}
            spans.append(FileSpan(
                uri=uri,
                path=uri.removeprefix("file://") if uri.startswith("file://") else None,
                start_line=md.get("start_line"),
                end_line=md.get("end_line"),
                content=content,
                reason="semantic:vector",
                score=float(node.score or 0.0),
                token_estimate=estimate_tokens(content),
                hash=hashlib.sha256(content.encode()).hexdigest(),
                retrieval_sources=["semantic"],
                chunk_kind=md.get("chunk_kind"),
                language=md.get("language"),
            ))
        return spans
```

### `src/rag/hybrid_retriever.py`

```py
import hashlib
from pathlib import Path
from libs.logger import logger
from libs.utils import is_local_uri, uri_to_path, path_to_uri
from models.rag import (
    FileSpan, RetrievalQuery, RetrievedContext, ContextCitation,
    SourceDocumentCompat,
)
from rag.context_budget import estimate_tokens
from rag.dedupe import dedupe_and_merge
from rag.reranker import rerank
from rag.exact_search import ExactSearch
from rag.symbol_index import search_symbols
# semantic + chat_history injected via constructor to avoid import cycles

class SymbolRetriever:
    def retrieve(self, query: RetrievalQuery) -> list[FileSpan]:
        if not is_local_uri(query.base_uri):
            return []
        # Pull each captured symbol token from query+selected_text+latest_error
        terms: set[str] = set()
        for src in (query.query, query.selected_text or "", query.latest_error or ""):
            for m in __import__("re").finditer(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", src):
                terms.add(m.group(0))
        spans: list[FileSpan] = []
        for term in list(terms)[:20]:
            for row in search_symbols(query.base_uri, term, limit=8):
                file_path = uri_to_path(row["file_uri"])
                if not file_path.exists():
                    continue
                try:
                    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                except OSError:
                    continue
                s, e = max(0, (row["start_line"] or 1) - 1), min(len(lines), (row["end_line"] or row["start_line"] or 1))
                content = "\n".join(lines[s:e])
                if not content:
                    continue
                spans.append(FileSpan(
                    uri=row["file_uri"],
                    path=str(file_path),
                    start_line=row["start_line"],
                    end_line=row["end_line"],
                    content=content,
                    reason=f"symbol:{row['symbol_kind']}:{row['symbol_name']}",
                    score=3.5,
                    token_estimate=estimate_tokens(content),
                    hash=hashlib.sha256(content.encode()).hexdigest(),
                    retrieval_sources=["symbol"],
                    chunk_kind=row["symbol_kind"],
                    language=row.get("language"),
                ))
        return spans

class HybridRetriever:
    def __init__(self, semantic, chat_history=None):
        self._semantic = semantic
        self._chat = chat_history
        self._exact = ExactSearch()
        self._symbol = SymbolRetriever()

    def retrieve(self, query: RetrievalQuery) -> RetrievedContext:
        spans: list[FileSpan] = []
        if is_local_uri(query.base_uri):
            base = uri_to_path(query.base_uri)
            if base.exists():
                spans.extend(self._exact.retrieve(query, base))
                spans.extend(self._symbol.retrieve(query))
        try:
            spans.extend(self._semantic.retrieve(query))
        except Exception as e:                                # noqa: BLE001
            logger.warning("semantic retrieval failed: %s", e)
        if self._chat and query.include_chat_history:
            try:
                spans.extend(self._chat.retrieve(query))
            except Exception as e:                            # noqa: BLE001
                logger.warning("chat_history retrieval failed: %s", e)

        deduped, saved = dedupe_and_merge(spans)
        scored = rerank(deduped, query)
        ranked_spans = [s for s, _ in scored]

        # Trim to top_k * 3 then later phases apply budget; here keep top_k * 3
        kept = ranked_spans[: max(query.top_k * 3, 8)]
        token_estimate = sum(s.token_estimate for s in kept)

        citations = [ContextCitation(
            uri=s.uri, path=s.path, start_line=s.start_line, end_line=s.end_line,
            reason=s.reason, retrieval_sources=s.retrieval_sources,
        ) for s in kept]

        sources_compat = [SourceDocumentCompat(uri=s.uri, content=s.content, score=s.score)
                          for s in kept]

        return RetrievedContext(
            spans=kept, sources=sources_compat, citations=citations,
            token_estimate=token_estimate, trace_id=None, response=None,
        )
```

### `src/api/rag.py` (extend)

```py
from models.rag import RagContextResponse, RetrievedContext
from rag.hybrid_retriever import HybridRetriever
from rag.semantic_search import SemanticRetriever
from rag.engine import get_index

_hybrid = HybridRetriever(semantic=SemanticRetriever(get_index))

@router.post("/retrieve", response_model=RetrievedContext)
async def rag_retrieve(query: RetrievalQuery):
    return _hybrid.retrieve(query)

@router.post("/context", response_model=RagContextResponse)
async def rag_context(query: RetrievalQuery):
    ctx = _hybrid.retrieve(query)
    blocks = []
    for s in ctx.spans:
        head = f"--- {s.path or s.uri}"
        if s.start_line and s.end_line:
            head += f":L{s.start_line}-L{s.end_line}"
        head += f"  ({s.reason})"
        blocks.append(f"{head}\n{s.content}")
    return RagContextResponse(
        context="\n\n".join(blocks),
        spans=ctx.spans,
        citations=ctx.citations,
        token_estimate=ctx.token_estimate,
        trace_id=ctx.trace_id or "",
        runtime_plan=None,
    )
```

Also rewrite `POST /api/v1/rag/search` (from Phase 2) to use
`HybridRetriever` instead of `ExactSearch` directly, so search returns
the same ranked set without generation.

## Acceptance

- `/rag/retrieve` returns spans from all channels with citations.
- `/rag/context` returns a packed text block + spans + citations.
- Removing the `index` (empty Chroma) still returns exact+symbol spans.

---

# PHASE 7 — Context budget + freshness + citations

## Context

`citations.py` is trivial — the Phase 6 code already builds
`ContextCitation`. Extract it to `src/rag/citations.py` for cleanliness.

Budgets per mode:

```py
BUDGETS = {
    "ask":        {"max_total_tokens": 6000,  "max_spans": 5,
                   "max_doc_tokens": 2000, "max_log_tokens": 500},
    "search":     {"max_total_tokens": 8000,  "max_spans": 8,
                   "max_doc_tokens": 3000, "max_log_tokens": 500},
    "edit-small": {"max_total_tokens": 10000, "max_spans": 6,
                   "max_doc_tokens": 2000, "max_log_tokens": 1000},
    "test-fix":   {"max_total_tokens": 12000, "max_spans": 8,
                   "max_doc_tokens": 1500, "max_log_tokens": 2000},
    "refactor":   {"max_total_tokens": 20000, "max_spans": 16,
                   "max_doc_tokens": 3000, "max_log_tokens": 1000},
}
```

Freshness signals: current branch (`git rev-parse --abbrev-ref HEAD`),
modified files (`git status --porcelain`), mtime, `indexing_history.timestamp`,
path markers (`/legacy/`, `/deprecated/`, `/old/`, `/archive/`),
generated markers (`/node_modules/`, `/vendor/`, `/dist/`, `/build/`,
`/target/`, `/.venv/`).

## Files to create

| File | Why |
| --- | --- |
| `src/rag/citations.py` | `build_citations(spans) -> list[ContextCitation]`. |
| `src/rag/freshness.py` | `compute_freshness(base_path) -> (stale_uris, recent_uris)`. |

### `src/rag/context_budget.py` (extend)

```py
from models.rag import FileSpan

BUDGETS = { ... }  # as above

class BudgetResult(BaseModel):  # add at top
    pass

def apply_budget(spans: list[FileSpan], mode: str,
                 override_total: int | None = None,
                 hardware_cap: "HardwareAwareRagBudget | None" = None
                 ) -> tuple[list[FileSpan], list[FileSpan]]:
    """Returns (kept, dropped). Spans must be pre-sorted by score desc."""
    cfg = BUDGETS.get(mode, BUDGETS["ask"])
    max_total = override_total or cfg["max_total_tokens"]
    max_spans = cfg["max_spans"]
    if hardware_cap:
        max_total = min(max_total, hardware_cap.max_retrieved_tokens)
        max_spans = min(max_spans, hardware_cap.max_spans)
    kept: list[FileSpan] = []; dropped: list[FileSpan] = []
    running = 0
    for s in spans:
        if len(kept) >= max_spans or running + s.token_estimate > max_total:
            dropped.append(s); continue
        kept.append(s); running += s.token_estimate
    return kept, dropped
```

### `src/rag/freshness.py`

```py
import subprocess, shutil
from pathlib import Path
from libs.utils import path_to_uri

GENERATED_MARKERS = ("/node_modules/", "/vendor/", "/dist/", "/build/",
                     "/target/", "/.venv/", "/.tox/", "/__pycache__/")
STALE_MARKERS = ("/legacy/", "/deprecated/", "/old/", "/archive/")
GIT = shutil.which("git")

def _git(args: list[str], cwd: Path) -> str:
    if not GIT: return ""
    try:
        p = subprocess.run([GIT, "-C", str(cwd), *args],
                           capture_output=True, text=True, timeout=5, check=False)
        return p.stdout
    except (subprocess.SubprocessError, OSError):
        return ""

def compute_freshness(base_path: Path) -> tuple[set[str], set[str]]:
    """Returns (stale_uris, recent_uris). 'stale' demotes; 'recent' boosts."""
    stale: set[str] = set(); recent: set[str] = set()
    porcelain = _git(["status", "--porcelain"], base_path)
    for line in porcelain.splitlines():
        if len(line) > 3:
            rel = line[3:].strip()
            recent.add(path_to_uri((base_path / rel).resolve()))
    for p in base_path.rglob("*"):
        if not p.is_file(): continue
        s = str(p)
        if any(m in s for m in GENERATED_MARKERS) or any(m in s for m in STALE_MARKERS):
            stale.add(path_to_uri(p))
    return stale, recent
```

## Files to modify

### `src/rag/hybrid_retriever.py`

Inject freshness into rerank and apply budget before returning:

```py
from rag.freshness import compute_freshness
from rag.context_budget import apply_budget

# inside HybridRetriever.retrieve(...), replacing the trim step:
stale, recent = (set(), set())
if is_local_uri(query.base_uri):
    stale, recent = compute_freshness(uri_to_path(query.base_uri))
if query.include_stale:
    stale = set()
scored = rerank(deduped, query, stale_uris=stale, recent_uris=recent)
ordered = [s for s, _ in scored]
kept, dropped = apply_budget(ordered, query.mode,
                             override_total=query.max_context_tokens,
                             hardware_cap=None)  # populated in Phase 12
```

### `src/rag/citations.py`

```py
from models.rag import FileSpan, ContextCitation
def build_citations(spans: list[FileSpan]) -> list[ContextCitation]:
    return [ContextCitation(
        uri=s.uri, path=s.path, start_line=s.start_line, end_line=s.end_line,
        reason=s.reason, retrieval_sources=s.retrieval_sources,
    ) for s in spans]
```

## Acceptance

- `/rag/context` never exceeds `max_total_tokens` for the mode.
- Files under `node_modules/` etc. are never in `kept` when other spans
  exist.
- `include_stale=true` re-enables them.

---

# PHASE 8 — Observability (JSONL trace)

## Context

Traces go to `${DATA_DIR}/traces/rag-YYYYMMDD.jsonl`. Each line is a
JSON object. `trace_id` is a `uuid4().hex`. Each retrieval call emits one
trace object.

## Files to create

| File | Why |
| --- | --- |
| `src/observability/__init__.py` | empty marker. |
| `src/observability/trace.py` | trace context manager + dataclass. |
| `src/observability/jsonl_exporter.py` | filesystem sink. |

### `src/observability/trace.py`

```py
import time, uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from models.rag import RerankScore

@dataclass
class RagTrace:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    query: str = ""
    mode: str = ""
    base_uri: str = ""
    retrieved_spans_count: int = 0
    inserted_spans_count: int = 0
    dropped_spans_count: int = 0
    retrieved_tokens: int = 0
    inserted_tokens: int = 0
    deduped_tokens_saved: int = 0
    rerank_scores: list[dict] = field(default_factory=list)
    freshness_stale_count: int = 0
    freshness_recent_count: int = 0
    context_budget_used: int = 0
    hardware_profile_hash: str | None = None
    backend_recommendation: dict | None = None
    retrieval_latency_ms: float = 0.0
    stages: list[dict] = field(default_factory=list)

    def add_stage(self, name: str, **kv):
        self.stages.append({"name": name, "t_ms": round((time.perf_counter() - self._t0) * 1000, 2), **kv})

    def to_dict(self) -> dict:
        return asdict(self)

@contextmanager
def start_trace(query: str, mode: str, base_uri: str):
    t = RagTrace(query=query, mode=mode, base_uri=base_uri)
    t._t0 = time.perf_counter()
    try:
        yield t
    finally:
        t.retrieval_latency_ms = round((time.perf_counter() - t._t0) * 1000, 2)
        from observability.jsonl_exporter import write_trace
        write_trace(t.to_dict())
```

### `src/observability/jsonl_exporter.py`

```py
import json
from datetime import datetime
from libs.configs import BASE_DATA_DIR

TRACE_DIR = BASE_DATA_DIR / "traces"
TRACE_DIR.mkdir(parents=True, exist_ok=True)

def write_trace(obj: dict) -> None:
    fname = TRACE_DIR / f"rag-{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
    line = json.dumps(obj, ensure_ascii=False, default=str)
    with fname.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
```

## Files to modify

### `src/rag/hybrid_retriever.py` — wrap retrieve

```py
from observability.trace import start_trace

class HybridRetriever:
    def retrieve(self, query: RetrievalQuery) -> RetrievedContext:
        with start_trace(query.query, query.mode, query.base_uri) as tr:
            # ... existing flow ...
            tr.retrieved_spans_count = len(deduped)
            tr.inserted_spans_count = len(kept)
            tr.dropped_spans_count = len(dropped)
            tr.retrieved_tokens = sum(s.token_estimate for s in ordered)
            tr.inserted_tokens = sum(s.token_estimate for s in kept)
            tr.deduped_tokens_saved = saved
            tr.context_budget_used = sum(s.token_estimate for s in kept)
            tr.freshness_stale_count = len(stale)
            tr.freshness_recent_count = len(recent)
            tr.rerank_scores = [sc.model_dump() for _, sc in scored[:20]]
            return RetrievedContext(
                spans=kept, ..., trace_id=tr.trace_id,
            )
```

## Acceptance

- Every `/rag/retrieve`, `/rag/search`, `/rag/context` call writes a line
  to `${DATA_DIR}/traces/rag-YYYYMMDD.jsonl`.
- `trace_id` returned in response matches the one in the JSONL file.

---

# PHASE 9 — Expansion + project profile

## Files to create

| File | Why |
| --- | --- |
| `src/rag/expansion.py` | depth-1 import / symbol expansion. |
| `src/rag/project_profile.py` | builds + caches `ProjectProfile`. |

### `src/rag/expansion.py`

```py
import re
from pathlib import Path
from libs.utils import path_to_uri, uri_to_path
from models.rag import FileSpan, RetrievalQuery
from rag.context_budget import estimate_tokens
from rag.symbol_index import search_symbols
import hashlib

class ExpansionBudget:
    max_depth = 1
    max_extra_spans = 4
    max_extra_tokens = 3000

IMPORT_RES = {
    "python": [re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.M)],
    "javascript": [re.compile(r"""(?:import|require)\s*\(?['"]([^'"]+)['"]""")],
    "typescript": [re.compile(r"""(?:import|require)\s*\(?['"]([^'"]+)['"]""")],
    "go": [re.compile(r'^\s*import\s+"([^"]+)"', re.M)],
    "rust": [re.compile(r"^\s*use\s+([\w:]+)", re.M)],
    "java": [re.compile(r"^\s*import\s+([\w.]+);", re.M)],
}

def expand(spans: list[FileSpan], query: RetrievalQuery) -> list[FileSpan]:
    extras: list[FileSpan] = []
    used_tokens = 0
    for s in spans:
        if used_tokens >= ExpansionBudget.max_extra_tokens: break
        if len(extras) >= ExpansionBudget.max_extra_spans: break
        if not s.language: continue
        pats = IMPORT_RES.get(s.language, [])
        symbols: set[str] = set()
        for p in pats:
            for m in p.finditer(s.content):
                for g in m.groups():
                    if g:
                        symbols.add(g.split(".")[-1])
        for sym in list(symbols)[:4]:
            for row in search_symbols(query.base_uri, sym, limit=1):
                file_path = uri_to_path(row["file_uri"])
                try:
                    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                except OSError:
                    continue
                a = max(0, (row["start_line"] or 1) - 1)
                b = min(len(lines), row["end_line"] or row["start_line"] or 1)
                content = "\n".join(lines[a:b])
                if not content: continue
                tok = estimate_tokens(content)
                if used_tokens + tok > ExpansionBudget.max_extra_tokens: continue
                extras.append(FileSpan(
                    uri=row["file_uri"], path=str(file_path),
                    start_line=row["start_line"], end_line=row["end_line"],
                    content=content, reason=f"expand:import:{sym}", score=2.0,
                    token_estimate=tok,
                    hash=hashlib.sha256(content.encode()).hexdigest(),
                    retrieval_sources=["expansion"],
                    chunk_kind=row["symbol_kind"], language=row.get("language"),
                ))
                used_tokens += tok
                if len(extras) >= ExpansionBudget.max_extra_spans: break
    return extras
```

### `src/rag/project_profile.py`

```py
import hashlib, json
from pathlib import Path
from libs.db import get_db_connection
from libs.utils import uri_to_path
from pydantic import BaseModel

class ProjectProfile(BaseModel):
    project_name: str
    stack: list[str] = []
    package_manager: str | None = None
    test_commands: list[str] = []
    build_commands: list[str] = []
    lint_commands: list[str] = []
    important_paths: list[str] = []
    generated_paths: list[str] = []
    conventions: list[str] = []
    test_patterns: list[str] = []
    updated_at: str

TRIGGER_FILES = ("package.json","pyproject.toml","requirements.txt","Cargo.toml",
                 "go.mod","flake.nix","shell.nix","Dockerfile","Makefile","README.md")

def _input_hash(base: Path) -> str:
    h = hashlib.sha256()
    for name in TRIGGER_FILES:
        p = base / name
        if p.exists():
            h.update(name.encode()); h.update(p.read_bytes()[:64_000])
    return h.hexdigest()

def get_or_build(resource_uri: str) -> ProjectProfile | None:
    base = uri_to_path(resource_uri)
    if not base.exists(): return None
    new_hash = _input_hash(base)
    with get_db_connection() as conn:
        row = conn.execute("SELECT profile_json, profile_hash FROM project_profiles WHERE resource_uri = ?",
                           (resource_uri,)).fetchone()
        if row and row["profile_hash"] == new_hash:
            return ProjectProfile.model_validate_json(row["profile_json"])
        profile = _build(base)
        conn.execute("""INSERT OR REPLACE INTO project_profiles(resource_uri, profile_json, profile_hash)
                        VALUES (?, ?, ?)""",
                     (resource_uri, profile.model_dump_json(), new_hash))
        conn.commit()
        return profile

def _build(base: Path) -> ProjectProfile:
    from datetime import datetime
    stack: list[str] = []
    pm = None; tests: list[str] = []; builds: list[str] = []; lint: list[str] = []
    if (base / "package.json").exists():
        stack.append("node"); pm = "npm"
        try:
            pkg = json.loads((base / "package.json").read_text())
            scripts = (pkg.get("scripts") or {})
            if "test" in scripts: tests.append("npm test")
            if "build" in scripts: builds.append("npm run build")
            if "lint" in scripts: lint.append("npm run lint")
        except (OSError, json.JSONDecodeError): pass
    if (base / "pyproject.toml").exists():
        stack.append("python"); pm = pm or "pip"
        tests.append("pytest")
    if (base / "Cargo.toml").exists():
        stack.append("rust"); pm = pm or "cargo"
        tests.append("cargo test"); builds.append("cargo build"); lint.append("cargo clippy")
    if (base / "go.mod").exists():
        stack.append("go"); tests.append("go test ./...")
    return ProjectProfile(
        project_name=base.name, stack=stack, package_manager=pm,
        test_commands=tests, build_commands=builds, lint_commands=lint,
        important_paths=["src", "lib"], generated_paths=["dist","build","target","node_modules",".venv"],
        conventions=[], test_patterns=[],
        updated_at=datetime.utcnow().isoformat(),
    )
```

## Files to modify

`src/rag/hybrid_retriever.py`: call `expand(spans, query)` after `rerank`,
extend `ordered` with results, re-sort, then apply budget. Record extras
in trace as `expanded_spans_count`.

## Acceptance

- Importing a function in retrieved code surfaces the definition span as
  an `expansion` source.
- `project_profiles` row created on first retrieval; not rebuilt when
  `package.json` etc. are unchanged.

---

# PHASE 10 — Log summarizer + sufficiency + agentic planner

## Files to create

| File | Why |
| --- | --- |
| `src/rag/log_summarizer.py` | strips logs to essentials. |
| `src/rag/sufficiency.py` | rules-based sufficiency check. |
| `src/rag/agentic_planner.py` | bounded multi-step retrieval. |
| `src/api/rag.py` (extend) | `POST /agentic-retrieve`. |

### `src/rag/log_summarizer.py`

```py
import re
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PROGRESS = re.compile(r"[\r\b].*")
SUCCESS = re.compile(r"(?i)^(installing|downloading|fetched|up to date)")

def summarize(log: str, max_tokens: int) -> str:
    log = ANSI.sub("", log)
    keep: list[str] = []
    seen: set[str] = set()
    for raw in log.splitlines():
        line = PROGRESS.sub("", raw).strip()
        if not line: continue
        if SUCCESS.search(line): continue
        if line in seen: continue
        seen.add(line)
        keep.append(line)
    # Keep first 50 + last 50 prioritizing stack frames
    if len(keep) > 100:
        keep = keep[:50] + ["..."] + keep[-50:]
    text = "\n".join(keep)
    # Hard cap by tokens (~4 chars/token)
    if len(text) > max_tokens * 4:
        text = text[: max_tokens * 4]
    return text
```

### `src/rag/sufficiency.py`

```py
from models.rag import RetrievalQuery, FileSpan, ContextSufficiency
import re

def check(query: RetrievalQuery, spans: list[FileSpan]) -> ContextSufficiency:
    if query.mode == "ask":
        ok = bool(spans)
        return ContextSufficiency(sufficient=ok, confidence=0.7 if ok else 0.3,
                                  missing=[] if ok else ["any_relevant_doc"])
    if query.mode == "test-fix":
        has_test = any(s.chunk_kind == "test" for s in spans)
        has_impl = any(s.chunk_kind in {"function","method","class"} for s in spans)
        missing = []
        if not has_test: missing.append("failing_test_span")
        if not has_impl: missing.append("implementation_span")
        return ContextSufficiency(sufficient=not missing, confidence=0.8 if not missing else 0.4,
                                  missing=missing,
                                  suggested_retrievals=["retrieve_exact(latest_error)", "inspect_tests"])
    if query.mode == "edit-small" and query.selected_text:
        sym = re.search(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", query.selected_text)
        if sym:
            name = sym.group(0)
            if not any(name in s.content for s in spans):
                return ContextSufficiency(sufficient=False, confidence=0.4,
                                          missing=[f"definition_of:{name}"],
                                          suggested_retrievals=["retrieve_symbol"])
    if query.mode == "refactor":
        # need at least 2 distinct files for callsites
        files = {s.uri for s in spans}
        ok = len(files) >= 2
        return ContextSufficiency(sufficient=ok, confidence=0.7 if ok else 0.4,
                                  missing=[] if ok else ["additional_callsites"])
    return ContextSufficiency(sufficient=bool(spans), confidence=0.6 if spans else 0.2)
```

### `src/rag/agentic_planner.py`

```py
from models.rag import RetrievalQuery, RetrievedContext
from rag.sufficiency import check
from rag.log_summarizer import summarize
from rag.context_budget import BUDGETS

AGENTIC_BUDGETS = {"ask":1, "search":1, "edit-small":2, "test-fix":4, "refactor":6}

class AgenticPlanner:
    def __init__(self, hybrid):
        self._hybrid = hybrid

    def run(self, query: RetrievalQuery) -> RetrievedContext:
        max_steps = AGENTIC_BUDGETS.get(query.mode, 1)
        if query.latest_error:
            cfg = BUDGETS.get(query.mode, BUDGETS["ask"])
            query = query.model_copy(update={
                "latest_error": summarize(query.latest_error, cfg["max_log_tokens"])
            })
        ctx = self._hybrid.retrieve(query)
        for step in range(1, max_steps):
            suff = check(query, ctx.spans)
            ctx.sufficiency = suff
            if suff.sufficient: break
            # Cheap re-query expansions
            if "definition_of" in (suff.missing[0] if suff.missing else ""):
                name = suff.missing[0].split(":",1)[1]
                query = query.model_copy(update={"query": name})
            elif "implementation_span" in suff.missing:
                query = query.model_copy(update={"include_chat_history": False})
            ctx = self._hybrid.retrieve(query)
        ctx.sufficiency = ctx.sufficiency or check(query, ctx.spans)
        return ctx
```

### `src/api/rag.py` (extend)

```py
from rag.agentic_planner import AgenticPlanner
_agentic = AgenticPlanner(_hybrid)

@router.post("/agentic-retrieve", response_model=RetrievedContext)
async def rag_agentic(query: RetrievalQuery):
    return _agentic.run(query)
```

## Acceptance

- `test-fix` mode parses `latest_error` and includes test+impl spans.
- Planner never makes more than `AGENTIC_BUDGETS[mode]` retrieval calls.
- `sufficiency.missing` is populated when context is insufficient.

---

# PHASE 11 — Hardware probe + `/api/v1/runtime/profile` + `scripts/probe_hardware.py`

## Files to create

| File | Why |
| --- | --- |
| `src/runtime/__init__.py` | empty marker. |
| `src/runtime/probe.py` | in-container probe; best-effort. |
| `src/runtime/hardware_profile.py` | DB load/store + merge. |
| `src/api/runtime.py` | router for `/runtime/*`. |
| `scripts/probe_hardware.py` | stdlib-only host probe. |

### `src/runtime/probe.py`

```py
import platform, shutil, subprocess, os, re
from datetime import datetime
from models.runtime import HardwareProfile, GPUDevice

def _run(cmd: list[str], timeout: int = 4) -> str:
    bin_ = shutil.which(cmd[0])
    if not bin_: return ""
    try:
        p = subprocess.run([bin_, *cmd[1:]], capture_output=True, text=True,
                           timeout=timeout, check=False)
        return p.stdout
    except (subprocess.SubprocessError, OSError):
        return ""

def _read(path: str) -> str:
    try:
        return open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return ""

def _cpu() -> tuple[str | None, int, int]:
    model = None
    cores = 0; threads = os.cpu_count() or 0
    info = _read("/proc/cpuinfo")
    if info:
        for line in info.splitlines():
            if line.startswith("model name") and not model:
                model = line.split(":",1)[1].strip()
            if line.startswith("cpu cores"):
                try: cores = max(cores, int(line.split(":",1)[1].strip()))
                except ValueError: pass
    if not model:
        model = platform.processor() or None
    return model, cores or threads, threads

def _ram() -> int:
    info = _read("/proc/meminfo")
    for line in info.splitlines():
        if line.startswith("MemTotal:"):
            kb = int(line.split()[1])
            return kb * 1024
    return 0

def _nvidia() -> list[GPUDevice]:
    out = _run(["nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.free,driver_version,compute_cap",
                "--format=csv,noheader,nounits"])
    gpus: list[GPUDevice] = []
    for row in out.splitlines():
        parts = [c.strip() for c in row.split(",")]
        if len(parts) < 6: continue
        try:
            gpus.append(GPUDevice(
                vendor="nvidia", name=parts[0], uuid=parts[1],
                vram_bytes=int(float(parts[2])) * 1024 * 1024,
                free_vram_bytes=int(float(parts[3])) * 1024 * 1024,
                driver=parts[4], compute_capability=parts[5],
                supports_cuda=True, supports_vulkan=True,
            ))
        except ValueError:
            continue
    return gpus

_GFX_RE = re.compile(r"gfx\d+")

def _amd() -> list[GPUDevice]:
    out = _run(["rocminfo"]) or _run(["rocm-smi", "--showproductname"])
    if not out: return []
    name = "AMD GPU"; gfx = None
    for line in out.splitlines():
        m = _GFX_RE.search(line)
        if m: gfx = m.group(0)
        if "Marketing Name" in line: name = line.split(":",1)[1].strip() or name
    return [GPUDevice(vendor="amd", name=name, gfx_target=gfx,
                     supports_rocm=True, supports_vulkan=True)]

def _vulkan_only() -> list[GPUDevice]:
    out = _run(["vulkaninfo", "--summary"])
    devs: list[GPUDevice] = []
    for line in out.splitlines():
        if "deviceName" in line:
            devs.append(GPUDevice(vendor="unknown",
                                  name=line.split("=",1)[-1].strip(),
                                  supports_vulkan=True))
    return devs

def probe(source: str = "container") -> HardwareProfile:
    warnings: list[str] = []
    cpu_model, cores, threads = _cpu()
    ram = _ram()
    gpus = _nvidia()
    if not gpus:
        amd = _amd()
        if amd: gpus.extend(amd)
        else: warnings.append("no nvidia-smi/rocminfo output")
    if not gpus:
        vk = _vulkan_only()
        if vk: gpus.extend(vk)
        else: warnings.append("no vulkaninfo output")
    return HardwareProfile(
        os=f"{platform.system()} {platform.release()}",
        detected_in=source, cpu_model=cpu_model,
        cpu_cores=cores, cpu_threads=threads, ram_bytes=ram, gpus=gpus,
        probe_warnings=warnings, captured_at=datetime.utcnow().isoformat(),
    )
```

### `src/runtime/hardware_profile.py`

```py
import hashlib, json
from libs.db import get_db_connection
from models.runtime import HardwareProfile

def save_profile(profile: HardwareProfile, source: str = "container") -> None:
    with get_db_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO hardware_profile(id, profile_json, source) VALUES (1, ?, ?)",
            (profile.model_dump_json(), source),
        )
        conn.commit()

def load_profile() -> HardwareProfile | None:
    with get_db_connection() as conn:
        row = conn.execute("SELECT profile_json, source FROM hardware_profile WHERE id = 1").fetchone()
    if not row: return None
    p = HardwareProfile.model_validate_json(row["profile_json"])
    p.detected_in = row["source"]
    return p

def hw_hash(profile: HardwareProfile) -> str:
    payload = {"cpu": profile.cpu_model, "ram": profile.ram_bytes,
               "gpus": [(g.vendor, g.name, g.vram_bytes) for g in profile.gpus]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

def get_or_probe() -> HardwareProfile:
    p = load_profile()
    if p: return p
    from runtime.probe import probe
    p = probe("container")
    save_profile(p, "container")
    return p
```

### `src/api/runtime.py`

```py
from fastapi import APIRouter
from models.runtime import HardwareProfile
from runtime.hardware_profile import get_or_probe, save_profile

router = APIRouter(prefix="/api/v1/runtime", tags=["runtime"])

@router.get("/profile", response_model=HardwareProfile)
async def get_runtime_profile():
    return get_or_probe()

@router.post("/profile", response_model=HardwareProfile)
async def submit_runtime_profile(profile: HardwareProfile):
    save_profile(profile, "host")
    return profile
```

### `scripts/probe_hardware.py`

Self-contained, stdlib-only mirror of `src/runtime/probe.py`. Prints
`HardwareProfile` JSON to stdout. Lua plugin runs it and POSTs it.

```py
#!/usr/bin/env python3
"""Host-side hardware probe; mirrors src/runtime/probe.py.
Stdlib-only so it runs without venv on the user's host."""
# (Copy the body of src/runtime/probe.py minus the imports of `models`
#  and emit a plain dict matching the HardwareProfile schema.)
```

## Files to modify

`src/main.py`: `from api import runtime; app.include_router(runtime.router)`.

## Acceptance

- `GET /api/v1/runtime/profile` returns valid JSON inside Docker with no
  GPU access (CPU + RAM populated, `gpus=[]`, warnings populated).
- `POST /api/v1/runtime/profile` with a host-derived profile is persisted
  and preferred on subsequent GETs (`detected_in="host"`).

---

# PHASE 12 — Hardware-aware budget + backend selector + VRAM + offload + multi-GPU + recommend

## Files to create

| File | Why |
| --- | --- |
| `src/rag/budget_from_hardware.py` | `compute_budget(profile) -> HardwareAwareRagBudget`. |
| `src/runtime/backend_selector.py` | picks backend per hardware. |
| `src/runtime/vram_estimator.py` | model + KV cache fit math. |
| `src/runtime/offload_planner.py` | emits `OffloadPlan` flags. |
| `src/runtime/multi_gpu_planner.py` | strategy + risk. |
| `src/runtime/router.py` | composes the recommendation. |
| `src/runtime/performance_modes.py` | mode → defaults. |
| `src/api/runtime.py` (extend) | `GET /runtime/recommend`. |

### `src/rag/budget_from_hardware.py`

```py
from models.runtime import HardwareProfile, HardwareAwareRagBudget

def compute_budget(profile: HardwareProfile) -> HardwareAwareRagBudget:
    if not profile.gpus:
        return HardwareAwareRagBudget(
            max_retrieved_tokens=6000, max_spans=6,
            max_tool_log_tokens=800, max_agentic_retrieval_steps=2,
            reason="CPU-only: keep context small to preserve TTFT",
        )
    best = max(profile.gpus, key=lambda g: g.vram_bytes or 0)
    vram_gb = (best.vram_bytes or 0) / (1024**3)
    if vram_gb >= 24:
        return HardwareAwareRagBudget(max_retrieved_tokens=32000, max_spans=20,
            max_tool_log_tokens=2000, max_agentic_retrieval_steps=6,
            reason=f"{vram_gb:.0f}GB VRAM: large context safe")
    if vram_gb >= 12:
        return HardwareAwareRagBudget(max_retrieved_tokens=18000, max_spans=14,
            max_tool_log_tokens=1500, max_agentic_retrieval_steps=4,
            reason=f"{vram_gb:.0f}GB VRAM: medium context")
    if vram_gb >= 8:
        return HardwareAwareRagBudget(max_retrieved_tokens=10000, max_spans=10,
            max_tool_log_tokens=1000, max_agentic_retrieval_steps=3,
            reason=f"{vram_gb:.0f}GB VRAM: prefer better retrieval over longer context")
    return HardwareAwareRagBudget(max_retrieved_tokens=6000, max_spans=6,
        max_tool_log_tokens=800, max_agentic_retrieval_steps=2,
        reason="low VRAM: keep context tight")
```

### `src/runtime/vram_estimator.py`

```py
from models.runtime import HardwareProfile, ModelRuntimePlan

QUANT_BPP = {"q4_0":4.5, "q4_k_m":4.8, "q5_k_m":5.6, "q6_k":6.6, "q8_0":8.5, "fp16":16, "bf16":16}

def estimate(model_params_b: float, quant: str, ctx_tokens: int,
             layers: int = 32, hidden: int = 4096, kv_heads: int = 32,
             batch: int = 1, concurrent: int = 1) -> int:
    bpp = QUANT_BPP.get(quant, 5.0)
    model_bytes = int(model_params_b * 1e9 * bpp / 8)
    # KV cache: 2 * layers * kv_heads * (hidden/kv_heads) * ctx * 2 bytes (fp16) * batch * concurrent
    kv_bytes = 2 * layers * hidden * ctx_tokens * 2 * batch * concurrent
    overhead = 600 * 1024 * 1024
    safety = 1.5 * 1024**3
    return model_bytes + kv_bytes + overhead + int(safety)

def plan(model_name: str, params_b: float, quant: str, ctx_tokens: int,
         profile: HardwareProfile, batch: int = 1, concurrent: int = 1) -> ModelRuntimePlan:
    required = estimate(params_b, quant, ctx_tokens, batch=batch, concurrent=concurrent)
    largest_vram = max((g.vram_bytes or 0) for g in profile.gpus) if profile.gpus else 0
    fits = required <= largest_vram
    rec = ("full GPU offload" if fits else
           "reduce RAG context first; then batch; then smaller model/quant; CPU offload only if allowed")
    return ModelRuntimePlan(
        model_name=model_name, quantization=quant,
        model_bytes=int(params_b * 1e9 * QUANT_BPP.get(quant,5.0)/8),
        context_tokens=ctx_tokens, batch_size=batch,
        expected_concurrent_requests=concurrent,
        kv_cache_bytes_estimate=required - int(params_b * 1e9 * QUANT_BPP.get(quant,5.0)/8),
        required_vram_bytes=required, fits_in_vram=fits, recommendation=rec,
    )
```

### `src/runtime/backend_selector.py`

```py
from models.runtime import HardwareProfile, BackendRecommendation

def select(profile: HardwareProfile, performance_mode: str = "balanced",
           prefer_vendor: str = "auto") -> BackendRecommendation:
    nvidia = [g for g in profile.gpus if g.vendor == "nvidia"]
    amd    = [g for g in profile.gpus if g.vendor == "amd"]
    if (prefer_vendor in {"auto","nvidia"}) and nvidia:
        if len(nvidia) > 1 and performance_mode != "quiet":
            return BackendRecommendation(backend="vllm", accelerator="cuda",
                reason="multi-GPU NVIDIA; vLLM/SGLang for tensor parallel",
                env={"CUDA_VISIBLE_DEVICES": ",".join(g.uuid or str(i) for i, g in enumerate(nvidia))},
                risk="medium")
        return BackendRecommendation(backend="ollama", accelerator="cuda",
            reason="single NVIDIA GPU; Ollama/llama.cpp ideal for dev",
            env={"CUDA_VISIBLE_DEVICES": nvidia[0].uuid or "0"}, risk="low")
    if (prefer_vendor in {"auto","amd"}) and amd:
        primary = amd[0]
        if primary.supports_rocm:
            return BackendRecommendation(backend="llama.cpp", accelerator="rocm",
                reason="AMD ROCm available", env={"ROCR_VISIBLE_DEVICES":"0"}, risk="low")
        if primary.supports_vulkan:
            return BackendRecommendation(backend="llama.cpp", accelerator="vulkan",
                reason="AMD without ROCm; falling back to Vulkan", env={}, risk="medium")
    return BackendRecommendation(backend="llama.cpp", accelerator="cpu",
        reason="no GPU detected; use small quantized model", env={}, risk="medium")
```

### `src/runtime/offload_planner.py`

```py
from models.runtime import OffloadPlan, HardwareProfile, ModelRuntimePlan

def plan_offload(model_plan: ModelRuntimePlan, profile: HardwareProfile) -> OffloadPlan:
    if not profile.gpus or not model_plan.fits_in_vram:
        return OffloadPlan(gpu_layers=0, context_size=model_plan.context_tokens,
                           batch_size=model_plan.batch_size, split_mode="none")
    if len(profile.gpus) == 1:
        return OffloadPlan(gpu_layers="all", main_gpu=0,
                           context_size=model_plan.context_tokens,
                           batch_size=model_plan.batch_size, split_mode="none")
    sizes = [g.vram_bytes or 0 for g in profile.gpus]
    total = sum(sizes) or 1
    split = [round(s/total, 3) for s in sizes]
    main = max(range(len(sizes)), key=lambda i: sizes[i])
    return OffloadPlan(gpu_layers="all", main_gpu=main, tensor_split=split,
                       split_mode="layer", context_size=model_plan.context_tokens,
                       batch_size=model_plan.batch_size)
```

### `src/runtime/multi_gpu_planner.py`

```py
from models.runtime import HardwareProfile
def strategy(profile: HardwareProfile, model_fits_one: bool, throughput_oriented: bool) -> tuple[str, str]:
    gs = profile.gpus
    if not gs:                    return "single_gpu", "no GPU"
    if len(gs) == 1:              return "single_gpu", "one GPU"
    if model_fits_one and throughput_oriented:
        return "data_parallel", "many requests; replicate per GPU"
    if not model_fits_one:
        sizes = sorted((g.vram_bytes or 0) for g in gs)
        if sizes[-1] >= 2 * (sizes[0] or 1):
            return "layer_split", "mismatched GPUs; use largest as main"
        return "tensor_parallel", "equal GPUs; tensor parallel"
    return "single_gpu", "default"
```

### `src/runtime/router.py`

```py
from models.runtime import BackendRecommendation
from runtime.backend_selector import select
from runtime.hardware_profile import get_or_probe
from runtime.performance_modes import load_mode

def recommend(prefer_vendor: str = "auto") -> BackendRecommendation:
    profile = get_or_probe()
    mode = load_mode()
    return select(profile, performance_mode=mode, prefer_vendor=prefer_vendor)
```

### `src/runtime/performance_modes.py`

```py
import os
def load_mode() -> str:
    return os.getenv("RAG_PERFORMANCE_MODE", "balanced")
```

### `src/api/runtime.py` (extend)

```py
from runtime.router import recommend

@router.get("/recommend", response_model=BackendRecommendation)
async def runtime_recommend(prefer_vendor: str = "auto"):
    return recommend(prefer_vendor)
```

### Wire HW budget into Phase 7

In `src/rag/hybrid_retriever.py` replace `hardware_cap=None` with:

```py
from rag.budget_from_hardware import compute_budget
from runtime.hardware_profile import get_or_probe
hardware_cap = compute_budget(get_or_probe())
```

Cache profile/budget at module load to avoid hitting DB per call.

## Acceptance

- `GET /api/v1/runtime/recommend` returns advisory `BackendRecommendation`;
  CPU-only profile yields `accelerator="cpu"`.
- RAG budget tightens on CPU-only / low-VRAM hardware (verify with
  hand-crafted profile via `POST /runtime/profile`).

---

# PHASE 13 — Benchmark + performance modes config

## Context

Benchmark talks to a **local OpenAI-compatible endpoint** (Ollama
default `http://localhost:11434/v1` or llama.cpp). User supplies
`backend, model, context_tokens, prompt`.

`psutil` is in `requirements.txt`. nvidia-smi may not exist — handle.

## Files to create

| File | Why |
| --- | --- |
| `src/runtime/benchmark.py` | runs the bench. |
| `src/api/runtime.py` (extend) | `POST /runtime/benchmark`. |
| `src/libs/configs.py` (extend) | TOML loader for `rag-service.toml`. |

### `src/runtime/benchmark.py`

```py
import json, time, shutil, subprocess
from urllib.request import Request, urlopen
from models.runtime import BenchmarkResult
from libs.db import get_db_connection
from runtime.hardware_profile import get_or_probe, hw_hash

def _peak_vram_bytes() -> int | None:
    rg = shutil.which("nvidia-smi")
    if not rg: return None
    try:
        p = subprocess.run([rg, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=2, check=False)
        return max(int(x.strip()) for x in p.stdout.splitlines()) * 1024 * 1024
    except (ValueError, subprocess.SubprocessError, OSError):
        return None

def run(backend: str, model: str, endpoint: str, context_tokens: int,
        prompt: str | None = None) -> BenchmarkResult:
    body = json.dumps({
        "model": model,
        "messages": [{"role":"user", "content": prompt or "Say hello briefly."}],
        "stream": True, "max_tokens": 64,
    }).encode()
    req = Request(endpoint.rstrip("/") + "/chat/completions",
                  data=body, headers={"Content-Type":"application/json"})
    t0 = time.perf_counter(); ttft = None; tokens_out = 0
    try:
        with urlopen(req, timeout=60) as resp:
            for chunk in resp:
                if ttft is None: ttft = (time.perf_counter() - t0) * 1000
                tokens_out += chunk.count(b"\n")  # heuristic on SSE
    except Exception as e:                                # noqa: BLE001
        return BenchmarkResult(backend=backend, model=model, context_tokens=context_tokens,
                               prompt_eval_tps=0, decode_tps=0, ttft_ms=0, **{"pass": False})
    total_ms = (time.perf_counter() - t0) * 1000
    decode_tps = tokens_out / max(0.001, (total_ms - (ttft or 0)) / 1000)
    result = BenchmarkResult(backend=backend, model=model, context_tokens=context_tokens,
                             prompt_eval_tps=0.0, decode_tps=decode_tps, ttft_ms=ttft or 0,
                             peak_vram_bytes=_peak_vram_bytes(), cpu_percent=0.0,
                             **{"pass": decode_tps > 0.5})
    profile = get_or_probe()
    with get_db_connection() as conn:
        conn.execute("""INSERT OR REPLACE INTO benchmark_cache
                        (hw_hash, backend, model, context_tokens, result_json)
                        VALUES (?, ?, ?, ?, ?)""",
                     (hw_hash(profile), backend, model, context_tokens, result.model_dump_json(by_alias=True)))
        conn.commit()
    return result
```

### `src/api/runtime.py` (extend)

```py
from pydantic import BaseModel
from runtime.benchmark import run as run_bench

class BenchmarkRequest(BaseModel):
    backend: str
    model: str
    endpoint: str
    context_tokens: int = 2048
    prompt: str | None = None

@router.post("/benchmark", response_model=BenchmarkResult)
async def runtime_benchmark(req: BenchmarkRequest):
    return run_bench(req.backend, req.model, req.endpoint, req.context_tokens, req.prompt)
```

### `src/libs/configs.py` (extend, append-only)

```py
import tomllib

RAG_TOML = BASE_DATA_DIR / "rag-service.toml"
def load_toml() -> dict:
    if not RAG_TOML.exists(): return {}
    try: return tomllib.loads(RAG_TOML.read_text())
    except Exception: return {}
```

## Acceptance

- `POST /api/v1/runtime/benchmark` against a running Ollama works and
  caches result by `hw_hash`.
- Subsequent calls with identical args return cached row in <50ms.

---

# PHASE 14 — Evals

## Files to create

| File | Why |
| --- | --- |
| `src/evals/__init__.py` | empty marker. |
| `src/evals/rag_cases.py` | load/save cases from `${DATA_DIR}/evals/rag_cases.jsonl`. |
| `src/evals/metrics.py` | recall@k, precision@k, MRR, etc. |
| `src/evals/runner.py` | runs cases against `HybridRetriever` (no LLM). |
| `src/api/evals.py` | `POST /api/v1/evals/rag/run`, `GET /report`. |

### `src/evals/metrics.py`

```py
def recall_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    if not expected: return 0.0
    top = retrieved[:k]
    return sum(1 for f in expected if f in top) / len(expected)

def precision_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    if k == 0: return 0.0
    top = retrieved[:k]
    return sum(1 for f in top if f in expected) / k

def mrr(retrieved: list[str], expected: set[str]) -> float:
    for i, f in enumerate(retrieved, 1):
        if f in expected: return 1.0 / i
    return 0.0
```

### `src/evals/rag_cases.py`

```py
import json
from libs.configs import BASE_DATA_DIR
from models.evals import RagEvalCase

EVAL_FILE = BASE_DATA_DIR / "evals" / "rag_cases.jsonl"
EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)

def load() -> list[RagEvalCase]:
    if not EVAL_FILE.exists(): return []
    out: list[RagEvalCase] = []
    for line in EVAL_FILE.read_text().splitlines():
        if not line.strip(): continue
        out.append(RagEvalCase.model_validate_json(line))
    return out

def append(case: RagEvalCase) -> None:
    with EVAL_FILE.open("a", encoding="utf-8") as f:
        f.write(case.model_dump_json() + "\n")
```

### `src/evals/runner.py`

```py
from models.evals import EvalReport, EvalRunResult, RagEvalCase
from models.rag import RetrievalQuery
from evals.metrics import recall_at_k, precision_at_k, mrr

def run(hybrid, cases: list[RagEvalCase], k: int = 10) -> EvalReport:
    results: list[EvalRunResult] = []
    trace_ids: list[str] = []
    for c in cases:
        q = RetrievalQuery(query=c.query, base_uri=c.base_uri, mode=c.mode,
                           current_file=c.current_file, latest_error=c.latest_error,
                           top_k=k, include_chat_history=False)
        ctx = hybrid.retrieve(q)
        retrieved = [s.path or s.uri for s in ctx.spans]
        expected = set(c.expected_files)
        bad = set(c.must_not_retrieve) & set(retrieved)
        irrelevant_tokens = sum(s.token_estimate for s in ctx.spans if (s.path or s.uri) not in expected)
        sym_hits = sum(1 for sy in c.expected_symbols for s in ctx.spans if sy in s.content)
        sym_rate = sym_hits / max(1, len(c.expected_symbols))
        results.append(EvalRunResult(
            case_id=c.id, recall_at_k=recall_at_k(retrieved, expected, k),
            precision_at_k=precision_at_k(retrieved, expected, k),
            mrr=mrr(retrieved, expected), expected_symbol_hit_rate=sym_rate,
            irrelevant_context_tokens=irrelevant_tokens,
            inserted_token_count=ctx.token_estimate,
            freshness_error_rate=len(bad) / max(1, len(retrieved)),
            dedupe_savings=0,
        ))
        if ctx.trace_id: trace_ids.append(ctx.trace_id)
    agg = {
        "recall@k": sum(r.recall_at_k for r in results) / max(1, len(results)),
        "precision@k": sum(r.precision_at_k for r in results) / max(1, len(results)),
        "mrr": sum(r.mrr for r in results) / max(1, len(results)),
        "avg_tokens": sum(r.inserted_token_count for r in results) / max(1, len(results)),
    }
    return EvalReport(results=results, aggregate=agg, trace_ids=trace_ids)
```

### `src/api/evals.py`

```py
from fastapi import APIRouter
from rag.hybrid_retriever import HybridRetriever  # ensure singleton
from evals.rag_cases import load as load_cases
from evals.runner import run as run_eval

router = APIRouter(prefix="/api/v1/evals/rag", tags=["evals"])
_last_report = None

@router.post("/run")
async def evals_run():
    global _last_report
    from api.rag import _hybrid    # reuse the live hybrid retriever
    _last_report = run_eval(_hybrid, load_cases())
    return _last_report

@router.get("/report")
async def evals_report():
    return _last_report or {"results": [], "aggregate": {}}
```

## Acceptance

- `POST /api/v1/evals/rag/run` over an empty case file returns
  `aggregate = {recall@k:0,...}` without errors.
- Runner never invokes any LLM (no generation in `HybridRetriever`).

---

# PHASE 15 — Chat-history index + endpoints

## Files to create

| File | Why |
| --- | --- |
| `src/rag/chat_history_index.py` | sanitize, embed, search chat history. Owns its own Chroma collection per resource. |
| `src/api/chat_history.py` | `/api/v1/chat-history/{upsert,delete,purge}`. |
| `src/services/chat_history.py` | SQLite layer for the `chat_history` table. |

### `src/services/chat_history.py`

```py
import hashlib
from libs.db import get_db_connection
from models.chat_history import ChatTurnUpsert
from rag.context_budget import estimate_tokens

_TOOL_PAYLOAD_RE = __import__("re").compile(r"<tool_payload>.*?</tool_payload>", __import__("re").S)
_BASE64_RE       = __import__("re").compile(r"\b[A-Za-z0-9+/]{200,}={0,2}\b")
_ANSI_RE         = __import__("re").compile(r"\x1b\[[0-9;]*[A-Za-z]")
_THINKING_RE     = __import__("re").compile(r"^\s*Thinking\.\.\.\s*$", __import__("re").M)

PASTE_LIMIT = 4096

def sanitize(text: str) -> str:
    t = _TOOL_PAYLOAD_RE.sub("<elided tool payload>", text)
    t = _BASE64_RE.sub("<elided base64>", t)
    t = _ANSI_RE.sub("", t)
    t = _THINKING_RE.sub("", t)
    if len(t) > PASTE_LIMIT:
        t = t[:PASTE_LIMIT] + f"\n<elided {len(t) - PASTE_LIMIT} bytes>"
    return t

def upsert(turn: ChatTurnUpsert) -> int:
    rows = []
    for i, m in enumerate(turn.messages):
        sanitized = sanitize(m.content)
        rows.append((
            turn.base_uri, turn.chat_id, i, m.role, sanitized,
            hashlib.sha256(sanitized.encode()).hexdigest(),
            estimate_tokens(sanitized), turn.title, m.timestamp,
        ))
    with get_db_connection() as conn:
        conn.execute("DELETE FROM chat_history WHERE resource_uri = ? AND chat_id = ?",
                     (turn.base_uri, turn.chat_id))
        conn.executemany(
            """INSERT INTO chat_history(resource_uri, chat_id, message_idx, role,
                  content_sanitized, content_hash, token_estimate, title, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?)""", rows)
        conn.commit()
    return len(rows)

def delete(resource_uri: str, chat_id: str) -> int:
    with get_db_connection() as conn:
        cur = conn.execute("DELETE FROM chat_history WHERE resource_uri=? AND chat_id=?",
                           (resource_uri, chat_id))
        conn.commit(); return cur.rowcount

def purge(resource_uri: str) -> int:
    with get_db_connection() as conn:
        cur = conn.execute("DELETE FROM chat_history WHERE resource_uri=?", (resource_uri,))
        conn.commit(); return cur.rowcount

def list_recent(resource_uri: str, limit: int = 200) -> list[dict]:
    with get_db_connection() as conn:
        rows = conn.execute(
            """SELECT * FROM chat_history WHERE resource_uri=?
               ORDER BY timestamp DESC LIMIT ?""",
            (resource_uri, limit)).fetchall()
    return [dict(r) for r in rows]
```

### `src/rag/chat_history_index.py`

```py
import hashlib, re
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import Document
from libs.configs import CHROMA_PERSIST_DIR
from libs.logger import logger
from models.rag import FileSpan, RetrievalQuery
from rag.context_budget import estimate_tokens
from services.chat_history import sanitize

_DEICTIC = re.compile(r"\b(we|earlier|before|previous(?:ly)?|you said|last time|recall|remember)\b", re.I)

def _collection_for(resource_uri: str) -> str:
    return "chat_" + hashlib.sha1(resource_uri.encode()).hexdigest()[:16]

class ChatHistoryIndex:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

    def _index_for(self, resource_uri: str):
        coll = self._client.get_or_create_collection(_collection_for(resource_uri))
        vs = ChromaVectorStore(chroma_collection=coll)
        sc = StorageContext.from_defaults(vector_store=vs)
        try:
            return VectorStoreIndex.from_vector_store(vs, storage_context=sc)
        except Exception:                              # noqa: BLE001
            return VectorStoreIndex([], storage_context=sc)

    def upsert(self, resource_uri: str, chat_id: str, messages: list[dict]) -> None:
        idx = self._index_for(resource_uri)
        docs = []
        for i, m in enumerate(messages):
            text = sanitize(m["content"])
            docs.append(Document(
                text=text, doc_id=f"{chat_id}#{i}",
                metadata={"resource_uri": resource_uri, "chat_id": chat_id,
                          "message_idx": i, "role": m["role"]},
            ))
        try: idx.refresh_ref_docs(docs)
        except Exception as e:                         # noqa: BLE001
            logger.warning("chat_history upsert failed: %s", e)

    def purge(self, resource_uri: str) -> None:
        try: self._client.delete_collection(_collection_for(resource_uri))
        except Exception:                              # noqa: BLE001
            pass

    def retrieve(self, query: RetrievalQuery) -> list[FileSpan]:
        if not query.include_chat_history: return []
        cond_conv = query.mode == "ask"
        cond_deictic = bool(_DEICTIC.search(query.query))
        if not (cond_conv or cond_deictic): return []
        idx = self._index_for(query.base_uri)
        try:
            engine = idx.as_query_engine(similarity_top_k=max(query.top_k, 6))
            result = engine.query(query.query)
        except Exception as e:                          # noqa: BLE001
            logger.warning("chat_history retrieve failed: %s", e); return []
        spans: list[FileSpan] = []
        cap = max(1, query.top_k // 4)
        for node in (result.source_nodes or [])[:cap]:
            content = str(node.node.get_content())
            md = getattr(node.node, "metadata", {}) or {}
            spans.append(FileSpan(
                uri=f"chat://{md.get('chat_id','?')}#{md.get('message_idx',0)}",
                path=None, start_line=None, end_line=None, content=content,
                reason=f"chat:{md.get('role','msg')}", score=1.0,
                token_estimate=estimate_tokens(content),
                hash=hashlib.sha256(content.encode()).hexdigest(),
                retrieval_sources=["chat_history"], chunk_kind="chat",
            ))
        return spans
```

### `src/api/chat_history.py`

```py
from fastapi import APIRouter
from pydantic import BaseModel
from models.chat_history import ChatTurnUpsert
from services import chat_history as chat_svc
from rag.chat_history_index import ChatHistoryIndex

router = APIRouter(prefix="/api/v1/chat-history", tags=["chat-history"])
_idx = ChatHistoryIndex()

class DeleteRequest(BaseModel):
    base_uri: str
    chat_id: str

class PurgeRequest(BaseModel):
    base_uri: str

@router.post("/upsert")
async def upsert(turn: ChatTurnUpsert):
    n = chat_svc.upsert(turn)
    _idx.upsert(turn.base_uri, turn.chat_id,
                [{"role": m.role, "content": m.content} for m in turn.messages])
    return {"status":"ok","messages_indexed":n}

@router.post("/delete")
async def delete(req: DeleteRequest):
    n = chat_svc.delete(req.base_uri, req.chat_id)
    return {"status":"ok","rows_deleted":n}

@router.post("/purge")
async def purge(req: PurgeRequest):
    n = chat_svc.purge(req.base_uri)
    _idx.purge(req.base_uri)
    return {"status":"ok","rows_deleted":n}
```

### Wire into hybrid retriever

In `src/api/rag.py`:

```py
from rag.chat_history_index import ChatHistoryIndex
_chat_idx = ChatHistoryIndex()
_hybrid = HybridRetriever(semantic=SemanticRetriever(get_index), chat_history=_chat_idx)
```

Underscore alias for backward compat:

```py
@router.post("/chat_history/upsert", include_in_schema=False)
async def _legacy_upsert(turn: ChatTurnUpsert):
    logger.warning("deprecated underscore route hit: /chat_history/upsert")
    return await upsert(turn)
# (repeat for delete/purge)
```

## Acceptance

- `POST /api/v1/chat-history/upsert` writes rows and embeds them.
- `RetrievalQuery(mode="ask", query="what did we decide about X")` surfaces
  matching chat spans with `retrieval_sources=["chat_history"]`.
- `purge` removes both DB rows and the Chroma collection.

---

# PHASE R — Lua client updates

## Context (existing exact code)

`lua/avante/rag_service.lua` (key lines):

- L432-L449 `M.indexing_status(uri)` posts to `/api/v1/indexing_status`
  (underscore, currently broken).
- L388 `M.retrieve(base_uri, query, on_complete)` posts to `/api/v1/retrieve`.
- L306 `M.add_resource(uri)` posts to `/api/v1/add_resource`.
- L466 `M.get_resources()` GETs `/api/v1/resources`.

`lua/avante/path.lua`:

- `History.save(bufnr, history)` at ~L179 — hook point.
- `History.delete(bufnr, filename)` at ~L188 — hook point.
- `generate_project_dirname_in_storage(bufnr)` at L15 — project key.

`lua/avante/config.lua` L361-L382 contains the `rag_service = { ... }` block.

## Files to modify

### `lua/avante/rag_service.lua`

1. **Fix the route mismatch:** change all underscore endpoints to hyphens.

```lua
-- L432-449 indexing_status
local resp = curl.post(M.get_rag_service_url() .. "/api/v1/indexing-status", { ... })
```

2. **Add new wrapper functions** (mirror `M.retrieve` style):

```lua
function M.rag_search(query_body, on_complete)
  -- POST /api/v1/rag/search   body = RetrievalQuery JSON
end
function M.rag_retrieve(query_body, on_complete)        -- /api/v1/rag/retrieve
function M.rag_context(query_body, on_complete)         -- /api/v1/rag/context
function M.rag_agentic_retrieve(query_body, on_complete)-- /api/v1/rag/agentic-retrieve
function M.rag_symbols(body, on_complete)               -- /api/v1/rag/symbols
function M.runtime_profile_get()                        -- GET  /api/v1/runtime/profile
function M.runtime_profile_post(profile_body)           -- POST /api/v1/runtime/profile
function M.runtime_recommend(prefer_vendor)             -- GET  /api/v1/runtime/recommend
function M.chat_history_upsert(turn)                    -- POST /api/v1/chat-history/upsert
function M.chat_history_delete(base_uri, chat_id)       -- POST /api/v1/chat-history/delete
function M.chat_history_purge(base_uri)                 -- POST /api/v1/chat-history/purge
```

All bodies use `vim.json.encode(body)` and `Content-Type: application/json`.
Rewrite URIs in spans via `M.to_local_uri` on responses (mirror existing
`M.retrieve` source-rewriting at L388-417).

3. **Map base_uri in chat history**: chat-history calls use the same
   `to_container_uri` path translation as code resources.

### `lua/avante/config.lua` — append to `rag_service` block

```lua
rag_service = {
  -- existing keys
  ...
  index_chat_history = true,
  chat_history_max_chats = 50,
  chat_history_max_age_days = 30,
  rag_chat_history_max_paste_bytes = 4096,
},
```

### New file `lua/avante/rag_chat_sync.lua`

```lua
local M = {}
local rag = require("avante.rag_service")
local Config = require("avante.config")
local Utils = require("avante.utils")

local pending = {}                -- bufnr -> timer
local DEBOUNCE_MS = 250

local function build_turn(history, project_root)
  local messages = {}
  for i, m in ipairs(history.messages or {}) do
    table.insert(messages, {
      role = m.role or "user",
      content = m.content or "",
      timestamp = m.timestamp or tostring(os.time()),
      tool_name = m.tool_name,
    })
  end
  return {
    base_uri = "file://" .. project_root,   -- to_container_uri done in rag_service.lua
    chat_id = (history.filename or ""):gsub("%.json$",""),
    title = history.title,
    project_root = project_root,
    messages = messages,
    updated_at = os.date("!%Y-%m-%dT%H:%M:%SZ"),
  }
end

function M.on_save(bufnr, history)
  if not Config.rag_service.index_chat_history then return end
  if pending[bufnr] then pending[bufnr]:stop() end
  pending[bufnr] = vim.defer_fn(function()
    pending[bufnr] = nil
    local root = Utils.get_project_root()
    if not root or root == "" then return end
    rag.chat_history_upsert(build_turn(history, root))
  end, DEBOUNCE_MS)
end

function M.on_delete(filename)
  if not Config.rag_service.index_chat_history then return end
  local root = Utils.get_project_root()
  if not root or root == "" then return end
  rag.chat_history_delete("file://" .. root, filename:gsub("%.json$",""))
end

return M
```

### Hook calls into `lua/avante/path.lua`

```lua
-- L179 inside History.save, after history_filepath:write(...):
local ok_sync, sync = pcall(require, "avante.rag_chat_sync")
if ok_sync then sync.on_save(bufnr, history) end

-- L188 inside History.delete, after vim.fs.rm(...):
local ok_sync, sync = pcall(require, "avante.rag_chat_sync")
if ok_sync then sync.on_delete(filename) end
```

## Acceptance

- Calling existing `M.indexing_status(uri)` returns valid JSON (route bug
  fixed).
- Saving a chat results in a `POST /api/v1/chat-history/upsert` (visible
  in `${DATA_DIR}/logs/...`).
- `M.rag_context({...}, cb)` returns ranked spans.

---

# PHASE T — Tests

## Files to create

```
py/rag-service/pytest.ini
py/rag-service/tests/__init__.py
py/rag-service/tests/conftest.py
py/rag-service/tests/test_routes_smoke.py
py/rag-service/tests/test_chunking.py
py/rag-service/tests/test_symbol_index.py
py/rag-service/tests/test_exact_search.py
py/rag-service/tests/test_dedupe.py
py/rag-service/tests/test_reranker.py
py/rag-service/tests/test_hybrid_retriever.py
py/rag-service/tests/test_context_budget.py
py/rag-service/tests/test_freshness.py
py/rag-service/tests/test_chat_history.py
py/rag-service/tests/test_probe.py
py/rag-service/tests/test_vram_estimator.py
py/rag-service/tests/test_backend_selector.py
py/rag-service/tests/test_planner.py
py/rag-service/tests/test_evals_runner.py
```

### `pytest.ini`

```ini
[pytest]
pythonpath = src
testpaths = tests
addopts = -ra -q
asyncio_mode = auto
```

### `conftest.py`

```py
import os, tempfile, pytest
from pathlib import Path

@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # Re-import to pick up new BASE_DATA_DIR
    import importlib, libs.configs, libs.db
    importlib.reload(libs.configs); importlib.reload(libs.db)
    libs.db.init_db()
    yield tmp_path

@pytest.fixture
def fake_repo(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "def foo():\n    return 1\n\nclass Bar:\n    def baz(self):\n        return 2\n"
    )
    (tmp_path / "src" / "test_main.py").write_text(
        "def test_foo():\n    assert foo() == 1\n"
    )
    return tmp_path
```

### Per-file content checklist

- `test_routes_smoke.py`: `from fastapi.testclient import TestClient; from main import app; client = TestClient(app)`. Hit `/api/health`, `/api/v1/readyz`, `/api/v1/resources`, `/api/v1/runtime/profile`. Assert 200.
- `test_chunking.py`: feed a Python file through `split_documents`; assert every chunk has `start_line`, `end_line`, `chunk_kind`, `text_hash`.
- `test_symbol_index.py`: call `replace_symbols_for_file` against the fake_repo Python file; query DB for `symbol_name='foo'` with `symbol_kind='function'`.
- `test_exact_search.py`: build `ExactSearch().retrieve(query=..., base_path=fake_repo)`; assert at least one `FileSpan` with `retrieval_sources=["exact"]`.
- `test_dedupe.py`: feed two `FileSpan`s with identical content; assert `len(deduped)==1` and `tokens_saved>0`.
- `test_reranker.py`: assert exact-tagged span outranks semantic-only span on identical content.
- `test_hybrid_retriever.py`: stub a fake `SemanticRetriever` (returns empty); ensure exact+symbol still flow through to `RetrievedContext`.
- `test_context_budget.py`: budget caps `max_total_tokens`; verify trim.
- `test_freshness.py`: create `node_modules/foo.js`; assert URI ends up in `stale_uris`.
- `test_chat_history.py`: upsert turn → search → purge → empty.
- `test_probe.py`: call `probe()`; assert `cpu_threads>0`, `ram_bytes>0`, no exceptions.
- `test_vram_estimator.py`: assert 70B fp16 model doesn't fit in 24GB profile.
- `test_backend_selector.py`: simulate CPU-only profile → `accelerator="cpu"`.
- `test_planner.py`: `AgenticPlanner` with `mode="ask"` makes exactly one call.
- `test_evals_runner.py`: run with a single hand-crafted case against fake_repo; recall@k > 0.

### Lua test updates

`tests/rag_service_spec.lua`: update any hard-coded `/api/v1/indexing_status`
expectations to `/api/v1/indexing-status`.

## Acceptance

- `pytest -q` inside `py/rag-service/` passes from a clean checkout.
- Lua busted suite still passes.

---

# DEFINITION OF DONE (rolled up)

- All listed phases land; existing `/api/v1/retrieve` response unchanged.
- `ripgrep`, `pciutils`, `vulkan-tools` present in the container image.
- Six new SQLite tables present; `_schema_meta.version='2'`.
- New endpoints respond with documented shapes:
  `/api/v1/rag/{search,retrieve,agentic-retrieve,symbols,context}`,
  `/api/v1/chat-history/{upsert,delete,purge}`,
  `/api/v1/runtime/{profile (GET/POST),recommend,benchmark}`,
  `/api/v1/evals/rag/{run,report}`.
- Underscore-aliased legacy endpoints log deprecation but still work.
- Every retrieval span carries `reason`, `retrieval_sources`,
  `start_line/end_line`, `hash`, `token_estimate`.
- Duplicates/overlapping spans merge; trace records `deduped_tokens_saved`.
- `test-fix` mode parses `latest_error` and surfaces stack-frame spans first.
- Agentic planner respects `AGENTIC_BUDGETS[mode]`.
- Hardware probe never raises; reports `unknown` gracefully; host-submitted
  profile is preferred.
- RAG budget tightens on CPU-only / low-VRAM hardware.
- JSONL traces in `${DATA_DIR}/traces/`.
- Eval runner produces recall/precision/MRR without invoking an LLM.
- Lua client uses hyphenated routes; chat-history sync from Lua works.

