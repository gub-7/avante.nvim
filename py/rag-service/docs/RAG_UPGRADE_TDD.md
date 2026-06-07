# TDD: Hardware-Aware Agentic RAG Upgrade for `rag-service`

Status: proposed, all phases in scope (no "minimal first patch" gate).
Owners: rag-service maintainers.
Scope: `py/rag-service/` (Python sidecar) + thin client changes in `lua/avante/rag_service.lua` + `lua/avante/path.lua`.

This document supersedes the earlier draft TDD by incorporating concrete answers
to the open questions about (1) chat-history-as-RAG-source, (2) the existing
client/server route mismatch, (3) phasing, (4) runtime/hardware detection
strategy, and (5) symbol-extraction language coverage.

---

## 0. Resolved design decisions

These are the binding answers that drive every section below.

### 0.1 Chat history as a retrieval source

- **Transport:** **push** from Lua → service. Adds endpoints
  `POST /api/v1/chat_history/upsert` and `POST /api/v1/chat_history/delete`.
  Avoids mounting `Config.history.storage_path` into the container and avoids a
  second host↔container path translation tree.
- **Association:** chat history is indexed as a **sub-namespace of the existing
  code resource** for the same project root. Internally a separate Chroma
  collection (`chat_<resource_id>`) joined at query time, but it is **not** a
  separate `resources` row.
  - Project association on the Lua side uses
    `generate_project_dirname_in_storage(bufnr)`. The Lua client maps that to
    the registered code resource URI before pushing; the service stores
    `(resource_uri, chat_id, message_idx)` keys.
- **Retention/cost controls (avoid overkill):**
  - Index only the **last 50 chats** OR **last 30 days**, whichever is smaller,
    per project. Configurable in `[rag.chat_history]`.
  - Strip before embedding: tool-call argument/result payloads, base64 blobs,
    pasted text larger than 4 KB (replaced with a `<elided n bytes>` marker),
    ANSI color codes, repeated assistant "Thinking..." frames.
  - Keep: user turns, assistant summaries, final code snippets, todo lists,
    error messages, file paths mentioned.
- **Retrieval policy:** chat-history chunks are tagged
  `retrieval_sources=["chat_history"]` and given a default channel weight of
  `1.0` (below `semantic`); they only surface when (a) the user's query is
  conversational ("what did we decide about X"), (b) `mode="ask"`, or
  (c) freshness signals (a recent chat mentions changed files in the query).
- **Privacy:** push is opt-in via `Config.rag_service.index_chat_history`
  (default `true`); a `/api/v1/chat_history/purge` endpoint allows wipe.

### 0.2 Existing route mismatch fix

- The Lua client currently calls `/api/v1/indexing_status` (underscore) while
  the server exposes `/api/v1/indexing-status` (hyphen). This route is dead
  today.
- **Standard going forward:** hyphens, FastAPI-style. Apply to **all** new
  endpoints.
- **Transition:** the server will additionally register underscore aliases for
  every existing/new route during this release cycle and log a deprecation
  warning when an underscore variant is hit. The Lua client is updated to use
  hyphens. Underscore aliases will be removed in the release after this one.

### 0.3 Phasing

All phases are in scope and will land as a single, well-tested upgrade.
Implementation order is preserved for reviewability (see §30), but no phase is
deferred. Sections 31 ("minimal first patch") and 32 ("definition of done")
from the original TDD are merged into §30 of this document.

### 0.4 Runtime / hardware detection

- **Best-effort, in-container first**: probe what is visible from inside the
  container; whatever cannot be detected is reported as `unknown` rather than
  failing the request.
- **Out-of-container host probe**: ship a small standalone script
  (`scripts/probe_hardware.py`, no third-party deps beyond Python stdlib) that
  the Lua plugin can run on the host on first launch and submit via
  `POST /api/v1/runtime/profile`. The submitted profile is cached and
  preferred over in-container detection.
- **Backend recommendations are advisory only.** The service never spawns
  llama.cpp / vLLM / SGLang / Ollama. It only emits
  `BackendRecommendation` JSON for the Lua side / user to act on.
- **RAG providers are assumed local**: the embed/LLM providers used *by the
  RAG service itself* (for indexing + agentic loops) are assumed to be local
  (Ollama, llama.cpp OpenAI-compatible). Remote/paid APIs remain *supported*
  (no code removal) but are not optimized for or included in hardware-aware
  budget logic.

### 0.5 Symbol extraction language coverage

- **v1 mandatory languages:** C, C++, C#, Python, Rust, JavaScript, TypeScript,
  Lua, Java, Go, PHP, Ruby, Swift.
- **ripgrep** is added as a hard dependency in the Dockerfile.
- **Test detection** uses **hard-coded per-language defaults** (filename
  patterns + AST node patterns), with an override hook on `ProjectProfile`
  (`test_patterns: list[str]`) for v1.

---

## 1. Target directory layout

```
py/rag-service/src/
  main.py                    # thin: app, lifespan, router wiring

  api/
    __init__.py
    resources.py             # add/remove/list resources
    retrieve.py              # legacy /api/v1/retrieve
    indexing_status.py
    rag.py                   # /api/v1/rag/{search,retrieve,agentic-retrieve,symbols,context}
    runtime.py               # /api/v1/runtime/{profile,recommend,benchmark}
    evals.py                 # /api/v1/evals/rag/{run,report}
    chat_history.py          # /api/v1/chat-history/{upsert,delete,purge}

  rag/
    engine.py                # orchestrator; owns watcher + indexers
    chunking.py              # structural + CodeSplitter fallback
    watcher.py               # FileSystemHandler + Observer plumbing
    exact_search.py          # ripgrep channel
    symbol_index.py          # tree-sitter symbols, SQLite-backed
    semantic_search.py       # Chroma/LlamaIndex channel
    chat_history_index.py    # chat-history channel (separate Chroma collection)
    hybrid_retriever.py
    reranker.py
    expansion.py             # import/symbol expansion
    dedupe.py
    freshness.py
    context_budget.py
    sufficiency.py
    project_profile.py
    log_summarizer.py
    agentic_planner.py
    citations.py
    budget_from_hardware.py

  runtime/
    probe.py                 # in-container probe
    hardware_profile.py      # cache + merge in-container + host-submitted
    backend_selector.py
    vram_estimator.py
    offload_planner.py
    multi_gpu_planner.py
    benchmark.py
    router.py
    performance_modes.py

  evals/
    rag_cases.py
    runner.py
    metrics.py

  observability/
    trace.py
    jsonl_exporter.py
    metrics.py

  libs/
    configs.py               # extended with RAG/runtime knobs
    db.py                    # schema additions
    logger.py
    utils.py

  models/
    resource.py
    indexing_history.py
    rag.py                   # FileSpan, RetrievalQuery, RetrievedContext, ...
    runtime.py               # HardwareProfile, GPUDevice, BackendRecommendation, ...
    chat_history.py          # ChatMessage, ChatTurnUpsert
    evals.py
```

`main.py` ends up <80 lines: lifespan, leader election, router includes.

---

## 2. Data models

### 2.1 RAG models (`src/models/rag.py`)

```py
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
    retrieval_sources: list[str] = []        # exact | symbol | semantic | chat_history | expansion
    chunk_kind: str | None = None            # function | class | section | test | ...
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

class RetrievedContext(BaseModel):
    response: str | None = None
    spans: list[FileSpan]
    sources: list[SourceDocument]
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
```

### 2.2 Chat history models (`src/models/chat_history.py`)

```py
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "tool", "system"]
    content: str
    timestamp: str
    tool_name: str | None = None      # if role=tool, name only; payload is stripped

class ChatTurnUpsert(BaseModel):
    base_uri: str                     # registered code resource URI for this project
    chat_id: str                      # filename without extension (e.g. "12")
    title: str | None = None
    project_root: str
    messages: list[ChatMessage]
    updated_at: str                   # ISO8601
```

### 2.3 Runtime models (`src/models/runtime.py`)

```py
class GPUDevice(BaseModel):
    vendor: Literal["nvidia", "amd", "intel", "apple", "unknown"]
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
    detected_in: Literal["host", "container", "merged", "unknown"]
    cpu_model: str | None
    cpu_cores: int
    cpu_threads: int
    ram_bytes: int
    gpus: list[GPUDevice]
    probe_warnings: list[str] = []
    captured_at: str

class BackendRecommendation(BaseModel):
    backend: Literal["ollama", "llama.cpp", "vllm", "sglang", "openai-compatible"]
    accelerator: Literal["cuda", "rocm", "vulkan", "cpu"]
    reason: str
    env: dict[str, str] = {}
    launch_args: list[str] = []
    risk: Literal["low", "medium", "high"]

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
    split_mode: Literal["none", "layer", "row"] = "none"
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
    peak_vram_bytes: int | None
    cpu_percent: float
    gpu_percent: float | None
    pass_: bool = Field(alias="pass")

class HardwareAwareRagBudget(BaseModel):
    max_retrieved_tokens: int
    max_spans: int
    max_tool_log_tokens: int
    max_agentic_retrieval_steps: int
    reason: str
```

---

## 3. Database schema additions (`src/libs/db.py`)

Add to `CREATE_TABLES_SQL` (all `IF NOT EXISTS`):

```sql
-- symbols extracted from indexed files
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

-- per-project profile cache
CREATE TABLE IF NOT EXISTS project_profiles (
    resource_uri TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    profile_hash TEXT NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- chat history (denormalised; embeddings live in Chroma)
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

-- cached hardware profile (host-submitted preferred)
CREATE TABLE IF NOT EXISTS hardware_profile (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    profile_json TEXT NOT NULL,
    source TEXT NOT NULL,           -- 'host' | 'container'
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- cached benchmark results keyed by (hardware hash, backend, model)
CREATE TABLE IF NOT EXISTS benchmark_cache (
    hw_hash TEXT NOT NULL,
    backend TEXT NOT NULL,
    model TEXT NOT NULL,
    context_tokens INTEGER NOT NULL,
    result_json TEXT NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (hw_hash, backend, model, context_tokens)
);
```

Migrations: idempotent — `init_db()` already runs `IF NOT EXISTS`. Add an
explicit `schema_version` row in a new `_schema_meta` table and bump from 1→2.

---

## 4. Phase 0 — Safe refactor of `main.py`

| Existing logic | Move to |
| --- | --- |
| `add_resource`, `remove_resource`, `list_resources` | `api/resources.py` |
| `retrieve` | `api/retrieve.py` |
| `get_indexing_status_for_resource` | `api/indexing_status.py` |
| `scan_directory`, `split_documents`, `process_document_batch` | `rag/chunking.py` + `rag/engine.py` |
| `FileSystemHandler` + Observer plumbing | `rag/watcher.py` |
| Chroma/LlamaIndex setup | `rag/semantic_search.py` |
| Provider init at startup | stays in `providers/factory.py`, called from `main.py` lifespan |

**Acceptance:** existing endpoints respond identically; `tests/rag_service_spec.lua`
passes; new pytest smoke (`py/rag-service/tests/test_routes_smoke.py`) starts
the app and hits each preserved route.

---

## 5. Phase 1 — Non-agentic hybrid RAG

### 5.1 Exact search (`rag/exact_search.py`)

- Use ripgrep (`rg --json --line-number --column --no-messages --hidden
  --glob '!.git' <query> <base_path>`). ripgrep already respects `.gitignore`,
  matching `get_pathspec` behavior.
- Fall back to a stdlib `pathlib`/`re` scan if `rg` missing (Dockerfile adds it
  so this only triggers in dev shells).
- Inputs: `query`, `latest_error` (parsed for paths/symbols), `selected_text`
  symbols, `current_file` basename.
- Output: `list[FileSpan]` with `retrieval_sources=["exact"]`.
- Ranking defaults: exact error line > exact symbol > exact filename > fuzzy
  text. Implemented as additive contributions to `RerankScore.exact`.

### 5.2 Symbol index (`rag/symbol_index.py`)

- Reuse `tree_sitter_language_pack` (already pulled in by `CodeSplitter`).
- Languages: C, C++, C#, Python, Rust, JavaScript, TypeScript, Lua, Java, Go,
  PHP, Ruby, Swift.
- Kinds: `function | method | class | interface | type | constant | variable |
  test | module | unknown`.
- Per-language tree-sitter queries live in `src/rag/queries/<lang>.scm`
  (small, hand-written; one query file per language).
- On index of a file: delete prior `symbols` rows where `file_uri=?`, insert
  new ones in a single transaction.
- Endpoint: `POST /api/v1/rag/symbols` with body
  `{ base_uri, q, kinds?: list[str], limit?: int }` → list of symbol hits
  convertible to `FileSpan`.

### 5.3 Structural chunking (`rag/chunking.py`)

- Prefer structural chunks (function/class for code, headings for markdown,
  test-case for tests, config-entry for `*.toml`/`*.yaml`/`*.json`/`*.ini`).
- Fall back to existing `CodeSplitter(chunk_lines=80, overlap=15,
  max_chars=1500)`.
- Metadata on every chunk:
  ```py
  metadata = {
    "uri": uri,
    "orig_doc_id": doc.doc_id,
    "chunk_number": i,
    "language": language,
    "chunk_kind": "function" | "class" | "section" | "test" | "config" | "code",
    "symbols": [...],
    "start_line": int,
    "end_line": int,
    "text_hash": "sha256:...",
  }
  ```

### 5.4 Hybrid retriever (`rag/hybrid_retriever.py`)

Flow: `RetrievalQuery → exact_search → symbol_index → semantic_search →
chat_history_index (if include_chat_history) → merge → dedupe → reranker →
freshness → context_budget → RetrievedContext`.

Channel weights:
```py
WEIGHTS = {
  "exact": 4.0,
  "symbol": 3.5,
  "semantic": 1.5,
  "chat_history": 1.0,
  "metadata": 1.0,
  "recent": 0.75,
  "test": 1.25,
}
```

### 5.5 Reranker (`rag/reranker.py`)

```
final_score = exact + symbol + semantic + chat_history
            + proximity + import_distance + recent_edit + test_relevance
            - token_penalty - duplicate_penalty - stale_penalty
```
Trace records the full `RerankScore` breakdown for every surviving span.

### 5.6 Dedupe (`rag/dedupe.py`)

- Normalize content (strip leading/trailing ws, collapse internal whitespace
  for hashing only, not for output).
- `hash = sha256(normalized)`.
- Merge overlapping `[start_line, end_line]` spans from the same `file_uri`.
- Preserve union of `reason` strings and `retrieval_sources`; keep highest
  `score`.
- Emit `deduped_tokens_saved` to trace.

### 5.7 Context budgeter (`rag/context_budget.py`)

```py
BUDGETS = {
  "ask":        {"max_total_tokens": 6000,  "max_spans": 5,  "max_doc_tokens": 2000, "max_log_tokens": 500},
  "search":     {"max_total_tokens": 8000,  "max_spans": 8,  "max_doc_tokens": 3000, "max_log_tokens": 500},
  "edit-small": {"max_total_tokens": 10000, "max_spans": 6,  "max_doc_tokens": 2000, "max_log_tokens": 1000},
  "test-fix":   {"max_total_tokens": 12000, "max_spans": 8,  "max_doc_tokens": 1500, "max_log_tokens": 2000},
  "refactor":   {"max_total_tokens": 20000, "max_spans": 16, "max_doc_tokens": 3000, "max_log_tokens": 1000},
}
```
Trim by ascending `final_score` until under budget; record dropped spans in
the trace.

### 5.8 Endpoints

- `POST /api/v1/rag/search` → spans only, no generation.
- `POST /api/v1/rag/retrieve` → hybrid one-pass, optional generation when
  `include_response=true`.
- `POST /api/v1/rag/context` → final packed context block + citations, no
  generation.

---

## 6. Phase 2 — Code-aware RAG

### 6.1 Freshness (`rag/freshness.py`)

Signals: current branch, `git status --porcelain` modified files, mtime,
`indexing_history.timestamp`, deprecated/archive path markers
(`/legacy/`, `/deprecated/`, `/old/`), generated/vendor markers
(`/node_modules/`, `/vendor/`, `/dist/`, `/build/`, `/target/`, `/.venv/`),
docs version markers (frontmatter `version:` / path `vN/`).

Behavior: exclude generated/vendor by default; demote stale docs; prefer
modified files. Override via `include_stale=true` on the query.

### 6.2 Import/symbol expansion (`rag/expansion.py`)

```py
class ExpansionBudget(BaseModel):
    max_depth: int = 1
    max_extra_spans: int = 4
    max_extra_tokens: int = 3000
```

Rules (v1, depth ≤ 1):
- Retrieved function body → pull definition spans of types/symbols it
  references but does not define.
- Unresolved symbol in selected span → pull its definition.
- Test failure span → pull tested function + adjacent fixtures.
- Import statement → pull only the imported symbol's definition, not the
  whole module.

### 6.3 Project profile cache (`rag/project_profile.py`)

```py
class ProjectProfile(BaseModel):
    project_name: str
    stack: list[str]
    package_manager: str | None
    test_commands: list[str]
    build_commands: list[str]
    lint_commands: list[str]
    important_paths: list[str]
    generated_paths: list[str]
    conventions: list[str]
    test_patterns: list[str] = []         # per §0.5: override for test detection
    updated_at: str
```

Triggers: changes to `package.json`, `pyproject.toml`, `requirements.txt`,
`Cargo.toml`, `go.mod`, `flake.nix`, `shell.nix`, `Dockerfile`, `Makefile`,
`README.md`. Hash inputs → `profile_hash`; rebuild only on change.

### 6.4 Citations (`rag/citations.py`)

Produce `path:Lstart-Lend` for every span; emit `ContextCitation` in
`RetrievedContext.citations`.

---

## 7. Phase 3 — Agentic RAG

### 7.1 Log summarizer (`rag/log_summarizer.py`)

Keep: command, exit code, file paths, line numbers, stack frames, assertion
diffs, first unique error, last relevant lines (default 20).
Drop: progress bars, duplicate stack traces, dep-install noise, ANSI codes,
repeated warnings, successful output.

Implementation: regex pipeline; cap output at `max_log_tokens` from budget.

### 7.2 Test-failure-first mode (`mode="test-fix"`)

Flow: parse `latest_error` → extract paths/symbols/line numbers → exact
search those first → retrieve test span → retrieve impl span → retrieve
fixtures/helpers → summarize logs → return compact context.

### 7.3 Sufficiency check (`rag/sufficiency.py`)

```
if mode="edit-small" and exact edited symbol missing:        insufficient
if mode="test-fix" and impl span for failing test missing:   insufficient
if mode="refactor"  and no callsites for renamed symbol:     insufficient
if mode="ask"       and any code or docs found:              sufficient
```
At most one extra retrieval round before returning insufficiency to caller.

### 7.4 Agentic planner (`rag/agentic_planner.py`)

Actions: `answer_now | retrieve_exact | retrieve_symbol | retrieve_semantic |
expand_imports | inspect_tests | summarize_logs | stop_insufficient`.

Budgets:
```py
AGENTIC_BUDGETS = {
  "ask": 1, "search": 1, "edit-small": 2, "test-fix": 4, "refactor": 6,
}
```
Planner is deterministic: a small rules engine over the current
`RetrievedContext` and `ContextSufficiency`. No LLM call required, but a
provider-backed planner can be enabled with
`RAG_AGENTIC_PLANNER=llm` (uses local provider only).

Endpoint: `POST /api/v1/rag/agentic-retrieve`.

---

## 8. Phase 4 — Hardware-aware runtime

### 8.1 Probe (`runtime/probe.py` + `scripts/probe_hardware.py`)

In-container probe runs:
- `platform.uname()`, `os.cpu_count()`, `/proc/cpuinfo`, `/proc/meminfo`
- `nvidia-smi` (if present)
- `rocminfo` / `rocm-smi` (if present)
- `vulkaninfo --summary` (if present)
- `lspci` (if present)

Each failure → warning into `HardwareProfile.probe_warnings`, field set to
`None`/`"unknown"`. **Detection is best-effort; the request never fails.**

Host probe (`scripts/probe_hardware.py`) is identical in behavior but runs
outside the container. The Lua plugin runs it once and submits via
`POST /api/v1/runtime/profile` with body
`{ "profile": HardwareProfile, "source": "host" }`. The submitted profile is
cached in `hardware_profile` (id=1) and is **preferred** when present.

`GET /api/v1/runtime/profile` returns the cached profile, merging
in-container detail (e.g. visible CUDA devices) when both exist
(`detected_in="merged"`).

### 8.2 Hardware-aware RAG budget (`rag/budget_from_hardware.py`)

```
CPU-only      → 4k–8k
8GB VRAM      → 8k–12k
12–16GB VRAM  → 12k–24k
24GB+ VRAM    → 24k–48k
multi-GPU     → prefer larger model / more throughput, do not blindly grow ctx
```
Caps the per-mode `BUDGETS` from §5.7. Recorded in trace
(`hardware_aware_cap_applied: bool`).

### 8.3 Backend selector + router (`runtime/backend_selector.py`, `runtime/router.py`)

| Hardware | Preferred path |
| --- | --- |
| NVIDIA single GPU | Ollama / llama.cpp for dev; vLLM / SGLang for serving |
| NVIDIA multi-GPU | vLLM / SGLang (tensor/data parallel) |
| AMD ROCm | llama.cpp ROCm, Ollama AMD, vLLM/SGLang ROCm if working |
| AMD Vulkan fallback | llama.cpp Vulkan |
| CPU-only | llama.cpp / Ollama CPU + small quant |

Endpoint: `GET /api/v1/runtime/recommend` → `BackendRecommendation`.
Always **advisory only** (per §0.4); no process spawning.

### 8.4 VRAM estimator (`runtime/vram_estimator.py`)

```
required_vram = model_weights
              + kv_cache(ctx, layers, hidden, kv_heads, dtype)
              + runtime_overhead (default 600 MB)
              + safety_margin   (default reserve_vram_gb)
```
Decision ladder: full GPU offload → reduce RAG context → reduce batch / use
smaller model / lower quant → controlled CPU offload only if allowed.

### 8.5 GPU offload planner (`runtime/offload_planner.py`)

Emits `OffloadPlan` with concrete llama.cpp-style flags
(`--n-gpu-layers`, `--main-gpu`, `--tensor-split`, `--split-mode`,
`--ctx-size`, `--batch-size`, `--ubatch-size`, `--cache-type-k/v`).

### 8.6 Device selection (NVIDIA/AMD)

Honors `[runtime.gpu]` config:
```toml
[runtime.gpu]
prefer_vendor = "auto"      # auto | nvidia | amd
visible_devices = []
allow_rocm_override = false
allow_cpu_offload = false
reserve_vram_gb = 1.5
```
Emits `CUDA_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` in
`BackendRecommendation.env`. Risky overrides (`HSA_OVERRIDE_GFX_VERSION`)
only emitted when `allow_rocm_override=true`.

### 8.7 Multi-GPU planner (`runtime/multi_gpu_planner.py`)

Strategies: `single_gpu | data_parallel | tensor_parallel | layer_split`.
Mismatched GPUs → main GPU = largest/fastest, do not equal-split. Detected
PCIe bottleneck (single-x4 link from `lspci -vv` when available) →
prefer smaller model over heavy split. Reported in `reason`.

### 8.8 Benchmark (`runtime/benchmark.py`)

`POST /api/v1/runtime/benchmark` body
`{ backend, model, context_tokens, prompt: str|None }`. Runs against any
local OpenAI-compatible endpoint (Ollama / llama.cpp), measures
prompt_eval_tps, decode_tps, ttft_ms, peak_vram (via `nvidia-smi
--query-gpu=memory.used`), cpu_percent (`psutil`), gpu_percent. Caches in
`benchmark_cache`. Detects performance collapse (decode_tps < 5% of
prompt_eval_tps) → marks `pass=false`.

### 8.9 Performance modes (`runtime/performance_modes.py`)

```toml
[runtime.performance]
mode = "balanced"          # quiet | balanced | max-power
prefer_gpu = true
allow_cpu_offload = false
allow_multi_gpu = true
reserve_vram_gb = 1.5
```
- `quiet`: fewer threads, smaller model, smaller batch.
- `balanced`: reserve desktop resources, moderate batch/ctx.
- `max-power`: maximize safe GPU use; never ignore VRAM/thermal caps.

---

## 9. Phase 5 — Observability + evals

### 9.1 Trace (`observability/trace.py`, `observability/jsonl_exporter.py`)

Stages: `rag.run → exact.search → symbol.search → semantic.search →
chat_history.search → rerank → expand → dedupe → freshness.filter →
context.pack → sufficiency.check → runtime.plan`.

Fields per trace: query, mode, base_uri, retrieved/inserted/dropped spans
counts, retrieved/inserted/deduped tokens, rerank breakdown, freshness
scores, context_budget_used, hardware_profile_hash, backend_recommendation,
retrieval_latency_ms, trace_id.

Sink: JSONL files at `${DATA_DIR}/traces/rag-YYYYMMDD.jsonl`, rotated
daily. `trace_id` returned on every new RAG endpoint response.

### 9.2 Evals (`evals/rag_cases.py`, `evals/runner.py`, `evals/metrics.py`)

```py
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
```

Metrics: `recall@k`, `precision@k`, `MRR`, `expected_symbol_hit_rate`,
`irrelevant_context_tokens`, `inserted_token_count`,
`freshness_error_rate`, `dedupe_savings`.

Endpoints: `POST /api/v1/evals/rag/run`, `GET /api/v1/evals/rag/report`.
Runner does **not** call any LLM (retrieval-only).

---

## 10. Chat history integration — concrete protocol

### 10.1 Lua side (`lua/avante/path.lua` + new `lua/avante/rag_chat_sync.lua`)

After each `History.save` (existing path), `rag_chat_sync` debounces (250ms),
loads the saved JSON, builds `ChatTurnUpsert`, calls
`POST /api/v1/chat-history/upsert`.

Mapping:
- `base_uri`: registered resource URI for `Utils.get_project_root()`. If not
  registered, skip silently.
- `chat_id`: filename stem (`12.json` → `"12"`).
- `messages`: filtered (`role in {user, assistant, system, tool}`),
  per-message sanitization done on the **service side** to keep policy
  centralized.

Opt-out: `Config.rag_service.index_chat_history = false`.

### 10.2 Service side (`api/chat_history.py` + `rag/chat_history_index.py`)

- Upsert: sanitize each message (see §0.1), token-estimate, store in
  `chat_history` table, embed sanitized content into Chroma collection
  `chat_<sha1(resource_uri)>` with metadata
  `{resource_uri, chat_id, message_idx, role, timestamp, title}`.
- Delete: by `(resource_uri, chat_id)`.
- Purge: by `resource_uri` (drops collection + table rows).
- Retention sweep: nightly task drops anything beyond `last 50 chats` or
  `> 30 days`.

### 10.3 Retrieval

`hybrid_retriever` calls `chat_history_index.retrieve()` only when
`include_chat_history=true` AND any of:
- `mode == "ask"`,
- query contains "we", "earlier", "before", "previous", "you said",
- `current_file` matches a path referenced in a chat within last 7 days.

Chat-history spans are tagged `retrieval_sources=["chat_history"]`, capped at
`max_spans // 4` (never dominate code spans), and given weight `1.0`.

---

## 11. Config additions (`src/libs/configs.py`)

New env-driven knobs (all optional, sane defaults):

```
RAG_CHAT_HISTORY_ENABLED          (default true)
RAG_CHAT_HISTORY_MAX_CHATS        (default 50)
RAG_CHAT_HISTORY_MAX_AGE_DAYS     (default 30)
RAG_CHAT_HISTORY_MAX_PASTE_BYTES  (default 4096)

RAG_PERFORMANCE_MODE              (quiet|balanced|max-power, default balanced)
RAG_PREFER_VENDOR                 (auto|nvidia|amd, default auto)
RAG_RESERVE_VRAM_GB               (default 1.5)
RAG_ALLOW_CPU_OFFLOAD             (default false)
RAG_ALLOW_ROCM_OVERRIDE           (default false)

RAG_AGENTIC_PLANNER               (rules|llm, default rules)
RAG_TRACE_ENABLED                 (default true)
```

A TOML loader (`tomllib`, stdlib) reads optional
`${DATA_DIR}/rag-service.toml` to expose richer `[runtime.gpu]`,
`[runtime.performance]`, `[rag.chat_history]` sections without polluting env.

---

## 12. Dockerfile changes

Add (and only add — keep existing layers):

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
      ripgrep \
      pciutils \
      vulkan-tools \
    && rm -rf /var/lib/apt/lists/*
```

`nvidia-smi` / `rocm-smi` are **not** installed; they are expected to be
provided via GPU passthrough when available. Their absence is handled
gracefully (per §0.4).

---

## 13. API compatibility plan

### 13.1 Preserved (unchanged response shape)

```
POST /api/v1/retrieve        -> RetrieveResponse { response, sources }
POST /api/v1/add_resource
POST /api/v1/remove_resource
GET  /api/v1/resources
GET  /api/v1/readyz
GET  /api/health
POST /api/v1/indexing-status (canonical)
POST /api/v1/indexing_status (deprecated alias; logs warning)
```

### 13.2 New endpoints

```
POST /api/v1/rag/search
POST /api/v1/rag/retrieve
POST /api/v1/rag/agentic-retrieve
POST /api/v1/rag/symbols
POST /api/v1/rag/context              -> RagContextResponse
POST /api/v1/chat-history/upsert
POST /api/v1/chat-history/delete
POST /api/v1/chat-history/purge
GET  /api/v1/runtime/profile
POST /api/v1/runtime/profile          (host-submitted)
GET  /api/v1/runtime/recommend
POST /api/v1/runtime/benchmark
POST /api/v1/evals/rag/run
GET  /api/v1/evals/rag/report
```

`RagContextResponse`:
```py
class RagContextResponse(BaseModel):
    context: str
    spans: list[FileSpan]
    citations: list[ContextCitation]
    token_estimate: int
    trace_id: str
    runtime_plan: BackendRecommendation | None
```

### 13.3 Lua client updates (`lua/avante/rag_service.lua`)

- Standardize all calls on hyphens; fix `indexing_status` → `indexing-status`.
- Add: `rag_search`, `rag_context`, `rag_agentic_retrieve`, `rag_symbols`,
  `runtime_profile`, `runtime_recommend`, `chat_history_upsert`,
  `chat_history_delete`.
- New `lua/avante/rag_chat_sync.lua` hooks `History.save`.

---

## 14. Testing strategy

- **Python**: new `py/rag-service/tests/` with pytest. Targets:
  - `test_routes_smoke.py` — every route returns expected shape on empty repo.
  - `test_chunking.py` — structural chunks carry `start_line/end_line/symbols`.
  - `test_symbol_index.py` — symbols extracted for each of the 13 languages.
  - `test_exact_search.py` — ripgrep and fallback paths.
  - `test_hybrid_retriever.py` — exact > symbol > semantic on synthetic repo.
  - `test_dedupe.py` — overlap merge + hash dedupe.
  - `test_context_budget.py` — never exceeds budget; drops lowest scores.
  - `test_chat_history.py` — upsert → retrieve → purge.
  - `test_probe.py` — best-effort probe never raises.
  - `test_vram_estimator.py` — decision ladder behavior.
  - `test_planner.py` — rules planner termination bounds.
  - `test_evals_runner.py` — recall/precision/MRR on tiny fixture set.
- **Lua**: extend `tests/rag_service_spec.lua` to cover URL standardization.
- CI: add a `pytest` job for `py/rag-service/` to `.pre-commit` or the
  existing GH workflow.

---

## 15. Implementation order (single delivery)

```
1.  Refactor main.py per §4 (no behavior change).
2.  Add Dockerfile deps (ripgrep, pciutils, vulkan-tools).
3.  Add models in src/models/{rag,runtime,chat_history,evals}.py.
4.  DB schema additions + _schema_meta version bump.
5.  rag/exact_search.py + /api/v1/rag/search.
6.  rag/chunking.py structural metadata.
7.  rag/symbol_index.py + /api/v1/rag/symbols (Python, Rust, Lua, JS/TS first;
    then C, C++, C#, Java, Go, PHP, Ruby, Swift).
8.  rag/dedupe.py.
9.  rag/reranker.py.
10. rag/hybrid_retriever.py + /api/v1/rag/retrieve + /api/v1/rag/context.
11. rag/context_budget.py.
12. observability/{trace,jsonl_exporter}.py wired into hybrid retriever.
13. rag/freshness.py.
14. rag/expansion.py.
15. rag/project_profile.py + triggers.
16. rag/citations.py wired into RagContextResponse.
17. rag/log_summarizer.py.
18. rag/sufficiency.py.
19. rag/agentic_planner.py + /api/v1/rag/agentic-retrieve.
20. runtime/probe.py + scripts/probe_hardware.py + /api/v1/runtime/profile.
21. rag/budget_from_hardware.py wired into context_budget.
22. runtime/{backend_selector,vram_estimator,offload_planner,multi_gpu_planner,
    router,performance_modes}.py + /api/v1/runtime/recommend.
23. runtime/benchmark.py + /api/v1/runtime/benchmark.
24. evals/{rag_cases,runner,metrics}.py + /api/v1/evals/rag/{run,report}.
25. api/chat_history.py + rag/chat_history_index.py +
    /api/v1/chat-history/{upsert,delete,purge}.
26. Lua: rag_chat_sync.lua + rag_service.lua URL standardization +
    new RPC wrappers.
27. Python pytest suite; Lua spec updates.
28. README + this TDD updates noting any deltas during build.
```

---

## 16. Definition of done

- All existing endpoints respond identically (Lua spec passes).
- All new endpoints land with pytest coverage (smoke + behavior).
- Chat history flows end-to-end from Lua save → service upsert → hybrid
  retrieval surfaces it under conditions in §10.3.
- Exact + symbol + semantic + chat_history retrieval available; no full files
  inserted by default; hard token budget enforced.
- Every span carries `reason`, `retrieval_sources`, `start_line/end_line`,
  and a citation.
- Duplicates and overlapping spans are merged; dedupe savings logged.
- `test-fix` mode parses `latest_error` and surfaces stack-trace spans first.
- Agentic retrieval terminates within `AGENTIC_BUDGETS[mode]` steps.
- Hardware probe never raises; gracefully reports `unknown` for invisible
  devices; host-submitted profile is preferred.
- Backend recommendation endpoint returns advisory JSON only (no process
  spawning anywhere in the service).
- RAG budget shrinks on CPU-only / low-VRAM detection.
- JSONL traces written for every RAG call.
- Eval runner produces recall/precision/MRR + token-cost report against the
  fixture corpus.
- ripgrep + pciutils + vulkan-tools present in the container image.
- Lua client uses hyphenated routes; server still accepts underscores with a
  deprecation warning for one release.

---

## 17. Out of scope (explicit non-goals)

- Spawning local LLM/embedding backends from the service.
- Optimizing hardware-aware budgets for paid remote APIs (still supported,
  just not the focus per §0.4).
- Cross-project chat-history retrieval (chat history is strictly per
  `resource_uri`).
- A second vector store; Chroma remains the only vector backend.
- Callgraph-aware expansion beyond depth 1 (deferred to a follow-up).

