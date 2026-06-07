"""Live activity dashboard for the RAG service.

Serves a self-contained HTML page at ``GET /`` that shows all RAG and routing
activity in real-time via Server-Sent Events (SSE).

The SSE endpoint polls the SQLite telemetry DB every second for new rows, so
it works correctly with multiple uvicorn workers (no shared in-memory state).
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, StreamingResponse

from observability.telemetry_db import init_telemetry_db

router = APIRouter(tags=["dashboard"])

# ---------------------------------------------------------------------------
# HTML dashboard (self-contained — no external dependencies)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Service — Live Activity</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --yellow: #d29922; --red: #f85149;
    --purple: #bc8cff; --teal: #39d353;
    --fast: #3fb950; --med: #d29922; --slow: #f85149;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font: 13px/1.5 'SF Mono', 'Cascadia Code', 'Fira Code', monospace; }

  header { padding: 16px 24px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 16px; font-weight: 600; color: var(--accent); }
  #status { font-size: 11px; color: var(--muted); margin-left: auto; display: flex; align-items: center; gap: 6px; }
  #dot { width: 8px; height: 8px; border-radius: 50%; background: var(--red); transition: background .3s; }
  #dot.live { background: var(--green); animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  .stats-bar { display: flex; gap: 1px; background: var(--border); border-bottom: 1px solid var(--border); }
  .stat { flex: 1; background: var(--surface); padding: 10px 16px; }
  .stat-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; }
  .stat-value { font-size: 20px; font-weight: 700; color: var(--text); margin-top: 2px; }
  .stat-value.accent { color: var(--accent); }
  .stat-value.green  { color: var(--green); }

  .toolbar { padding: 8px 16px; border-bottom: 1px solid var(--border); display: flex; gap: 8px; align-items: center; }
  .toolbar label { font-size: 11px; color: var(--muted); }
  .toolbar select, .toolbar input[type=text] {
    background: var(--surface); border: 1px solid var(--border); color: var(--text);
    padding: 4px 8px; border-radius: 4px; font: inherit; font-size: 12px;
  }
  #clear-btn {
    margin-left: auto; background: transparent; border: 1px solid var(--border);
    color: var(--muted); padding: 4px 10px; border-radius: 4px; cursor: pointer; font: inherit; font-size: 11px;
  }
  #clear-btn:hover { border-color: var(--accent); color: var(--accent); }

  #feed { overflow-y: auto; height: calc(100vh - 160px); padding: 8px; }

  .card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
    margin-bottom: 6px; overflow: hidden; transition: border-color .2s;
  }
  .card:hover { border-color: #484f58; }
  .card.new { animation: slide-in .25s ease-out; }
  @keyframes slide-in { from { opacity:0; transform:translateY(-8px); } to { opacity:1; transform:none; } }

  .card-header {
    display: flex; align-items: center; gap: 8px; padding: 8px 12px;
    cursor: pointer; user-select: none;
  }
  .card-header:hover { background: rgba(255,255,255,.03); }

  .badge {
    font-size: 10px; padding: 2px 7px; border-radius: 10px; font-weight: 600;
    white-space: nowrap; letter-spacing: .3px;
  }
  .badge-mode-ask       { background: #1f2d5a; color: #79b8ff; }
  .badge-mode-search    { background: #1b3a2d; color: #56d364; }
  .badge-mode-edit-small{ background: #3b2a1b; color: #e3b341; }
  .badge-mode-refactor  { background: #2e1b47; color: #bc8cff; }
  .badge-mode-test-fix  { background: #2a1e3a; color: #d2a8ff; }
  .badge-mode           { background: #21262d; color: var(--muted); }

  .badge-backend { background: #21262d; color: #79b8ff; }
  .badge-shadow  { background: #3b2a1b; color: #e3b341; }

  .query-text { flex: 1; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12px; }
  .latency { margin-left: auto; font-weight: 700; white-space: nowrap; }
  .latency.fast { color: var(--fast); }
  .latency.med  { color: var(--med);  }
  .latency.slow { color: var(--slow); }
  .chevron { color: var(--muted); font-size: 10px; transition: transform .2s; }
  .chevron.open { transform: rotate(90deg); }

  .card-body { border-top: 1px solid var(--border); padding: 10px 12px; display: none; }
  .card-body.open { display: block; }

  .meta-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; font-size: 11px; color: var(--muted); }
  .meta-row span { display: flex; align-items: center; gap: 4px; }

  .section-title { font-size: 10px; text-transform: uppercase; letter-spacing: .5px; color: var(--muted); margin: 10px 0 4px; }

  .stages-timeline { display: flex; flex-direction: column; gap: 4px; }
  .stage-row { display: flex; align-items: center; gap: 8px; font-size: 11px; }
  .stage-name { width: 140px; color: var(--muted); flex-shrink: 0; }
  .stage-bar-wrap { flex: 1; background: var(--bg); border-radius: 2px; height: 10px; overflow: hidden; }
  .stage-bar { height: 100%; border-radius: 2px; background: var(--accent); opacity: .7; }
  .stage-t { width: 55px; text-align: right; color: var(--muted); flex-shrink: 0; }

  .backend-runs { display: flex; flex-direction: column; gap: 4px; }
  .backend-run { background: var(--bg); border: 1px solid var(--border); border-radius: 4px; padding: 6px 10px; font-size: 11px; }
  .backend-run .br-header { display: flex; gap: 8px; align-items: center; }
  .br-name { font-weight: 600; color: var(--accent); }
  .br-tag  { background: #21262d; border-radius: 3px; padding: 1px 5px; color: var(--muted); font-size: 10px; }
  .br-tag.error { background: #3d1c1c; color: var(--red); }
  .br-latency { margin-left: auto; font-weight: 600; }

  .tokens-row { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 4px; }
  .tok { display: flex; flex-direction: column; align-items: center; }
  .tok-val { font-size: 16px; font-weight: 700; color: var(--text); }
  .tok-lbl { font-size: 9px; text-transform: uppercase; color: var(--muted); letter-spacing: .4px; }

  .time-label { font-size: 10px; color: var(--muted); }

  #empty { text-align: center; padding: 80px 20px; color: var(--muted); font-size: 13px; }
  #empty svg { opacity: .2; margin-bottom: 12px; }
</style>
</head>
<body>

<header>
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
    <circle cx="10" cy="10" r="9" stroke="#58a6ff" stroke-width="1.5"/>
    <path d="M6 10h8M10 6v8" stroke="#58a6ff" stroke-width="1.5" stroke-linecap="round"/>
  </svg>
  <h1>RAG Service — Live Activity</h1>
  <div id="status"><span id="dot"></span><span id="status-text">connecting…</span></div>
</header>

<div class="stats-bar">
  <div class="stat"><div class="stat-label">Requests</div><div class="stat-value accent" id="s-total">0</div></div>
  <div class="stat"><div class="stat-label">Avg Latency</div><div class="stat-value" id="s-latency">—</div></div>
  <div class="stat"><div class="stat-label">Tokens Inserted</div><div class="stat-value green" id="s-tokens">0</div></div>
  <div class="stat"><div class="stat-label">Dedup Saved</div><div class="stat-value" id="s-dedup">0</div></div>
  <div class="stat"><div class="stat-label">Backends Seen</div><div class="stat-value" id="s-backends">—</div></div>
</div>

<div class="toolbar">
  <label>Mode:</label>
  <select id="filter-mode">
    <option value="">All</option>
    <option value="ask">ask</option>
    <option value="search">search</option>
    <option value="edit-small">edit-small</option>
    <option value="refactor">refactor</option>
    <option value="test-fix">test-fix</option>
  </select>
  <label style="margin-left:8px">Filter:</label>
  <input type="text" id="filter-text" placeholder="query / backend…" style="width:200px">
  <button id="clear-btn" onclick="clearFeed()">Clear</button>
</div>

<div id="feed"><div id="empty">
  <svg width="40" height="40" viewBox="0 0 40 40"><circle cx="20" cy="20" r="18" stroke="#8b949e" stroke-width="1.5" fill="none"/><path d="M14 20h12M20 14v12" stroke="#8b949e" stroke-width="1.5" stroke-linecap="round"/></svg>
  <div>Waiting for RAG activity…</div>
  <div style="margin-top:6px;font-size:11px">Make a request to see live traces here.</div>
</div></div>

<script>
const feed = document.getElementById('feed');
const empty = document.getElementById('empty');
let events = [];
let stats = { total:0, latencies:[], tokens:0, dedup:0, backends: new Set() };
let es = null;

function latencyClass(ms) {
  if (ms < 200) return 'fast';
  if (ms < 1000) return 'med';
  return 'slow';
}

function modeBadge(mode) {
  const cls = 'badge-mode-' + (mode||'') || 'badge-mode';
  return `<span class="badge ${cls}">${mode||'?'}</span>`;
}

function fmtMs(ms) {
  if (ms == null) return '?';
  if (ms < 1000) return ms.toFixed(0) + 'ms';
  return (ms/1000).toFixed(2) + 's';
}

function fmtTime(iso) {
  if (!iso) return '';
  try { return new Date(iso).toLocaleTimeString(); } catch { return iso; }
}

function escHtml(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function buildCard(ev) {
  const lat = ev.retrieval_latency_ms || 0;
  const cls = latencyClass(lat);
  const shadow = ev.is_shadow ? `<span class="badge badge-shadow">shadow</span>` : '';
  const backend = ev.chosen_backend ? `<span class="badge badge-backend">${escHtml(ev.chosen_backend)}</span>` : '';

  // stages timeline
  let stagesHtml = '';
  const stages = ev.stages || [];
  if (stages.length) {
    const maxT = Math.max(...stages.map(s => s.t_ms || 0), 1);
    stagesHtml = `<div class="section-title">Pipeline Stages</div>
    <div class="stages-timeline">` +
      stages.map(s => {
        const pct = Math.round(((s.t_ms||0)/maxT)*100);
        return `<div class="stage-row">
          <span class="stage-name">${escHtml(s.name)}</span>
          <div class="stage-bar-wrap"><div class="stage-bar" style="width:${pct}%"></div></div>
          <span class="stage-t">${fmtMs(s.t_ms)}</span>
        </div>`;
      }).join('') +
    `</div>`;
  }

  // backend runs
  let runsHtml = '';
  const runs = ev.backend_runs || [];
  if (runs.length) {
    runsHtml = `<div class="section-title">Backend Runs</div>
    <div class="backend-runs">` +
      runs.map(r => {
        const err = r.error ? `<span class="br-tag error">error: ${escHtml(r.error)}</span>` : '';
        const shadow_tag = r.is_shadow ? `<span class="br-tag">shadow</span>` : '';
        const primary_tag = r.is_primary ? `<span class="br-tag">primary</span>` : '';
        const lc = latencyClass(r.latency_ms||0);
        return `<div class="backend-run">
          <div class="br-header">
            <span class="br-name">${escHtml(r.backend_name||'?')}</span>
            ${primary_tag}${shadow_tag}
            <span class="br-tag">top_k=${r.top_k||0}</span>
            <span class="br-tag">${r.result_count||0} results</span>
            ${err}
            <span class="br-latency ${lc}">${fmtMs(r.latency_ms)}</span>
          </div>
        </div>`;
      }).join('') +
    `</div>`;
  }

  // token stats
  const tokHtml = `<div class="section-title">Tokens</div>
  <div class="tokens-row">
    <div class="tok"><div class="tok-val">${ev.retrieved_tokens||0}</div><div class="tok-lbl">Retrieved</div></div>
    <div class="tok"><div class="tok-val">${ev.inserted_tokens||0}</div><div class="tok-lbl">Inserted</div></div>
    <div class="tok"><div class="tok-val">${ev.deduped_tokens_saved||0}</div><div class="tok-lbl">Dedup Saved</div></div>
    <div class="tok"><div class="tok-val">${ev.retrieved_spans_count||0}</div><div class="tok-lbl">Spans Retr.</div></div>
    <div class="tok"><div class="tok-val">${ev.inserted_spans_count||0}</div><div class="tok-lbl">Spans Ins.</div></div>
    <div class="tok"><div class="tok-val">${ev.context_budget_used||0}</div><div class="tok-lbl">Budget Used</div></div>
  </div>`;

  const id = 'card-' + escHtml(ev.trace_id || ev.request_id || Math.random().toString(36).slice(2));
  const uri = ev.base_uri || '';
  const shortUri = uri.replace(/^file:\\/\\//,'').replace(/^.*\\//,'') || uri;

  return `<div class="card new" id="${id}">
  <div class="card-header" onclick="toggleCard('${id}')">
    <span class="chevron" id="chev-${id}">▶</span>
    ${modeBadge(ev.mode)}
    ${backend}${shadow}
    <span class="query-text" title="${escHtml(ev.query)}">${escHtml(ev.query)}</span>
    <span class="latency ${cls}">${fmtMs(lat)}</span>
  </div>
  <div class="card-body" id="body-${id}">
    <div class="meta-row">
      <span>🕐 <span class="time-label">${fmtTime(ev._ts)}</span></span>
      ${shortUri ? `<span>📁 <span style="color:var(--muted);font-size:11px">${escHtml(shortUri)}</span></span>` : ''}
      ${ev.freshness_stale_count ? `<span>⚠️ ${ev.freshness_stale_count} stale</span>` : ''}
      ${ev.freshness_recent_count ? `<span>✅ ${ev.freshness_recent_count} fresh</span>` : ''}
      ${ev.rerank_scores && ev.rerank_scores.length ? `<span>🔀 ${ev.rerank_scores.length} reranked</span>` : ''}
    </div>
    ${stagesHtml}
    ${runsHtml}
    ${tokHtml}
  </div>
</div>`;
}

function toggleCard(id) {
  const body = document.getElementById('body-' + id);
  const chev = document.getElementById('chev-' + id);
  const open = body.classList.toggle('open');
  chev.classList.toggle('open', open);
}

function updateStats() {
  document.getElementById('s-total').textContent = stats.total;
  const avg = stats.latencies.length
    ? (stats.latencies.reduce((a,b)=>a+b,0)/stats.latencies.length)
    : null;
  document.getElementById('s-latency').textContent = avg != null ? fmtMs(avg) : '—';
  const lc = avg != null ? latencyClass(avg) : '';
  const lel = document.getElementById('s-latency');
  lel.className = 'stat-value' + (lc ? ' ' + lc : '');
  document.getElementById('s-tokens').textContent = stats.tokens.toLocaleString();
  document.getElementById('s-dedup').textContent = stats.dedup.toLocaleString();
  document.getElementById('s-backends').textContent = stats.backends.size ? [...stats.backends].join(', ') : '—';
}

function applyFilters() {
  const modeF = document.getElementById('filter-mode').value;
  const textF = document.getElementById('filter-text').value.toLowerCase();
  feed.innerHTML = '';
  const filtered = events.filter(ev => {
    if (modeF && ev.mode !== modeF) return false;
    if (textF && !JSON.stringify(ev).toLowerCase().includes(textF)) return false;
    return true;
  });
  if (!filtered.length) {
    feed.appendChild(empty);
    return;
  }
  filtered.forEach(ev => {
    feed.insertAdjacentHTML('afterbegin', buildCard(ev));
    // Remove 'new' animation class after it plays so it doesn't re-animate
    setTimeout(() => {
      const el = document.getElementById('card-' + (ev.trace_id || ev.request_id || ''));
      if (el) el.classList.remove('new');
    }, 400);
  });
}

function ingestEvent(ev) {
  events.unshift(ev); // newest first
  if (events.length > 500) events.pop();

  stats.total++;
  if (ev.retrieval_latency_ms) stats.latencies.push(ev.retrieval_latency_ms);
  if (stats.latencies.length > 200) stats.latencies.shift();
  stats.tokens += (ev.inserted_tokens||0);
  stats.dedup += (ev.deduped_tokens_saved||0);
  if (ev.chosen_backend) stats.backends.add(ev.chosen_backend);

  updateStats();

  // Only add to DOM if it passes the current filter
  const modeF = document.getElementById('filter-mode').value;
  const textF = document.getElementById('filter-text').value.toLowerCase();
  if (modeF && ev.mode !== modeF) return;
  if (textF && !JSON.stringify(ev).toLowerCase().includes(textF)) return;

  if (empty.parentNode === feed) feed.removeChild(empty);
  feed.insertAdjacentHTML('afterbegin', buildCard(ev));
  setTimeout(() => {
    const id = 'card-' + (ev.trace_id || ev.request_id || '');
    const el = document.getElementById(id);
    if (el) el.classList.remove('new');
  }, 400);
}

function clearFeed() {
  events = [];
  stats = { total:0, latencies:[], tokens:0, dedup:0, backends: new Set() };
  updateStats();
  feed.innerHTML = '';
  feed.appendChild(empty);
}

function connect() {
  if (es) { es.close(); es = null; }
  es = new EventSource('/api/v1/dashboard/events');

  es.addEventListener('history', e => {
    try {
      JSON.parse(e.data).forEach(ev => ingestEvent(ev));
    } catch {}
  });

  es.addEventListener('trace', e => {
    try { ingestEvent(JSON.parse(e.data)); } catch {}
  });

  es.onopen = () => {
    document.getElementById('dot').classList.add('live');
    document.getElementById('status-text').textContent = 'live';
  };
  es.onerror = () => {
    document.getElementById('dot').classList.remove('live');
    document.getElementById('status-text').textContent = 'reconnecting…';
    es.close(); es = null;
    setTimeout(connect, 3000);
  };
}

document.getElementById('filter-mode').addEventListener('change', applyFilters);
document.getElementById('filter-text').addEventListener('input', applyFilters);

connect();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

async def _event_stream(db_path: str | None = None) -> AsyncGenerator[str, None]:
    """Poll the telemetry DB every second and stream new traces as SSE.

    Yields SSE-formatted text. Emits:
    - ``event: history`` — the last 50 traces on first connect.
    - ``event: trace``   — each new trace as it arrives.
    - ``event: ping``    — a keepalive every 15 s when idle.
    """
    try:
        conn = init_telemetry_db(db_path)  # type: ignore[arg-type]
    except Exception:
        yield "event: ping\ndata: {}\n\n"
        return

    # Grab history (last 50 rows, newest first).
    # Use "SELECT rowid, *" so that the implicit SQLite rowid is available on
    # sqlite3.Row objects — "SELECT *" omits it for tables with a non-INTEGER
    # PRIMARY KEY, causing row["rowid"] to raise IndexError.
    try:
        rows = conn.execute(
            "SELECT rowid, * FROM retrieval_requests ORDER BY rowid DESC LIMIT 50"
        ).fetchall()
        history = [dict(r) for r in reversed(rows)]
        for item in history:
            item["_ts"] = item.get("created_at")
        yield f"event: history\ndata: {json.dumps(history)}\n\n"
        last_rowid: int = rows[0]["rowid"] if rows else 0
    except Exception:
        last_rowid = 0

    ping_counter = 0
    while True:
        await asyncio.sleep(1)
        ping_counter += 1

        try:
            new_rows = conn.execute(
                "SELECT rowid, * FROM retrieval_requests WHERE rowid > ? ORDER BY rowid ASC",
                (last_rowid,),
            ).fetchall()
        except Exception:
            # DB gone / closed — re-initialise next iteration
            try:
                conn = init_telemetry_db(db_path)  # type: ignore[arg-type]
            except Exception:
                pass
            continue

        for row in new_rows:
            last_rowid = row["rowid"]
            ev = dict(row)
            ev["_ts"] = ev.get("created_at")
            # Attach backend runs for this request
            try:
                runs = conn.execute(
                    "SELECT * FROM backend_search_runs WHERE request_id = ?",
                    (ev["request_id"],),
                ).fetchall()
                ev["backend_runs"] = [dict(r) for r in runs]
            except Exception:
                ev["backend_runs"] = []
            yield f"event: trace\ndata: {json.dumps(ev)}\n\n"

        if ping_counter % 15 == 0:
            yield "event: ping\ndata: {}\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_page() -> HTMLResponse:
    """Serve the live activity dashboard HTML page."""
    return HTMLResponse(content=_DASHBOARD_HTML)


@router.get("/api/v1/dashboard/events")
async def dashboard_events() -> StreamingResponse:
    """SSE stream of RAG trace events (history + live updates)."""
    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )

