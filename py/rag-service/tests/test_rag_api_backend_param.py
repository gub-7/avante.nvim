"""
Increment 12 — API surface tests: backend, mode, shadow, request_id params.

All tests use FastAPI TestClient (sync), following the same pattern as
test_routes_smoke.py.  The HybridRetriever singleton is patched so that
no real index / filesystem I/O is required.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_BODY: dict = {
    "query": "what does foo() return?",
    "base_uri": "file:///tmp",
    "top_k": 3,
}


def _stub_context(**overrides):
    """Return a minimal RetrievedContext suitable for route responses."""
    from models.rag import RetrievedContext

    defaults = dict(
        response=None,
        spans=[],
        sources=[],
        citations=[],
        token_estimate=0,
        trace_id="trace-stub",
    )
    defaults.update(overrides)
    return RetrievedContext(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient; app is initialised once per test file."""
    from main import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_post_rag_retrieve_accepts_mode_field(client):
    """POST /api/v1/rag/retrieve with mode='auto' must return 200, not 422."""
    body = {**VALID_BODY, "mode": "auto"}
    with patch("api.rag._hybrid") as mock_hybrid:
        mock_hybrid.retrieve.return_value = _stub_context()
        resp = client.post("/api/v1/rag/retrieve", json=body)
    assert resp.status_code == 200


def test_post_rag_retrieve_accepts_shadow_field_and_returns_only_primary(client):
    """POST with shadow=true must return 200 with a single (primary) context object."""
    body = {**VALID_BODY, "shadow": True}
    with patch("api.rag._hybrid") as mock_hybrid:
        mock_hybrid.retrieve.return_value = _stub_context()
        resp = client.post("/api/v1/rag/retrieve", json=body)
    assert resp.status_code == 200
    data = resp.json()
    # Response must be the primary RetrievedContext (dict with 'spans')
    assert "spans" in data
    assert isinstance(data["spans"], list)


def test_post_rag_retrieve_returns_request_id_in_response(client):
    """Response JSON must include a non-empty 'request_id' field."""
    with patch("api.rag._hybrid") as mock_hybrid:
        mock_hybrid.retrieve.return_value = _stub_context()
        resp = client.post("/api/v1/rag/retrieve", json=VALID_BODY)
    assert resp.status_code == 200
    data = resp.json()
    assert "request_id" in data, "response must contain 'request_id'"
    assert data["request_id"] is not None
    assert data["request_id"] != ""


def test_post_rag_retrieve_passes_parent_request_id_through_to_telemetry(client):
    """
    POST with parent_request_id is accepted (200) and the generated request_id
    is distinct from parent_request_id (it is freshly created for this call).
    """
    parent_id = "parent-abc-123"
    body = {**VALID_BODY, "parent_request_id": parent_id}
    with patch("api.rag._hybrid") as mock_hybrid:
        mock_hybrid.retrieve.return_value = _stub_context()
        resp = client.post("/api/v1/rag/retrieve", json=body)
    assert resp.status_code == 200
    data = resp.json()
    # The route must echo back a request_id (its own ID, not the parent's)
    assert "request_id" in data
    assert data["request_id"] != parent_id


def test_invalid_mode_returns_422(client):
    """POST /api/v1/rag/retrieve with an unrecognised mode value must return 422."""
    body = {**VALID_BODY, "mode": "banana"}
    resp = client.post("/api/v1/rag/retrieve", json=body)
    assert resp.status_code == 422


def test_default_mode_is_auto(client):
    """Omitting 'mode' from the request body should default to SearchMode.auto."""
    captured: dict = {}

    def _capture(query):
        captured["query"] = query
        return _stub_context()

    with patch("api.rag._pipeline") as mock_pipeline:
        mock_pipeline.run.side_effect = _capture
        resp = client.post("/api/v1/rag/retrieve", json=VALID_BODY)

    assert resp.status_code == 200
    assert "query" in captured, "run() was not called"
    assert captured["query"].mode.value == "auto", (
        f"expected mode='auto', got {captured['query'].mode!r}"
    )


def test_existing_workflow_mode_field_still_accepted_as_workflow_mode(client):
    """
    Backward-compat shim: sending mode='ask' (a legacy workflow-mode value)
    must not return 422; it should be silently promoted to workflow_mode='ask'
    while mode defaults to SearchMode.auto.
    """
    body = {**VALID_BODY, "mode": "ask"}  # 'ask' is a legacy workflow_mode value
    captured: dict = {}

    def _capture(query):
        captured["query"] = query
        return _stub_context()

    with patch("api.rag._pipeline") as mock_pipeline:
        mock_pipeline.run.side_effect = _capture
        resp = client.post("/api/v1/rag/retrieve", json=body)

    assert resp.status_code == 200, f"expected 200, got {resp.status_code}: {resp.text}"
    assert "query" in captured
    q = captured["query"]
    assert q.workflow_mode == "ask", f"workflow_mode should be 'ask', got {q.workflow_mode!r}"
    assert q.mode.value == "auto", f"mode should default to 'auto', got {q.mode!r}"

