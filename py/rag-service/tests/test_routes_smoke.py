"""Smoke tests: verify that core HTTP endpoints return 200."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a TestClient wrapping the FastAPI app.

    Scoped to module so the app is only initialised once per test file.
    The ``isolated_data_dir`` fixture (autouse) still provides per-test
    isolation for database state.
    """
    from main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200


def test_readyz(client):
    resp = client.get("/api/v1/readyz")
    assert resp.status_code == 200


def test_resources_list(client):
    resp = client.get("/api/v1/resources")
    assert resp.status_code == 200
    data = resp.json()
    assert "resources" in data


def test_runtime_profile_get(client):
    resp = client.get("/api/v1/runtime/profile")
    assert resp.status_code == 200

