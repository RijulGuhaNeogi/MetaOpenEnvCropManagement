"""HTTP transport integration tests for the Crop Management OpenEnv server.

Tests the actual FastAPI endpoints that judges and evaluators hit,
not just the in-process environment. Uses FastAPI TestClient (no server needed).

Run with:  python -m pytest tests/test_integration.py -v
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture(scope="module")
def client():
    """Shared TestClient instance for all tests in this module."""
    return TestClient(app)


# -----------------------------------------------------------------------
# Health & metadata endpoints
# -----------------------------------------------------------------------

def test_health_endpoint(client):
    """GET /health should return 200 with healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_tasks_endpoint(client):
    """GET /tasks should return all 3 task definitions."""
    response = client.get("/tasks")
    assert response.status_code == 200
    data = response.json()
    assert "tasks" in data
    assert len(data["tasks"]) == 3
    task_ids = {t["id"] for t in data["tasks"]}
    assert task_ids == {1, 2, 3}


def test_tasks_have_expected_fields(client):
    """Each task should include id, name, difficulty, and instructions."""
    response = client.get("/tasks")
    for task in response.json()["tasks"]:
        assert "id" in task
        assert "name" in task
        assert "difficulty" in task
        assert "instructions" in task


def test_docs_endpoint(client):
    """GET /docs should return the Swagger UI page."""
    response = client.get("/docs")
    assert response.status_code == 200


# -----------------------------------------------------------------------
# Baseline / ceiling endpoints
# -----------------------------------------------------------------------

def test_baseline_endpoint(client):
    """GET /baseline should return deterministic greedy scores for all tasks."""
    response = client.get("/baseline")
    assert response.status_code == 200
    data = response.json()
    assert data["seed"] == 42
    assert data["policy"] == "greedy"
    assert "tasks" in data
    assert "overall_mean" in data
    assert len(data["tasks"]) == 3
    for task_id_str, task_result in data["tasks"].items():
        assert 0.0 <= task_result["score"] <= 1.0
        assert task_result["steps"] > 0


def test_baseline_is_deterministic(client):
    """Two calls to /baseline should return identical scores."""
    r1 = client.get("/baseline").json()
    r2 = client.get("/baseline").json()
    assert r1["overall_mean"] == r2["overall_mean"]
    for task_id in r1["tasks"]:
        assert r1["tasks"][task_id]["score"] == r2["tasks"][task_id]["score"]


def test_ceiling_endpoint(client):
    """GET /ceiling should return deterministic oracle-ceiling scores."""
    response = client.get("/ceiling")
    assert response.status_code == 200
    data = response.json()
    assert data["seed"] == 42
    assert data["policy"] == "oracle_ceiling"
    assert len(data["tasks"]) == 3
    for task_result in data["tasks"].values():
        assert 0.0 <= task_result["score"] <= 1.0
        assert task_result["steps"] > 0


def test_ceiling_outperforms_greedy_on_hidden_tasks(client):
    """Oracle ceiling should beat the greedy baseline on the masked tasks."""
    baseline = client.get("/baseline").json()
    ceiling = client.get("/ceiling").json()

    assert ceiling["tasks"]["2"]["score"] > baseline["tasks"]["2"]["score"]
    assert ceiling["tasks"]["3"]["score"] > baseline["tasks"]["3"]["score"]


# -----------------------------------------------------------------------
# Reset endpoint (HTTP stateless — each call creates a new environment)
# -----------------------------------------------------------------------

def test_reset_endpoint(client):
    """POST /reset should return a valid observation."""
    response = client.post("/reset", json={"seed": 42, "task_id": 1})
    assert response.status_code == 200
    data = response.json()
    # OpenEnv returns the observation directly or nested
    assert "day" in data or "observation" in data
