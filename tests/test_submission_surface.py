"""Submission surface validation tests.

Validates the contract between the competition inference script and evaluator:
  - [START]/[STEP]/[END] stdout format compliance
  - /grader endpoint returns normalized score
  - Action string formatting (amount for spend actions)
  - Root inference.py importability
"""
from __future__ import annotations

import re
import subprocess
import sys
import textwrap

import httpx
import pytest

# ---------------------------------------------------------------------------
# Format helpers (imported from root inference.py)
# ---------------------------------------------------------------------------


def test_root_inference_importable():
    """Root inference.py can be imported without errors."""
    import inference  # noqa: F401

    assert hasattr(inference, "log_start")
    assert hasattr(inference, "log_step")
    assert hasattr(inference, "log_end")
    assert hasattr(inference, "run_task")
    assert hasattr(inference, "main")


def test_competition_log_format():
    """[START]/[STEP]/[END] lines match the mandatory format."""
    from inference import log_start, log_step, log_end
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        log_start(task="Basic Crop Growth", env="crop_management", model="test-model")
        log_step(step=1, action="wait", reward=0.0, done=False, error=None)
        log_step(step=2, action="irrigate(2.5)", reward=0.14, done=False, error="some error")
        log_step(step=3, action="harvest", reward=0.80, done=True, error=None)
        log_end(success=True, steps=3, score=0.869, rewards=[0.0, 0.14, 0.80])

    output = buf.getvalue()
    lines = [l for l in output.strip().split("\n") if l.strip()]

    # [START] line
    assert re.match(
        r"^\[START\] task=.+ env=.+ model=.+$", lines[0]
    ), f"Bad [START]: {lines[0]}"

    # [STEP] lines
    for step_line in lines[1:4]:
        assert re.match(
            r"^\[STEP\] step=\d+ action=\S+ reward=\d+\.\d{2} done=(true|false) error=.+$",
            step_line,
        ), f"Bad [STEP]: {step_line}"

    # [END] line
    assert re.match(
        r"^\[END\] success=(true|false) steps=\d+ score=\d+\.\d{3} rewards=[\d.,]+$",
        lines[4],
    ), f"Bad [END]: {lines[4]}"


def test_format_action_includes_amount_for_spend_actions():
    """irrigate and fertilize actions include amount in the string."""
    from inference import _format_action
    from models import CropAction

    assert _format_action(CropAction(action_type="irrigate", amount=2.5)) == "irrigate(2.5)"
    assert _format_action(CropAction(action_type="fertilize", amount=18.0)) == "fertilize(18.0)"
    assert _format_action(CropAction(action_type="harvest", amount=0.0)) == "harvest"
    assert _format_action(CropAction(action_type="wait", amount=0.0)) == "wait"


# ---------------------------------------------------------------------------
# /grader endpoint
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def server():
    """Start the server for grader endpoint testing and tear it down after."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app",
         "--host", "127.0.0.1", "--port", "8321"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    import time
    # Wait for startup
    for _ in range(30):
        try:
            httpx.get("http://127.0.0.1:8321/health", timeout=2.0)
            break
        except Exception:
            time.sleep(0.5)
    yield "http://127.0.0.1:8321"
    proc.terminate()
    proc.wait(timeout=10)


def test_grader_endpoint_returns_normalized_score(server):
    """POST /grader returns score in [0.0, 1.0] with breakdown dict."""
    payload = {
        "actual_yield": 3500.0,
        "target_yield": 5000.0,
        "total_water": 10.0,
        "total_n": 30.0,
        "total_cost": 200.0,
        "budget": 800.0,
        "harvest_dvs": 1.9,
        "harvested": True,
        "actions_taken": [],
        "task_id": 1,
    }
    resp = httpx.post(f"{server}/grader", json=payload, timeout=10.0)
    assert resp.status_code == 200
    data = resp.json()
    assert "score" in data
    assert "breakdown" in data
    assert 0.0 <= data["score"] <= 1.0
    assert isinstance(data["breakdown"], dict)


def test_grader_zero_yield(server):
    """Grader handles zero-yield (no harvest) correctly."""
    payload = {
        "actual_yield": 0.0,
        "target_yield": 5000.0,
        "total_water": 0.0,
        "total_n": 0.0,
        "total_cost": 0.0,
        "budget": 800.0,
        "harvest_dvs": 0.0,
        "harvested": False,
        "actions_taken": [],
        "task_id": 1,
    }
    resp = httpx.post(f"{server}/grader", json=payload, timeout=10.0)
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["score"] <= 1.0
    # No harvest should give a low score
    assert data["score"] < 0.5
