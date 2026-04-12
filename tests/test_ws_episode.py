"""Real WebSocket transport integration tests.

Runs a live uvicorn server in a subprocess and exercises the `/ws` endpoint
through CropEnvClient, matching the actual competition inference path.
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time

import httpx
import pytest

from agent.policy import greedy_action
from client import CropEnvClient
from models import CropAction


SEED = 42


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def ws_base_url() -> str:
    """Start a live uvicorn server for true /ws transport testing."""
    port = _find_free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    base_url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(50):
            try:
                response = httpx.get(f"{base_url}/health", timeout=1.0)
                if response.status_code == 200:
                    yield base_url
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        else:
            raise RuntimeError("Timed out waiting for uvicorn test server")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_ws_full_episode_via_client(ws_base_url: str):
    """Run a complete greedy episode over the real WebSocket transport."""
    sync_client = CropEnvClient(base_url=ws_base_url).sync()
    with sync_client:
        result = sync_client.reset(seed=SEED, task_id=1)
        obs = result.observation

        steps = 0
        while not result.done:
            action_dict = greedy_action(obs, {})
            result = sync_client.step(CropAction(**action_dict))
            obs = result.observation
            steps += 1

            if not result.done:
                assert obs.rubric_reward is None
                assert isinstance(result.reward, (int, float))

        assert result.done
        assert obs.done
        assert obs.rubric_reward is not None
        assert 0.0 <= obs.rubric_reward <= 1.0
        assert steps > 0


def test_ws_deterministic_across_runs(ws_base_url: str):
    """Same seed must produce identical rubric_reward across two real WS runs."""
    results = []
    for _ in range(2):
        sync_client = CropEnvClient(base_url=ws_base_url).sync()
        with sync_client:
            result = sync_client.reset(seed=SEED, task_id=2)
            obs = result.observation
            while not result.done:
                result = sync_client.step(CropAction(**greedy_action(obs, {})))
                obs = result.observation
            results.append(obs.rubric_reward)

    assert results[0] == results[1]


def test_ws_all_tasks_complete(ws_base_url: str):
    """Every public task must terminate over real /ws with a rubric score."""
    for task_id in (1, 2, 3):
        sync_client = CropEnvClient(base_url=ws_base_url).sync()
        with sync_client:
            result = sync_client.reset(seed=SEED, task_id=task_id)
            obs = result.observation
            while not result.done:
                result = sync_client.step(CropAction(**greedy_action(obs, {})))
                obs = result.observation

            assert obs.rubric_reward is not None, f"Task {task_id}: no rubric_reward"
            assert obs.rubric_reward > 0.0, f"Task {task_id}: zero score"
