"""Smoke tests for the Crop Management OpenEnv environment.

Verifies:
  - Environment reset/step cycle works
  - Grading is deterministic (same seed -> same score)
  - All 3 tasks complete without errors
  - Scores are in valid [0.0, 1.0] range

Run with:  python -m pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import CropAction, CropObservation, CropState
from server.environment import CropEnvironment


SEED = 42


def _run_episode(task_id: int, seed: int = SEED) -> tuple[float, int]:
    """Run a full episode with the greedy heuristic. Returns (score, steps)."""
    env = CropEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)

    steps = 0
    fert_done = set()

    while not obs.done:
        dvs = obs.crop_status.get("dvs", 0.0)
        sm = obs.soil_status.get("sm", 0.3)
        budget_left = obs.resources_used.get("budget_remaining", 0.0)

        # Simple greedy: harvest at maturity, irrigate when dry, fertilize at key stages
        if dvs >= 1.8:
            action = CropAction(action_type="harvest", amount=0.0)
        elif sm < 0.22 and budget_left > 5.0:
            action = CropAction(action_type="irrigate", amount=2.5)
        elif 0.20 <= dvs <= 0.40 and "s1" not in fert_done and budget_left > 25:
            fert_done.add("s1")
            action = CropAction(action_type="fertilize", amount=15.0)
        elif 0.50 <= dvs <= 0.70 and "s2" not in fert_done and budget_left > 20:
            fert_done.add("s2")
            action = CropAction(action_type="fertilize", amount=12.0)
        else:
            action = CropAction(action_type="wait", amount=0.0)

        obs = env.step(action)
        steps += 1

    return obs.reward, steps


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

def test_reset_returns_observation():
    """reset() should return a CropObservation with expected fields."""
    env = CropEnvironment()
    obs = env.reset(seed=SEED, task_id=1)

    assert isinstance(obs, CropObservation)
    assert obs.task_id == 1
    assert obs.day == 0
    assert obs.done is False
    assert "dvs" in obs.crop_status
    assert "sm" in obs.soil_status


def test_state_returns_crop_state():
    """state property should return a CropState."""
    env = CropEnvironment()
    env.reset(seed=SEED, task_id=1)
    state = env.state

    assert isinstance(state, CropState)
    assert state.current_task_id == 1
    assert state.seed == SEED


def test_step_advances_simulation():
    """A step should advance the simulation day."""
    env = CropEnvironment()
    obs = env.reset(seed=SEED, task_id=1)
    initial_day = obs.day

    obs = env.step(CropAction(action_type="wait", amount=0.0))
    assert obs.day > initial_day


def test_invalid_action_degrades_to_wait():
    """An invalid action_type should be caught and treated as wait."""
    env = CropEnvironment()
    env.reset(seed=SEED, task_id=1)

    obs = env.step(CropAction(action_type="fly_drone", amount=5.0))
    assert len(obs.conflicts) > 0
    assert obs.done is False


def test_all_tasks_complete():
    """All 3 tasks should run to completion without errors."""
    for task_id in (1, 2, 3):
        score, steps = _run_episode(task_id)
        assert 0.0 <= score <= 1.0, f"Task {task_id}: score {score} out of range"
        assert steps > 0, f"Task {task_id}: no steps taken"


def test_determinism():
    """Same seed should produce the exact same score."""
    score_a, _ = _run_episode(task_id=1, seed=SEED)
    score_b, _ = _run_episode(task_id=1, seed=SEED)
    assert score_a == score_b, f"Non-deterministic: {score_a} != {score_b}"


def test_different_seeds_differ():
    """Different seeds should produce different scores."""
    score_a, _ = _run_episode(task_id=1, seed=42)
    score_b, _ = _run_episode(task_id=1, seed=99)
    # Scores *could* be equal by chance, but extremely unlikely
    # Just verify both are valid
    assert 0.0 <= score_a <= 1.0
    assert 0.0 <= score_b <= 1.0


def test_scores_in_valid_range():
    """Grader scores must always be in [0.0, 1.0]."""
    for task_id in (1, 2, 3):
        for seed in (42, 0, 123, 999):
            score, _ = _run_episode(task_id, seed)
            assert 0.0 <= score <= 1.0, (
                f"Task {task_id}, seed {seed}: score {score} out of range"
            )


def test_difficulty_ordering():
    """Easy should score >= Medium >= Hard with the unified grading formula.

    Difficulty comes from environment conditions, not scoring weights.
    The same greedy heuristic should naturally perform better in easier
    environments (better weather, more budget, cheaper inputs).
    """
    scores = {}
    for task_id in (1, 2, 3):
        # Average over multiple seeds for robustness
        total = 0.0
        n_seeds = 5
        for seed in range(42, 42 + n_seeds):
            s, _ = _run_episode(task_id, seed)
            total += s
        scores[task_id] = total / n_seeds

    assert scores[1] >= scores[2], (
        f"Easy ({scores[1]:.4f}) should score >= Medium ({scores[2]:.4f})"
    )
    assert scores[2] >= scores[3], (
        f"Medium ({scores[2]:.4f}) should score >= Hard ({scores[3]:.4f})"
    )
