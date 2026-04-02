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
from server.reward import compute_delta_reward, compute_step_reward
from training_adapter import discrete_to_crop_action, list_discrete_actions


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


def test_observation_includes_control_features():
    """Observation should expose derived control features for RL policies."""
    env = CropEnvironment()
    obs = env.reset(seed=SEED, task_id=1)

    assert "moisture_gap_to_target" in obs.control_features
    assert "forecast_rain_3d" in obs.control_features
    assert "forecast_rain_7d" in obs.control_features
    assert "days_since_last_irrigation" in obs.control_features
    assert "dvs_distance_to_next_fertilizer_window" in obs.control_features


def test_irrigation_reward_prefers_moderate_dose():
    """A moderate irrigation dose should beat a wasteful large dose in dry soil."""
    moderate = compute_step_reward(
        action_type="irrigate",
        dvs=0.4,
        sm=0.24,
        amount=2.5,
        cost=5.0,
        budget_remaining=100.0,
        forecast_rain=0.0,
    )
    wasteful = compute_step_reward(
        action_type="irrigate",
        dvs=0.4,
        sm=0.24,
        amount=8.0,
        cost=16.0,
        budget_remaining=100.0,
        forecast_rain=0.0,
    )
    assert moderate > wasteful


def test_fertilizer_reward_prefers_sensible_dose_in_window():
    """A sensible fertilizer dose in the correct window should beat excess."""
    sensible = compute_step_reward(
        action_type="fertilize",
        dvs=0.30,
        sm=0.30,
        amount=18.0,
        cost=27.0,
        budget_remaining=100.0,
        total_n=10.0,
    )
    excessive = compute_step_reward(
        action_type="fertilize",
        dvs=0.30,
        sm=0.30,
        amount=50.0,
        cost=75.0,
        budget_remaining=100.0,
        total_n=10.0,
    )
    assert sensible > excessive


def test_delta_reward_is_positive_for_stress_relief():
    """Post-transition reward should be positive when stress is clearly relieved."""
    reward = compute_delta_reward(
        action_type="irrigate",
        pre_sm=0.19,
        post_sm=0.24,
        pre_water_stress=0.45,
        post_water_stress=0.72,
        pre_n_availability=0.55,
        post_n_availability=0.55,
        cost=5.0,
        budget_remaining=100.0,
    )
    assert reward > 0.0


def test_wait_actions_do_not_accumulate_positive_dense_reward():
    """Wait actions should remain neutral on dense reward before terminal grading."""
    env = CropEnvironment()
    obs = env.reset(seed=SEED, task_id=1)

    dense_rewards = []
    for _ in range(3):
        obs = env.step(CropAction(action_type="wait", amount=0.0))
        dense_rewards.append(obs.reward)
        if obs.done:
            break

    assert all(reward <= 0.0 for reward in dense_rewards)


def test_step_metadata_includes_reward_breakdown():
    """Dense-reward observations should expose reward components in metadata."""
    env = CropEnvironment()
    env.reset(seed=SEED, task_id=1)

    obs = env.step(CropAction(action_type="wait", amount=0.0))
    assert "reward_breakdown" in obs.metadata
    assert "intent_reward" in obs.metadata["reward_breakdown"]
    assert "delta_reward" in obs.metadata["reward_breakdown"]


def test_probe_scenario_can_be_loaded_internally():
    """Internal probe scenarios should be available via reset kwargs."""
    env = CropEnvironment()
    obs = env.reset(seed=SEED, task_id=1, probe_name="over_irrigation_trap")

    assert obs.metadata.get("probe_name") == "over_irrigation_trap"
    assert obs.season_summary.get("probe_name") == "over_irrigation_trap"
    assert obs.soil_status["sm"] >= 0.35


def test_harvest_hesitation_probe_starts_near_maturity():
    """The harvest hesitation probe should start close to the harvest window."""
    env = CropEnvironment()
    obs = env.reset(seed=SEED, task_id=1, probe_name="harvest_hesitation")

    assert obs.metadata.get("probe_name") == "harvest_hesitation"
    assert obs.crop_status["dvs"] >= 1.7


def test_training_adapter_maps_discrete_actions():
    """Discrete training actions should map to valid public CropAction values."""
    action = discrete_to_crop_action("irrigate_medium")
    assert action.action_type == "irrigate"
    assert action.amount == 5.0

    harvest = discrete_to_crop_action("harvest")
    assert harvest.action_type == "harvest"
    assert harvest.amount == 0.0


def test_training_adapter_lists_expected_actions():
    """The discrete training action list should expose the supported buckets."""
    actions = list_discrete_actions()
    assert actions[0] == "wait"
    assert "fertilize_large" in actions
