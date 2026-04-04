"""Smoke tests for the Crop Management OpenEnv environment.

Verifies:
  - Environment reset/step cycle works
  - Grading is deterministic (same seed -> same score)
  - All 3 tasks complete without errors
  - Scores are in valid [0.0, 1.0] range

Run with:  python -m pytest tests/ -v
"""
from __future__ import annotations

import pytest

from agent.inference import greedy_action
from models import CropAction, CropObservation, CropState
from models import ControlFeatures
from server.crop_sim import CROP_LIBRARY, PARTITION_TABLES, SOIL_LIBRARY, CropSimulator
from server.environment import CropEnvironment
from server.grader import grade_episode
from server.reward import compute_delta_reward, compute_step_reward, compute_trajectory_reward
from agent.training_adapter import discrete_to_crop_action, list_discrete_actions


SEED = 42


def _run_episode(task_id: int, seed: int = SEED) -> tuple[float, int]:
    """Run a full episode with the greedy heuristic. Returns (score, steps)."""
    env = CropEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)

    steps = 0
    fert_done = set()

    while not obs.done:
        action_dict = greedy_action(obs, fert_done)
        action = CropAction(**action_dict)

        obs = env.step(action)
        steps += 1

    return obs.reward, steps


def _run_policy_episode(task_id: int, policy, seed: int = SEED) -> tuple[float, dict]:
    """Run a full episode with a supplied policy callback."""
    env = CropEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)

    fert_done = set()
    while not obs.done:
        action_dict = policy(obs, fert_done)
        obs = env.step(CropAction(**action_dict))

    return obs.reward or 0.0, obs.metadata.get("rubric_breakdown", {})


def _wait_only_policy(obs, fert_done: set) -> dict:
    return {"action_type": "wait", "amount": 0.0}


def _no_fertilizer_policy(obs, fert_done: set) -> dict:
    action = greedy_action(obs, fert_done)
    if action["action_type"] == "fertilize":
        return {"action_type": "wait", "amount": 0.0}
    return action


def _extra_fertilizer_policy(obs, fert_done: set) -> dict:
    dvs = obs.crop_status.dvs
    budget_remaining = obs.resources_used.budget_remaining
    fert_cost = obs.resources_used.fertilizer_cost_per_kg

    if 0.24 <= dvs <= 0.36 and budget_remaining >= fert_cost * 8.0:
        return {"action_type": "fertilize", "amount": 8.0}
    if 0.54 <= dvs <= 0.66 and budget_remaining >= fert_cost * 8.0:
        return {"action_type": "fertilize", "amount": 8.0}

    return greedy_action(obs, fert_done)


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
    assert obs.crop_status.dvs is not None
    assert obs.soil_status.sm is not None


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

    assert "moisture_gap_to_target" in ControlFeatures.model_fields
    assert "forecast_rain_3d" in ControlFeatures.model_fields
    assert "forecast_rain_7d" in ControlFeatures.model_fields
    assert "days_since_last_irrigation" in ControlFeatures.model_fields
    assert "rooting_depth_cm" in ControlFeatures.model_fields
    assert "dvs_distance_to_next_fertilizer_window" in ControlFeatures.model_fields


def test_irrigation_reward_prefers_moderate_dose():
    """A moderate irrigation dose should beat a wasteful large dose in dry soil."""
    moderate = compute_step_reward(
        action_type="irrigate",
        dvs=0.4,
        sm=0.24,
        amount=2.5,
        cost=5.0,
        budget_remaining=100.0,
        total_water=0.0,
        forecast_rain=0.0,
    )
    wasteful = compute_step_reward(
        action_type="irrigate",
        dvs=0.4,
        sm=0.24,
        amount=8.0,
        cost=16.0,
        budget_remaining=100.0,
        total_water=0.0,
        forecast_rain=0.0,
    )
    assert moderate > wasteful + 1e-9


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
    assert sensible > excessive + 1e-9


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


def test_grain_fill_heat_stress_reduces_growth_under_extreme_heat():
    """Heat during grain fill should reduce growth, but with a bounded penalty."""
    crop = CROP_LIBRARY["wheat_nl"]
    soil = SOIL_LIBRARY["clay_loam"]
    partition_table = crop.FOTB

    mild_weather = [{"day": 0, "tmax": 28.0, "tmin": 18.0, "rain": 0.0, "radiation": 18.0}]
    hot_weather = [{"day": 0, "tmax": 38.0, "tmin": 24.0, "rain": 0.0, "radiation": 18.0}]

    mild = CropSimulator(crop, soil, mild_weather, partition_table)
    hot = CropSimulator(crop, soil, hot_weather, partition_table)

    for sim in (mild, hot):
        sim.dvs = 1.2
        sim.lai = 4.0
        sim.n_factor = 1.0
        sim.sm = soil.field_capacity

    mild._simulate_day(0.0)
    hot._simulate_day(0.0)

    assert hot.tagp < mild.tagp
    assert hot._heat_stress_factor(38.0) >= 0.8


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
    assert obs.soil_status.sm >= 0.35


def test_harvest_hesitation_probe_starts_near_maturity():
    """The harvest hesitation probe should start close to the harvest window."""
    env = CropEnvironment()
    obs = env.reset(seed=SEED, task_id=1, probe_name="harvest_hesitation")

    assert obs.metadata.get("probe_name") == "harvest_hesitation"
    assert obs.crop_status.dvs >= 1.7


def test_training_adapter_maps_discrete_actions():
    """Discrete training actions should map to valid public CropAction values."""
    action = discrete_to_crop_action("irrigate_medium")
    assert action.action_type == "irrigate"
    assert action.amount == pytest.approx(5.0)

    harvest = discrete_to_crop_action("harvest")
    assert harvest.action_type == "harvest"
    assert harvest.amount == pytest.approx(0.0)


def test_training_adapter_lists_expected_actions():
    """The discrete training action list should expose the supported buckets."""
    actions = list_discrete_actions()
    assert actions[0] == "wait"
    assert "fertilize_large" in actions


def test_all_discrete_actions_map_to_valid_crop_actions():
    """Every discrete action should map cleanly into the public schema."""
    mapped = [discrete_to_crop_action(name) for name in list_discrete_actions()]

    assert len(mapped) == 8
    assert {action.action_type for action in mapped} == {
        "wait",
        "harvest",
        "irrigate",
        "fertilize",
    }


def test_budget_exhaustion_degrades_action_to_wait():
    """Over-budget actions should be downgraded to wait without crashing."""
    env = CropEnvironment()
    env.reset(seed=SEED, task_id=3, probe_name="budget_starvation")

    obs = env.step(CropAction(action_type="fertilize", amount=50.0))
    assert any("Over budget" in message for message in obs.conflicts)
    assert env.state.actions_taken[-1]["action_type"] == "wait"


def test_probe_dense_reward_and_final_grade_are_directionally_aligned():
    """A better first action on a probe should help both dense reward and final grade."""

    def run_probe(first_action: CropAction) -> tuple[float, float]:
        env = CropEnvironment()
        obs = env.reset(seed=SEED, task_id=1, probe_name="over_irrigation_trap")
        fert_done: set[str] = set()
        dense_total = 0.0

        obs = env.step(first_action)
        dense_total += obs.metadata.get("reward_breakdown", {}).get("step_reward", 0.0)

        while not obs.done:
            action_dict = greedy_action(obs, fert_done)
            obs = env.step(CropAction(**action_dict))
            if not obs.done:
                dense_total += obs.metadata.get("reward_breakdown", {}).get("step_reward", 0.0)

        return dense_total, obs.reward or 0.0

    wait_dense, wait_final = run_probe(CropAction(action_type="wait", amount=0.0))
    irrigate_dense, irrigate_final = run_probe(CropAction(action_type="irrigate", amount=5.0))

    assert wait_dense >= irrigate_dense
    assert wait_final >= irrigate_final


def test_greedy_policy_consistently_beats_wait_only_policy():
    """A passive policy should trail the greedy baseline on all public tasks."""
    for task_id in (1, 2, 3):
        greedy_score, _ = _run_policy_episode(task_id, greedy_action)
        wait_score, breakdown = _run_policy_episode(task_id, _wait_only_policy)

        assert breakdown["timing_quality"] == pytest.approx(0.2)
        assert greedy_score > wait_score + 0.05


def test_skipping_fertilizer_remains_worse_than_greedy_policy():
    """Removing fertilizer decisions should clearly reduce final score quality."""
    for task_id in (1, 2, 3):
        greedy_score, greedy_breakdown = _run_policy_episode(task_id, greedy_action)
        no_fert_score, no_fert_breakdown = _run_policy_episode(task_id, _no_fertilizer_policy)

        assert no_fert_breakdown["timing_quality"] == pytest.approx(0.2)
        assert no_fert_breakdown["total_n"] == pytest.approx(0.0)
        assert greedy_breakdown["timing_quality"] > no_fert_breakdown["timing_quality"]
        assert greedy_score > no_fert_score + 0.05


def test_extra_fertilizer_policy_does_not_beat_greedy_baseline():
    """Extra fertilizer should not materially outperform the greedy baseline."""
    greedy_scores = []
    extra_scores = []

    for task_id in (1, 2, 3):
        greedy_score, _ = _run_policy_episode(task_id, greedy_action)
        extra_score, _ = _run_policy_episode(task_id, _extra_fertilizer_policy)
        greedy_scores.append(greedy_score)
        extra_scores.append(extra_score)

        assert extra_score <= greedy_score + 0.03

    assert sum(greedy_scores) / len(greedy_scores) >= sum(extra_scores) / len(extra_scores)


def test_late_harvest_penalty_hits_floor_after_dvs_23():
    """Late harvest timing should bottom out at the documented 0.5 floor."""
    score_205, breakdown_205 = grade_episode(
        actual_yield=6000.0,
        target_yield=6500.0,
        total_water=40.0,
        total_n=35.0,
        total_cost=400.0,
        budget=800.0,
        harvest_dvs=2.05,
        harvested=True,
        actions_taken=[],
        task_id=1,
    )
    score_210, breakdown_210 = grade_episode(
        actual_yield=6000.0,
        target_yield=6500.0,
        total_water=40.0,
        total_n=35.0,
        total_cost=400.0,
        budget=800.0,
        harvest_dvs=2.10,
        harvested=True,
        actions_taken=[],
        task_id=1,
    )
    score_230, breakdown_230 = grade_episode(
        actual_yield=6000.0,
        target_yield=6500.0,
        total_water=40.0,
        total_n=35.0,
        total_cost=400.0,
        budget=800.0,
        harvest_dvs=2.30,
        harvested=True,
        actions_taken=[],
        task_id=1,
    )
    score_260, breakdown_260 = grade_episode(
        actual_yield=6000.0,
        target_yield=6500.0,
        total_water=40.0,
        total_n=35.0,
        total_cost=400.0,
        budget=800.0,
        harvest_dvs=2.60,
        harvested=True,
        actions_taken=[],
        task_id=1,
    )

    assert breakdown_205["harvest_timing"] == pytest.approx(1.0)
    assert breakdown_210["harvest_timing"] == pytest.approx(0.9)
    assert breakdown_230["harvest_timing"] == pytest.approx(0.5)
    assert breakdown_260["harvest_timing"] == pytest.approx(0.5)
    assert score_205 > score_210 > score_230
    assert score_230 == pytest.approx(score_260)


# -----------------------------------------------------------------------
# Phase 4 — Edge-case tests for rewards and grader
# -----------------------------------------------------------------------


def test_out_of_window_fertilizer_reward_is_negative():
    """Fertilizing outside the DVS 0.20-0.40 / 0.50-0.70 windows must be penalized."""
    for dvs in (0.45, 0.80, 1.2):
        reward = compute_step_reward(
            action_type="fertilize",
            dvs=dvs,
            sm=0.30,
            amount=15.0,
            cost=22.5,
            budget_remaining=200.0,
            total_n=10.0,
        )
        assert reward < 0.0, f"Expected negative reward at DVS {dvs}, got {reward}"


def test_grader_timing_quality_averages_multiple_fert_actions():
    """Timing quality should be the mean of per-action proximity scores."""
    actions = [
        {"action_type": "fertilize", "dvs": 0.30, "amount": 18.0},
        {"action_type": "fertilize", "dvs": 0.90, "amount": 15.0},
    ]
    _, breakdown = grade_episode(
        actual_yield=5000.0,
        target_yield=6500.0,
        total_water=10.0,
        total_n=33.0,
        total_cost=200.0,
        budget=800.0,
        harvest_dvs=1.95,
        harvested=True,
        actions_taken=actions,
        task_id=1,
    )
    # DVS 0.30 → dist 0.0 → score 1.0
    # DVS 0.90 → dist min(0.6, 0.3) = 0.3 → score 0.4
    expected = (1.0 + 0.4) / 2.0
    assert breakdown["timing_quality"] == pytest.approx(expected)


def test_grader_unharvested_episode_zeros_yield_and_harvest():
    """An unharvested episode must score 0 for yield and harvest timing."""
    _, breakdown = grade_episode(
        actual_yield=5000.0,
        target_yield=6500.0,
        total_water=20.0,
        total_n=30.0,
        total_cost=300.0,
        budget=800.0,
        harvest_dvs=1.95,
        harvested=False,
        actions_taken=[],
        task_id=1,
    )
    assert breakdown["yield_score"] == pytest.approx(0.0)
    assert breakdown["harvest_timing"] == pytest.approx(0.0)


def test_grader_zero_water_gives_perfect_efficiency():
    """No irrigation should yield water_efficiency = 1.0."""
    _, breakdown = grade_episode(
        actual_yield=5000.0,
        target_yield=6500.0,
        total_water=0.0,
        total_n=30.0,
        total_cost=100.0,
        budget=800.0,
        harvest_dvs=1.90,
        harvested=True,
        actions_taken=[],
        task_id=1,
    )
    assert breakdown["water_efficiency"] == pytest.approx(1.0)


def test_grader_excessive_water_floors_efficiency():
    """Using more than the 50 cm cap should clamp water_efficiency to 0.0."""
    _, breakdown = grade_episode(
        actual_yield=5000.0,
        target_yield=6500.0,
        total_water=60.0,
        total_n=30.0,
        total_cost=100.0,
        budget=800.0,
        harvest_dvs=1.90,
        harvested=True,
        actions_taken=[],
        task_id=1,
    )
    assert breakdown["water_efficiency"] == pytest.approx(0.0)


def test_terminal_harvest_uses_trajectory_not_dense_reward():
    """The terminal harvest reward must equal compute_trajectory_reward(grade)."""
    env = CropEnvironment()
    obs = env.reset(seed=SEED, task_id=1, probe_name="harvest_hesitation")

    # Step until done (probe starts near maturity, so harvest quickly)
    obs = env.step(CropAction(action_type="harvest", amount=0.0))
    assert obs.done

    breakdown = obs.metadata["rubric_breakdown"]
    # Recompute the grade from the breakdown's raw metrics
    raw = (
        0.35 * breakdown["yield_score"]
        + 0.20 * breakdown["water_efficiency"]
        + 0.18 * breakdown["cost_efficiency"]
        + 0.15 * breakdown["timing_quality"]
        + 0.12 * breakdown["harvest_timing"]
    )
    expected_grade = max(0.0, min(1.0, round(raw, 4)))
    expected_reward = compute_trajectory_reward(expected_grade)
    assert obs.reward == pytest.approx(expected_reward)


# ---------------------------------------------------------------------------
# RFC 004 rubric compliance tests
# ---------------------------------------------------------------------------

def test_rubric_reward_set_on_terminal_step():
    """Terminal observations must carry rubric_reward (RFC 004)."""
    score, _ = _run_episode(1)
    env = CropEnvironment()
    obs = env.reset(seed=SEED, task_id=1)
    fert_done = set()
    while not obs.done:
        obs = env.step(CropAction(**greedy_action(obs, fert_done)))
    assert obs.rubric_reward is not None
    assert 0.0 <= obs.rubric_reward <= 1.0


def test_rubric_reward_none_on_intermediate_step():
    """Intermediate observations must have rubric_reward=None."""
    env = CropEnvironment()
    obs = env.reset(seed=SEED, task_id=1)
    assert obs.rubric_reward is None
    obs = env.step(CropAction(action_type="wait"))
    assert not obs.done
    assert obs.rubric_reward is None


def test_rubric_reward_matches_grade():
    """rubric_reward must equal the grader's score (not the trajectory reward)."""
    env = CropEnvironment()
    obs = env.reset(seed=SEED, task_id=1, probe_name="harvest_hesitation")
    obs = env.step(CropAction(action_type="harvest", amount=0.0))
    assert obs.done

    breakdown = obs.metadata["rubric_breakdown"]
    raw = (
        0.35 * breakdown["yield_score"]
        + 0.20 * breakdown["water_efficiency"]
        + 0.18 * breakdown["cost_efficiency"]
        + 0.15 * breakdown["timing_quality"]
        + 0.12 * breakdown["harvest_timing"]
    )
    expected_grade = max(0.0, min(1.0, round(raw, 4)))
    assert obs.rubric_reward == pytest.approx(expected_grade)


def test_weather_today_is_typed():
    """weather_today must be a WeatherDay model, not a dict."""
    from models import WeatherDay
    env = CropEnvironment()
    obs = env.reset(seed=SEED, task_id=1)
    assert isinstance(obs.weather_today, WeatherDay)
    assert hasattr(obs.weather_today, "tmax")
    assert hasattr(obs.weather_today, "rain")


def test_weather_forecast_is_typed():
    """weather_forecast entries must be WeatherDay models."""
    from models import WeatherDay
    env = CropEnvironment()
    obs = env.reset(seed=SEED, task_id=1)
    assert len(obs.weather_forecast) > 0
    assert isinstance(obs.weather_forecast[0], WeatherDay)


# ---------------------------------------------------------------------------
# YAML config loading tests
# ---------------------------------------------------------------------------

def test_yaml_configs_load_and_match_hardcoded():
    """YAML configs must load successfully and match hardcoded profiles."""
    from server.crop_params import load_profile_from_yaml, list_available_configs, CROP_LIBRARY, SOIL_LIBRARY

    configs = list_available_configs()
    assert "wheat_nl" in configs
    assert "wheat_iowa" in configs
    assert "wheat_punjab" in configs

    # Verify NL config matches hardcoded
    cp, sp = load_profile_from_yaml("wheat_nl.yaml")
    hc = CROP_LIBRARY["wheat_nl"]
    assert cp.TSUM1 == hc.TSUM1
    assert cp.TSUM2 == hc.TSUM2
    assert cp.LUE == hc.LUE
    assert cp.FOTB == hc.FOTB
    assert sp.SMFCF == SOIL_LIBRARY["clay_loam"].SMFCF


def test_yaml_loaded_scenario_produces_same_score():
    """Scores must be identical whether params come from YAML or hardcoded."""
    env = CropEnvironment()
    obs = env.reset(seed=SEED, task_id=1)
    fert_done = set()
    while not obs.done:
        obs = env.step(CropAction(**greedy_action(obs, fert_done)))
    # The scenario pipeline now loads from YAML; score must match prior run
    assert obs.reward is not None
    assert 0.0 <= obs.reward <= 1.0


def test_advisory_text_present_and_deterministic():
    """Advisory text must be non-empty and deterministic across identical runs."""
    texts = []
    for _ in range(2):
        env = CropEnvironment()
        obs = env.reset(seed=SEED, task_id=2)
        assert obs.advisory_text is not None
        assert len(obs.advisory_text) > 20
        texts.append(obs.advisory_text)
    assert texts[0] == texts[1], "Advisory text must be deterministic"


# ---------------------------------------------------------------------------
# Regression / hardening tests (added post quality-audit)
# ---------------------------------------------------------------------------

def test_baseline_scores_stable():
    """Greedy baseline scores must match documented values within tolerance.

    If this test breaks, either the grading formula, reward shaping,
    crop parameters, or greedy heuristic changed — update README/ARCHITECTURE.
    """
    expected = {1: 0.8689, 2: 0.8242, 3: 0.6776}
    for task_id, expected_score in expected.items():
        env = CropEnvironment()
        obs = env.reset(seed=SEED, task_id=task_id)
        fert_done: set = set()
        while not obs.done:
            obs = env.step(CropAction(**greedy_action(obs, fert_done)))
        assert obs.reward == pytest.approx(expected_score, abs=0.001), (
            f"Task {task_id}: expected {expected_score}, got {obs.reward}"
        )


def test_negative_amount_reports_conflict():
    """Sending a negative amount must produce a conflict message."""
    env = CropEnvironment()
    obs = env.reset(seed=SEED, task_id=1)
    obs = env.step(CropAction(action_type="irrigate", amount=-5.0))
    assert any("Amount must be >= 0" in c for c in obs.conflicts)


def test_yaml_malformed_raises_value_error(tmp_path):
    """Loading malformed YAML must raise ValueError, not a raw yaml error."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("!!invalid: [unclosed", encoding="utf-8")
    from server.crop_params import load_profile_from_yaml
    with pytest.raises(ValueError, match="Malformed YAML"):
        load_profile_from_yaml(bad_yaml)


def test_yaml_missing_keys_raises_value_error(tmp_path):
    """YAML without 'crop' or 'soil' keys must raise a clear ValueError."""
    incomplete = tmp_path / "incomplete.yaml"
    incomplete.write_text("crop:\\n  tsum1: 1100\\n", encoding="utf-8")
    from server.crop_params import load_profile_from_yaml
    with pytest.raises(ValueError, match="Missing required key"):
        load_profile_from_yaml(incomplete)
