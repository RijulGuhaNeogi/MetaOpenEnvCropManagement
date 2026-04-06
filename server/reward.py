"""Reward computation for Precision Agriculture Crop Management.

Dense per-step rewards give RL agents a richer learning landscape than
final-score-only feedback.  Each action gets a small reward/penalty based
on agronomic correctness.  The final episode score (from the grader) is
returned separately as the trajectory reward.

Design principles (informed by OpenEnv bootcamp):
  - Wait actions produce a state-quality delta signal so the agent
    learns the cost of inaction when the crop is suffering.
  - Good actions: +0.10 to +0.20
  - Neutral / acceptable: 0.0
  - Bad actions: -0.03 to -0.30
  - Symmetric penalties for early AND late harvest
  - Step rewards are gated by yield trajectory so they stay aligned
    with the terminal grader (which gates efficiency by yield_score).
"""
from __future__ import annotations

from server.constants import (
    DEFAULT_N_RECOV,
    FERT_MAX_KG_PER_STEP,
    FERT_TARGET_DVS_1,
    FERT_TARGET_DVS_2,
    FERT_WINDOW_1,
    FERT_WINDOW_2,
    HARVEST_DVS_HIGH,
    HARVEST_DVS_LOW,
    MAX_WATER_CM,
    SM_TARGET_HIGH,
    SM_TARGET_LOW,
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _ideal_n_dose(n_availability: float, n_recov: float) -> float:
    """Compute the N kg/ha that would fill n_factor to 1.0, capped at 50 kg."""
    deficit = max(0.0, 1.0 - n_availability)
    return min(FERT_MAX_KG_PER_STEP, deficit / max(n_recov, 0.001))


def _fertilizer_window_target(
    dvs: float, n_availability: float, n_recov: float,
) -> tuple[float | None, float | None, float]:
    if FERT_WINDOW_1[0] <= dvs <= FERT_WINDOW_1[1]:
        return _ideal_n_dose(n_availability, n_recov), FERT_TARGET_DVS_1, 0.14
    if FERT_WINDOW_2[0] <= dvs <= FERT_WINDOW_2[1]:
        return _ideal_n_dose(n_availability, n_recov), FERT_TARGET_DVS_2, 0.12
    return None, None, 0.0


def compute_step_reward(
    action_type: str,
    dvs: float,
    sm: float,
    amount: float,
    cost: float,
    budget_remaining: float,
    total_n: float = 0.0,
    total_water: float = 0.0,
    forecast_rain: float = 0.0,
    root_zone_depth_cm: float = 90.0,
    target_sm_low: float = SM_TARGET_LOW,
    target_sm_high: float = SM_TARGET_HIGH,
    water_stress: float = 1.0,
    n_availability: float = 1.0,
    n_recov: float = DEFAULT_N_RECOV,
) -> float:
    """Dense per-step reward based on agronomic correctness.

    Returns a reward in roughly [-0.3, +0.2] range.
    """
    if action_type == "wait":
        # Penalise inaction when the crop is suffering — the agent should
        # feel the cost of doing nothing while soil dries or N depletes.
        # Small magnitude so it never dominates an actual action reward.
        stress_penalty = max(0.0, 1.0 - water_stress) * -0.04   # up to -0.04 at full stress
        n_penalty = max(0.0, 0.5 - n_availability) * -0.03      # up to -0.015 at n=0.0
        # Penalise waiting inside a fert window when crop still needs N
        fert_window_penalty = 0.0
        if n_availability < 0.7 and (
            (FERT_WINDOW_1[0] <= dvs <= FERT_WINDOW_1[1])
            or (FERT_WINDOW_2[0] <= dvs <= FERT_WINDOW_2[1])
        ):
            fert_window_penalty = -0.015
        return _clamp(stress_penalty + n_penalty + fert_window_penalty, -0.06, 0.0)

    elif action_type in ("inspect_soil", "inspect_crop"):
        # Information-gathering: context-sensitive reward to encourage
        # strategic inspection on hidden tiers.
        # Higher reward for inspect_crop at ripening (harvest-timing critical)
        # and inspect_soil during fert window (dose calibration).
        if action_type == "inspect_crop" and dvs >= 1.5:
            return _clamp(0.03, -0.02, 0.04)  # Ripening — high-value
        if action_type == "inspect_soil" and (
            (FERT_WINDOW_1[0] <= dvs <= FERT_WINDOW_1[1])
            or (FERT_WINDOW_2[0] <= dvs <= FERT_WINDOW_2[1])
        ):
            return _clamp(0.02, -0.02, 0.03)  # In fert window — moderate-value
        return _clamp(0.01, -0.02, 0.02)

    elif action_type == "irrigate":
        target_sm = (target_sm_low + target_sm_high) / 2.0
        desired_amount = max(0.0, (target_sm - sm) * root_zone_depth_cm)
        water_pressure = min(0.04, max(0.0, total_water) / MAX_WATER_CM * 0.04)

        if sm >= target_sm_high:
            return _clamp(-0.04 - 0.01 * amount, -0.14, -0.02)

        if desired_amount <= 0.25:
            return _clamp(-0.01 - 0.006 * amount, -0.08, 0.0)

        dose_ratio = amount / max(desired_amount, 0.5)
        fit_score = max(0.0, 1.0 - min(abs(dose_ratio - 1.0), 1.5) / 1.5)
        dryness = max(0.0, target_sm - sm)
        dryness_score = min(1.0, dryness / 0.12)
        forecast_penalty = min(0.05, forecast_rain * 0.012)
        overshoot = max(0.0, amount - desired_amount * 1.2)
        overshoot_penalty = min(0.08, overshoot * 0.015)
        reward = 0.02 + 0.11 * fit_score * dryness_score
        reward -= forecast_penalty + overshoot_penalty + water_pressure
        return _clamp(reward, -0.12, 0.14)

    elif action_type == "fertilize":
        target_amount, target_dvs, window_bonus = _fertilizer_window_target(
            dvs, n_availability, n_recov,
        )
        projected_total_n = total_n + amount

        if target_amount is not None:
            timing_score = max(0.0, 1.0 - abs(dvs - (target_dvs or dvs)) / 0.10)
            timing_multiplier = 0.55 + 0.45 * timing_score
            dose_ratio = amount / max(target_amount, 1.0)
            fit_score = max(0.0, 1.0 - min(abs(dose_ratio - 1.0), 2.0) / 2.0)
            season_excess = max(0.0, projected_total_n - 70.0)
            excess_penalty = min(0.12, season_excess / 40.0 * 0.12)
            return _clamp(
                0.03 + window_bonus * fit_score * timing_multiplier - excess_penalty,
                -0.10,
                0.16,
            )
        elif dvs < 0.20:
            return -0.01  # Slightly wasteful: too early to matter much
        elif dvs > 1.5:
            return _clamp(-0.08 - 0.001 * amount, -0.14, -0.06)
        elif dvs > 1.0:
            return _clamp(-0.04 - 0.0008 * amount, -0.09, -0.03)
        # Between windows (0.40-0.50, 0.70-1.0) — not ideal
        return _clamp(-0.02 - 0.0006 * amount, -0.06, -0.01)

    elif action_type == "harvest":
        # NOTE: This branch is for standalone callers. The environment's
        # terminal harvest path uses compute_trajectory_reward(grade) directly.
        if HARVEST_DVS_LOW <= dvs <= HARVEST_DVS_HIGH:
            return 0.20   # Optimal harvest window
        elif 1.5 <= dvs < HARVEST_DVS_LOW:
            return -0.15  # Early, yield loss
        elif dvs < 1.5:
            return -0.30  # Way too early
        # DVS > HARVEST_DVS_HIGH — late harvest, grain shattering risk
        return max(-0.25, -0.20 * (dvs - HARVEST_DVS_HIGH) - 0.05)

    return 0.0


def compute_delta_reward(
    action_type: str,
    pre_sm: float,
    post_sm: float,
    pre_water_stress: float,
    post_water_stress: float,
    pre_n_availability: float,
    post_n_availability: float,
    cost: float,
    budget_remaining: float,
    total_cost: float = 0.0,
    budget: float = 0.0,
    pre_twso: float = 0.0,
    post_twso: float = 0.0,
    target_yield: float = 1.0,
) -> float:
    """Reward the consequence of an action after the transition.

    Positive reward reflects relief of stress; penalties capture waste and
    expensive low-impact actions.  A yield-progress component ensures step
    rewards stay aligned with the terminal grader's yield_score metric.

    Returns a value in roughly [-0.15, +0.15].
    """
    # Yield progress: mirrors grader's yield_score = actual/target.
    yield_delta = (post_twso - pre_twso) / max(target_yield, 1.0)
    yield_signal = _clamp(yield_delta * 0.5, -0.02, 0.04)

    if action_type in ("wait", "harvest"):
        if action_type == "harvest":
            return 0.0
        # Wait: state deterioration/recovery + yield progress
        stress_delta = post_water_stress - pre_water_stress
        n_delta = post_n_availability - pre_n_availability
        return _clamp(0.3 * stress_delta + 0.2 * n_delta + yield_signal, -0.08, 0.04)

    # Crop vigor: mirrors grader's max(yield_score, 0.1) gating of efficiency.
    # Early season (twso=0): vigor=1.0 (don't suppress). Late season with
    # poor yield: vigor drops, reducing efficiency reward / amplifying penalty.
    season_progress = min(1.0, post_twso / max(target_yield * 0.1, 1.0))
    crop_vigor = max(0.3, min(1.0, season_progress)) if post_twso > 0 else 1.0

    spend_ratio = cost / max(budget_remaining, 1.0)
    cost_penalty = min(0.03, spend_ratio * 0.03) * crop_vigor
    spend_pressure = 0.0
    if budget > 0.0:
        spend_pressure = min(0.06, max(0.0, total_cost) / budget * 0.06) * crop_vigor

    if action_type == "irrigate":
        stress_gain = post_water_stress - pre_water_stress
        overshoot_penalty = max(0.0, post_sm - 0.40) * 0.6
        no_effect_penalty = 0.02 if post_sm <= pre_sm + 0.002 else 0.0
        reward = (
            0.7 * stress_gain
            - overshoot_penalty
            - no_effect_penalty
            - cost_penalty
            - spend_pressure
            + yield_signal
        )
        return _clamp(reward, -0.15, 0.15)

    if action_type == "fertilize":
        n_gain = post_n_availability - pre_n_availability
        inefficiency_penalty = 0.02 if n_gain < 0.01 else 0.0
        reward = 0.6 * n_gain - inefficiency_penalty - cost_penalty - spend_pressure + yield_signal
        return _clamp(reward, -0.15, 0.15)

    return 0.0


def compute_trajectory_reward(final_grade: float) -> float:
    """Trajectory reward = final grader score, clamped [0, 1]."""
    return max(0.0, min(1.0, final_grade))
