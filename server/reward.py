"""Reward computation for Precision Agriculture Crop Management.

Dense per-step rewards give RL agents a richer learning landscape than
final-score-only feedback.  Each action gets a small reward/penalty based
on agronomic correctness.  The final episode score (from the grader) is
returned separately as the trajectory reward.

Design principles (informed by OpenEnv bootcamp):
  - Rewards must not incentivize doing nothing (no free +reward for wait)
  - Good actions: +0.10 to +0.20
  - Neutral / acceptable: 0.0
  - Bad actions: -0.03 to -0.30
  - Symmetric penalties for early AND late harvest
"""
from __future__ import annotations


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _fertilizer_window_target(dvs: float) -> tuple[float | None, float]:
    if 0.20 <= dvs <= 0.40:
        return 18.0, 0.14
    if 0.50 <= dvs <= 0.70:
        return 15.0, 0.12
    return None, 0.0


def compute_step_reward(
    action_type: str,
    dvs: float,
    sm: float,
    amount: float,
    cost: float,
    budget_remaining: float,
    total_n: float = 0.0,
    forecast_rain: float = 0.0,
    root_zone_depth_cm: float = 90.0,
    target_sm_low: float = 0.28,
    target_sm_high: float = 0.32,
) -> float:
    """Dense per-step reward based on agronomic correctness.

    Returns a reward in roughly [-0.3, +0.2] range.
    """
    if action_type == "wait":
        # Neutral — never reward doing nothing, to prevent lazy-wait policies
        return 0.0

    elif action_type == "irrigate":
        target_sm = (target_sm_low + target_sm_high) / 2.0
        desired_amount = max(0.0, (target_sm - sm) * root_zone_depth_cm)

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
        reward -= forecast_penalty + overshoot_penalty
        return _clamp(reward, -0.12, 0.14)

    elif action_type == "fertilize":
        target_amount, window_bonus = _fertilizer_window_target(dvs)
        projected_total_n = total_n + amount

        if target_amount is not None:
            dose_ratio = amount / max(target_amount, 1.0)
            fit_score = max(0.0, 1.0 - min(abs(dose_ratio - 1.0), 2.0) / 2.0)
            season_excess = max(0.0, projected_total_n - 45.0)
            excess_penalty = min(0.12, season_excess / 30.0 * 0.12)
            return _clamp(0.03 + window_bonus * fit_score - excess_penalty, -0.10, 0.16)
        elif dvs < 0.20:
            return -0.01  # Slightly wasteful: too early to matter much
        elif dvs > 1.5:
            return _clamp(-0.08 - 0.001 * amount, -0.14, -0.06)
        elif dvs > 1.0:
            return _clamp(-0.04 - 0.0008 * amount, -0.09, -0.03)
        # Between windows (0.40-0.50, 0.70-1.0) — not ideal
        return _clamp(-0.02 - 0.0006 * amount, -0.06, -0.01)

    elif action_type == "harvest":
        if 1.8 <= dvs <= 2.05:
            return 0.20   # Optimal harvest window
        elif 1.5 <= dvs < 1.8:
            return -0.15  # Early, yield loss
        elif dvs < 1.5:
            return -0.30  # Way too early
        # DVS > 2.05 — late harvest, grain shattering risk
        return max(-0.15, -0.05 * (dvs - 2.05) - 0.05)

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
) -> float:
    """Reward the consequence of an action after the transition.

    Positive reward reflects relief of stress; penalties capture waste and
    expensive low-impact actions. Returns a value in roughly [-0.15, +0.15].
    """
    if action_type in ("wait", "harvest"):
        return 0.0

    spend_ratio = cost / max(budget_remaining, 1.0)
    cost_penalty = min(0.03, spend_ratio * 0.03)

    if action_type == "irrigate":
        stress_gain = post_water_stress - pre_water_stress
        overshoot_penalty = max(0.0, post_sm - 0.40) * 0.6
        no_effect_penalty = 0.02 if post_sm <= pre_sm + 0.002 else 0.0
        reward = 0.7 * stress_gain - overshoot_penalty - no_effect_penalty - cost_penalty
        return _clamp(reward, -0.15, 0.15)

    if action_type == "fertilize":
        n_gain = post_n_availability - pre_n_availability
        inefficiency_penalty = 0.02 if n_gain < 0.01 else 0.0
        reward = 0.6 * n_gain - inefficiency_penalty - cost_penalty
        return _clamp(reward, -0.15, 0.15)

    return 0.0


def compute_trajectory_reward(final_grade: float) -> float:
    """Trajectory reward = final grader score, clamped [0, 1]."""
    return max(0.0, min(1.0, final_grade))
