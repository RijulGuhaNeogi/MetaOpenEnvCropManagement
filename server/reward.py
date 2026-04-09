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
    LEACH_RAIN_THRESHOLD,
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
    fert_events_count: int = 0,
    task_tier: int = 1,
) -> tuple[float, dict[str, float]]:
    """Dense per-step reward based on agronomic correctness.

    Returns (scalar_reward, breakdown_dict).
    Intensity hierarchy (by importance of the decision):
      Harvest timing:   ±0.15 – ±0.30  (most consequential)
      Fertilization:    ±0.05 – ±0.16  (yield depends on it)
      Irrigation:       ±0.04 – ±0.14  (moderate impact)
      Correct patience: +0.01 – +0.03  (small but consistent signal)
    """
    if action_type == "wait":
        # --- Context: where are we in the season? ---
        in_w1 = FERT_WINDOW_1[0] <= dvs <= FERT_WINDOW_1[1]
        in_w2 = FERT_WINDOW_2[0] <= dvs <= FERT_WINDOW_2[1]
        past_target = (in_w1 and dvs >= FERT_TARGET_DVS_1) or (in_w2 and dvs >= FERT_TARGET_DVS_2)
        in_harvest_window = dvs >= HARVEST_DVS_LOW

        # --- Patience bonus: reward correct inaction by growth phase ---
        patience_bonus = 0.0
        should_be_fertilizing = past_target and n_availability < 0.7 and fert_events_count < 2
        if not in_harvest_window and not should_be_fertilizing:
            if dvs < 0.15:
                patience_bonus = 0.01      # Pre-emergence: trivially correct
            elif (in_w1 and dvs < FERT_TARGET_DVS_1) or (in_w2 and dvs < FERT_TARGET_DVS_2):
                patience_bonus = 0.02      # Early in fert window: waiting for optimal DVS
            elif 1.0 <= dvs < 1.5:
                patience_bonus = 0.02      # Grain fill: crop growing, patience pays off
            elif not in_w1 and not in_w2:
                patience_bonus = 0.03      # Between windows: no action needed

        # --- Penalties: cost of inaction when crop is suffering ---
        stress_penalty = max(0.0, 1.0 - water_stress) * -0.10   # up to -0.09 at full stress
        n_penalty = max(0.0, 0.5 - n_availability) * -0.08      # up to -0.04 at n=0.0

        # Missed fert window — strong, gated by available fert slots
        fert_window_penalty = 0.0
        if past_target and fert_events_count < 2:
            if n_availability < 0.5:
                fert_window_penalty = -0.08    # Severely N-starved: big miss
            elif n_availability < 0.7:
                fert_window_penalty = -0.05    # Moderate deficiency

        # Missed irrigation opportunity
        irrigation_penalty = 0.0
        if sm < target_sm_low and forecast_rain < 0.3:
            irrigation_penalty = -0.04

        # Harvest urgency — STRONGEST penalties (most important timing)
        harvest_urgency = 0.0
        if HARVEST_DVS_LOW <= dvs <= HARVEST_DVS_HIGH:
            progress = (dvs - HARVEST_DVS_LOW) / max(HARVEST_DVS_HIGH - HARVEST_DVS_LOW, 0.01)
            harvest_urgency = -0.08 - 0.07 * progress  # -0.08 at 1.80, -0.15 at 2.00
        elif dvs >= HARVEST_DVS_HIGH:
            harvest_urgency = -0.15            # Post-maturity: grain shattering

        total = patience_bonus + stress_penalty + n_penalty + fert_window_penalty + irrigation_penalty + harvest_urgency
        reward = _clamp(total, -0.15, 0.04)
        breakdown = {
            "patience_bonus": round(patience_bonus, 4),
            "stress_penalty": round(stress_penalty, 4),
            "n_penalty": round(n_penalty, 4),
            "fert_window_penalty": round(fert_window_penalty, 4),
            "irrigation_penalty": round(irrigation_penalty, 4),
            "harvest_urgency": round(harvest_urgency, 4),
        }
        return reward, breakdown

    elif action_type in ("inspect_soil", "inspect_crop"):
        _empty = {"base_reward": 0.0, "tier_bonus": 0.0, "budget_factor": 1.0}
        if task_tier <= 1:
            return 0.0, _empty

        if action_type == "inspect_crop" and dvs >= 1.5:
            base_reward = 0.03
        elif action_type == "inspect_soil" and (
            (FERT_WINDOW_1[0] <= dvs <= FERT_WINDOW_1[1])
            or (FERT_WINDOW_2[0] <= dvs <= FERT_WINDOW_2[1])
        ):
            base_reward = 0.02
        else:
            base_reward = 0.01

        tier_bonus = 0.0
        if task_tier >= 3:
            tier_bonus = base_reward * 0.5
            base_reward *= 1.5

        budget_factor = min(1.0, budget_remaining / max(cost * 10, 1.0))
        reward = _clamp(base_reward * budget_factor, -0.02, 0.06)
        return reward, {
            "base_reward": round(base_reward, 4),
            "tier_bonus": round(tier_bonus, 4),
            "budget_factor": round(budget_factor, 4),
        }

    elif action_type == "irrigate":
        target_sm = (target_sm_low + target_sm_high) / 2.0
        desired_amount = max(0.0, (target_sm - sm) * root_zone_depth_cm)
        water_pressure = min(0.04, max(0.0, total_water) / MAX_WATER_CM * 0.04)

        if sm >= target_sm_high:
            reward = _clamp(-0.04 - 0.01 * amount, -0.14, -0.02)
            return reward, {"dose_fitness": 0.0, "dryness_score": 0.0,
                           "forecast_penalty": 0.0, "overshoot_penalty": round(0.01 * amount, 4),
                           "water_pressure": round(water_pressure, 4)}

        if desired_amount <= 0.25:
            reward = _clamp(-0.01 - 0.006 * amount, -0.08, 0.0)
            return reward, {"dose_fitness": 0.0, "dryness_score": 0.0,
                           "forecast_penalty": 0.0, "overshoot_penalty": round(0.006 * amount, 4),
                           "water_pressure": round(water_pressure, 4)}

        dose_ratio = amount / max(desired_amount, 0.5)
        fit_score = max(0.0, 1.0 - min(abs(dose_ratio - 1.0), 1.5) / 1.5)
        dryness = max(0.0, target_sm - sm)
        dryness_score = min(1.0, dryness / 0.12)
        forecast_penalty = min(0.05, forecast_rain * 0.012)
        overshoot = max(0.0, amount - desired_amount * 1.2)
        overshoot_penalty = min(0.08, overshoot * 0.015)
        reward = 0.02 + 0.11 * fit_score * dryness_score
        reward -= forecast_penalty + overshoot_penalty + water_pressure
        reward = _clamp(reward, -0.12, 0.14)
        return reward, {
            "dose_fitness": round(fit_score, 4),
            "dryness_score": round(dryness_score, 4),
            "forecast_penalty": round(forecast_penalty, 4),
            "overshoot_penalty": round(overshoot_penalty, 4),
            "water_pressure": round(water_pressure, 4),
        }

    elif action_type in ("fertilize", "fertilize_slow"):
        # Hard penalty for exceeding the 2-application cap
        if fert_events_count > 2:
            reward = _clamp(-0.06 - 0.001 * amount, -0.14, -0.04)
            return reward, {"timing_score": 0.0, "dose_fitness": 0.0,
                           "window_bonus": 0.0, "excess_penalty": 0.0,
                           "weather_awareness": 0.0, "cap_exceeded": 1.0}

        target_amount, target_dvs, window_bonus = _fertilizer_window_target(
            dvs, n_availability, n_recov,
        )
        projected_total_n = total_n + amount

        if target_amount is not None:
            timing_score = max(0.0, 1.0 - abs(dvs - (target_dvs or dvs)) / 0.10)
            timing_multiplier = 0.30 + 0.70 * timing_score
            dose_ratio = amount / max(target_amount, 1.0)
            fit_score = max(0.0, 1.0 - min(abs(dose_ratio - 1.0), 1.0))
            season_excess = max(0.0, projected_total_n - 55.0)
            excess_penalty = min(0.14, season_excess / 30.0 * 0.14)
            reward = window_bonus * fit_score * timing_multiplier - excess_penalty

            # Weather-awareness bonus/penalty for fert type choice
            weather_awareness = 0.0
            is_slow = (action_type == "fertilize_slow")
            if forecast_rain > LEACH_RAIN_THRESHOLD:
                weather_awareness = 0.02 if is_slow else -0.03
            elif forecast_rain < 0.2:
                weather_awareness = 0.01 if not is_slow else -0.02
            reward += weather_awareness

            reward = _clamp(reward, -0.10, 0.16)
            return reward, {
                "timing_score": round(timing_score, 4),
                "dose_fitness": round(fit_score, 4),
                "window_bonus": round(window_bonus, 4),
                "excess_penalty": round(excess_penalty, 4),
                "weather_awareness": round(weather_awareness, 4),
                "cap_exceeded": 0.0,
            }
        elif dvs < 0.20:
            reward = -0.01
        elif dvs > 1.5:
            reward = _clamp(-0.08 - 0.001 * amount, -0.14, -0.06)
        elif dvs > 1.0:
            reward = _clamp(-0.04 - 0.0008 * amount, -0.09, -0.03)
        else:
            # Between windows (0.40-0.50, 0.70-1.0) — not ideal
            reward = _clamp(-0.02 - 0.0006 * amount, -0.06, -0.01)
        return reward, {"timing_score": 0.0, "dose_fitness": 0.0,
                       "window_bonus": 0.0, "excess_penalty": 0.0,
                       "weather_awareness": 0.0, "cap_exceeded": 0.0}

    elif action_type == "harvest":
        # NOTE: This branch is for standalone callers. The environment's
        # terminal harvest path uses compute_trajectory_reward(grade) directly.
        if HARVEST_DVS_LOW <= dvs <= HARVEST_DVS_HIGH:
            # Big win — most important timing decision. Peak at DVS ~1.90.
            sweet_spot = (HARVEST_DVS_LOW + HARVEST_DVS_HIGH) / 2.0
            proximity = max(0.0, 1.0 - abs(dvs - sweet_spot) / 0.10)
            reward = 0.20 + 0.05 * proximity   # 0.20–0.25
        elif 1.5 <= dvs < HARVEST_DVS_LOW:
            reward = -0.15  # Early, yield loss
        elif dvs < 1.5:
            reward = -0.30  # Way too early — catastrophic
        else:
            # DVS > HARVEST_DVS_HIGH — late harvest, grain shattering risk
            reward = max(-0.25, -0.20 * (dvs - HARVEST_DVS_HIGH) - 0.05)
        return reward, {"dvs_proximity": round(dvs, 4)}

    return 0.0, {}


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
) -> tuple[float, dict[str, float]]:
    """Reward the consequence of an action after the transition.

    Positive reward reflects relief of stress; penalties capture waste and
    expensive low-impact actions.  A yield-progress component ensures step
    rewards stay aligned with the terminal grader's yield_score metric.

    Returns (scalar_reward, breakdown_dict).
    """
    # Yield progress: mirrors grader's yield_score = actual/target.
    # Coefficient boosted (2.0) so grain-fill growth produces visible signal.
    yield_delta = (post_twso - pre_twso) / max(target_yield, 1.0)
    yield_signal = _clamp(yield_delta * 2.0, -0.02, 0.06)

    if action_type in ("wait", "harvest"):
        if action_type == "harvest":
            return 0.0, {"yield_progress": 0.0, "stress_delta": 0.0, "n_delta": 0.0}
        # Wait: state deterioration/recovery + yield progress.
        stress_delta = post_water_stress - pre_water_stress
        n_delta = post_n_availability - pre_n_availability
        reward = _clamp(0.15 * stress_delta + 0.2 * n_delta + yield_signal, -0.10, 0.06)
        return reward, {
            "yield_progress": round(yield_signal, 4),
            "stress_delta": round(0.15 * stress_delta, 4),
            "n_delta": round(0.2 * n_delta, 4),
        }

    # Crop vigor: mirrors grader's max(yield_score, 0.1) gating of efficiency.
    season_progress = min(1.0, post_twso / max(target_yield * 0.1, 1.0))
    crop_vigor = max(0.3, min(1.0, season_progress)) if post_twso > 0 else 1.0

    spend_ratio = cost / max(budget_remaining, 1.0)
    cost_penalty = min(0.05, spend_ratio * 0.06) * crop_vigor
    spend_pressure = 0.0
    if budget > 0.0:
        spend_pressure = min(0.08, max(0.0, total_cost) / budget * 0.08) * crop_vigor

    if action_type == "irrigate":
        stress_gain = post_water_stress - pre_water_stress
        overshoot_penalty = max(0.0, post_sm - 0.40) * 0.6
        no_effect_penalty = 0.02 if post_sm <= pre_sm + 0.002 else 0.0
        effective_yield = yield_signal if stress_gain > 0 else 0.0
        reward = (
            0.7 * stress_gain
            - overshoot_penalty
            - no_effect_penalty
            - cost_penalty
            - spend_pressure
            + effective_yield
        )
        reward = _clamp(reward, -0.15, 0.15)
        return reward, {
            "stress_relief": round(0.7 * stress_gain, 4),
            "overshoot_penalty": round(overshoot_penalty, 4),
            "cost_pressure": round(cost_penalty + spend_pressure, 4),
            "yield_progress": round(effective_yield, 4),
        }

    if action_type == "fertilize" or action_type == "fertilize_slow":
        n_gain = post_n_availability - pre_n_availability
        inefficiency_penalty = 0.02 if n_gain < 0.01 else 0.0
        effective_yield = yield_signal if n_gain > 0 else 0.0
        reward = 0.6 * n_gain - inefficiency_penalty - cost_penalty - spend_pressure + effective_yield
        reward = _clamp(reward, -0.15, 0.15)
        return reward, {
            "n_recovery": round(0.6 * n_gain, 4),
            "inefficiency_penalty": round(inefficiency_penalty, 4),
            "cost_pressure": round(cost_penalty + spend_pressure, 4),
            "yield_progress": round(effective_yield, 4),
        }

    return 0.0, {}


def compute_trajectory_reward(final_grade: float) -> float:
    """Trajectory reward = final grader score, clamped [0, 1]."""
    return max(0.0, min(1.0, final_grade))
