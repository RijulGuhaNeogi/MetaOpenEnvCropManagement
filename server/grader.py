"""Deterministic grading for Precision Agriculture Crop Management.

Multi-metric scoring normalized to [0.0, 1.0]:
  - yield_score: actual yield / target (potential) yield
  - water_efficiency: less water used = better
  - cost_efficiency: under-budget = better
  - timing_quality: fertilize at correct growth stages
  - harvest_timing: harvest at maturity window

A single unified formula is used for ALL tasks.  Difficulty comes entirely
from the environment conditions (climate, budget, soil) not from different
scoring weights.  Easy environments naturally score higher because their
conditions make all metrics achievable; hard environments force trade-offs
that lower scores organically.
"""
from __future__ import annotations

from typing import Any


def grade_episode(
    actual_yield: float,
    target_yield: float,
    total_water: float,
    total_n: float,
    total_cost: float,
    budget: float,
    harvest_dvs: float,
    harvested: bool,
    actions_taken: list[dict[str, Any]],
    task_id: int,
) -> tuple[float, dict[str, Any]]:
    """Compute final grade. Returns (score, breakdown_dict).

    All metrics are deterministic — same inputs always produce the same score.
    """
    # ---------------------------------------------------------------
    # Metric 1: Yield score (0-1)
    # ---------------------------------------------------------------
    if target_yield > 0 and harvested:
        yield_score = min(1.0, actual_yield / target_yield)
    else:
        # No harvest or zero target — severe penalty
        yield_score = 0.0

    # ---------------------------------------------------------------
    # Metric 2: Water efficiency (0-1)
    # Less water = better. Max reasonable water ~50cm for wheat season.
    # ---------------------------------------------------------------
    max_water = 50.0  # cm — wasteful upper bound
    if total_water <= 0:
        water_efficiency = 1.0  # No irrigation = best water efficiency
    else:
        water_efficiency = max(0.0, 1.0 - total_water / max_water)

    # ---------------------------------------------------------------
    # Metric 3: Cost efficiency (0-1)
    # Under-budget = good.
    # ---------------------------------------------------------------
    if budget > 0:
        cost_efficiency = max(0.0, 1.0 - total_cost / budget)
    else:
        cost_efficiency = 1.0

    # ---------------------------------------------------------------
    # Metric 4: Timing quality (0-1)
    # Reward fertilizing near DVS 0.3 and DVS 0.6 (key growth stages)
    # ---------------------------------------------------------------
    fert_actions = [
        a for a in actions_taken if a.get("action_type") == "fertilize"
    ]
    if fert_actions:
        timing_scores = []
        for fa in fert_actions:
            dvs_at = fa.get("dvs", 0.0)
            # Distance to nearest target stage (0.3 or 0.6)
            dist_to_03 = abs(dvs_at - 0.3)
            dist_to_06 = abs(dvs_at - 0.6)
            best_dist = min(dist_to_03, dist_to_06)
            # Perfect at target = 1.0, degrades with distance
            score = max(0.0, 1.0 - best_dist / 0.5)
            timing_scores.append(score)
        timing_quality = sum(timing_scores) / len(timing_scores)
    else:
        # No fertilization — some penalty (missed opportunity)
        timing_quality = 0.2

    # ---------------------------------------------------------------
    # Metric 5: Harvest timing (0-1)
    # Optimal: DVS in [1.8, 2.05]
    # ---------------------------------------------------------------
    if not harvested:
        harvest_timing = 0.0
    elif 1.8 <= harvest_dvs <= 2.05:
        harvest_timing = 1.0
    elif 1.6 <= harvest_dvs < 1.8:
        # Slightly early
        harvest_timing = 0.5 + 0.5 * (harvest_dvs - 1.6) / 0.2
    elif harvest_dvs < 1.6:
        # Too early — linear penalty
        harvest_timing = max(0.0, harvest_dvs / 1.6 * 0.5)
    else:
        # DVS > 2.05 — slight penalty for late harvest
        harvest_timing = max(0.5, 1.0 - (harvest_dvs - 2.05) * 2.0)

    # ---------------------------------------------------------------
    # Unified scoring formula (same for ALL tasks)
    #
    # Difficulty comes from the environment, not from different weights.
    # Easy environments (Netherlands, $800) make all metrics achievable;
    # hard environments (Punjab, $300) force painful trade-offs that
    # organically lower the score.
    # ---------------------------------------------------------------
    raw = (
        0.35 * yield_score
        + 0.20 * water_efficiency
        + 0.18 * cost_efficiency
        + 0.15 * timing_quality
        + 0.12 * harvest_timing
    )

    final = max(0.0, min(1.0, round(raw, 4)))

    breakdown = {
        "yield_score": round(yield_score, 4),
        "water_efficiency": round(water_efficiency, 4),
        "cost_efficiency": round(cost_efficiency, 4),
        "timing_quality": round(timing_quality, 4),
        "harvest_timing": round(harvest_timing, 4),
        "actual_yield": round(actual_yield, 1),
        "target_yield": round(target_yield, 1),
        "total_water": round(total_water, 2),
        "total_n": round(total_n, 2),
        "total_cost": round(total_cost, 2),
        "budget": round(budget, 2),
        "harvest_dvs": round(harvest_dvs, 3),
        "harvested": harvested,
        "task_id": task_id,
    }

    return final, breakdown
