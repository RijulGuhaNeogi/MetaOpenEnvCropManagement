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

from server.constants import (
    FERT_TARGET_DVS_1,
    FERT_TARGET_DVS_2,
    HARVEST_DVS_HIGH,
    HARVEST_DVS_LOW,
    MAX_WATER_CM,
    WEIGHT_COST,
    WEIGHT_HARVEST,
    WEIGHT_TIMING,
    WEIGHT_WATER,
    WEIGHT_YIELD,
)


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
    explicit_harvest: bool = True,
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
    # Gated by yield_score to prevent zero-spend + poor yield from
    # earning perfect efficiency.
    # ---------------------------------------------------------------
    max_water = MAX_WATER_CM  # cm — wasteful upper bound
    if total_water <= 0:
        raw_water_efficiency = 1.0
    else:
        raw_water_efficiency = max(0.0, 1.0 - total_water / max_water)
    water_efficiency = raw_water_efficiency * max(yield_score, 0.1)

    # ---------------------------------------------------------------
    # Metric 3: Cost efficiency (0-1)
    # Under-budget = good.  Gated by yield_score (same rationale).
    # ---------------------------------------------------------------
    if budget > 0:
        raw_cost_efficiency = max(0.0, 1.0 - total_cost / budget)
    else:
        raw_cost_efficiency = 1.0
    cost_efficiency = raw_cost_efficiency * max(yield_score, 0.1)

    # ---------------------------------------------------------------
    # Metric 4: Timing quality (0-1)
    # Reward fertilizing near DVS 0.3 and DVS 0.6 (key growth stages)
    # ---------------------------------------------------------------
    fert_actions = [
        a for a in actions_taken if a.get("action_type") in ("fertilize", "fertilize_slow")
    ]
    if fert_actions:
        timing_scores = []
        for fa in fert_actions:
            dvs_at = fa.get("dvs", 0.0)
            # Distance to nearest target stage (0.3 or 0.6)
            dist_to_03 = abs(dvs_at - FERT_TARGET_DVS_1)
            dist_to_06 = abs(dvs_at - FERT_TARGET_DVS_2)
            best_dist = min(dist_to_03, dist_to_06)
            # Perfect at target = 1.0, degrades with distance
            score = max(0.0, 1.0 - best_dist / 0.5)
            timing_scores.append(score)
        timing_quality = sum(timing_scores) / len(timing_scores)
        # Penalize excess fertilization events (optimal is 2)
        excess_ferts = max(0, len(fert_actions) - 2)
        timing_quality *= max(0.4, 1.0 - 0.15 * excess_ferts)
    else:
        # No fertilization — zero credit (missed opportunity)
        timing_quality = 0.0

    # ---------------------------------------------------------------
    # Metric 5: Harvest timing (0-1)
    # Optimal: DVS in [1.8, 2.05]
    # Passive auto-harvest (DVS>=2.0, timeout, max-steps) gets minimal
    # credit — only an explicit agent harvest action earns full marks.
    # ---------------------------------------------------------------
    if not harvested:
        harvest_timing = 0.0
    elif not explicit_harvest:
        # Auto-terminated: crop matured or season ended without agent harvesting
        harvest_timing = 0.2
    elif HARVEST_DVS_LOW <= harvest_dvs <= HARVEST_DVS_HIGH:
        harvest_timing = 1.0
    elif 1.6 <= harvest_dvs < HARVEST_DVS_LOW:
        # Slightly early
        harvest_timing = 0.5 + 0.5 * (harvest_dvs - 1.6) / 0.2
    elif harvest_dvs < 1.6:
        # Too early — linear penalty
        harvest_timing = max(0.0, harvest_dvs / 1.6 * 0.5)
    else:
        # DVS > HARVEST_DVS_HIGH — slight penalty for late harvest
        harvest_timing = max(0.5, 1.0 - (harvest_dvs - HARVEST_DVS_HIGH) * 2.0)

    # ---------------------------------------------------------------
    # Unified scoring formula (same for ALL tasks)
    #
    # Difficulty comes from the environment, not from different weights.
    # Easy environments (Netherlands, $800) make all metrics achievable;
    # hard environments (Punjab, $300) force painful trade-offs that
    # organically lower the score.
    # ---------------------------------------------------------------
    raw = (
        WEIGHT_YIELD * yield_score
        + WEIGHT_WATER * water_efficiency
        + WEIGHT_COST * cost_efficiency
        + WEIGHT_TIMING * timing_quality
        + WEIGHT_HARVEST * harvest_timing
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
        "explicit_harvest": explicit_harvest,
        "task_id": task_id,
    }

    return final, breakdown
