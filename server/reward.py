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


def compute_step_reward(
    action_type: str,
    dvs: float,
    sm: float,
    cost: float,
    budget_remaining: float,
) -> float:
    """Dense per-step reward based on agronomic correctness.

    Returns a reward in roughly [-0.3, +0.2] range.
    """
    if action_type == "wait":
        # Neutral — never reward doing nothing, to prevent lazy-wait policies
        return 0.0

    elif action_type == "irrigate":
        if sm < 0.22:
            return 0.10   # Correct: irrigating dry soil
        elif sm < 0.30:
            return 0.03   # Acceptable
        elif sm > 0.40:
            return -0.05  # Wasteful: soil already wet
        return 0.0

    elif action_type == "fertilize":
        # Key growth stages: DVS ~0.25-0.35 (tillering) and ~0.55-0.65 (stem elongation)
        if 0.20 <= dvs <= 0.40 or 0.50 <= dvs <= 0.70:
            return 0.15  # Excellent timing
        elif dvs < 0.20:
            return 0.0   # Too early, neutral
        elif dvs > 1.5:
            return -0.10  # Wasteful: post grain-fill
        elif dvs > 1.0:
            return -0.05  # Late, diminishing returns
        # Between windows (0.40-0.50, 0.70-1.0) — not ideal
        return -0.03

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


def compute_trajectory_reward(final_grade: float) -> float:
    """Trajectory reward = final grader score, clamped [0, 1]."""
    return max(0.0, min(1.0, final_grade))
