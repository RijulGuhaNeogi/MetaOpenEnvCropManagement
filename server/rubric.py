"""RFC 004 rubric for Precision Agriculture Crop Management.

Provides a thin interface on top of the existing deterministic grader.
RL frameworks (TRL, torchforge) expect ``rubric_reward`` alongside
per-step ``reward`` so that trajectory-level scoring is unambiguous.
"""
from __future__ import annotations

from typing import Any

from server.grader import grade_episode


class CropManagementRubric:
    """Trajectory-level rubric that wraps the multi-metric grader.

    Usage::

        rubric = CropManagementRubric()
        score, breakdown = rubric.score_episode(
            actual_yield=..., target_yield=..., ...
        )
    """

    def score_episode(
        self,
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
        """Score a completed episode.

        Returns ``(score, rubric_breakdown)`` where *score* is in [0, 1]
        and *rubric_breakdown* is a dict of per-metric scores.
        """
        return grade_episode(
            actual_yield=actual_yield,
            target_yield=target_yield,
            total_water=total_water,
            total_n=total_n,
            total_cost=total_cost,
            budget=budget,
            harvest_dvs=harvest_dvs,
            harvested=harvested,
            actions_taken=actions_taken,
            task_id=task_id,
        )
