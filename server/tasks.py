"""Task definitions for Precision Agriculture Crop Management.

Each task represents a difficulty level with different environment conditions.
Scoring uses the SAME unified formula for all tasks — difficulty comes from
the climate, budget, and soil, not from different grading weights.

  Task 1 (Easy):   Netherlands — mild climate, good rainfall, generous budget
  Task 2 (Medium): Iowa — variable weather, drought risk, moderate budget
  Task 3 (Hard):   Punjab — hot, dry, tight budget, expensive inputs
"""
from __future__ import annotations


TASKS = {
    1: {
        "id": 1,
        "name": "Basic Crop Growth",
        "difficulty": "easy",
        "instructions": (
            "You are managing a wheat crop in the Netherlands. The climate is mild "
            "with regular rainfall, so the soil stays moist naturally most of the time. "
            "Your budget is generous ($800). The main challenge is timing: fertilize "
            "at the right growth stages — first window DVS 0.20–0.40 (target 0.30), "
            "second window DVS 0.50–0.70 (target 0.60) — and harvest "
            "when the crop reaches maturity (DVS in [1.80, 2.05]). Irrigation is rarely needed. "
            "Actions: irrigate (amount in cm), fertilize (amount in kg N/ha), harvest, "
            "or wait. "
            "\n\nScoring formula (same for all tasks): "
            "0.35×yield + 0.20×water_efficiency + 0.18×cost_efficiency "
            "+ 0.15×timing_quality + 0.12×harvest_timing. "
            "Yield = actual/target. Water efficiency = 1 - water_used/50cm. "
            "Cost efficiency = 1 - cost/budget. Timing = proximity of fertilization "
            "to DVS 0.3 and 0.6. Harvest timing = penalty if DVS outside [1.8, 2.05]."
        ),
    },
    2: {
        "id": 2,
        "name": "Water-Efficient Farming",
        "difficulty": "medium",
        "instructions": (
            "You are managing a wheat crop in Iowa, USA. The weather is variable — "
            "moderate rainfall with occasional drought periods that stress the crop. "
            "Your budget is moderate ($450) and input costs are higher than average. "
            "You must balance irrigation needs against water conservation and cost. "
            "Monitor soil moisture and weather forecasts carefully. Irrigate only when "
            "soil moisture drops below 0.22 and no rain is forecast in the next 2 days. "
            "Fertilize in the optimal windows: DVS 0.20–0.40 (target 0.30) and "
            "DVS 0.50–0.70 (target 0.60). Harvest at DVS in [1.80, 2.05]. "
            "Actions: irrigate (amount in cm), fertilize (amount in kg N/ha), harvest, "
            "or wait. "
            "\n\nScoring formula (same for all tasks): "
            "0.35×yield + 0.20×water_efficiency + 0.18×cost_efficiency "
            "+ 0.15×timing_quality + 0.12×harvest_timing. "
            "Yield = actual/target. Water efficiency = 1 - water_used/50cm. "
            "Cost efficiency = 1 - cost/budget. Timing = proximity of fertilization "
            "to DVS 0.3 and 0.6. Harvest timing = penalty if DVS outside [1.8, 2.05]."
        ),
    },
    3: {
        "id": 3,
        "name": "Precision Agriculture",
        "difficulty": "hard",
        "instructions": (
            "You are managing a wheat crop in Punjab, India — a hot, semi-arid region "
            "with very little rainfall during the growing season. Your budget is tight "
            "($300) and irrigation/fertilizer costs are the highest ($3/cm water, "
            "$2/kg N). The crop WILL suffer water stress without irrigation, but every "
            "irrigation event is expensive. You must make every dollar count: irrigate "
            "only when soil moisture is critically low (< 0.20) and no rain is expected, "
            "fertilize precisely in the optimal windows DVS 0.20–0.40 (target 0.30) "
            "and DVS 0.50–0.70 (target 0.60) with minimal amounts, and harvest "
            "at optimal maturity (DVS in [1.80, 2.05]). Wasteful spending leaves no budget for "
            "critical late-season irrigation. "
            "Actions: irrigate (amount in cm), fertilize (amount in kg N/ha), harvest, "
            "or wait. "
            "\n\nScoring formula (same for all tasks): "
            "0.35×yield + 0.20×water_efficiency + 0.18×cost_efficiency "
            "+ 0.15×timing_quality + 0.12×harvest_timing. "
            "Yield = actual/target. Water efficiency = 1 - water_used/50cm. "
            "Cost efficiency = 1 - cost/budget. Timing = proximity of fertilization "
            "to DVS 0.3 and 0.6. Harvest timing = penalty if DVS outside [1.8, 2.05]."
        ),
    },
}


def get_task_definition(task_id: int) -> dict:
    if task_id not in TASKS:
        raise ValueError(f"Unsupported task_id: {task_id}")
    return TASKS[task_id]
