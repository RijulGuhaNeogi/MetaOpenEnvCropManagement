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
        "observability_tier": 1,
        "hidden_fields": [],
        "instructions": (
            "Manage a wheat crop through one growing season in the Netherlands — a mild, "
            "well-watered climate with a generous budget ($800). Your goal is to maximize "
            "a composite score combining yield, water efficiency, cost efficiency, fertilizer "
            "timing, and harvest timing. The crop progresses through growth stages measured by "
            "DVS (Development Stage, 0→2). You observe soil moisture, weather forecasts, crop "
            "status, and budget each week. Available actions: irrigate (cm of water), fertilize "
            "(kg N/ha), harvest, or wait. You may also inspect_soil ($10) or inspect_crop ($20) "
            "for detailed field reports. The scoring formula is identical across all tasks: "
            "0.35×yield + 0.20×water_efficiency + 0.18×cost_efficiency "
            "+ 0.15×timing_quality + 0.12×harvest_timing."
        ),
    },
    2: {
        "id": 2,
        "name": "Water-Efficient Farming",
        "difficulty": "medium",
        "observability_tier": 2,
        "hidden_fields": ["dvs", "sm", "n_availability", "water_stress"],
        "instructions": (
            "Manage a wheat crop through one growing season in Iowa, USA — a continental "
            "climate with variable rainfall and occasional drought periods. Your budget is "
            "moderate ($450) with higher input costs. Some precise sensor readings are "
            "unavailable — you see coarsened bands (e.g. soil moisture 'low', nitrogen "
            "'adequate') and a natural-language weather summary instead of exact numbers. "
            "You can use inspect_soil ($10) or inspect_crop ($20) to reveal precise "
            "measurements for one step, but each costs budget and a week. Balance "
            "information-gathering against crop management. The scoring formula is identical "
            "across all tasks: 0.35×yield + 0.20×water_efficiency + 0.18×cost_efficiency "
            "+ 0.15×timing_quality + 0.12×harvest_timing."
        ),
    },
    3: {
        "id": 3,
        "name": "Precision Agriculture",
        "difficulty": "hard",
        "observability_tier": 3,
        "hidden_fields": ["dvs", "sm", "n_availability", "water_stress", "lai", "tagp", "twso"],
        "instructions": (
            "Manage a wheat crop through one growing season in Punjab, India — a hot, "
            "semi-arid region with very little rainfall. Your budget is tight ($300) and "
            "input costs are the highest across all tasks ($3/cm water, $2/kg N). Most "
            "precise sensor readings are unavailable — you see coarsened bands and a "
            "bucketed weather summary with reduced precision. You can use inspect_soil ($10) "
            "or inspect_crop ($20) to reveal precise measurements, but each costs budget "
            "and a week — a significant expense with a $300 budget. You must decide whether "
            "to invest in information or act on imprecise signals. The scoring formula is "
            "identical across all tasks: 0.35×yield + 0.20×water_efficiency "
            "+ 0.18×cost_efficiency + 0.15×timing_quality + 0.12×harvest_timing."
        ),
    },
}


def get_task_definition(task_id: int) -> dict:
    if task_id not in TASKS:
        raise ValueError(f"Unsupported task_id: {task_id}")
    return TASKS[task_id]
