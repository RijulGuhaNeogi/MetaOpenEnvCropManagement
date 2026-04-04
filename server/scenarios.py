"""Seeded scenario generator for Precision Agriculture Crop Management.

Generates weather data, crop/soil parameters, and budget constraints
programmatically. Same seed always produces the same scenario.
No external API calls or data files — fully self-contained.

Simplification: we focus on wheat across 3 climatic profiles:
  - Netherlands (mild, reliable rainfall)
  - Punjab, India (hot, dry, needs irrigation)
  - Iowa, USA (moderate, some drought risk)
"""
from __future__ import annotations

import math
import random
from typing import Any

from server.crop_sim import (
    CropSimulator,
    compute_potential_yield,
)
from server.crop_params import (
    CROP_LIBRARY,
    SOIL_LIBRARY,
)


# ---------------------------------------------------------------------------
# Embedded weather generators (deterministic, seed-based)
# ---------------------------------------------------------------------------

def _generate_weather_netherlands(rng: random.Random, n_days: int) -> list[dict]:
    """Mild maritime climate — regular rainfall, moderate temps."""
    weather = []
    for d in range(n_days):
        # Seasonal temperature curve (sowing in October, harvest in July)
        # Day 0 = Oct 15, Day 270 = ~Jul 12
        season_frac = d / 365.0
        base_temp = 5.0 + 12.0 * math.sin(math.pi * (season_frac + 0.2))
        tmax = base_temp + 3.0 + rng.gauss(0, 1.5)
        tmin = base_temp - 3.0 + rng.gauss(0, 1.2)
        tmin = min(tmin, tmax - 1.0)

        # Rainfall — frequent light rain
        rain_prob = 0.45 - 0.15 * math.sin(math.pi * (season_frac + 0.3))
        rain = 0.0
        if rng.random() < rain_prob:
            rain = rng.expovariate(1.0 / 0.6)  # mean 0.6 cm
            rain = min(rain, 3.0)

        # Solar radiation (MJ/m2/day)
        radiation = 8.0 + 12.0 * math.sin(math.pi * (season_frac + 0.15))
        radiation = max(2.0, radiation + rng.gauss(0, 2.0))

        weather.append({
            "day": d,
            "tmax": round(tmax, 1),
            "tmin": round(tmin, 1),
            "rain": round(max(0.0, rain), 2),
            "radiation": round(radiation, 1),
        })
    return weather


def _generate_weather_punjab(rng: random.Random, n_days: int) -> list[dict]:
    """Hot semi-arid — monsoon rains Jul-Sep, dry winter growing season.

    Punjab winter wheat (Nov-Apr) receives approximately 5-10 cm of rainfall.
    """
    weather = []
    for d in range(n_days):
        season_frac = d / 365.0
        base_temp = 18.0 + 12.0 * math.sin(math.pi * (season_frac + 0.15))
        tmax = base_temp + 6.0 + rng.gauss(0, 2.0)
        tmin = base_temp - 4.0 + rng.gauss(0, 1.5)
        tmin = min(tmin, tmax - 2.0)

        # Sparse but non-trivial rainfall during winter wheat season
        rain_prob = 0.12 + 0.03 * math.sin(math.pi * (season_frac + 0.5))
        rain = 0.0
        if rng.random() < rain_prob:
            rain = rng.expovariate(1.0 / 0.6)  # mean 0.6 cm per event
            rain = min(rain, 2.5)

        radiation = 12.0 + 8.0 * math.sin(math.pi * (season_frac + 0.1))
        radiation = max(4.0, radiation + rng.gauss(0, 1.5))

        weather.append({
            "day": d,
            "tmax": round(tmax, 1),
            "tmin": round(tmin, 1),
            "rain": round(max(0.0, rain), 2),
            "radiation": round(radiation, 1),
        })
    return weather


def _generate_weather_iowa(rng: random.Random, n_days: int) -> list[dict]:
    """Continental — cold winters, warm summers, moderate rainfall with dry spells."""
    weather = []
    for d in range(n_days):
        season_frac = d / 365.0
        base_temp = 3.0 + 16.0 * math.sin(math.pi * (season_frac + 0.2))
        tmax = base_temp + 4.0 + rng.gauss(0, 2.5)
        tmin = base_temp - 4.0 + rng.gauss(0, 2.0)
        tmin = min(tmin, tmax - 1.0)

        # Moderate rain with occasional dry spells
        rain_prob = 0.30 + 0.10 * math.sin(math.pi * (season_frac + 0.3))
        rain = 0.0
        if rng.random() < rain_prob:
            rain = rng.expovariate(1.0 / 0.5)
            rain = min(rain, 4.0)

        radiation = 7.0 + 13.0 * math.sin(math.pi * (season_frac + 0.15))
        radiation = max(2.0, radiation + rng.gauss(0, 2.5))

        weather.append({
            "day": d,
            "tmax": round(tmax, 1),
            "tmin": round(tmin, 1),
            "rain": round(max(0.0, rain), 2),
            "radiation": round(radiation, 1),
        })
    return weather


# ---------------------------------------------------------------------------
# Location configurations
# ---------------------------------------------------------------------------

LOCATIONS = {
    "netherlands": {
        "name": "Netherlands",
        "weather_fn": _generate_weather_netherlands,
        "soil": "clay_loam",
        "crop": "wheat_nl",
        "max_duration": 280,
    },
    "punjab": {
        "name": "Punjab, India",
        "weather_fn": _generate_weather_punjab,
        "soil": "sandy_loam",
        "crop": "wheat_punjab",
        "max_duration": 200,
    },
    "iowa": {
        "name": "Iowa, USA",
        "weather_fn": _generate_weather_iowa,
        "soil": "silt_loam",
        "crop": "wheat_iowa",
        "max_duration": 260,
    },
}


PROBE_SCENARIOS = {
    "over_irrigation_trap": {
        "task_id": 1,
        "budget": 250.0,
        "override_sm": 0.39,
        "force_forecast_rain": [1.1, 0.9, 0.7, 0.5, 0.4],
        "notes": "Soil already wet and rain is coming; irrigation should look unattractive.",
    },
    "late_fertilizer_temptation": {
        "task_id": 2,
        "start_at_dvs": 1.18,
        "override_n_factor": 0.52,
        "budget": 160.0,
        "notes": "Nitrogen looks low, but the crop is already too late for a profitable top-dress.",
    },
    "budget_starvation": {
        "task_id": 3,
        "start_at_dvs": 0.42,
        "override_sm": 0.18,
        "override_n_factor": 0.50,
        "budget": 32.0,
        "notes": "Budget only covers one meaningful intervention; waste is immediately punished.",
    },
    "harvest_hesitation": {
        "task_id": 1,
        "start_at_dvs": 1.76,
        "override_sm": 0.30,
        "budget": 120.0,
        "notes": "Crop is near maturity; delaying harvest should be an obvious mistake.",
    },
    "drought_rescue": {
        "task_id": 3,
        "start_at_dvs": 0.58,
        "override_sm": 0.12,
        "override_n_factor": 0.72,
        "budget": 140.0,
        "force_forecast_rain": [0.0, 0.0, 0.0, 0.1, 0.0],
        "notes": "Mid-season drought with enough budget to rescue the crop if irrigation is used well.",
    },
}


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------

def generate_scenario(seed: int, task_id: int) -> dict[str, Any]:
    """Generate a crop management scenario for the given task difficulty.

    The target_yield is the MAXIMUM potential yield across all 3 locations
    (computed with unlimited water and nitrogen on each location's weather).
    Using a universal target ensures fair scoring: easy locations naturally
    achieve a higher fraction of the target than hard locations, so
    difficulty ordering (Easy > Medium > Hard) is guaranteed without
    needing different grading weights.

    Returns dict with keys: crop_name, crop_params, soil_params,
    partition_table, weather, location, max_duration, budget,
    target_yield, step_days, irrigation_cost, fertilizer_cost.
    """
    if task_id not in (1, 2, 3):
        raise ValueError(f"task_id must be 1, 2, or 3, got {task_id}")

    rng = random.Random(seed)

    # Compute potential yield for ALL locations, use the max as universal target.
    # This means yield_score = actual / best_possible_anywhere, ensuring:
    #   - No task can exceed 1.0 (cap is never triggered)
    #   - Easy locations score higher naturally (better growing conditions)
    #   - Hard locations score lower naturally (worse conditions, not unfair target)
    universal_target = _compute_universal_target(seed)

    if task_id == 1:
        return _generate_easy(rng, seed, universal_target)
    elif task_id == 2:
        return _generate_medium(rng, seed, universal_target)
    else:
        return _generate_hard(rng, seed, universal_target)


def generate_probe_scenario(seed: int, probe_name: str) -> dict[str, Any]:
    """Generate an internal probe scenario for RL diagnostics.

    Probe scenarios are not public tasks. They modify the initial state or short
    weather horizon to expose specific failure modes.
    """
    if probe_name not in PROBE_SCENARIOS:
        raise ValueError(f"Unsupported probe_name: {probe_name}")

    probe = PROBE_SCENARIOS[probe_name]
    scenario = generate_scenario(seed, probe["task_id"])
    scenario["weather"] = [day.copy() for day in scenario["weather"]]
    scenario["probe_name"] = probe_name
    scenario["probe_notes"] = probe["notes"]

    if "budget" in probe:
        scenario["budget"] = probe["budget"]
    if "override_sm" in probe:
        scenario["override_sm"] = probe["override_sm"]
    if "override_n_factor" in probe:
        scenario["override_n_factor"] = probe["override_n_factor"]
    if "start_at_dvs" in probe:
        scenario["start_at_dvs"] = probe["start_at_dvs"]
    if "force_forecast_rain" in probe:
        scenario["force_forecast_rain"] = list(probe["force_forecast_rain"])

    return scenario


def _compute_universal_target(seed: int) -> float:
    """Compute the max potential yield across all 3 locations for this seed."""
    potentials = []
    for loc_key, loc in LOCATIONS.items():
        # Use the same seed derivation as each task's weather generator
        if loc_key == "netherlands":
            weather_seed = seed * 31 + 1
        elif loc_key == "iowa":
            weather_seed = seed * 37 + 2
        else:  # punjab
            weather_seed = seed * 41 + 3
        weather = loc["weather_fn"](
            random.Random(weather_seed), loc["max_duration"] + 30
        )
        pot = compute_potential_yield(loc["crop"], weather, loc["max_duration"])
        potentials.append(pot)
    return max(potentials)


def _generate_easy(rng: random.Random, seed: int, target_yield: float) -> dict[str, Any]:
    """Netherlands wheat, generous budget, good rainfall."""
    loc = LOCATIONS["netherlands"]
    crop_name = loc["crop"]
    crop_params = CROP_LIBRARY[crop_name]
    soil_params = SOIL_LIBRARY[loc["soil"]]
    max_duration = loc["max_duration"]

    weather = loc["weather_fn"](random.Random(seed * 31 + 1), max_duration + 30)

    return {
        "crop_name": crop_name,
        "crop_params": crop_params,
        "soil_params": soil_params,
        "partition_table": crop_params.FOTB,
        "weather": weather,
        "location": loc["name"],
        "max_duration": max_duration,
        "budget": 800.0,
        "target_yield": target_yield,
        "step_days": 7,
        "irrigation_cost": 2.0,
        "fertilizer_cost": 1.5,
    }


def _generate_medium(rng: random.Random, seed: int, target_yield: float) -> dict[str, Any]:
    """Iowa wheat, moderate budget, some drought risk."""
    loc = LOCATIONS["iowa"]
    crop_name = loc["crop"]
    crop_params = CROP_LIBRARY[crop_name]
    soil_params = SOIL_LIBRARY[loc["soil"]]
    max_duration = loc["max_duration"]

    weather = loc["weather_fn"](random.Random(seed * 37 + 2), max_duration + 30)

    return {
        "crop_name": crop_name,
        "crop_params": crop_params,
        "soil_params": soil_params,
        "partition_table": crop_params.FOTB,
        "weather": weather,
        "location": loc["name"],
        "max_duration": max_duration,
        "budget": 450.0,
        "target_yield": target_yield,
        "step_days": 7,
        "irrigation_cost": 2.5,
        "fertilizer_cost": 1.8,
    }


def _generate_hard(rng: random.Random, seed: int, target_yield: float) -> dict[str, Any]:
    """Punjab wheat, tight budget, drought-prone."""
    loc = LOCATIONS["punjab"]
    crop_name = loc["crop"]
    crop_params = CROP_LIBRARY[crop_name]
    soil_params = SOIL_LIBRARY[loc["soil"]]
    max_duration = loc["max_duration"]

    weather = loc["weather_fn"](random.Random(seed * 41 + 3), max_duration + 30)

    return {
        "crop_name": crop_name,
        "crop_params": crop_params,
        "soil_params": soil_params,
        "partition_table": crop_params.FOTB,
        "weather": weather,
        "location": loc["name"],
        "max_duration": max_duration,
        "budget": 300.0,
        "target_yield": target_yield,
        "step_days": 7,
        "irrigation_cost": 3.0,
        "fertilizer_cost": 2.0,
    }
