"""Pydantic models for the Precision Agriculture Crop Management OpenEnv.

Defines the three core types required by the OpenEnv framework:
  - CropAction:      what the agent sends each step
  - CropObservation:  what the agent receives each step
  - CropState:        full internal state for checkpointing / replay

All models inherit from openenv base classes (Action, Observation, State)
which are themselves Pydantic BaseModels with extra="forbid" / extra="allow".
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Agent action
# ---------------------------------------------------------------------------


class CropAction(Action):
    """Weekly management decision sent by the agent.

    action_type: one of 'irrigate', 'fertilize', 'harvest', 'wait'
    amount:      cm of water (irrigate) or kg N/ha (fertilize); 0 for others
    """
    action_type: str = "wait"
    amount: float = 0.0


# ---------------------------------------------------------------------------
# Typed observation sub-models
# ---------------------------------------------------------------------------


class CropStatus(BaseModel):
    """Crop growth state."""
    dvs: float = 0.0
    lai: float = 0.0
    tagp: float = 0.0
    twso: float = 0.0
    growth_stage: str = ""


class SoilStatus(BaseModel):
    """Soil moisture and nutrient state."""
    sm: float = 0.0
    water_deficit: bool = False
    water_stress: float = 0.0
    n_availability: float = 0.0
    field_capacity: float = 0.0
    wilting_point: float = 0.0


class ResourcesUsed(BaseModel):
    """Cumulative resource usage and budget."""
    total_water_cm: float = 0.0
    total_n_kg_ha: float = 0.0
    total_cost: float = 0.0
    budget_remaining: float = 0.0
    irrigation_cost_per_cm: float = 0.0
    fertilizer_cost_per_kg: float = 0.0


class ControlFeatures(BaseModel):
    """Derived RL-facing features for policy consumption."""
    moisture_gap_to_target: float = 0.0
    forecast_rain_3d: float = 0.0
    forecast_rain_7d: float = 0.0
    days_since_last_irrigation: int = 0
    days_since_last_fertilization: int = 0
    fertilizer_events_count: int = 0
    cumulative_n_applied: float = 0.0
    rooting_depth_cm: float = 0.0
    estimated_budget_to_finish: float = 0.0
    budget_remaining_ratio: float = 0.0
    dvs_distance_to_next_fertilizer_window: float = 0.0


# ---------------------------------------------------------------------------
# Agent observation (what the LLM sees each step)
# ---------------------------------------------------------------------------


class WeatherDay(BaseModel):
    """Single day of weather data."""
    day: int = 0
    tmax: float = 0.0
    tmin: float = 0.0
    rain: float = 0.0
    radiation: float = 0.0


class CropObservation(Observation):
    """Rich observation returned to the agent after each step.

    Contains crop status, soil moisture, weather (current + 5-day forecast),
    resource usage, season context, and any conflict feedback from the
    environment when an invalid action was attempted.

    RFC 004 compliance: ``rubric_reward`` carries the trajectory-level score
    (from the grader) at terminal steps, while ``reward`` carries the dense
    per-step signal.  Non-terminal steps have ``rubric_reward = None``.
    """
    task_id: int = 0
    task_name: str = ""
    instructions: str = ""
    day: int = 0                  # Current simulation day since sowing
    days_remaining: int = 0       # Days left before season ends

    crop_status: CropStatus = Field(default_factory=CropStatus)
    soil_status: SoilStatus = Field(default_factory=SoilStatus)
    weather_today: WeatherDay = Field(default_factory=WeatherDay)
    weather_forecast: list[WeatherDay] = Field(default_factory=list)
    resources_used: ResourcesUsed = Field(default_factory=ResourcesUsed)
    season_summary: dict[str, Any] = Field(default_factory=dict)
    control_features: ControlFeatures = Field(default_factory=ControlFeatures)
    conflicts: list[str] = Field(default_factory=list)
    advisory_text: Optional[str] = None
    rubric_reward: float | None = None


# ---------------------------------------------------------------------------
# Full internal state (for /state endpoint & checkpointing)
# ---------------------------------------------------------------------------


class CropState(State):
    """Complete simulator state — used for checkpointing and the /state endpoint.

    Mirrors the internal CropSimulator variables plus episode-level tracking
    (budget, actions taken, harvest status).
    """
    current_task_id: Optional[int] = None
    seed: Optional[int] = None
    current_day: int = 0
    total_days: int = 0
    crop_name: str = ""

    # Crop growth variables (from CropSimulator)
    dvs: float = 0.0              # Development stage: 0=sowing → 1=anthesis → 2=maturity
    lai: float = 0.0              # Leaf Area Index
    tagp: float = 0.0             # Total above-ground production (kg/ha)
    twso: float = 0.0             # Storage organ / grain weight (kg/ha)
    sm: float = 0.0               # Volumetric soil moisture (fraction)

    # Resource tracking
    total_water_applied: float = 0.0   # Cumulative irrigation (cm)
    total_n_applied: float = 0.0       # Cumulative nitrogen (kg N/ha)
    total_cost: float = 0.0            # Cumulative cost ($)
    budget: float = 0.0                # Total budget for the episode

    # Episode history
    actions_taken: list[dict[str, Any]] = Field(default_factory=list)
    harvested: bool = False
    harvest_dvs: float = 0.0           # DVS at time of harvest
    last_irrigation_day: Optional[int] = None
    last_fertilization_day: Optional[int] = None
    fertilizer_events_count: int = 0
