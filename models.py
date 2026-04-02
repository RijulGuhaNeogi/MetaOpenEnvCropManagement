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

from pydantic import Field
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
# Agent observation (what the LLM sees each step)
# ---------------------------------------------------------------------------


class CropObservation(Observation):
    """Rich observation returned to the agent after each step.

    Contains crop status, soil moisture, weather (current + 5-day forecast),
    resource usage, season context, and any conflict feedback from the
    environment when an invalid action was attempted.
    """
    task_id: int = 0
    task_name: str = ""
    instructions: str = ""
    day: int = 0                  # Current simulation day since sowing
    days_remaining: int = 0       # Days left before season ends

    # Nested dicts keep the observation flexible without rigid nesting models
    crop_status: dict[str, Any] = Field(default_factory=dict)
    soil_status: dict[str, Any] = Field(default_factory=dict)
    weather_today: dict[str, Any] = Field(default_factory=dict)
    weather_forecast: list[dict[str, Any]] = Field(default_factory=list)
    resources_used: dict[str, Any] = Field(default_factory=dict)
    season_summary: dict[str, Any] = Field(default_factory=dict)
    conflicts: list[str] = Field(default_factory=list)


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
