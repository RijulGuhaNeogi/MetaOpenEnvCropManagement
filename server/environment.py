"""Precision Agriculture Crop Management — OpenEnv Environment.

This is the core environment class that implements the OpenEnv interface.
An LLM agent manages a wheat growing season by making weekly decisions:
irrigation, fertilization, harvest, or wait.

Episode flow:
  1. reset(seed, task_id) → creates a CropSimulator with seeded weather
  2. step(action) → validates action, advances sim by 7 days, returns obs
  3. Episode ends when: agent harvests, DVS≥2.0 (auto-maturity),
     season duration exceeded, or MAX_STEPS reached
  4. Terminal observation includes the final grader score as reward

All internal state is deterministic: same seed + same actions = same result.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

from models import CropAction, CropObservation, CropState
from server.crop_sim import CropSimulator
from server.grader import grade_episode
from server.reward import compute_step_reward, compute_trajectory_reward
from server.scenarios import generate_scenario
from server.tasks import TASKS, get_task_definition


# Maximum number of agent steps before forced termination
MAX_STEPS = 60


class CropEnvironment(
    Environment[CropAction, CropObservation, CropState]
):
    def __init__(self) -> None:
        super().__init__()
        self._sim: Optional[CropSimulator] = None
        self._scenario: dict[str, Any] = {}
        self._state = CropState()

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CropObservation:
        task_id: int = kwargs.get("task_id", 1)
        task = get_task_definition(task_id)
        actual_seed = seed if seed is not None else 42

        scenario = generate_scenario(actual_seed, task_id)
        self._scenario = scenario

        # Create simulator
        self._sim = CropSimulator(
            crop_params=scenario["crop_params"],
            soil_params=scenario["soil_params"],
            weather_data=scenario["weather"],
            partition_table=scenario["partition_table"],
        )

        self._state = CropState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            current_task_id=task_id,
            seed=actual_seed,
            current_day=0,
            total_days=scenario["max_duration"],
            crop_name=scenario["crop_name"],
            dvs=0.0,
            lai=self._sim.lai,
            tagp=0.0,
            twso=0.0,
            sm=self._sim.sm,
            total_water_applied=0.0,
            total_n_applied=0.0,
            total_cost=0.0,
            budget=scenario["budget"],
            actions_taken=[],
            harvested=False,
            harvest_dvs=0.0,
        )

        return self._build_observation(task)

    def step(
        self,
        action: CropAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CropObservation:
        if self._sim is None or self._state.current_task_id is None:
            raise RuntimeError("Environment has not been reset.")

        if self._state.harvested:
            raise RuntimeError("Episode already done (harvested) — call reset().")

        task = get_task_definition(self._state.current_task_id)
        scenario = self._scenario
        conflicts: list[str] = []
        step_reward = 0.0

        self._state.step_count += 1

        action_type = action.action_type.lower().strip()
        amount = max(0.0, action.amount)

        # Validate action type
        if action_type not in ("irrigate", "fertilize", "harvest", "wait"):
            conflicts.append(
                f"Invalid action_type: '{action.action_type}'. "
                "Must be one of: irrigate, fertilize, harvest, wait."
            )
            action_type = "wait"
            amount = 0.0

        # Compute cost
        cost = 0.0
        irrig_cm = 0.0
        n_kg = 0.0

        if action_type == "irrigate":
            amount = min(amount, 10.0)  # Cap at 10 cm per step
            if amount <= 0:
                conflicts.append("Irrigation amount must be > 0. Treating as wait.")
                action_type = "wait"
                amount = 0.0
            else:
                cost = amount * scenario["irrigation_cost"]
                irrig_cm = amount

        elif action_type == "fertilize":
            amount = min(amount, 50.0)  # Cap at 50 kg N/ha per step
            if amount <= 0:
                conflicts.append("Fertilizer amount must be > 0. Treating as wait.")
                action_type = "wait"
                amount = 0.0
            else:
                cost = amount * scenario["fertilizer_cost"]
                n_kg = amount

        # Budget check
        budget_remaining = self._state.budget - self._state.total_cost
        if cost > budget_remaining and action_type in ("irrigate", "fertilize"):
            conflicts.append(
                f"Over budget: action costs ${cost:.1f} but only "
                f"${budget_remaining:.1f} remaining. Treating as wait."
            )
            action_type = "wait"
            amount = 0.0
            cost = 0.0
            irrig_cm = 0.0
            n_kg = 0.0

        # Record action
        action_record = {
            "step": self._state.step_count,
            "action_type": action_type,
            "amount": amount,
            "dvs": round(self._sim.dvs, 3),
            "sm": round(self._sim.sm, 3),
            "cost": round(cost, 2),
        }
        self._state.actions_taken.append(action_record)

        # Handle harvest
        if action_type == "harvest":
            self._state.harvested = True
            self._state.harvest_dvs = self._sim.dvs
            self._state.total_cost += cost

            # Compute final grade (step_reward is not used for terminal steps)
            grade, breakdown = grade_episode(
                actual_yield=self._sim.twso,
                target_yield=scenario["target_yield"],
                total_water=self._sim.total_water,
                total_n=self._sim.total_n,
                total_cost=self._state.total_cost,
                budget=self._state.budget,
                harvest_dvs=self._sim.dvs,
                harvested=True,
                actions_taken=self._state.actions_taken,
                task_id=self._state.current_task_id,
            )
            final_reward = compute_trajectory_reward(grade)

            self._sync_state()
            return self._build_observation(
                task, done=True, reward=final_reward,
                conflicts=conflicts, metadata={"grade_breakdown": breakdown},
            )

        # Compute step reward before advancing
        step_reward = compute_step_reward(
            action_type=action_type,
            dvs=self._sim.dvs,
            sm=self._sim.sm,
            cost=cost,
            budget_remaining=budget_remaining,
        )

        # Advance simulation
        step_days = scenario["step_days"]
        self._sim.advance(step_days, irrigation_cm=irrig_cm, n_kg_ha=n_kg)
        self._state.total_cost += cost
        self._sync_state()

        # Check termination conditions
        is_done = False
        final_reward = step_reward

        if self._sim.dvs >= 2.0:
            # Auto-harvest at full maturity
            self._state.harvested = True
            self._state.harvest_dvs = self._sim.dvs
            is_done = True

        if self._sim.current_day >= scenario["max_duration"]:
            # Season over — force end
            if not self._state.harvested:
                self._state.harvested = True
                self._state.harvest_dvs = self._sim.dvs
            is_done = True

        if self._state.step_count >= MAX_STEPS:
            # Safety cap on episode length
            if not self._state.harvested:
                self._state.harvested = True
                self._state.harvest_dvs = self._sim.dvs
            is_done = True

        if is_done:
            grade, breakdown = grade_episode(
                actual_yield=self._sim.twso,
                target_yield=scenario["target_yield"],
                total_water=self._sim.total_water,
                total_n=self._sim.total_n,
                total_cost=self._state.total_cost,
                budget=self._state.budget,
                harvest_dvs=self._state.harvest_dvs,
                harvested=self._state.harvested,
                actions_taken=self._state.actions_taken,
                task_id=self._state.current_task_id,
            )
            final_reward = compute_trajectory_reward(grade)
            return self._build_observation(
                task, done=True, reward=final_reward,
                conflicts=conflicts, metadata={"grade_breakdown": breakdown},
            )

        return self._build_observation(
            task, done=False, reward=step_reward, conflicts=conflicts,
        )

    @property
    def state(self) -> CropState:
        return self._state.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sync_state(self) -> None:
        """Sync simulator state into the Pydantic state model."""
        if self._sim is None:
            return
        self._state.current_day = self._sim.current_day
        self._state.dvs = round(self._sim.dvs, 4)
        self._state.lai = round(self._sim.lai, 3)
        self._state.tagp = round(self._sim.tagp, 1)
        self._state.twso = round(self._sim.twso, 1)
        self._state.sm = round(self._sim.sm, 4)
        self._state.total_water_applied = round(self._sim.total_water, 2)
        self._state.total_n_applied = round(self._sim.total_n, 2)

    def _build_observation(
        self,
        task: dict,
        done: bool = False,
        reward: float | None = None,
        conflicts: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CropObservation:
        sim = self._sim
        scenario = self._scenario

        if sim is None:
            return CropObservation(done=done, reward=reward)

        weather_today = sim.get_weather(sim.current_day)
        weather_forecast = sim.get_weather_forecast(
            sim.current_day + 1, n_days=5
        )

        budget_remaining = self._state.budget - self._state.total_cost

        return CropObservation(
            done=done,
            reward=reward,
            metadata=metadata or {},
            task_id=task["id"],
            task_name=task["name"],
            instructions=task["instructions"],
            day=sim.current_day,
            days_remaining=max(0, scenario["max_duration"] - sim.current_day),
            crop_status={
                "dvs": round(sim.dvs, 3),
                "lai": round(sim.lai, 2),
                "tagp": round(sim.tagp, 1),
                "twso": round(sim.twso, 1),
                "growth_stage": sim.growth_stage_name(),
            },
            soil_status={
                "sm": round(sim.sm, 3),
                "water_deficit": sim.sm < 0.22,
                "water_stress": round(sim._water_stress(), 3),
                "n_availability": round(sim.n_factor, 3),
                "field_capacity": scenario["soil_params"].field_capacity,
                "wilting_point": scenario["soil_params"].wilting_point,
            },
            weather_today={
                "tmax": weather_today["tmax"],
                "tmin": weather_today["tmin"],
                "rain": weather_today["rain"],
                "radiation": weather_today["radiation"],
            },
            weather_forecast=weather_forecast,
            resources_used={
                "total_water_cm": round(sim.total_water, 2),
                "total_n_kg_ha": round(sim.total_n, 2),
                "total_cost": round(self._state.total_cost, 2),
                "budget_remaining": round(budget_remaining, 2),
                "irrigation_cost_per_cm": scenario["irrigation_cost"],
                "fertilizer_cost_per_kg": scenario["fertilizer_cost"],
            },
            season_summary={
                "crop_name": scenario["crop_name"],
                "location": scenario["location"],
                "target_yield": round(scenario["target_yield"], 1),
                "budget": scenario["budget"],
                "step_days": scenario["step_days"],
            },
            conflicts=conflicts or [],
        )
