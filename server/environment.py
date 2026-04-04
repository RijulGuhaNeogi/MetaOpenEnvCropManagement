"""Precision Agriculture Crop Management — OpenEnv Environment.

This is the core environment class that implements the OpenEnv interface.
An LLM agent manages a wheat growing season by making weekly decisions:
irrigation, fertilization, harvest, or wait.

Episode flow:
  1. reset(seed, task_id) → creates a CropSimulator with seeded weather
  2. step(action) → validates action, advances sim by 7 days, returns obs
  3. Episode ends when: agent harvests, DVS≥2.0 (auto-maturity),
     season duration exceeded, or MAX_STEPS reached
  4. Terminal observation includes rubric_reward (grader score) and
      reward (trajectory reward for RL training)

All internal state is deterministic: same seed + same actions = same result.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

from models import CropAction, CropObservation, CropState
from models import ControlFeatures, CropStatus, ResourcesUsed, SoilStatus, WeatherDay
from server.advisory import generate_advisory
from server.crop_sim import CropSimulator
from server.reward import (
    compute_delta_reward,
    compute_step_reward,
    compute_trajectory_reward,
)
from server.rubric import CropManagementRubric
from server.scenarios import generate_probe_scenario, generate_scenario
from server.tasks import TASKS, get_task_definition


# Maximum number of agent steps before forced termination
MAX_STEPS = 60


class CropEnvironment(
    Environment[CropAction, CropObservation, CropState]
):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self) -> None:
        super().__init__()
        self._sim: Optional[CropSimulator] = None
        self._scenario: dict[str, Any] = {}
        self._state = CropState()
        self._rubric = CropManagementRubric()

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
        probe_name: Optional[str] = kwargs.get("probe_name")
        task = get_task_definition(task_id)
        actual_seed = seed if seed is not None else 42

        scenario = (
            generate_probe_scenario(actual_seed, probe_name)
            if probe_name
            else generate_scenario(actual_seed, task_id)
        )
        self._scenario = scenario

        # Create simulator
        self._sim = CropSimulator(
            crop_params=scenario["crop_params"],
            soil_params=scenario["soil_params"],
            weather_data=scenario["weather"],
            partition_table=scenario["partition_table"],
        )

        self._apply_start_state_overrides(scenario)

        self._state = CropState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            current_task_id=task_id,
            seed=actual_seed,
            current_day=self._sim.current_day,
            total_days=scenario["max_duration"],
            crop_name=scenario["crop_name"],
            dvs=round(self._sim.dvs, 4),
            lai=self._sim.lai,
            tagp=round(self._sim.tagp, 1),
            twso=round(self._sim.twso, 1),
            sm=self._sim.sm,
            total_water_applied=round(self._sim.total_water, 2),
            total_n_applied=round(self._sim.total_n, 2),
            total_cost=0.0,
            budget=scenario["budget"],
            actions_taken=[],
            harvested=False,
            harvest_dvs=0.0,
            last_irrigation_day=None,
            last_fertilization_day=None,
            fertilizer_events_count=0,
        )

        return self._build_observation(task, metadata=self._step_metadata())

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
        current_day = self._sim.current_day
        if action_type == "irrigate" and irrig_cm > 0.0:
            self._state.last_irrigation_day = current_day
        elif action_type == "fertilize" and n_kg > 0.0:
            self._state.last_fertilization_day = current_day
            self._state.fertilizer_events_count += 1

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
            grade, breakdown = self._rubric.score_episode(
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
                rubric_reward=grade,
                conflicts=conflicts,
                metadata={
                    **self._step_metadata(),
                    "rubric_breakdown": breakdown,
                },
            )

        # Compute intent reward before advancing
        forecast_rain_3d = sum(
            day["rain"]
            for day in self._sim.get_weather_forecast(self._sim.current_day + 1, n_days=3)
        )
        intent_reward = compute_step_reward(
            action_type=action_type,
            dvs=self._sim.dvs,
            sm=self._sim.sm,
            amount=amount,
            cost=cost,
            budget_remaining=budget_remaining,
            total_n=self._sim.total_n,
            total_water=self._sim.total_water,
            forecast_rain=forecast_rain_3d,
            root_zone_depth_cm=scenario["soil_params"].rooting_depth_mm / 10.0,
        )

        pre_sm = self._sim.sm
        pre_water_stress = self._sim._water_stress()
        pre_n_availability = self._sim.n_factor

        # Advance simulation
        step_days = scenario["step_days"]
        self._sim.advance(step_days, irrigation_cm=irrig_cm, n_kg_ha=n_kg)
        self._state.total_cost += cost
        self._sync_state()

        delta_reward = compute_delta_reward(
            action_type=action_type,
            pre_sm=pre_sm,
            post_sm=self._sim.sm,
            pre_water_stress=pre_water_stress,
            post_water_stress=self._sim._water_stress(),
            pre_n_availability=pre_n_availability,
            post_n_availability=self._sim.n_factor,
            cost=cost,
            budget_remaining=budget_remaining,
            total_cost=self._state.total_cost,
            budget=self._state.budget,
        )
        # Blend: 40% agronomic intent, 60% observed state change.
        # Validated against harvest_hesitation and drought_rescue probes.
        step_reward = 0.4 * intent_reward + 0.6 * delta_reward
        step_metadata = self._step_metadata(
            intent_reward=intent_reward,
            delta_reward=delta_reward,
            step_reward=step_reward,
        )

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
            grade, breakdown = self._rubric.score_episode(
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
                rubric_reward=grade,
                conflicts=conflicts,
                metadata={
                    **step_metadata,
                    "rubric_breakdown": breakdown,
                },
            )

        return self._build_observation(
            task,
            done=False,
            reward=step_reward,
            conflicts=conflicts,
            metadata=step_metadata,
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

    def _apply_start_state_overrides(self, scenario: dict[str, Any]) -> None:
        if self._sim is None:
            return

        start_at_dvs = scenario.get("start_at_dvs")
        if start_at_dvs is not None:
            while (
                self._sim.dvs < start_at_dvs
                and self._sim.current_day < scenario["max_duration"]
                and self._sim.dvs < 2.0
            ):
                self._sim.advance(1)

        if "override_sm" in scenario:
            field_capacity = scenario["soil_params"].field_capacity
            wilting_point = scenario["soil_params"].wilting_point
            self._sim.sm = max(wilting_point, min(field_capacity + 0.05, scenario["override_sm"]))

        if "override_n_factor" in scenario:
            self._sim.n_factor = max(0.3, min(1.0, scenario["override_n_factor"]))

        forced_forecast_rain = scenario.get("force_forecast_rain")
        if forced_forecast_rain:
            for offset, rain in enumerate(forced_forecast_rain, start=1):
                weather_index = self._sim.current_day + offset
                if weather_index < len(self._sim.weather):
                    self._sim.weather[weather_index]["rain"] = rain

    def _step_metadata(
        self,
        intent_reward: float | None = None,
        delta_reward: float | None = None,
        step_reward: float | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        probe_name = self._scenario.get("probe_name")
        if probe_name:
            metadata["probe_name"] = probe_name
            metadata["probe_notes"] = self._scenario.get("probe_notes", "")

        if intent_reward is not None or delta_reward is not None or step_reward is not None:
            metadata["reward_breakdown"] = {
                "intent_reward": round(intent_reward or 0.0, 4),
                "delta_reward": round(delta_reward or 0.0, 4),
                "step_reward": round(step_reward or 0.0, 4),
            }
        return metadata

    def _build_observation(
        self,
        task: dict,
        done: bool = False,
        reward: float | None = None,
        rubric_reward: float | None = None,
        conflicts: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CropObservation:
        sim = self._sim
        scenario = self._scenario

        if sim is None:
            return CropObservation(done=done, reward=reward, rubric_reward=rubric_reward)

        weather_today = sim.get_weather(sim.current_day)
        weather_forecast = sim.get_weather_forecast(
            sim.current_day + 1, n_days=5
        )
        extended_forecast = sim.get_weather_forecast(
            sim.current_day + 1, n_days=7
        )

        budget_remaining = self._state.budget - self._state.total_cost
        if sim.sm < 0.28:
            moisture_gap_to_target = round(0.28 - sim.sm, 3)
        elif sim.sm > 0.32:
            moisture_gap_to_target = round(0.32 - sim.sm, 3)
        else:
            moisture_gap_to_target = 0.0

        forecast_rain_3d = round(sum(day["rain"] for day in weather_forecast[:3]), 2)
        forecast_rain_7d = round(sum(day["rain"] for day in extended_forecast[:7]), 2)
        days_since_last_irrigation = (
            sim.current_day - self._state.last_irrigation_day
            if self._state.last_irrigation_day is not None
            else sim.current_day
        )
        days_since_last_fertilization = (
            sim.current_day - self._state.last_fertilization_day
            if self._state.last_fertilization_day is not None
            else sim.current_day
        )

        if sim.dvs < 0.20:
            dvs_distance_to_next_window = round(0.20 - sim.dvs, 3)
        elif 0.20 <= sim.dvs <= 0.40 or 0.50 <= sim.dvs <= 0.70:
            dvs_distance_to_next_window = 0.0
        elif sim.dvs < 0.50:
            dvs_distance_to_next_window = round(0.50 - sim.dvs, 3)
        else:
            dvs_distance_to_next_window = -1.0

        irrig_cost = scenario["irrigation_cost"]
        fert_cost = scenario["fertilizer_cost"]
        estimated_budget_to_finish = round(
            max(
                0.0,
                (forecast_rain_7d < 1.0) * irrig_cost * 4.0
                + (sim.n_factor < 0.6) * fert_cost * 15.0,
            ),
            2,
        )

        return CropObservation(
            done=done,
            reward=reward,
            rubric_reward=rubric_reward,
            metadata=metadata or {},
            task_id=task["id"],
            task_name=task["name"],
            instructions=task["instructions"],
            day=sim.current_day,
            days_remaining=max(0, scenario["max_duration"] - sim.current_day),
            crop_status=CropStatus(
                dvs=round(sim.dvs, 3),
                lai=round(sim.lai, 2),
                tagp=round(sim.tagp, 1),
                twso=round(sim.twso, 1),
                growth_stage=sim.growth_stage_name(),
            ),
            soil_status=SoilStatus(
                sm=round(sim.sm, 3),
                water_deficit=sim.sm < 0.22,
                water_stress=round(sim._water_stress(), 3),
                n_availability=round(sim.n_factor, 3),
                field_capacity=scenario["soil_params"].field_capacity,
                wilting_point=scenario["soil_params"].wilting_point,
            ),
            weather_today=WeatherDay(
                day=sim.current_day,
                tmax=weather_today["tmax"],
                tmin=weather_today["tmin"],
                rain=weather_today["rain"],
                radiation=weather_today["radiation"],
            ),
            weather_forecast=[
                WeatherDay(**day) for day in weather_forecast
            ],
            resources_used=ResourcesUsed(
                total_water_cm=round(sim.total_water, 2),
                total_n_kg_ha=round(sim.total_n, 2),
                total_cost=round(self._state.total_cost, 2),
                budget_remaining=round(budget_remaining, 2),
                irrigation_cost_per_cm=scenario["irrigation_cost"],
                fertilizer_cost_per_kg=scenario["fertilizer_cost"],
            ),
            season_summary={
                "crop_name": scenario["crop_name"],
                "location": scenario["location"],
                "target_yield": round(scenario["target_yield"], 1),
                "budget": scenario["budget"],
                "step_days": scenario["step_days"],
                "probe_name": scenario.get("probe_name"),
            },
            control_features=ControlFeatures(
                moisture_gap_to_target=moisture_gap_to_target,
                forecast_rain_3d=forecast_rain_3d,
                forecast_rain_7d=forecast_rain_7d,
                days_since_last_irrigation=days_since_last_irrigation,
                days_since_last_fertilization=days_since_last_fertilization,
                fertilizer_events_count=self._state.fertilizer_events_count,
                cumulative_n_applied=round(sim.total_n, 2),
                rooting_depth_cm=round(scenario["soil_params"].rooting_depth_mm / 10.0, 1),
                estimated_budget_to_finish=estimated_budget_to_finish,
                budget_remaining_ratio=round(
                    budget_remaining / max(self._state.budget, 1.0), 3
                ),
                dvs_distance_to_next_fertilizer_window=dvs_distance_to_next_window,
            ),
            conflicts=conflicts or [],
            advisory_text=generate_advisory(
                day=sim.current_day,
                days_remaining=max(0, scenario["max_duration"] - sim.current_day),
                step_days=scenario["step_days"],
                dvs=sim.dvs,
                lai=sim.lai,
                sm=sim.sm,
                field_capacity=scenario["soil_params"].field_capacity,
                wilting_point=scenario["soil_params"].wilting_point,
                water_stress=sim._water_stress(),
                n_availability=sim.n_factor,
                weather_today_tmax=weather_today["tmax"],
                forecast_rain_3d=forecast_rain_3d,
                forecast_rain_7d=forecast_rain_7d,
                total_water_cm=sim.total_water,
                total_n_kg_ha=sim.total_n,
                budget_remaining=budget_remaining,
                budget_total=scenario["budget"],
                location=scenario["location"],
            ),
        )
