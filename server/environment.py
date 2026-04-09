"""Precision Agriculture Crop Management — OpenEnv Environment.

This is the core environment class that implements the OpenEnv interface.
An LLM agent manages a wheat growing season by making weekly decisions:
irrigation, fertilization, harvest, or wait.

Episode flow:
  1. reset(seed, task_id) → creates a CropSimulator with seeded weather
  2. step(action) → validates action, advances sim by 7 days, returns obs
  3. Episode ends when: agent harvests, 2 steps after DVS≥2.0
     (auto-harvest with shattering penalty), season duration
     exceeded, or MAX_STEPS reached
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
from server.advisory import generate_advisory, generate_soil_report, generate_crop_report, weather_to_nl
from server.constants import (
    MAX_STEPS, REWARD_DELTA_WEIGHT, REWARD_INTENT_WEIGHT,
    STEP_REWARD_MIN, STEP_REWARD_MAX, STEP_REWARD_SCALE,
    INSPECT_SOIL_COST, INSPECT_CROP_COST,
    SM_BAND_CRITICAL, SM_BAND_LOW, SM_BAND_ADEQUATE,
    N_VISUAL_VERY_LOW, N_VISUAL_LOW, N_VISUAL_MODERATE, N_VISUAL_ADEQUATE,
    LAI_LOW, LAI_MODERATE,
    SLOW_RELEASE_COST_MULTIPLIER,
)
from server.crop_sim import CropSimulator
from server.reward import (
    compute_delta_reward,
    compute_step_reward,
    compute_trajectory_reward,
)
from server.rubric import CropManagementRubric
from server.scenarios import generate_probe_scenario, generate_scenario
from server.tasks import TASKS, get_task_definition

# Lazy import to avoid circular dependency — oracle_action is a pure function
# that reads an observation and returns an action dict.
_oracle_action = None

def _get_oracle_action():
    global _oracle_action
    if _oracle_action is None:
        from agent.inference import oracle_action
        _oracle_action = oracle_action
    return _oracle_action


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
        self._oracle_metadata_state: dict[str, Any] = {}
        self._oracle_policy_state: dict[str, Any] = {}

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
        self._oracle_metadata_state = {}
        self._oracle_policy_state = {}

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
            explicit_harvest=False,
            last_irrigation_day=None,
            last_fertilization_day=None,
            fertilizer_events_count=0,
            last_soil_report=None,
            last_crop_report=None,
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

        action_type = action.action_type.lower().strip()
        amount = action.amount
        if amount < 0:
            conflicts.append(
                f"Amount must be >= 0 (got {amount}). Clamping to 0."
            )
            amount = 0.0

        # Validate action type
        valid_actions = ("irrigate", "fertilize", "fertilize_slow", "harvest", "wait", "inspect_soil", "inspect_crop")
        if action_type not in valid_actions:
            conflicts.append(
                f"Invalid action_type: '{action.action_type}'. "
                f"Must be one of: {', '.join(valid_actions)}."
            )
            action_type = "wait"
            amount = 0.0

        # Handle inspect actions — free sub-action: no sim advance, no step
        # increment.  Charges budget, generates report (persisted), returns
        # observation so the agent can take a real action next call.
        if action_type in ("inspect_soil", "inspect_crop"):
            inspect_cost = INSPECT_SOIL_COST if action_type == "inspect_soil" else INSPECT_CROP_COST
            budget_remaining_now = self._state.budget - self._state.total_cost

            if inspect_cost > budget_remaining_now:
                conflicts.append(
                    f"Cannot afford {action_type} (${inspect_cost}) with "
                    f"${budget_remaining_now:.1f} remaining. Treating as wait."
                )
            else:
                # Valid inspect — charge budget, persist report, return immediately
                self._state.total_cost += inspect_cost

                # Record the inspect action
                action_record = {
                    "step": self._state.step_count,  # same step number
                    "action_type": action_type,
                    "amount": 0.0,
                    "dvs": round(self._sim.dvs, 3),
                    "sm": round(self._sim.sm, 3),
                    "cost": round(inspect_cost, 2),
                }
                self._state.actions_taken.append(action_record)

                # Generate and persist report
                if action_type == "inspect_soil":
                    self._state.last_soil_report = generate_soil_report(
                        sm=self._sim.sm,
                        water_deficit=self._sim.sm < 0.22,
                        field_capacity=scenario["soil_params"].field_capacity,
                        wilting_point=scenario["soil_params"].wilting_point,
                        n_availability=self._sim.n_factor,
                        water_stress=self._sim._water_stress(),
                    )
                else:
                    self._state.last_crop_report = generate_crop_report(
                        dvs=self._sim.dvs,
                        lai=self._sim.lai,
                        tagp=self._sim.tagp,
                        twso=self._sim.twso,
                        growth_stage=self._sim.growth_stage_name(),
                    )

                # Compute intent reward for the inspect
                forecast_rain_3d = sum(
                    day["rain"]
                    for day in self._sim.get_weather_forecast(self._sim.current_day + 1, n_days=3)
                )
                inspect_reward = compute_step_reward(
                    action_type=action_type,
                    dvs=self._sim.dvs,
                    sm=self._sim.sm,
                    amount=0.0,
                    cost=inspect_cost,
                    budget_remaining=budget_remaining_now,
                    total_n=self._sim.total_n,
                    total_water=self._sim.total_water,
                    forecast_rain=forecast_rain_3d,
                    root_zone_depth_cm=scenario["soil_params"].rooting_depth_mm / 10.0,
                    water_stress=self._sim._water_stress(),
                    n_availability=self._sim.n_factor,
                    n_recov=scenario["crop_params"].N_RECOV,
                    fert_events_count=self._state.fertilizer_events_count,
                    task_tier=task.get("observability_tier", 1),
                )
                inspect_reward, inspect_channels = inspect_reward

                return self._build_observation(
                    task, done=False, reward=inspect_reward,
                    conflicts=conflicts,
                    metadata=self._step_metadata(
                        intent_reward=inspect_reward,
                        delta_reward=0.0,
                        step_reward=inspect_reward,
                        intent_channels=inspect_channels,
                    ),
                    inspect_performed=action_type,
                )

            # If we reach here, inspect was rejected — treat as normal wait
            action_type = "wait"
            amount = 0.0

        # --- From here, action_type is irrigate / fertilize / harvest / wait ---
        # (inspect actions returned early above)
        inspect_performed: str | None = None  # Not an inspect at this point

        # Increment step only for real actions (inspects don't count)
        self._state.step_count += 1

        # Compute cost
        cost = 0.0
        irrig_cm = 0.0
        n_kg = 0.0
        slow_release = False

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

        elif action_type == "fertilize_slow":
            amount = min(amount, 50.0)
            if amount <= 0:
                conflicts.append("Fertilizer amount must be > 0. Treating as wait.")
                action_type = "wait"
                amount = 0.0
            else:
                cost = amount * scenario["fertilizer_cost"] * SLOW_RELEASE_COST_MULTIPLIER
                n_kg = amount
                slow_release = True

        # Budget check
        budget_remaining = self._state.budget - self._state.total_cost
        if cost > budget_remaining and action_type in ("irrigate", "fertilize", "fertilize_slow"):
            conflicts.append(
                f"Over budget: action costs ${cost:.1f} but only "
                f"${budget_remaining:.1f} remaining. Treating as wait."
            )
            action_type = "wait"
            amount = 0.0
            cost = 0.0
            irrig_cm = 0.0
            n_kg = 0.0
            slow_release = False

        # Record action
        record_action_type = inspect_performed if inspect_performed else action_type
        current_day = self._sim.current_day
        if action_type == "irrigate" and irrig_cm > 0.0:
            self._state.last_irrigation_day = current_day
        elif action_type in ("fertilize", "fertilize_slow") and n_kg > 0.0:
            self._state.last_fertilization_day = current_day
            self._state.fertilizer_events_count += 1

        action_record = {
            "step": self._state.step_count,
            "action_type": record_action_type,
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
            self._state.explicit_harvest = True
            self._state.total_cost += cost

            # Compute final grade
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
                explicit_harvest=True,
            )
            trajectory_reward = compute_trajectory_reward(grade)

            # Dense harvest timing signal — blend into terminal reward so the
            # agent gets immediate feedback on whether harvest timing was good.
            harvest_step_signal = compute_step_reward(
                action_type="harvest",
                dvs=self._sim.dvs,
                sm=self._sim.sm,
                amount=0.0,
                cost=0.0,
                budget_remaining=self._state.budget - self._state.total_cost,
                total_n=self._sim.total_n,
                total_water=self._sim.total_water,
            )
            harvest_step_signal, _ = harvest_step_signal
            # Map step signal from [-0.30, +0.25] to [0, 1] for blending
            normalized_harvest = (harvest_step_signal + 0.30) / 0.55
            normalized_harvest = max(0.0, min(1.0, normalized_harvest))
            final_reward = 0.7 * trajectory_reward + 0.3 * normalized_harvest

            self._sync_state()
            return self._build_observation(
                task, done=True, reward=final_reward,
                rubric_reward=grade,
                conflicts=conflicts,
                metadata={
                    **self._step_metadata(),
                    "rubric_breakdown": breakdown,
                },
                inspect_performed=inspect_performed,
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
            water_stress=self._sim._water_stress(),
            n_availability=self._sim.n_factor,
            n_recov=scenario["crop_params"].N_RECOV,
            fert_events_count=self._state.fertilizer_events_count,
        )
        # compute_step_reward now returns (float, dict)
        intent_reward, intent_channels = intent_reward

        pre_sm = self._sim.sm
        pre_water_stress = self._sim._water_stress()
        pre_n_availability = self._sim.n_factor
        pre_twso = self._sim.twso

        # Advance simulation
        step_days = scenario["step_days"]
        self._sim.advance(step_days, irrigation_cm=irrig_cm, n_kg_ha=n_kg,
                          slow_release=slow_release)
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
            pre_twso=pre_twso,
            post_twso=self._sim.twso,
            target_yield=scenario["target_yield"],
        )
        # compute_delta_reward now returns (float, dict)
        delta_reward, delta_channels = delta_reward
        # Blend: agronomic intent + observed state change.
        # Validated against harvest_hesitation and drought_rescue probes.
        step_reward = STEP_REWARD_SCALE * (
            REWARD_INTENT_WEIGHT * intent_reward + REWARD_DELTA_WEIGHT * delta_reward
        )
        step_reward = max(STEP_REWARD_MIN, min(STEP_REWARD_MAX, step_reward))

        # Dose quality feedback for fertilize actions
        dose_hint: str | None = None
        if action_type in ("fertilize", "fertilize_slow") and n_kg > 0:
            n_recov = scenario["crop_params"].N_RECOV
            ideal = min(50.0, max(0.0, 1.0 - pre_n_availability) / max(n_recov, 0.001))
            if ideal > 1.0:
                ratio = n_kg / ideal
                if ratio > 1.2:
                    dose_hint = f"overdosed — applied {n_kg:.0f}kg vs ~{ideal:.0f}kg ideal"
                elif ratio >= 0.8:
                    dose_hint = f"good dose — close to the ~{ideal:.0f}kg ideal"
                else:
                    dose_hint = f"underdosed — applied {n_kg:.0f}kg vs ~{ideal:.0f}kg ideal"

        step_metadata = self._step_metadata(
            intent_reward=intent_reward,
            delta_reward=delta_reward,
            step_reward=step_reward,
            intent_channels=intent_channels,
            delta_channels=delta_channels,
        )

        # Check termination conditions
        is_done = False
        final_reward = step_reward

        # Track when maturity first reached
        if self._sim.dvs >= 2.0 and self._state.maturity_reached_step is None:
            self._state.maturity_reached_step = self._state.step_count

        if (self._state.maturity_reached_step is not None
                and self._state.step_count - self._state.maturity_reached_step >= 2):
            # Auto-harvest: agent had 2 steps to harvest after maturity
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
                explicit_harvest=self._state.explicit_harvest,
            )
            # Unified terminal reward: same blend as explicit harvest,
            # but auto-termination gets a halved harvest-timing component
            # so the agent is always taught to harvest explicitly.
            harvest_step_signal = compute_step_reward(
                action_type="harvest",
                dvs=self._state.harvest_dvs,
                sm=self._sim.sm,
                amount=0.0,
                cost=0.0,
                budget_remaining=self._state.budget - self._state.total_cost,
                total_n=self._sim.total_n,
                total_water=self._sim.total_water,
            )
            harvest_step_signal, _ = harvest_step_signal
            normalized_harvest = (harvest_step_signal + 0.30) / 0.55
            normalized_harvest = max(0.0, min(1.0, normalized_harvest))
            auto_penalty = 0.5 if not self._state.explicit_harvest else 1.0
            final_reward = (
                0.7 * compute_trajectory_reward(grade)
                + 0.3 * normalized_harvest * auto_penalty
            )
            return self._build_observation(
                task, done=True, reward=final_reward,
                rubric_reward=grade,
                conflicts=conflicts,
                metadata={
                    **step_metadata,
                    "rubric_breakdown": breakdown,
                },
                inspect_performed=inspect_performed,
                dose_hint=dose_hint,
            )

        # Compute oracle reference action (for metadata — not visible to LLM)
        oracle_dict = self._compute_oracle_action(task, self._oracle_metadata_state)
        if oracle_dict:
            step_metadata["oracle_action"] = oracle_dict

        return self._build_observation(
            task,
            done=False,
            reward=step_reward,
            conflicts=conflicts,
            metadata=step_metadata,
            inspect_performed=inspect_performed,
            dose_hint=dose_hint,
        )

    @property
    def state(self) -> CropState:
        return self._state.model_copy(deep=True)

    def oracle_reference_action(self) -> dict[str, Any]:
        """Return the perfect-information oracle action for the current step.

        Unlike a policy operating on the public observation surface, this
        method evaluates the oracle on an internal tier-1 snapshot built from
        the simulator's exact state. It is the correct upper-bound reference
        for local baselines, tests, and diagnostics.
        """
        if self._sim is None or self._state.current_task_id is None:
            raise RuntimeError("Environment has not been reset.")

        task = get_task_definition(self._state.current_task_id)
        action_dict = self._compute_oracle_action(task, self._oracle_policy_state)
        if action_dict is None:
            raise RuntimeError("Failed to compute oracle reference action.")
        return action_dict

    # ------------------------------------------------------------------
    # Oracle reference (for metadata — not shown to LLM)
    # ------------------------------------------------------------------

    def _compute_oracle_action(self, task: dict, oracle_state: dict[str, Any]) -> dict[str, Any] | None:
        """Call oracle_action on a tier-1 observation snapshot.

        Returns the oracle's recommended action dict, or None on error.
        The result is stored in obs.metadata for offline analysis; the
        LLM never sees it in compress_observation.
        """
        try:
            oracle_fn = _get_oracle_action()
            tier1_task = dict(task)
            tier1_task["observability_tier"] = 1
            tier1_task["hidden_fields"] = []
            tier1_obs = self._build_observation(tier1_task, done=False)
            action_dict = oracle_fn(tier1_obs, oracle_state)
            return {
                "action_type": action_dict.get("action_type", "wait"),
                "amount": round(action_dict.get("amount", 0.0), 1),
            }
        except Exception:
            return None

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
        self._state.slow_release_pool = round(self._sim.slow_release_pool, 2)
        self._state.total_n_leached = round(self._sim.total_n_leached, 4)

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
        intent_channels: dict[str, float] | None = None,
        delta_channels: dict[str, float] | None = None,
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
            if intent_channels:
                metadata["reward_breakdown"]["intent_channels"] = intent_channels
            if delta_channels:
                metadata["reward_breakdown"]["delta_channels"] = delta_channels
        return metadata

    def _build_observation(
        self,
        task: dict,
        done: bool = False,
        reward: float | None = None,
        rubric_reward: float | None = None,
        conflicts: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inspect_performed: str | None = None,
        dose_hint: str | None = None,
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

        obs = CropObservation(
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
                n_leached=round(sim.total_n_leached, 4) if task.get("observability_tier", 1) <= 2 else None,
                slow_release_pool=round(sim.slow_release_pool, 2) if task.get("observability_tier", 1) <= 2 else None,
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
                slow_release_cost_per_kg=round(
                    scenario["fertilizer_cost"] * SLOW_RELEASE_COST_MULTIPLIER, 2
                ),
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
                fert_count=self._state.fertilizer_events_count,
                has_crop_report=self._state.last_crop_report is not None,
            ),
            dose_hint=dose_hint,
        )

        # ---------------------------------------------------------------
        # Partial observability coarsening (Phase E)
        # ---------------------------------------------------------------
        tier = task.get("observability_tier", 1)
        hidden_fields = set(task.get("hidden_fields", []))
        obs.observability_tier = tier

        if tier >= 2:
            # --- Compute bands from exact values (before hiding) ---
            exact_sm = sim.sm
            exact_n = sim.n_factor
            exact_lai = sim.lai

            if exact_sm < SM_BAND_CRITICAL:
                obs.sm_band = "critical"
            elif exact_sm < SM_BAND_LOW:
                obs.sm_band = "low"
            elif exact_sm < SM_BAND_ADEQUATE:
                obs.sm_band = "adequate"
            else:
                obs.sm_band = "high"

            if exact_n < N_VISUAL_VERY_LOW:
                obs.n_visual = "very_low"
            elif exact_n < N_VISUAL_LOW:
                obs.n_visual = "low"
            elif exact_n < N_VISUAL_MODERATE:
                obs.n_visual = "moderate"
            elif exact_n < N_VISUAL_ADEQUATE:
                obs.n_visual = "adequate"
            else:
                obs.n_visual = "surplus"

            if exact_lai < LAI_LOW:
                obs.lai_band = "sparse"
            elif exact_lai < LAI_MODERATE:
                obs.lai_band = "moderate"
            else:
                obs.lai_band = "dense"

            # --- Hide numeric fields ---
            if "dvs" in hidden_fields:
                obs.crop_status.dvs = -1.0
                obs.dvs_hidden = True
            if "sm" in hidden_fields:
                obs.soil_status.sm = -1.0
                obs.sm_hidden = True
            if "n_availability" in hidden_fields:
                obs.soil_status.n_availability = -1.0
            if "water_stress" in hidden_fields:
                obs.soil_status.water_stress = -1.0

            # --- Sentinel leaking control features ---
            obs.control_features.moisture_gap_to_target = 0.0
            obs.control_features.forecast_rain_3d = -1.0
            obs.control_features.forecast_rain_7d = -1.0
            # Preserve fert window signal (0.0 = inside, >0 = distance away)
            # so the LLM still gets in_fert_window=YES on hidden tiers
            # (dvs_distance_to_next_fertilizer_window is already set correctly)
            obs.control_features.estimated_budget_to_finish = -1.0

            # --- Replace weather forecast with NL summary ---
            obs.weather_summary = weather_to_nl(weather_forecast, tier)
            obs.weather_forecast = []

            # --- Tier-aware advisory (no exact numbers) ---
            obs.advisory_text = generate_advisory(
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
                tier=tier,
                fert_count=self._state.fertilizer_events_count,
                has_crop_report=self._state.last_crop_report is not None,
            )

        if tier >= 3:
            # --- Additionally hide crop growth internals ---
            if "lai" in hidden_fields:
                obs.crop_status.lai = -1.0
            if "tagp" in hidden_fields:
                obs.crop_status.tagp = -1.0
            if "twso" in hidden_fields:
                obs.crop_status.twso = -1.0

        # --- Inspection reports (persisted from state + fresh) ---
        # Fresh inspect reports are written to state before calling
        # _build_observation, so we always populate from state.
        if inspect_performed == "inspect_soil":
            # Fresh inspect — report was just persisted to state above
            obs.soil_report = self._state.last_soil_report
        elif self._state.last_soil_report:
            obs.soil_report = self._state.last_soil_report

        if inspect_performed == "inspect_crop":
            obs.crop_report = self._state.last_crop_report
        elif self._state.last_crop_report:
            obs.crop_report = self._state.last_crop_report

        return obs
