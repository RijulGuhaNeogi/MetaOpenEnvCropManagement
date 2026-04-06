#!/usr/bin/env python3
"""
Inference script for the Precision Agriculture Crop Management OpenEnv environment.

Uses the competition-mandated environment variables:
  API_BASE_URL  – LLM provider base URL
  MODEL_NAME    – model identifier
  HF_TOKEN      – authentication token

Can run against a local server (default http://localhost:8000) or a
remote HuggingFace Space URL passed via ENV_URL.

Uses the WebSocket-based EnvClient for multi-step episodes.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

log = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()  # Auto-load .env file if present

import httpx
from openai import OpenAI

from client import CropEnvClient
from models import CropAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

SEED = 42
TASKS = [1, 2, 3]

# Safety cap to prevent runaway episodes (server has MAX_STEPS=60 but
# we add a client-side guard as well)
MAX_CLIENT_STEPS = 200

# LLM usage tracking for cost / quota awareness
llm_calls = 0
llm_fallbacks = 0
llm_consecutive_errors = 0   # Stops calling LLM after N consecutive failures
LLM_ERROR_THRESHOLD = 3      # Max consecutive errors before disabling LLM
llm_last_error = ""
llm_credit_exhausted = False

# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

llm_client: OpenAI | None = None

if MODEL_NAME and HF_TOKEN:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


SYSTEM_PROMPT = """\
You are a precision agriculture advisor. You manage a wheat crop for one growing
season. Each step is one week. You MUST return a JSON object with:
  {"action_type": "...", "amount": ...}

ACTIONS (6 available):

1. HARVEST — if DVS >= 1.8 (or growth_stage is "ripening"/"mature"), ALWAYS harvest.
   → {"action_type": "harvest", "amount": 0}

2. FERTILIZE — you MUST fertilize exactly twice per season:
    • First application:  aim near DVS 0.30, not immediately at window start → 18 kg N/ha
    • Second application: aim near DVS 0.60, not immediately at window start → 15 kg N/ha
   Skipping fertilization severely hurts your score (15% of grade).
    → {"action_type": "fertilize", "amount": 18}

3. IRRIGATE — if soil moisture < 0.22 (or sm_band="critical"/"low") AND no rain coming:
    • Apply deficit-based irrigation toward 0.30, capped at 5.0 cm.
    • If critically dry (sm < 0.18 or sm_band="critical"), irrigate regardless.
    → {"action_type": "irrigate", "amount": 2.0}

4. INSPECT_SOIL — costs $10 + 1 week. Reveals exact soil moisture, nitrogen, water stress.
   Use when sm_band/n_visual are ambiguous and you need precision for a decision.
    → {"action_type": "inspect_soil", "amount": 0}

5. INSPECT_CROP — costs $20 + 1 week. Reveals exact DVS, LAI, biomass, grain weight.
   Use when you need precise growth stage to time fertilization or harvest.
    → {"action_type": "inspect_crop", "amount": 0}

6. WAIT — only if none of the above apply.
   → {"action_type": "wait", "amount": 0}

NOTE: On harder tasks, some numeric readings are replaced by bands (e.g. "low",
"adequate") and weather by NL summaries. Use inspect actions strategically if
you need exact values — but each costs budget and a week of growing time.

SCORING (same formula for all tasks):
  0.35×yield + 0.20×water_eff + 0.18×cost_eff + 0.15×timing + 0.12×harvest

BUDGET: Always check budget_remaining before spending.
Return ONLY valid JSON, no explanation."""


def compress_observation(obs) -> str:
    """Build a compact text representation of the observation for the LLM."""
    cs = obs.crop_status
    ss = obs.soil_status
    wt = obs.weather_today
    ru = obs.resources_used
    sm = obs.season_summary
    cf = obs.control_features
    tier = getattr(obs, "observability_tier", 1)

    lines = [
        f"Task: {obs.task_name} (tier {tier}) | Day {obs.day}/{obs.day + obs.days_remaining}",
        f"Crop: {sm.get('crop_name', 'wheat')} at {sm.get('location', '?')}",
    ]

    # Growth info — use bands when DVS hidden
    if cs.dvs >= 0:
        lines.append(
            f"Growth: DVS={cs.dvs:.3f} stage={cs.growth_stage} "
            f"LAI={cs.lai:.2f} Yield={cs.twso:.0f} kg/ha"
        )
    else:
        lai_str = f"LAI={cs.lai:.2f}" if cs.lai >= 0 else f"lai_band={getattr(obs, 'lai_band', '?')}"
        tagp_str = f"Biomass={cs.tagp:.0f}" if cs.tagp >= 0 else ""
        twso_str = f"Yield={cs.twso:.0f} kg/ha" if cs.twso >= 0 else ""
        extras = " ".join(filter(None, [lai_str, tagp_str, twso_str]))
        lines.append(f"Growth: stage={cs.growth_stage} {extras}")

    # Soil info — use bands when SM hidden
    if ss.sm >= 0:
        lines.append(
            f"Soil: moisture={ss.sm:.3f} deficit={ss.water_deficit} "
            f"water_stress={ss.water_stress:.2f} n_avail={ss.n_availability:.2f}"
        )
    else:
        sm_band = getattr(obs, "sm_band", "?")
        n_visual = getattr(obs, "n_visual", "?")
        lines.append(f"Soil: moisture_band={sm_band} nitrogen={n_visual}")

    # Weather today (always numeric)
    lines.append(
        f"Weather today: tmax={wt.tmax}°C tmin={wt.tmin}°C "
        f"rain={wt.rain} cm rad={wt.radiation} MJ/m2"
    )

    # Forecast — numeric or NL summary depending on tier
    fc = obs.weather_forecast
    weather_summary = getattr(obs, "weather_summary", None)
    if fc:
        fc_lines = []
        for f in fc[:4]:
            fc_lines.append(
                f"  day{f.day}: {f.tmax}/{f.tmin}°C "
                f"rain={f.rain}cm"
            )
        lines.append("Forecast:\n" + "\n".join(fc_lines))
    elif weather_summary:
        lines.append(f"Weather forecast: {weather_summary}")

    lines.append(
        f"Resources: water={ru.total_water_cm:.1f}cm "
        f"N={ru.total_n_kg_ha:.1f}kg "
        f"cost=${ru.total_cost:.1f} "
        f"remaining=${ru.budget_remaining:.1f}"
    )

    # Control features — only show non-sentinel values
    if cf:
        cf_parts = []
        if cf.moisture_gap_to_target != 0.0 or tier == 1:
            cf_parts.append(f"moist_gap={cf.moisture_gap_to_target}")
        if cf.forecast_rain_3d >= 0:
            cf_parts.append(f"rain3d={cf.forecast_rain_3d}cm")
        if cf.forecast_rain_7d >= 0:
            cf_parts.append(f"rain7d={cf.forecast_rain_7d}cm")
        cf_parts.append(f"budget_ratio={cf.budget_remaining_ratio}")
        cf_parts.append(f"root_depth={cf.rooting_depth_cm}cm")
        if cf.dvs_distance_to_next_fertilizer_window >= 0:
            cf_parts.append(f"next_fert_window={cf.dvs_distance_to_next_fertilizer_window}")
        if cf_parts:
            lines.append("Control: " + " ".join(cf_parts))

    lines.append(f"Target yield: {sm.get('target_yield', 0):.0f} kg/ha")

    if obs.conflicts:
        lines.append(f"Conflicts: {'; '.join(obs.conflicts)}")

    # Inspection reports
    soil_report = getattr(obs, "soil_report", None)
    crop_report = getattr(obs, "crop_report", None)
    if soil_report:
        lines.append(f"SOIL REPORT: {soil_report}")
    if crop_report:
        lines.append(f"CROP REPORT: {crop_report}")

    if getattr(obs, "advisory_text", None):
        lines.append(f"Advisory: {obs.advisory_text}")

    return "\n".join(lines)


def call_llm(obs) -> dict:
    """Ask the LLM for a crop management action.

    Returns a dict with 'action_type' and optionally 'amount', or an
    empty dict on any failure (network error, malformed response, etc.).
    Falls back to greedy heuristic when this returns {}.
    """
    global llm_calls, llm_fallbacks, llm_consecutive_errors
    global llm_last_error, llm_credit_exhausted

    if llm_client is None:
        return {}

    # Early-stop: if we've hit N consecutive errors (e.g. 402 credit exhaustion),
    # skip LLM entirely for the rest of the run to avoid wasting time
    if llm_consecutive_errors >= LLM_ERROR_THRESHOLD:
        return {}

    prompt = compress_observation(obs)

    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=128,
        )
        llm_calls += 1
        llm_consecutive_errors = 0  # Reset on success
        llm_last_error = ""
        llm_credit_exhausted = False
    except Exception as e:
        error_text = str(e)
        llm_consecutive_errors += 1
        llm_fallbacks += 1
        llm_last_error = error_text
        llm_credit_exhausted = (
            "Error code: 402" in error_text
            or "depleted your monthly included credits" in error_text
        )
        if llm_consecutive_errors >= LLM_ERROR_THRESHOLD:
            log.warning(
                "LLM disabled after %d consecutive errors — "
                "using greedy heuristic for remaining steps",
                LLM_ERROR_THRESHOLD,
            )
        else:
            log.warning("LLM error: %r — falling back to greedy", e)
        return {}

    text = response.choices[0].message.content or "{}"
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    # Try strict JSON parse first, then fall back to regex extraction
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Attempt to extract JSON object from mixed prose
        match = re.search(r'\{[^}]+\}', text)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                return {}
        else:
            return {}

    if "action_type" not in result:
        return {}
    return result


# ---------------------------------------------------------------------------
# Greedy heuristic (no LLM needed)
# ---------------------------------------------------------------------------

# Agronomic thresholds tuned for the three task scenarios
from server.constants import (
    FERT_HEURISTIC_OFFSET,
    FERT_TARGET_KG_1,
    FERT_TARGET_KG_2,
    FERT_WINDOW_1,
    FERT_WINDOW_2,
    GROWTH_STAGE_DVS_MAP,
    HARVEST_DVS_LOW,
    SM_BAND_MIDPOINT,
    SM_TARGET_HIGH,
    SM_TARGET_LOW,
    SM_WATER_DEFICIT,
)

SM_IRRIGATE_THRESHOLD = SM_WATER_DEFICIT  # Soil moisture below this = dry
SM_CRITICAL_THRESHOLD = 0.18    # Soil moisture below this = critically dry
RAIN_THRESHOLD_CM = 0.3         # Rain forecast above this = skip irrigation
HARVEST_DVS = HARVEST_DVS_LOW   # Minimum DVS for harvest
TARGET_SM = (SM_TARGET_HIGH + SM_TARGET_LOW) / 2  # Soil moisture target for irrigation dosing
MAX_IRRIGATION_CM = 5.0         # Do not irrigate more than this in one step
FERT_STAGE1_DVS = (FERT_WINDOW_1[0] + FERT_HEURISTIC_OFFSET, FERT_WINDOW_1[1])
FERT_STAGE2_DVS = (FERT_WINDOW_2[0] + FERT_HEURISTIC_OFFSET, FERT_WINDOW_2[1])
FERT_STAGE1_KG = FERT_TARGET_KG_1  # kg N/ha for first application
FERT_STAGE2_KG = FERT_TARGET_KG_2  # kg N/ha for second application


def greedy_action(obs, fert_stages_done: set) -> dict:
    """Rule-based crop management heuristic.

    Simple but effective baseline that makes agronomically sound decisions:
      1. Harvest when DVS >= 1.8 (crop is mature)
      2. Irrigate when soil is dry and no rain is forecast within 2 days
      3. Fertilize at key growth stages (once per stage)
      4. Otherwise wait (conservation — no free reward for waiting)

    On tier 2/3, DVS and SM may be hidden (-1.0). The greedy heuristic falls
    back to midpoint estimates from growth_stage and sm_band — intentionally
    imprecise, which degrades fert window 2 timing and over-irrigates.
    """
    cs = obs.crop_status
    ss = obs.soil_status
    ru = obs.resources_used
    fc = obs.weather_forecast
    cf = obs.control_features

    # Resolve DVS: use exact if available, else midpoint from growth stage
    dvs = cs.dvs
    if dvs < 0:
        dvs = GROWTH_STAGE_DVS_MAP.get(cs.growth_stage, 1.0)

    # Resolve SM: use exact if available, else midpoint from sm_band
    sm = ss.sm
    if sm < 0:
        sm = SM_BAND_MIDPOINT.get(obs.sm_band or "adequate", 0.285)

    budget_remaining = ru.budget_remaining
    irrig_cost = ru.irrigation_cost_per_cm
    fert_cost = ru.fertilizer_cost_per_kg
    rooting_depth_cm = cf.rooting_depth_cm

    # Check if rain is coming in next 2 days (unavailable on tier 2/3)
    rain_coming = False
    if fc:
        for f in fc[:2]:
            if f.rain > RAIN_THRESHOLD_CM:
                rain_coming = True
                break

    # 1. Harvest at maturity
    if dvs >= HARVEST_DVS:
        return {"action_type": "harvest", "amount": 0.0}

    # 2. Irrigate dry soil using a deficit-based dose
    desired_irrigation = max(
        0.5,
        min(MAX_IRRIGATION_CM, (TARGET_SM - sm) * rooting_depth_cm),
    )
    affordable_irrigation = budget_remaining / max(irrig_cost, 0.1)

    if sm < SM_CRITICAL_THRESHOLD:
        irrigation_amount = min(max(desired_irrigation, 3.0), affordable_irrigation)
        if irrigation_amount >= 0.5:
            return {"action_type": "irrigate", "amount": round(irrigation_amount, 2)}

    if sm < SM_IRRIGATE_THRESHOLD and not rain_coming:
        irrigation_amount = min(desired_irrigation, affordable_irrigation)
        if irrigation_amount >= 0.5:
            return {"action_type": "irrigate", "amount": round(irrigation_amount, 2)}

    # 3. Fertilize at key growth stages (once per stage)
    if FERT_STAGE1_DVS[0] <= dvs <= FERT_STAGE1_DVS[1] and "stage1" not in fert_stages_done and budget_remaining > fert_cost * FERT_STAGE1_KG:
        fert_stages_done.add("stage1")
        return {"action_type": "fertilize", "amount": FERT_STAGE1_KG}
    if FERT_STAGE2_DVS[0] <= dvs <= FERT_STAGE2_DVS[1] and "stage2" not in fert_stages_done and budget_remaining > fert_cost * FERT_STAGE2_KG:
        fert_stages_done.add("stage2")
        return {"action_type": "fertilize", "amount": FERT_STAGE2_KG}

    # 4. Wait (conservation)
    return {"action_type": "wait", "amount": 0.0}


def _record_transition(
    records: list[dict],
    task_id: int,
    step_num: int,
    policy_name: str,
    observation,
    action: CropAction,
    result,
) -> None:
    """Append a transition record for offline RL or imitation learning."""
    next_observation = result.observation
    records.append(
        {
            "task_id": task_id,
            "step": step_num,
            "policy": policy_name,
            "observation": observation.model_dump(),
            "action": action.model_dump(),
            "reward": result.reward,
            "done": result.done,
            "next_observation": next_observation.model_dump(),
            "metadata": next_observation.metadata,
        }
    )


def _write_trajectory_jsonl(path: str, records: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Crop Management inference.")
    parser.add_argument(
        "--trajectory-output",
        default=os.getenv("TRAJECTORY_OUTPUT", ""),
        help="Optional path to write JSONL transitions from the executed policy.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main loop using WebSocket client for multi-step episodes
# ---------------------------------------------------------------------------

def run():
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    trajectory_records: list[dict] = []

    # Quick HTTP health check (using context manager for clean socket handling)
    with httpx.Client(base_url=ENV_URL, timeout=30.0) as http:
        health = http.get("/health")
        health.raise_for_status()
        log.info("Connected to %s: %s", ENV_URL, health.json())

        tasks_resp = http.get("/tasks")
        tasks_resp.raise_for_status()
        available_tasks = {t["id"]: t for t in tasks_resp.json()["tasks"]}  
        log.info("Available tasks: %s", [t['name'] for t in available_tasks.values()])


    all_scores: dict[int, float] = {}

    for task_id in TASKS:
        if task_id not in available_tasks:
            log.info("Task %d not available, skipping", task_id)
            continue

        task = available_tasks[task_id]
        log.info("--- Task %d: %s (%s) ---", task_id, task['name'], task['difficulty'])

        sync_client = CropEnvClient(base_url=ENV_URL).sync()
        with sync_client:
            result = sync_client.reset(seed=SEED, task_id=task_id)
            obs = result.observation

            step_num = 0
            last_reward = 0.0
            fert_stages_done: set = set()

            while not result.done and step_num < MAX_CLIENT_STEPS:
                previous_obs = obs

                # Choose action: try LLM first, fall back to heuristic
                if llm_client is not None:
                    action_dict = call_llm(obs)
                    policy_name = "llm"
                    if not action_dict or "action_type" not in action_dict:
                        action_dict = greedy_action(obs, fert_stages_done)
                        policy_name = "greedy_fallback"
                else:
                    action_dict = greedy_action(obs, fert_stages_done)
                    policy_name = "greedy"

                # Ensure amount is present
                if "amount" not in action_dict:
                    action_dict["amount"] = 0.0

                action = CropAction(**action_dict)
                result = sync_client.step(action)
                obs = result.observation

                if args.trajectory_output:
                    _record_transition(
                        trajectory_records,
                        task_id=task_id,
                        step_num=step_num + 1,
                        policy_name=policy_name,
                        observation=previous_obs,
                        action=action,
                        result=result,
                    )

                step_num += 1
                rew_str = f"{result.reward:.4f}" if result.reward is not None else "n/a"
                act_str = f"{action.action_type}"
                if action.amount > 0:
                    act_str += f"({action.amount:.1f})"
                log.info(
                    "Step %d: %s | DVS=%.3f SM=%.3f Yield=%.0f | reward=%s done=%s",
                    step_num, act_str,
                    obs.crop_status.dvs, obs.soil_status.sm, obs.crop_status.twso,
                    rew_str, result.done,
                )

                if result.reward is not None:
                    last_reward = result.reward

        all_scores[task_id] = last_reward
        log.info("Task %d final score: %.4f", task_id, last_reward)

    # Summary (=== RESULTS === format required by hackathon evaluation)
    print("\n=== RESULTS ===")
    for tid in TASKS:
        if tid in all_scores:
            print(f"Task {tid}: score={all_scores[tid]:.4f}")
    if all_scores:
        overall = sum(all_scores.values()) / len(all_scores)
        print(f"Overall: {overall:.4f}")
    if args.trajectory_output:
        _write_trajectory_jsonl(args.trajectory_output, trajectory_records)
        log.info(
            "Trajectory export: wrote %d transitions to %s",
            len(trajectory_records), args.trajectory_output,
        )
    if llm_calls or llm_fallbacks:
        log.info("LLM stats: %d calls, %d fallbacks", llm_calls, llm_fallbacks)
        if llm_consecutive_errors >= LLM_ERROR_THRESHOLD:
            if llm_credit_exhausted:
                log.warning(
                    "LLM credits exhausted after %d successful calls. "
                    "Switched to greedy heuristic for remaining steps. "
                    "Regenerate your HF token or wait for monthly credit reset.",
                    llm_calls,
                )
            else:
                log.warning(
                    "LLM disabled after repeated errors and switched to greedy "
                    "heuristic for remaining steps. Last error: %s",
                    llm_last_error,
                )


if __name__ == "__main__":
    run()
