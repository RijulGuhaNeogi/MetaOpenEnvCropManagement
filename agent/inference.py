#!/usr/bin/env python3
"""
Inference script for the Precision Agriculture Crop Management OpenEnv environment.

Uses the competition-mandated environment variables:
  API_BASE_URL  – LLM provider base URL
  MODEL_NAME    – model identifier
  API_KEY       – authentication token (evaluator injects API_KEY; HF_TOKEN as fallback)

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

load_dotenv(override=False)  # Auto-load .env file if present; never override evaluator-injected vars

import httpx
from openai import OpenAI

from client import CropEnvClient
from models import CropAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

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

if API_KEY:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


SYSTEM_PROMPT = """\
Wheat crop advisor. 1 step=1 week. Reply ONLY with JSON: {"action_type":"...","amount":...}

RULE ZERO: If fert_count=2/2, NEVER fertilize again — skip to step 3.

CHECK IN ORDER EACH STEP:

1. HARVEST — NEVER harvest below DVS 1.80, even if growth_stage says "ripening". DVS>=1.80 → harvest. If DVS hidden: inspect_crop to learn exact DVS. Only harvest when DVS>=1.80 confirmed or advisory says "harvest window". MUST harvest explicitly; auto-harvest@DVS2.0=20% credit.

2. FERTILIZE — ONLY when fert_count<2 AND in_fert_window is OPTIMAL or LATE.
 If in_fert_window=EARLY → WAIT. The crop is in the window but not yet at target DVS. Waiting improves timing score.
 If in_fert_window=OPTIMAL → fertilize NOW. Best timing.
 If in_fert_window=LATE → fertilize NOW before window closes.
 If in_fert_window=YES (hidden tiers) → check advisory for timing hints: "Early in window"→WAIT, "Near optimal"→fertilize, "Late in window"→fertilize.
 Strategy: you have exactly 2 fert slots across 2 windows. Apply a larger dose in window 1, a smaller top-up in window 2.
 Dose by nitrogen status (5 bands):
  Tier1: n_avail≥0.9→SKIP(wait), 0.8-0.9→15kg, 0.65-0.8→30kg, 0.5-0.65→45kg, <0.5→50kg
  Tier2/3: "surplus"→SKIP(wait), "adequate"→15kg, "moderate"→30kg, "low"→45kg, "very_low"→50kg
 After fertilizing, check the Dose feedback line — it tells you if you dosed correctly.
 Yield is 35% of your score — prioritize crop nutrition. Stay within budget but don't under-fertilize.

3. IRRIGATE — Tier1: moisture<0.28 & rain3d<0.3→irrigate. Tier2/3: moisture_band="low"/"critical" & no rain forecast. Dose: 3cm default, or sm_gap_to_optimal×90 if known.

4. INSPECT (tier2/3 only) — inspect does NOT cost a week — you get results immediately and act on them next step. inspect_soil($10): reveals EXACT nitrogen level. inspect_crop($20): reveals EXACT DVS. Results persist in all future observations as SOIL REPORT / CROP REPORT — use them to calibrate subsequent decisions. Budget is the only limit on inspects.

5. WAIT — if nothing above applies. Waiting when crop is healthy is correct.

Positive reward=good decision. Negative reward=wrong action, bad timing, or wasteful spend. JSON only."""


def compress_observation(obs, prev_action: str | None = None, prev_reward: float | None = None) -> str:
    """Build a compact text representation of the observation for the LLM."""
    cs = obs.crop_status
    ss = obs.soil_status
    wt = obs.weather_today
    ru = obs.resources_used
    sm = obs.season_summary
    cf = obs.control_features
    tier = getattr(obs, "observability_tier", 1)

    lines = []

    # Reward feedback from previous action — with specific diagnostic
    if prev_action is not None and prev_reward is not None:
        sign = "+" if prev_reward >= 0 else ""
        if prev_reward > 0.10:
            hint = "strong positive — good decision"
        elif prev_reward > 0.02:
            hint = "mildly positive"
        elif prev_reward > -0.02:
            hint = "neutral — no harm, no benefit"
        elif prev_reward > -0.06:
            hint = "negative — wasteful or mistimed action"
        else:
            hint = "strongly negative — wrong action, bad timing, or budget waste"
        lines.append(f"Last action: {prev_action} → reward: {sign}{prev_reward:.3f} ({hint})")

    lines.extend([
        f"Task: {obs.task_name} (tier {tier}) | Day {obs.day}/{obs.day + obs.days_remaining}",
        f"Crop: {sm.get('crop_name', 'wheat')} at {sm.get('location', '?')}",
    ])

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

    fert_count = cf.fertilizer_events_count if cf else 0
    lines.append(
        f"Resources: water={ru.total_water_cm:.1f}cm "
        f"N={ru.total_n_kg_ha:.1f}kg "
        f"fert_count={fert_count}/2 "
        f"cost=${ru.total_cost:.1f} "
        f"remaining=${ru.budget_remaining:.1f}"
    )

    # Control features — only show non-sentinel values
    if cf:
        cf_parts = []
        if cf.moisture_gap_to_target != 0.0 or tier == 1:
            cf_parts.append(f"sm_gap_to_optimal={cf.moisture_gap_to_target}")
        if cf.forecast_rain_3d >= 0:
            cf_parts.append(f"rain3d={cf.forecast_rain_3d}cm")
        if cf.forecast_rain_7d >= 0:
            cf_parts.append(f"rain7d={cf.forecast_rain_7d}cm")
        cf_parts.append(f"budget_ratio={cf.budget_remaining_ratio}")
        cf_parts.append(f"root_depth={cf.rooting_depth_cm}cm")
        # Convert fert window distance to EARLY/OPTIMAL/LATE signal
        fwd = cf.dvs_distance_to_next_fertilizer_window
        if fwd == 0.0:
            # Inside a fert window — determine position relative to target
            dvs_val = cs.dvs if cs.dvs >= 0 else -1.0
            if dvs_val >= 0:
                if 0.20 <= dvs_val <= 0.40:
                    if dvs_val < 0.25:
                        cf_parts.append("in_fert_window=EARLY(target=0.30,wait_for_optimal)")
                    elif dvs_val <= 0.35:
                        cf_parts.append("in_fert_window=OPTIMAL(target=0.30)")
                    else:
                        cf_parts.append("in_fert_window=LATE(target=0.30,act_now)")
                elif 0.50 <= dvs_val <= 0.70:
                    if dvs_val < 0.55:
                        cf_parts.append("in_fert_window=EARLY(target=0.60,wait_for_optimal)")
                    elif dvs_val <= 0.65:
                        cf_parts.append("in_fert_window=OPTIMAL(target=0.60)")
                    else:
                        cf_parts.append("in_fert_window=LATE(target=0.60,act_now)")
                else:
                    cf_parts.append("in_fert_window=YES")
            else:
                # DVS hidden (T2/T3) — derive timing from advisory text
                adv = getattr(obs, "advisory_text", "") or ""
                if "Early in window" in adv:
                    cf_parts.append("in_fert_window=EARLY")
                elif "Near optimal" in adv:
                    cf_parts.append("in_fert_window=OPTIMAL")
                elif "Late in window" in adv:
                    cf_parts.append("in_fert_window=LATE")
                else:
                    cf_parts.append("in_fert_window=YES")
        elif fwd > 0:
            cf_parts.append(f"in_fert_window=NO({fwd:.2f}_DVS_away)")
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

    # Dose quality feedback
    dose_hint = getattr(obs, "dose_hint", None)
    if dose_hint:
        lines.append(f"Dose feedback: {dose_hint}")

    if getattr(obs, "advisory_text", None):
        lines.append(f"Advisory: {obs.advisory_text}")

    return "\n".join(lines)


def call_llm(obs, prev_action: str | None = None, prev_reward: float | None = None) -> dict:
    """Ask the LLM for a crop management action.

    Returns a dict with 'action_type' and optionally 'amount', or an
    empty dict on any failure (network error, malformed response, etc.).
    Falls back to oracle baseline when this returns {}.
    """
    global llm_calls, llm_fallbacks, llm_consecutive_errors
    global llm_last_error, llm_credit_exhausted

    if llm_client is None:
        return {}

    # Early-stop: if we've hit N consecutive errors (e.g. 402 credit exhaustion),
    # skip LLM entirely for the rest of the run to avoid wasting time
    if llm_consecutive_errors >= LLM_ERROR_THRESHOLD:
        return {}

    prompt = compress_observation(obs, prev_action=prev_action, prev_reward=prev_reward)

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
                "using oracle baseline for remaining steps",
                LLM_ERROR_THRESHOLD,
            )
        else:
            log.warning("LLM error: %r — falling back to oracle baseline", e)
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
    # Normalize case — some models return "WAIT" / "Fertilize" etc.
    result["action_type"] = result["action_type"].lower().strip()
    return result


# ---------------------------------------------------------------------------
# Perfect-information oracle baseline (sets the scoring ceiling)
# ---------------------------------------------------------------------------

# The oracle knows every model parameter, all weather, the full simulation
# internals.  Observability tiers only hinder the *LLM* agent; the oracle
# tracks DVS and N-factor from first principles regardless of what is hidden.

from server.constants import (
    FERT_TARGET_DVS_1,
    FERT_TARGET_DVS_2,
    FERT_WINDOW_1,
    FERT_WINDOW_2,
    HARVEST_DVS_LOW,
    SM_BAND_MIDPOINT,
    SM_TARGET_HIGH,
    SM_TARGET_LOW,
)
from server.crop_params import CROP_LIBRARY, WOFOSTCropParams

# Location key → crop params (oracle knows the full model)
_LOC_CROP_KEY = {
    "Netherlands": "wheat_nl",
    "Iowa, USA": "wheat_iowa",
    "Punjab, India": "wheat_punjab",
}

HARVEST_DVS = 1.90
SM_IRRIGATE_THRESHOLD = SM_TARGET_LOW   # 0.28
SM_CRITICAL_THRESHOLD = 0.18
RAIN_THRESHOLD_CM = 0.3
TARGET_SM = (SM_TARGET_HIGH + SM_TARGET_LOW) / 2  # 0.30
MAX_IRRIGATION_CM = 5.0


def _get_crop_params(obs) -> WOFOSTCropParams:
    """Resolve crop parameters from the observation's location."""
    loc_name = obs.season_summary.get("location", "Netherlands")
    key = _LOC_CROP_KEY.get(loc_name, "wheat_nl")
    return CROP_LIBRARY[key]


def oracle_action(obs, oracle_state: dict) -> dict:
    """Perfect-information oracle baseline.

    This baseline has *complete knowledge* of the crop simulation model:
    TSUM values, N recovery/depletion rates, shattering parameters, soil
    hydrology, and all future weather.  It computes DVS via thermal-time
    accumulation and tracks nitrogen factor from first principles, so it
    is unaffected by observability-tier masking.

    It represents the theoretical scoring ceiling: an RL/LLM agent must
    discover these dynamics from observation signals and step rewards.
    """
    cs = obs.crop_status
    ss = obs.soil_status
    ru = obs.resources_used
    fc = obs.weather_forecast           # [] on tier 2-3
    cf = obs.control_features

    # ── Initialise persistent state on first call ──────────────────────
    if "dvs" not in oracle_state:
        cp = _get_crop_params(obs)
        oracle_state.update({
            "dvs": 0.0,
            "n_factor": cp.N_FACTOR_INIT,
            "prev_day": obs.day,
            "prev_forecast": [],        # WeatherDay objects from last step
            "fert_done": set(),
            "pending_fert_kg": 0.0,     # fert decided last step, applied this
            "cp": cp,
        })

    cp: WOFOSTCropParams = oracle_state["cp"]

    # ── Advance internal DVS & N-factor for elapsed days ──────────────
    days_elapsed = obs.day - oracle_state["prev_day"]
    if days_elapsed > 0:
        # Apply pending fertiliser (sim applies it before daily loop)
        if oracle_state["pending_fert_kg"] > 0:
            oracle_state["n_factor"] = min(
                1.0,
                oracle_state["n_factor"]
                + oracle_state["pending_fert_kg"] * cp.N_RECOV,
            )
            oracle_state["pending_fert_kg"] = 0.0

        prev_fc = oracle_state["prev_forecast"]
        for d in range(days_elapsed):
            # Best available temperature for this day
            if d < len(prev_fc):
                tavg = (prev_fc[d].tmax + prev_fc[d].tmin) / 2.0
            else:
                tavg = (obs.weather_today.tmax + obs.weather_today.tmin) / 2.0
            eff_temp = max(0.0, tavg - cp.TBASE)

            # DVS phenology
            if oracle_state["dvs"] < 1.0:
                oracle_state["dvs"] += eff_temp / cp.TSUM1
            else:
                oracle_state["dvs"] += eff_temp / cp.TSUM2
            oracle_state["dvs"] = min(2.0, oracle_state["dvs"])

            # N depletion
            loss = cp.N_LOSS_PRE if oracle_state["dvs"] < 1.0 else cp.N_LOSS_POST
            oracle_state["n_factor"] = max(cp.N_FACTOR_FLOOR, oracle_state["n_factor"] - loss)

    # Calibrate from exact values when visible (tier 1)
    if cs.dvs >= 0:
        oracle_state["dvs"] = cs.dvs
    if ss.n_availability >= 0:
        oracle_state["n_factor"] = ss.n_availability

    # Snapshot for this step
    dvs = oracle_state["dvs"]
    n_factor = oracle_state["n_factor"]
    fert_done = oracle_state["fert_done"]

    # Store state for next step's thermal-time computation
    oracle_state["prev_day"] = obs.day
    oracle_state["prev_forecast"] = list(fc)      # may be [] on tier 2-3

    # ── Predict DVS after next 7-day advance (for timing optimisation) ─
    def _predict_next_dvs() -> float:
        """Estimate DVS after the next 7-day step using forecast or proxy."""
        fut_dvs = dvs
        for d in range(7):
            if d < len(fc):
                tavg = (fc[d].tmax + fc[d].tmin) / 2.0
            else:
                tavg = (obs.weather_today.tmax + obs.weather_today.tmin) / 2.0
            eff = max(0.0, tavg - cp.TBASE)
            if fut_dvs < 1.0:
                fut_dvs += eff / cp.TSUM1
            else:
                fut_dvs += eff / cp.TSUM2
        return min(2.0, fut_dvs)

    next_dvs = _predict_next_dvs()

    # ── Resolve SM ─────────────────────────────────────────────────────
    sm = ss.sm
    if sm < 0:
        sm = SM_BAND_MIDPOINT.get(obs.sm_band or "adequate", 0.285)

    budget_remaining = ru.budget_remaining
    irrig_cost = ru.irrigation_cost_per_cm
    fert_cost = ru.fertilizer_cost_per_kg
    rooting_depth = cf.rooting_depth_cm

    rain_coming = any(f.rain > RAIN_THRESHOLD_CM for f in (fc or [])[:2])

    # ── 1. HARVEST at peak yield ───────────────────────────────────────
    if dvs >= HARVEST_DVS:
        return {"action_type": "harvest", "amount": 0.0}
    # If next step would trigger auto-harvest (DVS ≥ 2.0), harvest now
    if next_dvs >= 2.0 and dvs >= HARVEST_DVS_LOW:
        return {"action_type": "harvest", "amount": 0.0}

    # ── 2. IRRIGATE proactively ────────────────────────────────────────
    desired_irrig = max(
        0.5,
        min(MAX_IRRIGATION_CM, (TARGET_SM - sm) * rooting_depth),
    )
    affordable_irrig = budget_remaining / max(irrig_cost, 0.1)

    if sm < SM_CRITICAL_THRESHOLD:
        amt = min(max(desired_irrig, 3.0), affordable_irrig)
        if amt >= 0.5:
            return {"action_type": "irrigate", "amount": round(amt, 2)}

    if sm < SM_IRRIGATE_THRESHOLD and not rain_coming:
        amt = min(desired_irrig, affordable_irrig)
        if amt >= 0.5:
            return {"action_type": "irrigate", "amount": round(amt, 2)}

    # ── 3. FERTILISE at optimal timing ─────────────────────────────────
    # The oracle picks the step whose DVS is closest to the grader's
    # target (0.30 / 0.60), maximising timing_quality.  It computes the
    # exact N amount to fill n_factor to 1.0, minimising cost.
    #
    # N-factor tracking uses obs.day (exact) + known depletion rates,
    # which is precise regardless of tier (unlike DVS which requires
    # temperature data).  This avoids the divergence that would occur
    # if we relied on weather-proxy thermal-time for N depletion.

    def _should_fert_now(window, target_dvs) -> bool:
        """True when this step is the best moment inside *window*."""
        if not (window[0] <= dvs <= window[1]):
            return False
        dist_now = abs(dvs - target_dvs)
        dist_next = abs(next_dvs - target_dvs)
        return (dvs >= target_dvs
                or next_dvs > window[1]
                or dist_now <= dist_next)

    def _fert_kg() -> float:
        """N required to fill to 1.0, computed from day-exact tracking.
        Respects the environment's 50 kg/ha per-step cap."""
        # Use exact n_factor if visible (tier 1); otherwise track via days
        if ss.n_availability >= 0:
            nf = ss.n_availability
        else:
            # Reconstruct n_factor from sowing day + known history
            nf = cp.N_FACTOR_INIT
            prev_day = 0
            for fert_day, fert_kg in oracle_state.get("fert_history", []):
                # Deplete from prev_day to fert_day (all pre-anthesis at these DVS)
                days_gap = fert_day - prev_day
                nf -= days_gap * cp.N_LOSS_PRE
                nf = max(cp.N_FACTOR_FLOOR, nf)
                # Apply fert (respecting the env's 50 kg cap)
                applied = min(fert_kg, 50.0)
                nf = min(1.0, nf + applied * cp.N_RECOV)
                prev_day = fert_day
            # Deplete from last event to now
            days_gap = obs.day - prev_day
            nf -= days_gap * cp.N_LOSS_PRE
            nf = max(cp.N_FACTOR_FLOOR, nf)
        deficit = max(0.0, 1.0 - nf)
        # Cap the request to 50 kg (env will clip anyway)
        return min(50.0, round(deficit / cp.N_RECOV, 1))

    # Ensure fert_history exists
    if "fert_history" not in oracle_state:
        oracle_state["fert_history"] = []

    # Stage 1
    if "stage1" not in fert_done and _should_fert_now(FERT_WINDOW_1, FERT_TARGET_DVS_1):
        kg = _fert_kg()
        if kg > 0 and budget_remaining > fert_cost * kg:
            fert_done.add("stage1")
            oracle_state["pending_fert_kg"] = kg
            oracle_state["fert_history"].append((obs.day, kg))  # capped inside _fert_kg
            return {"action_type": "fertilize", "amount": kg}

    # Stage 2
    if "stage2" not in fert_done and _should_fert_now(FERT_WINDOW_2, FERT_TARGET_DVS_2):
        kg = _fert_kg()
        if kg > 0 and budget_remaining > fert_cost * kg:
            fert_done.add("stage2")
            oracle_state["pending_fert_kg"] = kg
            oracle_state["fert_history"].append((obs.day, kg))  # capped inside _fert_kg
            return {"action_type": "fertilize", "amount": kg}

    # ── 4. WAIT ────────────────────────────────────────────────────────
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
            oracle_state: dict = {}
            prev_action_str: str | None = None
            prev_step_reward: float | None = None

            while not result.done and step_num < MAX_CLIENT_STEPS:
                previous_obs = obs

                # Choose action: try LLM first, fall back to heuristic
                if llm_client is not None:
                    action_dict = call_llm(obs, prev_action=prev_action_str, prev_reward=prev_step_reward)
                    policy_name = "llm"
                    if not action_dict or "action_type" not in action_dict:
                        action_dict = oracle_action(obs, oracle_state)
                        policy_name = "oracle_fallback"
                else:
                    action_dict = oracle_action(obs, oracle_state)
                    policy_name = "oracle"

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

                # Track previous action/reward for LLM feedback
                prev_action_str = act_str
                prev_step_reward = result.reward

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
                    "Switched to oracle baseline for remaining steps. "
                    "Regenerate your HF token or wait for monthly credit reset.",
                    llm_calls,
                )
            else:
                log.warning(
                    "LLM disabled after repeated errors and switched to oracle "
                    "baseline for remaining steps. Last error: %s",
                    llm_last_error,
                )


if __name__ == "__main__":
    run()
