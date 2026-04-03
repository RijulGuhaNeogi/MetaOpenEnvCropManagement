## Key Findings That Shape the PCSE Migration Plan

### PCSE Integration Risks (from deep dive)

| Concern | Risk | Detail |
|---------|------|--------|
| **Step-by-step action injection** | **HIGH** | PCSE wants pre-scheduled agromanagement. Our env does `advance(7, irrigation_cm=X, n_kg_ha=Y)` — PCSE has no clean equivalent. Requires a hybrid adapter that wraps PCSE growth with manual water/N injection. |
| **Nitrogen modeling (n_factor)** | **MEDIUM-HIGH** | WOFOST-NPK module exists but is poorly documented and unstable. Our simple `n_factor` (injected linearly, depletes over time) would need an approximation layer. |
| **Weather from dicts** | **LOW** | Custom `WeatherDataProvider` from our existing generated dicts is straightforward. No network needed. |
| **DVS/LAI/TAGP/TWSO/SM access** | **LOW** | PCSE exposes these directly via output variables. |
| **Determinism** | **LOW** | PCSE is fully deterministic — same inputs, same outputs. |
| **Docker packaging** | **LOW** | Pure Python, ~50MB extra. Self-contained if crop/soil params are bundled. |
| **No network at runtime** | **OK** | As long as we use a custom weather provider and bundle crop/soil data. Hackathon rules explicitly forbid external API calls. |

### Hackathon Constraints That Affect PCSE

| Constraint | Source | Impact on PCSE |
|-----------|--------|---------------|
| **vcpu=2, 8GB RAM** | Preparation docs | PCSE is lightweight — OK |
| **inference < 20 min** | Preparation docs | PCSE per-season sim ~0.5s — OK |
| **No external API calls at runtime** | README | Must bundle all PCSE data, use custom weather provider |
| **python:3.11-slim Docker** | Dockerfile | PCSE supports 3.11 — OK |
| **Deterministic bit-exact reproducibility** | Grading protocol | PCSE is deterministic — OK if adapter doesn't introduce float drift |
| **33 smoke tests must pass** | Current test suite | Calibration drift is the main risk |

### What Our Current Simulator Does That PCSE Would Need to Match

1. **Simple advance(days, irrigation_cm, n_kg_ha)** — PCSE doesn't have this; needs hybrid adapter
2. **n_factor as a 0-1 multiplier** — PCSE has nitrogen state variables but not this exact abstraction
3. **Weather forecast with deterministic noise** — Custom; must be preserved outside PCSE
4. **Partition tables for grain allocation** — PCSE does this internally (better)
5. **Heat stress** — PCSE models this internally (better)
6. **Water stress as a simple ratio** — PCSE has more complex water balance; adapter must expose simplified version

### Recommended PCSE Approach (Hybrid Adapter)

The **hybrid adapter** is the only viable path:
- Use PCSE for phenology (DVS progression) and biomass accumulation
- Manually inject irrigation/fertilizer effects via state manipulation between PCSE steps
- Derive `n_factor`, `_water_stress()` from PCSE internal state or keep current approximations
- Keep existing weather generation, wrap in custom `WeatherDataProvider`
- Keep existing grader, rewards, tasks, environment loop almost unchanged

### Parameters to Potentially Drop or Simplify

- **n_factor**: Consider keeping the current simple linear model rather than fighting WOFOST-NPK. The grader only uses `timing_quality` (distance to DVS 0.3/0.6) for fertilizer — it doesn't directly grade nitrogen efficiency. The reward's `n_gain` in delta_reward depends on `n_factor`, but this could use a proxy.
- **Heat stress**: PCSE handles this internally — better to let PCSE do it than keep our simplified version.

Now I'm ready to write the updated FUTURE_SCOPE_PCSE.md. I'm requesting write access.   

Analyzing your answers...

I don't have file editing tools available in this mode. Let me provide you the complete rewritten file content so you can apply it.

Here is the complete updated FUTURE_SCOPE_PCSE.md:

```markdown
# Future Scope: PCSE Migration Plan

This document is a handoff artifact for a future migration from the current custom WOFOST-inspired simulator to PCSE/WOFOST.

It is written to support a later implementation pass by Claude Opus or another strong coding LLM.

## Current State (as of April 2026)

The current project is a working precision agriculture OpenEnv environment with:
- deterministic seeded scenarios for 3 locations (Netherlands, Iowa, Punjab)
- 3 tasks with unified grader weights across all difficulty levels
- universal target yield calibration (max potential across all 3 locations)
- dense per-step rewards split into intent (40%) and delta (60%) components, validated against probe scenarios
- late-harvest reward slope retuned to -0.20/DVS with floor -0.25 (matches grader signal direction)
- harvest intent reward branch in `compute_step_reward` documented as dead code in the environment path (present for external callers only)
- stable greedy baseline: Task1=0.8442, Task2=0.8155, Task3=0.7046
- 33 passing smoke tests including edge-case coverage for grader boundaries and reward behavior
- 5 internal probe scenarios for RL diagnostics (over_irrigation_trap, late_fertilizer_temptation, budget_starvation, harvest_hesitation, drought_rescue)
- optional LLM-driven inference with HF Router and automatic greedy fallback
- control_features in observations (moisture_gap, forecast_rain, budget_ratio, dvs_window_distance, etc.)

The current simulator is custom and lives in `server/crop_sim.py`.

### DVS Windows (Current Calibration Targets)

These are the DVS-based windows that PCSE migration must match or justify deviation from:
- Fertilizer reward windows: DVS 0.20-0.40 (peak at 0.30) and DVS 0.50-0.70 (peak at 0.60)
- Grader timing target: proximity to DVS 0.30 and 0.60, linear decay over distance 0.5
- Harvest reward optimal: DVS 1.8-2.05 (+0.20 reward)
- Harvest reward late penalty: DVS > 2.05, slope -0.20/DVS, floor -0.25
- Grader harvest timing optimal: DVS 1.8-2.05 (score 1.0), floor 0.5 for DVS > 2.30
- Auto-harvest termination: DVS >= 2.0

### Current Reward Structure

- Intent reward: agronomic correctness before state transition (irrigation fit, fertilizer timing, harvest window)
- Delta reward: observed stress relief / waste after state transition
- Blend: `step_reward = 0.4 * intent_reward + 0.6 * delta_reward` (validated, documented)
- Terminal: `compute_trajectory_reward(grade_episode(...))` replaces step reward on harvest/end

## Goal of a Future PCSE Migration

Replace the crop simulation internals with PCSE while preserving:
- OpenEnv interface and task loop
- current action space: `irrigate`, `fertilize`, `harvest`, `wait`
- current observation schema including control_features
- deterministic behavior across seeds (bit-exact reproducibility required by hackathon)
- current grading philosophy and reward structure (grader is frozen)
- current inference output contract (`=== RESULTS ===`)
- current baseline score ordering: Easy >= Medium >= Hard

## Why Consider PCSE

### Benefits
- scientific credibility: PCSE is the reference WOFOST implementation
- better judge appeal: "uses PCSE/WOFOST" is stronger than "WOFOST-inspired"
- richer agronomic dynamics: phenology, water balance, biomass partitioning, heat stress
- stronger real-world utility narrative for the hackathon rubric (30% of score)

### Costs / Risks
- setup complexity: crop/site/soil/agromanagement configuration is more involved
- action injection: PCSE is designed for pre-scheduled agromanagement, not step-by-step agent control (see Risk Assessment below)
- nitrogen modeling: WOFOST-NPK module is immature and poorly documented
- calibration drift: DVS timing, yield levels, and stress behavior will shift relative to current logic
- debugging cost: more opaque than the current pure-Python simulator (~200 lines)

## PCSE Risk Assessment

### Critical: Step-by-step action injection — RISK: HIGH

PCSE is optimized for pre-scheduled agromanagement calendars. Our environment loop does:
```
agent decides → env calls sim.advance(7 days, irrigation_cm=X, n_kg_ha=Y) → observe
```
PCSE has no clean equivalent to `advance(days, irrigation_cm, n_kg_ha)`. It expects management events to be declared upfront or injected via state-events between simulation days.

**Mitigation:** Hybrid adapter (see Migration Strategy below). Run PCSE day-by-day, inject water/N effects through state manipulation between days. This works but is fragile and may introduce subtle behavioral differences.

### Medium-High: Nitrogen factor (n_factor) — RISK: MEDIUM-HIGH

Our current simulator uses a simple `n_factor` (0.0-1.0) that:
- Increases linearly by `n_kg_ha * 0.008` when fertilizer is applied
- Depletes at 0.0003/day pre-anthesis, 0.0015/day post-anthesis
- Floors at 0.3
- Directly multiplies biomass growth

WOFOST-NPK exists in PCSE but is poorly documented and exposes nitrogen state through different abstractions. Our grader does NOT directly grade nitrogen efficiency — it only grades `timing_quality` (proximity of fertilize actions to DVS 0.3/0.6). The reward's `delta_reward` uses `n_factor` change, but this could use a proxy.

**Recommendation:** Keep a simplified n_factor proxy derived from PCSE's nitrogen state, or continue using the current linear model alongside PCSE's growth engine. Do not depend on WOFOST-NPK stability for grading correctness.

### Low: Weather from Python dicts — RISK: LOW

PCSE supports custom `WeatherDataProvider` implementations. Our weather is generated as deterministic Python dicts in `server/scenarios.py`. Wrapping these in a PCSE-compatible provider is straightforward and requires no network access.

### Low: State access (DVS, LAI, TAGP, TWSO, SM) — RISK: LOW

PCSE exposes these directly as output variables. DVS, LAI, TAGP, TWSO, SM are all standard WOFOST outputs.

### Low: Determinism — RISK: LOW

PCSE is fully deterministic given the same inputs. No hidden random seeds or stochastic processes.

### Low: Docker / Dependencies — RISK: LOW

PCSE is pure Python, ~50MB install. No C extensions required. Works on python:3.11-slim. Crop/soil parameter data can be bundled (no network access needed after install).

## Hackathon Constraints That Affect PCSE

| Constraint | Source | Impact |
|-----------|--------|--------|
| vcpu=2, 8GB RAM | Preparation docs | PCSE is lightweight — no issue |
| Inference < 20 min total | Preparation docs | PCSE sim ~0.5s per season — no issue |
| No external API calls at runtime | README / evaluation rules | Must use custom WeatherDataProvider and bundle all crop/soil data |
| python:3.11-slim Docker base | Dockerfile | PCSE supports Python 3.11 |
| Deterministic bit-exact reproducibility | Grading protocol | PCSE is deterministic; adapter must not introduce float drift |
| Health check every 30s, 5s timeout | Dockerfile | PCSE init must be fast (bundle data, don't download at startup) |
| Docker image size ~5GB soft limit | HF Spaces | PCSE adds ~50-200MB — fits comfortably |
| No dependency blacklist | Hackathon rules | PCSE is allowed as long as it's pip-installable |

## Recommendation Threshold

Proceed with PCSE only if there are at least 3 to 5 focused days available for:
- integration (hybrid adapter construction)
- calibration (DVS timing, yield levels, stress behavior)
- testing (33 tests must pass, difficulty ordering must hold)
- Docker verification (build, health check, inference run)
- documentation refresh

If time is tighter than that, keep the current simulator. It is stable, deterministic, already calibrated, and lower risk.

## Current File Map

These are the files most relevant to the migration:
- `server/crop_sim.py` — current simulator (CropSimulator, CropParams, SoilParams, CROP_LIBRARY, SOIL_LIBRARY, PARTITION_TABLES, compute_potential_yield)
- `server/environment.py` — OpenEnv environment loop (CropEnvironment, MAX_STEPS=60, 7-day step)
- `server/scenarios.py` — deterministic scenario and weather generation (3 locations, 5 probe scenarios, universal target yield)
- `server/grader.py` — final scoring logic (FROZEN — 5 metrics, unified weights, deterministic)
- `server/reward.py` — dense step rewards (intent + delta, late-harvest slope -0.20/DVS, harvest branch is dead code in env path)
- `server/tasks.py` — task descriptions and instructions (harvest window documented as [1.8, 2.05])
- `models.py` — Pydantic action/observation/state models
- `inference.py` — greedy + LLM inference client with fallback
- `training_adapter.py` — discrete RL action adapter (8 actions)
- `server/Dockerfile` — container runtime (python:3.11-slim, port 7860)
- `requirements.txt` — dependencies (no PCSE currently)
- `openenv.yaml` — metadata and inference env vars
- `tests/test_smoke.py` — 33 passing tests (smoke + RL-focused + grader edge cases)

## Required Simulator Compatibility Contract

A PCSE-backed adapter should preserve the interface expected by `server/environment.py`.

The environment currently expects a simulator object that provides:
- constructor inputs:
  - `crop_params` (CropParams dataclass)
  - `soil_params` (SoilParams dataclass)
  - `weather_data` (list[dict] with keys: day, tmax, tmin, rain, radiation)
  - `partition_table` (list[tuple[float, float]] — DVS to grain fraction)
- state-like attributes:
  - `current_day` (int)
  - `dvs` (float, 0.0-2.0)
  - `lai` (float)
  - `tagp` (float, kg/ha — total above-ground production)
  - `twso` (float, kg/ha — grain yield)
  - `sm` (float — volumetric soil moisture)
  - `n_factor` (float, 0.0-1.0 — nitrogen availability multiplier)
  - `total_water` (float, cm — cumulative irrigation)
  - `total_n` (float, kg/ha — cumulative nitrogen applied)
- methods:
  - `advance(days, irrigation_cm=0.0, n_kg_ha=0.0)` — advance simulation, apply inputs
  - `get_weather(day)` — return weather dict for a given day
  - `get_weather_forecast(start_day, n_days=5)` — return forecast with deterministic noise
  - `growth_stage_name()` — return phenological stage label
  - `_water_stress()` — return 0.1-1.0 stress factor (called directly by environment for delta reward)

If PCSE does not expose one of these fields directly, the adapter should derive or approximate it and document the mapping.

### Environment API Surface on Simulator (Exact Calls)

These are every attribute and method the environment accesses on the simulator object:

**During reset:** `crop_params`, `soil_params`, `weather_data`, `partition_table` (constructor)
**During step (pre-advance):** `dvs`, `sm`, `current_day`, `n_factor`, `total_water`, `total_n`, `get_weather_forecast()`, `_water_stress()`
**During step (advance):** `advance(step_days, irrigation_cm, n_kg_ha)`
**During step (post-advance):** `dvs`, `sm`, `n_factor`, `_water_stress()`, `current_day`, `lai`, `tagp`, `twso`, `total_water`, `total_n`
**During grading:** `twso` (yield), `total_water`, `total_n`, `dvs` (harvest DVS)
**During observation building:** `dvs`, `lai`, `tagp`, `twso`, `sm`, `current_day`, `n_factor`, `_water_stress()`, `growth_stage_name()`, `get_weather()`, `get_weather_forecast()`

## Migration Strategy

### Phase 1: Read and Freeze Current Behavior

Before changing anything:
- read all files listed in the file map above
- record the current behavioral contract (documented in this file)
- run `python -m pytest tests/ -v` and save baseline results
- run `python -m benchmark_sweep.py` across seeds 42-46 for all 3 tasks and record scores

Key DVS windows to verify post-migration:
- Typical DVS at day 100 for each location (phenology speed)
- Typical yield at maturity for each location (growth capacity)
- Water stress behavior at SM levels 0.15, 0.25, 0.35 for each soil type

### Phase 2: Build a Hybrid PCSE Adapter

The hybrid adapter is the recommended approach because PCSE does not support arbitrary per-step action injection.

Architecture:
- keep `server/environment.py` intact — it continues to call `sim.advance(7, irrigation_cm=X, n_kg_ha=Y)`
- replace `CropSimulator` in `server/crop_sim.py` (or add `server/pcse_adapter.py`)
- the adapter internally:
  1. Runs PCSE day-by-day using a custom engine step loop
  2. Before each batch of days, injects irrigation as soil moisture addition
  3. Before each batch of days, injects nitrogen as n_factor boost (using current linear model or PCSE proxy)
  4. After each batch, reads DVS, LAI, TAGP, TWSO, SM from PCSE state
  5. Computes `_water_stress()` from PCSE soil moisture state using the current formula
  6. Computes `n_factor` as a simplified proxy (current linear model or mapped from PCSE NPK state)

The adapter should expose the same interface listed in the compatibility contract above.

**Critical design decision:** For nitrogen, prefer the current simple linear model (`n_factor += n_kg_ha * 0.008`, depletes at phenology-dependent rate, floor 0.3) unless WOFOST-NPK proves stable and well-behaved. The grader does not directly grade nitrogen efficiency.

### Phase 3: Deterministic Weather Provider

Current weather is generated deterministically as Python dicts in `server/scenarios.py`.

Approach:
- wrap the current generated weather in a custom PCSE-compatible `WeatherDataProvider`
- this preserves current seeded scenario generation and requires zero external data
- do NOT use `NASAPowerWeatherDataProvider` or any network-based provider
- the custom provider should accept the existing `list[dict]` format and expose it in whatever interface PCSE's engine expects

### Phase 4: Crop / Soil / Site / Agromanagement Setup

PCSE migration will need:
- crop parameters for winter wheat (bundle from PCSE's included data or replicate current CropParams)
- soil parameters for 3 representative soil types matching current SOIL_LIBRARY (clay_loam, sandy_loam, silt_loam)
- site data (latitude/longitude for radiation calculations if PCSE requires it)
- agromanagement skeleton (sowing date, but no pre-scheduled irrigation/fertilizer — those are injected per-step)

Important:
- keep only wheat for now
- do not broaden crop scope during the migration
- keep 3 locations and current budget difficulty scaling
- bundle all parameter files in the Docker image — no downloads at runtime

### Phase 5: Action Mapping

Current actions are:
- `irrigate(amount_cm)` — capped at 10 cm/step
- `fertilize(amount_kg_n_ha)` — capped at 50 kg/step
- `harvest` — terminal, triggers grading
- `wait` — no-op, 7 days advance

The adapter must simulate these effects within PCSE's framework:
- irrigation: add water to soil compartment (directly modify SM or use PCSE water balance input)
- fertilization: boost n_factor (simplified model) or inject N via PCSE-NPK if stable
- harvest: environment handles this — adapter just needs to have accumulated yield in `twso`
- wait: advance without inputs

PCSE should not silently take over episode termination policy. Auto-harvest at DVS >= 2.0 and max-duration termination remain controlled by `environment.py`.

### Phase 6: Observation Mapping

Preserve the current observation schema:
- `crop_status`: dvs, lai, tagp, twso, growth_stage
- `soil_status`: sm, water_deficit, water_stress, n_availability, field_capacity, wilting_point
- `weather_today`: tmax, tmin, rain, radiation
- `weather_forecast`: 5-day list with deterministic noise
- `resources_used`: total_water, total_n, total_cost, budget_remaining, unit costs
- `season_summary`: crop, location, target_yield, budget, step_days
- `control_features`: moisture_gap_to_target, forecast_rain_3d, forecast_rain_7d, days_since_last_irrigation, days_since_last_fertilization, fertilizer_events_count, cumulative_n_applied, rooting_depth_cm, estimated_budget_to_finish, budget_remaining_ratio, dvs_distance_to_next_fertilizer_window
- `conflicts`: list of validation messages

If PCSE gives richer state than needed, do not expand the schema during the first migration pass.

### Phase 7: Recalibrate Target Yield

Current project uses a universal target yield:
- compute potential yield across all 3 locations for a given seed (unlimited water and nitrogen)
- use the max as the target for all tasks

This must be preserved with PCSE:
- replace current `compute_potential_yield` with a PCSE-backed version
- run potential yield simulation with no water/N limitation for each location
- take the maximum as universal target
- verify `Easy >= Medium >= Hard` still holds across seeds 42-46 at minimum
- if PCSE yield levels differ significantly from current (~5000-7000 kg/ha range for wheat), adjust budgets or other scenario parameters to maintain difficulty ordering

### Phase 8: Verify Reward Windows Against PCSE Phenology

Dense rewards in `server/reward.py` use DVS windows that have already been calibrated:
- fertilize near `0.20-0.40` (peak 0.30)
- fertilize near `0.50-0.70` (peak 0.60)
- harvest optimal `1.8-2.05`
- late harvest penalty: slope -0.20/DVS, floor -0.25

PCSE may shift DVS timing (wheat phenology depends on temperature sum thresholds which differ between our simplified model and PCSE's full WOFOST). If shifts are small (< 0.1 DVS), keep current windows. If shifts are large, adjust window centers to match PCSE's typical DVS-at-stage values.

Rule:
- do not change the reward philosophy
- prefer minimal numeric retuning over conceptual reward redesign
- grader weights are FROZEN — do not change

### Phase 9: Docker and Packaging

Additions to `requirements.txt`:
- `pcse` (plus any transitive dependencies it pulls in)

Verification requirements:
- Docker image still builds cleanly on python:3.11-slim
- All crop/soil parameter data is bundled (no network downloads at startup)
- Server still exposes `/health`, `/reset`, `/step`, `/state`, `/tasks`
- Health check responds within 5 seconds
- Image remains under ~2GB (current is ~500MB-1GB, PCSE adds ~50-200MB)
- `python inference.py` completes within 20 minutes across all 3 tasks

## Rough Time Estimate

With a strong coding LLM:
- best case: 2 to 3 days (if PCSE action injection works cleanly)
- realistic case: 4 to 6 days (hybrid adapter + calibration + test fixes)
- bad case: 7+ days if nitrogen modeling or calibration drift fight back

The bottleneck is not code generation. It is:
- hybrid adapter construction (working around PCSE's pre-scheduled management model)
- calibration (matching current DVS timing and yield levels)
- nitrogen proxy design (avoiding WOFOST-NPK instability)
- deterministic behavior verification
- Docker verification with bundled data

## Suggested Acceptance Criteria

A future PCSE migration should not be considered complete unless all of the following are true:
- project runs end to end with PCSE-backed simulation
- all 33 tests in `tests/test_smoke.py` pass (or justified replacements exist)
- scores remain in `[0.0, 1.0]`
- difficulty ordering remains `Easy >= Medium >= Hard` across seeds 42-46
- baseline scores remain within ±0.05 of current values (or justified drift is documented)
- inference still prints `=== RESULTS ===` and completes within 20 minutes
- Docker builds on python:3.11-slim and serves `/health` within 5 seconds
- current `.env` / HF Router / OpenAI client flow remains intact
- no external network calls during simulation (weather, crop data, etc.)
- documentation is updated consistently
- `openenv validate` still passes

## Explicit Non-Goals for First PCSE Pass

Do not do these during the first migration pass:
- add new crops
- redesign the grader (FROZEN)
- redesign the reward structure (already calibrated)
- redesign the observation schema
- change the task difficulty philosophy
- expand the action space
- add notebook-based workflows
- use PCSE's built-in network weather providers
- fully integrate WOFOST-NPK if it introduces instability

## Claude Prompt: Full Migration

Use this as the main prompt for Claude Opus:

```text
You are migrating an existing OpenEnv hackathon project from a custom pure-Python crop simulator to PCSE/WOFOST, with minimal architectural disruption.

Repository context:
- This is a precision agriculture OpenEnv environment.
- Current simulator is custom and lives in server/crop_sim.py.
- Environment loop is in server/environment.py.
- Scenario generation is in server/scenarios.py.
- Final grading is in server/grader.py (FROZEN — do not modify).
- Dense step rewards are in server/reward.py (already calibrated — modify only if PCSE phenology requires window adjustments).
- Inference client/agent is in inference.py.
- Current code already works, 33 tests pass, and docs are updated.
- You must preserve determinism, OpenEnv compatibility, and the existing grader/reward/task structure.

Current calibration targets to preserve:
- Harvest optimal window: DVS 1.8-2.05
- Fertilizer windows: DVS 0.20-0.40 (peak 0.30), DVS 0.50-0.70 (peak 0.60)
- Late-harvest penalty slope: -0.20/DVS, floor -0.25
- Intent/delta blend: 0.4/0.6 (validated)
- Baseline scores (seed=42): Task1=0.8442, Task2=0.8155, Task3=0.7046
- 33 passing tests

Hackathon constraints:
- vcpu=2, 8GB RAM, no GPU
- Inference must complete within 20 minutes
- No external API calls at runtime (weather, crop data, etc.)
- python:3.11-slim Docker base
- Health check must respond within 5 seconds
- Bit-exact deterministic reproducibility required

Goal:
Replace the simulator internals with PCSE using a HYBRID ADAPTER approach:
1. server/environment.py remains unchanged (still calls sim.advance(7, irrigation_cm=X, n_kg_ha=Y))
2. The adapter runs PCSE day-by-day internally, injecting irrigation/fertilizer between days
3. n_factor uses a simplified proxy (current linear model) unless WOFOST-NPK proves stable
4. Weather uses a custom WeatherDataProvider wrapping existing generated dicts
5. Tasks, grader, and reward logic remain mostly unchanged
6. Docker remains self-contained with bundled crop/soil data

Critical constraints:
- Do NOT rewrite the whole project from scratch.
- Do NOT modify server/grader.py.
- Do NOT change reward philosophy — only adjust DVS window numbers if PCSE phenology demonstrably shifts them.
- Keep the current universal target-yield approach.
- Preserve the existing observation schema including control_features.
- Keep the current action space: irrigate (max 10 cm), fertilize (max 50 kg), harvest, wait.
- Keep current stdout format in inference.py, especially the === RESULTS === summary.
- Prefer keeping the current n_factor linear model over depending on WOFOST-NPK.
- Do not introduce external data downloads at runtime.

Implementation strategy:
Phase 1: Analyze current code and freeze behavioral baselines
Phase 2: Build hybrid PCSE adapter preserving CropSimulator's interface
Phase 3: Build custom deterministic WeatherDataProvider from existing weather dicts
Phase 4: Bundle PCSE crop/soil/site parameters in the Docker image
Phase 5: Map irrigate/fertilize actions to PCSE state manipulation
Phase 6: Recalibrate compute_potential_yield using PCSE
Phase 7: Verify all 33 tests pass, difficulty ordering holds, scores remain within ±0.05
Phase 8: Update Docker, requirements.txt, and documentation

Expected output:
1. Brief migration plan
2. Files changed
3. Key adapter decisions (especially n_factor strategy)
4. Any blockers or assumptions
5. Final verification results (test pass/fail, baseline scores, difficulty ordering)

If PCSE integration turns out to be too fragile for a full migration in one pass, stop after Phase 2 or 3 and implement a compatibility scaffold plus explicit TODOs rather than forcing a broken end state.
```

## Claude Prompt: Safer First Pass

Use this first if you want a lower-risk migration start:

```text
Do ONLY Phase 1 and Phase 2:
- analyze the current simulator contract (documented in FUTURE_SCOPE_PCSE.md)
- design and implement a PCSE hybrid adapter scaffold
- the adapter should run PCSE internally day-by-day but expose the same advance/state interface
- use a custom WeatherDataProvider wrapping the existing weather dicts
- keep n_factor as the current linear model (do not integrate WOFOST-NPK yet)
- do not yet migrate scenario generation or target yield calibration
- leave the project in a compilable state where all 33 tests pass
- document exactly what remains for full migration
```

## Claude Prompt: Hybrid Fallback Approach

Use this if Claude gets stuck on PCSE signals or management events:

```text
Do not force native PCSE action signaling if it is unstable.
Instead, build a hybrid adapter:
- use PCSE for phenology (DVS advancement) and biomass/yield accumulation
- inject irrigation effects by directly modifying soil moisture in PCSE state between day steps
- keep the current n_factor linear model entirely outside PCSE
- preserve deterministic behavior and interface compatibility first
- explain clearly which parts are true PCSE and which parts are wrapper-level approximations
- run all 33 tests before declaring the adapter ready
```

## Prompting Guidance

Best sequence:
1. start with the safer first-pass prompt
2. inspect the adapter design before allowing full migration
3. only then run the full migration prompt
4. keep the hybrid fallback prompt ready if PCSE signal wiring becomes fragile

## Implementation Notes for a Future Engineer

When reviewing the PCSE adapter work, specifically check:
- whether the adapter preserves the environment-facing contract (every attribute and method listed above)
- whether the weather provider is deterministic (same seed produces identical weather sequence)
- whether DVS progression speed roughly matches current phenology (typical anthesis around day 100-130 for Netherlands)
- whether yield levels are in the same ballpark as current (5000-7000 kg/ha for wheat at maturity)
- whether `target_yield` still uses universal max potential logic
- whether n_factor proxy behaves similarly (starts 0.55, increases with N application, depletes post-anthesis)
- whether `_water_stress()` returns 1.0/0.5/0.1 at correct SM ratios
- whether the final warning and fallback logic in inference.py remain untouched
- whether Docker starts quickly (health check within 5 seconds) and answers `/health`
- whether no unrelated files were refactored needlessly
- whether all 33 tests pass without test modifications

## Future Validation Checklist

Run these after migration:
- `python -m pytest tests/ -v` — all 33 tests pass
- `python benchmark_sweep.py` — multi-seed score sampling for tasks 1, 2, 3
- local server start via `python -m uvicorn server.app:app --host 0.0.0.0 --port 8000`
- `python inference.py` — completes within 20 minutes, prints === RESULTS ===
- Docker build and health check (port 7860)
- verify difficulty ordering: Easy >= Medium >= Hard across seeds 42-46
- verify baseline scores within ±0.05 of current values

## Current Decision

As of now, the project remains on the custom simulator because it is:
- stable and well-tested (33 passing tests)
- deterministic
- easier to debug (~200 lines of transparent Python)
- already calibrated (reward slopes, DVS windows, yield targets)
- lower risk for the current hackathon timeline

PCSE is a future-scope enhancement, not a current blocker. The hybrid adapter approach is the recommended path if pursued.
```
