# Future Scope: PCSE Migration Plan

This document is a handoff artifact for a future migration from the current custom WOFOST-inspired simulator to PCSE/WOFOST.

It is written to support a later implementation pass by Claude Opus or another strong coding LLM.

## Current State

The current project is a working precision agriculture OpenEnv environment with:
- deterministic seeded scenarios
- 3 tasks: Netherlands, Iowa, Punjab
- unified grader weights across tasks
- universal target yield calibration
- dense per-step rewards
- stable greedy baseline
- optional LLM-driven inference with HF Router

The current simulator is custom and lives in `server/crop_sim.py`.

## Goal of a Future PCSE Migration

Replace the crop simulation internals with PCSE while preserving:
- OpenEnv interface and task loop
- current action space: `irrigate`, `fertilize`, `harvest`, `wait`
- current observation schema as closely as possible
- deterministic behavior across seeds
- current grading philosophy and reward structure
- current inference output contract (`=== RESULTS ===`)

## Why Consider PCSE

### Benefits
- scientific credibility: PCSE is the reference WOFOST implementation
- better judge appeal: "uses PCSE/WOFOST" is stronger than "WOFOST-inspired"
- richer agronomic dynamics: phenology, water balance, nutrient effects, crop management realism
- stronger real-world utility narrative for the hackathon rubric

### Costs / Risks
- setup complexity: crop/site/soil/agromanagement configuration is more involved
- integration complexity: irrigation/fertilizer actions may need PCSE signals or management event handling
- dependency weight: PCSE brings numpy/pandas and related runtime overhead
- Docker complexity: need to ensure self-contained deployment and data availability
- calibration drift: DVS timing, yield levels, and stress behavior will shift relative to current logic
- debugging cost: more opaque than the current pure-Python simulator

## Recommendation Threshold

Proceed with PCSE only if there are at least 3 to 5 focused days available for:
- integration
- calibration
- testing
- Docker verification
- documentation refresh

If time is tighter than that, keep the current simulator.

## Current File Map

These are the files most relevant to the migration:
- `server/crop_sim.py` — current simulator
- `server/environment.py` — OpenEnv environment loop
- `server/scenarios.py` — deterministic scenario and weather generation
- `server/grader.py` — final scoring logic
- `server/reward.py` — dense step rewards
- `server/tasks.py` — task descriptions and instructions
- `models.py` — Pydantic action/observation/state models
- `inference.py` — greedy + LLM inference client
- `server/Dockerfile` — container runtime
- `requirements.txt` — dependencies
- `openenv.yaml` — metadata and inference env vars
- `tests/test_smoke.py` — current smoke suite

## Required Simulator Compatibility Contract

A PCSE-backed adapter should preserve the interface expected by `server/environment.py`.

The environment currently expects a simulator object that provides:
- constructor inputs:
  - `crop_params`
  - `soil_params`
  - `weather_data`
  - `partition_table`
- state-like attributes:
  - `current_day`
  - `dvs`
  - `lai`
  - `tagp`
  - `twso`
  - `sm`
  - `n_factor`
  - `total_water`
  - `total_n`
- methods:
  - `advance(days, irrigation_cm=0.0, n_kg_ha=0.0)`
  - `get_weather(day)`
  - `get_weather_forecast(start_day, n_days=5)`
  - `growth_stage_name()`

If PCSE does not expose one of these fields directly, the adapter should derive or approximate it and document the mapping.

## Migration Strategy

### Phase 1: Read and Freeze Current Behavior

Before changing anything:
- read `server/crop_sim.py`
- read `server/environment.py`
- read `server/scenarios.py`
- read `server/grader.py`
- read `server/reward.py`
- read `models.py`
- read `tests/test_smoke.py`

Record the current behavioral contract:
- observation fields
- action caps
- DVS windows used in rewards and tasks
- current greedy baseline scores
- determinism guarantees

### Phase 2: Build a PCSE Adapter Instead of Rewriting the Environment

Preferred approach:
- keep `server/environment.py` mostly intact
- replace the internals behind `CropSimulator`
- either:
  - reimplement `server/crop_sim.py` as a PCSE adapter, or
  - add `server/pcse_adapter.py` and have `environment.py` use it

The adapter should expose the same high-level contract listed above.

### Phase 3: Deterministic Weather Provider

Current weather is generated deterministically as Python dicts in `server/scenarios.py`.

Future PCSE migration options:
- Option A: wrap the current generated weather in a custom PCSE-compatible WeatherDataProvider
- Option B: pre-bundle PCSE weather files generated from the same deterministic logic

Preferred option:
- Option A, because it preserves current seeded scenario generation and minimizes external data requirements

### Phase 4: Crop / Soil / Site / Agromanagement Setup

PCSE migration will need:
- crop parameters for winter wheat
- soil parameters for the 3 locations or representative soil classes
- site data
- agromanagement calendar / sowing / emergence / harvest assumptions

Important:
- keep only wheat for now
- do not broaden crop scope during the migration
- keep 3 locations and current budget difficulty scaling

### Phase 5: Action Mapping

Current actions are:
- `irrigate(amount_cm)`
- `fertilize(amount_kg_n_ha)`
- `harvest`
- `wait`

The adapter must map these to PCSE behavior.

Preferred rule:
- irrigation and fertilization remain controlled at the environment layer
- harvest remains a terminal environment action
- PCSE should not silently take over episode termination policy beyond crop maturity signals already used by environment logic

### Phase 6: Observation Mapping

Preserve the current observation schema:
- `crop_status`
  - `dvs`
  - `lai`
  - `tagp`
  - `twso`
  - `growth_stage`
- `soil_status`
  - `sm`
  - `water_deficit`
  - `water_stress`
  - `n_availability`
  - `field_capacity`
  - `wilting_point`
- `weather_today`
- `weather_forecast`
- `resources_used`
- `season_summary`
- `conflicts`

If PCSE gives richer state than needed, do not expand the schema during the first migration pass.

### Phase 7: Recalibrate Target Yield

Current project uses a universal target yield:
- compute potential yield across all 3 locations for a given seed
- use the max as the target for all tasks

This should be preserved with PCSE.

Required future work:
- replace current `compute_potential_yield` with a PCSE-backed version
- recompute baseline scores
- verify `Easy >= Medium >= Hard` still holds across multiple seeds

### Phase 8: Recalibrate Reward Windows If Necessary

Dense rewards in `server/reward.py` use DVS windows like:
- fertilize near `0.20-0.40`
- fertilize near `0.50-0.70`
- harvest near `1.8-2.05`

These may need mild adjustment if PCSE shifts stage timing materially.

Rule:
- do not change the reward philosophy unless clearly necessary
- prefer minimal numeric retuning over conceptual reward redesign

### Phase 9: Docker and Packaging

Future dependency additions likely include:
- `pcse`
- transitive scientific stack dependencies

Verification requirements:
- Docker image still builds cleanly
- server still exposes `/health`, `/reset`, `/step`, `/state`, `/tasks`
- health check still works
- image remains reasonable in size
- inference still completes within hackathon runtime expectations

## Rough Time Estimate

With a strong coding LLM:
- best case: 1.5 to 2.5 days
- realistic case: 3 to 5 days
- bad case: 6+ days if PCSE integration and calibration fight back

The bottleneck is not code generation. It is:
- integration
- debugging
- calibration
- deterministic behavior
- Docker verification

## Suggested Acceptance Criteria

A future PCSE migration should not be considered complete unless all of the following are true:
- project runs end to end with PCSE-backed simulation
- `tests/test_smoke.py` passes
- scores remain in `[0.0, 1.0]`
- difficulty ordering remains `Easy >= Medium >= Hard`
- inference still prints `=== RESULTS ===`
- Docker builds and serves `/health`
- current `.env` / HF Router / OpenAI client flow remains intact
- documentation is updated consistently

## Explicit Non-Goals for First PCSE Pass

Do not do these during the first migration pass:
- add new crops
- redesign the grader
- redesign the reward structure
- redesign the observation schema
- change the task difficulty philosophy
- add notebook-based workflows
- expand the project beyond the current hackathon scope

## Claude Prompt: Full Migration

Use this as the main prompt for Claude Opus:

```text
You are migrating an existing OpenEnv hackathon project from a custom pure-Python crop simulator to PCSE/WOFOST, with minimal architectural disruption.

Repository context:
- This is a precision agriculture OpenEnv environment.
- Current simulator is custom and lives in server/crop_sim.py.
- Environment loop is in server/environment.py.
- Scenario generation is in server/scenarios.py.
- Final grading is in server/grader.py.
- Dense step rewards are in server/reward.py.
- Inference client/agent is in inference.py.
- Current code already works, tests pass, and docs are updated.
- You must preserve determinism, OpenEnv compatibility, and the existing grader/reward/task structure as much as possible.

Goal:
Replace the simulator internals with PCSE while keeping the public behavior and architecture stable enough that:
1. server/environment.py remains the main OpenEnv environment
2. tasks, grader, and reward logic remain mostly unchanged
3. deterministic seeded scenarios still work
4. Docker remains self-contained
5. inference.py still runs without changes to output format

Critical constraints:
- Do NOT rewrite the whole project from scratch.
- Do NOT change the overall task definitions or scoring philosophy unless necessary.
- Keep the current universal target-yield idea unless a better calibration is needed.
- Preserve the existing observation schema as closely as possible.
- Keep the current action space: irrigate, fertilize, harvest, wait.
- Keep current stdout format in inference.py, especially the === RESULTS === summary.
- Keep current .env / HF Router / OpenAI client logic unchanged.
- Prefer an adapter layer over invasive rewrites.

Implementation strategy:
Phase 1: Analyze current code
- Read server/crop_sim.py, server/environment.py, server/scenarios.py, server/grader.py, server/reward.py, models.py, requirements.txt, server/Dockerfile.
- Identify exactly what current environment.py expects from CropSimulator.
- Produce a short compatibility contract for the simulator object:
  - constructor inputs
  - properties accessed
  - methods used
  - weather forecast methods used

Phase 2: Design a PCSE adapter
- Create a new adapter in server/crop_sim.py or a new file if cleaner.
- The adapter must expose the SAME or nearly same interface currently used by environment.py:
  - current_day
  - dvs
  - lai
  - tagp
  - twso
  - sm
  - n_factor if possible, or a reasonable proxy
  - total_water
  - total_n
  - get_weather(day)
  - get_weather_forecast(start_day, n_days)
  - advance(days, irrigation_cm=..., n_kg_ha=...)
  - growth_stage_name()
- If PCSE does not expose one field directly, compute or approximate it and document the mapping in comments.

Phase 3: PCSE scenario plumbing
- Update server/scenarios.py to provide PCSE-compatible crop/site/soil/agromanagement/weather inputs.
- Keep the 3 existing locations and seeded deterministic weather behavior if feasible.
- If true PCSE weather providers are too heavy, build a lightweight deterministic custom WeatherDataProvider from the existing generated weather dicts.
- Preserve the same task budgets and approximate climatic difficulty ordering.

Phase 4: Action integration
- Map irrigate/fertilize actions into PCSE-compatible management signals or agromanagement updates.
- Preserve current action caps and budget logic in environment.py where possible.
- Ensure harvest remains controlled by environment.py, even if PCSE can run beyond that point.

Phase 5: Target yield calibration
- Re-implement compute_potential_yield using PCSE.
- Preserve the universal target approach: target_yield = max potential yield across all 3 task locations for a given seed.
- Ensure yield_score remains fair and Easy >= Medium >= Hard on average.

Phase 6: Minimal environment changes
- Modify server/environment.py only as needed to work with the adapter.
- Keep the observation schema stable.
- Keep water_stress and n_availability fields if possible; if PCSE does not expose them directly, derive reasonable proxies.

Phase 7: Tests and verification
- Update/add tests only where behavior necessarily changed.
- Run or specify checks for:
  - determinism
  - all tasks complete successfully
  - scores remain in [0, 1]
  - difficulty ordering Easy >= Medium >= Hard
  - inference.py still works
- If the migration causes scoring drift, recalibrate target yield or scenario parameters instead of weakening tests.

Phase 8: Documentation
- Update README.md, plan.md, openenv.yaml, and requirements.txt to reflect PCSE usage only if the migration is completed and stable.
- Do not leave stale docs claiming pure-Python simulator if PCSE is now used.
- Document any adapter approximations clearly.

Coding requirements:
- Make minimal, focused changes.
- Add concise comments only where the mapping from PCSE to current abstractions is non-obvious.
- Preserve existing style and naming where possible.
- Do not remove current safety guards such as max steps or LLM fallback behavior.
- Do not change inference.py behavior except if absolutely required by observation changes.
- Do not introduce unrelated refactors.

Expected output format from you:
1. Brief migration plan
2. Files changed
3. Key adapter decisions
4. Any blockers or assumptions
5. Final verification results

If PCSE integration turns out to be too fragile for a full migration in one pass, stop after Phase 2 or 3 and implement a compatibility scaffold plus explicit TODOs rather than forcing a broken end state.
```

## Claude Prompt: Safer First Pass

Use this first if you want a lower-risk migration start:

```text
Do ONLY Phase 1 and Phase 2:
- analyze the current simulator contract
- design and implement a PCSE adapter scaffold
- do not yet fully migrate scenario generation or action signals if that would make the project unstable
- leave the project in a compilable state
- document exactly what remains for full migration
```

## Claude Prompt: Hybrid Fallback Approach

Use this if Claude gets stuck on PCSE signals or management events:

```text
Do not force native PCSE action signaling if it is unstable.
Instead, build a hybrid adapter:
- use PCSE for baseline crop growth and state evolution
- inject irrigation/fertilizer effects through a controlled wrapper if necessary
- preserve deterministic behavior and interface compatibility first
- explain clearly which parts are true PCSE and which parts are wrapper-level approximations
```

## Prompting Guidance for Claude

Best sequence:
1. start with the safer first-pass prompt
2. inspect the adapter design before allowing full migration
3. only then run the full migration prompt
4. keep the hybrid fallback prompt ready if PCSE signal wiring becomes fragile

## Implementation Notes for a Future Engineer

When reviewing Claude's PCSE work, specifically check:
- whether the adapter preserves the environment-facing contract
- whether the weather provider is deterministic
- whether DVS values remain sensible relative to current reward windows
- whether `target_yield` still uses universal max potential logic
- whether the final warning and fallback logic in `inference.py` remain untouched
- whether Docker still starts quickly and answers `/health`
- whether no unrelated files were refactored needlessly

## Future Validation Checklist

Run these after migration:
- `python -m pytest tests/ -v`
- local server start via `python -m uvicorn server.app:app --host 0.0.0.0 --port 8000`
- `python inference.py`
- Docker build and health check
- multi-seed score sampling for tasks 1, 2, 3

## Current Decision

As of now, the project remains on the custom simulator because it is:
- stable
- deterministic
- easier to debug
- already calibrated
- lower risk for the current hackathon timeline

PCSE is a future-scope enhancement, not a current blocker.
