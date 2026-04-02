

   

# Precision Agriculture Crop Management RL Environment — Implementation Plan

**Team Hijibiji** | Meta PyTorch OpenEnv Hackathon | Deadline: April 7, 2026

**TL;DR:** A crop management RL environment where an LLM agent manages a wheat growing season — deciding when to irrigate, fertilize, and harvest to maximize yield while minimizing resource use. Uses a custom WOFOST-inspired pure-Python crop simulator with deterministic weather generation. No external dependencies or API calls at runtime.

> **Status (April 2026):** All phases implemented. See changes below.

---

## Competition Scoring Rubric

| Criterion | Weight |
|-----------|--------|
| Real-world utility | 30% |
| Task & grader quality | 25% |
| Environment design | 20% |
| Code quality | 15% |
| Creativity | 10% |

**Estimated score: 85-93/100** (up from 60-76 with ML Scheduler)

---

## Implementation Status

### Audit Sweep (21 fixes across 5 phases) — COMPLETED

**Phase 1 — Reward Signal Quality** ✅
- Wait reward: +0.05 → 0.0 (prevents lazy-wait policy exploitation)
- Late harvest: +0.10 → proportional penalty (prevents delay-for-reward)
- Between-window fertilization: +0.05 → -0.03 (prevents spray-more strategies)
- Removed dead budget check in reward.py (already enforced by environment.py)

**Phase 2 — Agronomic Accuracy** ✅
- LAI senescence rate: 0.03 → 0.10 (realistic 7-10 day senescence period)
- Punjab rainfall: increased rain_prob 0.08→0.12, mean rain 0.3→0.6cm (5-10cm/season)
- N-depletion: flat 0.0008 → phenology-aware (0.0003 pre-anthesis, 0.0015 post-anthesis)
- Added heat stress: >35°C during flowering (DVS 0.8-1.2) reduces growth up to 70%

**Phase 3 — Inference Robustness** ✅
- httpx.Client uses `with` context manager (clean socket handling)
- Client-side MAX_CLIENT_STEPS=200 guard against runaway episodes
- try/except around all LLM calls with graceful fallback to greedy heuristic
- Replaced `assert llm_client is not None` with safe check
- Regex-based JSON extraction for malformed LLM responses
- LLM usage tracking (call count + fallback count printed in summary)

**Phase 4 — Observation Enrichment** ✅
- Added `water_stress` (0-1 continuous) and `n_availability` (0-1) to soil_status
- Scoring formula text added to all 3 task instructions (transparency for LLM agents)

**Phase 5 — Code Quality Polish** ✅
- Removed unreachable `elif harvested: yield_score = 0.5` in grader
- Added Dockerfile HEALTHCHECK for orchestrator readiness detection
- Extracted greedy heuristic thresholds as named constants

---

## Key Technical Decisions

- **Custom WOFOST-inspired simulator:** Pure Python (~280 lines), no external deps, deterministic
- **Step size:** 7 days (weekly decisions) — gives ~15-20 steps per season, good for LLM reasoning
- **Weather data:** Pre-downloaded and embedded (no runtime API calls — Docker must be self-contained)
- **No CropGym or PCSE used** — custom pure-Python simulator avoids heavy dependencies and complex setup
- **Target yields** set to *universal* maximum potential production across all 3 locations — guarantees yield_score < 1.0 in water-limited scenarios and ensures Easy ≥ Medium ≥ Hard ordering
- **No existing crop env in OpenEnv** — verified across all 25+ reference environments (sumo_rl already exists, killing traffic signal option)

---

## Phase 1: Models & Data Structures

**File: `models.py`** — Replace all existing models

### CropState (Pydantic BaseModel)
- `episode_id`, `step_count`, `current_task_id`, `seed`
- `current_day: int` — simulation day (0-indexed from sowing)
- `total_days: int` — max season length
- `crop_name: str` — e.g. "wheat", "maize"
- `dvs: float` — development stage (0=sowing, 1=anthesis, 2=maturity)
- `lai: float` — leaf area index
- `tagp: float` — total above-ground production (kg/ha)
- `twso: float` — yield/grain weight (kg/ha)
- `sm: float` — soil moisture (fraction)
- `rain_today: float` — today's rainfall (cm)
- `temp_max: float`, `temp_min: float` — today's temps (°C)
- `total_water_applied: float` — cumulative irrigation (cm)
- `total_n_applied: float` — cumulative nitrogen (kg/ha)
- `total_cost: float` — cumulative costs ($)
- `budget: float` — total budget
- `actions_taken: list[dict]` — history of all agent actions

### CropAction (Pydantic BaseModel)
- `action_type: str` — one of: `"irrigate"`, `"fertilize"`, `"harvest"`, `"wait"`
- `amount: float` — irrigation cm or N kg/ha (0 for wait/harvest)

### CropObservation (Pydantic BaseModel)
- `done: bool`, `reward: float | None`, `metadata: dict`
- `task_id: int`, `task_name: str`, `instructions: str`
- `day: int`, `days_remaining: int`
- `crop_status: dict` — dvs, lai, tagp, twso, growth_stage_name (e.g. "vegetative", "flowering", "grain_fill", "mature")
- `soil_status: dict` — sm, water_deficit (bool)
- `weather_today: dict` — tmax, tmin, rain, radiation, wind
- `weather_forecast: list[dict]` — next 3-5 days forecast (from real data, with noise)
- `resources_used: dict` — total_water, total_n, total_cost, budget_remaining
- `season_summary: dict` — crop_name, location, target_yield
- `conflicts: list[str]` — any invalid action feedback

---

## Phase 2: Environment Core

**File: `server/environment.py`** — Replace `SchedulerEnvironment` with `CropEnvironment`

### reset(seed, task_id)
1. Load scenario from `generate_scenario(seed, task_id)` — returns crop params, soil type, location, budget, weather data
2. Initialize PCSE WOFOST model: `Wofost72_WLP_FD(params, weather, agromanagement)`
3. Run PCSE to sowing day, capture initial state
4. Return observation with initial crop/soil/weather state

### step(action)
1. Validate action (can't irrigate negative, can't fertilize after maturity, etc.)
2. If `irrigate`: send `model._send_signal(signal=pcse.signals.irrigate, amount=X, efficiency=0.7)`
3. If `fertilize`: send `model._send_signal(signal=pcse.signals.apply_n, amount=X, recovery=0.7)`
4. If `harvest`: end episode, compute final grade
5. If `wait`: no intervention
6. Advance PCSE by `step_days` days (7 days per step = weekly decisions, ~20 steps per season)
7. Read new state from PCSE: `model.get_variable('DVS')`, `'LAI'`, `'TAGP'`, `'TWSO'`, `'SM'`
8. Compute step reward via `reward.py`
9. Auto-terminate if DVS >= 2.0 (maturity) or day exceeds season
10. Build and return observation

**Key design: 7-day steps.** Agent makes ~15-20 decisions per growing season. This matches the "text reasoning" paradigm — weekly farm management decisions, not daily micro-control. LLM reads weather forecast + crop state + budget and reasons about trade-offs.

---

## Phase 3: Scenarios

**File: `server/scenarios.py`** — Complete rewrite

### Data sourcing (embedded, no runtime API calls)
- Pre-download 5-10 location weather sets from NASA POWER using `pcse.db.NASAPowerWeatherDataProvider` during development
- Save as serialized PCSE-compatible weather objects or CABO files inside `server/data/weather/`
- Use PCSE's built-in crop parameter database (`pcse.util.WOFOST72SiteDataProvider`, crop YAML files)
- Locations: Netherlands (wheat), Iowa USA (maize), Punjab India (wheat), Brazil (maize), Kenya (maize) — diverse climates

### generate_scenario(seed, task_id) returns:
```python
{
    "crop_name": "wheat",
    "crop_params": {...},       # PCSE ParameterProvider
    "soil_params": {...},       # from PCSE soil DB
    "weather": weather_provider, # pre-loaded weather data
    "location": "Netherlands",
    "sowing_date": date(1999, 10, 15),
    "max_duration": 300,        # days
    "budget": 500.0,            # $ per hectare
    "target_yield": 7000.0,     # kg/ha (reference potential yield)
    "step_days": 7,             # days per agent step
    "irrigation_cost": 2.0,     # $ per cm
    "fertilizer_cost": 1.5,     # $ per kg N
}
```

### Difficulty scaling
- **Task 1 (Easy):** Single crop (wheat), one location (Netherlands — mild, predictable), generous budget (2x cost), no water stress. Agent just needs to learn basic fertilization timing.
- **Task 2 (Medium):** Two possible crops, variable locations, tighter budget, some drought periods. Agent must balance irrigation vs budget.
- **Task 3 (Hard):** Random crop/location, tight budget (1.2x minimum cost), drought-prone locations (Kenya, Punjab), must optimize ALL three: yield + water + cost simultaneously.

---

## Phase 4: Grader

**File: `server/grader.py`** — Complete rewrite with multi-metric grading

### Metrics (all normalized 0-1)

| Metric | Formula | Purpose |
|--------|---------|---------|
| `yield_score` | `min(1.0, actual_yield / target_yield)` | Did the crop grow? |
| `water_efficiency` | `1.0 - min(1.0, water_used / max_water)` | Less water = better |
| `cost_efficiency` | `max(0, 1.0 - budget_used / budget)` | Under-budget is good |
| `timing_quality` | Bonus for fertilizing near DVS 0.3 and 0.6 | Domain-correct timing |
| `harvest_timing` | Penalty if too early (DVS<1.8) or too late (DVS>2.1) | Harvest at maturity |

### Per-task formulas (guaranteed non-trivial grading)

All tasks use the same **unified weights** — difficulty comes from environment conditions (climate, budget, soil), not scoring formula:

| yield | water | cost | timing | harvest |
|-------|-------|------|--------|---------|
| 35%   | 20%   | 18%  | 15%    | 12%     |

**Why this won't saturate:** Maximizing yield requires irrigation/fertilization, but water_efficiency and cost_efficiency penalize over-use. The Pareto frontier means a perfect 1.0 is mathematically impossible (target_yield is the universal max potential across all locations — an unreachable theoretical max under budget constraints). Harder locations naturally score lower because their climate makes it harder to approach the universal target.

---

## Phase 5: Rewards

**File: `server/reward.py`**

### compute_step_reward(action_type, dvs, sm, lai_change, cost, budget_remaining)
| Condition | Reward |
|-----------|--------|
| `wait` (always neutral) | 0.0 (prevents lazy-wait exploitation) |
| `irrigate` when SM < 0.3 (water stress) | +0.1 (correct response) |
| `irrigate` when SM > 0.5 (wasteful) | -0.05 |
| `fertilize` near DVS 0.3 or 0.6 (key growth stages) | +0.15 |
| `fertilize` at DVS > 1.5 (post-grain-fill, wasteful) | -0.1 |
| `fertilize` between key windows | -0.03 (discourages spray-more) |
| `harvest` at DVS ∈ [1.8, 2.1] (optimal window) | +0.2 |
| `harvest` at DVS < 1.5 (too early, yield lost) | -0.3 |
| Late harvest (DVS > 2.05) | proportional penalty up to -0.15 |
| Invalid/over-budget action | -0.1 |

### compute_trajectory_reward(grade_score)
Direct mapping: returns `grade_score` (as in current env).

---

## Phase 6: Tasks

**File: `server/tasks.py`**

```python
TASKS = {
    1: {
        "id": 1,
        "name": "Basic Crop Management",
        "difficulty": "easy",
        "instructions": "Manage a wheat crop in a temperate climate. Decide weekly whether to irrigate, fertilize, or wait. Harvest when the crop is mature (DVS ≈ 2.0). Budget is generous. Focus on maximizing yield."
    },
    2: {
        "id": 2,
        "name": "Water-Efficient Farming",
        "difficulty": "medium",
        "instructions": "Manage a crop under water scarcity. Balance yield against water conservation. Irrigation is expensive. Fertilize at key growth stages (DVS ~0.3 vegetative, ~0.6 reproductive). Budget is limited."
    },
    3: {
        "id": 3,
        "name": "Precision Agriculture",
        "difficulty": "hard",
        "instructions": "Maximize economic return from a crop in a challenging climate. Optimize all three: yield, water usage, and cost. Drought periods are likely. Fertilize and irrigate strategically based on crop development stage, soil moisture, and weather forecast."
    },
}
```

---

## Phase 7: Inference Agent

**File: `inference.py`** — Greedy heuristic + optional LLM reasoning

### Baseline greedy heuristic (no LLM needed)
```python
if dvs >= 1.8: harvest
elif sm < 0.22 and budget_remaining > irrigation_cost: irrigate(2.0 cm)
elif dvs in [0.25-0.35, 0.55-0.65] and budget_remaining > fert_cost: fertilize(12 kg)
else: wait
```
All thresholds extracted as named constants (`SM_IRRIGATE_THRESHOLD`, `IRRIGATE_AMOUNT`, etc.).

### LLM-enhanced agent (for better scores)
- Structured system prompt with numbered priority list emphasizing fertilization timing
- Regex-based JSON extraction for malformed LLM responses
- `llm_consecutive_errors` counter with `LLM_ERROR_THRESHOLD=3` — auto-fallback to greedy after 3 consecutive failures (handles credit exhaustion gracefully)
- End-of-run WARNING printed if credit exhaustion detected
- `python-dotenv` auto-loads `.env` file for API credentials
- LLM usage tracking: call count + fallback count printed in summary

---

## Phase 8: Docker & Config

### server/Dockerfile
- Base: `python:3.11-slim`
- `pip install` existing deps (FastAPI, Pydantic, uvicorn, python-dotenv)
- Pure Python simulator, no GPU needed
- HEALTHCHECK for orchestrator readiness detection

### openenv.yaml
```yaml
env_name: crop_management
description: "Precision agriculture environment where an LLM agent manages crop irrigation, fertilization, and harvest timing to maximize yield while minimizing water and cost."
tasks: [1, 2, 3]
env_vars: [API_BASE_URL, MODEL_NAME, HF_TOKEN]
```

### requirements.txt
Key deps: `fastapi`, `pydantic>=2.7`, `uvicorn`, `openai>=1.0`, `httpx>=0.25`, `python-dotenv>=1.0`

---

## Phase 9: Testing & Polish

1. Run all 3 tasks with greedy baseline, verify scores: Task 1 ~0.82, Task 2 ~0.80, Task 3 ~0.70
2. Verify difficulty ordering: Easy ≥ Medium ≥ Hard (enforced in test_smoke.py)
3. Run with LLM agent, verify improvement over baseline
4. Verify Docker builds and runs on vcpu=2/8GB
5. Verify inference completes < 20 minutes
6. Write README with scoring rubric explanation
7. Final audit: check grader doesn't saturate, scenarios are seeded/deterministic

---

## Files to Modify

| File | Action | Key Changes |
|------|--------|-------------|
| `models.py` | **Rewrite** | CropAction, CropObservation, CropState |
| `server/environment.py` | **Rewrite** | CropEnvironment with water_stress & n_availability in observations |
| `server/crop_sim.py` | **New** | WOFOST-inspired pure-Python crop simulator (~280 lines) |
| `server/scenarios.py` | **Rewrite** | Deterministic weather generation, 3 locations, universal target yield |
| `server/grader.py` | **Rewrite** | Unified multi-metric grading (same weights all tasks) |
| `server/reward.py` | **Rewrite** | Dense step rewards (wait=0.0, late harvest penalty, between-window fert=-0.03) |
| `server/tasks.py` | **Rewrite** | 3 tasks with scoring formula in instructions |
| `inference.py` | **Rewrite** | Greedy heuristic + LLM with dotenv, early-stop, regex JSON extraction |
| `server/app.py` | **Minor edit** | Update import names (CropEnvironment, CropAction, CropObservation) |
| `openenv.yaml` | **Edit** | Update env_name, description, tasks, dependencies |
| `requirements.txt` | **Edit** | fastapi, pydantic, uvicorn, openai, httpx, python-dotenv |
| `server/Dockerfile` | **Edit** | HEALTHCHECK, pure-Python deps only |
| `tests/test_smoke.py` | **New** | 9 tests: determinism, reset/step, scoring range, difficulty ordering |
| `README.md` | **Rewrite** | Full documentation |

---

## Verification Plan

1. **Unit test grader:** Feed known yield/water/cost values, verify scores match expected formulas
2. **Saturation test:** Run greedy baseline on all 3 tasks × 5 seeds — confirm Task 2 < 0.85, Task 3 < 0.70 (no saturation)
3. **Determinism test:** Same seed produces identical episode — verify PCSE is deterministic with fixed weather
4. **Docker test:** `docker build` + `docker run` + hit `/reset` and `/step` endpoints
5. **Inference test:** Full `python inference.py` run completes in < 20 minutes, stdout format matches `=== RESULTS ===` block
6. **LLM reasoning test:** Verify LLM agent outperforms greedy by ≥ 0.05 on at least 2 tasks

---

## Team Split (5 days)

| Day | Roudraneel | Rijul | Tirthajoti |
|-----|-----------|-------|------------|
| 1 | models.py + environment.py skeleton | scenarios.py + download weather data | tasks.py + study PCSE API |
| 2 | environment.py (PCSE integration) | grader.py + reward.py | Help with scenarios / test PCSE |
| 3 | Docker + app.py wiring | Grader unit tests + saturation tests | inference.py (greedy + LLM) |
| 4 | Integration testing | End-to-end testing all 3 tasks | README + openenv.yaml |
| 5 | Bug fixes + polish | Final audit | Final inference tuning + submission |

---

## Competition Constraints (verified ✅)

- ✅ Docker self-contained (no runtime API calls except LLM inference via HF Router)
- ✅ Environment variables: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` — used via OpenAI client
- ✅ inference.py uses `openai.OpenAI` client with HF Router
- ✅ stdout format: `=== RESULTS ===` block with task scores
- ✅ vcpu=2/8GB compatible (pure Python simulator, no external scientific libraries)
- ✅ Inference < 20 minutes (simulator runs in milliseconds per step)
- ✅ `.env` auto-loaded by python-dotenv (no manual export needed)
- ✅ LLM early-stop after 3 consecutive errors (graceful credit exhaustion handling)

### Team Split (5 days)

| Day | Roudraneel | Rijul | Tirthajoti |
|-----|-----------|-------|------------|
| 1 | models.py + environment.py skeleton | scenarios.py + download weather data | tasks.py + study PCSE API |
| 2 | environment.py (PCSE integration) | grader.py + reward.py | Help with scenarios / test PCSE |
| 3 | Docker + app.py wiring | Grader unit tests + saturation tests | inference.py (greedy + LLM) |
| 4 | Integration testing | End-to-end testing all 3 tasks | README + openenv.yaml |
| 5 | Bug fixes + polish | Final audit | Final inference tuning + submission |
