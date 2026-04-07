# Architecture — Precision Agriculture Crop Management

> **Last updated:** April 2026
> **Version:** 2.5 — Reward shaping v2: 8 dense-reward fixes (wait penalties, inspect tier-gating, dose curve, terminal unification, delta yield-signal gating) + harvest urgency within [1.80, 2.00] window, post-maturity grace period (2 extra steps with grain shattering), step reward clamp [−0.9, +0.9]

---

## 1. System Overview

This project is a deterministic, multi-step precision agriculture environment built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework. An AI agent manages a wheat growing season—deciding weekly when to irrigate, fertilize, and harvest to maximize yield while minimizing water use and cost.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AGENT LAYER                                      │
│  agent/inference.py  ── LLM policy + oracle baseline + orchestration   │
│  agent/training_adapter.py ── discrete RL action vocabulary            │
│  agent/benchmark_sweep.py  ── multi-seed evaluation                    │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │  WebSocket (/ws) or HTTP (/reset, /step)
┌───────────────────────────────▼─────────────────────────────────────────┐
│                       CLIENT LAYER                                      │
│  client.py  ── CropEnvClient (EnvClient subclass, WebSocket transport) │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│                       SERVER LAYER                                      │
│  server/app.py  ── FastAPI entry point via create_app() + /tasks       │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ENVIRONMENT INTERFACE                                          │   │
│  │  server/environment.py ── CropEnvironment (reset/step/state)    │   │
│  └────────┬──────────┬──────────┬──────────┬──────────┬────────────┘   │
│           │          │          │          │          │                  │
│  ┌────────▼──┐ ┌─────▼────┐ ┌──▼───┐ ┌───▼──┐ ┌────▼─────┐           │
│  │ crop_sim  │ │ scenarios│ │ tasks│ │grader│ │  reward  │           │
│  │(simulator)│ │(weather) │ │(defs)│ │(score)│ │ (dense)  │           │
│  └───────────┘ └──────────┘ └──────┘ └──────┘ └──────────┘           │
└─────────────────────────────────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│                       DATA MODELS LAYER                                 │
│  models.py ── CropAction, CropObservation, CropState (Pydantic)        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Layer Model

The codebase is organized into **four logical layers**, each with a clear responsibility boundary:

| Layer | Responsibility | Key Files | Depends On |
|-------|---------------|-----------|------------|
| **Data Models** | Pydantic schemas for action, observation, state | `models.py` | OpenEnv core types |
| **Server / Environment** | OpenEnv interface, HTTP/WS endpoints, episode loop | `server/app.py`, `server/environment.py` | Models, Simulation Domain |
| **Simulation Domain** | Pure crop science: growth model, weather, grading, rewards | `server/crop_sim.py`, `server/scenarios.py`, `server/tasks.py`, `server/grader.py`, `server/reward.py` | Models (for type hints only) |
| **Agent / Training** | Policies, RL adapters, evaluation | `agent/inference.py`, `agent/training_adapter.py`, `agent/benchmark_sweep.py` | Models, Client |

**Dependency rule:** Each layer depends only on layers below it. The Agent layer never imports from `server/` internals (except `benchmark_sweep.py` which uses `CropEnvironment` directly for fast local evaluation).

---

## 3. Component Descriptions

### 3.1 Data Models — `models.py`

| Class | Base | Key Fields | Purpose |
|-------|------|-----------|---------|
| `CropAction` | `Action` | `action_type` (str), `amount` (float) | Agent's weekly decision |
| `CropObservation` | `Observation` | 12 nested dicts: crop/soil/weather/resources/control_features/conflicts + task context + `dose_hint` | Rich observation for LLM or RL |
| `CropState` | `State` | Full simulator state + episode tracking + `maturity_reached_step`, `last_soil_report`, `last_crop_report` | Serializable checkpoint |

All models use Pydantic `BaseModel` with `extra="forbid"` and inherit from OpenEnv base types.

### 3.2 Server — `server/app.py`

- Creates the FastAPI app via `create_app(CropEnvironment, CropAction, CropObservation)`
- Auto-registers: `/health`, `/reset`, `/step`, `/state`, `/ws`, `/docs`, `/web`
- Custom endpoints:
  - `/tasks` (GET) — lists available task definitions
  - `/grader` (POST) — grades an episode given metrics: returns `{score, breakdown}`
  - `/baseline` (GET) — deterministic greedy baseline scores for all tasks (seed=42, cached)
  - `/ceiling` (GET) — deterministic oracle ceiling scores for all tasks (seed=42, cached)
- Entry point: `uvicorn server.app:app --host 0.0.0.0 --port 8000`

### 3.3 Environment Interface — `server/environment.py`

| Method | Responsibility |
|--------|---------------|
| `reset(seed, task_id)` | Load scenario → create `CropSimulator` → apply probe overrides → return initial observation |
| `step(action)` | Validate action → if inspect: deduct budget, store report, return early (no sim advance) → else: compute intent reward → advance sim 7 days → compute delta reward → check termination → return observation |
| `state` (property) | Return serializable `CropState` |

**Inspect actions** are handled as **free sub-actions**: they deduct budget and store their report in `CropState` (`last_soil_report` / `last_crop_report`), but do **not** advance the simulation, increment the step counter, or consume a week. Budget is the only constraint on inspects (no artificial cap). Reports persist in all subsequent observations.

**Termination conditions** (checked in order):
1. Agent sends `harvest` action → terminal, compute final grade with blended reward (0.7 × trajectory + 0.3 × normalized harvest step signal)
2. 2 steps after DVS first reaches 2.0 → auto-harvest with shattering penalty (grain yield degrades ~23%/step post-maturity; agent gets 2 extra steps to harvest explicitly before forced termination)
3. Day ≥ max_duration → forced season end
4. Step ≥ MAX_STEPS (60) → safety cap

### 3.4 Simulation Engine — `server/crop_sim.py`

Pure-Python WOFOST-inspired model (~330 LOC):

| Component | Implementation |
|-----------|---------------|
| Phenology | Temperature-sum DVS (0→1→2), TSUM1/TSUM2 control |
| Biomass | Light-use efficiency: PAR × LUE × LAI interception |
| Water balance | Rainfall + irrigation − ET (Hargreaves), bounded [wilting, field_capacity] |
| Water stress | 0.1–1.0 factor, reduces growth when SM < threshold |
| Heat stress | Pollen sterility >35°C at anthesis; grain fill penalty >32°C |
| Nitrogen | Linear n_factor model (0.3–1.0), phenology-aware depletion |
| Partitioning | DVS-dependent table: grain fraction increases post-anthesis |
| LAI dynamics | Log-linear growth vegetative, senescence post-DVS 1.5 |

**Key method:** `advance(days, irrigation_cm, n_kg_ha)` — advances the model by `days` time-steps, applying interventions. Post-maturity (DVS ≥ 2.0): biomass growth stops (`actual_growth = 0`), but grain shattering continues (`SHATTER_RATE = 0.25/day` above `SHATTER_DVS = 1.85`), causing ~23% yield loss per 7-day step. This natural consequence, combined with the 2-step grace period before auto-termination, teaches the agent that delaying harvest past maturity is costly.

**Data libraries** (module-level dicts, sourced from `server/crop_params.py`):
- `CROP_LIBRARY` — region-specific WOFOST wheat profiles (wheat_nl, wheat_iowa, wheat_punjab)
- `SOIL_LIBRARY` — clay_loam, sandy_loam, silt_loam with published FAO/ISRIC parameters
- `PARTITION_TABLES` — DVS→grain_fraction tuples auto-generated from FOTB

### 3.4a Crop & Soil Parameters — `server/crop_params.py`

Centralized WOFOST parameter library with frozen dataclasses:

| Class | Key Fields | Source |
|-------|-----------|--------|
| `WOFOSTCropParams` | tsum1, tsum2, lue, tdwi, laiem, slatb, fotb, etc. | Boogaard et al. (2014), de Wit et al. (2019) |
| `WOFOSTSoilParams` | SMFCF, SMW, RDMSOL, SM0, CRAIRC | ISRIC, FAO soil classification |

3 wheat profiles: `WHEAT_NL`, `WHEAT_IOWA`, `WHEAT_PUNJAB` (region-calibrated).
3 soil profiles: `SOIL_CLAY_LOAM`, `SOIL_SANDY_LOAM`, `SOIL_SILT_LOAM`.

YAML override: `load_profile_from_yaml(path)` loads custom profiles from `configs/` directory, falling back to hardcoded defaults.

### 3.4b Advisory Text — `server/advisory.py`

Deterministic template-based advisory generator. `generate_advisory()` takes `has_crop_report` and `budget_remaining` parameters to gate inspect recommendations and produces factual, neutral text describing current crop state, stress conditions, and resource usage — never prescribing actions. Uses 5-band nitrogen status (very_low/low/moderate/adequate/surplus) aligned with the oracle dose formula. Covers all DVS ranges including the DVS 1.50–1.70 gap ("NOT yet in harvest window") for all tiers. Included in `CropObservation.advisory_text`.

### 3.5 Scenario Generation — `server/scenarios.py`

Deterministic by seed, no external data:

| Location | Climate | Soil | Budget | Max Days |
|----------|---------|------|--------|----------|
| Netherlands | Mild maritime, 45% rain probability | clay_loam | $800 | 280 |
| Iowa | Continental, 30% rain probability | silt_loam | $450 | 260 |
| Punjab | Hot semi-arid, 12% rain probability | sandy_loam | $300 | 200 |

**Weather generators:** Temperature (seasonal sine + Gaussian noise), rainfall (stochastic Poisson events), radiation (seasonal curve + noise). All seeded by `seed × prime + offset`.

**5 probe scenarios** for RL diagnostics:
1. `over_irrigation_trap` — wet soil + rain forecast
2. `late_fertilizer_temptation` — low N but DVS > 1.18
3. `budget_starvation` — $32 budget, dry soil
4. `harvest_hesitation` — DVS = 1.76
5. `drought_rescue` — DVS = 0.58, SM = 0.12

### 3.6 Task Definitions — `server/tasks.py`

Three tasks with the **same grading formula** but different environment difficulty:

| Task | Difficulty | Location | Budget | Key Challenge |
|------|-----------|----------|--------|---------------|
| 1 | Easy | Netherlands | $800 | Timing (rainfall handles water) |
| 2 | Medium | Iowa | $450 | Balance yield/efficiency/cost |
| 3 | Hard | Punjab | $300 | Every decision is a trade-off |

### 3.7 Grading — `server/grader.py` (FROZEN)

Single unified formula for all tasks:

```
score = 0.35 × yield_score
      + 0.20 × water_efficiency
      + 0.18 × cost_efficiency
      + 0.15 × timing_quality
      + 0.12 × harvest_timing
```

| Metric | Formula | Range |
|--------|---------|-------|
| yield_score | min(1.0, actual_yield / target_yield) | [0, 1] |
| water_efficiency | max(0, 1 − total_water / 50cm) | [0, 1] |
| cost_efficiency | max(0, 1 − total_cost / budget) | [0, 1] |
| timing_quality | Mean proximity of N actions to DVS 0.3 & 0.6 | [0.2, 1.0] |
| harvest_timing | 1.0 if DVS ∈ [1.8, 2.00], penalty otherwise | [0, 1] |

`target_yield` = max potential yield across all 3 locations for that seed (universal target).

### 3.8 Dense Rewards — `server/reward.py`

Three-tier reward architecture:

| Tier | Function | Range | When |
|------|----------|-------|------|
| **Intent** | `compute_step_reward()` | [−0.3, +0.2] | Before advancing sim |
| **Delta** | `compute_delta_reward()` | [−0.15, +0.15] | After advancing sim |
| **Trajectory** | `compute_trajectory_reward()` | [0.0, 1.0] | At episode end (= final grade, blended with harvest step signal) |

Step reward blend: `0.4 × intent + 0.6 × delta`, clamped to [`STEP_REWARD_MIN`, `STEP_REWARD_MAX`] (default [−0.9, +0.9]).

Terminal reward blend: `0.7 × trajectory_reward + 0.3 × normalized_harvest_step_signal` — gives immediate credit for good harvest timing at the terminal step. Auto-termination (non-explicit harvest) applies a 0.5× multiplier on the harvest-timing component.

**Reward shaping highlights (v2):**
- Wait penalties doubled for stress/N deficit; harvest urgency ramps from −0.05 to −0.10 inside [1.80, 2.00] DVS window; −0.10 flat during post-maturity grace period
- Inspect reward gated by observability tier (tier 1 = 0.0) and scaled by budget pressure
- Fertilizer dose curve steepened (divisor halved) for sharper dose sensitivity
- Delta yield-signal suppressed when the primary action effect is negative (prevents credit for passive yield growth during harmful actions)
- Wait delta rain-luck confound halved (0.3→0.15 coefficient)

### 3.9 Client — `client.py`

`CropEnvClient` extends OpenEnv's `EnvClient[CropAction, CropObservation, CropState]`:
- Handles WebSocket serialization/deserialization
- Methods: `reset()`, `step()`, `state`
- Usage: `CropEnvClient(base_url="...").sync()` for synchronous operation

### 3.10 Agent — `agent/inference.py`

Three policy paths:
1. **LLM-only** — if API_KEY (or HF_TOKEN) + API_BASE_URL + MODEL_NAME are set
2. **LLM + fallback** — on LLM errors, falls back to the observation-limited greedy heuristic (auto-disables after 3 consecutive failures)
3. **Greedy heuristic only** — default when no LLM credentials
4. **Oracle baseline only** — reserved for ceiling measurement, benchmarks, and diagnostics

**Greedy heuristic** (`greedy_action()`) — the public-surface fallback policy:

The greedy heuristic is intentionally constrained to the same information
surface that the LLM sees: exact numeric values on tier 1, and masked bands,
weather summaries, advisory text, and persisted inspect reports on tier 2/3.
It does not read simulator internals, hidden DVS, or oracle-only state.

**Oracle baseline** (`oracle_reference_action()` in the environment) — the theoretically optimal policy:

The oracle baseline has perfect knowledge of the crop model (WOFOST parameters,
thermal time constants, N recovery rates) and computes the best possible action
at every step from an internal tier-1 snapshot, independent of observability
tier. It serves as the upper-bound reference trajectory.

**Decision logic (priority order):**
1. Tracks DVS via exact internal state from the simulator snapshot
2. Tracks N-factor via exact internal state from the simulator snapshot
3. **Harvest** inside the optimal grading window 1.80–2.00, as late as safely possible
4. **Irrigate** when SM < 0.28 (or critical < 0.18) and no significant rain forecast
5. **Fertilize** at optimal DVS within windows [0.20–0.40] and [0.50–0.70]:
   - Waits for the step with DVS closest to target (0.30 / 0.60)
   - Computes exact N amount to fill n_factor to 1.0 (respects 50 kg/ha per-step cap)
6. **Wait** otherwise

**Oracle reference in metadata:**
At each non-terminal step, the environment calls the perfect-information oracle
on an internal tier-1 snapshot and stores
the result in `obs.metadata["oracle_action"]`.  This is **not visible** to the
LLM (not included in `compress_observation`) — it exists for offline analysis
and alignment measurement.

**Theoretical score ceilings:**
The oracle achieves ~98% of the theoretical maximum.  Five hard constraints
prevent reaching 1.0:

| Constraint | Impact | Mitigable? |
|------------|--------|------------|
| Post-anthesis N depletion (5× faster after DVS 1.0) | −2–5% yield | No — both fert windows are pre-anthesis |
| Grain shattering after DVS 1.85 | −1.25% yield at DVS 1.90 | Marginal — earlier harvest trades grain fill |
| Water/cost efficiency gated by yield_score in grader | −2–3% | No — grader design choice |
| 50 kg/ha per-step fertilizer cap | Split across 2 applications | No — hard environment cap |
| Heat stress (Punjab: 35°C threshold) | Up to −70% reproductive growth | No — environmental |

**Oracle scores (seed=42):**

| Task | Score | Timing | Yield | Cost |
|------|-------|--------|-------|------|
| 1 (NL, tier 1) | 0.9593 | 0.975 | 0.951 | 0.847 |
| 2 (Iowa, tier 2) | 0.9409 | 0.952 | 0.949 | 0.728 |
| 3 (Punjab, tier 3) | 0.9067 | 1.000 | 0.937 | 0.550 |

### 3.11 Step Reward Alignment with Oracle

The dense reward system (`server/reward.py`) is designed so that the oracle's
actions receive maximum reward:

- **Fertilizer dose:** The reward computes the ideal dose dynamically from
  `n_availability` and `N_RECOV` (how much N to fill n_factor to 1.0, capped
  at 50 kg/ha) — matching the oracle's calculation exactly.
- **Fertilizer timing:** Rewards peak at DVS 0.30 and 0.60 (the oracle's targets).
- **Harvest:** +0.20 reward in the [1.80, 2.00] DVS window.
- **Irrigation:** Rewards proportional to soil moisture deficit relief.
- **Advisory text:** Provides contextual hints (fertilizer window status, soil
  moisture vs optimal range, harvest readiness) without prescribing actions.

### 3.12 Training Adapter — `agent/training_adapter.py`

Discrete action vocabulary (8 buckets) for RL training:

| Action ID | Maps To | Amount |
|-----------|---------|--------|
| wait | wait | 0 |
| harvest | harvest | 0 |
| irrigate_small / medium / large | irrigate | 2 / 5 / 8 cm |
| fertilize_small / medium / large | fertilize | 15 / 30 / 50 kg |

### 3.13 Benchmark Sweep — `agent/benchmark_sweep.py`

Multi-seed evaluation utility that runs the oracle policy directly against `CropEnvironment` (no HTTP) for fast reproducible comparisons. Outputs per-task mean/std/min/max and verifies difficulty ordering.

---

## 4. Data Flow

### 4.1 Reset Pipeline

```
Agent calls reset(seed=42, task_id=1)
  │
  ▼
CropEnvironment.reset()
  ├── generate_scenario(seed=42, task_id=1)  ← scenarios.py
  │     ├── select location (Netherlands for task 1)
  │     ├── generate deterministic weather (seed × 31)
  │     ├── compute universal target yield
  │     └── return {weather, crop, soil, budget, target_yield, ...}
  ├── CropSimulator(crop_params, soil_params, weather)  ← crop_sim.py
  ├── _apply_start_state_overrides()  ← if probe scenario
  ├── _build_observation()  ← construct rich obs + 12 control features
  └── return CropObservation(done=False, reward=None)
```

### 4.2 Step Pipeline

```
Agent sends CropAction(action_type='irrigate', amount=2.5)
  │
  ▼
CropEnvironment.step(action)
  ├── Validate action (type, amount, budget)
  │     └── Degrade invalid/over-budget → wait
  ├── If inspect_soil / inspect_crop:
  │     ├── Check budget >= inspect cost
  │     ├── Deduct budget
  │     ├── Store report in CropState (persists)
  │     └── Return observation immediately (no sim advance)
  ├── compute_step_reward(action, state)  ← Intent reward BEFORE
  │     reward.py
  ├── sim.advance(7, irrigation_cm, n_kg_ha)  ← crop_sim.py
  │     ├── water balance (rain + irrig - ET)
  │     ├── stress factors (water, heat, nitrogen)
  │     ├── biomass growth (PAR × LUE × stress)
  │     ├── DVS advance (temperature sum)
  │     └── partitioning → grain yield
  ├── compute_delta_reward(before, after)  ← Delta reward AFTER
  │     reward.py
  ├── step_reward = 0.4 × intent + 0.6 × delta
  ├── Check termination:
  │     ├── harvest action → grade_episode() → blend 0.7×traj + 0.3×harvest_signal → terminal
  │     ├── 2 steps after DVS first hits 2.0 → auto-harvest (with shattering) → terminal
  │     ├── day ≥ max_duration → forced end → terminal
  │     └── step ≥ MAX_STEPS → safety cap → terminal
  ├── _build_observation()  ← new obs + control features
  └── return CropObservation(reward=step_reward, done=...)
```

### 4.3 Grading Pipeline (Terminal Step)

```
Episode termination triggered
  │
  ▼
grade_episode()  ← grader.py
  ├── yield_score = min(1.0, actual_yield / target_yield)
  ├── water_efficiency = max(0, 1 - total_water / 50)
  ├── cost_efficiency = max(0, 1 - total_cost / budget)
  ├── timing_quality = mean_proximity(fert_actions, DVS [0.3, 0.6])
  ├── harvest_timing = window([1.8, 2.00])
  └── return weighted sum → [0.0, 1.0]
         │
         ▼
  compute_trajectory_reward(grade)  ← reward.py
         │
         ▼
  Observation.reward = blended_reward (0.7 × trajectory + 0.3 × harvest_signal)
  Observation.done = True
  Observation.metadata = {rubric_breakdown: {...}}
```

---

## 5. Folder Structure

```
MetaHackathonPrep/
├── models.py                      # [OpenEnv required] Pydantic Action/Observation/State
├── client.py                      # [OpenEnv required] WebSocket EnvClient subclass
├── inference.py                   # [OpenEnv required] Competition inference script
├── openenv.yaml                   # [OpenEnv required] Environment metadata
├── Dockerfile                     # Container image for HuggingFace Spaces
├── pyproject.toml                 # Package configuration
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
│
├── server/                        # Server layer (OpenEnv environment + domain logic)
│   ├── __init__.py                # Package marker
│   ├── app.py                     # FastAPI entry point
│   ├── advisory.py                # Deterministic advisory text generator
│   ├── constants.py               # Shared numeric thresholds and weights
│   ├── crop_params.py             # WOFOST crop/soil parameter library + YAML loader
│   ├── crop_sim.py                # WOFOST-inspired crop simulator
│   ├── environment.py             # CropEnvironment (OpenEnv interface)
│   ├── scenarios.py               # Deterministic scenario/weather generation
│   ├── grader.py                  # FROZEN — deterministic 5-metric scoring
│   ├── rubric.py                  # RFC 004 rubric (CropManagementRubric)
│   ├── reward.py                  # Dense per-step + trajectory rewards
│   └── tasks.py                   # Task definitions (3 difficulty levels)
│
├── agent/                         # Agent-side code (policies, training, evaluation)
│   ├── __init__.py                # Package exports
│   ├── inference.py               # LLM + greedy fallback + oracle ceiling helpers
│   ├── training_adapter.py        # Discrete RL action vocabulary (8 buckets)
│   └── benchmark_sweep.py         # Multi-seed evaluation utility
│
├── docs/                          # Documentation
│   ├── ARCHITECTURE.md            # This document
│   ├── HACKATHON_MASTER.md        # Hackathon requirements synthesis & checklist
│   ├── REFERENCES.md              # Scientific references (WOFOST, Boogaard et al.)
│   ├── SUBMISSION_READINESS.md    # Pre-submission compliance report
│   └── HackathonSubmissionUpdates # Submission feedback log
│
├── configs/                       # YAML crop/soil profiles (override hardcoded defaults)
│   ├── wheat_nl.yaml
│   ├── wheat_iowa.yaml
│   └── wheat_punjab.yaml
│
├── examples/                      # Runnable examples
│   ├── direct_benchmark.py        # Fast local benchmark (no HTTP)
│   └── client_greedy_run.py       # WebSocket client example
│
├── tests/                         # Test suite
│   ├── test_smoke.py              # Smoke + RL + rubric/weather tests (64 tests)
│   ├── test_integration.py        # HTTP endpoint integration tests (7 tests)
│   ├── test_submission_surface.py # Competition format compliance tests (6 tests)
│   └── test_ws_episode.py         # Real WebSocket transport tests (3 tests)
│
├── ProblemDetails                 # Problem statement (file)
├── Samples                        # Reference env samples (file)
```

### Separation of Concerns Map

| Concern | Files | Layer |
|---------|-------|-------|
| Data contracts | `models.py` | Data Models |
| Network transport | `client.py`, `server/app.py` | Client / Server |
| OpenEnv interface | `server/environment.py` | Server |
| Crop physics | `server/crop_sim.py`, `server/crop_params.py` | Simulation Domain |
| Advisory text | `server/advisory.py` | Simulation Domain |
| Shared constants | `server/constants.py` | Simulation Domain |
| Weather & scenarios | `server/scenarios.py` | Simulation Domain |
| Task configuration | `server/tasks.py` | Simulation Domain |
| Episode scoring | `server/grader.py` | Simulation Domain |
| Per-step rewards | `server/reward.py` | Simulation Domain |
| AI policies | `agent/inference.py` | Agent |
| RL training adapter | `agent/training_adapter.py` | Agent |
| Policy evaluation | `agent/benchmark_sweep.py` | Agent |
| Container deployment | `Dockerfile` (root) | Infrastructure |

---

## 6. OpenEnv Compliance

### Required Scaffold (must not move)

| File | OpenEnv Role |
|------|-------------|
| `models.py` at root | `openenv.yaml` → `models: models` |
| `client.py` at root | `openenv.yaml` → `client: client` |
| `server/app.py` | `openenv.yaml` → `server: server.app:app` |
| `Dockerfile` (root) | Container build target |
| `openenv.yaml` | Framework metadata |

### Endpoints (auto-registered by `create_app`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check → `{"status": "healthy"}` |
| `/reset` | POST | Start new episode |
| `/step` | POST | Execute action |
| `/state` | GET | Current environment state |
| `/ws` | WebSocket | Multi-step episode (preferred) |
| `/tasks` | GET | Custom: list task definitions |
| `/grader` | POST | Custom: grade an episode → `{score, breakdown}` |
| `/baseline` | GET | Custom: greedy baseline scores for all tasks (seed=42, cached) |
| `/ceiling` | GET | Custom: oracle ceiling scores for all tasks (seed=42, cached) |

### Compliance Checklist

- [x] Pydantic models inherit from OpenEnv base types (Action, Observation, State)
- [x] Environment subclasses `Environment[CropAction, CropObservation, CropState]`
- [x] `create_app()` used for server creation
- [x] Deterministic: same seed + same actions = same result
- [x] 3+ tasks with graders
- [x] Normalized scores [0.0, 1.0]
- [x] Docker image builds and runs
- [x] Port 7860 for HF Spaces
- [x] `openenv validate` passes
- [x] `=== RESULTS ===` stdout format in inference

---

## 7. Dependency Graph

```
models.py ← (no internal deps)
  ▲
  │
  ├── client.py ← models
  │
  ├── server/app.py ← models, server/environment, server/tasks
  │
│  ├── server/environment.py ← models, server/constants, server/crop_sim,
  │                            server/advisory, server/rubric,
  │                            server/reward, server/scenarios, server/tasks
  │
  ├── server/rubric.py ← server/grader
  │
  ├── server/crop_sim.py ← server/crop_params (standalone simulator)
  │
  ├── server/crop_params.py ← (no internal deps; WOFOST parameter library)
  │
  ├── server/advisory.py ← (no internal deps; template generator)
  │
  ├── server/constants.py ← (no internal deps; shared thresholds)
  │
  ├── server/scenarios.py ← server/crop_sim, server/crop_params
  │
  ├── server/tasks.py ← (no internal deps)
  │
  ├── server/grader.py ← (no internal deps)
  │
  ├── server/reward.py ← (no internal deps)
  │
  ├── agent/inference.py ← models, client
  │
  ├── agent/training_adapter.py ← models
  │
  └── agent/benchmark_sweep.py ← models, server/environment,
                                  agent/inference (oracle_action)
```

---

## 8. Known Limitations & Improvement Areas

### Current Architecture Limitations

1. **`server/` is a flat directory** — mixes environment interface (`environment.py`) and domain logic (`crop_sim.py`, `scenarios.py`, `grader.py`, `reward.py`, `tasks.py`). Acceptable at current size but should be sub-packaged if more simulation modules are added (e.g., PCSE integration).

2. **`agent/inference.py` has multiple responsibilities** — LLM policy, greedy heuristic, trajectory export, and episode orchestration in one file. A future split into `llm_policy.py`, `greedy_policy.py`, `trajectory.py`, and `runner.py` would improve testability.

3. **Configuration partially externalized** — crop/soil parameters now live in `server/crop_params.py` (frozen dataclasses) with YAML overrides in `configs/`. Task definitions and weather parameters remain in code. Further extraction is possible but not currently needed.

4. **`benchmark_sweep.py` crosses the layer boundary** — imports directly from `server/environment.py` for fast local evaluation. This is an intentional performance trade-off (no HTTP overhead) but breaks the clean Agent→Client→Server dependency chain.

### Documented Improvement Plans

- Potential future migration from custom simulator to PCSE/WOFOST library for higher biophysical fidelity

### Key Metrics (Current Baseline)

| Task | Greedy Score (seed=42) | Oracle Score (seed=42) |
|------|----------------------|----------------------|
| 1 (Easy) | 0.7464 | 0.9593 |
| 2 (Medium) | 0.5515 | 0.9409 |
| 3 (Hard) | 0.3143 | 0.8769 |
| **Overall** | **0.5374** | **0.9257** |

---

## 9. Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Pure-Python simulator** | No PCSE dependency → smaller Docker image, faster startup, deterministic by construction |
| **Universal target yield** | `target_yield = max(potential across all locations)` → yield_score never exceeds 1.0, difficulty ordering preserved |
| **Same grading weights for all tasks** | Difficulty from environment conditions, not scoring manipulation |
| **Dense step rewards** | Intent + delta split gives RL agents per-step signal, not just sparse terminal reward |
| **5 probe scenarios** | Internal RL diagnostics without changing public task definitions |
| **Discrete action adapter** | 8-bucket vocabulary for RL exploration; public CropAction schema stays continuous |
| **WebSocket-first for inference** | HTTP endpoints are stateless per-request; WebSocket maintains session for multi-step episodes |
| **Greedy heuristic as fallback** | Ensures inference always produces valid results even if LLM API fails |
