# Architecture — Precision Agriculture Crop Management

> **Last updated:** April 3, 2026
> **Version:** 1.0 — initial architecture document

---

## 1. System Overview

This project is a deterministic, multi-step precision agriculture environment built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework. An AI agent manages a wheat growing season—deciding weekly when to irrigate, fertilize, and harvest to maximize yield while minimizing water use and cost.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AGENT LAYER                                      │
│  agent/inference.py  ── LLM policy + greedy heuristic + orchestration  │
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
| `CropObservation` | `Observation` | 12 nested dicts: crop/soil/weather/resources/control_features/conflicts + task context | Rich observation for LLM or RL |
| `CropState` | `State` | Full simulator state + episode tracking | Serializable checkpoint |

All models use Pydantic `BaseModel` with `extra="forbid"` and inherit from OpenEnv base types.

### 3.2 Server — `server/app.py`

- Creates the FastAPI app via `create_app(CropEnvironment, CropAction, CropObservation)`
- Auto-registers: `/health`, `/reset`, `/step`, `/state`, `/ws`, `/docs`, `/web`
- Custom endpoint: `/tasks` — lists available task definitions
- Entry point: `uvicorn server.app:app --host 0.0.0.0 --port 8000`

### 3.3 Environment Interface — `server/environment.py`

| Method | Responsibility |
|--------|---------------|
| `reset(seed, task_id)` | Load scenario → create `CropSimulator` → apply probe overrides → return initial observation |
| `step(action)` | Validate action → compute intent reward → advance sim 7 days → compute delta reward → check termination → return observation |
| `state` (property) | Return serializable `CropState` |

**Termination conditions** (checked in order):
1. Agent sends `harvest` action → terminal, compute final grade
2. DVS ≥ 2.0 → auto-harvest maturity
3. Day ≥ max_duration → forced season end
4. Step ≥ MAX_STEPS (60) → safety cap

### 3.4 Simulation Engine — `server/crop_sim.py`

Pure-Python WOFOST-inspired model (~250 LOC):

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

**Key method:** `advance(days, irrigation_cm, n_kg_ha)` — advances the model by `days` time-steps, applying interventions.

**Data libraries** (module-level dicts):
- `CROP_LIBRARY` — wheat params: tsum1=1100, tsum2=1000, lue=2.5
- `SOIL_LIBRARY` — clay_loam, sandy_loam, silt_loam
- `PARTITION_TABLES` — DVS→grain_fraction tuples

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
| harvest_timing | 1.0 if DVS ∈ [1.8, 2.05], penalty otherwise | [0, 1] |

`target_yield` = max potential yield across all 3 locations for that seed (universal target).

### 3.8 Dense Rewards — `server/reward.py`

Three-tier reward architecture:

| Tier | Function | Range | When |
|------|----------|-------|------|
| **Intent** | `compute_step_reward()` | [−0.3, +0.2] | Before advancing sim |
| **Delta** | `compute_delta_reward()` | [−0.15, +0.15] | After advancing sim |
| **Trajectory** | `compute_trajectory_reward()` | [0.0, 1.0] | At episode end (= final grade) |

Step reward blend: `0.4 × intent + 0.6 × delta`

### 3.9 Client — `client.py`

`CropEnvClient` extends OpenEnv's `EnvClient[CropAction, CropObservation, CropState]`:
- Handles WebSocket serialization/deserialization
- Methods: `reset()`, `step()`, `state`
- Usage: `CropEnvClient(base_url="...").sync()` for synchronous operation

### 3.10 Agent — `agent/inference.py`

Three policy paths:
1. **LLM-only** — if HF_TOKEN + API_BASE_URL + MODEL_NAME are set
2. **LLM + fallback** — on LLM errors, falls back to greedy (auto-disables after 3 consecutive failures)
3. **Greedy heuristic only** — default when no LLM credentials

**Greedy heuristic** (`greedy_action()`):
1. Harvest when DVS ≥ 1.8
2. Irrigate when SM < 0.22 AND no rain forecast (or SM < 0.18 critical)
3. Fertilize at DVS [0.27–0.40] and [0.57–0.70] (once per stage)
4. Wait otherwise

Includes trajectory export (JSONL) for offline RL.

### 3.11 Training Adapter — `agent/training_adapter.py`

Discrete action vocabulary (8 buckets) for RL training:

| Action ID | Maps To | Amount |
|-----------|---------|--------|
| wait | wait | 0 |
| harvest | harvest | 0 |
| irrigate_small / medium / large | irrigate | 2 / 5 / 8 cm |
| fertilize_small / medium / large | fertilize | 15 / 30 / 50 kg |

### 3.12 Benchmark Sweep — `agent/benchmark_sweep.py`

Multi-seed evaluation utility that runs the greedy policy directly against `CropEnvironment` (no HTTP) for fast reproducible comparisons. Outputs per-task mean/std/min/max and verifies difficulty ordering.

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
  │     ├── harvest action → grade_episode() → terminal
  │     ├── DVS ≥ 2.0 → auto-harvest → terminal
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
  ├── harvest_timing = window([1.8, 2.05])
  └── return weighted sum → [0.0, 1.0]
         │
         ▼
  compute_trajectory_reward(grade)  ← reward.py
         │
         ▼
  Observation.reward = trajectory_reward (overrides step reward)
  Observation.done = True
  Observation.metadata = {grade_breakdown: {...}}
```

---

## 5. Folder Structure

```
MetaHackathonPrep/
├── models.py                      # [OpenEnv required] Pydantic Action/Observation/State
├── client.py                      # [OpenEnv required] WebSocket EnvClient subclass
├── openenv.yaml                   # [OpenEnv required] Environment metadata
├── pyproject.toml                 # Package configuration
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
│
├── server/                        # Server layer (OpenEnv environment + domain logic)
│   ├── __init__.py                # Package marker
│   ├── app.py                     # FastAPI entry point
│   ├── environment.py             # CropEnvironment (OpenEnv interface)
│   ├── crop_sim.py                # WOFOST-inspired crop simulator
│   ├── scenarios.py               # Deterministic scenario/weather generation
│   ├── tasks.py                   # Task definitions (3 difficulty levels)
│   ├── grader.py                  # FROZEN — deterministic 5-metric scoring
│   ├── reward.py                  # Dense per-step + trajectory rewards
│   └── Dockerfile                 # Container for HuggingFace Spaces
│
├── agent/                         # Agent-side code (policies, training, evaluation)
│   ├── __init__.py                # Package exports
│   ├── inference.py               # LLM + greedy policy + episode orchestration
│   ├── training_adapter.py        # Discrete RL action vocabulary (8 buckets)
│   └── benchmark_sweep.py         # Multi-seed evaluation utility
│
├── docs/                          # Documentation
│   ├── ARCHITECTURE.md            # This document
│   ├── hackathonBriefing.md       # Bootcamp alignment & checklist
│   ├── IMPROVEMENT_PLAN.md        # Enhancement roadmap (Phases A–C)
│   └── FUTURE_SCOPE_PCSE.md       # PCSE migration plan
│
├── examples/                      # Runnable examples
│   ├── direct_benchmark.py        # Fast local benchmark (no HTTP)
│   └── client_greedy_run.py       # WebSocket client example
│
├── tests/                         # Test suite
│   └── test_smoke.py              # 33 smoke + RL-focused tests
│
├── Preparation/                   # Hackathon prep materials
├── ProblemDetails/                # Problem statement
├── Samples/                       # Reference env samples
└── studymaterialLinks/            # OpenEnv tutorial links
```

### Separation of Concerns Map

| Concern | Files | Layer |
|---------|-------|-------|
| Data contracts | `models.py` | Data Models |
| Network transport | `client.py`, `server/app.py` | Client / Server |
| OpenEnv interface | `server/environment.py` | Server |
| Crop physics | `server/crop_sim.py` | Simulation Domain |
| Weather & scenarios | `server/scenarios.py` | Simulation Domain |
| Task configuration | `server/tasks.py` | Simulation Domain |
| Episode scoring | `server/grader.py` | Simulation Domain |
| Per-step rewards | `server/reward.py` | Simulation Domain |
| AI policies | `agent/inference.py` | Agent |
| RL training adapter | `agent/training_adapter.py` | Agent |
| Policy evaluation | `agent/benchmark_sweep.py` | Agent |
| Container deployment | `server/Dockerfile` | Infrastructure |

---

## 6. OpenEnv Compliance

### Required Scaffold (must not move)

| File | OpenEnv Role |
|------|-------------|
| `models.py` at root | `openenv.yaml` → `models: models` |
| `client.py` at root | `openenv.yaml` → `client: client` |
| `server/app.py` | `openenv.yaml` → `server: server.app:app` |
| `server/Dockerfile` | Container build target |
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
  ├── server/environment.py ← models, server/crop_sim, server/grader,
  │                            server/reward, server/scenarios, server/tasks
  │
  ├── server/crop_sim.py ← (no internal deps; standalone simulator)
  │
  ├── server/scenarios.py ← server/crop_sim
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
                                  agent/inference (greedy_action)
```

---

## 8. Known Limitations & Improvement Areas

### Current Architecture Limitations

1. **`server/` is a flat directory** — mixes infrastructure (`app.py`, `Dockerfile`), environment interface (`environment.py`), and domain logic (`crop_sim.py`, `scenarios.py`, `grader.py`, `reward.py`, `tasks.py`). Acceptable at current size but should be sub-packaged if more simulation modules are added (e.g., PCSE integration).

2. **`agent/inference.py` has multiple responsibilities** — LLM policy, greedy heuristic, trajectory export, and episode orchestration in one file. A future split into `llm_policy.py`, `greedy_policy.py`, `trajectory.py`, and `runner.py` would improve testability.

3. **Configuration embedded in code** — crop/soil parameter libraries, task definitions, and weather parameters are hardcoded as module-level dicts. Extracting to YAML/JSON configuration files would improve extensibility.

4. **`benchmark_sweep.py` crosses the layer boundary** — imports directly from `server/environment.py` for fast local evaluation. This is an intentional performance trade-off (no HTTP overhead) but breaks the clean Agent→Client→Server dependency chain.

### Documented Improvement Plans

- **[IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md)** — Phases A–C: reward alignment, heuristic lift, bug fixes
- **[FUTURE_SCOPE_PCSE.md](FUTURE_SCOPE_PCSE.md)** — Migration from custom simulator to PCSE/WOFOST library

### Key Metrics (Current Baseline)

| Task | Greedy Score (seed=42) |
|------|----------------------|
| 1 (Easy) | 0.8442 |
| 2 (Medium) | 0.8155 |
| 3 (Hard) | 0.7046 |
| **Overall** | **0.7881** |

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
