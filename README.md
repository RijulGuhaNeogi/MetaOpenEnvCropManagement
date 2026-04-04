---
title: Crop Management Environment
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# Precision Agriculture Crop Management — OpenEnv

> **Meta PyTorch OpenEnv Hackathon — Round 1 Submission**
> **Team Hijibiji** — Roudraneel, Rijul, Tirthajoti

A deterministic, multi-step precision agriculture environment built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework. An AI agent manages a wheat growing season — deciding weekly when to irrigate, fertilize, and harvest to maximize yield while minimizing water use and cost.

The current environment is optimized for both public-task grading quality and RL learnability: dense rewards are post-transition aware, observations include policy-native control features, and internal probe scenarios exist for diagnosing reward/behavior failures without changing the public tasks.

---

## Why Precision Agriculture?

Precision agriculture is a $12B+ industry where farmers make weekly decisions about irrigation, fertilization, and harvest timing under uncertainty (weather, soil conditions, budgets). This environment models that exact decision process:

- **Real-world task:** Every farmer does this — the environment simulates decisions that affect millions of hectares of cropland globally
- **Multi-objective optimization:** Maximizing yield conflicts with minimizing water/cost — a genuine Pareto frontier problem
- **Weather uncertainty:** Agents must read forecasts and plan ahead, not just react
- **Novel domain:** No existing crop management environment in the OpenEnv ecosystem

The WOFOST-inspired crop growth simulator captures real agricultural dynamics: temperature-sum phenology, water stress, nitrogen response, and biomass partitioning.

---

## Architecture

```
+-------------+     WebSocket      +------------------+
|   agent/    | <---------------> |  server/app.py   |
| inference.py |                   |  (FastAPI server) |
+-------------+                    +--------+---------+
       |                                    |
       v                                    v
+-------------+                    +------------------+
|  client.py  |                    |  environment.py  |
| (WebSocket  |                    |  (CropEnvironment)|
|  client)    |                    +---+----+----+----+
+-------------+                        |    |    |
                                       v    v    v
                              crop_sim  grader  reward
                              scenarios tasks
```

**No external dependencies or API calls at runtime.** Weather data is generated deterministically from the seed. The crop growth simulator is a lightweight pure-Python implementation (~200 lines) inspired by the WOFOST model.

---

## Tasks

| ID | Name | Difficulty | Location | Budget | Key Challenge |
|----|------|------------|----------|--------|---------------|
| 1 | Basic Crop Growth | Easy | Netherlands | $800 | Fertilize at right growth stages, harvest at maturity |
| 2 | Water-Efficient Farming | Medium | Iowa, USA | $450 | Balance yield vs water conservation under variable weather |
| 3 | Precision Agriculture | Hard | Punjab, India | $300 | Maximize yield + minimize water + stay in budget in drought conditions |

**Why Task 3 is genuinely hard:**
- Punjab has minimal rainfall during the wheat season — irrigation is essential but expensive
- Budget is tight ($300) — every irrigation/fertilization decision must be justified
- Scoring weights yield (35%), water efficiency (20%), cost efficiency (18%), timing (15%), and harvest timing (12%) — no single strategy dominates
- Target yield is the *universal* maximum potential production across all three locations — Task 3 creates genuine planning tension because the tight budget forces agents to prioritize which interventions matter most, and harder locations score lower by design

---

## Action Space

```json
{
  "action_type": "irrigate",
  "amount": 2.5
}
```

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `action_type` | str | `irrigate`, `fertilize`, `harvest`, `wait` | Weekly management decision |
| `amount` | float | 0-10 cm (irrigate), 0-50 kg N/ha (fertilize), 0 (others) | Resource amount |

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `day` | int | Current simulation day |
| `days_remaining` | int | Days left in the season |
| `crop_status` | `CropStatus` | DVS (development stage 0→2), LAI, biomass, yield, growth stage name |
| `soil_status` | `SoilStatus` | Soil moisture, water deficit flag, water stress (0-1), N availability (0-1), field capacity, wilting point |
| `weather_today` | `WeatherDay` | Temperature (max/min), rainfall, radiation |
| `weather_forecast` | `list[WeatherDay]` | 5-day forecast (with slight noise for realism) |
| `resources_used` | `ResourcesUsed` | Total water, nitrogen, cost, budget remaining, unit costs |
| `season_summary` | dict | Crop name, location, target yield, budget, step size |
| `control_features` | `ControlFeatures` | Derived RL-facing features (see below) |
| `advisory_text` | str \| None | Deterministic factual summary of current crop state |
| `conflicts` | list[str] | Feedback on invalid actions |

All typed sub-models (`CropStatus`, `SoilStatus`, `ResourcesUsed`, `ControlFeatures`) are Pydantic models with explicit field types, providing IDE autocomplete and serialization safety.

### Key control features

- `moisture_gap_to_target`
- `forecast_rain_3d`
- `forecast_rain_7d`
- `days_since_last_irrigation`
- `days_since_last_fertilization`
- `fertilizer_events_count`
- `cumulative_n_applied`
- `budget_remaining_ratio`
- `rooting_depth_cm`
- `dvs_distance_to_next_fertilizer_window`
- `estimated_budget_to_finish`

---

## Grading

All grading is **deterministic** — same inputs always produce the same score (0.0–1.0).

### Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| `yield_score` | `min(1.0, actual_yield / target_yield)` | Did the crop grow? |
| `water_efficiency` | `1.0 - water_used / 50cm` | Less water = better |
| `cost_efficiency` | `1.0 - cost / budget` | Under-budget = better |
| `timing_quality` | Distance to DVS 0.3/0.6 targets | Fertilize at right stages |
| `harvest_timing` | Penalty for early/late harvest | Harvest at DVS 1.8-2.0 |

### Unified Weights (same for all tasks)

Difficulty comes from **environment conditions** (climate, budget, soil), not from different scoring weights. The same formula is used for all tasks:

| yield | water | cost | timing | harvest |
|-------|-------|------|--------|--------|
| 35% | 20% | 18% | 15% | 12% |

### Step Rewards

Dense per-step signals are split into:

- **Intent reward:** evaluates whether the action is agronomically sensible before transition
- **Delta reward:** evaluates whether the action actually relieved stress or wasted resources after transition

This reward breakdown is exposed in observation metadata for diagnostics and offline RL.

Current shaping highlights:

- **Irrigation**
  - rewards closing the moisture deficit toward the target band around 0.30
  - penalizes overshoot, irrigating ahead of forecast rain, cumulative water waste, and expensive low-impact irrigation
- **Fertilization**
  - rewards both correct DVS timing and sensible nitrogen dose
  - peaks near DVS 0.30 and 0.60 rather than rewarding the full window equally
  - penalizes excess seasonal nitrogen and late ineffective application
- **Harvest**
  - rewards harvesting in the maturity window around DVS 1.8–2.05
  - penalizes early and late harvest
- **Wait**
  - remains neutral to avoid lazy-wait exploitation

### Rubric System (RFC 004)

The environment follows the [OpenEnv RFC 004](https://github.com/meta-pytorch/OpenEnv/blob/main/rfcs/004-rubrics.md) convention for delayed rewards:

- **`obs.reward`** — Dense per-step signal (intent + delta blend) for RL training
- **`obs.rubric_reward`** — Trajectory-level score from the grader at terminal steps; `None` on intermediate steps
- **`obs.metadata["rubric_breakdown"]`** — Per-metric scores at terminal step (yield, water, cost, timing, harvest)

This separation lets RL frameworks (TRL, GRPO, torchforge) distinguish immediate feedback from episode-level evaluation without ambiguity.

The rubric is provided by `CropManagementRubric` (in `server/rubric.py`), a thin wrapper around the deterministic grader.

---

## Baseline Scores (Greedy Heuristic, seed=42)

| Task | Score |
|------|-------|
| 1 (Easy) | 0.8689 |
| 2 (Medium) | 0.8242 |
| 3 (Hard) | 0.6776 |
| **Overall** | **0.7902** |

These scores are produced by the current greedy heuristic, which uses deficit-based irrigation, WOFOST-calibrated fertilizer timing/amounts, and constants shared with the reward module.

---

## Project Structure

```
MetaHackathonPrep/
├── agent/
│   ├── __init__.py         # Agent package marker
│   ├── inference.py        # Greedy heuristic + LLM inference + optional trajectory export
│   ├── training_adapter.py # Discrete RL action adapter for training-only workflows
│   └── benchmark_sweep.py  # Reusable multi-seed greedy benchmark utility
├── docs/
│   ├── ARCHITECTURE.md     # Comprehensive architecture document
│   ├── hackathonBriefing.md# Bootcamp alignment & checklist
│   ├── FUTURE_SCOPE_PCSE.md# PCSE migration plan
│   └── REFERENCES.md       # Scientific references (WOFOST, Boogaard et al.)
├── examples/
│   ├── direct_benchmark.py # Minimal direct-environment benchmark example
│   └── client_greedy_run.py# Minimal WebSocket client example
├── configs/
│   ├── wheat_nl.yaml       # Netherlands wheat WOFOST profile
│   ├── wheat_iowa.yaml     # Iowa wheat WOFOST profile
│   └── wheat_punjab.yaml   # Punjab wheat WOFOST profile
├── server/
│   ├── __init__.py         # Package marker
│   ├── app.py              # FastAPI server via create_app() + /tasks, /grader, /baseline
│   ├── advisory.py         # Deterministic advisory text generator
│   ├── constants.py        # Shared numeric thresholds and weights
│   ├── crop_params.py      # WOFOST crop/soil parameter library + YAML loader
│   ├── crop_sim.py         # WOFOST-inspired crop growth simulator
│   ├── environment.py      # CropEnvironment (reset/step/state)
│   ├── grader.py           # Multi-metric deterministic scoring
│   ├── reward.py           # Dense step + trajectory reward computation
│   ├── rubric.py           # RFC 004 rubric (CropManagementRubric)
│   ├── scenarios.py        # Seeded weather + scenario generator (3 locations)
│   └── tasks.py            # Task definitions (3 difficulty levels)
├── tests/
│   ├── test_smoke.py       # Smoke + RL + rubric tests
│   ├── test_integration.py # HTTP endpoint integration tests
│   └── test_ws_episode.py  # WebSocket full-episode tests
├── models.py               # CropAction, CropObservation, CropState
├── client.py               # WebSocket EnvClient subclass
├── inference.py            # Competition inference script (root entrypoint)
├── openenv.yaml            # OpenEnv environment metadata
├── Dockerfile              # Docker image for HuggingFace Spaces
├── pyproject.toml          # Package configuration
├── requirements.txt        # Python dependencies
├── .env                    # Local env vars (API keys — git-ignored)
├── .env.example            # Documented env var template
├── .gitignore              # Git ignore rules
├── .dockerignore           # Docker build exclusions
└── README.md               # This file
```

---

## Usage Paths

The repo supports three primary workflows:

- **Inference path** — run the full WebSocket client plus heuristic or LLM policy against a live server
- **Direct benchmark path** — evaluate the built-in greedy policy directly against the environment without starting the server
- **Test path** — validate determinism, reward behavior, and policy regressions via the smoke suite

Use the inference path when you want end-to-end OpenEnv behavior, the direct benchmark path when you want fast reproducible comparisons, and the test path when you want regression protection.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full architecture documentation, data flow diagrams, and layer model.

---

## Setup & Usage

### Prerequisites

- Python 3.11+
- pip

### Local Installation

```bash
pip install -r requirements.txt
```

### Start the Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Verify

```bash
curl http://localhost:8000/health
# {"status": "healthy"}

curl http://localhost:8000/tasks
# {"tasks": [{"id": 1, "name": "Basic Crop Growth", ...}, ...]}
```

---

## Running Inference

### With LLM (competition mode)

Create a `.env` file (auto-loaded by `python-dotenv`):

```dotenv
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
HF_TOKEN=hf_your_token_here
```

Then run:

```bash
python inference.py
```

Run a single task:

```bash
TASK_ID=1 python inference.py
```

Optional trajectory export:

```bash
TRAJECTORY_OUTPUT=trajectories/run python inference.py
```

This writes JSONL transitions with observation, action, reward, next observation, done flag, and metadata for offline RL or imitation-learning bootstrapping.

Or export the variables directly:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token_here"
python inference.py
```

> **Note:** If the LLM API returns 3+ consecutive errors (e.g. credit exhaustion), inference automatically falls back to the greedy heuristic for the rest of the episode and prints a warning at the end.

### Without LLM (heuristic baseline)

```bash
python inference.py
```

Minimal client example against a running server:

```bash
python examples/client_greedy_run.py --base-url http://localhost:8000 --task-id 1 --seed 42
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | For LLM mode | LLM API endpoint (OpenAI-compatible) |
| `MODEL_NAME` | For LLM mode | Model identifier |
| `HF_TOKEN` | For LLM mode | HuggingFace / API authentication token |
| `ENV_URL` | No | Server URL (default: `http://localhost:8000`) |
| `TASK_ID` | No | Run a single task by ID (default: all 3) |
| `SEED` | No | Random seed for reproducibility (default: 42) |
| `TRAJECTORY_OUTPUT` | No | Optional JSONL path prefix for transition export |

All variables can be set in a `.env` file in the project root (auto-loaded via `python-dotenv`).

---

## Deployment

### Hugging Face Spaces (Recommended)

```bash
# From the project directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --repo-id your-username/crop-management --private
```

The deployed Space provides:
- Web Interface at `/web`
- API Documentation at `/docs`
- Health Check at `/health`
- WebSocket at `/ws` for persistent sessions

### Local Docker

```bash
docker build -t crop-management .
docker run -p 7860:7860 crop-management
ENV_URL=http://localhost:7860 python inference.py
```

### Quick Baseline Check

```bash
# After server is running:
curl http://localhost:8000/baseline
# Returns deterministic greedy scores for all 3 tasks
```

## Testing

```bash
python -m pytest tests/test_smoke.py -q
```

Direct-environment benchmark sweep:

```bash
python -m agent.benchmark_sweep --start-seed 42 --count 10
```

Optional JSON output for scripting:

```bash
python -m agent.benchmark_sweep --start-seed 42 --count 10 --json
```

Optional JSON file export:

```bash
python -m agent.benchmark_sweep --start-seed 42 --count 10 --output benchmarks/sweep_42_10.json
```

Minimal direct benchmark example:

```bash
python examples/direct_benchmark.py
```

### Interpreting Single-Seed vs Multi-Seed Results

- **Single-seed result** means evaluating one deterministic scenario instance, such as seed 42
- **Multi-seed sweep** means evaluating the same policy across many deterministic scenario instances and summarizing mean, spread, and ordering

Use a single seed for quick local checks and regressions. Use a multi-seed sweep to judge policy stability and difficulty calibration.

In the current benchmark, Task 1 is very stable, Task 2 has moderate scenario sensitivity, and Task 3 has the largest variance because weather and budget pressure interact more strongly there. That variance is expected; what matters is that difficulty ordering remains stable on aggregate.

Current test coverage includes:

- environment reset/step/state basics
- determinism and difficulty ordering
- control feature presence
- reward monotonicity for irrigation and fertilizer
- delta-reward sanity for stress relief
- neutral wait behavior
- reward breakdown metadata
- internal probe scenario loading
- discrete training adapter mapping
- budget exhaustion handling
- directional alignment between probe dense reward and terminal grade
- passive-policy and extra-fertilizer regression checks
- late-harvest boundary regression checks

The full test suite has **51 passing tests** (41 smoke/rubric/weather + 7 HTTP integration + 3 real WebSocket transport).

## Limitations

This environment is intentionally **WOFOST-inspired**, not a full scientific crop model.

- It models one crop family used in the public benchmark and does not attempt crop rotation or multi-season farm planning.
- Root growth, pests, disease, salinity, and management operations beyond irrigation, fertilization, harvest, and wait are intentionally omitted.
- Weather is deterministic by seed and forecast noise is deterministic by day; this improves reproducibility and RL debugging at the cost of richer stochastic realism.
- The simulator is tuned for clear reward signal and grading stability, not for exact agronomic calibration to a specific field trial dataset.

Those tradeoffs are intentional. The benchmark is optimized for deterministic evaluation, transparent grading, and trainable sequential decision-making rather than maximum biophysical fidelity.

## Internal RL Utilities

These are useful for RL development but do not change the public task interface:

- **Probe scenarios** via `reset(..., probe_name=...)`
  - `over_irrigation_trap`
  - `late_fertilizer_temptation`
  - `budget_starvation`
  - `harvest_hesitation`
  - `drought_rescue`
- **Discrete action adapter** in `agent/training_adapter.py`
  - `wait`
  - `harvest`
  - `irrigate_small`, `irrigate_medium`, `irrigate_large`
  - `fertilize_small`, `fertilize_medium`, `fertilize_large`

These utilities are intended for diagnostics and training convenience only. The public OpenEnv action schema remains `action_type + amount`.

---

## Crop Growth Model

The simulator implements key dynamics from the WOFOST (World Food Studies) model:

- **Phenology:** Temperature-sum driven development stages (DVS 0→1 vegetative, 1→2 reproductive)
- **Biomass:** Light-use-efficiency model with PAR interception via LAI
- **Water balance:** Rainfall + irrigation - evapotranspiration (Hargreaves ET)
- **Water stress:** Reduces growth when soil moisture drops below threshold
- **Heat stress:** Extreme heat sharply penalizes flowering and sustained heat mildly penalizes grain fill, with bounded deterministic effects strongest in hot scenarios like Punjab
- **Nitrogen response:** Fertilization increases growth factor (0.3→1.0); phenology-aware depletion (slow pre-anthesis, fast post-anthesis)
- **Partitioning:** DVS-dependent allocation to storage organs (grain yield)
- **Senescence:** Realistic LAI decline after DVS > 1.5 (~7-10 day period)

This is a **WOFOST-inspired** simulator, not the PCSE reference implementation. That tradeoff is intentional: the current environment favors deterministic behavior, transparent grading, small deployment footprint, and strong RL signal quality over additional simulator complexity.

Three climatic profiles with deterministic weather generation:
- **Netherlands:** Mild maritime, reliable rainfall (easy)
- **Iowa, USA:** Continental, moderate with dry spells (medium)
- **Punjab, India:** Hot semi-arid, minimal winter rain (hard)

### WOFOST Parameter Sources

All crop and soil parameters are sourced from published WOFOST literature:

- **Crop parameters:** Boogaard et al. (2014) *WOFOST Data Set*, de Wit et al. (2019) *25 Years of WOFOST*
- **Soil parameters:** ISRIC World Soil Information, FAO soil classification
- **Thermal sums:** van Diepen et al. (1989) original WOFOST specification
- **Partitioning tables (FOTB):** de Wit et al. (2019), Table 4

Parameters are stored in `server/crop_params.py` (frozen dataclasses with inline citation tags) and can be overridden via YAML configs in the `configs/` directory. See [docs/REFERENCES.md](docs/REFERENCES.md) for the full reference list.

### YAML Configuration

Crop and soil profiles can be customized via YAML files in `configs/`:

```yaml
# configs/wheat_nl.yaml
crop:
  tsum1: 1100
  tsum2: 1000
  lue: 2.8
  # ... full WOFOST parameter set
soil:
  SMFCF: 0.36
  SMW: 0.10
  RDMSOL: 1200
```

The environment loads YAML configs first, falling back to hardcoded profiles if no matching YAML exists. To add a new crop profile, create a YAML file and reference it from `server/scenarios.py`.

---

Built for the [Meta PyTorch OpenEnv Hackathon](https://github.com/meta-pytorch/OpenEnv) — Round 1.
