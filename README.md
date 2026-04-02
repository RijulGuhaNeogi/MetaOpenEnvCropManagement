# Precision Agriculture Crop Management — OpenEnv

> **Meta PyTorch OpenEnv Hackathon — Round 1 Submission**
> **Team Hijibiji** — Roudraneel, Rijul, Tirthajoti

A deterministic, multi-step precision agriculture environment built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework. An AI agent manages a wheat growing season — deciding weekly when to irrigate, fertilize, and harvest to maximize yield while minimizing water use and cost.

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
| inference.py | <---------------> |  server/app.py   |
| (AI agent)   |                   |  (FastAPI server) |
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
- Target yield is the *universal* maximum potential production across all three locations — reaching it is mathematically impossible under budget constraints, and harder locations score lower by design

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
| `crop_status` | dict | DVS (development stage 0→2), LAI, biomass, yield, growth stage name |
| `soil_status` | dict | Soil moisture, water deficit flag, water stress (0-1), N availability (0-1), field capacity, wilting point |
| `weather_today` | dict | Temperature (max/min), rainfall, radiation |
| `weather_forecast` | list[dict] | 5-day forecast (with slight noise for realism) |
| `resources_used` | dict | Total water, nitrogen, cost, budget remaining, unit costs |
| `season_summary` | dict | Crop name, location, target yield, budget, step size |
| `conflicts` | list[str] | Feedback on invalid actions |

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

Dense per-step signals during the episode (designed to resist reward hacking):
- **Irrigate dry soil:** +0.10 | **Irrigate wet soil:** -0.05
- **Fertilize at key DVS:** +0.15 | **Fertilize late:** -0.10
- **Harvest at maturity:** +0.20 | **Harvest too early:** -0.30
- **Wait:** 0.00 (neutral — no free reward for doing nothing)
- **Late harvest (DVS > 2.05):** proportional penalty up to -0.15
- **Between-window fertilization:** -0.03 (discourages spray-more strategies)

---

## Baseline Scores (Greedy Heuristic, seed=42)

| Task | Score |
|------|-------|
| 1 (Easy) | 0.8166 |
| 2 (Medium) | 0.8049 |
| 3 (Hard) | 0.6986 |
| **Overall** | **0.7734** |

---

## Project Structure

```
MetaHackathonPrep/
├── server/
│   ├── __init__.py         # Package marker
│   ├── app.py              # FastAPI server via create_app() + /tasks endpoint
│   ├── environment.py      # CropEnvironment (reset/step/state)
│   ├── crop_sim.py         # WOFOST-inspired crop growth simulator
│   ├── grader.py           # Multi-metric deterministic scoring
│   ├── reward.py           # Dense step + trajectory reward computation
│   ├── scenarios.py        # Seeded weather + scenario generator (3 locations)
│   ├── tasks.py            # Task definitions (3 difficulty levels)
│   └── Dockerfile          # Docker image for HuggingFace Spaces
├── tests/
│   └── test_smoke.py       # Smoke tests (determinism, reset/step, scoring range)
├── models.py               # CropAction, CropObservation, CropState
├── client.py               # WebSocket EnvClient subclass
├── inference.py            # Greedy heuristic + LLM inference script
├── openenv.yaml            # OpenEnv environment metadata
├── pyproject.toml          # Package configuration
├── requirements.txt        # Python dependencies
├── .env                    # Local env vars (API keys — git-ignored)
├── .gitignore              # Git ignore rules
├── .dockerignore           # Docker build exclusions
└── README.md               # This file
```

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

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | For LLM mode | LLM API endpoint (OpenAI-compatible) |
| `MODEL_NAME` | For LLM mode | Model identifier |
| `HF_TOKEN` | For LLM mode | HuggingFace / API authentication token |
| `ENV_URL` | No | Server URL (default: `http://localhost:8000`) |

All variables can be set in a `.env` file in the project root (auto-loaded via `python-dotenv`).

---

## Docker

```bash
docker build -f server/Dockerfile -t crop-management .
docker run -p 7860:7860 crop-management
ENV_URL=http://localhost:7860 python inference.py
```

---

## Crop Growth Model

The simulator implements key dynamics from the WOFOST (World Food Studies) model:

- **Phenology:** Temperature-sum driven development stages (DVS 0→1 vegetative, 1→2 reproductive)
- **Biomass:** Light-use-efficiency model with PAR interception via LAI
- **Water balance:** Rainfall + irrigation - evapotranspiration (Hargreaves ET)
- **Water stress:** Reduces growth when soil moisture drops below threshold
- **Heat stress:** Above 35°C during flowering (DVS 0.8–1.2), pollen sterility reduces growth up to 70%
- **Nitrogen response:** Fertilization increases growth factor (0.3→1.0); phenology-aware depletion (slow pre-anthesis, fast post-anthesis)
- **Partitioning:** DVS-dependent allocation to storage organs (grain yield)
- **Senescence:** Realistic LAI decline after DVS > 1.5 (~7-10 day period)

Three climatic profiles with deterministic weather generation:
- **Netherlands:** Mild maritime, reliable rainfall (easy)
- **Iowa, USA:** Continental, moderate with dry spells (medium)
- **Punjab, India:** Hot semi-arid, minimal winter rain (hard)

---

Built for the [Meta PyTorch OpenEnv Hackathon](https://github.com/meta-pytorch/OpenEnv) — Round 1.
