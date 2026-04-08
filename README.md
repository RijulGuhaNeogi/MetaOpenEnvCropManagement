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
> **Team Hijibiji** 

A deterministic, multi-step precision agriculture environment built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework. An AI agent manages a wheat growing season — deciding weekly when to irrigate, fertilize, and harvest to maximize yield while minimizing water use and cost while negotiating leaching .

The current environment is optimized for both public-task grading quality and RL learnability: dense rewards are post-transition aware, observations include policy-native control features, and internal probe scenarios exist for diagnosing reward/behavior failures without changing the public tasks.

---

## Why Precision Agriculture?

Precision agriculture is a $12B+ industry where farmers make weekly decisions about irrigation, fertilization, and harvest timing under uncertainty (weather, soil conditions, budgets). This environment models that exact decision process:

- **Real-world task:** Every farmer does this — the environment simulates decisions that affect millions of hectares of cropland globally
- **Multi-objective optimization:** Maximizing yield conflicts with minimizing water/cost — a genuine Pareto frontier problem
- **Weather uncertainty:** Agents must read forecasts and plan ahead, not just react
- **Novel domain:** No existing crop management environment in the OpenEnv ecosystem

The WOFOST-inspired crop growth simulator captures real agricultural dynamics: temperature-sum phenology, water stress, nitrogen response, and biomass partitioning.

**Scientifically grounded:** The simulator's equations and parameters are drawn from 17 peer-reviewed agronomic references — including the original WOFOST model (van Diepen et al. 1989), Hargreaves evapotranspiration, Beer–Lambert light interception, Feddes water-stress functions, and FAO/IIASA regional calibrations for all three task locations. See [docs/REFERENCES.md](docs/REFERENCES.md) for the full bibliography.

---

## Key Design Strengths

| Strength | Detail |
|----------|--------|
| **Organic difficulty progression** | All 3 tasks use the **same scoring formula**. Difficulty comes from climate, budget, input costs, and observability — not inflated weights |
| **Dense, aligned reward shaping** | Every step produces a blended intent + delta reward that mirrors the terminal grader — step rewards and final grades push the same behavior |
| **Anti-exploit reward gating** | Efficiency metrics are multiplied by `max(yield_score, 0.1)` — a do-nothing agent cannot score high on water/cost efficiency |
| **Crop vigor gating** | Delta rewards scale with `crop_vigor = f(twso / target)` — late-season efficiency actions on a failing crop get diminished credit, mirroring the grader |
| **Explicit harvest incentive** | Agent-initiated harvest gets full timing credit; auto-harvest gets only 20% (0.2× penalty) — agents must learn *when* to stop |
| **Natural consequences** | Grain shattering (~23% yield loss per 7-day step past DVS 1.85) provides a physics-based penalty for harvest delay, not an arbitrary rule |
| **N leaching & slow-release** | Wet soil leaches applied N; `fertilize_slow` (1.5× cost) resists leaching — agents must read weather forecasts to pick the right fertilizer type |
| **Information-action tradeoff** | Inspect actions reveal hidden state but cost budget; economically dominated by real actions so they can't be spammed for free reward |
| **Deterministic + diverse** | Same seed → identical weather, growth, grading, rewards, forecasts — fully reproducible. **Different seeds produce entirely different weather seasons** (varying rainfall, temperature, drought/flood events), so agents must generalize across conditions, not memorize one scenario. For example: seed 2 (Task 1) produces heavy rain during fertilizer windows (rain₃d = 1.63cm), triggering N leaching that demands `fertilize_slow`; seed 14 (Task 3) combines Punjab's tight $300 budget with wet spells that punish cheap fertilizer; seed 42 stays dry, rewarding cost-efficient regular fertilizer. No single hard-coded strategy works across seeds |
| **Region-calibrated stress** | Punjab has lower heat thresholds (34°C/31°C vs 35°C/32°C), faster N depletion, and sandy soil — difficulty is agronomically grounded |
| **Comprehensive penalties** | Overwatering, over-fertilization (>2 apps hard-capped), late/early fertilization, inaction during stress, harvest urgency, budget exhaustion |
| **Advisory without prescription** | Rich NL advisory describes field conditions factually but never tells the agent what to do — the agent must reason |
| **Probe diagnostics** | 5 edge-case probe scenarios (over-irrigation trap, late-fertilizer temptation, budget starvation, harvest hesitation, drought rescue) validate reward directional correctness without altering public tasks |

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

| ID | Name | Difficulty | Location | Budget | Observability | Key Challenge |
|----|------|------------|----------|--------|---------------|---------------|
| 1 | Basic Crop Growth | Easy | Netherlands | $800 | Tier 1 (full numeric) | Fertilize at right growth stages, harvest at maturity |
| 2 | Water-Efficient Farming | Medium | Iowa, USA | $450 | Tier 2 (hidden DVS/SM, bands + NL weather) | Balance yield vs water conservation with partial information |
| 3 | Precision Agriculture | Hard | Punjab, India | $300 | Tier 3 (most fields hidden, bucketed weather) | Maximize yield + minimize water + stay in budget under information scarcity |

### Observability Tiers as Curriculum Design

The three tasks form a natural **reasoning curriculum** — not just a difficulty ladder. Tier 1 gives full numeric state, so agents learn optimal policy with complete information. Tier 2 hides key variables behind coarsened bands and natural-language weather, forcing the agent to generalize under partial observability. Tier 3 further restricts signals to bucketed summaries and introduces an **information-action tradeoff**: two inspect actions (`inspect_soil` at $10, `inspect_crop` at $20) reveal exact hidden values but consume budget that could fund real interventions. Unlike typical RL environments where observation quality is fixed, agents here must learn *when information is worth paying for* — a decision-theoretic challenge with no analogue in game or benchmark environments.

**Why Task 3 is genuinely hard:**
- Punjab has minimal rainfall during the wheat season — irrigation is essential but expensive
- Budget is tight ($300) — every irrigation/fertilization decision must be justified
- Most precise sensor readings are hidden — the agent sees coarsened bands and bucketed weather
- Two inspect actions (`inspect_soil` at $10, `inspect_crop` at $20) reveal exact values but cost budget; results persist across the episode
- Scoring weights yield (35%), water efficiency (20%), cost efficiency (18%), timing (15%), and harvest timing (12%) — no single strategy dominates
- An LLM agent that strategically inspects and reasons over NL observations can outperform the greedy heuristic, which operates blindly on midpoint estimates

### Why This Environment Challenges Even Frontier LLMs

Beyond observability tiers, the environment embeds a **weather-contingent economic tradeoff** that requires multi-step causal reasoning — not pattern matching:

| | Dry forecast (rain₃d < 0.2cm) | Wet forecast (rain₃d > 0.5cm) |
|---|---|---|
| **Regular `fertilize`** | ✅ Cheap, full N uptake | ❌ N leaches out of root zone — wasted money and lost yield |
| **Slow-release `fertilize_slow`** | ❌ Paying 1.5× for protection you don't need | ✅ 70% leach-resistant — N retained through the rain event |

The agent must **read the 3-day rain forecast, estimate leaching risk, weigh the 1.5× cost premium against expected N loss, and choose the correct fertilizer type** — all within a tight budget. The advisory warns about leaching risk but never prescribes an action.

**Why this separates frontier from weak LLMs:**
- A greedy heuristic always picks cheap regular fertilizer → loses N in wet weather → yield drops
- A weak LLM may learn "rain → slow-release" but overapplies it in borderline cases, wasting budget
- A frontier LLM must reason about **forecast magnitude vs cost vs remaining budget vs crop growth stage** — a 4-variable conditional decision with no single threshold that always works
- The oracle achieves ~0.94 on Task 1; a strong LLM (Llama 3.3 70B) scores ~0.82 — the remaining 12-point gap requires precisely this kind of economic reasoning under uncertainty

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
| `action_type` | str | `irrigate`, `fertilize`, `fertilize_slow`, `harvest`, `wait`, `inspect_soil`, `inspect_crop` | Weekly management decision |
| `amount` | float | 0-10 cm (irrigate), 0-50 kg N/ha (fertilize/fertilize_slow), 0 (others) | Resource amount |

**Inspect actions:** `inspect_soil` ($10) reveals exact soil moisture, nitrogen, and water stress. `inspect_crop` ($20) reveals exact DVS, LAI, biomass, and grain weight. Inspects are **free sub-actions** — they cost budget but do **not** advance the simulation or consume a week. The agent gets results immediately and can take a real action on the next call within the same logical step. Budget is the only constraint on inspects (no artificial cap). Results **persist** in all subsequent observations as `soil_report` / `crop_report`. Available on all tiers, but most valuable on Tier 2/3 where numeric readings are hidden.

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
| `observability_tier` | int | 1=full numeric, 2=mixed bands+NL, 3=NL-heavy |
| `sm_band` | str \| None | Coarsened soil moisture band (tier 2/3): critical/low/adequate/high |
| `n_visual` | str \| None | Coarsened nitrogen band (tier 2/3): very_low/low/moderate/adequate/surplus |
| `lai_band` | str \| None | Coarsened canopy band (tier 2/3): sparse/moderate/dense |
| `weather_summary` | str \| None | NL weather forecast (tier 2: exact per-day, tier 3: bucketed) |
| `soil_report` | str \| None | Exact soil readings (persists after inspect_soil) |
| `crop_report` | str \| None | Exact crop readings (persists after inspect_crop) |
| `dose_hint` | str \| None | Dose quality feedback after fertilize actions |

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

Difficulty comes from **environment conditions** (climate, budget, soil), not from different scoring weights. The same formula is used for all 3 tasks — a key design choice that keeps grading fair and lets difficulty emerge naturally from the scenario:


| yield | water | cost | timing | harvest |
|-------|-------|------|--------|--------|
| 35% | 20% | 18% | 15% | 12% |

### Step Rewards

Unlike environments with binary pass/fail grading or sparse terminal-only rewards, every step in this environment produces a **dense, agronomically-grounded reward** that shapes the same behavior the terminal grader scores. This means an RL agent can learn from every transition, not just episode outcomes.

Dense per-step signals are split into:

- **Intent reward:** evaluates whether the action is agronomically sensible before transition
- **Delta reward:** evaluates whether the action actually relieved stress or wasted resources after transition
- **Terminal harvest blending:** at episode end, the reward is `0.7 × trajectory_reward + 0.3 × normalized_harvest_step_signal`, giving immediate feedback on harvest timing quality

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
  - rewards harvesting in the maturity window DVS 1.8–2.0
  - penalizes early and late harvest
  - 2-step grace period after DVS reaches 2.0: grain shattering (~23% yield loss/step) provides natural consequence for delaying; auto-termination fires after 2 extra steps
- **Wait**
  - penalizes inaction when the crop is suffering (water stress or nitrogen deficiency)
  - adds a fert-window penalty when N is low and DVS is past the target
  - strong harvest-urgency ramp (−0.05 to −0.10) inside the harvest window [1.80, 2.00]; flat −0.10 during post-maturity grace period
  - small magnitude outside harvest context so it never dominates an actual action reward

All blended step rewards are clamped to [−0.9, +0.9] as a safety net.

### Anti-Exploit Design

- **Yield-gated efficiency:** Water and cost efficiency scores are multiplied by `max(yield_score, 0.1)`. An agent that uses zero water and zero fertilizer gets near-zero final score, not a high efficiency score.
- **Crop vigor scaling:** Delta rewards for irrigation/fertilization are scaled by crop vigor (`twso / target`). Late-season actions on a failing crop get diminished credit, preventing reward farming on doomed episodes.
- **Inspection budget pressure:** Inspect reward is scaled by `budget_remaining / (cost × 10)`. Repeated inspections on a drained budget yield near-zero reward.
- **Fertilizer hard cap:** More than 2 fertilizer applications trigger a hard penalty (−0.04 to −0.14), regardless of timing or dose quality.
- **Auto-harvest penalty:** If the agent fails to harvest explicitly, auto-termination applies a 0.5× multiplier to the harvest-timing reward component (only 20% credit vs 100%). Combined with grain shattering losses, passivity is doubly penalized.

### Rubric System (RFC 004)

The environment follows the [OpenEnv RFC 004](https://github.com/meta-pytorch/OpenEnv/blob/main/rfcs/004-rubrics.md) convention for delayed rewards:

- **`obs.reward`** — Dense per-step signal (intent + delta blend) for RL training
- **`obs.rubric_reward`** — Trajectory-level score from the grader at terminal steps; `None` on intermediate steps
- **`obs.metadata["rubric_breakdown"]`** — Per-metric scores at terminal step (yield, water, cost, timing, harvest)

This separation lets RL frameworks (TRL, GRPO, torchforge) distinguish immediate feedback from episode-level evaluation without ambiguity.

The rubric is provided by `CropManagementRubric` (in `server/rubric.py`), a thin wrapper around the deterministic grader.

---

## Baseline Scores (seed=42)

Two reference policies bracket the achievable score range. They use **different information** and serve **different purposes**:

| | Oracle Ceiling | Greedy Baseline |
|---|---|---|
| **What it sees** | Full internal simulator state (exact DVS, soil moisture, N-factor, weather) | Only the public observation the LLM would see (masked bands, NL weather, advisory text on Tier 2/3) |
| **What it does** | Plans optimal timing using thermal-sum lookahead, exact N-deficit tracking, and perfect harvest targeting | Applies simple threshold rules to whatever is visible — irrigate if dry, fertilize if in window, harvest if ripe |
| **Purpose** | Upper bound — the best any policy could do with perfect information | Fair comparison target — shares the same information constraint as the LLM agent |

### Scores

| Task | Oracle Ceiling | Greedy Baseline | Gap |
|------|---------------|-----------------|-----|
| 1 (Easy) | 0.9593 | 0.9588 | 0.0005 |
| 2 (Medium) | 0.9409 | 0.5298 | 0.4111 |
| 3 (Hard) | 0.9067 | 0.4224 | 0.4843 |

**Reading the gap:** On Task 1 (full observability), the greedy heuristic nearly matches the oracle — information isn't the bottleneck. On Tasks 2–3, the gap widens dramatically because the greedy heuristic cannot see the hidden state it needs for precise decisions. This gap is the **observability challenge** — the space where an LLM that reasons over natural-language cues and strategically uses inspect actions can outperform the blind heuristic.

A **wait-only (do-nothing) policy** scores 0.37 / 0.35 / 0.17 on Tasks 1/2/3 respectively — the grader's anti-passivity calibration ensures that agents must take meaningful actions to score well.

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
│   ├── app.py              # FastAPI server via create_app() + /tasks, /grader, /baseline, /ceiling
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
│   ├── test_smoke.py       # Smoke + RL + rubric tests (65 tests)
│   ├── test_integration.py # HTTP endpoint integration tests (9 tests)
│   ├── test_submission_surface.py  # Competition format compliance tests (5 tests)
│   └── test_ws_episode.py  # WebSocket full-episode tests (3 tests)
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
- **Direct benchmark path** — evaluate the oracle ceiling directly against the environment without starting the server
- **Test path** — validate determinism, reward behavior, and policy regressions via the smoke suite

Use the inference path when you want end-to-end OpenEnv behavior with greedy or LLM decisions, the direct benchmark path when you want fast oracle-ceiling comparisons, and the test path when you want regression protection.

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

curl http://localhost:8000/baseline
# Deterministic greedy baseline scores for all 3 tasks (cached, seed=42)

curl http://localhost:8000/ceiling
# Deterministic oracle ceiling scores for all 3 tasks (cached, seed=42)

curl -X POST http://localhost:8000/grader -H 'Content-Type: application/json' \
  -d '{"actual_yield": 5000, "target_yield": 8000, "total_water": 10, "total_n": 60, "total_cost": 100, "budget": 800, "harvest_dvs": 1.9, "harvested": true, "actions_taken": [], "task_id": 1}'
# {"score": ..., "breakdown": {...}}
```

---

## Running Inference

### With LLM (competition mode)

Create a `.env` file (auto-loaded by `python-dotenv`):

```dotenv
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
API_KEY=your_api_key_here          # preferred (evaluator injects this)
HF_TOKEN=hf_your_token_here        # fallback for local development
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
export API_KEY="your_api_key_here"        # or use HF_TOKEN as fallback
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
| `API_KEY` | For LLM mode | API authentication token (evaluator injects this) |
| `HF_TOKEN` | No | Fallback for `API_KEY` — accepted for local development |
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

The direct benchmark uses the environment's perfect-information oracle reference. The `/baseline` endpoint exposes the observation-limited greedy baseline. If you call `oracle_action(obs, state)` directly on masked tier-2/3 observations, that path is a reconstruction heuristic rather than the true oracle ceiling.

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

The full test suite has **82 passing tests** (65 smoke/rubric/weather + 9 HTTP integration + 5 submission surface + 3 real WebSocket transport).

## Limitations

This environment is intentionally **WOFOST-inspired**, not a full scientific crop model.

- It models one crop family used in the public benchmark and does not attempt crop rotation or multi-season farm planning.
- Root growth, pests, disease, salinity, and management operations beyond irrigation, fertilization, harvest, and wait are intentionally omitted.
- Weather is deterministic by seed and forecast noise is deterministic by day; this improves reproducibility and RL debugging at the cost of richer stochastic realism.
- The simulator is tuned for clear reward signal and grading stability, not for exact agronomic calibration to a specific field trial dataset.

Those tradeoffs are intentional. The benchmark is optimized for deterministic evaluation, transparent grading, and trainable sequential decision-making rather than maximum biophysical fidelity.

---

## RL Learnability

This environment is designed to produce a **learnable reward landscape**, not just a grading function:

- **Reward–grader alignment:** Step-level intent and delta rewards are calibrated to push the same behavior the terminal grader scores. An agent that maximizes cumulative step reward will also score well on the final rubric.
- **Shaped, not sparse:** Every action type produces a non-zero reward signal. Wait is always ≤ 0; irrigate and fertilize range from −0.14 to +0.16 depending on timing and dose quality; harvest yields +0.20 at optimal DVS. No action is reward-silent.
- **Smooth gradients:** Fertilize reward peaks sharply at target DVS (0.30, 0.60) with linear decay — the agent gets a clear gradient toward optimal timing. Irrigation reward scales with dose accuracy and soil dryness — the agent learns precise amounts, not just thresholds.
- **Multi-signal terminal:** The final reward blends 70% trajectory grade (5 rubric metrics) with 30% harvest-timing signal, giving both cumulative and pointwise feedback at episode end.
- **Curriculum-ready:** The 3 tasks form a natural curriculum: Tier 1 (full state, generous budget) → Tier 2 (hidden state, moderate budget) → Tier 3 (minimal state, tight budget). Agents can train on easy tasks first and transfer.
- **Consistent constants:** All threshold values (harvest DVS, fertilizer windows, SM targets) are centralized in `server/constants.py` and verified consistent across reward, grader, and environment modules. No hidden misalignments.
- **Trajectory export:** `TRAJECTORY_OUTPUT` env var enables JSONL export with `(observation, action, reward, next_observation, done, metadata)` tuples for offline RL / imitation learning.
- **Training adapter:** `agent/training_adapter.py` provides an 11-action discrete mapping (`wait`, `harvest`, `irrigate_small/medium/large`, `fertilize_small/medium/large`, `fertilize_slow_small/medium/large`) for standard RL frameworks.

---

## Internal RL Utilities

### Reward-Alignment Probe Scenarios

Five purpose-built edge-case scenarios validate that dense step rewards push the same direction as the terminal grader — without modifying public tasks. Each probe isolates a single failure mode an RL agent might exploit or stumble on:

| Probe | What It Tests |
|-------|---------------|
| `over_irrigation_trap` | Agent faces saturated soil + forecast rain — correct reward must penalize unnecessary irrigation, not reward "doing something" |
| `late_fertilizer_temptation` | DVS is past the optimal window — fertilizing now wastes budget with minimal yield benefit; reward must be negative despite the action "looking productive" |
| `budget_starvation` | Budget is nearly exhausted — agent must choose between an inspect (information) and a real action (intervention); reward must reflect opportunity cost |
| `harvest_hesitation` | Crop is mature and shattering has begun — every wait step must carry escalating penalty; reward must ramp urgency |
| `drought_rescue` | Severe water stress mid-season — immediate irrigation must produce a large positive delta reward to overcome the "save budget" heuristic |

Activate via `reset(..., probe_name="harvest_hesitation")`. Probes share the same reward and grading code as public tasks — they only override the initial scenario state.

### Discrete Action Adapter

`agent/training_adapter.py` provides an 11-action discrete mapping for standard RL frameworks:

`wait` · `harvest` · `irrigate_small` · `irrigate_medium` · `irrigate_large` · `fertilize_small` · `fertilize_medium` · `fertilize_large` · `fertilize_slow_small` · `fertilize_slow_medium` · `fertilize_slow_large`

The public OpenEnv action schema remains `action_type + amount`.

---

## Crop Growth Model

The simulator implements key dynamics from the WOFOST (World Food Studies) model:

- **Phenology:** Temperature-sum driven development stages (DVS 0→1 vegetative, 1→2 reproductive)
- **Biomass:** Light-use-efficiency model with PAR interception via LAI
- **Water balance:** Rainfall + irrigation - evapotranspiration (Hargreaves ET)
- **Water stress:** Reduces growth when soil moisture drops below threshold
- **Heat stress:** Two separate mechanisms — pollen sterility during flowering (DVS 0.8–1.2, triggered >35°C) and kernel weight reduction during grain-fill (DVS 1.0–1.5, triggered >32°C). Punjab uses **lower thresholds** (34°C/31°C) for region-realistic heat sensitivity.
- **Nitrogen response:** Fertilization increases the N-factor (0.3→1.0) with intentionally low recovery (0.008 per kg applied) — timing matters more than volume. Depletion is phenology-aware: slow pre-anthesis (0.0003/day), fast post-anthesis (0.0015/day; Punjab: 0.0020/day).
- **N leaching:** When fertilizer is applied and soil is wet (near field capacity) or rain falls, a fraction of applied N washes below the root zone. Regular fertilizer loses 15–40% in wet conditions.
- **Slow-release fertilizer:** `fertilize_slow` costs 1.5× base rate but resists leaching (70% resistant). Delivers 70% of N immediately, with the remaining 30% dripping in over 14 days. The agent must decide: cheap regular fert in dry weather, or expensive slow-release when rain is coming.
- **Partitioning:** DVS-dependent allocation to storage organs (grain yield)
- **Senescence:** Realistic LAI decline after DVS > 1.5 (~7-10 day period)

This is a **WOFOST-inspired** simulator, not the PCSE reference implementation. That tradeoff is intentional: the current environment favors deterministic behavior, transparent grading, small deployment footprint, and strong RL signal quality over additional simulator complexity.

Three climatic profiles with deterministic season-based weather generation:

| Location | Rainfall | Temperature | Season | Soil Type | Key Constraint |
|----------|----------|-------------|--------|-----------|----------------|
| **Netherlands** | 50–70 cm (45% rain days) | Mild 5–17°C | 280 days (Oct–Jul) | Clay loam (FC 0.43) | None — favorable baseline |
| **Iowa, USA** | 30–50 cm (30% rain days) | Variable 3–19°C | 260 days | Silt loam (FC 0.40) | Drought spells, variable dry periods |
| **Punjab, India** | 5–10 cm (12% rain days) | Hot 10–30°C | 200 days (Nov–Apr) | Sandy loam (FC 0.35) | Irrigation-dependent, heat stress, fast N depletion |

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
