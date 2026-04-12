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

A deterministic, multi-step precision agriculture environment built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework based on the scientifically grounded WOFOST model. An AI agent manages a wheat growing season — deciding weekly when to irrigate, fertilize, and harvest to maximize yield while minimizing water use and cost and negotiating adverse factors like shattering, leaching and heat stress.

---

## Summary


- This environment models a **real $12B+ industry workflow** — weekly crop management decisions that affect millions of hectares globally — not a toy classification task
- The crop simulator is **scientifically grounded** in the WOFOST model with parameters drawn from **17 peer-reviewed agronomic references**
- All 3 tasks use the **same scoring formula**; difficulty comes naturally from climate, budget, soil, and observability — not inflated weights
- Dense, aligned step rewards let RL agents learn from **every transition**, not just episode outcomes
- **Different seeds produce entirely different weather seasons**, so agents must generalize — no single hard-coded strategy works
- The repo is self-contained (no external APIs at runtime), fully deterministic, Docker-ready, and has **82 passing tests**

---

## Why Precision Agriculture?

- **Real-world task:** Every farmer makes these decisions — the environment simulates irrigation, fertilization, and harvest timing under weather uncertainty
- **Multi-objective optimization:** Maximizing yield conflicts with minimizing water/cost — a genuine Pareto frontier problem
- **Weather uncertainty:** Agents must read forecasts and plan ahead, not just react
- **Novel domain:** No existing crop management environment in the OpenEnv ecosystem

**Scientifically grounded:** The simulator's equations and parameters are drawn from 17 peer-reviewed references — including the original WOFOST model (van Diepen et al. 1989), Hargreaves evapotranspiration, Beer–Lambert light interception, Feddes water-stress functions, and FAO/IIASA regional calibrations. See [docs/REFERENCES.md](docs/REFERENCES.md) for the full bibliography.

---

## Key Design Strengths

| Strength | Detail |
|----------|--------|
| **Organic difficulty progression** | Same scoring formula across all tasks. Difficulty comes from climate, budget, and observability |
| **Dense, aligned reward shaping** | Every step produces a blended intent + delta reward that mirrors the terminal grader |
| **Anti-exploit gating** | Yield-gated efficiency, crop vigor scaling, fertilizer hard cap, auto-harvest penalty — [details](docs/ARCHITECTURE.md) |
| **N leaching & slow-release** | Wet soil leaches applied N; `fertilize_slow` (1.5× cost) resists leaching — agents must read weather forecasts to choose |
| **Deterministic + diverse seeds** | Same seed → identical episode. Different seeds → entirely different weather seasons (e.g., seed 2 produces heavy rain triggering N leaching; seed 14 combines tight budget with wet spells; seed 42 stays dry). No strategy memorization possible |
| **Region-calibrated stress** | Punjab has lower heat thresholds, faster N depletion, sandy soil — difficulty is agronomically grounded |
| **Advisory without prescription** | Rich NL advisory describes field conditions factually but never tells the agent what to do |
| **Probe diagnostics** | 5 edge-case probe scenarios validate reward correctness without altering public tasks |

---

## Tasks

| ID | Name | Difficulty | Location | Budget | Observability | Key Challenge |
|----|------|------------|----------|--------|---------------|---------------|
| 1 | Basic Crop Growth | Easy | Netherlands | $800 | Tier 1 (full numeric) | Fertilize at right growth stages, harvest at maturity |
| 2 | Water-Efficient Farming | Medium | Iowa, USA | $450 | Tier 2 (hidden DVS/SM, bands + NL weather) | Balance yield vs water under partial information |
| 3 | Precision Agriculture | Hard | Punjab, India | $300 | Tier 3 (most fields hidden, bucketed weather) | Maximize yield + minimize water + stay in budget under info scarcity |

### Observability Tiers

The tasks form a **reasoning curriculum**:

- **Tier 1** — full numeric state; agents learn optimal policy with complete information
- **Tier 2** — key variables hidden behind coarsened bands and NL weather; agents must generalize under partial observability
- **Tier 3** — bucketed summaries + an **information-action tradeoff**: `inspect_soil` ($10) and `inspect_crop` ($20) reveal hidden values but consume budget. Agents must learn *when information is worth paying for*

### Weather-Contingent Fertilizer Tradeoff

The environment embeds a tradeoff that requires multi-step causal reasoning:

| | Dry forecast (rain₃d < 0.2cm) | Wet forecast (rain₃d > 0.5cm) |
|---|---|---|
| **Regular `fertilize`** | ✅ Cheap, full N uptake | ❌ N leaches — wasted money and yield |
| **Slow-release `fertilize_slow`** | ❌ Paying 1.5× for unneeded protection | ✅ 70% leach-resistant |

The agent must read the forecast, estimate leaching risk, weigh costs against expected N loss, and choose — all within a tight budget. No single threshold always works.

**Why this challenges frontier LLMs:** The decision requires jointly reasoning over forecast magnitude, soil moisture, crop growth stage, remaining budget, and N availability — a 4-variable conditional decision. The oracle achieves ~0.94 on Task 1; a strong LLM (Llama 3.3 70B) scores ~0.82 — the remaining gap requires precisely this kind of economic reasoning under uncertainty.

---

## Action Space

```json
{ "action_type": "irrigate", "amount": 2.5 }
```

| Action | Amount | Description |
|--------|--------|-------------|
| `irrigate` | 0–10 cm | Apply water |
| `fertilize` | 0–50 kg N/ha | Standard nitrogen fertilizer |
| `fertilize_slow` | 0–50 kg N/ha | Slow-release (1.5× cost, 70% leach-resistant) |
| `harvest` | — | Harvest the crop |
| `wait` | — | Do nothing this week |
| `inspect_soil` | — | $10: reveal exact soil state (persists) |
| `inspect_crop` | — | $20: reveal exact crop state (persists) |

Inspect actions are **free sub-actions** — they cost budget but do not advance the simulation. Results persist in all subsequent observations.

---

## Observation Space

| Field | Description |
|-------|-------------|
| `day`, `days_remaining` | Simulation timeline |
| `crop_status` | DVS, LAI, biomass, yield, growth stage name |
| `soil_status` | Soil moisture, water stress, N availability, field capacity |
| `weather_today`, `weather_forecast` | Current + 5-day forecast (with slight noise) |
| `resources_used` | Total water, nitrogen, cost, budget remaining |
| `control_features` | 11 derived RL features (moisture gap, forecast rain, budget ratio, etc.) |
| `advisory_text` | Factual NL summary of crop state (never prescriptive) |
| `sm_band`, `n_visual`, `lai_band` | Coarsened bands (tier 2/3) |
| `weather_summary` | NL weather forecast (tier 2/3) |
| `soil_report`, `crop_report` | Exact readings after inspect actions (persists) |
| `observability_tier` | 1=full numeric, 2=mixed, 3=NL-heavy |

All sub-models are Pydantic types with explicit fields. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full schema.

---

## Grading

Deterministic scoring — same inputs always produce the same score in [0.0, 1.0].

| Metric | Weight | Formula |
|--------|--------|---------|
| Yield | 35% | `min(1.0, actual_yield / target_yield)` |
| Water efficiency | 20% | `1.0 - water_used / 50cm` |
| Cost efficiency | 18% | `1.0 - cost / budget` |
| Timing quality | 15% | Proximity of fertilizer actions to DVS 0.3 / 0.6 |
| Harvest timing | 12% | Peak at DVS 1.8–2.0, penalty otherwise |

**Same weights for all tasks.** Difficulty emerges from environment conditions, not scoring manipulation.

### Step Rewards

Every step produces **dense, agronomically-grounded reward** (intent + delta blend):

- **Irrigation** — rewards closing moisture deficit; penalizes overshoot and irrigating before rain
- **Fertilization** — rewards correct DVS timing and dose; peaks near DVS 0.30 and 0.60
- **Harvest** — rewards DVS 1.8–2.0; grain shattering (~23% loss/step) penalizes delay naturally
- **Wait** — penalizes inaction during stress; harvest-urgency ramp inside maturity window

Terminal: `0.7 × trajectory_grade + 0.3 × harvest_timing_signal`. All step rewards clamped to [−0.9, +0.9].

**RL learnability:** Rewards are shaped (not sparse), every action produces a non-zero signal, fertilizer/irrigation rewards have smooth gradients toward optimal timing and dose, and the 3-tier task ladder is curriculum-ready. A discrete 11-action adapter and JSONL trajectory export support standard RL frameworks. For full reward architecture, anti-exploit mechanisms, and learnability details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Baseline Scores (seed=42)

| | Oracle Ceiling | Llama 3.3 70B | Greedy(Heuristic Baseline) |
|---|---|---|---|
| **Info** | Full simulator state | NL observations + LLM reasoning | Public observation only |
| **Strategy** | Thermal-sum lookahead, exact N-deficit tracking | Prompt-based crop management decisions | Simple threshold rules on visible data |

| Task | Oracle | Llama 3.3 70B | Greedy(Heuristic Baseline) | LLM vs Greedy |
|------|--------|---------------|--------|---------------|
| 1 (Easy) | 0.959 | 0.941 | 0.959 | −0.018 |
| 2 (Medium) | 0.941 | 0.885 | 0.530 | +0.355 |
| 3 (Hard) | 0.907 | 0.780 | 0.422 | +0.358 |
| **Average** | **0.936** | **0.869** | **0.637** | **+0.232** |

On Task 1 (full observability), the greedy heuristic nearly matches the oracle. On Tasks 2–3, the gap widens dramatically — this is the **observability challenge** where LLM reasoning over NL cues and strategic inspection can outperform the blind heuristic.

A **do-nothing policy** scores 0.37 / 0.35 / 0.17 — anti-passivity calibration ensures agents must act.

### LLM Agent Comparison (seed=190)

Seed 190 selected for maximum oracle action diversity — it triggers all 5 action types (`wait`, `fertilize`, `fertilize_slow`, `irrigate`, `harvest`) across the 3 tasks, including 2 irrigations in Task 3 (most seeds require 0–1).

| Task | Oracle | Llama 3.3 70B | Greedy(Heuristic Baseline) | LLM vs Greedy |
|------|--------|---------------|--------|---------------|
| 1 (Easy) | 0.940 | 0.847 | 0.846 | +0.001 |
| 2 (Medium) | 0.922 | 0.868 | 0.528 | +0.340 |
| 3 (Hard) | 0.857 | 0.680 | 0.405 | +0.275 |
| **Average** | **0.906** | **0.798** | **0.593** | **+0.205** |

The LLM agent outperforms greedy by **+34% on average**, demonstrating that LLM reasoning over NL observations, weather forecasts, and budget constraints adds substantial value — especially on the harder tasks where observability is limited.

---

## Crop Growth Model

WOFOST-inspired pure-Python simulator (~330 LOC). Key dynamics:

| Component | Implementation |
|-----------|---------------|
| Phenology | Temperature-sum DVS (0→1→2) |
| Biomass | Light-use efficiency: PAR × LUE × LAI interception |
| Water balance | Rainfall + irrigation − ET (Hargreaves) |
| Water/heat stress | Growth reduction, pollen sterility, grain-fill penalty |
| Nitrogen | Linear N-factor (0.3→1.0), phenology-aware depletion |
| N leaching | Wet soil leaches N; slow-release resists at 30% rate |
| Partitioning | DVS-dependent grain fraction; senescence post-DVS 1.5 |

Three region-calibrated climate profiles:

| Location | Rainfall | Temperature | Season | Soil | Key Constraint |
|----------|----------|-------------|--------|------|----------------|
| Netherlands | 50–70 cm | Mild 5–17°C | 280 days | Clay loam | Favorable baseline |
| Iowa | 30–50 cm | Variable 3–19°C | 260 days | Silt loam | Drought spells |
| Punjab | 5–10 cm | Hot 10–30°C | 200 days | Sandy loam | Irrigation-dependent, heat stress |

**Parameters sourced from:** Boogaard et al. (2014), de Wit et al. (2019), van Diepen et al. (1989), ISRIC, FAO. See [docs/REFERENCES.md](docs/REFERENCES.md).

---

## Repository Layout

```
MetaHackathonPrep/
├── models.py               # CropAction, CropObservation, CropState (Pydantic)
├── client.py               # WebSocket EnvClient subclass
├── inference.py            # Competition inference entrypoint
├── openenv.yaml            # OpenEnv metadata
├── Dockerfile              # HuggingFace Spaces container
├── pyproject.toml          # Package config
├── requirements.txt        # Dependencies
├── server/
│   ├── app.py              # FastAPI server + /tasks, /grader, /baseline, /ceiling
│   ├── environment.py      # CropEnvironment (reset/step/state)
│   ├── crop_sim.py         # WOFOST-inspired simulator
│   ├── crop_params.py      # WOFOST parameter library + YAML loader
│   ├── scenarios.py        # Seeded weather + scenario generation
│   ├── grader.py           # Multi-metric deterministic scoring
│   ├── reward.py           # Dense step + trajectory rewards
│   ├── rubric.py           # RFC 004 rubric wrapper
│   ├── advisory.py         # Deterministic advisory text
│   ├── constants.py        # Shared thresholds and weights
│   └── tasks.py            # Task definitions (3 levels)
├── agent/
│   ├── inference.py        # Greedy heuristic + LLM inference
│   ├── training_adapter.py # Discrete RL action adapter (11 actions)
│   └── benchmark_sweep.py  # Multi-seed evaluation utility
├── configs/                # YAML crop/soil profiles (wheat_nl, wheat_iowa, wheat_punjab)
├── docs/
│   ├── ARCHITECTURE.md     # Deep technical documentation
│   └── REFERENCES.md       # 17 scientific references
├── examples/               # Runnable client + benchmark examples
└── tests/                  # 82 tests (smoke, integration, submission, WebSocket)
```

---

## Setup & Usage

### Install

```bash
pip install -r requirements.txt
```

### Start Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Verify

```bash
curl http://localhost:7860/health    # {"status": "healthy"}
curl http://localhost:7860/tasks     # List all 3 tasks
curl http://localhost:7860/baseline  # Greedy scores (seed=190)
curl http://localhost:7860/ceiling   # Oracle scores (seed=190)
```

---

## Running Inference

### Heuristic mode (no LLM)

```bash
python inference.py
```

### LLM mode

Create a `.env` file (auto-loaded by `python-dotenv`):

```dotenv
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
API_KEY=your_api_key_here
```

```bash
python inference.py              # all tasks
TASK_ID=1 python inference.py    # single task
```

Optional trajectory export for offline RL:

```bash
TRAJECTORY_OUTPUT=trajectories/run python inference.py
```

> If the LLM API returns 3+ consecutive errors, inference automatically falls back to the greedy heuristic.

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | LLM mode | OpenAI-compatible endpoint |
| `MODEL_NAME` | LLM mode | Model identifier |
| `API_KEY` | LLM mode | Auth token (evaluator injects this) |
| `HF_TOKEN` | No | Fallback for `API_KEY` |
| `ENV_URL` | No | Server URL (default: `http://localhost:7860`) |
| `TASK_ID` | No | Single task ID (default: all 3) |
| `SEED` | No | Random seed (default: 190) |
| `TRAJECTORY_OUTPUT` | No | JSONL export path for offline RL |

---

## Deployment

### Docker

```bash
docker build -t crop-management .
docker run -p 7860:7860 crop-management
```

### Hugging Face Spaces

```bash
openenv push
```

### API Surface

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check |
| POST | `/reset` | Start new episode |
| POST | `/step` | Submit action |
| GET | `/state` | Internal state |
| GET | `/tasks` | Task metadata |
| GET | `/baseline` | Greedy scores |
| GET | `/ceiling` | Oracle scores |
| POST | `/grader` | Grade an episode |
| WS | `/ws` | Persistent session |
| GET | `/web` | HF Space UI |
| GET | `/docs` | Interactive API docs |

### Pre-Submission Validation

```bash
bash scripts/validate-submission.sh https://rijulgn-crop-management-env.hf.space
```

---

## Testing

```bash
python -m pytest tests/test_smoke.py -q          # 65 smoke/RL/rubric tests
python -m pytest tests/ -q                        # all 82 tests
python -m agent.benchmark_sweep --start-seed 190 --count 10  # multi-seed sweep
python examples/direct_benchmark.py               # minimal oracle benchmark
```

The full suite covers: determinism, difficulty ordering, reward monotonicity, delta-reward stress relief, probe alignment, budget exhaustion, passive-policy regression, and WebSocket transport.

---
