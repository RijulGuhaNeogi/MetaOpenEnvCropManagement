# Codebase Audit — Precision Agriculture Crop Management OpenEnv

**Date:** April 2, 2026  
**Auditor:** Automated  
**Baseline scores (seed=42):** Task1=0.6586, Task2=0.7803, Task3=0.7799, Overall=0.7396

---

## 1. BUGS

### BUG-1: Inverted Difficulty Scores (HIGH)
**Files:** `server/grader.py`, `server/scenarios.py`  
**Problem:** Task 1 (Easy) scores 0.6586 but Task 2 (Medium) scores 0.7803. The greedy heuristic performs *better* on harder tasks. This signals broken difficulty calibration.  
**Root cause:** Task 1 weights yield at 70%, but the Netherlands scenario's `compute_potential_yield()` with unlimited water sets a very high target — the actual yield under normal conditions can't approach it. Meanwhile Tasks 2-3 diversify weight across water_efficiency and cost_efficiency, which the greedy heuristic easily scores well on (it's conservative by design).  
**Fix:** Adjust Task 1 weights or lower the target yield multiplier for easy scenarios. Also ensure compute_potential_yield uses comparable conditions.

### BUG-2: Action Record Logs Original Amount on Budget Failure (LOW)
**File:** `server/environment.py` (line ~147)  
**Problem:** When an action fails the budget check, `action_type` is overwritten to "wait" and `cost/irrig_cm/n_kg` are zeroed, but the `action_record["amount"]` still contains the original requested amount. This makes the action log misleading.  
**Fix:** Set `amount = 0.0` when the action degrades to "wait" due to budget or validation failure.

### BUG-3: Missing Max-Steps Guard (MEDIUM)
**File:** `server/environment.py`  
**Problem:** If `max_duration` is large and weather is cold (no crop development), an agent that always waits could create an extremely long episode. The only termination is DVS>=2.0 or `current_day >= max_duration`. With 7-day steps, Netherlands can have up to 40 steps (280/7), which is fine — but there's no hard cap on step_count.  
**Fix:** Add a `MAX_STEPS = 60` constant and terminate if step_count exceeds it.

### BUG-4: Forecast Noise Repeats Predictably (LOW)
**File:** `server/crop_sim.py` (get_weather_forecast method)  
**Problem:** The "noise" added to forecasts uses `((d * 17) % 7 - 3) * 0.4` which produces a repeating 7-day pattern. An LLM could learn to perfectly de-noise forecasts.  
**Fix:** Cosmetic — acceptable for hackathon scope, but could use a seed-based hash for more varied noise.

---

## 2. ISSUES / CODE QUALITY

### ISSUE-1: Maize Params Unused (Dead Code)
**File:** `server/crop_sim.py`  
**Problem:** `CROP_LIBRARY["maize"]` and `MAIZE_PARTITION` are defined but never used — all 3 tasks are wheat.  
**Keep as-is:** Shows extensibility. Just add a comment noting it's for future use.

### ISSUE-2: pyproject.toml Has Stale Comment
**File:** `pyproject.toml`  
**Problem:** Contains `# Concern 10 fix` comment which is leftover from the old ML Scheduler project.  
**Fix:** Remove the stale comment.

### ISSUE-3: No .gitignore or .dockerignore
**Problem:** `__pycache__/` directories get checked in and Docker builds copy them.  
**Fix:** Add both files.

### ISSUE-4: No Tests
**Problem:** Zero test files. Hackathon rubric gives 15% for code quality.  
**Fix:** Add a basic `tests/test_smoke.py` that verifies reset/step/grade determinism.

### ISSUE-5: Instructions Repeated Every Observation
**File:** `server/environment.py` (_build_observation)  
**Problem:** Full task instructions are included in every observation. For LLM inference, this wastes tokens.  
**Accepted:** The OpenEnv spec expects observations to be self-contained. Keep as-is.

### ISSUE-6: server/__init__.py is Empty
**File:** `server/__init__.py`  
**OK:** Required as a package marker. No change needed.

---

## 3. HACKATHON COMPLIANCE CHECK

| Requirement | Status | Notes |
|-------------|--------|-------|
| OpenEnv spec (Environment, Action, Observation, State) | ✅ | Inherits correctly from openenv base classes |
| create_app() usage | ✅ | Uses create_app(CropEnvironment, CropAction, CropObservation) |
| /health endpoint | ✅ | Provided by create_app() |
| /reset POST | ✅ | Returns CropObservation |
| /step POST | ✅ | Returns CropObservation with done/reward |
| /state GET | ✅ | Returns CropState |
| /tasks custom endpoint | ✅ | Lists 3 tasks |
| WebSocket multi-step | ✅ | EnvClient works |
| Dockerfile builds | ✅ | python:3.11-slim, port 7860 |
| No runtime API calls from env | ✅ | All data generated from seed |
| Deterministic scoring | ✅ | Same seed → same score |
| 3+ tasks with graders | ✅ | 3 tasks, all graded 0.0-1.0 |
| inference.py uses env vars | ✅ | API_BASE_URL, MODEL_NAME, HF_TOKEN |
| inference.py outputs === RESULTS === | ✅ | Correct format |
| Inference < 20 minutes | ✅ | ~10 seconds locally |
| Self-contained Docker (no external data) | ✅ | Weather/params generated in-process |
| openenv.yaml present | ✅ | Complete metadata |
| requirements.txt present | ✅ | All deps listed |

### Potential Compliance Risks:
1. **Port mismatch:** Dockerfile exposes 7860, but app.py __main__ uses 8000. The CMD uses 7860 — this is correct for HF Spaces.
2. **ENV_URL default:** inference.py defaults to localhost:8000. During competition, ENV_URL should be set. This is standard.

---

## 4. WHY NOT PCSE?

The original plan.md explicitly called for PCSE (`Wofost72_WLP_FD`). We opted for a custom simulator instead. Here's the trade-off:

### Advantages of PCSE:
- **Scientific credibility:** PCSE is the reference WOFOST implementation, peer-reviewed and validated
- **Richer dynamics:** Full NPK cycles, soil water layers, vernalization, detailed phenology
- **Judging appeal:** "Uses real WOFOST model" sounds better than "inspired by WOFOST"

### Why We Didn't Use PCSE:
- **Complex setup:** PCSE requires YAML crop parameter files, soil databases, agromanagement calendars, and a WeatherDataProvider object. Setting up `pcse.Models.Wofost72_WLP_FD` needs ~50 lines of boilerplate config
- **Data files:** PCSE's crop/soil databases are large (~30MB) and need to be bundled in the Docker image
- **Signal API complexity:** `model._send_signal(signal=pcse.signals.irrigate)` is fragile and versions can break
- **Docker image size:** PCSE + dependencies (PyYAML, pandas, numpy) adds ~150MB to the image
- **Debuggability:** A custom 200-line simulator is fully transparent — judges can read and understand every line
- **License:** PCSE uses EUPL-1.2, which is compatible but adds legal review overhead

### Recommendation:
**Consider switching to PCSE** if there's time. The plan.md explicitly called for it and the "Real-world utility" rubric (30%) would score higher with a validated crop model. The custom simulator is adequate but PCSE would strengthen the submission.

If switching to PCSE:
1. `pip install pcse` (~5MB, pulls numpy/pandas)
2. Embed crop YAML files from `pcse.util` into `server/data/`
3. Use `pcse.Models.Wofost72_WLP_FD` with custom WeatherDataProvider
4. Keep our grader/reward/tasks — only replace crop_sim.py internals

---

## 5. IMPROVEMENTS TO IMPLEMENT

### Priority 1: Fix difficulty calibration
- Adjust compute_potential_yield to use moderate (not unlimited) resources
- OR adjust Task 1 weights to 60% yield, 20% harvest, 20% timing

### Priority 2: Add basic test suite
- `tests/test_determinism.py`: same seed → same scores
- `tests/test_env.py`: reset/step cycle completes without error

### Priority 3: Add .gitignore, .dockerignore

### Priority 4: Clean up pyproject.toml, requirements.txt

### Priority 5: Add docstrings/comments throughout
