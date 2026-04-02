# Improvement Plan — Audit-Driven Score & RL Quality Fixes

**Date:** April 3, 2026  
**Baseline (seed=42):** Task1=0.8166, Task2=0.8049, Task3=0.6986, Overall=0.7734  
**Constraint:** `server/grader.py` is FROZEN — all changes target the reward/environment/inference layer only.

---

## Summary

Audit of the current codebase reveals **3 high-impact score-limiting issues** and several RL signal gaps.
The step-reward layer doesn't align with the grader's actual metric weights (35% yield, 20% water, 18% cost, 15% timing, 12% harvest), causing RL agents to optimize for the wrong thing.
The greedy heuristic and LLM prompt also leave easy points on the table with fixed irrigation doses and early fertilization timing.

**Execution guardrail:** scientific accuracy is useful only insofar as it strengthens the environment for training and evaluation. Do **not** introduce realism-driven changes that weaken grader alignment, reduce reward clarity, or make the RL signal noisier.

---

## Phase A: Reward–Grader Alignment *(highest ROI for RL quality)*

### A1. Fertilizer timing reward should peak at DVS 0.3 / 0.6, not be flat across windows

| | |
|---|---|
| **Problem** | `_fertilizer_window_target()` in `server/reward.py` gives flat reward for DVS 0.20–0.40 and 0.50–0.70. But `grader.py` scores `1.0 - \|dvs - 0.3\| / 0.5` — a triangular peak at 0.3 and 0.6. Agent is incentivized to fertilize at window edges where grader penalizes. |
| **Fix** | Add a proximity bonus that peaks at 0.30 and 0.60 within the window, e.g. `proximity = 1.0 - abs(dvs - center) / 0.10` scaled into the window bonus. |
| **Files** | `server/reward.py` — `_fertilizer_window_target()` + fertilize branch of `compute_step_reward()` |

### A2. Add cumulative-water pressure to irrigation reward

| | |
|---|---|
| **Problem** | Grader gives 20% weight to `water_efficiency = 1 - total_water / 50`. Step reward has **zero** signal about cumulative water usage — agent doesn't learn to conserve. |
| **Fix** | Add `water_pressure = min(0.04, total_water / 50 * 0.04)` penalty to irrigation rewards. Requires passing `total_water` into `compute_step_reward()`. Keep the penalty bounded at **max 0.04** so it nudges conservation without overwhelming the local agronomic signal. |
| **Why this is safe** | Water efficiency is currently only enforced by the sparse terminal grade. This adds a dense per-step learning signal, not a second terminal objective. |
| **Files** | `server/reward.py` — signature + irrigate branch; `server/environment.py` — call site |

### A3. Add budget pressure to step reward

| | |
|---|---|
| **Problem** | Grader gives 18% weight to `cost_efficiency = 1 - cost / budget`. Step reward only has `cost_penalty = min(0.03, spend_ratio * 0.03)` in delta_reward — max 0.03. Agent gets almost no signal to conserve budget. |
| **Fix** | Scale cumulative cost pressure, e.g. `spend_pressure = min(0.06, (total_cost / budget) * 0.06)` applied to non-wait actions. Keep the ceiling at **max 0.06** so obviously beneficial actions can still remain net-positive. |
| **Why this is safe** | Cost efficiency is also sparse today. This adds bounded dense pressure against overspending rather than changing the grader itself. |
| **Files** | `server/reward.py` — `compute_delta_reward()` or `compute_step_reward()` |

---

## Phase B: Greedy Heuristic & LLM Prompt *(direct score lift)*

### B1. Compute deficit-optimal irrigation dose instead of fixed amounts

| | |
|---|---|
| **Problem** | Greedy uses fixed 2.5 / 3.0 cm. If deficit is only 1.2 cm, we overshoot (wastes water + budget — 20%+18% of grade). |
| **Prerequisite** | `greedy_action()` does not currently receive root-zone depth. Expose `rooting_depth_cm` in `control_features` from `_build_observation()` before implementing the heuristic. |
| **Fix** | Once exposed, compute `desired = max(0.5, min(5.0, (0.30 - sm) * rooting_depth_cm))` and use that instead of fixed 2.5 / 3.0 cm. Mirror the same logic in the LLM prompt. |
| **Files** | `server/environment.py` — `_build_observation()` control features; `inference.py` — `greedy_action()` irrigation branch |

### B2. Align fertilizer timing and amount with reward targets

| | |
|---|---|
| **Problem** | Two independent misalignments exist. First, greedy fertilizes as soon as DVS enters the window at 0.20, while grader timing peaks nearer 0.3 and 0.6. Second, `inference.py` uses 15 / 12 kg but `reward.py` currently targets 18 / 15 kg inside `_fertilizer_window_target()`. |
| **Fix** | Tighten greedy timing windows to `(0.27, 0.40)` and `(0.57, 0.70)`. Also align the heuristic amounts with the reward targets by changing `FERT_STAGE1_KG = 18.0` and `FERT_STAGE2_KG = 15.0`. Mirror both changes in the test heuristic. |
| **Files** | `inference.py` — `FERT_STAGE1_DVS`, `FERT_STAGE2_DVS`, `FERT_STAGE1_KG`, `FERT_STAGE2_KG`; `tests/test_smoke.py` — `_run_episode()` mirror |

### B3. Improve LLM SYSTEM_PROMPT with grader-aware guidance

| | |
|---|---|
| **Problem** | Prompt says "fertilize at ~0.3 and ~0.6" loosely but doesn't explain scoring decay, that water_efficiency is 20% of grade, or that the preferred fertilizer amounts should match reward targets. |
| **Fix** | Add explicit budget-conservation advice, precise timing targets, deficit-based irrigation guidance, and align prompt fertilizer amounts with the reward targets. |
| **Files** | `inference.py` — `SYSTEM_PROMPT` |

---

## Phase C: Bug Fixes & Signal Quality

### C1. Fix `dvs_distance_to_next_window` semantics after DVS 0.70

| | |
|---|---|
| **Problem** | After DVS 0.70, returns `dvs - 0.70` (distance FROM window, grows forever). Misleading for RL agent. |
| **Fix** | After both windows passed, return `-1.0` (sentinel: "no more windows"). |
| **Files** | `server/environment.py` — `_build_observation()` control features block |

### C2. Fix `estimated_budget_to_finish` to use actual scenario costs

| | |
|---|---|
| **Problem** | Uses magic numbers `12.0` and `18.0` instead of actual `irrigation_cost` and `fertilizer_cost` from scenario. |
| **Fix** | `(forecast_rain_7d < 1.0) * irrig_cost * 4.0 + (sim.n_factor < 0.6) * fert_cost * 15.0` |
| **Files** | `server/environment.py` — `_build_observation()` |

### C3. `budget_remaining_ratio` is already implemented

| | |
|---|---|
| **Finding** | `budget_remaining_ratio` already exists in `control_features`, so no code change is required. |
| **Action** | Remove this from the implementation backlog and keep it only as a verified-complete note. |
| **Files** | `server/environment.py` — control features dict |

### C4. Fix probe `force_forecast_rain` to apply relative to sim start position

| | |
|---|---|
| **Problem** | `force_forecast_rain` overwrites weather days 0,1,2... but probes with `start_at_dvs` fast-forward past those days, so forced rain is already in the past. |
| **Fix** | Store `force_forecast_rain` and apply in `_apply_start_state_overrides()` relative to `sim.current_day`. |
| **Files** | `server/scenarios.py` — `generate_probe_scenario()`; `server/environment.py` — `_apply_start_state_overrides()` |

### C5. Sync test greedy with inference greedy

| | |
|---|---|
| **Problem** | Test uses `"s1"/"s2"` keys, inference uses `"stage1"/"stage2"`. Different fertilizer windows too (test: 0.20, inference: 0.20). Will diverge as B2 tightens inference windows. |
| **Fix** | Import `greedy_action` from `inference.py` into tests, or extract shared constants. |
| **Files** | `tests/test_smoke.py` — `_run_episode()`; `inference.py` — constants |

---

## Phase D: Test Coverage Gaps

| ID | Description |
|----|-------------|
| **D1** | Test reward-grader alignment: run a full episode, verify higher cumulative step reward correlates with higher final grade. |
| **D2** | Test all 8 discrete actions in adapter (currently only 2 of 8 tested). |
| **D3** | Add budget exhaustion test: tiny budget, verify actions gracefully degrade to wait without crash. |
| **D4** | Use `pytest.approx()` for reward monotonicity assertions to avoid float precision failures. |

---

## File Impact Map

| File | Items |
|------|-------|
| `server/reward.py` | A1, A2, A3 |
| `server/environment.py` | A2 call site, B1 prerequisite, C1, C2, C4 |
| `inference.py` | B1, B2, B3 |
| `server/scenarios.py` | C4 |
| `tests/test_smoke.py` | B2 mirror, C5, D1–D4 |
| `server/grader.py` | **FROZEN** — reference only |

---

## Verification Checklist

- [ ] `python -m pytest tests/test_smoke.py -q` — all tests pass
- [ ] Baseline scores improve (especially Task3 which is budget/water constrained)
- [ ] `test_difficulty_ordering` still passes
- [ ] `git diff server/grader.py` shows zero modifications

---

## Priority

1. **Phase A** — fixes RL reward signal quality (misaligned incentives hurt training)
2. **Phase B** — lifts baseline scores directly (better heuristic = higher immediate scores)
3. **Phase C** — correctness fixes (wrong semantics, magic numbers, stale rain data)
4. **Phase D** — test coverage polish (prevents regressions)

**Guardrail for all phases:** if a scientifically motivated refinement introduces weaker grader alignment, noisier step rewards, or worse baseline behavior, reject it even if it appears more realistic on paper.

---

## Obsolete Documentation

Review these files for outdated content after implementation:

| File | Status | Notes |
|------|--------|-------|
| `AUDIT.md` | Partially outdated | Baseline scores and bug list predate RL changes. BUG-3 (MAX_STEPS) is fixed. ISSUE-4 (no tests) is fixed. |
| `issuesDetected.md` | Review needed | May duplicate AUDIT.md findings or contain stale references. |
| `RLimprovementPlan.md` | Completed | All 5 phases implemented. Can be archived or marked as done. |
| `FUTURE_SCOPE_PCSE.md` | Still relevant | PCSE migration guide — not acted on, still valid for future. |
| `plan.md` | Partially outdated | Original hackathon plan; PCSE references never implemented. |
| `hackathonBriefing.md` | Review needed | May contain stale timelines or scope assumptions. |
