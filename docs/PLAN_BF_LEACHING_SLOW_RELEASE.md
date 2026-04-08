# Execution Plan: N Leaching + `fertilize_slow` (Combo B+F)

**Branch**: `RGN_llmDifficultyIncrease`  
**Estimated effort**: ~3-4 hours (implement + test)  
**Prompt bloat**: +6 lines (~3% increase to system prompt)  
**Risk**: Main branch untouched — safe fallback if anything breaks

---

## 1. Design Summary

### N Leaching (internal mechanic — no new action)
When fertilizer is applied and soil is wet (SM near field capacity) or rain falls during the 7-day advance, a fraction of applied N washes below the root zone. This is deterministic (same weather + SM + fert = same leaching).

- **Regular fertilizer**: Loses 15-40% of applied N in wet conditions
- **Slow-release fertilizer**: Loses only 5-12% (70% leach-resistant)

### `fertilize_slow` (visible new action)
Same fert windows, same 50kg cap, same 2-application limit. Different economics:

| Property | `fertilize` | `fertilize_slow` |
|----------|-------------|-------------------|
| Cost per kg | base rate | 1.5× base rate |
| Immediate N recovery | 100% of N_RECOV | 70% of N_RECOV |
| Remaining 30% | — | Released over 14 days (slow-release pool) |
| Leach resistance | None (full leaching rate) | 70% resistant (0.30× leaching rate) |

**Decision matrix the agent must solve:**

| | Dry forecast | Wet forecast |
|---|---|---|
| **Regular fert** | ✅ Cheap + full N | ❌ N washes out |
| **Slow-release** | ❌ Overpaying for protection | ✅ N retained |

---

## 2. Expected Score Impact

| Policy | Task 3 Before | Task 3 After | Why |
|--------|---------------|--------------|-----|
| **Oracle** | ~0.88 | ~0.88-0.90 | Weather-aware fert type preserves N better |
| **Greedy** | ~0.70 | ~0.62-0.66 | Always uses regular → N leaches in rain → lower yield |
| **Weak LLM** | ~0.80 | ~0.72-0.78 | Follows "rain → slow-release" but pays premium even when dry |
| **Frontier LLM** | ~0.80 | ~0.80-0.85 | Correctly evaluates rain magnitude vs premium cost |

**New gaps**: Greedy↔Frontier: ~20-23pts. Frontier↔Weak: ~5-10pts. Frontier↔Oracle: ~5-8pts.

---

## 3. System Prompt Addition (~6 lines, inside existing Step 2)

```
NEW: Two fertilize types:
  "fertilize" = cheap, full immediate N boost, but N leaches if soil wet or rain coming.
  "fertilize_slow" = 1.5× cost, resists leaching. Use when rain forecast >0.5cm.
  Rule: rain3d>0.5 → fertilize_slow. rain3d<0.2 → regular fertilize. Between → judgment call.
  The advisory will warn you about leaching risk when conditions are wet.
```

---

## 4. Phase-by-Phase Implementation

### Phase 1: Simulator Core

#### `server/crop_params.py` — 3 new params in WOFOSTCropParams
```python
# N leaching & slow-release parameters
LEACH_RATE: float = 0.35            # fraction of excess-water N lost per cm
SLOW_RELEASE_IMMEDIATE: float = 0.70 # fraction of slow-release N applied immediately
SLOW_RELEASE_LEACH_FACTOR: float = 0.30  # leaching multiplier (0.30 = 70% resistant)
```
Add after existing N_FACTOR_INIT (around line 84). No changes to WHEAT_NL/WHEAT_IOWA/WHEAT_PUNJAB configs (inherit defaults).

#### `server/crop_sim.py` — Core leaching + slow-release mechanics

**New state** (in `__init__`, after `self.total_n`):
```python
self.slow_release_pool: float = 0.0    # pending slow-release N (kg/ha)
self.total_n_leached: float = 0.0      # cumulative N lost to leaching
self._n_applied_this_advance: float = 0.0  # for leaching calc within advance
self._slow_release_this_advance: bool = False
```

**Modified `advance()`** (around line 169):
```python
def advance(self, days: int, irrigation_cm: float = 0.0,
            n_kg_ha: float = 0.0, slow_release: bool = False):
    # Nitrogen injection
    if n_kg_ha > 0:
        if slow_release:
            immediate_frac = self.crop.SLOW_RELEASE_IMMEDIATE  # 0.70
            immediate_kg = n_kg_ha * immediate_frac
            deferred_kg = n_kg_ha - immediate_kg
            increase = immediate_kg * self.crop.N_RECOV
            self.n_factor = min(1.0, self.n_factor + increase)
            self.slow_release_pool += deferred_kg
        else:
            increase = n_kg_ha * self.crop.N_RECOV
            self.n_factor = min(1.0, self.n_factor + increase)
        self.total_n += n_kg_ha
        self._n_applied_this_advance = n_kg_ha
        self._slow_release_this_advance = slow_release
    else:
        self._n_applied_this_advance = 0.0
        self._slow_release_this_advance = False

    # existing irrigation + simulate_day loop unchanged
```

**Modified `_simulate_day()`** — add leaching after water balance (after SM clamp, around line 190):
```python
# N leaching: if recently fertilized and soil is wet
if self._n_applied_this_advance > 0:
    excess_water = max(0.0, sm - sp.SMFCF)
    if excess_water > 0.01:
        leach_factor = (self.crop.SLOW_RELEASE_LEACH_FACTOR
                        if self._slow_release_this_advance else 1.0)
        n_lost = excess_water * self.crop.LEACH_RATE * leach_factor * 0.01
        self.n_factor = max(self.crop.N_FACTOR_FLOOR,
                            self.n_factor - n_lost)
        self.total_n_leached += n_lost

# Slow-release daily drip (pool releases over ~14 days)
if self.slow_release_pool > 0:
    daily_release = self.slow_release_pool / 14.0
    self.n_factor = min(1.0, self.n_factor + daily_release * self.crop.N_RECOV)
    self.slow_release_pool = max(0.0, self.slow_release_pool - daily_release)
```

After the 7-day loop completes, reset: `self._n_applied_this_advance = 0.0`

---

### Phase 2: Constants & Models

#### `server/constants.py` — 2 new constants (append after line 19)
```python
SLOW_RELEASE_COST_MULTIPLIER = 1.5
LEACH_RAIN_THRESHOLD = 0.5   # cm rain in 3d above which leaching intent penalty fires
```

#### `models.py`
- **CropAction docstring** (line 27): Add `"fertilize_slow"` to valid types
- **CropState** (after line 176): Add `slow_release_pool: float = 0.0` and `total_n_leached: float = 0.0`
- **ResourcesUsed** (after line 66): Add `slow_release_cost_per_kg: float = 0.0`

---

### Phase 3: Environment

#### `server/environment.py` — Handle `fertilize_slow` action

**Action validation** — add `"fertilize_slow"` to valid action types (wherever the set is defined).

**Action handling** (after the existing `elif action_type == "fertilize":` block, ~line 286):
```python
elif action_type == "fertilize_slow":
    amount = min(amount, 50.0)
    if amount <= 0:
        conflicts.append("Fertilizer amount must be > 0. Treating as wait.")
        action_type = "wait"
        amount = 0.0
    else:
        cost = amount * scenario["fertilizer_cost"] * SLOW_RELEASE_COST_MULTIPLIER
        n_kg = amount
        slow_release = True
```

**Budget check** (line 289): Add `"fertilize_slow"` to the tuple:
```python
if cost > budget_remaining and action_type in ("irrigate", "fertilize", "fertilize_slow"):
```

**Recording** (line 305): Add `"fertilize_slow"` as equivalent to fertilize for event counting:
```python
elif action_type in ("fertilize", "fertilize_slow") and n_kg > 0.0:
```

**Advance call** (line 400): Pass slow_release flag:
```python
self._sim.advance(step_days, irrigation_cm=irrig_cm, n_kg_ha=n_kg,
                  slow_release=slow_release)
```

**Dose hint** (~line 425): Extend condition to include `"fertilize_slow"`.

**ResourcesUsed** — set `slow_release_cost_per_kg` field.

**CropState sync** — write `slow_release_pool` and `total_n_leached` from sim.

---

### Phase 4: Reward

#### `server/reward.py` — Weather-aware fert type bonus/penalty

In the `elif action_type == "fertilize":` block (line 150), change condition to:
```python
elif action_type in ("fertilize", "fertilize_slow"):
```

After computing the base reward (before `return _clamp(...)` at line 163), add:
```python
# Weather-awareness bonus/penalty for fert type choice
is_slow = (action_type == "fertilize_slow")
if forecast_rain_3d > LEACH_RAIN_THRESHOLD:
    reward += 0.02 if is_slow else -0.03   # wet: slow-release correct
elif forecast_rain_3d < 0.2:
    reward += 0.01 if not is_slow else -0.02  # dry: regular correct
```

Add `forecast_rain_3d: float = 0.0` parameter to `compute_step_reward()` if not already present.

**Delta reward**: No changes needed — leaching naturally reduces `post_n - pre_n`, so the delta self-corrects.

---

### Phase 5: Advisory

#### `server/advisory.py` — Leaching warnings + slow-release mention

In fert window blocks (lines 132-141 and 170-179), add after existing guidance:
```python
# Leaching risk + slow-release hint
if sm > sp_smfcf - 0.03 or forecast_rain_3d > 0.5:
    parts.append("⚠ Leaching risk: soil is wet or rain expected. "
                 "Consider slow-release fertilizer (fertilize_slow) to protect N.")
else:
    parts.append("Low leaching risk — regular fertilizer is cost-effective.")
```

---

### Phase 6: Agent

#### `agent/inference.py` — Oracle, Greedy, System Prompt

**System prompt** (after line 89, inside step 2 FERTILIZE): Add the 6 lines shown in Section 3 above.

**Greedy policy** (~line 551): **No change**. Greedy always uses `"fertilize"`. This is intentional — greedy degrades when rain causes leaching.

**Oracle policy** (~line 782): Add weather-aware fert type selection:
```python
# Choose fert type based on 3-day rain forecast
rain_3d = sum(getattr(f, "rain", 0.0) for f in forecast[:3]) if forecast else 0.0
fert_type = "fertilize_slow" if rain_3d > 0.5 else "fertilize"
return {"action_type": fert_type, "amount": kg}
```

**`compress_observation()`**: When in fert window, add line showing slow-release option + cost.

---

### Phase 7: Supporting Files

#### `agent/training_adapter.py` — 3 new discrete actions
```python
"fertilize_slow_small": ("fertilize_slow", 15.0),
"fertilize_slow_medium": ("fertilize_slow", 30.0),
"fertilize_slow_large": ("fertilize_slow", 50.0),
```

#### `server/grader.py` (~line 88) — Include fertilize_slow in timing filter:
```python
action["action_type"] in ("fertilize", "fertilize_slow")
```

---

### Phase 8: Tests

#### `tests/test_smoke.py` — 5 new tests

1. **`test_n_leaching_reduces_n_factor_in_wet_soil`** — Apply regular fert at high SM → verify reduced n_gain vs dry soil
2. **`test_slow_release_resists_leaching`** — Same wet conditions, slow_release=True → verify higher n_factor retained
3. **`test_slow_release_costs_more`** — Step with fertilize_slow → verify cost = 1.5× fertilizer_cost_per_kg
4. **`test_slow_release_pool_drains_gradually`** — Apply slow-release → advance 2 steps → verify pool decreases and n_factor rises
5. **`test_fertilize_slow_valid_action`** — Step with fertilize_slow → no error → fert_events_count increments

---

### Phase 9: Documentation

- **README.md**: Add N Leaching and Slow-Release to Features. Update Actions table. Update discrete actions.
- **openenv.yaml**: Add `fertilize_slow` to action type description.
- **docs/ARCHITECTURE.md**: Brief leaching mechanics section.

---

## 5. File Impact Summary

| # | File | Lines Changed | Risk | Phase |
|---|------|--------------|------|-------|
| 1 | server/crop_params.py | +3 lines | Very Low | 1 |
| 2 | server/crop_sim.py | +30 lines | **Medium** | 1 |
| 3 | server/constants.py | +2 lines | Very Low | 2 |
| 4 | models.py | +4 lines | Low | 2 |
| 5 | server/environment.py | +20 lines | **Medium** | 3 |
| 6 | server/reward.py | +8 lines | Low | 4 |
| 7 | server/advisory.py | +6 lines | Low | 5 |
| 8 | agent/inference.py | +12 lines (oracle) +6 lines (prompt) | **Medium** | 6 |
| 9 | agent/training_adapter.py | +3 lines | Very Low | 7 |
| 10 | server/grader.py | +1 line | Very Low | 7 |
| 11 | tests/test_smoke.py | +60 lines | Low | 8 |
| 12 | README.md | +15 lines | Low | 9 |
| 13 | openenv.yaml | +1 line | Very Low | 9 |

**Total**: ~170 new/modified lines across 13 files.

---

## 6. Dependency Chain

```
Phase 1: crop_params.py → crop_sim.py → [unit test leaching]
Phase 2: constants.py + models.py (parallel with Phase 1)
Phase 3: environment.py (depends on Phase 1+2)
Phase 4: reward.py (depends on Phase 2 constants)
Phase 5: advisory.py (independent)
Phase 6: inference.py (depends on Phase 3)
Phase 7: training_adapter.py + grader.py (depend on Phase 2)
Phase 8: tests (depend on Phase 3+4)
Phase 9: docs (last, independent)
```

---

## 7. Verification Checklist

- [ ] `pytest tests/` — all existing + 5 new tests pass
- [ ] Same-seed determinism still holds (leaching is deterministic from weather+SM)
- [ ] Regular fert in rain → visibly lower n_gain in delta reward
- [ ] Slow-release in rain → n_gain preserved
- [ ] Slow-release pool drains over 2 steps (gradual release visible)
- [ ] Greedy still works (uses regular fert, just scores lower)
- [ ] Oracle uses slow-release in wet, regular in dry
- [ ] Cost accounting: fertilize_slow charges 1.5× per kg
- [ ] LLM sees slow-release option in system prompt and advisory
- [ ] `fert_events_count` treats fertilize_slow same as fertilize (max 2)
- [ ] Task difficulty ordering preserved: Task1 > Task2 > Task3 scores

---

## 8. Rollback Plan

If anything breaks submission gates:
1. `git checkout main` — original passing submission is untouched
2. Submit from main with zero risk

The `RGN_llmDifficultyIncrease` branch is purely additive. No existing behavior is removed—only augmented.
