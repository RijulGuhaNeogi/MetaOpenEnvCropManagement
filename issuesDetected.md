

Let me do a thorough re-read of every file with fresh eyes, informed by the bootcamp insights.  

Audit all server-side files

Audit client and inference files

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/server)

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/models.py)

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/client.py)

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/inference.py)

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/openenv.yaml)

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/pyproject.toml)

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/tests/test_smoke.py)

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/README.md)

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/server/Dockerfile)

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/requirements.txt)

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/server/crop_sim.py)

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/server/environment.py)

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/server/grader.py)

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/server/reward.py)

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/server/scenarios.py)

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/server/tasks.py)

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/server/app.py)

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/models.py)

Created memory file [](file:///c%3A/Users/rguhaneogi/AppData/Roaming/Code/User/workspaceStorage/2f1d98ed382d762b16cb893c28af5c19/GitHub.copilot-chat/memory-tool/memories/OGE4ZGU4OGYtMzIzZS00MDk5LWI1MzMtYTgyNGI2ZDIzYzVm/audit-findings.md)

Read [](file:///c%3A/Users/rguhaneogi/AppData/Roaming/Code/User/workspaceStorage/2f1d98ed382d762b16cb893c28af5c19/GitHub.copilot-chat/chat-session-resources/8a8de88f-323e-4099-b533-a824b6d23c5f/toolu_bdrk_01XMgdVyzuevmnjmoVn64JbQ__vscode-1774808439952/content.txt#1-1), lines 1 to 250

Read [](file:///c%3A/Users/rguhaneogi/AppData/Roaming/Code/User/workspaceStorage/2f1d98ed382d762b16cb893c28af5c19/GitHub.copilot-chat/chat-session-resources/8a8de88f-323e-4099-b533-a824b6d23c5f/toolu_bdrk_01KQe2nv8VP82weeyFxc8GQW__vscode-1774808439953/content.txt#1-1), lines 1 to 250

Read [](file:///c%3A/Users/rguhaneogi/AppData/Roaming/Code/User/workspaceStorage/2f1d98ed382d762b16cb893c28af5c19/GitHub.copilot-chat/chat-session-resources/8a8de88f-323e-4099-b533-a824b6d23c5f/toolu_bdrk_01XMgdVyzuevmnjmoVn64JbQ__vscode-1774808439952/content.txt#250-250), lines 250 to 300

Read [](file:///c%3A/Users/rguhaneogi/AppData/Roaming/Code/User/workspaceStorage/2f1d98ed382d762b16cb893c28af5c19/GitHub.copilot-chat/chat-session-resources/8a8de88f-323e-4099-b533-a824b6d23c5f/toolu_bdrk_01KQe2nv8VP82weeyFxc8GQW__vscode-1774808439953/content.txt#250-250), lines 250 to 330

Now let me synthesize both audit reports into a prioritized plan. 

Analyzing your answers...

Good — full sweep it is. On the action validation: I recommend **keeping runtime validation** (current approach) because the bootcamp explicitly showed agents might send malformed/unexpected output during RL training. Rejecting at the Pydantic level would crash the episode; runtime degradation-to-wait with a conflict message gives a learning signal instead.

Let me build the comprehensive plan: 

Created memory file [](file:///c%3A/Users/rguhaneogi/AppData/Roaming/Code/User/workspaceStorage/2f1d98ed382d762b16cb893c28af5c19/GitHub.copilot-chat/memory-tool/memories/OGE4ZGU4OGYtMzIzZS00MDk5LWI1MzMtYTgyNGI2ZDIzYzVm/plan.md)

Deleted memory [](file:///c%3A/Users/rguhaneogi/AppData/Roaming/Code/User/workspaceStorage/2f1d98ed382d762b16cb893c28af5c19/GitHub.copilot-chat/memory-tool/memories/OGE4ZGU4OGYtMzIzZS00MDk5LWI1MzMtYTgyNGI2ZDIzYzVm/plan.md)

Created memory file [](file:///c%3A/Users/rguhaneogi/AppData/Roaming/Code/User/workspaceStorage/2f1d98ed382d762b16cb893c28af5c19/GitHub.copilot-chat/memory-tool/memories/OGE4ZGU4OGYtMzIzZS00MDk5LWI1MzMtYTgyNGI2ZDIzYzVm/plan.md)

## Plan: Full Codebase Improvement Sweep

21 targeted fixes across 5 phases. No architectural changes — all improvements to existing files.

---

### Phase 1: Reward Signal Quality (RL-Critical)
*These directly affect whether an RL agent can learn from this environment.*

1. **Fix "wait" reward bias** — reward.py: `wait` when soil moist gives +0.05 free reward. Agent learns to do nothing. Change to 0.0 (neutral).
2. **Fix harvest reward asymmetry** — reward.py: Late harvest (DVS>2.05) currently gets +0.10 reward (!) while early gets -0.15. Late should penalize too.
3. **Tighten fertilization timing** — reward.py: Fertilizing between optimal windows gets +0.05 (free reward). Change to -0.03 to prevent "spray more = more reward."
4. **Remove duplicate step_reward computation** — environment.py: Reward computed once before harvest check, then again after. First one is discarded — confusing dead code.

### Phase 2: Agronomic Accuracy
*Strengthens "Real-world utility" (30% of rubric).*

5. **Fix LAI senescence rate** — crop_sim.py: Rate 0.03 is 3-5x too slow. Real wheat fully senesces in 7-10 days post-anthesis, not 50+.
6. **Fix Punjab rainfall** — scenarios.py: Generates ~0.5cm/season. Real Punjab winter wheat gets 5-8cm. Rain frequency 0.08→0.12, mean per event 0.3→0.6cm.
7. **Make N-depletion phenology-aware** — crop_sim.py: Currently flat 0.0008/day loss. Should be slower pre-anthesis (0.0003), faster post-anthesis (0.0015).
8. **Add heat stress for Punjab** — crop_sim.py: If tmax > 35°C during flowering (DVS 0.8-1.2), growth reduced 50%. Makes Task 3 genuinely harder and more realistic.

### Phase 3: Inference Robustness (Judging Survival)
*Prevents crashes during hackathon evaluation.*

9. **Use httpx context manager** — inference.py: Currently `http.close()` can be skipped if exception occurs.
10. **Add max-step guard** — inference.py: `while not result.done` could run forever. Add `for step_num in range(200):` with break.
11. **Add try/except around LLM call** — inference.py: Network timeouts, rate limits, 500 errors crash the script.
12. **Replace assert with if-check** — inference.py: `assert llm_client is not None` gets optimized away with `python -O`.
13. **Improve JSON parsing** — inference.py: Fragile `split/rsplit` for markdown code blocks. Use regex instead.
14. **Add LLM usage tracking** — inference.py: Count LLM successes vs greedy fallbacks. Print summary so judges can verify LLM integration works.

### Phase 4: Observation Enrichment
*Help LLM agents reason better — "Environment design" (20% rubric).*

15. **Add `water_stress` to soil_status** — environment.py: Currently only binary `water_deficit` flag. Add continuous 0-1 value from `sim._water_stress()`.
16. **Add `n_availability` to crop_status** — environment.py: Agent can't see nitrogen status. Add `sim.n_factor` to observation.
17. **Add scoring formula to task instructions** — tasks.py: Tell agents: "35% yield + 20% water eff + 18% cost eff + 15% fert timing + 12% harvest timing." Bootcamp said agents need to understand the reward landscape.

### Phase 5: Code Quality Polish

18. **Remove unreachable grader code** — grader.py: `elif harvested: yield_score = 0.5` can never execute (target_yield is always >0).
19. **Remove dead reward budget check** — reward.py: Already enforced by environment.py before reward is called.
20. **Add HEALTHCHECK to Dockerfile** — Dockerfile: Container silently crashes if import fails.
21. **Extract greedy magic numbers** — inference.py: Named constants for thresholds (0.22, 0.18, 2.5, etc.).

---

**Relevant files:** reward.py, environment.py, crop_sim.py, scenarios.py, grader.py, tasks.py, inference.py, Dockerfile

**Verification:**
1. `python -m pytest tests/ -v` — all tests pass
2. Run inference.py — confirm Task1 ≥ Task2 ≥ Task3 ordering maintained
3. Verify no episode exceeds 200 steps
4. Confirm LLM fallback logging prints to stderr

**Decisions:**
- Keep runtime action validation (not Pydantic Literal types) — LLMs need graceful degradation, not crashes
- Keep unified grading formula — bootcamp confirms environment-driven difficulty is correct
- Skip multi-year/crop rotation, frost modeling — too much scope for deadline
