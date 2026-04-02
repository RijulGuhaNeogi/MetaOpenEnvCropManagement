I’m consolidating the prior RL roadmap into a tighter execution plan tied to the actual code entry points, then I’ll persist the final phased plan in session memory.

Searched for regex `def compute_step_reward|def _build_observation|class CropObservation|class CropAction|def step\(|def grade_episode|def generate_scenario` (`**/*.{py,md}`), 11 results

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/models.py)

Read memory [](file:///c%3A/Users/rguhaneogi/AppData/Roaming/Code/User/workspaceStorage/2f1d98ed382d762b16cb893c28af5c19/GitHub.copilot-chat/memory-tool/memories/ODUxNTM2YjEtMWFiYi00YzdhLWE1NTQtNjZkZGQyMjQ2ZGJm/plan.md)

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/server/reward.py)

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/server/environment.py#1-1), lines 1 to 260

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/server/scenarios.py#1-1), lines 1 to 260

Read [](file:///c%3A/Users/rguhaneogi/source/repos/MetaHackathonPrep/tests/test_smoke.py)

I’ve mapped the plan to the concrete code seams: reward work centers on compute_step_reward and step, observation work centers on CropObservation and _build_observation, and testing can extend the current greedy smoke harness. Next I’m replacing the draft with a final phased plan that is execution-ready.

Deleted memory [](file:///c%3A/Users/rguhaneogi/AppData/Roaming/Code/User/workspaceStorage/2f1d98ed382d762b16cb893c28af5c19/GitHub.copilot-chat/memory-tool/memories/ODUxNTM2YjEtMWFiYi00YzdhLWE1NTQtNjZkZGQyMjQ2ZGJm/plan.md)

Created memory file [](file:///c%3A/Users/rguhaneogi/AppData/Roaming/Code/User/workspaceStorage/2f1d98ed382d762b16cb893c28af5c19/GitHub.copilot-chat/memory-tool/memories/ODUxNTM2YjEtMWFiYi00YzdhLWE1NTQtNjZkZGQyMjQ2ZGJm/plan.md)

## Plan: RL Learnability Upgrade

Keep the current custom simulator and improve the RL-facing layer around it. The crop model in crop_sim.py is already realistic enough for the hackathon: it has phenology, biomass growth, water balance, nitrogen limitation, LAI dynamics, and heat stress. The highest-ROI work is to make rewards consequence-aware, expose more control-relevant observation features, add RL-specific evaluation, and only then add training-only extras. This improves learnability without risking a late PCSE migration or changing benchmark comparability.

**Phases**
1. Phase 0: Freeze scope and baseline behavior.
   Record baseline metrics before any RL-facing changes.
   Use test_smoke.py as the baseline harness across all three tasks and multiple seeds.
   Capture final score, step count, total water, total nitrogen, total cost, and harvest DVS from environment.py and grader.py.
   Treat grader.py as frozen during Phases 1-3 so the public benchmark stays comparable.
   Explicitly exclude PCSE migration, new crops, and public API redesign from this pass.

2. Phase 1: Redesign dense rewards around action quality and outcome.
   Update compute_step_reward in reward.py so irrigation reward depends on both pre-action moisture deficit and dose size rather than only simple soil-moisture bands.
   Use a target moisture band, roughly 0.28 to 0.32 soil moisture, and reward closing the deficit while penalizing overshoot and spending when forecast rain already reduces the need.
   Update fertilizer shaping in reward.py so reward depends on DVS window, amount applied, and cumulative nitrogen already used, with diminishing returns inside good windows and stronger penalties for obviously excessive late-season nitrogen.
   Keep harvest timing logic largely intact because it is already directionally correct and not the main RL bottleneck.
   In CropEnvironment.step in environment.py, change the reward flow so part of the reward is computed after `sim.advance` rather than fully before transition.
   Capture pre-transition values for soil moisture, water stress proxy, nitrogen availability proxy, DVS, and budget remaining before calling `advance`.
   After the transition, compute delta-based shaping from post-transition values, prioritizing reduction in water stress, improved nitrogen status, and penalties for oversaturation or waste.
   Blend the reward into two interpretable parts: an intent component from pre-action timing and a consequence component from post-transition deltas.
   Preserve the final terminal reward path through `compute_trajectory_reward` and `grade_episode` so the terminal benchmark definition does not change.

3. Phase 2: Enrich observations with policy-native control features.
   Extend CropObservation in models.py with derived control features rather than only raw state buckets.
   Add `moisture_gap_to_target`, `forecast_rain_3d`, `forecast_rain_7d`, `days_since_last_irrigation`, `days_since_last_fertilization`, `fertilizer_events_count`, `cumulative_n_applied`, `budget_remaining_ratio`, and `dvs_distance_to_next_fertilizer_window`.
   Compute these features in CropEnvironment._build_observation in environment.py using existing simulator state plus short summaries from `weather_forecast`.
   Add minimal internal bookkeeping to CropState in models.py and to the environment in environment.py for last irrigation day, last fertilization day, and fertilizer event count.
   Keep the existing nested observation structure intact so current LLM-facing clients are not broken. Only add fields.
   This phase depends on Phase 1 reward semantics being clear, but much of the observation work can run in parallel once the fields are agreed.

4. Phase 3: Add RL-specific tests and probe evaluation.
   Expand test_smoke.py or add a focused reward test file under tests to validate monotonicity and anti-hacking properties.
   Add tests asserting that, from the same starting state, moderate irrigation in dry soil does not score worse than larger wasteful irrigation.
   Add tests asserting that nitrogen in the correct DVS window with a sensible dose scores better than an excessive dose, and that late-season fertilizer remains negative or clearly inferior.
   Add tests asserting that post-transition relief of severe stress is rewarded more than acting when stress is mild.
   Add tests asserting that repeated `wait` or repeated tiny near-no-op actions cannot accumulate an exploitably positive dense reward.
   Add internal probe scenarios in scenarios.py for over-irrigation, late-fertilizer temptation, budget starvation, harvest hesitation, and drought rescue.
   Keep these probe scenarios internal to evaluation and training diagnostics. Do not expose them as new public tasks.
   This phase depends on Phase 1 and should complete before any optional training-only features.

5. Phase 4: Add training-only reward decomposition and optional action adapters.
   Surface decomposed reward components in observation metadata from environment.py, for example `intent_reward`, `delta_reward`, and terminal grade breakdown.
   Keep the scalar `Observation.reward` path unchanged so OpenEnv behavior and benchmark compatibility remain intact.
   If needed for RL experimentation, add a thin training adapter outside the public schema that maps discrete buckets to the current CropAction format defined in models.py.
   Recommended bucket mapping is `wait`, `harvest`, `irrigate_small`, `irrigate_medium`, `irrigate_large`, `fertilize_small`, `fertilize_medium`, `fertilize_large`, converted internally to calibrated amounts.
   Treat this adapter as a training convenience only. Do not replace the public `action_type` plus `amount` API used by the hackathon client.
   If time remains after the above, extend inference.py to generate labeled trajectories from the heuristic policy and noisy perturbations for offline RL or imitation-learning bootstrap.
   This phase is valuable but non-blocking for the hackathon deliverable.

**Execution order**
1. Phase 0 must finish first because it establishes the baseline and locks benchmark invariants.
2. Phase 1 is the critical blocking phase because reward semantics drive the rest of the RL design.
3. Phase 2 can begin in parallel with the latter half of Phase 1 once the derived reward signals and needed state summaries are agreed.
4. Phase 3 depends on Phase 1 and should start as soon as the first reward redesign is stable enough to test.
5. Phase 4 is optional and should only start after Phase 3 confirms the reward is monotonic, non-exploitable, and baseline-compatible.

**Relevant files**
- reward.py — update `compute_step_reward` to make irrigation and fertilizer shaping amount-sensitive and partially outcome-aware
- environment.py — update CropEnvironment.step reward flow and CropEnvironment._build_observation derived features
- models.py — extend CropObservation and CropState with derived RL-facing fields and light history tracking
- scenarios.py — add probe scenarios and later, if desired, internal curriculum hooks
- grader.py — preserve as-is during the main pass; reuse its breakdown for terminal evaluation and metadata
- test_smoke.py — reuse as the baseline harness and extend with RL-specific assertions or split into a dedicated reward-focused test file
- inference.py — optional later phase for trajectory generation and RL data bootstrapping
- RLimprovementPlan.md — use as the design rationale source, not as an implementation artifact

**Verification**
1. Re-run the existing test suite in tests after each phase; all current smoke and determinism checks must stay green.
2. Re-run the greedy baseline over multiple seeds and verify final grader scores remain in a similar range because grader.py is unchanged.
3. Inspect per-step reward traces before and after Phase 1 to confirm the new shaping differentiates moderate versus wasteful doses.
4. Confirm that the same seed and same action sequence remain deterministic after adding any new state bookkeeping.
5. Validate that new observation fields are populated consistently and do not break current clients expecting the existing nested dict structure.
6. Run internal probe scenarios and verify expected ordering, for example drought rescue beats unnecessary irrigation, and sensible fertilizer beats excessive fertilizer.
7. If Phase 4 is executed, confirm the metadata reward breakdown is present without altering the public scalar reward contract.
8. Verify Docker build and end-to-end inference output format remain unchanged.

**Decisions**
- Include: reward shaping, post-transition consequence reward, observation enrichment, probe scenarios, and RL-specific tests
- Include later if time permits: decomposed training reward, discrete training adapter, offline trajectory generation
- Exclude: PCSE migration, new crops, public task redesign, grader weight changes, and public action schema changes
- Preserve: deterministic seeded behavior, OpenEnv contract, terminal scoring semantics, and public client compatibility

The final phased plan is saved in session memory at `/memories/session/plan.md`.

1. If you want, I can now condense this into a strict 3-day execution order.
2. If you want to hand this to an implementation agent, Phase 1 is the correct starting point.