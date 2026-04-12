"""Ablation study: measure score impact of removing each environment mechanic.

Runs modified policies against the environment to show that each design
decision (leaching, slow-release, inspections, irrigation, fertilization)
contributes meaningfully to score differentiation.

Usage:
    python -m scripts.ablation_study
"""
from __future__ import annotations

from statistics import mean

from agent.policy import greedy_action, oracle_action
from models import CropAction
from server.environment import CropEnvironment

SEEDS = list(range(42, 52))  # 10 seeds for statistical robustness
TASKS = [1, 2, 3]


def run_episode(task_id: int, seed: int, policy_fn) -> float:
    env = CropEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    state: dict = {}
    steps = 0
    while not obs.done:
        action_dict = policy_fn(env, obs, state)
        obs = env.step(CropAction(**action_dict))
        steps += 1
        if steps > 80:
            break
    return float(obs.reward or 0.0)


# ── Policy functions ──────────────────────────────────────────────────

def oracle_policy(env, obs, state):
    return env.oracle_reference_action()


def greedy_policy(env, obs, state):
    return greedy_action(obs, state)


def no_fertilize_policy(env, obs, state):
    action = env.oracle_reference_action()
    if action["action_type"] in ("fertilize", "fertilize_slow"):
        return {"action_type": "wait", "amount": 0.0}
    return action


def no_irrigation_policy(env, obs, state):
    action = env.oracle_reference_action()
    if action["action_type"] == "irrigate":
        return {"action_type": "wait", "amount": 0.0}
    return action


def no_slow_release_policy(env, obs, state):
    action = env.oracle_reference_action()
    if action["action_type"] == "fertilize_slow":
        action = {**action, "action_type": "fertilize"}
    return action


def wait_only_policy(env, obs, state):
    return {"action_type": "wait", "amount": 0.0}


# ── Main ──────────────────────────────────────────────────────────────

POLICIES = {
    "Oracle (full)": oracle_policy,
    "Greedy heuristic": greedy_policy,
    "Oracle − fertilization": no_fertilize_policy,
    "Oracle − irrigation": no_irrigation_policy,
    "Oracle − slow-release": no_slow_release_policy,
    "Wait-only (passive)": wait_only_policy,
}


def main():
    results: dict[str, dict[int, float]] = {}

    for name, policy_fn in POLICIES.items():
        results[name] = {}
        for task_id in TASKS:
            scores = [run_episode(task_id, seed, policy_fn) for seed in SEEDS]
            results[name][task_id] = round(mean(scores), 3)

    # Print markdown table
    print("| Policy Variant | Task 1 | Task 2 | Task 3 | Avg | Δ vs Oracle |")
    print("|----------------|--------|--------|--------|-----|-------------|")

    oracle_avg = mean(results["Oracle (full)"][t] for t in TASKS)
    for name in POLICIES:
        t1 = results[name][1]
        t2 = results[name][2]
        t3 = results[name][3]
        avg = round(mean([t1, t2, t3]), 3)
        delta = round(avg - oracle_avg, 3)
        delta_str = f"{delta:+.3f}" if name != "Oracle (full)" else "—"
        print(f"| {name} | {t1:.3f} | {t2:.3f} | {t3:.3f} | {avg:.3f} | {delta_str} |")


if __name__ == "__main__":
    main()
