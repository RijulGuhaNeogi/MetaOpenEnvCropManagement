#!/usr/bin/env python3
"""
Competition inference script — Precision Agriculture Crop Management.

MANDATORY ENV VARS:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    API_KEY        Authentication token (evaluator injects API_KEY; HF_TOKEN as fallback).

STDOUT FORMAT:
    [START] task=<task_name> env=crop_management model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv(override=False)  # never override evaluator-injected env vars

from openai import OpenAI

from client import CropEnvClient
from models import CropAction

# Import reusable logic from the agent module (no duplication)
from agent.inference import (
    call_llm,
    oracle_action,
    _record_transition,
    _write_trajectory_jsonl,
    llm_client,
    MAX_CLIENT_STEPS,
)

# ---------------------------------------------------------------------------
# Configuration (competition-mandated env vars)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
TASK_ID = os.getenv("TASK_ID", "")  # empty = run all 3
SEED = int(os.getenv("SEED", "42"))
BENCHMARK = "crop_management"
SUCCESS_THRESHOLD = 0.1
TRAJECTORY_OUTPUT = os.getenv("TRAJECTORY_OUTPUT", "")


# ---------------------------------------------------------------------------
# Strict stdout helpers — match sample format exactly
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Format action string for [STEP] line
# ---------------------------------------------------------------------------
def _format_action(action: CropAction) -> str:
    if action.action_type in ("irrigate", "fertilize") and action.amount > 0:
        return f"{action.action_type}({action.amount:.1f})"
    return action.action_type


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------
def run_task(task_id: int, task_name: str) -> float:
    """Run a single task and return the rubric score."""
    model_name = MODEL_NAME or "oracle-baseline"
    log_start(task=task_name, env=BENCHMARK, model=model_name)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    trajectory_records: list[dict] = []

    try:
        sync_client = CropEnvClient(base_url=ENV_URL).sync()
        with sync_client:
            result = sync_client.reset(seed=SEED, task_id=task_id)
            obs = result.observation
            oracle_state: dict = {}

            while not result.done and steps_taken < MAX_CLIENT_STEPS:
                previous_obs = obs

                # Choose action: LLM with oracle fallback
                if llm_client is not None:
                    action_dict = call_llm(obs)
                    policy_name = "llm"
                    if not action_dict or "action_type" not in action_dict:
                        action_dict = oracle_action(obs, oracle_state)
                        policy_name = "oracle_fallback"
                else:
                    action_dict = oracle_action(obs, oracle_state)
                    policy_name = "oracle"

                if "amount" not in action_dict:
                    action_dict["amount"] = 0.0

                action = CropAction(**action_dict)
                result = sync_client.step(action)
                obs = result.observation

                steps_taken += 1
                reward = result.reward if result.reward is not None else 0.0
                rewards.append(reward)

                # Emit [STEP] line
                error = getattr(obs, "last_action_error", None)
                if error is None and obs.conflicts:
                    error = "; ".join(obs.conflicts)
                log_step(
                    step=steps_taken,
                    action=_format_action(action),
                    reward=reward,
                    done=result.done,
                    error=error if error else None,
                )

                if TRAJECTORY_OUTPUT:
                    _record_transition(
                        trajectory_records,
                        task_id=task_id,
                        step_num=steps_taken,
                        policy_name=policy_name,
                        observation=previous_obs,
                        action=action,
                        result=result,
                    )

        # Score = rubric_reward from terminal observation (trajectory-level grader)
        rubric = getattr(obs, "rubric_reward", None)
        if rubric is not None:
            score = float(rubric)
        else:
            # Fallback: last dense reward (shouldn't happen)
            score = rewards[-1] if rewards else 0.0
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} error: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    if TRAJECTORY_OUTPUT and trajectory_records:
        suffix = f"_task{task_id}" if TASK_ID == "" else ""
        _write_trajectory_jsonl(f"{TRAJECTORY_OUTPUT}{suffix}.jsonl", trajectory_records)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    import httpx

    # Health check
    with httpx.Client(base_url=ENV_URL, timeout=30.0) as http:
        health = http.get("/health")
        health.raise_for_status()

        tasks_resp = http.get("/tasks")
        tasks_resp.raise_for_status()
        available = {t["id"]: t for t in tasks_resp.json()["tasks"]}

    # Determine which tasks to run
    if TASK_ID:
        task_ids = [int(TASK_ID)]
    else:
        task_ids = sorted(available.keys())

    scores: dict[int, float] = {}
    for tid in task_ids:
        if tid not in available:
            print(f"[DEBUG] Task {tid} not available, skipping", file=sys.stderr, flush=True)
            continue
        task_name = available[tid]["name"]
        scores[tid] = run_task(tid, task_name)

    # Print summary to stderr (not stdout — stdout is reserved for [START]/[STEP]/[END])
    if len(scores) > 1:
        overall = sum(scores.values()) / len(scores)
        print(f"[DEBUG] Overall: {overall:.4f}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
