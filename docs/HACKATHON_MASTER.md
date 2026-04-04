# Meta PyTorch OpenEnv Hackathon Master Reference

## Purpose

This document is the working source of truth for the hackathon. It combines the official Round 1 requirements, the preparation notes, the study-material links, and the bootcamp transcript guidance into one place.

Use it for three things:

1. To decide whether a feature or design choice helps or hurts our submission.
2. To evaluate whether the current project is likely to clear the automated gate.
3. To judge whether the environment is actually competitive on the rubric, not just technically valid.

## Primary Sources

This guide is synthesized from the following repository sources:

1. [docs/HackathonInformationVideoTranscriptsAndDumps.txt](docs/HackathonInformationVideoTranscriptsAndDumps.txt)
2. [Preparation](Preparation)
3. [ProblemDetails](ProblemDetails)
4. [studymaterialLinks](studymaterialLinks)
5. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## One-Sentence Summary

Build a deployable OpenEnv environment for a real-world task, with typed models, meaningful rewards, at least three graded tasks of increasing difficulty, and a reproducible baseline inference script that runs within evaluator constraints.

## Dates And Timeline

1. Registration: March 14 to April 3, 2026
2. Preparation phase: through March 25, 2026
3. Round 1 build and submit: March 25 to April 7, 2026
4. Submission deadline: April 7, 2026 at 11:59 PM IST
5. Results: April 10, 2026
6. Grand finale: April 25 to 26, 2026

## What The Judges Actually Care About

The public scoring rubric says the following weights apply:

1. Real-world utility: 30%
2. Task and grader quality: 25%
3. Environment design: 20%
4. Code quality and spec compliance: 15%
5. Creativity and novelty: 10%

The practical reading of that rubric is:

1. Utility, graders, and environment design make up 75% of the score.
2. Creativity matters, but it is not the main lever.
3. Passing validation is necessary but not sufficient.
4. A technically correct but shallow toy environment will still score badly.

## Core Objective

The environment must simulate a real-world task, not a game or toy. The transcript repeatedly frames a good environment as one an RL or post-training engineer could plausibly plug into a real training run.

Strong example domains mentioned across the material:

1. Email triage
2. Code review workflows
3. Data cleaning pipelines
4. Scheduling and planning
5. Customer support routing
6. Content moderation

The common property is not the industry label. It is that the task is something people genuinely do, where an agent must make decisions over a trajectory and can be judged with meaningful reward and grading logic.

## Non-Negotiable Submission Requirements

The submission must include the following:

1. A public GitHub repository with the environment code
2. A deployed Hugging Face Space that runs the environment
3. A working Dockerfile
4. A valid openenv.yaml file
5. A root-level inference.py file
6. A README with setup, usage, task descriptions, schema descriptions, and baseline scores
7. At least three tasks with graders

The environment must also satisfy the following behavior requirements:

1. Implement typed Pydantic models for Action, Observation, and State
2. Implement step, reset, and state correctly
3. Produce deterministic grader outputs between 0.0 and 1.0
4. Provide a meaningful reward signal through the trajectory
5. Build and run cleanly in Docker
6. Deploy and respond correctly on Hugging Face Spaces

## Phase 1 Automated Gate

This is the pass or fail barrier. If this fails, the submission is effectively out.

The gate checks for:

1. Hugging Face Space deploys and responds
2. OpenEnv spec compliance
3. Docker build succeeds
4. Baseline inference reproduces and completes
5. Three or more tasks exist with graders
6. Grader scores stay in the 0.0 to 1.0 range

Operational conclusion:

1. First optimize for reliable deployment and reproducible execution.
2. Then optimize for score quality.

## Disqualification And High-Risk Failure Modes

The materials explicitly mention or strongly imply the following as fatal or near-fatal:

1. Environment does not deploy or respond
2. No inference.py script
3. Graders always return the same score
4. Plagiarized or trivially modified environment
5. Toy or game-like problem with weak real-world value
6. Sparse terminal-only rewards with no useful trajectory signal
7. Hardcoded environment assumptions that break evaluator execution
8. Broken Dockerfile or runtime-only deployment issues

## Required OpenEnv Shape

The expected OpenEnv structure includes:

1. Pydantic models for Action, Observation, and State
2. Server-side environment logic implementing reset, step, and state
3. A FastAPI app exposing the environment
4. A client wrapper for interacting with the environment
5. Metadata through openenv.yaml

Expected standard endpoints include:

1. /health
2. /reset
3. /step
4. /state
5. /ws

Additional endpoints mentioned in the hackathon material:

1. /tasks
2. /grader
3. /baseline

These extra endpoints are useful because they make the environment easier to inspect, evaluate, and demo, even if only some are strictly enforced.

## Important OpenEnv Implementation Note

For this repository specifically, there is an important implementation nuance already captured in repo memory: HTTP endpoints are stateless in the underlying OpenEnv setup, while multi-step episodes should be handled through WebSocket or EnvClient session semantics.

That means:

1. The environment must still answer health and reset correctly over HTTP.
2. Any multi-step evaluation logic must not assume persistent HTTP server-side episode state unless the implementation explicitly handles it.
3. Session design errors can easily break correctness even if the API surface looks fine.

## Task Design Requirements

At least three tasks are required, and they should progress from easy to medium to hard.

The materials strongly imply the following structure:

1. Easy task: straightforward, narrow, should be solvable by a decent baseline
2. Medium task: requires a clearer chain of decisions or trade-offs
3. Hard task: should challenge frontier-class models with planning, edge cases, or competing objectives

High-quality tasks have these properties:

1. Clear objective
2. Deterministic grading
3. Meaningful difference in difficulty
4. Realistic workflow context
5. Opportunity for partial progress reward

Weak tasks usually fail because:

1. The hard task is only a larger easy task
2. The grader is binary and shallow
3. The agent is not forced to reason over time
4. The environment has no meaningful trade-offs

## Graders: What Good Looks Like

Graders are not just technical validation. They are a major part of the submission score.

Good graders:

1. Return scores normalized to 0.0 through 1.0
2. Are deterministic and reproducible
3. Reflect the actual task objective
4. Distinguish partial success from complete success
5. Do not collapse all reasonable trajectories into the same score

Bad graders:

1. Return mostly identical values across runs and tasks
2. Reward trivial exploits
3. Ignore quality dimensions that matter in the real workflow
4. Only score the final state and ignore the trajectory entirely

## Reward Design Requirements

The transcript guidance is especially strong here. Reward shaping is central.

The reward function should:

1. Provide signal across the full trajectory
2. Reward partial progress
3. Penalize undesirable behavior such as loops, wasted actions, or destructive choices
4. Encourage efficiency where appropriate
5. Stay aligned with the final grader rather than fighting it

Reward-design principles implied by the transcript:

1. If the agent can never realistically reach reward, the environment is poor for RL
2. Sparse end-only rewards are weak unless the task is extremely well structured
3. Milestone-based or delta-based rewards are stronger than pure binary success signals
4. Reward hacking must be considered during design, not after deployment

## Inference Script Requirements

The project must include a root-level inference.py script.

Constraints gathered from the materials:

1. Use the OpenAI client style for LLM calls
2. Read credentials and model configuration from environment variables
3. Support variables such as API_BASE_URL, MODEL_NAME, and HF_TOKEN
4. Some materials also mention OPENAI_API_KEY or API_KEY fallback usage patterns
5. Finish within 20 minutes
6. Be runnable on a machine with 2 vCPU and 8 GB RAM
7. Produce reproducible baseline scores

The inference script is not there to impress with a huge score. Its main job is to be reliable, compatible with evaluator infrastructure, and representative.

## Deployment Constraints

The deployment target is a Hugging Face Space.

This implies:

1. The Dockerfile must work cleanly in the target environment
2. Cold-start behavior matters
3. The service must become healthy and respond to checks
4. Any missing environment variables or hidden local assumptions can sink the submission

Practical rule:

1. If the environment only works on the author machine, it does not count as working.

## What The Bootcamp Transcript Adds Beyond The Formal Spec

The transcript provides important interpretation that is not just restating the rules.

### 1. Real-world usefulness beats novelty for novelty's sake

The environment should feel like something that would belong in a future agent product or RL training suite.

### 2. Long-running tasks are valuable

The transcript repeatedly favors tasks where agents can make mistakes, adapt, recover, and continue.

### 3. Curriculum and difficulty scaling matter

The best environments make it possible to train or evaluate at increasing difficulty rather than only one flat challenge level.

### 4. Reward hacking is a real concern

Models will exploit weak graders or reward shortcuts. Good design anticipates this.

### 5. A realistic achievable task is better than an impossible one

If the environment is so hard that the baseline can never access useful reward, the RL setup is poor even if the idea is interesting.

## Hidden Evaluator Heuristics

Based on the transcript and briefing, evaluators are likely to ask these implicit questions:

1. Is this environment something the RL community would actually care about?
2. Do the tasks create useful training or benchmarking signal?
3. Does the hard task expose reasoning or planning weaknesses in strong models?
4. Is the reward function thoughtful, not just present?
5. Can the system be reproduced and run without hand-holding?
6. Is the design robust against exploitative or degenerate agent behavior?

## Study Material Interpretation

The provided study path suggests how the organizers expect teams to approach the work:

1. Module 1 explains why environments matter
2. Module 2 covers how to use existing environments
3. Module 3 covers deployment and scaling
4. Module 4 is the most important for Round 1 because it focuses on building your own environment

Practical reading:

1. Modules 1 to 3 provide context and deployment literacy
2. Module 4 is the implementation-critical module for the actual submission

## Strategy Implications For This Repository

This repository already appears to be centered on a crop-management environment. That can be a strong fit for the rubric if the implementation is framed correctly.

Potential strengths of the current direction:

1. Agriculture is a real-world domain
2. The environment can support planning, trade-offs, and delayed outcomes
3. Difficulty can be increased through climate, resource, and timing complexity
4. The workflow is meaningful enough to justify RL evaluation

Possible risks for the current direction:

1. The hard task may not yet be hard enough for frontier models
2. Rewards may be more simulator-centric than agent-training-centric
3. The grader may need clearer normalization and stronger differentiation across tasks
4. The narrative in the README may need to better justify immediate agent-training value

## Project Evaluation Checklist

Use this checklist before any major submission milestone.

### A. Automated Gate Checklist

1. Does the Hugging Face Space deploy successfully?
2. Does the service become healthy without manual intervention?
3. Does openenv.yaml exist and match the environment?
4. Does the Dockerfile build from a clean checkout?
5. Does inference.py exist at the repository root?
6. Does inference.py complete successfully within the runtime budget?
7. Are there at least three tasks?
8. Do all graders return normalized scores from 0.0 to 1.0?
9. Do reset, step, and state behave correctly?

### B. Real-World Utility Checklist

1. Can we explain in one sentence what real job this environment simulates?
2. Would an RL engineer or agent-evaluation engineer plausibly use this environment?
3. Does the environment model actual constraints, trade-offs, and failure modes from the domain?
4. Is the task more than a disguised toy problem?

### C. Task And Grader Quality Checklist

1. Are easy, medium, and hard tasks genuinely different?
2. Does the hard task require planning or edge-case handling?
3. Can the grader distinguish partial progress from full success?
4. Are grader outcomes deterministic for the same setup?
5. Are there clear ways for agents to do poorly, adequately, and excellently?

### D. Reward Design Checklist

1. Does the agent receive reward before the episode ends?
2. Are progress milestones rewarded?
3. Are loops or wasteful behaviors penalized?
4. Is the reward aligned with the final grader?
5. Have we thought through likely exploit paths?

### E. Inference And Reproducibility Checklist

1. Is the script fully environment-variable driven?
2. Is the base URL configurable?
3. Are model calls using the expected OpenAI client pattern?
4. Can evaluators reproduce the same outcome class without hidden setup?
5. Does the script avoid local-only assumptions?

### F. Documentation Checklist

1. Does the README explain why the environment matters?
2. Are action, observation, and task definitions documented clearly?
3. Are setup and run instructions complete?
4. Are baseline scores shown?
5. Is the difficulty progression explained?

## Strong Submission Principles

If there is disagreement about product or implementation decisions, these rules should break ties:

1. Prefer real-world fidelity over gimmicks.
2. Prefer reliable deployment over clever but fragile engineering.
3. Prefer better graders over cosmetic UI or novelty.
4. Prefer reward functions that teach something over rewards that merely exist.
5. Prefer a hard task with meaningful failure modes over an artificially bigger easy task.

## Recommended Next Use Of This Document

The next high-value step is to score the current repository against this checklist and produce a gap report before submission.

That review should focus on:

1. Whether the current crop-management tasks satisfy true easy, medium, and hard progression
2. Whether rewards are trajectory-shaped enough for the hackathon expectations
3. Whether inference.py and deployment details match the latest requirements exactly
4. Whether the README and openenv.yaml argue the real-world utility strongly enough
_____________________________________________________________________
IMPORTANT!!!!

Disqualification Criteria

Environment does not deploy or respond

Plagiarized or trivially modified existing environments

Graders that always return the same score

No baseline inference script
____________________________________________________________________

IMPORTANT!!!!!!!

Pre-Submission Checklist  — all must pass or you're disqualified

HF Space deploys

Automated ping to the Space URL — must return 200 and respond to reset()

OpenEnv spec compliance

Validate openenv.yaml, typed models, step()/reset()/state() endpoints

Dockerfile builds

Automated docker build on the submitted repo

Baseline reproduces

Run the submitted inference script — must complete without error and produce scores

3+ tasks with graders

Enumerate tasks, run each grader, verify scores/reward in 0.0–1.0 range

Mandatory Additional Instructions

Before submitting, ensure the following variables are defined in your environment configuration:

API_BASE_URL   The API endpoint for the LLM.

MODEL_NAME     The model identifier to use for inference.

HF_TOKEN       Your Hugging Face / API key.

The inference script must be named `inference.py` and placed in the root directory of the project

Participants must use OpenAI Client for all LLM calls using above variables

Participants must emit structured stdout logs strictly following the [START], [STEP], and [END] format defined in the sample inference.py provided below. Any deviation in field names, ordering, or formatting will result in incorrect evaluation scoring. Refer to the Sample Inference Script for the complete format specification and examples.

Infra Restrictions

Runtime of inference script should be less than 20min 

Make sure your env and inference can run on a machine with vcpu=2, memory=8gb

Validator

Run the pre-submission validation script before submitting
_______________________________________________________________

Sample inference script:

"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from my_env_v4 import MyEnvV4Action, MyEnvV4Env
IMAGE_NAME = os.getenv("IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "echo")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

# Max possible reward: each token contributes 0.1, across all steps
_MAX_REWARD_PER_STEP = MAX_TOKENS * 0.1
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are interacting with a simple echo environment.
    Each turn you must send a message. The environment will echo it back.
    Reward is proportional to message length: reward = len(message) * 0.1
    Your goal is to maximize total reward by sending meaningful, substantive messages.
    Reply with exactly one message string — no quotes, no prefixes, just the message text.
    """
).strip()


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
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Last echoed message: {last_echoed!r}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Send your next message.
        """
    ).strip()


def get_model_message(client: OpenAI, step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, last_echoed, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "hello"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "hello"


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await MyEnvV4Env.from_docker_image(IMAGE_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset() # OpenENV.reset()
        last_echoed = result.observation.echoed_message
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            message = get_model_message(client, step, last_echoed, last_reward, history)

            result = await env.step(MyEnvV4Action(message=message))
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_echoed = obs.echoed_message
            last_reward = reward

            log_step(step=step, action=message, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {message!r} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
_____________________________________________________________________

IMPORTANT!!!!!!!

Pre validation script:

#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator
#
# Checks that your HF Space is live, Docker image builds, and openenv validate passes.
#
# Prerequisites:
#   - Docker:       https://docs.docker.com/get-docker/
#   - openenv-core: pip install openenv-core
#   - curl (usually pre-installed)
#
# Run:
#   curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/validate-submission.sh | bash -s -- <ping_url> [repo_dir]
#
#   Or download and run locally:
#     chmod +x validate-submission.sh
#     ./validate-submission.sh <ping_url> [repo_dir]
#
# Arguments:
#   ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)
#   repo_dir   Path to your repo (default: current directory)
#
# Examples:
#   ./validate-submission.sh https://my-team.hf.space
#   ./validate-submission.sh https://my-team.hf.space ./my-repo
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  printf "\n"
  printf "  ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)\n"
  printf "  repo_dir   Path to your repo (default: current directory)\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi
PING_URL="${PING_URL%/}"
export PING_URL
PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf "\n"

log "${BOLD}Step 1/3: Pinging HF Space${NC} ($PING_URL/reset) ..."

CURL_OUTPUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")
HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>"$CURL_OUTPUT" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /reset"
elif [ "$HTTP_CODE" = "000" ]; then
  fail "HF Space not reachable (connection failed or timed out)"
  hint "Check your network connection and that the Space is running."
  hint "Try: curl -s -o /dev/null -w '%%{http_code}' -X POST $PING_URL/reset"
  stop_at "Step 1"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
  hint "Make sure your Space is running and the URL is correct."
  hint "Try opening $PING_URL in your browser first."
  stop_at "Step 1"
fi

log "${BOLD}Step 2/3: Running docker build${NC} ..."

if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 2"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

log "${BOLD}Step 3/3: Running openenv validate${NC} ..."

if ! command -v openenv &>/dev/null; then
  fail "openenv command not found"
  hint "Install it: pip install openenv-core"
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0

________________________________________________________________
