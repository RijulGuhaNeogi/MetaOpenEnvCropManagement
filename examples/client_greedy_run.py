"""Minimal WebSocket client example.

Starts from a running server and executes the built-in greedy heuristic for one
task. This mirrors the real OpenEnv client path rather than the direct oracle
benchmark path.
"""
from __future__ import annotations

import argparse

from client import CropEnvClient
from agent.policy import greedy_action
from models import CropAction


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one oracle episode through the WebSocket client.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:7860",
        help="OpenEnv server base URL.",
    )
    parser.add_argument("--task-id", type=int, default=1, help="Task id to run.")
    parser.add_argument("--seed", type=int, default=190, help="Scenario seed.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with CropEnvClient(base_url=args.base_url).sync() as env:
        result = env.reset(seed=args.seed, task_id=args.task_id)
        obs = result.observation
        steps = 0

        while not result.done:
            action = CropAction(**greedy_action(obs, {}))
            result = env.step(action)
            obs = result.observation
            steps += 1

        print(
            {
                "task_id": args.task_id,
                "seed": args.seed,
                "steps": steps,
                "final_score": result.reward,
                "rubric_breakdown": obs.metadata.get("rubric_breakdown", {}),
            }
        )


if __name__ == "__main__":
    main()