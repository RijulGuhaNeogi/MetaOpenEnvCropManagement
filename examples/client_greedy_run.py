"""Minimal WebSocket client example.

Starts from a running server and executes the built-in greedy policy for one task.
This mirrors the real OpenEnv client path rather than the direct benchmark path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from client import CropEnvClient
from inference import greedy_action
from models import CropAction


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one greedy episode through the WebSocket client.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="OpenEnv server base URL.",
    )
    parser.add_argument("--task-id", type=int, default=1, help="Task id to run.")
    parser.add_argument("--seed", type=int, default=42, help="Scenario seed.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    fert_done = set()

    with CropEnvClient(base_url=args.base_url).sync() as env:
        result = env.reset(seed=args.seed, task_id=args.task_id)
        obs = result.observation
        steps = 0

        while not result.done:
            action = CropAction(**greedy_action(obs, fert_done))
            result = env.step(action)
            obs = result.observation
            steps += 1

        print(
            {
                "task_id": args.task_id,
                "seed": args.seed,
                "steps": steps,
                "final_score": result.reward,
                "grade_breakdown": obs.metadata.get("grade_breakdown", {}),
            }
        )


if __name__ == "__main__":
    main()