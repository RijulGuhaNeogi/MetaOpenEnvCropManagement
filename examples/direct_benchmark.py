"""Minimal local benchmark example.

Runs a short direct-environment sweep without starting the HTTP server.
Useful when validating environment logic or comparing policy changes.
"""
from __future__ import annotations

from agent.benchmark_sweep import build_result


def main() -> None:
    task_ids = [1, 2, 3]
    seeds = list(range(190, 193))
    result = build_result(task_ids, seeds)

    print("Direct benchmark example")
    print(
        {
            "seeds": seeds,
            "task_mean_scores": {
                task_id: result["tasks"][task_id]["mean_score"]
                for task_id in task_ids
            },
            "overall_mean": result["overall_mean"],
            "difficulty_ordering_holds": result["difficulty_ordering_holds"],
        }
    )


if __name__ == "__main__":
    main()