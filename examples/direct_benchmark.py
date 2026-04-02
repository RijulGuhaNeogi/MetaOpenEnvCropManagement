"""Minimal local benchmark example.

Runs a short direct-environment sweep without starting the HTTP server.
Useful when validating environment logic or comparing policy changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark_sweep import build_result


def main() -> None:
    task_ids = [1, 2, 3]
    seeds = list(range(42, 45))
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