"""Run a direct multi-seed oracle baseline sweep.

This utility evaluates the current oracle policy directly against the
CropEnvironment without going through the HTTP/WebSocket inference path.
That makes it the fastest and cleanest way to measure policy stability
across many deterministic seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

from agent.policy import oracle_action
from models import CropAction
from server.environment import CropEnvironment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a direct multi-seed oracle baseline sweep.",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=42,
        help="First seed in the sweep (default: 42).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of consecutive seeds to evaluate (default: 10).",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Task ids to evaluate (default: 1 2 3).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of formatted text.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to write the full sweep result as JSON.",
    )
    return parser.parse_args()


def run_episode(task_id: int, seed: int) -> dict:
    env = CropEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    steps = 0

    while not obs.done:
        action = CropAction(**env.oracle_reference_action())
        obs = env.step(action)
        steps += 1
        if steps > 80:
            raise RuntimeError(
                f"Unexpected long episode: task={task_id}, seed={seed}, steps={steps}"
            )

    breakdown = obs.metadata.get("rubric_breakdown", {})
    return {
        "seed": seed,
        "score": float(obs.reward or 0.0),
        "steps": steps,
        "yield_score": float(breakdown.get("yield_score", 0.0)),
        "water_efficiency": float(breakdown.get("water_efficiency", 0.0)),
        "cost_efficiency": float(breakdown.get("cost_efficiency", 0.0)),
        "timing_quality": float(breakdown.get("timing_quality", 0.0)),
        "harvest_timing": float(breakdown.get("harvest_timing", 0.0)),
        "total_water": float(breakdown.get("total_water", 0.0)),
        "total_n": float(breakdown.get("total_n", 0.0)),
        "total_cost": float(breakdown.get("total_cost", 0.0)),
    }


def summarize_runs(runs: list[dict]) -> dict:
    scores = [run["score"] for run in runs]
    steps = [run["steps"] for run in runs]
    total_water = [run["total_water"] for run in runs]
    total_n = [run["total_n"] for run in runs]
    total_cost = [run["total_cost"] for run in runs]

    return {
        "mean_score": round(mean(scores), 4),
        "std_score": round(pstdev(scores), 4),
        "min_score": round(min(scores), 4),
        "max_score": round(max(scores), 4),
        "mean_steps": round(mean(steps), 2),
        "mean_water_cm": round(mean(total_water), 2),
        "mean_n_kg_ha": round(mean(total_n), 2),
        "mean_cost": round(mean(total_cost), 2),
        "per_seed_scores": {
            run["seed"]: round(run["score"], 4)
            for run in runs
        },
    }


def build_result(task_ids: list[int], seeds: list[int]) -> dict:
    task_summaries: dict[int, dict] = {}

    for task_id in task_ids:
        runs = [run_episode(task_id, seed) for seed in seeds]
        summary = summarize_runs(runs)
        summary["runs"] = runs
        task_summaries[task_id] = summary

    overall_mean = round(
        mean(summary["mean_score"] for summary in task_summaries.values()),
        4,
    )
    return {
        "start_seed": seeds[0],
        "count": len(seeds),
        "tasks": task_summaries,
        "overall_mean": overall_mean,
        "difficulty_ordering_holds": all(
            task_summaries[earlier]["mean_score"] >= task_summaries[later]["mean_score"]
            for earlier, later in zip(task_ids, task_ids[1:])
        ) if task_ids == sorted(task_ids) else None,
    }


def _write_result(path_str: str, result: dict) -> None:
    output_path = Path(path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")

    seeds = list(range(args.start_seed, args.start_seed + args.count))
    result = build_result(args.tasks, seeds)

    if args.output:
        _write_result(args.output, result)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print(f"Seeds: {seeds[0]}-{seeds[-1]} ({len(seeds)} total)\n")
    for task_id in args.tasks:
        print(f"TASK {task_id}")
        summary = result["tasks"][task_id]
        print(
            {
                key: value
                for key, value in summary.items()
                if key not in {"per_seed_scores", "runs"}
            }
        )
        print("per_seed_scores", summary["per_seed_scores"])
        print("")

    print("OVERALL")
    print(
        {
            "task_mean_scores": {
                task_id: result["tasks"][task_id]["mean_score"]
                for task_id in args.tasks
            },
            "overall_mean": result["overall_mean"],
            "difficulty_ordering_holds": result["difficulty_ordering_holds"],
        }
    )
    if args.output:
        print(f"\nSaved full sweep result to {args.output}")


if __name__ == "__main__":
    main()