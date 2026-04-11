#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
END_SCORE_RE = re.compile(r"^\[END\].*\bscore=([0-9]+(?:\.[0-9]+)?)\b")


def parse_seed_spec(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid inclusive range: {token}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    if not values:
        raise ValueError("At least one seed must be provided")
    return values


def parse_task_ids(raw: str) -> list[int]:
    task_ids = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not task_ids:
        raise ValueError("At least one task id must be provided")
    return task_ids


def _tail(text: str, lines: int = 30) -> str:
    chunks = text.strip().splitlines()
    if not chunks:
        return "<empty>"
    return "\n".join(chunks[-lines:])


def run_inference_once(*, env_url: str, seed: int, task_id: int, timeout_s: int) -> tuple[str, str, str]:
    env = os.environ.copy()
    env["ENV_URL"] = env_url
    env["SEED"] = str(seed)
    env["TASK_ID"] = str(task_id)
    # Set to empty string (not absent) so load_dotenv(override=False)
    # in inference.py won't restore credentials from .env
    env["API_KEY"] = ""
    env["HF_TOKEN"] = ""
    env["API_BASE_URL"] = ""
    env["TRAJECTORY_OUTPUT"] = ""

    result = subprocess.run(
        [sys.executable, "inference.py"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )

    stdout = result.stdout
    stderr = result.stderr
    if result.returncode != 0:
        raise RuntimeError(
            "inference.py failed for "
            f"seed={seed} task_id={task_id} exit_code={result.returncode}\n"
            f"stdout:\n{_tail(stdout)}\n\n"
            f"stderr:\n{_tail(stderr)}"
        )

    stdout_lines = stdout.splitlines()
    if not any(line.startswith("[START]") for line in stdout_lines):
        raise RuntimeError(f"Missing [START] line for seed={seed} task_id={task_id}")
    if not any(line.startswith("[STEP]") for line in stdout_lines):
        raise RuntimeError(f"Missing [STEP] line for seed={seed} task_id={task_id}")

    end_lines = [line for line in stdout_lines if line.startswith("[END]")]
    if len(end_lines) != 1:
        raise RuntimeError(
            f"Expected exactly 1 [END] line for seed={seed} task_id={task_id}, got {len(end_lines)}\n"
            f"stdout:\n{_tail(stdout)}"
        )

    match = END_SCORE_RE.match(end_lines[0])
    if match is None:
        raise RuntimeError(
            f"Could not parse score from [END] line for seed={seed} task_id={task_id}: {end_lines[0]}"
        )

    return match.group(1), stdout, stderr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic baseline reproducibility checks against inference.py.",
    )
    parser.add_argument(
        "--env-url",
        default=os.getenv("ENV_URL", "http://127.0.0.1:7860"),
        help="Base URL for the running environment server.",
    )
    parser.add_argument(
        "--seeds",
        default="42-44",
        help="Comma-separated seeds and/or inclusive ranges, for example 42-46 or 42,99.",
    )
    parser.add_argument(
        "--task-ids",
        default="1,2,3",
        help="Comma-separated task ids to evaluate.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="Number of repeated runs per seed/task pair.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-run timeout in seconds for inference.py.",
    )
    parser.add_argument(
        "--expect-min",
        type=float,
        default=None,
        help="Optional lower bound for every parsed score.",
    )
    parser.add_argument(
        "--expect-max",
        type=float,
        default=None,
        help="Optional upper bound for every parsed score.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    seeds = parse_seed_spec(args.seeds)
    task_ids = parse_task_ids(args.task_ids)
    if args.repeat < 2:
        raise ValueError("--repeat must be at least 2 to check reproducibility")

    baseline_scores: dict[tuple[int, int], str] = {}
    total_runs = len(seeds) * len(task_ids) * args.repeat
    completed = 0

    print(
        f"Running reproducibility check against {args.env_url} for seeds={seeds} task_ids={task_ids} repeat={args.repeat}",
        flush=True,
    )

    for run_idx in range(1, args.repeat + 1):
        for seed in seeds:
            for task_id in task_ids:
                completed += 1
                print(
                    f"[{completed}/{total_runs}] run={run_idx} seed={seed} task_id={task_id}",
                    flush=True,
                )
                score_str, _stdout, _stderr = run_inference_once(
                    env_url=args.env_url,
                    seed=seed,
                    task_id=task_id,
                    timeout_s=args.timeout,
                )
                score = float(score_str)

                if args.expect_min is not None and score < args.expect_min:
                    raise RuntimeError(
                        f"Score {score:.3f} fell below expect-min={args.expect_min:.3f} "
                        f"for seed={seed} task_id={task_id}"
                    )
                if args.expect_max is not None and score > args.expect_max:
                    raise RuntimeError(
                        f"Score {score:.3f} exceeded expect-max={args.expect_max:.3f} "
                        f"for seed={seed} task_id={task_id}"
                    )

                key = (seed, task_id)
                previous = baseline_scores.get(key)
                if previous is None:
                    baseline_scores[key] = score_str
                elif previous != score_str:
                    raise RuntimeError(
                        "Non-reproducible score detected for "
                        f"seed={seed} task_id={task_id}: first={previous} later={score_str}"
                    )

    print("Reproducibility check passed.", flush=True)
    for seed in seeds:
        for task_id in task_ids:
            print(
                f"seed={seed} task_id={task_id} score={baseline_scores[(seed, task_id)]}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())