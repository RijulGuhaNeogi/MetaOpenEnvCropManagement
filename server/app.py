"""FastAPI application entry point.

Uses OpenEnv's create_app() to automatically register the standard
endpoints: /health, /reset, /step, /state, /ws, and /docs.

The custom /tasks endpoint lists available task definitions.

Note: HTTP endpoints (/reset, /step) are stateless per-request — a new
CropEnvironment instance is created for each request.  For multi-step
episodes, use the WebSocket endpoint (/ws), which maintains a persistent
session across steps.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""
from typing import Any

from fastapi.responses import RedirectResponse

from openenv.core.env_server import create_app
from pydantic import BaseModel

from models import CropAction, CropObservation
from server.environment import CropEnvironment
from server.grader import grade_episode
from server.tasks import TASKS


class GradeRequest(BaseModel):
    actual_yield: float
    target_yield: float
    total_water: float
    total_n: float
    total_cost: float
    budget: float
    harvest_dvs: float
    harvested: bool
    actions_taken: list[dict[str, Any]] = []
    task_id: int = 1
    explicit_harvest: bool = True

app = create_app(
    CropEnvironment,
    CropAction,
    CropObservation,
    env_name="crop_management",
)


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/docs", status_code=307)


@app.get("/web", include_in_schema=False)
def web_redirect():
    return RedirectResponse(url="/docs", status_code=307)


@app.post("/grader")
def grade(req: GradeRequest):
    score, breakdown = grade_episode(
        actual_yield=req.actual_yield,
        target_yield=req.target_yield,
        total_water=req.total_water,
        total_n=req.total_n,
        total_cost=req.total_cost,
        budget=req.budget,
        harvest_dvs=req.harvest_dvs,
        harvested=req.harvested,
        actions_taken=req.actions_taken,
        task_id=req.task_id,
        explicit_harvest=req.explicit_harvest,
    )
    return {"score": score, "breakdown": breakdown}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": t["id"],
                "name": t["name"],
                "difficulty": t["difficulty"],
                "instructions": t["instructions"],
            }
            for t in TASKS.values()
        ]
    }


_baseline_cache: dict | None = None
_ceiling_cache: dict | None = None


@app.get("/baseline")
def run_baseline():
    """Return deterministic greedy-baseline scores for all tasks (seed=42).

    Results are cached after first computation since they are fully
    deterministic — same seed + same greedy policy = same scores.
    """
    global _baseline_cache
    if _baseline_cache is not None:
        return _baseline_cache

    from agent.policy import greedy_action

    results = {}
    for task_id in sorted(TASKS.keys()):
        env = CropEnvironment()
        obs = env.reset(seed=42, task_id=task_id)
        steps = 0
        while not obs.done:
            action = CropAction(**greedy_action(obs, {}))
            obs = env.step(action)
            steps += 1
        results[task_id] = {
            "task_id": task_id,
            "name": TASKS[task_id]["name"],
            "score": obs.rubric_reward if obs.rubric_reward is not None else obs.reward,
            "steps": steps,
        }

    scores = [r["score"] for r in results.values()]
    _baseline_cache = {
        "seed": 42,
        "policy": "greedy",
        "tasks": results,
        "overall_mean": round(sum(scores) / len(scores), 4),
    }
    return _baseline_cache


@app.get("/ceiling")
def run_ceiling():
    """Return deterministic oracle-ceiling scores for all tasks (seed=42)."""
    global _ceiling_cache
    if _ceiling_cache is not None:
        return _ceiling_cache

    results = {}
    for task_id in sorted(TASKS.keys()):
        env = CropEnvironment()
        obs = env.reset(seed=42, task_id=task_id)
        steps = 0
        while not obs.done:
            action = CropAction(**env.oracle_reference_action())
            obs = env.step(action)
            steps += 1
        results[task_id] = {
            "task_id": task_id,
            "name": TASKS[task_id]["name"],
            "score": obs.rubric_reward if obs.rubric_reward is not None else obs.reward,
            "steps": steps,
        }

    scores = [r["score"] for r in results.values()]
    _ceiling_cache = {
        "seed": 42,
        "policy": "oracle_ceiling",
        "tasks": results,
        "overall_mean": round(sum(scores) / len(scores), 4),
    }
    return _ceiling_cache


def main():
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)


if __name__ == "__main__":
    main()
