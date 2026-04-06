"""FastAPI application entry point.

Uses OpenEnv's create_app() to automatically register the standard
endpoints: /health, /reset, /step, /state, /ws, /docs, /web.

The custom /tasks endpoint lists available task definitions.

Note: HTTP endpoints (/reset, /step) are stateless per-request — a new
CropEnvironment instance is created for each request.  For multi-step
episodes, use the WebSocket endpoint (/ws), which maintains a persistent
session across steps.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""
from typing import Any

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


@app.get("/baseline")
def run_baseline():
    """Return deterministic oracle scores for all tasks (seed=42).

    Results are cached after first computation since they are fully
    deterministic — same seed + same oracle policy = same scores.
    """
    global _baseline_cache
    if _baseline_cache is not None:
        return _baseline_cache

    from agent.inference import oracle_action as _oracle_action

    results = {}
    for task_id in sorted(TASKS.keys()):
        env = CropEnvironment()
        obs = env.reset(seed=42, task_id=task_id)
        oracle_state: dict = {}
        steps = 0
        while not obs.done:
            action = CropAction(**_oracle_action(obs, oracle_state))
            obs = env.step(action)
            steps += 1
        results[task_id] = {
            "task_id": task_id,
            "name": TASKS[task_id]["name"],
            "score": obs.reward,
            "steps": steps,
        }

    scores = [r["score"] for r in results.values()]
    _baseline_cache = {
        "seed": 42,
        "tasks": results,
        "overall_mean": round(sum(scores) / len(scores), 4),
    }
    return _baseline_cache


def main():
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
