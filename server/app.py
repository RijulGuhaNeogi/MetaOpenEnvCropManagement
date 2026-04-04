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
from openenv.core.env_server import create_app

from models import CropAction, CropObservation
from server.environment import CropEnvironment
from server.tasks import TASKS

app = create_app(
    CropEnvironment,
    CropAction,
    CropObservation,
    env_name="crop_management",
)


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
    """Return deterministic greedy scores for all tasks (seed=42).

    Results are cached after first computation since they are fully
    deterministic — same seed + same greedy policy = same scores.
    """
    global _baseline_cache
    if _baseline_cache is not None:
        return _baseline_cache

    from agent.inference import greedy_action as _greedy_action

    results = {}
    for task_id in sorted(TASKS.keys()):
        env = CropEnvironment()
        obs = env.reset(seed=42, task_id=task_id)
        fert_done: set[str] = set()
        steps = 0
        while not obs.done:
            action = CropAction(**_greedy_action(obs, fert_done))
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)
