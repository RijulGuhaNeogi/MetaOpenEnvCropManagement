"""FastAPI application entry point.

Uses OpenEnv's create_app() to automatically register the standard
endpoints: /health, /reset, /step, /state, /ws, /docs, /web.

The custom /tasks endpoint lists available task definitions.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `models` and `server` are importable
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)
