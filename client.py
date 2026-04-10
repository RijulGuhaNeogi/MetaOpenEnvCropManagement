"""WebSocket client for the Crop Management OpenEnv.

Extends the generic EnvClient to handle CropAction serialization and
CropObservation / CropState deserialization. Used by agent/inference.py for
multi-step episodes over WebSocket (the HTTP endpoints are stateless).

Usage:
    sync_client = CropEnvClient(base_url="http://localhost:7860").sync()
    with sync_client:
        result = sync_client.reset(seed=42, task_id=1)
        while not result.done:
            result = sync_client.step(CropAction(action_type="wait"))
"""
from __future__ import annotations

from typing import Any, Dict

from openenv.core.env_client import EnvClient, StepResult

from models import CropAction, CropObservation, CropState


class CropEnvClient(
    EnvClient[CropAction, CropObservation, CropState]
):
    """Typed EnvClient for the Crop Management environment."""

    def _step_payload(self, action: CropAction) -> Dict[str, Any]:
        """Serialize a CropAction into the JSON dict sent over WebSocket."""
        return action.model_dump(exclude_none=True)

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[CropObservation]:
        """Deserialize the server response into a StepResult."""
        obs_data = dict(payload.get("observation", payload))
        for field in ("reward", "done", "metadata", "rubric_reward"):
            if field in payload:
                obs_data[field] = payload[field]
        obs = CropObservation.model_validate(obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs.reward),
            done=payload.get("done", obs.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CropState:
        """Deserialize the /state response into a CropState."""
        return CropState.model_validate(payload)
