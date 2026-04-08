"""Training-side action adapter for discrete RL policies.

This module keeps the public OpenEnv action schema unchanged while offering a
small discrete action vocabulary that is easier for RL exploration.
"""
from __future__ import annotations

from models import CropAction


DISCRETE_ACTION_MAP: dict[str, tuple[str, float]] = {
    "wait": ("wait", 0.0),
    "harvest": ("harvest", 0.0),
    "irrigate_small": ("irrigate", 2.0),
    "irrigate_medium": ("irrigate", 5.0),
    "irrigate_large": ("irrigate", 8.0),
    "fertilize_small": ("fertilize", 15.0),
    "fertilize_medium": ("fertilize", 30.0),
    "fertilize_large": ("fertilize", 50.0),
    "fertilize_slow_small": ("fertilize_slow", 15.0),
    "fertilize_slow_medium": ("fertilize_slow", 30.0),
    "fertilize_slow_large": ("fertilize_slow", 50.0),
}


def list_discrete_actions() -> list[str]:
    """Return the supported discrete training actions in stable order."""
    return list(DISCRETE_ACTION_MAP.keys())


def discrete_to_crop_action(discrete_action: str) -> CropAction:
    """Convert a discrete training action into the public CropAction schema."""
    if discrete_action not in DISCRETE_ACTION_MAP:
        valid = ", ".join(list_discrete_actions())
        raise ValueError(
            f"Unsupported discrete_action '{discrete_action}'. Valid actions: {valid}"
        )

    action_type, amount = DISCRETE_ACTION_MAP[discrete_action]
    return CropAction(action_type=action_type, amount=amount)
