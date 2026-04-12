"""Agent-side code: policies, training adapters, and evaluation utilities."""
from agent.policy import greedy_action  # noqa: F401
from agent.policy import oracle_action  # noqa: F401
from agent.training_adapter import (  # noqa: F401
    discrete_to_crop_action,
    list_discrete_actions,
)
