"""OpenEnv WebSocket client for AgriEnv."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import Action, AgriState, Observation


class AgriEnvClient(EnvClient[Action, Observation, AgriState]):
    """Typed client for a running AgriEnv OpenEnv server."""

    def _step_payload(self, action: Action) -> Dict[str, Any]:
        payload = action.to_dict()
        payload.pop("metadata", None)
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[Observation]:
        obs_data = dict(payload.get("observation", {}))
        obs_data["done"] = payload.get("done", obs_data.get("done", False))
        obs_data["reward"] = payload.get("reward", obs_data.get("reward"))
        observation = Observation.from_mapping(obs_data)
        reward = payload.get("reward")
        return StepResult(
            observation=observation,
            reward=float(reward) if reward is not None else None,
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> AgriState:
        return AgriState.from_mapping(payload)
