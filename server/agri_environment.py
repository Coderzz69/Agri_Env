"""OpenEnv server wrapper for AgriEnv."""

from __future__ import annotations

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from agri_env.env import AgriEnv
from agri_env.models import Action, AgriState, Observation


class AgriEnvironment(Environment[Action, Observation, AgriState]):
    """FastAPI/OpenEnv-facing environment wrapper around the local simulator."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._env = AgriEnv()

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs) -> Observation:
        task = kwargs.get("task")
        return self._env.reset(seed=seed, task=task, episode_id=episode_id)

    def step(self, action: Action, timeout_s: float | None = None, **kwargs) -> Observation:
        del timeout_s, kwargs
        observation, _reward, _done, _info = self._env.step(action)
        return observation

    @property
    def state(self) -> AgriState:
        return self._env.state()

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="AgriEnv",
            description="Precision agriculture greenhouse control environment with typed OpenEnv APIs.",
            version="1.0.0",
            author="AgriEnv Hackathon Submission",
            documentation_url="https://huggingface.co/spaces/<username>/agri-env",
        )
