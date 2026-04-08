"""Typed OpenEnv models for AgriEnv."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from openenv.core.env_server.types import Action as OpenEnvAction
from openenv.core.env_server.types import Observation as OpenEnvObservation
from openenv.core.env_server.types import State as OpenEnvState
from pydantic import BaseModel, Field


class ModelHelpers:
    """Compatibility helpers shared by Pydantic models."""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]):
        if hasattr(cls, "model_validate"):
            return cls.model_validate(value)  # type: ignore[attr-defined]
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump()  # type: ignore[attr-defined]
        return self.dict()  # type: ignore[attr-defined]


class Action(ModelHelpers, OpenEnvAction):
    """Continuous control action for the greenhouse."""

    irrigation: float = Field(..., ge=0.0, le=3000.0)
    nitrogen_injection: float = Field(..., ge=0.0, le=0.5)
    phosphorus_injection: float = Field(..., ge=0.0, le=0.5)
    potassium_injection: float = Field(..., ge=0.0, le=0.5)
    co2_ppm: float = Field(..., ge=300.0, le=1200.0)
    pesticide: float = Field(..., ge=0.0, le=1.0)

    @classmethod
    def from_any(cls, value: Any) -> "Action":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls.from_mapping(value)
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            values = list(value)
            if len(values) != 6:
                raise ValueError("Action sequence must contain 6 values.")
            return cls(
                irrigation=float(values[0]),
                nitrogen_injection=float(values[1]),
                phosphorus_injection=float(values[2]),
                potassium_injection=float(values[3]),
                co2_ppm=float(values[4]),
                pesticide=float(values[5]),
            )
        raise TypeError(f"Unsupported action type: {type(value)!r}")

    def clipped(self) -> "Action":
        return Action(
            irrigation=min(max(float(self.irrigation), 0.0), 3000.0),
            nitrogen_injection=min(max(float(self.nitrogen_injection), 0.0), 0.5),
            phosphorus_injection=min(max(float(self.phosphorus_injection), 0.0), 0.5),
            potassium_injection=min(max(float(self.potassium_injection), 0.0), 0.5),
            co2_ppm=min(max(float(self.co2_ppm), 300.0), 1200.0),
            pesticide=min(max(float(self.pesticide), 0.0), 1.0),
            metadata=dict(self.metadata),
        )

    def to_vector(self) -> list[float]:
        return [
            self.irrigation,
            self.nitrogen_injection,
            self.phosphorus_injection,
            self.potassium_injection,
            self.co2_ppm,
            self.pesticide,
        ]


class Observation(ModelHelpers, OpenEnvObservation):
    """Continuous observation returned to the agent."""

    task_id: str = Field(default="hard")
    difficulty: str = Field(default="hard")
    task_title: str = Field(default="Full Yield and Cost Optimization")
    stage_name: str = Field(default="seed")
    soil_moisture: float = Field(..., ge=0.0, le=1.0)
    nitrogen: float = Field(..., ge=0.0, le=1.0)
    phosphorus: float = Field(..., ge=0.0, le=1.0)
    potassium: float = Field(..., ge=0.0, le=1.0)
    temperature_c: float = Field(..., ge=0.0, le=50.0)
    humidity: float = Field(..., ge=0.0, le=100.0)
    pest_density: float = Field(..., ge=0.0, le=1.0)
    energy_price: float = Field(..., ge=0.0)
    water_budget_remaining: float = Field(..., ge=0.0, le=1.0)
    growth_stage_progress: float = Field(..., ge=0.0, le=1.0)
    cumulative_yield: float = Field(default=0.0, ge=0.0)
    total_cost: float = Field(default=0.0, ge=0.0)

    def to_vector(self) -> list[float]:
        return [
            self.soil_moisture,
            self.nitrogen,
            self.phosphorus,
            self.potassium,
            self.temperature_c,
            self.humidity,
            self.pest_density,
            self.energy_price,
            self.water_budget_remaining,
            self.growth_stage_progress,
        ]

    @property
    def nutrient_mean(self) -> float:
        return (self.nitrogen + self.phosphorus + self.potassium) / 3.0


class AgriState(ModelHelpers, OpenEnvState):
    """Full latent environment state used by OpenEnv clients."""

    task_id: str = Field(default="hard")
    difficulty: str = Field(default="hard")
    task_title: str = Field(default="Full Yield and Cost Optimization")
    horizon: int = Field(default=140, ge=1)
    growth_stage: str = Field(default="seed")
    growth_stage_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    growth_stage_multiplier: float = Field(default=1.0, ge=0.0)
    soil_moisture: float = Field(default=0.0, ge=0.0, le=1.0)
    nitrogen: float = Field(default=0.0, ge=0.0, le=1.0)
    phosphorus: float = Field(default=0.0, ge=0.0, le=1.0)
    potassium: float = Field(default=0.0, ge=0.0, le=1.0)
    temperature_c: float = Field(default=0.0, ge=0.0, le=50.0)
    humidity: float = Field(default=0.0, ge=0.0, le=100.0)
    pest_density: float = Field(default=0.0, ge=0.0, le=1.0)
    energy_price: float = Field(default=0.0, ge=0.0)
    water_budget_remaining: float = Field(default=0.0, ge=0.0, le=1.0)
    water_used_liters: float = Field(default=0.0, ge=0.0)
    biomass: float = Field(default=0.0, ge=0.0)
    cumulative_yield: float = Field(default=0.0, ge=0.0)
    total_cost: float = Field(default=0.0, ge=0.0)
    episode_summary: dict[str, Any] = Field(default_factory=dict)
    grader: dict[str, Any] = Field(default_factory=dict)


class Reward(ModelHelpers, BaseModel):
    """Structured reward and shaping components."""

    total: float = Field(...)
    crop_growth: float = Field(...)
    moisture_alignment: float = Field(...)
    nutrient_alignment: float = Field(...)
    efficiency_bonus: float = Field(...)
    stability_bonus: float = Field(...)
    task_bonus: float = Field(...)
    operational_cost: float = Field(...)
    resource_penalty: float = Field(...)
    pest_penalty: float = Field(...)


class EpisodeSummary(ModelHelpers, BaseModel):
    """Deterministic episode metrics consumed by task graders."""

    task_id: str = Field(...)
    steps: int = Field(..., ge=0)
    cumulative_yield: float = Field(..., ge=0.0)
    total_cost: float = Field(..., ge=0.0)
    average_efficiency: float = Field(..., ge=0.0)
    moisture_rmse: float = Field(..., ge=0.0)
    nutrient_rmse: float = Field(..., ge=0.0)
    mean_pest_density: float = Field(..., ge=0.0)
    stability_index: float = Field(..., ge=0.0, le=1.0)
    water_budget_remaining: float = Field(..., ge=0.0, le=1.0)
    reward_mean: float = Field(...)
    reward_std: float = Field(..., ge=0.0)


class GraderResult(ModelHelpers, BaseModel):
    """Standard task score in [0, 1]."""

    task_id: str = Field(...)
    score: float = Field(..., ge=0.0, le=1.0)
    passed: bool = Field(...)
    metrics: dict[str, float] = Field(default_factory=dict)
