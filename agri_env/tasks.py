"""Task definitions for AgriEnv."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    difficulty: str
    title: str
    description: str
    horizon: int
    alpha: float
    seed: int
    water_budget_liters: float
    sensor_noise: float
    weather_variability: float
    target_moisture: float = 0.70
    target_nutrient: float = 0.65
    expected_yield: float = 30.0
    expected_efficiency: float = 8.0
    pass_threshold: float = 0.70


TASKS: dict[str, TaskConfig] = {
    "easy": TaskConfig(
        task_id="easy",
        difficulty="easy",
        title="Maintain Optimal Soil Moisture",
        description="Keep soil moisture near 0.70 with smooth, economical irrigation.",
        horizon=60,
        alpha=0.20,
        seed=11,
        water_budget_liters=90000.0,
        sensor_noise=0.008,
        weather_variability=0.55,
        expected_yield=37.0,
        expected_efficiency=12.5,
        pass_threshold=0.80,
    ),
    "medium": TaskConfig(
        task_id="medium",
        difficulty="medium",
        title="Balance Nutrients and Suppress Pests",
        description="Balance NPK, keep pests low, and maintain stable greenhouse conditions.",
        horizon=90,
        alpha=0.28,
        seed=23,
        water_budget_liters=120000.0,
        sensor_noise=0.012,
        weather_variability=0.85,
        expected_yield=56.0,
        expected_efficiency=12.2,
        pass_threshold=0.77,
    ),
    "hard": TaskConfig(
        task_id="hard",
        difficulty="hard",
        title="Full Yield and Cost Optimization",
        description="Maximize yield under cost pressure, noisy sensors, and a tighter resource budget.",
        horizon=140,
        alpha=0.36,
        seed=37,
        water_budget_liters=150000.0,
        sensor_noise=0.018,
        weather_variability=1.10,
        expected_yield=86.0,
        expected_efficiency=12.3,
        pass_threshold=0.80,
    ),
}


def get_task(task_id: str) -> TaskConfig:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        supported = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task '{task_id}'. Expected one of: {supported}.") from exc
