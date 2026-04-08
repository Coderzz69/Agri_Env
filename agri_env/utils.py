"""Shared helpers and constants for AgriEnv."""

from __future__ import annotations

import json
import math
from typing import Iterable


ACTION_BOUNDS = {
    "irrigation": (0.0, 3000.0),
    "nitrogen_injection": (0.0, 0.5),
    "phosphorus_injection": (0.0, 0.5),
    "potassium_injection": (0.0, 0.5),
    "co2_ppm": (300.0, 1200.0),
    "pesticide": (0.0, 1.0),
}

OBSERVATION_BOUNDS = {
    "soil_moisture": (0.0, 1.0),
    "nitrogen": (0.0, 1.0),
    "phosphorus": (0.0, 1.0),
    "potassium": (0.0, 1.0),
    "temperature_c": (7.5, 45.0),
    "humidity": (0.0, 100.0),
    "pest_density": (0.0, 1.0),
    "energy_price": (1.0, 17.0),
    "water_budget_remaining": (0.0, 1.0),
    "growth_stage_progress": (0.0, 1.0),
}

GROWTH_STAGES: list[tuple[float, str, float]] = [
    (0.00, "seed", 0.55),
    (0.20, "vegetative", 0.85),
    (0.50, "flowering", 1.00),
    (0.75, "fruiting", 1.15),
    (0.95, "harvest", 0.90),
]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def gaussian_score(value: float, target: float, width: float) -> float:
    safe_width = max(width, 1e-6)
    return math.exp(-((value - target) ** 2) / (2.0 * safe_width ** 2))


def mean(values: Iterable[float]) -> float:
    data = list(values)
    if not data:
        return 0.0
    return sum(data) / len(data)


def stddev(values: Iterable[float]) -> float:
    data = list(values)
    if len(data) < 2:
        return 0.0
    avg = mean(data)
    variance = sum((value - avg) ** 2 for value in data) / len(data)
    return math.sqrt(variance)


def rmse(errors: Iterable[float]) -> float:
    data = list(errors)
    if not data:
        return 0.0
    return math.sqrt(sum(error ** 2 for error in data) / len(data))


def nutrient_balance_score(nitrogen: float, phosphorus: float, potassium: float, target: float = 0.65) -> float:
    target_fit = mean(
        [
            gaussian_score(nitrogen, target, 0.18),
            gaussian_score(phosphorus, target, 0.18),
            gaussian_score(potassium, target, 0.18),
        ]
    )
    imbalance = (abs(nitrogen - phosphorus) + abs(phosphorus - potassium) + abs(nitrogen - potassium)) / 3.0
    symmetry = clamp(1.0 - imbalance / 0.5, 0.0, 1.0)
    return clamp(0.65 * target_fit + 0.35 * symmetry, 0.0, 1.0)


def stage_at_step(step: int, horizon: int) -> tuple[str, float, float]:
    progress = clamp(step / max(horizon - 1, 1), 0.0, 1.0)
    current_name = GROWTH_STAGES[-1][1]
    current_multiplier = GROWTH_STAGES[-1][2]
    for threshold, name, multiplier in GROWTH_STAGES:
        if progress >= threshold:
            current_name = name
            current_multiplier = multiplier
    return current_name, progress, current_multiplier


def compact_json(payload: object) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def safe_error_text(message: str | None) -> str:
    if not message:
        return "null"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", ".", ":"} else "_" for ch in message)[:160]
