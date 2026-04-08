"""Deterministic graders for the AgriEnv tasks."""

from __future__ import annotations

from typing import Callable

from .models import EpisodeSummary, GraderResult
from .tasks import TASKS, TaskConfig
from .utils import clamp


def _yield_score(summary: EpisodeSummary, task: TaskConfig) -> float:
    return clamp(summary.cumulative_yield / max(task.expected_yield, 1e-6), 0.0, 1.0)


def _efficiency_score(summary: EpisodeSummary, task: TaskConfig) -> float:
    return clamp(summary.average_efficiency / max(task.expected_efficiency, 1e-6), 0.0, 1.0)


def _stability_score(summary: EpisodeSummary) -> float:
    reward_component = clamp(1.0 - summary.reward_std / 0.45, 0.0, 1.0)
    return clamp(0.65 * summary.stability_index + 0.35 * reward_component, 0.0, 1.0)


def _moisture_control_score(summary: EpisodeSummary) -> float:
    return clamp(1.0 - summary.moisture_rmse / 0.16, 0.0, 1.0)


def _nutrient_control_score(summary: EpisodeSummary) -> float:
    return clamp(1.0 - summary.nutrient_rmse / 0.18, 0.0, 1.0)


def _pest_control_score(summary: EpisodeSummary) -> float:
    return clamp(1.0 - summary.mean_pest_density / 0.35, 0.0, 1.0)


def _budget_score(summary: EpisodeSummary) -> float:
    return clamp(0.35 + 0.65 * summary.water_budget_remaining, 0.0, 1.0)


def grade_easy(summary: EpisodeSummary) -> GraderResult:
    task = TASKS["easy"]
    metrics = {
        "yield_score": _yield_score(summary, task),
        "efficiency_score": _efficiency_score(summary, task),
        "stability_score": _stability_score(summary),
        "moisture_control_score": _moisture_control_score(summary),
    }
    score = clamp(
        0.40 * metrics["moisture_control_score"]
        + 0.25 * metrics["yield_score"]
        + 0.15 * metrics["efficiency_score"]
        + 0.20 * metrics["stability_score"],
        0.0,
        1.0,
    )
    return GraderResult(task_id=task.task_id, score=score, passed=score >= task.pass_threshold, metrics=metrics)


def grade_medium(summary: EpisodeSummary) -> GraderResult:
    task = TASKS["medium"]
    metrics = {
        "yield_score": _yield_score(summary, task),
        "efficiency_score": _efficiency_score(summary, task),
        "stability_score": _stability_score(summary),
        "nutrient_control_score": _nutrient_control_score(summary),
        "pest_control_score": _pest_control_score(summary),
    }
    score = clamp(
        0.22 * metrics["yield_score"]
        + 0.14 * metrics["efficiency_score"]
        + 0.14 * metrics["stability_score"]
        + 0.28 * metrics["nutrient_control_score"]
        + 0.22 * metrics["pest_control_score"],
        0.0,
        1.0,
    )
    return GraderResult(task_id=task.task_id, score=score, passed=score >= task.pass_threshold, metrics=metrics)


def grade_hard(summary: EpisodeSummary) -> GraderResult:
    task = TASKS["hard"]
    metrics = {
        "yield_score": _yield_score(summary, task),
        "efficiency_score": _efficiency_score(summary, task),
        "stability_score": _stability_score(summary),
        "moisture_control_score": _moisture_control_score(summary),
        "nutrient_control_score": _nutrient_control_score(summary),
        "pest_control_score": _pest_control_score(summary),
        "budget_score": _budget_score(summary),
    }
    score = clamp(
        0.22 * metrics["yield_score"]
        + 0.18 * metrics["efficiency_score"]
        + 0.16 * metrics["stability_score"]
        + 0.10 * metrics["moisture_control_score"]
        + 0.10 * metrics["nutrient_control_score"]
        + 0.12 * metrics["pest_control_score"]
        + 0.12 * metrics["budget_score"],
        0.0,
        1.0,
    )
    return GraderResult(task_id=task.task_id, score=score, passed=score >= task.pass_threshold, metrics=metrics)


TASK_GRADERS: dict[str, Callable[[EpisodeSummary], GraderResult]] = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


def grade_episode(summary: EpisodeSummary) -> GraderResult:
    try:
        grader = TASK_GRADERS[summary.task_id]
    except KeyError as exc:
        supported = ", ".join(sorted(TASK_GRADERS))
        raise ValueError(f"Unsupported task '{summary.task_id}'. Expected one of: {supported}.") from exc
    return grader(summary)
