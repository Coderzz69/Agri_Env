"""OpenEnv-compatible greenhouse environment for precision agriculture."""

from __future__ import annotations

import math
import random
from typing import Any
from uuid import uuid4

try:  # pragma: no cover - optional compatibility layer.
    import numpy as np
    from gymnasium import spaces
except ModuleNotFoundError:  # pragma: no cover
    np = None
    spaces = None

from .graders import grade_episode
from .models import Action, AgriState, EpisodeSummary, Observation, Reward
from .tasks import TaskConfig, get_task
from .utils import ACTION_BOUNDS, OBSERVATION_BOUNDS, clamp, gaussian_score, mean, nutrient_balance_score, rmse, stage_at_step, stddev


class AgriEnv:
    """OpenEnv-style precision agriculture environment with deterministic task grading."""

    metadata = {"name": "agri-env", "render_modes": ["human"]}

    def __init__(self, task: str = "hard", seed: int | None = None, render_mode: str | None = None) -> None:
        self.render_mode = render_mode
        self.task: TaskConfig = get_task(task)
        self._base_seed = seed if seed is not None else self.task.seed
        self._rng = random.Random(self._base_seed)
        self._latent_state: dict[str, float] = {}
        self._metrics: dict[str, list[float]] = {}
        self.current_step = 0
        self.cumulative_yield = 0.0
        self.total_cost = 0.0
        self._last_summary: EpisodeSummary | None = None
        self._episode_id = str(uuid4())

        if spaces is not None and np is not None:
            self.action_space = spaces.Box(
                low=np.array([bounds[0] for bounds in ACTION_BOUNDS.values()], dtype=np.float32),
                high=np.array([bounds[1] for bounds in ACTION_BOUNDS.values()], dtype=np.float32),
                dtype=np.float32,
            )
            self.observation_space = spaces.Box(
                low=np.array([bounds[0] for bounds in OBSERVATION_BOUNDS.values()], dtype=np.float32),
                high=np.array([bounds[1] for bounds in OBSERVATION_BOUNDS.values()], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = ACTION_BOUNDS
            self.observation_space = OBSERVATION_BOUNDS

    def reset(self, seed: int | None = None, task: str | None = None, episode_id: str | None = None) -> Observation:
        """Reset the environment and return the first observation."""

        if task is not None:
            self.task = get_task(task)
            if seed is None:
                self._base_seed = self.task.seed
        if seed is not None:
            self._base_seed = seed
        self._rng = random.Random(self._base_seed)
        self.current_step = 0
        self.cumulative_yield = 0.0
        self.total_cost = 0.0
        self._last_summary = None
        self._episode_id = episode_id or str(uuid4())
        self._metrics = {
            "yield": [],
            "cost": [],
            "moisture_error": [],
            "nutrient_error": [],
            "pest": [],
            "reward": [],
            "stability": [],
        }

        task_bias = {"easy": 0.0, "medium": 0.03, "hard": 0.06}[self.task.task_id]
        self._latent_state = {
            "soil_moisture": self._rng.uniform(0.58 - task_bias, 0.74 - task_bias / 2.0),
            "nitrogen": self._rng.uniform(0.42 - task_bias, 0.62),
            "phosphorus": self._rng.uniform(0.40 - task_bias, 0.60),
            "potassium": self._rng.uniform(0.44 - task_bias, 0.66),
            "temperature_c": self._rng.uniform(20.0, 27.5),
            "humidity": self._rng.uniform(48.0, 68.0),
            "pest_density": self._rng.uniform(0.04 + task_bias, 0.14 + task_bias * 1.8),
            "energy_price": self._rng.uniform(3.0, 7.0),
            "water_used_liters": 0.0,
            "biomass": 0.18,
        }
        self._temp_phase = self._rng.uniform(0.0, math.tau)
        self._humidity_phase = self._rng.uniform(0.0, math.tau)
        self._energy_phase = self._rng.uniform(0.0, math.tau)
        return self._observe(done=False, reward=None, extra_metadata={"event": "reset"})

    def step(self, action: Action | dict[str, float] | list[float] | tuple[float, ...]) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        """Advance one timestep and return observation, reward, done, and info."""

        if not self._latent_state:
            raise RuntimeError("Call reset() before step().")

        action_model = Action.from_any(action).clipped()
        stage_name, stage_progress, stage_multiplier = stage_at_step(self.current_step, self.task.horizon)
        previous_state = dict(self._latent_state)
        irrigation_norm = action_model.irrigation / ACTION_BOUNDS["irrigation"][1]
        nutrient_mean_injection = (
            action_model.nitrogen_injection
            + action_model.phosphorus_injection
            + action_model.potassium_injection
        ) / 3.0
        co2_norm = clamp((action_model.co2_ppm - 400.0) / 800.0, 0.0, 1.0)

        temp_trend = 1.8 * math.sin((self.current_step / 8.0) + self._temp_phase) * self.task.weather_variability
        humidity_trend = 6.5 * math.cos((self.current_step / 9.5) + self._humidity_phase) * self.task.weather_variability
        energy_trend = 1.2 * math.sin((self.current_step / 6.5) + self._energy_phase) * self.task.weather_variability

        target_temp = 23.0 + temp_trend + 0.45 * co2_norm
        temp_noise = self._rng.uniform(-0.35, 0.35) * self.task.weather_variability
        self._latent_state["temperature_c"] += 0.18 * (target_temp - self._latent_state["temperature_c"]) + temp_noise

        target_humidity = 58.0 + humidity_trend + irrigation_norm * 16.0 - max(self._latent_state["temperature_c"] - 25.0, 0.0) * 1.1
        humidity_noise = self._rng.uniform(-1.4, 1.4) * self.task.weather_variability
        self._latent_state["humidity"] += 0.16 * (target_humidity - self._latent_state["humidity"]) + humidity_noise

        evaporation = (
            0.016
            + max(self._latent_state["temperature_c"] - 22.0, 0.0) * 0.0017
            + max(55.0 - self._latent_state["humidity"], 0.0) * 0.00045
            + 0.0025 * stage_multiplier
        )
        moisture_noise = self._rng.uniform(-0.010, 0.010) * self.task.weather_variability
        self._latent_state["soil_moisture"] += irrigation_norm * 0.58 - evaporation + moisture_noise

        leaching = max(irrigation_norm - 0.65, 0.0) * 0.08
        nutrient_draw = 0.012 + stage_multiplier * 0.010
        self._latent_state["nitrogen"] += action_model.nitrogen_injection * 0.55 - nutrient_draw * 0.90 - leaching + self._rng.uniform(-0.008, 0.008)
        self._latent_state["phosphorus"] += action_model.phosphorus_injection * 0.45 - nutrient_draw * 0.65 - leaching * 0.55 + self._rng.uniform(-0.007, 0.007)
        self._latent_state["potassium"] += action_model.potassium_injection * 0.50 - nutrient_draw * 0.80 - leaching * 0.70 + self._rng.uniform(-0.007, 0.007)

        pest_pressure = 0.008 + max(self._latent_state["humidity"] - 68.0, 0.0) * 0.0008 + stage_multiplier * 0.006
        pesticide_effect = action_model.pesticide * 0.18
        oversaturation_pest_risk = max(self._latent_state["soil_moisture"] - 0.84, 0.0) * 0.08
        pest_noise = self._rng.uniform(-0.006, 0.006)
        self._latent_state["pest_density"] += pest_pressure + oversaturation_pest_risk - pesticide_effect - 0.010 + pest_noise

        target_energy_price = 5.2 + energy_trend
        energy_noise = self._rng.uniform(-0.28, 0.28)
        self._latent_state["energy_price"] += 0.20 * (target_energy_price - self._latent_state["energy_price"]) + energy_noise

        self._latent_state["soil_moisture"] = clamp(self._latent_state["soil_moisture"], 0.0, 1.0)
        self._latent_state["nitrogen"] = clamp(self._latent_state["nitrogen"], 0.0, 1.0)
        self._latent_state["phosphorus"] = clamp(self._latent_state["phosphorus"], 0.0, 1.0)
        self._latent_state["potassium"] = clamp(self._latent_state["potassium"], 0.0, 1.0)
        self._latent_state["temperature_c"] = clamp(self._latent_state["temperature_c"], 7.5, 45.0)
        self._latent_state["humidity"] = clamp(self._latent_state["humidity"], 25.0, 100.0)
        self._latent_state["pest_density"] = clamp(self._latent_state["pest_density"], 0.0, 1.0)
        self._latent_state["energy_price"] = clamp(self._latent_state["energy_price"], 1.0, 17.0)

        water_cost = irrigation_norm * 0.18
        nutrient_cost = nutrient_mean_injection * 0.42
        co2_cost = 0.22 * co2_norm * (self._latent_state["energy_price"] / 10.0)
        pesticide_cost = action_model.pesticide * 0.20
        operational_cost = water_cost + nutrient_cost + co2_cost + pesticide_cost

        moisture_alignment = gaussian_score(self._latent_state["soil_moisture"], self.task.target_moisture, 0.11)
        nutrient_alignment = nutrient_balance_score(
            self._latent_state["nitrogen"],
            self._latent_state["phosphorus"],
            self._latent_state["potassium"],
            target=self.task.target_nutrient,
        )
        temperature_alignment = gaussian_score(self._latent_state["temperature_c"], 24.0, 4.5)
        humidity_alignment = gaussian_score(self._latent_state["humidity"], 62.0, 13.0)
        pest_factor = clamp(1.0 - self._latent_state["pest_density"] ** 1.1, 0.0, 1.0)
        water_remaining = clamp(
            1.0 - (self._latent_state["water_used_liters"] + action_model.irrigation) / self.task.water_budget_liters,
            0.0,
            1.0,
        )
        growth = (
            stage_multiplier
            * moisture_alignment
            * nutrient_alignment
            * temperature_alignment
            * humidity_alignment
            * pest_factor
            * (0.88 + 0.12 * co2_norm)
            * (0.85 + 0.15 * water_remaining)
        )

        moisture_bonus = 0.18 * moisture_alignment
        nutrient_bonus = 0.16 * nutrient_alignment
        efficiency_bonus = 0.12 * clamp(growth / max(operational_cost + 0.05, 1e-6), 0.0, 2.0)
        state_shift = (
            abs(self._latent_state["soil_moisture"] - previous_state["soil_moisture"]) * 1.6
            + abs(self._latent_state["pest_density"] - previous_state["pest_density"]) * 1.4
            + abs(
                (
                    self._latent_state["nitrogen"]
                    + self._latent_state["phosphorus"]
                    + self._latent_state["potassium"]
                )
                / 3.0
                - (
                    previous_state["nitrogen"]
                    + previous_state["phosphorus"]
                    + previous_state["potassium"]
                )
                / 3.0
            )
        )
        stability_signal = clamp(1.0 - state_shift, 0.0, 1.0)
        stability_bonus = 0.10 * stability_signal

        if self.task.task_id == "easy":
            task_bonus = 0.18 * moisture_alignment
        elif self.task.task_id == "medium":
            task_bonus = 0.10 * nutrient_alignment + 0.08 * pest_factor
        else:
            task_bonus = 0.10 * growth + 0.06 * water_remaining

        over_irrigation_penalty = 0.14 * max(irrigation_norm - 0.55, 0.0) ** 2
        over_pesticide_penalty = 0.18 * max(action_model.pesticide - 0.35, 0.0) ** 2
        budget_penalty = 0.22 * max(1.0 - water_remaining - 0.08, 0.0) ** 2
        resource_penalty = over_irrigation_penalty + over_pesticide_penalty + budget_penalty
        pest_penalty = 0.14 * self._latent_state["pest_density"]

        reward_total = (
            growth
            + moisture_bonus
            + nutrient_bonus
            + efficiency_bonus
            + stability_bonus
            + task_bonus
            - self.task.alpha * operational_cost
            - resource_penalty
            - pest_penalty
        )

        reward = Reward(
            total=reward_total,
            crop_growth=growth,
            moisture_alignment=moisture_bonus,
            nutrient_alignment=nutrient_bonus,
            efficiency_bonus=efficiency_bonus,
            stability_bonus=stability_bonus,
            task_bonus=task_bonus,
            operational_cost=operational_cost,
            resource_penalty=resource_penalty,
            pest_penalty=pest_penalty,
        )

        self._latent_state["biomass"] += growth
        self._latent_state["water_used_liters"] += action_model.irrigation
        self.cumulative_yield += growth
        self.total_cost += operational_cost
        self.current_step += 1

        self._metrics["yield"].append(growth)
        self._metrics["cost"].append(operational_cost)
        self._metrics["moisture_error"].append(self._latent_state["soil_moisture"] - self.task.target_moisture)
        nutrient_mean = (
            self._latent_state["nitrogen"] + self._latent_state["phosphorus"] + self._latent_state["potassium"]
        ) / 3.0
        self._metrics["nutrient_error"].append(nutrient_mean - self.task.target_nutrient)
        self._metrics["pest"].append(self._latent_state["pest_density"])
        self._metrics["reward"].append(reward.total)
        self._metrics["stability"].append(stability_signal)

        done = self.current_step >= self.task.horizon
        info: dict[str, Any] = {
            "task_id": self.task.task_id,
            "difficulty": self.task.difficulty,
            "stage": stage_name,
            "stage_progress": stage_progress,
            "cumulative_yield": self.cumulative_yield,
            "total_cost": self.total_cost,
            "water_budget_remaining": water_remaining,
            "reward_breakdown": reward.to_dict(),
            "action": action_model.to_dict(),
        }
        if done:
            summary = self.episode_summary()
            grader_result = grade_episode(summary)
            info["episode_summary"] = summary.to_dict()
            info["grader"] = grader_result.to_dict()
        observation = self._observe(done=done, reward=reward, extra_metadata=info)
        return observation, reward, done, info

    def state(self) -> AgriState:
        """Return the latent environment state without observation noise."""

        if not self._latent_state:
            raise RuntimeError("Call reset() before state().")
        stage_name, stage_progress, stage_multiplier = stage_at_step(self.current_step, self.task.horizon)
        water_budget_remaining = clamp(
            1.0 - self._latent_state["water_used_liters"] / self.task.water_budget_liters,
            0.0,
            1.0,
        )
        summary = self.episode_summary().to_dict() if self.current_step > 0 else {}
        grader = grade_episode(self.episode_summary()).to_dict() if self.current_step >= self.task.horizon else {}
        return AgriState(
            episode_id=self._episode_id,
            step_count=self.current_step,
            task_id=self.task.task_id,
            difficulty=self.task.difficulty,
            task_title=self.task.title,
            horizon=self.task.horizon,
            growth_stage=stage_name,
            growth_stage_progress=stage_progress,
            growth_stage_multiplier=stage_multiplier,
            soil_moisture=self._latent_state["soil_moisture"],
            nitrogen=self._latent_state["nitrogen"],
            phosphorus=self._latent_state["phosphorus"],
            potassium=self._latent_state["potassium"],
            temperature_c=self._latent_state["temperature_c"],
            humidity=self._latent_state["humidity"],
            pest_density=self._latent_state["pest_density"],
            energy_price=self._latent_state["energy_price"],
            water_budget_remaining=water_budget_remaining,
            water_used_liters=self._latent_state["water_used_liters"],
            biomass=self._latent_state["biomass"],
            cumulative_yield=self.cumulative_yield,
            total_cost=self.total_cost,
            episode_summary=summary,
            grader=grader,
        )

    def episode_summary(self) -> EpisodeSummary:
        """Return deterministic end-of-episode metrics for grading."""

        if self._last_summary is not None and self.current_step >= self.task.horizon:
            return self._last_summary
        summary = EpisodeSummary(
            task_id=self.task.task_id,
            steps=self.current_step,
            cumulative_yield=self.cumulative_yield,
            total_cost=self.total_cost,
            average_efficiency=self.cumulative_yield / max(self.total_cost, 1e-6),
            moisture_rmse=rmse(self._metrics["moisture_error"]),
            nutrient_rmse=rmse(self._metrics["nutrient_error"]),
            mean_pest_density=mean(self._metrics["pest"]),
            stability_index=clamp(mean(self._metrics["stability"]), 0.0, 1.0),
            water_budget_remaining=clamp(
                1.0 - self._latent_state["water_used_liters"] / self.task.water_budget_liters,
                0.0,
                1.0,
            ),
            reward_mean=mean(self._metrics["reward"]),
            reward_std=stddev(self._metrics["reward"]),
        )
        if self.current_step >= self.task.horizon:
            self._last_summary = summary
        return summary

    def render(self) -> None:
        if self.render_mode != "human":
            return
        state = self.state()
        print(
            "Step {step:03d} | stage={stage} | moisture={moisture:.3f} | "
            "NPK=({n:.3f},{p:.3f},{k:.3f}) | pest={pest:.3f} | energy={energy:.2f} | yield={yield_:.2f}".format(
                step=state.step_count,
                stage=state.growth_stage,
                moisture=state.soil_moisture,
                n=state.nitrogen,
                p=state.phosphorus,
                k=state.potassium,
                pest=state.pest_density,
                energy=state.energy_price,
                yield_=state.cumulative_yield,
            )
        )

    def _observe(self, done: bool, reward: Reward | None, extra_metadata: dict[str, Any] | None) -> Observation:
        stage_name, stage_progress, _ = stage_at_step(self.current_step, self.task.horizon)
        water_budget_remaining = clamp(
            1.0 - self._latent_state["water_used_liters"] / self.task.water_budget_liters,
            0.0,
            1.0,
        )

        def noisy(name: str, noise_scale: float) -> float:
            low, high = OBSERVATION_BOUNDS[name]
            span = high - low
            value = self._latent_state[name]
            noise = self._rng.uniform(-noise_scale, noise_scale) * span
            return clamp(value + noise, low, high)

        metadata = dict(extra_metadata or {})
        if reward is not None:
            metadata.setdefault("reward_breakdown", reward.to_dict())

        return Observation(
            done=done,
            reward=None if reward is None else reward.total,
            metadata=metadata,
            task_id=self.task.task_id,
            difficulty=self.task.difficulty,
            task_title=self.task.title,
            stage_name=stage_name,
            soil_moisture=noisy("soil_moisture", self.task.sensor_noise),
            nitrogen=noisy("nitrogen", self.task.sensor_noise),
            phosphorus=noisy("phosphorus", self.task.sensor_noise),
            potassium=noisy("potassium", self.task.sensor_noise),
            temperature_c=noisy("temperature_c", self.task.sensor_noise * 0.65),
            humidity=noisy("humidity", self.task.sensor_noise * 1.2),
            pest_density=noisy("pest_density", self.task.sensor_noise),
            energy_price=noisy("energy_price", self.task.sensor_noise * 0.9),
            water_budget_remaining=water_budget_remaining,
            growth_stage_progress=stage_progress,
            cumulative_yield=self.cumulative_yield,
            total_cost=self.total_cost,
        )
