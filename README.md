---
title: AgriEnv OpenEnv Server
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /
tags:
  - openenv
  - reinforcement-learning
  - sustainability
  - agriculture
---

# AgriEnv

AgriEnv is a precision agriculture reinforcement learning environment built in the OpenEnv client/server pattern. It simulates greenhouse control under water, nutrient, pest, and energy tradeoffs, and exposes the standard OpenEnv `reset()`, `step()`, and `state()` interface over HTTP and WebSocket.

## Why this version is the final product

This repo now follows the OpenEnv course architecture instead of only shipping a local Gym-style simulator:

- `agri_env/models.py`: typed OpenEnv contracts for `Action`, `Observation`, and `AgriState`
- `agri_env/client.py`: typed `EnvClient` for training code and notebooks
- `agri_env/env.py`: core simulator, observation normalization, and SB3 wrapper
- `agri_env/gui.py`: real-time monitoring dashboard
- `server/agri_environment.py`: OpenEnv server wrapper
- `server/app.py`: FastAPI app (OpenEnv main server)
- `server/flask_app.py`: Secondary Flask REST API
- `train_rl.py`: SB3 PPO training pipeline
- `Dockerfile`: Space-ready container entrypoint
- `openenv.yaml`: OpenEnv manifest
- `pyproject.toml`: package metadata and script entry points

## Problem motivation

Greenhouse operators need to manage irrigation, NPK dosing, CO2 enrichment, and pesticide use while staying cost-efficient and stable under noisy sensing and weather drift. AgriEnv turns that real control problem into a deterministic, typed RL environment that can be trained locally or deployed as an OpenEnv microservice.

## Observation, action, and reward design

### Observation

`Observation` features are normalized to the `[0, 1]` range for optimal Reinforcement Learning performance:

- `soil_moisture`: [0, 1] (0.70 is optimal)
- `nitrogen`: [0, 1]
- `phosphorus`: [0, 1]
- `potassium`: [0, 1]
- `temperature_c`: [0, 1] (normalized from [7.5, 45.0])
- `humidity`: [0, 1] (normalized from [0.0, 100.0])
- `pest_density`: [0, 1]
- `energy_price`: [0, 1] (normalized from [1.0, 17.0])
- `water_budget_remaining`: [0, 1]
- `growth_stage_progress`: [0, 1]

It also includes task metadata and uses the inherited OpenEnv `done`, `reward`, and `metadata` fields.
Detailed reward breakdowns are available in the observation metadata.

`Action` controls:

- `irrigation`
- `nitrogen_injection`
- `phosphorus_injection`
- `potassium_injection`
- `co2_ppm`
- `pesticide`

### Reward

The shaped reward balances:

- crop growth
- moisture alignment
- nutrient alignment
- efficiency bonus
- stability bonus
- task-specific bonus
- operational cost
- overuse penalties
- pest pressure

The scalar reward is returned through the OpenEnv observation contract, and a detailed reward breakdown is stored in observation metadata.

## Tasks

- `easy`: maintain soil moisture near `0.70`
- `medium`: balance NPK and reduce pest pressure
- `hard`: full yield-cost optimization under noise, drift, and water-budget pressure

Each task has a deterministic grader in `[0, 1]`.

The grader score combines:

- cumulative yield
- efficiency (`yield / cost`)
- stability
- task-specific control quality such as moisture accuracy, nutrient balance, pest suppression, and budget retention

## OpenEnv usage

### Local server

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m server.app --port 8000
```

### Validate the environment

```bash
.venv/bin/openenv validate .
.venv/bin/openenv validate --url http://127.0.0.1:8000
python3 -m unittest discover -s tests -v
```

### Use the typed client

```python
from agri_env import Action
from agri_env.client import AgriEnvClient

with AgriEnvClient(base_url="http://127.0.0.1:8000").sync() as env:
    result = env.reset(task="easy", seed=11)
    result = env.step(
        Action(
            irrigation=120.0,
            nitrogen_injection=0.05,
            phosphorus_injection=0.04,
            potassium_injection=0.05,
            co2_ppm=500.0,
            pesticide=0.02,
        )
    )
    print(result.observation.soil_moisture, result.reward, result.done)
```

### Run the Dashboard

The integrated dashboard allows you to monitor simulation state, reward signals, and variance in real-time.

```bash
agri-gui
```

### Reinforcement Learning Training

AgriEnv is optimized for RL with stable reward signals and normalized observations. You can train a PPO agent out-of-the-box:

```bash
# Start training
train-rl

# Monitor via TensorBoard
tensorboard --logdir=logs/
```

### Secondary Flask API

For standard web integrations that don't require the full OpenEnv FastAPI spec, a lightweight Flask API is available:

```bash
flask-api
```

### Run the baseline locally

```bash
python3 inference.py --policy baseline
```

`--policy baseline` is the deterministic local heuristic. It is useful for smoke tests and reproducible offline scoring.

### Run the baseline against a running OpenEnv server

```bash
python3 inference.py --task hard --policy baseline --base-url http://127.0.0.1:8000
```

### Run the LLM controller

```bash
export API_KEY=your_proxy_key
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
python3 inference.py --task hard --policy llm --base-url http://127.0.0.1:8000
```

The inference runner reads:

- `API_BASE_URL` with a default of `https://router.huggingface.co/v1`
- `MODEL_NAME` with a default of `meta-llama/Llama-3.1-8B-Instruct`
- `API_KEY` with no default

For compatibility with common local setups, it also accepts `HF_TOKEN` or `OPENAI_API_KEY` as fallbacks, but the validator-facing path uses `API_KEY`.

By default, `inference.py` runs the LLM controller across all three tasks and emits one `[START] ... [STEP] ... [END]` block per task. This default path is the one intended for hackathon evaluation because it routes requests through the injected LiteLLM proxy.

## Deployment

### Docker

```bash
docker build -t agri-env .
docker run --rm -p 8000:8000 agri-env
```

### Hugging Face Spaces

```bash
.venv/bin/openenv push --repo-id <your-username>/agri-env --exclude push-exclude.txt
```

If you keep your virtual environment inside the repo, use `--exclude push-exclude.txt`. The current OpenEnv push command does not automatically reuse `.gitignore` patterns during staging, so excluding `.venv/` explicitly avoids large uploads and 504 errors.

After deployment:

- app UI: `https://<your-username>-agri-env.hf.space/web`
- docs: `https://<your-username>-agri-env.hf.space/docs`
- health: `https://<your-username>-agri-env.hf.space/health`
- websocket endpoint: `wss://<your-username>-agri-env.hf.space/ws`

## Example baseline results

Default seeded baseline scores:

- `easy`: `0.8470`
- `medium`: `0.8084`
- `hard`: `0.8425`

Example final log line format:

```text
[END] success=true steps=140 score=0.843 rewards=0.64,0.68,0.70,...
```

## Notes

- The environment is deterministic for a fixed task and seed.
- The server supports concurrent sessions.
- The Space container enables the OpenEnv web interface through `ENABLE_WEB_INTERFACE=true`.
