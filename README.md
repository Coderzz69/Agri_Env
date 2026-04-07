# AgriEnv: Precision Agriculture RL Environment

AgriEnv is a biologically inspired and economically aware OpenAI Gymnasium-compatible reinforcement learning environment designed for precision agriculture in greenhouse systems.

## Objective

The goal for an RL agent in this environment is to control key greenhouse parameters—irrigation, nutrient injection, CO2 levels, and pesticide usage—to maximize crop growth while minimizing operational costs and environmental impact.

## Features

- **Gymnasium Compatible**: Follows the standard `reset()` and `step()` API.
- **Biologically Inspired Dynamics**: 
  - Soil moisture increases with irrigation and decreases via temperature-dependent evaporation.
  - Nutrients (N, P, K) increase with injection and face natural decay.
  - Pest density increases in high humidity and decreases with pesticide application.
- **Economic Awareness**: Includes variable energy prices and material costs (water, CO2).
- **Stochasticity**: Small random fluctuations in weather (temperature/humidity) and energy prices to provide a more robust training challenge.
- **Shaping Penalties**: Discourages excessive resource usage (e.g., over-irrigation or heavy pesticide use).

## Environment Specifications

### Observation Space (8-dim Continuous Box)
1. `soil_moisture`: (0.0 to 1.0)
2. `nitrogen`: (0.0 to 1.0)
3. `phosphorus`: (0.0 to 1.0)
4. `potassium`: (0.0 to 1.0)
5. `temperature`: (10.0 to 50.0 °C)
6. `humidity`: (0.0 to 100.0 %)
7. `pest_density`: (0.0 to 1.0)
8. `energy_price`: (1.0 to 10.0)

### Action Space (6-dim Continuous Box)
1. `irrigation`: (0.0 to 500.0 ml/hr)
2. `nitrogen_injection`: (0.0 to 0.5)
3. `phosphorus_injection`: (0.0 to 0.5)
4. `potassium_injection`: (0.0 to 0.5)
5. `co2_level`: (300.0 to 800.0 ppm)
6. `pesticide_usage`: (0.0 to 1.0)

### Reward Function
The reward represents the net profit:
`reward = growth - (alpha * cost) - penalties`
- **Growth**: High when soil moisture is near optimal (0.6), nutrients are high, and pests are low.
- **Cost**: Derived from water consumption and CO2 enrichment.
- **Penalties**: Small penalties for excessive irrigation (>400) or pesticide usage (>0.8).

## Getting Started

### Prerequisites
Ensure you have Python 3.8+ and `pip` installed.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Coderzz69/Agri_Env.git
   cd Agri_Env
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install numpy gymnasium
   ```

### Usage
Run the built-in random agent demo:
```bash
python agri_env.py
```

## Example Code
```python
import gymnasium as gym
from agri_env import AgriEnv

env = AgriEnv(render_mode="human")
obs, info = env.reset()

for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```
