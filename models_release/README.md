# AI Grand Prix Trained Models

## Models

### aigp_8gates_final.zip (504 KB)
**8-Gate Racer** — State-based specialist for aggressive 8-gate racing.
- Type: State-based (privileged 24D observation)
- Training: 100M steps on HPC
- Best for: High-performance gate racing
- Usage: `vision_mode=False`

### aigp_racer_final.zip (504 KB)
**Swift Racer (100M Extended)** — Championship-level state-based racer.
- Type: State-based
- Training: 100M steps (extended Swift baseline)
- Best for: Maximum performance benchmarking
- Usage: `vision_mode=False`

### aigp_distill_final.zip (13 MB)
**Vision Policy** — CNN-based policy with FPV + event camera.
- Type: Vision-based (48×48 RGB + 14-channel event histogram)
- Training: 52M steps via distillation from state experts
- Best for: Sim-to-real transfer, actual drone deployment
- Usage: `vision_mode=True`

## Quick Start

```python
from stable_baselines3 import PPO
from drone_race_env import DroneRaceEnv

# Load state-based model
model = PPO.load('aigp_8gates_final.zip')
env = DroneRaceEnv(render_mode=None, vision_mode=False)

# Or load vision model
model = PPO.load('aigp_distill_final.zip')
env = DroneRaceEnv(render_mode=None, vision_mode=True)

# Run
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, trunc, info = env.step(action)
    if done or trunc:
        break
```

## Sim-to-Real Transfer

**Recommended approach:**
1. Use `aigp_distill_final.zip` (vision-based)
2. Fine-tune on real drone with domain randomization
3. Match camera parameters (FOV, resolution, tilt)
4. Validate gate detection accuracy

## Performance

- **aigp_8gates_final**: Completes 8/8 gates consistently
- **aigp_racer_final**: Championship-level performance
- **aigp_distill_final**: ~50M+ steps of vision training, proven on HPC

## Citation

If using these models in research:
```
@misc{aigrandprix2026,
  title={Autonomous Drone Racing: AI Grand Prix Competition Models},
  author={[Your Team]},
  year={2026},
  url={https://github.com/click-b8/AI-GrandPrix}
}
```
