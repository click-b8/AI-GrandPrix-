#!/usr/bin/env python3
"""Generate training data from vision expert (imitation learning).

Runs the aigp_distill_final.zip model and saves observations + actions
for training a smaller/faster student model via behavioral cloning.
"""
from stable_baselines3 import PPO
from drone_race_env import DroneRaceEnv
import numpy as np
import pickle
import os

OUTPUT_DIR = "./expert_trajectories"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("GENERATING EXPERT TRAJECTORIES")
print("Collecting data from: aigp_distill_final.zip (vision model)")
print("=" * 70)

model = PPO.load("../aigp_distill_final.zip")
env = DroneRaceEnv(
    render_mode=None,
    domain_rand=False,
    difficulty=1.0,
    vision_mode=True
)

trajectories = []
total_steps = 0
successful_laps = 0

print("\nCollecting expert demonstrations...")

for lap in range(20):  # 20 successful laps
    obs, _ = env.reset()
    done = False
    steps = 0
    lap_data = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs_next, reward, done, trunc, info = env.step(action)

        # Store transition
        lap_data.append({
            "observation": obs,
            "action": action,
            "observation_next": obs_next,
            "reward": reward,
            "gates_passed": info.get("gates_passed", 0),
        })

        obs = obs_next
        steps += 1
        total_steps += 1

        if done or trunc:
            break

    # Only keep successful laps (8/8 gates)
    if info.get("gates_passed", 0) == 8:
        trajectories.append(lap_data)
        successful_laps += 1
        print(f"  ✓ Lap {lap+1}: {len(lap_data)} steps (SUCCESS)")
    else:
        print(f"  ✗ Lap {lap+1}: {len(lap_data)} steps (FAILED - {info.get('gates_passed', 0)}/8)")

    if successful_laps >= 10:
        break

env.close()

# Save trajectories
output_path = os.path.join(OUTPUT_DIR, "expert_data.pkl")
with open(output_path, "wb") as f:
    pickle.dump(trajectories, f)

print("\n" + "=" * 70)
print(f"✅ DATA COLLECTION COMPLETE")
print(f"  • Successful trajectories: {successful_laps}")
print(f"  • Total steps collected: {total_steps}")
print(f"  • Saved to: {output_path}")
print("=" * 70)

print("\nNext steps:")
print("1. Use this data for behavioral cloning:")
print("   python3 train_from_expert.py")
print("2. Or use for DAgger (expert + model learning):")
print("   python3 train_with_dagger.py")
