#!/usr/bin/env python3
"""Test trained models on HPC."""
from stable_baselines3 import PPO
from drone_race_env import DroneRaceEnv
import time

print("=" * 60)
print("TESTING HPC TRAINED MODELS")
print("=" * 60)

# Test 1: 8-gates
print("\n✓ Testing aigp_8gates_final.zip...")
model = PPO.load('../aigp_8gates_final.zip')
env = DroneRaceEnv(render_mode=None, domain_rand=False, difficulty=1.0, vision_mode=False)

for lap in range(3):
    obs, _ = env.reset()
    done = False
    steps = 0
    gates = 0
    start = time.time()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        gates = info.get('gates_passed', 0)
        steps += 1
        if done or trunc:
            break
    elapsed = time.time() - start
    print(f"  Lap {lap+1}: {gates}/8 gates | {steps} steps | {elapsed:.1f}s")
env.close()

# Test 2: racer
print("\n✓ Testing aigp_racer_final.zip...")
model = PPO.load('../aigp_racer_final.zip')
env = DroneRaceEnv(render_mode=None, domain_rand=False, difficulty=1.0, vision_mode=False)

for lap in range(3):
    obs, _ = env.reset()
    done = False
    steps = 0
    gates = 0
    start = time.time()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        gates = info.get('gates_passed', 0)
        steps += 1
        if done or trunc:
            break
    elapsed = time.time() - start
    print(f"  Lap {lap+1}: {gates}/8 gates | {steps} steps | {elapsed:.1f}s")
env.close()

print("\n" + "=" * 60)
print("TESTS COMPLETE ✅")
print("=" * 60)
