#!/usr/bin/env python3
"""Benchmark all HPC-trained models locally."""
from stable_baselines3 import PPO
from drone_race_env import DroneRaceEnv
import numpy as np
import time

models = [
    ("aigp_8gates_final.zip", False, "8-Gate Racer"),
    ("aigp_racer_final.zip", False, "Swift Racer (100M)"),
    ("aigp_distill_final.zip", True, "Vision Policy"),
]

print("=" * 70)
print("BENCHMARKING HPC MODELS")
print("=" * 70)

results = {}

for model_path, vision_mode, name in models:
    print(f"\n📊 {name}")
    print(f"   Model: {model_path}")
    print(f"   Vision: {vision_mode}")
    print("-" * 70)

    try:
        model = PPO.load(f"../{model_path}")
        env = DroneRaceEnv(
            render_mode=None,
            domain_rand=False,
            difficulty=1.0,
            vision_mode=vision_mode
        )

        gates_passed = []
        times = []

        for lap in range(5):
            obs, _ = env.reset()
            done = False
            steps = 0
            start = time.time()

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, trunc, info = env.step(action)
                gates = info.get('gates_passed', 0)
                steps += 1
                if done or trunc:
                    break

            elapsed = time.time() - start
            gates_passed.append(gates)
            times.append(elapsed)
            print(f"   Lap {lap+1}: {gates}/8 gates | {elapsed:.1f}s | {steps} steps")

        env.close()

        # Summary stats
        avg_gates = np.mean(gates_passed)
        avg_time = np.mean(times)
        consistency = (sum(1 for g in gates_passed if g == 8) / len(gates_passed)) * 100

        results[name] = {
            "avg_gates": avg_gates,
            "avg_time": avg_time,
            "consistency": consistency,
            "gates_per_lap": gates_passed,
        }

        print(f"\n   SUMMARY:")
        print(f"   • Avg gates: {avg_gates:.1f}/8")
        print(f"   • Avg time: {avg_time:.1f}s per lap")
        print(f"   • Consistency (8/8): {consistency:.0f}%")

    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("RANKING")
print("=" * 70)

ranked = sorted(results.items(), key=lambda x: x[1]["consistency"], reverse=True)
for i, (name, stats) in enumerate(ranked, 1):
    print(f"{i}. {name}")
    print(f"   Consistency: {stats['consistency']:.0f}% | Avg gates: {stats['avg_gates']:.1f} | Time: {stats['avg_time']:.1f}s")

print("\n✅ Benchmark complete!")
