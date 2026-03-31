#!/usr/bin/env python3
"""
Benchmark locally-trained models on current hardware.

Tests all available locally-trained models to find best performer
for May qualifier. Uses architecture-compatible models only.

Run with: python3 benchmark_local_models.py
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

try:
    from stable_baselines3 import PPO
    from drone_race_env import DroneRaceEnv
    import numpy as np
except ImportError as e:
    print(f"ERROR: {e}")
    exit(1)

# Locally-trained models we can actually load
MODELS = [
    ("trained_swift_100m/best_model/best_model.zip", False, "Swift 100M (state-based)"),
    ("trained_fast_safe/best_model/best_model.zip", True, "Fast-Safe (vision-based)"),
    ("trained_vision_events/best_model/best_model.zip", True, "Vision+Events"),
    ("trained_allgates/best_model/best_model.zip", False, "All Gates (state)"),
]

LAPS_PER_MODEL = 10  # Local testing - fewer laps
OUTPUT_DIR = "./benchmark_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS = {
    "timestamp": datetime.now().isoformat(),
    "test_type": "local_models",
    "laps_per_model": LAPS_PER_MODEL,
    "models": {},
}

print("=" * 80)
print("LOCAL MODEL BENCHMARKING FOR MAY QUALIFIER")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print(f"\nTesting locally-trained models ({LAPS_PER_MODEL} laps each)")
print("(HPC models have architecture incompatibilities)\n")

for model_path, vision_mode, model_name in MODELS:
    print(f"📊 {model_name}")
    print(f"   Path: {model_path}")
    print(f"   Vision: {vision_mode}")
    print("-" * 80)

    model_results = {
        "name": model_name,
        "path": model_path,
        "vision_mode": vision_mode,
        "laps": [],
        "summary": {},
    }

    try:
        if not os.path.exists(model_path):
            print(f"   ❌ Not found: {model_path}\n")
            RESULTS["models"][model_name] = model_results
            continue

        model = PPO.load(model_path)
        env = DroneRaceEnv(
            render_mode=None,
            domain_rand=False,
            difficulty=1.0,
            vision_mode=vision_mode,
        )

        gates_all = []
        times_all = []
        steps_all = []

        for lap_num in range(1, LAPS_PER_MODEL + 1):
            obs, _ = env.reset()
            done = False
            steps = 0
            start_time = time.time()
            gates = 0

            while not done and steps < 5000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, trunc, info = env.step(action)
                gates = info.get("gates_passed", 0)
                steps += 1
                if done or trunc:
                    break

            elapsed = time.time() - start_time
            gates_all.append(gates)
            times_all.append(elapsed)
            steps_all.append(steps)

            status = "✓" if gates == 8 else "✗"
            print(f"   Lap {lap_num:2d}: {status} {gates}/8 gates | {elapsed:6.2f}s | {steps:4d} steps")

            model_results["laps"].append({
                "lap_num": lap_num,
                "gates": gates,
                "time": elapsed,
                "steps": steps,
                "success": gates == 8,
            })

        env.close()

        # Summary stats
        avg_gates = np.mean(gates_all)
        std_gates = np.std(gates_all)
        avg_time = np.mean(times_all)
        std_time = np.std(times_all)
        success_rate = (sum(1 for g in gates_all if g == 8) / len(gates_all)) * 100

        model_results["summary"] = {
            "avg_gates": float(avg_gates),
            "std_gates": float(std_gates),
            "success_rate_percent": float(success_rate),
            "avg_time_seconds": float(avg_time),
            "std_time_seconds": float(std_time),
            "min_time_seconds": float(np.min(times_all)),
            "max_time_seconds": float(np.max(times_all)),
        }

        print(f"\n   SUMMARY:")
        print(f"   ├─ Success: {success_rate:.1f}% ({sum(1 for g in gates_all if g == 8)}/{LAPS_PER_MODEL})")
        print(f"   ├─ Avg gates: {avg_gates:.2f} ± {std_gates:.2f}/8")
        print(f"   └─ Avg time: {avg_time:.2f}s\n")

    except Exception as e:
        print(f"   ❌ ERROR: {e}\n")
        model_results["error"] = str(e)

    RESULTS["models"][model_name] = model_results

# Save results
output_file = os.path.join(OUTPUT_DIR, f"local_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
with open(output_file, "w") as f:
    json.dump(RESULTS, f, indent=2)

print("=" * 80)
print("✅ BENCHMARKING COMPLETE")
print("=" * 80)
print(f"Results saved: {output_file}\n")

# Ranking
print("RANKING BY SUCCESS RATE:")
print("=" * 80)
ranked = sorted(
    [
        (name, data) for name, data in RESULTS["models"].items()
        if "summary" in data
    ],
    key=lambda x: x[1]["summary"]["success_rate_percent"],
    reverse=True,
)

for i, (name, data) in enumerate(ranked, 1):
    s = data["summary"]
    print(f"{i}. {name:35s} Success: {s['success_rate_percent']:5.1f}% | Time: {s['avg_time_seconds']:6.2f}s")

print()
