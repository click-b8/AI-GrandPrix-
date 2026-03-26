#!/usr/bin/env python3
"""
Comprehensive HPC benchmarking for May qualifier.

Runs all 3 trained models 20x each to establish baseline performance:
- Lap completion rate (8/8 gates)
- Average lap time
- Consistency and reliability
- Failure modes and weak gates

This script is designed for SLURM submission.
Run with: python3 benchmark_hpc_final.py
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

try:
    from stable_baselines3 import PPO
    from drone_race_env import DroneRaceEnv
    import numpy as np
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install stable-baselines3 gymnasium mujoco")
    sys.exit(1)

# Configuration
MODELS = [
    ("aigp_8gates_final.zip", False, "8-Gate Racer (state-based)"),
    ("aigp_racer_final.zip", False, "Swift Racer 100M (state-based)"),
    ("aigp_distill_final.zip", True, "Vision Policy (vision-based)"),
]

LAPS_PER_MODEL = 20
OUTPUT_DIR = "./benchmark_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS = {
    "timestamp": datetime.now().isoformat(),
    "laps_per_model": LAPS_PER_MODEL,
    "models": {},
}

print("=" * 80)
print("AI GRAND PRIX - HPC MODEL BENCHMARKING")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Laps per model: {LAPS_PER_MODEL}")
print(f"  Models to test: {len(MODELS)}")
print(f"  Output directory: {OUTPUT_DIR}")
print("\n" + "=" * 80 + "\n")

for model_path, vision_mode, model_name in MODELS:
    print(f"📊 TESTING: {model_name}")
    print(f"   File: {model_path}")
    print(f"   Vision Mode: {vision_mode}")
    print("-" * 80)

    model_results = {
        "name": model_name,
        "path": model_path,
        "vision_mode": vision_mode,
        "laps": [],
        "summary": {},
    }

    try:
        # Load model
        if not os.path.exists(f"../{model_path}"):
            print(f"   ❌ Model not found: ../{model_path}")
            RESULTS["models"][model_name] = model_results
            continue

        model = PPO.load(f"../{model_path}")
        env = DroneRaceEnv(
            render_mode=None,
            domain_rand=False,
            difficulty=1.0,
            vision_mode=vision_mode,
        )

        gates_all = []
        times_all = []
        steps_all = []
        failed_gates = []  # Track which gates are problematic

        for lap_num in range(1, LAPS_PER_MODEL + 1):
            obs, _ = env.reset()
            done = False
            steps = 0
            start_time = time.time()
            gates = 0

            while not done and steps < 5000:  # Safety limit
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, trunc, info = env.step(action)
                gates = info.get("gates_passed", 0)
                steps += 1

                if done or trunc:
                    break

            elapsed = time.time() - start_time

            # Record metrics
            gates_all.append(gates)
            times_all.append(elapsed)
            steps_all.append(steps)

            # Track failures
            if gates < 8:
                failed_gates.append(gates)

            # Status output
            status = "✓" if gates == 8 else "✗"
            print(f"   Lap {lap_num:2d}: {status} {gates}/8 gates | {elapsed:6.2f}s | {steps:4d} steps")

            # Save lap data
            model_results["laps"].append({
                "lap_num": lap_num,
                "gates": gates,
                "time": elapsed,
                "steps": steps,
                "success": gates == 8,
            })

        env.close()

        # Calculate summary statistics
        avg_gates = np.mean(gates_all)
        std_gates = np.std(gates_all)
        avg_time = np.mean(times_all)
        std_time = np.std(times_all)
        success_rate = (sum(1 for g in gates_all if g == 8) / len(gates_all)) * 100
        min_time = np.min(times_all)
        max_time = np.max(times_all)

        model_results["summary"] = {
            "avg_gates": float(avg_gates),
            "std_gates": float(std_gates),
            "success_rate_percent": float(success_rate),
            "avg_time_seconds": float(avg_time),
            "std_time_seconds": float(std_time),
            "min_time_seconds": float(min_time),
            "max_time_seconds": float(max_time),
            "total_laps": LAPS_PER_MODEL,
            "failed_gates_count": len(failed_gates),
        }

        # Print summary
        print(f"\n   SUMMARY:")
        print(f"   ├─ Success rate: {success_rate:.1f}% ({sum(1 for g in gates_all if g == 8)}/{LAPS_PER_MODEL})")
        print(f"   ├─ Avg gates: {avg_gates:.2f} ± {std_gates:.2f}/8")
        print(f"   ├─ Avg lap time: {avg_time:.2f}s ± {std_time:.2f}s")
        print(f"   ├─ Time range: {min_time:.2f}s - {max_time:.2f}s")
        print(f"   └─ Failed gates: {len(failed_gates)}")

    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        model_results["error"] = str(e)

    RESULTS["models"][model_name] = model_results
    print()

# Save results to JSON
output_file = os.path.join(OUTPUT_DIR, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
with open(output_file, "w") as f:
    json.dump(RESULTS, f, indent=2)

print("=" * 80)
print("✅ BENCHMARKING COMPLETE")
print("=" * 80)
print(f"\nResults saved to: {output_file}")

# Print ranking
print("\n" + "=" * 80)
print("FINAL RANKING (by success rate)")
print("=" * 80 + "\n")

ranked = sorted(
    [
        (name, data)
        for name, data in RESULTS["models"].items()
        if "summary" in data
    ],
    key=lambda x: x[1]["summary"]["success_rate_percent"],
    reverse=True,
)

for rank, (name, data) in enumerate(ranked, 1):
    summary = data["summary"]
    print(f"{rank}. {name}")
    print(f"   ├─ Success: {summary['success_rate_percent']:.1f}% ({int(summary['success_rate_percent']/100*LAPS_PER_MODEL)}/{LAPS_PER_MODEL})")
    print(f"   ├─ Avg gates: {summary['avg_gates']:.2f}/8")
    print(f"   └─ Avg time: {summary['avg_time_seconds']:.2f}s\n")

print("=" * 80)
print("RECOMMENDATION FOR MAY QUALIFIER")
print("=" * 80)
if ranked:
    best_name, best_data = ranked[0]
    print(f"\n✅ Primary model: {best_name}")
    print(f"   Success rate: {best_data['summary']['success_rate_percent']:.1f}%")
    print(f"\nNext steps:")
    print("  1. If success rate >= 95%: Use this model as-is")
    print("  2. If 85-95%: Consider fine-tuning with domain randomization")
    print("  3. If < 85%: Investigate failure modes, may need retraining")
    print(f"\nFor backup strategy, test #2 rank model as fallback")

print("\n")
