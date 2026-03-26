#!/usr/bin/env python3
"""
Measure inference latency for all models on current hardware.

Tests inference time (model.predict) for each model to ensure <50ms
requirement is met. Samples 100+ inferences per model.

Run with: python3 measure_inference_time.py
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

try:
    from stable_baselines3 import PPO
    from drone_race_env import DroneRaceEnv
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    sys.exit(1)

MODELS = [
    ("aigp_8gates_final.zip", False, "8-Gate Racer"),
    ("aigp_racer_final.zip", False, "Swift Racer 100M"),
    ("aigp_distill_final.zip", True, "Vision Policy"),
]

INFERENCE_SAMPLES = 200  # Number of inference calls to measure
WARMUP_CALLS = 10  # Warmup to stabilize

print("=" * 70)
print("INFERENCE LATENCY MEASUREMENT")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Warmup calls: {WARMUP_CALLS}")
print(f"  Measurement samples: {INFERENCE_SAMPLES}")
print(f"  Target latency: <50ms per decision")
print("\n" + "=" * 70 + "\n")

results = {}

for model_path, vision_mode, model_name in MODELS:
    print(f"📊 {model_name}")
    print(f"   Model: {model_path}")
    print(f"   Vision: {vision_mode}")
    print("-" * 70)

    try:
        if not os.path.exists(f"../{model_path}"):
            print(f"   ❌ Model not found: ../{model_path}\n")
            continue

        model = PPO.load(f"../{model_path}")
        env = DroneRaceEnv(
            render_mode=None,
            domain_rand=False,
            difficulty=1.0,
            vision_mode=vision_mode,
        )

        obs, _ = env.reset()

        # Warmup
        print(f"   Warming up ({WARMUP_CALLS} calls)...", end=" ", flush=True)
        for _ in range(WARMUP_CALLS):
            action, _ = model.predict(obs, deterministic=True)
        print("✓")

        # Measure inference time
        print(f"   Measuring latency ({INFERENCE_SAMPLES} samples)...", end=" ", flush=True)
        latencies = []

        for _ in range(INFERENCE_SAMPLES):
            start = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            latencies.append(elapsed)

            # Step environment for next observation
            obs, _, done, trunc, _ = env.step(action)
            if done or trunc:
                obs, _ = env.reset()

        print("✓")
        env.close()

        # Calculate statistics
        latencies = np.array(latencies)
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        max_latency = np.max(latencies)
        min_latency = np.min(latencies)

        # Check if meets requirement
        meets_requirement = mean_latency < 50
        status = "✓ PASS" if meets_requirement else "✗ FAIL"

        results[model_name] = {
            "mean": mean_latency,
            "std": std_latency,
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "min": min_latency,
            "max": max_latency,
            "meets_requirement": meets_requirement,
        }

        # Print results
        print(f"\n   LATENCY METRICS:")
        print(f"   ├─ Mean:  {mean_latency:7.2f}ms ± {std_latency:6.2f}ms {status}")
        print(f"   ├─ P50:   {p50:7.2f}ms (median)")
        print(f"   ├─ P95:   {p95:7.2f}ms (95th percentile)")
        print(f"   ├─ P99:   {p99:7.2f}ms (99th percentile)")
        print(f"   └─ Range: {min_latency:7.2f}ms - {max_latency:7.2f}ms\n")

    except Exception as e:
        print(f"   ❌ ERROR: {e}\n")

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70 + "\n")

if results:
    for model_name, metrics in results.items():
        status = "✓" if metrics["meets_requirement"] else "✗"
        print(f"{status} {model_name:30s}: {metrics['mean']:6.2f}ms (p95: {metrics['p95']:6.2f}ms)")

    print("\n" + "=" * 70)
    all_pass = all(m["meets_requirement"] for m in results.values())
    if all_pass:
        print("✅ All models meet <50ms requirement")
    else:
        print("⚠️  Some models exceed 50ms requirement")
        print("   Consider: quantization, distillation, or hardware optimization")
else:
    print("❌ No successful measurements")

print()
