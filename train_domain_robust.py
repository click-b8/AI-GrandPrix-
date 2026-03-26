#!/usr/bin/env python3
"""
Train robust models with aggressive domain randomization.

Fine-tunes existing HPC models with increased domain randomization to improve
hardware robustness and sim-to-real transfer without real drone access.

Aggressive parameters:
- Mass: ±25%
- Drag: ±30%
- Motor latency: +10ms
- Observation noise: camera blur, latency
- Random initial conditions

Run with: python3 train_domain_robust.py
"""

import os
import json
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from drone_race_env import DroneRaceEnv

OUTPUT_DIR = "./trained_domain_robust"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Models to fine-tune
MODELS_TO_FINETUNE = [
    ("aigp_racer_final.zip", False, "Swift Racer (domain robust)"),
    ("aigp_distill_final.zip", True, "Vision Policy (domain robust)"),
]

print("=" * 70)
print("DOMAIN RANDOMIZATION FINE-TUNING")
print("=" * 70)
print("\nObjective: Make models robust to hardware variations")
print("Strategy: Aggressive domain randomization without real drone data")
print("\nParameters:")
print("  Mass variation: ±25%")
print("  Drag variation: ±30%")
print("  Motor latency: +10ms")
print("  Difficulty: 1.0 (full Swift challenge)")
print("  Training steps: 10M (fine-tuning)")
print("\n" + "=" * 70 + "\n")

for model_name, vision_mode, description in MODELS_TO_FINETUNE:
    print(f"🔄 Fine-tuning: {description}")
    print(f"   Source: {model_name}")
    print(f"   Vision mode: {vision_mode}")
    print("-" * 70)

    try:
        # Create environment with aggressive domain randomization
        env = DroneRaceEnv(
            render_mode=None,
            # Aggressive domain randomization
            domain_rand=True,
            domain_rand_params={
                "mass_scale": (0.75, 1.25),      # ±25%
                "drag_scale": (0.70, 1.30),      # ±30%
                "motor_latency": (0.01, 0.01),   # 10ms added
                "obs_noise_scale": 0.05,         # 5% observation noise
                "init_pos_noise": 0.1,           # Random start variations
                "init_yaw_noise": 0.3,           # Random starting orientation
            },
            difficulty=1.0,  # Full difficulty
            vision_mode=vision_mode,
        )

        # Load base model from HPC
        model = PPO.load(f"../{model_name}", env=env)

        # Set up aggressive training parameters
        model.learning_rate = 1e-4  # Lower LR for fine-tuning
        model.ent_coef = 0.001  # More exploration
        model.n_steps = 2048

        # Create model save directory
        model_save_dir = os.path.join(OUTPUT_DIR, model_name.replace(".zip", ""))
        os.makedirs(model_save_dir, exist_ok=True)

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=100000,  # Save every 100k steps
            save_path=model_save_dir,
            name_prefix="model",
        )

        print("\n   Training with aggressive domain randomization...")
        print("   (This may take 30-60 minutes)")

        # Fine-tune for 10M steps with domain randomization
        model.learn(
            total_timesteps=10_000_000,
            callback=checkpoint_callback,
            progress_bar=True,
        )

        # Save final model
        final_path = os.path.join(model_save_dir, "final_model.zip")
        model.save(final_path)

        print(f"\n   ✅ Training complete!")
        print(f"   Final model: {final_path}")

        # Save config for reference
        config = {
            "source_model": model_name,
            "description": description,
            "vision_mode": vision_mode,
            "training_steps": 10_000_000,
            "domain_randomization": {
                "mass_scale": "0.75-1.25",
                "drag_scale": "0.70-1.30",
                "motor_latency_ms": 10,
                "obs_noise_scale": 0.05,
                "init_pos_noise": 0.1,
                "init_yaw_noise": 0.3,
            },
            "learning_rate": 1e-4,
            "ent_coef": 0.001,
        }

        config_path = os.path.join(model_save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"   Config saved: {config_path}")

        env.close()

    except FileNotFoundError as e:
        print(f"   ❌ Model not found: {model_name}")
        print(f"      Error: {e}")
    except Exception as e:
        print(f"   ❌ Training error: {e}")
        import traceback
        traceback.print_exc()

    print()

print("=" * 70)
print("✅ DOMAIN RANDOMIZATION TRAINING COMPLETE")
print("=" * 70)
print(f"\nModels saved to: {OUTPUT_DIR}/")
print("\nNext steps:")
print("1. Benchmark these domain-robust models locally")
print("2. Compare vs original HPC models")
print("3. Use best variant for May qualifier submission")
print("\n")
