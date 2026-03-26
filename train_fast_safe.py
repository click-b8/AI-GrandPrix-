#!/usr/bin/env python3
"""Train fast & safe state-based expert (speed + safety + completeness focus).

Optimizes for:
- Speed: Bonus for completing gates quickly, time penalty encourages aggressive flying
- Safety: High crash penalties, smooth control, body rate limits
- Completeness: Extra reward for finishing all 8 gates

Uses state observation (privileged info) for rapid convergence, then can be
distilled to vision policy later.
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

from drone_race_env import DroneRaceEnv
from config import EPISODE_LENGTH_SEC, CTRL_FREQ

N_ENVS = 8
TOTAL_TIMESTEPS = 100_000_000
SAVE_DIR = './trained_fast_safe'

# Curriculum: ramp difficulty as agent improves
CURRICULUM = [
    (0.0, 0.2),    # start very easy
    (0.15, 0.5),   # ramp up
    (0.35, 0.8),   # harder
    (0.60, 1.0),   # full difficulty
    (1.0, 1.0),    # stay at full
]

def get_difficulty(progress: float) -> float:
    """Map training progress (0-1) to difficulty (0-1)."""
    for i in range(len(CURRICULUM) - 1):
        t0, d0 = CURRICULUM[i]
        t1, d1 = CURRICULUM[i + 1]
        if t0 <= progress <= t1:
            alpha = (progress - t0) / (t1 - t0) if t1 > t0 else 1.0
            return d0 + alpha * (d1 - d0)
    return CURRICULUM[-1][1]


class CurriculumCallback:
    """Callback to update environment difficulty during training."""
    def __init__(self, env):
        self.env = env
        self.last_progress = 0.0

    def __call__(self, locals_dict):
        """Called by on_step callback."""
        num_timesteps = locals_dict.get('self').num_timesteps
        total_ts = TOTAL_TIMESTEPS
        progress = min(1.0, num_timesteps / total_ts)

        difficulty = get_difficulty(progress)

        # Update all envs in the vectorized env
        if hasattr(self.env, 'envs'):
            for env in self.env.envs:
                env.set_difficulty(difficulty)
        else:
            self.env.set_difficulty(difficulty)

        # Log progress periodically
        if progress - self.last_progress >= 0.1:
            print(f"Progress: {progress*100:.0f}% | Difficulty: {difficulty:.2f}")
            self.last_progress = progress


def make_train_env():
    """Create a single training environment."""
    return DroneRaceEnv(
        render_mode=None,
        domain_rand=True,
        difficulty=0.2,  # start easy
        vision_mode=True,  # VISION MODE: FPV camera + event camera input
    )


def make_eval_env():
    """Create a single evaluation environment (deterministic)."""
    return DroneRaceEnv(
        render_mode=None,
        domain_rand=False,
        difficulty=1.0,  # always test at full difficulty
        vision_mode=True,  # VISION MODE: FPV camera + event camera input
    )


def lr_schedule(progress_remaining):
    """Learning rate schedule: decay from 3e-4 to 1e-5."""
    return 3e-4 * progress_remaining


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 70)
    print("TRAINING FAST & SAFE VISION EXPERT")
    print("=" * 70)
    print(f"Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Envs: {N_ENVS} parallel")
    print(f"Input: FPV camera (48x48 RGB) + Event camera")
    print(f"Focus: Speed + Safety + Completeness")
    print("=" * 70)

    # Create vectorized environments
    vec_env = make_vec_env(make_train_env, n_envs=N_ENVS)
    eval_env = make_vec_env(make_eval_env, n_envs=1)

    # Load model if resuming, else create new
    model_path = os.path.join(SAVE_DIR, 'best_model', 'best_model.zip')
    if os.path.exists(model_path):
        print(f"\nResuming from {model_path}...")
        model = PPO.load(model_path, env=vec_env)
    else:
        print("\nCreating new PPO model...")
        model = PPO(
            'MultiInputPolicy',  # Dict observation space (image + state)
            vec_env,
            learning_rate=lr_schedule,
            n_steps=2048,  # collect 2048 steps per update (8 envs x 256 steps)
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.001,  # low entropy (focused policy, not exploratory)
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
        )

    # Slightly tune for speed/safety focus
    model.ent_coef = 0.0005  # very low entropy = deterministic fast flying
    model.learning_rate = lr_schedule

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,  # save every 100k steps
        save_path=os.path.join(SAVE_DIR, 'checkpoints'),
        name_prefix='aigp_fast_safe',
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVE_DIR, 'best_model'),
        log_path=os.path.join(SAVE_DIR, 'eval_logs'),
        eval_freq=50_000,
        n_eval_episodes=10,
        deterministic=True,
    )

    # Curriculum callback
    curriculum = CurriculumCallback(vec_env)

    print(f"\nTraining for {TOTAL_TIMESTEPS:,} timesteps...")
    print("Curriculum: easy → full difficulty over 60% of training\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
        log_interval=10,
    )

    final_path = os.path.join(SAVE_DIR, 'aigp_fast_safe_final')
    model.save(final_path)
    print(f'\n✅ Final model saved to {final_path}.zip')

    vec_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
