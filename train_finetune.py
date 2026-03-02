"""Fine-tune the 50M best model (5/8 gates) with gradual noise introduction.

Strategy: Start from a model that already races well, then slowly add
Swift realism so it retains its racing skill while becoming robust.

Phase 1 (0-5M):   difficulty 0.0 — warm up, re-adapt to training
Phase 2 (5-15M):  difficulty 0.0→0.3 — gentle noise introduction
Phase 3 (15-30M): difficulty 0.3→0.7 — moderate realism
Phase 4 (30-50M): difficulty 0.7→1.0 — full Swift
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from drone_race_env import DroneRaceEnv


TOTAL_TIMESTEPS = 50_000_000
SAVE_DIR = './trained_finetune'
SOURCE_MODEL = './trained_50m/best_model.zip'

# Slower curriculum — gentle ramp from a strong starting point
CURRICULUM = [
    (0.0, 0.0),    # warm up
    (0.10, 0.0),   # stay easy for 10%
    (0.30, 0.3),   # gentle noise by 30%
    (0.60, 0.7),   # moderate by 60%
    (1.0, 1.0),    # full Swift by end
]


def get_difficulty(progress: float) -> float:
    for i in range(len(CURRICULUM) - 1):
        t0, d0 = CURRICULUM[i]
        t1, d1 = CURRICULUM[i + 1]
        if t0 <= progress <= t1:
            alpha = (progress - t0) / (t1 - t0) if t1 > t0 else 1.0
            return d0 + alpha * (d1 - d0)
    return CURRICULUM[-1][1]


class CurriculumCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=1):
        super().__init__(verbose)
        self._total_timesteps = total_timesteps
        self._last_difficulty = -1.0

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self._total_timesteps
        difficulty = get_difficulty(progress)
        if abs(difficulty - self._last_difficulty) > 0.01:
            self._last_difficulty = difficulty
            env = self.training_env
            for i in range(env.num_envs):
                env.env_method('set_difficulty', difficulty, indices=[i])
            if self.verbose:
                print(f'\n[Curriculum] Step {self.num_timesteps:,} '
                      f'({progress:.1%}) -> difficulty={difficulty:.2f}')
        return True


def make_train_env():
    return DroneRaceEnv(render_mode=None, domain_rand=True, difficulty=0.0)


def make_eval_env():
    return DroneRaceEnv(render_mode=None, domain_rand=False, difficulty=1.0)


def lr_schedule(progress_remaining):
    # Lower LR for fine-tuning: 1e-4 max, decaying to 0
    return 1e-4 * progress_remaining


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    vec_env = make_vec_env(make_train_env, n_envs=8)
    eval_env = make_vec_env(make_eval_env, n_envs=1)

    # Load the 5-gate model and fine-tune
    print(f'Loading base model: {SOURCE_MODEL}')
    model = PPO.load(SOURCE_MODEL, env=vec_env,
                     custom_objects={'learning_rate': lr_schedule, 'lr_schedule': lr_schedule})
    model.ent_coef = 0.0005  # very low entropy — preserve learned policy
    model.learning_rate = lr_schedule
    model.tensorboard_log = os.path.join(SAVE_DIR, 'tb_logs')

    curriculum_cb = CurriculumCallback(TOTAL_TIMESTEPS)
    checkpoint_cb = CheckpointCallback(
        save_freq=25_000,
        save_path=os.path.join(SAVE_DIR, 'checkpoints'),
        name_prefix='aigp_finetune',
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVE_DIR, 'best_model'),
        log_path=os.path.join(SAVE_DIR, 'eval_logs'),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f'Fine-tuning for {TOTAL_TIMESTEPS:,} timesteps')
    print(f'Starting from 5/8 gate model, gradually adding Swift noise')
    print(f'LR: 1e-4 (lower for fine-tuning), entropy: 0.0005')
    print(f'Eval always at full difficulty (1.0)')

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[curriculum_cb, checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    final_path = os.path.join(SAVE_DIR, 'aigp_finetune_final')
    model.save(final_path)
    print(f'\nFinal model saved to {final_path}.zip')

    vec_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
