"""Train to pass all 8 gates — fine-tune from the 5/8 gate model.

Strategy:
- Start from the fine-tuned model (5/8 gates @ full Swift)
- Train at full difficulty from the start (already robust)
- Higher gate reward to incentivize pushing through later gates
- Longer training (100M steps) to master the full track
- Moderate LR with slow decay to allow continued learning
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback,
)
from drone_race_env import DroneRaceEnv


TOTAL_TIMESTEPS = 100_000_000
SAVE_DIR = './trained_allgates'
SOURCE_MODEL = './trained_finetune/best_model/best_model.zip'


def make_train_env():
    return DroneRaceEnv(render_mode=None, domain_rand=True, difficulty=1.0)


def make_eval_env():
    return DroneRaceEnv(render_mode=None, domain_rand=False, difficulty=1.0)


def lr_schedule(progress_remaining):
    # Moderate LR with slow decay for continued learning
    return 1.5e-4 * progress_remaining


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    vec_env = make_vec_env(make_train_env, n_envs=8)
    eval_env = make_vec_env(make_eval_env, n_envs=1)

    print(f'Loading base model: {SOURCE_MODEL}')
    model = PPO.load(SOURCE_MODEL, env=vec_env,
                     custom_objects={'learning_rate': lr_schedule, 'lr_schedule': lr_schedule})
    model.ent_coef = 0.001  # moderate exploration to discover new gate paths
    model.learning_rate = lr_schedule
    model.tensorboard_log = os.path.join(SAVE_DIR, 'tb_logs')

    checkpoint_cb = CheckpointCallback(
        save_freq=25_000,
        save_path=os.path.join(SAVE_DIR, 'checkpoints'),
        name_prefix='aigp_allgates',
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVE_DIR, 'best_model'),
        log_path=os.path.join(SAVE_DIR, 'eval_logs'),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f'Training for {TOTAL_TIMESTEPS:,} timesteps at full Swift difficulty')
    print(f'Goal: all 8/8 gates')
    print(f'LR: 1.5e-4 with decay, entropy: 0.001')

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    final_path = os.path.join(SAVE_DIR, 'aigp_allgates_final')
    model.save(final_path)
    print(f'\nFinal model saved to {final_path}.zip')

    vec_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
