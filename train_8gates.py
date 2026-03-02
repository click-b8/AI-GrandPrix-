"""Push for 8/8 gates — fine-tune with relaxed roll limit + progressive rewards.

Changes from previous:
- Roll/pitch crash limit: 90° → 120° (allows aggressive banking)
- Progressive gate reward: later gates worth more
- Start from best 5-gate model, train at full Swift difficulty
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from drone_race_env import DroneRaceEnv


TOTAL_TIMESTEPS = 100_000_000
SAVE_DIR = './trained_8gates'
SOURCE_MODEL = './trained_allgates/best_model/best_model.zip'


def make_train_env():
    return DroneRaceEnv(render_mode=None, domain_rand=True, difficulty=1.0)


def make_eval_env():
    return DroneRaceEnv(render_mode=None, domain_rand=False, difficulty=1.0)


def lr_schedule(progress_remaining):
    return 2e-4 * progress_remaining


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    vec_env = make_vec_env(make_train_env, n_envs=8)
    eval_env = make_vec_env(make_eval_env, n_envs=1)

    print(f'Loading base model: {SOURCE_MODEL}')
    model = PPO.load(SOURCE_MODEL, env=vec_env,
                     custom_objects={'learning_rate': lr_schedule, 'lr_schedule': lr_schedule})
    model.ent_coef = 0.002  # slightly more exploration to find gate 6-8 paths
    model.learning_rate = lr_schedule
    model.tensorboard_log = os.path.join(SAVE_DIR, 'tb_logs')

    checkpoint_cb = CheckpointCallback(
        save_freq=25_000,
        save_path=os.path.join(SAVE_DIR, 'checkpoints'),
        name_prefix='aigp_8gates',
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVE_DIR, 'best_model'),
        log_path=os.path.join(SAVE_DIR, 'eval_logs'),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f'Training for {TOTAL_TIMESTEPS:,} timesteps')
    print(f'Roll limit: 120° (was 90°), progressive gate rewards')
    print(f'LR: 2e-4, entropy: 0.002 (more exploration)')

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    final_path = os.path.join(SAVE_DIR, 'aigp_8gates_final')
    model.save(final_path)
    print(f'\nFinal model saved to {final_path}.zip')
    vec_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
