"""Resume training from best Swift model for 100M more steps."""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from drone_race_env import DroneRaceEnv


def lr_schedule(progress_remaining):
    return 3e-4 * progress_remaining


def make_train_env():
    return DroneRaceEnv(render_mode=None, domain_rand=True)


def make_eval_env():
    return DroneRaceEnv(render_mode=None, domain_rand=False)


save_dir = './trained_swift_100m'
os.makedirs(save_dir, exist_ok=True)

vec_env = make_vec_env(make_train_env, n_envs=8)
eval_env = make_vec_env(make_eval_env, n_envs=1)

model = PPO.load('./trained_swift/best_model/best_model.zip', env=vec_env,
                 custom_objects={'learning_rate': lr_schedule, 'lr_schedule': lr_schedule})
model.ent_coef = 0.001

checkpoint_cb = CheckpointCallback(
    save_freq=25_000,
    save_path=os.path.join(save_dir, 'checkpoints'),
    name_prefix='aigp_ppo',
)
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(save_dir, 'best_model'),
    log_path=os.path.join(save_dir, 'eval_logs'),
    eval_freq=10_000,
    n_eval_episodes=5,
    deterministic=True,
)

total = 100_000_000
print(f'Resuming training for {total:,} timesteps from best Swift model...')
print(f'Saving to: {save_dir}')
model.learn(
    total_timesteps=total,
    callback=[checkpoint_cb, eval_cb],
    progress_bar=True,
)
final_path = os.path.join(save_dir, 'aigp_racer_final')
model.save(final_path)
print(f'Final model saved to {final_path}.zip')
vec_env.close()
eval_env.close()
