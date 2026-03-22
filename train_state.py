"""Stage 1: Train a state-based expert policy (privileged, no vision).

Uses full 24D state observation including ground-truth gate positions.
This converges MUCH faster than vision-based training (~10-20M steps).
The expert policy is then used to generate demonstrations for Stage 2
(behavioral cloning + DAgger to train the vision policy).

Based on Swift (Nature 2023): train with privileged info first, then distill.
"""

import os
import glob
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from drone_race_env import DroneRaceEnv
from config import EPISODE_LENGTH_SEC, CTRL_FREQ

N_ENVS = 8
TOTAL_TIMESTEPS = 50_000_000  # Extended to master full difficulty
SAVE_DIR = './trained_state_expert'


# Curriculum: ramp difficulty faster since state-based learns quickly
CURRICULUM = [
    (0.0, 0.3),    # start easy
    (0.10, 0.5),   # medium
    (0.25, 0.8),   # harder
    (0.40, 1.0),   # full difficulty — spend 60% of training here
    (1.0, 1.0),
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
    """State-based env: no vision, with domain randomization."""
    return DroneRaceEnv(
        render_mode=None, domain_rand=True, difficulty=0.3, vision_mode=False,
    )


def make_eval_env():
    """Eval env: no randomization, full difficulty."""
    return DroneRaceEnv(
        render_mode=None, domain_rand=False, difficulty=1.0, vision_mode=False,
    )


def lr_schedule(progress_remaining):
    """Exponential decay from 3e-4 to 3e-5."""
    return 3e-4 * (0.1 ** (1 - progress_remaining))


def find_latest_checkpoint(save_dir):
    ckpt_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        return None, 0
    ckpts = glob.glob(os.path.join(ckpt_dir, 'aigp_state_*_steps.zip'))
    if not ckpts:
        return None, 0
    def get_steps(path):
        name = os.path.basename(path)
        return int(name.split('_')[-2])
    latest = max(ckpts, key=get_steps)
    steps = get_steps(latest)
    return latest, steps


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    vec_env = make_vec_env(make_train_env, n_envs=N_ENVS, vec_env_cls=DummyVecEnv)
    eval_env = make_vec_env(make_eval_env, n_envs=1)

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
    )

    resume_path, resume_steps = find_latest_checkpoint(SAVE_DIR)
    remaining_timesteps = TOTAL_TIMESTEPS - resume_steps

    if resume_path and remaining_timesteps > 0:
        print(f'Resuming from checkpoint: {resume_path} ({resume_steps:,} steps done)')
        model = PPO.load(
            resume_path,
            env=vec_env,
            custom_objects={'learning_rate': lr_schedule},
            tensorboard_log=os.path.join(SAVE_DIR, 'tb_logs'),
        )
        model.set_env(vec_env)
    elif remaining_timesteps <= 0:
        print(f'Training already complete ({resume_steps:,} >= {TOTAL_TIMESTEPS:,})')
        vec_env.close()
        eval_env.close()
        return
    else:
        print('Starting fresh state-based expert training')
        model = PPO(
            policy='MlpPolicy',
            env=vec_env,
            learning_rate=lr_schedule,
            n_steps=2048,
            batch_size=1024,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=os.path.join(SAVE_DIR, 'tb_logs'),
        )

    curriculum_cb = CurriculumCallback(TOTAL_TIMESTEPS)
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(SAVE_DIR, 'checkpoints'),
        name_prefix='aigp_state',
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVE_DIR, 'best_model'),
        log_path=os.path.join(SAVE_DIR, 'eval_logs'),
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
    )

    print(f'State-based expert training: {remaining_timesteps:,} timesteps')
    print(f'Observation: 24D (pos + vel + rpy + angvel + gate_rel + prev_action)')
    print(f'Action: 4D CTBR (thrust + body rates)')
    print(f'Network: MLP [256, 256, 128]')
    print(f'Curriculum: easy(0-15%) -> medium(15-40%) -> hard(40-60%) -> full(60%+)')
    print(f'Envs: {N_ENVS} (DummyVecEnv)')

    model.learn(
        total_timesteps=remaining_timesteps,
        callback=[curriculum_cb, checkpoint_cb, eval_cb],
        progress_bar=True,
        reset_num_timesteps=False if resume_path else True,
    )

    final_path = os.path.join(SAVE_DIR, 'aigp_state_final')
    model.save(final_path)
    print(f'\nFinal expert model saved to {final_path}.zip')

    vec_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
