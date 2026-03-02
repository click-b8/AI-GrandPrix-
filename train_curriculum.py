"""Curriculum training: start easy, ramp up to full Swift difficulty.

Phase 1 (0-20M):   difficulty 0.0 — no noise, no motor lag, no delay
Phase 2 (20-40M):  difficulty 0.0→0.5 — gradually add realism
Phase 3 (40-70M):  difficulty 0.5→1.0 — ramp to full Swift
Phase 4 (70-100M): difficulty 1.0 — full Swift, fine-tune

This lets the agent first learn the racing line, then adapt to noise.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from drone_race_env import DroneRaceEnv


TOTAL_TIMESTEPS = 100_000_000
SAVE_DIR = './trained_curriculum'

# Curriculum schedule: (timestep_fraction, difficulty)
CURRICULUM = [
    (0.0, 0.0),    # start easy
    (0.20, 0.0),   # stay easy for first 20%
    (0.40, 0.5),   # ramp to medium by 40%
    (0.70, 1.0),   # ramp to full by 70%
    (1.0, 1.0),    # stay at full
]


def get_difficulty(progress: float) -> float:
    """Interpolate difficulty from curriculum schedule."""
    for i in range(len(CURRICULUM) - 1):
        t0, d0 = CURRICULUM[i]
        t1, d1 = CURRICULUM[i + 1]
        if t0 <= progress <= t1:
            alpha = (progress - t0) / (t1 - t0) if t1 > t0 else 1.0
            return d0 + alpha * (d1 - d0)
    return CURRICULUM[-1][1]


class CurriculumCallback(BaseCallback):
    """Update environment difficulty based on training progress."""

    def __init__(self, total_timesteps, verbose=1):
        super().__init__(verbose)
        self._total_timesteps = total_timesteps
        self._last_difficulty = -1.0

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self._total_timesteps
        difficulty = get_difficulty(progress)

        # Only update when difficulty changes significantly
        if abs(difficulty - self._last_difficulty) > 0.01:
            self._last_difficulty = difficulty
            # Update all vectorized envs
            env = self.training_env
            for i in range(env.num_envs):
                env.env_method('set_difficulty', difficulty, indices=[i])
            if self.verbose:
                print(f'\n[Curriculum] Step {self.num_timesteps:,} '
                      f'({progress:.1%}) -> difficulty={difficulty:.2f}')
        return True


# Global difficulty holder for env creation
_current_difficulty = [0.0]


def make_train_env():
    return DroneRaceEnv(render_mode=None, domain_rand=True, difficulty=_current_difficulty[0])


def make_eval_env():
    return DroneRaceEnv(render_mode=None, domain_rand=False, difficulty=1.0)


def lr_schedule(progress_remaining):
    return 3e-4 * progress_remaining


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    _current_difficulty[0] = 0.0  # start easy
    vec_env = make_vec_env(make_train_env, n_envs=8)
    eval_env = make_vec_env(make_eval_env, n_envs=1)

    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        learning_rate=lr_schedule,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=os.path.join(SAVE_DIR, 'tb_logs'),
    )

    curriculum_cb = CurriculumCallback(TOTAL_TIMESTEPS)
    checkpoint_cb = CheckpointCallback(
        save_freq=25_000,
        save_path=os.path.join(SAVE_DIR, 'checkpoints'),
        name_prefix='aigp_curriculum',
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVE_DIR, 'best_model'),
        log_path=os.path.join(SAVE_DIR, 'eval_logs'),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f'Curriculum training: {TOTAL_TIMESTEPS:,} timesteps')
    print(f'Schedule: easy(0-20M) -> medium(20-40M) -> full(40-70M) -> fine-tune(70-100M)')
    print(f'Eval always at full difficulty (1.0)')

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[curriculum_cb, checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    final_path = os.path.join(SAVE_DIR, 'aigp_curriculum_final')
    model.save(final_path)
    print(f'\nFinal model saved to {final_path}.zip')

    vec_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
