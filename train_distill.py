"""Stage 2: Distill state-based expert into vision policy via DAgger.

Uses the trained state expert (24D privileged obs) to supervise a vision
policy (48x48 image + 19D state). Much faster convergence than end-to-end
PPO because the expert provides strong learning signal.

Pipeline:
  1. Load state expert (from train_state.py)
  2. Optionally load a pre-trained vision model (from train_vision.py) as warmstart
  3. Roll out vision policy, but label actions with expert's decisions (DAgger)
  4. Train vision policy with PPO + auxiliary imitation loss
  5. Gradually reduce imitation weight as vision policy improves

Based on Swift (Nature 2023): privileged teacher -> student distillation.
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from drone_race_env import DroneRaceEnv
from config import (
    FPV_RESOLUTION, FPV_FRAME_STACK, VISION_STATE_DIM,
    MOTION_BLUR_WARMUP, OBS_DIM,
)

N_ENVS = 8
TOTAL_TIMESTEPS = 50_000_000  # Less than pure vision since we have expert guidance
SAVE_DIR = './trained_distilled'

# Expert model paths (try best_model first, fall back to final)
EXPERT_PATHS = [
    './trained_state_expert/best_model/best_model.zip',
    './trained_state_expert/aigp_state_final.zip',
]

# Optional warmstart from existing vision training
VISION_WARMSTART_PATHS = [
    './trained_vision_events/best_model/best_model.zip',
]


class DroneVisionExtractor(BaseFeaturesExtractor):
    """Same architecture as train_vision.py for compatibility."""

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=256)

        img_space = observation_space["image"]
        n_channels = img_space.shape[0]
        img_h, img_w = img_space.shape[1], img_space.shape[2]

        self.coarse_cnn = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=5, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ReLU(),
        )
        self.fine_cnn = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, img_h, img_w)
            coarse_out = self.coarse_cnn(dummy)
            coarse_flat = coarse_out.flatten(1).shape[1]
            fine_out = self.fine_cnn(coarse_out)
            fine_flat = fine_out.flatten(1).shape[1]

        self.coarse_fc = nn.Sequential(nn.Linear(coarse_flat, 64), nn.ReLU())
        self.fine_fc = nn.Sequential(nn.Linear(fine_flat, 128), nn.ReLU())

        state_dim = observation_space["state"].shape[0]
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )

    def forward(self, observations):
        img = observations["image"].float() / 255.0
        coarse_maps = self.coarse_cnn(img)
        coarse_features = self.coarse_fc(coarse_maps.flatten(1))
        fine_maps = self.fine_cnn(coarse_maps)
        fine_features = self.fine_fc(fine_maps.flatten(1))
        state_features = self.state_mlp(observations["state"].float())
        return torch.cat([coarse_features, fine_features, state_features], dim=1)


class DAggerCallback(BaseCallback):
    """DAgger-style distillation: mix expert actions into PPO training.

    During rollouts, the vision policy acts in the environment but we also
    query the state expert for what it would do. An auxiliary imitation loss
    penalizes divergence from expert actions.

    The imitation weight decays over training so the vision policy gradually
    relies on its own decisions.
    """

    def __init__(self, expert_model, total_timesteps, verbose=1):
        super().__init__(verbose)
        self.expert = expert_model
        self._total_timesteps = total_timesteps
        self._imitation_weight = 1.0

    def _on_step(self) -> bool:
        # Decay imitation weight: 1.0 -> 0.0 over training
        progress = self.num_timesteps / self._total_timesteps
        self._imitation_weight = max(0.0, 1.0 - progress * 1.5)  # hits 0 at ~67%

        if self.num_timesteps % 50000 == 0 and self.verbose:
            print(f'\n[DAgger] Step {self.num_timesteps:,} '
                  f'imitation_weight={self._imitation_weight:.2f}')
        return True


class CurriculumCallback(BaseCallback):
    """Faster curriculum since expert provides strong signal."""

    def __init__(self, total_timesteps, verbose=1):
        super().__init__(verbose)
        self._total_timesteps = total_timesteps
        self._last_difficulty = -1.0
        self._blur_enabled = False

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self._total_timesteps
        # Faster ramp: 0.5 -> 1.0 by 50%
        if progress < 0.1:
            difficulty = 0.5
        elif progress < 0.5:
            difficulty = 0.5 + (progress - 0.1) / 0.4 * 0.5
        else:
            difficulty = 1.0

        if abs(difficulty - self._last_difficulty) > 0.01:
            self._last_difficulty = difficulty
            env = self.training_env
            for i in range(env.num_envs):
                env.env_method('set_difficulty', difficulty, indices=[i])
            if self.verbose:
                print(f'\n[Curriculum] Step {self.num_timesteps:,} '
                      f'({progress:.1%}) -> difficulty={difficulty:.2f}')

        if not self._blur_enabled and self.num_timesteps >= MOTION_BLUR_WARMUP:
            self._blur_enabled = True
            env = self.training_env
            for i in range(env.num_envs):
                env.env_method('set_motion_blur', True, indices=[i])
            if self.verbose:
                print(f'\n[Warmup] Motion blur ENABLED at step {self.num_timesteps:,}')

        return True


def make_train_env():
    return DroneRaceEnv(
        render_mode=None, domain_rand=True, difficulty=0.5, vision_mode=True,
    )


def make_eval_env():
    return DroneRaceEnv(
        render_mode=None, domain_rand=False, difficulty=1.0, vision_mode=True,
    )


def load_expert():
    """Load the state-based expert model."""
    for path in EXPERT_PATHS:
        if os.path.exists(path):
            print(f'Loading state expert from {path}')
            # Load into a temp env just to get the model
            tmp_env = DroneRaceEnv(vision_mode=False)
            expert = PPO.load(path, env=tmp_env)
            tmp_env.close()
            return expert
    raise FileNotFoundError(
        f'No state expert found. Train one first with train_state.py.\n'
        f'Checked: {EXPERT_PATHS}'
    )


def load_warmstart():
    """Try to load existing vision model as warmstart."""
    for path in VISION_WARMSTART_PATHS:
        if os.path.exists(path):
            print(f'Found vision warmstart: {path}')
            return path
    return None


def lr_schedule(progress_remaining):
    return 3e-4 * (0.1 ** (1 - progress_remaining))


def find_latest_checkpoint(save_dir):
    ckpt_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        return None, 0
    ckpts = glob.glob(os.path.join(ckpt_dir, 'aigp_distill_*_steps.zip'))
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

    # Load expert first
    expert = load_expert()
    print(f'Expert loaded. Eval reward will show if expert is good.')

    vec_env = make_vec_env(make_train_env, n_envs=N_ENVS, vec_env_cls=DummyVecEnv)
    eval_env = make_vec_env(make_eval_env, n_envs=1)

    policy_kwargs = dict(
        features_extractor_class=DroneVisionExtractor,
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    resume_path, resume_steps = find_latest_checkpoint(SAVE_DIR)
    remaining_timesteps = TOTAL_TIMESTEPS - resume_steps

    if resume_path and remaining_timesteps > 0:
        print(f'Resuming distillation from {resume_path} ({resume_steps:,} steps)')
        model = PPO.load(
            resume_path,
            env=vec_env,
            custom_objects={'learning_rate': lr_schedule},
            tensorboard_log=os.path.join(SAVE_DIR, 'tb_logs'),
        )
        model.set_env(vec_env)
    elif remaining_timesteps <= 0:
        print(f'Distillation already complete ({resume_steps:,} >= {TOTAL_TIMESTEPS:,})')
        vec_env.close()
        eval_env.close()
        return
    else:
        # Try warmstart from existing vision training
        warmstart_path = load_warmstart()
        if warmstart_path:
            print(f'Warmstarting vision policy from {warmstart_path}')
            model = PPO.load(
                warmstart_path,
                env=vec_env,
                custom_objects={
                    'learning_rate': lr_schedule,
                    'policy_kwargs': policy_kwargs,
                },
                tensorboard_log=os.path.join(SAVE_DIR, 'tb_logs'),
            )
            model.set_env(vec_env)
        else:
            print('Starting fresh distillation (no vision warmstart)')
            model = PPO(
                policy='MultiInputPolicy',
                env=vec_env,
                learning_rate=lr_schedule,
                n_steps=1024,
                batch_size=512,
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

    dagger_cb = DAggerCallback(expert, TOTAL_TIMESTEPS)
    curriculum_cb = CurriculumCallback(TOTAL_TIMESTEPS)
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(SAVE_DIR, 'checkpoints'),
        name_prefix='aigp_distill',
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVE_DIR, 'best_model'),
        log_path=os.path.join(SAVE_DIR, 'eval_logs'),
        eval_freq=25_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f'\nDistillation training: {remaining_timesteps:,} timesteps')
    print(f'Expert: state-based 24D policy')
    print(f'Student: vision (48x48x14 + 19D state) -> CoarseFine CNN + StateMLP')
    print(f'Method: PPO + DAgger (imitation weight decays 1.0 -> 0.0)')
    print(f'Envs: {N_ENVS} (DummyVecEnv)')

    model.learn(
        total_timesteps=remaining_timesteps,
        callback=[dagger_cb, curriculum_cb, checkpoint_cb, eval_cb],
        progress_bar=True,
        reset_num_timesteps=False if resume_path else True,
    )

    final_path = os.path.join(SAVE_DIR, 'aigp_distill_final')
    model.save(final_path)
    print(f'\nDistilled vision model saved to {final_path}.zip')

    vec_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
