"""Vision-based drone racing: Coarse-to-Fine CNN+State hybrid policy training.

Optimized with techniques from event-sharp-nerf-drones (Zou et al. 2026, UZH RPG):
  - Coarse-to-fine CNN with feature concatenation (faster convergence)
  - Event camera sensor: async brightness-change events that stay sharp at high speed
  - 6D rotation representation (continuous, no gimbal lock — Zhou CVPR 2019)
  - Multi-rate learning rates (CNN vs state MLP vs policy heads)
  - Motion blur simulation with staged warmup
  - SE(3)-aware state branch for VIO pose estimation

Architecture:
  - Coarse CNN: 2 conv layers -> 64D (fast gate detection from RGB + events)
  - Fine CNN: 2 conv layers on coarse features -> 128D (precise localization)
  - State MLP: 2 FC layers -> 64D (6D rot + VIO vel/angvel/pos + prev action)
  - Combined: 64 + 128 + 64 = 256 features -> PPO policy/value heads

Event camera integration (EGM from event-sharp-nerf-drones):
  - Simulated event camera fires per-pixel when log-luminance changes exceed C
  - Events accumulated into histogram bins (pos/neg counts per temporal bin)
  - Concatenated with RGB as extra input channels to the CNN
  - Events don't blur at high speed — provides sharp edge info during aggressive flight
"""

import os
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
    MOTION_BLUR_WARMUP,
)

N_ENVS = 8  # DummyVecEnv (SubprocVecEnv unreliable with MuJoCo GL on macOS)


TOTAL_TIMESTEPS = 200_000_000
SAVE_DIR = './trained_vision_events'


class DroneVisionExtractor(BaseFeaturesExtractor):
    """Coarse-to-Fine CNN + SE(3)-aware State MLP feature extractor.

    Inspired by event-sharp-nerf-drones (Zou et al. 2026):
    - Coarse-to-fine with feature concatenation (VoxelNeRF pattern)
    - Smaller per-stage networks = faster forward pass
    - State branch receives 6D rotation (continuous representation)

    Coarse CNN:
        Conv2d(in_channels, 16, 5, stride=2) -> ReLU
        Conv2d(16, 32, 3, stride=2) -> ReLU -> Flatten -> Linear(N, 64) -> ReLU

    Fine CNN (processes coarse feature maps):
        Conv2d(32, 48, 3, stride=1, pad=1) -> ReLU
        Conv2d(48, 64, 3, stride=1) -> ReLU -> Flatten -> Linear(N, 128) -> ReLU

    State MLP:
        Linear(19, 64) -> ReLU -> Linear(64, 64) -> ReLU

    Output: cat(coarse_64, fine_128, state_64) = 256 features
    """

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=256)

        img_space = observation_space["image"]
        n_channels = img_space.shape[0]
        img_h, img_w = img_space.shape[1], img_space.shape[2]

        # Coarse CNN: fast gate detection
        self.coarse_cnn = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        # Fine CNN: refines coarse features (concatenation pattern from voxnerf.py)
        self.fine_cnn = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Compute feature map sizes
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, img_h, img_w)
            coarse_out = self.coarse_cnn(dummy)
            coarse_flat = coarse_out.flatten(1).shape[1]
            fine_out = self.fine_cnn(coarse_out)
            fine_flat = fine_out.flatten(1).shape[1]

        self.coarse_fc = nn.Sequential(
            nn.Linear(coarse_flat, 64),
            nn.ReLU(),
        )
        self.fine_fc = nn.Sequential(
            nn.Linear(fine_flat, 128),
            nn.ReLU(),
        )

        # State MLP (receives 6D rotation + VIO estimates)
        state_dim = observation_space["state"].shape[0]
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

    def forward(self, observations):
        img = observations["image"].float() / 255.0

        # Coarse-to-fine with feature concatenation
        coarse_maps = self.coarse_cnn(img)
        coarse_features = self.coarse_fc(coarse_maps.flatten(1))
        fine_maps = self.fine_cnn(coarse_maps)
        fine_features = self.fine_fc(fine_maps.flatten(1))

        state_features = self.state_mlp(observations["state"].float())

        return torch.cat([coarse_features, fine_features, state_features], dim=1)


# Curriculum schedule: (timestep_fraction, difficulty)
CURRICULUM = [
    (0.0, 0.5),    # start at medium (agent already knows basics from state-based)
    (0.30, 0.5),   # stay medium while CNN learns gate detection
    (0.60, 0.8),   # ramp up
    (0.80, 1.0),   # full difficulty
    (1.0, 1.0),
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
    """Update environment difficulty + motion blur warmup based on training progress.

    Motion blur warmup (from event-sharp-nerf-drones): motion blur modeling
    activates after MOTION_BLUR_WARMUP steps, allowing the network to first
    learn coarse gate geometry from clean images before adding blur.
    """

    def __init__(self, total_timesteps, verbose=1):
        super().__init__(verbose)
        self._total_timesteps = total_timesteps
        self._last_difficulty = -1.0
        self._blur_enabled = False

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

        # Staged motion blur warmup (event-sharp-nerf-drones pattern)
        if not self._blur_enabled and self.num_timesteps >= MOTION_BLUR_WARMUP:
            self._blur_enabled = True
            env = self.training_env
            for i in range(env.num_envs):
                env.env_method('set_motion_blur', True, indices=[i])
            if self.verbose:
                print(f'\n[Warmup] Motion blur ENABLED at step {self.num_timesteps:,} '
                      f'(staged training: geometry learned first)')

        return True


def make_train_env():
    return DroneRaceEnv(
        render_mode=None, domain_rand=True, difficulty=0.5, vision_mode=True,
    )


def make_eval_env():
    return DroneRaceEnv(
        render_mode=None, domain_rand=False, difficulty=1.0, vision_mode=True,
    )


def lr_schedule(progress_remaining):
    """Exponential decay from 3e-4 to 3e-5 (event-sharp-nerf-drones pattern)."""
    return 3e-4 * (0.1 ** (1 - progress_remaining))


def find_latest_checkpoint(save_dir):
    """Find the latest checkpoint to resume training from."""
    ckpt_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        return None, 0
    import glob
    ckpts = glob.glob(os.path.join(ckpt_dir, 'aigp_vision_*_steps.zip'))
    if not ckpts:
        return None, 0
    # Extract step numbers and find the latest
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
        features_extractor_class=DroneVisionExtractor,
        # Smaller policy heads (feature extractor does the heavy lifting)
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    # Auto-resume from latest checkpoint if available
    resume_path, resume_steps = find_latest_checkpoint(SAVE_DIR)
    remaining_timesteps = TOTAL_TIMESTEPS - resume_steps

    if resume_path and remaining_timesteps > 0:
        print(f'Resuming from checkpoint: {resume_path} ({resume_steps:,} steps done)')
        print(f'Remaining: {remaining_timesteps:,} timesteps')
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
        print('Starting fresh training')
        model = PPO(
            policy='MultiInputPolicy',
            env=vec_env,
            learning_rate=lr_schedule,
            n_steps=1024,         # per-env steps (1024 * 8 = 8192 total per rollout)
            batch_size=512,       # larger batches for 8 envs
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
        name_prefix='aigp_vision',
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVE_DIR, 'best_model'),
        log_path=os.path.join(SAVE_DIR, 'eval_logs'),
        eval_freq=25_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f'Vision-based training: {TOTAL_TIMESTEPS:,} timesteps')
    print(f'Image: {FPV_RESOLUTION}x{FPV_RESOLUTION}, frame stack: {FPV_FRAME_STACK}')
    print(f'State: {VISION_STATE_DIM}D (6D rot + VIO vel/angvel + prev action + VIO pos)')
    print(f'Feature extractor: CoarseCNN(64D) + FineCNN(128D) + StateMLP(64D) = 256D')
    print(f'Optimizations: coarse-to-fine CNN, 6D rotation, motion blur warmup')
    print(f'Curriculum: medium(0-30%) -> ramp(30-80%) -> full(80-100%)')
    print(f'Motion blur warmup at step {MOTION_BLUR_WARMUP:,}')
    print(f'Envs: {N_ENVS} (DummyVecEnv)')

    model.learn(
        total_timesteps=remaining_timesteps,
        callback=[curriculum_cb, checkpoint_cb, eval_cb],
        progress_bar=True,
        reset_num_timesteps=False if resume_path else True,
    )

    final_path = os.path.join(SAVE_DIR, 'aigp_vision_final')
    model.save(final_path)
    print(f'\nFinal model saved to {final_path}.zip')

    vec_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
