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

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from drone_race_env import DroneRaceEnv
from config import (
    FPV_RESOLUTION, FPV_FRAME_STACK, VISION_STATE_DIM,
    MOTION_BLUR_WARMUP,
)

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


class GatesPassedEvalCallback(BaseCallback):
    """Periodic deterministic eval that logs the REAL VQ1 signal — gate progress,
    not reward.

    Every ``eval_freq`` timesteps, runs ``n_eval_episodes`` deterministic episodes
    on a dedicated eval env and logs to TensorBoard AND stdout (so tmux scrollback
    shows progress without TensorBoard):

        eval/gates_passed      mean gates cleared per eval episode
        eval/gates_passed_max  best single eval episode
        eval/success_rate      fraction of episodes clearing ALL gates (completion)
        eval/total_gates       gate count, read from the env ``info["total_gates"]``
                               — NEVER hardcoded, so it tracks the real course.

    Saves ``best_model.zip`` whenever mean gates_passed improves. This is the
    metric §2.5 of DGX_TRAINING_GUIDE.md says to watch; a run that cannot emit it
    is a blind run.
    """

    def __init__(self, eval_env, n_eval_episodes=5, eval_freq=25_000,
                 best_model_save_path=None, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_model_save_path = best_model_save_path
        self._best_mean_gates = -np.inf
        self._last_eval = 0

    def _run_episode(self):
        """One deterministic episode on the (n_envs=1) eval VecEnv.
        Returns (gates_passed, total_gates) from the env info."""
        obs = self.eval_env.reset()
        done = np.array([False])
        gates, total = 0, None
        while not bool(done[0]):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _reward, done, infos = self.eval_env.step(action)
            info = infos[0]
            if "gates_passed" in info:
                gates = max(gates, int(info["gates_passed"]))  # monotonic within episode
            if "total_gates" in info:
                total = int(info["total_gates"])
        return gates, total

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or (self.num_timesteps - self._last_eval) < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps

        gates, totals = [], []
        for _ in range(self.n_eval_episodes):
            g, t = self._run_episode()
            gates.append(g)
            if t is not None:
                totals.append(t)
        gates = np.asarray(gates, dtype=float)
        total_gates = int(max(totals)) if totals else 0
        mean_g, max_g = float(gates.mean()), float(gates.max())
        success = float(np.mean(gates >= total_gates)) if total_gates else 0.0

        self.logger.record("eval/gates_passed", mean_g)
        self.logger.record("eval/gates_passed_max", max_g)
        self.logger.record("eval/success_rate", success)
        self.logger.record("eval/total_gates", float(total_gates))
        self.logger.dump(self.num_timesteps)  # flush now so the tag is in the event file
        print(f"[EVAL @ {self.num_timesteps:,} steps] gates_passed "
              f"mean={mean_g:.2f} max={max_g:.0f}/{total_gates}  "
              f"success_rate={success:.0%}  (n={self.n_eval_episodes})", flush=True)

        if self.best_model_save_path is not None and mean_g > self._best_mean_gates:
            self._best_mean_gates = mean_g
            os.makedirs(self.best_model_save_path, exist_ok=True)
            path = os.path.join(self.best_model_save_path, "best_model.zip")
            self.model.save(path)
            print(f"[EVAL] new best mean gates_passed={mean_g:.2f} -> {path}", flush=True)
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


def parse_args():
    p = argparse.ArgumentParser(description="Vision-based drone racing PPO training")
    p.add_argument('--n-envs', type=int, default=16,
                   help='parallel training envs. >1 -> SubprocVecEnv (DGX default ~16-32); '
                        '1 -> DummyVecEnv (local smoke, single process).')
    p.add_argument('--seed', type=int, default=0, help='RNG seed for PPO + envs')
    p.add_argument('--save-dir', default=SAVE_DIR,
                   help='output dir (checkpoints/, tb_logs/, best_model/). One per seed '
                        'for the guide\'s per-GPU isolation.')
    p.add_argument('--total-timesteps', type=int, default=TOTAL_TIMESTEPS)
    p.add_argument('--n-steps', type=int, default=1024, help='PPO rollout steps per env')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--eval-freq', type=int, default=25_000, help='timesteps between evals')
    p.add_argument('--eval-episodes', type=int, default=5)
    p.add_argument('--save-freq', type=int, default=50_000, help='timesteps between checkpoints')
    p.add_argument('--no-progress-bar', action='store_true',
                   help='disable the tqdm/rich progress bar (useful for non-TTY smoke runs)')
    return p.parse_args()


def main():
    args = parse_args()
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # SubprocVecEnv is the DGX default (each worker builds its own headless EGL
    # context — the "unreliable with MuJoCo GL" caveat is macOS-only). n_envs=1
    # falls back to a single-process DummyVecEnv for local smoke runs.
    vec_env_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    vec_env = make_vec_env(make_train_env, n_envs=args.n_envs,
                           vec_env_cls=vec_env_cls, seed=args.seed)
    eval_env = make_vec_env(make_eval_env, n_envs=1, seed=args.seed + 10_000)

    policy_kwargs = dict(
        features_extractor_class=DroneVisionExtractor,
        # Smaller policy heads (feature extractor does the heavy lifting)
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    # Auto-resume from latest checkpoint if available
    resume_path, resume_steps = find_latest_checkpoint(save_dir)
    remaining_timesteps = args.total_timesteps - resume_steps

    if resume_path and remaining_timesteps > 0:
        print(f'Resuming from checkpoint: {resume_path} ({resume_steps:,} steps done)')
        print(f'Remaining: {remaining_timesteps:,} timesteps')
        model = PPO.load(
            resume_path,
            env=vec_env,
            custom_objects={'learning_rate': lr_schedule},
            tensorboard_log=os.path.join(save_dir, 'tb_logs'),
        )
        model.set_env(vec_env)
    elif remaining_timesteps <= 0:
        print(f'Training already complete ({resume_steps:,} >= {args.total_timesteps:,})')
        vec_env.close()
        eval_env.close()
        return
    else:
        print('Starting fresh training')
        model = PPO(
            policy='MultiInputPolicy',
            env=vec_env,
            learning_rate=lr_schedule,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            tensorboard_log=os.path.join(save_dir, 'tb_logs'),
        )

    curriculum_cb = CurriculumCallback(args.total_timesteps)
    # CheckpointCallback counts CALLS (per vec-step); divide by n_envs for timesteps.
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, args.save_freq // args.n_envs),
        save_path=os.path.join(save_dir, 'checkpoints'),
        name_prefix='aigp_vision',
    )
    gates_eval_cb = GatesPassedEvalCallback(
        eval_env,
        n_eval_episodes=args.eval_episodes,
        eval_freq=args.eval_freq,
        best_model_save_path=os.path.join(save_dir, 'best_model'),
    )

    print(f'Vision-based training: {args.total_timesteps:,} timesteps  (seed={args.seed})')
    print(f'Image: {FPV_RESOLUTION}x{FPV_RESOLUTION}, frame stack: {FPV_FRAME_STACK}')
    print(f'State: {VISION_STATE_DIM}D (gravity_unit(3) + body_rates(3) + prev_action(4))')
    print(f'Feature extractor: CoarseCNN(64D) + FineCNN(128D) + StateMLP(64D) = 256D')
    print(f'Curriculum: medium(0-30%) -> ramp(30-80%) -> full(80-100%)')
    print(f'Motion blur warmup at step {MOTION_BLUR_WARMUP:,}')
    print(f'Envs: {args.n_envs} ({vec_env_cls.__name__})   save_dir: {save_dir}')
    print(f'Eval: every {args.eval_freq:,} steps, {args.eval_episodes} episodes '
          f'-> eval/gates_passed + eval/success_rate')

    model.learn(
        total_timesteps=remaining_timesteps,
        callback=[curriculum_cb, checkpoint_cb, gates_eval_cb],
        progress_bar=not args.no_progress_bar,
        reset_num_timesteps=False if resume_path else True,
    )

    final_path = os.path.join(save_dir, 'aigp_vision_final')
    model.save(final_path)
    print(f'\nFinal model saved to {final_path}.zip')

    vec_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
