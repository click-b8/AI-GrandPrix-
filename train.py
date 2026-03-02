"""Train a drone racing agent with PPO.

Matches Swift (Nature 2023) training setup:
- PPO algorithm
- 2-layer MLP, 128 neurons (configurable)
- Domain randomization enabled during training
- Headless (no GUI) for speed
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from drone_race_env import DroneRaceEnv


def make_train_env():
    """Training env with domain randomization."""
    return DroneRaceEnv(render_mode=None, domain_rand=True)


def make_eval_env():
    """Evaluation env without domain randomization."""
    return DroneRaceEnv(render_mode=None, domain_rand=False)


def train(total_timesteps: int, save_dir: str, resume_from: str = None, n_envs: int = 8):
    os.makedirs(save_dir, exist_ok=True)

    vec_env = make_vec_env(make_train_env, n_envs=n_envs)
    eval_env = make_vec_env(make_eval_env, n_envs=1)

    # Network architecture matching Swift: 2-layer MLP, 128 units
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    # Linear learning rate decay: starts at 3e-4, decays to 0
    def lr_schedule(progress_remaining: float) -> float:
        return 3e-4 * progress_remaining

    if resume_from and os.path.isfile(resume_from):
        print(f"Resuming from: {resume_from}")
        model = PPO.load(resume_from, env=vec_env)
        model.learning_rate = lr_schedule
        model.ent_coef = 0.001
    else:
        model = PPO(
            policy="MlpPolicy",
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
            tensorboard_log=os.path.join(save_dir, "tb_logs"),
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=25_000,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="aigp_ppo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best_model"),
        log_path=os.path.join(save_dir, "eval_logs"),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f"Training for {total_timesteps:,} timesteps with {n_envs} parallel envs...")
    print(f"Network: MLP [128, 128] (Swift architecture)")
    print(f"Domain randomization: ON")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    final_path = os.path.join(save_dir, "aigp_racer_final")
    model.save(final_path)
    print(f"Final model saved to {final_path}.zip")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI Grand Prix drone racing agent")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Total training timesteps")
    parser.add_argument("--save-dir", type=str, default="./trained", help="Save directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from .zip checkpoint")
    parser.add_argument("--envs", type=int, default=8, help="Number of parallel environments")
    args = parser.parse_args()

    train(args.timesteps, args.save_dir, args.resume, args.envs)
