"""Quick test to debug HPC segfault."""
import torch
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

from train_vision import make_train_env, make_eval_env, DroneVisionExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

print("Step 1: single env")
env = make_train_env()
obs, _ = env.reset()
print(f"  OK: {obs['image'].shape}")
env.close()

print("Step 2: vec env (8 envs)")
vec_env = make_vec_env(make_train_env, n_envs=8, vec_env_cls=DummyVecEnv)
obs = vec_env.reset()
print(f"  OK: {obs['image'].shape}")

print("Step 3: create PPO model")
policy_kwargs = dict(
    features_extractor_class=DroneVisionExtractor,
    net_arch=dict(pi=[128, 64], vf=[128, 64]),
)
model = PPO(
    policy='MultiInputPolicy',
    env=vec_env,
    n_steps=64,
    batch_size=64,
    policy_kwargs=policy_kwargs,
    verbose=0,
    device='cuda' if torch.cuda.is_available() else 'cpu',
)
print(f"  OK on device: {model.device}")

print("Step 4: short training (512 steps)")
model.learn(total_timesteps=512)
print("  OK")

vec_env.close()
print("All tests passed!")
