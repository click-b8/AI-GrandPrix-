"""Run a trained drone racing agent with 3D visualization."""

import argparse
import time

import numpy as np
from stable_baselines3 import PPO

from config import CTRL_FREQ
from drone_race_env import DroneRaceEnv


def race(model_path: str, num_laps: int = 1, slow_motion: float = 1.0):
    """Load a trained model and fly it through the track with GUI rendering."""
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    env = DroneRaceEnv(render_mode="human", domain_rand=False)
    step_delay = (1.0 / CTRL_FREQ) * slow_motion

    for lap in range(num_laps):
        print(f"\n--- Lap {lap + 1}/{num_laps} ---")
        obs, info = env.reset()
        total_reward = 0.0
        step = 0
        max_speed = 0.0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            max_speed = max(max_speed, info.get("speed", 0))

            time.sleep(step_delay)

            if terminated or truncated:
                gates = info.get("gates_passed", 0)
                total = info.get("total_gates", "?")
                result = "FINISHED" if gates == total else "DNF"
                print(f"  {result} | Gates: {gates}/{total} | "
                      f"Steps: {step} | Reward: {total_reward:.1f} | "
                      f"Max speed: {max_speed:.1f} m/s")
                break

    env.close()
    print("\nRace complete!")


def demo_hover(seconds: int = 10):
    """Hover test — thrust at ~hover level, zero body rates."""
    from config import HOVER_THRUST_NORMALIZED
    print(f"Hover test for {seconds}s (thrust={HOVER_THRUST_NORMALIZED:.3f})...")
    env = DroneRaceEnv(render_mode="human", domain_rand=False)
    obs, info = env.reset()

    for i in range(seconds * CTRL_FREQ):
        action = np.array([HOVER_THRUST_NORMALIZED, 0.0, 0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(1.0 / CTRL_FREQ)
        if terminated:
            print(f"  Crashed at step {i}")
            break
    else:
        print(f"  Hover stable! Final pos: {obs[:3]}")

    env.close()


def demo_random(steps: int = 500):
    """Random actions to stress-test the environment."""
    print("Running random-action demo...")
    env = DroneRaceEnv(render_mode="human", domain_rand=False)
    obs, info = env.reset()

    for i in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(1.0 / CTRL_FREQ)

        if terminated or truncated:
            print(f"  Episode ended at step {i} | Gates: {info.get('gates_passed', 0)}")
            obs, info = env.reset()

    env.close()
    print("Demo complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Grand Prix — Drone Race")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model .zip")
    parser.add_argument("--hover", action="store_true",
                        help="Run hover stability test")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--slow", type=float, default=1.0,
                        help="Slow-motion factor (2.0 = half speed)")
    args = parser.parse_args()

    if args.model:
        race(args.model, num_laps=args.laps, slow_motion=args.slow)
    elif args.hover:
        demo_hover()
    else:
        demo_random()
