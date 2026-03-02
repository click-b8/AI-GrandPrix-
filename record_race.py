"""Record a drone race as video (no GUI needed)."""

import argparse
import numpy as np
import mujoco
from stable_baselines3 import PPO

from drone_race_env import DroneRaceEnv
from config import CTRL_FREQ


def record(model_path: str, output_path: str = "race.mp4", width: int = 640, height: int = 480):
    """Record a race episode to video file."""
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    env = DroneRaceEnv(render_mode=None, domain_rand=False)
    obs, info = env.reset()

    # Set up offscreen renderer
    renderer = mujoco.Renderer(env._model, height=height, width=width)
    frames = []

    # Camera setup
    cam = mujoco.MjvCamera()
    cam.distance = 25.0
    cam.azimuth = 60
    cam.elevation = -25
    cam.lookat[:] = [0, 7, 2.5]

    total_reward = 0.0
    step = 0
    max_speed = 0.0

    print("Running episode...")
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        max_speed = max(max_speed, info.get("speed", 0))

        # Render frame (every 2 steps for 50fps video from 100Hz control)
        if step % 2 == 0:
            renderer.update_scene(env._data, cam)
            frame = renderer.render()
            frames.append(frame.copy())

        if terminated or truncated:
            gates = info.get("gates_passed", 0)
            total = info.get("total_gates", "?")
            result = "FINISHED" if gates == total else "DNF"
            print(f"  {result} | Gates: {gates}/{total} | "
                  f"Steps: {step} | Reward: {total_reward:.1f} | "
                  f"Max speed: {max_speed:.1f} m/s")
            break

    env.close()
    del renderer

    # Save as video
    print(f"Saving {len(frames)} frames to {output_path}...")
    try:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 50.0, (width, height))
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"Video saved: {output_path}")
    except ImportError:
        # Fallback: save as numpy array
        npy_path = output_path.replace('.mp4', '_frames.npy')
        np.save(npy_path, np.array(frames))
        print(f"OpenCV not available. Frames saved as: {npy_path}")
        print("Install opencv-python to get MP4 output.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record drone race video")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .zip")
    parser.add_argument("--output", type=str, default="race.mp4", help="Output video path")
    args = parser.parse_args()
    record(args.model, args.output)
