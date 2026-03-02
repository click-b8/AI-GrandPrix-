"""Launch interactive MuJoCo viewer with trained drone agent.

The agent runs in a background thread, continuously stepping the env
and copying state into the viewer's model/data. The MuJoCo interactive
GUI runs on the main thread (required by macOS Cocoa).
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import threading

from stable_baselines3 import PPO
from drone_race_env import DroneRaceEnv
from config import CTRL_FREQ

MODEL_PATH = './trained_8gates/best_model/best_model.zip'

print(f'Loading model: {MODEL_PATH}')
agent = PPO.load(MODEL_PATH)
env = DroneRaceEnv(render_mode=None, domain_rand=False, difficulty=1.0)
obs, info = env.reset()

# Shared state
running = True
lock = threading.Lock()


def agent_loop(viewer_model, viewer_data):
    """Run the trained agent in a background thread."""
    global obs, running
    lap = 0
    step_delay = 1.0 / CTRL_FREQ

    while running:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Copy env physics state into viewer's data (thread-safe)
        with lock:
            np.copyto(viewer_data.qpos, env._data.qpos)
            np.copyto(viewer_data.qvel, env._data.qvel)
            mujoco.mj_forward(viewer_model, viewer_data)

        if terminated or truncated:
            gates = info.get('gates_passed', 0)
            speed = info.get('speed', 0)
            lap += 1
            print(f'Lap {lap}: {gates}/8 gates | speed {speed:.0f} m/s')
            obs, _ = env.reset()
            with lock:
                np.copyto(viewer_data.qpos, env._data.qpos)
                np.copyto(viewer_data.qvel, env._data.qvel)
                mujoco.mj_forward(viewer_model, viewer_data)
            time.sleep(1.0)  # pause between laps

        time.sleep(step_delay)


# Start agent thread before launching viewer
thread = threading.Thread(target=agent_loop, args=(env._model, env._data), daemon=True)
thread.start()

print('Interactive MuJoCo viewer launching...')
print('Mouse: rotate/zoom/pan. Close window to stop.')

# This blocks on the main thread — full interactive MuJoCo GUI
mujoco.viewer.launch(env._model, env._data)

running = False
thread.join(timeout=2)
env.close()
print('Done.')
