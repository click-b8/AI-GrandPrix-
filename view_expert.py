"""View the state-based expert flying in MuJoCo."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from drone_race_env import DroneRaceEnv
from config import CTRL_FREQ

# Find best model
for path in [
    './trained_state_expert/best_model/best_model.zip',
    './trained_state_expert/aigp_state_final.zip',
]:
    if os.path.exists(path):
        model_path = path
        break
else:
    print("No expert model found. Run train_state.py first.")
    sys.exit(1)

print(f"Loading {model_path}...")
fpv = '--fpv' in sys.argv
env = DroneRaceEnv(render_mode="human", domain_rand=False, difficulty=1.0, vision_mode=False, fpv_view=fpv)
if fpv:
    print("FPV mode: first-person camera view")
model = PPO.load(model_path, env=env)

step_delay = 1.0 / CTRL_FREQ  # real-time playback

input("Press ENTER to start flying...")

for lap in range(5):
    print(f"--- Lap {lap+1}/5 ---")
    obs, info = env.reset()
    total_reward = 0.0
    step = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        time.sleep(step_delay)
        if terminated or truncated:
            gates = info.get("gates_passed", 0)
            total = info.get("total_gates", "?")
            result = "FINISHED" if gates == total else "DNF"
            print(f"  {result} | Gates: {gates}/{total} | Steps: {step} | Reward: {total_reward:.1f}")
            time.sleep(2)  # pause between laps
            break

input("Press ENTER to close...")
env.close()
print("Done!")
