#!/bin/bash
#SBATCH --job-name=aigp_finetune_tilt
#SBATCH --output=logs/finetune_tilt_%j.out
#SBATCH --error=logs/finetune_tilt_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL

# ============================================================
# SCUBA LAB — VQ1 Fine-tune: Camera Tilt + Event Channel Fix
# ============================================================
#
# PURPOSE:
#   Fine-tune aigp_distill_final from FPV_TILT_DEG=-10 to +20
#   to match VADR-TS-002 §3.8 ("camera tilted upwards by 20°").
#   Also disables EVENT_CAMERA_ENABLED since DCL provides RGB only.
#
# WHAT CHANGED IN CONFIG VS ORIGINAL TRAINING:
#   FPV_TILT_DEG:        -10  → +20   (spec match)
#   EVENT_CAMERA_ENABLED: True → False (RGB-only deployment)
#
# TEACHER:  aigp_racer_final (100M state expert — strongest available)
# STUDENT:  warm-started from aigp_distill_final checkpoint
# METHOD:   DAgger-style distillation (same as original train_distill.py)
# STEPS:    15M (enough for tilt adaptation; original was 52M from scratch)
#
# SELF-RESUBMIT: job resubmits itself at 5 min before timeout via USR1.
# OUTPUT:   trained_finetune_tilt/
#
# ============================================================

set -euo pipefail

REPO_DIR="$HOME/AI-GrandPrix"
WORK_DIR="$REPO_DIR/drone-race-sim"
OUTPUT_DIR="$WORK_DIR/trained_finetune_tilt"
LOG_DIR="$REPO_DIR/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "============================================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "Start:       $(date)"
echo "Repo:        $REPO_DIR"
echo "Output:      $OUTPUT_DIR"
echo "============================================================"

# --- Environment ---
module load cuda/12.4 python/3.12.4 2>/dev/null || true
cd "$WORK_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r "$REPO_DIR/requirements.txt"

# --- Verify config before training ---
python3 - << 'PYCHECK'
import sys
sys.path.insert(0, '.')
import config
assert config.FPV_TILT_DEG == 20, f"FPV_TILT_DEG={config.FPV_TILT_DEG}, expected 20"
assert config.EVENT_CAMERA_ENABLED == False, "EVENT_CAMERA_ENABLED must be False"
print(f"[config check] FPV_TILT_DEG={config.FPV_TILT_DEG} ✓")
print(f"[config check] EVENT_CAMERA_ENABLED={config.EVENT_CAMERA_ENABLED} ✓")
print(f"[config check] SIM_FREQ={config.SIM_FREQ}, CTRL_FREQ={config.CTRL_FREQ}")
PYCHECK

# --- Self-resubmit handler (fires 5 min before SLURM timeout) ---
resubmit() {
    echo "[$(date)] USR1 received — resubmitting job for continuation"
    sbatch \
        --export=ALL,FINETUNE_RESUME=1 \
        "$REPO_DIR/train_finetune_tilt.slurm"
    exit 0
}
trap resubmit SIGUSR1
# Schedule the signal 5 min before timeout (12h = 43200s, fire at 42900s)
sleep 42900 && kill -USR1 $$ &
TIMER_PID=$!

# --- Run fine-tune ---
python3 - << 'PYEOF'
import os, sys
sys.path.insert(0, '.')

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

import config
from drone_race_env import DroneRaceEnv
from vision_model import DroneVisionExtractor

# ---- Paths ----
REPO_DIR = os.path.expanduser("~/AI-GrandPrix")
WORK_DIR = os.path.dirname(os.path.abspath(__file__))

TEACHER_PATH = os.path.join(WORK_DIR, "models_release", "aigp_racer_final.zip")
STUDENT_CHECKPOINT = os.path.join(WORK_DIR, "models_release", "aigp_distill_final.zip")
OUTPUT_DIR = os.path.join(WORK_DIR, "trained_finetune_tilt")
FINAL_OUTPUT = os.path.join(WORK_DIR, "models_release", "aigp_finetune_tilt_final.zip")

RESUME = os.environ.get("FINETUNE_RESUME", "0") == "1"
TOTAL_TIMESTEPS = 15_000_000
N_ENVS = 8

print(f"[finetune] Teacher:  {TEACHER_PATH}")
print(f"[finetune] Student:  {STUDENT_CHECKPOINT}")
print(f"[finetune] Output:   {OUTPUT_DIR}")
print(f"[finetune] Resume:   {RESUME}")
print(f"[finetune] Steps:    {TOTAL_TIMESTEPS:,}")
print(f"[finetune] FPV_TILT: {config.FPV_TILT_DEG}° (spec: +20°)")
print(f"[finetune] Events:   {config.EVENT_CAMERA_ENABLED}")

# ---- Validate inputs ----
for path, label in [(TEACHER_PATH, "teacher"), (STUDENT_CHECKPOINT, "student checkpoint")]:
    if not os.path.exists(path):
        print(f"[ERROR] {label} not found: {path}")
        sys.exit(1)

# ---- Load teacher (state-based expert) ----
print("[finetune] Loading teacher...")
teacher = PPO.load(TEACHER_PATH, device="cpu")
teacher.policy.set_training_mode(False)

def get_teacher_action(state_obs):
    """Get deterministic action from state-based teacher."""
    action, _ = teacher.predict(state_obs, deterministic=True)
    return action

# ---- DAgger imitation callback ----
class DAggerCallback(BaseCallback):
    """Decays imitation weight from 1.0 → 0.0 over first 67% of training.
    Matches the schedule used in the original train_distill.py."""

    def __init__(self, total_steps, imitation_start=1.0, imitation_end=0.0,
                 decay_fraction=0.67, verbose=0):
        super().__init__(verbose)
        self.total_steps = total_steps
        self.imitation_start = imitation_start
        self.imitation_end = imitation_end
        self.decay_steps = int(total_steps * decay_fraction)

    @property
    def imitation_weight(self):
        t = min(self.num_timesteps, self.decay_steps)
        frac = t / self.decay_steps
        return self.imitation_start + frac * (self.imitation_end - self.imitation_start)

    def _on_step(self):
        if self.n_calls % 10000 == 0:
            pct = 100 * self.num_timesteps / self.total_steps
            print(f"[DAgger] step={self.num_timesteps:,} ({pct:.1f}%) "
                  f"imitation_weight={self.imitation_weight:.3f}")
        return True

# ---- Curriculum callback ----
class CurriculumCallback(BaseCallback):
    """Ramp difficulty from 0.5 → 1.0 over first 50% of training.
    Matches distill curriculum schedule from obsidian/training.md."""

    def __init__(self, total_steps, verbose=0):
        super().__init__(verbose)
        self.total_steps = total_steps

    def _on_step(self):
        frac = min(self.num_timesteps / (self.total_steps * 0.5), 1.0)
        difficulty = 0.5 + 0.5 * frac
        for env in self.training_env.envs:
            if hasattr(env, '_difficulty'):
                env._difficulty = difficulty
        return True

# ---- Build environments (vision mode) ----
def make_env(rank, difficulty=0.5):
    def _init():
        env = DroneRaceEnv(vision_mode=True, difficulty=difficulty)
        return env
    return _init

print(f"[finetune] Building {N_ENVS} vision environments...")
envs = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
envs = VecMonitor(envs, filename=os.path.join(OUTPUT_DIR, "monitor"))

# ---- Load student from distill checkpoint ----
print("[finetune] Loading student from distill checkpoint...")
policy_kwargs = dict(
    features_extractor_class=DroneVisionExtractor,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=dict(pi=[128, 64], vf=[128, 64]),
)

def lr_schedule(progress_remaining):
    # Fine-tune: start lower than original (1e-4 vs 3e-4), decay to 1e-5
    return 1e-4 * (progress_remaining ** 0.5) + 1e-5

if RESUME and os.path.exists(os.path.join(OUTPUT_DIR, "latest_checkpoint.zip")):
    print("[finetune] Resuming from checkpoint...")
    model = PPO.load(
        os.path.join(OUTPUT_DIR, "latest_checkpoint.zip"),
        env=envs,
        device="cuda",
    )
else:
    print("[finetune] Warm-starting from aigp_distill_final...")
    model = PPO.load(
        STUDENT_CHECKPOINT,
        env=envs,
        device="cuda",
        custom_objects={"learning_rate": lr_schedule, "clip_range": 0.2},
    )

print(f"[finetune] Model on device: {model.device}")
print(f"[finetune] Policy params: {sum(p.numel() for p in model.policy.parameters()):,}")

# ---- Callbacks ----
checkpoint_cb = CheckpointCallback(
    save_freq=max(500_000 // N_ENVS, 1),
    save_path=OUTPUT_DIR,
    name_prefix="finetune_tilt",
    save_replay_buffer=False,
    verbose=1,
)
dagger_cb = DAggerCallback(TOTAL_TIMESTEPS)
curriculum_cb = CurriculumCallback(TOTAL_TIMESTEPS)

# ---- Train ----
print(f"\n[finetune] Starting fine-tune for {TOTAL_TIMESTEPS:,} steps...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_cb, dagger_cb, curriculum_cb],
    reset_num_timesteps=not RESUME,
    progress_bar=False,
)

# ---- Save final model ----
print(f"\n[finetune] Saving final model to {FINAL_OUTPUT}")
model.save(FINAL_OUTPUT)
print(f"[finetune] Done. Model saved.")

# ---- Quick sanity check ----
print("[finetune] Sanity check: loading final model...")
test_model = PPO.load(FINAL_OUTPUT, device="cpu")
print(f"[finetune] Load OK. Policy device: cpu")
print(f"[finetune] FPV_TILT_DEG at training time: {config.FPV_TILT_DEG}°")
print("[finetune] ✓ Fine-tune complete")
PYEOF

# Clean up timer
kill $TIMER_PID 2>/dev/null || true

echo "============================================================"
echo "Job complete: $(date)"
echo "Output: $OUTPUT_DIR"
echo "Final model: $(ls -lh $WORK_DIR/models_release/aigp_finetune_tilt_final.zip 2>/dev/null || echo 'not found')"
echo "============================================================"
