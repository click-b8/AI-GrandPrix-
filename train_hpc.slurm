#!/bin/bash
#SBATCH --job-name=aigp-expert
#SBATCH --output=train_hpc_%j.log
#SBATCH --error=train_hpc_%j.err
#SBATCH --partition=shortq7-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:55:00
#SBATCH --signal=B:USR1@300
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nbrande2020@fau.edu

# Trap USR1 signal (sent 5 min before time limit) to resubmit
resubmit() {
    echo "Time limit approaching, resubmitting..."
    sbatch ~/drone-race-sim/train_hpc.slurm
    exit 0
}
trap resubmit USR1

echo "Job started: $(date)"
echo "Node: $(hostname)"

# Load modules
module load cuda/12.4.0-gcc-13.2.0-shyinv2
module load python/3.12.4-gcc-13.2.0-ryucfix

nvidia-smi

# MuJoCo headless rendering
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH}

# Activate venv
source ~/venv_aigp/bin/activate

# Verify GPU
python3 -c 'import torch; print("CUDA:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'

cd ~/drone-race-sim

# === 3-STAGE PIPELINE ===
# Stage 1: State expert (fast, ~20M steps, no rendering needed)
# Stage 2: Distill to vision policy (50M steps, uses expert + optional warmstart)
# Each stage auto-skips if already complete.

echo "=== Stage 1: State-based expert ==="
python3 train_state.py &
TRAIN_PID=$!
wait $TRAIN_PID

echo "=== Stage 2: Distill expert -> vision policy ==="
python3 train_distill.py &
TRAIN_PID=$!
wait $TRAIN_PID

echo "Job finished: $(date)"

# Resubmit if stages not yet complete
sbatch ~/drone-race-sim/train_hpc.slurm
