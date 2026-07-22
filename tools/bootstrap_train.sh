#!/usr/bin/env bash
# Zero-Claude-involvement bootstrap for AI Grand Prix VQ1 training on a
# fresh Linux machine (DGX or otherwise, unknown prior state).
#
# venv -> pinned installs -> CUDA check (warn, don't die) -> checkpoint
# SHA256 verify -> fast test subset + 100-step smoke -> launch the seed-1
# resume under tools/guardian.py. Safe to re-run.
#
# Usage: tools/bootstrap_train.sh [--save-dir DIR] [--seed N] [--n-envs N]
#                                  [--skip-tests] [--skip-smoke]

set -eo pipefail

SAVE_DIR="surface_seed1"
SEED=1
N_ENVS=8   # Linux/EGL: independent per-worker headless contexts, no WGL race
           # (see bootstrap_train.ps1). DGX multi-GPU boxes can raise this to
           # 16-32 per DGX_TRAINING_GUIDE.md section 2.3 once this default is
           # confirmed working.
SKIP_TESTS=0
SKIP_SMOKE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --save-dir) SAVE_DIR="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --n-envs) N_ENVS="$2"; shift 2 ;;
        --skip-tests) SKIP_TESTS=1; shift ;;
        --skip-smoke) SKIP_SMOKE=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

step() { printf '\n=== %s ===\n' "$1"; }
warn() { printf 'WARNING: %s\n' "$1" >&2; }

if [ ! -f "drone_race_env.py" ]; then
    echo "drone_race_env.py not found next to tools/ -- this script must live at <repo>/tools/bootstrap_train.sh" >&2
    exit 1
fi

# --- 1. Python + venv ---
step "Python venv"
PYTHON_BIN=""
for cand in python3.12 python3.11 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        PYTHON_BIN="$cand"
        break
    fi
done
if [ -z "$PYTHON_BIN" ]; then
    echo "No python interpreter found on PATH (tried python3.12, python3.11, python3, python)." >&2
    exit 1
fi
PY_VER="$("$PYTHON_BIN" --version 2>&1)"
echo "Using: $PYTHON_BIN ($PY_VER)"
case "$PY_VER" in
    *"3.11."*|*"3.12."*) ;;
    *) warn "DGX_TRAINING_GUIDE.md recommends Python 3.11/3.12 (NOT 3.14) for wheel availability. Found: $PY_VER. Continuing anyway -- this exact stack (torch/mujoco/sb3) is proven working on 3.14 on the desktop machine. If a pip install below fails, install 3.12 and rerun." ;;
esac

VENV_PATH="$REPO_ROOT/.venv"
if [ ! -d "$VENV_PATH" ]; then
    "$PYTHON_BIN" -m venv "$VENV_PATH"
else
    echo "venv already exists at $VENV_PATH -- reusing"
fi
VENV_PY="$VENV_PATH/bin/python"

# --- 2. Pinned installs (desktop's known-good stack) ---
step "Installing pinned stack"
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu126
"$VENV_PY" -m pip install "mujoco>=3.10" "stable-baselines3>=2.9" gymnasium Pillow tensorboard "pymavlink>=2.4" pytest

# --- 3. CUDA check: warn and continue, never die ---
step "CUDA check"
CUDA_OUT="$("$VENV_PY" -c 'import torch; print(torch.cuda.is_available())')"
if [ "$CUDA_OUT" != "True" ]; then
    warn "CUDA not available (torch.cuda.is_available() = $CUDA_OUT). Training will run on CPU -- slower, but expected on some machines. Continuing."
else
    echo "CUDA available."
fi

# --- 4. MANDATORY headless render gotcha ---
step "Headless render (MUJOCO_GL=egl)"
export MUJOCO_GL=egl
echo "MUJOCO_GL=egl exported for this process and the guardian/train_vision.py children it launches."

# --- 5. SHA256-verify the resume checkpoint ---
step "Verifying models_release/seed1_570k_resume.zip"
# Update this hash any time the checkpoint file at this path is intentionally
# replaced (e.g. a newer resume snapshot committed under the same name).
EXPECTED_HASH="e581edd95228bfd164a7a3c5730b3d9edda8704a2e1a7450bfbe16bed24d990f"
CKPT_PATH="models_release/seed1_570k_resume.zip"
if [ ! -f "$CKPT_PATH" ]; then
    echo "Checkpoint not found: $CKPT_PATH" >&2
    exit 1
fi
if command -v sha256sum >/dev/null 2>&1; then
    ACTUAL_HASH="$(sha256sum "$CKPT_PATH" | awk '{print $1}')"
else
    ACTUAL_HASH="$("$VENV_PY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$CKPT_PATH")"
fi
ACTUAL_HASH_LOWER="$(echo "$ACTUAL_HASH" | tr '[:upper:]' '[:lower:]')"
if [ "$ACTUAL_HASH_LOWER" != "$EXPECTED_HASH" ]; then
    echo "SHA256 mismatch for seed1_570k_resume.zip" >&2
    echo "  expected: $EXPECTED_HASH" >&2
    echo "  actual:   $ACTUAL_HASH_LOWER" >&2
    echo "Do not train on a corrupted/unexpected checkpoint -- re-pull from origin." >&2
    exit 1
fi
echo "Checkpoint hash OK."

# --- 6. Fast test subset ---
if [ "$SKIP_TESTS" -eq 0 ]; then
    step "Fast test subset"
    # Pure env/observation/attitude/track-transform logic -- no mock MAVLink
    # socket, no deploy-path fixtures. Excludes test_mavlink_compliance.py /
    # test_vq1_readiness.py (deploy-path, heavier setup) -- run those
    # separately when validating a deploy, not a fresh training-machine
    # bootstrap.
    "$VENV_PY" -m pytest tests/test_observation_contract.py tests/test_vision_obs_10d.py tests/test_track_transform.py tests/test_attitude_filter.py -v
else
    echo "Skipping tests (--skip-tests)"
fi

# --- 7. 100-step smoke test ---
if [ "$SKIP_SMOKE" -eq 0 ]; then
    step "100-step smoke test"
    SMOKE_DIR="$REPO_ROOT/_bootstrap_smoke"
    rm -rf "$SMOKE_DIR"
    "$VENV_PY" train_vision.py --save-dir "$SMOKE_DIR" --n-envs 1 --total-timesteps 100 \
        --n-steps 64 --batch-size 32 --eval-freq 0 --save-freq 100 --keep-last 1 --no-progress-bar
    rm -rf "$SMOKE_DIR"
    echo "Smoke test OK."
else
    echo "Skipping smoke test (--skip-smoke)"
fi

# --- 8. Launch under the guardian ---
step "Launching guardian-supervised training"
echo "Starting at --n-envs $N_ENVS. Linux/EGL workers build independent headless"
echo "contexts (no WGL race) -- see DGX_TRAINING_GUIDE.md section 2.3 to raise this"
echo "toward 16-32 on a multi-GPU DGX box."

exec "$VENV_PY" tools/guardian.py --save-dir "$SAVE_DIR" -- --seed "$SEED" --n-envs "$N_ENVS" --keep-last 50
