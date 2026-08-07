<#
.SYNOPSIS
    Zero-Claude-involvement bootstrap for AI Grand Prix VQ1 training on a
    fresh Windows machine (work render box, unknown prior state).
.DESCRIPTION
    venv -> pinned installs -> CUDA check (warn, don't die) -> checkpoint
    SHA256 verify -> fast test subset + 100-step smoke -> launch the seed-1
    resume under tools/guardian.py. Safe to re-run (venv/checkpoint steps
    skip if already satisfied).
.PARAMETER SaveDir
    Training save dir (checkpoints/, tb_logs/, best_model/). Default matches
    the existing surface_seed1 convention.
.PARAMETER Seed
    RNG seed passed to train_vision.py.
.PARAMETER NEnvs
    Parallel envs. Default 4 -- see the WGL resume-race note before raising
    this on Windows (DGX_TRAINING_GUIDE.md has the full writeup).
.PARAMETER SkipTests
    Skip the fast pytest subset (not recommended; use only for a quick rerun
    once the machine is already known-good).
.PARAMETER SkipSmoke
    Skip the 100-step smoke test.
#>
param(
    [string]$SaveDir = "surface_seed1",
    [int]$Seed = 1,
    [int]$NEnvs = 4,
    [switch]$SkipTests,
    [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Warn2($msg) { Write-Host "WARNING: $msg" -ForegroundColor Yellow }

if (-not (Test-Path (Join-Path $RepoRoot "drone_race_env.py"))) {
    Write-Error "drone_race_env.py not found next to tools/ -- this script must live at <repo>/tools/bootstrap_train.ps1"
}

# --- 1. Python + venv ---
Write-Step "Python venv"
$candidates = @(
    @{Exe = "py"; Args = @("-3.12") },
    @{Exe = "py"; Args = @("-3.11") },
    @{Exe = "py"; Args = @("-3") },
    @{Exe = "python3"; Args = @() },
    @{Exe = "python"; Args = @() }
)
$Selected = $null
foreach ($c in $candidates) {
    if (Get-Command $c.Exe -ErrorAction SilentlyContinue) {
        try {
            $verOut = & $c.Exe @($c.Args + @("--version")) 2>&1
            if ($LASTEXITCODE -eq 0) { $Selected = $c; $Selected.Ver = "$verOut"; break }
        }
        catch {}
    }
}
if (-not $Selected) {
    Write-Error "No python interpreter found on PATH (tried py -3.12, py -3.11, py -3, python3, python)."
}
Write-Host "Using: $($Selected.Exe) $($Selected.Args -join ' ')  ($($Selected.Ver))"
if ($Selected.Ver -notmatch "3\.(11|12)\.") {
    Write-Warn2 "DGX_TRAINING_GUIDE.md recommends Python 3.11/3.12 (NOT 3.14) for wheel availability. Found: $($Selected.Ver). Continuing anyway -- this exact stack (torch/mujoco/sb3) is proven working on 3.14 on the desktop machine. If a pip install below fails, install 3.12 and rerun."
}

$venvPath = Join-Path $RepoRoot ".venv"
if (-not (Test-Path $venvPath)) {
    & $Selected.Exe @($Selected.Args + @("-m", "venv", $venvPath))
    if ($LASTEXITCODE -ne 0) { Write-Error "venv creation failed" }
}
else {
    Write-Host "venv already exists at $venvPath -- reusing"
}
$venvPy = Join-Path $venvPath "Scripts\python.exe"

# --- 2. Pinned installs (desktop's known-good stack) ---
Write-Step "Installing pinned stack"
& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install torch --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { Write-Error "torch install failed" }
& $venvPy -m pip install "mujoco>=3.10" "stable-baselines3>=2.9" gymnasium Pillow tensorboard "pymavlink>=2.4" pytest
if ($LASTEXITCODE -ne 0) { Write-Error "pip install of core stack failed" }

# --- 3. CUDA check: warn and continue, never die (some render boxes are CPU-only) ---
Write-Step "CUDA check"
$cudaOut = (& $venvPy -c "import torch; print(torch.cuda.is_available())").Trim()
if ($cudaOut -ne "True") {
    Write-Warn2 "CUDA not available (torch.cuda.is_available() = $cudaOut). Training will run on CPU -- slower, but expected on some machines. Continuing."
}
else {
    Write-Host "CUDA available."
}

# --- 4. SHA256-verify the resume checkpoint ---
Write-Step "Verifying models_release/seed1_570k_resume.zip"
# Update this hash any time the checkpoint file at this path is intentionally
# replaced (e.g. a newer resume snapshot committed under the same name).
$ExpectedHash = "E581EDD95228BFD164A7A3C5730B3D9EDDA8704A2E1A7450BFBE16BED24D990F"
$ckptPath = Join-Path $RepoRoot "models_release\seed1_570k_resume.zip"
if (-not (Test-Path $ckptPath)) {
    Write-Error "Checkpoint not found: $ckptPath"
}
$actualHash = (Get-FileHash -Algorithm SHA256 $ckptPath).Hash
if ($actualHash -ne $ExpectedHash) {
    Write-Error "SHA256 mismatch for seed1_570k_resume.zip`n  expected: $ExpectedHash`n  actual:   $actualHash`nDo not train on a corrupted/unexpected checkpoint -- re-pull from origin."
}
Write-Host "Checkpoint hash OK."

# --- 5. Fast test subset ---
if (-not $SkipTests) {
    Write-Step "Fast test subset"
    # Pure env/observation/attitude/track-transform logic -- no MuJoCo GL
    # context, no mock MAVLink socket, no deploy-path fixtures. Excludes
    # test_mavlink_compliance.py / test_vq1_readiness.py (deploy-path,
    # heavier setup) -- run those separately when validating a deploy, not a
    # fresh training-machine bootstrap.
    & $venvPy -m pytest tests/test_observation_contract.py tests/test_vision_obs_10d.py tests/test_track_transform.py tests/test_attitude_filter.py -v
    if ($LASTEXITCODE -ne 0) { Write-Error "Fast test subset failed -- fix before training on this machine." }
}
else {
    Write-Host "Skipping tests (-SkipTests)"
}

# --- 6. 100-step smoke test ---
if (-not $SkipSmoke) {
    Write-Step "100-step smoke test"
    $smokeDir = Join-Path $RepoRoot "_bootstrap_smoke"
    if (Test-Path $smokeDir) { Remove-Item -Recurse -Force $smokeDir }
    & $venvPy train_vision.py --save-dir $smokeDir --n-envs 1 --total-timesteps 100 --n-steps 64 --batch-size 32 --eval-freq 0 --save-freq 100 --keep-last 1 --no-progress-bar
    $smokeExit = $LASTEXITCODE
    if (Test-Path $smokeDir) { Remove-Item -Recurse -Force $smokeDir }
    if ($smokeExit -ne 0) { Write-Error "Smoke test failed -- training mechanics are broken on this machine." }
    Write-Host "Smoke test OK."
}
else {
    Write-Host "Skipping smoke test (-SkipSmoke)"
}

# --- 7. Launch under the guardian ---
Write-Step "Launching guardian-supervised training"
# Windows note -- the resume-path WGL race: SubprocVecEnv workers each build
# their own WGL context. On RESUME (PPO.load() + N worker processes all
# initializing near-simultaneously) high n-envs has raced the Windows GL
# driver's context creation and crashed the run before the first rollout.
# n-envs=4 has been reliable; raise it only after a resume has succeeded once
# at this count, and bump gradually. Linux/EGL workers create independent
# headless contexts and do not hit this -- see bootstrap_train.sh, which
# defaults higher.
Write-Host "Starting at -NEnvs $NEnvs (conservative, Windows WGL resume-race guard). See DGX_TRAINING_GUIDE.md troubleshooting section before raising it."

& $venvPy tools\guardian.py --save-dir $SaveDir -- --seed $Seed --n-envs $NEnvs --keep-last 50
