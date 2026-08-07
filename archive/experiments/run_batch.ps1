# run_batch.ps1 — batch runner for the deterministic -0.6 leg-2 hold build.
# Runs the filt88 golden command in BLOCKS (sim drifts; restart the sim between blocks),
# auto-tags each run's rate, prints a Gate-3 verdict, and stops+beeps on registration.
#
# Usage (from the repo root):
#   cd "C:\Users\nohab\Desktop\AI GP\AI-GrandPrix-"
#   .\run_batch.ps1 -Block 4 -Start 89.1              # run 4, tags filt89.1..89.4
#   .\run_batch.ps1 -Block 4 -Start 89.5 -HoldDeg -0.4   # sweep variant
#
# RESTART THE SIM before each new block (every 3-4 runs) or runs go off-rate.

param(
    [int]    $Block   = 4,        # runs in this block (keep 3-4; sim drifts)
    [string] $Start   = "89.1",   # first tick-log index -> filt89.1.csv ...
    [double] $HoldDeg = -0.6      # --gate2-hold-fixed-deg (default -0.6, no need to pass)
)

$ErrorActionPreference = "Stop"
$maj, $min = $Start.Split(".")
$idx = [int]$min

for ($i = 0; $i -lt $Block; $i++) {
    $tag = "filt$maj.$idx"
    Write-Host "`n=== RUN $tag  (hold=$HoldDeg) ===" -ForegroundColor Cyan

    python tools/schedule_flier.py --coast-tube `
        --const-thrust 0.275 `
        --descent-bias-leg0 -0.005 --descent-bias-leg1 0.039 --descent-bias-leg2 0.024 `
        --gate-vert --gate-vert-size-min 0.05 `
        --vert-auth-down 0.045,0.015 --vert-auth-up 0.06,0.10 `
        --k-thrust-v 0.16 --kd-v 0.1 --k-gate-bank 1.0 `
        --post-gate-hold-s 1.5 --post-gate-hold-decay 0.5 `
        --thrust-slew 0.6 --thrust-slew-down 0.6 `
        --gate-max-bank-deg 11 --gate-max-bank-ag2 9 --gate-max-bank-ag2-left 0 `
        --post-gate1-bank 5 --post-gate2-bank -4.5 `
        --gate2-hold-fixed-deg $HoldDeg `
        --gate-bank-size-min-ag2 0.05 --gate-bank-full-size-ag2 0.15 `
        --gate2-exit-level-size 0.35 --gate2-exit-level-tau 0.10 `
        --gate-commit-size 0.16 --gate-commit-align 0.08 --gate-commit-stable-frames 3 `
        --gate-vert-commit-size 0.18 --gate-vert-commit-to-ag 0 `
        --leg2-entry-arrest 0.12 --leg2-entry-arrest-s 1.6 `
        --no-pitch-hold --log-tube `
        --tick-log "$tag.csv" 2>&1 | Tee-Object "$tag.log" | Out-Null

    # verdict
    python flight_report.py "$tag.csv"
    if ($LASTEXITCODE -eq 3) {
        Write-Host ">>> GATE 3 REGISTERED on $tag  <<<" -ForegroundColor Green
        [console]::beep(880,600)
        break
    }
    $idx++
}

Write-Host "`nBlock done. RESTART THE SIM before the next block." -ForegroundColor Yellow
Write-Host "Summary so far:" -ForegroundColor Yellow
Get-ChildItem "filt$maj.*.csv" | Sort-Object Name | ForEach-Object { python flight_report.py $_.Name }
