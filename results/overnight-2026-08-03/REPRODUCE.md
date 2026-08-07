# Reproducing the overnight batch — 2026-08-03

Everything needed to re-run or re-analyze the final qualification campaign. **Simulator access note:** the AI-GP simulator requires an online account and qualifier access, which ended at the competition deadline (2026-08-03). No post-deadline flights were possible; the *analysis* pipeline below remains fully reproducible from the shipped data.

## Environment assumptions

- Windows 10/11, PowerShell 5.1, Python 3.11+ with `numpy` and `pymavlink` (`pip install -r requirements.txt`).
- AI-GP Simulator v1.0.3385 installed; `FlightSim.exe` is a launcher — the real process is `FlightSim\Binaries\Win64\DCGame-Win64-Shipping.exe`, which owns UDP 14560 (MAVLink) + 5601 (vision). The batch script's `-SimExe` parameter points at the launcher.
- Simulator logged in manually, qualifier course loaded, drone at the start gate, fullscreen, machine set to never sleep, lock screen off.
- Hardware used: ARM Surface (x86 translation) — loop rate 51–112 Hz; on native x86 expect different rate behavior (see docs/findings.md §rate).

## The exact batch command

```powershell
.\automation\run_qualifier_batch.ps1 -KeepSimAlive -Mode holdblind -HoldSeconds 0.30
```

Defaults that matter: `-MaxAttempts 1000`, `-RateBallast:$true` (adds `--log-tube` as deterministic per-frame load), `-RestartDelaySeconds 2.0`, reset = `Esc`,`Down`,`Enter` via SendInput. The batch stops **only** on the sim's race-finish signal, `MaxAttempts`, or SIM LOST.

## The frozen controller command it flies

```
python -u tools/schedule_flier.py --coast-tube --const-thrust 0.275
  --descent-bias-leg0 -0.005 --descent-bias-leg1 0.039 --descent-bias-leg2 0.024
  --gate-vert --gate-vert-size-min 0.05 --vert-auth-down 0.045,0.015 --vert-auth-up 0.06,0.10
  --k-thrust-v 0.16 --kd-v 0.1 --k-gate-bank 1.0 --post-gate-hold-s 1.5 --post-gate-hold-decay 0.5
  --thrust-slew 0.6 --thrust-slew-down 0.6 --gate-max-bank-deg 11 --gate-max-bank-ag2 9
  --gate-max-bank-ag2-left 0 --post-gate1-bank 5 --post-gate2-bank -4.5 --gate2-hold-fixed-deg -0.6
  --gate-bank-size-min-ag2 0.05 --gate-bank-full-size-ag2 0.15
  --gate2-exit-level-size 0.35 --gate2-exit-level-tau 0.10
  --gate-commit-size 0.16 --gate-commit-align 0.08 --gate-commit-stable-frames 3
  --gate-commit-rate-max 0.13 --gate-commit-stable-s 0.15
  --gate-vert-commit-size 0.18 --gate-vert-commit-to-ag 0
  --leg2-entry-arrest 0.12 --leg2-entry-arrest-s 1.6
  --gate3-vert-descent --gate3-vert-descent-delta 0.015 --gate3-vert-descent-size 0.25
  --gate3-vert-descent-hold-s 0.30
  --no-pitch-hold --log-tube --tick-log <run>.csv
```

(The batch's `Get-FlightArgs` builds exactly this; treat that function as the source of truth.)

## Data in this directory

| item | contents |
|---|---|
| `batch_summary.csv` | 597 rows (546 overnight from 03:30:43; earlier rows are pre-fix debugging — filter `timestamp >= 2026-08-03T03:30:43`) — one row per attempt, 38 columns, written before any purge |
| `json/` | per-run `flight_report --json` records (552) |
| `top10_traces/` | full tick CSV+log for the 10 closest approaches (runs 329, 508, 241, 61, 104, 570, 509, 62, 478, 156) |
| `keepers/` | the 3 clean-verdict deep approaches (169, 198, 412) |
| `gate3_leg_traces/` *(Release asset only)* | all 302 Gate-3-leg full traces + strays — see below |

**Full raw dataset:** `vq1-overnight-raw-dataset-2026-08-03.tar.gz` — GitHub Release asset, SHA-256 recorded alongside it in the release notes and in `*.sha256`. Contains this entire directory including `gate3_leg_traces/` and `diagnostic/`.

## Analysis pipeline

```bash
# one-run verdict + JSON (what the batch ran per attempt)
python analysis/flight_report.py <run>.csv --json <run>.json --descent-size 0.25

# the funnel / clustering / postmortem numbers: aggregate batch_summary.csv
#   overnight filter: rows from timestamp 2026-08-03T03:30:43 onward
# the forensic replay: per-tick reconstruction over gate3_leg_traces/*.csv
#   (relevant columns: t, active_gate, u_f, sz_f, size_frac, du_f, rate_ok, align_ok,
#    commit_stable_s, commit_latched, commit_k, gate_cmd_deg, des_roll_deg, est_roll_deg,
#    loop_hz, det_hz, race_fin)
```

The published figures (docs/img/) derive from: the funnel and factor histogram from `batch_summary.csv`; the duration/scatter/latch figures from per-run features extracted from the leg traces (leg window = first `active_gate==2` tick to the first sustained 3-tick `size_frac==0` loss after a deep detection; true crossing = deepest stable frame before that loss).

## Known simulator limitations

- Online PGOS login is manual and cannot be automated headlessly; session tokens live in the Windows profile.
- One race per reset; the pause-menu RESTART is the only programmatic-adjacent reset (keystrokes).
- Race state is authoritative from the sim: gate registration (`active_gate` advance) and finish (`race_fin`) come from its ENCAPSULATED_DATA stream; no ground-truth pose is ever exposed.
- Loop rate is a property of the host machine, not the sim, and materially affects controller behavior (docs/findings.md).
