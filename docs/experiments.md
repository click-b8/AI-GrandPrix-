# Experiment history — what was tried, what survived

Every mechanism below shipped as a default-off flag, was tested in live flights (one change per test), and was either **frozen** (validated, guarded by tests) or **disproven** (left in code for reproducibility, documented here). This is the compressed log of ~600 flights of manual tuning plus the 546-attempt overnight batch.

## Eras

1. **RL era** (archived in `archive/rl-training/`): PPO/distillation training against a MuJoCo course model, DGX cluster guides, seed sweeps. Superseded when sim realities (vision-only observation contract, no position) made the direct-control vision-servo approach dominant. Kept as provenance.
2. **Open-loop era:** scheduled thrust/bank segments, no vision in the loop. Established the coast physics (no climb authority; pitch coasts to ~−17.8°) and the per-leg descent-bias backbone.
3. **Vision-servo era (`--coast-tube`)** — everything below.

## Validated and frozen

| mechanism | verdict |
|---|---|
| Altitude ladder (per-leg descent biases) | Backbone of Gates 1–2; legs 0–2 tuned |
| Post-gate holds (+5° / −4.5°) | Fixed both post-gate veers |
| Gate-1 exit climb fix, Gate-2 descent floor | Root-caused and frozen (see archived notes) |
| **Held-bank lever** — Gate-2 crossing bank predicts Gate-3 lateral (r = 0.87) → `--gate2-hold-fixed-deg -0.6` | The single most valuable tuning discovery; deterministic replacement for a run-dependent latched value |
| **Derivative-aware commit** (rate 0.13 / dwell 0.15 s) | Centered genuine latches; frozen. Post-campaign replay found a strictly better rule (never flown — see [findings.md](findings.md)) |
| **Gate-3 terminal descent** (Δ0.015 @ sz 0.25) | Converted the +0.53 high arrival to +0.28 and sinking |
| **Blind hold** (0.30 s) | Sustains the sink through the terminal blind coast |

## Disproven (kept in code, default-off)

| mechanism | why it failed |
|---|---|
| Upward flare before the gate (`--gate3-vert-flare*`) | Wrong model — the drone arrives *high*, not low; replay killed it before it cost more flights |
| Rail-tube lateral steering (`--tube-lateral`) | The rail detector rebuild changed the signal contract; the old law didn't transfer — hard-disarmed pending recalibration that never became the priority |
| Pitch-hold | Pitch is not usefully controllable in coast; qualifier always flew `--no-pitch-hold` |
| Frame-count commit stability (`--gate-commit-stable-frames`) | "3 frames" is 3× longer at 20 Hz than 60 Hz — superseded by the seconds-based dwell; inert in the frozen config (pinned by test) |
| More-descent-fixes-high (bigger deltas / size ramps) | Terminal traces showed the drone still sinking at gate loss; projections straddled center — more descent pushes fast-sink runs *low* |
| **"Median rate < 88 Hz selects the miss mode"** | The most instructive failure: a 77.8%-accurate split that dissolved under per-leg analysis — leg *transit time* separates the modes at 97.6%, and the best centered runs were low-rate machines flying the fast leg. Correlation ≠ mechanism, even at n = 302 |
| Ballast rate-pinning (`--log-tube` as deterministic load) | Reached the Gate-3 leg reliably (3/3 vs 1/3) but added its own load spike exactly at the crossing; bimodal, not in-band |
| PnP pose estimation | Frozen at feasibility-audit stage — right idea, wrong risk profile against the deadline |

## Method notes

- **Offline replay before live tests:** every threshold that mattered (commit rate/dwell, descent size, hold duration) was swept against logged runs before a flag was flown. The final latch investigation replayed 187k ticks across 302 runs and flew nothing.
- **Metric integrity:** the "max-size frame" crossing metric silently measured post-plane spurious detections; correcting it to the true crossing (deepest stable pre-loss frame) inverted a major conclusion. Lesson: audit the metric before trusting the trend.
- **One variable per flight; freeze-and-guard:** validated behavior immediately became the baseline that later experiments were forbidden to touch, enforced by ~530 tests including bit-for-bit default-off checks.
