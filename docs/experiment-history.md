# Experiment history — controller evolution

The campaign as a sequence of engineering decisions. Each stage: **Problem → Hypothesis → Change → Result → Decision.** Superseded eras (RL training, open-loop scheduling) are summarized at the end; family-by-family verdicts live in [experiments.md](experiments.md).

| # | stage | problem | hypothesis | change | result | decision |
|---|---|---|---|---|---|---|
| 1 | Baseline vision-servo | RL pivot left no flyable controller | A classical servo on blob centroid can fly gates | `--coast-tube`: per-leg schedule + gate-centering servo | Flew, erratic vertical | Adopt as platform |
| 2 | Altitude ladder | Uncontrolled altitude drift between gates | Per-leg descent biases can shape the profile without climb authority | `--descent-bias-legN` backbone | Gates 1–2 reachable | Freeze as vertical backbone |
| 3 | Gate-1 exit climb | Post-G1 balloon broke the G2 approach | Post-gate bank/hold shapes the exit | `--post-gate1-bank 5` + hold/decay | G1→G2 leg stabilized | Freeze |
| 4 | Gate-2 descent floor | Systematic G2 undershoot | Descent bias floor bug | Root-caused, corrected | G2 pass reliable | Freeze |
| 5 | Leg-2 entry arrest | High-energy G2 crossings destabilized the leg | Brief thrust arrest at entry smooths it | `--leg2-entry-arrest 0.12/1.6s` | Cleaner G3 approaches | Freeze |
| 6 | Exit-level latch | Bank residue at G2 exit varied | Level the bank once G2 fills the frame | `--gate2-exit-level-*` | Reduced exit spread | Freeze |
| 7 | G3 lateral variance | Same config → wildly different G3 arrival | It's not noise — some latent state carries over | Instrumented; regression hunt | **Held bank at G2 crossing predicts G3 u_f, r = 0.87** | The "lever" — biggest single discovery |
| 8 | Deterministic hold | Latched hold value varied run to run | Fix the hold to the regression's zero | `--gate2-hold-fixed-deg -0.6` | G3 lateral centered on clean runs | Freeze |
| 9 | Size-only commit | Terminal servo chased parallax spikes | Freeze steering when close | commit on `sz_f ≥ 0.16` | Latched on G2 residual (before G3 even seen) | Add arm precondition |
| 10 | Conditional commit + stability window | Latched on one-frame zero crossings | Require a centered *streak* | `--gate-commit-stable-frames 3` | Still latched on fast transients; frame counts are rate-dependent | Replace with rate-aware rule |
| 11 | **Derivative-aware commit** | A streak can't tell settled from fast-crossing | The *rate* of u_f can | `--gate-commit-rate-max 0.13 --gate-commit-stable-s 0.15` (real-time dwell) | Genuine latches centered (B2: −0.026); false latches rejected | Freeze; guard with tests |
| 12 | Terminal vertical | Arrives high at G3 (corrected metric — a max-size-frame bug had said low) | Cut thrust on the deep approach | `--gate3-vert-descent` Δ0.015 @ sz 0.25 | +0.53 → +0.28 at loss, still sinking | Freeze |
| 13 | **Blind descent hold** | Descent released at gate loss, 2.4 m short | Hold the same cut through the blind coast | `--gate3-vert-descent-hold-s 0.30` | Sink sustained; vertical-low misses: 3/302 | Freeze |
| 14 | Rate instability | Every G3-reaching run off-rate; crossings dipped to 37–48 Hz | Stray sim processes + power state | PID-tracked lifecycle + port isolation; power plans; ballast experiments | Merged-stream class eliminated; rate never fully pinned | Lifecycle frozen; rate → analysis variable |
| 15 | **Automated overnight batch** | Manual attempts too slow for the deadline | If variance can qualify, volume finds it; if not, the dataset proves why | `run_qualifier_batch.ps1 -KeepSimAlive`: arm-before-GO, SendInput reset, classification, purge | **546 attempts, 0 registrations, 0 interventions** | Volume hypothesis falsified — by design, informatively |
| 16 | **302-run forensic replay** | Why zero? | The answer is in the tick logs | Offline replay: population analysis, latch reconstruction, 256-rule sweep | Two deterministic populations; fork precedes commit; better commit rule found (86:1) | Findings frozen ([findings.md](findings.md)); flights ended before the discriminating experiment |

## Superseded eras

**RL training** (`archive/rl-training/`): PPO + distillation against a MuJoCo course model, DGX-scale runs. Ended when the qualifier's true observation contract (vision-only, no pose) made end-to-end learning the higher-risk path against the deadline. **Open-loop era**: scheduled thrust/bank with no vision feedback; established the coast physics (no climb authority; −17.8° pitch coast) and the per-leg structure that the ladder later formalized.

## Method invariants

One variable per flight test; replay before fly for every threshold; validated behavior frozen immediately and guarded by regression tests; every experimental mechanism a default-off flag so any historical configuration reproduces from a command line.
