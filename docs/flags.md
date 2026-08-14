# CLI flag reference — `tools/schedule_flier.py`

The controller exposes ~152 flags. That number is the experiment log, not sprawl: every mechanism ever tested ships as a **default-off flag**, so any historical configuration reproduces from a command line, and disproven ideas remain runnable rather than deleted. This page triages them; the exact frozen VQ1 command is in [REPRODUCE.md](../results/overnight-2026-08-03/REPRODUCE.md).

**Status legend:** ✅ active in the final VQ1 controller · 🧪 experimental (default-off, functional) · ❌ ruled out (default-off, kept for reproducibility) · 🔧 diagnostic only

## Core flight mode

| flag | purpose | default | VQ1 value | status |
|---|---|---|---|---|
| `--coast-tube` | The competition mode: vision-servo coast pipeline | off | on | ✅ |
| `--const-thrust` | Base thrust | — | 0.275 | ✅ |
| `--no-pitch-hold` | Disable pitch control (pitch is uncontrollable in coast) | — | on | ✅ |
| `--max-s` | Flight timeout | 35 | 35 | ✅ |
| `--tick-log` | Telemetry CSV path (~94 columns) | — | per-run | ✅ |

## Vertical control

| flag | purpose | default | VQ1 value | status |
|---|---|---|---|---|
| `--descent-bias-leg0/1/2` | Per-leg descent ladder (the vertical backbone) | none | −0.005 / 0.039 / 0.024 | ✅ |
| `--descent-bias-leg3/4/5` | Later legs — never reached tuning | none | unset | 🧪 |
| `--gate-vert`, `--gate-vert-size-min` | Vision vertical trim enable + gate | off / — | on / 0.05 | ✅ |
| `--k-thrust-v`, `--kd-v` | PD gains on `v_f` | — | 0.16 / 0.1 | ✅ |
| `--vert-auth-down`, `--vert-auth-up` | Asymmetric trim authority | — | 0.045,0.015 / 0.06,0.10 | ✅ |
| `--thrust-slew`, `--thrust-slew-down` | Thrust slew limits (per-second) | — | 0.6 / 0.6 | ✅ |
| `--gate-vert-commit-size`, `--gate-vert-commit-to-ag` | Vertical commit scoping | — | 0.18 / 0 | ✅ |

## Lateral control

| flag | purpose | default | VQ1 value | status |
|---|---|---|---|---|
| `--k-gate-bank` | Gate-centering servo gain | — | 1.0 | ✅ |
| `--gate-max-bank-deg` | Global servo clamp | — | 11° | ✅ |
| `--gate-max-bank-ag2`, `--gate-max-bank-ag2-left` | Gate-3-leg clamp, asymmetric (+0/−9°) | — | 9 / 0 | ✅ |
| `--gate-bank-size-min-ag2`, `--gate-bank-full-size-ag2` | Servo authority ramp by gate size | — | 0.05 / 0.15 | ✅ |
| `--kd-lat` | Lateral derivative term | 0 | unset | 🧪 |
| `--tube-lateral` (+ family) | Rail-tube steering — hard-disarmed after the detector rebuild changed its signal contract | off | off | ❌ |

## Post-gate behavior

| flag | purpose | default | VQ1 value | status |
|---|---|---|---|---|
| `--post-gate-hold-s`, `--post-gate-hold-decay` | Decaying scheduled bank after each gate | — | 1.5 / 0.5 | ✅ |
| `--post-gate1-bank`, `--post-gate2-bank` | Per-gate hold values | — | +5° / −4.5° | ✅ |
| `--gate2-hold-fixed-deg` | **The lever**: deterministic held bank at the G2 crossing (predicts G3 lateral, r = 0.87) | latched | **−0.6°** | ✅ |
| `--gate2-exit-level-size`, `--gate2-exit-level-tau` | Level the bank as G2 fills the frame | — | 0.35 / 0.10 | ✅ |
| `--leg2-entry-arrest`, `--leg2-entry-arrest-s` | Thrust arrest entering leg 2 | — | 0.12 / 1.6 | ✅ |

## Commit logic (close-range servo fade)

| flag | purpose | default | VQ1 value | status |
|---|---|---|---|---|
| `--gate-commit-size` | Eligibility size (arms only after the previous gate's residual clears below it) | 0.16 | 0.16 | ✅ |
| `--gate-commit-align` | Alignment band \|u_f\| | — | 0.08 | ✅ |
| `--gate-commit-rate-max`, `--gate-commit-stable-s` | Derivative-aware rule: low-rate + aligned for a real-time dwell | 0 / 0 | **0.13 / 0.15** | ✅ |
| `--gate-commit-stable-frames` | Frame-count streak — superseded; **inert when the seconds rule is active** (pinned by test) | 3 | 3 (inert) | ❌ |
| *(replay-validated, never flown: size 0.20 / align 0.12 / rate 0.16 — see [findings.md](findings.md))* | | | | |

## Terminal Gate-3 logic

| flag | purpose | default | VQ1 value | status |
|---|---|---|---|---|
| `--gate3-vert-descent`, `--gate3-vert-descent-delta`, `--gate3-vert-descent-size` | Fixed thrust cut on the deep G3 approach | off | on / 0.015 / 0.25 | ✅ |
| `--gate3-vert-descent-hold-s` | Bounded blind hold: sustain the cut after gate loss | 0 | **0.30** | ✅ |
| `--gate3-vert-flare*` family | Upward flare before the gate — wrong sign of the problem (drone arrives high) | off | off | ❌ |

## Diagnostics / logging

| flag | purpose | status |
|---|---|---|
| `--log-tube` | Rail-detector logging pass (~3.4 ms/frame). Flight-inert; doubled late-campaign as a deterministic load "ballast" in rate experiments | 🔧 (on in VQ1 command) |
| `--far-gate-reject`, `--gate-filter-tau` | Detector hygiene | ✅ |
| Probe/calibration-era flags (`--calibrate`, `--cruise`, `--probe-*`, pitch-hold family, `--ff-backbone` era, open-loop turn flags) | Earlier eras; superseded | ❌ |

## Reading guide

If a flag isn't in the frozen VQ1 command and isn't marked ✅ here, it is historical. The ❌ families are documented, not deleted, because reproducing *any* past run — including the failed experiments — from its logged command line was a campaign invariant.
