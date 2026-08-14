# Engineering Postmortem — AI-GP Virtual Qualifier R1

**Outcome: did not qualify. Gate 3 never registered in 546 automated attempts.** This document is the formal account: what was attempted, what was learned, what remains open. Every number traces to `results/overnight-2026-08-03/` or the test suite.

## Objective

Fly the 6-waypoint VQ1 course (start, gates 1–4, finish) fully autonomously, vision-only, before the qualification deadline (2026-08-03).

## Constraints

Camera-only sensing (blob centroid + size — no pose); no climb authority (thrust ceiling with an uncontrollable −17.8° nose-down coast); bank ±11° (Gate-3 leg: asymmetric +0/−9°); terminal blindness (gate exits the frame ~2.4 m before its plane); host loop rate 39–117 Hz and thermally unstable; one race per manual sim reset until the reset was automated; hard deadline.

## Final architecture

Classical vision-servo pipeline (see [architecture.md](architecture.md), [controller.md](controller.md)): per-leg feed-forward bank + decaying post-gate holds; lateral gate-centering servo with a derivative-aware close-range commit latch; per-leg descent ladder + PD vertical trim + Gate-3 terminal descent with a bounded blind hold; ~94-column tick telemetry; unattended batch campaign automation; offline replay forensics.

## Major bottlenecks, in the order they were broken

1. **Gates 1–2 tuning** — solved by the altitude ladder, post-gate holds, and two root-caused fixes (Gate-1 exit climb; Gate-2 descent floor).
2. **Run-to-run lateral variance at Gate 3** — traced to the *signed bank held at the Gate-2 crossing* predicting Gate-3 lateral arrival (r = 0.87); fixed deterministically (`--gate2-hold-fixed-deg -0.6`).
3. **False commit latches** — a frame-count stability rule latched on transients and meant different things at different loop rates; replaced by the derivative-aware, real-time-dwell rule.
4. **Vertical high-arrival at Gate 3** — a metric bug (post-plane parallax spikes polluting the "crossing" measurement) had inverted the diagnosis; the corrected true-crossing metric revealed a tight systematic high bias, addressed by the terminal descent + blind hold.
5. **Environment nondeterminism** — stray simulator processes merging UDP streams (fixed by PID-tracked lifecycle management with port-isolation assertions) and battery/power-plan-dependent loop rate (never fully controlled; became an analysis variable).
6. **Gate-3 registration** — never broken. See root cause below.

## Hypotheses tested (selected)

| hypothesis | verdict | evidence |
|---|---|---|
| Gate-3 lateral variance is random noise | **Rejected** — deterministic function of held bank at G2 | r = 0.87 regression across the tuning campaign |
| Drone arrives low at Gate 3 (flare needed) | **Rejected** — arrives high | corrected true-crossing metric |
| More descent fixes the high arrival | **Rejected** — projections straddle center; more descent pushes fast-sink runs low | terminal traces of clean deep runs |
| Registration is a volume game (variance harvest) | **Rejected** — 0-for-546 caps p(success) < 0.5%/attempt | overnight campaign |
| Median loop rate ≥ 88 Hz selects the miss mode | **Rejected** — 77.8% proxy; leg transit time separates at 97.6%; best centered runs were low-rate | per-leg reanalysis of 302 runs |
| Commit latch causes the straight trajectory | **Rejected** — 67 unlatched runs flew straight; 14 latched runs veered | latch replay (187k ticks) |
| Commit latch improves terminal precision | **Confirmed** — median |u_f| 0.088 latched vs 0.254 unlatched | same replay |
| A better commit rule exists | **Confirmed offline, never flown** — 0.20/0.12/0.16 rule, 86:1 confusion | 256-combination sweep |

## What worked

The Gates-1–2 stack (92.3% / 55.3% pass rates); the held-bank lever; the derivative-aware commit as a terminal instrument; the terminal descent + blind hold (vertical-low misses: 3 of 302 — the descent never overshot); the automation (546 attempts, zero interventions, zero runaways); the offline-replay methodology (every threshold that mattered was swept against logged data before flying; the final investigation flew nothing at all).

## What failed

The upward flare (wrong sign of the problem); rail-tube lateral steering (signal contract changed under it); frame-count commit stability (rate-dependent); rate ballast as a stabilizer (bimodal, added its own crossing-load spike); the volume-harvest strategy (falsified by its own dataset — which was the graceful-failure design working as intended).

## Instrumentation & automation

The ~94-column tick log made the entire forensic layer possible — every controller decision input is reconstructable offline. `flight_report.py` gave every run a machine-readable verdict (the batch's classification contract). The batch system contributed the lifecycle discovery (launcher vs shipping process; port-isolation assertions), SendInput scancode injection with arm-before-GO sequencing, and self-purging disk management that recorded before deleting.

## The overnight dataset & root-cause findings

546 attempts; 302 full Gate-3-leg traces. Findings, stated at their supported strength:

1. **Two deterministic approach populations** (97.6% separable by G2→G3 transit time; empty gap 3.5–6.0 s):
   - **FAST/straight** (~3.12 s, σ ≈ 0.03 s): laterally centered, residual vertical-high (+0.15…+0.37) — the transit is too short for the descent to finish.
   - **SLOW/veer** (~6.67 s): vertically converged, large same-sign lateral miss (u_f ≈ +0.42) — the servo fights the veer in the correct direction but arrests rather than reverses it.
2. **The fork precedes commit eligibility.** Populations are identical through t+1.5 s under identical commands and diverge in u_f before any latch decision or differential steering exists; the only measured physical differential is ~0.5° of roll tracking. **The commit latch is a terminal-precision mechanism, not the fork's cause.** What seeds the fork is *not resolved* — machine rate survives as a correlate (not a determinant; the best centered runs were low-rate), and the designed discriminating experiment (pinned-rate A/B) was never run.
3. **The registration gap.** The drone was visually observed to traverse the Gate-3 opening on several runs; 53 runs physically struck the gate frame (they reached the plane); 23 crossed centered on both measured axes at gate loss. **The simulator's registration signal never fired on any of the 546.** The registration criterion (tolerance, timing, geometry) was never characterized — vision-only, with the gate out of frame for the final 2.4 m, the true plane-crossing position was not measurable, and the sim exposes no ground truth. This is the project's deepest open question: the residual between "centered at the last measurable moment" and "whatever the simulator requires" was never closed, and cannot be characterized without either sim telemetry or a terminal-vision upgrade.

## What I would do differently

Characterize the registration criterion *first* — even crudely (deliberate offset sweeps early in the campaign) — before optimizing the approach to an uncharacterized acceptance test. Run the environment-isolation fixes (process lifecycle, power plan) in week one, not the last day: half the early tuning signal was noise. Freeze the crossing *metric* before tuning against it. And build the automation earlier — 546 clean attempts in one night dwarfed weeks of manual iteration.

## Next architecture, if simulator access returned

In order: (1) the pinned-rate A/B (closed-loop rate limiter, designed, unflown) to resolve the fork's seed; (2) fly the replay-validated commit rule; (3) registration-criterion characterization via deliberate offset sweeps; (4) terminal state projection through the blind coast (propagate last-valid state + sink rate to a predicted plane crossing); (5) only then the PnP corner-pose rebuild — it replaces the parallax-limited terminal signal but competes with everything above for validation flights.
