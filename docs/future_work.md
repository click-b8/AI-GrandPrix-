# Future work

Ordered by expected value, all grounded in the overnight dataset. The first two were fully designed and ready to run when simulator access ended.

## 1. The pinned-rate A/B (designed, never run)

A closed-loop rate limiter (target a period; sleep only when ahead; never skip control/detector updates; log requested vs actual dt) pinned at ~72 Hz × 3 runs and ~100 Hz × 3 runs, all flight parameters frozen. Classify each run by leg transit time (3.1 s vs 6.7 s — 97.6% separable). Outcome either proves machine rate causes the trajectory fork (→ rate-pinning becomes a flight requirement) or exonerates it (→ the fork seed is in Gate-2-crossing vehicle state, searchable in the kept traces). One evening of flights closes the project's biggest open question if it were still accessible. 

## 2. Fly the replay-validated commit rule

`size 0.20 / align 0.12 / rate 0.16 / dwell 0.15` — offline-proven strictly dominant (86:1 confusion, all golden runs preserved, arming blocker cured). A/B against the frozen rule on the standard batch. Expected effect: latch rate on straight approaches 52% → ~64%, tripling-precision terminal blinding on the runs that were arriving unlatched.

## 3. Close the terminal residual

The top-10 family shares one defect: +0.12…+0.23 high at gate loss with ~2.4 m still to fly. Candidates, in rising ambition: start the terminal descent earlier (size-threshold sweep — replay first), a slightly larger blind-hold cut scaled by loss-time v_f (bounded, replay first), or terminal state projection — propagate the last valid (u_f, v_f, sink rate) through the blind coast and shape the hold to a predicted plane-crossing rather than a fixed cut.

## 4. Instrument the fork seed

If rate is exonerated by (1): correlate Gate-2 crossing vehicle state (roll, roll rate, the exit-level latch timing) against subsequent population membership across the 302 traces. The ~0.5° roll-tracking differential under identical commands must come from somewhere measurable.

## 5. Legs 3–5

Untuned, unflown. Known issues queued: per-leg descent biases default to None; leg 4 demands a −25° bank reversal (the course's sharpest) entered at whatever state Gate 3 exits; `commit_k`'s lagged recovery re-enters vision authority slowly at exactly that moment. The overnight harness already classifies post-Gate-3 progress (`POST_G3_FAILURE` with failure-point bucketing) — the tooling is ready the day Gate 3 falls.

## 6. Vision upgrades (kept deliberately last)

The PnP feasibility branch (frozen at audit) becomes worthwhile only after the blob-servo ceiling is proven: full pose from gate corners would replace the parallax-distorted terminal signal that currently makes the last 2.4 m blind. The audit's conclusion stands — it is a rebuild, not a patch, and it competes with everything above for validation flights.

## Non-goals

Re-tuning Gates 1–2 (frozen, proven, and every historical regression traces to touching them), and any variance-harvesting rerun of the current configuration — the dataset already caps its success probability below 0.5% per attempt.
