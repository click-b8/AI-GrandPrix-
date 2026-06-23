# VQ1 Current Plan — as of 2026-06-23

**Status: integration harness COMPLETE and verified. No checkpoint flies the course.**

Across multiple distinct model checkpoints, with every input axis verified
correct (z-up frame, FLU rates, 19-D layout matching `_get_minimal_state`), the
drone thrashes thrust 0↔1, spins, and crashes within ~15s. `active_gate` has
never advanced past 0. Conclusion: the failure is the policy/weights, not the
adapter. The existing checkpoints were never trained against this deployment
observation spec.

---

## TASK 1 — Certify the harness (in progress)

Fly the stale-clock start-gate fix once. Success = one clean legal start:
countdown arms (delta > 0), GO fires at "Go!" not before, no DQ. Purpose is NOT
to evaluate the model — it's to confirm the harness is deployment-ready for a
NEW model. One run, then push the fix.

## TASK 2 — DGX training run (CRITICAL PATH — highest EV, start ASAP)

The README's own "DGX run: not yet started" is the real deliverable. Train a
model to convergence against the EXACT observation our adapter now produces:

- 48×48 RGB input (NOT 14-ch event — RGB only, matches DCL stream)
- +20° camera tilt
- 19-D state: dims 0-5 rot_6d (z-up), 6-8 lin_vel (z-up), 9-11 ang_rates
  (FLU body), 12-15 prev_action, 16-18 position (z-up)

This spec is the verified deployment observation. Training to it = deploys
through the harness with zero frame surprises.

Fresh start from a clean checkpoint, LR ~1e-5. Checkpoint frequently so partial
runs are usable.

## TASK 3 — Parallel cleanup while TASK 2 trains

- **Fix README:** `aigp_distill_11600000_steps.zip` and
  `aigp_finetune_tilt_final.zip` are byte-identical (same SHA) — the "two model"
  distinction is false. Document it.
- **Fix cross-run state leak:** stale state/race values bleed into the next
  run's first frames (visible in logs).
- **Optional:** lateral/heading frame analysis (NWU vs ENU) to fully close the
  input question — low priority, spin-not-drift already argues it's not heading.
- **Known:** CPU inference is emulated x86 PyTorch on ARM (Snapdragon X). Not
  the navigation blocker, but a jitter/headroom issue for later.
