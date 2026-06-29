# VQ1 Run Notes — 2026-06-28

## Harness verified end-to-end

This run confirmed the full deployment harness works:

- **Clean legal start** — countdown armed (delta > 0), GO fired at delta ≈ 0
  (at on-screen "Go!", not before), no DQ. The stale-clock start-gate fix held.
- **Model in control** — drone armed and flew under policy output.
- Captured on screen recording: `Recording_2026-06-28_VQ1_Good_Start.mp4`
  (kept out of the git tree; attached to a GitHub Release — see below).

## Failure mode (confirmed by the FPV recording)

At launch the gates are **large, red, centered, with a clear racing line** — the
perception input is good and the correct action is obvious. Yet the drone
**banks off the line within ~2s of GO and diverges**. This is an
**attitude-control / policy failure**, not:

- ❌ vision — gates clearly visible and centered in the FPV frame
- ❌ frame/observation — z-up, FLU rates, 19-D layout all verified vs
  `_get_minimal_state`
- ❌ start gate — clean legal GO this run
- ❌ model selection — reproduced across distinct checkpoints

## Conclusion

**The weights are the bottleneck.** Every part of the harness around the policy
is verified; the policy itself cannot fly the verified deployment observation
because no existing checkpoint was trained against it.

## Next action

DGX training run against the verified deployment observation spec:

- 48×48 RGB input (RGB only — matches DCL stream, not 14-ch event)
- +20° camera tilt
- 19-D state: dims 0-5 rot_6d (z-up), 6-8 lin_vel (z-up), 9-11 ang_rates
  (FLU body), 12-15 prev_action, 16-18 position (z-up)

See [[current-plan]] TASK 2 for the full training plan.
