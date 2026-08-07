#future-work

# Future work

Things that would improve the project but are not required for
VQ1/VQ2 submission. Ordered by rough impact; each one is a pointer,
not a plan.

## Physical drone validation

The single biggest unknown: does any of this transfer to real
hardware? Before the physical qualifier (September 2026), get a
competition-class drone and run:

- Camera calibration match (FOV, tilt, resolution).
- Flight controller MAVLink integration (emit our output directly
  into the FC's offboard mode).
- Gate detection on real imagery (may require lighting normalization).
- Latency characterization (sensor-to-actuator loop).

`COMPETITION_STRATEGY.md` has flagged this as "CRITICAL" since March
2026. No progress has been recorded.

## Ensemble voting across models

Have three trained models (`aigp_8gates`, `aigp_racer`, `aigp_distill`).
Two are state-based (need gates); one is vision-based (deployable).
For the virtual qualifiers the state-based ones are not submissible.
But for benchmarking or for a hybrid architecture with a gate
detector providing pseudo-gate-positions, we could run all three and
vote or average.

Risk: different action distributions; naive averaging may produce
incoherent commands.

## Model quantization

`aigp_distill_final` is 13 MB `.zip` / 4 MB raw weights. Probably
fine for any target hardware; quantization to int8 would shave size
further. Not a priority unless the 100 TOPS target shows memory
pressure.

## Sim-to-sim validation harness

Once the DCL sim ships, write a thin wrapper that:

1. Runs `DroneRaceEnv` and the DCL sim side-by-side.
2. Feeds the same action sequence to both.
3. Measures state divergence over time.

Useful to quantify how bad the 200 Hz → 120 Hz physics-rate mismatch
is, and whether our training dynamics translate. Also a good
regression-test base for post-VQ1 iterations.

## VQ2 difficulty adaptation

VQ2 (June 2026) is described as "complex, low signal-to-noise, no
visual aids". Our distill model was trained with full domain
randomization and motion blur, so in principle it's VQ2-ready.
But if VQ1 reveals specific weaknesses, VQ2 is the place to
fine-tune:

- If gates are detected late at high speed → increase the event
  channel contribution, or retrain with stronger motion blur.
- If the policy is slow → re-tune reward weights (higher time
  penalty).
- If crashes concentrate at specific gates → add gate-specific
  domain randomization.

Plan fine-tuning for 10–50M additional steps, starting from the
current distill checkpoint.

## Real event-camera integration on physical drone

If the physical drone in September has (or can be retrofitted with)
an event camera, the zero-padding sim-to-real gap goes away. Real
event cameras (Prophesee, iniVation, etc.) produce event streams
directly; the EGM model we used in simulation should match the real
sensor's output format within tolerance.

Low probability the September drone has one; include in the "wish
list" conversation with Anduril / DCL.

## Strict MAVLink library dependency

Currently `dcl_mavlink_adapter.py` hand-codes MAVLink frames and
`dcl_mavlink_client.py` uses MAVSDK. A unified approach using
`pymavlink` (for raw frame encoding) or `mavsdk` (for high-level
offboard control, but over spec-compliant messages) would:

- Get correct CRC computation for free.
- Get correct message packing for free.
- Avoid the semantic confusion between body rates and attitude angles
  by using the library's typed message interface.

Recommendation: move to `pymavlink` for the raw path; keep MAVSDK
only if telemetry subscription is easier there.

## Better local viewer tooling

`race.py`, `view_vision.py`, `live_fpv.py`, etc. are all slight
variations on the same theme. Consolidating them into one tool with
flags (`--view-mode=3d|fpv|events`, `--model=<path>`, etc.) would
reduce the maintenance footprint.

## CI / test hygiene

No CI. No proper test suite. `test_dcl_adapter.py` is tautological
(see [[fragilities#Test suite does not exercise the real model]]).
A minimal CI that runs:

1. `strict=True` weight load test against the current model.
2. A deterministic inference rollout reproducibility check.
3. A linter pass (ruff + mypy).

...would have caught the adapter load bug the day it shipped.

## Documentation consolidation

Currently seven top-level Markdown docs (`README.md`,
`README_SCUBA_LAB.md`, `DCL_INTEGRATION.md`, `COMPETITION_STRATEGY.md`,
`README_COMPETITION.md`, `COMPETITION_CHECKLIST.md`, plus per-model
READMEs) with substantial overlap and inconsistencies. This wiki is
the start of consolidation. Post-VQ1, replace the top-level docs
with pointers to the wiki.

## See also

- [[open-questions]] — things we don't know that might reshape the plan
- [[submission-readiness]] — the near-term blocker list
- [[decisions-log]] — the decisions this work might revisit
