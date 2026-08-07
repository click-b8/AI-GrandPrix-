#competition

# Competition specification

Reference for the Anduril AI Grand Prix / A2RL × DCL 2025–2026 series
as Scuba Lab understands it. Some of what's in the project docs is
vague or contradictory; this file resolves to what we believe the DCL
actually requires, and flags what we don't yet know.

**Today:** 2026-04-21. **Primary source:** VADR-TS-001 Issue 00.01,
dated 2026-03-09. **Secondary:** `DCL_INTEGRATION.md`,
`README_SCUBA_LAB.md`, `COMPETITION_CHECKLIST.md`.

## Timeline

| Stage | Month | Format |
|---|---|---|
| Virtual Qualifier 1 (VQ1) | May 2026 | Code submission, DCL sim, "desaturated / simplified / highlighted gates" |
| Virtual Qualifier 2 (VQ2) | June 2026 | Code submission, DCL sim, "complex, low signal-to-noise, no visual aids" |
| Physical Qualifier | September 2026 | In-person, Southern California |
| Final Championship | November 2026 | Ohio |

The DCL simulator is expected to release in May 2026 alongside VQ1.
**We have not tested against the actual simulator.**

## Control interface (VADR-TS-001)

- **Protocol:** MAVLink v2 over UDP.
- **Accepted messages:** `SET_POSITION_TARGET_LOCAL_NED` or
  `SET_ATTITUDE_TARGET`. The spec lists both; we don't know from the
  docs which is primary, or whether both are accepted simultaneously.
  Ask DCL at sim release.
- **Command rate:** 50–120 Hz. We plan to run at 50 Hz (matches
  `dcl_mavlink_client.py:target_fps=50`).
- **Heartbeat:** ≥ 2 Hz required.
- **TIMESYNC:** required; synchronization with sim clock expected.
- **Max run duration:** 8 minutes (480 s).
- **Deterministic sensors:** no stochastic sensor outputs permitted;
  courses are identical across all teams.
- **Zero human intervention** during a timed run.

## Sensor interface (DCL-provided)

Per `DCL_INTEGRATION.md:77-88`:

Inputs to our agent:
- **Telemetry:** position (x, y, z), velocity (vx, vy, vz), orientation
  (roll, pitch, yaw).
- **Visual:** forward-facing FPV camera stream.
- **No depth, no engine RPM, no battery state.**

Outputs our agent must produce:
- **Throttle:** 0–1.
- **Roll, pitch, yaw:** −1 to 1.

The output ranges are as the competition expects them — these are
what the DCL simulator ultimately applies. What they *mean*
(attitude setpoint vs body-rate command) depends on how we encode
the MAVLink message. Our trained policy emits body rates, so the
correct mapping is into the `body_*_rate` fields of
`SET_ATTITUDE_TARGET` with the attitude-quaternion bit masked out.
See [[fragilities#MAVLink action semantics mismatch]].

## Hardware (VADR-TS-001 minimum)

| Component | Spec |
|---|---|
| CPU | Intel Core i5-10400F or AMD Ryzen 5 3600 |
| GPU | Nvidia RTX 2060 Super or AMD 9060XT |
| RAM | 16 GB |
| Storage | 60 GB |
| Python | 3.14.2+ |

"Optimized for ~100 TOPS" is mentioned in `DCL_INTEGRATION.md:97` —
this is the target for the physical phase where the agent runs on an
embedded compute module attached to the drone, not the virtual phase.
VQ1/VQ2 run on organizer-provided desktop/cloud-ish hardware.

Note: we train on SB3 2.7.1 / Python 3.12, and locally run Python
3.12 as well. If DCL strictly requires 3.14, we may have packaging
work ahead (re-pinning torch/SB3 for 3.14 compatibility).

## Submission format

Per `COMPETITION_CHECKLIST.md:98-106`, the anticipated submission
package is:

```
submission/
├── dcl_mavlink_client.py   (entry point)
├── dcl_adapter.py          (inference adapter)
├── vision_model.py         (feature extractor)
├── README.md               (quick-start)
├── requirements.txt        (pinned deps)
└── models/
    └── aigp_distill_final.zip
```

This is plausible but has not been confirmed against actual DCL
submission rules. We haven't seen a real submission template.

## IP and ownership

Per `README_SCUBA_LAB.md:132-137`:

- Scuba Lab retains full IP ownership of algorithm, source code,
  documentation.
- Anduril receives permission to use code for competition operations
  only, for the competition period.
- No transfer; Anduril cannot commercialize the code.

Worth confirming against the actual competition legal document before
submission. The README_SCUBA_LAB.md wording is plausible but is not
itself an agreement.

## What VQ1 will look like (educated guess)

From `DCL_INTEGRATION.md:70-74` and `README_COMPETITION.md:233-237`:

- **Format:** structured 3D racecourse, standardized gates, possibly
  the same 8-gate layout as our training track or similar.
- **Scoring:** time-based; fastest valid run wins.
- **Goal:** pass all gates in correct order.
- **Visual conditions:** "desaturated imagery, realistic physics" —
  meaning: the FPV stream may be in muted colors, but otherwise no
  tricks.
- **Gate tolerance (per `README_COMPETITION.md:122`):** "0.5 m"
  center-pass tolerance. Our training uses 1.0 m (`GATE_TOLERANCE`
  in `config.py`). If VQ1 enforces 0.5 m, we have a gap — the policy
  has never been rewarded for tight passes. Worth validating.

## What we're waiting on

- The simulator itself.
- Which of `SET_ATTITUDE_TARGET` / `SET_POSITION_TARGET_LOCAL_NED` is
  primary.
- Whether the simulator speaks raw MAVLink v2 or MAVSDK abstractions.
- The exact image-stream API (encoding, rate, topic).
- The exact submission-package spec.
- Any hardware differences between virtual-phase and physical-phase
  runs.

## Discrepancies resolved

A few places where the project docs disagreed and this wiki picks a
canonical answer:

- **Physics rate:** our sim is 200 Hz; DCL is 120 Hz. This wiki tells
  the truth (see [[simulation]] and [[fragilities#Training / deployment physics-rate mismatch]]).
  The `DCL_INTEGRATION.md:133` remark that "200Hz control loop
  possible" is just claiming our inference is fast enough, not that
  DCL runs at 200 Hz.
- **Gate tolerance:** our sim uses 1.0 m, `README_COMPETITION.md`
  mentions 0.5 m for VQ1. Canonical: DCL decides; we tell the truth
  about what we trained against (1.0 m).
- **System type in heartbeat:** our code uses `MAV_TYPE_FIXED_WING = 1`.
  Should be `MAV_TYPE_QUADROTOR = 2`. See [[submission-readiness]].

## See also

- [[submission-readiness]] — the punch list before VQ1
- [[deployment]] — how our code maps to this interface
- [[fragilities]] — what's wrong with the current mapping
- [[open-questions]] — what we still need to ask DCL
