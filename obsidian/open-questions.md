#open-question

# Open questions

Things we don't know that we need to find out. Tagged with urgency:
🔴 blocks VQ1 submission, 🟡 affects VQ1 performance, 🟢 later-phase
concern.

## 🔴 Blocking VQ1

### Which MAVLink control message does DCL prefer?

VADR-TS-001 lists both `SET_ATTITUDE_TARGET` and
`SET_POSITION_TARGET_LOCAL_NED`. Our trained policy emits CTBR (body
rates + thrust), which maps naturally to `SET_ATTITUDE_TARGET` with
the attitude bit of `type_mask` set to ignore. But:

- Is one primary and the other secondary?
- Does the DCL sim accept both, or only one?
- If we emit the "wrong" one, do we get rejected or just silently
  ignored?

Ask at sim release. Until we know, [[deployment]] assumes
`SET_ATTITUDE_TARGET` with body-rate interpretation.

### What is the image stream API?

`drone-race-sim/dcl_mavlink_client.py` currently feeds zero arrays as
`vision_frame`. Once the sim ships, we need to know: what format, what
frame rate, what topic (MAVLink camera trigger? gstreamer? HTTP? a
file-watch?). All plausible, none documented in the specs we have.

### Will the DCL sim validate `MAVLink v2` CRCs strictly?

If yes — and any real MAVLink parser does — our zeroed CRC blocks
every message. If for some reason the sim is permissive, we could
submit before fixing the CRC (though we still shouldn't). See
[[fragilities#MAVLink CRC is a zeroed placeholder]] and
[[submission-readiness]] item 3.

### Does DCL enforce Python 3.14.2+?

`DCL_INTEGRATION.md:99` says yes. We train and test on 3.12.
`stable_baselines3` and `torch` wheels for 3.14 may or may not be
stable. If enforced, we need to pin and package for 3.14 at
submission. Worth confirming before spending the packaging hours.

### VQ1 gate tolerance — 0.5 m or 1.0 m?

`README_COMPETITION.md:122` mentions 0.5 m center-pass tolerance, but
doesn't cite VADR-TS-001. Our training uses 1.0 m (`config.py`).
Tight passes were not a trained skill. If VQ1 enforces 0.5 m, we may
see gate-pass failures the sim doesn't tell us about. Confirm at sim
release and consider whether to fine-tune with a tighter
`GATE_TOLERANCE`.

## 🟡 Affects VQ1 performance

### ~~Does the zero-padding of event channels hurt deployment performance?~~ RESOLVED 2026-05-14

Resolved by decision: EVENT_CAMERA_ENABLED set to False in fine-tune config.
DCL provides RGB only; the fine-tuned model (aigp_finetune_tilt_final) trains
without event channels entirely, removing the deployment gap. The adapter
auto-detects 6ch vs 14ch input based on CNN in_channels. Original distill model
still falls back to zero-padding for compatibility.

### What is the trained policy's actual behavior on black images?

The current deployment path feeds `np.zeros` to a model with random
policy weights. We don't know what the correctly-loaded policy does
when shown black — does it hover? Spin? Translate? This matters for
sanity-checking the sim connection: if the sim's vision stream is
quietly broken and we get all-black frames, we need to recognize the
failure mode rather than assume the policy is "racing".

### Does the training/deployment physics-rate mismatch matter?

We train at 200 Hz physics, 100 Hz control. DCL runs at 120 Hz
physics, 50–120 Hz command. The rate controller (PD on body rates)
was tuned for 200 Hz substeps. There's no reason in principle the
policy can't run at 50 Hz command in DCL, but we haven't validated
that its learned body-rate commands produce the same closed-loop
dynamics at 120 Hz physics + 50 Hz command.

Blocked on sim availability. Once it ships, sim-to-sim replay is the
test.

### Does motion blur behave the same in DCL's rendering?

We simulate blur via SLERP sub-exposure averaging. DCL's rendering
may include real motion blur; or none at all; or a different blur
model. Training against one blur model and deploying against another
could shift feature statistics enough to degrade gate detection.

### Is `aigp_distill_final` actually better than the state experts on a vision-equivalent task?

We have no head-to-head. It's the only deployable-on-DCL model
regardless. But if it's worse than we think, distilling from a
stronger teacher (`aigp_racer_final`, 100M state) or for longer
might be worthwhile before VQ1. Requires the adapter fix first.

### Is the sign of `FPV_TILT_DEG` correct (camera up vs down)?

`config.py:87` declares `FPV_TILT_DEG = 20` commented "VADR-TS-002 §3.8:
camera tilted upwards 20°", but the env applies the NEGATIVE at
`drone_race_env.py:81` (`tilt_rad = np.radians(-FPV_TILT_DEG)`), feeding the
camera `xyaxes` in the MJCF.

Flagged from the 2026-07-11 course-wiring smoke render: an FPV frame from the
new spawn put the START gate (≈level, ~0.5 m above the drone, ~23 m ahead) low
in the frame. That is *consistent with an UP tilt* (optic axis above a level
target pushes it below center), i.e. possibly correct — but MuJoCo `xyaxes`
sign conventions are easy to get backwards, and a hand-trace is not proof. Do
NOT "fix" the sign on the strength of the render alone; an off-by-sign here
would point the camera 40° away from spec and silently wreck gate detection.

Resolve against ONE real DCL sim frame (the sniff/de-risk session will produce
one): put a gate of known height at known range in view and read whether it
sits above or below center. That fixes the sign unambiguously. Related but
distinct from the resolved `−10 vs 0` magnitude question below.

**Update (2026-07-11, read-only de-risk sniff).** Captured one real 640×360 sim
FPV frame (`tools/imu_check.py`). Lead gate clearly visible, horizontally
centered, sitting SLIGHTLY ABOVE vertical center (~48% down). **Still not
conclusive for our env's tilt sign** — and here's the newly-understood confound:
the real drone sits at −17.8° nose-down body pitch at spawn (see the IMU entry
below and the resting-attitude block in [[observation-spec]]), which shifts the
gate vertically *on top of* whatever the camera tilt does; the two partially
cancel. A clean sign check now needs our env spawned at the measured −17.8°
pitch with matched gate geometry, then compare the vertical gate position to
this frame. So resolving the spawn-pitch (B5) is a prerequisite to closing this.
Frame kept for reference.

**RESOLVED (2026-07-11, powered live-GO run) — sign is CORRECT (camera tilts
UP).** See the powered de-risk resolution below. In short: at rest the body is
−17.8° nose-down yet the ~level START gate sits just-above-center, which requires
a camera tilted UP ~+20° relative to the body (nearly cancelling the nose-down
to give a near-level view) — exactly §3.8's +20°. A level body with that up-tilt
would put the gate LOW, which is what our env shows at its level spawn, so our
`−FPV_TILT_DEG` up-tilt matches the real sim. Residual: quantitative cross-check
once B5 bakes the −17.8° spawn (expect the gate at ~48%).

### IMU + telemetry availability / health (de-risk sniff 2026-07-11)

Read-only 25 s MAVLink capture (`udpin:0.0.0.0:14550`, `tools/imu_check.py`) with
the drone sitting at spawn. Nothing was sent to the sim. Findings vs expectation:

- **Telemetry availability — CONFIRMED (resolves the A3 IMU-only-scope risk).**
  `ATTITUDE`, `LOCAL_POSITION_NED`, `ODOMETRY` are all ABSENT on the live v3385
  stream; `HIGHRES_IMU` streams at **114.6 Hz** (expected ~115). The
  [[observation-spec]] premise — build the obs from `HIGHRES_IMU` + vision +
  `prev_action` only — holds against the real sim. Full type mix: HIGHRES_IMU
  114.6, ACTUATOR_OUTPUT_STATUS 92.7, HEARTBEAT 9.9, ENCAPSULATED_DATA 4.0 Hz.
  (The "40 B ATTITUDE-sized" packets are NOT ATTITUDE — decode shows none.)

- **Rest attitude — value confirmed; A-vs-B NOT yet resolved.** Mean accel
  `[-2.999, -0.003, -9.340]`, `|a| = 9.810` (pure 1 g), implied pitch **−17.80°**,
  roll ~0. This INDEPENDENTLY REPRODUCES the A1 resting-attitude reading in
  [[observation-spec]] to 3 d.p., and answers sub-question (i): the reading is
  **on the ground at spawn**. It does NOT decide (A) non-level body spawn vs (B)
  IMU mount offset — those are identical at rest by construction; sub-question
  (ii) "does the vector stay pure-pitch under maneuvering?" needs the in-motion
  de-risk FLIGHT, which a read-only rest session cannot provide. `TODO(pitch)`
  value/sign is pinned (−17.8° nose-down); the mechanism attribution is still open.

- **🔴-if-confirmed — gyro channel unverified.** `xgyro/ygyro/zgyro` were exactly
  `0.0` across all 2865 HIGHRES_IMU samples. ~0 is expected at rest, but *exactly*
  zero (no noise) means we cannot confirm the gyro channel is live. Both the A2
  gravity filter (gyro prediction) and `body_rates` (obs dims 3–5) depend on it —
  if it's dead, the whole state vector is compromised. MUST confirm nonzero gyro
  under motion in the de-risk flight before trusting either. Escalate to 🔴 if the
  flight shows gyro stuck at zero.

- **🟡 — `time_usec` not strictly monotonic as received.** HIGHRES_IMU `time_usec`
  deltas: mean 8729 µs (→114.6 Hz, correct) but std 10731 µs and monotonic=False
  (some ≤0 deltas) — occasional out-of-order/duplicate arrival, likely UDP
  reordering on the shared socket. The A2 filter integrates `ġ = −ω×g` over `dt`
  from these stamps, so its dt handling must reject nonpositive/outlier deltas
  rather than integrate them.

### Powered de-risk attempt (2026-07-11) — control needs a live GO; gyro STILL open

Follow-up POWERED session (`tools/powered_derisk.py`, authorized dev run, not a
timed attempt): arm → TIMESYNC → command a gentle pure-pitch burst (0.30 rad/s,
thrust 0.30, 1.5 s) via `SET_ATTITUDE_TARGET type_mask=128`, recording IMU +
vision throughout. **The maneuver did NOT achieve controlled motion, so the gyro
question is NOT resolved.** What we learned:

- **Control is ineffective without a live race GO.** After a fresh sim relaunch,
  no GO countdown fired (the `ENCAPSULATED_DATA` race-status stream is present but
  never crosses into GO on its own). With `--force-no-go` I commanded the burst
  anyway: the FPV view was **identical** at rest and mid-pitch, accel was static
  (rest == burst to 3 d.p.), and gyro stayed exactly 0. The drone did not respond
  to `SET_ATTITUDE_TARGET`. This is *why* `run_vq1` gates control on GO — the sim
  only honors control when the race is live. **A genuine race start is required**
  to command the drone (someone/something must trigger the countdown).

- **`HIGHRES_IMU` streams only after ARM.** Idle/pre-race the sim emits only
  HEARTBEAT + ENCAPSULATED_DATA; IMU begins once armed. (Explains why a no-arm
  listen now returns zero IMU, unlike the earlier session that was race-active.)

- **Gyro liveness — STILL UNVERIFIED (do NOT read as "dead").** Gyro was 0.0
  throughout, but the drone never actually rotated, so this tells us nothing about
  the channel. The prior 🔴-if-confirmed flag stays OPEN, pending a run where the
  drone genuinely moves (real GO). Between runs the rest attitude drifted from
  −17.8° to ~+47° and the FPV view is now off-course scenery (no gate/path) — the
  drone is displaced/lodged in geometry from the forced attempts; **the sim needs
  a race reset before any clean at-spawn capture.**

- **`time_usec` refined (while armed).** 0% true reordering (`<0` deltas) and no
  >1 s gaps — the earlier "non-monotonic" read was DUPLICATE stamps (`delta==0`),
  measured at **~12–28% of samples** while armed (higher, ~50–70%, in the near-
  frozen pre-race state). That is ≫1%, so per the brief I converted `run_vq1`'s
  warn-once dt fallback to a **counted warning** (`_MAVLinkReceiver._warn_dt`).
  The A2 filter already falls back to nominal 1/115 on `delta<=0`, which is the
  right behavior for duplicate stamps.

- **A-vs-B (pitch mechanism) and FPV tilt-sign remain OPEN** — both needed the
  in-motion / at-spawn frames this session failed to produce.

Raw IMU log + frames saved by the tool (`derisk_imu.csv`, `derisk_*.png`) so a
future run is re-analyzable without re-flying.

### Powered de-risk RESOLVED (2026-07-11, real race GO)

Re-ran `tools/powered_derisk.py` waiting on a genuine race the operator started;
the countdown armed and GO fired (`race_start` 2915 ms ahead → GO), the drone was
at a CLEAN spawn (rest accel back to `[-2.999, -0.002, -9.340]`, −17.8°), and this
time control was effective — the maneuver actually moved the drone. Three
questions close:

- **✅ Gyro is LIVE (clears the 🔴 flag).** During the pitch burst `max|gyro| =
  0.848 rad/s`, dominant on the **pitch (y)** axis with x/z gyro **exactly 0** —
  a clean pure-pitch rotation, not a tumble. The earlier all-zero gyro was purely
  "drone not moving," never a dead channel. `body_rates` (obs dims 3–5) and the A2
  filter's gyro prediction have a real signal.

- **✅ Pitch A-vs-B → hypothesis A (non-level body spawn).** Two independent lines
  now favor A over B (fixed IMU/camera mount offset):
  (1) *pure-pitch under maneuver* — gyro stays pure pitch (x,z ≡ 0), accel-y stays
  at its rest value (max dev 0.004 m/s²), no roll coupling;
  (2) *rest camera geometry* — B (level body + up-tilted ~20° camera/IMU mount)
  predicts the level gate should sit LOW in frame, but it sits just-above-center,
  which requires the body to actually be pitched −17.8° nose-down. So the −17.8°
  is the drone's real body attitude at spawn. **Fix (B5): bake a −17.8° nose-down
  pitch into the env spawn quaternion; keep the deploy filter `R_mount = identity`**
  — matches the working hypothesis in [[observation-spec]], now confirmed.

- **✅ FPV tilt-sign correct** (camera up ~+20°, matches §3.8) — see the resolution
  appended to the tilt-sign entry above. Our env's `−FPV_TILT_DEG` up-tilt matches
  the real sim; the "gate low in our level-spawn render" is exactly the expected
  up-tilt behavior, not a sign bug.

- **time_usec (armed, live race):** 12.7% duplicate stamps, **0% reordering**, no
  >1 s gaps — consistent with the prior armed captures; the counted dt-fallback
  warning (`run_vq1`) stands.

Still open: (ii)-adjacent — the in-flight gravity-filter residual envelope for B5
noise calibration (needs the A2 filter logging under aggressive flight, a bigger
run than this gentle burst).

### Attitude-termination envelope — tilt-from-vertical proposal (PENDING desktop ep_len data, NOT applied)

Spawn-envelope diagnostic (2026-07-13): under PPO-init action noise, fresh-spawn
episodes terminate in ~7 steps (~0.07 s), and this is **independent of the −17.8°
spawn pitch** (level 7.17 vs pitched 7.10 steps; 100% terminate via the ROLL
condition, 0% via pitch). Two mechanisms:

- `_quat_to_rpy` extracts pitch via `arcsin(clip(...))`, clamped to ±90°, so the
  `abs(pitch) > 120°` termination limb (`drone_race_env.py`) is **unreachable** —
  attitude termination is effectively roll-only; a nose-over past 90° flips into
  the roll condition via the euler gimbal.
- Roll starts at 0° for any spawn, and random ±12 rad/s (`MAX_BODY_RATE`) roll
  commands walk it through ±120° in ~7 steps.

So the −17.8° spawn is **exonerated** (keep it), and the ~0.07 s episodes are a
general random-init property, not a spawn-pitch artifact.

**PROPOSAL (do NOT apply yet):** replace the euler roll/pitch attitude termination
with a single gimbal-free **tilt-from-vertical** criterion — terminate when the
body-up axis deviates more than Θ from world-up, i.e. `R[2,2] < cos(Θ)`. At
Θ ≈ 120–135° this is symmetric, well-defined, and fixes the dead pitch-limb; a
modest Θ widening also gives early-learning "oxygen" if the short episodes prove
to be starving PPO. (A small per-step survival bonus is a weaker alternative —
risks loitering against `REWARD_TIME_PENALTY`.)

**STATUS — pending desktop ep_len data.** The Surface seed-0 run (237k steps)
showed ep_len trending 7→10, i.e. PPO *is* clawing survival up, so the envelope
may not be the bottleneck. Decide from the desktop seed-0 run (more steps): if its
ep_len is ALSO stuck near 7–10 well past a few hundred k, the envelope is limiting
early learning and this change is warranted; if ep_len keeps climbing, it's normal
random-init and no change is needed. (Tilt-from-vertical is a cleaner "crashed"
proxy than the euler limits regardless, so it's low-risk even if not strictly
needed.)

## 🟢 Later-phase

### What's inference time on target hardware?

Our only number is 0.26 ms mean on Apple M4 CPU. The competition
target is ~100 TOPS embedded. Numbers between desktop CPU and
embedded edge-AI can differ by 10×+ in either direction depending
on memory bandwidth and compiler support.

Not actionable until we have access to target hardware, which is a
Physical Qualifier (September) concern.

### Does the policy transfer to a real drone?

Literally no physical validation has been done. If the model works
perfectly on the DCL sim but fails on the physical drone in
September, it will be late to start figuring out why. Options
include: fine-tune with real flight data (requires drone time we
don't have), tighten sim-to-real domain randomization, deploy the
state-based `aigp_racer_final` against a Mocap-fed state estimator
(doesn't require vision to work).

This is the biggest unknown for the physical phase and has been
flagged in `COMPETITION_STRATEGY.md` since before the audit.

### Is the curriculum ramp rate right?

We have three curriculum schedules across three training scripts.
None have been ablated. Slower ramps may improve final performance;
faster ramps may underprepare the policy. No data either way.

### Are the reward weights well-tuned?

The reward terms and their relative weights
(`REWARD_GATE_PASSED = 100`, `REWARD_PROGRESS = 10`, etc.) are from
the initial Swift-aligned design. They have not been ablated. A
reward re-tune for speed (lower progress weight, higher time
penalty) might produce faster laps; one for reliability (higher gate
bonus, lower time penalty) might improve completion rate under
harder conditions. Unknown which is the bottleneck.

### Is there a better place to put the adapter's Tanh?

`PolicyNet` ends with a Tanh in `dcl_adapter.py`. Training does not.
When we fix the adapter, should the fixed adapter have Tanh? SB3's
`MultiInputPolicy` applies no Tanh; action squashing happens via
clipping at the env boundary. Deployment should probably match: no
Tanh in the adapter, clip ranges at `action_to_dcl_command`. This is
a minor but real decision that affects output distributions near the
edges of the action space.

## Things we *thought* were open questions but aren't

- **PolicyNet 275D quirk.** Resolved by the audit's shape check. It's
  not a quirk; the adapter is wrong. See
  [[fragilities#The policy head is untrained at deployment]].
- **`FPV_TILT_DEG` −10 vs 0.** Resolved by git investigation. The
  deployed model was trained with `−10`; the `drone-race-sim/`
  subdir copy has an uncommitted local drift. Use `−10`.
- **Inference time "5 ms" vs "50 ms" disagreement.** Resolved on Mac
  by measurement (0.26 ms). Still open for target hardware but now a
  concrete lower bound exists.

## See also

- [[fragilities]] — what we know is broken
- [[submission-readiness]] — what blocks submission today
- [[future-work]] — things we'd do with more time
