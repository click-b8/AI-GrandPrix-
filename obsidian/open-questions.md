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
