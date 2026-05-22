#deployment #competition

# Submission readiness

Operational checklist. If VQ1 submission were required tomorrow, this
is what would break, in priority order.

**Last updated: 2026-05-22**
**VQ1 deadline:** May 2026.

## Status summary

All deployment-layer blockers resolved as of 2026-04-22 (branch
`fix/mavlink-complete`, merged to main). Spec gap analysis completed
2026-05-14 against VADR-TS-002. Camera tilt fine-tune active on HPC
(job 4656258). One blocker remains: vision stream (blocked on DCL sim).

---

## Blocking issues

### 1. ~~Policy weights do not load~~ — RESOLVED 2026-04-21
Branch `fix/policy-weight-load`, merged. `strict=True`, layer stds
confirmed above Kaiming floor. See [[fragilities#The policy head is untrained at deployment]].

### 2. ~~MAVLink output interprets body rates as attitude angles~~ — RESOLVED 2026-04-22
`dcl_mavlink_adapter.py` rewritten. CTBR semantics correct:
`body_*_rate = policy_output * MAX_BODY_RATE (12.0 rad/s)`,
`type_mask=128` (IGNORE_ATTITUDE). See [[fragilities#MAVLink action semantics mismatch]].

### 3. ~~MAVLink CRC is zero~~ — RESOLVED 2026-04-22
CRC-16-CCITT with correct crc_extra seeds (SET_ATTITUDE_TARGET=49,
HEARTBEAT=50) via pymavlink. 11/11 compliance tests passing.

### 4. ~~MAVLink payload is malformed~~ — RESOLVED 2026-04-22
Uses `pymavlink MAVLink_set_attitude_target_message` for correct wire
encoding. Byte-exact match against pymavlink reference encoder.

### 5. ~~Advertised model path is wrong~~ — RESOLVED 2026-04-22
All doc references updated. `run_vq1.py` uses canonical path with
loud failure if missing.

### 6. Vision stream is a placeholder — BLOCKED on DCL sim
`run_vq1.py: _get_vision_frame()` returns `np.zeros((48,48,3))`.
DCL simulator not yet released (expected May 2026).
**Fix when sim ships:** replace stub with real camera stream.
Resize from 640×360 → 48×48 is already handled in `dcl_adapter.py`.

### 7. ~~`dcl_mavlink_client.py` uses wrong MAVLink message~~ — RESOLVED 2026-04-22
Patched to use `MAVLinkFrameBuilder` from corrected adapter.
`set_actuator_control` removed.

### 8. ~~Silent `except: pass` in MAVLink send~~ — RESOLVED 2026-04-22
Replaced with logged + counted `send_errors`. No silent swallowing.

### 9. ~~Test suite is tautological~~ — RESOLVED 2026-04-22
28/28 tests passing including byte-exact pymavlink comparison,
300-frame UDP roundtrip, weight-load validation.

---

## Spec gaps (from VADR-TS-002 analysis, 2026-05-14)

### Camera tilt mismatch — IN PROGRESS
Spec §3.8: +20° upward. HPC training had 0°. Root config had -10°.
**Fix:** Fine-tune job 4656258 running on HPC. `FPV_TILT_DEG=20`,
`EVENT_CAMERA_ENABLED=False`, 15M steps from `aigp_state_final` teacher.
Output: `trained_finetune_tilt/aigp_distill_final.zip`.
`run_vq1.py` will auto-prefer `aigp_finetune_tilt_final.zip` when present.

### Event channel gap — RESOLVED by decision 2026-05-14
DCL provides RGB only. Original distill trained with 14-channel input
(RGB + event channels). Fine-tune sets `EVENT_CAMERA_ENABLED=False`
producing 6-channel model (RGB only × 2 frames). Removes zero-padding
deployment gap entirely. Adapter auto-detects 14ch vs 6ch by inspecting
CNN `in_channels`.

### Camera resolution — RESOLVED in adapter
DCL streams 640×360. Model expects 48×48. `dcl_adapter.process_observation()`
resizes via `PIL.Image.BILINEAR` before inference. No retrain needed.

### Physics rate mismatch — BLOCKED on DCL sim
Training: 200 Hz physics / 100 Hz control.
DCL: 120 Hz physics / 50–120 Hz command.
**Fix:** Replay deterministic rollout against DCL sim when it ships.
Retune `MOTOR_TAU`/`RATE_KP`/`RATE_KD` if divergence > 0.5m at gates.

---

## What to ask DCL when the simulator ships

1. Exact MAVLink message accepted (we use SET_ATTITUDE_TARGET — confirm)
2. Image topic, frame rate, encoding for FPV stream
3. Whether TIMESYNC handshake is required before commands accepted
4. Whether MAV_TYPE in heartbeat affects anything (we send QUADROTOR=2)

---

## Decision: what to ship for VQ1

**Preferred:** wait for fine-tune job 4656258 to complete, copy
`aigp_finetune_tilt_final.zip` to `models_release/`, push to GitHub.
`run_vq1.py` picks it up automatically. Integrate vision stream when
sim ships. Submit.

**Local validation (2026-05-22):** Pipeline confirmed on Surface (Python 3.14.3,
Windows 11, CPU-only). 28/28 tests passing, 0 send errors, ~30 Hz throughput on CPU
(50 Hz expected on competition GPU). Model loads strict=True.

**Fallback if fine-tune fails:** submit with `aigp_distill_final.zip`
(0° tilt, 14ch). Performance will be degraded but submission is
spec-compliant. First submission is a measurement, not a validation.

**Do not submit the pre-April-22 code in any form.** The three silent
failures (wrong weights, wrong MAVLink semantics, zeroed CRC) compose
into a submission that burns an entry with nothing to learn from.

## See also

- [[fragilities]] — long-form discussion of each item
- [[vq1-execution-plan]] — the sequenced branch plan
- [[experiments-log]] — active training run details
