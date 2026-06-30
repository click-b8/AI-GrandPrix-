# Observation Contract — VQ1 Vision + IMU (10-D)

**Status:** authoritative. This is the single source of truth that the training
env (`drone_race_env._get_minimal_state` / `_get_vision_obs`) and the deploy
adapter (`dcl_adapter.process_observation`) must BOTH implement, byte-identically.
A divergence here is the failure mode that has cost us weeks (z-axis, FRD/FLU,
faked frame stack). Train == deploy by construction, or not at all.

Built **only** from sources un-blocked in the latest sim: `HIGHRES_IMU`
(accel + gyro), the FPV vision stream, and our own previous action. No
`ATTITUDE`, no `LOCAL_POSITION_NED`, no `ODOMETRY`, no `GATE_INFO`.

---

## Observation = Dict { image, state }

### `state` — 10-D float32

| Idx | Field | Frame | Definition | Level-attitude value |
|-----|-------|-------|------------|----------------------|
| 0–2 | `gravity_unit` | **FLU body** | unit vector pointing **along gravity (down)** in body frame | `[0, 0, -1]` |
| 3–5 | `body_rates` | **FLU body** | angular velocity, rad/s | `[0, 0, 0]` at rest |
| 6–9 | `prev_action` | n/a | last policy output `[throttle, roll, pitch, yaw]`, normalized | `[·,·,·,·]` |

`state = concatenate([gravity_unit, body_rates, prev_action])` → shape `(10,)`.

#### `gravity_unit` (dims 0–2)
Unit vector in the body frame pointing in the direction gravity pulls (down).
At level attitude it is `[0, 0, -1]` (FLU body z is up, so "down" is −z).

- **Train (`_get_minimal_state`):** compute the CLEAN value directly from the
  true body→world rotation `R` (z-up world):
  `gravity_unit = R.T @ [0, 0, -1]`, then add gravity-direction noise whose
  magnitude is set by the **measured deploy-filter residual envelope** (see
  de-risk step below — do not guess this).
- **Deploy (A2 `GravityEstimator`):** complementary/Mahony filter fuses gyro +
  accel → gravity-down unit in **FRD body**, then map FRD→FLU:
  `gravity_unit = C @ gravity_unit_frd`, with `C = diag(1, -1, -1)`.
  Sanity: at rest the accel reads ~+g "up" → gravity-down in FRD is `[0,0,+1]`;
  `C @ [0,0,1] = [0,0,-1]` ✓ matches train's level value.

**Worked NON-LEVEL example — this is the one that matters.** The old z-axis bug
passed at identity and only failed under rotation, so the contract (and the
test) must agree at a tilted attitude. At **roll = 30°, pitch = 30°, yaw = 0**:

- Train: `R.T @ [0,0,-1] = [0.5, -0.433013, -0.75]` (unit; verified).
- Deploy: perfect static accel in FRD `= C @ (-[0.5,-0.433,-0.75])`; filter
  recovers gravity-down in FRD, `C @ (·)` maps it back to
  `[0.5, -0.433013, -0.75]` ✓.

Both sides MUST produce `[0.5, -0.433013, -0.75]` here. Agreement only at level
is NOT sufficient evidence of a correct transform.

#### `body_rates` (dims 3–5)
Angular velocity in FLU body frame, rad/s.
- **Train:** `qvel[3:6]` (verified body-frame) + gyro noise.
- **Deploy:** `C @ gyro_frd` (the FRD→FLU conversion already proven correct for
  the old dims 9–11). `C = diag(1, -1, -1)`.

#### `prev_action` (dims 6–9)
The previous step's policy output, normalized: `throttle ∈ [0,1]`,
`roll/pitch/yaw ∈ [-1,1]`. Stored pre-scaling (before ×MAX_BODY_RATE).
- **Train:** `self._prev_action`. **Deploy:** `self._prev_action`.
Both initialize to `zeros(4)` at episode/session start.

---

### `image` — stacked RGB FPV

| Property | Value |
|----------|-------|
| Per-frame resolution | 48 × 48 |
| Channels per frame | 3 (RGB, `EVENT_CAMERA_ENABLED = False`) |
| Frame stack | `FPV_FRAME_STACK = 3` (consecutive real frames) |
| Total channels | `3 × 3 = 9` |
| dtype / range | `uint8`, `[0, 255]` (the CNN divides by 255 internally) |
| Decode | JPEG → **RGB** (PIL `convert("RGB")`). NOT cv2/BGR. |

#### Channel order — the part that silently breaks
The env builds **HWC** `(48, 48, 9)` by
`np.concatenate(list(frame_buffer), axis=2)`, where `frame_buffer` is a
`deque(maxlen=3)` appended newest-to-the-right. `list(deque)` iterates
**oldest → newest**, so the channel axis is:

```
[ f_{t-2}:R,G,B , f_{t-1}:R,G,B , f_t:R,G,B ]   (oldest first, newest last)
```

SB3 `VecTransposeImage` flips HWC→**CHW** `(9, 48, 48)` before the policy,
preserving that channel order.

**Deploy MUST reproduce CHW `(9,48,48)` with the identical order.** Maintain a
`deque(maxlen=FPV_FRAME_STACK)` of the last N **genuinely consecutive** frames
(do NOT duplicate one frame — that makes velocity unobservable, defeating the
only motion cue the policy has). For each frame oldest→newest, transpose
`(H,W,3)→(3,H,W)` and concatenate along axis 0:

```
image = concat([CHW(f_{t-2}), CHW(f_{t-1}), CHW(f_t)], axis=0)   # (9,48,48)
```

A reversed deque (newest-first) is a silent train/deploy mismatch — assert the
order in the contract test.

---

## Sources (un-blocked only)

| Obs field | Source message | Notes |
|-----------|----------------|-------|
| `gravity_unit` | `HIGHRES_IMU` accel + gyro → filter | accel `xacc/yacc/zacc` (A1 adds capture), gyro for prediction |
| `body_rates` | `HIGHRES_IMU` gyro | `xgyro/ygyro/zgyro` |
| `prev_action` | self | our last command |
| `image` | vision UDP stream | header `<IHHIIQ`, port 5600, PIL RGB decode |

Heading (yaw), position, and linear velocity are **not** in the observation —
they are unobservable from un-blocked sources and must be inferred by the policy
from the image sequence. This is intentional and is why the real frame stack
(above) and visual domain randomization are load-bearing.

---

## Verification test (next artifact, before any Part-A/B code lands)

A single test feeds one known input — `(gyro, accel, prev_action, frame
sequence)` — into BOTH builders and asserts the emitted observations are
**byte-identical**. Two non-negotiable conditions on the input:

- **Tilted attitude, not level.** Use **roll = 30°, pitch = 30°, yaw = 0**
  (gravity target `[0.5, -0.433013, -0.75]`). A test that only checks level
  would have passed the old z-axis bug. Level may be included as an extra case,
  but the tilted case is mandatory.
- **Distinguishable frames.** `frame0 ≠ frame1 ≠ frame2`, distinct across both
  frame index AND RGB channel (e.g. `frame_k[...,c] = 10*(k+1)+c`). Three
  identical frames would let a reversed deque pass silently — the whole point of
  the order check.

Steps:
1. Drive the deploy path: A2 filter (static accel + zero gyro, settled) + A3
   builder → `{image, state}`.
2. Drive the train path: env `_get_minimal_state` with a quaternion/qvel that
   yield the same true gravity + rates; stack the same frame sequence the way
   `_get_vision_obs` does (`concatenate(list(deque), axis=2)` → VecTranspose).
3. Assert `state` equal within float tolerance AND `image` exactly equal
   (shape `(9,48,48)`, dtype uint8, channel order oldest→newest).
4. **Negative control:** a reversed frame stack must FAIL the image assertion —
   prove the test can detect an order flip.

This test is the guardrail. It must pass before training time is committed and
must stay green for the life of the project.

---

## Implements-this contract

| Side | File / function |
|------|-----------------|
| Deploy state | `dcl_adapter.process_observation` (A3) |
| Deploy gravity | `attitude_filter.GravityEstimator` (A2) |
| Deploy accel capture | `run_vq1._MAVLinkReceiver` (A1) |
| Deploy frame stack | `dcl_adapter` rolling deque (A3) |
| Train state | `drone_race_env._get_minimal_state` (B5) |
| Train image | `drone_race_env._get_vision_obs` / `_frame_buffer` |
| Shared constants | `config.VISION_STATE_DIM = 10`, `FPV_FRAME_STACK = 3`, `EVENT_CAMERA_ENABLED = False` |

## Gravity-noise calibration (sets B5 from measurement, not guess)

After A1+A2 are live, fly a few real races logging the filter's gravity estimate
and its internal gyro-vs-accel disagreement under aggressive flight. The
measured residual-error envelope sets the gravity-direction noise injected in
B5's `_get_minimal_state`, so we train on the noise we will actually deploy
into. Local, pre-DGX.
