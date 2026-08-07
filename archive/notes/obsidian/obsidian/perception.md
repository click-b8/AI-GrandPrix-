#perception #fragility

# Perception

Three simulated sensors feed the policy: an FPV RGB camera, a
co-located event camera, and a VIO pipeline that integrates a noisy
IMU with periodic visual corrections.

## FPV RGB camera

- **Resolution:** 48 × 48 (`FPV_RESOLUTION`).
- **FOV:** 90° (`FPV_FOV`).
- **Tilt:** −10° (camera tilted down from forward). **Silent drift:**
  the nested `drone-race-sim/config.py` has `0` uncommitted; the
  deployed model was trained with `−10`. See [[fragilities#FPV_TILT_DEG local edit does not reflect training]].
- **Frame stacking:** 2 consecutive frames (`FPV_FRAME_STACK`),
  concatenated along the channel dim.
- **Rendering:** MuJoCo `Renderer` with a camera mounted at the drone
  body.

Rationale for 48×48: small enough that the CNN is cheap to forward and
cheap to render (MuJoCo rendering dominates step time at larger
resolutions), large enough that a 1.5 m gate at 5–10 m fills enough
pixels to be detectable. The 2-frame stack provides first-order
temporal context without paying the cost of a longer history.

## Motion blur simulation

Modeled after `event-sharp-nerf-drones` (Zou et al. 2026, UZH RPG).
At training step `MOTION_BLUR_WARMUP = 50_000`, a staged callback
enables sub-exposure averaging:

- `MOTION_BLUR_SAMPLES = 3` sub-exposure renders.
- Pose interpolated between `prev_qpos` and current via linear
  position + **SLERP** quaternion (`_slerp` in `drone_race_env.py`).
- Each sub-exposure is rendered at the interpolated pose, then averaged.

The warmup lets the CNN first learn clean gate geometry before adding
blur, which is a form of curriculum.

## Event camera simulation

Same paper (Zou et al. 2026). The `EventCameraSensor` class implements
the Event Generation Model (EGM): per-pixel events fire when log-luminance
changes exceed a threshold.

- Positive events (ON): `log L(t) − log L(ref) ≥ +C_pos`, where
  `C_pos = 0.2` with additive noise std `0.03`.
- Negative events (OFF): `log L(t) − log L(ref) ≤ −C_neg`, same.
- After firing, the reference log-luma advances by `count × C`, so
  subsequent fires require a further change.
- Events are accumulated into a histogram with `EVENT_FRAME_BINS = 2`
  temporal bins × 2 polarities = **4 event channels per frame**.
- Concatenated with the 3 RGB channels = 7 channels per frame.
- Stacked ×2 frames = **14 input channels** to the CNN.

Rec.601 luminance weights (0.299, 0.587, 0.114) are used to collapse
RGB to luma, matching Zou et al.

### Why model events at all

Events remain sharp at high drone velocity where RGB frames suffer
motion blur. In principle, the policy can learn to rely on events for
edge/gate localization when RGB is unusable. At training time this is
cheap (the EGM is a few lines of numpy on an already-rendered frame).

### The deployment gap — event channels are zero-padded at inference

DCL provides RGB only. `dcl_adapter.process_observation` builds the
14-channel input by concatenating the RGB frame with 4 zero channels:

```python
event_channels = np.zeros((4, 48, 48), dtype=np.float32)
frame_with_events = np.concatenate([rgb_frame, event_channels], axis=0)  # 7 ch
image_14ch = np.concatenate([frame_with_events, frame_with_events], axis=0)  # 14 ch
```

**This has never been validated.** We have no experiment showing the
trained policy's performance when the event channels are zero vs
populated from a real (or simulated) event stream. Given that event
channels are a documented training signal, zeroing them at inference
is a genuine sim-to-real gap; the bet is that the RGB-dominant features
still carry enough information. It may or may not be true for our
trained model.

It's also currently moot, because the policy head is running on random
weights (see [[fragilities#The policy head is untrained at deployment]]). Once
that's fixed, run a side-by-side: policy with simulated events vs
policy with zero event channels. The answer belongs in
[[experiments-log]].

## VIO simulation

`VIOSimulator` in `drone_race_env.py` models a MEMS IMU + visual
correction pipeline, loosely modeled on Intel RealSense T265 / the
Swift paper's VIO setup.

### IMU integration (physics rate, ~200 Hz)

At every physics step:

1. Update gyro bias as a random walk with instability `VIO_GYRO_BIAS_INSTABILITY = 0.003`.
2. Update accel bias similarly, with `VIO_ACCEL_BIAS = 0.02`.
3. Measure angular velocity as `true + bias + white noise` (`VIO_GYRO_WHITE_NOISE = 0.01`).
4. Measure acceleration as `true + bias + white noise` (`VIO_ACCEL_NOISE = 0.05`).
5. Integrate acceleration → velocity; add velocity drift `VIO_VEL_DRIFT_RATE = 0.02` m/s/s.
6. Dead-reckon position from velocity; add position drift `VIO_POS_DRIFT_RATE = 0.05` m/s.

### Visual correction (camera rate, ~30 Hz)

Every `CTRL_FREQ // VIO_UPDATE_RATE = 3` control steps:
- Snap position estimate 70% toward ground truth plus 0.03·difficulty
  noise.
- Snap velocity estimate 70% toward ground truth plus 0.05·difficulty
  noise.

This is not a real VIO (no features, no poses, no bundle adjustment);
it's a phenomenological model calibrated to produce drift characteristic
of monocular VIO at high speed.

### What the policy actually sees

The **state branch** of the vision network receives VIO estimates, not
ground truth:

- VIO velocity (3D)
- VIO angular rates (3D)
- VIO position (3D)

Plus a 6D rotation from ground-truth body-to-world (we do not simulate
rotation estimation error separately — the VIO model's angular-rate
noise propagates to angular integration in the env physics, and the
rotation is taken from MuJoCo state).

### Reward vs observation

**Rewards are computed on clean observations** (see `step()`:
`pos = obs_clean[:3]`, etc.). **Observations delivered to the policy
are noised.** This is intentional: training signal stays clean;
perception is forced to be robust.

## See also

- [[vision-model]] — how these 14 channels + 19D state become actions
- [[simulation]] — the environment that invokes these sensors
- [[fragilities]] — event-channel padding and FPV tilt drift
- [[decisions-log]] — 6D rotation vs quaternions
