#architecture #training

# Simulation environment

`drone_race_env.py` — a MuJoCo-backed `gymnasium.Env` that models a
quadrotor, a simulated IMU+VIO pipeline, an FPV camera, an event
camera, and an 8-gate race course. This is everything the policy sees
during training.

## Action space: CTBR

4-dimensional, Box:

- index 0: normalized collective thrust, `[0, 1]`
- indices 1–3: body rates `[−1, 1]`, scaled by `MAX_BODY_RATE = 12.0`
  rad/s (≈ 690°/s max) before being fed to the rate controller.

Inside `step()`, the normalized thrust becomes
`collective_thrust = thrust_normalized * MAX_COLLECTIVE_THRUST * self._thrust_scale`,
where `MAX_COLLECTIVE_THRUST = 4.5 * DRONE_MASS * GRAVITY` (thrust-to-weight
ratio of 4.5). Body rates become `target_body_rates = delayed_action[1:] * MAX_BODY_RATE`
and are tracked by a Betaflight-style PD rate controller running at
physics rate (see `_apply_forces`).

Motor dynamics are a first-order lag: `MOTOR_TAU = 0.02` s,
scaled by curriculum difficulty (tau → 0 at easy, tau → 0.02 at full).

**CTBR was chosen over direct motor control** because it abstracts the
rate loop, matches the published drone-racing literature (Swift, Nature
2023), and is what real Betaflight-style flight controllers expose.
See [[decisions-log#CTBR over direct motor control]].

## Observation spaces

There are **two observation modes**, selected by the `vision_mode` flag.

### State mode (24D Box)

Used by `train_state.py` → `aigp_state_final.zip` (the privileged
expert) and by `train_8gates.py` / `train_resume.py` → the
`aigp_8gates_final.zip` / `aigp_racer_final.zip` artifacts.

24 dimensions, in this exact order:

| Slice | Content |
|---|---|
| [0:3] | position (x, y, z) |
| [3:6] | linear velocity |
| [6:9] | Euler RPY |
| [9:12] | angular velocity, world frame |
| [12:15] | angular velocity, body frame |
| [15:18] | relative gate position (gate − drone) |
| [18] | relative gate yaw |
| [19:23] | previous action |
| [23] | gates remaining / total gates |

This is **privileged** — `position` and `relative_gate_*` are
ground-truth, which is why state-expert training converges fast
(~10–20M steps) and why we use it as the teacher in DAgger
distillation.

Observation noise is added *only* to the obs delivered to the policy;
rewards are computed on the clean observation. This is intentional and
matches the Swift methodology: train with noisy perception but reward
against ground truth to keep the reward signal clean.

### Vision mode (Dict)

Used by `train_vision.py` (abandoned end-to-end run) and `train_distill.py`
→ `aigp_distill_final.zip`.

```
{
  "image": Box(uint8, shape=(48, 48, 14)),   # 2 stacked frames × (3 RGB + 4 event)
  "state": Box(float32, shape=(19,)),        # 6D rotation, VIO estimates, prev action
}
```

The 19D state layout (from `_get_minimal_state`):

| Slice | Content |
|---|---|
| [0:6] | 6D rotation (first two columns of body-to-world rotmat) |
| [6:9] | VIO velocity estimate |
| [9:12] | VIO angular rates |
| [12:16] | previous action |
| [16:19] | VIO position estimate |

See [[vision-model]] for why 6D rotation and why VIO estimates instead
of ground truth.

## Rewards (`_reward_*` in config, summed in `step()`)

| Term | Config value | Trigger |
|---|---|---|
| Gate passed | `+100.0 × multiplier` | Drone inside `GATE_TOLERANCE = 1.0` m of gate center |
| Progress | `+10.0 × Δdistance` | Every step, rewards closing distance to next gate |
| Perception | `−0.5` | Bearing error > 90° (gate out of FOV) |
| Cmd smoothness | `−0.02 × ‖action[1:]‖₁` | Every step, discourages jerky rate commands |
| Body rate penalty | `−0.01 × ‖ω‖²` | Every step, encourages smooth attitude |
| Time penalty | `−0.1` | Every step, speed incentive |
| Crash | `−200.0` | z < 0.1, z > 20, out of bounds, or roll/pitch > 120° |

The gate multiplier scales with gate index (`1.0, 1.1, ..., 1.7`) so
later gates are worth more — this reduces the incentive to "settle"
into a safe local loop and rewards full-track completion.

The **root** `drone_race_env.py` also has an alignment bonus multiplier
(`0.5 + 0.5 × |v̂·ĝ|`) applied on gate-pass that rewards flying through
the gate aligned with the gate's forward direction rather than
crabbing through sideways. The `drone-race-sim/` copy of this file
does **not** have that term — that's the older version. See
[[fragilities#Canonical copy of each file]].

## Domain randomization

Enabled by default (`DOMAIN_RAND = True`), scaled by curriculum
difficulty so randomization starts small and grows:

| Param | Range |
|---|---|
| Mass | ±5–10% (`MASS_RANGE = (0.9, 1.05)`) |
| Thrust noise | ±5% per-episode scale factor |
| Action latency | 0–2 control-step delay |
| Aerodynamic drag | 0.0–0.3 coeff |
| Observation delay | 0–2 control-step delay |
| Per-axis observation noise | See config `OBS_NOISE_*` |

All noise terms are gated behind `self._difficulty`, which the
curriculum callback advances from 0.5 up to 1.0 over training.

## Silent load-bearing constants

Things in `config.py` that affect training but rarely get discussed:

- `SIM_FREQ = 200` vs `CTRL_FREQ = 100` — physics is 2× control rate.
  Changing this re-tunes the entire rate controller. The DCL sim is
  120 Hz, which we don't match. See [[fragilities#Training / deployment physics-rate mismatch]].
- `MOTOR_TAU = 0.02` — first-order motor lag. Scaled by difficulty.
  Changing this changes the effective control bandwidth and interacts
  with `RATE_KP = 60`, `RATE_KD = 2`.
- `EPISODE_LENGTH_SEC = 30` — training horizon. Race horizon is 480 s.
- `FPV_TILT_DEG = -10` — camera tilted 10° down from forward. **The
  uncommitted `drone-race-sim` copy has `0` and is wrong.** See
  [[fragilities#FPV_TILT_DEG local edit does not reflect training]].
- `EVENT_CAMERA_ENABLED = True`, `EVENT_FRAME_BINS = 2` — events are
  always on during training. Total image channels is
  `(3 + 2*2) × 2 = 14` per stacked observation. See [[perception]].
- `MOTION_BLUR_WARMUP = 50_000` — blur is off for the first 50K steps
  so the CNN learns clean geometry first. Staged curriculum.
- `VIO_UPDATE_RATE = 30` Hz — visual corrections are applied at camera
  rate, IMU integration runs at physics rate.

## Physics integration details

- MuJoCo free-body quadrotor with `xfrc_applied` for direct
  thrust/torque application (no motor actuators; we compute torques
  via PD on body rates and apply them).
- No aerodynamic table; drag is linear-in-`|v|·v` with a randomized
  coefficient.
- Arena: 40 m × 40 m, no walls; out-of-bounds is a crash.
- Gates: 1.5 m × 1.5 m, 8 in sequence, z varying from 1.5 to 4.5 m.
  See `track.py`.

## See also

- [[perception]] — FPV and event camera in detail
- [[vision-model]] — how these obs feed into the network
- [[training]] — curriculum schedules and hyperparameters
- [[fragilities]] — silent training/deployment drifts
