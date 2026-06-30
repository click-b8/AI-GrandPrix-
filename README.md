# AI Grand Prix — SCUBA Lab
**Anduril AI Grand Prix** | FAU SCUBA Lab / MPCR Lab | VQ1 active

> **Source of truth for a fresh session.** Read this before re-deriving
> anything. The deployment harness is solved and verified; the bottleneck
> is the policy/weights, not the integration. Don't re-debug the layers
> marked verified below.

---

## ✅ Status — harness verified, policy is the bottleneck (June 2026)

The deployment harness is **verified end-to-end.** The integration is **not**
the problem.

**Verified correct (do not re-debug):**

| Layer | Evidence |
|---|---|
| MAVLink transport (UDP 14550) | ATTITUDE + LOCAL_POSITION_NED parsed; commands land |
| Race start | **Clean legal GO**, confirmed on FPV recording — stale-clock start-gate fix requires a genuine future countdown before arming and ignores stale `race_start` echoes |
| Vertical / z-axis | NED z-down → training z-up conversion verified |
| Rotation (rot_6d) | Two-sided similarity `C·R·C` checked against the training `_quat_to_rotmat` oracle |
| Body rates (dims 9–11) | **Proven correct.** The `qvel[3:6]` body-frame check confirmed MuJoCo free-joint angular velocity is body-frame; training feeds FLU body rates; deployment `C·ω` (FRD→FLU) matches |

**The actual failure:** the existing model checkpoints **fail to fly the
course.** From a clean, legal race start the drone goes to **full thrust and
tumbles within ~2 s** (confirmed by FPV recording). Every observation axis
fed to the model is provably frame-correct, so this is a **policy / weights
problem, not an adapter problem** — the available checkpoints were never
trained to a policy that flies the deployment spec.

See [`obsidian/current-plan.md`](obsidian/current-plan.md) and
[`obsidian/vq1-run-notes.md`](obsidian/vq1-run-notes.md) for detail.

---

## 🎯 Active priority — VQ1

VQ1 is the **current target and closes soon.** In VQ1 the simulator exposes
**full telemetry** — ATTITUDE, LOCAL_POSITION_NED, etc., **no restrictions.**
The bar is **course completion within 8 minutes (§8.1), not speed.** A slow,
clean lap qualifies.

**Path to passing:**

1. **Hover-thrust calibration probe** — in the VQ1 sim, measure DCL's actual
   hover throttle fraction and check it against the trained **0.22**. A
   mismatch here is a prime suspect for the full-thrust tumble: if DCL needs a
   different fraction to hover, a policy calibrated to 0.22 will saturate
   thrust immediately.
2. **Fresh state-primary training run on the DGX** — calibrated to the
   measured hover-thrust, leaning on **position / velocity / attitude** (which
   transfer cleanly in VQ1's full-telemetry regime) with **vision as a
   secondary gate-finding signal.** This is the realistic route to a flying
   policy before VQ1 closes, given the existing vision checkpoints don't fly.

**To run:**
```bash
python3 run_vq1.py --host <dcl_ip> --port <dcl_port>
```

---

## 🤖 Model inventory — read before any "model swap"

⚠️ **`aigp_distill_11600000_steps.zip` and `aigp_finetune_tilt_final.zip` are
the SAME FILE** — byte-identical, same SHA256 (`1C5F6D3C…`), both 12,868,185
bytes. The finetune run diverged numerically (~14M steps) and its "final"
artifact is just the 11.6M pre-divergence snapshot under a second name.
**Switching between these two is a no-op** — do not re-run that "swap" and
expect different flight behavior; it has already cost a wasted flight.

The **only genuinely different vision model on disk** is
**`aigp_distill_final.zip`** (14-channel event-cam distill, 13,210,546 bytes,
SHA256 `81EF9BDF…`).

`run_vq1.py` loads `aigp_distill_final.zip` (`CANONICAL_MODEL_PATH = _DISTILL_PATH`,
`run_vq1.py:79`). It flies (clean takeoff under model control) but **does not
navigate the course** — same policy/weights limitation as above.

| File | SHA256 | Bytes | Note |
|---|---|---|---|
| `aigp_distill_final.zip` | `81EF9BDF…` | 13,210,546 | **Loaded by run_vq1.py.** Only genuinely distinct vision model |
| `aigp_finetune_tilt_final.zip` | `1C5F6D3C…` | 12,868,185 | **Byte-identical to** `aigp_distill_11600000_steps.zip`. Diverged finetune |
| `aigp_distill_11600000_steps.zip` (repo root) | `1C5F6D3C…` | 12,868,185 | Same file as above under a different name |
| `aigp_racer_final.zip` | — | 516,544 | State-based teacher (PPO) |
| `aigp_8gates_final.zip` | — | 516,485 | State-based reference |

---

## 🔭 Future work — VQ2 (NOT current; scoped only)

Per spec **VADR-TS-003 §9.3**, VQ2 qualification **blocks ATTITUDE,
LOCAL_POSITION_NED, ODOMETRY, and GATE_INFO** — forcing a **vision-only**
policy. This is the **next effort after VQ1**, not current work.

- **HIGHRES_IMU survives** (gyro + accelerometer), enabling an
  IMU-proprioceptive observation: **gravity-vector + body-rates + prev_action
  ≈ 10-D** (accel → gravity-down → roll/pitch; gyro → body rates).
- **Visual domain randomization** becomes the primary sim-to-DCL transfer
  lever and is **currently absent** from the env — lighting, gate colors, and
  background are all fixed (only motion blur varies). DCL has dynamic
  lighting; the CNN will not transfer without this.
- **Open confirmations needed from the Race Director** before committing DGX
  time to VQ2: that HIGHRES_IMU stays available in competitive runs, and that
  no hardcoded course prior is permitted (GATE_INFO is blocked and track
  validation is enforced).

---

## 🧠 Architecture

Vision-based racing trained via **privileged distillation** (Swift, Nature 2023):

```
DCL FPV Camera (640×360, UDP 5600)
        ↓ DCLVisionReceiver → resize 48×48
  Coarse-to-Fine CNN → 256D features
        +
  State (19D) ─────────────────────→ PPO Policy → CTBR [throttle, roll, pitch, yaw]
                                                       ↓ × MAX_BODY_RATE (12 rad/s)
                                            SET_ATTITUDE_TARGET (MAVLink v2, UDP)
                                                       ↓
                                            DCL Simulator Flight Controller

DCL MAVLink telemetry (UDP 14550)
        ↓ _MAVLinkReceiver → ATTITUDE / HIGHRES_IMU / LOCAL_POSITION_NED
        → frame-converted 19D state (verified) feeds the policy above
```

The 19-D state is `rot_6d(0-5) + lin_vel(6-8) + ang_rates(9-11) +
prev_action(12-15) + position(16-18)`, mirroring `_get_minimal_state()` in
`drone_race_env.py`. The architecture (`DroneVisionExtractor`) **always**
consumes both image and state — it is not camera-only (relevant to VQ2).

**Key files:**
```
run_vq1.py              ← VQ1 entry point (start here)
dcl_vision_receiver.py  ← Threaded UDP JPEG frame receiver
dcl_adapter.py          ← Vision model wrapper + frame-correct observation build
dcl_mavlink_adapter.py  ← MAVLink v2 encoder
config.py               ← All hyperparameters
drone_race_env.py       ← Training env (observation source of truth)
models_release/         ← Deployable model artifacts
tests/                  ← Compliance + frame-conversion oracle tests
obsidian/               ← Full project wiki (plan, run notes, fragilities)
```

**CLI:**
```
--host          DCL simulator IP (default: 127.0.0.1)
--port          DCL MAVLink control port
--hz            Control rate in Hz
--vision-port   UDP port for FPV stream (default: 5600)
--telem-port    Local UDP port for MAVLink telemetry (default: 14550)
--allow-stub-vision   DEV ONLY — black frames, no receivers started
```

---

## 📚 References

- **Swift** — Champion-level drone racing (Nature 2023). CTBR action space, privileged distillation
- **SkyDreamer** — TU Delft, world-model pixel-to-motor, A2RL Multi-Drone Race 2026 (arXiv 2510.14783)
- **MonoRace** — TU Delft, A2RL × DCL 2025 winner. Competition-spec drone parameters
- **VADR-TS-002 / VADR-TS-003** — DCL technical specifications (in project files; TS-003 defines VQ2)

## 📖 Full Wiki

```
obsidian/
├── current-plan.md         ← Current prioritized plan (READ THIS)
├── vq1-run-notes.md        ← Latest run outcome — harness verified, policy is the bottleneck
├── fragilities.md          ← Known failure modes
├── models.md               ← Model artifact details
├── experiments-log.md      ← Training history
└── open-questions.md       ← Known unknowns
```

---

*SCUBA Lab, FAU | Machine Perception and Cognitive Robotics Laboratory*
