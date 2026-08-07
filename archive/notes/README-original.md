# AI Grand Prix — SCUBA Lab
**Anduril AI Grand Prix** | FAU SCUBA Lab / MPCR Lab | VQ1 — vision + IMU

> **Source of truth for a fresh session.** Read this before re-deriving
> anything. The transport/harness layer is solved and verified byte-equivalent
> to the official sample client. The problem is now exactly one thing: the
> updated sim blocks the state telemetry, so VQ1 needs a **vision + IMU policy
> on a 10-D observation, retrained from scratch.** Don't re-debug the verified
> transport layers; don't try to fly an existing checkpoint — it can't.

---

## 🛑 What changed — the VQ1 sim now blocks state telemetry (June 2026)

Confirmed against the official **v1.0.3379** sample client (`PyAIPilotExample-v2`):
`mavlink_rx.py` marks **ATTITUDE**, **LOCAL_POSITION_NED**, and **ODOMETRY** as
*"disabled in the latest version of the simulator,"* and gate positions are
*"nulled."* **HIGHRES_IMU (accel + gyro)** and the **vision stream** survive.
(The VQ2 spec VADR-TS-003 §9.3 blocks are now live in VQ1.)

**Consequence — every existing checkpoint is unflyable.** Our model requires a
19-D state; **12 of those dims** (rot_6d, lin_vel, position) come from the now-
dead ATTITUDE/LOCAL_POSITION_NED. With their source messages gone, those dims
are pinned to startup defaults, so the model is fed a constant **"level / still
/ at origin" lie** every frame while the drone is actually tilting and
accelerating. This is **not a bad-weights problem** — it is an
**observation-availability problem.** No checkpoint of the old observation can
fly here, because the observation it was trained on no longer exists.

Only **7 of 19** dims survive: body_rates (HIGHRES_IMU gyro) + prev_action
(self). VQ1 and VQ2 have **converged on the same vision-only problem.**

---

## ✅ Verified and standing — the transport/harness layer

Audited byte-for-byte against the v1.0.3379 sample client; this work stands:

| Layer | Evidence |
|---|---|
| MAVLink transport (UDP 14550) | connection/arm/heartbeat match the sample |
| Race start | **Clean legal GO** (FPV-confirmed); stale-clock gate; ENCAPSULATED_DATA race-status parse `<BQqqIq` matches the sample exactly |
| Control send | `SET_ATTITUDE_TARGET` `type_mask=128`, identity quat, rates+thrust — **identical** to the sample's attitude path |
| Vision header | `"<IHHIIQ"` (24 B) — **identical** to the sample |
| Timesync | `timesync_send(now_ns, 0)` @ 10 Hz — identical |
| Body-rate frame | `qvel[3:6]` body-frame check confirmed; deploy `C·ω` (FRD→FLU) correct — **carries forward** to the new obs |

The transport is not the bottleneck and is not in question. What's invalidated
is the **observation**, not the harness.

---

## 🎯 Active priority — VQ1 as a vision + IMU policy

The path forward is a policy on a **10-D observation built only from un-blocked
sources**, retrained from scratch:

```
state (10-D) = gravity_unit[3] (FLU) + body_rates[3] (FLU) + prev_action[4]
image        = (FPV_FRAME_STACK × 3, 48, 48) RGB, consecutive frames
```

- `gravity_unit` ← HIGHRES_IMU accel + gyro via a complementary/Mahony filter
  (gravity-down unit vector in body frame; recovers roll/pitch).
- `body_rates` ← HIGHRES_IMU gyro. `prev_action` ← self.
- **Yaw, position, and velocity are unobservable** from un-blocked sources →
  the **image sequence carries all navigation**, so a real multi-frame stack and
  **visual domain randomization are mandatory and load-bearing** (the env
  currently has none — fixed lighting/gate-colors/background).

**The train==deploy contract is [`obsidian/observation-spec.md`](obsidian/observation-spec.md)** —
the single spec both the env builder and the deploy adapter must implement
byte-identically. Its guardrail, `tests/test_observation_contract.py`, **must
be green (cross-builder layer un-skipped and passing) before any GPU time.**

**Work order:** A1 accelerometer capture (done) → A2 gravity filter → A3 10-D
adapter + real frame stack → A4 IMU-based hover probe → measure filter residual
envelope (local) → B5 env emits the same 10-D obs → B6 visual DR → B7 frame
stack 3–4 → B8 DGX run (`MUJOCO_GL=egl`, with an early eval gate-1 kill-criterion).

```bash
python3 run_vq1.py --host <dcl_ip> --port <dcl_port>   # [IMU] line confirms accel/gyro live
```

---

## 🤖 Model inventory — all current checkpoints are now unflyable

All checkpoints below were trained on the old 19-D state and **cannot fly the
updated sim** (12 of 19 dims dead, per above). They are retained for reference
only; the next flying model comes from the vision+IMU retrain.

⚠️ **`aigp_distill_11600000_steps.zip` and `aigp_finetune_tilt_final.zip` are
the SAME FILE** — byte-identical, SHA256 `1C5F6D3C…`, 12,868,185 bytes. Swapping
between them is a no-op; do not re-run that "swap."

| File | SHA256 | Bytes | Note |
|---|---|---|---|
| `aigp_distill_final.zip` | `81EF9BDF…` | 13,210,546 | Loaded by `run_vq1.py`; only genuinely distinct vision model. Unflyable on dead state. |
| `aigp_finetune_tilt_final.zip` | `1C5F6D3C…` | 12,868,185 | Byte-identical to `aigp_distill_11600000_steps.zip`; diverged finetune |
| `aigp_distill_11600000_steps.zip` (repo root) | `1C5F6D3C…` | 12,868,185 | Same file as above, different name |
| `aigp_racer_final.zip` | — | 516,544 | State-based teacher (PPO) |
| `aigp_8gates_final.zip` | — | 516,485 | State-based reference |

---

## 🧠 Architecture (target: vision + IMU, 10-D)

```
DCL FPV Camera (640×360, UDP 5600)
        ↓ DCLVisionReceiver → resize 48×48, rolling stack of N consecutive frames
  Coarse-to-Fine CNN ┐
                     ├─→ PPO Policy → CTBR [throttle, roll, pitch, yaw]
  State (10-D) ──────┘                 ↓ × MAX_BODY_RATE (12 rad/s)
   gravity_unit + body_rates + prev_action   SET_ATTITUDE_TARGET (MAVLink v2)
        ↑                                            ↓
  HIGHRES_IMU (accel+gyro) → complementary filter   DCL Flight Controller
```

The 19-D state path (`rot_6d + lin_vel + ang_rates + prev_action + position`)
is **retired** — its ATTITUDE/LOCAL_POSITION_NED sources are blocked. The new
observation is defined in `obsidian/observation-spec.md`.

**Key files:**
```
run_vq1.py              ← VQ1 entry point; MAVLink RX (now captures IMU accel)
dcl_vision_receiver.py  ← Threaded UDP JPEG frame receiver ("<IHHIIQ")
dcl_adapter.py          ← Vision model wrapper + observation build (→ 10-D, A3)
dcl_mavlink_adapter.py  ← MAVLink v2 control send (verified vs sample client)
drone_race_env.py       ← Training env (observation source of truth; → 10-D, B5)
config.py               ← All hyperparameters
obsidian/observation-spec.md  ← train==deploy 10-D contract (READ THIS)
tests/test_observation_contract.py  ← guardrail; green before GPU time
models_release/         ← Deployable model artifacts
```

---

## 📚 References

- **Swift** — Champion-level drone racing (Nature 2023). CTBR, privileged distillation
- **SkyDreamer** — TU Delft, world-model pixel-to-motor, A2RL Multi-Drone Race 2026 (arXiv 2510.14783)
- **MonoRace** — TU Delft, A2RL × DCL 2025 winner; proves vision+IMU racing on this recipe
- **VADR-TS-002 / VADR-TS-003** — DCL technical specs (TS-003 §9.3 = the telemetry block, now live in VQ1)
- **PyAIPilotExample-v2 (sim v1.0.3379)** — official sample client; transport ground truth

## 📖 Full Wiki

```
obsidian/
├── observation-spec.md     ← train==deploy 10-D vision+IMU contract (authoritative)
├── current-plan.md         ← Prioritized plan
├── vq1-run-notes.md        ← Run history
├── fragilities.md          ← Known failure modes
├── models.md               ← Model artifact details
├── experiments-log.md      ← Training history
└── open-questions.md       ← Known unknowns
```

---

*SCUBA Lab, FAU | Machine Perception and Cognitive Robotics Laboratory*
