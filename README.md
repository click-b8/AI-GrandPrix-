# AI Grand Prix — SCUBA Lab
**Anduril AI Grand Prix** | FAU SCUBA Lab / MPCR Lab | VQ1 — May 2026

---

# 🔧 ACTIVE DEBUGGING — Live Integration (June 15, 2026)

> **Read this before the status tables below.** The "✅ Current Status"
> and "Live Stack Validation" sections further down predate live
> simulator integration and are partially superseded by the findings
> here. The system is **not yet flying the course.** Do not treat the
> deployment as submission-ready.

## Where we are

The client now connects, arms, and the drone **lifts off under model
control** — but it does **not navigate the course.** `active_gate`
stays at 0 for the entire run. Root cause identified (below); fix in
progress.

## Confirmed working (verified against live DCL sim)

| Item | Evidence |
|------|----------|
| MAVLink transport on UDP 14550 | ATTITUDE + LOCAL_POSITION_NED parsed; commands land |
| Race-state detection | ENCAPSULATED_DATA sub-type 1 parsed; `race_start_boot_time_ms` flips ≥0 on green flag |
| Arm/thrust gate | Drone stays `armed=True` (base_mode 0xc1) through standby→race; no longer disarms |
| Model takes control at race start | First real command ~46 ms after green flag |
| Liftoff | Drone leaves the ground under model control, every run |

## Root cause of "won't fly the course" — STATE VECTOR MISMATCH

`process_observation()` in `dcl_adapter.py` feeds the model a 19-D state
whose layout **does not match** the training layout in
`_get_minimal_state()`. Every dimension is wrong:

| Dims | Training expects | We currently send |
|------|------------------|-------------------|
| 0–5  | 6D rotation (two cols of body→world R) | position (3) + angular rates (3) |
| 6–8  | linear velocity (world) | Euler angles |
| 9–11 | body angular rates | zeros |
| 12–15| previous action | zeros |
| 16–18| position | zeros |

The model receives noise, so it floors thrust (~1.0) and commands
sustained ~3 rad/s roll — it thrashes rather than tracking gate 0. The
24-D non-vision state has a gate slot, but the **19-D vision state does
not** — gate targeting is done entirely by the CNN on the FPV image, so
no gate-position feed is required; the fix is purely the state layout.

**Next action:** rewrite `process_observation()` to mirror
`_get_minimal_state()` exactly (units, frames, 6D-rotation
construction). Adapter-only change; model untouched.

## Open items found during live debugging

| Item | Detail | Severity |
|------|--------|----------|
| State-vector layout | All 19 dims mismatched vs training (above) | **BLOCKER** |
| Inference on CPU | `Device: cpu` caps control loop at ~20 Hz; move to GPU | High |
| Vision chunk drops | 370→1150 dropped as loop falls behind 30 Hz stream | High (symptom of CPU bottleneck) |
| Control rate vs README | Logs show 250 Hz / port 14550; README says 50 Hz / 14540 — reconcile | Medium |
| 250 Hz vs spec | VADR-TS-002 §4.4 caps command rate at <100 Hz; clamp before submission | Medium |
| Shutdown traceback | KeyboardInterrupt/CancelledError uncaught on teardown (cosmetic) | Low |

## Diagnostics currently in the code (remove/quiet before submission)

- `[Race Status]` — race_start_boot_time_ms / active_gate / race_finish_ns
- `[Sim HB]` — armed / base_mode / system_status
- `[CMD out]` — race_started / thrust / body rates sent to wire
- `[State 19D fed to model]` — per-dimension state dump

---

## ⚡ Current Status (May 28, 2026)

| Item | Status |
|---|---|
| Deployment pipeline | ✅ 28/28 tests passing |
| MAVLink compliance | ✅ Spec-compliant (VADR-TS-002) |
| Deployment model | ✅ `aigp_distill_11600000_steps.zip` — 11.6M steps, reward −162, stable |
| Vision stream | ✅ DCLVisionReceiver wired — UDP port 5600, threaded, VADR-TS-002 s4.6 |
| MAVLink telemetry | ✅ `_MAVLinkTelemetryReceiver` wired — ATTITUDE + HIGHRES_IMU parsed |
| Camera tilt | ✅ +20° (fine-tune complete) |
| RGB only / event camera | ✅ Event camera disabled |
| Fine-tune training | ⚠️ 3 HPC runs complete; 3rd diverged at ~14M steps — DGX run pending |
| DGX training | ⏳ Not yet started — fresh start from 11.6M warmstart, LR ~1e-5 |

**To run:**
```bash
python3 run_vq1.py --host <dcl_ip> --port <dcl_port>
```
One command. Loads the best available model automatically. No other setup needed.

---

## Live Stack Validation (May 29, 2026)

| Check | Result |
|---|---|
| Control loop Hz | ✅ 50.0 Hz (target 50 Hz) |
| Frames sent (30s) | ✅ 1418 |
| Send errors | ✅ 0 |
| UDP packets absorbed | ✅ 1474 (1418 control + 56 heartbeats) |
| Model load time | ✅ ~1.7s |
| Shutdown | ✅ Clean — finally block fired, no hang |
| Test method | In-process threading.Timer, UDP sink on localhost:14540 |

---

## 🤖 Model Inventory

| Model | Location | Steps | Reward | Status |
|---|---|---|---|---|
| `aigp_distill_11600000_steps.zip` | `models_release/` ✅ | 11.6M | −162 | **DEPLOY THIS** — stable pre-collapse checkpoint |
| `aigp_finetune_tilt_final.zip` | `models_release/` ✅ | ~4–5M | — | Backup only (earlier checkpoint from tilt fine-tune) |
| `best_model.zip` | repo root | — | — | ❌ DO NOT USE — from a collapsed run |
| `best_model_tilt.zip` | repo root | ~4–5M | — | Backup only — same as finetune_tilt_final |
| `aigp_racer_final.zip` | `models_release/` ✅ | — | — | Teacher model (state-based PPO) |
| `aigp_8gates_final.zip` | `models_release/` ✅ | — | — | Reference model |

**`run_vq1.py` automatically prefers `aigp_finetune_tilt_final.zip` when present, falls back to `aigp_distill_final.zip`.** No config change needed.

### Fine-tune training history
Three HPC runs completed (job IDs 4656682, 4656743, 4657222). Third run diverged numerically at ~14M steps — training stopped. Best stable checkpoint is the 11.6M-step pre-collapse snapshot deployed above. DGX run is the next step: fresh start from the 11.6M warmstart, LR ~1e-5.

---

## 🚀 VQ1 Day 1 Checklist

```bash
python3 run_vq1.py --host <dcl_ip> --port <dcl_port>
```

1. **Confirm heartbeat accepted** — look for `[SCUBA Lab MAVLink]` in logs, no send errors
2. **Confirm vision frames arriving** — `[DCL Vision] Frames received:` should increment
3. **Confirm telemetry non-zero** — attitude/velocity should leave 0.0 within a few seconds
4. **First live run**

**If telemetry port is wrong:** DCL may not send to default 14550 — override with:
```bash
python3 run_vq1.py --host <dcl_ip> --port <dcl_port> --telem-port <actual_port>
```

**If vision port is wrong:**
```bash
python3 run_vq1.py --host <dcl_ip> --port <dcl_port> --vision-port <actual_port>
```

### Questions to confirm with DCL on Day 1
1. Is `SET_ATTITUDE_TARGET` accepted with `type_mask=128` (body rates)?
2. FPV stream port? (we default to UDP 5600 per VADR-TS-002 s4.6)
3. MAVLink telemetry port? (we default to 14550, standard GCS port)
4. Is TIMESYNC handshake required before commands accepted?

---

## 🔧 Spec Compliance (VADR-TS-002)

| Parameter | Spec | Ours | Status |
|---|---|---|---|
| Camera tilt | +20° upward | +20° | ✅ |
| Camera resolution | 640×360 | resized to 48×48 in receiver | ✅ |
| FOV | 90° | 90° | ✅ |
| Event camera | RGB only | Disabled | ✅ |
| Gate inner size | 1500×1500mm | 1500×1500mm | ✅ |
| Control rate | 50–120 Hz | 50 Hz | ✅ |
| Coordinate frame | NED | NED | ✅ |
| MAVLink message | SET_ATTITUDE_TARGET | type_mask=128 (body rates) | ✅ |
| Physics rate | 120 Hz (DCL sim) | 200 Hz (training) | ⚠️ Validate on first contact |

---

## 🧠 Architecture

Vision-based autonomous racing trained end-to-end via **privileged distillation** (Swift, Nature 2023):

```
DCL FPV Camera (640×360, UDP port 5600)
        ↓ DCLVisionReceiver (background thread, chunked JPEG per VADR-TS-002 s4.6)
        ↓ resize to 48×48
  Coarse-to-Fine CNN → 256D features
        +
  VIO State (19D) ──────────────────→ PPO Policy → CTBR [throttle, roll, pitch, yaw]
                                                         ↓ × MAX_BODY_RATE (12 rad/s)
                                              SET_ATTITUDE_TARGET (MAVLink v2, UDP)
                                                         ↓
                                              DCL Simulator Flight Controller

DCL MAVLink telemetry (UDP port 14550)
        ↓ _MAVLinkTelemetryReceiver (background thread)
        ↓ ATTITUDE → (roll, pitch, yaw) + angular rates
        ↓ HIGHRES_IMU → body angular rates (high-frequency override)
        → _latest_telemetry (thread-safe, feeds VIO State above)
```

**Training approach:** State-based expert (full privileged state) supervises vision student (pixels + partial state) via DAgger imitation decay. Same lineage as Swift (Nature 2023) and MonoRace (A2RL 2025 winner).

**Key files:**
```
run_vq1.py              ← VQ1 entry point (start here)
dcl_vision_receiver.py  ← Threaded UDP JPEG frame receiver (VADR-TS-002 s4.6)
dcl_adapter.py          ← Vision model wrapper + inference
dcl_mavlink_adapter.py  ← MAVLink v2 encoder (spec-compliant)
dcl_mavlink_client.py   ← MAVSDK telemetry + control loop
config.py               ← All hyperparameters
models_release/         ← Deployable model artifacts
tests/                  ← 28 compliance tests (run before submitting)
obsidian/               ← Full project wiki (fragilities, decisions, plan)
```

**CLI reference:**
```
--host          DCL simulator IP (default: 127.0.0.1)
--port          DCL MAVLink control port (default: 14540)
--hz            Control rate in Hz (default: 50)
--vision-port   UDP port for FPV stream (default: 5600)
--telem-port    Local UDP port to bind for MAVLink telemetry (default: 14550)
--allow-stub-vision   DEV ONLY — black frames, no receivers started
```

---

## ⚠️ Known Open Items

| Item | Detail |
|---|---|
| Physics rate mismatch | Trained at 200 Hz, DCL sim runs at 120 Hz — validate behavior on first contact |
| DCL telemetry port | Default 14550; confirm with DCL on Day 1, override via `--telem-port` |
| DGX training run | Not yet started — fresh from 11.6M warmstart, LR ~1e-5, once DGX access granted |

---

## 📚 References

- **Swift** — Champion-level drone racing (Nature 2023). CTBR action space, privileged distillation, sim-to-real
- **SkyDreamer** — TU Delft, world-model pixel-to-motor policy, won A2RL Multi-Drone Race 2026 (arXiv 2510.14783)
- **MonoRace** — TU Delft, A2RL x DCL 2025 winner. Competition-spec drone parameters
- **VADR-TS-002** — DCL technical specification (in project files)
- **event-sharp-nerf-drones** — Event camera model (Zou et al., IEEE T-RO 2026, UZH RPG)

## 📖 Full Wiki

All audit findings, decisions, fragilities, training history, and open questions:
```
obsidian/
├── fragilities.md          ← Read this first — known failure modes
├── submission-readiness.md ← Operational checklist
├── vq1-execution-plan.md   ← Sequenced branch plan
├── experiments-log.md      ← Training history + active jobs
├── models.md               ← Model artifact details
└── open-questions.md       ← Known unknowns
```

---

*SCUBA Lab, FAU | Machine Perception and Cognitive Robotics Laboratory*
