# AI Grand Prix — SCUBA Lab
**Anduril AI Grand Prix** | FAU SCUBA Lab / MPCR Lab | VQ1 launching end of May 2026

---

## ⚡ Current Status (May 22, 2026)

| Item | Status |
|---|---|
| VQ1 deadline | **End of next week** |
| Deployment pipeline | ✅ Clean — 25 tests passing |
| MAVLink compliance | ✅ Spec-compliant (VADR-TS-002) |
| Active model | ✅ `aigp_distill_final.zip` — deployable now |
| Tilt fine-tune | ⏳ Training on FAU HPC (job 4654527) |
| Vision stream | ⏳ Placeholder — wires in on Day 1 when sim drops |

**To run:**
```bash
python3 run_vq1.py --host <dcl_ip> --port <dcl_port>
```
That's it. One command. The script loads the best available model automatically.

---

## 🤖 Model Inventory

| Model | Location | Type | Tilt | Channels | Status |
|---|---|---|---|---|---|
| `aigp_distill_final.zip` | `models_release/` ✅ in repo | Vision PPO | 0° | 14ch RGB+events | **Fallback — deploy now** |
| `aigp_finetune_tilt_final.zip` | HPC → `models_release/` ⏳ | Vision PPO | +20° (spec) | 6ch RGB | **Primary — pending HPC copy** |
| `aigp_racer_final.zip` | `models_release/` ✅ in repo | State PPO | — | State only | Teacher model |
| `aigp_8gates_final.zip` | `models_release/` ✅ in repo | State PPO | — | State only | Reference |

**`run_vq1.py` automatically prefers `aigp_finetune_tilt_final.zip` when present, falls back to `aigp_distill_final.zip`.** No config change needed.

### Getting the fine-tune model into the repo
Once HPC access is restored:
```bash
# On HPC
cp ~/drone-race-sim/trained_finetune_tilt/aigp_distill_final.zip \
   ~/drone-race-sim/models_release/aigp_finetune_tilt_final.zip

# Then push to GitHub from local machine
git add models_release/aigp_finetune_tilt_final.zip
git commit -m "feat: add tilt-corrected fine-tune model (FPV_TILT=+20, RGB-only)"
git push origin main
```

---

## 🚀 VQ1 Day-One Integration Sprint

The DCL simulator ships concurrent with VQ1. When credentials arrive:

**Hour 1 — Connect:**
```bash
python3 run_vq1.py --host <dcl_ip> --port <dcl_port>
```
Watch logs — confirm heartbeat accepted and telemetry returning.

**Hour 2-3 — Wire vision stream:**
Replace `_get_vision_frame()` stub in `run_vq1.py` with real DCL camera feed.
Resize 640×360 → 48×48 already handled in `dcl_adapter.py`.

**Hour 3-4 — First live run:**
Confirm drone moves toward gate 1. Check `adapter.send_errors` counter in logs if not.

**Day 2 — Submit.**

### Questions to ask DCL on Day 1
1. Is `SET_ATTITUDE_TARGET` accepted? (we use `type_mask=128`, body rates)
2. FPV stream port and encoding? (spec says UDP:5600, JPEG — confirm)
3. Is TIMESYNC handshake required before commands accepted?

---

## 🔧 Spec Compliance (VADR-TS-002)

| Parameter | Spec | Ours | Status |
|---|---|---|---|
| Camera tilt | +20° upward | +20° (fine-tune) / 0° (fallback) | ⏳ Fine-tune pending |
| Camera resolution | 640×360 | 48×48 (adapter resizes) | ✅ |
| FOV | 90° | 90° | ✅ |
| Gate inner size | 1500×1500mm | 1500×1500mm | ✅ |
| Control rate | 50–120 Hz | 50 Hz | ✅ |
| Coordinate frame | NED | NED | ✅ |
| Event camera | RGB only | Disabled in fine-tune | ⏳ Fine-tune pending |
| Physics rate | 120 Hz | 200 Hz training | Validate at sim launch |

---

## 🧠 Architecture

Vision-based autonomous racing trained end-to-end via **privileged distillation** (Swift, Nature 2023):

```
DCL FPV Camera (640×360)
        ↓ resize to 48×48
  Coarse-to-Fine CNN → 256D features
        +
  VIO State (19D) ──────────────────→ PPO Policy → CTBR [throttle, roll, pitch, yaw]
                                                         ↓ × MAX_BODY_RATE (12 rad/s)
                                              SET_ATTITUDE_TARGET (MAVLink v2, UDP)
                                                         ↓
                                              DCL Simulator Flight Controller
```

**Training approach:** State-based expert (full privileged state) supervises vision student (pixels + partial state) via DAgger imitation decay. Same lineage as Swift (Nature 2023) and MonoRace (A2RL 2025 winner).

**Key files:**
```
run_vq1.py              ← VQ1 entry point (start here)
dcl_adapter.py          ← Vision model wrapper + inference
dcl_mavlink_adapter.py  ← MAVLink v2 encoder (spec-compliant)
dcl_mavlink_client.py   ← MAVSDK telemetry + control loop
config.py               ← All hyperparameters
models_release/         ← Deployable model artifacts
tests/                  ← 25 compliance tests (run before submitting)
obsidian/               ← Full project wiki (fragilities, decisions, plan)
```

---

## 📋 For Dr. Pratik — Quick HPC Check

Noah is will be in Nicaragua obtain his sign-in password, check training status with:
```bash
ssh nbrande2020@athenelogin.hpc.fau.edu
squeue -u nbrande2020
tail -50 ~/drone-race-sim/train_hpc_4654527.log | grep -E "ep_rew|timesteps|imitat|config|rror"
```

If job is no longer running and `trained_finetune_tilt/aigp_distill_final.zip` exists — training completed successfully. Copy it to `models_release/aigp_finetune_tilt_final.zip` and push.

If job failed — resubmit:
```bash
cd ~/drone-race-sim && sbatch train_finetune_tilt.slurm
```

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
├── fragilities.md        ← Read this first — known failure modes
├── submission-readiness.md ← Operational checklist
├── vq1-execution-plan.md ← Sequenced branch plan
├── experiments-log.md    ← Training history + active jobs
├── models.md             ← Model artifact details
└── open-questions.md     ← Known unknowns
```

---

*SCUBA Lab, FAU | Machine Perception and Cognitive Robotics Laboratory*
