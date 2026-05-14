#training #measurement

# Experiments log

Honest inventory of what measurement trail exists and what's lost.
The docs at root make specific claims about training dynamics (final
reward ~-170, "converged stably", etc.) but most of the underlying
logs are not in this checkout. This file records what can actually be
reconstructed.

## What we have

### Checkpoint ladders (reconstructible training curves)

- **`drone-race-sim/trained_swift_100m/checkpoints/`** — 18 `.zip`
  files at steps 3.4M, 5.0M, 12.6M, 14.2M, 27.8M, 29.2M, 42.4M, 53.2M,
  55.6M, 60.8M, 68.6M, 81.8M, 89.6M, 98M, plus `best_model/best_model.zip`.
  Produces `aigp_racer_final.zip` at 100M. **This is the most detailed
  training trace in the repo.**

- **`drone-race-sim/trained_finetune/`** — `aigp_finetune_final.zip`
  plus checkpoints up to 9.8M steps (confirmed 2026-05-14). Policy class:
  `ActorCriticPolicy` (state-based `MlpPolicy`), Box observation space.
  **Not a vision model — cannot be used as vision warmstart.**

- **`drone-race-sim/trained_distilled/`** — final weights only
  (`policy.pth`, `policy.optimizer.pth`, `pytorch_variables.pth`).
  No checkpoint ladder. Reward curve not locally reconstructible.

### Recorded measurements

- **`drone-race-sim/train_events.log`** — exists, gitignored. Worth
  grepping for "ep_rew_mean" to pull the distill reward curve.

- **MODELS.json** — records training_steps and training dates for the
  three final artifacts. Canonical registry.

- **`COMMIT_EDITMSG`** — one commit message claims 8gates model
  finishes "8/8 gates in 4.27 s". Single observation, not stats-backed.

### Measurements taken during the audit (2026-04-21)

- **Inference time on Apple M4 CPU, end-to-end adapter.step():** mean
  0.26 ms, p95 0.30 ms, p99 0.35 ms, min-max 0.23–0.40 ms over 500
  samples with 20-sample warmup.

- **Policy-head weight check (pre-fix):** pi_net[0].weight std 0.0347,
  max 0.060 — Kaiming-uniform at fan_in 275. Policy was running random
  init. Fixed on branch `fix/policy-weight-load` (merged).

- **FPV_TILT_DEG investigation (2026-04-21):** HPC config had 0° at
  training time (uncommitted local edit). Root config had -10°. Model
  was trained at 0°, not -10° as previously believed.

### Spec gap analysis (2026-05-14)

Cross-referenced training config against VADR-TS-002:

| Parameter | Spec | Training (original) | Resolution |
|---|---|---|---|
| FPV_TILT_DEG | +20° upward | 0° (HPC) / -10° (root) | **Retrain: set to +20°** |
| Camera resolution | 640×360 px | 48×48 px | Adapter resizes — no retrain needed |
| Physics rate | 120 Hz | 200 Hz | Validate at sim release |
| Event camera | RGB only | Enabled (14ch) | **Retrain: disabled (6ch)** |
| Gate inner size | 1500×1500mm | 1500×1500mm | ✅ Match |
| FOV | 90° | 90° | ✅ Match |
| Control rate | 50–120 Hz | 100 Hz | ✅ Within spec |

## Active training run (2026-05-14)

**Job 4654271** on FAU HPC (shortq7-gpu, nodegpu partition):
- Script: `train_finetune_tilt.py` (wrapper over `train_distill.py`)
- Teacher: `trained_state_expert/aigp_state_final.zip`
- Warmstart: `trained_vision_events/best_model/` if available, else scratch
- Steps: 15,000,000
- Config: `FPV_TILT_DEG=20`, `EVENT_CAMERA_ENABLED=False`
- Output: `trained_finetune_tilt/aigp_distill_final.zip`
- Self-resubmits via SIGUSR1 at 5 min before 5:55 wall time
- Expected duration: ~6 hours on V100

**Warmstart note:** `trained_finetune/aigp_finetune_final.zip` was
confirmed state-based (`ActorCriticPolicy`, Box obs space) on
2026-05-14. Removed from `VISION_WARMSTART_PATHS` in wrapper.
`train_distill.py` will use `trained_vision_events` if present,
otherwise trains from scratch with expert guidance.

## What we don't have

### Reward curves for the deployed model

No reward-vs-step trace for `aigp_distill_final`. `train_events.log`
may contain this.

### Completion-rate statistics

No batch evaluation exists. "8/8 in 4.27 s" is a single observation.

### Inference time on target hardware

Measured on Apple M4 CPU only. Competition target is ~100 TOPS embedded.

## Experimental runs (inferred from directory structure)

| Run | Directory | What it probably was |
|---|---|---|
| Early 5-gate baseline | `trained_old_5gate/` | Original 5-gate course |
| 10M-step | `trained_10m/` | Short ablation run |
| 50M-step | `trained_50m/` | Mid-length ablation |
| All-gates | `trained_allgates/` | 5→8 gate transition |
| Safe/slow | `trained_fast_safe/` | Safety-focused experiment |
| Fine-tune (state) | `trained_finetune/` | State-based fine-tune — confirmed MlpPolicy |
| Swift 100M | `trained_swift_100m/` | Final state expert; `aigp_racer_final` |
| Distilled | `trained_distilled/` | Final distill run; `aigp_distill_final` |
| Vision events | `trained_vision_events/` | Pure vision PPO warmstart for distill |
| **Tilt fine-tune** | `trained_finetune_tilt/` | **Active — tilt +20°, events off, 15M steps** |

## See also

- [[models]] — what artifacts we have
- [[training]] — how they were trained
- [[fragilities]] — measurements that contradicted existing claims
- [[open-questions]] — measurement gaps to close
