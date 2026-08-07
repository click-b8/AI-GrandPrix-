#training

# Training

The project has **three training lineages**, one of which was abandoned.
The actual competition model came from lineage 2. The docs at root
conflate all three.

## Lineage 1: State-based experts (privileged observation)

**Script:** `drone-race-sim/train_state.py`
**Output:** `aigp_state_final.zip` (referenced but not present on disk)
→ fed into `train_8gates.py` and `train_resume.py` →
`aigp_8gates_final.zip`, `aigp_racer_final.zip`.

- PPO (`MlpPolicy`), 24D state observation, 4D CTBR action.
- Network: `[256, 256, 128]` for both π and V.
- 8 envs, `DummyVecEnv` (no SubprocVecEnv — see [[fragilities#Canonical copy of each file]]
  for the macOS note).
- Rollout: `n_steps=2048`, `batch_size=1024`, `n_epochs=10`.
- LR schedule: exponential decay 3e-4 → 3e-5 (the `lr_schedule`
  callable appears identically in all training scripts).
- Curriculum: start at 0.3 difficulty, ramp to 1.0 by 40% of training,
  then hold. Faster ramp than the vision curriculum because state-based
  learning converges in ~10–20M steps.
- `TOTAL_TIMESTEPS = 50_000_000` in `train_state.py`. The downstream
  8-gate and Swift-resume scripts extended to 100M per `MODELS.json`.

Final artifacts:

| Model | Steps | Source |
|---|---|---|
| `aigp_8gates_final.zip` | 100M | `train_8gates.py` on HPC, 2026-03-24 |
| `aigp_racer_final.zip` | 100M | `train_resume.py` on HPC, 2026-02-23 |

Both are state-based, i.e., use privileged 24D observation with
ground-truth gate positions. They **cannot be deployed** on a real
drone or in the DCL sim because DCL does not provide gate positions.
They are useful only as teachers for distillation and as oracles for
benchmarking.

## Lineage 2: Distilled vision (the VQ1 submission candidate)

**Script:** `drone-race-sim/train_distill.py`
**Output:** `aigp_distill_final.zip` (13 MB, lives at
`drone-race-sim/models_release/`).

- Teacher: a state-based expert loaded from
  `./trained_state_expert/best_model/best_model.zip` or similar.
- Student: vision+state policy using `DroneVisionExtractor` →
  `MlpExtractor(128, 64)`.
- Method: PPO with DAgger-style imitation loss. The `DAggerCallback`
  decays `imitation_weight` from 1.0 down to 0.0 over the first ~67%
  of training, so the student gradually shifts from "copy the expert"
  to "run on its own".
- Curriculum: medium (0.5) for first 10%, ramp to full (1.0) by 50%,
  hold thereafter. Faster than the abandoned vision-only curriculum.
- `TOTAL_TIMESTEPS = 50_000_000`. `MODELS.json` records **52M** for
  the deployed artifact, so training ran slightly past target.

The script supports warmstarting from `./trained_vision_events/best_model/best_model.zip`
— this path points to the abandoned lineage 3 below, used to
bootstrap the distillation.

### Reconciling the "50M + 50M" claim

The top-level docs say "2-stage: 50M (state expert) + 50M (distilled
vision)". Reading this literally, you'd expect a 100M step total. What
actually happened:

- State expert training: 50M steps (per `train_state.py`). But the
  deployed *state-based* artifacts (`aigp_8gates`, `aigp_racer`)
  reached 100M via follow-on scripts.
- Distillation: ~52M steps.

So the "50M + 50M" claim is closer to "52M distillation on top of
a state expert that was first trained for ~50M" — and the teacher
expert itself may have been at 100M by the time distillation ran.
Canonical step count for deployment purposes: **52M for the vision
policy**, trained from a state-expert teacher. See [[models]].

## Lineage 3: End-to-end vision PPO (abandoned)

**Script:** `train_vision.py` (present at root and in `drone-race-sim/`,
identical).
**Intended output:** `aigp_vision_final.zip` — does not exist on disk.

- PPO on vision+state policy from scratch, no expert.
- `TOTAL_TIMESTEPS = 200_000_000`.
- Same feature extractor as the distill lineage.

The root `README.md` describes this lineage ("200M timesteps target")
but the model it produced was never shipped, and there is no
checkpoint record for it in the repo. Likely abandoned in favor of
distillation, which reaches similar quality in a quarter of the
compute. The root README is historically stale on this point.

A `trained_vision_events/best_model/best_model.zip` is referenced as a
warmstart source in `train_distill.py`, suggesting the vision lineage
got far enough to produce a usable-ish checkpoint before being
repurposed as a warmstart and then superseded.

## HPC setup

- **Cluster:** FAU HPC (Florida Atlantic University). Linux 4.18,
  CUDA 12.4, Python 3.12.4, 1× Tesla V100.
- **Jobs:** `drone-race-sim/train_hpc.slurm` orchestrates:
  1. Segfault pre-flight test.
  2. Stage A: `test_distilled_model.py` (validates an existing distill
     model, does not train).
  3. Stage B: `train_8gates.py` if `trained_8gates/aigp_8gates_final.zip`
     doesn't exist.
  4. Stage C: `train_resume.py` if `trained_swift_100m/aigp_racer_final.zip`
     doesn't exist.
- Uses SLURM `USR1` signal to self-resubmit at 5 min before timeout,
  for multi-day training.
- `train_vision_hpc.slurm` exists but has a hardcoded macOS path in
  `cd /Users/mpcrmini2/Desktop/AI\ GrandPrix/drone-race-sim` — this
  script would fail on any HPC node. It is not a working HPC script.

**No SLURM script in this repo produces `aigp_distill_final.zip`.**
The distill model was trained interactively on the HPC or via a script
that was never committed. Its precise provenance is unreconstructible
from this checkout.

## Hyperparameter summary (identical across all PPO runs)

```
learning_rate = lr_schedule (3e-4 → 3e-5 exp decay)
n_steps       = 1024 (vision), 2048 (state)
batch_size    = 512 (vision),  1024 (state)
n_epochs      = 10
gamma         = 0.99
gae_lambda    = 0.95
clip_range    = 0.2
ent_coef      = 0.005
max_grad_norm = 0.5
```

## Curriculum note

The curriculum schedules are hand-tuned and differ per lineage:

| Lineage | Schedule |
|---|---|
| State expert | easy → medium → hard → full by 40%, hold 60% |
| Vision (abandoned) | medium 0–30%, ramp 30–80%, full 80–100% |
| Distill | medium 0–10%, ramp 10–50%, full 50%+ |

The distill lineage ramps fastest because the teacher provides strong
learning signal; the vision-only lineage ramps slowest because it
needs time to learn coarse CNN features before confronting harder
observations.

## SB3 and Python versions

**Training (HPC):** SB3 2.7.1, PyTorch 2.6.0+cu124, Python 3.12.4.
Recorded in `drone-race-sim/trained_distilled/system_info.txt`.

**Local (Mac):** SB3 2.8.0, PyTorch 2.11.0, Python 3.12.13 arm64. The
version gap is why `dcl_adapter.py` loads `policy.pth` directly
instead of using `PPO.load()`. See [[decisions-log#Direct weight loading over PPO.load]].

## See also

- [[models]] — the concrete registry of what's on disk
- [[experiments-log]] — what we have logs for and what's lost
- [[vision-model]] — the architecture all three lineages trained
- [[simulation]] — the environment they trained against
