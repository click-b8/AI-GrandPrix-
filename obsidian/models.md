#architecture #deployment

# Model registry

What's actually on disk, where it lives, how it was trained, and
whether it can be deployed.

## aigp_distill_final.zip — VQ1 candidate

- **Path (canonical):** `drone-race-sim/models_release/aigp_distill_final.zip`
- **Docs path (wrong):** `~/Desktop/AI GrandPrix/AI_GrandPrix_Models/aigp_distill_final.zip`
  — file does not exist here. See [[fragilities#Silent failure chain]].
- **Size:** 13 MB (compressed `.zip`; inner `policy.pth` is 4 MB).
- **Type:** vision-based PPO, distilled via DAgger from the state-based
  expert.
- **Training:** `drone-race-sim/train_distill.py`, HPC, 2026-03-12 to
  2026-03-24. `MODELS.json` records 52,000,000 steps.
- **Teacher:** state-based expert (see `aigp_state_final` below, which
  is not on local disk but presumably lives on HPC).
- **Observation:** Dict with `image` (14 × 48 × 48 uint8) and `state`
  (19D). See [[perception]], [[vision-model]].
- **Action:** 4D CTBR — `[thrust ∈ [0,1], roll_rate, pitch_rate,
  yaw_rate ∈ [−1,1] × MAX_BODY_RATE]`.
- **Validated:** not end-to-end. The `test_dcl_adapter.py` script that
  claims to validate it points at a wrong path, and the adapter it
  would validate loads only features, not policy. See
  [[fragilities#The policy head is untrained at deployment]].

**To deploy:** do not use the current `dcl_adapter.py` as-is. See
[[submission-readiness]] for the list of fixes.

## aigp_8gates_final.zip — state-based 8-gate racer

- **Paths:** `drone-race-sim/models_release/aigp_8gates_final.zip`,
  also present at `AI_GrandPrix_Models/aigp_8gates_final.zip` (the
  only model at that path).
- **Size:** 504 KB.
- **Type:** state-based PPO (`MlpPolicy`, 24D privileged observation,
  no CNN).
- **Training:** `drone-race-sim/train_8gates.py`, HPC, 2026-03-24.
  `MODELS.json` records 100M steps.
- **Recorded performance (from recent git commit 5130652):**
  "8/8 gates in 4.27 s". This is a measurement on the sim, with
  privileged observation, privileged gate positions, no vision.
- **Deployable?** Only against sims that accept a 24D state
  observation including ground-truth gate positions. Not usable for
  DCL (DCL doesn't provide gates). Useful as a speed baseline and as
  a teacher for further distillation.

## aigp_racer_final.zip — state-based Swift-baseline racer

- **Path:** `drone-race-sim/models_release/aigp_racer_final.zip`
  (not at root).
- **Size:** 504 KB.
- **Type:** state-based PPO, `[256, 256, 128]` network.
- **Training:** 100M steps, 2026-02-23. Extended from an earlier
  Swift-style baseline via `train_resume.py`. Has a full checkpoint
  ladder in `drone-race-sim/trained_swift_100m/checkpoints/` from
  3.4M to 98M steps.
- **Deployable?** Same caveat as `aigp_8gates`: state-based, requires
  gate positions.
- **Use:** the most-tuned state-based expert; probably the strongest
  teacher for future distillation.

## aigp_state_final.zip — referenced but not on local disk

- **Path expected:** `./trained_state_expert/best_model/best_model.zip`
  or `./trained_state_expert/aigp_state_final.zip` (per
  `train_distill.py:43-46`).
- **Does not exist** in the local checkout.
- Referenced in `README_SCUBA_LAB.md:35` as "Available on HPC, not
  primary competition model".
- Presumably the direct output of `train_state.py` before any
  follow-on training. `train_distill.py` loads it (or the
  `best_model`) as its teacher.

If you need to retrain the distill model, you need this file. If it's
lost from HPC, `aigp_racer_final` can serve as a substitute teacher
(it's a longer-trained state expert with the same observation space).

## Other checkpoint directories on disk

Presence of these directories tells you the training lineage existed
locally at some point:

- `drone-race-sim/trained_swift_100m/` — checkpoint ladder for
  `aigp_racer_final`. Full 3.4M–98M `.zip` chain. Most detailed
  training trace in the repo.
- `drone-race-sim/trained_distilled/` — raw weights (`policy.pth`,
  `policy.optimizer.pth`, `pytorch_variables.pth`) for
  `aigp_distill_final`. **This is what `dcl_adapter.py` actually
  loads.**
- `drone-race-sim/trained_vision_events/` — best-model folder
  referenced by `train_distill.py` warmstart. Used to bootstrap the
  distill.
- `drone-race-sim/trained/`, `trained_10m/`, `trained_50m/`,
  `trained_allgates/`, `trained_fast_safe/`, `trained_finetune/`,
  `trained_old_5gate/` — earlier experimental runs. See
  [[experiments-log]].

## Registry source of truth

`drone-race-sim/models_release/MODELS.json`. Canonical values:

```json
{
  "aigp_8gates_final.zip":  { "training_steps": 100000000, "trained_on": "HPC (Mar 24, 2026)" },
  "aigp_racer_final.zip":   { "training_steps": 100000000, "trained_on": "HPC (Feb 23, 2026)" },
  "aigp_distill_final.zip": { "training_steps":  52000000, "trained_on": "HPC (Mar 12-24, 2026)" }
}
```

The 50M vs 52M discrepancy between `MODELS.json` and the free-text
READMEs is harmless rounding. The 200M claim in the root README is
about the abandoned end-to-end vision lineage, not any shipped model;
that claim should be updated or removed.

## Which model to ship

For VQ1 (vision-based, DCL sim, no gate positions):

- **`aigp_distill_final.zip`** is the only deployable option.
- State-based models cannot be used because DCL does not provide gates.
- But the current deployment adapter (`dcl_adapter.py`) silently
  discards the policy-head weights from this model. Fix the adapter
  before trusting the submission.

For benchmark/comparison (privileged sim, does not need vision):

- `aigp_racer_final` — most reliable state baseline.
- `aigp_8gates_final` — measured 8/8 in 4.27 s, likely faster but
  less tuned.

## See also

- [[training]] — how each model was produced
- [[deployment]] — how any of them reach a simulator
- [[fragilities]] — why the advertised path is wrong and the weights don't load
- [[experiments-log]] — checkpoint ladders and what we have logs for

## aigp_finetune_tilt_final.zip — VQ1 candidate (pending)

- **Path (target):** `drone-race-sim/models_release/aigp_finetune_tilt_final.zip`
- **Status:** Training in progress — HPC job 4654271, FAU shortq7-gpu
- **Type:** Vision-based PPO, distilled via DAgger from state expert
- **Training:** `train_finetune_tilt.py` (wrapper over `train_distill.py`),
  15M steps, warm-start from `trained_vision_events` if available else scratch
- **Teacher:** `trained_state_expert/aigp_state_final.zip`
- **Config delta from aigp_distill_final:**
  - `FPV_TILT_DEG`: 0° → **+20°** (VADR-TS-002 §3.8 match)
  - `EVENT_CAMERA_ENABLED`: True → **False** (RGB-only deployment)
  - Input channels: 14 → **6** (3 RGB × 2 stacked frames)
- **Deployable:** Yes, once training completes and copied to `models_release/`
- **Adapter:** `dcl_adapter.py` auto-detects 6ch input via CNN `in_channels`
- **Entry point:** `run_vq1.py` prefers this over `aigp_distill_final` when present
