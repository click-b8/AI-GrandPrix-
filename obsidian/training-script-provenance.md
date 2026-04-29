#training #provenance

# Training script provenance

Per-script audit of every `train_*.py` file under `drone-race-sim/`,
plus `train_vision.py` (which now lives at the project root after
`refactor/deduplicate-root-subdir` but is part of the same lineage).
Investigation pass on 2026-04-27 against `origin/main` at `c557756`.

This document is a **read-only audit**. Nothing was modified. The
investigator wanted to do nothing outside the scope; no deferred
actions are listed at the top.

## Overlap with existing wiki

[[training]] already documents the three lineages (state expert →
state-based racers → distill, plus the abandoned end-to-end vision
run). [[experiments-log]] inventories which `trained_*/` directories
exist locally. [[models]] documents the three deployable artifacts
in `models_release/`. This file does **not** repeat that material at
the lineage level. It instead audits each `train_*.py` script
individually so a reader can decide per-file what to do with it.

Where this audit and the existing wiki disagree (none yet), trust
the wiki.

## Script-by-script

Order is roughly "most load-bearing first". The single load-bearing
script for VQ1 is `train_distill.py`; everything else is either an
ancestor in the lineage or a divergent experiment.

---

### 1. `drone-race-sim/train_distill.py`

1. **Goal.** Stage 2 of the Swift-style two-stage pipeline:
   distill a state-based expert (privileged 24D observation) into a
   vision policy (48×48 image + 19D state) via PPO + DAgger-style
   imitation. Decays an `imitation_weight` from 1.0 to 0.0 over the
   first ~67% of training, so the student gradually becomes
   self-reliant.
2. **Curriculum / reward / config.** Curriculum starts at 0.5
   difficulty for the first 10%, ramps to 1.0 by 50%, holds. Motion
   blur warmup at `MOTION_BLUR_WARMUP` steps. PPO with
   `MultiInputPolicy`, `n_steps=1024`, `batch_size=512`,
   `n_epochs=10`, `ent_coef=0.005`, LR exp decay 3e-4 → 3e-5.
   `TOTAL_TIMESTEPS = 50_000_000`. Reward signal is whatever
   `DroneRaceEnv` provides (this script doesn't customize rewards).
3. **Action / observation.** Action: 4D CTBR (thrust + 3 rates).
   Observation: Dict — `image` (14 × 48 × 48 uint8: 2 stacked frames
   of 7 channels each = 3 RGB + 4 event histogram bins) and `state`
   (19D: 6D rotation + VIO vel/angvel + prev action + VIO pos).
4. **Output existence.** Yes. The deployed VQ1 candidate
   `models_release/aigp_distill_final.zip` was produced by this
   script (HPC, 2026-03-12 to 2026-03-24, ~52M steps per
   `MODELS.json`). The unzipped weights live at
   `trained_distilled/` (root, after dedupe) — `policy.pth`,
   `policy.optimizer.pth`, `pytorch_variables.pth`, plus
   `system_info.txt` confirming SB3 2.7.1 / PyTorch 2.6.0+cu124 /
   Python 3.12.4 / Linux (matches the HPC config). No checkpoint
   ladder survived; only the final state.
5. **Live-code dependency.** `dcl_adapter.py` (root) loads
   `policy.pth` from this directory. The deployed VQ1 entry point
   `run_vq1.py` instantiates `SCUBALabAdapter`, which loads this
   artifact. `test_dcl_adapter.py` and `measure_inference_time.py`
   exercise the same. **This is the single most load-bearing
   training output in the repo.**
6. **Classification: (a) produced a still-used artifact.**
   **Recommendation: archive in place.** The script itself is not
   re-run from `main` (training is done; the artifact is what
   matters), but it's the canonical record of how the deployed
   model came to be. Keep it where it lives. Do not promote it out
   of `drone-race-sim/` — the rest of `drone-race-sim/` is the
   training-side environment, and pulling this script out would
   strand its imports (`drone_race_env`, `config`).

---

### 2. `drone-race-sim/train_state.py`

1. **Goal.** Stage 1 of the Swift-style pipeline: train a
   state-based "privileged" expert with 24D observation and ground-
   truth gate positions, intended as the teacher for `train_distill.py`.
2. **Curriculum / reward / config.** Faster ramp than vision
   curricula: 0.3 → 0.5 → 0.8 → 1.0 over first 40% of training,
   then full for the remaining 60%. PPO `MlpPolicy` with
   `[256, 256, 128]` for both π and V. `n_steps=2048`,
   `batch_size=1024`, `n_epochs=10`, `ent_coef=0.005`, LR exp
   decay 3e-4 → 3e-5. `TOTAL_TIMESTEPS = 50_000_000`.
   `vision_mode=False` for both train and eval. Has resume-from-
   latest-checkpoint logic.
3. **Action / observation.** Action: 4D CTBR. Observation: 24D
   privileged state (`pos + vel + rpy + angvel + gate_rel +
   prev_action`).
4. **Output existence.** **No local artifact.** Expected output
   `./trained_state_expert/best_model/best_model.zip` or
   `./trained_state_expert/aigp_state_final.zip` does not exist on
   this checkout. Per [[models#aigp_state_final.zip — referenced
   but not on local disk]], the artifact lives only on the HPC.
   `train_distill.py:43-46` will fail with `FileNotFoundError` if
   run locally because the teacher isn't present.
5. **Live-code dependency.** Six archive-side scripts under
   `drone-race-sim/` reference
   `trained_state_expert/best_model/best_model.zip` or
   `trained_state_expert/aigp_state_final.zip` directly:
   `train_distill.py:44-45` (the load-bearing forward link —
   the teacher source for the distill run), `view_expert.py:11-12`,
   `view_expert_opencv.py:15`, `launch_mujoco.py:15`,
   `mujoco_viewer.py:15`, `find_working_model.py:13`. All would
   raise `FileNotFoundError` locally because the artifact is
   HPC-only. No deployment-path code references this script's
   output. **Which teacher checkpoint actually trained the deployed
   `aigp_distill_final.zip` is unknowable from this checkout** —
   the HPC training was done either against this script's output
   or against the longer-trained `aigp_racer_final` (per #3); the
   audit cannot pin down which from local files alone.
6. **Classification: (a) produced a still-used artifact**, but
   the artifact only exists on HPC, not in this repo.
   **Recommendation: archive in place.** This is the recipe for
   reproducing the teacher; if the HPC artifact is lost or the
   distill needs to be retrained from scratch, this script is the
   starting point. Note in [[training]] that the script's
   *output* would have to be recovered from HPC before
   `train_distill.py` can run locally.

---

### 3. `drone-race-sim/train_resume.py`

1. **Goal.** Continue training a previously-trained Swift-style
   state-based PPO model (`./trained_swift/best_model/best_model.zip`)
   for another 100M steps to produce a "championship-level" state-
   based racer.
2. **Curriculum / reward / config.** No explicit curriculum;
   uses `domain_rand=True` for train, `False` for eval, default
   difficulty. `ent_coef=0.001`, LR `3e-4 * progress_remaining`
   linear decay. `total_timesteps = 100_000_000`. Inherits
   policy/value architecture from the loaded base model.
3. **Action / observation.** State-based: 4D CTBR action, 24D
   privileged observation (inherited from
   `DroneRaceEnv(vision_mode=False)`, the default). Same as #2.
4. **Output existence.** Yes. `models_release/aigp_racer_final.zip`
   (504 KB) on disk. The full checkpoint ladder survived in
   `drone-race-sim/trained_swift_100m/checkpoints/` (steps 3.4M
   through ~98M, per [[experiments-log]]) — the most detailed
   training trace anywhere in the repo. Per `MODELS.json`: 100M
   steps, HPC, 2026-02-23.
5. **Live-code dependency.** Listed in `MODELS.json` as a
   shipped artifact. **State-based: not deployable to DCL** because
   DCL doesn't provide gate positions. Useful as a benchmarking
   oracle and as an alternate teacher for re-distillation.
   `train_hpc.slurm` Stage C runs this script if its output is
   missing. `train_domain_robust.py` lists it as a fine-tuning
   source. No live *deployment* code references it.
6. **Classification: (a) produced a still-used artifact** — the
   `.zip` is shipped in `models_release/` and the checkpoint ladder
   is the project's only reconstructible reward curve.
   **Recommendation: archive in place.** Keep the script alongside
   its checkpoints. The script is needed if the HPC ever needs to
   resume training; the checkpoint ladder is needed for
   retrospective reward-curve reconstruction noted in
   [[experiments-log#What we don't have]].

   Note: the script depends on `./trained_swift/best_model/best_model.zip`,
   which **does not exist locally** (no `trained_swift/` directory,
   only `trained_swift_100m/`). The 50M predecessor lives somewhere
   in HPC history or was renamed. This is documented in
   [[experiments-log]] as expected.

---

### 4. `drone-race-sim/train_8gates.py`

1. **Goal.** Push a 5-gate model up to 8/8 gate completion via
   fine-tuning with a relaxed roll/pitch crash limit (90° → 120°,
   "aggressive banking") and progressive gate rewards (later gates
   weighted higher).
2. **Curriculum / reward / config.** No curriculum; trains at
   `difficulty=1.0` from the start (the source model is already
   robust). Source model: `./trained_allgates/best_model/best_model.zip`.
   `ent_coef=0.002`, LR `2e-4 * progress_remaining`,
   `total_timesteps = 100_000_000`. Has resume-from-checkpoint
   logic. The "progressive gate reward" is implemented inside
   `DroneRaceEnv`, not in the script — the script just loads a
   model and trains; the reward shaping is environmental.
3. **Action / observation.** State-based, default
   `DroneRaceEnv(vision_mode=False)`: 4D CTBR action, 24D state.
4. **Output existence.** Yes. `models_release/aigp_8gates_final.zip`
   (504 KB) on disk. Also exists at the third path
   `AI_GrandPrix_Models/aigp_8gates_final.zip` (per
   [[models]]). Per `MODELS.json`: 100M steps, HPC, 2026-03-24.
   The local `trained_8gates/` directory **does not exist** — the
   intermediate checkpoint ladder is HPC-only.
5. **Live-code dependency.** Listed in `MODELS.json`. Same caveat
   as #3: state-based, not DCL-deployable; useful as benchmark and
   teacher. `train_hpc.slurm` Stage B runs this script if its
   output is missing. Commit `5130652` ("Restore aigp_8gates_final
   model (state-based expert, 8/8 gates in 4.27s)" — verified by
   `git log --grep "8/8"` against the outer repo) is the
   provenance for the "8/8 in 4.27 s" claim — the only concrete
   performance number anywhere in the repo. One archive-side
   reference: `drone-race-sim/live_sim.py:18` has
   `MODEL_PATH = './trained_8gates/best_model/best_model.zip'`,
   but the directory `drone-race-sim/trained_8gates/` does not
   exist locally — the path is **stale**, and `live_sim.py` would
   fail to load the model on this checkout. No live *deployment*
   code references this script's output.
6. **Classification: (a) produced a still-used artifact**.
   **Recommendation: archive in place.** Keep alongside `train_resume.py`
   as the state-based-racer pair.

---

### 5. `drone-race-sim/train_allgates.py`

1. **Goal.** Fine-tune from a 5/8-gate model to all 8 gates. Same
   target as #4 but a *predecessor* — its output feeds into
   `train_8gates.py` as the latter's source model.
2. **Curriculum / reward / config.** No curriculum; trains at
   full Swift difficulty from the start. Source:
   `./trained_finetune/best_model/best_model.zip` (output of #6).
   `ent_coef=0.001`, LR `1.5e-4 * progress_remaining`,
   `total_timesteps = 100_000_000`. No resume-from-checkpoint
   logic.
3. **Action / observation.** State-based, default observation:
   4D CTBR, 24D state.
4. **Output existence.** Yes locally — `drone-race-sim/trained_allgates/`
   exists with `aigp_allgates_final.zip` (516 KB), a `best_model/`,
   and a 502-file checkpoint ladder. `aigp_allgates_final.zip` is
   **not in `models_release/`** — it's a stepping stone, not a
   shipped artifact.
5. **Live-code dependency.** Verified by complete grep
   (`grep -rn "trained_allgates" drone-race-sim/ --include="*.py"`):
   the artifact is the **most-loaded model in the entire `drone-race-sim/`
   archive directory**. 14 archive-side scripts reference it
   (11 call `PPO.load("trained_allgates/aigp_allgates_final.zip")`
   directly; 3 include it in a candidate-paths list):

   ```
   debug_takeoff.py, debug_gates.py, race_visualization.py,
   watch_race_mujoco.py, render_race_to_video.py, run_race_fixed.py,
   capture_mujoco_race.py, show_gates_live.py, run_autonomous_race.py,
   create_race_animation.py, run_working_race.py,
   benchmark_local_models.py, run_best_model.py, find_working_model.py
   ```

   Plus `train_8gates.py:19` as the forward source-link (the
   8gates training reads this script's `best_model/`). All 15
   references are inside `drone-race-sim/` (archive-side); **none
   are on the deployment path**. The `aigp_allgates_final` model
   was superseded as a *training-pipeline* stepping stone (8gates
   has moved past it), but the local benchmark / debug / render /
   visualization tooling has not migrated to a different model.
6. **Classification: (b) produced an artifact that's been
   superseded.** The `aigp_allgates_final.zip` is a transitional
   model in the 5→8 gate fine-tuning chain, no longer needed for
   anything except retrospective audit.
   **Recommendation: archive in place.** Worth keeping for now —
   the checkpoint ladder is one of the few intact training traces
   in the repo, and `train_8gates.py` documents the lineage by
   reference. Could be deleted post-VQ1 if disk pressure ever
   matters; would not affect any deployment-side code.

---

### 6. `drone-race-sim/train_finetune.py`

1. **Goal.** Fine-tune the 50M-step "5/8 gates" model
   (`./trained_50m/best_model.zip`) with gradual noise introduction
   to make it robust without losing racing skill. The output feeds
   into #5 (`train_allgates.py`).
2. **Curriculum / reward / config.** Curriculum: 0.0 (warm up,
   first 10%) → 0.3 (by 30%) → 0.7 (by 60%) → 1.0 (by end).
   `ent_coef=0.0005` ("very low entropy — preserve learned
   policy"), LR `1e-4 * progress_remaining` (lower than #5).
   `total_timesteps = 50_000_000`. No resume-from-checkpoint
   logic.
3. **Action / observation.** State-based, default observation.
4. **Output existence.** Yes locally —
   `drone-race-sim/trained_finetune/` with `aigp_finetune_final.zip`
   (516 KB), a `best_model/`, and a 252-file checkpoint ladder.
   Not in `models_release/`.
5. **Live-code dependency.** Three references (verified by
   `grep -rn "trained_finetune" drone-race-sim/`):
   - `train_allgates.py:23` — forward source-link (reads
     `best_model/best_model.zip`).
   - `drone-race-sim/run_final.py:27` — loads
     `trained_finetune/checkpoints/aigp_finetune_9600000_steps.zip`
     (a specific intermediate 9.6M-step checkpoint from the 252-
     file ladder).
   - `drone-race-sim/run_no_render.py:13` — loads the same
     9.6M-step checkpoint.

   Both `run_final.py` and `run_no_render.py` are archive-side;
   neither is on the deployment path. They preserve a working
   reference to a specific mid-training snapshot, not the final
   model.
6. **Classification: (b) produced an artifact that's been
   superseded.** Same situation as #5 — earlier in the chain, also
   superseded by downstream training.
   **Recommendation: archive in place.** Same reasoning as #5.

---

### 7. `train_vision.py` (project root after dedupe)

1. **Goal.** End-to-end vision PPO from scratch, no expert. The
   "Lineage 3" run that the project's root `README.md` describes
   ("200M timesteps target") and that was eventually abandoned in
   favor of the distillation lineage.
2. **Curriculum / reward / config.** Curriculum: 0.5 (first 30%)
   → 0.8 (by 60%) → 1.0 (by 80%), hold. PPO `MultiInputPolicy`
   with the `DroneVisionExtractor` (Coarse-to-Fine CNN +
   SE(3)-aware state MLP, 256-dim features). Motion blur warmup
   at `MOTION_BLUR_WARMUP`. `n_steps=1024`, `batch_size=512`,
   `n_epochs=10`, `ent_coef=0.005`, LR exp decay 3e-4 → 3e-5.
   `TOTAL_TIMESTEPS = 200_000_000`. Has resume-from-checkpoint
   logic.
3. **Action / observation.** Action: 4D CTBR. Observation: same
   Dict shape as `train_distill.py` — `image` 14×48×48 uint8 +
   `state` 19D.
4. **Output existence.** Partial. The intended final
   `aigp_vision_final.zip` does **not** exist on disk anywhere.
   `drone-race-sim/trained_vision_events/` exists locally with a
   `best_model/` and a (now-empty) `checkpoints/` directory plus
   `tb_logs/`, `eval_logs/`. So training got far enough to write
   a "best so far" but never reached completion (200M target).
5. **Live-code dependency.** Verified by complete grep
   (`grep -rn "trained_vision_events"` and
   `grep -rn "from train_vision"`). Two distinct dependency
   classes:

   *Module imports (use the `train_vision.py` module itself):*
   - `drone-race-sim/test_hpc.py:5` and
     `drone-race-sim/test_hpc_segfault.py:5` import
     `make_train_env`, `make_eval_env`, `DroneVisionExtractor`.
   - `train_distill.py:55` documents
     "Same architecture as `train_vision.py` for compatibility"
     — the file defines the architecture-of-record for the
     vision feature extractor.

   *Artifact loads (use the `trained_vision_events/` output):*
   - `train_distill.py:50` — `VISION_WARMSTART_PATHS` warmstart
     for the distill run.
   - `README.md:57` (top-level project README) —
     `mjpython race.py --model trained_vision_events/best_model/best_model.zip`
     in the quickstart section.
   - `README.md:74` — same directory listed in the project-
     structure tree.
   - `drone-race-sim/_view_now.py:53`,
     `drone-race-sim/benchmark_local_models.py:29`,
     `drone-race-sim/live_fpv.py:22`,
     `drone-race-sim/run_best_model.py:12`,
     `drone-race-sim/run_vision_race.py:17` — each loads
     `best_model/best_model.zip`.
   - `drone-race-sim/view_march19.py:16` — loads a specific
     checkpoint (`aigp_vision_32000000_steps.zip`) which is **no
     longer present**: the `checkpoints/` directory is empty. Stale
     path.

   The deployed `dcl_adapter.py` does **not** import this script
   — but the trained vision policy it loads uses the same feature-
   extractor topology that this script defines. Whether the top-
   level README's references to `trained_vision_events/best_model/`
   point at currently-correct or stale artifacts is outside this
   audit's scope; the references are recorded here as live-code-
   dependency facts only.
6. **Classification: (b) produced an artifact that's been
   superseded** — `trained_vision_events/best_model/` exists but
   was used only as warmstart for the distill run, and the distill
   final eclipses it.
   **Recommendation: keep at root, document as architecture-of-
   record.** Even though end-to-end vision PPO was abandoned as a
   training method, this file's `DroneVisionExtractor` is the
   shared definition that `train_distill.py` references textually
   and that the deployed `aigp_distill_final.zip`'s checkpoint
   weights are shaped to fit. Removing it would orphan the
   compatibility comment in `train_distill.py:55`. Note: this is
   the only `train_*.py` file currently at the repo root rather
   than under `drone-race-sim/`; that asymmetry is a side effect
   of `refactor/deduplicate-root-subdir` and is by design.

---

### 8. `drone-race-sim/train_fast_safe.py`

1. **Goal.** Train a vision-mode policy with reward shaping
   focused on speed + safety + completeness ("high crash penalties,
   smooth control, body rate limits, extra reward for finishing
   all 8 gates"). The docstring says it can be distilled later,
   but its execution path uses `MultiInputPolicy` directly — i.e.,
   it's a *vision-mode end-to-end* run, not a state-based one
   despite the docstring's "state observation (privileged info)
   for rapid convergence" line.
2. **Curriculum / reward / config.** Curriculum: 0.2 → 0.5 →
   0.8 → 1.0 over the first 60%. `ent_coef=0.0005` ("very low
   entropy = deterministic fast flying"), LR `3e-4 *
   progress_remaining` linear decay. `total_timesteps =
   100_000_000`. The reward shaping language ("high crash
   penalties, completeness bonus") is **not implemented in the
   script** — those rewards either live in `DroneRaceEnv` (so
   this script just runs against whatever rewards
   `DroneRaceEnv(vision_mode=True)` provides) or were aspirational
   text in the docstring that didn't make it into code. The
   `CurriculumCallback` defined in this script is also
   *unused* — it's instantiated as `curriculum = CurriculumCallback(vec_env)`
   but **not passed** into `model.learn(callback=...)`, which
   only includes `[checkpoint_cb, eval_cb]`. The curriculum is
   inert.
3. **Action / observation.** Vision-mode: action 4D CTBR,
   observation Dict (`image` + 19D state).
4. **Output existence.** Yes locally —
   `drone-race-sim/trained_fast_safe/` exists with `best_model/`,
   `eval_logs/`, and an empty `checkpoints/` directory. **No
   `aigp_fast_safe_final.zip`** — training did not run to
   completion. The `best_model/` was created mid-run but
   `checkpoints/` is empty (`save_freq=100_000` means a 100M-step
   target should have produced ~1000 checkpoints; zero implies
   training was interrupted before the first checkpoint cadence
   landed any files, or those files were cleaned up).
5. **Live-code dependency.** Nothing on `main` references this
   script's output. It's mentioned in `COMPETITION_STRATEGY.md`
   ("`train_fast_safe.py` - Training now") and in
   `push_when_done.sh` (a one-shot shell script that committed it),
   but no current code path loads its model.
6. **Classification: (c) never produced a committed artifact /
   status unclear.** The `best_model/` directory exists but has no
   accompanying final `.zip` and an empty checkpoint dir; the
   training was started and abandoned. The script has two latent
   bugs (curriculum callback never wired into `model.learn`, and
   the docstring's reward shaping isn't in the code) which suggest
   the script was authored mid-iteration and not fully completed.
   **Recommendation: archive in place** for the historical record
   (it documents an attempted approach), but mark in
   [[experiments-log]] that the run never produced a usable
   artifact and that its curriculum is inert. **Do not** restart
   it without first fixing the curriculum-callback bug; otherwise
   it would silently train at a fixed `difficulty=0.2`.

---

### 9. `drone-race-sim/train_curriculum.py`

1. **Goal.** Earliest curriculum-style state-based PPO from
   scratch (no source model, no expert teacher). Phases defined
   inline in the docstring: easy(0–20M) → medium(20–40M) →
   hard(40–70M) → fine-tune(70–100M).
2. **Curriculum / reward / config.** Implements `CurriculumCallback`
   correctly (passes into `model.learn`, unlike `train_fast_safe.py`).
   PPO `MlpPolicy`, `[128, 128]` for both heads (smaller than #2's
   `[256, 256, 128]`). `ent_coef=0.001`, LR `3e-4 *
   progress_remaining` linear, `total_timesteps = 100_000_000`,
   `n_steps=2048`, `batch_size=256`, `n_epochs=10`.
3. **Action / observation.** State-based default: 4D CTBR, 24D
   state.
4. **Output existence.** **No.** The expected output directory
   `./trained_curriculum/` does not exist locally. No
   `aigp_curriculum_final.zip` anywhere in the repo. Either the
   script was never run, or its output was deleted.
5. **Live-code dependency.** None. Nothing references
   `trained_curriculum/` or `aigp_curriculum_final.zip` on
   `main`. It is referenced *only* in `obsidian/training.md`'s
   curriculum-comparison table.
6. **Classification: (d) experimental dead end** (or, less
   confidently, (c) — the line is fuzzy). It looks like an early
   attempt at the curriculum approach that `train_state.py` later
   superseded with a faster ramp and a `[256, 256, 128]` net.
   **Recommendation: archive in place.** It documents an early
   design choice (smaller `[128, 128]` net, slower curriculum)
   that the project moved past. Useful as a baseline-for-
   comparison artifact in the historical record. No deletion
   urgency.

---

### 10. `drone-race-sim/train_domain_robust.py`

1. **Goal.** Fine-tune the *already-shipped* models
   (`aigp_racer_final` and `aigp_distill_final`) with aggressive
   domain randomization (mass ±25%, drag ±30%, +10ms motor
   latency, observation noise, random initial conditions) to
   improve hardware/sim-to-real robustness without needing a
   real drone.
2. **Curriculum / reward / config.** No curriculum;
   `difficulty=1.0` from the start. Loops over both shipped
   models and fine-tunes each for 10M steps with `learning_rate=1e-4`,
   `ent_coef=0.001`, `n_steps=2048`. Saves a `config.json`
   alongside each output recording the domain-rand parameters.
3. **Action / observation.** Inherits from each model loaded:
   `aigp_racer_final` is state-based (24D), `aigp_distill_final`
   is vision-mode (Dict). Action: 4D CTBR for both.
4. **Output existence.** **No.** Expected output
   `./trained_domain_robust/` does not exist locally. No fine-
   tuned variants exist.
5. **Live-code dependency.** None. Nothing on `main` references
   `trained_domain_robust/` or any output of this script. It is
   not referenced by any other training script, deployment script,
   or test.
6. **Classification: (c) never produced a committed artifact / status
   unclear.** The script has functional `try`/`except` paths
   around model-loading (multi-path lookup that searches
   `models_release/`, `../`, and `.`) and around training; if it
   was run and failed, the `except Exception` would have printed
   a traceback rather than silently no-op'ing — but no log of
   that exists either. Most likely never executed.
   **Recommendation: archive in place.** It encodes a sensible
   forward-looking idea (sim-to-real robustness via domain rand)
   that may be worth resurrecting post-VQ1 once a real drone is
   accessible. The configured parameters are reasonable and
   self-documenting via `config.json`. Do **not** delete: this
   is the only script in the project that even attempts to bridge
   the sim-to-real gap that [[fragilities#Real-drone data]] flags
   as a critical gap.

   Note: the script's broad-exception block on training failure
   (`drone-race-sim/train_domain_robust.py:152-155`) wraps the
   entire model-load + train + save sequence in `try / except
   FileNotFoundError / except Exception`. The `except Exception`
   prints `❌ Training error: {e}` and a full traceback before
   continuing to the next model in the loop — so this is a
   **broad continue-on-error pattern, not a silent-failure
   pattern**: errors are visible. The concern is the breadth of
   the catch and the lack of re-raise (per the project's
   "observability first; re-raise or return a typed result"
   guardrail in [[session-prompt-template#Guardrails]]), not
   stealth. If this script is ever revived for real use, the
   `except Exception` should be narrowed (or replaced with
   logging + re-raise) per the project's standard.

---

## Summary table

| # | Script | Output exists? | Used by live code on `main`? | Classification | Recommendation |
|---|---|---|---|---|---|
| 1 | `train_distill.py` | yes (`aigp_distill_final.zip`, `trained_distilled/`) | **yes** — `dcl_adapter.py` loads this | (a) still-used | archive in place |
| 2 | `train_state.py` | only on HPC (no local artifact) | indirectly — produces teacher for #1 | (a) still-used | archive in place |
| 3 | `train_resume.py` | yes (`aigp_racer_final.zip`, full ladder) | shipped in `MODELS.json`, not deployed | (a) still-used | archive in place |
| 4 | `train_8gates.py` | yes (`aigp_8gates_final.zip`) | shipped in `MODELS.json`, not deployed | (a) still-used | archive in place |
| 5 | `train_allgates.py` | yes (`trained_allgates/`, not shipped) | 14 archive-side scripts load it + `train_8gates.py` source-link | (b) superseded | archive in place |
| 6 | `train_finetune.py` | yes (`trained_finetune/`, not shipped) | `train_allgates.py` source-link + 2 archive scripts (run_final.py, run_no_render.py) load 9.6M ckpt | (b) superseded | archive in place |
| 7 | `train_vision.py` (root) | partial (`trained_vision_events/best_model/`) | architecture-of-record; warmstart for #1 | (b) superseded | keep at root |
| 8 | `train_fast_safe.py` | partial (`trained_fast_safe/best_model/`, no final, no checkpoints) | none | (c) status unclear | archive in place |
| 9 | `train_curriculum.py` | no | none | (d) dead end | archive in place |
| 10 | `train_domain_robust.py` | no | none | (c) never produced | archive in place |

Legend recap: (a) produced a still-used artifact; (b) artifact
superseded; (c) never produced a committed artifact / status
unclear; (d) experimental dead end.

## Overall recommendation for the batch

**Archive everything in place. Do not delete anything.** The
`drone-race-sim/` subdirectory is intentionally treated as a
"historical working repo" by the prior audit (`obsidian/architecture.md`
labels these scripts as "(archive)"); this audit confirms that
characterization for every individual script. There is no
deletion pressure: the scripts collectively are <50 KB of source,
and the trained_*/ directories that haven't been re-derived from
HPC are small (≤ 1 MB each except `trained_swift_100m/` and
`trained_vision_events/`).

The only file in this set that is currently load-bearing for
deployment is `train_distill.py`, and only via its *output*
(`trained_distilled/policy.pth` plus `models_release/aigp_distill_final.zip`).
The script itself is not invoked from `main`; it's the recipe for
re-deriving the artifact if HPC is re-engaged.

Two scripts (`train_fast_safe.py`, `train_domain_robust.py`)
contain latent issues that would matter if anyone tried to *run*
them again:

- `train_fast_safe.py` instantiates a `CurriculumCallback` it
  never passes to `model.learn`, so the curriculum is inert. If
  resurrected, the callback wiring needs fixing first.
- `train_domain_robust.py` wraps the entire training step in
  `except Exception`. The block prints `❌ Training error: {e}`
  and a full traceback, so it's a **broad continue-on-error**
  pattern, not silent — but per the project's "re-raise or return
  a typed result" guardrail it's still a deviation. If
  resurrected, the broad catch should be narrowed (or replaced
  with logging + re-raise).

These notes are for the user's awareness; the audit takes no
action on either issue.

Post-VQ1, if the user wants to flatten `drone-race-sim/` into
the outer repo per [[fragilities#Nested repo shares the outer
repos GitHub remote]] mitigation options, the keep/archive split
documented here doesn't change — everything would just move up
a level. No rearrangement of the training scripts is needed
before that decision.

## Ambiguous or surprising

Nothing rises to the level of a `PROMINENT_ISSUE_FOUND.md` halt.
A few items worth flagging without alarm:

1. **`train_fast_safe.py` curriculum callback is wired wrong.**
   `curriculum = CurriculumCallback(vec_env)` is instantiated
   on line 159 but the call to `model.learn(callback=[checkpoint_cb,
   eval_cb], …)` on lines 164–169 omits it. Any run of this
   script trains at `difficulty=0.2` permanently. The class also
   doesn't subclass `BaseCallback` (unlike #2/#3/#4/#7), it's a
   plain class with `__call__`, so SB3 wouldn't have accepted it
   even if it were passed. This is a real bug, but the script
   was apparently never run to completion (no final `.zip`, no
   checkpoints), so the bug never bit anyone. Noting for the
   record only — *no fix is in scope for this audit.*

2. **`train_fast_safe.py` docstring describes reward shaping
   that isn't in the code.** "Bonus for completing gates quickly,
   time penalty encourages aggressive flying", "high crash
   penalties, smooth control, body rate limits", "extra reward
   for finishing all 8 gates" — none of these are implemented in
   the script. They might be in `DroneRaceEnv` (the script's
   environment), which would be fine, but they might just be
   aspirational docstring text. Resolving this would require
   reading `drone_race_env.py` to confirm; that's out of scope
   for a `train_*.py` audit.

3. **`train_curriculum.py` uses a `[128, 128]` network whereas
   `train_state.py` uses `[256, 256, 128]`.** The two scripts
   share a curriculum-style approach but produce architecturally
   different policies. The network-size choice in
   `train_curriculum.py` is smaller, suggesting it predates
   `train_state.py` and was iterated upon. Consistent with
   "experimental dead end" classification.

4. **The chain `train_finetune.py → train_allgates.py →
   train_8gates.py` is documented by source-model file paths**
   (each script's `SOURCE_MODEL` constant points at the previous
   step's `best_model/best_model.zip`). The chain is real, but
   the predecessor of `train_finetune.py` —
   `./trained_50m/best_model.zip` — is referenced and exists
   locally as `drone-race-sim/trained_50m/`, but the script that
   *produced* `trained_50m/` is not in this checkout. The
   training that produced `trained_50m/` happened before `git
   log` of the nested repo was kept clean. Not a bug, just a
   provenance gap.

5. **`train_domain_robust.py` references `aigp_distill_final.zip`
   at `models_release/{name}` (relative).** After dedupe, the
   models actually live at the root `models_release/` and at
   `drone-race-sim/models_release/`-the-empty-shell-after-dedupe
   (which has no zips). If run from `drone-race-sim/`, the
   relative path lookup would search:
   `models_release/aigp_distill_final.zip` (empty subdir post-
   dedupe — would fail) → `../aigp_distill_final.zip` (root, but
   the file actually lives in `models_release/`, not at the root
   itself — would fail) → `aigp_distill_final.zip` (cwd — would
   fail). The script's multi-path lookup does not actually find
   the post-dedupe location of the model. If anyone tries to run
   this script as-is, it will print
   `❌ Model not found: aigp_distill_final.zip` and skip. That's
   load-failure-with-loud-error rather than silent-failure, so
   it's not a fragility entry, but it does mean the script needs
   a path update before it would work in the post-dedupe layout.

6. **`train_state.py`'s output is referenced but absent locally.**
   `train_distill.py:43-46` will throw `FileNotFoundError` if run
   on this checkout. This is documented in
   [[models#aigp_state_final.zip — referenced but not on local
   disk]] and is by design (the teacher lives on HPC). Calling
   it out here only because someone reading just this document
   without the rest of the wiki might be surprised.

None of the above is the "load-bearing undocumented script" or
"embedded credentials" or "missing referenced files in deployed
code" that the task framing flagged as a stop-condition. The
prior audit had already characterized this directory as "archive"
and this per-script pass confirms that characterization holds.

## See also

- [[training]] — the lineage-level view this document complements
- [[experiments-log]] — what `trained_*/` directories survived locally
- [[models]] — the three deployable artifacts in `models_release/`
- [[architecture#Training-side files]] — confirms `drone-race-sim/`
  is treated as the training archive
