#decision

# Decisions log

Decisions that shaped the project, with alternatives considered. Dates
are best-guess from git history and file timestamps; where unknown,
"~" prefix. The decisions below reflect the project's trajectory, not
necessarily still-current reasoning — see [[open-questions]] for
things that probably deserve revisiting.

---

## CTBR over direct motor control

- **Date:** Initial (pre-2026-02, commit `241e904` and earlier).
- **Decision:** use `[thrust, roll_rate, pitch_rate, yaw_rate]` as the
  action space rather than 4-motor thrusts.
- **Alternatives considered:** per-motor thrust (what MuJoCo could
  support directly), direct torques (what `_apply_forces` eventually
  does under the hood).
- **Why:**
  - Matches Swift (Nature 2023) and MonoRace, the two reference
    systems for this class of problem. Papers in this area standardize
    on CTBR.
  - Matches what real Betaflight / Cleanflight flight controllers
    accept. Sim-to-real transfer benefits: the learned policy outputs
    map directly to the inputs a real drone's FC expects.
  - Decouples the policy from motor-level dynamics. The rate
    controller (PD on body rates with inertia-scaled torque) handles
    the fast loop; the policy thinks in a slower, more abstract space.
- **Consequence:** the rate controller runs at physics rate inside
  `_apply_forces`. `RATE_KP = 60, RATE_KD = 2` are hand-tuned for
  stability. Tuning these values changes the effective bandwidth of
  the policy's body-rate commands.

---

## 6D rotation over quaternions in the state branch

- **Date:** vision pipeline commit, 2026-03-22 (`be4f89d`).
- **Decision:** the 19D state vector represents orientation as the
  first two columns of the body-to-world rotation matrix (6 values),
  not as a unit quaternion or Euler angles.
- **Alternatives considered:** unit quaternion (4 values), Euler RPY
  (3 values).
- **Why:**
  - Zhou et al. (CVPR 2019) showed that unit quaternions and Euler
    angles are **topologically discontinuous** as regression targets,
    which hurts gradient flow and learning when the network must
    regress-to-predict pose.
  - A CNN-coupled state branch is doing pose-adjacent regression.
    Quaternion double-cover (`q ≡ -q`) can cause learning instability
    in narrow ways that are hard to debug.
  - 6D representation is continuous, avoids gimbal lock, and recovers
    a full rotation matrix via Gram-Schmidt (cheap, differentiable).
- **Downside:** 6 dims instead of 4 or 3 — trivially more parameters in
  `state_mlp`'s first layer. Not meaningful.
- **Still-current:** yes. Reuse if rebuilding.

---

## Coarse-to-fine CNN over a single convnet

- **Date:** 2026-03-22 (`be4f89d`).
- **Decision:** two CNN stages where the fine stage **reuses the
  coarse stage's feature maps** rather than re-processing the raw
  image.
- **Alternatives considered:** single deeper CNN; a ResNet-lite stem.
- **Why:**
  - Inherited from `event-sharp-nerf-drones` (Zou et al. 2026) and the
    VoxelNeRF decomposition pattern.
  - Cheaper forward pass: the fine stage sees 32-channel 11×11 maps
    instead of 14-channel 48×48 images. Forward-pass cost is
    dominated by the coarse stage.
  - Implicit multi-scale supervision: coarse features must carry
    enough signal to support the fine stage.
- **Not validated:** we haven't ablated this against a single CNN of
  comparable parameter count. Treat as a reasoned design choice, not
  a measured win.
- **See:** [[vision-model]].

---

## Distillation from a state-based expert over end-to-end vision PPO

- **Date:** ~2026-03-09 (commit `434abcb`, "Add Swift-style privileged
  teacher distillation pipeline").
- **Decision:** the vision policy (`aigp_distill_final`) is trained via
  DAgger from a state-based expert teacher, not end-to-end with PPO
  alone.
- **Alternatives considered:** end-to-end vision PPO (`train_vision.py`,
  200M step target). Actually tried and abandoned.
- **Why:**
  - State-based PPO converges in ~10–20M steps; vision PPO needs
    100M+ to reach comparable reward.
  - With a strong teacher, DAgger compresses the learning timeline
    dramatically — the student gets dense gradient signal from every
    rollout even when its own actions are poor.
  - The teacher has ground-truth gate positions the student doesn't;
    the student must learn to hallucinate gates from pixels, but the
    teacher tells it what the right action would be if it could.
- **Current artifact:** `aigp_distill_final.zip`, 52M steps, 2-phase
  imitation-weight decay (1.0 → 0.0 over first ~67% of training).
- **See:** [[training]].

---

## Direct weight loading over `PPO.load()`

- **Date:** ~2026-03-25, around the creation of the `dcl_adapter.py`
  framework.
- **Decision:** `dcl_adapter.py` unzips the `.zip` and loads `policy.pth`
  directly into a handwritten torch module, bypassing
  `stable_baselines3.PPO.load()`.
- **Alternatives considered:** PPO.load() with `custom_objects`; using
  the same SB3 version on every machine; building a wheel with pinned
  deps.
- **Why:**
  - HPC trained with SB3 2.7.1 / PyTorch 2.6 / Python 3.12.4.
  - Local/deployment machines run SB3 2.8.0 / PyTorch 2.11 / Python
    3.12.13 (macOS) or whatever DCL ships with.
  - `PPO.load()` unpickles objects; class structure changes across SB3
    versions caused load failures in practice.
  - Direct `torch.load` of `policy.pth` is version-agnostic for the
    tensor payload and robust to SB3 internals changing.
- **Cost:** the adapter must mirror the trained network topology by
  hand. When the handwritten mirror is wrong — which it currently is
  (see [[fragilities#The policy head is untrained at deployment]]) —
  `load_state_dict(..., strict=False)` silently accepts whatever subset
  of keys match. **This is the source of the single worst bug in the
  project.**
- **Retain the decision, change the implementation:** direct weight
  loading is still correct. Use `strict=True` and write the mirror
  topology against the actual checkpoint shapes. `dcl_adapter_sb3.py`
  (the PPO.load variant, currently listed as deprecated in
  `README_COMPETITION.md`) is a distraction.

---

## One archived predecessor: `dcl_adapter_sb3.py`

- **Date of deprecation:** ~2026-04-07 (`README_COMPETITION.md:94`).
- **What it was:** a version of the adapter that called `PPO.load()`.
- **Why deprecated:** the SB3 version gap caused architecture mismatch
  errors at load.
- **Why noted here:** so that the next person who inherits this project
  knows there were two paths and which lost. The deprecation is not
  enforced (the file still exists), which is the kind of thing that
  trips up a stranger running a grep.

---

## Zero-pad event channels at deployment

- **Date:** ~2026-03-25, with `dcl_adapter.py`.
- **Decision:** at inference, fill the event-histogram channels with
  zeros because DCL provides RGB only.
- **Alternatives considered:** run a live EGM at deployment against the
  DCL RGB stream (would require tracking a reference log-luma
  per-pixel, manageable); drop the event channels and retrain without
  them.
- **Why:**
  - Retraining is expensive.
  - Running a live EGM requires us to match the DCL stream's frame
    rate and avoid rollout desync.
  - The bet is that the policy's learned features still function on
    RGB alone — event channels provided regularization during training
    but aren't strictly necessary at inference.
- **Not validated.** See [[perception#The deployment gap — event channels are zero-padded at inference]]
  and [[open-questions]]. Once the deployment adapter loads real
  weights, this is one of the first ablations to run.

---

## Duplicate the curated subset of files to the repo root

- **Date:** ~2026-04-16 (recent commits: "Clean main branch: remove
  training scripts, benchmarks, and non-essential files"; "Final
  cleanup: remove _archive folder, duplicates, and unnecessary files").
- **Decision:** the top-level of the repo contains a curated subset of
  files intended for deployment; the training scaffolding and
  experimental scripts remain in the `drone-race-sim/` subdirectory
  (which is its own git repo).
- **Alternatives considered:** move everything to root; treat the
  subdirectory as submodule.
- **Why:** present a clean, small, public-facing project at the root,
  while preserving all training history and experimental files.
- **Incomplete:** `dcl_mavlink_client.py`, `test_dcl_adapter.py`,
  `models_release/aigp_distill_final.zip`, `measure_inference_time.py`,
  and the SLURM scripts all stayed in the subdirectory. The root is
  not actually self-sufficient for deployment. See
  [[fragilities#Canonical copy of each file]].
- **To resolve:** either complete the migration or state in every
  README that deployment requires files from the subdirectory.

---

## Motion blur warmup at 50K steps

- **Date:** 2026-03-22 (`be4f89d`).
- **Decision:** motion blur is disabled for the first 50,000 training
  steps, then enabled by a callback.
- **Why:** the CNN learns clean gate geometry first, then is asked to
  handle blur as a perturbation. Staged training — the curriculum
  equivalent of "learn to walk before you run".
- **See:** [[perception#Motion blur simulation]].

---

## DummyVecEnv over SubprocVecEnv on Mac

- **Date:** implicit in `train_vision.py:44` (`N_ENVS = 8  # DummyVecEnv
  (SubprocVecEnv unreliable with MuJoCo GL on macOS)`).
- **Decision:** use in-process environments for parallelism during
  training.
- **Why:** MuJoCo's OpenGL rendering is unreliable across forked
  processes on macOS; rendering failures were observed. DummyVecEnv
  is single-threaded but doesn't fork.
- **Cost:** slower rollouts locally. On HPC (Linux) with
  `MUJOCO_GL=egl`, SubprocVecEnv works fine but we never switched
  because the code was shared.
- **Still-current:** yes for Mac. For HPC retraining, switching to
  SubprocVecEnv would meaningfully speed rollouts.

---

## 2026-04-23 — Accepted one-time plan-structure deviation

- **Date:** 2026-04-23.
- **Decision:** accept the process deviation on `fix/mavlink-complete`
  (commit `3f34b57` + merge `7975ccf` + patch `f433ddf`) as a one-time
  exception; do not require re-structuring into the originally-planned
  three-branch sequence (F: protocol compliance, G: body-rate
  semantics, H: client entry point).
- **What was deviated from:** the VQ1 execution plan explicitly
  required MAVLink fixes to be broken into at least two branches
  ("one for the semantic mismatch, one for the transport/protocol
  compliance — do not bundle semantics with protocol, they fail
  differently and need different tests"). A single commit on
  `fix/mavlink-complete` bundled protocol compliance, CTBR
  semantics, client rewrite, and tests. A subsequent merge pulled
  `refactor/deduplicate-root-subdir` into `fix/mavlink-complete`
  retroactively, rather than sequencing the branches per plan.
- **Why accept:** the resulting code is correct and well-tested (25
  passing tests covering CRC, sequence, semantics, byte-exact
  pymavlink roundtrip, end-to-end mock receiver). The test split is
  in fact cleaner than what the plan required: `test_mavlink_compliance.py`
  isolates protocol concerns and `test_vq1_readiness.py` covers
  semantics + integration. The outcome met the plan's technical
  intent; the deviation was in process, not in artifact quality.
- **Root cause:** the prompt the overnight session received was
  under-specified ("follow the plan"), not the session's execution
  being sloppy. Vague prompts are themselves a silent-failure
  category — they produce outputs that look correct, pass tests,
  match the plan's intent, and whose process deviations only surface
  under audit. See [[fragilities#Silent failure chain]] for the
  general category.
- **Mitigation:** [[session-prompt-template]] was created the same
  day to force future prompts to specify a single named branch,
  forbidden vague phrasings, an explicit Evidence-step STOP that
  auto-mode doesn't override, and a Do-NOT list that names
  "collapsing multiple plan-branches into one" and "merging
  another plan-branch into this one retroactively" as scope
  violations verbatim. Subsequent branches must use the template.
- **Still-current:** yes. Future branches get specific, single-scope
  prompts.

---

## 2026-04-23 — Author identity uniformity

- **Date:** 2026-04-23.
- **Decision:** all commits on this project use the git identity that
  `git config user.name` and `git config user.email` return from the
  committer's machine. Non-person identities (team-bot-style names
  like `SCUBA Lab <scubalab@vq1.local>`, `Bot <bot@...>`, CI
  service accounts, etc.) are not permitted; Co-Authored-By
  trailers to bot identities are not permitted.
- **Why:** the project's established convention across all prior
  commits was a single-person hostname-derived identity. The three
  commits on `fix/mavlink-complete` (`3f34b57`, `7975ccf`, `f433ddf`)
  authored as `SCUBA Lab <scubalab@vq1.local>` broke that
  convention. Non-person identities in the permanent history:
  (a) obscure who actually did the work — is this a human
  collaborator, an AI agent, a shared machine? (b) break
  `git shortlog -sn` and `git log --author=<person>` searches,
  (c) set a precedent that erodes the project's one-identity
  invariant over time.
- **Mitigation applied same day:** rebased the three SCUBA Lab
  commits on `fix/mavlink-complete` to the project's standard
  identity using `git rebase` with an env-filter that rewrites
  author/committer name+email while preserving timestamps. The
  `fix/vision-stub-guard` branch was authored with the correct
  identity from the start; no rewrite needed there.
- **Codified in:** [[session-prompt-template#Commit identity]],
  which requires a runnable `git config user.name/user.email`
  check before the first commit of any branch, and explicitly
  forbids non-person identities.
- **Still-current:** yes. Applies to all future branches.

---

## See also

- [[training]] — the training pipelines these decisions shaped
- [[deployment]] — the deployment pipeline
- [[fragilities]] — where these decisions left silent footguns
- [[open-questions]] — decisions that probably deserve revisiting
