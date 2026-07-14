# DGX Training Guide — Anduril AI Grand Prix VQ1

**For whoever is at the DGX** (Linux, multi-GPU). We share this one box: Noah is
driving it solo for the first few days of fast iteration; Pratik joins later.
**This document is fully self-contained — it assumes zero GitHub / project
context**, so you do not need to read any other file to get running.

**How to read it:** **Part 1** (environment prep) is identical for everyone — do
it once per machine/account. **Part 2** is the training workflow; it is written
for **fast, monitored solo iteration** (short runs you watch and course-correct),
not a hands-off monolithic job — see §2.4.

---

## Context (read once)

This is the **Anduril AI Grand Prix, VQ1** — an autonomous drone-racing
qualifier. The updated competition sim **blocks** the "cheat" telemetry streams
(`ATTITUDE`, `LOCAL_POSITION_NED`, `ODOMETRY`), so the policy must fly from
**vision + IMU only**:

- **State (target 10-D):** gravity vector (from IMU, 3) + body angular rates (3)
  + previous action (4). No global position, no velocity, no attitude quaternion.
- **Image:** a stacked **48×48 RGB** frame (target **9×48×48** = 3 frames × RGB).

We are training a policy **from scratch in MuJoCo** with the goal of **zero-shot
transfer** to DCL's competition sim. The observation the model trains on must be
byte-for-byte the same contract it sees at deployment — hence the hard gate in
Part 2.

### What "success" means for VQ1 — read this before you tune anything

**VQ1 is the completion qualifier, it is the DELIBERATELY-EASY round, and it is
far more forgiving than it sounds:**

- **Single drone.** No opponent on the course.
- **Intentionally simple by design.** Per the official site: a **small number of
  gates, a desaturated environment, gates visually HIGHLIGHTED**, **no wind, no
  disturbances.** The hard visual-transfer problem is **VQ2, not VQ1.** Don't
  over-engineer for a round that was built to be passable (see §2.2).
- **Completion, not speed.** Fly all the gates in order; you are **not** ranked on
  lap time. (Our course currently has **8 gates** — confirm the real count from
  the downloaded sim; see §2.2.)
- **UNLIMITED attempts.** Per Anduril's **6/30 email**, VQ1 remains **open**, and
  successful teams passed only after **"dozens to hundreds of attempts"** and
  **thousands of runs** against the sim.

The strategic consequence — **internalize this, it shapes every decision below:**

> We do **NOT** need to perfect the vision-only transfer model before flying. We
> need the **first competent vision+IMU policy that completes the course**, then
> we iterate attempts against the deterministic sim. Attempts are free. A policy
> that completes **1 in 20** attempts still qualifies given unlimited tries.

So: train to **first competent completion**, deploy, and grind attempts — do
**not** train to convergence before we try to qualify.

### Out of scope for VQ1 (don't build these yet)

The following are **VQ2 / competitive research references only — NOT VQ1
inputs.** VQ1 has **no opponent**, so anything about interaction, blocking, or
multi-agent game theory does not apply to this qualifier:

- **TU Delft interaction-aware thesis (Papuc, 2025)** — multi-agent / opponent-aware.
- **MonoRace** — competitive-racing reference.

Multi-agent / blocking / Nash-equilibrium methods are **out of scope for VQ1.**
Park them for VQ2.

---

# PART 1 — DO NOW  ✅

Everything here is safe to do immediately, in parallel with the local env work. It
gets the DGX fully ready so that the moment the launch gates in §2.1 are green you
can start training within minutes.

## 1.1 Create the conda environment

Use **Python 3.11 or 3.12**. Do **not** use 3.14 — SB3 / torch wheels are not
reliably available/stable there yet.

```bash
conda create -n aigp python=3.11 -y
conda activate aigp
```

## 1.2 Clone the repo

```bash
git clone https://github.com/click-b8/AI-GrandPrix-.git
cd AI-GrandPrix-
```

## 1.3 Install dependencies

Install torch **with the CUDA build that matches your DGX driver** first, then
the rest. Pick the CUDA tag from https://pytorch.org that matches `nvidia-smi`.
Example for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then the core stack:

```bash
pip install "mujoco>=3.0" gymnasium "stable-baselines3>=2.1" Pillow tensorboard
```

**Drop `pymavlink`** — it's deploy-only (talks to the DCL sim over MAVLink) and
is not needed for training. If you `pip install -r requirements.txt`, that's fine
too, but pymavlink there is irrelevant on the training box.

Sanity check versions:

```bash
python -c "import mujoco, gymnasium, stable_baselines3, torch; \
print('mujoco', mujoco.__version__); \
print('sb3', stable_baselines3.__version__); \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

`torch.cuda.is_available()` must print `True`.

## 1.4 MANDATORY headless gotcha — set `MUJOCO_GL`

The DGX has **no display**. Off-screen 48×48 rendering will fail to create a GL
context unless you tell MuJoCo to use EGL:

```bash
export MUJOCO_GL=egl
```

Put it in your shell profile or prepend it to every command (shown throughout
this doc). If you ever see errors like *"Failed to create GL context"* or
*"cannot open display"*, this is the cause. `osmesa` is a slower fallback if EGL
is unavailable, but **EGL is what you want on the DGX.**

## 1.5 Smoke test — prove the environment works NOW

This uses the **real** training-env class, constructed exactly the way
`train_vision.py` builds it. Copy-paste — it should work first try:

```bash
MUJOCO_GL=egl python - <<'PY'
from drone_race_env import DroneRaceEnv

# Identical construction to train_vision.py's make_train_env():
env = DroneRaceEnv(render_mode=None, domain_rand=True, difficulty=0.5, vision_mode=True)
obs, info = env.reset()
for k, v in obs.items():
    print(f"{k:6s} shape={v.shape} dtype={v.dtype}")
env.close()
print("OK — env constructed, reset, rendered offscreen.")
PY
```

### What shapes to expect

- **RIGHT NOW (before the vision+IMU update, "B5"), you will see:**
  - `image` → **(48, 48, 6)**  (channels-last: 48×48, 2 stacked RGB frames)
  - `state` → **(19,)**
  **This is expected — it is not a bug.** The current state is the 19-D VIO
  contract; the current image is 2-frame channels-last.
- **After B5 lands, the target contract is:**
  - `image` → **(9, 48, 48)**  (channels-first: 3 stacked RGB frames)
  - `state` → **(10,)**  (gravity vector + body rates + prev action)

  So the state dim collapses **19 → 10** and the image moves to a 3-frame
  channels-first layout. If the smoke test prints 19 / (48,48,6) today, you are
  correctly set up; it will update to 10 / (9,48,48) when B5 lands.

Once the smoke test prints shapes without error, **Part 1 is complete.** Don't
start training until the launch gates in §2.1 are green.

---

# PART 2 — LAUNCH (only once the gates in §2.1 are green)  ⛔

**Do not start training until both launch gates below pass.** They go green when
B5 has landed: the 10-D vision+IMU obs, the SubprocVecEnv default, per-seed save
dirs, and the gates-passed eval logging (see §2.1, Gate 2) are all in. If you're
driving both the code and the DGX, *you* confirm these yourself before launching —
they are not optional.

## 2.1 THE LAUNCH GATES — both must be satisfied before you start

There are **two** hard, required-before-launch gates. Do not start training until
**both** are confirmed green.

### Gate 1 — the observation-contract test is GREEN

**DO NOT launch training until the observation-contract test passes against the
real env builder:**

```bash
MUJOCO_GL=egl python -m pytest tests/test_observation_contract.py -v
```

This test guarantees **train == deploy**: that the 10-D vision+IMU observation
the model trains on is exactly the observation it will receive in DCL's sim. If
you launch before it passes, you train a model whose inputs won't match
deployment — the weights are silently useless and **you will have burned
GPU-weeks for nothing.**

### Gate 2 — gates-passed eval logging is in (BLOCKING B5 item)

**DO NOT launch until the run emits `eval/gates_passed` to TensorBoard.** As the
code stands today, `train_vision.py` uses stock SB3 `EvalCallback`, which logs
**only** `eval/mean_reward` and `eval/mean_ep_length` — it does **not** record
gates-passed, even though the environment reports `gates_passed` in its step
`info`. **This makes the entire VQ1 gate-milestone plan unmeasurable**, and a run
you cannot measure is a dead run that burns days silently before anyone notices.

We are adding a **custom eval callback (B5, required before launch)** that logs
`eval/gates_passed` (mean gates cleared in eval) and sets an `is_success` flag =
**all-8 completion**, which SB3 then surfaces as `eval/success_rate`. The GO
signal confirms this is live. **When training starts, verify `eval/gates_passed`
appears in TensorBoard. If it does NOT, stop and tell us — do not run blind.**

## 2.2 Domain randomization — MINIMAL for VQ1 (this is the deliberately-easy round)

Per the official site, **VQ1 is intentionally simple**: a **small number of
gates, a desaturated environment, gates visually HIGHLIGHTED by design, no wind,
no disturbances.** The hard realistic-rendering transfer problem is **VQ2, not
VQ1.** So we are **not** fighting a realistic-rendering gap here.

**Scope visual DR DOWN drastically for VQ1:** minimal lighting/color jitter is
enough. Concretely —

- **Keep:** small brightness/contrast jitter, mild camera noise, minor gate-color
  jitter. That's it.
- **Cut for VQ1:** heavy texture randomization, aggressive lighting sweeps,
  photorealistic clutter, worst-case transfer robustness. **Over-randomizing a
  desaturated, highlighted-gate scene just slows convergence and delays our first
  completion.**
- **Defer to VQ2:** the exhaustive visual DR. Save it for the round that actually
  needs it.

### Course geometry & camera — lock these to the real sim before/with the contract test

- **Gate count:** our course (`track.py:RACE_TRACK`) currently has **8 gates**,
  but the site says VQ1 has a "small number" — possibly fewer. **Confirm the
  actual count and gate poses from the downloaded sim and update `RACE_TRACK`**;
  the env reads `_num_gates = len(get_gate_positions())`, so everything (episode
  termination, the milestones in §2.5) follows automatically once the real
  geometry is in.
- **Camera FoV (train==deploy-critical):** we currently render `fovy=90°` on a
  square 48×48 — that follows the *stated* FoV. The Elodin practice-rig
  intrinsics imply **HFoV=90°, VFoV≈58.72°, 16:9** (deploy squashes 360×640 →
  48×48). **Our camera must follow the INTRINSICS, not the stated 90° VFoV:** set
  vertical FoV ≈ **58.72°** and render 16:9 before the 48×48 resize. A mismatch
  here distorts what the policy sees vs. deployment — fold this into the contract
  test (§2.1, Gate 1) so it can't silently drift.
- **Motor RPM (optional, don't block):** the site lists motor-RPM readouts as
  *likely* available. If the downloaded sim exposes them, they're an extra
  proprioceptive signal we could add to the state vector — but that grows the
  observation, so it must pass the contract test. **Do not block the first run on
  this.**

## 2.3 Vectorized envs — use `SubprocVecEnv` (this is decided, not an A/B)

The code carries a comment that `SubprocVecEnv` is "unreliable with MuJoCo GL."
**That caveat is macOS-specific** — it comes from macOS requiring the GL context
on the main thread. **On the Linux DGX with `MUJOCO_GL=egl` it does not apply:**
each worker process creates its own independent headless EGL context, which is
exactly what EGL is for. Vision-from-scratch is **sample-hungry**, so more
parallel envs = more throughput = first completion sooner.

**Fixed recommendation: `SubprocVecEnv` with ~16–32 envs on the DGX.** We will
make this the default in `train_vision.py` before GO, so you do not have to edit
code. The only requirement on your side: **`export MUJOCO_GL=egl` must be set
before launch** (each worker inherits it and builds its own context). If you ever
hit a rare fork/GL interaction, set the start method to `spawn` — but with EGL on
Linux this is not expected.

## 2.4 The workflow — FAST SOLO ITERATION (not a monolithic hands-off job)

VQ1 is the easy round and we have unlimited attempts, so the winning strategy for
the next few days is **many short, monitored runs you course-correct**, *not* one
200M-step job you kick off and walk away from. The mental model:

> Launch several short seeds in parallel → watch `eval/gates_passed` → **kill the
> stalled ones fast** → pour GPUs into whichever seed shows gate progress → the
> instant any checkpoint completes the course in eval, deploy it to the grinder
> (§2.7). Meanwhile the grinder is **always running** against the current best
> model, so we're accumulating qualifying attempts the whole time.

`train_vision.py`'s `TOTAL_TIMESTEPS = 200_000_000` is just a ceiling — with
checkpoints every 50k and eval every 25k, **you are meant to watch the early evals
and stop early**, not run to 200M. Treat each seed as a short probe (first few M
steps) that has to *earn* more compute by showing gate-1 progress.

### Launch a seed (in `tmux` so it survives disconnects)

```bash
tmux new -s aigp
export MUJOCO_GL=egl
python train_vision.py
```

Detach with `Ctrl-b d`; reattach with `tmux attach -t aigp`. Kill a stalled seed
with `Ctrl-c` (auto-resume means you can always continue a promising one later).

### Parallel short seeds — one per GPU, each with its OWN save directory

PPO is **seed-sensitive**; a single seed can fail by luck, so run several at once
and let the early evals decide which survives. **Each seed must write to its own
directory or they overwrite each other's checkpoints.** `train_vision.py` writes
to `./trained_vision_events` relative to the working directory, so the simplest
guaranteed-isolated method is **one checkout per seed**:

```bash
# Seed 0 on GPU 0
git clone https://github.com/click-b8/AI-GrandPrix-.git seed0 && cd seed0
tmux new -s seed0
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python train_vision.py    # Ctrl-b d to detach

# Seed 1 on GPU 1 (repeat per GPU: seed2, seed3, …)
git clone https://github.com/click-b8/AI-GrandPrix-.git seed1 && cd seed1
tmux new -s seed1
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl python train_vision.py
```

Each directory keeps its own `trained_vision_events/` (checkpoints, TensorBoard,
eval logs), so seeds never collide. (B5 also adds a `--save-dir`/seed flag to make
this a one-liner without separate clones; the clone method above always works.)
Run TensorBoard once over the parent dir to watch all seeds together:
`tensorboard --logdir . --bind_all`.

## 2.5 Monitoring & the REAL success signal

The script is already wired for:

- **Checkpoints:** every **50k** steps → `trained_vision_events/checkpoints/`
- **Eval:** every **25k** steps (5 episodes)
- **Auto-resume:** on restart it finds the latest checkpoint and continues
  automatically — if a run dies, just re-run the same launch command.
- **TensorBoard:** `trained_vision_events/tb_logs/`:

  ```bash
  tensorboard --logdir trained_vision_events/tb_logs --bind_all
  ```

### Watch gates-passed, NOT reward

**Do not judge the run by reward curve.** Reward can climb steadily while the
drone still never clears **gate 1**. The only signal that means the policy is
actually flying the course is **gates passed in eval.**

> ⚠️ This metric only exists because of **Launch Gate 2 (§2.1)** — the custom
> eval callback we add in B5. Watch **`eval/gates_passed`** and
> **`eval/success_rate`** (= all-8 completion rate) in TensorBoard. **Confirm
> `eval/gates_passed` is present the moment training starts** — if it is missing,
> Gate 2 didn't land; stop and tell us rather than running blind.

### Aggressive early-eval milestones (this is the VQ1 plan)

Track these three, in order, from eval checkpoints:

1. **Gate 1 pass** — the drone can fly and clear the first gate.
2. **Gate 3 pass** — it's stringing gates together, not fluking gate 1.
3. **All-8 completion** — a full course completion in eval.

**The moment an intermediate checkpoint completes the course in eval — even
inconsistently — deploy it.** Do **not** wait for convergence or a clean,
high-completion-rate policy. Given unlimited attempts, a checkpoint that completes
**1 in 20** eval runs is a qualifying candidate. Ship the first competent
completer to the grinder (§2.7), then keep iterating better seeds behind it.

## 2.6 Aggressive early-kill — free the GPU for the next seed

Because you're **watching** these runs (not running hands-off), kill hard and
early. Two tiers:

- **Solo-iteration kill (your day-to-day):** a seed showing **no gate-1 progress
  in `eval/gates_passed` by ~1–2M steps** is almost certainly dead — **kill it and
  relaunch a fresh seed on that GPU.** Don't nurse a flat line; the GPU is better
  spent on another draw. A seed that *is* progressing (gates ticking up) earns
  more compute.
- **Hard backstop:** if **no seed clears gate 1 by ~5M steps**, stop and
  re-examine the setup — that's a systemic problem (obs contract, camera FoV,
  reward), not bad luck. For calibration, our prior non-vision runs didn't diverge
  until ~14M, but a healthy vision-from-scratch seed on this easy course should
  show gate-1 movement **well before 5M**, typically in the first 1–2M.

Rule of thumb: **keep ~1 promising seed growing, and keep the other GPUs cycling
fresh short seeds.** Every stalled seed you kill early is a free extra draw.

## 2.7 Keep the grinder running in parallel — the whole time

Independently of training, run the automated attempt-grinder against the **current
best deployed model** continuously (it's a separate deploy-side process — see
`ATTEMPT_RUNNER_DESIGN.md`):

```bash
python attempt_runner.py --max-attempts 200
python summarize_attempts.py --in attempts.jsonl   # gate histogram + completion rate
```

Why it runs the entire time: (1) VQ1 is won by **volume of attempts**, so every
hour the grinder isn't running is qualifying attempts left on the table; (2) it
gives us a **gate-histogram baseline** of how far the current model actually gets;
(3) when a new training checkpoint beats the deployed one, swap it in
(`models_release/aigp_distill_final.zip`) and the grind continues against the
better policy. Training and grinding are **concurrent workstreams**, not
sequential phases.

---

## Quick reference

| Thing | Value |
|---|---|
| Repo | https://github.com/click-b8/AI-GrandPrix-.git |
| Python | 3.11 or 3.12 (NOT 3.14) |
| Headless render | `export MUJOCO_GL=egl` (mandatory) |
| Smoke-test env | `DroneRaceEnv(render_mode=None, domain_rand=True, difficulty=0.5, vision_mode=True)` |
| Obs NOW (pre-B5) | `image (48,48,6)`, `state (19,)` |
| Obs target (post-B5) | `image (9,48,48)`, `state (10,)` |
| Launch gate 1 (must be GREEN) | `pytest tests/test_observation_contract.py` |
| Launch gate 2 (must be present) | `eval/gates_passed` in TensorBoard (B5 custom eval callback) |
| Vec envs | `SubprocVecEnv`, ~16–32 envs (decided — macOS caveat doesn't apply on Linux+EGL) |
| Workflow | **fast solo iteration** — short monitored seeds, kill stalled ones fast; NOT a 200M hands-off job |
| Launch | `MUJOCO_GL=egl python train_vision.py` (in tmux) |
| Parallel seeds | one checkout per seed → isolated `trained_vision_events/`; pin with `CUDA_VISIBLE_DEVICES` |
| Checkpoints / Eval | every 50k / every 25k |
| TensorBoard | `tensorboard --logdir . --bind_all` (parent dir shows all seeds) |
| Success signal | eval **gates passed** (`eval/gates_passed`, added in B5) — NOT reward |
| VQ1 milestones | gate-1 → gate-3 → all-8; **deploy first completer** (1-in-20 qualifies) |
| Early-kill | no gate-1 by ~1–2M → kill seed, relaunch; no seed by ~5M → systemic problem, re-examine |
| Visual DR | **minimal** — VQ1 is desaturated + highlighted gates, no wind (save exhaustive DR for VQ2) |
| Gate count | confirm from downloaded sim (site: "small number"); currently **8** in `track.py:RACE_TRACK` |
| Camera FoV | follow **intrinsics**: vertical ≈**58.72°**, 16:9 → 48×48 — NOT the stated 90° VFoV |
| Motor RPM | optional extra obs if sim exposes it; don't block (changes contract) |
| Grinder (parallel) | `python attempt_runner.py --max-attempts 200` running the whole time vs. current best model |
| Out of scope (VQ2 research only) | Papuc 2025 (TU Delft, interaction-aware), MonoRace — no opponent in VQ1 |
