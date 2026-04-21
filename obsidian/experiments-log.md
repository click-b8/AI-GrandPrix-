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
  training trace in the repo.** Loading any two of these and diffing
  policy weights, or running both in eval, could reconstruct a reward
  curve retrospectively.

- Other `trained_*/` directories exist (`trained_10m/`, `trained_50m/`,
  `trained_allgates/`, `trained_fast_safe/`, `trained_finetune/`,
  `trained_old_5gate/`) but I haven't opened them during the audit;
  they likely contain checkpoint subsets from earlier experiments.

- **`drone-race-sim/trained_distilled/`** — final weights only
  (`policy.pth`, `policy.optimizer.pth`, `pytorch_variables.pth`),
  `_stable_baselines3_version` file, `system_info.txt`. **No checkpoint
  ladder.** So the reward curve for the deployed distill model is
  not locally reconstructible.

### Recorded measurements

- **`drone-race-sim/train_events.log`** — exists, gitignored, has not
  been inspected for this audit. Worth grepping for
  "ep_rew_mean" to pull out the distill reward curve if SB3's
  default logging was enabled.

- **MODELS.json** — records training_steps and training dates for the
  three final artifacts. Canonical registry.

- **`COMMIT_EDITMSG`** in both git repos — one recent commit message
  claims the 8gates model finishes "8/8 gates in 4.27 s". This is the
  most concrete performance number we have, but without the raw
  measurement script or logs we can't reproduce.

### Measurements taken during the audit (2026-04-21)

- **Inference time on Apple M4 CPU, end-to-end adapter.step():** mean
  0.26 ms, p95 0.30 ms, p99 0.35 ms, min-max 0.23–0.40 ms over 500
  samples with 20-sample warmup. See [[fragilities#Inference time]] for
  hardware context and caveats.

- **Policy-head weight check:** `PolicyNet.pi_net[0].weight` after
  `SCUBALabAdapter.__init__` returns: std 0.0347, max 0.060, mean
  0.00012. Consistent with Kaiming-uniform at fan_in 275. **The policy
  head is running random init**; no weights loaded. See
  [[fragilities#The policy head is untrained at deployment]].

- **`drone-race-sim/config.py` git state:** `FPV_TILT_DEG` was committed
  at `-10` (commit `be4f89d`, 2026-03-22); locally modified to `0`
  (file mtime 2026-03-24 19:07); change never committed; still shows
  as "not staged" in `git status`. HPC training completed 2026-03-25
  09:23 from its own clone, so the trained model saw `-10`. The root
  `config.py` correctly has `-10`.

## What we don't have

### Reward curves for the deployed model

We do not have a reward-vs-step trace for `aigp_distill_final`. The
`MODELS.json` entry claims "Performance: Realistic perception-based
flight" but no numbers. `train_events.log` may contain this if we
inspect it. Otherwise the curve is lost — training was done
interactively on HPC or via a script that was not committed.

### Completion-rate statistics

`COMPETITION_STRATEGY.md:97` specifies "100% gate completion" as a
success criterion and discusses "run 20+ times" and "run 100x trials"
as evaluation protocols. **No record of any such run** exists in the
repo. The "8/8 in 4.27 s" figure from commit `5130652` is a single
observation, not a statistics-backed claim.

### Comparison across the three lineages

We don't have side-by-side evaluation of `aigp_8gates`, `aigp_racer`,
`aigp_distill` against the same evaluation protocol. The adapter load
bug also means any adapter-mediated evaluation of `aigp_distill` to
date was measuring random-init behavior.

### Real-drone data

None. Zero physical validation. `COMPETITION_STRATEGY.md` lists this
as a "CRITICAL" action item and has since April 2026.

### Inference time on target hardware

We measured on Apple M4 CPU. The competition target is ~100 TOPS
embedded. No measurement on comparable hardware exists.

## What would close these gaps

Before VQ1:

1. Fix the adapter (see [[submission-readiness]]).
2. Re-run `measure_inference_time.py` with a **fixed** adapter.
3. Run a batch of 20+ evaluation episodes on the distill model in
   `vision_mode=True` with `domain_rand=False`; record
   ep_rew_mean, gate-completion rate, lap time.
4. Grep `train_events.log` for training-curve data; capture in this
   file.

After VQ1 submission is wired up:

5. Replay a trained state-based expert's rollouts against the DCL sim
   once available; measure sim-to-sim divergence.
6. Measure inference on competition-class hardware.

## Experimental runs we have artifacts from (inferred from directory structure)

Order is approximate, reconstructed from directory names:

| Run | Directory | What it probably was |
|---|---|---|
| early 5-gate baseline | `trained_old_5gate/` | Original 5-gate course |
| 10M-step training | `trained_10m/` | Ablation / short run |
| 50M-step training | `trained_50m/` | Ablation / mid-length run |
| all-gates training | `trained_allgates/` | Transition from 5 to 8 gates |
| safe/slow training | `trained_fast_safe/` | `train_fast_safe.py` lineage; from COMPETITION_STRATEGY.md, this was a safety-focused experiment |
| fine-tune | `trained_finetune/` | `train_finetune.py`; likely an iteration on an earlier expert |
| Swift 100M | `trained_swift_100m/` | Final state-expert lineage; produced `aigp_racer_final` |
| distilled | `trained_distilled/` | Final distill run; produced `aigp_distill_final` |
| vision events | `trained_vision_events/` | Pure vision PPO (abandoned); fed into distill as warmstart |

None of these have been opened during this audit; writing this row is
inference from directory names, which can lie. Before adding any of
these to a wiki claim, inspect the directory.

## See also

- [[models]] — what artifacts we have
- [[training]] — how they were trained in theory
- [[fragilities]] — the measurements that contradicted existing claims
- [[open-questions]] — measurement gaps to close
