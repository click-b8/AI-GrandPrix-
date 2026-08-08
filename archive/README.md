# Archive

Superseded eras and bulk experiment data, preserved rather than deleted. Nothing here is needed to run or understand the current system — each area exists because it explains where the project has been. Bulk data directories are git-ignored (local/Release-asset only); code and documents are tracked.

| area | what it is | why it ended |
|---|---|---|
| `rl-training/` | The first era: PPO training + policy distillation against a MuJoCo course model (`drone_race_env.py`, `train_*.py`), DGX/SLURM guides, model checkpoints and seed sweeps (git-ignored binaries). | Superseded by the direct vision-servo controller once the qualifier's observation contract (vision-only, no pose) made end-to-end RL the riskier path. |
| `dcl-hardware/` | DCL AI-vector-module integration: MAVLink adapters, vision receiver, hardware docs — the physical-qualifier interface. | Sim qualifier ended before the physical round; kept intact for any future hardware phase. |
| `tuning-runs/` | ~560 CSV/log pairs from the manual tuning campaign (the `filt*`/`flt*` series) plus `tune_logs/`. Every claim in `docs/experiments.md` traces to files here. Git-ignored (bulk). | Campaign concluded; the overnight batch superseded manual iteration. |
| `experiments/` | Abandoned or superseded tooling: `attempt_runner.py` (+ design doc), the open-loop tuning era artifacts, `run_batch.ps1` (the manual-restart batch), scratch scripts, automation-script backups. | Each superseded by the current automation/analysis stack. |
| `frames/` | Vision-debug frame dumps (`leg1_frames/`, probe frames, rail overlays, detector snapshots). Git-ignored (bulk); `leg1_frames.zip` is the compact form. | Detector debugging concluded. |
| `notes/` | Raw working notes: `obsidian/` (decisions log, experiments log, fragilities), `design-specs/` (control redesign, altitude ladder, protocol notes), the original competition strategy. `docs/` is the curated form of this material. | Distilled into `docs/`. |

Git history note: stale branches were preserved as tags before deletion — `archive/in-process-branch`, `archive/fix-policy-weight-load`, `archive/gate3-pnp-probe-frozen` (the PnP feasibility branch, frozen at audit stage by decision).
