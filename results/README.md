# results/

- `overnight-2026-08-03/` — the final qualification campaign (546 attempts).
  `batch_summary.csv` (one row per attempt; 633 rows total — the 546-attempt
  campaign slice is `03:30:43 … 10:28:03`, the rest being pre-campaign setup
  runs and post-campaign empty cycles), `json/` (per-run records),
  `top10_traces/` (full tick CSV+log for the 10 closest approaches),
  `keepers/` (clean-verdict deep approaches), `REPRODUCE.md` (exact commands).
  The complete raw dataset (302 Gate-3-leg traces, 187k ticks) is the GitHub
  Release asset `vq1-overnight-raw-dataset-2026-08-03.tar.gz`
  (SHA-256 in `*.sha256`).
- `analysis-intermediates/` — per-run feature tables derived from the raw
  traces (`legfeat.csv`, `latch.csv`); inputs to the figures in `docs/img/`.

File naming in the raw set: `filt_auto_NNN.*` = attempt NNN of the automated
overnight batch. (Historical manual-era names like `filt83`, `filt94B` exist
only in `archive/tuning-runs/`.)
