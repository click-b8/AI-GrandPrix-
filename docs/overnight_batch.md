# The overnight batch — 2026-08-03

The final campaign: an unattended best-of-N qualification attempt, run in the last hours of simulator access. Design goal: *if* qualification was reachable by variance, harvest it; if not, produce the dataset that says why not. It produced the second.

## Configuration

- Frozen controller (see [REPRODUCE.md](../results/overnight-2026-08-03/REPRODUCE.md) for the exact 40-flag command).
- `automation/run_qualifier_batch.ps1 -KeepSimAlive -Mode holdblind -HoldSeconds 0.30` — sim logged in once and kept alive; keystroke reset per attempt; stop only on the sim's race-finish signal or `MaxAttempts` 1000.
- Hardware note: ARM Surface translating the x86 sim — loop rate 51–112 Hz across runs, which became a first-class variable in the analysis.

## Execution

| metric | value |
|---|---|
| Attempts | **546** over 7.0 h (03:30–10:24) |
| Cadence | median 44 s/attempt, min 43 s |
| Operator interventions | 0 |
| Automation failures | 0 (no runaway cycles, no stray processes, no SIM LOST) |
| Data captured | 546 summary rows + JSONs; 302 full Gate-3-leg tick traces; discards self-purged after recording |

## Results

![funnel](img/funnel.png)

| stage | runs | % |
|---|---|---|
| Spawn | 546 | 100% |
| Passed Gate 1 | 504 | 92.3% |
| Passed Gate 2 | 302 | 55.3% |
| **Registered Gate 3** | **0** | **0.0%** |
| Gate 4 / 5 / finish | 0 | — |

![factors](img/gate3_factors.png)

Physical outcomes on the Gate-3 leg (sim collision log): 53 runs struck Gate 3's frame at t≈11.5 s — they *reached the plane* and clipped it; ~124 missed the opening and hit ground ~3.5 s later; 50 clipped Gate 2's frame while passing it.

## The statistical verdict

Crossing positions sampled the entire error space — laterally −0.63…+0.59, vertically −0.62…+0.91, including 23 runs centered on both axes — and **none registered**. If per-attempt registration probability were even 1%, the chance of 0-for-546 is ~0.4%; the data caps p below ~0.5% per attempt for this configuration. **The failure was systematic. More attempts could not have qualified this controller**, and the batch's real product is that certainty plus the forensic dataset that located the mechanism — see [findings.md](findings.md).

## Top-10 closest approaches

![top10](img/top10.png)

All ten: laterally centered (|u_f| ≤ 0.05), deep (size 0.50–0.59), a shared +0.12…+0.23 vertical residual at gate loss, commit latched at t+2.1–2.6 with settled dwells. One family, one residual — the tightest description of "what stood between this controller and Gate 3."
