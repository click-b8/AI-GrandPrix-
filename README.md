# AI Grand Prix — VQ1 Autonomous Racing Drone

Vision-only autonomous drone racing for the [AI Grand Prix](https://www.theaigrandprix.com) (Anduril / DCL / Neros) Virtual Qualifier — a controller that flies a simulated racing drone through a gate course using **only a forward camera feed over MAVLink**: no GPS, no position telemetry, no absolute coordinates.

![Qualification funnel](docs/img/funnel.png)

## Outcome, honestly

**We did not qualify.** The final overnight campaign ran **546 unattended attempts**: 92% passed Gate 1, 55% passed Gate 2, and **zero registered Gate 3** — the binding constraint. What the campaign produced instead is the more interesting artifact: a 302-run forensic dataset proving the failures were **systematic, not random** — two deterministic trajectory populations (a 3.12 s "straight" transit and a 6.67 s "veer", separated at 97.6% by leg duration alone), a commit-latch mechanism shown to be a *symptom* of the fork rather than its cause, and a replay-validated improvement to the commit rule that was never flown because simulator access ended at the deadline. The full analysis is in [docs/findings.md](docs/findings.md).

![Crossing scatter](docs/img/crossing_scatter.png)

## What's here

| area | contents |
|---|---|
| `tools/schedule_flier.py` | The flight controller: 3,900 lines, vision-servo "coast-tube" pipeline — gate detection → lateral bank servo with a derivative-aware commit latch → vertical trim with terminal descent + bounded blind hold. ~150 CLI flags encode the tuning campaign's full experiment history. |
| `vq1_vision_servo.py` | Vision pipeline: HSV blob gate detector → filtered (u_f, v_f, sz_f) servo signals. |
| `automation/` | `run_qualifier_batch.ps1` — unattended best-of-N harness: simulator process lifecycle (PID-tracked, port-isolation asserted), SendInput keystroke race resets, arm-before-restart sequencing, per-run classification, self-purging disk management. |
| `analysis/` | `flight_report.py` (per-run verdict + JSON contract) and trajectory tooling. |
| `tests/` | ~530 tests: controller logic, latch behavior, batch lifecycle, MAVLink compliance. |
| `docs/` | [Architecture](docs/architecture.md) · [Controller](docs/controller.md) · [Vision](docs/vision.md) · [Automation](docs/automation.md) · [Experiments](docs/experiments.md) · [Overnight batch](docs/overnight_batch.md) · [Findings](docs/findings.md) · [Lessons learned](docs/lessons_learned.md) · [Future work](docs/future_work.md) |
| `results/overnight-2026-08-03/` | The final experiment: summary CSV, per-run JSON, top-10 closest traces, and [REPRODUCE.md](results/overnight-2026-08-03/REPRODUCE.md). Full 1 GB raw dataset ships as a GitHub Release asset. |
| `archive/` | Earlier eras, preserved and indexed: RL training, DCL hardware integration, ~560 manual tuning runs, abandoned experiments. |

## The problem in one paragraph

The simulated drone coasts nose-down (~−17.8° pitch, uncontrollable) with a hard thrust ceiling that removes climb authority: the controller can steer **bank** and modulate **descent** only, from a camera whose gate detector is a color-blob centroid — no corners, no pose. Every gate approach is therefore a one-shot ballistic intercept steered by two error scalars, ending with the gate leaving the frame ~2.4 m before the plane and the final half-second flown blind. Loop rate on the target hardware swung 39–117 Hz, which became its own research problem.

## Headline findings

![Duration bimodality](docs/img/duration_bimodal.png)

1. **The Gate-3 miss modes are two deterministic trajectories, not noise.** Leg transit time is perfectly bimodal (3.12 s vs 6.67 s, empty gap between); the fast population arrives laterally centered but high, the slow population veers left but vertically centered. Where a run crossed the gate plane sampled the entire error space across 302 runs — and registered zero — proving a systematic cause, not variance to be harvested with more attempts.
2. **The commit latch is a symptom, not the fork.** 67 fast runs never latched yet flew straight; 14 slow runs latched yet veered. The populations diverge in u_f *before* the latch is even eligible. But the latch matters as a terminal instrument: latched fast runs crossed at |u_f| 0.088 median vs 0.254 unlatched — it blinds the servo to a terminal parallax spike.
3. **The dominant latch blocker was structural:** in 43% of runs the arm precondition (filtered gate size dipping below the commit threshold between gates) never occurred. An offline replay across all 302 runs found a strictly dominating rule (size 0.20 / align 0.12 / rate 0.16) — 86:1 confusion, all top-10 runs preserved — that was never flown.
4. **"Median loop rate selects the miss mode" was a proxy, not a cause** — believed for a day, then falsified by the same dataset that suggested it (the best centered runs came from low-rate machines flying the fast leg).

## Reproducing the final experiment

See [results/overnight-2026-08-03/REPRODUCE.md](results/overnight-2026-08-03/REPRODUCE.md) for the exact frozen controller flags, the batch command, environment assumptions, and the analysis pipeline. Note: the official simulator requires an online account login and qualifier access, which **ended at the competition deadline** — the dataset and replay tooling here are what remain reproducible.

## License & provenance

Built by a one-person team for AI-GP Virtual Qualifier Round 1 (July–August 2026). The simulator and its assets are the property of the AI Grand Prix and are **not** included here.
