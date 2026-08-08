# AI Grand Prix — VQ1 Autonomous Racing Drone

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20PowerShell%205.1-0078D6?logo=windows&logoColor=white)](#5-quick-start)
[![Tests](https://img.shields.io/badge/tests-423%20passed%2C%20114%20skipped-success)](#5-quick-start)
[![Competition](https://img.shields.io/badge/AI%20Grand%20Prix-Virtual%20Qualifier%20R1-E4572E)](https://www.theaigrandprix.com)
[![Result](https://img.shields.io/badge/result-did%20not%20qualify-lightgrey)](#6-results--findings)

Vision-only autonomous drone racing for the [AI Grand Prix](https://www.theaigrandprix.com)
(Anduril / DCL / Neros) Virtual Qualifier — a controller that flies a simulated racing
drone through a gate course using **only a forward camera feed over MAVLink**: no GPS, no
position telemetry, no absolute coordinates.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Software Map](#3-software-map)
4. [The Problem](#4-the-problem)
5. [Quick Start](#5-quick-start)
6. [Results & Findings](#6-results--findings)
7. [Repository Structure](#7-repository-structure)
8. [Data & Reproducibility](#8-data--reproducibility)
9. [Lessons Learned & Future Work](#9-lessons-learned--future-work)
10. [Acknowledgments](#10-acknowledgments)

---

## 1. System Overview

The task: fly a 6-gate course autonomously, with zero human intervention, from a single
forward camera. The simulator publishes no position, no velocity and no gate coordinates —
the only spatial information available is where a gate's colour blob sits in a 320×180
frame.

![Qualification funnel](docs/img/funnel.png)

| Specification | Value |
|---|---|
| Sensing | Forward FPV camera only (640×360 source, 320×180 to the detector) |
| Position telemetry | **None** — no GPS, no `LOCAL_POSITION_NED`, no gate coordinates |
| Control authority | Bank (roll) + thrust. Pitch coasts nose-down at ~−17.8°, uncontrollable |
| Gate detection | HSV colour-blob centroid → `(u_f, v_f, sz_f)` — no corners, no pose |
| Control loop | 39–117 Hz observed; all laws written in per-second units because the rate is not constant |
| Course | 6 gates (START, g1–g4, FINISH), 6 legs, slopes +1.5° to +17.1° |
| Final campaign | 546 unattended attempts over one night |
| Outcome | **Did not qualify** — 92% Gate 1, 55% Gate 2, 0% Gate 3 |

---

## 2. Architecture

Three processes, one direction of authority: the simulator owns race state, the flier owns
the control loop, the batch runner owns everything the flier cannot see.

```
┌─────────────────────────────┐        UDP 14560 (MAVLink)       ┌──────────────────────┐
│  AI-GP Simulator            │ ─────────────────────────────▶   │  schedule_flier.py   │
│  (DCGame-Win64-Shipping)    │   ATTITUDE / TIMESYNC / race     │  --coast-tube        │
│                             │                                  │                      │
│   renders FPV camera        │        UDP 5601 (vision)         │  ┌────────────────┐  │
│   owns race state           │ ─────────────────────────────▶   │  │ vq1_vision_    │  │
│                             │      ENCAPSULATED_DATA           │  │ servo detector │  │
│                             │                                  │  └───────┬────────┘  │
│                             │   SET_ATTITUDE_TARGET            │          ▼           │
│                             │ ◀─────────────────────────────   │  control law (bank,  │
└─────────────────────────────┘     (roll, pitch=free, thrust)   │  thrust) @ loop rate │
                                                                 └──────────────────────┘
                    ▲                                                        ▲
                    │  Esc / ↓ / Enter (SendInput)                           │ launches, watches,
            ┌───────┴──────────────────────────────────────────────────────┬┘ classifies
            │              automation/run_qualifier_batch.ps1              │
            │   sim lifecycle · race reset · best-of-N loop · purge        │
            └──────────────────────────────┬───────────────────────────────┘
                                           ▼
                          analysis/flight_report.py  →  batch_summary.csv + per-run JSON
```

Full walkthrough, including the constraints that shaped each layer:
**[docs/architecture.md](docs/architecture.md)**.

---

## 3. Software Map

| Component | Path | Role |
|---|---|---|
| Flight controller | [`tools/schedule_flier.py`](tools/schedule_flier.py) | The entry point. ~3,900 lines, 158 CLI flags encoding the full experiment history: gate detection → lateral bank servo with a derivative-aware commit latch → vertical trim with terminal descent and a bounded blind hold. |
| Vision pipeline | [`vq1_vision_servo.py`](vq1_vision_servo.py) | HSV blob gate detector and cyan rail detector → filtered servo signals `(u_f, v_f, sz_f)`. |
| Batch automation | [`automation/run_qualifier_batch.ps1`](automation/run_qualifier_batch.ps1) | Unattended campaign harness: PID-tracked simulator lifecycle, UDP port-isolation assertions, `SendInput` race resets, arm-before-restart sequencing, per-run classification, self-purging disk management. |
| Per-run analysis | [`analysis/flight_report.py`](analysis/flight_report.py) | Reduces a ~94-column tick log to a verdict line plus a JSON record — the classification contract the batch runner consumes. |
| Course geometry | [`analysis/track.py`](analysis/track.py), [`nav_frames.py`](nav_frames.py) | UE → MuJoCo course transform and navigation frames, validated against the extracted course JSON. |
| Tests | [`tests/`](tests) | 423 passed, 114 skipped. Controller logic, latch behaviour, batch lifecycle, MAVLink compliance, course transforms. |

---

## 4. The Problem

The simulated drone coasts nose-down (~−17.8° pitch, uncontrollable) with a hard thrust
ceiling that removes climb authority: the controller can steer **bank** and modulate
**descent** only, from a camera whose gate detector is a colour-blob centroid — no corners,
no pose. Every gate approach is therefore a one-shot ballistic intercept steered by two
error scalars, ending with the gate leaving the frame ~2.4 m before the plane and the final
half-second flown blind. Loop rate on the target hardware swung 39–117 Hz, which became its
own research problem.

---

## 5. Quick Start

> The official simulator requires an online account and qualifier access, which **ended at
> the competition deadline**. The dataset, replay tooling and tests below remain fully
> reproducible; live flight does not.

**1. Install dependencies** (Python 3.11+):

```bash
pip install -r requirements.txt
```

**2. Run the test suite** — no simulator required:

```bash
python -m pytest tests -q
```

Expect **423 passed, 114 skipped**. The skips are tests gated on bulk flight logs
(`archive/tuning-runs/`, ~489 MB) and model checkpoints that are deliberately not tracked.

**3. Fly one attempt** — requires the simulator running, logged in, drone at the start gate.
The exact frozen flag set is documented in
[REPRODUCE.md](results/overnight-2026-08-03/REPRODUCE.md):

```powershell
python tools/schedule_flier.py --coast-tube --const-thrust 0.275 `
  --gate-commit-rate-max 0.13 --gate-commit-stable-s 0.15 `
  --gate3-vert-descent --gate3-vert-descent-delta 0.015 --gate3-vert-descent-size 0.25 `
  --gate3-vert-descent-hold-s 0.30 `
  --no-pitch-hold --tick-log run.csv
```

**4. Score the run:**

```bash
python analysis/flight_report.py run.csv            # one-line verdict
python analysis/flight_report.py run.csv --json -   # machine-readable record
```

**5. Run an unattended campaign** (Windows, PowerShell 5.1):

```powershell
.\automation\run_qualifier_batch.ps1 -KeepSimAlive -Mode holdblind -HoldSeconds 0.30
```

---

## 6. Results & Findings

**We did not qualify.** The final overnight campaign ran **546 unattended attempts**:

| Milestone | Rate |
|---|---|
| Reached the Gate-1 leg | 100% |
| Passed Gate 1 | 92% |
| Passed Gate 2 | 55% |
| **Registered Gate 3** | **0%** |

What the campaign produced instead is the more interesting artifact: a 302-run forensic
dataset proving the failures were **systematic, not random**.

![Crossing scatter](docs/img/crossing_scatter.png)

![Duration bimodality](docs/img/duration_bimodal.png)

1. **The Gate-3 miss modes are two deterministic trajectories, not noise.** Leg transit time
   is perfectly bimodal — 3.12 s vs 6.67 s with an empty gap, 97.6% separable on duration
   alone. The fast population arrives laterally centred but high; the slow population veers
   left but vertically centred. Across 302 runs the crossings sampled the entire error space
   and registered zero — a systematic cause, not variance to be harvested with more attempts.
2. **The commit latch is a symptom, not the fork.** 67 fast runs never latched yet flew
   straight; 14 slow runs latched yet veered. The populations diverge in `u_f` *before* the
   latch is eligible. It still matters as a terminal instrument: latched fast runs crossed at
   |u_f| 0.088 median versus 0.254 unlatched.
3. **The dominant latch blocker was structural.** In 43% of runs the arm precondition never
   occurred. An offline replay across all 302 runs found a strictly dominating rule
   (size 0.20 / align 0.12 / rate 0.16) — 86:1 confusion, all top-10 runs preserved — that
   was never flown, because simulator access ended first.
4. **"Median loop rate selects the miss mode" was a proxy, not a cause** — believed for a
   day, then falsified by the same dataset that suggested it.

Full analysis: **[docs/findings.md](docs/findings.md)**.

---

## 7. Repository Structure

```
AI-GrandPrix-/
├── tools/schedule_flier.py     # flight controller — the entry point (158 flags)
├── vq1_vision_servo.py         # gate + rail detectors, servo signal filtering
├── automation/                 # unattended batch harness (lifecycle, resets, classify)
├── analysis/                   # flight_report.py (verdict + JSON), track.py (geometry)
├── tests/                      # 537 tests; fixtures/ carries the detector frame
├── docs/                       # architecture, controller, vision, automation, findings
│   └── img/                    # figures used by the docs and this README
├── results/overnight-2026-08-03/
│   ├── batch_summary.csv       # one row per attempt (546)
│   ├── json/                   # per-run structured records
│   ├── top10_traces/           # the ten closest Gate-3 approaches
│   └── REPRODUCE.md            # frozen flags, environment, analysis pipeline
├── archive/                    # earlier eras, indexed: RL training, DCL hardware,
│                               #   tuning runs, abandoned experiments, design specs
├── config.py                   # camera intrinsics, course constants
└── course_gates_cm.json        # extracted gate geometry (UE world space, cm)
```

---

## 8. Data & Reproducibility

The published subset — `batch_summary.csv`, per-run JSON, keepers, the top-10 traces and
[REPRODUCE.md](results/overnight-2026-08-03/REPRODUCE.md) — is tracked in this repository.

The **complete raw dataset** (302 Gate-3-leg tick traces, per-run JSON, diagnostics;
1,310 files) ships as a Release asset rather than a repository blob:

- **Release:** [`overnight-2026-08-03`](https://github.com/click-b8/AI-GrandPrix-/releases/tag/overnight-2026-08-03)
- **Asset:** `vq1-overnight-raw-dataset-2026-08-03.tar.gz` (56,466,296 bytes)
- **SHA-256:** `2957f75dccf5d113e56b752d2be0edc9a8e7f53ce9c80053c7e5a41263d57081`

Verify after download:

```bash
sha256sum -c results/vq1-overnight-raw-dataset-2026-08-03.sha256
```

Extract alongside `REPRODUCE.md`, which documents the exact frozen controller flags, the
batch command, environment assumptions and the analysis pipeline.

---

## 9. Lessons Learned & Future Work

- **A proxy that predicts is not a cause.** Loop rate correlated with the miss mode for a
  full day before the same dataset falsified it. → [docs/lessons_learned.md](docs/lessons_learned.md)
- **Replay beats flying.** The entire commit-latch investigation and its replacement rule
  were derived offline from logged state, flying zero additional attempts — which is why the
  work survived losing simulator access. → [docs/findings.md](docs/findings.md)
- **Automation is where the experiment actually lives.** More engineering went into making
  attempts *comparable* — process isolation, deterministic resets, honest classification —
  than into the control law itself. → [docs/automation.md](docs/automation.md)
- **The unflown fix.** The dominating commit rule, terminal corner-based pose, and closing
  the blind final 2.4 m are the three threads that were live at the deadline.
  → [docs/future_work.md](docs/future_work.md)

---

## 10. Acknowledgments

Built by a one-person team for the AI Grand Prix Virtual Qualifier Round 1
(July–August 2026).

The AI Grand Prix is organised by **Anduril Industries** in partnership with the
**Drone Champions League (DCL)** and **Neros Technologies**. The simulator, its assets and
the course are their property and are **not** included in this repository — only the
extracted gate coordinates required to reproduce the analysis.
