# Architecture

## System overview

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

## The three layers

**1. Flight (`tools/schedule_flier.py` + `vq1_vision_servo.py`).** A single-process control loop: MAVLink connect → heartbeat → timesync → ARM → wait for the sim's race-start (GO) → fly a 6-segment schedule (START, gates 1–4, FINISH) with vision-servo corrections. The loop runs as fast as the machine allows (39–117 Hz observed); all control laws are written in real-time units (per-second gains, seconds-based dwells) precisely because the rate is not constant. Detailed pipeline: [controller.md](controller.md), [vision.md](vision.md).

**2. Automation (`automation/run_qualifier_batch.ps1`).** Turns one manual flight into an unattended campaign. Owns what the flier cannot: the simulator process (launcher vs the real shipping binary), UDP port isolation, the in-game race reset (keystrokes via Win32 `SendInput`), the arm-before-restart ordering that a race countdown demands, per-attempt classification, disk hygiene, and stop conditions (the only success stop is the sim's own race-finish signal). Details: [automation.md](automation.md).

**3. Analysis (`analysis/flight_report.py` + offline replay).** Every run emits a ~94-column tick log. `flight_report.py` reduces one CSV to a verdict line and a JSON record (the batch's classification contract). The forensic layer replays logged state against candidate control rules offline — the entire commit-latch investigation ([findings.md](findings.md)) flew zero additional flights.

## Constraints that shaped everything

- **Vision-only:** no position, velocity, or map. Two filtered scalars (u_f lateral, v_f vertical) plus blob size are the entire state estimate.
- **No climb authority:** thrust ceiling ~0.335 with the drone coasting nose-down ~−17.8°; the vertical problem is strictly *descent management*.
- **Asymmetric, capped bank:** ±11° schedule bank, tighter (+0/−9°) on the Gate-3 leg.
- **Terminal blindness:** the gate leaves the camera frame at ~2.4 m out (blob size ~0.55); the last ~0.5–0.75 s of every approach is open-loop.
- **Non-deterministic loop rate:** ARM-translated x86 sim on battery-managed hardware; rate became a first-class experimental variable.
- **One shot per run, ~40 s per attempt:** no mid-flight resets; the sim's race state is authoritative.
