# Automated VQ1 Attempt-Runner — Design (OUR SIDE)

**Not for Pratik's runbook.** This is the Windows deploy-side workstream.

## Why this exists

Per Anduril's 6/30 email, **VQ1 is won by volume of attempts** — unlimited tries;
teams passed after "dozens to hundreds of attempts" and thousands of runs. A
policy that completes even **~1-in-20** qualifies. So once we have a competent
checkpoint the bottleneck is **throughput of attempts**, not model quality.
Manual launching does not scale to hundreds of runs; we need an unattended loop.

## Scope — BUILD AND START NOW (not "while the DGX trains")

This harness depends **only on `run_vq1.py` + the sim's race-status telemetry —
both verified working today.** It does **NOT** depend on B5 or the new
vision+IMU model. Build it now and start a grind against the **current**
checkpoint **tonight**, because:

1. **Battle-test the pipeline before the good checkpoint arrives.** Deploy →
   outcome-detection → reset/relaunch is the risky part, not the model. Shaking
   it out now means that when the completer lands we **point it at the new
   checkpoint and walk away** instead of debugging plumbing under time pressure.
2. **Get data we do not currently have.** We have only ever *watched individual
   runs*. A gate-histogram over **100–200 automated attempts** on the current
   models tells us **how far they actually get** (median gate reached,
   best-case, failure modes) — a real baseline instead of anecdotes.
3. **Answer the one open sim question** (does the sim auto-rearm between races —
   see Linchpin 2) that only an unattended back-to-back grind can settle.

Target: `attempt_runner.py` + `attempts.jsonl` + `summarize_attempts.py`, running
a grind against the current checkpoint tonight.

---

## Linchpin 1 — outcome classifier (RESOLVED)

**Source of truth:** `run_vq1.py`, `_MAVLinkReceiver._loop()`. (Note:
`dcl_vision_receiver.py` is vision-frames only; `dcl_mavlink_client.py`'s
`run_race()` is a stub timer — neither carries race status.)

Race status arrives as a MAVLink **`ENCAPSULATED_DATA`** message whose payload
`data[0] == 1` (discriminator), unpacked as:

```python
disc, sim_boot_ms, race_start_ms, race_finish_ns, active_gate, _ = \
    struct.unpack_from("<BQqqIq", raw)
```

| field | type | meaning |
|---|---|---|
| `sim_boot_ms` | uint64 | server clock, ms since sim boot |
| `race_start_ms` | int64 | clock at which race goes live; `delta = race_start_ms - sim_boot_ms`. `delta > 0` = countdown; `delta <= 0` = **race live** |
| **`race_finish_ns`** | int64 | **completion signal — non-zero/positive = race FINISHED (all-8 cleared)**; 0/unset during the race |
| **`active_gate`** | uint32 | **gate-progress signal — index of the current target gate**; advances as gates are cleared |

**There is no explicit crash/failure field in the telemetry**, so the classifier
is inferential:

- **COMPLETED** ← `race_finish_ns > 0` (canonical). Cross-check `active_gate == 8`.
- **TIMEOUT** ← race has been live (`delta <= 0`) longer than the DCL race window
  (**480 s** default, from `dcl_mavlink_client.run_race(max_duration=480.0)`) with
  `race_finish_ns` still 0.
- **FAILED** ← the run ends or `active_gate` stalls below 8 before timeout with no
  finish (crash/out-of-bounds inferred — there is no in-band crash flag).
- **ERROR** ← connection lost / no race-status messages / `run_vq1.py` exits
  non-zero / never reaches GO.

**Progress metric to log every attempt:** `gates_passed = active_gate` (max value
observed during the attempt), plus the terminal `race_finish_ns`.

## Linchpin 2 — reset (RESOLVED)

**No client-issued in-band episode reset exists.** `run_vq1.py` resets only its
own module globals for a fresh session (`_race_started/_countdown_armed/
_armed_race_start_ms`) and then waits for the sim's **next genuine `race_start`
countdown**, using `STALE_DELTA_MS` to reject stale prior-race echoes on the wire.
`adapter.run_loop()` does **not** self-terminate on finish.

**Therefore a fresh attempt = a fresh `run_vq1.py` process.** The runner:
launch → wait for GO (`_race_started`) → monitor race status → classify outcome →
**hard-kill the process** → relaunch. The client's stale-echo guard already
handles leftover prior-race state at startup, so relaunching is safe.

> **Residual uncertainty to settle empirically tonight (this is why we build
> now):** whether the sim **auto-rearms** a new race countdown after each
> finish/timeout, or needs an external trigger between attempts. The code proves
> multiple races occur on one sim (the "prior race" stale echoes), but not that
> they loop unattended. The first ~10–20 back-to-back relaunches will show
> whether each catches a fresh countdown. If it does NOT auto-rearm, the runner
> grows a sim re-arm/restart step here.

---

## Design

**Component: `attempt_runner.py`** — a supervisor loop that:

1. **Launch** one attempt: `python run_vq1.py [passthrough args]` as a child
   process, capturing stderr (INFO logs) to a per-attempt log. **`run_vq1.py` has
   no `--checkpoint` flag** — the model is hardcoded to `CANONICAL_MODEL_PATH =
   models_release/aigp_distill_final.zip`. To grind a different checkpoint, swap
   that file and record which one via `--model-note`.
2. **Wait for GO**, then **detect outcome** by tailing race status (either parse
   the child's `[Race Status]` log line, or — cleaner — snoop the same
   `ENCAPSULATED_DATA` stream). Classify COMPLETED / TIMEOUT / FAILED / ERROR per
   Linchpin 1.
3. **Log** one JSONL record per attempt (see schema below), flushed to disk
   immediately.
4. **Reset**: hard-kill the child, confirm the process/socket is gone, then loop
   (Linchpin 2). On repeated ERROR, escalate to a sim restart.
5. **Repeat** until a stop condition:
   - `--max-attempts N`,
   - `--stop-on-first-completion` (for the qualifying submission run),
   - `--stop-after-k-completions K` (confirm a completer reproduces).

### `attempts.jsonl` record schema

```json
{
  "attempt_id": 42,
  "model_note": "aigp_distill_final.zip",
  "start_ts": "2026-07-06T21:03:11Z",
  "end_ts": "2026-07-06T21:11:31Z",
  "outcome": "FAILED",
  "gates_passed": 3,
  "race_finish_ns": 0,
  "duration_s": 500.0,
  "note": "process exited at gate 3, no finish"
}
```

### `summarize_attempts.py`

Reads `attempts.jsonl` → prints **completion rate**, a **gate histogram**
(how many attempts reached gate 0,1,…,8), median/best gate reached, and
outcome breakdown (COMPLETED/TIMEOUT/FAILED/ERROR). This is the artifact that
tells us, at a glance, whether a checkpoint is worth submitting.

## Reliability requirements (where unattended grinds die)

- **Per-attempt watchdog:** hard-kill any attempt exceeding the race window +
  margin (e.g. 480 s + 60 s); mark `ERROR`/`TIMEOUT` so one hang can't stall the
  overnight grind.
- **Crash-safe logging:** append + flush/fsync every record; never memory-buffer.
  Survive a mid-grind reboot with the attempt history intact.
- **Clean process teardown:** kill the whole child process tree; verify the
  MAVLink/vision UDP ports are released before relaunch (stale sockets silently
  poison every subsequent attempt).
- **Auto-restart on ERROR streaks:** N consecutive ERRORs → restart the sim
  process (if we control it), then resume.
- **Idempotent attempt IDs + resumable tally:** a restarted runner continues the
  count instead of double-counting.
- **Log seed/initial conditions:** the sim is deterministic; if a completer only
  wins under specific conditions, we want that in the data.

## Tonight's concrete steps

1. Build `attempt_runner.py`, `attempts.jsonl` writer, `summarize_attempts.py`. ✅ **DONE** (compile-clean; `--help` and empty-log summary verified).
2. Confirm the DCL sim is running (the runner just relaunches `run_vq1.py`, which
   loads the fixed `aigp_distill_final.zip`). Then launch:
   `python attempt_runner.py --max-attempts 200`
3. Grind **100–200 attempts** unattended; in the morning read the gate-histogram
   from `python summarize_attempts.py --in attempts.jsonl` and confirm the
   auto-rearm behavior (Linchpin 2 — do back-to-back relaunches each catch a
   fresh countdown, or does the sim need an external re-arm between races?).
