#!/usr/bin/env python3
"""Automated VQ1 attempt-runner — grind hundreds of unattended race attempts.

See ATTEMPT_RUNNER_DESIGN.md for the full rationale. Short version: VQ1 is won by
volume of attempts (unlimited tries), so once we have a competent checkpoint the
bottleneck is throughput, not model quality. This supervisor launches run_vq1.py
in a loop, classifies each attempt from race-status telemetry, logs it, resets by
hard-killing the child, and repeats.

Outcome classifier (from run_vq1.py `_MAVLinkReceiver`, ENCAPSULATED_DATA race
status, payload `<BQqqIq`):
    race_finish_ns  -> non-zero  => COMPLETED (all 8 gates cleared)
    active_gate     -> index of current target gate => gate-progress metric
    delta<=0        => race live; race window is 480 s (DCL default)

    COMPLETED : race_finish_ns > 0            (cross-check active_gate == 8)
    TIMEOUT   : race live longer than window, race_finish_ns still 0
    FAILED    : run ends / active_gate stalls < 8 before timeout, no finish
    ERROR     : never reached GO / no race status / child exits abnormally

Reset (Linchpin 2): there is NO in-band episode reset. A fresh attempt == a fresh
run_vq1.py process; the sim re-arms the next race countdown and run_vq1's stale-
echo guard rejects the prior race. So we hard-kill and relaunch.

NOTE: run_vq1.py has no --checkpoint flag. The model is the fixed
models_release/aigp_distill_final.zip (CANONICAL_MODEL_PATH). To grind a
different checkpoint, swap that file and pass --model-note to record which one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from queue import Empty, Queue

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_VQ1 = os.path.join(HERE, "run_vq1.py")
DEFAULT_MODEL = os.path.join(HERE, "models_release", "aigp_distill_final.zip")

try:
    from track import get_gate_positions
    _DEFAULT_NUM_GATES = len(get_gate_positions())   # course-agnostic: follows RACE_TRACK
except Exception:
    _DEFAULT_NUM_GATES = 8                            # fallback if track import fails

# run_vq1.py log-line patterns (logger emits INFO to stderr).
RE_GO = re.compile(r"\[Race\] GO")
RE_STATUS = re.compile(r"active_gate=(-?\d+).*?race_finish_ns=(-?\d+)")

_stop_requested = False


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _reader_thread(stream, q: "Queue[str]"):
    """Pump child stderr lines into a queue so the main loop can poll with a deadline."""
    try:
        for line in iter(stream.readline, ""):
            q.put(line.rstrip("\n"))
    finally:
        q.put(None)  # sentinel: stream closed / process exited


def _kill_tree(proc: subprocess.Popen):
    """Hard-kill the child and any descendants; verify it's gone before returning."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


def _count_existing(path: str) -> int:
    """Resume attempt numbering from the existing log so restarts don't double-count."""
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def _write_record(path: str, record: dict):
    """Append one JSONL record and force it to disk (survive a mid-grind reboot)."""
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def run_one_attempt(attempt_id: int, args) -> dict:
    """Launch one run_vq1.py attempt, watch its telemetry, classify the outcome."""
    cmd = [sys.executable, "-u", RUN_VQ1,
           "--port", str(args.port), "--vision-port", str(args.vision_port)]
    if args.host:
        cmd += ["--host", args.host]
    if args.control_mode:
        cmd += ["--control-mode", args.control_mode]
    if args.device:
        cmd += ["--device", args.device]

    start_wall = time.time()
    start_ts = _utcnow()
    proc = subprocess.Popen(
        cmd, cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, bufsize=1,
        creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
    )
    q: "Queue[str]" = Queue()
    threading.Thread(target=_reader_thread, args=(proc.stderr, q), daemon=True).start()

    go_seen = False
    go_wall = None
    max_gate = 0
    race_finish_ns = 0
    outcome = None
    note = ""

    while outcome is None:
        now = time.time()
        # Startup watchdog: never reached GO.
        if not go_seen and (now - start_wall) > args.startup_timeout:
            outcome, note = "ERROR", f"no GO within {args.startup_timeout:.0f}s (sim up?)"
            break
        # Race-window watchdog: live too long without a finish -> timeout.
        if go_seen and (now - go_wall) > args.race_window:
            outcome, note = "TIMEOUT", f"no finish within {args.race_window:.0f}s race window"
            break
        try:
            line = q.get(timeout=0.5)
        except Empty:
            if proc.poll() is not None:
                line = None
            else:
                continue
        if line is None:  # stream closed / process exited
            if race_finish_ns > 0 or max_gate >= args.num_gates:
                outcome, note = "COMPLETED", "process exited after finish"
            elif go_seen:
                outcome, note = "FAILED", f"process exited at gate {max_gate}, no finish"
            else:
                outcome, note = "ERROR", "process exited before GO"
            break

        if not go_seen and RE_GO.search(line):
            go_seen, go_wall = True, time.time()
            continue
        m = RE_STATUS.search(line)
        if m:
            gate = int(m.group(1))
            finish = int(m.group(2))
            max_gate = max(max_gate, gate)
            if finish > 0:
                race_finish_ns = finish
                outcome, note = "COMPLETED", "race_finish_ns > 0"
                break

    _kill_tree(proc)
    return {
        "attempt_id": attempt_id,
        "model_note": args.model_note,
        "start_ts": start_ts,
        "end_ts": _utcnow(),
        "outcome": outcome,
        "gates_passed": max_gate,
        "race_finish_ns": race_finish_ns,
        "duration_s": round(time.time() - start_wall, 1),
        "note": note,
    }


def main():
    p = argparse.ArgumentParser(description="Automated VQ1 attempt-runner")
    p.add_argument("--out", default=os.path.join(HERE, "attempts.jsonl"),
                   help="JSONL attempt log (append; default attempts.jsonl)")
    p.add_argument("--max-attempts", type=int, default=200,
                   help="Stop after this many attempts (default 200)")
    p.add_argument("--stop-on-first-completion", action="store_true",
                   help="Stop as soon as one attempt COMPLETED (submission run)")
    p.add_argument("--stop-after-k-completions", type=int, default=None,
                   help="Stop after K completions (confirm a completer reproduces)")
    p.add_argument("--race-window", type=float, default=480.0,
                   help="Max live-race seconds before TIMEOUT (DCL default 480)")
    p.add_argument("--startup-timeout", type=float, default=120.0,
                   help="Max seconds to reach GO before ERROR (default 120)")
    p.add_argument("--inter-attempt-gap", type=float, default=3.0,
                   help="Seconds to wait between attempts (let sim re-arm)")
    p.add_argument("--error-streak-abort", type=int, default=10,
                   help="Abort the grind after this many consecutive ERRORs")
    p.add_argument("--model-note", default="aigp_distill_final.zip",
                   help="Which checkpoint is loaded (recorded per attempt; run_vq1 "
                        "has no --checkpoint flag, so swap the file to change it)")
    p.add_argument("--num-gates", type=int, default=_DEFAULT_NUM_GATES,
                   help="Gates in the course; defaults to len(track.RACE_TRACK). "
                        "Completion is confirmed by race_finish_ns from the sim; "
                        "this only sizes the cross-check + histogram.")
    # run_vq1.py passthrough
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=14550)
    p.add_argument("--vision-port", type=int, default=5600)
    p.add_argument("--control-mode", default=None, choices=[None, "attitude", "rates", "actuator"])
    p.add_argument("--device", default=None, choices=[None, "cuda", "mps", "cpu"])
    args = p.parse_args()

    def _handle_sigint(signum, frame):
        global _stop_requested
        _stop_requested = True
        print("\n[runner] stop requested — finishing current attempt, then exiting.")
    signal.signal(signal.SIGINT, _handle_sigint)

    done = _count_existing(args.out)
    completions = 0
    error_streak = 0
    print(f"[runner] logging to {args.out} (resuming at attempt #{done + 1})")

    # Main grind loop.
    attempts_this_session = 0
    while not _stop_requested and attempts_this_session < args.max_attempts:
        attempt_id = done + attempts_this_session + 1
        print(f"[runner] attempt #{attempt_id} starting…")
        rec = run_one_attempt(attempt_id, args)
        _write_record(args.out, rec)
        attempts_this_session += 1
        print(f"[runner] attempt #{attempt_id}: {rec['outcome']} "
              f"(gate {rec['gates_passed']}/{args.num_gates}, {rec['duration_s']}s) — {rec['note']}")

        if rec["outcome"] == "ERROR":
            error_streak += 1
            if error_streak >= args.error_streak_abort:
                print(f"[runner] ABORT: {error_streak} consecutive ERRORs — is the sim up?")
                break
        else:
            error_streak = 0

        if rec["outcome"] == "COMPLETED":
            completions += 1
            if args.stop_on_first_completion:
                print("[runner] first completion — stopping (submission run).")
                break
            if args.stop_after_k_completions and completions >= args.stop_after_k_completions:
                print(f"[runner] reached {completions} completions — stopping.")
                break

        if not _stop_requested:
            time.sleep(args.inter_attempt_gap)

    print(f"[runner] session done: {attempts_this_session} attempts, {completions} completions.")
    print(f"[runner] summarize with:  python summarize_attempts.py --in {args.out}")


if __name__ == "__main__":
    main()
