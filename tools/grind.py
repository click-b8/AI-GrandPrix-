#!/usr/bin/env python3
"""Thinnest VQ1 grinder -- loop the EXISTING vision-servo flight, unattended.

Phase 0 confirmed the DCL sim AUTO-RESTARTS races, so no GO trigger is needed:
each fresh run_vq1.py connection catches the next countdown on its own (its
stale-echo guard waits for a genuine future race_start). This wraps that in a
loop to PROVE THE HARNESS -- no controller changes:

    launch run_vq1 --controller vision-servo  ->  watch its race-status log  ->
    detect the race END  ->  append one line to attempts.jsonl  ->  hard-kill ->
    relaunch for the next auto-race  ->  repeat.

Race-status source: run_vq1's own "[Race Status]" log line (~1 Hz), which carries
delta = race_start - sim_boot, active_gate, and race_finish_ns. We do NOT bind
the MAVLink port -- run_vq1 owns it.

Race-END detection (the transition that must be clean, or we miss/double-log):
  COMPLETED : race_finish_ns > 0.
  FAILED    : we were LIVE (delta<=0 seen) and delta goes back > 0 -- a fresh
              countdown == the previous race ended and the next is arming. This
              is the auto-restart edge; it fires in ~1 s, not after the 480 s
              window. Exactly ONE per race, so no double-logging.
  TIMEOUT   : live longer than --race-window (fallback only).
  ERROR     : never reached GO, or the child exited before GO.
On race-END: log, hard-kill run_vq1, relaunch. The fresh process waits for the
next genuine countdown, so we never fly a stale race (no missed/duplicated
attempts). Throughput note: one race per (fly + relaunch + wait-for-next); to
fly EVERY back-to-back race without relaunch, run_vq1 would need to re-arm
in-process -- a controller change deliberately deferred until the harness is
proven.

    python tools/grind.py --max-attempts 10
    python tools/grind.py --out attempts.jsonl --device cpu
"""
import argparse
import os
import re
import signal
import sys
import time
from queue import Empty, Queue
from subprocess import PIPE, STDOUT, Popen
from threading import Thread

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# GRIND_RUN_VQ1 lets a test point the grinder at a fake race emitter (no sim).
RUN_VQ1 = os.environ.get("GRIND_RUN_VQ1", os.path.join(ROOT, "run_vq1.py"))

sys.path.insert(0, ROOT)
# Reuse attempt_runner's tested process/log helpers (kill-tree, fsync'd append).
from attempt_runner import _kill_tree, _reader_thread, _count_existing, _write_record, _utcnow  # noqa: E402

try:
    from track import get_gate_positions
    NUM_GATES = len(get_gate_positions())
except Exception:
    NUM_GATES = 6

RE_GO = re.compile(r"\[Race\] GO")
# "[Race Status] sim_boot=.. race_start=.. delta=D .. active_gate=G .. race_finish_ns=F"
RE_STATUS = re.compile(r"delta=(-?\d+).*?active_gate=(-?\d+).*?race_finish_ns=(-?\d+)")

_stop = False


def run_attempt(attempt_id, args):
    """One race: launch run_vq1, watch its log, classify the outcome, kill it."""
    cmd = [sys.executable, "-u", RUN_VQ1, "--controller", "vision-servo",
           "--port", str(args.port)]
    if args.device:
        cmd += ["--device", args.device]

    start = time.time()
    start_ts = _utcnow()
    proc = Popen(cmd, cwd=ROOT, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1,
                 creationflags=(0x00000200 if os.name == "nt" else 0))  # NEW_PROCESS_GROUP
    q = Queue()
    Thread(target=_reader_thread, args=(proc.stdout, q), daemon=True).start()

    go_seen = go_wall = None
    was_live = False
    max_gate = -1
    finish = 0
    outcome = note = None

    while outcome is None:
        now = time.time()
        if not go_seen and (now - start) > args.startup_timeout:
            outcome, note = "ERROR", f"no GO within {args.startup_timeout:.0f}s (sim up? auto-race on?)"
            break
        if go_seen and (now - go_wall) > args.race_window:
            outcome, note = "TIMEOUT", f"live > {args.race_window:.0f}s window"
            break
        try:
            line = q.get(timeout=0.5)
        except Empty:
            if proc.poll() is not None:
                line = None
            else:
                continue
        if line is None:  # process exited
            if finish > 0 or max_gate >= NUM_GATES:
                outcome, note = "COMPLETED", "process exited after finish"
            elif go_seen:
                outcome, note = "FAILED", f"process exited at gate {max_gate}"
            else:
                outcome, note = "ERROR", "process exited before GO"
            break

        if not go_seen and RE_GO.search(line):
            go_seen, go_wall = True, time.time()
            continue
        m = RE_STATUS.search(line)
        if m:
            delta, gate, fin = int(m.group(1)), int(m.group(2)), int(m.group(3))
            max_gate = max(max_gate, gate)
            if delta <= 0:
                was_live = True
            if fin > 0:
                finish = fin
                outcome, note = "COMPLETED", "race_finish_ns > 0"
                break
            # THE clean race-END edge: we were live, now a fresh countdown -> the
            # race ended (crash/DQ/timeout) and the next is arming.
            if was_live and delta > 0:
                outcome, note = "FAILED", f"race ended at gate {max_gate} (next countdown armed)"
                break

    _kill_tree(proc)
    return {
        "attempt_id": attempt_id,
        "controller": "vision-servo",
        "start_ts": start_ts,
        "end_ts": _utcnow(),
        "outcome": outcome,
        "gates_passed": max_gate,
        "race_finish_ns": finish,
        "duration_s": round(time.time() - start, 1),
        "note": note,
    }


def main():
    p = argparse.ArgumentParser(description="Thin unattended VQ1 grinder (vision-servo).")
    p.add_argument("--out", default=os.path.join(ROOT, "attempts.jsonl"))
    p.add_argument("--max-attempts", type=int, default=10)
    p.add_argument("--startup-timeout", type=float, default=120.0,
                    help="max s to reach GO before ERROR (sim up + auto-race on?)")
    p.add_argument("--race-window", type=float, default=480.0,
                    help="fallback TIMEOUT if a race stays live this long (DCL default 480)")
    p.add_argument("--settle", type=float, default=3.0,
                    help="pause after kill before relaunch, so the sim/port settle")
    p.add_argument("--port", type=int, default=14550)
    p.add_argument("--device", default=None, choices=[None, "cuda", "mps", "cpu"])
    args = p.parse_args()

    def _sigint(_s, _f):
        global _stop
        _stop = True
        print("\n[grind] stop requested -- finishing current attempt, then exiting.")
    signal.signal(signal.SIGINT, _sigint)

    done = _count_existing(args.out)
    print(f"[grind] logging to {args.out} (resuming at attempt #{done + 1}); "
          f"{NUM_GATES} gates; Ctrl-C to stop.")
    n = 0
    completions = 0
    while not _stop and n < args.max_attempts:
        aid = done + n + 1
        print(f"[grind] attempt #{aid} starting...")
        rec = run_attempt(aid, args)
        _write_record(args.out, rec)
        n += 1
        completions += (rec["outcome"] == "COMPLETED")
        print(f"[grind] #{aid}: {rec['outcome']} (gate {rec['gates_passed']}/{NUM_GATES}, "
              f"{rec['duration_s']}s) -- {rec['note']}")
        if not _stop:
            time.sleep(args.settle)

    print(f"[grind] done: {n} attempts this session, {completions} completions. "
          f"Summary: python summarize_attempts.py --in {args.out}")


if __name__ == "__main__":
    main()
