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


def _graceful_stop(proc, wait=5.0):
    """End run_vq1 GRACEFULLY so its finally-block DISARM runs (a hard-kill skips
    it, which may be why the sim won't auto-arm the next race). Send CTRL_BREAK
    (Windows) / SIGINT (POSIX), wait for it to disarm+exit, then hard-kill as a
    fallback."""
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
    except Exception:
        pass
    try:
        proc.wait(timeout=wait)
    except Exception:
        pass
    _kill_tree(proc)   # ensure it's gone regardless


def run_attempt(attempt_id, args):
    """One race: launch run_vq1, watch its log, classify the outcome, kill it."""
    cmd = [sys.executable, "-u", RUN_VQ1, "--controller", args.controller,
           "--port", str(args.port)]
    if args.device:
        cmd += ["--device", args.device]

    start = time.time()
    start_ts = _utcnow()
    proc = Popen(cmd, cwd=ROOT, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1,
                 creationflags=(0x00000200 if os.name == "nt" else 0))  # NEW_PROCESS_GROUP
    q = Queue()
    Thread(target=_reader_thread, args=(proc.stdout, q), daemon=True).start()

    # Save the FULL child output per attempt so we can read the [servo]/[hybrid]
    # trace afterwards (the classifier only regex-scans lines; without this they're
    # discarded and the flight is unrecoverable).
    os.makedirs(args.logdir, exist_ok=True)
    logpath = os.path.join(args.logdir, f"attempt_{attempt_id:04d}.log")
    logf = open(logpath, "w", encoding="utf-8")
    logf.write(f"# attempt {attempt_id}  controller={args.controller}  start={start_ts}\n")
    logf.flush()

    go_seen = go_wall = None
    was_live = False
    max_gate = -1
    finish = 0
    outcome = note = None
    last_status = None            # latest [Race Status] fields (for the waiting print)
    last_status_print = 0.0

    while outcome is None:
        now = time.time()
        # OBSERVABLE WAIT: while there's no GO yet, echo the sim's race status every
        # few seconds so "waiting for a countdown that never comes" is obvious from
        # the console alone (delta>0 armed=True cycling = countdown coming; stale=True
        # or no status = the sim isn't arming -> the (B) re-arm problem).
        if not go_seen and (now - last_status_print) >= args.status_interval:
            last_status_print = now
            print(f"[grind #{attempt_id}] waiting for GO ({now - start:.0f}s) | "
                  f"{last_status or 'NO race status received yet -- did the sim arm a countdown?'}")
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

        logf.write(line + "\n")   # full trace to the per-attempt log

        if "[Race Status]" in line:
            last_status = line.split("[Race Status]", 1)[1].strip()

        if not go_seen and RE_GO.search(line):
            go_seen, go_wall = True, time.time()
            print(f"[grind #{attempt_id}] GO -- flying")
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

    _graceful_stop(proc)   # graceful (lets run_vq1 disarm) then hard-kill fallback
    logf.write(f"# outcome={outcome} gates_passed={max_gate} note={note}\n")
    logf.close()
    return {
        "attempt_id": attempt_id,
        "controller": args.controller,
        "start_ts": start_ts,
        "end_ts": _utcnow(),
        "outcome": outcome,
        "gates_passed": max_gate,
        "race_finish_ns": finish,
        "duration_s": round(time.time() - start, 1),
        "note": note,
        "log": logpath,
    }


def main():
    p = argparse.ArgumentParser(description="Thin unattended VQ1 grinder (vision-servo).")
    p.add_argument("--out", default=os.path.join(ROOT, "attempts.jsonl"))
    p.add_argument("--max-attempts", type=int, default=10)
    p.add_argument("--startup-timeout", type=float, default=120.0,
                    help="max s to reach GO before ERROR (sim up + auto-race on?)")
    p.add_argument("--race-window", type=float, default=480.0,
                    help="fallback TIMEOUT if a race stays live this long (DCL default 480)")
    p.add_argument("--settle", type=float, default=5.0,
                    help="pause after graceful stop before relaunch, so the sim resets the "
                         "drone to the grid + arms the next race ((B) re-arm needs a clean gap)")
    p.add_argument("--status-interval", type=float, default=5.0,
                    help="while waiting for GO, echo the sim's race status this often (s)")
    p.add_argument("--controller", default="vision-servo",
                   choices=["vision-servo", "hybrid"],
                   help="which hardcoded controller to grind (default vision-servo)")
    p.add_argument("--port", type=int, default=14550)
    p.add_argument("--device", default=None, choices=[None, "cuda", "mps", "cpu"])
    p.add_argument("--logdir", default=os.path.join(ROOT, "grind_logs"),
                   help="per-attempt full flight logs are written here")
    args = p.parse_args()

    def _sigint(_s, _f):
        global _stop
        _stop = True
        print("\n[grind] stop requested -- finishing current attempt, then exiting.")
    signal.signal(signal.SIGINT, _sigint)

    done = _count_existing(args.out)
    print(f"[grind] controller={args.controller.upper()}  ->  run_vq1 --controller {args.controller}")
    print(f"[grind] logging to {args.out} (resuming at attempt #{done + 1}); "
          f"per-attempt flight logs in {args.logdir}/; {NUM_GATES} gates; Ctrl-C to stop.")
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
