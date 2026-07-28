#!/usr/bin/env python3
"""Standalone training supervisor -- outlives any Claude session.

Launches a training script (default train_vision.py) as a subprocess,
relaunches it if it exits abnormally, and logs periodic health (ep_len_mean,
gates_passed, disk-free-GB) by tailing the training stdout log.

Auto-resume and checkpoint pruning are NOT reimplemented here -- they already
live in train_vision.py (find_latest_checkpoint + RetentionCheckpointCallback,
driven by --keep-last). This process's only job is: keep that script running,
and watch disk space so a relaunch never races an ENOSPC-corrupted checkpoint
(see the 2026-07-14 seed-1 incident RetentionCheckpointCallback's docstring
references).

Usage:
    python tools/guardian.py --save-dir surface_seed1 -- \
        --seed 1 --n-envs 4 --keep-last 50

Everything after `--` is passed to --train-script verbatim (n-envs, seed,
keep-last, motion-blur, etc). --save-dir is also forwarded to the training
script and is where guardian.log / train.log live.
"""
import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

EP_LEN_RE = re.compile(r"ep_len_mean\s*\|\s*([\d.]+)")
GATES_RE = re.compile(r"gates_passed\s*mean=([\d.]+)")
TIMESTEPS_RE = re.compile(r"total_timesteps\s*\|\s*([\d,]+)")


def log(path, msg):
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def disk_free_gb(path):
    _total, _used, free = shutil.disk_usage(path)
    return free / (1024 ** 3)


def tail_metrics(train_log_path, n_bytes=16384):
    """Best-effort scrape of the last ep_len_mean / gates_passed values from
    the training stdout log (SB3's own progress table +
    GatesPassedEvalCallback's print line)."""
    if not os.path.isfile(train_log_path):
        return None, None
    try:
        with open(train_log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n_bytes))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None, None
    ep_len = None
    gates = None
    timesteps = None
    for m in EP_LEN_RE.finditer(tail):
        ep_len = m.group(1)
    for m in GATES_RE.finditer(tail):
        gates = m.group(1)
    for m in TIMESTEPS_RE.finditer(tail):
        timesteps = m.group(1)
    return ep_len, gates, timesteps


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-script", default="train_vision.py")
    p.add_argument("--save-dir", required=True,
                    help="passed through to the training script as --save-dir; "
                         "also where guardian.log / train.log live")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--check-interval", type=float, default=60.0,
                    help="seconds between health-log lines")
    p.add_argument("--min-free-gb", type=float, default=2.0,
                    help="if free disk drops below this, stop the child BEFORE "
                         "it can corrupt an in-flight checkpoint; guardian exits "
                         "rather than looping against a full disk")
    p.add_argument("--restart-backoff", type=float, default=10.0,
                    help="seconds to wait before relaunching after a crash")
    p.add_argument("--stall-timeout", type=float, default=1800.0,
                    help="if total_timesteps in train.log hasn't advanced for this many "
                         "seconds while the child is still alive (proc.poll() == None), "
                         "treat it as hung -- e.g. a wedged GPU driver, see the 2026-07-26 "
                         "surface_seed1 incident where the child sat alive-but-frozen for "
                         "7+ hours -- and kill it so the outer loop relaunches")
    p.add_argument("train_args", nargs=argparse.REMAINDER,
                    help="everything after -- is passed to the training script verbatim")
    args = p.parse_args()

    train_args = args.train_args
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]

    if not os.path.isfile(args.train_script):
        print(f"guardian: --train-script not found: {args.train_script}", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.save_dir, exist_ok=True)
    guardian_log = os.path.join(args.save_dir, "guardian.log")
    train_log = os.path.join(args.save_dir, "train.log")

    cmd = [args.python, args.train_script, "--save-dir", args.save_dir] + train_args
    log(guardian_log, f"guardian starting. cmd={cmd!r}")

    stop = {"flag": False}

    def _handle_stop(signum, _frame):
        stop["flag"] = True
        log(guardian_log, f"received signal {signum}, stopping after current child exits")

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    while not stop["flag"]:
        free_gb = disk_free_gb(args.save_dir)
        if free_gb < args.min_free_gb:
            log(guardian_log, f"CRITICAL: {free_gb:.2f} GB free < --min-free-gb "
                               f"{args.min_free_gb}; refusing to launch. Free space and rerun.")
            sys.exit(1)

        log(guardian_log, f"launching child (free disk: {free_gb:.2f} GB)")
        with open(train_log, "a", encoding="utf-8") as tlog:
            proc = subprocess.Popen(cmd, stdout=tlog, stderr=subprocess.STDOUT)

            last_check = 0.0
            last_progress_val = None
            last_progress_time = time.time()
            while True:
                ret = proc.poll()
                if ret is not None:
                    break
                now = time.time()
                if now - last_check >= args.check_interval:
                    last_check = now
                    free_gb = disk_free_gb(args.save_dir)
                    ep_len, gates, timesteps = tail_metrics(train_log)
                    log(guardian_log,
                        f"health: ep_len_mean={ep_len or 'n/a'} "
                        f"gates_passed={gates or 'n/a'} free_disk_gb={free_gb:.2f} "
                        f"total_timesteps={timesteps or 'n/a'}")
                    if timesteps is not None and timesteps != last_progress_val:
                        last_progress_val = timesteps
                        last_progress_time = now
                    elif now - last_progress_time > args.stall_timeout:
                        log(guardian_log,
                            f"CRITICAL: total_timesteps stuck at "
                            f"{last_progress_val or 'n/a'} for over "
                            f"{args.stall_timeout:.0f}s while child is still alive -- "
                            f"treating as hung (e.g. wedged GPU driver) and killing it")
                        proc.terminate()
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        break
                    if free_gb < args.min_free_gb:
                        log(guardian_log, f"CRITICAL: {free_gb:.2f} GB free -- "
                                           f"terminating child before it corrupts a checkpoint")
                        proc.terminate()
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        break
                if stop["flag"]:
                    log(guardian_log, "stop requested -- terminating child")
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
                time.sleep(1.0)

        ret = proc.returncode
        if stop["flag"]:
            log(guardian_log, f"child exited (code={ret}) after stop request; guardian exiting")
            break
        if ret == 0:
            log(guardian_log, "child exited 0 (training complete) -- guardian exiting")
            break
        log(guardian_log, f"child exited abnormally (code={ret}) -- "
                           f"relaunching in {args.restart_backoff:.0f}s "
                           f"(train_vision.py's own auto-resume picks up the latest checkpoint)")
        time.sleep(args.restart_backoff)

    log(guardian_log, "guardian stopped")


if __name__ == "__main__":
    main()
