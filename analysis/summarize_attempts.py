#!/usr/bin/env python3
"""Summarize an attempts.jsonl grind: completion rate + gate histogram.

This is the artifact that tells us, at a glance, whether a checkpoint is worth
submitting and how far the current models actually get — the baseline we lack
because we've only ever watched individual runs.

Usage:
    python summarize_attempts.py --in attempts.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from track import get_gate_positions
    _DEFAULT_NUM_GATES = len(get_gate_positions())   # course-agnostic: follows RACE_TRACK
except Exception:
    _DEFAULT_NUM_GATES = 8                            # fallback if track import fails


def load(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    p = argparse.ArgumentParser(description="Summarize VQ1 attempt log")
    p.add_argument("--in", dest="infile", default="attempts.jsonl")
    p.add_argument("--num-gates", type=int, default=_DEFAULT_NUM_GATES)
    args = p.parse_args()

    if not os.path.exists(args.infile) or not load(args.infile):
        print(f"No attempts logged yet in {args.infile}.")
        return

    recs = load(args.infile)
    n = len(recs)
    outcomes = Counter(r.get("outcome") for r in recs)
    gates = [int(r.get("gates_passed", 0)) for r in recs]
    completions = outcomes.get("COMPLETED", 0)

    print(f"=== VQ1 Attempt Summary ({n} attempts) ===")
    print(f"Log: {args.infile}")
    models = sorted({r.get("model_note", "?") for r in recs})
    print(f"Checkpoint(s): {', '.join(models)}")
    print()

    print("Outcomes:")
    for k in ("COMPLETED", "TIMEOUT", "FAILED", "ERROR"):
        c = outcomes.get(k, 0)
        print(f"  {k:10s} {c:4d}  ({100.0 * c / n:5.1f}%)")
    print()

    rate = 100.0 * completions / n
    print(f"COMPLETION RATE: {completions}/{n} = {rate:.1f}%")
    if completions:
        print(f"  -> ~1 completion per {n / completions:.1f} attempts")
    print()

    # Gate histogram: how many attempts reached at least gate g.
    print(f"Gate histogram (max gate reached, 0..{args.num_gates}):")
    hist = Counter(gates)
    peak = max((hist.get(g, 0) for g in range(args.num_gates + 1)), default=0) or 1
    for g in range(args.num_gates + 1):
        c = hist.get(g, 0)
        bar = "#" * int(round(40 * c / peak))
        print(f"  gate {g:>2}: {c:4d}  {bar}")
    print()

    srt = sorted(gates)
    median = srt[n // 2]
    print(f"Best gate reached: {max(gates)} / {args.num_gates}")
    print(f"Median gate reached: {median} / {args.num_gates}")
    print(f"Mean gate reached: {sum(gates) / n:.2f} / {args.num_gates}")


if __name__ == "__main__":
    main()
