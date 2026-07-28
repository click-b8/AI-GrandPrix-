#!/usr/bin/env python3
"""PASSIVE survey of everything the DCL sim actually sends.

Receive-only: no ARM, no SET_ATTITUDE_TARGET, no commands of any kind. Safe to run
alongside a real flight -- it binds its own UDP port, so give it a DIFFERENT --port from
the flier, or run it on its own during a race.

WHY THIS EXISTS. The flier reads ONE number out of the telemetry (active_gate) and
assumes no position data exists. ENCAPSULATED_DATA's payload is a fixed 253-byte array
and we unpack only the first 37 ('<BQqqIq'), so 216 bytes have never been examined. This
tool answers, from data rather than assumption:

  * which MAVLink message types the sim emits, and at what rate;
  * every field of every one of them;
  * which BYTE OFFSETS of the ENCAPSULATED_DATA payload CHANGE while the drone is
    flying -- the only reliable way to tell live telemetry from a stale buffer.

THE DISCRIMINATOR IS MOTION. Captured while the sim sits idle, the payload tail is
non-zero but frozen, which is exactly what uninitialised buffer bytes look like. Run this
DURING a race: any offset carrying position, velocity, bearing or distance MUST vary as
the drone moves. Offsets that stay frozen through a whole flight are not telemetry.

    python tools/dump_race_stream.py --seconds 60          # during a race
    python tools/dump_race_stream.py --seconds 20 --quiet  # idle baseline
"""
import argparse
import collections
import struct
import sys
import time

from pymavlink import mavutil

HEADER_FMT = "<BQqqIq"          # what the flier unpacks today
HEADER_LEN = struct.calcsize(HEADER_FMT)     # 37


def survey(port, seconds, quiet):
    conn = mavutil.mavlink_connection(f"udpin:127.0.0.1:{port}")
    print(f"[dump] listening udpin:127.0.0.1:{port} for {seconds:.0f}s "
          f"(PASSIVE -- no ARM, no commands)")
    counts = collections.Counter()
    sample = {}
    payloads = []
    gate_events = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        msg = conn.recv_match(blocking=True, timeout=1.0)
        if msg is None:
            continue
        t = msg.get_type()
        if t == "BAD_DATA":
            counts["BAD_DATA"] += 1
            continue
        counts[t] += 1
        sample.setdefault(t, msg)
        if t == "ENCAPSULATED_DATA":
            raw = bytes(msg.data)
            payloads.append((time.time() - t0, raw))
            if len(raw) >= HEADER_LEN:
                ag = struct.unpack_from(HEADER_FMT, raw)[4]
                if not gate_events or gate_events[-1][1] != ag:
                    gate_events.append((time.time() - t0, ag))
                    if not quiet:
                        print(f"[dump]   t={time.time()-t0:6.2f}s  active_gate -> {ag}")
    dur = time.time() - t0

    print(f"\n{'='*78}\nMESSAGE TYPES in {dur:.1f}s\n{'='*78}")
    print(f"{'type':<28}{'count':>8}{'Hz':>9}")
    for t, n in counts.most_common():
        print(f"{t:<28}{n:>8}{n/dur:>9.1f}")

    print(f"\n{'='*78}\nEVERY FIELD of every type\n{'='*78}")
    for t in sorted(sample):
        print(f"\n--- {t} ---")
        for f in sample[t].get_fieldnames():
            v = getattr(sample[t], f, None)
            if isinstance(v, (list, tuple, bytes, bytearray)) and len(v) > 12:
                print(f"   {f:<24} <{type(v).__name__} len={len(v)}>")
            else:
                print(f"   {f:<24} {v!r}")

    print(f"\n{'='*78}\nENCAPSULATED_DATA: {len(payloads)} payloads\n{'='*78}")
    if not payloads:
        print("none captured -- is a race running?")
        return
    print(f"lengths: {dict(collections.Counter(len(r) for _, r in payloads))}")
    print(f"type byte: {dict(collections.Counter(r[0] for _, r in payloads if r))}")
    print(f"active_gate timeline: {[(round(t,2), g) for t, g in gate_events]}")

    for tb in sorted({r[0] for _, r in payloads if r}):
        subset = [r for _, r in payloads if r and r[0] == tb]
        r = subset[-1]
        print(f"\n### type byte {tb} -- {len(subset)} msgs, last payload:")
        for off in range(0, len(r), 16):
            print(f"  {off:04d}: {r[off:off+16].hex(' ')}")
        h = struct.unpack_from(HEADER_FMT, r)
        print(f"\n  header '{HEADER_FMT}' (bytes 0..{HEADER_LEN-1}) -- what the flier reads:")
        print(f"    [ 0]    type        = {h[0]}")
        print(f"    [ 1: 9] sim_boot_ms = {h[1]}")
        print(f"    [ 9:17] race_start  = {h[2]}")
        print(f"    [17:25] race_finish = {h[3]}")
        print(f"    [25:29] ACTIVE_GATE = {h[4]}      <- the only field we use")
        print(f"    [29:37] trailing q  = {h[5]}      <- PARSED BUT DISCARDED"
              f"  (= {h[5]/1e9:.3f} s if ns)")

        # THE KEY MEASUREMENT: what moves while the drone moves?
        first = subset[0]
        n = min(len(first), *(len(x) for x in subset))
        varying = [i for i in range(n) if any(x[i] != first[i] for x in subset)]
        head_var = [i for i in varying if i < HEADER_LEN]
        tail_var = [i for i in varying if i >= HEADER_LEN]
        tail_nz = sorted({i for x in subset for i in range(HEADER_LEN, len(x)) if x[i]})
        print("\n  offsets that VARY across the capture:")
        print(f"    in the known header (0..{HEADER_LEN-1}): {head_var}")
        print(f"    PAST the header  ({HEADER_LEN}..{n-1}): {tail_var if tail_var else 'NONE'}")
        print(f"  offsets past the header that are ever NON-ZERO: {tail_nz if tail_nz else 'none'}")
        if tail_nz and not tail_var:
            print("\n  VERDICT: the tail is non-zero but COMPLETELY FROZEN. Bytes that never\n"
                  "  change while the drone moves are not position/velocity/bearing --\n"
                  "  this is the signature of an uninitialised buffer, not telemetry.")
        elif tail_var:
            print("\n  VERDICT: bytes past the header CHANGE during flight -- decode them.\n"
                  "  Candidate reads at each varying offset:")
            for i in sorted(set(tail_var))[:24]:
                for fmt, name in (("<f", "f32"), ("<i", "i32"), ("<h", "i16")):
                    try:
                        (v,) = struct.unpack_from(fmt, r, i)
                    except struct.error:
                        continue
                    if v and abs(v) < 1e9:
                        print(f"    off {i:3d} {name} = {v:+.6g}")
        else:
            print("\n  VERDICT: everything past the header is ZERO in every message.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=14550)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--quiet", action="store_true", help="no per-gate progress lines")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    survey(a.port, a.seconds, a.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
