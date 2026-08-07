#!/usr/bin/env python3
"""Does the sim send position telemetry WHEN ASKED? (The passive tools never asked.)

WHY THIS EXISTS. tools/imu_check.py and tools/dump_race_stream.py are strictly
receive-only -- by design, they never transmit. So when they report
LOCAL_POSITION_NED / ODOMETRY / ATTITUDE as ABSENT, that is evidence about what the sim
STREAMS UNSOLICITED, and nothing at all about what it can be asked for. Most MAVLink
endpoints do not emit position on their own: you request it, via
MAV_CMD_SET_MESSAGE_INTERVAL (511) or the legacy REQUEST_DATA_STREAM. A passive absence
has therefore been read as "the interface has no position", which does not follow.

This probe closes that gap and nothing else:
  1. bind + wait for a heartbeat (so we know the target sys/comp);
  2. ask for the position/attitude messages explicitly, both the modern and legacy way;
  3. listen and report which of them actually arrive.

It sends ONLY stream requests. No ARM, no attitude/position setpoint, no mode change --
nothing that can move the aircraft.

    python tools/probe_request_position.py [--seconds 12] [--rate-hz 20]
"""
import argparse
import sys
import time
from collections import Counter

from pymavlink import mavutil

# The messages a position-based controller would need, in priority order.
WANTED = {
    30:  "ATTITUDE",
    31:  "ATTITUDE_QUATERNION",
    32:  "LOCAL_POSITION_NED",
    33:  "GLOBAL_POSITION_INT",
    331: "ODOMETRY",
    63:  "GLOBAL_POSITION_INT_COV",
    64:  "LOCAL_POSITION_NED_COV",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=14550)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--rate-hz", type=float, default=20.0)
    ap.add_argument("--hb-timeout", type=float, default=15.0)
    args = ap.parse_args()

    conn = mavutil.mavlink_connection(f"udpin:{args.ip}:{args.port}")
    print(f"[probe] bound udpin:{args.ip}:{args.port}; waiting for heartbeat ...")
    hb = conn.wait_heartbeat(timeout=args.hb_timeout)
    if hb is None:
        print("[probe] NO HEARTBEAT -- is the sim running? (nothing was sent)")
        return 2
    tgt_sys, tgt_comp = conn.target_system, conn.target_component
    print(f"[probe] heartbeat: sys={tgt_sys} comp={tgt_comp} "
          f"type={hb.type} autopilot={hb.autopilot} "
          f"base_mode={hb.base_mode} status={hb.system_status}")

    # ---- baseline: what arrives WITHOUT asking -------------------------------------
    base = Counter()
    t_end = time.time() + 3.0
    while time.time() < t_end:
        m = conn.recv_match(blocking=True, timeout=0.5)
        if m is not None:
            base[m.get_type()] += 1
    print(f"[probe] baseline (3 s, nothing requested): "
          f"{dict(sorted(base.items(), key=lambda kv: -kv[1]))}")

    # ---- ASK -----------------------------------------------------------------------
    interval_us = int(1e6 / max(args.rate_hz, 1e-6))
    print(f"[probe] requesting {len(WANTED)} message types at {args.rate_hz:.0f} Hz "
          f"via MAV_CMD_SET_MESSAGE_INTERVAL (511) ...")
    for msgid, name in WANTED.items():
        conn.mav.command_long_send(
            tgt_sys, tgt_comp,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            float(msgid), float(interval_us), 0, 0, 0, 0, 0)
        time.sleep(0.05)

    # Legacy fallback: some stacks only honour REQUEST_DATA_STREAM.
    print("[probe] also sending legacy REQUEST_DATA_STREAM (POSITION + EXTRA1 + ALL) ...")
    for stream in (mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                   mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
                   mavutil.mavlink.MAV_DATA_STREAM_ALL):
        conn.mav.request_data_stream_send(tgt_sys, tgt_comp, stream,
                                          int(args.rate_hz), 1)
        time.sleep(0.05)

    # ---- LISTEN --------------------------------------------------------------------
    got = Counter()
    acks = []
    samples = {}
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        m = conn.recv_match(blocking=True, timeout=0.5)
        if m is None:
            continue
        t = m.get_type()
        got[t] += 1
        if t == "COMMAND_ACK":
            acks.append((m.command, m.result))
        if t in WANTED.values() and t not in samples:
            samples[t] = m.to_dict()

    print(f"\n================ AFTER REQUESTING ({args.seconds:.0f} s) ================")
    for t, n in sorted(got.items(), key=lambda kv: -kv[1]):
        print(f"  {t:<28} {n:6d}   {n / args.seconds:6.1f} Hz")

    if acks:
        print("\n-- COMMAND_ACK --")
        for cmd, res in acks:
            meaning = {0: "ACCEPTED", 1: "TEMP_REJECTED", 2: "DENIED",
                       3: "UNSUPPORTED", 4: "FAILED"}.get(res, str(res))
            print(f"  command {cmd} -> result {res} ({meaning})")
    else:
        print("\n-- COMMAND_ACK -- NONE (the sim did not acknowledge the requests at all)")

    print("\n-- verdict per wanted message --")
    any_new = False
    for msgid, name in WANTED.items():
        n = got.get(name, 0)
        if n:
            any_new = True
            print(f"  {name:<24} PRESENT  ({n} msgs)  sample: "
                  f"{ {k: v for k, v in list(samples.get(name, {}).items())[:6]} }")
        else:
            print(f"  {name:<24} absent")

    print("\n" + "=" * 70)
    if any_new:
        print("RESULT: the sim DOES provide position/attitude telemetry when asked.")
        print("        The passive-listener conclusion was an artefact of never asking.")
    else:
        print("RESULT: still nothing after an explicit request (and the legacy stream")
        print("        request). Combined with the passive capture, that is real evidence")
        print("        this build does not expose position telemetry over MAVLink.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
