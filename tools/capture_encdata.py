#!/usr/bin/env python3
"""Capture EVERY ENCAPSULATED_DATA payload during a live lap, for offline decode.

PASSIVE listener (no ARM, no setpoints -- safe). Logs each 253-byte race-status
payload as hex + wall time + the parsed active_gate, to a CSV. Run it DURING an
actual armed race so the drone moves through the known gate positions; then the
payload's varying bytes can be correlated against course_gates_cm.json to decide,
definitively, whether drone POSITION is hidden in the stream.

    python tools/capture_encdata.py --seconds 120 --out encap_capture.csv
"""
import argparse, struct, time, csv, sys
from pymavlink import mavutil

HEADER_FMT = "<BQqqIq"   # type, sim_boot_ms, race_start, race_finish, active_gate, trailing

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=14550)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out", default="encap_capture.csv")
    a = ap.parse_args()
    conn = mavutil.mavlink_connection(f"udpin:{a.ip}:{a.port}")
    print(f"[cap] listening udpin:{a.ip}:{a.port} for {a.seconds}s (PASSIVE). Fly a lap NOW.")
    conn.wait_heartbeat(timeout=15)
    print("[cap] heartbeat ok")
    t0 = time.time(); n = 0; last_ag = None
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["t", "active_gate", "payload_hex"])
        while time.time() - t0 < a.seconds:
            m = conn.recv_match(type="ENCAPSULATED_DATA", blocking=True, timeout=1.0)
            if m is None:
                continue
            raw = bytes(m.data[:253])
            try:
                ag = struct.unpack_from(HEADER_FMT, raw)[4]
            except Exception:
                ag = -1
            t = time.time() - t0
            w.writerow([f"{t:.4f}", ag, raw.hex()])
            n += 1
            if ag != last_ag:
                print(f"[cap]   t={t:6.2f}s  active_gate -> {ag}")
                last_ag = ag
    print(f"[cap] done. {n} payloads -> {a.out}")

if __name__ == "__main__":
    main()
