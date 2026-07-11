#!/usr/bin/env python3
"""De-risk telemetry listener for the DCL sim — READ ONLY, sends nothing.

Binds passively to the MAVLink stream (udpin:0.0.0.0:14550) and, optionally, the
vision stream (UDP 5600), listens for a fixed window with the drone sitting at
spawn, and reports:

  1. Message-type mix + per-type Hz. Is ATTITUDE actually streaming? Do
     LOCAL_POSITION_NED / ODOMETRY appear?
  2. HIGHRES_IMU rate, time_usec monotonicity (delta mean/std), field population.
  3. Rest attitude: mean accel, |a|, implied pitch = atan2(xacc, -zacc).
  4. zacc sign at rest (must be NEGATIVE for gravity = -accel).
  + Optional: reassemble one vision frame, save PNG, report resolution.

Uses `udpin:` (bind + listen only). It never calls any *_send / request_* /
wait_heartbeat-with-send path, so nothing is transmitted to the sim.

    python tools/imu_check.py [--seconds 25] [--no-vision]
"""
import argparse
import math
import socket
import struct
import sys
import time
from collections import Counter

MAV_ADDR = "udpin:0.0.0.0:14550"
VISION_PORT = 5600
VISION_HEADER = struct.Struct("<IHHIIQ")  # frame_id u32, chunk_id u16, total_chunks u16,
#                                            jpeg_size u32, payload_size u32, sim_time_ns u64
assert VISION_HEADER.size == 24


def listen_mavlink(seconds):
    from pymavlink import mavutil

    print(f"[mavlink] binding {MAV_ADDR} (passive, no TX) for {seconds}s ...")
    conn = mavutil.mavlink_connection(MAV_ADDR)

    counts = Counter()
    hr_t = []          # HIGHRES_IMU time_usec
    hr_acc = []        # (xacc, yacc, zacc)
    hr_gyro = []       # (xgyro, ygyro, zgyro)
    hr_fields_seen = Counter()

    start = time.time()
    first_pkt = None
    while time.time() - start < seconds:
        msg = conn.recv_match(blocking=True, timeout=1.0)
        if msg is None:
            # No traffic. If we've heard nothing at all after 5s, the stream is quiet.
            if first_pkt is None and time.time() - start > 5.0:
                return None
            continue
        mtype = msg.get_type()
        if mtype == "BAD_DATA":
            continue
        if first_pkt is None:
            first_pkt = time.time()
        counts[mtype] += 1
        if mtype == "HIGHRES_IMU":
            hr_t.append(msg.time_usec)
            hr_acc.append((msg.xacc, msg.yacc, msg.zacc))
            hr_gyro.append((msg.xgyro, msg.ygyro, msg.zgyro))
            for f in ("xacc", "yacc", "zacc", "xgyro", "ygyro", "zgyro"):
                if getattr(msg, f) != 0.0:
                    hr_fields_seen[f] += 1

    elapsed = time.time() - start
    return dict(elapsed=elapsed, counts=counts, hr_t=hr_t, hr_acc=hr_acc,
                hr_gyro=hr_gyro, hr_fields_seen=hr_fields_seen)


def report_mavlink(r):
    print("\n================ MAVLINK TELEMETRY ================")
    elapsed = r["elapsed"]
    counts = r["counts"]
    total = sum(counts.values())
    print(f"window: {elapsed:.1f}s   total messages: {total}\n")

    print(f"{'MESSAGE TYPE':<26}{'COUNT':>8}{'Hz':>10}")
    print("-" * 44)
    for mtype, c in counts.most_common():
        print(f"{mtype:<26}{c:>8}{c/elapsed:>10.1f}")

    # a. Key availability questions
    print("\n-- availability --")
    for key in ("ATTITUDE", "LOCAL_POSITION_NED", "ODOMETRY", "ATTITUDE_QUATERNION"):
        present = counts.get(key, 0)
        print(f"  {key:<22} {'PRESENT ('+str(present)+')' if present else 'ABSENT'}")

    # b. HIGHRES_IMU
    hr_t = r["hr_t"]
    print("\n-- HIGHRES_IMU --")
    if not hr_t:
        print("  NO HIGHRES_IMU RECEIVED")
        return None
    n = len(hr_t)
    print(f"  count={n}  rate={n/elapsed:.1f} Hz  (expect ~115 Hz)")
    deltas = [hr_t[i+1] - hr_t[i] for i in range(n - 1)]
    monotonic = all(d > 0 for d in deltas)
    if deltas:
        mean_d = sum(deltas) / len(deltas)
        std_d = (sum((d - mean_d) ** 2 for d in deltas) / len(deltas)) ** 0.5
        print(f"  time_usec monotonic: {monotonic}   delta mean={mean_d:.1f}us std={std_d:.1f}us "
              f"(=> {1e6/mean_d:.1f} Hz)")
    fs = r["hr_fields_seen"]
    print(f"  fields nonzero over window: " +
          ", ".join(f"{f}={fs.get(f,0)}/{n}" for f in ("xacc","yacc","zacc","xgyro","ygyro","zgyro")))

    # c. rest attitude
    acc = r["hr_acc"]
    mean_ax = sum(a[0] for a in acc) / n
    mean_ay = sum(a[1] for a in acc) / n
    mean_az = sum(a[2] for a in acc) / n
    amag = math.sqrt(mean_ax**2 + mean_ay**2 + mean_az**2)
    pitch_deg = math.degrees(math.atan2(mean_ax, -mean_az))
    print("\n-- rest attitude (mean accel over window) --")
    print(f"  mean accel = [{mean_ax:+.3f} {mean_ay:+.3f} {mean_az:+.3f}] m/s^2")
    print(f"  |a| = {amag:.3f} m/s^2  (expect ~9.81)")
    print(f"  implied pitch = atan2(xacc, -zacc) = {pitch_deg:+.2f} deg")

    # d. zacc sign
    print("\n-- zacc sign at rest --")
    if mean_az < 0:
        print(f"  zacc mean = {mean_az:+.3f}  -> NEGATIVE (OK for gravity = -accel convention)")
        sign_ok = True
    else:
        print(f"  *** zacc mean = {mean_az:+.3f}  -> POSITIVE ***")
        print("  *** FLAG: attitude_filter assumes gravity = -accel (zacc<0 at rest). ***")
        print("  *** A positive zacc at rest is a SIGN BUG waiting to happen. ***")
        sign_ok = False

    return dict(pitch_deg=pitch_deg, amag=amag, mean_az=mean_az, sign_ok=sign_ok,
                rate=n/elapsed, monotonic=monotonic)


def listen_vision(seconds, out_png):
    print(f"\n[vision] binding 0.0.0.0:{VISION_PORT} for up to {seconds}s ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", VISION_PORT))
    sock.settimeout(1.0)

    frames = {}          # frame_id -> {chunk_id: payload}
    meta = {}            # frame_id -> (total_chunks, jpeg_size)
    got_jpeg = None
    got_meta = None
    start = time.time()
    pkts = 0
    while time.time() - start < seconds and got_jpeg is None:
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            if pkts == 0 and time.time() - start > 4.0:
                break
            continue
        except OSError:
            continue
        if len(data) < VISION_HEADER.size:
            continue
        pkts += 1
        frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_ns = \
            VISION_HEADER.unpack_from(data, 0)
        payload = data[VISION_HEADER.size:VISION_HEADER.size + payload_size]
        frames.setdefault(frame_id, {})[chunk_id] = payload
        meta[frame_id] = (total_chunks, jpeg_size)
        if len(frames[frame_id]) == total_chunks:
            blob = b"".join(frames[frame_id][i] for i in sorted(frames[frame_id]))
            jpeg = blob[:jpeg_size]
            if jpeg[:2] == b"\xff\xd8":       # JPEG SOI
                got_jpeg = jpeg
                got_meta = (frame_id, total_chunks, jpeg_size, sim_ns)
    sock.close()

    print(f"[vision] packets received: {pkts}")
    if got_jpeg is None:
        if pkts == 0:
            print("[vision] no packets on 5600 — vision stream not detected.")
        else:
            print("[vision] packets seen but no full frame reassembled in window.")
        return None

    from PIL import Image
    import io
    img = Image.open(io.BytesIO(got_jpeg))
    img.save(out_png)
    fid, tc, js, sim_ns = got_meta
    print(f"[vision] reassembled frame_id={fid} ({tc} chunks, {js} B jpeg) -> {out_png}")
    print(f"[vision] resolution: {img.size[0]}x{img.size[1]}  (expect 640x360)")
    return dict(size=img.size, png=out_png, frame_id=fid)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=int, default=25, help="mavlink listen window")
    ap.add_argument("--vision-seconds", type=int, default=10, help="vision listen window")
    ap.add_argument("--no-vision", action="store_true", help="skip the vision listener")
    ap.add_argument("--png", default="vision_frame.png", help="output PNG path for the reassembled frame")
    args = ap.parse_args()

    r = listen_mavlink(args.seconds)
    if r is None:
        print("\n*** STREAM QUIET: no MAVLink packets on 14550 within 5s. ***")
        print("*** Sim likely closed/timed out — relaunch and retry. Stopping. ***")
        sys.exit(2)
    report_mavlink(r)

    if not args.no_vision:
        listen_vision(args.vision_seconds, args.png)

    print("\n[done] read-only session complete; nothing was sent to the sim.")


if __name__ == "__main__":
    main()
