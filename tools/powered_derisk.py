#!/usr/bin/env python3
"""POWERED de-risk session — dev run, NOT a submitted timed attempt.

Reuses the verified run_vq1 control path (arm -> TIMESYNC -> wait GO ->
SET_ATTITUDE_TARGET type_mask=128 body-rate command) to command ONE brief,
gentle PURE-PITCH rotation while recording HIGHRES_IMU throughout, then analyzes:

  1. Does gyro go NONZERO under motion? (resolves the 🔴-if-confirmed flag)
  2. During pure-pitch rotation, does accel stay pure-pitch (yacc ~ rest)?
     -> observation-spec's A-vs-B in-motion discriminator.
  3. Vision frames at rest + mid-pitch (feeds the FPV tilt-sign question).
  4. time_usec delta histogram: fraction <= 0 (how often A3's dt fallback fires).

Maneuver is bounded: a short settle, a <=2s pitch-rate burst at a small rate and
modest thrust, a stop, then DISARM. Safe by construction; --no-control does a
pure listen (no arm, no TX) for a dry check.

    python tools/powered_derisk.py [--pitch-rate 0.3] [--thrust 0.30]
                                   [--burst 1.5] [--go-timeout 60] [--no-control]
"""
import argparse
import struct
import sys
import threading
import time

import numpy as np
from pymavlink import mavutil

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from dcl_mavlink_adapter import DCLTimesync
from dcl_vision_receiver import DCLVisionReceiver

# --- shared state ---
_lock = threading.Lock()
_imu = []                       # (t_wall, time_usec, ax, ay, az, gx, gy, gz)
_running = True
_race_started = threading.Event()
_armed = {"armed": False, "race_start": -1}
STALE_DELTA_MS = -10000
START_MARGIN_MS = 100


def rx_loop(conn):
    global _running
    while _running:
        try:
            msg = conn.recv_match(blocking=False)
        except (ConnectionResetError, OSError):
            break
        if msg is None:
            time.sleep(0.0005)
            continue
        t = msg.get_type()
        if t == "HIGHRES_IMU":
            with _lock:
                _imu.append((time.time(), getattr(msg, "time_usec", None),
                             msg.xacc, msg.yacc, msg.zacc,
                             msg.xgyro, msg.ygyro, msg.zgyro))
        elif t == "ENCAPSULATED_DATA":
            raw = bytes(msg.data)
            if raw and raw[0] == 1:
                try:
                    _, sim_boot, rstart, rfin, gate, _ = struct.unpack_from("<BQqqIq", raw)
                except struct.error:
                    continue
                delta = rstart - sim_boot
                is_stale = rstart >= 0 and delta < STALE_DELTA_MS
                if is_stale:
                    continue
                if not _armed["armed"] and rstart >= 0 and delta > 0:
                    _armed["armed"] = True
                    _armed["race_start"] = rstart
                    print(f"[race] countdown armed: race_start={rstart} is {delta}ms ahead")
                if (_armed["armed"] and not _race_started.is_set()
                        and sim_boot >= _armed["race_start"] + START_MARGIN_MS):
                    _race_started.set()
                    print(f"[race] GO (sim_boot={sim_boot} >= race_start+{START_MARGIN_MS})")


def window(t0, t1):
    """IMU rows with wall time in [t0, t1]."""
    with _lock:
        return [r for r in _imu if t0 <= r[0] <= t1]


def stats_block(rows, label):
    if not rows:
        print(f"  [{label}] no samples"); return None
    A = np.array([(r[2], r[3], r[4]) for r in rows], float)
    G = np.array([(r[5], r[6], r[7]) for r in rows], float)
    print(f"  [{label}] n={len(rows)}")
    print(f"    accel mean=[{A[:,0].mean():+.3f} {A[:,1].mean():+.3f} {A[:,2].mean():+.3f}] "
          f"|a|={np.linalg.norm(A.mean(0)):.3f}")
    print(f"    gyro  mean=[{G[:,0].mean():+.4f} {G[:,1].mean():+.4f} {G[:,2].mean():+.4f}]  "
          f"max|g|=[{np.abs(G[:,0]).max():.4f} {np.abs(G[:,1]).max():.4f} {np.abs(G[:,2]).max():.4f}]")
    return dict(accel=A, gyro=G)


def main():
    global _running
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=14550)
    ap.add_argument("--vision-port", type=int, default=5600)
    ap.add_argument("--pitch-rate", type=float, default=0.30, help="rad/s body pitch rate (gentle)")
    ap.add_argument("--thrust", type=float, default=0.30, help="collective thrust [0,1] during maneuver")
    ap.add_argument("--burst", type=float, default=1.5, help="pitch-burst duration s (<=2)")
    ap.add_argument("--go-timeout", type=float, default=60.0)
    ap.add_argument("--force-no-go", action="store_true",
                    help="if GO never fires, command the maneuver anyway (dev de-risk only)")
    ap.add_argument("--no-control", action="store_true", help="listen only; no arm, no TX")
    ap.add_argument("--png-prefix", default="derisk")
    args = ap.parse_args()
    args.burst = min(args.burst, 2.0)

    print(f"[conn] udpin:0.0.0.0:{args.port} — waiting for sim heartbeat ...")
    conn = mavutil.mavlink_connection(f"udpin:0.0.0.0:{args.port}")
    hb = conn.wait_heartbeat(timeout=15)
    if hb is None:
        print("*** No heartbeat in 15s — sim not streaming. Relaunch and retry. ***")
        sys.exit(2)
    tsys, tcomp = conn.target_system, conn.target_component
    print(f"[conn] heartbeat: system={tsys} component={tcomp}")

    vis = DCLVisionReceiver(port=args.vision_port)
    vis.start()
    rx = threading.Thread(target=rx_loop, args=(conn,), daemon=True)
    rx.start()

    timesync = None
    frames = {}
    maneuver_ran = False
    burst_t0 = burst_t1 = None
    try:
        if not args.no_control:
            conn.mav.command_long_send(tsys, tcomp,
                                       mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                       0, 1, 0, 0, 0, 0, 0, 0)
            print("[tx] ARM sent")
            timesync = DCLTimesync(conn)
            timesync.start()

        # baseline at rest
        rest_t0 = time.time()
        time.sleep(3.0)
        rest_t1 = time.time()
        try:
            if vis.frames_received:
                frames["rest"] = vis.get_latest_frame()
        except Exception:
            pass

        if args.no_control:
            print("[no-control] listen-only; skipping GO wait + maneuver.")
        else:
            print(f"[race] waiting for GO (timeout {args.go_timeout}s) ...")
            go = _race_started.wait(timeout=args.go_timeout)
            if not go and not args.force_no_go:
                print("*** No GO within timeout — not commanding without a race. Disarming. ***")
                print("*** (pass --force-no-go to command the maneuver anyway for the dev de-risk.) ***")
                conn.mav.command_long_send(tsys, tcomp,
                                           mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                           0, 0, 0, 0, 0, 0, 0, 0)
            else:
                if not go:
                    print("[force-no-go] GO never fired — commanding the maneuver anyway "
                          "(authorized dev de-risk, NOT a timed attempt).")
                # Re-ARM at GO in case the race reset disarmed us, so control lands.
                conn.mav.command_long_send(tsys, tcomp,
                                           mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                           0, 1, 0, 0, 0, 0, 0, 0)
                boot_ms = int(time.time() * 1000)

                def send(roll, pitch, yaw, thr):
                    conn.mav.set_attitude_target_send(
                        int(time.time() * 1000) - boot_ms, tsys, tcomp,
                        128, [1, 0, 0, 0], roll, pitch, yaw, thr)

                def phase(dur, roll, pitch, yaw, thr, tag, grab_png=None):
                    print(f"[maneuver] {tag}: {dur:.2f}s rpy_rate=[{roll} {pitch} {yaw}] thr={thr}")
                    t_end = time.time() + dur
                    grabbed = False
                    while time.time() < t_end:
                        send(roll, pitch, yaw, thr)
                        if grab_png and not grabbed and time.time() > t_end - dur / 2:
                            try:
                                if vis.frames_received:
                                    frames[grab_png] = vis.get_latest_frame()
                            except Exception:
                                pass
                            grabbed = True
                        time.sleep(0.02)  # 50 Hz

                man_t0 = time.time()
                phase(0.5, 0, 0, 0, args.thrust, "settle")
                burst_t0 = time.time()
                phase(args.burst, 0, args.pitch_rate, 0, args.thrust, "PITCH burst", grab_png="pitch")
                burst_t1 = time.time()
                phase(0.5, 0, 0, 0, args.thrust, "stop")
                phase(0.3, 0, 0, 0, 0.0, "idle")
                man_t1 = time.time()
                maneuver_ran = True

                conn.mav.command_long_send(tsys, tcomp,
                                           mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                           0, 0, 0, 0, 0, 0, 0, 0)
                print("[tx] DISARM sent")
    finally:
        _running = False
        time.sleep(0.2)
        if timesync:
            timesync.stop()
        vis.stop()
        conn.close()

    # ---- analysis ----
    print("\n================ POWERED DE-RISK ANALYSIS ================")
    with _lock:
        n_total = len(_imu)
    print(f"HIGHRES_IMU samples captured: {n_total}")
    if n_total == 0:
        print("no IMU — nothing to analyze."); return

    rest_rows = window(rest_t0, rest_t1)
    print("\n-- REST baseline --")
    rest = stats_block(rest_rows, "rest")

    burst = None
    if maneuver_ran and burst_t0 is not None:
        burst_rows = window(burst_t0, burst_t1)
        print("\n-- PITCH BURST --")
        burst = stats_block(burst_rows, "burst")

        # Q1: gyro nonzero under motion?
        if burst:
            gymax = np.abs(burst["gyro"]).max()
            print("\n[Q1] gyro NONZERO under motion? "
                  f"max|gyro| over burst = {gymax:.4f} rad/s -> "
                  f"{'YES — channel is LIVE' if gymax > 1e-3 else '*** NO — gyro STUCK AT ZERO, channel DEAD ***'}")
            # Q2: pure-pitch? ygyro should dominate, yacc should stay ~rest.
            gy = burst["gyro"]
            dom = "pitch(y)" if np.abs(gy[:, 1]).mean() >= max(np.abs(gy[:, 0]).mean(), np.abs(gy[:, 2]).mean()) else "NOT pitch"
            yacc_rest = rest["accel"][:, 1].mean() if rest else float("nan")
            yacc_burst_dev = np.abs(burst["accel"][:, 1] - yacc_rest).max()
            print(f"[Q2] rotation axis dominant = {dom} "
                  f"(mean|g|=[{np.abs(gy[:,0]).mean():.4f} {np.abs(gy[:,1]).mean():.4f} {np.abs(gy[:,2]).mean():.4f}])")
            print(f"     accel pure-pitch? max |yacc - yacc_rest| over burst = {yacc_burst_dev:.3f} m/s^2 "
                  f"(small => stays pure-pitch => favors hypothesis A body-attitude)")

    # Q4: time_usec delta histogram over the WHOLE session
    with _lock:
        usec = [r[1] for r in _imu if r[1] is not None]
    print("\n-- time_usec delta histogram (whole session) --")
    if len(usec) > 1:
        d = np.diff(np.array(usec, dtype=np.int64))
        eq0 = int((d == 0).sum())
        lt0 = int((d < 0).sum())
        le0 = eq0 + lt0
        gap = int((d > 1_000_000).sum())
        frac_bad = (le0 + gap) / len(d)
        print(f"  deltas n={len(d)}  mean={d.mean():.1f}us std={d.std():.1f}us")
        print(f"  ==0 (dup): {eq0} ({eq0/len(d)*100:.2f}%)   <0 (reorder): {lt0} ({lt0/len(d)*100:.2f}%)"
              f"   >1s: {gap} ({gap/len(d)*100:.2f}%)")
        print(f"  A3 dt-fallback fraction = {frac_bad*100:.2f}%  "
              f"-> {'>1%: bump warn-once to COUNTED warning' if frac_bad > 0.01 else '<=1%: warn-once is fine'}")
        for lo, hi in [(-10**9, 0), (0, 4000), (4000, 8000), (8000, 12000), (12000, 20000), (20000, 10**9)]:
            c = int(((d > lo) & (d <= hi)).sum())
            print(f"    ({lo:>11}, {hi:>11}] : {c:>5} ({c/len(d)*100:5.1f}%)")

    # save raw IMU log (so re-analysis never needs a re-fly)
    import os
    outdir = os.environ.get("DERISK_OUT", ".")
    csv_path = os.path.join(outdir, f"{args.png_prefix}_imu.csv")
    with open(csv_path, "w", newline="") as fh:
        fh.write("t_wall,time_usec,xacc,yacc,zacc,xgyro,ygyro,zgyro\n")
        with _lock:
            for r in _imu:
                fh.write(",".join("" if v is None else repr(v) for v in r) + "\n")
    marks = f"rest=[{rest_t0},{rest_t1}] burst=[{burst_t0},{burst_t1}]"
    print(f"[log] {n_total} IMU rows -> {csv_path}   windows: {marks}")

    # save vision frames
    from PIL import Image
    for tag, fr in frames.items():
        p = os.path.join(outdir, f"{args.png_prefix}_{tag}.png")
        Image.fromarray(fr).save(p)
        print(f"[vision] saved {tag} frame {fr.shape[1]}x{fr.shape[0]} -> {p}")

    print("\n[done] powered de-risk complete; drone disarmed.")


if __name__ == "__main__":
    main()
