#!/usr/bin/env python3
"""Cruise-speed calibration probe: fly FIXED commands straight, time the gate ticks.

Everything in the motion schedule times off cruise speed. This measures it the only
way v3385 allows (no position telemetry): fly the 0.40-probe way -- fixed thrust,
zero body rates, riding the spawn tilt -- and record the wall-clock when the sim's
active_gate ticks past each gate. Each tick + the known spawn->gate distance gives a
leg speed and a running average.

    python tools/probe_cruise.py                 # thrust 0.29, ride spawn tilt
    python tools/probe_cruise.py --thrust 0.31   # if it sinks too fast / too slow

Run ON THE SIM BOX during a real race. Read-only telemetry; the only things sent are
ARM, TIMESYNC, and the fixed SET_ATTITUDE_TARGET under test. Nothing steers.
"""
import argparse
import math
import os
import struct
import sys
import threading
import time

from pymavlink import mavutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vq1_vision_servo import gravity_to_roll_pitch  # noqa: E402
from tools.motion_schedule import segments          # noqa: E402  (cumulative gate distances)


def cumulative_distances():
    """spawn->gate[i] 3D distance, cumulative, for i=0..5."""
    cum, run = [], 0.0
    for s in segments():
        run += s["d3d"]
        cum.append(run)
    return cum   # cum[i] = distance from spawn to gate i


class RX(threading.Thread):
    def __init__(self, conn):
        super().__init__(daemon=True)
        self.conn = conn
        self.lock = threading.Lock()
        self.running = True
        self.active_gate = -1
        self.race_started = False
        self.countdown_armed = False
        self.armed_race_start_ms = -1
        self.go_wall = None
        self.gate_ticks = []          # (active_gate_value, t_since_go)
        self.pitch_deg = 0.0

    def run(self):
        STALE, MARGIN = -10000, 100
        while self.running:
            try:
                msg = self.conn.recv_match(blocking=False)
            except (ConnectionResetError, OSError):
                break
            if msg is None:
                time.sleep(0.001); continue
            t = msg.get_type()
            if t == "HIGHRES_IMU":
                # gravity vector = NEGATED specific force (zacc ~ -9.81 at rest); feeding
                # raw +accel makes pitch 180 deg off. Match run_vq1's GravityEstimator sign.
                _, pitch = gravity_to_roll_pitch((-msg.xacc, -msg.yacc, -msg.zacc))
                with self.lock:
                    self.pitch_deg = math.degrees(pitch)
            elif t == "ENCAPSULATED_DATA":
                raw = bytes(msg.data)
                if not raw or raw[0] != 1:
                    continue
                try:
                    _, sim_boot, race_start, _f, ag, _ = struct.unpack_from("<BQqqIq", raw)
                except struct.error:
                    continue
                with self.lock:
                    delta = race_start - sim_boot
                    if race_start >= 0 and delta < STALE:
                        continue
                    if (not self.countdown_armed) and race_start >= 0 and delta > 0:
                        self.countdown_armed = True
                        self.armed_race_start_ms = race_start
                    if (self.countdown_armed and not self.race_started
                            and sim_boot >= self.armed_race_start_ms + MARGIN):
                        self.race_started = True
                        self.go_wall = time.time()
                    if int(ag) != self.active_gate:
                        self.active_gate = int(ag)
                        if self.race_started:
                            self.gate_ticks.append((int(ag), time.time() - self.go_wall))

    def stop(self):
        self.running = False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=14550)
    ap.add_argument("--thrust", type=float, default=0.29)
    ap.add_argument("--max-s", type=float, default=30.0, help="max flight seconds after GO")
    ap.add_argument("--go-timeout", type=float, default=180.0)
    args = ap.parse_args()

    cum = cumulative_distances()
    print(f"[cruise] fixed thrust={args.thrust}, zero rates (ride spawn tilt). "
          f"spawn->gate cumulative dist (m): {[round(c, 1) for c in cum]}")

    conn = mavutil.mavlink_connection(f"udpin:{args.ip}:{args.port}")
    print("[cruise] waiting for heartbeat ...")
    conn.wait_heartbeat()
    print(f"[cruise] heartbeat: sys {conn.target_system}")

    stop = threading.Event()

    def _timesync():
        while not stop.is_set():
            conn.mav.timesync_send(int(time.time_ns()), 0)
            time.sleep(0.1)
    threading.Thread(target=_timesync, daemon=True).start()
    rx = RX(conn); rx.start()

    boot0 = int(time.time() * 1000)
    conn.mav.command_long_send(conn.target_system, conn.target_component,
                               mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                               0, 1, 0, 0, 0, 0, 0, 0)
    print("[cruise] ARM sent; waiting for GO ...")
    t0 = time.time()
    while not rx.race_started:
        if time.time() - t0 > args.go_timeout:
            print("[cruise] no GO within timeout."); stop.set(); rx.stop(); return 2
        time.sleep(0.02)
    print("[cruise] GO -- flying fixed command straight.")

    last_report = 0.0
    while time.time() - rx.go_wall < args.max_s:
        conn.mav.set_attitude_target_send(
            int(time.time() * 1000) - boot0, conn.target_system, conn.target_component,
            128, [1.0, 0.0, 0.0, 0.0], 0.0, 0.0, 0.0, args.thrust)
        now = time.time()
        if now - last_report >= 1.0:
            last_report = now
            with rx.lock:
                print(f"[cruise] t={now-rx.go_wall:4.1f}s active_gate={rx.active_gate} "
                      f"pitch~{rx.pitch_deg:+.1f}deg")
        time.sleep(1.0 / 250.0)
        if rx.active_gate >= len(cum):    # passed FINISH
            break

    # graceful throttle-0 + disarm
    for _ in range(5):
        conn.mav.set_attitude_target_send(0, conn.target_system, conn.target_component,
                                          128, [1.0, 0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0)
        time.sleep(0.02)
    conn.mav.command_long_send(conn.target_system, conn.target_component,
                               mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                               0, 0, 0, 0, 0, 0, 0, 0)
    stop.set(); rx.stop()

    print("\n================= CRUISE PROBE RESULT =================")
    ticks = rx.gate_ticks
    if not ticks:
        print("active_gate never advanced -- it didn't pass gate 0. Try adjusting --thrust,")
        print("or the fixed spawn tilt isn't aimed well enough (needs the schedule's turns).")
        print("======================================================")
        return 0
    print("tick  active_gate  passed_gate  t_since_go   cum_dist   avg_speed   leg_speed")
    prev_t, prev_d = 0.0, 0.0
    for (ag, t) in ticks:
        passed = ag - 1                       # active_gate->ag means gate (ag-1) was passed
        if passed < 0 or passed >= len(cum):
            continue
        d = cum[passed]
        avg = d / t if t > 1e-6 else float("nan")
        leg = (d - prev_d) / (t - prev_t) if (t - prev_t) > 1e-6 else float("nan")
        print(f"        {ag:>2}          {passed:>2}       {t:7.2f}s   {d:7.1f}m   "
              f"{avg:7.2f}    {leg:7.2f} m/s")
        prev_t, prev_d = t, d
    last_ag, last_t = ticks[-1]
    passed = last_ag - 1
    if 0 <= passed < len(cum):
        print(f"\n>>> CRUISE ESTIMATE ~ {cum[passed]/last_t:.1f} m/s "
              f"(over {passed+1} gate(s), {cum[passed]:.0f} m in {last_t:.1f}s)")
        print(f">>> rebuild the schedule: python tools/motion_schedule.py --cruise "
              f"{cum[passed]/last_t:.0f}")
    print("======================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
