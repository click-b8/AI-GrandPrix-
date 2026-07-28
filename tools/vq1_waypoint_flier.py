#!/usr/bin/env python3
"""Hardcoded position-waypoint qualifying flier for DCL sim v3385.

Position control is CONFIRMED working (tools/probe_position_exact.py: a D=-3
LOCAL_NED setpoint climbed the drone, motors 0.13). This flies the 6 known gates
as LOCAL_NED position setpoints and lets the SIM's own controller fly to them.
No vision, no reactive attitude loop, no flip risk.

Init sequence is IDENTICAL to probe_position_exact -- the sequence that unlocked
position control (our old probe skipped it and was ignored):
    udpin -> wait_heartbeat -> TIMESYNC 10 Hz -> ARM -> stream setpoints @ 250 Hz

Sequencer = the sim. Every tick the setpoint is gates_ned[clamp(active_gate,0,5)];
when the drone passes a gate the sim increments active_gate and the setpoint
auto-advances. Done when active_gate > 5 or race_finish is set.

Frame (see nav_frames.py):
  origin = spawn, D = -dZ_ue          -- BOTH confirmed live by the climb probe.
  horizontal 'world' (N=+dX_ue, E=+dY_ue) -- the hypothesis consistent with the
     DGX right-of-nose gate bearing (facing South to reach N<0, gate at E<0 = West
     = the drone's right). Safety net: if active_gate hasn't left 0 within
     --resolve-s, flip N once and retry (covers a wrong horizontal-sign guess).

NOTE: v3385 publishes NO position telemetry, so we log the COMMANDED setpoint +
active_gate (the sim's gate trigger IS the arrival signal), not a measured pose.

    python tools/vq1_waypoint_flier.py                 # fly the race (world axes)
    python tools/vq1_waypoint_flier.py --axes world_negN --no-auto-resolve
    python tools/vq1_waypoint_flier.py --no-go-wait     # stream immediately (debug)
"""
import argparse
import struct
import sys
import threading
import time

from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavc

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav_frames import gate_ned_setpoints  # noqa: E402

# position-only mask: x/y/z ACTIVE, velocity+accel+yaw ignored (== probe_position_exact POS_MASK)
POS_MASK = (mavc.POSITION_TARGET_TYPEMASK_VX_IGNORE | mavc.POSITION_TARGET_TYPEMASK_VY_IGNORE
            | mavc.POSITION_TARGET_TYPEMASK_VZ_IGNORE | mavc.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavc.POSITION_TARGET_TYPEMASK_AY_IGNORE | mavc.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavc.POSITION_TARGET_TYPEMASK_YAW_IGNORE | mavc.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE)

N_GATES = 6  # START..FINISH, indices 0..5


class RX:
    """Background receiver: race status (active_gate/GO/finish), motors, IMU, collisions."""
    def __init__(self, conn):
        self.conn = conn
        self.lock = threading.Lock()
        self.running = True
        self.active_gate = -1
        self.race_started = False
        self.countdown_armed = False
        self.armed_race_start_ms = -1
        self.race_finish_ns = 0
        self.go_wall_t = None
        self.motor_mag = 0.0
        self.gyro_mag = 0.0
        self.collisions = []   # (wall_t, id, threat_level, impulse)

    def start(self):
        threading.Thread(target=self._loop, daemon=True, name="FlierRX").start()

    def stop(self):
        self.running = False

    def _loop(self):
        STALE_DELTA_MS, START_MARGIN_MS = -10000, 100
        while self.running:
            try:
                msg = self.conn.recv_match(blocking=False)
            except (ConnectionResetError, OSError):
                break
            if msg is None:
                time.sleep(0.001)
                continue
            t = msg.get_type()
            if t == "BAD_DATA":
                continue
            if t == "ACTUATOR_OUTPUT_STATUS":
                vals = list(getattr(msg, "actuator", []) or [])
                n = int(getattr(msg, "active", 0)) or 0
                used = vals[:n] if n else [v for v in vals if v != 0.0]
                if used:
                    with self.lock:
                        self.motor_mag = sum(abs(v) for v in used) / len(used)
            elif t == "HIGHRES_IMU":
                g = (msg.xgyro**2 + msg.ygyro**2 + msg.zgyro**2) ** 0.5
                with self.lock:
                    self.gyro_mag = float(g)
            elif t == "COLLISION":
                with self.lock:
                    self.collisions.append((time.time(), int(getattr(msg, "id", 0)),
                                            int(getattr(msg, "threat_level", 0)),
                                            float(getattr(msg, "horizontal_minimum_delta", 0.0))))
            elif t == "ENCAPSULATED_DATA":
                self._race(msg, STALE_DELTA_MS, START_MARGIN_MS)

    def _race(self, msg, STALE_DELTA_MS, START_MARGIN_MS):
        raw = bytes(msg.data)
        if not raw or raw[0] != 1:
            return
        try:
            _, sim_boot_ms, race_start_ms, fin_ns, active_gate, _ = struct.unpack_from("<BQqqIq", raw)
        except struct.error:
            return
        with self.lock:
            self.active_gate = int(active_gate)
            if fin_ns > 0:
                self.race_finish_ns = int(fin_ns)
            delta = race_start_ms - sim_boot_ms
            if race_start_ms >= 0 and delta < STALE_DELTA_MS:
                return
            if (not self.countdown_armed) and race_start_ms >= 0 and delta > 0:
                self.countdown_armed = True
                self.armed_race_start_ms = race_start_ms
            if (self.countdown_armed and not self.race_started
                    and sim_boot_ms >= self.armed_race_start_ms + START_MARGIN_MS):
                self.race_started = True
                self.go_wall_t = time.time()

    def snapshot(self):
        with self.lock:
            return (self.active_gate, self.race_started, self.race_finish_ns,
                    self.motor_mag, self.gyro_mag, list(self.collisions))


def arm(conn):
    conn.mav.command_long_send(conn.target_system, conn.target_component,
                               mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                               0, 1, 0, 0, 0, 0, 0, 0)


def disarm(conn):
    conn.mav.command_long_send(conn.target_system, conn.target_component,
                               mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                               0, 0, 0, 0, 0, 0, 0, 0)


def timesync_thread(conn, stop):
    while not stop.is_set():
        conn.mav.timesync_send(int(time.time_ns()), 0)
        time.sleep(1.0 / 10.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=14550)
    ap.add_argument("--axes", default="world", choices=["world", "world_negN", "spawn_fwd"])
    ap.add_argument("--rate", type=float, default=250.0, help="setpoint Hz (sample = 250)")
    ap.add_argument("--resolve-s", type=float, default=10.0,
                    help="if active_gate stays 0 this long after GO, flip N once (safety net)")
    ap.add_argument("--no-auto-resolve", action="store_true", help="disable the N-sign flip")
    ap.add_argument("--no-go-wait", action="store_true", help="stream immediately, don't wait for GO")
    ap.add_argument("--go-timeout", type=float, default=180.0)
    ap.add_argument("--max-flight-s", type=float, default=90.0, help="hard stop safety timeout")
    args = ap.parse_args()

    # --- waypoints (mutable so the auto-resolve can flip N in place) ---
    wps = [list(map(float, r["ned"])) for r in gate_ned_setpoints(axes=args.axes)]
    print(f"[flier] axes={args.axes}  {N_GATES} waypoints (N,E,D):")
    for i, (tag, ned) in enumerate((r["tag"], r["ned"]) for r in gate_ned_setpoints(axes=args.axes)):
        print(f"[flier]   g{i} {str(tag or ''):<7} N={ned[0]:+8.2f} E={ned[1]:+7.2f} D={ned[2]:+7.2f}")

    # --- EXACT proven init: connect -> heartbeat -> timesync -> arm ---
    print(f"[flier] connecting udpin:{args.ip}:{args.port} ...")
    conn = mavutil.mavlink_connection(f"udpin:{args.ip}:{args.port}")
    print("[flier] waiting for heartbeat ...")
    conn.wait_heartbeat()
    print(f"[flier] heartbeat: system {conn.target_system} component {conn.target_component}")

    stop = threading.Event()
    threading.Thread(target=timesync_thread, args=(conn, stop), daemon=True, name="Timesync").start()
    rx = RX(conn)
    rx.start()

    boot0_ms = int(time.time() * 1000)
    arm(conn)
    print("[flier] ARM sent")

    if not args.no_go_wait:
        print("[flier] waiting for race GO ...")
        t0 = time.time()
        while not rx.race_started:
            if time.time() - t0 > args.go_timeout:
                print("[flier] ERROR: no GO within timeout. Start a race, retry.")
                stop.set(); rx.stop(); disarm(conn); return 2
        ag, *_ = rx.snapshot()
        print(f"[flier] GO. active_gate={ag}")

    # --- control loop: setpoint = wps[clamp(active_gate,0,5)] @ rate ---
    dt = 1.0 / args.rate
    flight_t0 = time.time()
    go_t = rx.go_wall_t or flight_t0
    resolved = args.no_auto_resolve       # True => never flip
    flipped = False
    last_gate = -1
    last_log = 0.0
    rc = 0
    print(f"[flier] flying. setpoint = wps[active_gate] @ {args.rate}Hz "
          f"(auto-resolve={'off' if args.no_auto_resolve else f'{args.resolve_s}s'})")
    try:
        while True:
            ag, started, finish, motors, gyro, cols = rx.snapshot()

            # end conditions
            if finish > 0 or ag > N_GATES - 1:
                print(f"[flier] *** RACE COMPLETE *** active_gate={ag} finish_ns={finish}")
                break
            if time.time() - flight_t0 > args.max_flight_s:
                print(f"[flier] max-flight-s ({args.max_flight_s}s) hit -- stopping. active_gate={ag}")
                rc = 1
                break

            idx = max(0, min(ag, N_GATES - 1))
            n, e, d = wps[idx]
            conn.mav.set_position_target_local_ned_send(
                int(time.time() * 1000) - boot0_ms,
                conn.target_system, conn.target_component,
                mavc.MAV_FRAME_LOCAL_NED, POS_MASK,
                n, e, d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

            # gate advance -> lock the frame sign, log
            if ag != last_gate:
                if ag > 0:
                    resolved = True   # something advanced: current sign is correct
                print(f"[flier] >>> active_gate {last_gate} -> {ag}  (t={time.time()-go_t:5.1f}s "
                      f"since GO)  now targeting g{idx} N={n:+.1f} E={e:+.1f} D={d:+.1f}")
                last_gate = ag

            # N-sign auto-resolve: still on gate 0 after resolve_s -> flip N once
            if (not resolved) and (not flipped) and ag == 0 and (time.time() - go_t) > args.resolve_s:
                for w in wps:
                    w[0] = -w[0]
                flipped = True
                resolved = True
                print(f"[flier] !!! active_gate stuck at 0 for {args.resolve_s}s -- FLIPPING N sign. "
                      f"g0 now N={wps[0][0]:+.1f} E={wps[0][1]:+.1f} D={wps[0][2]:+.1f}")

            # collision surfacing
            if cols:
                for (ct, cid, lvl, imp) in cols[-3:]:
                    kind = {1001: "GATE", 1002: "ENVIRONMENT"}.get(cid, str(cid))
                    print(f"[flier] COLLISION {kind} level={lvl} impulse={imp:.2f} kg*m/s")
                with rx.lock:
                    rx.collisions.clear()

            now = time.time()
            if now - last_log >= 0.5:
                last_log = now
                print(f"[flier]  t={now-go_t:5.1f}s ag={ag} tgt=g{idx} "
                      f"NED=({n:+.1f},{e:+.1f},{d:+.1f}) motors={motors:.3f} gyro={gyro:.2f}")
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[flier] interrupted -- disarming.")
        rc = 130
    finally:
        stop.set(); rx.stop()
        disarm(conn)
        ag, *_ , cols = rx.snapshot()
        print(f"[flier] DISARM sent. final active_gate={ag}. "
              f"gates passed = {max(ag, 0)}/{N_GATES}.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
