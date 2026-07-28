#!/usr/bin/env python3
"""DECISIVE position-control test: replicate the OFFICIAL sample client EXACTLY.

Motivation
----------
The PyAIPilotExample v2 sample client contains a `set_position_target_local_ned`
call, which suggests position/velocity control should work. But our earlier probe
(tools/probe_position_target.py) sent position targets on v3385 twice and they were
IGNORED (motors ~0.013, no motion). This test removes every difference between our
probe and the sample so the result is unambiguous:

  IF the sample's EXACT sequence makes the drone move  -> position control works and
     our old probe was missing something (mask/rate).  -> build the waypoint flier.
  IF it is STILL ignored with the exact sequence       -> v3385 genuinely does not
     honour position/velocity control.                 -> stop chasing it for good.

What the sample client actually does (verified against controller.py / main.py /
setup.py / timesync.py / mavlink_rx.py / vision_rx.py) -- the ONLY things it ever
SENDS are:
    1. TIMESYNC  (timesync_send(time_ns, 0)) at 10 Hz from a background thread
    2. ARM       (MAV_CMD_COMPONENT_ARM_DISARM, param1=1)
    3. the setpoint (set_position_target_local_ned) at 250 Hz, continuously
It sets NO flight mode (no DO_SET_MODE / OFFBOARD / GUIDED) and sends NO
VISION_POSITION_ESTIMATE (VisionRX only RECEIVES JPEG frames). This test does the
same, in that order. `--vpe` optionally ALSO streams a dummy VISION_POSITION_ESTIMATE
-- that is BEYOND the sample (a separate "does the controller need a pose source?"
hypothesis), clearly labelled, off by default.

The sample's own `update_position_flight_control` is really a VELOCITY command (its
mask IGNORES x/y/z and commands vx=2). `--cmd velfwd` reproduces it byte-for-byte.
`--cmd climb` sends an actual POSITION target (D=-3, origin-agnostic climb) through
the identical machinery.

We have NO position/velocity telemetry on v3385, so "did it move" is read from what
we CAN see: ACTUATOR_OUTPUT_STATUS motor magnitude, HIGHRES_IMU accel/gyro,
COMMAND_ACK, STATUSTEXT, a POSITION_TARGET_LOCAL_NED echo, and active_gate.

Run ON THE SIM BOX, during a real race (default streams across the GO; use
--wait-go to only stream after GO, or --no-go to stream immediately like the raw
sample and never wait):

    python tools/probe_position_exact.py --cmd velfwd      # the sample EXACTLY
    python tools/probe_position_exact.py --cmd climb        # position target, same seq
    python tools/probe_position_exact.py --cmd climb --vpe  # + dummy pose (beyond sample)
"""
import argparse
import statistics
import struct
import sys
import threading
import time
from collections import deque

from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavc

# ---- the sample client's EXACT velocity mask (position IGNORED, velocity ACTIVE) --
VEL_MASK = (mavc.POSITION_TARGET_TYPEMASK_X_IGNORE | mavc.POSITION_TARGET_TYPEMASK_Y_IGNORE
            | mavc.POSITION_TARGET_TYPEMASK_Z_IGNORE | mavc.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavc.POSITION_TARGET_TYPEMASK_AY_IGNORE | mavc.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavc.POSITION_TARGET_TYPEMASK_YAW_IGNORE | mavc.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE)
# ---- position-only mask (position ACTIVE, velocity IGNORED) -----------------------
POS_MASK = (mavc.POSITION_TARGET_TYPEMASK_VX_IGNORE | mavc.POSITION_TARGET_TYPEMASK_VY_IGNORE
            | mavc.POSITION_TARGET_TYPEMASK_VZ_IGNORE | mavc.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavc.POSITION_TARGET_TYPEMASK_AY_IGNORE | mavc.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavc.POSITION_TARGET_TYPEMASK_YAW_IGNORE | mavc.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE)


class RX:
    """Background receiver -- phase-tagged motor/IMU + race status + acks/echoes."""
    def __init__(self, conn):
        self.conn = conn
        self.lock = threading.Lock()
        self.running = True
        self.phase = "init"
        self.act = {"baseline": [], "stream": []}
        self.acc = {"baseline": [], "stream": []}
        self.gyro = {"baseline": [], "stream": []}
        self.statustexts = []
        self.command_acks = []
        self.pos_target_echoes = 0
        # race status (same stale-echo GO logic as run_vq1 / old probe)
        self.race_started = False
        self.countdown_armed = False
        self.armed_race_start_ms = -1
        self.active_gate = -1
        self.go_wall_t = None

    def start(self):
        threading.Thread(target=self._loop, daemon=True, name="ProbeRX").start()

    def stop(self):
        self.running = False

    def set_phase(self, ph):
        with self.lock:
            self.phase = ph

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
            with self.lock:
                ph = self.phase
                if t == "ACTUATOR_OUTPUT_STATUS":
                    vals = list(getattr(msg, "actuator", []) or [])
                    n = int(getattr(msg, "active", 0)) or 0
                    used = vals[:n] if n else [v for v in vals if v != 0.0]
                    if used and ph in self.act:
                        self.act[ph].append(sum(abs(v) for v in used) / len(used))
                elif t == "HIGHRES_IMU":
                    a = (float(msg.xacc), float(msg.yacc), float(msg.zacc))
                    g = (msg.xgyro**2 + msg.ygyro**2 + msg.zgyro**2) ** 0.5
                    if ph in self.acc:
                        self.acc[ph].append(a)
                        self.gyro[ph].append(g)
                elif t == "STATUSTEXT":
                    txt = getattr(msg, "text", "")
                    if isinstance(txt, (bytes, bytearray)):
                        txt = txt.decode("utf-8", "replace")
                    self.statustexts.append((ph, txt.strip("\x00").strip()))
                elif t == "COMMAND_ACK":
                    self.command_acks.append((ph, int(msg.command), int(msg.result)))
                elif t == "POSITION_TARGET_LOCAL_NED":
                    self.pos_target_echoes += 1
                elif t == "ENCAPSULATED_DATA":
                    self._race(msg, STALE_DELTA_MS, START_MARGIN_MS)

    def _race(self, msg, STALE_DELTA_MS, START_MARGIN_MS):
        raw = bytes(msg.data)
        if not raw or raw[0] != 1:
            return
        try:
            _, sim_boot_ms, race_start_ms, _fin, active_gate, _ = struct.unpack_from("<BQqqIq", raw)
        except struct.error:
            return
        self.active_gate = int(active_gate)
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

    def summary(self, ph):
        with self.lock:
            act = list(self.act.get(ph, []))
            acc = list(self.acc.get(ph, []))
            gyro = list(self.gyro.get(ph, []))
        act_mean = statistics.fmean(act) if act else float("nan")
        amag = [(x[0]**2 + x[1]**2 + x[2]**2) ** 0.5 for x in acc]
        amag_std = statistics.pstdev(amag) if len(amag) > 1 else 0.0
        az_mean = statistics.fmean(x[2] for x in acc) if acc else float("nan")
        gyro_max = max(gyro) if gyro else 0.0
        return dict(n_act=len(act), act_mean=act_mean, n_imu=len(acc),
                    amag_std=amag_std, az_mean=az_mean, gyro_max=gyro_max)


def timesync_thread(conn, stop):
    """Exactly timesync.py: timesync_send(time_ns, 0) at 10 Hz."""
    while not stop.is_set():
        conn.mav.timesync_send(int(time.time_ns()), 0)
        time.sleep(1.0 / 10.0)


def vpe_send(conn, boot_us):
    """DUMMY VISION_POSITION_ESTIMATE (zero pose). BEYOND the sample -- only with --vpe."""
    usec = int(time.time() * 1e6) - boot_us
    try:  # newer pymavlink wants covariance[21] + reset_counter
        conn.mav.vision_position_estimate_send(usec, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                               [0.0] * 21, 0)
    except TypeError:
        conn.mav.vision_position_estimate_send(usec, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="127.0.0.1", help="bind IP (sample uses 127.0.0.1)")
    ap.add_argument("--port", type=int, default=14550)
    ap.add_argument("--cmd", default="velfwd", choices=["velfwd", "climb", "north", "east"],
                    help="velfwd = sample's EXACT velocity call; climb = position D=-3; "
                         "north/east = PURE HORIZONTAL position setpoint at fixed altitude "
                         "(the missing analog to climb: proves N/E control, not just altitude)")
    ap.add_argument("--height", type=float, default=3.0, help="climb height m for --cmd climb")
    ap.add_argument("--dist", type=float, default=5.0, help="horizontal distance m for north/east")
    ap.add_argument("--rate", type=float, default=250.0, help="setpoint Hz (sample = 250)")
    ap.add_argument("--baseline-s", type=float, default=2.5)
    ap.add_argument("--stream-s", type=float, default=15.0)
    ap.add_argument("--vpe", action="store_true",
                    help="ALSO stream a dummy VISION_POSITION_ESTIMATE (beyond the sample)")
    ap.add_argument("--wait-go", action="store_true",
                    help="only stream AFTER race GO (default: stream across the GO)")
    ap.add_argument("--no-go", action="store_true",
                    help="stream immediately like the raw sample; never reference GO")
    ap.add_argument("--go-timeout", type=float, default=180.0)
    args = ap.parse_args()

    if args.cmd == "velfwd":
        mask, pos, vel = VEL_MASK, (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)
        label = "velfwd: sample-EXACT vel mask, vx=2.0 m/s fwd (pos ignored)"
    elif args.cmd == "climb":
        mask, pos, vel = POS_MASK, (0.0, 0.0, -abs(args.height)), (0.0, 0.0, 0.0)
        label = f"climb: pos mask, N=0 E=0 D={-abs(args.height)} (vel ignored)"
    elif args.cmd == "north":
        mask, pos, vel = POS_MASK, (abs(args.dist), 0.0, 0.0), (0.0, 0.0, 0.0)
        label = (f"north: pos mask, N={abs(args.dist)} E=0 D=0 -- PURE HORIZONTAL. If the "
                 f"drone translates/tilts, N/E position control works (not just altitude).")
    else:  # east
        mask, pos, vel = POS_MASK, (0.0, abs(args.dist), 0.0), (0.0, 0.0, 0.0)
        label = (f"east: pos mask, N=0 E={abs(args.dist)} D=0 -- PURE HORIZONTAL. If the "
                 f"drone translates/tilts, N/E position control works (not just altitude).")

    print(f"[exact] {label}")
    print(f"[exact] MAV_FRAME_LOCAL_NED  type_mask={mask} (0x{mask:03x})  rate={args.rate}Hz "
          f"vpe={args.vpe}")

    # ---- EXACT sample sequence: connect -> heartbeat -> timesync -> arm -> stream ----
    print(f"[exact] connecting udpin:{args.ip}:{args.port} ...")
    conn = mavutil.mavlink_connection(f"udpin:{args.ip}:{args.port}")
    print("[exact] waiting for heartbeat ...")
    conn.wait_heartbeat()
    print(f"[exact] heartbeat: system {conn.target_system} component {conn.target_component}")

    stop = threading.Event()
    threading.Thread(target=timesync_thread, args=(conn, stop), daemon=True,
                     name="Timesync").start()
    rx = RX(conn)
    rx.start()

    boot0_ms = int(time.time() * 1000)         # = sample's system_boot_ms
    boot0_us = int(time.time() * 1e6)

    conn.mav.command_long_send(conn.target_system, conn.target_component,
                               mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                               0, 1, 0, 0, 0, 0, 0, 0)
    print("[exact] ARM sent")

    if args.wait_go and not args.no_go:
        print("[exact] waiting for race GO ...")
        t0 = time.time()
        while not rx.race_started:
            if time.time() - t0 > args.go_timeout:
                print("[exact] ERROR: no GO within timeout. Start a race, retry.")
                stop.set(); rx.stop(); return 2
            time.sleep(0.05)
        print(f"[exact] GO. active_gate={rx.active_gate}")

    def send_once():
        conn.mav.set_position_target_local_ned_send(
            int(time.time() * 1000) - boot0_ms,
            conn.target_system, conn.target_component,
            mavc.MAV_FRAME_LOCAL_NED, mask,
            pos[0], pos[1], pos[2], vel[0], vel[1], vel[2],
            0.0, 0.0, 0.0, 0.0, 0.0)
        if args.vpe:
            vpe_send(conn, boot0_us)

    # baseline: send NOTHING (measure motors/IMU at rest)
    rx.set_phase("baseline")
    print(f"[exact] BASELINE {args.baseline_s}s (no setpoint) ...")
    time.sleep(args.baseline_s)
    base = rx.summary("baseline")

    # stream the setpoint at `rate`, continuously (crosses the GO unless --wait-go)
    rx.set_phase("stream")
    gate_start = rx.active_gate
    print(f"[exact] STREAM {args.stream_s}s at {args.rate}Hz ...")
    dt = 1.0 / args.rate
    t0 = time.time()
    last_log = 0.0
    while time.time() - t0 < args.stream_s:
        send_once()
        now = time.time()
        if now - last_log >= 1.0:
            last_log = now
            s = rx.summary("stream")
            go = "GO" if rx.race_started else "pre-GO"
            print(f"[exact]  t={now - t0:4.1f}s {go:6s} motors={s['act_mean']:.3f} "
                  f"gyro_max={s['gyro_max']:.2f} active_gate={rx.active_gate}")
        time.sleep(dt)
    strm = rx.summary("stream")

    stop.set(); rx.stop()

    # ---- verdict ----
    print("\n================= EXACT-SAMPLE PROBE RESULT =================")
    print(f"cmd={args.cmd}  vpe={args.vpe}  race_started={rx.race_started}")
    print(f"phase       motors(|out|)   IMU|a|std   gyro_max   az_mean   n_imu")
    for name, d in (("baseline", base), ("stream", strm)):
        print(f"{name:9s}   {d['act_mean']:>11.3f}   {d['amag_std']:>8.3f}   "
              f"{d['gyro_max']:>8.3f}   {d['az_mean']:>7.2f}   {d['n_imu']}")
    d_motor = strm["act_mean"] - base["act_mean"]
    d_gyro = strm["gyro_max"] - base["gyro_max"]
    d_astd = strm["amag_std"] - base["amag_std"]
    gate_adv = rx.active_gate - gate_start
    print(f"\ndeltas: motors {d_motor:+.3f}   gyro_max {d_gyro:+.3f}   |a|std {d_astd:+.3f}   "
          f"active_gate {gate_start}->{rx.active_gate} ({gate_adv:+d})")
    print(f"POSITION_TARGET echoes: {rx.pos_target_echoes}")
    for ph, txt in rx.statustexts:
        print(f"  STATUSTEXT[{ph}]: {txt}")
    for ph, cmd, res in rx.command_acks:
        print(f"  COMMAND_ACK[{ph}]: command={cmd} result={res}")

    moved = (abs(d_motor) > 0.05 or d_gyro > 0.2 or d_astd > 0.3
             or gate_adv != 0 or rx.pos_target_echoes > 0)
    rejected = any(k in t.lower() for _, t in rx.statustexts
                   for k in ("reject", "unsupported", "deny", "not support"))
    print("\nVERDICT:")
    if rejected:
        print("  REJECTED -- sim explicitly refused (see STATUSTEXT). Position/velocity")
        print("  control is not accepted with this mask/frame.")
    elif moved:
        print("  *** SIM RESPONDED to the sample's EXACT sequence. *** Position/velocity")
        print("  control WORKS -- our old probe was missing this. Next: --cmd climb (if this")
        print("  was velfwd) to confirm POSITION, then build the gate-waypoint flier.")
    else:
        print("  IGNORED -- even the sample's EXACT sequence produced no motor/IMU/gate")
        print("  response. v3385 does NOT honour position/velocity control. Stop chasing it;")
        print("  the only live control path is SET_ATTITUDE_TARGET body-rates after GO.")
        if not args.vpe:
            print("  (Optional last stone: re-run with --vpe to test the 'needs a pose source'")
            print("   hypothesis before closing this for good.)")
    print("=============================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
