#!/usr/bin/env python3
"""Probe whether the DCL sim honours SET_POSITION_TARGET_LOCAL_NED.

Run ON THE SIM BOX. If the sim's own controller flies to position setpoints, the
whole VQ1 qualifying script becomes: send the 6 gate coordinates in order, watch
active_gate. This probe answers, cheaply and unmistakably, whether that path is
open -- BEFORE anyone calibrates the vision servo.

Strategy (the pure-climb test, per design)
-------------------------------------------
We have NO position/velocity telemetry (measured: only HIGHRES_IMU +
ACTUATOR_OUTPUT_STATUS + ENCAPSULATED_DATA on this sim). So we can't watch the
drone "arrive". Instead:

  1. Wait for a real race GO (control is only honoured live).
  2. BASELINE window: send nothing; record motor outputs (ACTUATOR_OUTPUT_STATUS,
     ~93 Hz) and IMU (accel/gyro) at rest-under-race.
  3. SETPOINT window: stream a PURE-CLIMB setpoint (N=0,E=0,D=-height) in
     MAV_FRAME_LOCAL_NED. Zero horizontal cancels ALL axis ambiguity: if the
     drone climbs, position control works AND local-NED origin is spawn-relative.
  4. Verdict from what we CAN see:
       - motor outputs spool up vs baseline      -> controller acted on it
       - IMU shows vertical accel / any motion    -> it moved
       - a COMMAND_ACK / STATUSTEXT arrives        -> explicit accept/reject
       - POSITION_TARGET_LOCAL_NED echo            -> sim latched the target
       - silence + no motor/IMU change             -> IGNORED (try --mask/--frame)

If the climb works, re-run with `--setpoint gate1` to fly toward gate 1 and watch
active_gate advance -- the whole qualifying script in embryo.

READ-ONLY on telemetry; the only things sent are ARM, TIMESYNC, and the setpoint
under test. Nothing here flies the drone except the setpoint you asked for.

    python tools/probe_position_target.py                 # pure 3 m climb, LOCAL_NED, pos-only mask
    python tools/probe_position_target.py --mask posvel   # position + zero-velocity mask
    python tools/probe_position_target.py --frame offset  # LOCAL_OFFSET_NED (relative to current)
    python tools/probe_position_target.py --setpoint gate1 --axes world   # after a good climb
    python tools/probe_position_target.py --setpoint custom -20 0 -3
"""
import argparse
import os
import statistics
import sys
import threading
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymavlink import mavutil  # noqa: E402
from pymavlink.dialects.v20 import common as mavc  # noqa: E402

from dcl_mavlink_adapter import DCLTimesync  # noqa: E402
from nav_frames import climb_setpoint, gate_ned_setpoints  # noqa: E402

# position-only: ignore vel + accel + yaw + yaw_rate, use x/y/z.
MASK_POS = (mavc.POSITION_TARGET_TYPEMASK_VX_IGNORE | mavc.POSITION_TARGET_TYPEMASK_VY_IGNORE
            | mavc.POSITION_TARGET_TYPEMASK_VZ_IGNORE | mavc.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavc.POSITION_TARGET_TYPEMASK_AY_IGNORE | mavc.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavc.POSITION_TARGET_TYPEMASK_YAW_IGNORE | mavc.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE)
# position + velocity (velocity setpoints = 0): some controllers need the vel
# fields present, not ignored, or a supported message no-ops.
MASK_POSVEL = (mavc.POSITION_TARGET_TYPEMASK_AX_IGNORE | mavc.POSITION_TARGET_TYPEMASK_AY_IGNORE
               | mavc.POSITION_TARGET_TYPEMASK_AZ_IGNORE | mavc.POSITION_TARGET_TYPEMASK_YAW_IGNORE
               | mavc.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE)

FRAMES = {"local": mavc.MAV_FRAME_LOCAL_NED,       # 1: relative to local origin (spawn?)
          "offset": mavc.MAV_FRAME_LOCAL_OFFSET_NED,  # 7: relative to CURRENT position
          "body": mavc.MAV_FRAME_BODY_NED}          # 8: relative to body


class Probe:
    def __init__(self, conn):
        self.conn = conn
        self.lock = threading.Lock()
        self.running = True
        # GO detection (mirrors run_vq1 stale-echo logic)
        self.race_started = False
        self.countdown_armed = False
        self.armed_race_start_ms = -1
        self.active_gate = -1
        self.race_finish_ns = 0
        # phase-tagged collectors
        self.phase = "init"
        self.act = {"baseline": [], "setpoint": []}   # per-sample mean actuator magnitude
        self.imu_acc = {"baseline": [], "setpoint": []}  # (ax,ay,az)
        self.imu_gyro = {"baseline": [], "setpoint": []}  # |gyro|
        self.statustexts = []
        self.command_acks = []
        self.pos_target_echoes = 0
        self._accel_window = deque(maxlen=20)
        # timestamped IMU trace (t_wall, gx, gy, gz, ax, ay, az) for every sample,
        # phase-independent -- used by tools/probe_rate_vs_angle.py for the gyro
        # time-series analysis. Bounded so a long session can't grow unbounded.
        self.imu_trace = deque(maxlen=20000)

    def start(self):
        threading.Thread(target=self._loop, daemon=True, name="ProbeRX").start()

    def stop(self):
        self.running = False

    def _loop(self):
        STALE_DELTA_MS = -10000
        START_MARGIN_MS = 100
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
                    acc = (float(msg.xacc), float(msg.yacc), float(msg.zacc))
                    gyro = (float(msg.xgyro), float(msg.ygyro), float(msg.zgyro))
                    self._accel_window.append(acc)
                    self.imu_trace.append((time.time(), gyro[0], gyro[1], gyro[2],
                                           acc[0], acc[1], acc[2]))
                    if ph in self.imu_acc:
                        self.imu_acc[ph].append(acc)
                        self.imu_gyro[ph].append((gyro[0]**2 + gyro[1]**2 + gyro[2]**2) ** 0.5)
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
                    self._race_status(msg, STALE_DELTA_MS, START_MARGIN_MS)

    def _race_status(self, msg, STALE_DELTA_MS, START_MARGIN_MS):
        import struct
        raw = bytes(msg.data)
        if not raw or raw[0] != 1:
            return
        try:
            _, sim_boot_ms, race_start_ms, _fin_ns, active_gate, _ = \
                struct.unpack_from("<BQqqIq", raw)
        except struct.error:
            return
        self.active_gate = int(active_gate)
        if _fin_ns > 0:
            self.race_finish_ns = int(_fin_ns)
        delta = race_start_ms - sim_boot_ms
        is_stale = race_start_ms >= 0 and delta < STALE_DELTA_MS
        if is_stale:
            return
        if (not self.countdown_armed) and race_start_ms >= 0 and delta > 0:
            self.countdown_armed = True
            self.armed_race_start_ms = race_start_ms
        if (self.countdown_armed and not self.race_started
                and sim_boot_ms >= self.armed_race_start_ms + START_MARGIN_MS):
            self.race_started = True

    # --- phase helpers ---
    def set_phase(self, ph):
        with self.lock:
            self.phase = ph

    def summary(self, ph):
        with self.lock:
            act = list(self.act.get(ph, []))
            acc = list(self.imu_acc.get(ph, []))
            gyro = list(self.imu_gyro.get(ph, []))
        act_mean = statistics.fmean(act) if act else float("nan")
        az_mean = statistics.fmean(a[2] for a in acc) if acc else float("nan")
        amag = [(a[0]**2 + a[1]**2 + a[2]**2) ** 0.5 for a in acc]
        amag_std = statistics.pstdev(amag) if len(amag) > 1 else 0.0
        gyro_max = max(gyro) if gyro else 0.0
        return dict(n_act=len(act), act_mean=act_mean, n_imu=len(acc),
                    az_mean=az_mean, amag_std=amag_std, gyro_max=gyro_max)


def build_setpoint(args):
    if args.setpoint == "climb":
        ned = climb_setpoint(args.height)
        label = f"pure climb {args.height} m up (LOCAL_NED N=0 E=0 D={ned[2]})"
    elif args.setpoint == "gate1":
        row = gate_ned_setpoints(axes=args.axes)[1]  # index 1 = first gate after START
        ned = row["ned"]
        label = f"gate1 NED {ned} (axes={args.axes})"
    elif args.setpoint == "custom":
        ned = args.custom
        label = f"custom NED {ned}"
    else:
        raise ValueError(args.setpoint)
    return [float(ned[0]), float(ned[1]), float(ned[2])], label


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=14550)
    ap.add_argument("--setpoint", default="climb", choices=["climb", "gate1", "custom"])
    ap.add_argument("--height", type=float, default=3.0, help="climb height (m) for --setpoint climb")
    ap.add_argument("--custom", type=float, nargs=3, metavar=("N", "E", "D"),
                    help="custom NED setpoint for --setpoint custom")
    ap.add_argument("--axes", default="world", choices=["world", "world_negN", "spawn_fwd"],
                    help="UE->NED axis hypothesis for --setpoint gate1")
    ap.add_argument("--mask", default="pos", choices=["pos", "posvel"])
    ap.add_argument("--frame", default="local", choices=["local", "offset", "body"])
    ap.add_argument("--rate", type=float, default=30.0, help="setpoint send rate Hz")
    ap.add_argument("--baseline-s", type=float, default=2.5)
    ap.add_argument("--setpoint-s", type=float, default=6.0)
    ap.add_argument("--go-timeout", type=float, default=180.0)
    ap.add_argument("--no-go-wait", action="store_true",
                    help="send without waiting for GO (diagnostic; control likely inert)")
    args = ap.parse_args()

    if args.setpoint == "custom" and not args.custom:
        ap.error("--setpoint custom requires --custom N E D")

    ned, label = build_setpoint(args)
    type_mask = MASK_POS if args.mask == "pos" else MASK_POSVEL
    frame = FRAMES[args.frame]
    print(f"[probe] setpoint: {label}")
    print(f"[probe] type_mask={type_mask} (0x{type_mask:03x}, {args.mask})  "
          f"coordinate_frame={frame} ({args.frame})  rate={args.rate} Hz")

    print(f"[probe] binding udpin:0.0.0.0:{args.port} ...")
    conn = mavutil.mavlink_connection(f"udpin:0.0.0.0:{args.port}")
    conn.wait_heartbeat()
    print(f"[probe] heartbeat: system {conn.target_system}, component {conn.target_component}")

    conn.mav.command_long_send(conn.target_system, conn.target_component,
                               mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                               0, 1, 0, 0, 0, 0, 0, 0)
    print("[probe] ARM sent")

    ts = DCLTimesync(conn)
    ts.start()
    probe = Probe(conn)
    probe.start()

    # 1. wait for GO
    if not args.no_go_wait:
        print("[probe] waiting for race GO ...")
        t0 = time.time()
        while not probe.race_started:
            if time.time() - t0 > args.go_timeout:
                print("[probe] ERROR: no GO within timeout. Start a race and retry.")
                ts.stop(); probe.stop(); return 2
            time.sleep(0.05)
        print(f"[probe] GO. active_gate={probe.active_gate}")
    else:
        print("[probe] --no-go-wait: proceeding without GO (control likely inert).")

    boot0 = int(time.time() * 1000)

    def send_once():
        conn.mav.set_position_target_local_ned_send(
            int(time.time() * 1000) - boot0,
            conn.target_system, conn.target_component, frame, type_mask,
            ned[0], ned[1], ned[2], 0, 0, 0, 0, 0, 0, 0, 0)

    # 2. baseline (send nothing)
    probe.set_phase("baseline")
    print(f"[probe] BASELINE {args.baseline_s}s (no setpoint) ...")
    time.sleep(args.baseline_s)
    base = probe.summary("baseline")

    # 3. setpoint window
    probe.set_phase("setpoint")
    gate_at_start = probe.active_gate
    print(f"[probe] SETPOINT {args.setpoint_s}s streaming at {args.rate} Hz ...")
    dt = 1.0 / args.rate
    t0 = time.time()
    while time.time() - t0 < args.setpoint_s:
        send_once()
        time.sleep(dt)
    sp = probe.summary("setpoint")
    probe.set_phase("done")

    # 4. verdict
    ts.stop(); probe.stop()
    print("\n================= PROBE RESULT =================")
    print(f"phase        motors(|out|)   IMU |a|std   IMU gyro_max   az_mean   n")
    print(f"baseline     {base['act_mean']:>10.3f}   {base['amag_std']:>9.3f}   "
          f"{base['gyro_max']:>11.3f}   {base['az_mean']:>7.2f}   {base['n_imu']}")
    print(f"setpoint     {sp['act_mean']:>10.3f}   {sp['amag_std']:>9.3f}   "
          f"{sp['gyro_max']:>11.3f}   {sp['az_mean']:>7.2f}   {sp['n_imu']}")

    d_motor = sp["act_mean"] - base["act_mean"]
    d_gyro = sp["gyro_max"] - base["gyro_max"]
    d_astd = sp["amag_std"] - base["amag_std"]
    gate_adv = probe.active_gate - gate_at_start

    print(f"\ndeltas: motors {d_motor:+.3f}   gyro_max {d_gyro:+.3f}   "
          f"|a|std {d_astd:+.3f}   active_gate {gate_at_start}->{probe.active_gate} ({gate_adv:+d})")
    print(f"POSITION_TARGET echoes: {probe.pos_target_echoes}")
    for ph, txt in probe.statustexts:
        print(f"  STATUSTEXT[{ph}]: {txt}")
    for ph, cmd, res in probe.command_acks:
        print(f"  COMMAND_ACK[{ph}]: command={cmd} result={res}")

    motion = (abs(d_motor) > 0.05 or d_gyro > 0.2 or d_astd > 0.3
              or gate_adv != 0 or probe.pos_target_echoes > 0)
    rejected = any("reject" in t.lower() or "unsupported" in t.lower() or "deny" in t.lower()
                   for _, t in probe.statustexts)
    print("\nVERDICT:")
    if rejected:
        print("  REJECTED — sim sent a rejection STATUSTEXT (see above). Position "
              "targets not accepted with this mask/frame.")
    elif motion:
        print("  *** SIM RESPONDED to the position target. *** Position control looks")
        print("  supported. If this was --setpoint climb: origin is spawn-relative and")
        print("  the frame works -> re-run with --setpoint gate1 and watch active_gate.")
        print("  Then build the waypoint script (send the 6 gate NEDs in order).")
    else:
        print("  IGNORED — no motor/IMU/gate change and no echo/ack/statustext. The")
        print("  message was likely a silent no-op. Try: --mask posvel, --frame offset,")
        print("  a bigger --height. If ALL variants are inert, position targets are not")
        print("  honoured -> fall back to the vision servo (calibrate HSV on a real frame).")
    print("================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
