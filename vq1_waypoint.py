#!/usr/bin/env python3
"""VQ1 hardcoded WAYPOINT script — send gate coordinates, let the sim fly.

GATED: only useful if tools/probe_position_target.py confirmed the sim honours
SET_POSITION_TARGET_LOCAL_NED. If the probe said IGNORED/REJECTED, use the
vision servo (run_vq1.py --controller vision-servo) instead.

If position targets DO work, this is the whole qualifying run with no perception:
  arm -> TIMESYNC -> wait GO -> each tick, stream the NED setpoint for the sim's
  CURRENT target gate (index = active_gate from race status) -> the sim's own
  controller flies there -> active_gate advances -> repeat through FINISH.

Because we always send the setpoint for `active_gate` (the sim's own idea of the
current target), gate sequencing is automatic: when the sim counts a gate passed
and increments active_gate, the very next tick targets the next gate. No position
telemetry needed.

Pass whatever the probe confirmed:
  --axes  {world|world_negN|spawn_fwd}   axis mapping the climb+gate1 probe validated
  --mask  {pos|posvel}                    type_mask that produced motion
  --frame {local|offset|body}             coordinate_frame that produced motion

    python vq1_waypoint.py --axes world --mask pos --frame local
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pymavlink import mavutil  # noqa: E402

from dcl_mavlink_adapter import DCLTimesync  # noqa: E402
from nav_frames import gate_ned_setpoints  # noqa: E402
# Reuse the probe's verified RX/GO/race-status plumbing instead of duplicating it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
from probe_position_target import Probe, MASK_POS, MASK_POSVEL, FRAMES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=14550)
    ap.add_argument("--axes", default="world", choices=["world", "world_negN", "spawn_fwd"])
    ap.add_argument("--mask", default="pos", choices=["pos", "posvel"])
    ap.add_argument("--frame", default="local", choices=["local", "offset", "body"])
    ap.add_argument("--rate", type=float, default=30.0, help="setpoint send rate Hz")
    ap.add_argument("--go-timeout", type=float, default=180.0)
    ap.add_argument("--gate-timeout", type=float, default=60.0,
                    help="abort if active_gate does not advance within this many seconds")
    ap.add_argument("--race-window", type=float, default=480.0,
                    help="hard cap on live-race seconds (DCL default 480)")
    args = ap.parse_args()

    gates = gate_ned_setpoints(axes=args.axes)
    n_gates = len(gates)
    type_mask = MASK_POS if args.mask == "pos" else MASK_POSVEL
    frame = FRAMES[args.frame]
    print(f"[wp] {n_gates} gates, axes={args.axes}, mask={args.mask} "
          f"(0x{type_mask:03x}), frame={args.frame} ({frame}), rate={args.rate} Hz")
    for g in gates:
        print(f"  gate {g['index']} {str(g['tag'] or ''):<7} NED={g['ned']}")

    conn = mavutil.mavlink_connection(f"udpin:0.0.0.0:{args.port}")
    conn.wait_heartbeat()
    print(f"[wp] heartbeat: sys {conn.target_system} comp {conn.target_component}")
    conn.mav.command_long_send(conn.target_system, conn.target_component,
                               mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                               0, 1, 0, 0, 0, 0, 0, 0)
    print("[wp] ARM sent")
    ts = DCLTimesync(conn)
    ts.start()
    probe = Probe(conn)   # RX-only: tracks race_started, active_gate, race_finish_ns
    probe.set_phase("run")
    probe.start()

    print("[wp] waiting for GO ...")
    t0 = time.time()
    while not probe.race_started:
        if time.time() - t0 > args.go_timeout:
            print("[wp] ERROR: no GO within timeout."); ts.stop(); probe.stop(); return 2
        time.sleep(0.05)
    print(f"[wp] GO. active_gate={probe.active_gate}")

    def send(ned):
        conn.mav.set_position_target_local_ned_send(
            int(time.time() * 1000), conn.target_system, conn.target_component,
            frame, type_mask, ned[0], ned[1], ned[2], 0, 0, 0, 0, 0, 0, 0, 0)

    go_wall = time.time()
    last_gate = max(0, probe.active_gate)
    last_advance = time.time()
    dt = 1.0 / args.rate
    outcome = None
    while outcome is None:
        now = time.time()
        if probe.race_finish_ns > 0:
            outcome = f"COMPLETED (race_finish_ns={probe.race_finish_ns}, active_gate={probe.active_gate})"
            break
        if now - go_wall > args.race_window:
            outcome = f"TIMEOUT (race window {args.race_window}s, reached gate {probe.active_gate})"
            break
        ag = probe.active_gate
        if ag != last_gate:
            print(f"[wp] active_gate {last_gate} -> {ag}")
            last_gate = ag
            last_advance = now
        if now - last_advance > args.gate_timeout:
            outcome = f"STALLED at gate {ag} (no advance in {args.gate_timeout}s)"
            break
        # Target the sim's CURRENT gate; clamp into range (pre-race/-1 -> first gate).
        idx = ag if 0 <= ag < n_gates else (0 if ag < 0 else n_gates - 1)
        send(gates[idx]["ned"])
        time.sleep(dt)

    ts.stop(); probe.stop()
    print(f"\n[wp] OUTCOME: {outcome}")
    return 0 if outcome and outcome.startswith("COMPLETED") else 1


if __name__ == "__main__":
    sys.exit(main())
