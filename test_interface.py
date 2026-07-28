#!/usr/bin/env python3
"""
VQ1 interface bisection test  --  NO MODEL, NO VISION.
Purpose: prove whether the sim accepts SET_ATTITUDE_TARGET and produces motion.

If the drone climbs when you run this, your control interface is correct and the
bug lives entirely in your state/model pipeline. If it does NOT climb, the bug is
interface-level (message type / type_mask / rate / arming).

Usage:
    python test_interface.py --conn udpin:0.0.0.0:14550 --thrust 0.7 --secs 4

Spec anchors (VADR-TS-002):
  - Control interface = SET_ATTITUDE_TARGET (actuator control is NOT in 4.3)
  - Command rate < 100 Hz  -> we send at 80 Hz
  - Heartbeat >= 2 Hz
  - Coordinate frame: NED  (down is +Z, so thrust pushes you UP in body frame)
"""
import argparse, math, time, threading
from pymavlink import mavutil

# ATTITUDE_TARGET type_mask: ignore the 3 body rates, USE attitude(q) + thrust.
# bit0=roll_rate, bit1=pitch_rate, bit2=yaw_rate -> 0b00000111 = 7
IGNORE_RATES_USE_ATT_THRUST = 0b00000111

def euler_to_quat(roll, pitch, yaw):
    cy, sy = math.cos(yaw*0.5), math.sin(yaw*0.5)
    cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
    cr, sr = math.cos(roll*0.5), math.sin(roll*0.5)
    return [cr*cp*cy + sr*sp*sy,   # w
            sr*cp*cy - cr*sp*sy,   # x
            cr*sp*cy + sr*cp*sy,   # y
            cr*cp*sy - sr*sp*cy]   # z

def heartbeat_loop(m, stop):
    while not stop.is_set():
        m.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        time.sleep(0.25)  # 4 Hz, > spec minimum of 2 Hz

def send_attitude(m, q, thrust):
    m.mav.set_attitude_target_send(
        int(time.time()*1000) & 0xFFFFFFFF,
        m.target_system, m.target_component,
        IGNORE_RATES_USE_ATT_THRUST,
        q, 0.0, 0.0, 0.0,
        float(thrust))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", default="udpin:0.0.0.0:14550")
    ap.add_argument("--thrust", type=float, default=0.7)  # > hover (~0.5)
    ap.add_argument("--secs", type=float, default=4.0)
    ap.add_argument("--rate", type=float, default=80.0)   # < 100 Hz per spec
    ap.add_argument("--arm-thrust", type=float, default=0.05)  # low at arm
    ap.add_argument("--prearm-secs", type=float, default=2.0)
    args = ap.parse_args()

    print(f"[*] connecting {args.conn}")
    m = mavutil.mavlink_connection(args.conn)
    m.wait_heartbeat()
    print(f"[+] heartbeat: sys={m.target_system} comp={m.target_component}")

    # log every distinct message type the sim sends (confirms LOCAL_POSITION_NED absence)
    seen = {}
    def watch():
        while True:
            msg = m.recv_match(blocking=True)
            if msg is None: continue
            t = msg.get_type()
            seen[t] = seen.get(t, 0) + 1
    threading.Thread(target=watch, daemon=True).start()

    stop = threading.Event()
    threading.Thread(target=heartbeat_loop, args=(m, stop), daemon=True).start()

    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0)
    print("[*] ARM sent")

    level = euler_to_quat(0, 0, 0)
    dt = 1.0/args.rate

    # PRE-ARM HOLD: low thrust so the 'throttle down' safety gate is satisfied
    print(f"[*] pre-arm hold {args.prearm_secs}s at thrust={args.arm_thrust}")
    t_end = time.time() + args.prearm_secs
    while time.time() < t_end:
        send_attitude(m, level, args.arm_thrust); time.sleep(dt)

    # CLIMB: level attitude, thrust above hover
    print(f"[*] CLIMB {args.secs}s at thrust={args.thrust} ({args.rate:.0f} Hz)")
    n, t_end = 0, time.time() + args.secs
    while time.time() < t_end:
        send_attitude(m, level, args.thrust); n += 1; time.sleep(dt)

    print(f"[*] settle: thrust back to {args.arm_thrust}")
    for _ in range(int(args.rate)):
        send_attitude(m, level, args.arm_thrust); time.sleep(dt)

    stop.set()
    print(f"\n[+] sent {n} attitude cmds")
    print("[+] message types received from sim:")
    for t, c in sorted(seen.items(), key=lambda x: -x[1]):
        print(f"      {t:<28} x{c}")
    if "LOCAL_POSITION_NED" not in seen:
        print("\n[!] LOCAL_POSITION_NED NOT received -> your 19D state position is zero.")
        print("    Per spec 3.3/4.5 the sim does not expose position. Estimate it.")
    print("\n[?] Did the drone climb? YES -> interface OK, bug is in model/state.")
    print("                          NO  -> bug is interface (mask/thrust/arming).")

if __name__ == "__main__":
    main()
