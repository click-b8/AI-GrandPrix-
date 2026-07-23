#!/usr/bin/env python3
"""Is the DCL sim's controller RATE mode or ANGLE mode? (decisive, on the sim box)

Why this matters
----------------
The sim UI reports FLIGHT MODE: ANGLE. We send SET_ATTITUDE_TARGET with
type_mask=128 (IGNORE_ATTITUDE), which per MAVLink means "use body RATES +
thrust". If the controller actually treats those body_*_rate fields as ANGLE
setpoints, our whole control model is wrong:
  - the RL policy (trained on rates) is mis-scaled, and
  - the hardcoded vision servo's inner loop (a rate PD off the gravity estimate)
    is the wrong architecture -- it should send desired ANGLES via the quaternion
    (type_mask=0) and drop the inner loop entirely.
The position-target path can't rescue us: SET_POSITION_TARGET_LOCAL_NED was
probed live on v3385 and is INERT (motors 0.013, |a|std 0.000). So this question
has to be answered, not sidestepped.

The test
--------
After a real GO, command a CONSTANT body rate on one axis (default pitch,
0.30 rad/s) for a few seconds and watch the gyro on that axis (HIGHRES_IMU):

  RATE control  -> gyro rises to ~command and HOLDS (the drone keeps rotating);
                   integrated angle grows without bound.
  ANGLE control -> gyro SPIKES then DECAYS toward ~0 as the drone reaches the
                   implied angle and holds it; integrated angle plateaus.

We quantify with first-quarter vs last-quarter mean gyro over the command window
and the integrated angle. Clear separation either way.

Sends only ARM, TIMESYNC, and the SET_ATTITUDE_TARGET rate command under test.
NOT a timed attempt -- an authorised diagnostic. Keep --thrust modest.

    python tools/probe_rate_vs_angle.py                 # pitch, 0.30 rad/s, 2.5 s
    python tools/probe_rate_vs_angle.py --axis roll --rate 0.40
    python tools/probe_rate_vs_angle.py --thrust 0.25 --duration 3.0
"""
import argparse
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pymavlink import mavutil  # noqa: E402

from dcl_mavlink_adapter import DCLTimesync, MAX_BODY_RATE  # noqa: E402
from probe_position_target import Probe  # noqa: E402

AXIS_IDX = {"roll": 0, "pitch": 1, "yaw": 2}  # gyro/body-rate component index


def integ_angle(samples, idx):
    """Trapezoidal integral of gyro[idx] over the sample times -> radians."""
    if len(samples) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(samples[:-1], samples[1:]):
        dt = b[0] - a[0]
        total += 0.5 * (a[1 + idx] + b[1 + idx]) * dt
    return total


def window(trace, t0, t1):
    return [s for s in trace if t0 <= s[0] <= t1]


def classify(commanded, first_q, last_q, measured, scale_tol=(0.85, 1.15)):
    """Pure classifier from the gyro summary stats on the commanded axis.

    Returns a dict with verdict (RATE/ANGLE/INCONCLUSIVE), signed plant gain K =
    measured/commanded, sign check, and the list of calibration issues. RATE vs
    ANGLE is magnitude-based (sign-agnostic); sign + scale are reported SEPARATELY
    so a sign-inverted, mis-scaled RATE response is never mislabelled 'no change'.
    """
    signed_gain = measured / commanded if commanded != 0 else float("nan")
    sign_ok = (measured * commanded) > 0
    holds = abs(last_q) > 0.5 * abs(first_q) and abs(measured) > 0.30 * abs(commanded)
    decays = abs(first_q) > 0.15 and abs(last_q) < 0.4 * abs(first_q)
    if decays and not holds:
        verdict = "ANGLE"
    elif holds:
        verdict = "RATE"
    else:
        verdict = "INCONCLUSIVE"
    issues = []
    if verdict == "RATE":
        if not sign_ok:
            issues.append("SIGN INVERTED")
        if not (scale_tol[0] <= abs(signed_gain) <= scale_tol[1]):
            issues.append(f"SCALE {abs(signed_gain):.2f}x")
    return {"verdict": verdict, "K": signed_gain, "sign_ok": sign_ok,
            "holds": holds, "decays": decays, "issues": issues}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=14550)
    ap.add_argument("--axis", default="pitch", choices=["roll", "pitch", "yaw"])
    ap.add_argument("--rate", type=float, default=0.30,
                    help="commanded body rate rad/s on --axis (well below MAX_BODY_RATE=12)")
    ap.add_argument("--thrust", type=float, default=0.20, help="constant thrust during the test")
    ap.add_argument("--duration", type=float, default=2.5, help="command-window seconds")
    ap.add_argument("--baseline-s", type=float, default=1.0)
    ap.add_argument("--hz", type=float, default=250.0, help="command send rate")
    ap.add_argument("--go-timeout", type=float, default=180.0)
    ap.add_argument("--no-go-wait", action="store_true")
    args = ap.parse_args()

    idx = AXIS_IDX[args.axis]
    norm = args.rate / MAX_BODY_RATE
    if abs(norm) > 1.0:
        ap.error(f"--rate {args.rate} exceeds MAX_BODY_RATE {MAX_BODY_RATE}")
    print(f"[rate-probe] axis={args.axis} rate={args.rate} rad/s (norm {norm:+.4f}) "
          f"thrust={args.thrust} duration={args.duration}s")

    conn = mavutil.mavlink_connection(f"udpin:0.0.0.0:{args.port}")
    conn.wait_heartbeat()
    print(f"[rate-probe] heartbeat: sys {conn.target_system} comp {conn.target_component}")
    conn.mav.command_long_send(conn.target_system, conn.target_component,
                               mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                               0, 1, 0, 0, 0, 0, 0, 0)
    print("[rate-probe] ARM sent")
    ts = DCLTimesync(conn)
    ts.start()
    probe = Probe(conn)
    probe.set_phase("run")
    probe.start()

    if not args.no_go_wait:
        print("[rate-probe] waiting for GO ...")
        t0 = time.time()
        while not probe.race_started:
            if time.time() - t0 > args.go_timeout:
                print("[rate-probe] ERROR: no GO within timeout."); ts.stop(); probe.stop(); return 2
            time.sleep(0.05)
        print("[rate-probe] GO.")

    boot0 = int(time.time() * 1000)
    rates = [0.0, 0.0, 0.0]
    rates[idx] = args.rate

    def send(rvec, thrust):
        conn.mav.set_attitude_target_send(
            int(time.time() * 1000) - boot0, conn.target_system, conn.target_component,
            128, [1, 0, 0, 0], rvec[0], rvec[1], rvec[2], thrust)

    # baseline: no command
    t_base0 = time.time()
    print(f"[rate-probe] BASELINE {args.baseline_s}s ...")
    time.sleep(args.baseline_s)

    # command window: constant rate on the chosen axis
    t_cmd0 = time.time()
    print(f"[rate-probe] COMMAND {args.duration}s: {args.axis}_rate={args.rate} rad/s ...")
    dt = 1.0 / args.hz
    while time.time() - t_cmd0 < args.duration:
        send(rates, args.thrust)
        time.sleep(dt)
    t_cmd1 = time.time()
    # release: zero rates, cut thrust
    for _ in range(10):
        send([0.0, 0.0, 0.0], 0.0)
        time.sleep(0.01)

    ts.stop(); probe.stop()
    trace = list(probe.imu_trace)

    base = window(trace, t_base0, t_cmd0)
    cmd = window(trace, t_cmd0, t_cmd1)
    if len(cmd) < 8:
        print(f"[rate-probe] ERROR: only {len(cmd)} IMU samples in the command window "
              f"(IMU streaming? armed?). Cannot judge.")
        return 3

    def gm(samples):
        return statistics.fmean(s[1 + idx] for s in samples) if samples else float("nan")

    q = max(1, len(cmd) // 4)
    first_q = gm(cmd[:q])
    last_q = gm(cmd[-q:])
    # STEADY measured rate: mean gyro over the last 75% of the window (skip the
    # startup ramp). This is the signed number we calibrate against.
    measured = gm(cmd[q:])
    peak = max((abs(s[1 + idx]) for s in cmd), default=0.0)
    base_g = gm(base) if base else 0.0
    ang = integ_angle(cmd, idx)

    commanded = args.rate
    r = classify(commanded, first_q, last_q, measured)
    K = r["K"]

    print("\n================= RATE-vs-ANGLE RESULT =================")
    print(f"axis={args.axis}  ({len(cmd)} cmd samples, {len(base)} baseline)")
    print(f"  commanded rate = {commanded:+.3f} rad/s   (what we put on the wire)")
    print(f"  MEASURED  rate = {measured:+.3f} rad/s   (steady gyro mean)")
    print(f"  first_quarter={first_q:+.3f}  last_quarter={last_q:+.3f}  peak={peak:.3f}  "
          f"baseline={base_g:+.3f}")
    print(f"  signed gain K = measured/commanded = {K:+.2f}   (K=+1 is a perfect follower)")
    print(f"  integrated angle over window = {ang:+.3f} rad ({math.degrees(ang):+.1f} deg)")

    print("\nVERDICT:")
    if r["verdict"] == "ANGLE":
        print("  ANGLE control. |gyro| spiked then DECAYED toward zero -> the")
        print("  body_*_rate field is tracked as an ANGLE setpoint, not a rate.")
        print("   - vision servo: send desired ANGLES via the quaternion (type_mask=0),")
        print("     drop the gravity-PD inner loop.  - re-examine the RL CTBR mapping.")
    elif r["verdict"] == "RATE":
        if r["issues"]:
            print(f"  RATE control, but MIS-CALIBRATED: {', '.join(r['issues'])}.")
            print(f"    commanded {commanded:+.3f} -> measured {measured:+.3f} rad/s")
            print(f"    per-axis plant gain  K[{args.axis}] = {K:+.2f}")
            print(f"    correction: to achieve an intended rate R, send wire = R / K")
            print(f"                = R * {1.0/K:+.3f}   (folds in sign + scale)")
        else:
            print(f"  RATE control, calibrated (K={K:+.2f} ~ +1). No correction "
                  f"needed on {args.axis}.")
    else:
        print("  INCONCLUSIVE. Neither a clean hold nor a clean decay. Re-run with a")
        print("  larger --rate or longer --duration, or confirm a clean at-spawn GO.")

    print(f"\n[SUMMARY] axis={args.axis} cmd={commanded:+.3f} measured={measured:+.3f} "
          f"K={K:+.2f} sign={'OK' if r['sign_ok'] else 'INVERTED'} "
          f"scale={abs(K):.2f}x verdict={r['verdict']}")
    print("========================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
