#!/usr/bin/env python3
"""Pure OPEN-LOOP motion schedule for VQ1 (no vision, no position, no dead reckoning).

v3385 gives us attitude/rate control only (position is inert). So the qualifying
run becomes a TIMED sequence of SET_ATTITUDE_TARGET commands computed from the known
course geometry (course_gates_cm.json) and our measured plant (thrust->sink map,
cruise pitch, per-axis actuator K). The 0.40 probe proved fixed commands fly
straight; this computes the RIGHT fixed commands, segment by segment.

Per segment we hold [thrust, pitch, yaw-rate turn] for [duration]:
  thrust    -- from the required sink rate (= cruise_speed * tan(descent slope)) via
               the measured THRUST_SINK_MAP.
  pitch     -- held at cruise (~-18 deg): the forward-lean that drives cruise speed.
  yaw-rate  -- a brief feedforward turn at segment entry (yaw_rate = heading_change /
               turn_time) to swing the pitched-forward thrust vector onto the new
               bearing. This is the ONLY heading authority we have open-loop (no yaw
               telemetry), so it's the least-certain term -- validate live.
  duration  -- segment_length / cruise_speed.
Advance on active_gate (ground truth) with the duration as a time fallback.

EVERYTHING times off CRUISE SPEED, which is unknown until measured. Run the
calibration probe first (tools/probe_cruise.py): fixed pitch/thrust straight flight,
time the active_gate ticks -> m/s. Then rebuild with --cruise <measured>.

    python tools/motion_schedule.py --cruise 8            # show the schedule
    python tools/motion_schedule.py --cruise 8 --out motion_schedule.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nav_frames import load_spawn_and_gates          # noqa: E402
from vq1_vision_servo import thrust_for_sink, THRUST_SINK_MAP, ServoConfig  # noqa: E402


def segments(path=None):
    """Per-segment geometry from spawn through the 6 gates. Frame-agnostic: only
    distances / descent / RELATIVE heading change matter for an attitude schedule."""
    spawn_cm, gates = load_spawn_and_gates(path) if path else load_spawn_and_gates()
    names = ["SPAWN"] + [(t or f"g{i}") for i, (t, _) in enumerate(gates)]
    pts = [spawn_cm / 100.0] + [p / 100.0 for _, p in gates]
    segs = []
    prev_brg = None
    for i in range(1, len(pts)):
        dv = pts[i] - pts[i - 1]
        dx, dy, dz = float(dv[0]), float(dv[1]), float(dv[2])
        dhoriz = math.hypot(dx, dy)
        d3d = float(np.linalg.norm(dv))
        drop = -dz                                   # + = descends
        slope = math.degrees(math.atan2(drop, dhoriz))
        brg = math.degrees(math.atan2(dy, dx))
        turn = 0.0 if prev_brg is None else ((brg - prev_brg + 180.0) % 360.0 - 180.0)
        prev_brg = brg
        segs.append({"seg": i - 1, "from": names[i - 1], "to": names[i],
                     "d3d": d3d, "dhoriz": dhoriz, "drop": drop,
                     "slope_deg": slope, "turn_deg": turn})
    return segs


def build_schedule(cruise_mps, cfg=None, turn_time_s=1.0):
    cfg = cfg or ServoConfig()
    max_rate = cfg.ol_max_rate_rad_s
    out = []
    for s in segments():
        sink = cruise_mps * math.tan(math.radians(s["slope_deg"]))   # + = need to descend
        thr = float(thrust_for_sink(sink))
        sink_capped = sink > max(p[1] for p in THRUST_SINK_MAP)       # steep -> map floor thrust
        duration = s["d3d"] / max(cruise_mps, 1e-3)
        t_turn = min(turn_time_s, duration)
        yaw_rate = math.radians(s["turn_deg"]) / max(t_turn, 1e-3)
        yaw_clamped = abs(yaw_rate) > max_rate
        yaw_rate = float(np.clip(yaw_rate, -max_rate, max_rate))
        out.append({
            "seg": s["seg"], "from": s["from"], "to": s["to"],
            "thrust": round(thr, 3),
            "pitch_deg": round(cfg.cruise_pitch_deg, 1),
            "roll_deg": 0.0,
            "yaw_rate_rad_s": round(yaw_rate, 3),
            "turn_time_s": round(t_turn, 2),
            "duration_s": round(duration, 2),
            "req_sink_mps": round(sink, 2),
            "turn_deg": round(s["turn_deg"], 1),
            "slope_deg": round(s["slope_deg"], 1),
            "d3d_m": round(s["d3d"], 1),
            "sink_capped": bool(sink_capped),
            "yaw_capped": bool(yaw_clamped),
        })
    return out


G = 9.81


def lateral_offsets(path=None):
    """Per-segment lateral translation (metres, RIGHT positive) the drone must slide
    to go from the previous gate's Y to this gate's Y. All gates share -90 deg
    heading, so the course is pure lateral offsets on a fixed forward heading -- no
    rotation. RIGHT = -Y (a right gate has dy<0, matching the +2.2 deg to gate 1)."""
    spawn_cm, gates = load_spawn_and_gates(path) if path else load_spawn_and_gates()
    ys = [spawn_cm[1] / 100.0] + [p[1] / 100.0 for _, p in gates]
    return [-(ys[i] - ys[i - 1]) for i in range(1, len(ys))]   # right positive


def build_bank_schedule(cruise_mps, cfg=None, maneuver_frac=0.7):
    """Bank-to-TRANSLATE schedule (no yaw, no heading drift). Each segment does a
    symmetric bang-bang bank: +phi for t/2 then -phi for t/2, netting the segment's
    lateral offset with zero end lateral velocity and unchanged heading."""
    cfg = cfg or ServoConfig()
    lat = lateral_offsets()
    out = []
    for s in segments():
        d = lat[s["seg"]]                                   # right-positive offset (m)
        duration = s["d3d"] / max(cruise_mps, 1e-3)
        t_man = min(maneuver_frac * duration, duration)
        half = max(t_man / 2.0, 1e-3)
        phi = math.degrees(math.atan(abs(d) / (G * half * half)))
        capped = phi > cfg.ol_max_bank_deg
        phi = min(phi, cfg.ol_max_bank_deg)
        bank = math.copysign(phi, d)                        # signed, right positive
        sink = cruise_mps * math.tan(math.radians(s["slope_deg"]))
        thr = float(thrust_for_sink(sink))
        out.append({
            "seg": s["seg"], "from": s["from"], "to": s["to"],
            "lateral_m": round(d, 2), "bank_deg": round(bank, 1),
            "t_man_s": round(t_man, 2), "thrust": round(thr, 3),
            "pitch_deg": round(cfg.cruise_pitch_deg, 1),
            "duration_s": round(duration, 2), "req_sink_mps": round(sink, 2),
            "bank_capped": bool(capped), "slope_deg": round(s["slope_deg"], 1),
            "d3d_m": round(s["d3d"], 1),
        })
    return out


def print_bank_schedule(sched, cruise_mps):
    print(f"\nBANK-TO-TRANSLATE SCHEDULE  (cruise = {cruise_mps} m/s, pitch held "
          f"{sched[0]['pitch_deg']} deg, heading FIXED forward -- no yaw)")
    print("seg from->to      lateral  bank(+=right)  t_man   thrust  duration  req_sink  notes")
    total = 0.0
    for r in sched:
        total += r["duration_s"]
        notes = []
        if r["bank_capped"]:
            notes.append("BANK-CAPPED(can't translate enough in window)")
        print(f" {r['seg']}  {r['from'][:5]:>5}->{r['to'][:5]:<5} {r['lateral_m']:+6.2f}m  "
              f"{r['bank_deg']:+6.1f} deg    {r['t_man_s']:.2f}s  {r['thrust']:.3f}   "
              f"{r['duration_s']:5.2f}s   {r['req_sink_mps']:+5.2f}   {' '.join(notes)}")
    print(f"TOTAL scheduled time = {total:.1f} s.  bang-bang: +phi for t_man/2 then -phi for "
          f"t_man/2 -> net slide, zero end lateral velocity, heading unchanged.")
    if any(r["bank_capped"] for r in sched):
        print("NOTE: BANK-CAPPED segments need more slide than the fence bank allows in the "
              "window -> widen maneuver_frac (more of the segment) or lower cruise.")


def print_schedule(sched, cruise_mps):
    print(f"\nMOTION SCHEDULE  (cruise = {cruise_mps} m/s, pitch held at "
          f"{sched[0]['pitch_deg']} deg)")
    print("seg from->to      thrust  pitch  yaw_rate(turn)  t_turn  duration  req_sink  notes")
    total = 0.0
    for r in sched:
        total += r["duration_s"]
        notes = []
        if r["sink_capped"]:
            notes.append("SINK-CAPPED(can't descend fast enough)")
        if r["yaw_capped"]:
            notes.append("YAW-CAPPED")
        print(f" {r['seg']}  {r['from'][:5]:>5}->{r['to'][:5]:<5} {r['thrust']:.3f}  "
              f"{r['pitch_deg']:+5.1f}  {r['yaw_rate_rad_s']:+5.2f} rad/s ({r['turn_deg']:+5.1f})  "
              f"{r['turn_time_s']:.2f}s   {r['duration_s']:5.2f}s   {r['req_sink_mps']:+5.2f}   "
              f"{' '.join(notes)}")
    print(f"TOTAL scheduled time = {total:.1f} s  (race timing starts at GO)")
    if any(r["sink_capped"] for r in sched):
        print("NOTE: SINK-CAPPED segments need more descent than thrust=0.20 delivers at this "
              "cruise -> they'll under-descend. Lower --cruise, or extend the sink map, or "
              "add nose-down pitch on those segments.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cruise", type=float, default=8.0,
                    help="cruise speed m/s (MEASURE with tools/probe_cruise.py first)")
    ap.add_argument("--turn-time", type=float, default=1.0,
                    help="s over which each segment-entry heading change is executed")
    ap.add_argument("--out", help="write the schedule to this JSON (for the executor)")
    ap.add_argument("--bank", action="store_true",
                    help="bank-to-TRANSLATE schedule (fixed heading, no yaw/drift) "
                         "instead of the yaw-turn schedule")
    ap.add_argument("--maneuver-frac", type=float, default=0.7,
                    help="fraction of each segment used for the bang-bang bank (--bank)")
    args = ap.parse_args()
    if args.bank:
        sched = build_bank_schedule(args.cruise, maneuver_frac=args.maneuver_frac)
        print_bank_schedule(sched, args.cruise)
        kind = "bank"
    else:
        sched = build_schedule(args.cruise, turn_time_s=args.turn_time)
        print_schedule(sched, args.cruise)
        kind = "yaw"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"cruise_mps": args.cruise, "kind": kind, "segments": sched}, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
