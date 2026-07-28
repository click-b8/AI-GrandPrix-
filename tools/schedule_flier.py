#!/usr/bin/env python3
"""Open-loop MOTION-SCHEDULE flier for VQ1 (SET_ATTITUDE_TARGET, no vision/position).

Two modes:

  --calibrate : FIRST run. Hold pitch -18 / thrust 0.29 straight after GO and time the
                active_gate ticks against the known spawn->gate distances -> measured
                cruise m/s (+ how far a pure-straight flight reaches). Feeds --cruise.

  (default)   : Fly the full 6-segment schedule computed at --cruise from the course
                geometry + measured plant (thrust->sink map). Per segment hold
                [thrust, pitch -18, entry turn], advancing on active_gate (primary)
                with each segment's duration as a TIME FALLBACK. Segment 0 carries the
                +2.2 deg right bearing to gate 1 as a small right bank. Hard attitude
                fence + rate clamp on (can't flip).

Reuses the proven arm/TIMESYNC/GO/SET_ATTITUDE_TARGET path and the open-loop flier's
attitude fence/hold + PLANT_RATE_CALIB wire conversion (one rule, same as run_vq1).

    python tools/schedule_flier.py --calibrate            # measure cruise FIRST
    python tools/schedule_flier.py --cruise 8             # then fly the schedule
    python tools/schedule_flier.py --cruise 8 --flip-turns  # if turns go the wrong way live

Lateral/turn SIGNS (roll bank + yaw turns) use the verified LATERAL_SIGN convention
but are the least-certain open-loop term -- if it banks/turns the wrong way, add
--flip-turns. Run ON THE SIM BOX during a real race.
"""
import argparse
import json
import math
import os
import struct
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime

import numpy as np
from pymavlink import mavutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vq1_vision_servo import (  # noqa: E402
    gravity_to_roll_pitch, ServoConfig, MAX_BODY_RATE, LATERAL_SIGN, apply_openloop_tune,
    TubeDetector, GateDetector, SPAWN_PITCH_DEG)
from dcl_mavlink_adapter import wire_body_rate  # noqa: E402
from attitude_filter import GravityEstimator  # noqa: E402
from dcl_vision_receiver import DCLVisionReceiver  # noqa: E402
from tools.motion_schedule import build_bank_schedule, segments  # noqa: E402

G = 9.81


def cumulative_distances():
    cum, run = [], 0.0
    for s in segments():
        run += s["d3d"]
        cum.append(run)
    return cum


def initial_bank_deg(cruise_mps, turn_time_s):
    """Small bank for segment 0's initial bearing to gate 1 (~+2.2 deg RIGHT of the
    spawn heading). Coordinated-turn: phi = atan(dpsi * V / (g * t))."""
    data = json.load(open(os.path.join(ROOT, "course_gates_cm.json"), encoding="utf-8"))
    spawn = data["spawn"]
    spawn_yaw = float(spawn["yaw_deg"])
    import numpy as np
    sp = np.asarray(spawn["pos_cm"], float)
    g0 = np.asarray(data["gates"][0]["pos_cm"], float)
    d = (g0 - sp)
    seg0_brg = math.degrees(math.atan2(d[1], d[0]))
    offset = (seg0_brg - spawn_yaw + 180.0) % 360.0 - 180.0     # + = gate RIGHT of nose
    bank = math.degrees(math.atan(math.radians(offset) * cruise_mps / (G * max(turn_time_s, 1e-3))))
    return offset, bank


class RX(threading.Thread):
    def __init__(self, conn):
        super().__init__(daemon=True)
        self.conn = conn
        self.lock = threading.Lock()
        self.running = True
        self.active_gate = -1
        self.race_started = False
        self.race_finish = 0
        self.countdown_armed = False
        self.armed_race_start_ms = -1
        self.go_wall = None
        self.gravity = (0.0, 0.0, 9.81)
        self.gyro = (0.0, 0.0, 0.0)
        self.grav_est = GravityEstimator()   # gyro-aided, accel magnitude-gated (valid under thrust)
        self._last_imu_t = None
        self.accel_g = 1.0
        self.gate_ticks = []
        self.collisions = []          # (t_since_go, id) -- 1001=Gate, 1002=Env/ground
        self.imu_count = 0            # total HIGHRES_IMU seen (freeze detector)

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
                # Feed the A2 GravityEstimator (same as run_vq1), NOT raw -accel. Under
                # thrust the accelerometer is specific force + kinematic accel, so raw
                # -accel is a GARBAGE gravity estimate (measured: est_pitch decoupled from
                # the gyro, erratic +70/-66/+128 deg/s while gyro held -35). The filter
                # gyro-propagates gravity and only trusts the accel when |accel|~1g, so it
                # stays valid in flight. It does the -accel negation internally.
                accel = (float(msg.xacc), float(msg.yacc), float(msg.zacc))
                gyro = (float(msg.xgyro), float(msg.ygyro), float(msg.zgyro))
                now = time.time()
                if self._last_imu_t is None:
                    self.grav_est.reset(gravity_frd=(-accel[0], -accel[1], -accel[2]))
                    dt = None
                else:
                    dt = now - self._last_imu_t
                self._last_imu_t = now
                g = self.grav_est.update(accel, gyro, dt)
                amag_g = (accel[0]**2 + accel[1]**2 + accel[2]**2) ** 0.5 / 9.81
                with self.lock:
                    self.gravity = (float(g[0]), float(g[1]), float(g[2]))
                    self.gyro = gyro
                    self.accel_g = amag_g   # |accel| in g: outside [0.85,1.15] => accel gated OUT
                    self.imu_count += 1
            elif t == "COLLISION":
                # DCL crash telemetry. A hit here (esp. 1002=Env/ground on the seg1
                # descent onto g1) explains an IMU freeze + ag stuck: the run ended.
                cid = int(getattr(msg, "id", 0))
                with self.lock:
                    tgo = (time.time() - self.go_wall) if self.go_wall else -1.0
                    self.collisions.append((tgo, cid))
                what = {1001: "GATE", 1002: "ENV/ground"}.get(cid, f"id={cid}")
                print(f"[coast] !!! COLLISION {what} at t={tgo:.1f}s (ag={self.active_gate})",
                      flush=True)
            elif t == "ENCAPSULATED_DATA":
                raw = bytes(msg.data)
                if not raw or raw[0] != 1:
                    continue
                try:
                    _, sim_boot, race_start, fin, ag, _ = struct.unpack_from("<BQqqIq", raw)
                except struct.error:
                    continue
                with self.lock:
                    if fin > 0:
                        self.race_finish = int(fin)
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

    def snap(self):
        with self.lock:
            return (self.active_gate, self.race_finish, self.gravity, self.gyro)

    def stop(self):
        self.running = False


def lag_step(k, dt_s, target, tau_s):
    """One tick of a first-order lag from `k` toward `target` with time constant `tau_s`.

    The shared primitive behind every smooth weight in this file: it never steps, it
    approaches from wherever it is (so a reversal mid-transition is continuous), and it
    is symmetric -- the same tau ramps a weight back IN as ramps it out. tau <= 0 means
    no lag at all: take the target immediately.

    dt is clamped the same way the thrust slew limiter clamps it, so one long stall
    between ticks cannot grant an unbounded jump.
    """
    if tau_s <= 0.0:
        return target
    dt_s = min(max(dt_s, 0.0), 0.5)
    return k + (dt_s / (tau_s + dt_s)) * (target - k)


def ff_lat_step(k, dt_s, gate_alive, hold_s, tau_s):
    """CHANGE A -- decay the feed-forward lateral bank to level once the gate is gone.

    Returns the next value of the feed-forward WEIGHT k in [0, 1], which multiplies the
    whole lateral feed-forward (schedule backbone + --post-gate1-bank).

    WHY: filt4.csv holds des_roll_deg at exactly +5.00 for all 1210 ticks after the last
    live detection at t=12.762 -- 7.07 s of commanded bank with nothing watching. A
    feed-forward is only trustworthy while something can correct it; once vision is gone
    the honest command is wings-level, not a standing turn.

    `gate_alive` is True while a detection is live OR still inside the hold window; the
    caller supplies that so this stays a pure function of (k, dt).

    First-order lag, not a closed form off a loss anchor: it decays as exp(-t/tau) from a
    held k=1 either way, and it also ramps the feed-forward back IN over the same tau on
    re-acquisition instead of stepping the bank back on in one tick.
    """
    return lag_step(k, dt_s, 1.0 if gate_alive else 0.0, tau_s)


class RailSignal:
    """LPF + hold-then-fade on the TUBE's (u_tube, curvature) -- the RAIL signal.

    WHY the tube and not the gate (flt9 ADDENDUM): a gate blob LUNGES sideways from
    parallax as you close on it. Measured on filt9's gate-3 leg, raw gate u_err ran
    +0.20 -> +0.42 -> +0.95 over 0.6 s while the tube moved only +0.17 -> +0.23 ->
    +0.37 across the same ticks. The gate law spent 30 of the 46 close-range ticks
    pinned to its -11 deg clamp, steering hard off a decoy. The tube is the rail.

    WHY it needs its own continuity: the tube is found on only ~8% of ticks on that
    leg. Without a hold-then-fade the rail command would chatter on and off at that
    duty cycle. Same shape and the same time constants as the gate measurement filter
    (that filter is NOT modified -- this is a second, parallel one), so a lost rail
    holds briefly and then decays to zero = coast straight, never a step and never a
    slam.

    `k` is the decay weight: 1.0 while live or holding, easing to 0 once the tube has
    been gone longer than the hold. Fading is CLOSED FORM from the values at loss, not
    iteratively from the live value -- iterating compounds every tick and races to
    zero (a bug this file has already been bitten by once, on the v-trim).
    """

    def __init__(self, tau_s, hold_s, decay_s):
        self.tau = max(tau_s, 1e-3)
        self.hold = hold_s
        self.decay = max(decay_s, 1e-3)
        self.u = self.curv = 0.0
        self._u_loss = self._c_loss = 0.0
        self.det_t = None          # last SOLID reading; None = never seen the rail
        self.k = 0.0

    def update(self, now, dt_s, solid, u_tube, curvature):
        if solid:
            if self.det_t is None:
                self.u, self.curv = float(u_tube), float(curvature)   # SEED on first sight
            else:
                a = dt_s / (self.tau + dt_s)
                self.u += a * (float(u_tube) - self.u)
                self.curv += a * (float(curvature) - self.curv)
            self.det_t, self.k = now, 1.0
            self._u_loss, self._c_loss = self.u, self.curv
        elif self.det_t is not None:
            gap = now - self.det_t
            if gap <= self.hold:
                self.k = 1.0                        # HOLD: the rail is simply frozen
            else:
                self.k = math.exp(-(gap - self.hold) / self.decay)
                self.u, self.curv = self._u_loss * self.k, self._c_loss * self.k
                if self.k <= 0.02:                  # DEAD -> coast straight
                    self.u = self.curv = 0.0
        return self.k

    @property
    def alive(self):
        return self.det_t is not None and self.k > 0.02

    def control_u(self, lead):
        """The steering signal: near-band centre + anticipatory curvature lead."""
        return self.u + lead * self.curv


def build_descent_ladder(sched, anchor_bias, anchor_leg=2, scale=1.0, overrides=None):
    """ALTITUDE LADDER -- per-LEG vertical feed-forward, as thrust BELOW the hover
    baseline, indexed by course leg (leg i is flown while active_gate == i).

    WHY a table and not one number: --post-gate1-descent applied ONE value to every leg
    from ag>=1 onward. That number was tuned against the STEEP middle legs (17.1 / 16.2
    deg), and the last two legs are nearly flat (1.9 / 1.5 deg) -- holding the steep-leg
    descent across them flies the drone into the ground past gate 3.

    DERIVED from the course geometry, never hardcoded: each leg gets the anchor's proven
    descent scaled by its own slope against the anchor leg's slope,

        bias[i] = scale * anchor_bias * slope[i] / slope[anchor_leg]

    so regenerating course_gates_cm.json regenerates the ladder. Two rules on top:

    * leg 0 (SPAWN->START) is PINNED to 0.0. That leg is the empirically-proven
      const-thrust START pass and nothing here may touch it. (Its slope is negative
      anyway -- it climbs slightly -- so the formula would give 0 regardless.)
    * a bias may only SINK (>= 0). A climbing leg gets 0, not a thrust boost: the
      vertical feed-forward is open loop, and --gate-vert is what earns altitude back.

    `overrides` is {leg_index: absolute_bias}. An override is taken EXACTLY as given and
    is deliberately NOT multiplied by `scale` -- it is an answer for that leg, not a
    starting point for the one-knob trim.
    """
    if not sched:
        return []
    ref_i = max(0, min(len(sched) - 1, anchor_leg))
    ref = float(sched[ref_i]["slope_deg"])
    if ref <= 0.0:
        raise ValueError(f"--descent-anchor-leg {anchor_leg} has slope {ref:+.1f} deg, "
                         "which is not a descending leg -- it cannot anchor the ladder.")
    out = []
    for r in sched:
        slope = float(r["slope_deg"])
        b = 0.0 if r["seg"] == 0 else scale * anchor_bias * max(slope, 0.0) / ref
        out.append(round(max(b, 0.0), 4))
    for i, v in (overrides or {}).items():
        if 0 <= i < len(out):
            out[i] = float(v)
    return out


def hold_to_norm(cfg, des_roll, des_pitch, yaw_ff, gravity, gyro, pitch_lim_rad_s=None):
    """Gentle attitude hold -> normalised body rates, hard-clamped, with the fence.
    Identical law to OpenLoopFlier. `pitch_lim_rad_s` (default = ol_max_rate_rad_s)
    lets PITCH authority be raised alone while roll/yaw stay anti-flip clamped.
    Returns (roll_n, pitch_n, yaw_n, est_roll_deg, est_pitch_deg, fenced)."""
    est_roll, est_pitch = gravity_to_roll_pitch(gravity)
    p, q, r = float(gyro[0]), float(gyro[1]), float(gyro[2])
    fence = math.radians(cfg.ol_fence_deg)
    fenced = abs(est_roll) > fence or abs(est_pitch - des_pitch) > fence
    if fenced:
        des_roll = 0.0
        yaw_ff = 0.0
    roll_rate = cfg.ol_kp_att * (des_roll - est_roll) - cfg.ol_kd_att * p
    pitch_rate = cfg.ol_kp_att * (des_pitch - est_pitch) - cfg.ol_kd_att * q  # STANDARD (reverted)
    yaw_rate = yaw_ff - cfg.ol_kd_att * r
    lim = cfg.ol_max_rate_rad_s / MAX_BODY_RATE
    plim = (pitch_lim_rad_s or cfg.ol_max_rate_rad_s) / MAX_BODY_RATE
    clip = lambda x, L=lim: max(-L, min(L, x / MAX_BODY_RATE))
    return (clip(roll_rate), clip(pitch_rate, plim), clip(yaw_rate),
            math.degrees(est_roll), math.degrees(est_pitch), fenced)


def send(conn, boot0, thrust, roll_n, pitch_n, yaw_n):
    # time_boot_ms is uint32. Wall clock (time.time) can jump BACKWARD on an NTP sync
    # mid-flight -> negative elapsed -> struct.error ('I' requires 0..4294967295), which
    # crashed the flier at ~t=35s after 139 good sends. Use the MONOTONIC clock (never
    # rewinds) and mask to uint32 so this field can never take the sender down.
    tb = int((time.monotonic() - boot0) * 1000) & 0xFFFFFFFF
    conn.mav.set_attitude_target_send(
        tb, conn.target_system, conn.target_component,
        128, [1.0, 0.0, 0.0, 0.0],
        wire_body_rate("roll", roll_n), wire_body_rate("pitch", pitch_n),
        wire_body_rate("yaw", yaw_n), float(thrust))


def _harness(args):
    """Connect -> heartbeat -> timesync -> arm -> wait GO. Returns (conn, rx, boot0, stop)."""
    conn = mavutil.mavlink_connection(f"udpin:{args.ip}:{args.port}")
    print("[sched] waiting for heartbeat ...")
    conn.wait_heartbeat()
    print(f"[sched] heartbeat sys {conn.target_system}")
    stop = threading.Event()

    def _ts():
        while not stop.is_set():
            conn.mav.timesync_send(int(time.time_ns()), 0)
            time.sleep(0.1)
    threading.Thread(target=_ts, daemon=True).start()
    rx = RX(conn); rx.start()
    boot0 = time.monotonic()   # seconds; send() computes uint32 ms elapsed from this
    conn.mav.command_long_send(conn.target_system, conn.target_component,
                               mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                               0, 1, 0, 0, 0, 0, 0, 0)
    print("[sched] ARM sent; waiting for GO ...")
    t0 = time.time()
    while not rx.race_started:
        if time.time() - t0 > args.go_timeout:
            print("[sched] no GO within timeout."); stop.set(); rx.stop(); return None
        time.sleep(0.02)
    print("[sched] GO.")
    return conn, rx, boot0, stop


def _shutdown(conn, rx, stop):
    for _ in range(5):
        send(conn, 0, 0.0, 0.0, 0.0, 0.0)   # throttle-0 idle so the sim re-arms next race
        time.sleep(0.02)
    conn.mav.command_long_send(conn.target_system, conn.target_component,
                               mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                               0, 0, 0, 0, 0, 0, 0, 0)
    stop.set(); rx.stop()
    print("[sched] throttle-0 + DISARM sent.")


def mode_calibrate(args, cfg):
    h = _harness(args)
    if h is None:
        return 2
    conn, rx, boot0, stop = h
    cum = cumulative_distances()
    des_pitch_deg = args.des_pitch if args.des_pitch is not None else cfg.cruise_pitch_deg
    des_pitch = math.radians(des_pitch_deg)
    plim = args.pitch_max_rate
    print(f"[sched] CALIBRATE: hold pitch {des_pitch_deg} / thrust {args.thrust} / "
          f"pitch_rate_clamp {plim or cfg.ol_max_rate_rad_s} rad/s straight. "
          f"cumulative spawn->gate (m): {[round(c, 1) for c in cum]}")
    last = 0.0
    while time.time() - rx.go_wall < args.max_s:
        ag, fin, grav, gyro = rx.snap()
        if args.const_pitch_rate is not None:
            # OPEN-LOOP: fixed pitch rate, NO attitude feedback. Isolates the plant's
            # natural pitch behavior + the rate->attitude sign from the loop. rate=0 =>
            # pure coast (ride the spawn tilt). If the nose pitches up here with ZERO
            # command, the aero-moment/authority theory is real; if it holds ~-18, the
            # arc was the loop/estimate, and more authority would be wrong.
            rn = yn = 0.0
            pn = max(-1.0, min(1.0, args.const_pitch_rate / MAX_BODY_RATE))
            ep = math.degrees(gravity_to_roll_pitch(grav)[1])
            fenced = False
        else:
            rn, pn, yn, er, ep, fenced = hold_to_norm(cfg, 0.0, des_pitch, 0.0, grav, gyro,
                                                      pitch_lim_rad_s=plim)
        send(conn, boot0, args.thrust, rn, pn, yn)
        now = time.time()
        if now - last >= 0.25:      # 4x/s so a pitch OSCILLATION is resolvable (not aliased)
            last = now
            gate = "accel-IN " if 0.85 <= rx.accel_g <= 1.15 else "accel-OUT"
            print(f"[sched]  t={now-rx.go_wall:5.2f}s ag={ag} est_pitch={ep:+.1f} "
                  f"gyro_q={gyro[1]:+.2f} |a|={rx.accel_g:.2f}g {gate} "
                  f"{'FENCE' if fenced else ''}")
        if ag >= len(cum) or fin > 0:
            break
        time.sleep(1.0 / 250.0)
    _shutdown(conn, rx, stop)

    print("\n================= CRUISE CALIBRATION =================")
    ticks = rx.gate_ticks
    if not ticks:
        print("active_gate never advanced -- pure-straight flight didn't reach gate 0.")
        print("The schedule's seg-0 bank/turns are load-bearing even early.")
    else:
        prev_t = prev_d = 0.0
        print("passed_gate  t_since_go  cum_dist  avg_speed  leg_speed")
        for ag, t in ticks:
            g = ag - 1
            if 0 <= g < len(cum):
                d = cum[g]
                avg = d / t if t > 1e-6 else float("nan")
                leg = (d - prev_d) / (t - prev_t) if (t - prev_t) > 1e-6 else float("nan")
                print(f"    {g:>2}       {t:7.2f}s  {d:7.1f}m  {avg:7.2f}   {leg:7.2f} m/s")
                prev_t, prev_d = t, d
        g = ticks[-1][0] - 1
        if 0 <= g < len(cum):
            v = cum[g] / ticks[-1][1]
            print(f"\n>>> CRUISE ~ {v:.1f} m/s. Next:  python tools/schedule_flier.py --cruise {v:.0f}")
        threaded = max(a for a, _ in ticks)     # highest active_gate reached flying STRAIGHT
        print(f"\n>>> STRAIGHT flight threaded {threaded}/6 gates (no turns commanded).")
        print(">>> Gates sit within +/-3.9 m of the straight line. If this count is HIGH, the")
        print(">>> open-loop TURNS are likely UNNECESSARY and net-harmful (they risk >10 deg")
        print(">>> compounding heading drift). If it stalls early at g2/g4 (the -3.6/-3.9 m")
        print(">>> gates), only those need a heading correction -- report the count.")
    print("=====================================================")
    return 0


def mode_schedule(args, cfg):
    sched = build_bank_schedule(args.cruise, cfg=cfg, maneuver_frac=args.maneuver_frac)
    sign = -1.0 if args.flip_turns else 1.0
    print(f"[sched] FLY bank-to-translate schedule @ cruise {args.cruise} m/s "
          f"(FIXED heading, no yaw). flip={args.flip_turns}")
    for r in sched:
        print(f"[sched]   seg{r['seg']} {r['from'][:5]}->{r['to'][:5]} lat={r['lateral_m']:+.2f}m "
              f"bank={r['bank_deg']:+.1f} t_man={r['t_man_s']:.2f} thr={r['thrust']:.3f} "
              f"dur={r['duration_s']:.2f}")

    h = _harness(args)
    if h is None:
        return 2
    conn, rx, boot0, stop = h

    seg = 0
    seg_t0 = rx.go_wall
    last_log = 0.0
    while True:
        ag, fin, grav, gyro = rx.snap()
        now = time.time()
        if fin > 0 or ag > 5:
            print(f"[sched] *** COMPLETE *** active_gate={ag} finish={fin}")
            break
        if now - rx.go_wall > args.max_s:
            print(f"[sched] max flight {args.max_s}s -- stopping. seg={seg} ag={ag}")
            break
        # segment pointer: active_gate LEADS (position resync); duration*mult time fallback
        row = sched[seg]
        if ag > seg and ag <= 5:
            seg = ag; seg_t0 = now
            print(f"[sched] >>> seg -> {seg} (active_gate) t={now-rx.go_wall:.1f}s")
            row = sched[seg]
        elif (now - seg_t0) > row["duration_s"] * args.fallback_mult and seg < 5:
            seg += 1; seg_t0 = now
            print(f"[sched] >>> seg -> {seg} (TIME fallback) t={now-rx.go_wall:.1f}s")
            row = sched[seg]

        t_in = now - seg_t0
        # BANG-BANG bank: +phi first half of t_man, -phi second half, then level. Heading
        # never changes (yaw always 0). LATERAL_SIGN maps right-positive bank -> sim roll.
        t_man = row["t_man_s"]
        bank_rad = math.radians(row["bank_deg"])           # signed, right positive
        if t_in < t_man / 2.0:
            des_roll = LATERAL_SIGN * sign * bank_rad
            phase = "bank"
        elif t_in < t_man:
            des_roll = -LATERAL_SIGN * sign * bank_rad      # counter-bank: null lateral vel
            phase = "counter"
        else:
            des_roll = 0.0
            phase = "level"
        des_pitch = math.radians(row["pitch_deg"])
        rn, pn, yn, er, ep, fenced = hold_to_norm(cfg, des_roll, des_pitch, 0.0, grav, gyro)
        send(conn, boot0, row["thrust"], rn, pn, yn)

        if now - last_log >= 0.25:
            last_log = now
            print(f"[sched] seg{seg} {phase:7s} t_in={t_in:4.1f}/{row['duration_s']:.1f}s ag={ag} "
                  f"thr={row['thrust']:.3f} desRoll={math.degrees(des_roll):+.1f} "
                  f"estP={ep:+.1f} estR={er:+.1f} {'FENCE' if fenced else ''}")
        time.sleep(1.0 / 250.0)

    _shutdown(conn, rx, stop)
    print(f"[sched] final active_gate={rx.active_gate} gates passed={max(rx.active_gate,0)}/6")
    return 0


def mode_coast_schedule(args, cfg):
    """The PROVEN coast (pitch rate = 0, open-loop) + per-segment THRUST (sink map) and
    a gentle roll-hold BANK, both stepped by active_gate. Pitch is never fed back --
    only thrust and lateral bank are scheduled. This is the coast that passed START,
    extended to thread the descending, laterally-offset course."""
    sched = build_bank_schedule(args.cruise, cfg=cfg, maneuver_frac=args.maneuver_frac)
    # ANCHORED thrust (the sink-map absolutes disagree with the measured coast: 0.275 was
    # ~level, map implies ~0.9 m/s sink). Anchor to the measured level thrust (0 sink) and
    # use ONLY the map slope: thr = level - sink_needed/sink_slope. (This was mistakenly
    # only in mode_coast_tube before -- the bug that made --coast-schedule fly raw thrust.)
    for r in sched:
        sink = args.cruise * math.tan(math.radians(r["slope_deg"]))
        # Descending course: the anchor may only LOWER thrust (to sink), never RAISE it
        # above the measured level. A negative slope (slight climb, e.g. seg0 -1.3deg to
        # START which is +0.51m) otherwise pushes thr to 0.281 and flew OVER START -- but
        # 0.275 is the PROVEN thrust that passes START and ticks ag->1. Cap at level.
        thr = args.level_thrust - max(sink, 0.0) / args.sink_slope
        r["thrust"] = round(max(cfg.ol_thrust_lo, min(cfg.ol_thrust_hi, thr)), 3)
    sched[0]["thrust"] = args.level_thrust  # seg0 = the empirically-proven START pass
    sign = -1.0 if args.flip_turns else 1.0
    print(f"[coast] cruise {args.cruise} m/s, thrust_bias {args.thrust_bias:+.3f}, "
          f"PITCH=coast(rate0), roll-hold bank, flip={args.flip_turns}")
    for r in sched:
        print(f"[coast]   seg{r['seg']} {r['from'][:5]}->{r['to'][:5]} "
              f"thr={r['thrust']+args.thrust_bias:.3f} bank={r['bank_deg']:+.1f} "
              f"t_man={r['t_man_s']:.2f} dur={r['duration_s']:.2f}")
    # --- BOUNDED TUBE TRIM (the one new loop) --------------------------------------
    # The fixed banks stay the open-loop backbone; the tube may only NUDGE the residual
    # with hard-clamped authority, so a dropout or a spike stops trimming instead of
    # veering. Off unless --tube-trim is passed: the bare schedule is unchanged.
    vision = tube_det = None
    if args.tube_trim:
        vision = DCLVisionReceiver(port=args.vision_port)
        vision.start()
        tube_det = TubeDetector(cfg)
        print(f"[coast] TUBE TRIM on: area_min={args.tube_trim_area_min:.3f} "
              f"k={args.k_tube_trim} lead={cfg.k_tube_lead} "
              f"clamp={args.trim_clamp_deg:+.1f}deg k_att={args.k_trim_att} "
              f"phase={'all' if args.trim_during_pulse else 'coast-only'}")

    h = _harness(args)
    if h is None:
        if vision is not None:
            vision.stop()
        return 2
    conn, rx, boot0, stop = h
    seg, seg_t0, last = 0, rx.go_wall, 0.0
    prev_imu = 0
    lim = cfg.ol_max_rate_rad_s / MAX_BODY_RATE
    trim_clamp = math.radians(args.trim_clamp_deg)
    last_frame_n, tube = -1, None
    n_tick = n_det = 0
    rate_t0, hz, det_hz = time.time(), 0.0, 0.0
    tick_fh = None
    if args.tick_log:
        tick_fh = open(args.tick_log, "w", encoding="utf-8")
        tick_fh.write("t,seg,active_gate,phase,thrust,roll_n,trim_n,des_trim_deg,"
                      "tube_found,u_tube,curvature,area_frac,trim_on,"
                      "est_roll_deg,est_pitch_deg,loop_hz,det_hz\n")
        print(f"[coast] per-tick log -> {args.tick_log}")
    while True:
        ag, fin, grav, gyro = rx.snap()
        now = time.time()
        n_tick += 1
        if now - rate_t0 >= 1.0:
            hz, det_hz = n_tick / (now - rate_t0), n_det / (now - rate_t0)
            n_tick = n_det = 0
            rate_t0 = now
        if fin > 0 or ag > 5:
            print(f"[coast] *** COMPLETE *** active_gate={ag} finish={fin}"); break
        if now - rx.go_wall > args.max_s:
            print(f"[coast] max flight {args.max_s}s -- stop. seg={seg} ag={ag}"); break
        row = sched[seg]
        if ag > seg and ag <= 5:
            seg = ag; seg_t0 = now; row = sched[seg]
            print(f"[coast] >>> seg -> {seg} (active_gate) t={now-rx.go_wall:.1f}s")
        elif (now - seg_t0) > row["duration_s"] * args.fallback_mult and seg < 5:
            seg += 1; seg_t0 = now; row = sched[seg]
            print(f"[coast] >>> seg -> {seg} (TIME fallback) t={now-rx.go_wall:.1f}s")

        # bang-bang bank ANGLE via a GENTLE roll hold; PITCH stays pure coast (rate 0).
        # Segments below --banks-from (or under --no-steer) fly PURE COAST on roll too
        # (roll cmd = 0, NOT a wings-level hold) -- the exact lateral behavior of the
        # standalone coast that passed gate 1. This protects that pass and lets banks be
        # added one segment at a time.
        est_roll, est_pitch = gravity_to_roll_pitch(grav)
        t_in = now - seg_t0
        if args.no_steer or seg < args.banks_from or seg > args.banks_to:
            roll_n = 0.0; roll_cmd = 0.0; phase = "coast"
        else:
            # OPEN-LOOP ROLL PULSE (NO level-hold): +rate for pulse_s to bank toward the
            # offset, -rate for pulse_s to level back, then roll=0 pure coast for the rest.
            # Coast is preserved everywhere except the brief slide -- this deliberately
            # avoids the wings-level hold that perturbed the gate-1 pass.
            target = LATERAL_SIGN * sign * math.radians(row["bank_deg"])   # bank angle to reach
            prate = target / max(args.pulse_s, 1e-3)                        # rad/s to reach it
            if t_in < args.pulse_s:
                roll_cmd = prate; phase = "roll+"
            elif t_in < 2.0 * args.pulse_s:
                roll_cmd = -prate; phase = "roll-"
            else:
                roll_cmd = 0.0; phase = "coast"      # banked -> slid -> leveled, now pure coast
            roll_n = max(-lim, min(lim, roll_cmd / MAX_BODY_RATE))

        # --- the ONE new loop: bounded tube trim on top of the open-loop bank -------
        # roll_n is a RATE, so the trim CANNOT be a raw rate addition -- a sustained
        # rate integrates without bound and "+/-5 deg of bank" would be meaningless.
        # Instead the tube sets a bank-ANGLE offset (hard-clamped to trim_clamp) and we
        # command the rate that closes onto it. Authority is then bounded by the ANGLE,
        # which is what "nudge, never slam" actually requires.
        # Applied only in the bang-bang's coast phase by default: there the open-loop
        # intent is bank=0, so est_roll IS the residual the tube should trim. During the
        # pulses the banks own the roll axis and the trim stays out of the way.
        des_trim = trim_n = 0.0
        trim_on = False
        if tube_det is not None:
            frame_n = vision.frames_received
            if frame_n != last_frame_n or tube is None:
                last_frame_n = frame_n
                tube = tube_det.measure(vision.get_latest_frame())
                n_det += 1
            if (tube.found and tube.area_frac >= args.tube_trim_area_min
                    and (args.trim_during_pulse or phase == "coast")):
                u = LATERAL_SIGN * sign * (tube.u_tube + cfg.k_tube_lead * tube.curvature)
                des_trim = max(-trim_clamp, min(trim_clamp, args.k_tube_trim * u))
                trim_rate = args.k_trim_att * (des_trim - est_roll)      # rad/s onto it
                trim_n = max(-lim, min(lim, trim_rate / MAX_BODY_RATE))
                trim_on = True
            # NOT found / too weak / wrong phase -> trim_n stays 0.0: pure coast, NOT a
            # wings-level hold. A dropout can only stop trimming, never command a roll.
        roll_n = max(-lim, min(lim, roll_n + trim_n))

        pitch_n = 0.0                                  # COAST -- no pitch feedback, ever
        thrust = float(max(cfg.ol_thrust_lo, min(cfg.ol_thrust_hi, row["thrust"] + args.thrust_bias)))
        send(conn, boot0, thrust, roll_n, pitch_n, 0.0)

        if tick_fh is not None:
            tick_fh.write(
                f"{now - rx.go_wall:.4f},{seg},{ag},{phase},{thrust:.4f},"
                f"{roll_n:+.5f},{trim_n:+.5f},{math.degrees(des_trim):+.3f},"
                f"{1 if (tube is not None and tube.found) else 0},"
                f"{(tube.u_tube if tube is not None else 0.0):+.5f},"
                f"{(tube.curvature if tube is not None else 0.0):+.5f},"
                f"{(tube.area_frac if tube is not None else 0.0):.5f},{int(trim_on)},"
                f"{math.degrees(est_roll):+.3f},{math.degrees(est_pitch):+.3f},"
                f"{hz:.1f},{det_hz:.1f}\n")

        if now - last >= 0.25:
            last = now
            # imu delta since last print: 0 => HIGHRES_IMU stream stalled (sim ended/crashed),
            # which is why |a|/estP flatline. fin>0 => race registered a finish.
            dimu = rx.imu_count - prev_imu; prev_imu = rx.imu_count
            frozen = " IMU-FROZEN" if dimu == 0 else ""
            trim_s = ("" if tube_det is None else
                      (f" | TRIM on des={math.degrees(des_trim):+.1f}deg "
                       f"tn={trim_n:+.3f} area={tube.area_frac:.3f}" if trim_on else
                       f" | TRIM off (found={int(tube.found) if tube else 0} "
                       f"area={tube.area_frac if tube else 0.0:.3f} {phase})"))
            print(f"[coast] seg{seg} {phase} t_in={t_in:4.1f}/{row['duration_s']:.1f} ag={ag} "
                  f"thr={thrust:.3f} rollCmd={roll_n*MAX_BODY_RATE:+.2f}rad/s "
                  f"estR={math.degrees(est_roll):+.1f} estP={math.degrees(est_pitch):+.1f} "
                  f"|a|={rx.accel_g:.2f}g dimu={dimu} fin={fin}{frozen}"
                  f"{trim_s} loop={hz:.0f}Hz det={det_hz:.0f}Hz")
        time.sleep(1.0 / 250.0)
    if tick_fh is not None:
        tick_fh.close()
    if vision is not None:
        vision.stop()
    _shutdown(conn, rx, stop)
    print(f"[coast] final active_gate={rx.active_gate}  gates passed={max(rx.active_gate,0)}/6")
    return 0


def mode_coast_tube(args, cfg):
    """The ONE-closed-loop flier: coast pitch (rate 0) + open-loop thrust schedule +
    CLOSED-LOOP lateral steering on the BLUE TUBE (the racing line). The tube is the
    only feedback -- it senses the actual course, so it self-corrects, unlike the blind
    dead-reckon banks. Geometry banks are dropped entirely."""
    sched = build_bank_schedule(args.cruise, cfg=cfg, maneuver_frac=args.maneuver_frac)  # banks + slope
    # ANCHORED thrust: the sink-map absolutes disagree with the measured coast (0.275 was
    # ~level, map implies ~0.9 m/s sink there). So anchor to the measured level thrust
    # (0 sink) and use ONLY the map's slope: thrust = level - sink_needed / sink_slope.
    for r in sched:
        sink = args.cruise * math.tan(math.radians(r["slope_deg"]))   # + = descend
        thr = args.level_thrust - max(sink, 0.0) / args.sink_slope    # anchor may only LOWER
        r["thrust"] = round(max(cfg.ol_thrust_lo, min(cfg.ol_thrust_hi, thr)), 3)
    sched[0]["thrust"] = args.level_thrust  # seg0 = the empirically-proven START pass
    sign = -1.0 if args.flip_turns else 1.0
    # --tube-steer-area-min defines a SOLID tube reading for the LOG only -- the tube
    # has no steering authority on any leg (flt10 broke gate 1, flt12 crashed into it).
    # GATE thresholds. size_frac is resolution-invariant (sqrt(area)/H), so the half-res
    # gate detect is directly comparable to full res. MEASURED on base.csv/hold.csv:
    # size>=0.10 holds on 7-14% of flight (p50 ~0.07-0.09, max 0.42).
    # These now bound the gate FINE-TRIM (and the --lateral-gate-centring revert path),
    # not the primary steering term -- the rail owns that.
    gate_bank = math.radians(args.gate_max_bank_deg)
    # Per-leg VISION clamp for the ag==2 leg. Defaults to the global one, so unless it is
    # passed the command is bit-identical everywhere. Tightening it lets the ag==2
    # feed-forward (--post-gate2-bank) LEAD and leaves vision only the residual.
    gate_bank_ag2 = math.radians(args.gate_max_bank_ag2
                                 if args.gate_max_bank_ag2 is not None
                                 else args.gate_max_bank_deg)
    gate_size_min = args.gate_bank_size_min
    # CONST THRUST: replaces the whole sink-map schedule with one number. Segments still
    # advance (they drive seg tracking / the time fallback) but their thrust is ignored.
    const_thrust = None
    if args.const_thrust is not None:
        const_thrust = float(max(cfg.ol_thrust_lo, min(cfg.ol_thrust_hi, args.const_thrust)))
        if abs(const_thrust - args.const_thrust) > 1e-9:
            print(f"[tube] *** --const-thrust {args.const_thrust:.3f} CLAMPED to "
                  f"{const_thrust:.3f} (ol bounds {cfg.ol_thrust_lo}..{cfg.ol_thrust_hi}) ***")
        if abs(args.thrust_bias) > 1e-9:
            print(f"[tube] note: --thrust-bias {args.thrust_bias:+.3f} IGNORED under "
                  f"--const-thrust (X is used exactly).")
    # --- FEED-FORWARD BACKBONE tables, one entry per segment ------------------------
    # LATERAL: the schedule's own bank_deg, bracketed by --banks-from/--banks-to.
    # VERTICAL: the sink each leg's slope needs, as a thrust DELTA below level. Taken
    # from the map SLOPE only (level_thrust - sched_thrust), never the map's absolute
    # value -- the absolutes are what put 0.135 on the steep legs and dropped the drone.
    ff_on = args.ff_backbone
    ff_bank_deg = [(r["bank_deg"] if args.banks_from <= r["seg"] <= args.banks_to else 0.0)
                   if ff_on else 0.0 for r in sched]
    # SIGN: RAW, the same path --post-gate1-bank used (positive bank_deg -> positive
    # des_roll -> LEFT), NOT the LATERAL_SIGN negation mode_coast_schedule applies to
    # these same numbers. Those two disagree, and the flight evidence decides it: on the
    # leg after ag==1, gate centring -- closed loop, self-correcting -- commanded desR
    # ~ +11 LEFT, and the schedule's seg-1 bank is +10.8. Raw matches the eye; negated
    # would command a 10.8 deg turn into the opposite wall. --flip-turns still inverts
    # the whole backbone in one shot if the sim disagrees.
    ff_bank_rad = [sign * math.radians(b) for b in ff_bank_deg]
    # DESCENT delta from the UNCLAMPED slope thrust. Reading it off r["thrust"] instead
    # silently loses the slope: that value is already clamped at cfg.ol_thrust_lo (0.18),
    # so segs 1/2/3 -- slopes 12.1/17.1/16.2, genuinely different descents -- all
    # collapsed to an identical -0.095. --ff-descent-floor is the ONLY floor here.
    ff_thrust_delta = []
    for r in sched:
        sink = args.cruise * math.tan(math.radians(r["slope_deg"]))   # + = descend
        ff_thrust_delta.append(-max(sink, 0.0) / args.sink_slope)     # <= 0, may only sink
    ff_thrust_delta[0] = 0.0        # seg0 = the empirically-proven START pass, level
    ff_bank_max = math.radians(args.ff_max_bank_deg)
    # --- ALTITUDE LADDER (replaces the single post-gate-1 descent) -------------------
    # --post-gate1-descent is now the ANCHOR of the table rather than the value flown on
    # every leg, so the proven 0.026 stays on the command line in the same place and
    # keeps its meaning on the leg it was tuned for.
    desc_overrides = {}
    for _i in range(len(sched)):
        _v = getattr(args, f"descent_bias_leg{_i}", None)
        if _v is not None:
            desc_overrides[_i] = float(_v)
    descent_bias = build_descent_ladder(sched, args.post_gate1_descent,
                                        anchor_leg=args.descent_anchor_leg,
                                        scale=args.descent_scale,
                                        overrides=desc_overrides)
    # --ff-backbone still owns the vertical axis where it is on; the ladder is the
    # non-backbone path. It is "on" whenever any leg actually asks for a descent.
    ladder_on = (not ff_on) and any(b > 0.0 for b in descent_bias)
    # Between gates the LADDER owns the sink, so the v-trim decays to ZERO -- i.e. it
    # HOLDS the leg's feed-forward baseline instead of coasting anywhere.
    # --gate-vert-coast-bias would add a second, leg-INDEPENDENT descent on top of the
    # per-leg one: exactly the flat-leg over-descent this table exists to remove.
    coast_bias = 0.0 if ladder_on else args.gate_vert_coast_bias
    if ladder_on and abs(args.gate_vert_coast_bias) > 1e-9:
        print(f"[tube] *** --gate-vert-coast-bias {args.gate_vert_coast_bias:+.3f} FORCED "
              f"to 0.000: the altitude ladder owns the between-gate descent and a "
              f"leg-independent bias on top of it double-counts. ***")
    steepest = max(abs(b) for b in ff_bank_deg) if ff_on else 0.0
    if ff_on:
        print(f"[tube] FF LATERAL SIGN = RAW (+bank -> +desR -> LEFT), matching the "
              f"desR~+11 LEFT that gate centring commanded on the seg-1 leg whose "
              f"schedule bank is +10.8. NOTE --coast-schedule NEGATES these same numbers "
              f"(LATERAL_SIGN={LATERAL_SIGN:+.0f}); the two modes disagree ON PURPOSE. "
              f"If it turns into the wall, --flip-turns.")
    if ff_on and steepest > args.ff_max_bank_deg + 1e-9:
        print(f"[tube] *** WARNING: --ff-max-bank-deg {args.ff_max_bank_deg:.0f} is BELOW "
              f"the schedule's steepest leg ({steepest:.1f}) -- that leg's backbone will "
              f"be TRUNCATED and it will under-turn. ***")
    pitch_s = "PURE COAST (hold removed)"
    thrust_s = (f"CONST {const_thrust:.3f} (sink-map schedule OVERRIDDEN)"
                if const_thrust is not None else "sched(open-loop)")
    if ff_on:
        thrust_s += (f" | FF DESCENT per-seg, floor {args.ff_descent_floor:.3f}"
                     f"{'' if not args.post_gate1_descent else f' [altitude ladder anchor {args.post_gate1_descent:.3f} SUPERSEDED]'}")
    elif ladder_on:
        thrust_s += (f" | ALTITUDE LADDER per-leg [{', '.join(f'{b:.3f}' for b in descent_bias)}]"
                     f" anchor {args.post_gate1_descent:.3f} @ leg{args.descent_anchor_leg} "
                     f"({sched[args.descent_anchor_leg]['slope_deg']:+.1f}deg) "
                     f"scale {args.descent_scale:.2f}"
                     f"{f' overrides {desc_overrides}' if desc_overrides else ''}"
                     f" (leg index = {'SEG' if args.post_gate1_on_seg else 'ACTIVE_GATE'})")
    if args.thrust_slew or args.thrust_slew_down:
        thrust_s += (f" | slew up {args.thrust_slew:.2f}/s "
                     f"down {args.thrust_slew_down:.2f}/s")
    if args.gate_vert:
        thrust_s += (f" + GATE-VERT trim [{-args.gate_vert_down_auth:+.3f},"
                     f"{args.gate_vert_up_auth:+.3f}] kv={args.k_thrust_v} "
                     f"kdv={args.kd_v} size>={args.gate_vert_size_min:.2f} "
                     f"dead-decay={args.gate_vert_decay_s}s "
                     f"-> bias {coast_bias:+.3f}"
                     f"{' (= HOLD the leg baseline)' if ladder_on else ''}")
    filt_s = (f"GATE FILTER tau={args.gate_filter_tau:.2f}s hold={args.gate_filter_hold_s:.2f}s "
              f"fade={args.gate_filter_decay_s:.2f}s (shared by BOTH loops)")
    if args.kd_lat > 0.0:
        print(f"[tube] *** WARNING: --kd-lat {args.kd_lat:.2f} is ON. flt6 measured this "
              f"NET-NEGATIVE: gate-2 lateral error is MONOTONIC, so D reinforces P and "
              f"drove the bank to the clamp (gate#2 u_err +0.919 vs +0.843 without). "
              f"Default is 0.0. ***")
    leg1_s = ""
    if args.post_gate1_bank:
        leg1_trig = "seg" if args.post_gate1_on_seg else "ag"
        leg1_dir = "LEFT" if args.post_gate1_bank > 0 else "RIGHT"
        leg1_s += (f" | LEG1 BANK {args.post_gate1_bank:+.1f}deg {leg1_dir} "
                   f"(on {leg1_trig}==1, summed with centring then clamped)")
    if args.post_gate2_bank or args.gate_max_bank_ag2 is not None:
        leg1_s += (f" | LEG2 (ag==2) PRE-TURN {args.post_gate2_bank:+.1f}deg "
                   f"{'LEFT' if args.post_gate2_bank > 0 else 'RIGHT' if args.post_gate2_bank < 0 else '--'}"
                   f", vision clamp {math.degrees(gate_bank_ag2):.0f}deg"
                   f"{' (TIGHTER than the global %.0f -- feed-forward leads)' % args.gate_max_bank_deg if math.degrees(gate_bank_ag2) < args.gate_max_bank_deg - 1e-9 else ''}")
    steer_s = (f"GATE-CENTRING(closed) k={args.k_gate_bank} "
               f"clamp {args.gate_max_bank_deg:.0f}deg size>={gate_size_min:.2f} "
               f"-- the ONLY steering law. TUBE DETECTOR "
               f"{'ON (--log-tube: +13.5 ms/frame, EXPECT LOWER/CHOPPIER LOOP RATE)'
                  if args.log_tube else 'OFF (not computed at all)'}"
               f"; the tube has had ZERO authority since flt12 (its handoff CRASHED "
               f"gate 1 -- curvature lead spiked the rail to -11 deg at close range)")
    if args.gate_commit_size > 0.0:
        steer_s += (f" + CLOSE-RANGE COMMIT, ag>={args.gate_commit_from_ag} ONLY: "
                    f"sz_f>={args.gate_commit_size:.2f} slews "
                    f"{'the VISION TRIM' if args.gate_commit_vision_only else 'des_roll'}"
                    f" -> 0 over tau={args.gate_commit_tau:.2f}s (reversible, not a "
                    f"latch). ag<{args.gate_commit_from_ag} UNTOUCHED: commit_k forced to "
                    f"exactly 1.0 = bit-for-bit the --gate-commit-size 0 law (flt9's; "
                    f"NOT flt11's, which ran the commit unscoped on ag 0-1)")
        if args.gate_commit_from_ag <= 1:
            print(f"[tube] *** WARNING: --gate-commit-from-ag {args.gate_commit_from_ag} "
                  f"lets the commit alter the GATE-1/GATE-2 approaches, which pass "
                  f"TODAY. Default is 2. ***")
    else:
        steer_s += " + commit DISABLED (--gate-commit-size 0)"
    lat_s = (f"FF BACKBONE (schedule banks, segs {args.banks_from}..{args.banks_to}, "
             f"fence {args.ff_max_bank_deg:.0f}deg, gated on "
             f"{'SEG (incl. TIME fallback -- may bank mid-approach)' if args.post_gate1_on_seg else 'ACTIVE_GATE'}"
             f") + {steer_s}"
             if ff_on else f"{steer_s}, NO backbone")
    print(f"[tube] cruise {args.cruise} m/s bias {args.thrust_bias:+.3f} | PITCH={pitch_s}, "
          f"THRUST={thrust_s}, LATERAL={lat_s} flip={args.flip_turns}"
          f"{leg1_s} | {filt_s}"
          f"{' | TUBE FILTER shares those constants (log-only)' if args.log_tube else ''} | "
          f"kp_att={cfg.ol_kp_att} kd_att={cfg.ol_kd_att}")
    # Per-segment backbone table: EXACTLY what the open loop will command, including
    # where the floor bites, printed before the flight so it is checkable on the ground.
    for r in sched:
        i = r["seg"]
        if ff_on:
            raw = (const_thrust if const_thrust is not None
                   else r["thrust"] + args.thrust_bias) + ff_thrust_delta[i]
            thr_show = max(args.ff_descent_floor, raw)
            floored = " FLOOR" if thr_show > raw + 1e-9 else ""
            d = math.degrees(ff_bank_rad[i])
            bank_show = (f"bank {ff_bank_deg[i]:+6.1f} -> desR {d:+6.1f} "
                         f"({'LEFT ' if d > 0 else 'RIGHT' if d < 0 else '  -- '}"
                         f"{', lat_m ' + format(float(r['lateral_m']), '+.1f') if 'lateral_m' in r else ''})")
        else:
            base = const_thrust if const_thrust is not None else r["thrust"] + args.thrust_bias
            thr_show = base - descent_bias[i]
            raw, floored, bank_show = thr_show, "", "bank    -- (backbone off)"
        delta = -descent_bias[i] if not ff_on else ff_thrust_delta[i]
        over = " OVERRIDE" if (not ff_on and i in desc_overrides) else ""
        print(f"[tube]   seg{i} {r['from'][:5]:>5}->{r['to'][:5]:<5} "
              f"slope {r['slope_deg']:+5.1f} d{delta:+.4f}{over} "
              f"thr {thr_show:.3f}{floored:6} | {bank_show}")

    # PITCH HOLD: REMOVED. Every variant (spawn-referenced drift arrest, seg-gated,
    # the nose-up slowdown) destabilised the flight, so the axis is pure coast and the
    # setpoint machinery is gone rather than left as a switch that can be flipped back.
    # --hold-pitch / --hold-pitch-from-seg are accepted but inert; --no-pitch-hold is
    # now the only behaviour.
    if args.hold_pitch is not None or args.hold_pitch_from_seg:
        print("[tube] *** --hold-pitch/--hold-pitch-from-seg IGNORED: the old pitch hold "
              "is removed. Use --pitch-hold-deg (CHANGE D). ***")
    # --- CHANGE D: gentle near-equilibrium pitch hold (slowdown) --------------------
    des_pitch_hold = None
    pitch_lim = math.radians(args.pitch_max_rate_deg) / MAX_BODY_RATE
    if args.pitch_hold_deg is not None:
        des_pitch_hold = math.radians(args.pitch_hold_deg)
        off = args.pitch_hold_deg - SPAWN_PITCH_DEG
        cmd0 = abs(args.pitch_kp * math.radians(off)) / MAX_BODY_RATE
        print(f"[tube] CHANGE D pitch hold @ {args.pitch_hold_deg:+.1f} deg "
              f"({off:+.1f} from the {SPAWN_PITCH_DEG:.1f} spawn equilibrium) "
              f"kp={args.pitch_kp} kd={args.pitch_kd} "
              f"rate clamp +/-{args.pitch_max_rate_deg:.0f} deg/s "
              f"({pitch_lim:.4f} norm) | standing command at equilibrium = "
              f"{cmd0:.4f} norm = {100 * cmd0 / pitch_lim:.0f}% of that clamp")
        if abs(off) > 5.0:
            print(f"[tube] *** WARNING: {off:+.1f} deg off equilibrium. -10 (7.8 off) "
                  f"TUMBLED at kp 3.0. Change D is specified near-equilibrium. ***")
        if args.pitch_kp > 1.5:
            print(f"[tube] *** WARNING: --pitch-kp {args.pitch_kp} is not the SOFT gain "
                  f"Change D specifies (1.0); 3.0 is what tumbled. ***")

    vision = DCLVisionReceiver(port=args.vision_port)
    vision.start()
    # TUBE DETECTOR: OFF unless --log-tube. It has had zero control authority since the
    # rail was removed, but it was still running EVERY new frame at FULL resolution --
    # MEASURED on vision_frame.png, 13.47 ms/call against the gate detector's 5.45, i.e.
    # 71% of the per-frame vision budget spent on a signal nothing reads. That cost is
    # what drove the loop-rate variance (34.9 Hz median but a 16.2 Hz floor on the
    # gate-2 leg), and an inconsistent loop rate is what makes gate 2 inconsistent.
    tube_det = TubeDetector(cfg) if args.log_tube else None
    # GATE channel -- INSTRUMENTATION ONLY on flight 1 (no thrust authority).
    # Half-res (320x180) with dilate radius 2: GateDetector is 20 ms/call at full res,
    # which on top of TubeDetector's 12.8 ms would drop this loop to ~30 Hz and
    # destabilise the pitch hold we are adding. Measured on vision_frame.png (7/27
    # 19:16): half-res r=2 gives v_err -0.0325 vs the full-res r=3 truth of -0.0294,
    # at 5.1 ms. r=1 SPLITS gate 1 on the cyan tube glow (v_err -0.071, 2.4x wrong);
    # third-res merges gate 1 with gate 2 and flips the sign. Tube stays FULL res so
    # tube->roll is numerically unchanged.
    gate_cfg = replace(cfg, gate_dilate_px=2)
    gate_det = GateDetector(gate_cfg)
    gate_prefer = None          # last ACCEPTED (u, v, size) -> near/far gate continuity

    h = _harness(args)
    if h is None:
        vision.stop(); return 2
    conn, rx, boot0, stop = h
    seg, seg_t0, last = 0, rx.go_wall, 0.0
    lim = cfg.ol_max_rate_rad_s / MAX_BODY_RATE
    # Detection is FRAME-RATE work, not loop-rate work: re-running a detector on an
    # unchanged frame returns identical numbers, so gating on frames_received is an
    # exact numerical no-op that buys back the budget the gate channel costs.
    last_frame_n, tube, gate = -1, None, None
    far_rej = False       # persists between frames, like `gate` itself
    ff_lat_k = 1.0        # CHANGE A: feed-forward lateral weight, 1 = full backbone
    commit_k = 1.0        # CLOSE-RANGE COMMIT weight, 1 = full lateral authority
    # --- gate -> THRUST vertical state (--gate-vert) --------------------------------
    # v_filt/last_v_t drive the same dirty-derivative HybridController uses
    # (deriv_tau_s). vtrim is the LAST gate-driven trim; last_gate_t stamps when we last
    # had a usable gate, so the no-gate path can hold briefly then decay to the coast
    # descent bias rather than snapping back to level thrust between gates.
    vtrim, gate_trim, last_gate_t = 0.0, 0.0, None
    # --- GATE MEASUREMENT FILTER (shared by BOTH loops) ------------------------------
    # One EMA on (u_err, v_err, size_frac) at --gate-filter-tau, updated every tick
    # against the most recent detection. Both axes read the SAME filtered numbers, so
    # they move together and the same approach reproduces the same path instead of each
    # loop chasing its own per-frame flicker.
    # u_f/v_f/sz_f  = filtered signal;  v_f2 = second-stage EMA of v_f, so the vertical
    # derivative is (v_f - v_f2)/tau -- a dirty derivative of an ALREADY-smooth signal
    # rather than of raw detections.
    # *_loss        = the filtered values at the moment the gate went away, so the decay
    #                 is CLOSED FORM from a fixed anchor. Decaying iteratively from the
    #                 live value compounds every tick and races to zero (a bug this file
    #                 has already been bitten by once, on the v-trim).
    u_f = v_f = sz_f = v_f2 = u_f2 = 0.0
    u_loss = v_loss = sz_loss = v2_loss = u2_loss = 0.0
    # --- RAIL (tube) signal: its OWN filter, same constants, separate state -----------
    # The gate filter above is untouched; this is a parallel one so the two signals can
    # be lost and re-acquired independently (on filt9's gate-3 leg the gate was live for
    # most of the leg while the tube was solid on ~8% of ticks).
    rail = (RailSignal(args.gate_filter_tau, args.gate_filter_hold_s,
                       args.gate_filter_decay_s) if args.log_tube else None)
    filt_t = None               # last filter update (dt source)
    det_t = None                # last LIVE detection; None = never seen a gate
    lat_latch = vert_latch = False   # which loops were engaged when the gate was lost
    sig_k = 0.0                 # decay weight, 1.0 while live or holding
    tau_f = max(args.gate_filter_tau, 1e-3)
    prev_thrust, last_thrust_t = None, None   # slew-limiter state (--thrust-slew)
    # Loop-rate instrumentation. TubeDetector is 21.5 ms/call at full res and the
    # SHIPPED code ran it EVERY tick despite the 250 Hz sleep -- i.e. coast-tube has
    # really been closing roll at ~39 Hz. Frame-gating removes the redundant calls;
    # these counters report what rate we ACTUALLY achieve so flight 1 measures it
    # instead of us guessing (if hz is still low, detection wants its own thread).
    n_tick = n_det = 0
    rate_t0, hz, det_hz = time.time(), 0.0, 0.0
    tick_fh = None
    if args.tick_log:
        tick_fh = open(args.tick_log, "w", encoding="utf-8")
        tick_fh.write("t,seg,active_gate,thrust,thrust_cmd,base_thrust,vtrim,"
                      "descent_on,desc_leg,descent_bias,gate_v_ok,"
                      "gate_found,u_err,v_err,size_frac,cy,"
                      "u_f,u_f2,du_lat,v_f,sz_f,sig_k,live,far_rej,"
                      "gate_cmd_deg,committed,commit_k,"
                      "steer,bank_seg,ff_bank_deg,trim_deg,des_roll_deg,roll_n,"
                      "est_pitch_deg,gyro_pitch,des_pitch_deg,pitch_n,"
                      "tube_found,u_tube,curvature,area_frac,est_roll_deg,hold_on,"
                      "loop_hz,det_hz\n")
        print(f"[tube] per-tick log -> {args.tick_log}")
    try:
        while True:
            ag, fin, grav, gyro = rx.snap()
            now = time.time()
            n_tick += 1
            if now - rate_t0 >= 1.0:
                hz, det_hz = n_tick / (now - rate_t0), n_det / (now - rate_t0)
                n_tick = n_det = 0
                rate_t0 = now
            if fin > 0 or ag > 5:
                print(f"[tube] *** COMPLETE *** active_gate={ag} finish={fin}"); break
            if now - rx.go_wall > args.max_s:
                print(f"[tube] max flight {args.max_s}s -- stop. seg={seg} ag={ag}"); break
            row = sched[seg]
            if ag > seg and ag <= 5:
                seg = ag; seg_t0 = now; row = sched[seg]
                print(f"[tube] >>> seg -> {seg} (active_gate) t={now-rx.go_wall:.1f}s")
            elif (now - seg_t0) > row["duration_s"] * args.fallback_mult and seg < 5:
                seg += 1; seg_t0 = now; row = sched[seg]
                print(f"[tube] >>> seg -> {seg} (TIME fallback) t={now-rx.go_wall:.1f}s")

            # Course position, resolved ONCE per tick so the descent and the leg-1 bank
            # can never disagree about where we are. leg1 is ==1 (that leg alone). The
            # old post_gate1 (>=1, "the whole rest of the course") is gone with the
            # single latched descent it gated -- the ladder indexes the leg directly.
            leg1 = (seg == 1) if args.post_gate1_on_seg else (ag == 1)
            # LEG-2 PRE-TURN scope. Keyed to RAW active_gate, deliberately NOT to seg or
            # to --post-gate1-on-seg: the segment pointer also advances on the TIME
            # fallback, which fires mid-approach and would apply this bank while gate 2 is
            # still dead ahead. That is exactly how the close-range commit flew into gate 2
            # (flt13). ag == 2 means gate 2 is confirmed PASSED and gate 3 is the target,
            # so gates 1 and 2 cannot be touched by this term under any timing.
            leg2 = (ag == 2)
            # BANK index is driven by ACTIVE_GATE, not by seg. seg also advances on the
            # TIME fallback, which fires while we are still on approach and would start
            # the turn for the NEXT gate with the current one still dead ahead -- turning
            # off the gate we are lined up on. Keying to ag means segment i's bank engages
            # only when its ENTRY gate is confirmed passed (ag >= i), the same trigger
            # that made --post-gate1-bank reliable. seg >= ag always (seg only ever runs
            # AHEAD via the fallback), so this can only ever DELAY a bank, never skip one.
            bank_seg = seg if args.post_gate1_on_seg else max(0, min(len(sched) - 1, ag))
            # ALTITUDE LADDER leg index -- the SAME course-position index as the banks.
            # This is a deliberate REVERSAL of the old single descent, which keyed off
            # `seg` on the argument that arriving late at a descent is recoverable. With
            # a per-LEG table that argument inverts: `seg` also advances on the TIME
            # fallback, so it would drop the baseline to the NEXT leg's descent while the
            # current gate is still dead ahead -- and on this course the next leg's
            # descent can be an order of magnitude different (0.026 -> 0.003 at gate 3).
            # ag is the ground truth for which leg we are actually on.
            desc_leg = bank_seg

            # --- LATERAL: closed loop on the tube (the ONLY feedback) ---
            est_roll, est_pitch = gravity_to_roll_pitch(grav)
            bank_bias = trim = 0.0
            live = sig_alive = False        # --no-steer never runs the filter
            du_lat = 0.0
            gate_cmd = 0.0                  # the gate-centring command (the only one)
            tube_solid = committed = False
            gate_v_ok = False               # stays False when --gate-vert is off
            if args.no_steer:
                # ISOLATION: ZERO roll command AND no tube processing -- a true pure-coast
                # replicate (same as the standalone coast that passed gate 1: roll=0, no
                # wings-level hold, no per-tick vision load slowing the loop). Tests
                # whether steering (or its overhead) is what breaks gate 1.
                tube = None
                steer = False
                u_lat = des_roll = roll_n = 0.0
            else:
                # Detect ONLY on a new frame; otherwise reuse the last measurement
                # (bit-identical to recomputing it, and ~18 ms/tick cheaper).
                frame_n = vision.frames_received
                if frame_n != last_frame_n or gate is None:
                    last_frame_n = frame_n
                    frame = vision.get_latest_frame()
                    if tube_det is not None:
                        tube = tube_det.measure(frame)             # FULL res, --log-tube
                    # ascontiguousarray: a bare [::2,::2] VIEW measured ~1 ms/call slower
                    # through the HSV conversion than a packed copy.
                    gate = gate_det.detect(np.ascontiguousarray(frame[::2, ::2]),
                                           prefer=gate_prefer)
                    # FAR-GATE REJECT. Once the near gate leaves frame the detector
                    # re-locks onto the NEXT gate down the course, a blob at the
                    # vanishing point. MEASURED on filt.csv: at t=11.12 size_frac fell
                    # 0.1124 -> 0.0373 in ONE frame while v_err flipped -0.996 -> +0.246,
                    # and it then sat at 0.036-0.045 for the remaining 2.4 s of vision.
                    # Both loops were still steering to it. Treating it as no-detection
                    # feeds the shared filter's hold-then-fade, so the loops coast out
                    # smoothly instead of chasing a target 30 m away.
                    far_rej = (args.far_gate_reject and gate.found
                               and gate.size_frac < args.far_gate_size
                               and (args.far_gate_verr <= 0.0
                                    or abs(gate.v_err) > args.far_gate_verr))
                    # prefer= is detector continuity; seeding it with the far gate is how
                    # a momentary far lock becomes a persistent one.
                    if (gate.found and not far_rej
                            and gate.size_frac >= cfg.gate_size_accept):
                        gate_prefer = (gate.u_err, gate.v_err, gate.size_frac)
                    n_det += 1
                # --- FILTER STEP (every tick, not every frame) ------------------------
                # Running at loop rate against a 30 Hz staircase is what makes tau mean
                # 0.2 SECONDS rather than 0.2 * (whatever the frame rate happened to be).
                dt_f = min(max(now - filt_t, 0.0), 0.5) if filt_t is not None else 0.0
                filt_t = now
                live = (gate.found and not far_rej
                        and gate.size_frac >= cfg.gate_size_accept)
                if live:
                    if det_t is None:
                        # SEED on first sight -- easing up from 0 would ramp the very
                        # first gate in over ~0.2 s of wrong target.
                        u_f, v_f, sz_f = gate.u_err, gate.v_err, gate.size_frac
                        v_f2 = v_f
                        u_f2 = u_f          # P1: lateral 2nd stage, seeded like v_f2
                    else:
                        a = dt_f / (tau_f + dt_f)
                        u_f += a * (gate.u_err - u_f)
                        v_f += a * (gate.v_err - v_f)
                        sz_f += a * (gate.size_frac - sz_f)
                        v_f2 += a * (v_f - v_f2)
                        u_f2 += a * (u_f - u_f2)
                    det_t, sig_k = now, 1.0
                    u_loss, v_loss, sz_loss, v2_loss = u_f, v_f, sz_f, v_f2
                    u2_loss = u_f2
                elif det_t is not None:
                    gap = now - det_t
                    if gap <= args.gate_filter_hold_s:
                        sig_k = 1.0             # HOLD: the target is simply frozen
                        # ...but the lateral SECOND stage keeps converging on it, so the
                        # derivative decays to zero over tau. Freezing BOTH stages -- what
                        # v_f2 does -- freezes du at whatever it was, i.e. the loop keeps
                        # asserting "still moving at 0.97/s" while the signal sits still.
                        # MEASURED on filt5 t=11.20-11.46: frozen du held 0.972 for 0.26 s
                        # and des_roll then slammed +3.52 -> -7.62 in ONE 25 ms tick when
                        # the gate returned. A held signal has zero measured rate.
                        u_f2 += (dt_f / (tau_f + dt_f)) * (u_f - u_f2)
                        u2_loss = u_f2
                    else:
                        # then fade the whole signal toward zero -- never a step to 0
                        sig_k = math.exp(-(gap - args.gate_filter_hold_s)
                                         / max(args.gate_filter_decay_s, 1e-3))
                        u_f, v_f = u_loss * sig_k, v_loss * sig_k
                        sz_f, v_f2 = sz_loss * sig_k, v2_loss * sig_k
                        u_f2 = u2_loss * sig_k      # both lateral stages fade together
                # DEAD: faded to nothing. Release the latches so a re-acquisition starts
                # clean; u_f/v_f are ~2%% of their loss values here, so nothing jumps.
                sig_alive = det_t is not None and sig_k > 0.02
                if not sig_alive:
                    lat_latch = vert_latch = False
                    if det_t is not None and sig_k <= 0.02:
                        u_f = v_f = sz_f = v_f2 = 0.0
                # --- RAIL FILTER: the tube's own LPF + hold-then-fade -----------------
                # SOLID = a real, wide tube reading. The area gate is the load-bearing
                # guard: MEASURED on filt9's gate-3 leg, readings at area>=0.015 keep
                # |u_tube| <= 0.222, while dropping the gate to 0.005 admits readings out
                # to 0.673 -- slivers of tube glimpsed past the gate frame at close
                # range, which is precisely the slam this loop must not make. 0.015 is
                # also the independently-measured median area when the tube is found
                # (base.csv/hold.csv), so it is not fitted to one window.
                if rail is not None:
                    tube_solid = bool(tube is not None and tube.found
                                      and tube.area_frac >= args.tube_steer_area_min)
                    rail.update(now, dt_f, tube_solid,
                                tube.u_tube if tube is not None else 0.0,
                                tube.curvature if tube is not None else 0.0)
                # ENGAGE on the FILTERED size while a detection is live; through a blink,
                # stay engaged on the latch and let the fading u_f retire the trim
                # smoothly. Thresholding sz_f during the fade instead would cut the trim
                # off at whatever value it still held -- the exact jump the filter exists
                # to remove.
                if live:
                    steer = sz_f >= gate_size_min
                    lat_latch = steer
                else:
                    steer = lat_latch and sig_alive
                # LATERAL BACKBONE = the schedule's per-segment bank, fed forward so the
                # drone PRE-TURNS onto each leg instead of waiting for a gate to grow big
                # enough to centre on. Vision then only trims the residual. Same sign path
                # the coast-schedule uses (LATERAL_SIGN * sign * bank_rad), so the
                # schedule's right-positive geometry lands in the sim's left-positive roll
                # frame exactly as it does there -- and matches the hand-tuned
                # --post-gate1-bank it generalises (seg1 +10.8 deg -> LEFT, same way).
                # CHANGE A: fade the whole lateral feed-forward out once the gate has been
                # gone longer than --ff-lat-hold-s. Applies to the schedule backbone AND
                # --post-gate1-bank -- both are open-loop bank, and neither is safe to
                # hold blind. gate_alive uses the SAME det_t the measurement filter keeps,
                # so the two stay in step, but its own hold window so it can be tuned
                # without touching the LPF.
                # det_t None = no gate seen YET (pre-first-acquisition). That is not a
                # LOST gate: fading here would kill the seg-0 backbone before the drone
                # has ever had anything to see. Only a gate that existed can be lost.
                ff_alive = live or det_t is None or (now - det_t) <= args.ff_lat_hold_s
                ff_lat_k = ff_lat_step(ff_lat_k, dt_f, ff_alive,
                                       args.ff_lat_hold_s, args.ff_lat_decay_s)
                ff_bank = ff_bank_rad[bank_seg]
                bank_bias = ff_lat_k * (ff_bank
                                        + (math.radians(args.post_gate1_bank) if leg1 else 0.0)
                                        + (math.radians(args.post_gate2_bank) if leg2 else 0.0))
                # --- LATERAL = GATE CENTRING. The ONLY steering law. -------------------
                # The tube has NO authority on any leg. Two attempts to give it some both
                # failed and the second one crashed:
                #   flt10  rail-primary everywhere, gate demoted to a 3 deg fine-trim ->
                #          broke gate 1 (tube solid on 1.7% of ticks: nothing to steer on)
                #   flt12  priority handoff, rail only while the tube is solid ->
                #          CRASHED GATE 1. The curvature lead spiked the rail to -11 deg
                #          at close range with auth 0.95, and the drone hit the gate.
                # The curvature term is the specific hazard: (upper band - lower band)
                # goes large and noisy exactly when the near tube fills the lower band at
                # close range, which is when a bank command does the most damage.
                # PERMANENT: gate centring steers. The tube is MEASURED and LOGGED (it
                # answers "did the ladder put us on the line?") and that is all it does.
                #
                # TWO clamps, deliberately different: vision is clamped to +/-gate_bank,
                # the load-bearing guard that has kept a flickering detection from
                # slamming the rail. PRIORITY 1: derivative damping taken on the SAME
                # signal the P term uses (sign chain applied to both), so it opposes the
                # P term's own rate of change. Default kd_lat=0 -- flt6 measured it
                # net-negative.
                du_lat = LATERAL_SIGN * sign * (u_f - u_f2) / tau_f
                # The VISION clamp is per-leg: ag==2 may use a tighter bound so the
                # feed-forward leads and vision only trims the residual.
                gb = gate_bank_ag2 if leg2 else gate_bank
                gate_cmd = (max(-gb, min(gb,
                                         args.k_gate_bank * LATERAL_SIGN * sign * u_f
                                         - args.kd_lat * du_lat))
                            if steer else 0.0)
                u_lat = LATERAL_SIGN * sign * u_f if steer else 0.0
                # --- CLOSE-RANGE COMMIT, scoped to ag >= --gate-commit-from-ag ---------
                # Past --gate-commit-size the gate stops being a usable lateral reference:
                # it is close enough that PARALLAX, not position error, dominates u_err.
                # MEASURED on filt9/filt11's gate-3 leg, u_f runs +0.22 -> +0.74 over the
                # last 0.8 s while the drone is already lined up, and des_roll sat pinned
                # to the -11 deg clamp for 46 of 63 ticks -- steering sideways into the
                # terminal sweep. So at close range we COMMIT: slew the command to
                # wings-level over --gate-commit-tau and fly the approach heading through
                # the last stretch. Not a latch -- drop back below the size and the same
                # lag ramps authority back in.
                #
                # THE SCOPE IS LOAD-BEARING. Gates 1 and 2 (ag 0 and 1) pass TODAY and
                # every change that touched their approaches has cost a pass. Out of
                # scope, commit_k is forced to EXACTLY 1.0 -- not merely lagged back
                # toward it -- so the command on those legs is bit-for-bit the law flt11
                # flew, with no floating-point residue from a decayed weight.
                # Keyed to RAW active_gate, deliberately NOT to desc_leg/bank_seg: under
                # --post-gate1-on-seg the segment pointer also advances on the TIME
                # fallback, which fires mid-approach and would arm the commit while gate 2
                # is still dead ahead. flt13 is what that costs -- the commit fired on the
                # ag=1 approach, retired the leg-1 bank, and the drone clipped gate 2 at
                # t=7.7. active_gate is ground truth for which gate we are flying at.
                in_commit_scope = (args.gate_commit_size > 0.0
                                   and ag >= args.gate_commit_from_ag)
                if in_commit_scope:
                    committed = sz_f >= args.gate_commit_size
                    commit_k = lag_step(commit_k, dt_f, 0.0 if committed else 1.0,
                                        args.gate_commit_tau)
                else:
                    committed, commit_k = False, 1.0
                trim = commit_k * gate_cmd
                if steer or bank_bias:
                    # The BACKBONE is trusted geometry, so the sum only meets the outer
                    # --ff-max-bank-deg fence, which must exceed the schedule's steepest
                    # leg (-21) or the backbone would be silently truncated. The vision
                    # term is already clamped on its own above.
                    # The open-loop BANK BIAS is retired by the commit too: holding the
                    # approach HEADING is the point, and any commanded bank -- trim or
                    # backbone -- turns the drone off it. --gate-commit-vision-only leaves
                    # the backbone standing. Out of commit scope both scale by 1.0.
                    bb_scale = 1.0 if args.gate_commit_vision_only else commit_k
                    des_roll = max(-ff_bank_max,
                                   min(ff_bank_max, bb_scale * bank_bias + trim))
                    roll_rate = (cfg.ol_kp_att * (des_roll - est_roll)
                                 - cfg.ol_kd_att * float(gyro[0]))
                    roll_n = max(-lim, min(lim, roll_rate / MAX_BODY_RATE))
                else:
                    # roll = 0 EXACTLY: pure coast. NOT des_roll=0 through the hold law,
                    # which would be a wings-level hold -- the thing this file repeatedly
                    # notes perturbed the proven gate-1 pass. Only reachable with the
                    # backbone off for this segment (--banks-from/--banks-to).
                    u_lat = du_lat = trim = des_roll = roll_n = 0.0
            # --- PITCH: coast by default; CHANGE D adds a gentle near-equilibrium hold.
            # Every EARLIER hold destabilised the flight, and the difference is authority,
            # not the idea: those ran cfg.ol_kp_att = 3.0 against a setpoint up to 7.8 deg
            # off equilibrium (-10 tumbled). This one is deliberately soft AND near-
            # equilibrium: kp 1.0 at -15 is 2.8 deg off the -17.8 spawn attitude, so the
            # standing command is 0.0041 normalised -- 23% of this loop's OWN 12 deg/s
            # clamp, against the 0.0340 (1.9x that clamp) the -10/kp-3.0 hold demanded.
            # A nudge that bleeds airspeed, not an attitude the plant has to fight.
            # Rate damping on gyro[1] (q) is what keeps the nudge from ringing.
            if des_pitch_hold is None:
                pitch_n = 0.0
            else:
                pitch_rate = (args.pitch_kp * (des_pitch_hold - est_pitch)
                              - args.pitch_kd * float(gyro[1]))    # gyro[1] = PITCH (q)
                pitch_n = max(-pitch_lim, min(pitch_lim, pitch_rate / MAX_BODY_RATE))
            base_thrust = (const_thrust if const_thrust is not None else
                           float(max(cfg.ol_thrust_lo, min(cfg.ol_thrust_hi,
                                                           row["thrust"] + args.thrust_bias))))
            # --- VERTICAL BACKBONE = per-segment feed-forward descent -----------------
            # The schedule already knows each leg's slope; ff_thrust_delta[seg] is the
            # sink that slope needs, expressed as thrust BELOW level, applied to the
            # --const-thrust baseline (never to the schedule's absolute values, whose
            # steep legs bottom out near 0.135 and simply dropped the drone).
            # The FLOOR is the whole point: it caps how much descent the open loop may
            # command, so a steep leg sinks briskly but can never freefall. Where the
            # floor binds the backbone under-descends ON PURPOSE and --gate-vert makes up
            # the rest from what it can actually see.
            if ff_on:
                base_thrust = max(args.ff_descent_floor, base_thrust + ff_thrust_delta[seg])
                descent_on = ff_thrust_delta[seg] < 0.0
                leg_bias = -ff_thrust_delta[seg]
            else:
                # --- ALTITUDE LADDER: this leg's vertical feed-forward ----------------
                # One table lookup, evaluated every tick against the CURRENT leg index,
                # so the instant ag increments the baseline IS the new leg's -- no ramp,
                # no state, nothing to get stuck. (The total-command slew limiter still
                # shapes the step: a 0.026 change at 0.6/s lands in ~43 ms.)
                # The old behaviour -- one value latched on from ag>=1 for the rest of
                # the course -- is what flew the near-flat final legs into the ground.
                leg_bias = descent_bias[desc_leg]
                descent_on = leg_bias > 0.0
                base_thrust = base_thrust - leg_bias
            base_thrust = float(max(cfg.ol_thrust_lo, min(cfg.ol_thrust_hi, base_thrust)))
            # --- VERTICAL: gate v-error trims the baseline (--gate-vert) -------------
            # SIGN: v_err > 0 == gate BELOW frame centre == we are looking DOWN at it ==
            # too HIGH -> trim NEGATIVE -> less thrust -> descend. (Same sign as the
            # shipped HybridController law, thrust = ff - k*v_err.)
            if not args.gate_vert:
                thrust_cmd = base_thrust
            else:
                # Engagement mirrors the lateral: threshold the FILTERED size while live,
                # ride the latch through a blink. The measurement filter now owns ALL
                # continuity -- the old trim-level hold/decay is gone, because two
                # cascaded hold-then-decay stages compound into a lag neither one
                # describes, and the two loops would no longer be reading one target.
                if live:
                    vert_latch = sz_f >= args.gate_vert_size_min
                    gate_v_ok = vert_latch
                else:
                    gate_v_ok = vert_latch and sig_alive
                if gate_v_ok:
                    # Dirty derivative of the ALREADY-smooth signal: v_f2 is a second EMA
                    # of v_f at the same tau, so this never differentiates raw detections
                    # and cannot kick across a detection hole -- both stages fade together.
                    dv = (v_f - v_f2) / tau_f
                    gate_trim = max(-args.gate_vert_down_auth,
                                    min(args.gate_vert_up_auth,
                                        -(args.k_thrust_v * v_f + args.kd_v * dv)))
                    vtrim = gate_trim
                    last_gate_t = now
                elif last_gate_t is None:
                    vtrim = 0.0                    # never seen a gate -> pure baseline
                else:
                    # Signal fully faded. v_f is ~0 by now so gate_trim is already ~0;
                    # ease from there to `coast_bias`. Under the ALTITUDE LADDER that
                    # target is 0.0, which means HOLD this leg's feed-forward baseline
                    # -- the leg's own slope is already the between-gate descent, so
                    # there is nothing left for a trim to add. Closed form from
                    # gate_trim (the value at handoff), never iteratively from vtrim --
                    # that compounds.
                    gap = now - last_gate_t
                    k = 1.0 - math.exp(-gap / max(args.gate_vert_decay_s, 1e-3))
                    vtrim = gate_trim + k * (coast_bias - gate_trim)
                thrust_cmd = float(max(cfg.ol_thrust_lo,
                                       min(cfg.ol_thrust_hi, base_thrust + vtrim)))

            # --- SLEW LIMIT on the TOTAL command (ASYMMETRIC) ------------------------
            # Any thrust change (the post-gate-1 step, a trim jump, a gate re-acquire)
            # ramps instead of stepping in one tick -- that is what removes the ~0.5 g
            # lurch that was throwing the course out of frame.
            # DOWN is fast (--thrust-slew-down) so the descent commits promptly; UP is
            # gentle (--thrust-slew) so the recovery cannot bounce the drone back up and
            # set off a vertical oscillation against the gate trim.
            # FIRST tick seeds prev_thrust directly -- seeding from 0 would ramp in over
            # ~1.8 s at GO and drop the drone.
            if prev_thrust is None:
                thrust = thrust_cmd
            else:
                going_down = thrust_cmd < prev_thrust
                rate = args.thrust_slew_down if going_down else args.thrust_slew
                if rate <= 0:
                    thrust = thrust_cmd          # that direction is unlimited
                else:
                    # No lower floor on dt: flooring it at 1e-4 grants a bigger step than
                    # the elapsed time earns on sub-0.1ms ticks, which measurably broke
                    # the bound (0.167/s against a 0.15 limit). dt==0 -> step 0 -> hold;
                    # the next tick has real elapsed time, so this cannot lock up.
                    dt_t = min(max(now - last_thrust_t, 0.0), 0.5)
                    step = rate * dt_t
                    thrust = prev_thrust + max(-step, min(step, thrust_cmd - prev_thrust))
            prev_thrust, last_thrust_t = thrust, now
            send(conn, boot0, thrust, roll_n, pitch_n, 0.0)

            # --- per-tick instrumentation: the (v_err, est_pitch) pairs flight 2 needs
            #     to fit the pitch-compensation residual before the trim is enabled.
            if tick_fh is not None:
                tf = now - rx.go_wall
                gf = 1 if (gate is not None and gate.found) else 0
                tick_fh.write(
                    f"{tf:.6f},{seg},{ag},{thrust:.6f},{thrust_cmd:.6f},"
                    f"{base_thrust:.4f},{vtrim:+.5f},{int(descent_on)},"
                    f"{desc_leg},{leg_bias:.4f},"          # ALTITUDE LADDER: leg + bias
                    f"{int(bool(args.gate_vert and gate_v_ok))},"   # the REAL engagement
                    f"{gf},"
                    f"{(gate.u_err if gf else 0.0):+.5f},"
                    f"{(gate.v_err if gf else 0.0):+.5f},"
                    f"{(gate.size_frac if gf else 0.0):.5f},"
                    f"{(gate.cy * 2.0 if gf else 0.0):.2f},"   # x2: detect ran at half res
                    f"{u_f:+.5f},{u_f2:+.5f},{du_lat:+.5f},"
                    f"{v_f:+.5f},{sz_f:.5f},{sig_k:.4f},{int(live)},"
                    f"{int(far_rej)},"
                    f"{math.degrees(gate_cmd):+.3f},{int(committed)},{commit_k:.4f},"
                    f"{int(steer)},{bank_seg},{math.degrees(bank_bias):+.3f},"
                    f"{math.degrees(trim):+.3f},{math.degrees(des_roll):+.3f},{roll_n:+.5f},"
                    f"{math.degrees(est_pitch):+.3f},{float(gyro[1]):+.5f},"
                    f"{(args.pitch_hold_deg if des_pitch_hold is not None else 0.0):+.2f},"
                    f"{pitch_n:+.5f},"
                    f"{1 if (tube is not None and tube.found) else 0},"
                    f"{(tube.u_tube if tube is not None else 0.0):+.5f},"
                    f"{(tube.curvature if tube is not None else 0.0):+.5f},"
                    f"{(tube.area_frac if tube is not None else 0.0):.5f},"
                    f"{math.degrees(est_roll):+.3f},0,"    # hold_on: hold removed
                    f"{hz:.1f},{det_hz:.1f}\n")

            if now - last >= 0.25:
                last = now
                who = "COMMIT" if committed else ("commit" if commit_k < 0.99 else "steer ")
                tube_s = ("TUBE off" if tube is None or rail is None else
                          f"TUBE(log-only) solid={int(tube_solid)} "
                          f"area={tube.area_frac:.4f} u={rail.u:+.3f}")
                gate_s = ("GATE --" if gate is None or not gate.found else
                          f"GATE u={gate.u_err:+.3f} v={gate.v_err:+.3f} "
                          f"sz={gate.size_frac:.3f} -> {math.degrees(gate_cmd):+.1f}deg "
                          f"x{commit_k:.2f}")
                lad_s = f" LADDER leg{desc_leg} -{leg_bias:.4f}"
                vert_s = ("" if not args.gate_vert else
                          f" VERT trim={vtrim:+.4f}"
                          f"{'' if last_gate_t is None else ('' if (now-last_gate_t) < 1e-3 else f'(coasting {now-last_gate_t:.1f}s)')}")
                print(f"[tube] seg{seg}"
                      f"{f'/bank{bank_seg}' if bank_seg != seg else ''} "
                      f"ag={ag} thr={thrust:.3f}{lad_s}{vert_s} | [{who}] {tube_s} | {gate_s} | "
                      f"uLat={u_lat:+.3f} desR={math.degrees(des_roll):+.1f}"
                      f"(ff{math.degrees(bank_bias):+.1f}{math.degrees(trim):+.1f}) "
                      f"estR={math.degrees(est_roll):+.1f} estP={math.degrees(est_pitch):+.1f} "
                      f"|a|={rx.accel_g:.2f}g dimu={rx.imu_count} "
                      f"loop={hz:.0f}Hz det={det_hz:.0f}Hz")
            time.sleep(1.0 / 250.0)
    finally:
        if tick_fh is not None:
            tick_fh.close()
        vision.stop()
        _shutdown(conn, rx, stop)
    print(f"[tube] final active_gate={rx.active_gate}  gates passed={max(rx.active_gate,0)}/6")
    return 0


def _stamp(args):
    """CODE VERSION STAMP -- first lines of every run. The self-hash PROVES which code
    is executing (stale sim-box copies have cost us 4 diagnosis rounds). Compare the
    printed sha to the expected one before trusting any flight."""
    import hashlib
    import subprocess
    path = os.path.abspath(__file__)
    try:
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:10]
    except Exception:
        sha = "????"
    try:
        mt = datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")
    except Exception:
        mt = "?"
    try:
        git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=os.path.dirname(path), stderr=subprocess.DEVNULL,
                                      text=True).strip() or "none"
    except Exception:
        git = "none"
    mode = ("calibrate" if args.calibrate else "coast-tube" if args.coast_tube
            else "coast-schedule" if args.coast_schedule else "schedule")
    print(f"[STAMP] EXEC FILE = {path}")
    print(f"[STAMP] schedule_flier.py sha={sha} mtime={mt} git={git}")
    print(f"[STAMP] mode={mode} cruise={args.cruise} level_thrust={args.level_thrust} "
          f"sink_slope={args.sink_slope} banks=[{args.banks_from}..{args.banks_to}] "
          f"pulse_s={args.pulse_s} thrust_bias={args.thrust_bias} flip={args.flip_turns}")
    # The resolved per-segment thrust+bank it will actually fly (schedule modes):
    if args.coast_schedule or args.coast_tube:
        try:
            sched = build_bank_schedule(args.cruise, cfg=ServoConfig(),
                                        maneuver_frac=args.maneuver_frac)
            for r in sched:
                sink = args.cruise * math.tan(math.radians(r["slope_deg"]))
                thr = round(max(0.0, min(1.0, args.level_thrust - max(sink, 0.0) / args.sink_slope)), 3)
                if r["seg"] == 0:
                    thr = args.level_thrust  # seg0 pinned to the proven START pass
                print(f"[STAMP]   seg{r['seg']} {r['from'][:5]}->{r['to'][:5]} "
                      f"thr={thr:.3f} bank={r['bank_deg']:+.1f} slope={r['slope_deg']:+.1f}")
        except Exception as exc:
            print(f"[STAMP]   (schedule preview failed: {exc})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=14550)
    ap.add_argument("--calibrate", action="store_true", help="measure cruise (FIRST run)")
    ap.add_argument("--cruise", type=float, default=8.0, help="cruise m/s for the schedule")
    ap.add_argument("--thrust", type=float, default=0.29, help="fixed thrust for --calibrate")
    ap.add_argument("--des-pitch", type=float, default=None,
                    help="override held pitch (deg) for the SWEEP: -24, -30, -36 (--calibrate)")
    ap.add_argument("--pitch-max-rate", type=float, default=None,
                    help="raise PITCH rate clamp alone (rad/s); roll/yaw stay anti-flip clamped")
    ap.add_argument("--const-pitch-rate", type=float, default=None,
                    help="OPEN-LOOP diagnostic (--calibrate): command this fixed pitch rate "
                         "(rad/s), NO attitude feedback. 0 = coast/ride spawn tilt. Isolates "
                         "plant behavior + rate->attitude sign from the loop.")
    ap.add_argument("--maneuver-frac", type=float, default=0.7,
                    help="fraction of each segment used for the bang-bang bank slide")
    ap.add_argument("--fallback-mult", type=float, default=1.4,
                    help="advance a segment on time if active_gate stalls past duration*this")
    ap.add_argument("--coast-schedule", action="store_true",
                    help="fly the PROVEN coast (pitch rate 0) + per-segment thrust(sink map) "
                         "+ gentle roll-hold bank, stepped by active_gate. Threads the course.")
    ap.add_argument("--coast-tube", action="store_true",
                    help="ONE closed loop: coast pitch + thrust schedule + STEER TO TUBE "
                         "(blue path) center. Drops the geometry banks. Needs the vision feed.")
    ap.add_argument("--vision-port", type=int, default=5600, help="DCL FPV UDP port (--coast-tube)")
    ap.add_argument("--const-thrust", type=float, default=None, metavar="X",
                    help="--coast-tube: fly CONSTANT thrust X every tick, replacing the "
                         "whole sink-map schedule (clamped to ol_thrust_lo/hi; "
                         "--thrust-bias is ignored). Segments still advance for seg "
                         "tracking and the time fallback, but their thrust is unused.")
    ap.add_argument("--tube-max-bank-deg", type=float, default=15.0, metavar="DEG",
                    help="--coast-tube lateral bank cap. Default 15 (was cfg's 25, which "
                         "coast_tube.log slammed on tube spikes).")
    ap.add_argument("--tube-area-min", type=float, default=0.03, metavar="F",
                    help="--coast-tube: masked tube area required to steer at all; below "
                         "it roll=0 (pure coast). Default 0.03 (was cfg's 0.01). NOTE: "
                         "measured on base.csv/hold.csv, area>=0.03 held on ~16%% of "
                         "found ticks (~1-2%% of flight), so steering is rare by design.")
    ap.add_argument("--post-gate1-descent", type=float, default=0.035, metavar="A",
                    help="--coast-tube ALTITUDE LADDER ANCHOR. This is no longer the "
                         "descent flown on every leg from ag>=1 -- that single latched "
                         "value over-descended the near-flat final legs (1.9/1.5 deg) "
                         "and flew the drone into the ground past gate 3. It is now the "
                         "bias for the ANCHOR leg (--descent-anchor-leg, the 17.1 deg "
                         "g1->g2 leg where 0.026 is proven); every other leg is DERIVED "
                         "from it by slope ratio. The proven number keeps its meaning on "
                         "the leg it was tuned for. 0 disables the ladder entirely.")
    ap.add_argument("--descent-scale", type=float, default=1.0, metavar="S",
                    help="ALTITUDE LADDER: one-knob trim multiplying EVERY derived leg "
                         "bias (default 1.0). Scales the whole ladder up/down while "
                         "keeping the slope ratios between legs. Does NOT apply to "
                         "--descent-bias-legN overrides, which are absolute.")
    ap.add_argument("--descent-anchor-leg", type=int, default=2, metavar="N",
                    help="ALTITUDE LADDER: which leg --post-gate1-descent is the proven "
                         "bias FOR (default 2 = g1->g2, slope 17.1 deg). Its slope is "
                         "read from course_gates_cm.json, not hardcoded, so the ratios "
                         "track the course file. Must be a DESCENDING leg.")
    for _leg in range(6):
        ap.add_argument(f"--descent-bias-leg{_leg}", type=float, default=None, metavar="A",
                        help=(f"ALTITUDE LADDER: override leg {_leg}'s derived bias with "
                              f"this ABSOLUTE value (ignores --descent-scale). Leg index "
                              f"= active_gate: 0=SPAWN->START, 1=START->g1, 2=g1->g2, "
                              f"3=g2->g3, 4=g3->g4, 5=g4->FINISH."
                              if _leg == 0 else
                              f"ALTITUDE LADDER: absolute bias override for leg {_leg} "
                              f"(see --descent-bias-leg0)."))
    ap.add_argument("--ff-backbone", action="store_true",
                    help="--coast-tube: fly the schedule as a FEED-FORWARD BACKBONE on "
                         "BOTH axes, with vision trimming around it. LATERAL: the "
                         "segment's bank_deg is commanded outright (+10.8/-13/+12.4/-21/"
                         "+18.4 on segs 1-5) so the drone PRE-TURNS onto each leg instead "
                         "of waiting for a gate to grow big enough to centre on; gate "
                         "centring adds at most +/---gate-max-bank-deg on top. Each leg's "
                         "bank is gated on ACTIVE_GATE (segment i banks only once ag>=i, "
                         "i.e. its entry gate is confirmed passed) -- NOT on the segment "
                         "pointer, which the TIME fallback advances mid-approach and would "
                         "turn the drone off the gate it is lined up on. VERTICAL: a "
                         "per-segment descent from the schedule's slope (seg-driven -- a "
                         "late descent is recoverable by the v-trim, an early bank is "
                         "not), floored at --ff-descent-floor; --gate-vert trims that. "
                         "Supersedes --post-gate1-descent.")
    ap.add_argument("--ff-descent-floor", type=float, default=0.20, metavar="A",
                    help="--ff-backbone: thrust the feed-forward descent may never go "
                         "below (default 0.20). LOAD-BEARING -- the schedule's steep legs "
                         "ask for ~0.135, which is a freefall, and that is what dropped "
                         "the earlier raw-schedule runs. Where the floor binds, the "
                         "backbone under-descends on purpose and --gate-vert makes up the "
                         "difference from what it can see.")
    ap.add_argument("--ff-max-bank-deg", type=float, default=25.0, metavar="DEG",
                    help="--ff-backbone: outer fence on backbone+trim (default 25 = the "
                         "cfg attitude cap). Must EXCEED the schedule's steepest leg (21) "
                         "or that leg is truncated and under-turns. Note this is NOT "
                         "--gate-max-bank-deg, which still bounds the VISION trim alone.")
    ap.add_argument("--post-gate1-on-seg", action="store_true",
                    help="trigger EVERY post-gate-1 behaviour (descent, --post-gate1-bank, "
                         "--post-gate1-slowdown) on the SEGMENT pointer "
                         "(seg>=1, which the time fallback also advances) instead of "
                         "active_gate>=1. NOTE: ag never advanced past 0 in base.csv or "
                         "hold.csv, so the default ag>=1 trigger may never fire.")
    ap.add_argument("--thrust-slew", type=float, default=0.15, metavar="A_PER_S",
                    help="--coast-tube: max UPWARD rate of change of the TOTAL thrust "
                         "command (per second). Kept gentle so a recovery cannot bounce "
                         "the drone back up and oscillate against the gate trim. 0 = "
                         "unlimited UP only (it no longer disables the whole limiter -- "
                         "see --thrust-slew-down).")
    ap.add_argument("--thrust-slew-down", type=float, default=0.6, metavar="A_PER_S",
                    help="--coast-tube: max DOWNWARD rate of change of the TOTAL thrust "
                         "command (per second). Faster than the up-rate so the descent "
                         "commits promptly while the recovery stays gentle. 0 = "
                         "unlimited DOWN.")
    ap.add_argument("--gate-vert", action="store_true",
                    help="--coast-tube: close THRUST on the gate's vertical error. "
                         "Baseline = --const-thrust (else the schedule row); the gate "
                         "trims it within [-down-auth, +up-auth]. v_err>0 (gate below "
                         "centre = we are too high) -> negative trim -> descend.")
    ap.add_argument("--gate-vert-size-min", type=float, default=0.05, metavar="F",
                    help="gate size_frac required to trim thrust (default 0.05; measured "
                         "on base.csv/hold.csv this holds on 95-100%% of found ticks).")
    ap.add_argument("--k-thrust-v", type=float, default=0.16, metavar="K",
                    help="thrust per unit gate v_err (default 0.16).")
    ap.add_argument("--kd-v", type=float, default=0.10, metavar="K",
                    help="thrust per unit filtered dv/dt. Default 0.10. NOTE this is an "
                         "INCREASE from the previous 0.04 -- see the --gate-filter-tau "
                         "note; if the intent was less damping now that the signal is "
                         "smooth, 0.02 is the value that halves it.")
    ap.add_argument("--gate-vert-down-auth", type=float, default=0.06, metavar="A",
                    help="max DOWNWARD (negative) thrust trim. Default 0.06 -> floor "
                         "0.215 off a 0.275 baseline. Asymmetric with up-auth on "
                         "purpose: the course descends.")
    ap.add_argument("--gate-vert-up-auth", type=float, default=0.03, metavar="A",
                    help="max UPWARD (positive) thrust trim (default 0.03).")
    ap.add_argument("--gate-vert-coast-bias", type=float, default=-0.03, metavar="A",
                    help="trim decayed toward when no gate is visible -- a mild descent "
                         "so the drone keeps sinking BETWEEN gates (default -0.03).")
    ap.add_argument("--gate-vert-hold-s", type=float, default=0.5, metavar="S",
                    help="SUPERSEDED by --gate-filter-hold-s: continuity now lives in the "
                         "shared measurement filter, not in a second trim-level stage. "
                         "Accepted and ignored.")
    ap.add_argument("--gate-vert-decay-s", type=float, default=1.5, metavar="S",
                    help="first-order time constant of the decay to the coast bias.")
    ap.add_argument("--pitch-hold-deg", type=float, default=None, metavar="DEG",
                    help="--coast-tube CHANGE D: hold pitch at this attitude (deg, "
                         "nose-down negative) to bleed airspeed. Omit = pure coast "
                         "(default, unchanged). Spec value -15, i.e. ~3 deg nose-up of "
                         f"the {SPAWN_PITCH_DEG:.1f} equilibrium. NEAR-EQUILIBRIUM is the "
                         "whole design: -10 (7.8 deg off) tumbled at kp 3.0.")
    ap.add_argument("--pitch-kp", type=float, default=1.0, metavar="K",
                    help="CHANGE D proportional gain, rad/s per rad. Default 1.0 -- "
                         "deliberately SOFT, a third of the cfg.ol_kp_att=3.0 the roll "
                         "channel uses and that tumbled the earlier holds.")
    ap.add_argument("--pitch-kd", type=float, default=0.15, metavar="K",
                    help="CHANGE D rate damping on gyro q, rad/s per rad/s. Default 0.15: "
                         "3x the roll channel's 0.05, because this loop is a slow nudge "
                         "and wants to be over-damped rather than ring. Damping ratio at "
                         "kp=1.0 is kd/(2*sqrt(kp)) = 0.075 of critical on the rate loop "
                         "alone -- the plant supplies the rest.")
    ap.add_argument("--pitch-max-rate-deg", type=float, default=12.0, metavar="D",
                    help="CHANGE D hard rate clamp (deg/s, default 12). SEPARATE from the "
                         "roll channel's ol_max_rate_rad_s so a pitch experiment can never "
                         "spend the roll axis's authority budget. This is the tumble stop.")
    ap.add_argument("--kd-lat", type=float, default=0.0, metavar="K",
                    help="--coast-tube PRIORITY 1: derivative damping on the LPF lateral "
                         "signal. des_roll = clamp(k_gate_bank*u_lat - kd_lat*du_lat), "
                         "du_lat from a second-stage EMA u_f2 at --gate-filter-tau. "
                         "DEFAULT 0.0 = OFF, opt-in only. flt6 flew it and it was "
                         "NET-NEGATIVE: on the gate-2 approach the lateral error grows "
                         "MONOTONICALLY, so a derivative reinforces the P term instead of "
                         "opposing it -- the bank ran to the -11 clamp and clipped the "
                         "right post (protocol.py flt6 gate#2 u_err +0.919 vs filt5's "
                         "+0.843 without it). The code is kept for a leg where the error "
                         "actually oscillates; do not enable it blind.")
    ap.add_argument("--ff-lat-hold-s", type=float, default=0.4, metavar="S",
                    help="--coast-tube CHANGE A: hold the lateral feed-forward bank at "
                         "full strength this long after the gate is lost, then fade it to "
                         "level over --ff-lat-decay-s. Applies to the --ff-backbone "
                         "schedule bank AND --post-gate1-bank. MEASURED on filt4.csv: "
                         "without this, des_roll_deg held +5.00 for 1210 ticks / 7.07 s "
                         "after the last live detection -- an open-loop turn with nothing "
                         "watching it. Default 0.4 (matches --gate-filter-hold-s).")
    ap.add_argument("--ff-lat-decay-s", type=float, default=0.4, metavar="S",
                    help="--coast-tube CHANGE A: time constant of the fade to wings-level "
                         "(default 0.4). Also the ramp-back-IN constant on re-acquisition, "
                         "so the bank is never stepped on in a single tick. 0 = no lag "
                         "(instant on/off).")
    ap.add_argument("--far-gate-reject", action="store_true",
                    help="--coast-tube: treat a too-small gate as NO detection, so both "
                         "loops coast out through the filter's hold-then-fade instead of "
                         "steering to the next gate at the vanishing point. MEASURED on "
                         "filt.csv: at t=11.12 size_frac fell 0.1124->0.0373 in one frame "
                         "(v_err flipped -0.996->+0.246) and stayed at 0.036-0.045 for "
                         "the last 2.4 s of vision, with both loops still tracking it.")
    ap.add_argument("--far-gate-size", type=float, default=0.055, metavar="F",
                    help="--far-gate-reject: size_frac below this is a far gate. Default "
                         "0.055. On filt.csv this splits the run perfectly: 0 rejects "
                         "across all 333 good near-gate ticks, 82/82 on the far-gate "
                         "lock.")
    ap.add_argument("--far-gate-verr", type=float, default=0.0, metavar="V",
                    help="--far-gate-reject: OPTIONAL extra condition -- also require "
                         "|v_err| above this to reject. 0 (default) = size alone. NOTE "
                         "0.5 was the originally-specified value but on filt.csv it "
                         "makes the rule a NO-OP: the far gate sat at |v_err| 0.25-0.34 "
                         "(0 of 82 ticks above 0.5) while the NEAR gate ran at a median "
                         "of 0.72. High |v_err| tracked the near gate, not the far one.")
    ap.add_argument("--gate-filter-tau", type=float, default=0.2, metavar="S",
                    help="--coast-tube: EMA time constant on the gate's u_err/v_err/"
                         "size_frac (default 0.2 s). ONE filtered signal feeds BOTH the "
                         "lateral and vertical loops, so they move together and the same "
                         "approach reproduces the same path. Updated at LOOP rate against "
                         "the latest detection, so tau is in seconds regardless of frame "
                         "rate. 0 is not accepted -- use a tiny value to effectively "
                         "disable.")
    ap.add_argument("--gate-filter-hold-s", type=float, default=0.4, metavar="S",
                    help="--coast-tube: hold the last filtered gate values unchanged this "
                         "long after the gate blinks out (default 0.4 s) before fading. "
                         "Never zeroes and never steps.")
    ap.add_argument("--gate-filter-decay-s", type=float, default=0.5, metavar="S",
                    help="--coast-tube: time constant of the fade to zero after the hold "
                         "expires. Both loops retire their trim smoothly as it fades; the "
                         "signal is declared dead at 2%% (hold + ~4 tau).")
    ap.add_argument("--log-tube", action="store_true",
                    help="--coast-tube: run the TUBE detector and log its columns "
                         "(tube_solid/u_tube_f/curv_f/rail_k/tube_found/u_tube/curvature/"
                         "area_frac). OFF BY DEFAULT and it costs real flight quality to "
                         "turn on: MEASURED on vision_frame.png the tube detector is "
                         "13.47 ms/call at full resolution against the gate detector's "
                         "5.45, so it is 71%% of the per-frame vision budget -- for a "
                         "signal that has had ZERO control authority since the rail was "
                         "removed. With it on, the loop ran a 34.9 Hz median with a "
                         "16.2 Hz FLOOR on the gate-2 leg, and that VARIANCE is what "
                         "made gate 2 inconsistently pass-or-collide. Enable only to "
                         "answer 'did the altitude ladder put us on the line?', and "
                         "expect the flight to be slower and less repeatable while it is "
                         "on. Changes NO control law either way.")
    ap.add_argument("--tube-steer-area-min", type=float, default=0.015, metavar="F",
                    help="--coast-tube: masked tube area at which a reading counts as "
                         "SOLID. LOG ONLY -- the tube has no steering authority (flt10 "
                         "broke gate 1, flt12 crashed into it). This is the threshold "
                         "the tube_solid column and claude/protocol.py use to answer "
                         "'did the altitude ladder put us on the line?'. Default 0.015 = "
                         "the measured median area when the tube is found.")
    ap.add_argument("--lateral-gate-centring", action="store_true",
                    help="NO-OP, kept so existing command lines still parse. Gate "
                         "centring is now the only lateral law; the tube is log-only.")
    ap.add_argument("--gate-commit-size", type=float, default=0.25, metavar="F",
                    help="--coast-tube CLOSE-RANGE COMMIT: filtered gate size_frac above "
                         "which the gate stops being a usable lateral reference and the "
                         "lateral command is slewed to WINGS-LEVEL over "
                         "--gate-commit-tau, so the drone flies its approach heading "
                         "through the last stretch instead of chasing the terminal "
                         "parallax sweep. Default 0.25. MEASURED on filt9's gate-3 leg: "
                         "u_f runs +0.22 -> +0.67 over the last 0.8 s with the drone "
                         "already lined up, and des_roll sat pinned to the -11 deg clamp "
                         "for 30 of 46 close-range ticks; at 0.25 the commit takes that "
                         "to 0 pinned ticks and an 8.9 deg peak (6.6 one tau in). This "
                         "is NOT a latch -- drop back below the size and the same lag "
                         "ramps authority back in. 0 disables (never commits).")
    ap.add_argument("--gate-commit-from-ag", type=int, default=2, metavar="N",
                    help="--coast-tube: apply the close-range commit ONLY from this leg "
                         "index (active_gate) onward. Default 2 = the gate-3 leg. THE "
                         "SCOPE IS LOAD-BEARING: gates 1 and 2 (ag 0 and 1) pass today "
                         "and every change that has touched their approaches has cost a "
                         "pass, so out of scope the commit weight is forced to EXACTLY "
                         "1.0 and the command is bit-for-bit the law flt11 flew. 0 would "
                         "apply it everywhere -- do not, without a flight that says so.")
    ap.add_argument("--gate-commit-tau", type=float, default=0.2, metavar="S",
                    help="--coast-tube: time constant of the slew to wings-level at "
                         "close range, and of the ramp back in below the commit size "
                         "(default 0.2 s). 0 = instant (a STEP -- the thing the lag "
                         "exists to avoid).")
    ap.add_argument("--gate-commit-vision-only", action="store_true",
                    help="--coast-tube: scope the close-range commit to the VISION trim "
                         "and leave the open-loop bank (the --ff-backbone leg bank and "
                         "--post-gate1-bank) standing. Default is the full command -> 0, "
                         "because holding the approach HEADING is the point and any "
                         "commanded bank turns the drone off it. Use this if zeroing the "
                         "leg-1 bank at close range costs the gate-2 pass.")
    ap.add_argument("--k-gate-bank", type=float, default=1.0, metavar="K",
                    help="--coast-tube: desired bank ANGLE (rad) per unit gate u_err, "
                         "before the --gate-max-bank-deg clamp. Same units as cfg.k_bank "
                         "(1.0), so u_err=0.175 reaches the 10 deg clamp.")
    ap.add_argument("--gate-bank-size-min", type=float, default=0.10, metavar="F",
                    help="--coast-tube: gate size_frac required to steer at all. Below it "
                         "roll=0 (coast straight) with NO tube fallback. Measured on "
                         "base.csv/hold.csv: >=0.10 holds on 7-14%% of flight.")
    ap.add_argument("--gate-max-bank-deg", type=float, default=10.0, metavar="DEG",
                    help="--coast-tube: HARD clamp on the gate-centring bank. Default 10.")
    ap.add_argument("--post-gate1-bank", type=float, default=0.0, metavar="DEG",
                    help="--coast-tube: EXTRA standing left bank on the gate-1->gate-2 leg "
                         "only (ag==1, or seg==1 under --post-gate1-on-seg), on top of the "
                         "backbone. Default is now 0 because --ff-backbone generalises it: "
                         "seg 1 already gets the schedule's +10.8 the same way. Use only to "
                         "hand-bias that one leg. POSITIVE = left; NEGATIVE = right.")
    ap.add_argument("--post-gate2-bank", type=float, default=0.0, metavar="DEG",
                    help="--coast-tube: EXTRA standing pre-turn bank on the ag==2 leg "
                         "ONLY, summed with the gate-centring trim and then clamped -- "
                         "exactly the mechanism --post-gate1-bank uses on ag==1 to line "
                         "up the next gate. POSITIVE = LEFT, NEGATIVE = RIGHT (same sign "
                         "convention as --post-gate1-bank). Default 0 = OFF, so the "
                         "command is unchanged until you set it. "
                         "PURPOSE: pre-turn toward the next gate's KNOWN position so "
                         "vision only trims the residual, instead of letting u_err run "
                         "out to the -11 deg clamp chasing the terminal parallax sweep. "
                         "SCOPE: keyed to RAW active_gate == 2, NOT to the segment "
                         "pointer and NOT affected by --post-gate1-on-seg, so the TIME "
                         "fallback can never apply it while gate 2 is still ahead. "
                         "NOTE ON THE VALUE: the leg flown at ag==2 is the schedule's "
                         "g1->g2 segment (seg2), whose lateral offset is -3.70 m with a "
                         "computed bank of -13.0 deg. The +6.30 m / +12.4 deg figures "
                         "belong to seg3, which is flown at ag==3. Check which leg you "
                         "mean before picking a number -- `python tools/motion_schedule.py "
                         "--cruise 8 --bank` prints the table.")
    ap.add_argument("--gate-max-bank-ag2", type=float, default=None, metavar="DEG",
                    help="--coast-tube: HARD clamp on the VISION centring bank for the "
                         "ag==2 leg alone. Defaults to --gate-max-bank-deg, i.e. no "
                         "change unless passed. Set it TIGHTER than the global clamp to "
                         "let --post-gate2-bank's feed-forward lead and leave vision only "
                         "the residual. Bounds the vision term only -- the feed-forward "
                         "bank is added on top and the sum meets --ff-max-bank-deg.")
    ap.add_argument("--tube-trim", action="store_true",
                    help="--coast-schedule: add the BOUNDED tube trim (the one new loop). "
                         "Fixed banks stay the open-loop backbone; the tube may only nudge "
                         "the residual within --trim-clamp-deg. Off = schedule unchanged.")
    ap.add_argument("--tube-trim-area-min", type=float, default=0.015, metavar="F",
                    help="masked tube area fraction required to trim. MEASURED on "
                         "base.csv/hold.csv: median area when found is 0.015-0.019 and p90 "
                         "is ~0.04, so 0.03-0.05 fires on ~1%% of the flight (effectively "
                         "off). 0.015 keeps roughly the stronger half. Default 0.015.")
    ap.add_argument("--k-tube-trim", type=float, default=0.35, metavar="K",
                    help="bank ANGLE (rad) per unit tube control-u, before the clamp. "
                         "0.35 vs the k_bank=1.0 that saturated coast-tube.")
    ap.add_argument("--trim-clamp-deg", type=float, default=5.0, metavar="DEG",
                    help="HARD clamp on the trim's bank angle. This is the load-bearing "
                         "guard -- see --tube-trim-area-min help for why area is not.")
    ap.add_argument("--k-trim-att", type=float, default=1.0, metavar="K",
                    help="P gain closing est_roll onto the trim angle (rad/s per rad).")
    ap.add_argument("--trim-during-pulse", action="store_true",
                    help="also trim during the bang-bang roll+/roll- pulses. Default is "
                         "coast-phase only, where the open-loop intent is bank=0 so "
                         "est_roll IS the residual and the trim cannot fight the bank.")
    ap.add_argument("--hold-pitch", type=float, default=None, metavar="DEG",
                    help="REMOVED (--coast-tube): the pitch hold destabilised the flight "
                         "and is gone. Accepted so old command lines still parse; passing "
                         "it prints a warning and changes nothing.")
    ap.add_argument("--no-pitch-hold", action="store_true",
                    help="--coast-tube: NO-OP, kept so existing command lines still run. "
                         "Pure-coast pitch (pitch_n=0) is now the only behaviour.")
    ap.add_argument("--hold-pitch-from-seg", type=int, default=0, metavar="N",
                    help="REMOVED (--coast-tube) -- see --hold-pitch.")
    ap.add_argument("--tick-log", default=None, metavar="CSV",
                    help="--coast-tube: write EVERY tick to this CSV (t, seg, active_gate, "
                         "thrust, gate v_err/size/cy, est_pitch, gyro_pitch, tube, roll). "
                         "This is the flight-1 calibration data for the gate v-trim.")
    ap.add_argument("--no-steer", action="store_true",
                    help="ISOLATION (--coast-tube): ZERO roll command (pure coast on roll, no "
                         "wings-level hold). Tests whether steering is what breaks gate 1.")
    ap.add_argument("--thrust-bias", type=float, default=0.0,
                    help="global thrust offset on the schedule (e.g. -0.015 to align map to "
                         "the measured coast that passed START at 0.275)")
    ap.add_argument("--banks-from", type=int, default=0,
                    help="(--coast-schedule, and --coast-tube --ff-backbone) apply banks "
                         "only for seg >= N; segs below fly "
                         "PURE COAST (roll=0). N>=6 => all banks off (isolation); N=2 => banks "
                         "from seg2 on. Adds banks one segment at a time.")
    ap.add_argument("--banks-to", type=int, default=5,
                    help="(--coast-schedule, and --coast-tube --ff-backbone) apply banks "
                         "only for seg <= N. With --banks-from "
                         "this brackets a SINGLE segment (e.g. --banks-from 1 --banks-to 1 = only "
                         "seg1) for strict one-at-a-time rollout; all others fly pure coast.")
    ap.add_argument("--pulse-s", type=float, default=0.5,
                    help="(--coast-schedule) OPEN-LOOP roll pulse half-duration (s): +rate for "
                         "pulse_s then -rate for pulse_s to bank+level, then roll=0 coast. NO "
                         "level-hold. Bigger => bigger slide.")
    ap.add_argument("--level-thrust", type=float, default=0.275,
                    help="(--coast-schedule) measured level-flight thrust at coast attitude "
                         "(0 sink). Thrust schedule anchors here: thr = level - sink/sink_slope.")
    ap.add_argument("--sink-slope", type=float, default=17.6,
                    help="(--coast-schedule) m/s of extra sink per unit thrust reduction "
                         "(from the sink-map SLOPE). Lower thrust => more sink.")
    ap.add_argument("--flip-turns", action="store_true", help="invert lateral/turn sign if wrong live")
    ap.add_argument("--max-s", type=float, default=35.0, help="hard flight timeout after GO")
    ap.add_argument("--go-timeout", type=float, default=180.0)
    args = ap.parse_args()

    # Line-buffer stdout so every line flushes live to a Tee/pipe (Python block-buffers
    # a pipe by default, which lost the whole trace last run -- only 1 line reached disk).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    _stamp(args)   # FIRST lines: code sha + config, so stale runs are obvious at a glance

    cfg = ServoConfig()
    applied = apply_openloop_tune(cfg)   # inherit tuned kp/etc from openloop_tune.json
    if applied:
        print(f"[sched] tune overrides: {applied}")
    if args.calibrate:
        return mode_calibrate(args, cfg)
    if args.coast_tube:
        return mode_coast_tube(args, cfg)
    if args.coast_schedule:
        return mode_coast_schedule(args, cfg)
    return mode_schedule(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
