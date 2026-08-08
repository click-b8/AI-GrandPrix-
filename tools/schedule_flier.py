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
import collections
import json
import math
import os
import queue
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
    TubeDetector, GateDetector, SPAWN_PITCH_DEG, draw_rail_overlay)
from dcl_mavlink_adapter import wire_body_rate  # noqa: E402
from attitude_filter import GravityEstimator  # noqa: E402
from dcl_vision_receiver import DCLVisionReceiver  # noqa: E402
from tools.motion_schedule import build_bank_schedule, segments  # noqa: E402

G = 9.81

# --- POST-GATE PATH HOLD tuning (see --post-gate-hold-s) ----------------------------
# Promote to flags if either needs to move per-leg.
POST_GATE_HOLD_SIZE = 0.20   # gate size_frac at/above which steering is reliable -> resume
POST_GATE_HOLD_CORR = 0.3    # fraction of the LIVE gate_cmd admitted during the hold
POST_GATE_HOLD_CAPTURE_MAX = 0.35   # capture only BELOW this size_frac: past it the gate
#                                     is close enough that parallax dominates u_err, so a
#                                     command captured there is a spike, not a heading
HOLD_MAX = math.radians(2.5)        # hard cap on the HELD command. A trajectory hold is a
#                                     nudge, not a turn: filt57 latched +10.0 deg (the bank
#                                     clamp) off the next gate's first sighting and flew it
#                                     for 1.5 s, stacking with the leg bank to pin the
#                                     15 deg fence. Capped, even a bad capture is survivable.


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
        # OBSERVATIONAL ONLY: the most recent RAW race-status payload (type byte 1).
        # We unpack '<BQqqIq' = 37 of its 253 bytes and discard the remaining 216, which
        # are the bytes a position field could be hiding in. Kept whole here so a lap can
        # be dumped and decoded offline. Nothing reads this for control.
        self.encap_raw = b""
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
                    # Stash the FULL payload before any of the filtering below can
                    # `continue` past it -- the stale-race skip in particular would
                    # otherwise drop payloads we want for offline decoding. Assignment
                    # only; no control path reads it.
                    self.encap_raw = raw
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

    def encap_hex(self):
        """Latest raw race-status payload as hex, or '' if none has arrived yet.

        Separate from snap() on purpose: snap()'s 4-tuple is unpacked positionally by
        every mode, so widening it would touch control code. This is read only by the
        tick logger.
        """
        with self.lock:
            return self.encap_raw.hex()

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


class FrameDumper(threading.Thread):
    """Saves raw FPV frames + rail overlays DURING a live race, on a worker thread.

    WHY A THREAD, NOT AN INLINE SAVE. A 640x360 PNG costs ~29.5 ms to encode and write,
    and each sample is TWO of them (raw + overlay) plus a detector call: ~68 ms, against
    a ~28 ms tick. Writing that inline at 4 Hz would stall the loop for more than two
    ticks at a time, on the g1->g2 leg -- which IS the gate-2 approach, the leg whose
    loop-rate variance (35 Hz median, 16 Hz floor) already makes gate 2 inconsistently
    pass-or-collide. That would corrupt the very flight we are trying to observe. So the
    control loop only hands over a reference and moves on.

    Frames are safe to share without copying: DCLVisionReceiver.get_latest_frame()
    already returns a fresh array per call, and neither the detector nor the overlay
    drawer mutates its input.

    BOUNDED, DROP-OLDEST. If the disk cannot keep up the queue must never grow without
    limit or block the flight -- so a full queue drops the sample and counts it. A
    missing frame is a nuisance; a stalled control loop is a crash.

    PRE-GATE RING. Frames on the ag==0 approach are held in a small ring buffer and
    written only when active_gate ticks over to 1 -- i.e. retroactively, once we know
    the drone actually reached gate 1. That is the only way to reliably catch "the last
    second BEFORE the gate": the alternative, guessing from a rising size_frac, fires on
    every glimpse of a gate and misses the transition when the detector blinks.
    """

    def __init__(self, outdir, cfg, every_s=0.25, pre_s=1.0, queue_max=64):
        super().__init__(daemon=True)
        self.outdir, self.cfg = outdir, cfg
        self.every_s, self.pre_s = every_s, pre_s
        self.q = queue.Queue(maxsize=queue_max)
        self.ring = collections.deque(maxlen=max(1, int(pre_s / max(every_s, 1e-3)) + 1))
        self._last_t = {}          # tag -> last accepted sample time
        self.dropped = self.written = 0
        self._manifest = None
        self._stop = threading.Event()

    def offer(self, tag, t, frame, force=False):
        """Called from the CONTROL LOOP. Must stay cheap: a clock check and a queue put.
        `force` bypasses the rate limit (used to flush the pre-gate ring)."""
        if not force:
            last = self._last_t.get(tag)
            if last is not None and (t - last) < self.every_s:
                return False
            self._last_t[tag] = t
        try:
            self.q.put_nowait((tag, t, frame))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def hold_pre_gate(self, t, frame):
        """Buffer an approach frame without writing it. Rate-limited like offer()."""
        last = self._last_t.get("_ring")
        if last is not None and (t - last) < self.every_s:
            return
        self._last_t["_ring"] = t
        self.ring.append((t, frame))

    def flush_pre_gate(self):
        """active_gate reached 1: commit the buffered approach frames."""
        n = 0
        while self.ring:
            t, frame = self.ring.popleft()
            n += int(self.offer("ag0pre", t, frame, force=True))
        return n

    def run(self):
        from PIL import Image
        os.makedirs(self.outdir, exist_ok=True)
        det = TubeDetector(self.cfg)
        man = open(os.path.join(self.outdir, "manifest.csv"), "w", encoding="utf-8")
        man.write("t,tag,lock,u_tube,v_converge,curvature,rail_sep,n_rows,fit_rms,"
                  "area_frac,raw_file,overlay_file\n")
        self._manifest = man
        try:
            while not (self._stop.is_set() and self.q.empty()):
                try:
                    tag, t, frame = self.q.get(timeout=0.2)
                except queue.Empty:
                    continue
                m = det.measure(frame)
                # The lock state lives in the FILENAME so the folder can be scanned
                # without opening anything: the failures sort right next to the frame
                # that produced them.
                stem = (f"{tag}_t{t:08.3f}_{'LOCK' if m.found else 'NOLOCK'}"
                        f"_u{m.u_tube:+.3f}_rows{m.n_rows:03d}_rms{m.fit_rms:04.1f}")
                raw_f, ovl_f = stem + "_raw.png", stem + "_ovl.png"
                Image.fromarray(frame).save(os.path.join(self.outdir, raw_f))
                Image.fromarray(draw_rail_overlay(frame, m, self.cfg)).save(
                    os.path.join(self.outdir, ovl_f))
                man.write(f"{t:.3f},{tag},{int(m.found)},{m.u_tube:+.5f},"
                          f"{m.v_converge:+.5f},{m.curvature:+.5f},{m.rail_sep:.5f},"
                          f"{m.n_rows},{m.fit_rms:.3f},{m.area_frac:.5f},"
                          f"{raw_f},{ovl_f}\n")
                man.flush()
                self.written += 1
        finally:
            man.close()

    def stop(self, drain_s=10.0):
        """Signal the writer and let it finish what is already queued."""
        self._stop.set()
        self.join(timeout=drain_s)
        return self.written, self.dropped


def rail_bank_cmd(u_rail, curv_rail, k_bank, lead_gain, bank_max, lead_max):
    """RAIL steering command (rad) from ALREADY sign-corrected rail signals.

    des_roll = clamp(k_bank * (u_tube + lead_gain * curvature)) -- the original coast-tube
    "LATERAL=TUBE(closed)" law, with ONE change that is the whole flt12 fix: the curvature
    lead gets its OWN clamp before the sum, rather than only meeting the outer bank clamp.

    WHY the lead needs a separate bound: curvature is (upper band centre - lower band
    centre). It goes large AND noisy exactly when the near tube fills the lower band at
    close range -- i.e. when a bank command does the most damage. In flt12 the lead alone
    spiked the rail to the -11 deg clamp at authority 0.95 and the drone hit gate 1.
    Bounded on its own, the lead stays what it is supposed to be -- an anticipatory nudge
    onto the coming bend -- and can never dominate the u_tube term it is leading.
    """
    lead = max(-lead_max, min(lead_max, k_bank * lead_gain * curv_rail))
    return max(-bank_max, min(bank_max, k_bank * u_rail + lead))


def gate_fine_trim(gate_cmd, sz_f, u_f, size_min, uerr_max, fine_max):
    """THE ANTI-DECOY RULE: while the rail steers, the gate may trim only when it is
    LARGE and NEAR-CENTRED, and then only as a clamped nudge.

    A big gate that is far off-centre is not a large error signal -- it is the parallax
    lunge. MEASURED on filt9's gate-3 leg, gate u_err sweeps to +0.95 while the tube moves
    to +0.37; steering on that swept the bank into its clamp for 65% of the close-range
    ticks. So the window closes as the sweep begins: on that leg every admitted tick
    precedes every rejected one, and the trim retires and stays retired.
    """
    if sz_f < size_min or abs(u_f) > uerr_max:
        return 0.0
    return max(-fine_max, min(fine_max, gate_cmd))


def blend_lateral(rail_auth, rail_cmd, gate_fine, commit_k, gate_cmd):
    """Weighted handoff between the RAIL law and the GATE law.

    `rail_auth` is the rail filter's own decay weight, NOT a boolean: the rail owns the
    bank while the tube is solid or held, and as it fades the gate law (with its
    close-range commit) fades back in over the same tau. Never a step, never a slam.

    The two endpoints are exact, and that exactness is the safety property:
      auth 1.0 -> rail_cmd + gate_fine, the specified tube-following law
      auth 0.0 -> commit_k * gate_cmd, bit-for-bit the law that passes gates 1 and 2
    A leg where the tube is never seen therefore flies today's proven law rather than
    flying blind -- which matters, because whether the altitude ladder actually brings the
    tube into view on the gate-3 leg is still an open question a flight has to answer.
    """
    return (rail_auth * (rail_cmd + gate_fine)
            + (1.0 - rail_auth) * commit_k * gate_cmd)


class RailVertical:
    """The rail's VERTICAL loop: v_converge -> a thrust trim, with hold-then-decay.

    WHAT IT SERVOS. v_converge is the frame row where the two fitted rails MEET -- the
    path's vanishing point. Its position is set by the path's slope relative to the
    camera axis, and since pitch is pinned at the coast equilibrium (PERMANENT, flt8),
    that makes it a direct readout of "am I descending at the path's slope?":
        v ABOVE target  (more negative)  the path runs UP out of frame  -> CLIMB
        v BELOW target  (more positive)  the path dives away below      -> DESCEND
    So the drone tracks the COURSE's descent instead of reacting to a red gate that only
    becomes a usable reference in the last metre.

    SIGN: trim = -k * (v - target), matching the gate law's thrust = ff - k*v_err. A
    positive error (convergence below target = we are riding high) gives a NEGATIVE
    trim = less thrust = descend.

    THE TARGET IS NOT ZERO, and that is the whole subtlety. MEASURED on vision_frame.png:
    with the gate vertically centred (v_err -0.03, the exact condition the working
    gate-vert loop drives to) the rail converges at v = -0.346. Targeting 0 would
    therefore command a standing climb of k*0.346 -- with k=0.16 that is 0.055, nearly
    double the up-authority clamp, so it would sit pinned to the clamp and fly the drone
    off the top of the course. The default target is the measured equilibrium.

    HOLD THEN DECAY TO THE FEED-FORWARD. The rail blinks out at close range and through
    the gate. On loss the trim is HELD briefly, then decays CLOSED FORM to 0.0 -- and 0.0
    means "fly this leg's altitude-ladder baseline", because vtrim is a trim on a
    base_thrust that already carries the ladder's slope-sized descent. So the vertical
    command is always either the live rail or the slope-matched feed-forward. Never flat,
    never blind, and never a step.

    Closed form from the value AT LOSS, never iterated from the live value -- iterating
    compounds every tick and races to zero, a bug this file has already been bitten by
    once on this very trim.
    """

    def __init__(self, tau_s, hold_s, decay_s, k, target, up_auth, down_auth):
        self.tau = max(tau_s, 1e-3)
        self.hold, self.decay = hold_s, max(decay_s, 1e-3)
        self.k, self.target = k, target
        self.up, self.down = up_auth, down_auth
        self.v = 0.0               # filtered v_converge
        self.trim = 0.0            # the commanded trim
        self._trim_loss = 0.0      # trim at the moment the lock was lost
        self.v_loss = 0.0          # filtered v at that same moment (decays WITH the trim)
        self.det_t = None          # last LOCK; None = never locked
        self.k_w = 0.0             # 1 while live or holding, decaying after
        self.live = False

    def _clamp(self, t):
        return max(-self.down, min(self.up, t))

    def update(self, now, dt_s, locked, v_converge, target=None):
        if target is not None:
            self.target = target
        self.live = bool(locked)
        if locked:
            if self.det_t is None:
                self.v = float(v_converge)            # SEED on first lock
            else:
                a = dt_s / (self.tau + dt_s)
                self.v += a * (float(v_converge) - self.v)
            self.trim = self._clamp(-self.k * (self.v - self.target))
            self.det_t, self.k_w = now, 1.0
            self._trim_loss, self.v_loss = self.trim, self.v
        elif self.det_t is not None:
            gap = now - self.det_t
            if gap <= self.hold:
                self.k_w = 1.0                        # HOLD the last rail descent
            else:
                # Decay the HELD TRIM toward 0 = this leg's ladder feed-forward.
                self.k_w = math.exp(-(gap - self.hold) / self.decay)
                self.trim = self._trim_loss * self.k_w
                # DECAY THE FILTER STATE WITH IT. self.v is what the trim is computed
                # from, so letting it sit at its pre-loss value while the trim decays
                # desynchronises the two: the moment the rail came back, the trim was
                # recomputed from the stale v and JUMPED straight back to full
                # magnitude -- measured at 0.025 of thrust in a single tick, which is
                # the slam this class exists to avoid, arriving at re-acquisition
                # instead of at the loss. Decaying v toward the target keeps
                # trim == -k*(v - target) true at every instant, so re-acquisition is
                # just the LPF easing back over tau, with no extra state and no step.
                self.v = self.target + (self.v_loss - self.target) * self.k_w
                if self.k_w <= 0.02:
                    self.trim = 0.0                   # fully on the ladder now
                    self.v = self.target
        return self.trim

    @property
    def alive(self):
        """True while the rail still owns the vertical -- locked, holding, or decaying
        to the feed-forward. Once dead, the gate-vertical loop resumes."""
        return self.det_t is not None and self.k_w > 0.02


def rail_vert_targets(sched, anchor_target, anchor_leg=2, per_deg=0.0):
    """Per-leg v_converge targets, derived from the course slopes like the ladder.

    WHY IT CAN BE PER-LEG AT ALL: the vanishing point sits where it does because of the
    path's slope against a FIXED camera pitch, so a steeper leg puts it LOWER in frame.
    A single target fitted on one leg is therefore slightly wrong on the others.

    `per_deg` is v-units per degree of extra slope and DEFAULTS TO 0.0 -- i.e. one fixed
    target everywhere -- because the scale has not been measured: it needs the camera's
    vertical FOV, or a flight that logs v_converge across two legs of known slope. That
    measurement is one --log-tube flight away (read the per-leg v_converge medians out of
    archive/notes/design-specs/protocol.py); until then a derived table would be invented precision.
    """
    if not sched:
        return []
    ref_i = max(0, min(len(sched) - 1, anchor_leg))
    ref = float(sched[ref_i]["slope_deg"])
    return [round(anchor_target + per_deg * (float(r["slope_deg"]) - ref), 4)
            for r in sched]


def build_vertical_schedule(n_legs, down_spec, up_spec, down_default, up_default):
    """PER-LEG gate-vertical authority, indexed by ACTIVE_GATE -- a table, like the
    descent ladder, not a hardcoded gate-1/gate-2 fence.

    The control law must not know which gate is "the near-level one" and which is "the
    steep one": that is a property of a COURSE, and it changes with the track and the
    gate count. So authority is a vector the caller supplies, the loop just indexes it,
    and a different course is a different table rather than a different if-statement.

    `down_spec` / `up_spec` are comma-separated lists (or None). Rules:
      * None            -> the scalar default is used on every leg (behaviour unchanged)
      * shorter than    -> the LAST entry fills the remaining legs, so "0.045,0.08" means
        n_legs             "gentle on the first approach, aggressive thereafter" on a
                           course of any length
      * index past the  -> clamped to the last entry (same rule, applied at lookup)
        end

    Returns [(down, up)] of length n_legs, all values positive magnitudes.
    """
    def parse(spec, dflt):
        if spec is None:
            vals = [float(dflt)]
        else:
            vals = [float(x) for x in str(spec).replace(" ", "").split(",") if x != ""]
            if not vals:
                vals = [float(dflt)]
        if any(v < 0.0 for v in vals):
            raise ValueError(f"vertical authority must be a positive magnitude: {vals}")
        return [vals[min(i, len(vals) - 1)] for i in range(max(n_legs, 1))]

    return list(zip(parse(down_spec, down_default), parse(up_spec, up_default)))


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
    # --tube-steer-area-min defines a SOLID tube reading: the threshold at which the rail
    # may steer under --tube-lateral, and the one the tube_solid log column uses either
    # way. Without --tube-lateral the tube is log-only, as it has been since flt12.
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
    # ASYMMETRIC ag==2 clamp. Gate 3 is always RIGHT of gate 2, so on that leg a LEFT
    # command is chasing parallax or noise and spends runway the leg does not have
    # (filt63: +2.5 deg left out of gate 2, then again mid-leg, before reversing right).
    # None = symmetric, i.e. the upper bound stays gb and the command is bit-identical.
    gate_bank_ag2_left = (math.radians(args.gate_max_bank_ag2_left)
                          if args.gate_max_bank_ag2_left is not None else None)
    gate_size_min = args.gate_bank_size_min
    # LEG-2 EARLY ENGAGE. Gate 3 is only ever a small distant centroid on that leg, so the
    # global 0.10 threshold keeps vision silent through most of it and the drone flies the
    # leg open loop. A lower threshold lets the real drift be read; the size RAMP below is
    # what keeps a tiny centroid from slamming the bank. Defaults to the global value, so
    # unless it is passed the ag==2 leg is bit-for-bit unchanged too.
    gate_size_min_ag2 = (args.gate_bank_size_min_ag2
                         if args.gate_bank_size_min_ag2 is not None
                         else args.gate_bank_size_min)
    # --- RAIL LATERAL IS DISARMED PENDING DETECTOR VALIDATION ------------------------
    # The tube detector was rebuilt to extract the two cyan RAIL LINES (curve fits)
    # instead of masking a cyan fill and measuring its area. That changes what u_tube
    # MEANS -- it is now the path centre at a fixed look-ahead row, from a quadratic fit,
    # not a band average of a blob -- and it retires area_frac as a validity test.
    # Every number that tuned the rail lateral law (the |u_tube| <= 0.222 range that set
    # --k-tube-bank 0.6, the area>=0.015 solidity gate, the 3.4% solid rate) was measured
    # against the OLD signal and does not transfer. Re-arming the law on the new signal
    # without re-measuring is exactly how flt10 and flt12 broke gate 1.
    # So this is a hard stop, not a warning: the flag parses (old command lines still
    # work) but refuses to fly until the detector is validated on real gate-3-leg frames.
    if args.tube_lateral:
        raise SystemExit(
            "[tube] --tube-lateral is DISARMED: the rail detector was rebuilt "
            "(rails-as-curves, no area gate) and every constant that tuned the lateral "
            "law was fitted to the OLD signal. Validate the detector on captured frames "
            "first:  python tools/tube_rail_check.py  -- then re-derive --k-tube-bank / "
            "--tube-max-bank-deg from the new u_tube range before re-arming. "
            "Use --log-tube to MEASURE the rails in flight with zero control authority.")
    # RAIL clamps, in rad. tube_lead_max is deliberately its OWN clamp and not a fraction
    # of tube_bank: the curvature lead is the term that crashed flt12, and it is bounded
    # on its own so it can never dominate the u_tube term it is meant to lead.
    tube_bank = math.radians(args.tube_max_bank_deg)
    tube_lead_max = math.radians(args.tube_lead_max_deg)
    gate_fine_max = math.radians(args.gate_fine_max_deg)
    rail_lat_bank = math.radians(args.rail_lat_max_deg)
    # GATE-CENTRE FLOOR, clamped to the open-loop bounds: a floor outside them could
    # never be reached, so it would silently be no floor at all.
    gate_floor_thrust = float(max(cfg.ol_thrust_lo,
                                  min(cfg.ol_thrust_hi, args.gate_vert_floor_thrust)))
    # RAIL DETECTOR geometry overrides -- band edges, look-ahead row, and how far the
    # look-ahead is held below the convergence. Applied to a COPY so the shared cfg (and
    # every other consumer of it) is untouched.
    _band = list(cfg.rail_band)
    if args.rail_band_top is not None:
        _band[0] = args.rail_band_top
    if args.rail_band_bottom is not None:
        _band[1] = args.rail_band_bottom
    rail_cfg = replace(
        cfg, rail_band=tuple(_band),
        rail_lookahead=(cfg.rail_lookahead if args.rail_lookahead is None
                        else args.rail_lookahead),
        rail_merge_recover=args.rail_merge_recover,
        rail_lookahead_below_conv=(cfg.rail_lookahead_below_conv
                                   if args.rail_lookahead_below_conv is None
                                   else args.rail_lookahead_below_conv))
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
    # RAIL VERTICAL per-leg v_converge targets (one fixed target unless a per-degree
    # scale is supplied -- see rail_vert_targets for why that defaults to off).
    vert_auth = build_vertical_schedule(len(sched), args.vert_auth_down,
                                        args.vert_auth_up,
                                        args.gate_vert_down_auth,
                                        args.gate_vert_up_auth)
    rail_v_target = rail_vert_targets(sched, args.rail_vert_target,
                                      anchor_leg=args.descent_anchor_leg,
                                      per_deg=args.rail_vert_target_per_deg)
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
        thrust_s += (f" | LEVEL-LOCK "
                     f"{f'band {args.gate_vert_level_band:.2f} (descent x min(1,|v_f|/band) while a gate is live and v_f>=0, on every descending leg)' if args.gate_vert_level_band > 0.0 else 'off (--gate-vert-level-band 0)'}")
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
        _tbl = (args.vert_auth_down is not None or args.vert_auth_up is not None)
        thrust_s += (f" + GATE-VERT trim "
                     + (f"PER-LEG by active_gate "
                        + " ".join(f"ag{i}[{-d:+.3f},{u:+.3f}]"
                                   for i, (d, u) in enumerate(vert_auth))
                        + " (last entry fills any higher ag)"
                        if _tbl else
                        f"[{-args.gate_vert_down_auth:+.3f},"
                        f"{args.gate_vert_up_auth:+.3f}] (global)")
                     + f" kv={args.k_thrust_v} "
                     f"kdv={args.kd_v} size>={args.gate_vert_size_min:.2f} "
                     f"dead-decay={args.gate_vert_decay_s}s "
                     f"-> bias {coast_bias:+.3f}"
                     f"{' (= HOLD the leg baseline)' if ladder_on else ''}")
        if args.gate_vert_commit_size > 0.0:
            thrust_s += (f" + VERT COMMIT sz_f>={args.gate_vert_commit_size:.2f} on "
                         f"ag<={args.gate_vert_commit_to_ag} ONLY: freeze the PD, ease "
                         f"the held trim to the leg baseline over "
                         f"tau={args.gate_vert_commit_tau:.2f}s (reversible, not a "
                         f"latch). ag>{args.gate_vert_commit_to_ag} keeps the LIVE PD so "
                         f"the arrest can fire (flt40 plunged at gate 2 with the commit "
                         f"applied everywhere)")
        else:
            thrust_s += " + VERT COMMIT off (--gate-vert-commit-size 0)"
    if args.gate_vert_floor:
        thrust_s += (
            f" | *** GATE-CENTRE FLOOR {gate_floor_thrust:.3f} *** engages while a gate "
            f"is live AND v_f <= {args.gate_vert_floor_vthresh:+.2f} (a hair BEFORE dead "
            f"centre, so it FLARES rather than catches); thrust is then held AT OR ABOVE "
            f"the floor -- max(), so the up-trim may still climb, only sinking is denied. "
            f"OVERRIDES the ladder descent and the VCOMMIT freeze (both sit upstream), "
            f"and is RE-APPLIED past the thrust slew limiter -- on segs 1-3 the "
            f"baseline sits at ol_thrust_lo {cfg.ol_thrust_lo:.3f}, so through the "
            f"{args.thrust_slew:.2f}/s up-slew alone the floor would take "
            f"{(gate_floor_thrust - cfg.ol_thrust_lo) / max(args.thrust_slew, 1e-9):.2f}s "
            f"= ~{(gate_floor_thrust - cfg.ol_thrust_lo) / max(args.thrust_slew, 1e-9) * args.cruise:.1f}m "
            f"of travel to arrive. Releases as each gate is passed. ALL gates.")
    if args.rail_vert:
        thrust_s += (
            f" | RAIL VERT (ag>={args.rail_vert_from_ag}) v_converge -> trim: "
            f"target {args.rail_vert_target:+.2f}"
            f"{f' (per-leg {rail_v_target})' if args.rail_vert_target_per_deg else ' (SAME on every leg)'}"
            f" k={args.rail_vert_k} auth [-{args.rail_vert_down_auth:.3f},"
            f"+{args.rail_vert_up_auth:.3f}] tau={args.rail_vert_tau:.2f}s; "
            f"on LOSS hold {args.rail_vert_hold_s:.2f}s then decay to the LADDER FF over "
            f"{args.rail_vert_decay_s:.2f}s (0.0 trim = this leg's slope-sized descent, "
            f"NOT level); handoff tau {args.rail_auth_tau:.2f}s. "
            f"ag<{args.rail_vert_from_ag} UNTOUCHED: authority forced to EXACTLY 0.0. "
            f"*** TARGET IS A ONE-FRAME MEASUREMENT (gate centred -> v=-0.346 on the "
            f"START->g1 leg). Steeper legs sit HIGHER; read this leg's median v_converge "
            f"out of the log and retune. ***")
    filt_s = (f"GATE FILTER tau={args.gate_filter_tau:.2f}s hold={args.gate_filter_hold_s:.2f}s "
              f"fade={args.gate_filter_decay_s:.2f}s (shared by BOTH loops)")
    if args.kd_lat > 0.0:
        print(f"[tube] *** WARNING: --kd-lat {args.kd_lat:.2f} is ON. flt6 measured this "
              f"NET-NEGATIVE: gate-2 lateral error is MONOTONIC, so D reinforces P and "
              f"drove the bank to the clamp (gate#2 u_err +0.919 vs +0.843 without). "
              f"Default is 0.0. ***")
    leg1_s = ""
    if args.post_gate_hold_s > 0.0:
        leg1_s += (f" | POST-GATE HOLD: hold gate-1 exit cmd for "
                   f"{args.post_gate_hold_s:.1f}s, decay tau "
                   f"{args.post_gate_hold_decay:.1f}s, resume at size>="
                   f"{POST_GATE_HOLD_SIZE:.2f}, live-corr "
                   f"{POST_GATE_HOLD_CORR * 100:.0f}%.")
    if args.approach_bank:
        leg1_s += (f" | APPROACH BANK {args.approach_bank:+.1f}deg "
                   f"{'LEFT' if args.approach_bank > 0 else 'RIGHT'} (on ag==0, the "
                   f"gate-1 approach; summed with centring then clamped)")
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
               f"clamp {args.gate_max_bank_deg:.0f}deg size>={gate_size_min:.2f}")
    if args.rail_lateral_primary:
        steer_s += (
            f" | *** LATERAL=RAIL PRIMARY from ag>={args.rail_lateral_from_ag} *** "
            f"u_tube -> 0, k={args.rail_lat_k} clamp {args.rail_lat_max_deg:.0f}deg, NO "
            f"curvature lead (the flt12 term), NO gate fine-trim. GATE-CENTRING is the "
            f"FALLBACK and fades back in over {args.tube_auth_tau:.2f}s whenever the "
            f"rail lock drops. "
            f"RAIL BAND {rail_cfg.rail_band[0]:.2f}..{rail_cfg.rail_band[1]:.2f}H, "
            f"look-ahead {rail_cfg.rail_lookahead:.2f}H held >= "
            f"{rail_cfg.rail_lookahead_below_conv:.2f}H BELOW the convergence "
            f"(adaptive: keeps the lock as the path sinks). "
            f"NOTE ag=0 authority is what flt10/flt12 had when they broke gate 1 -- on "
            f"the OLD detector. VERIFY GATE 1 STILL PASSES.")
    if args.tube_lateral:
        steer_s += (f" | LATERAL=TUBE(closed) from ag>={args.tube_lateral_from_ag}: "
                    f"k={args.k_tube_bank} clamp {args.tube_max_bank_deg:.0f}deg, "
                    f"curvature lead {cfg.k_tube_lead} clamped SEPARATELY to "
                    f"{args.tube_lead_max_deg:.0f}deg (the flt12 fix), area>="
                    f"{args.tube_steer_area_min:.3f}; authority = the rail's own decay "
                    f"weight, so a lost tube HOLDS then fades the GATE law back in. "
                    f"GATE demoted to a {args.gate_fine_max_deg:.1f}deg FINE-TRIM, "
                    f"admitted only while sz_f>={args.gate_fine_size_min:.2f} AND "
                    f"|u_f|<={args.gate_fine_uerr_max:.2f} (the anti-decoy rule). "
                    f"ag<{args.tube_lateral_from_ag} UNTOUCHED: rail authority forced to "
                    f"EXACTLY 0.0 -- gates 1 and 2 fly bit-for-bit the shipped law, and "
                    f"the tube detector is not even RUN there, so their loop rate is "
                    f"unchanged too")
    elif not args.rail_lateral_primary:
        steer_s += " -- the ONLY steering law (no rail lateral flag passed)"
    # RAIL DETECTOR: report where it actually runs and at what resolution -- the reader
    # needs to know whether a leg's missing rail columns mean "no lock" or "not measured".
    if args.log_tube:
        det_where = "EVERY leg (--log-tube)"
    else:
        _from = [a for a, on in
                 ((args.tube_lateral_from_ag, args.tube_lateral),
                  (args.rail_vert_from_ag, args.rail_vert),
                  (args.rail_lateral_from_ag, args.rail_lateral_primary)) if on]
        det_where = (f"ag>={min(_from)} onward" if _from else None)
    steer_s += (f". RAIL DETECTOR "
                + ("OFF (not computed at all)" if det_where is None else
                   f"ON {det_where}, "
                   f"{'FULL res ~8.0 ms/frame (--rail-full-res)' if args.rail_full_res else 'HALF res ~3.4 ms/frame (reuses the gate detector array; agrees with full res to 0.0002 in offset)'}"))
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
    # --tube-lateral needs the detector too, but ONLY on the legs the rail actually
    # steers (see the per-leg gate at the measure() call). --log-tube keeps its old
    # meaning: run it on EVERY leg, for instrumentation.
    tube_det = (TubeDetector(rail_cfg)
                if (args.log_tube or args.tube_lateral or args.rail_vert
                    or args.rail_lateral_primary) else None)
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

    # FRAME DUMP (--dump-frames-leg1): log-only diagnostic, its own writer thread.
    dumper = None
    if args.dump_frames_leg1:
        dumper = FrameDumper(os.path.abspath(args.dump_frames_dir), rail_cfg,
                             every_s=args.dump_frames_every, pre_s=args.dump_frames_pre,
                             queue_max=args.dump_frames_queue)
        dumper.start()
        print(f"[tube] FRAME DUMP -> {dumper.outdir} every {args.dump_frames_every:.2f}s "
              f"while ag==1, plus the last {args.dump_frames_pre:.1f}s before gate 1 "
              f"(written retroactively). Writer is a BACKGROUND thread; control loop "
              f"unaffected, no control law changed.")

    h = _harness(args)
    if h is None:
        vision.stop()
        if dumper is not None:
            dumper.stop()
        return 2
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
    exit_k = 1.0          # GATE-2 EXIT LEVEL weight (ag==1 only), 1 = full authority
    exit_latched = False  # GATE-2 EXIT LEVEL: armed by the first size crossing on ag==1
    leg2_arrest_t0 = None # LEG-2 ENTRY ARREST: wall time of the ag->2 crossing
    gate3_desc_armed = False  # GATE-3 TERMINAL DESCENT: hysteresis arm/release latch
    g3hold_active = False     # GATE-3 BLIND HOLD: sinking on through the post-loss coast
    g3hold_t0 = None          # ...wall time the hold armed (for the bounded timer)
    g3_last_valid_vf = 0.0    # v_f on the last VALID Gate-3 frame
    g3_last_valid_sz = 0.0    # sz_f on the last VALID Gate-3 frame
    g3_loss_t = None          # wall time of the valid->invalid transition
    g3_desc_was_active = False  # was the descent cutting on that last valid frame?
    commit_latched = False  # CLOSE-RANGE COMMIT: locked once close AND centred
    commit_armed = False    # ...and only after sz_f cleared the PREVIOUS gate's residual
    commit_align_count = 0  # consecutive centred frames toward the stability window
    u_f_prev = None         # previous filtered u_f, for du_f/dt on the commit qualifier
    commit_stable_s = 0.0   # accumulated align+rate dwell, SECONDS (derivative path)
    cm_prev_ag = 0          # for resetting both on a gate change
    prev_ag = -1          # FRAME DUMP: for the 0 -> 1 transition (ring flush)
    rail_v_auth = 0.0     # RAIL VERTICAL authority, 0 = the gate loop owns the trim
    rail_auth_k = 0.0     # RAIL lateral authority, 0 = the gate law owns the bank
    gate_fine_k = 0.0     # RAIL lateral: the lagged gate fine-trim (rad)
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
    # --- POST-GATE PATH HOLD state (--post-gate-hold-s) ------------------------------
    pg_latched = 0.0            # LATCHED pre-close heading of the gate being flown (rad)
    pg_cap_open = True          # latch open? shuts once the gate passes CAPTURE_MAX
    pg_hold_cmd = 0.0           # the held command, captured+capped at the gate pass
    pg_hold_t0 = None           # wall time of that capture; None = no gate passed yet
    pg_prev_ag = 0              # for edge-detecting a gate advance
    u_loss = v_loss = sz_loss = v2_loss = u2_loss = 0.0
    # --- RAIL (tube) signal: its OWN filter, same constants, separate state -----------
    # The gate filter above is untouched; this is a parallel one so the two signals can
    # be lost and re-acquired independently (on filt9's gate-3 leg the gate was live for
    # most of the leg while the tube was solid on ~8% of ticks).
    rail = (RailSignal(args.gate_filter_tau, args.gate_filter_hold_s,
                       args.gate_filter_decay_s)
            if (args.log_tube or args.tube_lateral or args.rail_vert
                or args.rail_lateral_primary) else None)
    # --- RAIL VERTICAL (--rail-vert): v_converge -> thrust trim ----------------------
    rail_v = (RailVertical(args.rail_vert_tau, args.rail_vert_hold_s,
                           args.rail_vert_decay_s, args.rail_vert_k,
                           args.rail_vert_target, args.rail_vert_up_auth,
                           args.rail_vert_down_auth) if args.rail_vert else None)
    filt_t = None               # last filter update (dt source)
    # dt_f is assigned inside the vision branch, which --no-steer skips entirely. The
    # rail-vertical loop reads it every tick, so it needs a defined value from tick one.
    dt_f = 0.0
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
                      "descent_on,desc_leg,descent_bias,descent_gain,v_up_auth,v_down_auth,gate_v_ok,v_committed,floor_on,"
                      "gate_found,u_err,v_err,size_frac,cy,"
                      "u_f,u_f2,du_lat,v_f,sz_f,sig_k,live,far_rej,"
                      "gate_cmd_deg,committed,commit_k,"
                      "steer,bank_seg,ff_bank_deg,trim_deg,des_roll_deg,roll_n,"
                      "est_pitch_deg,gyro_pitch,des_pitch_deg,pitch_n,"
                      "tube_found,u_tube,curvature,area_frac,est_roll_deg,hold_on,"
                      "tube_solid,u_tube_f,curv_f,rail_k,rail_auth,rail_cmd_deg,"
                      # RAIL DETECTOR: the lock and its geometry. v_converge is the
                      # vanishing-point row; rail_sep/fit_rms/n_rows say WHY it locked
                      # or did not, which is what a lock-rate post-mortem needs.
                      "v_converge,rail_sep,fit_rms,rail_rows,rail_ylook,rail_merged,rail_ok,"
                      # RAIL VERTICAL: the filtered v, its target, the trim it asked for
                      # and how much authority it actually had.
                      "rail_v_f,rail_v_tgt,rail_v_trim,rail_v_auth,rail_v_live,"
                      # encap_hex is LAST so every existing column keeps its position
                      # and older readers/parsers are unaffected.
                      "loop_hz,det_hz,encap_hex,"
                      # POST-GATE PATH HOLD. tube_found is already a column above, so it
                      # is not duplicated here. Appended AFTER encap_hex so no existing
                      # column index shifts.
                      "post_gate_hold_active,hold_bank_cmd,gate2_size_frac,"
                      # ag==2 size-ramped gate authority; 1.0 on every other leg.
                      "ag2_auth,"
                      # GATE-2 EXIT LEVEL weight; 1.0 except on ag==1 at high size.
                      "gate2_exit_k,"
                      # LEG-2 ENTRY ARREST: the thrust boost currently applied.
                      "leg2_arrest,"
                      # CLOSE-RANGE COMMIT: alignment latch (0/1) and the consecutive
                      # centred-frame streak that arms it.
                      "commit_latched,commit_align_count,"
                      # GATE-3 TERMINAL FLARE: engaged flag, the thrust it holds, and the
                      # raw correction it requested before the max()/ceiling.
                      "gate3_flare_on,gate3_flare_thr,gate3_flare_uncl,"
                      # GATE-3 TERMINAL DESCENT: engaged flag and the signed thrust cut.
                      "gate3_desc_on,gate3_desc_raw,"
                      # COMMIT QUALIFIER: the u_f derivative and the two conditions the
                      # seconds-based dwell is accumulated from.
                      "du_f,rate_ok,align_ok,commit_stable_s,"
                      # GATE-3 BLIND HOLD: arm precondition, whether it is cutting, the
                      # timer, and the last-valid state the arm test read.
                      # g3_loss_t is on the RUN clock (same basis as the t column), not
                      # the raw epoch wall time the state holds.
                      "g3hold_armed,g3hold_on,g3hold_elapsed,"
                      "g3_last_vf,g3_last_sz,g3_loss_t,"
                      # The signed cut actually requested this tick, from EITHER path
                      # (they are mutually exclusive: valid-only vs invalid-only).
                      # Requested, not delivered -- the ol_thrust_lo clamp can absorb part
                      # of it; read the thrust column for what was commanded.
                      "g3_delta_app,"
                      # AUTHORITATIVE COURSE-COMPLETE SIGNAL. `fin` is the race-finish
                      # field the sim ships in the SAME ENCAPSULATED_DATA payload as
                      # active_gate; fin>0 is what ends the flight (see the COMPLETE
                      # break). It was previously visible only in the console line, so a
                      # CSV alone could not tell a finished course from a timeout.
                      "race_fin\n")
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
            # FRAME DUMP: gate 1 is confirmed passed -- commit the buffered approach
            # frames, which are now known to be the real last second before the gate.
            if dumper is not None and ag >= 1 and prev_ag == 0:
                n = dumper.flush_pre_gate()
                print(f"[tube] FRAME DUMP: flushed {n} pre-gate-1 approach frame(s)")
            prev_ag = ag

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
            # --- PER-LEG GATE-VERT AUTHORITY (a TABLE, indexed by active_gate) ----
            # One global clamp cannot serve a course whose legs differ: a near-level
            # approach needs a gentle vertical or it clips the gate, while a steep leg
            # needs aggressive descent AND a strong arrest. MEASURED on filt31: the
            # gate-1 approach already drives the trim to +0.048 against a 0.06 clamp
            # with |v_f| only 0.024 (so a bigger global clamp buys only over-climb),
            # while the next leg sat pinned to the DOWN clamp for 83% of its length and
            # still rode high. Those are opposite tunings.
            # The law does NOT know which leg is which -- it indexes a table the caller
            # supplies, clamped to the last entry, so a different course is a different
            # table rather than a different branch.
            v_down_auth, v_up_auth = vert_auth[min(ag, len(vert_auth) - 1)]

            # --- LATERAL: closed loop on the tube (the ONLY feedback) ---
            est_roll, est_pitch = gravity_to_roll_pitch(grav)
            bank_bias = trim = 0.0
            live = sig_alive = False        # --no-steer never runs the filter
            du_lat = 0.0
            gate_cmd = 0.0                  # the gate-centring command (the only one)
            tube_solid = committed = False
            gate_v_ok = False               # stays False when --gate-vert is off
            v_committed = False             # VERTICAL close-range commit engaged
            descent_gain = 1.0              # LEVEL-LOCK scale on the ladder descent
            rail_auth = rail_cmd = gate_fine = 0.0   # RAIL lateral (--tube-lateral)
            ag2_auth = 1.0                  # ag==2 size-ramped authority (1.0 elsewhere)
            post_gate_hold_active = 0       # POST-GATE PATH HOLD engaged this tick
            hold_bank_cmd = 0.0             # the held gate-1-exit command (radians)
            floor_on = False                # GATE-CENTRE ALTITUDE FLOOR active
            rail_ok = False                 # rail passed the STEERING-QUALITY gate
            # COMMIT QUALIFIER, per-tick display values. Reset here (not carried) so a
            # tick that never reaches the commit block -- --no-steer, or out of commit
            # scope -- logs what was actually evaluated rather than the last leg's
            # residue. commit_stable_s is persistent state and is NOT reset here.
            du_f = 0.0
            rate_ok = True
            align_ok = False
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
                    # FRAME DUMP: hand the frame to the writer thread and move on. Only
                    # on a NEW frame -- re-dumping an unchanged frame would just make
                    # duplicate files. get_latest_frame() already returns a private copy
                    # and nothing downstream mutates it, so there is no race and no
                    # second copy. ag==0 goes to the ring (written only if we reach gate
                    # 1); ag==1 is written as it happens.
                    if dumper is not None:
                        t_flight = now - rx.go_wall
                        if ag == 0:
                            dumper.hold_pre_gate(t_flight, frame)
                        elif ag == 1:
                            dumper.offer("ag1", t_flight, frame)
                    # PER-LEG detector gate. The tube detector is 13.47 ms/call against
                    # the gate detector's 5.45 -- 71% of the per-frame vision budget --
                    # and that cost is what drove the loop to a 16.2 Hz FLOOR on the
                    # gate-2 leg, which is what made gate 2 inconsistently pass-or-collide.
                    # So under --tube-lateral we only pay it on the legs where the rail
                    # actually steers: legs 0-1 keep the fast loop and the gate-1/gate-2
                    # approaches are untouched, numerically AND in timing. --log-tube is
                    # the explicit "measure everywhere" override, and keeps its old cost.
                    # ascontiguousarray: a bare [::2,::2] VIEW measured ~1 ms/call slower
                    # through the HSV conversion than a packed copy. Hoisted so the rail
                    # detector can REUSE it -- the downsample is then free.
                    half = np.ascontiguousarray(frame[::2, ::2])
                    if tube_det is not None and (args.log_tube
                                                 or ag >= args.tube_lateral_from_ag
                                                 or (args.rail_vert
                                                     and ag >= args.rail_vert_from_ag)
                                                 or (args.rail_lateral_primary
                                                     and ag >= args.rail_lateral_from_ag)):
                        # HALF res by default: 3.4 ms vs 8.0, and measured to agree to
                        # 0.0002 in offset / 0.01 in v_converge. That matters because
                        # --rail-vert runs this every frame on the g1->g2 leg, which is
                        # the gate-2 approach -- the leg whose loop-rate variance already
                        # makes gate 2 inconsistently pass-or-collide.
                        tube = tube_det.measure(frame if args.rail_full_res else half)
                    elif tube_det is not None:
                        tube = None
                    gate = gate_det.detect(half, prefer=gate_prefer)
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
                    # SOLID = the detector LOCKED (both rails fitted). The area test that
                    # used to stand here is gone: area is the wrong question about a thin
                    # bright curve -- a fully visible rail a pixel wide has ~zero area, so
                    # that test reported "not seen" on frames where the path is plainly
                    # visible. That false negative IS the sub-4% lock rate. tube.found now
                    # carries the fit quality, the rail separation and the row count;
                    # area_frac is logged and gates nothing.
                    tube_solid = bool(tube is not None and tube.found)
                    rail.update(now, dt_f, tube_solid,
                                tube.u_tube if tube is not None else 0.0,
                                tube.curvature if tube is not None else 0.0)
                # ENGAGE on the FILTERED size while a detection is live; through a blink,
                # stay engaged on the latch and let the fading u_f retire the trim
                # smoothly. Thresholding sz_f during the fade instead would cut the trim
                # off at whatever value it still held -- the exact jump the filter exists
                # to remove.
                # ag==2 may engage EARLIER (--gate-bank-size-min-ag2). Every other leg
                # uses the same gate_size_min it always has, so ag<2 is untouched.
                size_min_eff = gate_size_min_ag2 if leg2 else gate_size_min
                if live:
                    steer = sz_f >= size_min_eff
                    lat_latch = steer
                else:
                    steer = lat_latch and sig_alive
                # SIZE-RAMPED AUTHORITY, ag==2 only. Engaging early is only safe if the
                # command engages GRADUALLY: at the new threshold the gate is a handful
                # of pixels and its u_err is mostly noise, so authority starts at ~0 and
                # reaches full by --gate-bank-full-size-ag2. On every other leg this is
                # exactly 1.0, and multiplying by 1.0 is bit-identical.
                if leg2:
                    _span = max(args.gate_bank_full_size_ag2 - size_min_eff, 1e-6)
                    ag2_auth = max(0.0, min(1.0, (sz_f - size_min_eff) / _span))
                else:
                    ag2_auth = 1.0
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
                                        + (math.radians(args.approach_bank) if ag == 0 else 0.0)
                                        + (math.radians(args.post_gate1_bank) if leg1 else 0.0)
                                        + (math.radians(args.post_gate2_bank) if leg2 else 0.0))
                # --- LATERAL = GATE CENTRING (the DEFAULT law) -------------------------
                # This is what flies unless --tube-lateral is passed, and it is what still
                # flies on legs 0-1 even then. Below, the rail may take it over from
                # ag>=--tube-lateral-from-ag; see that block for how the two blend and why
                # the two earlier rail attempts (flt10, flt12) failed without a leg fence
                # and a separate clamp on the curvature lead.
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
                # POSITIVE = LEFT, so the ag==2 asymmetry caps only the UPPER bound. The
                # lower bound stays -gb: full RIGHT authority toward gate 3 is untouched.
                # Off leg 2, or with the flag unset, gb_hi IS gb -- bit-for-bit unchanged.
                gb_hi = (gate_bank_ag2_left
                         if (leg2 and gate_bank_ag2_left is not None) else gb)
                gate_cmd = (max(-gb, min(gb_hi,
                                         args.k_gate_bank * LATERAL_SIGN * sign * u_f
                                         - args.kd_lat * du_lat))
                            if steer else 0.0) * ag2_auth
                u_lat = LATERAL_SIGN * sign * u_f if steer else 0.0
                # --- POST-GATE PATH HOLD (--post-gate-hold-s) -------------------------
                # Immediately after gate 1 the next gate is a small distant centroid, and
                # chasing it turns the drone off the heading that just flew gate 1
                # cleanly. So HOLD the gate-1-exit command and let it decay, admitting
                # only a fraction of the live command, until the next gate is big enough
                # to steer on. Nothing downstream changes -- this only reshapes gate_cmd,
                # in RADIANS, the same units it already carries.
                # `gate` is None until the first frame arrives, so every read of it is
                # guarded; the spec's bare `gate.found` would raise on the opening ticks.
                g_found = bool(gate is not None and gate.found)
                # CAPTURE = a LATCH that freezes as the gate closes, not a running
                # last-value. MEASURED on filt57, the running version took the LAST
                # in-window tick before the pass -- but by then the NEXT gate was already
                # acquired, small and off to one side, and its command ramped -1.52 ->
                # +7.92 deg in 70 ms. The latch captured +10.0 (the bank clamp) and flew
                # it for 1.5 s. A size window cannot separate those: both gates pass
                # through 0.10-0.35, so the contamination is a different GATE, not a
                # different size. Latching on the way IN and freezing at
                # POST_GATE_HOLD_CAPTURE_MAX holds the pre-close heading of the gate
                # actually being flown, and the next gate's first sightings arrive after
                # the latch has already shut.
                if g_found:
                    if gate.size_frac >= POST_GATE_HOLD_CAPTURE_MAX:
                        pg_cap_open = False           # gate close -> freeze the latch
                    elif (gate.size_frac >= args.gate_bank_size_min and pg_cap_open):
                        pg_latched = gate_cmd         # RAW cmd, captured pre-reshape
                if ag > pg_prev_ag:                   # ANY gate advance -> re-arm
                    if ag == 2:
                        # LEG-2 hold is deterministic: empirical Gate-3 centering
                        # optimum, NOT captured from vision (that path carries entry
                        # variance; r=0.87 shows the held bank sets the Gate-3 crossing
                        # offset).
                        pg_hold_cmd = math.radians(args.gate2_hold_fixed_deg)
                    else:
                        pg_hold_cmd = max(-HOLD_MAX, min(HOLD_MAX, pg_latched))
                    pg_hold_t0 = now
                    # Reopen and CLEAR for the next gate. Clearing is the fail-safe: a
                    # leg where no mid-range command is ever seen holds STRAIGHT rather
                    # than re-flying the previous gate's stale heading.
                    pg_cap_open = True
                    pg_latched = 0.0
                pg_prev_ag = ag
                # ag >= 1, not ag == 1: the re-arm above now fires on EVERY gate advance,
                # but leaving this fenced to ag==1 would re-arm on 1->2 and then never
                # apply, so the hold would still only exist on the one leg. >= 1 is what
                # actually generalises it. (ag 0 is the spawn->gate-1 approach, where
                # there is no preceding gate pass to hold a heading from.)
                if (args.post_gate_hold_s > 0.0 and ag >= 1
                        and pg_hold_t0 is not None):
                    elapsed = now - pg_hold_t0
                    weak = (not g_found) or (gate.size_frac < POST_GATE_HOLD_SIZE)
                    if elapsed < args.post_gate_hold_s and weak:
                        post_gate_hold_active = 1
                        w = math.exp(-elapsed / max(args.post_gate_hold_decay, 1e-3))
                        gate_cmd = (w * pg_hold_cmd
                                    + (1.0 - w) * (POST_GATE_HOLD_CORR * gate_cmd))
                        hold_bank_cmd = pg_hold_cmd
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
                # ALIGNMENT GATE (--gate-commit-align). Size alone is the wrong trigger:
                # MEASURED on filt81, the commit fired mid-turn and decayed commit_k to
                # ~0 while the drone was still off-centre, so as it drifted left vision
                # saturated (gate_cmd -9 deg) but des_roll stayed ~0 -- it coasted off
                # centre, deaf to its own correction. On the near-pass at 80.5 it happened
                # to be centred when the commit fired and locked a good heading. So lock
                # only when CLOSE **and** CENTRED.
                # THE LATCH IS LOAD-BEARING: once locked it stays locked for the rest of
                # the leg, even if |u_f| later spikes. That spike is the terminal parallax
                # as the gate fills; re-opening on it would hand the reaction back to
                # vision and reintroduce the exact end-of-leg swing this removes.
                # Reset only on a gate change.
                if ag != cm_prev_ag:
                    commit_latched = False
                    commit_armed = False
                    commit_align_count = 0
                    commit_stable_s = 0.0
                    u_f_prev = None         # no du_f across a leg boundary
                cm_prev_ag = ag
                if in_commit_scope:
                    size_ok = sz_f >= args.gate_commit_size
                    # DERIVATIVE-AWARE QUALIFICATION (--gate-commit-rate-max /
                    # --gate-commit-stable-s). MEASURED on filt93B: the latch fired at
                    # sz~0.16 while u_f was CROSSING zero at du_f/dt +0.13 and still
                    # accelerating -- a transient, not alignment. commit_k then decayed to
                    # 0 and swallowed the vision's full -9 deg correction, and the drone
                    # missed left. A frame streak cannot tell a settled signal from a fast
                    # zero-crossing; its RATE can. So require the aligned condition to be
                    # low-rate too, and to hold for a REAL-TIME dwell rather than N frames
                    # (N frames is 3x longer at 20 Hz than at 60 Hz -- the same rule meant
                    # different things run to run).
                    # Replay over the recent Gate-3 approaches: rate 0.11 / dwell 0.15 s
                    # rejects the false latches (filt93B +0.13, filt89.1 +0.19) and keeps
                    # the golden ones (filt87.1 +0.096, filt80.5 -0.012).
                    deriv_on = (args.gate_commit_rate_max > 0.0
                                or args.gate_commit_stable_s > 0.0)
                    # `or deriv_on` only widens the branch when a new flag is set; with
                    # both at their 0.0 defaults this is the original condition exactly.
                    if args.gate_commit_align is not None or deriv_on:
                        # ARM GATE. sz_f does not reset when active_gate advances -- the
                        # PREVIOUS gate's size is still decaying through the filter, and
                        # it is large and (having just been flown through) well centred.
                        # MEASURED on filt84: latched at t=8.30 on sz_f 0.43 with
                        # u_f -0.015, which was gate-2 leftover, BEFORE gate 3 was
                        # acquired at all -- so it committed at the leg START and never
                        # turned toward the next gate.
                        # So require sz_f to first fall BELOW the commit size: the old
                        # gate has cleared and the new one is not big yet. Gate N+1 always
                        # starts small, so this arms naturally every leg and only ever
                        # blocks the residual.
                        if not size_ok:
                            commit_armed = True
                        # STABILITY WINDOW. A single centred frame is not alignment:
                        # MEASURED on filt85 the latch fired on a one-frame crossing at
                        # u_f +0.06, and parallax then amplified that residual to +0.76
                        # by the gate plane. Requiring a STREAK locks only while stably
                        # centred, i.e. during the pre-parallax window, so neither a lone
                        # centred frame nor a blip can trigger it.
                        # The reset only matters BEFORE the latch: a spike that breaks the
                        # streak delays or prevents a premature lock, and can never
                        # un-lock a good one (commit_latched is never cleared here).
                        # du_f/dt on the FILTERED signal over the real loop dt. u_f_prev is
                        # None on the first in-scope tick of a leg, so du_f is 0 there
                        # rather than a spike off the previous leg's heading.
                        du_f = ((u_f - u_f_prev) / dt_f
                                if (u_f_prev is not None and dt_f > 0) else 0.0)
                        # align_ok tolerates --gate-commit-align being unset: the rate rule
                        # can then stand on its own instead of raising on `abs(u_f) <= None`.
                        align_ok = (args.gate_commit_align is None
                                    or abs(u_f) <= args.gate_commit_align)
                        rate_ok = (abs(du_f) < args.gate_commit_rate_max
                                   if args.gate_commit_rate_max > 0.0 else True)
                        if deriv_on:
                            # DERIVATIVE-AWARE PATH: seconds-based dwell. Either failure
                            # zeroes the dwell immediately, so the latch tick is always
                            # itself aligned AND low-rate, with the whole window behind it.
                            if align_ok and rate_ok:
                                commit_stable_s += dt_f
                            else:
                                commit_stable_s = 0.0
                            stable_ok = commit_stable_s >= args.gate_commit_stable_s
                            if (commit_armed and size_ok and stable_ok
                                    and not commit_latched):
                                commit_latched = True
                        elif commit_armed and size_ok:
                            # EXISTING FRAME-BASED PATH -- unchanged.
                            if abs(u_f) <= args.gate_commit_align:
                                commit_align_count += 1
                            else:
                                commit_align_count = 0
                            if commit_align_count >= args.gate_commit_stable_frames:
                                commit_latched = True
                        u_f_prev = u_f
                        committed = commit_latched  # once latched, stays committed
                    else:
                        committed = size_ok         # legacy size-only, unchanged
                    commit_k = lag_step(commit_k, dt_f, 0.0 if committed else 1.0,
                                        args.gate_commit_tau)
                else:
                    committed, commit_k = False, 1.0
                # --- RAIL LATERAL (--tube-lateral): the tube steers, the gate trims -----
                # des_roll = clamp(k_bank * (u_tube + lead * curvature)), the original
                # coast-tube "LATERAL=TUBE(closed)" law, with the two guards the previous
                # attempts lacked:
                #   1. the LEG FENCE (--tube-lateral-from-ag). flt10 armed the rail on
                #      every leg and broke gate 1 -- the tube was solid on 1.7% of ticks
                #      there, so "steer on the tube" meant "steer on nothing".
                #   2. a SEPARATE clamp on the curvature lead (--tube-lead-max-deg). That
                #      term is (upper band - lower band); it goes large and noisy exactly
                #      when the near tube fills the lower band at close range, and in
                #      flt12 it spiked the rail to -11 deg at auth 0.95 and crashed into
                #      gate 1. Clamped on its own it can never dominate the u_tube term.
                # AUTHORITY IS A WEIGHT, NOT A SWITCH: a lagged weight chasing rail.k
                # (the rail filter's own hold-then-fade). Solid tube -> the rail owns the
                # bank; tube lost -> it holds briefly, then fades, and the gate law (with
                # its close-range commit) fades back in. So a leg where the tube is never
                # seen flies bit-for-bit today's proven law rather than flying blind --
                # the honest hedge, because whether the ladder actually brings the tube
                # into view on the gate-3 leg is still an open question a flight has to
                # answer (on flt9 it was solid on 3.4% of that leg's ticks).
                # The LAG is load-bearing in its own right: rail.k is 1.0 the INSTANT the
                # tube is first seen -- it only shapes the fade on loss -- so handing over
                # on rail.k alone steps from one law to the other in a single tick, 8.9
                # deg on filt9's gate-3 leg. That is the slam this law exists to remove,
                # arriving at the handoff instead of at the gate. Same primitive and the
                # same reasoning as ff_lat_step's ramp-back-in.
                # --- RAIL LATERAL PRIMARY (--rail-lateral-primary) --------------------
                # The rail is the STEERING REFERENCE and the gate is the fallback, the
                # reverse of the default law. Built FRESH on the rebuilt u_tube rather
                # than inheriting --tube-lateral's constants, which were fitted to the
                # old fill-and-area signal and do not transfer.
                # The command is pure band-centring, u_tube -> 0. NO curvature lead:
                # that term is what spiked the rail to the clamp and flew flt12 into
                # gate 1, and holding the path centred does not need it.
                # AUTHORITY is the rail filter's hold-then-fade weight, so a lost rail
                # HOLDS briefly and then hands back to gate-centring over the same tau
                # -- the fallback is continuous, never a switch.
                # STEERING-QUALITY GATE. tube.found means "both rails fitted"; it does
                # NOT mean the fit is good enough to bank on. MEASURED on filt29's
                # g1->g2 leg, the 2.7% of ticks that did lock had rms p50 2.76 px (vs
                # 0.93 on the gate-1 leg), separation p50 0.090 (vs 0.483) and |u_tube|
                # p90 0.623 -- which at the shipped gain is a FULL-CLAMP bank command
                # taken off a poorly-conditioned fit of a distant sliver. Steering on
                # those ticks is the "bad signal" case, and it is worse than gate
                # centring. So authority additionally requires a well-conditioned fit.
                rail_ok = bool(tube is not None and tube.found
                               and tube.fit_rms <= args.rail_lat_max_rms
                               and tube.rail_sep >= args.rail_lat_min_sep)
                rail_cmd = gate_fine = 0.0
                if (args.rail_lateral_primary and rail is not None and rail.alive
                        and rail_ok and ag >= args.rail_lateral_from_ag):
                    # The CLOSE-RANGE COMMIT outranks the rail: past --gate-commit-size
                    # the drone flies its approach heading straight through the gate
                    # centre, and a rail bank at that range would steer it off. commit_k
                    # runs 1 -> 0 as the commit engages, so the rail's authority is
                    # retired by the same weight that retires the gate's, and both go to
                    # wings-level together.
                    rail_target = rail.k * commit_k
                    rail_cmd = rail_bank_cmd(LATERAL_SIGN * sign * rail.u, 0.0,
                                             args.rail_lat_k, 0.0, rail_lat_bank, 0.0)
                else:
                    rail_target = (rail.k if (args.tube_lateral and rail is not None
                                              and rail.alive
                                              and ag >= args.tube_lateral_from_ag)
                                   else 0.0)
                rail_auth_k = lag_step(rail_auth_k, dt_f, rail_target, args.tube_auth_tau)
                # Snap to EXACTLY 0 once the lag has retired, so an out-of-scope leg is
                # bit-for-bit the gate law rather than the gate law plus a 1e-9 residue.
                if rail_target == 0.0 and rail_auth_k < 1e-4:
                    rail_auth_k = 0.0
                rail_auth = rail_auth_k
                if rail_auth > 0.0 and not args.rail_lateral_primary:
                    # --tube-lateral's law. (Disarmed at startup, so this cannot run in
                    # flight today; kept intact for when its gains are re-derived.)
                    # Same sign path as the gate term (LATERAL_SIGN * sign), so the rail
                    # lands in the sim's left-positive roll frame exactly as u_f does.
                    rail_cmd = rail_bank_cmd(LATERAL_SIGN * sign * rail.u,
                                             LATERAL_SIGN * sign * rail.curv,
                                             args.k_tube_bank, cfg.k_tube_lead,
                                             tube_bank, tube_lead_max)
                    # The fine trim reads the UNCOMMITTED gate_cmd: the commit exists to
                    # retire the gate as the PRIMARY law, whereas here the u_err window
                    # already retires the trim before the sweep starts. Applying both
                    # would double-retire it.
                    fine_target = gate_fine_trim(gate_cmd, sz_f, u_f,
                                                 args.gate_fine_size_min,
                                                 args.gate_fine_uerr_max, gate_fine_max)
                else:
                    # PRIMARY law adds no gate fine-trim: the whole point is that the
                    # rail alone holds the path centred, so mixing the gate back in at
                    # close range would reintroduce the parallax lunge it replaces.
                    fine_target = 0.0
                # LAGGED, because the admission window is a HARD test on sz_f and |u_f|:
                # the tick it opens or closes, an unlagged trim would step by the full
                # clamp. Measured on filt9's gate-3 leg that is a 3.4 deg jump on a
                # rail-owned tick -- a slam introduced by the anti-decoy guard itself.
                # Lagging the trim VALUE (not a separate weight) is smooth in both
                # directions and needs no extra state.
                gate_fine_k = lag_step(gate_fine_k, dt_f, fine_target, args.tube_auth_tau)
                gate_fine = gate_fine_k
                trim = blend_lateral(rail_auth, rail_cmd, gate_fine, commit_k, gate_cmd)
                # --- GATE-2 EXIT LEVEL (--gate2-exit-level-size), ag == 1 ONLY ---------
                # The drone crosses gate 2 still banked ~5-8 deg left (the
                # --post-gate1-bank approach bank), carries that left momentum onto the
                # gate-2->3 leg, and 11 deg of right vision authority cannot reverse it
                # before gate 3 leaves the FOV -- u_err runs to +0.99 and it crashes
                # around t=14, identically on filt57/58/60.
                # So retire the WHOLE lateral command as the gate fills: exit_k slews
                # 1 -> 0 over --gate2-exit-level-tau and scales the backbone AND the trim
                # together, easing des_roll to 0 right at the gate plane.
                # THE HIGH THRESHOLD IS THE POINT. The +5 left bank must fly the entire
                # approach and only let go in the last ~0.2 s. flt13 levelled mid-approach
                # and clipped gate 2; this is deliberately later than that.
                # Reversible, not a latch: drop back below the size and exit_k eases home.
                # SCOPE IS LOAD-BEARING: ag 0 and ag >= 2 force exit_k to EXACTLY 1.0, so
                # those legs are bit-for-bit unchanged (1.0 * x == x in IEEE-754), and the
                # ag >= 2 close-range commit is untouched.
                # LATCHED, not live-gated. filt62: the size test released the moment the
                # gate swept past -- still on ag==1, before active_gate ticked to 2 --
                # exit_k eased back to 1.0 and des_roll slammed to +11.5 deg LEFT (the
                # approach bank plus vision chasing the small next gate), re-imparting
                # exactly the left momentum the levelling had just removed. So the FIRST
                # crossing arms a latch and the target stays 0.0 for the rest of the leg,
                # holding the whole command level through the tail.
                if ag != 1:
                    exit_latched = False          # the latch lives only on the ag==1 leg
                if args.gate2_exit_level_size > 0.0 and ag == 1:
                    if sz_f >= args.gate2_exit_level_size:
                        exit_latched = True       # first crossing arms it, for good
                    exit_k = lag_step(exit_k, dt_f, 0.0 if exit_latched else 1.0,
                                      args.gate2_exit_level_tau)
                else:
                    exit_k = 1.0
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
                                   min(ff_bank_max,
                                       exit_k * (bb_scale * bank_bias + trim)))
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
            # GATE-VERTICAL ENGAGEMENT, resolved BEFORE the baseline so the level-lock
            # below can read it. Mirrors the lateral: threshold the FILTERED size while a
            # detection is live, ride the latch through a blink. The measurement filter
            # owns ALL continuity -- the old trim-level hold/decay is gone, because two
            # cascaded hold-then-decay stages compound into a lag neither one describes,
            # and the two loops would no longer be reading one target.
            # (Hoisted from the --gate-vert block; it is a pure function of the filter
            # state, so computing it earlier changes nothing about what it evaluates to.)
            if live:
                vert_latch = sz_f >= args.gate_vert_size_min
                gate_v_ok = vert_latch
            else:
                gate_v_ok = vert_latch and sig_alive
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
                # --- LEVEL-LOCK (--gate-vert-level-band) --------------------------
                # The ladder's descent is a CONSTANT per leg: it keeps sinking even once
                # the gate is vertically centred, so the drone arrives at the gate plane
                # still going down and passes below the middle. The gate-vert trim can
                # only fight that with its own authority; the descent itself never lets
                # up. So fade the descent out as the gate rises to centre:
                #   v_f  > 0  gate BELOW centre = we are still too HIGH -> keep descending
                #   v_f -> 0  gate AT centre    = we are AT the gate's height -> level off
                # gain is 1.0 at |v_f| >= band and eases linearly to 0 at v_f == 0, so the
                # descent is given up smoothly rather than switched off.
                # ONLY for v_f >= 0. A strongly negative v_f means the gate is ABOVE
                # centre -- the drone is already LOW -- and there the ladder's descent is
                # not what needs trimming; suppressing it would be suppressing lift the
                # v-trim is separately trying to add. That case keeps normal behaviour.
                # Guarded on band > 0.0, which is both the OFF switch and the
                # divide-by-zero guard.
                if (args.gate_vert_level_band > 0.0 and gate_v_ok
                        and descent_on and v_f >= 0.0):
                    descent_gain = min(1.0, abs(v_f) / args.gate_vert_level_band)
                base_thrust = base_thrust - leg_bias * descent_gain
            base_thrust = float(max(cfg.ol_thrust_lo, min(cfg.ol_thrust_hi, base_thrust)))
            # --- VERTICAL: gate v-error trims the baseline (--gate-vert) -------------
            # SIGN: v_err > 0 == gate BELOW frame centre == we are looking DOWN at it ==
            # too HIGH -> trim NEGATIVE -> less thrust -> descend. (Same sign as the
            # shipped HybridController law, thrust = ff - k*v_err.)
            if not args.gate_vert:
                # vtrim persists ACROSS ticks, so it must be re-zeroed here rather than
                # left holding whatever the rail blend wrote last tick -- otherwise the
                # blend's own output would feed back into its next input.
                vtrim = 0.0
                thrust_cmd = base_thrust
            else:
                # --- VERTICAL CLOSE-RANGE COMMIT (--gate-vert-commit-size) ------------
                # The vertical analogue of the lateral commit. Past the commit size the
                # gate fills the frame, v_f gets noisy and the kd_v*dv term differentiates
                # that noise -- it can spike the trim into the down-clamp in the last
                # metres and fly the drone into the gate plane. There is nothing useful
                # left to servo on there anyway: at that range the altitude is already
                # made or missed.
                # So we FREEZE the PD and ease the trim it had earned to the leg baseline
                # (0.0 = fly the ladder's own descent), over --gate-vert-commit-tau.
                # NOT a latch: drop back below the size and the live PD resumes on the
                # next tick, from wherever the trim has eased to -- so re-engagement is
                # continuous, never a step.
                # SCOPE: every gate, deliberately including gate 1 -- unlike the lateral
                # commit's ag>=2 fence. Gate 1 is exactly where a premature dive kills us,
                # and the vertical axis has no equivalent of the leg-1 bank for the fence
                # to protect.
                # SCOPE (--gate-vert-commit-to-ag, an UPPER bound on active_gate). The
                # commit is REQUIRED at gate 1 and FATAL after it, proven both ways by
                # our own runs:
                #   flt41, commit OFF -> GATE 1 CLIPS. The kd_v term reads the gate's
                #     geometric fall in frame as sink, drives the trim to the +0.06
                #     clamp and thrust to 0.340, and the drone climbs into the top bar:
                #     COLLISION at t=4.6 s, tumble.
                #   flt40, commit ON everywhere -> GATE 2 PLUNGES. The PD is frozen
                #     through the crossing, so the trim sits at ~0 (thrust 0.249) while
                #     v runs +0.17 -> -0.90 and the arrest never fires at all.
                # The difference is that gate 1 is a near-level approach where the
                # close-range D term is pure noise, while the descending legs arrive
                # with real sink that something has to stop. So: ON for ag==0, OFF from
                # ag>=1, which is the default.
                v_committed = (args.gate_vert_commit_size > 0.0
                               and ag <= args.gate_vert_commit_to_ag
                               and sz_f >= args.gate_vert_commit_size)
                if gate_v_ok and v_committed:
                    # Hold the last pre-commit trim and slew it to the baseline. Iterating
                    # on gate_trim IS the specified first-order slew here (unlike the
                    # no-gate branch below, which must anchor to a fixed handoff value
                    # because a second decay stage is already acting on it).
                    gate_trim = lag_step(gate_trim, dt_f, 0.0, args.gate_vert_commit_tau)
                    vtrim = gate_trim
                    last_gate_t = now
                elif gate_v_ok:
                    # Dirty derivative of the ALREADY-smooth signal: v_f2 is a second EMA
                    # of v_f at the same tau, so this never differentiates raw detections
                    # and cannot kick across a detection hole -- both stages fade together.
                    dv = (v_f - v_f2) / tau_f
                    gate_trim = max(-v_down_auth,
                                    min(v_up_auth,
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
            # --- RAIL VERTICAL (--rail-vert): the path's own slope drives the trim -----
            # Runs AFTER the gate-vertical block so `vtrim` above is the gate loop's
            # answer, and this blends over it. Out of scope, or before the rail has ever
            # locked, the weight is EXACTLY 0.0 and vtrim is bit-for-bit the gate law --
            # which is what keeps the gate-1 approach untouched.
            # The weight is LAGGED for the same reason the lateral handoff is: the lock
            # is a hard boolean, so switching on it would step the thrust trim.
            if rail_v is not None:
                in_v_scope = ag >= args.rail_vert_from_ag
                v_locked = bool(in_v_scope and tube is not None and tube.found)
                rail_v.update(now, dt_f, v_locked,
                              tube.v_converge if tube is not None else 0.0,
                              target=rail_v_target[min(desc_leg,
                                                       len(rail_v_target) - 1)])
                # The rail keeps the vertical while it is locked, HOLDING, or decaying to
                # the ladder feed-forward. Only once it is fully decayed (trim ~0, i.e.
                # already flying the ladder) does it release to the gate loop, so the
                # handback is from a value that is already the feed-forward -- never a
                # jump back to whatever the gate happens to want.
                rail_v_auth = lag_step(rail_v_auth, dt_f,
                                       1.0 if (in_v_scope and rail_v.alive) else 0.0,
                                       args.rail_auth_tau)
                if not (in_v_scope and rail_v.alive) and rail_v_auth < 1e-4:
                    rail_v_auth = 0.0          # EXACT 0 -> bit-for-bit the gate law
                if rail_v_auth > 0.0:
                    vtrim = rail_v_auth * rail_v.trim + (1.0 - rail_v_auth) * vtrim
            # --- LEG-2 ENTRY ARREST (--leg2-entry-arrest), ag==2 entry ONLY -----------
            # Straight after gate 2 the drone free-sinks at ~0.23 thrust for ~1.2 s
            # before the vertical PD engages, and that early dive is why it arrives at
            # gate 3 below the window (v_err ~ -0.85, consistently). There is no climb
            # authority to recover with, so the sink is PREVENTED rather than corrected:
            # a boost that fires on the crossing and decays linearly to 0 across the
            # window. Linear, not exponential, so it reaches exactly 0 at the end rather
            # than trailing a tail into the leg.
            # The latch is the first tick at ag==2 and clears whenever ag leaves 2.
            if ag != 2:
                leg2_arrest_t0 = None
            elif leg2_arrest_t0 is None:
                leg2_arrest_t0 = now          # the crossing
            leg2_arrest = 0.0
            if (args.leg2_entry_arrest > 0.0 and ag == 2
                    and leg2_arrest_t0 is not None):
                _el = now - leg2_arrest_t0
                if _el < args.leg2_entry_arrest_s:
                    leg2_arrest = args.leg2_entry_arrest * (
                        1.0 - _el / max(args.leg2_entry_arrest_s, 1e-6))
            # Added INSIDE the clamp so it goes through the same ol_thrust_hi ceiling and
            # the slew limiter below -- it can raise the command, never bypass its bounds.
            thrust_cmd = float(max(cfg.ol_thrust_lo,
                                   min(cfg.ol_thrust_hi,
                                       base_thrust + vtrim + leg2_arrest)))

            # --- GATE-CENTRE ALTITUDE FLOOR (--gate-vert-floor) ----------------------
            # ONCE THE DRONE HAS COME DOWN TO A GATE'S LEVEL, IT MAY NOT SINK FURTHER
            # UNTIL IT IS THROUGH. The failure this fixes: on the approach v_f runs
            # +0.6 -> 0 -> -0.7, i.e. the drone descends onto the gate line and straight
            # past it, and the climb reaction only fires once v_f is already negative --
            # by which time there is sink velocity built up that the trim cannot arrest
            # in the metre that is left. That is the consistent under-gate-2 (and -3) hit.
            #
            # v_f > 0 == gate BELOW frame centre == still above the gate. Engaging at
            # v_f <= vthresh (a hair BEFORE dead centre, default 0.10) is what makes it a
            # FLARE rather than a catch: the sink is stopped while the drone is still
            # above the bar, instead of after it has crossed it.
            #
            # max(), never assignment: the floor forbids DESCENT, it does not command an
            # altitude. The up-trim can still climb above it; only sinking is denied.
            #
            # Placed AFTER thrust_cmd is fully formed, so it overrides every term that
            # could pull thrust down -- the altitude ladder's per-leg descent (which is
            # baked into base_thrust) and the VCOMMIT freeze (which owns vtrim at exactly
            # this range) both sit upstream of it and neither can dig under the floor.
            floor_on = False
            if args.gate_vert_floor and gate_v_ok and v_f <= args.gate_vert_floor_vthresh:
                floor_on = True
                thrust_cmd = max(thrust_cmd, gate_floor_thrust)

            # --- GATE-3 TERMINAL FLARE (--gate3-vert-flare), ag==2 ONLY ------------------
            # Gate 3's steep dive loses the gate at v_f~+0.2-0.3, before the floor's 0.10
            # threshold or the live PD's climb reaction can fire; the drone then sinks below.
            # Engage on SIZE + a higher v_f threshold so the sink is arrested while the gate is
            # still tracked. Reversible max(), never a latch; clears the instant ag leaves 2 or
            # the detection drops. Bounded < ceiling, slewed by the limiter below.
            # NOTE: the spec names `gf` here, but that variable is assigned only inside the
            # tick-log block ~200 lines below, so it is NOT in scope at this point. The
            # spec defines it as bool(gate is not None and gate.found); that expression is
            # inlined instead, which is the same test without the NameError.
            gate3_flare_on = False
            gate3_flare_thr = float(max(cfg.ol_thrust_lo, min(cfg.ol_thrust_hi,
                                                              args.gate3_vert_flare_thrust)))
            if (args.gate3_vert_flare and ag == 2 and gate_v_ok
                    and gate is not None and gate.found
                    and sz_f >= args.gate3_vert_flare_size
                    and 0.0 <= v_f <= args.gate3_vert_flare_vthresh):
                gate3_flare_on = True
                thrust_cmd = max(thrust_cmd, gate3_flare_thr)
            # state reset: nothing persists across ticks (no latch), so ag!=2 / invalid
            # detection => condition simply False next tick => bit-for-bit baseline.

            # --- GATE-3 TERMINAL DESCENT (--gate3-vert-descent), ag==2 ONLY --------------
            # Clean runs cross Gate 3 HIGH (v_f > 0 = drone above). A tiny thrust cut near the
            # plane drops the crossing toward centre. Hysteresis arm/release so it can never
            # push the drone below: arm only when clearly high (v_f>=vhi), release the moment
            # v_f eases to the near-centre band (v_f<=vrelease). Reversible; clears when ag
            # leaves 2 or detection drops.
            gate3_desc_on = False
            gate3_desc_raw = 0.0
            _g3d_valid = (args.gate3_vert_descent and ag == 2 and gate_v_ok
                          and gate is not None and gate.found
                          and sz_f >= args.gate3_vert_descent_size)
            if _g3d_valid and v_f >= args.gate3_vert_descent_vhi:
                gate3_desc_armed = True
            if v_f <= args.gate3_vert_descent_vrelease or not _g3d_valid:
                gate3_desc_armed = False
            if _g3d_valid and gate3_desc_armed:
                gate3_desc_on = True
                gate3_desc_raw = -args.gate3_vert_descent_delta
                thrust_cmd = max(cfg.ol_thrust_lo,
                                 thrust_cmd - args.gate3_vert_descent_delta)
            if ag != 2:
                gate3_desc_armed = False    # hard reset off-leg

            # --- GATE-3 DESCENT BLIND HOLD (--gate3-vert-descent-hold-s), ag==2 ONLY --
            # MEASURED on the clean descent-enabled runs: the visible arm->loss window is
            # only ~0.23-0.25 s, and the gate goes away while v_f is still HIGH
            # (+0.38..+0.45) and falling at ~-0.5/s. The blind coast from there to the
            # gate plane is another ~0.54-0.75 s. The descent above releases the instant
            # the detection drops, so the drone stops sinking while still high and half a
            # second short of the plane -- the cut is spent before it can land.
            # So carry the SAME cut through the blind coast, for a bounded time.
            # Replay -> 0.30 s: crossing +0.28 -> ~+0.20, no low-miss exposure, and it
            # expires before the earliest spurious reacquisition (0.38 s).
            #
            # A SEPARATE STAGE, deliberately: the block above only ever fires on a VALID
            # detection, and this one only ever fires on an INVALID one, so the two can
            # never both cut on the same tick and the delta cannot be applied twice.
            g3hold_on = False
            if args.gate3_vert_descent_hold_s > 0.0 and ag == 2:
                valid = (gate_v_ok and gate is not None and gate.found and sz_f > 0.0)
                if valid:
                    # Track the last VALID state; the transition test below reads it.
                    g3_last_valid_vf = v_f
                    g3_last_valid_sz = sz_f
                    g3_desc_was_active = bool(gate3_desc_on)
                    # NOTE: the spec has a low-reacquisition release
                    # (--gate3-vert-descent-hold-release-vf) here, and then clears the
                    # hold unconditionally on the next line. The unconditional clear
                    # subsumes it: ANY valid frame ends the blind hold and hands the
                    # aircraft back to the live descent path, which owns the low case via
                    # its own vrelease hysteresis. Both statements are kept as specified;
                    # the release-vf flag therefore has no observable effect today.
                    if g3hold_active and v_f <= args.gate3_vert_descent_hold_release_vf:
                        g3hold_active = False
                    g3hold_active = False
                    g3_loss_t = None
                else:
                    # ARM only on the valid->invalid TRANSITION, and only if the descent
                    # was actually cutting into a still-high gate at close range. g3_loss_t
                    # is cleared only by a valid frame or by leaving the leg, so a run of
                    # blind ticks cannot re-arm and extend the timer.
                    if (not g3hold_active and g3_loss_t is None
                            and g3_desc_was_active
                            and g3_last_valid_sz >= args.gate3_vert_descent_size
                            and g3_last_valid_vf > 0.0):
                        g3hold_active = True
                        g3hold_t0 = now
                        g3_loss_t = now
                    if g3hold_active:
                        if (now - g3hold_t0) >= args.gate3_vert_descent_hold_s:
                            g3hold_active = False          # timer expiry
                        else:
                            g3hold_on = True
                            # The SAME signed delta that was active at loss, never larger,
                            # and NOT scaled by the stale v_f -- a fixed cut through the
                            # coast. Same clamp and the same slew limiter below.
                            thrust_cmd = max(cfg.ol_thrust_lo,
                                             thrust_cmd - args.gate3_vert_descent_delta)
            if ag != 2:
                # Leaving the leg (including the advance to 3) releases everything.
                g3hold_active = False
                g3hold_t0 = None
                g3_loss_t = None
                g3_desc_was_active = False

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
            # THE FLOOR IS RE-APPLIED PAST THE SLEW LIMITER, and that is what makes it a
            # floor rather than a wish. MEASURED: on the steep legs the ladder baseline is
            # 0.145-0.155, so reaching a 0.250 floor through the 0.15/s UP slew takes
            # 0.63-0.70 s -- at 8 m/s the drone travels ~5 m, and the gate is long past.
            # The slew limiter would therefore have defeated this clamp on precisely the
            # gate-2 and gate-3 approaches it exists to fix.
            # Stepping thrust up is the ARRESTING direction, and the lurch the slew
            # limiter was added to remove was a descent step throwing the course out of
            # frame -- so bypassing it upward, only while the floor is active, buys the
            # flare at the one moment it is worth a step.
            # prev_thrust then tracks the FLOORED value, so on release the ordinary
            # down-slew (0.60/s) walks it back to the leg baseline in ~0.18 s instead of
            # dropping in one tick.
            if floor_on:
                thrust = max(thrust, gate_floor_thrust)
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
                    f"{desc_leg},{leg_bias:.4f},{descent_gain:.4f},"   # LADDER + LEVEL-LOCK
                    f"{v_up_auth:.4f},{v_down_auth:.4f},"   # PER-LEG gate-vert authority
                    f"{int(bool(args.gate_vert and gate_v_ok))},"   # the REAL engagement
                    f"{int(v_committed)},"                 # VERTICAL commit engaged
                    f"{int(floor_on)},"                    # GATE-CENTRE FLOOR active
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
                    # RAIL: the filtered signal, its decay weight, and what it commanded
                    f"{int(tube_solid)},"
                    f"{(rail.u if rail is not None else 0.0):+.5f},"
                    f"{(rail.curv if rail is not None else 0.0):+.5f},"
                    f"{(rail.k if rail is not None else 0.0):.4f},"
                    f"{rail_auth:.4f},{math.degrees(rail_cmd):+.3f},"
                    f"{(tube.v_converge if tube is not None else 0.0):+.5f},"
                    f"{(tube.rail_sep if tube is not None else 0.0):.5f},"
                    f"{(tube.fit_rms if tube is not None else 0.0):.3f},"
                    f"{(tube.n_rows if tube is not None else 0)},"
                    f"{(tube.y_look if tube is not None else 0.0):.1f},"
                    f"{(tube.n_merged if tube is not None else 0)},{int(rail_ok)},"
                    f"{(rail_v.v if rail_v is not None else 0.0):+.5f},"
                    f"{(rail_v.target if rail_v is not None else 0.0):+.4f},"
                    f"{(rail_v.trim if rail_v is not None else 0.0):+.5f},"
                    f"{rail_v_auth:.4f},"
                    f"{int(rail_v.live) if rail_v is not None else 0},"
                    # encap_hex LAST: the full raw race-status payload, for offline
                    # decoding of the 216 bytes past the header. Bare hex, no sign or
                    # precision formatting, and '' until the first payload arrives.
                    f"{hz:.1f},{det_hz:.1f},{rx.encap_hex()},"
                    # POST-GATE PATH HOLD: engaged flag, the held command in DEGREES
                    # (gate_cmd is radians internally), and the live gate size it is
                    # waiting on.
                    f"{post_gate_hold_active},"
                    f"{math.degrees(hold_bank_cmd):+.3f},"
                    f"{(gate.size_frac if gf else 0.0):.5f},"
                    f"{ag2_auth:.4f},"
                    f"{exit_k:.4f},"
                    f"{leg2_arrest:.5f},"
                    f"{int(commit_latched)},{commit_align_count},"
                    f"{int(gate3_flare_on)},{gate3_flare_thr:.3f},"
                    f"{max(0.0, gate3_flare_thr - (base_thrust + vtrim + leg2_arrest)):.3f},"
                    f"{int(gate3_desc_on)},{gate3_desc_raw:+.3f},"
                    f"{du_f:+.4f},{int(rate_ok)},{int(align_ok)},{commit_stable_s:.3f},"
                    f"{int(g3_desc_was_active and ag == 2)},{int(g3hold_on)},"
                    f"{(now - g3hold_t0) if g3hold_t0 is not None else 0.0:.3f},"
                    f"{g3_last_valid_vf:+.3f},{g3_last_valid_sz:.3f},"
                    f"{(g3_loss_t - rx.go_wall) if g3_loss_t is not None else 0.0:.3f},"
                    f"{(gate3_desc_raw - args.gate3_vert_descent_delta) if g3hold_on else gate3_desc_raw:+.4f},"
                    f"{int(fin)}"
                    "\n")

            if now - last >= 0.25:
                last = now
                who = "COMMIT" if committed else ("commit" if commit_k < 0.99 else "steer ")
                # RAIL LOCK STATE, every tick line: found/rms/rows/sep and whether it is
                # actually STEERING. 'lock-REJ' means the rails were fitted but the
                # steering-quality gate refused them -- the case that matters most when
                # reading a flight, because it is the difference between "the path is
                # not visible" and "the path is visible but not good enough to bank on".
                if tube is None or rail is None:
                    tube_s = "RAIL off"
                else:
                    if rail_auth > 0.0:
                        st = f"STEER x{rail_auth:.2f}"
                    elif tube.found:
                        st = "lock-REJ" if not rail_ok else "lock"
                    else:
                        st = "NO-LOCK"
                    tube_s = (f"RAIL[{st}] found={int(tube.found)} "
                              f"rms={tube.fit_rms:.2f}px rows={tube.n_rows} "
                              f"sep={tube.rail_sep:.3f} u={rail.u:+.3f} "
                              f"vconv={tube.v_converge:+.3f}"
                              f"{f' mrg={tube.n_merged}' if tube.n_merged else ''}"
                              f"{f' -> {math.degrees(rail_cmd):+.1f}deg' if rail_auth > 0.0 else ''}")
                gate_s = ("GATE --" if gate is None or not gate.found else
                          f"GATE u={gate.u_err:+.3f} v={gate.v_err:+.3f} "
                          f"sz={gate.size_frac:.3f} -> {math.degrees(gate_cmd):+.1f}deg "
                          f"x{commit_k:.2f}")
                lad_s = (f" LADDER leg{desc_leg} -{leg_bias * descent_gain:.4f}"
                         f"{f' [LEVEL] x{descent_gain:.2f}' if descent_gain < 0.9 else ''}")
                floor_s = (f" [FLOOR {gate_floor_thrust:.3f} v_f={v_f:+.3f}]"
                           if floor_on else "")
                if rail_v is not None and rail_v_auth > 0.0:
                    vert_s = (f" RAILVERT x{rail_v_auth:.2f} "
                              f"{'live' if rail_v.live else 'HOLD/decay'} "
                              f"v={rail_v.v:+.3f}->{rail_v.target:+.2f} "
                              f"trim={rail_v.trim:+.4f} | VERT trim={vtrim:+.4f}")
                else:
                    vert_s = ("" if not args.gate_vert else
                              f"{' [VCOMMIT]' if v_committed else ''} "
                              f"VERT trim={vtrim:+.4f}"
                              f"{'' if last_gate_t is None else ('' if (now-last_gate_t) < 1e-3 else f'(coasting {now-last_gate_t:.1f}s)')}")
                print(f"[tube] seg{seg}"
                      f"{f'/bank{bank_seg}' if bank_seg != seg else ''} "
                      f"ag={ag} thr={thrust:.3f}{floor_s}{lad_s}{vert_s} | [{who}] {tube_s} | {gate_s} | "
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
        # Let the writer finish what is queued -- the last samples are the ones nearest
        # gate 2, i.e. the most interesting ones, and they are still in flight here.
        if dumper is not None:
            wrote, dropped = dumper.stop()
            print(f"[tube] FRAME DUMP: wrote {wrote} sample(s) "
                  f"({2 * wrote} PNGs) to {dumper.outdir}"
                  + (f"  *** DROPPED {dropped} (writer could not keep up -- raise "
                     f"--dump-frames-every or --dump-frames-queue) ***" if dropped else ""))
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


def build_parser():
    """The CLI, built separately from main() so tests can read the shipped DEFAULTS
    directly instead of pattern-matching --help text. Nothing here runs a flight."""
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
    ap.add_argument("--tube-max-bank-deg", type=float, default=10.0, metavar="DEG",
                    help="--coast-tube: HARD clamp on the RAIL bank command under "
                         "--tube-lateral. Default 10 -- the SAME bound --gate-max-bank-deg "
                         "gives the law the rail replaces, deliberately: the tube is a "
                         "better reference than the gate, which is a reason to trust it "
                         "more, not a reason to let it bank harder. (Was 15, a leftover "
                         "from the original tube-steering design that no control path "
                         "read; at 15 the rail could out-bank the gate law it takes over "
                         "from, which is the wrong direction given flt12 crashed on a "
                         "rail spike.) Replayed on filt9's gate-3 leg the rail law never "
                         "reaches this clamp on any tick, where the gate law sat pinned "
                         "to its own for 65%% of them -- see tests/test_rail_lateral.py.")
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
                         "g1->g2 leg); every other leg is DERIVED from it by slope ratio, "
                         "so the anchor keeps its meaning on the leg it was tuned for. "
                         "Default 0.035 is the value the CURRENT gates-1-and-2 baseline "
                         "flies, and it supersedes the 0.026 in "
                         "archive/notes/design-specs/altitude-ladder-spec.md -- that figure was the anchor "
                         "before the ladder was flown, and the spec was not updated when "
                         "0.035 proved out. Pass 0.026 to reproduce the spec's table "
                         "exactly. 0 disables the ladder entirely.")
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
    ap.add_argument("--gate-vert-level-band", type=float, default=0.0, metavar="V",
                    help="--coast-tube LEVEL-LOCK: fade the ALTITUDE LADDER's per-leg "
                         "descent to zero as the gate rises to vertical centre, so the "
                         "drone levels off AT the gate's height instead of sinking "
                         "through it. Default 0.0 = OFF (opt-in per run), and the same "
                         "test is the divide-by-zero guard. "
                         "The ladder's descent is a CONSTANT per leg -- it keeps sinking "
                         "even once v_f is ~0, so the drone reaches the gate plane still "
                         "going down and passes below the middle; only the v-trim opposes "
                         "it, and only within its own authority. With this set, the "
                         "applied descent is scaled by min(1, |v_f| / BAND): full ladder "
                         "descent while the gate sits a full band below centre (drone "
                         "still too high), easing linearly to LEVEL as v_f -> 0. "
                         "Engages while a gate is live (gate_v_ok) on every DESCENDING "
                         "leg, all active_gate values -- no leg fence. "
                         "ONE-SIDED: applied only for v_f >= 0 (gate at or below centre). "
                         "A strongly negative v_f means the drone is already LOW, where "
                         "cutting the descent would be suppressing lift the v-trim is "
                         "separately trying to add -- that case keeps normal behaviour. "
                         "Applies to the ALTITUDE LADDER path only; --ff-backbone's own "
                         "vertical feed-forward is unaffected.")
    ap.add_argument("--gate-vert-commit-size", type=float, default=0.0, metavar="F",
                    help="--coast-tube VERTICAL CLOSE-RANGE COMMIT: filtered gate "
                         "size_frac at or above which the vertical PD is FROZEN -- the "
                         "trim it had earned is held and eased to the leg baseline (0.0) "
                         "over --gate-vert-commit-tau, instead of being recomputed from "
                         "v_f/dv. Default 0.0 = OFF (opt-in per run). "
                         "WHY: past close range the gate fills the frame, v_f gets noisy "
                         "and the kd_v*dv term differentiates that noise, which can spike "
                         "the trim into the down-clamp in the last metres and fly the "
                         "drone into the gate plane -- the vertical analogue of the "
                         "lateral parallax sweep. At that range the altitude is already "
                         "made or missed; there is nothing useful left to servo on. "
                         "NOT a latch: below the size the live PD resumes on the next "
                         "tick, from wherever the trim eased to, so re-engagement is "
                         "continuous. SCOPE: see --gate-vert-commit-to-ag, which "
                         "defaults to GATE 1 ONLY (ag==0). It is required there and "
                         "fatal after it -- flt41 clipped gate 1 without it, flt40 "
                         "plunged at gate 2 with it applied everywhere.")
    ap.add_argument("--gate-vert-commit-to-ag", type=int, default=0, metavar="N",
                    help="--coast-tube: apply the VERTICAL close-range commit only up to "
                         "and including this leg (active_gate). Default 0 = GATE 1 ONLY. "
                         "An UPPER bound, unlike the lateral --gate-commit-from-ag, "
                         "because the vertical commit is needed at the FIRST gate and "
                         "harmful after it -- proven both ways by our own runs: "
                         "flt41 with the commit OFF clipped gate 1 (the kd_v term read "
                         "the gate's geometric fall in frame as sink, drove the trim to "
                         "the +0.06 clamp and thrust to 0.340, and the drone climbed "
                         "into the top bar -- collision at t=4.6 s); flt40 with the "
                         "commit ON everywhere plunged at gate 2 (the PD was frozen "
                         "through the crossing, trim ~0 at thrust 0.249, while v ran "
                         "+0.17 -> -0.90 and the arrest never fired). "
                         "Gate 1 is a near-level approach where the close-range D term "
                         "is pure noise; the descending legs arrive with real sink that "
                         "something has to stop. Raise this only with a flight that says "
                         "so.")
    ap.add_argument("--gate-vert-commit-tau", type=float, default=0.20, metavar="S",
                    help="--coast-tube: time constant of the vertical commit's slew to "
                         "the leg baseline (default 0.20 s). 0 = drop the trim to the "
                         "baseline instantly (a STEP -- the thing the lag exists to "
                         "avoid). Only has effect when --gate-vert-commit-size > 0.")
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
                    help="--coast-tube: run the TUBE detector on EVERY leg and log its "
                         "columns (tube_solid/u_tube_f/curv_f/rail_k/tube_found/u_tube/"
                         "curvature/area_frac). Note --tube-lateral already runs the "
                         "detector on the legs the rail steers; this is the override that "
                         "also measures it on legs 0-1, where it buys instrumentation at "
                         "the cost of their loop rate. "
                         "OFF BY DEFAULT, but MUCH cheaper than it used to be: the old "
                         "fill-and-area detector ran a full-frame HSV conversion at 13.47 "
                         "ms/call against the gate detector's 5.45 (71%% of the vision "
                         "budget), which is what drove the 16.2 Hz loop FLOOR on the "
                         "gate-2 leg and made gate 2 inconsistently pass-or-collide. The "
                         "rebuilt RAIL detector uses an integer chroma test and two curve "
                         "fits -- no HSV -- and measures 8.0 ms at full resolution and "
                         "3.4 ms at half, i.e. CHEAPER than the gate detection the loop "
                         "already pays for every frame. The old cost argument no longer "
                         "applies; the default stays off only because nothing reads the "
                         "signal yet. Changes NO control law either way.")
    ap.add_argument("--vert-auth-down", default=None, metavar="A[,A...]",
                    help="--coast-tube PER-LEG gate-vert DOWN (descend) authority: a "
                         "comma-separated table indexed by ACTIVE_GATE, e.g. "
                         "\"0.045,0.08\". Default None = the single "
                         "--gate-vert-down-auth on every leg (behaviour unchanged). "
                         "A short list is EXTENDED BY ITS LAST ENTRY, and an "
                         "active_gate past the end clamps to the last entry -- so "
                         "\"0.045,0.08\" reads as 'gentle on the first approach, "
                         "aggressive on every descending leg after it' on a course of "
                         "ANY gate count. "
                         "This is a table and not a gate-1/gate-2 fence on purpose: "
                         "which leg is near-level and which is steep is a property of "
                         "the COURSE, so it belongs in data, not in the control law.")
    ap.add_argument("--vert-auth-up", default=None, metavar="A[,A...]",
                    help="--coast-tube PER-LEG gate-vert UP (climb/arrest) authority, "
                         "same table form as --vert-auth-down, e.g. \"0.05,0.10\". "
                         "Default None = --gate-vert-up-auth everywhere. "
                         "This is the ARREST that stops a sink before the gate plane, "
                         "and it is exactly what cannot be raised globally: MEASURED on "
                         "filt31 the first approach already reaches +0.048 of trim "
                         "against a 0.06 clamp while the gate is centred (|v_f| 0.024), "
                         "so a larger global clamp only buys over-climb into that "
                         "gate's top bar. cfg.ol_thrust_hi is the real ceiling on what "
                         "any climb trim can command.")
    ap.add_argument("--gate-vert-floor", action="store_true",
                    help="--coast-tube GATE-CENTRE ALTITUDE FLOOR: once the drone has "
                         "descended to a gate's vertical centre, FORBID any further sink "
                         "until it is through. OFF BY DEFAULT (opt-in per run). "
                         "THE DEFECT IT FIXES: on the approach the filtered v_f runs "
                         "+0.6 -> 0 -> -0.7 -- the drone descends onto the gate line and "
                         "straight past it, and the climb reaction only fires once v_f "
                         "is already negative, by which point there is sink velocity "
                         "built up that the trim cannot arrest in the metre remaining. "
                         "That is the consistent under-gate-2 (and gate-3) hit. "
                         "While a gate is live and v_f <= --gate-vert-floor-vthresh, the "
                         "commanded thrust is held at or above --gate-vert-floor-thrust. "
                         "It is a MAX, not an assignment: the drone may hold or CLIMB, "
                         "it may only never descend further. Applies to ALL gates and "
                         "releases by itself as each gate is passed and the detection "
                         "goes away. Logs a [FLOOR] tag while active.")
    ap.add_argument("--gate-vert-floor-vthresh", type=float, default=0.10, metavar="V",
                    help="--coast-tube: filtered v_f at or below which the floor engages "
                         "(default 0.10). v_f > 0 means the gate sits BELOW frame centre, "
                         "i.e. the drone is still above it, so a small POSITIVE threshold "
                         "engages a hair BEFORE dead centre. That is deliberate and is "
                         "what makes this a FLARE instead of a catch: the sink is stopped "
                         "while the drone is still above the bar rather than after it has "
                         "crossed. 0.0 would engage exactly at centre, which is already "
                         "too late to arrest built-up sink.")
    ap.add_argument("--gate-vert-floor-thrust", type=float, default=0.250, metavar="X",
                    help="--coast-tube: the thrust the floor holds (default 0.250, the "
                         "plant's approximate hover). Clamped to the open-loop thrust "
                         "bounds. NOTE this is well ABOVE the steep legs' ladder "
                         "baseline, which is exactly why the floor is also re-applied "
                         "PAST the thrust slew limiter: on segs 1-3 the baseline sits at "
                         "ol_thrust_lo (0.180 -- the ladder's descent is swallowed by "
                         "that clamp there), so through the 0.15/s up-slew alone the "
                         "floor takes 0.47 s to arrive, which at 8 m/s is ~3.7 m of "
                         "travel. The gate-2 approach does not have 3.7 m left once v_f "
                         "has fallen to the engage threshold.")
    ap.add_argument("--rail-lateral-primary", action="store_true",
                    help="--coast-tube RAIL LATERAL PRIMARY: the RAIL is the steering "
                         "reference and gate-centring is the FALLBACK -- the reverse of "
                         "the default law. While the rail is locked the roll command "
                         "drives u_tube -> 0 (hold the path centred); when the rail is "
                         "lost the gate law fades back in over --tube-auth-tau. OFF BY "
                         "DEFAULT (opt-in per run). Engages from "
                         "--rail-lateral-from-ag (default 0 = from spawn). "
                         "Built FRESH on the rebuilt u_tube -- it does NOT inherit "
                         "--tube-lateral's constants, which were fitted to the old "
                         "fill-and-area signal. Pure band-centring, NO curvature lead: "
                         "that term is what spiked the rail to its clamp and flew flt12 "
                         "into gate 1, and holding the path centred does not need it. "
                         "*** READ THIS BEFORE FLYING: giving the rail authority from "
                         "ag==0 is what flt10 and flt12 did, and both broke gate 1. The "
                         "argument for retrying is that they steered on a detector that "
                         "reported area instead of rails; that argument is reasonable "
                         "but UNPROVEN in flight. The mitigations are the gentle gain, "
                         "the clamp, and a continuous fade back to the gate law -- not a "
                         "guarantee. Fly it with --dump-frames-leg1 and check gate 1 "
                         "still passes before trusting it downstream. ***")
    ap.add_argument("--rail-lateral-from-ag", type=int, default=0, metavar="N",
                    help="--coast-tube: leg (active_gate) from which the rail may steer "
                         "under --rail-lateral-primary. Default 0 = from spawn, which is "
                         "the point of the flag: stay on the path from the start. Set 1 "
                         "or 2 to keep the proven gate-1 (and gate-2) approach on the "
                         "gate law while still using the rail downstream.")
    ap.add_argument("--rail-lat-k", type=float, default=0.5, metavar="K",
                    help="--coast-tube: bank ANGLE (rad) per unit u_tube under "
                         "--rail-lateral-primary. Default 0.5, deliberately gentle. "
                         "MEASURED: u_tube reads +-0.005 when the drone is centred on "
                         "the path and reaches ~0.38 at a 120 px lateral displacement of "
                         "the frame, so 0.5 puts a large-but-real offset (0.384) exactly "
                         "at the 11 deg clamp -- the rail stays PROPORTIONAL across its "
                         "whole working range instead of steering on a stop.")
    ap.add_argument("--rail-lat-max-deg", type=float, default=11.0, metavar="DEG",
                    help="--coast-tube: HARD clamp on the rail primary bank command "
                         "(default 11).")
    ap.add_argument("--rail-lat-max-rms", type=float, default=2.0, metavar="PX",
                    help="--coast-tube STEERING-QUALITY GATE: the rail may steer only "
                         "while its fit residual is at or below this (default 2.0 px). "
                         "tube.found means 'both rails were fitted', NOT 'the fit is "
                         "good enough to bank on'. MEASURED on filt29's g1->g2 leg, the "
                         "2.7%% of ticks that did lock had rms p50 2.76 px against 0.93 "
                         "on the gate-1 leg -- a poorly-conditioned fit of a distant "
                         "sliver. Banking on that is the 'steer to a bad signal' case, "
                         "which is worse than gate centring, so those ticks are refused "
                         "and the gate law keeps the bank.")
    ap.add_argument("--rail-lat-min-sep", type=float, default=0.15, metavar="F",
                    help="--coast-tube STEERING-QUALITY GATE: minimum rail separation "
                         "(frac of frame width) at the look-ahead row before the rail "
                         "may steer (default 0.15). Small separation means the two rails "
                         "are nearly coincident -- a DISTANT path seen as a thin wedge, "
                         "where the centre has a tiny baseline and large lever arm. "
                         "MEASURED on filt29's g1->g2 leg: separation p50 0.090 on the "
                         "locked ticks, against 0.483 on the gate-1 leg where the path "
                         "fills the lower frame. 0 disables this gate.")
    ap.add_argument("--rail-merge-recover", action="store_true",
                    help="--coast-tube: recover rows where the two rails have MERGED "
                         "into one narrow run because the path is distant. Those rows "
                         "are otherwise discarded as 'one rail, unusable alone' -- which "
                         "on filt29's g1->g2 leg threw away 86.2%% of ticks while the "
                         "path was plainly in frame (area_frac 0.006, rail_rows 0). "
                         "DEFAULT OFF and NOT YET VALIDATED: a merged run yields a "
                         "BEARING to a distant path, not a rail separation, so it still "
                         "will not satisfy the separation test and will not by itself "
                         "produce a steering lock. It is a DIAGNOSTIC for now -- turn it "
                         "on with --log-tube to see how much of a leg is merged-distant "
                         "rather than genuinely unseen.")
    ap.add_argument("--rail-band-top", type=float, default=None, metavar="F",
                    help="--coast-tube: TOP of the rail search band, frac of H (config "
                         "default 0.30). Lower it to look further up toward the horizon.")
    ap.add_argument("--rail-band-bottom", type=float, default=None, metavar="F",
                    help="--coast-tube: BOTTOM of the rail search band, frac of H. "
                         "NOTE: the default is ALREADY 1.00, the last row of the frame, "
                         "so there is nothing below it to extend into -- the band is not "
                         "what runs out when the path sinks as the nose rises. MEASURED "
                         "on vision_frame.png rolled down 150 px: rows 50 and rms 0.40, "
                         "both healthy, but NO LOCK, because the look-ahead row had "
                         "ended up ABOVE the convergence where the rails have crossed "
                         "and the separation goes negative. "
                         "--rail-lookahead-below-conv is the knob that actually fixes "
                         "that, and it is on by default.")
    ap.add_argument("--rail-lookahead", type=float, default=None, metavar="F",
                    help="--coast-tube: row where the lateral offset is measured, frac "
                         "of H (config default 0.75).")
    ap.add_argument("--rail-lookahead-below-conv", type=float, default=None, metavar="F",
                    help="--coast-tube: keep the look-ahead row at least this far (frac "
                         "of H) BELOW the convergence row, default 0.12. THIS IS WHAT "
                         "KEEPS THE PATH LOCKED AS IT SINKS IN FRAME. The rails diverge "
                         "downward from the vanishing point, so above it they have "
                         "crossed and the separation test fails on a perfectly good fit. "
                         "As the nose relaxes the path sinks and the convergence slides "
                         "down past a fixed 0.75H -- exactly that failure. Adapting the "
                         "look-ahead extends the lock from 120 px of sink to 180 px "
                         "(measured), and leaves the nominal frame bit-identical because "
                         "the adaptation only binds once the convergence is low. 0 pins "
                         "the look-ahead at --rail-lookahead.")
    ap.add_argument("--rail-vert", action="store_true",
                    help="--coast-tube RAIL VERTICAL: drive the thrust trim off the "
                         "RAIL's vanishing point (v_converge) so the drone tracks the "
                         "COURSE's descent directly, instead of only reacting to a red "
                         "gate that is a usable reference for the last metre. OFF BY "
                         "DEFAULT (opt-in per run). "
                         "The path's convergence row reads the slope of the path ahead "
                         "against a FIXED camera pitch: convergence LOW = the path dives "
                         "away below = descend; HIGH = the path runs up out of frame = "
                         "climb. This is also what keeps the rail IN FRAME, which is why "
                         "the lock collapses on leg 1 today -- nothing steers vertically "
                         "toward it. "
                         "Engages only from --rail-vert-from-ag (default 1), so the "
                         "gate-1 vertical that works today is untouched. On loss the "
                         "trim HOLDS then decays to the altitude ladder's slope-sized "
                         "feed-forward -- never flat, never blind.")
    ap.add_argument("--rail-vert-from-ag", type=int, default=0, metavar="N",
                    help="--coast-tube: engage the rail vertical from this leg "
                         "(active_gate) onward. Default 0 = FROM SPAWN, so the path is "
                         "held vertically on the gate-1 approach too, not just after it "
                         "-- which is also what keeps the rail in frame early enough to "
                         "be useful. Set 1 to fence it off the gate-1 approach and leave "
                         "that vertical law bit-for-bit as it flies today. "
                         "Only has effect with --rail-vert, which is itself off by "
                         "default; below the fence the rail's vertical authority is "
                         "EXACTLY 0.0.")
    ap.add_argument("--rail-vert-target", type=float, default=-0.35, metavar="V",
                    help="--coast-tube: the v_converge the rail vertical holds. "
                         "DEFAULT IS NOT 0 AND MUST NOT BE: MEASURED on vision_frame.png, "
                         "when the gate is vertically CENTRED (v_err -0.03, the exact "
                         "condition the working gate-vert loop drives to) the rail "
                         "converges at v=-0.346. Targeting 0 would command a standing "
                         "climb of k*0.346 = 0.055 at the default gain -- nearly DOUBLE "
                         "the up-authority clamp, so it would pin the clamp and fly the "
                         "drone off the top of the course. -0.35 is that measured "
                         "equilibrium. "
                         "CAVEAT WORTH A FLIGHT: it was measured on the START->g1 leg "
                         "(12.1 deg). A steeper leg puts the vanishing point LOWER in "
                         "frame, so the true equilibrium on g1->g2 (17.1 deg) is somewhat "
                         "higher than -0.35. Fly once with --log-tube, read the leg's "
                         "median v_converge out of archive/notes/design-specs/protocol.py, and set this to "
                         "it. See --rail-vert-target-per-deg for deriving the rest.")
    ap.add_argument("--rail-vert-target-per-deg", type=float, default=0.0, metavar="V",
                    help="--coast-tube: v_converge target shift per degree of leg slope "
                         "away from --descent-anchor-leg, so each leg gets its own target "
                         "the way the altitude ladder gets its own descent. Default 0.0 "
                         "= ONE fixed target on every leg, because this scale has not "
                         "been measured -- it needs the camera's vertical FOV or a flight "
                         "logging v_converge across two legs of known slope. Deriving a "
                         "table before then would be invented precision.")
    ap.add_argument("--rail-vert-k", type=float, default=0.10, metavar="K",
                    help="--coast-tube: thrust trim per unit v_converge error (default "
                         "0.10, DELIBERATELY BELOW the gate loop's --k-thrust-v 0.16). "
                         "The rail vertical is a TRIM on a baseline that already carries "
                         "the leg's slope-sized ladder descent, so it only has to correct "
                         "the residual; and the target itself is a one-frame measurement, "
                         "so a soft gain bounds what a wrong target can do.")
    ap.add_argument("--rail-vert-up-auth", type=float, default=0.020, metavar="A",
                    help="--coast-tube: max UPWARD (climb) rail thrust trim, default "
                         "0.020 -- tighter than the gate loop's 0.030. Climb authority is "
                         "the dangerous direction here: an over-high target commands a "
                         "standing climb, and this is the bound on how far that can go.")
    ap.add_argument("--rail-vert-down-auth", type=float, default=0.040, metavar="A",
                    help="--coast-tube: max DOWNWARD (descend) rail thrust trim, default "
                         "0.040 (gate loop uses 0.060). Descent is the safer direction --"
                         " the ladder is already descending -- so it gets more room.")
    ap.add_argument("--rail-vert-tau", type=float, default=0.25, metavar="S",
                    help="--coast-tube: LPF time constant on v_converge (default 0.25 s). "
                         "The vanishing point is an EXTRAPOLATION of two curve fits, so "
                         "it is noisier than the fits themselves and wants smoothing "
                         "before it moves the thrust.")
    ap.add_argument("--rail-vert-hold-s", type=float, default=0.4, metavar="S",
                    help="--coast-tube: after the rail lock drops, HOLD the last "
                         "rail-derived trim this long before decaying (default 0.4 s). "
                         "This is what carries the drone through the gate, where the "
                         "rails are occluded by the gate frame itself.")
    ap.add_argument("--rail-vert-decay-s", type=float, default=0.6, metavar="S",
                    help="--coast-tube: time constant of the decay from the held rail "
                         "trim to 0.0 (default 0.6 s). 0.0 is not level -- it is this "
                         "leg's ALTITUDE LADDER baseline, which is already sized to the "
                         "course slope. That is what 'never flat, never blind' means. "
                         "Once fully decayed the rail releases the vertical and the "
                         "gate-vertical loop resumes.")
    ap.add_argument("--rail-auth-tau", type=float, default=0.3, metavar="S",
                    help="--coast-tube: time constant over which the rail vertical takes "
                         "the thrust trim over from the gate-vertical loop and hands it "
                         "back (default 0.3 s). Same reason as --tube-auth-tau: the "
                         "lock is a hard boolean, so an unlagged handoff steps the trim.")
    ap.add_argument("--rail-full-res", action="store_true",
                    help="--coast-tube: run the rail detector on the FULL-resolution "
                         "frame. Default is HALF resolution, which reuses the array the "
                         "gate detector already builds (so the downsample is free), costs "
                         "3.4 ms instead of 8.0, and is measured to agree with full res "
                         "to 0.0002 in offset and 0.01 in v_converge "
                         "(tests/test_rail_detector.py). Half res matters because "
                         "--rail-vert runs the detector every frame on the g1->g2 leg -- "
                         "which IS the gate-2 approach, the leg whose loop-rate variance "
                         "already makes gate 2 inconsistently pass-or-collide.")
    ap.add_argument("--dump-frames-leg1", action="store_true",
                    help="--coast-tube DIAGNOSTIC: during a live race, save the raw FPV "
                         "frame AND the rail-detector overlay (fitted curves + lock "
                         "status, drawn exactly as tools/tube_rail_check.py draws them) "
                         "every --dump-frames-every seconds while active_gate==1 (the "
                         "g1->g2 leg), plus the last --dump-frames-pre seconds of the "
                         "gate-1 approach so the transition THROUGH gate 1 is captured. "
                         "Answers 'why does the rail lock collapse on that leg?' with "
                         "pictures instead of inference. "
                         "LOG-ONLY: changes no control law, and the detector runs on the "
                         "WRITER THREAD, so the loop never even pays for the measurement. "
                         "Writing is off-thread because a PNG costs ~29.5 ms to encode "
                         "and each sample is two of them -- inline that would stall the "
                         "loop for longer than two ticks on the gate-2 approach, the leg "
                         "whose loop-rate variance already makes gate 2 inconsistent. "
                         "The approach frames are written RETROACTIVELY, when ag reaches "
                         "1, so they are the real last second before the gate rather "
                         "than a guess from a rising size_frac. "
                         "Costs disk and some CPU on a background thread; expect ~4-8 "
                         "files/second while it is active. Output: --dump-frames-dir, "
                         "including a manifest.csv of every sample's lock/offset/fit.")
    ap.add_argument("--dump-frames-dir", default="leg1_frames", metavar="DIR",
                    help="--coast-tube: where --dump-frames-leg1 writes (default "
                         "leg1_frames/). Created if missing. Filenames carry flight "
                         "time, LOCK/NOLOCK, offset, row count and fit residual, so the "
                         "folder can be scanned without opening anything.")
    ap.add_argument("--dump-frames-every", type=float, default=0.25, metavar="S",
                    help="--coast-tube: seconds between dumped samples (default 0.25). "
                         "Lower means more frames and more background CPU/disk.")
    ap.add_argument("--dump-frames-pre", type=float, default=1.0, metavar="S",
                    help="--coast-tube: seconds of the gate-1 approach (ag==0) held in "
                         "the ring buffer and written once gate 1 is passed (default "
                         "1.0). This is what captures the transition through the gate.")
    ap.add_argument("--dump-frames-queue", type=int, default=64, metavar="N",
                    help="--coast-tube: max samples queued for the writer (default 64). "
                         "When full, new samples are DROPPED and counted rather than "
                         "blocking the control loop -- a missing frame is a nuisance, a "
                         "stalled loop is a crash.")
    ap.add_argument("--tube-steer-area-min", type=float, default=0.015, metavar="F",
                    help="DEPRECATED AND UNUSED -- kept so existing command lines still "
                         "parse. The tube_solid column is now simply the detector's LOCK "
                         "(both rails fitted), because area is the wrong question about a "
                         "thin bright curve: a fully visible rail one pixel wide has "
                         "essentially zero area, so an area threshold reported 'not seen' "
                         "on frames where the path is plainly visible. That FALSE "
                         "NEGATIVE is the sub-4%% lock rate this rebuild fixes. Fit "
                         "quality, rail separation and row count carry the validity now "
                         "(see ServoConfig.rail_*); area_frac is still logged, and still "
                         "gates nothing.")
    ap.add_argument("--lateral-gate-centring", action="store_true",
                    help="NO-OP, kept so existing command lines still parse. Gate "
                         "centring is the lateral law unless --tube-lateral is passed.")
    ap.add_argument("--tube-lateral", action="store_true",
                    help="--coast-tube RAIL LATERAL: steer on the TUBE (u_tube + a "
                         "clamped curvature lead) instead of gate-centring, on the legs "
                         "selected by --tube-lateral-from-ag. "
                         "*** CURRENTLY DISARMED -- passing this EXITS with an error. *** "
                         "The tube detector was rebuilt to fit the two cyan RAIL LINES "
                         "rather than mask a fill, so u_tube now means a different "
                         "quantity and area_frac is no longer a validity test. Every "
                         "constant that tuned this law (--k-tube-bank 0.6 from the old "
                         "|u_tube|<=0.222 range, the area>=0.015 solidity gate) was "
                         "fitted to the OLD signal and does not carry over. Validate with "
                         "tools/tube_rail_check.py on real gate-3-leg frames and "
                         "re-derive the gains before re-arming. The law itself is intact "
                         "and tested (tests/test_rail_lateral.py). "
                         "OFF BY DEFAULT (opt-in per run). Implies the tube detector on "
                         "those legs. "
                         "WHY: a gate blob LUNGES sideways from parallax as you close on "
                         "it. MEASURED on filt9's gate-3 leg, gate u_err ran +0.20 -> "
                         "+0.42 -> +0.95 over 0.6 s while the tube moved only +0.17 -> "
                         "+0.23 -> +0.37; des_roll sat pinned to the -11 deg clamp for 30 "
                         "of 46 close-range ticks, steering off a decoy. The gate is the "
                         "decoy, the tube is the rail. "
                         "AUTHORITY IS THE RAIL's OWN DECAY WEIGHT, not a switch: the "
                         "rail owns the bank while it is solid or holding, and as it "
                         "fades the previous gate law (with its close-range commit) fades "
                         "back in over the same tau. A leg where the tube is never seen "
                         "therefore flies EXACTLY today's law rather than flying blind -- "
                         "which matters because the tube's visibility on the gate-3 leg "
                         "is precisely what the altitude ladder is supposed to fix and "
                         "has not yet been confirmed in flight.")
    ap.add_argument("--tube-lateral-from-ag", type=int, default=2, metavar="N",
                    help="--coast-tube: give the rail steering authority ONLY from this "
                         "leg index (active_gate) onward. Default 2 = the gate-3 leg, the "
                         "first leg downstream of the two gates that pass today. "
                         "THE SCOPE IS LOAD-BEARING -- this is the exact fence whose "
                         "absence sank both previous attempts: flt10 gave the rail "
                         "authority on EVERY leg and broke gate 1 (the tube was solid on "
                         "1.7%% of ticks there -- nothing to steer on), and flt12 crashed "
                         "into gate 1 the same way. Out of scope the rail's authority is "
                         "forced to EXACTLY 0.0, so the command on legs 0-1 is "
                         "bit-for-bit the law that passes gates 1 and 2 today. 0 would "
                         "arm it everywhere -- that is what crashed; do not, without a "
                         "flight that says so.")
    ap.add_argument("--tube-auth-tau", type=float, default=0.3, metavar="S",
                    help="--coast-tube: time constant over which the RAIL takes the bank "
                         "over from the gate law, and gives it back (default 0.3 s). "
                         "WHY IT EXISTS: the rail's own weight is 1.0 the instant the "
                         "tube is first seen -- it only shapes the fade on LOSS -- so "
                         "handing over on that alone STEPS from one law to the other in "
                         "one tick. Replayed on filt9's gate-3 leg that step is 8.9 deg: "
                         "the exact slam this law exists to remove, arriving at the "
                         "handoff instead of at the gate. 0 = instant (that step).")
    ap.add_argument("--k-tube-bank", type=float, default=0.6, metavar="K",
                    help="--coast-tube: desired bank ANGLE (rad) per unit filtered "
                         "u_tube, before the --tube-max-bank-deg clamp. Default 0.6, and "
                         "deliberately NOT --k-gate-bank's 1.0. THIS IS WHAT MAKES THE "
                         "RAIL COMMAND BOUNDED, and the boundedness is earned by the "
                         "gearing, not by a generous clamp: MEASURED on filt9's gate-3 "
                         "leg |u_tube| peaks at 0.222 on solid readings, which at k=1.0 "
                         "asks for 12.7 deg of bank -- MORE than the gate law's own 10 "
                         "deg clamp, so the rail would sit pinned on 24%% of the "
                         "close-range ticks, which is the same blind-steering failure "
                         "with a different sensor. At 0.6 that full-scale deflection maps "
                         "to 7.6 deg, so the rail stays PROPORTIONAL across its entire "
                         "observed range and never reaches its clamp on that leg (0 "
                         "pinned ticks, 8.7 deg peak including the lead and fine-trim).")
    ap.add_argument("--tube-lead-max-deg", type=float, default=3.0, metavar="DEG",
                    help="--coast-tube: SEPARATE clamp on the curvature-lead part of the "
                         "rail command (default 4). THIS IS THE flt12 FIX. The lead is "
                         "(upper band centre - lower band centre), which goes large AND "
                         "noisy exactly when the near tube fills the lower band at close "
                         "range -- i.e. when a bank command does the most damage. flt12 "
                         "let the lead spike the rail to the -11 deg clamp at auth 0.95 "
                         "and flew into gate 1. Clamping the lead on its own keeps it an "
                         "anticipatory nudge that can never dominate the u_tube term it "
                         "is supposed to lead. 0 disables the lead entirely (pure "
                         "band-centring).")
    ap.add_argument("--gate-fine-max-deg", type=float, default=2.5, metavar="DEG",
                    help="--coast-tube: clamp on the gate FINE-TRIM that rides on top of "
                         "the rail command (default 2.5 deg, a QUARTER of the rail's 10). "
                         "While the rail is steering the gate is demoted to a nudge -- it "
                         "may refine the rail's centring, never drive the bank. The trim "
                         "is also LAGGED in over --tube-auth-tau rather than switched: "
                         "its admission window is a hard test on sz_f and |u_f|, so an "
                         "unlagged trim would step by the full clamp on the tick the "
                         "window opens (3.4 deg on filt9's gate-3 leg) -- a slam "
                         "introduced by the anti-decoy guard itself.")
    ap.add_argument("--gate-fine-size-min", type=float, default=0.15, metavar="F",
                    help="--coast-tube: filtered gate size_frac BELOW which the gate "
                         "fine-trim is not applied at all (default 0.15). A small/distant "
                         "gate is not a lateral reference worth trimming on.")
    ap.add_argument("--gate-fine-uerr-max", type=float, default=0.15, metavar="F",
                    help="--coast-tube: |filtered gate u_err| ABOVE which the fine-trim "
                         "retires (default 0.15). THE ANTI-DECOY RULE: the gate may trim "
                         "only while LARGE and NEAR-CENTRED. A large gate that is far "
                         "off-centre is not an error signal -- it is the parallax lunge, "
                         "and following it is the flt9 defect. On filt9's gate-3 leg "
                         "every admitted tick precedes every rejected one: the trim "
                         "retires as the sweep begins and stays retired.")
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
    ap.add_argument("--gate-commit-align", type=float, default=None, metavar="F",
                    help="--coast-tube: |filtered u_err| below which the drone counts as "
                         "ALIGNED, gating the close-range lateral commit. Default None = "
                         "OFF, i.e. the legacy size-only trigger, bit-for-bit. "
                         "WHY: size alone is the wrong trigger. MEASURED on filt81, the "
                         "commit fired MID-TURN and decayed commit_k to ~0 while the "
                         "drone was still off-centre -- as it drifted left, vision "
                         "saturated at gate_cmd -9 deg but des_roll stayed ~0, so it "
                         "coasted off-centre deaf to its own correction. With this set, "
                         "the commit locks only when CLOSE **and** CENTRED. "
                         "LATCHED: once locked it stays locked for the rest of the leg "
                         "even if |u_f| spikes -- that spike is the terminal parallax as "
                         "the gate fills, and re-opening on it would hand the reaction "
                         "back to vision and restore the end-of-leg swing this removes. "
                         "Resets only on a gate change. Scope is unchanged "
                         "(--gate-commit-from-ag).")
    ap.add_argument("--gate-commit-stable-frames", type=int, default=1, metavar="N",
                    help="--coast-tube: consecutive frames the drone must be centred "
                         "(|u_f| <= --gate-commit-align) before the close-range commit "
                         "latches. Default 1 = latch on the first centred frame, "
                         "bit-for-bit the current behaviour. "
                         "WHY RAISE IT: a single centred frame is not alignment. MEASURED "
                         "on filt85 the latch fired on a ONE-FRAME crossing at u_f +0.06, "
                         "and parallax amplified that residual to +0.76 by the gate "
                         "plane. A streak locks only while STABLY centred -- during the "
                         "pre-parallax window -- so neither a lone centred frame nor a "
                         "blip can trigger it. "
                         "The streak counter resets on any off-centre frame, but only "
                         "BEFORE the latch: a spike can delay or prevent a premature "
                         "lock, never un-lock a good one. Only has effect with "
                         "--gate-commit-align set.")
    ap.add_argument("--gate-commit-rate-max", type=float, default=0.0, metavar="R",
                    help="--coast-tube close-range commit (ag >= --gate-commit-from-ag): "
                         "require |du_f/dt| < R (filtered u_f, real dt) IN ADDITION to the "
                         "align band, so the latch cannot fire on a transient zero-"
                         "crossing. 0.0 (default) = OFF, the frame-based rule is unchanged "
                         "and bit-for-bit. "
                         "WHY: MEASURED on filt93B the latch fired at sz~0.16 while u_f "
                         "was CROSSING zero at du_f/dt +0.13 and still accelerating; "
                         "commit_k decayed to 0, swallowed the vision's full -9 deg "
                         "correction, and the drone missed left. A frame streak cannot "
                         "tell a settled signal from a fast crossing -- its rate can. "
                         "Replay-derived value: 0.11 (rejects filt93B +0.13 and filt89.1 "
                         "+0.19, keeps filt87.1 +0.096 and filt80.5 -0.012). "
                         "Setting this (or --gate-commit-stable-s) switches the latch to "
                         "the seconds-based dwell and retires --gate-commit-stable-frames.")
    ap.add_argument("--gate-commit-stable-s", type=float, default=0.0, metavar="S",
                    help="--coast-tube close-range commit: require align+rate continuously "
                         "for S SECONDS (real dt) before latching, instead of "
                         "--gate-commit-stable-frames. 0.0 (default) = OFF, use the frame "
                         "count. "
                         "WHY SECONDS: a frame count means different things at different "
                         "loop rates -- 3 frames is 0.15 s at 20 Hz but 0.05 s at 60 Hz, "
                         "so the same rule qualified differently run to run. "
                         "Replay-derived value: 0.15. The genuine settled window before "
                         "the gate plane is only ~0.15 s; do NOT exceed it or the golden "
                         "latches are lost.")
    ap.add_argument("--leg2-entry-arrest", type=float, default=0.0, metavar="A",
                    help="--coast-tube LEG-2 ENTRY ARREST: thrust ADDED the moment "
                         "active_gate advances to 2, decaying to 0 over "
                         "--leg2-entry-arrest-s. Default 0.0 = OFF. "
                         "WHY: straight after gate 2 the drone free-sinks at ~0.23 "
                         "thrust for ~1.2 s before the vertical PD engages, and that "
                         "early dive is why it reaches gate 3 below the window (v_err "
                         "~ -0.85, consistently). There is no climb authority to recover "
                         "with, so the sink is PREVENTED instead of corrected. "
                         "SCOPE: the ag==2 entry only -- ag<2 and ag>2 are untouched, "
                         "and the latch clears if ag leaves 2. The boost is added inside "
                         "the normal thrust clamp and passes through the slew limiter, "
                         "so it can raise the command but never escape its bounds.")
    ap.add_argument("--leg2-entry-arrest-s", type=float, default=1.2, metavar="S",
                    help="--coast-tube: window over which the leg-2 entry arrest decays "
                         "LINEARLY from full to 0 (default 1.2 s, the measured duration "
                         "of the free-sink). Linear rather than exponential so it reaches "
                         "exactly 0 at the end instead of trailing a tail into the leg.")
    ap.add_argument("--gate-max-bank-ag2-left", type=float, default=None, metavar="DEG",
                    help="--coast-tube: ASYMMETRIC vision bank clamp on the ag==2 leg. "
                         "Caps the LEFT (positive) side of gate_cmd at this many degrees "
                         "while leaving full RIGHT authority at -gate-max-bank-ag2. "
                         "0 blocks all left bank on that leg. Default None = symmetric, "
                         "bit-for-bit unchanged. "
                         "WHY: gate 3 is always RIGHT of gate 2, so on that leg any LEFT "
                         "command is chasing parallax or noise and burns a short runway "
                         "-- filt63 banked +2.5 deg left out of gate 2 and again mid-leg "
                         "before reversing right. "
                         "SCOPE: ag==2 ONLY. ag<2, the close-range commit, and the "
                         "bank_bias backbone are all untouched.")
    ap.add_argument("--gate2-exit-level-size", type=float, default=0.0, metavar="F",
                    help="--coast-tube GATE-2 EXIT LEVEL: filtered gate size_frac at or "
                         "above which the ENTIRE ag==1 lateral command (backbone bank AND "
                         "gate trim) is slewed to WINGS-LEVEL, so the drone exits gate 2 "
                         "straight instead of banked. Default 0.0 = OFF. "
                         "WHY: the drone crosses gate 2 still banked ~5-8 deg left from "
                         "the --post-gate1-bank approach bank, carries that momentum onto "
                         "the gate-2->3 leg, and 11 deg of right vision authority cannot "
                         "reverse it before gate 3 leaves the FOV (u_err -> +0.99, crash "
                         "~t=14, identical on filt57/58/60). "
                         "KEEP THE THRESHOLD HIGH -- the approach bank must fly the WHOLE "
                         "approach and only retire in the last ~0.2 s at the gate plane. "
                         "flt13 levelled mid-approach and clipped gate 2; this is "
                         "deliberately later than that failure. "
                         "SCOPED TO ag==1 ONLY: ag 0 and ag>=2 force the weight to "
                         "EXACTLY 1.0 and are bit-for-bit unchanged, and the ag>=2 "
                         "close-range commit is untouched. Reversible, not a latch.")
    ap.add_argument("--gate2-exit-level-tau", type=float, default=0.20, metavar="S",
                    help="--coast-tube: time constant of the gate-2 exit-level slew to "
                         "wings-level, and of the ramp back in below the size (default "
                         "0.20 s). Same lag_step primitive as the close-range commit. "
                         "0 = instant (a STEP -- the thing the lag exists to avoid).")
    ap.add_argument("--gate-bank-size-min-ag2", type=float, default=None, metavar="F",
                    help="--coast-tube: gate size_frac required to steer ON THE ag==2 LEG "
                         "ONLY. Default None = use --gate-bank-size-min everywhere "
                         "(ag==2 then bit-for-bit unchanged). Gate 3 stays a small "
                         "distant centroid for most of that leg, so the global 0.10 keeps "
                         "vision silent and the drone flies it open loop; ~0.05 lets the "
                         "real drift be read. Safe only together with the size ramp -- "
                         "see --gate-bank-full-size-ag2.")
    ap.add_argument("--gate-bank-full-size-ag2", type=float, default=0.15, metavar="F",
                    help="--coast-tube: size_frac at which the ag==2 gate command reaches "
                         "FULL authority (default 0.15). Below it the command is scaled "
                         "linearly from 0 at --gate-bank-size-min-ag2, so a few-pixel "
                         "centroid nudges instead of slamming the bank. Logged as "
                         "ag2_auth. No effect on any other leg.")
    ap.add_argument("--gate-max-bank-deg", type=float, default=10.0, metavar="DEG",
                    help="--coast-tube: HARD clamp on the gate-centring bank. Default 10.")
    ap.add_argument("--gate3-vert-flare", action="store_true",
                    help="ag==2 ONLY: terminal anti-sink flare for Gate 3. Once the drone "
                         "is close (sz_f>=--gate3-vert-flare-size) and nearing gate centre "
                         "from above (0 <= v_f <= --gate3-vert-flare-vthresh) with a valid "
                         "Gate-3 detection, hold thrust at >= --gate3-vert-flare-thrust to "
                         "arrest the dive sink before the under-gate hit. Reversible max() "
                         "(like --gate-vert-floor), not a latch. Default off.")
    ap.add_argument("--gate3-vert-flare-thrust", type=float, default=0.285, metavar="X",
                    help="Thrust the Gate-3 flare holds (default 0.285; must stay < "
                         "ol_thrust_hi ceiling).")
    ap.add_argument("--gate3-vert-flare-size", type=float, default=0.30, metavar="F",
                    help="sz_f at/above which the Gate-3 flare may engage (default 0.30).")
    ap.add_argument("--gate3-vert-flare-vthresh", type=float, default=0.40, metavar="V",
                    help="filtered v_f at/below which the flare engages (default 0.40 -- "
                         "higher than the 0.10 floor because Gate 3 is lost early on the "
                         "steep dive). Only v_f>=0 (drone still above centre); a negative "
                         "v_f is already-below and not flared.")
    ap.add_argument("--gate3-vert-descent", action="store_true",
                    help="ag==2 ONLY: terminal descent nudge for Gate 3. When close "
                         "(sz_f>=--gate3-vert-descent-size) with a valid Gate-3 detection "
                         "and the drone verified HIGH (v_f>=--gate3-vert-descent-vhi), "
                         "subtract --gate3-vert-descent-delta from thrust to drop the high "
                         "crossing toward centre. Releases the instant "
                         "v_f<=--gate3-vert-descent-vrelease (near centre) so it can never "
                         "drive the drone low. Reversible, not a latch. Default off.")
    ap.add_argument("--gate3-vert-descent-delta", type=float, default=0.015, metavar="X",
                    help="Thrust subtracted while active (default 0.015; keep small -- "
                         "plant cannot recover after going low).")
    ap.add_argument("--gate3-vert-descent-size", type=float, default=0.25, metavar="F",
                    help="sz_f at/above which the descent nudge may engage (default 0.25).")
    ap.add_argument("--gate3-vert-descent-vhi", type=float, default=0.15, metavar="V",
                    help="v_f at/above which the drone is 'high' and the nudge arms "
                         "(default 0.15).")
    ap.add_argument("--gate3-vert-descent-vrelease", type=float, default=0.10, metavar="V",
                    help="v_f at/below which the nudge releases (near centre; default "
                         "0.10). Hysteresis with --gate3-vert-descent-vhi prevents chatter "
                         "and guarantees no low overshoot.")
    ap.add_argument("--gate3-vert-descent-hold-s", type=float, default=0.0, metavar="S",
                    help="--coast-tube, ag==2 ONLY: after a valid Gate-3 detection is LOST "
                         "while the descent was active and the drone was still high, keep "
                         "applying the SAME descent delta for up to S seconds (the blind "
                         "coast to the plane). 0.0 (default) = OFF, bit-for-bit unchanged. "
                         "WHY: MEASURED, the visible arm->loss window is only ~0.23-0.25 s "
                         "and the gate is lost at v_f +0.38..+0.45 still falling, with "
                         "another ~0.54-0.75 s of blind coast to the plane -- so the cut "
                         "releases while the drone is still high and half a second short. "
                         "Replay-recommended 0.30 (crossing +0.28 -> ~+0.20, no low-miss "
                         "exposure, expires before the earliest spurious reacquisition at "
                         "0.38 s). Never increases the delta; releases on ag advance, "
                         "timer expiry, any valid reacquisition, or reset.")
    ap.add_argument("--gate3-vert-descent-hold-release-vf", type=float, default=0.10,
                    metavar="V",
                    help="If a VALID Gate-3 frame reacquires during the blind hold with "
                         "v_f <= this (drone no longer high), release immediately rather "
                         "than over-descending. Default 0.10. NOTE: currently INERT -- any "
                         "valid frame already ends the hold unconditionally and hands back "
                         "to the live descent path, which owns the low case through its "
                         "own --gate3-vert-descent-vrelease hysteresis.")
    ap.add_argument("--gate2-hold-fixed-deg", type=float, default=-0.6, metavar="DEG",
                    help="LEG-2 ONLY: force the post-gate held bank to this fixed value "
                         "(deg; negative = right bank) instead of capturing it from "
                         "vision. Empirical Gate-3 centering optimum is -0.6 "
                         "(u_f@Gate3 = 0.176*hold + 0.10, r=0.87). Leaves leg-1 and all "
                         "other legs on the captured-latch path.")
    ap.add_argument("--post-gate-hold-s", type=float, default=1.5, metavar="S",
                    help="Max seconds after gate-1 pass to HOLD the gate-1-exit lateral "
                         "command instead of chasing the small distant gate-2 centroid. "
                         "0 = OFF (bit-identical to before).")
    ap.add_argument("--post-gate-hold-decay", type=float, default=0.5, metavar="S",
                    help="Exponential time-constant (s) of the held command's decay "
                         "during the hold.")
    ap.add_argument("--approach-bank", type=float, default=0.0, metavar="DEG",
                    help="Constant feed-forward bank on the gate-1 approach (ag==0), "
                         "+ = LEFT. Cancels the +2.21deg right-of-nose spawn bearing that "
                         "the reactive centring is too laggy to null before the gate.")
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
    return ap


def main():
    args = build_parser().parse_args()

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
