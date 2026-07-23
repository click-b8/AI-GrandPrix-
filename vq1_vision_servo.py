"""Hardcoded VQ1 vision-servo controller — NO RL, NO neural net.

Why this exists
---------------
The DCL deploy MAVLink stream on v3385 carries NO position, velocity, or
attitude quaternion (measured, read-only sniff — see obsidian/open-questions.md
"IMU + telemetry availability"). Only HIGHRES_IMU (accel+gyro), the FPV vision
stream, and ENCAPSULATED_DATA race status (which exposes `active_gate`).

That rules out a waypoint/position-PID controller: there is nothing to close a
position loop on. Blind dead-reckoning (double-integrate accel) or an open-loop
time/thrust profile both drift multiple metres over the ~161 m course, far wider
than the ~1 m gate pass-trigger. The only signal tied to real gate geometry in
real time is the camera: the gate is visible and roughly centred in the FPV
frame. So we servo on it.

Two-loop design (this is the crux)
----------------------------------
The plant is RATE-controlled: SET_ATTITUDE_TARGET type_mask=128 takes body
roll/pitch/yaw RATES (rad/s) + thrust. You cannot hold a steady forward lean by
commanding a constant rate. So:

  OUTER (guidance, from vision)  -> desired roll ANGLE (bank toward gate),
                                    desired pitch ANGLE (forward cruise lean),
                                    desired yaw RATE (turn toward gate),
                                    thrust (vertical framing).
  INNER (stabiliser, from IMU)   -> body RATES that drive the measured roll/pitch
                                    (from the gravity-down estimate) toward the
                                    desired angles, damped by the gyro.

The IMU inputs are the SAME ones run_vq1 already computes:
  telemetry["gravity_frd"] = gravity-down in FRD, |g|~=9.81 (A2 GravityEstimator)
  telemetry["velocity"]    = (p, q, r) body rates rad/s (raw gyro)

Angle extraction from gravity-down g=[gx,gy,gz] (FRD; x-fwd, y-right, z-down):
  roll  phi   = atan2(gy, gz)      (level -> 0)
  pitch theta = atan2(-gx, gz)     (level -> 0; nose-down -> negative; reproduces
                                    the measured -17.8 deg rest reading)

Everything tunable is in ServoConfig with a CALIBRATE banner. NOTHING here is
tuned against the real sim yet — the gate-colour thresholds and every gain must
be calibrated on the machine that runs the DCL sim, against real frames. The
hover-thrust fraction (ServoConfig.hover_cruise) is UNMEASURED; run
`run_vq1.py --hover-probe` first to bracket it.

The command contract matches the model path exactly:
  {"throttle": [0,1], "roll": [-1,1], "pitch": [-1,1], "yaw": [-1,1]}
roll/pitch/yaw are normalised body rates (multiplied by MAX_BODY_RATE=12 rad/s
downstream in dcl_mavlink_adapter.py).
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("vq1_vision_servo")

# Must match dcl_mavlink_adapter.MAX_BODY_RATE — normalised rate 1.0 == this rad/s.
MAX_BODY_RATE = 12.0

# Measured resting/spawn body pitch on v3385 (nose-down). cruise_pitch_deg must
# stay near this or the attitude loop holds a constant pitch correction and the
# drone climbs/backs up instead of flying the course (flight 5). Guard below.
SPAWN_PITCH_DEG = -17.8
CRUISE_PITCH_TOL_DEG = 5.0


# ===========================================================================
# ============================  CALIBRATE ME  ===============================
# Gate DETECTION thresholds are CALIBRATED (2026-07-22, vision_frame.png) -- see
# the gate-detector block. The GAINS and hover fraction below are still STARTING
# GUESSES and must be tuned on the sim box. Tune order:
#   1. [DONE] Gate detection calibrated from a real frame. Re-verify per sim
#      build with tools/vision_servo_dryrun.py --image <frame> --overlay out.png.
#   2. hover_cruise: run `run_vq1.py --hover-probe 0.15/0.20/0.25/...`; set
#      hover_cruise to the thrust where climb-rate ~= 0, then a hair BELOW it
#      (the course descends).
#   3. Attitude inner loop (kp_att/kd_att, cruise_pitch_deg): fly a single gate,
#      watch it hold a stable forward lean without oscillating. NOTE: this whole
#      inner loop assumes RATE control -- if tools/probe_rate_vs_angle.py finds
#      the sim is ANGLE mode, replace it with a direct-angle (quaternion) command.
#   4. Guidance gains (k_yaw/k_bank/k_thrust_v): widen until it centres gates
#      briskly without overshoot.
# ===========================================================================
@dataclass
class ServoConfig:
    # ---- image geometry ----
    img_w: int = 640
    img_h: int = 360

    # ---- gate detector (HSV thresholds; hue in degrees 0..360) ----
    # CALIBRATED against a real v3385 FPV frame (vision_frame.png, 2026-07-22):
    # gates are bright RED squares centred on hue ~0/360 (measured median ~359,
    # p10..p90 = -10..+6 wrap-adjusted), sat median 0.53, val median 0.95 (min
    # 0.67). This is a RED-WRAPAROUND band (lo>hi triggers the wrap branch in
    # _mask). The dim red horizon grid (val ~0.38) is rejected by gate_val_min;
    # the big cyan guidance tube (hue ~198) is far outside the band. The
    # brightness fallback is OFF: the tube's white core and grey wireframe would
    # otherwise be masked. Verified 0 tube / 0 wireframe contamination on-frame.
    gate_hue_lo: float = 340.0    # RED wraparound band: hue >= 340 OR hue <= 20
    gate_hue_hi: float = 20.0
    gate_sat_min: float = 0.30    # 0..1 (gate sat min ~0.26; 0.30 with margin)
    gate_val_min: float = 0.55    # 0..1 (rejects val~0.38 grid; gates are val>=0.67)
    gate_use_brightness_fallback: bool = False  # OFF: would grab the cyan tube / wireframe
    bright_val_min: float = 0.85
    bright_sat_max: float = 0.25
    min_area_frac: float = 0.0002  # ~46 px: keeps DISTANT gates (a far gate frame is
                                   # ~30-90 px). Safe because the hue+val mask is clean.
    trim_iters: int = 2            # outlier-trim passes (legacy path, use_largest_blob=False)
    trim_sigma: float = 2.0
    # nearest-gate selection: target the LARGEST connected blob, not the centroid
    # of all red pixels (which blends near+far gates and aims between them).
    use_largest_blob: bool = True
    dilate_gaps: bool = True            # 1-px dilation bridges hollow-square JPEG breaks
    largest_blob_tol: float = 0.70      # blobs within this frac of max area are tie-broken
    max_mask_frac_reject: float = 0.20  # if the mask covers >20% it isn't gates -> reject

    # ---- guidance tube (cyan) -- MEASUREMENT ONLY, no guidance wiring yet ----
    # Measured on vision_frame.png: tube hue ~198 (176-239), sat med 0.72, val
    # med 0.52, ~5% of frame, continuous down the course centreline.
    tube_hue_lo: float = 180.0
    tube_hue_hi: float = 215.0
    tube_sat_min: float = 0.30
    tube_val_min: float = 0.40
    tube_lower_band: tuple = (0.60, 0.85)  # frac of H: nearest/widest -> most stable u
    tube_upper_band: tuple = (0.35, 0.55)  # frac of H: curvature reference (bend ahead)
    tube_min_band_frac: float = 0.004      # min masked frac of a band to trust its centre

    # ---- guidance (outer loop, from vision) ----
    # Flight 2: YAW was MASKING lateral error -- it recentres the gate in-frame
    # without translating the drone, so u stayed small (0.02->0.18 over 1.7s)
    # while the real ~0.9 m offset never closed, then blew up from parallax and we
    # passed BESIDE gate 1. Only BANK translates. So bank does the correction; yaw
    # only aligns the nose. Geometry: ~0.9 m over ~2 s ~ 0.45 m/s^2 ~ 2.6 deg bank;
    # k_bank*u at u=0.05 must give ~0.045 rad -> k_bank ~ 1.0.
    # cruise_pitch MUST match the ~-17.8 deg spawn attitude, or the attitude loop
    # holds a CONSTANT pitch correction (flight 5: -8 made it command +10 deg
    # nose-up forever -> climbed and backed up instead of flying the course). To
    # slow the approach, lower hover_cruise (thrust drives forward speed via the
    # tilted thrust vector) -- NOT pitch, which is the same axis holding attitude.
    cruise_pitch_deg: float = -18.0   # forward lean; ~matches spawn so the loop sits quiet
    k_bank: float = 1.0               # desired roll ANGLE (rad) per unit control-u (0.45->1.0)
    max_bank_deg: float = 35.0
    k_yaw: float = 0.15               # nose ALIGNMENT only, not the correction (0.70->0.15)
    hover_cruise: float = 0.30        # 0.32->0.30 to SLOW the approach via thrust (not pitch).
                                      # Still above the ~0.27 rest hover so it won't sink; the
                                      # lower thrust means a smaller forward component off the
                                      # -18 deg tilt -> gentler approach, wider correction window.
    k_thrust_v: float = 0.06          # thrust change per unit v_err. 0.027 too weak (flight 3
                                      # passed below), 0.15 DIVERGED (flight 4: positive
                                      # feedback collapsed thrust to 0.134). 0.06 splits them.
    min_thrust: float = 0.05
    max_thrust: float = 0.60
    # Hard clamp on |thrust - hover_cruise|. The vertical channel must NEVER be
    # able to collapse the flight regardless of gain or sign (flight 4 ran thrust
    # to 0.134 via positive feedback). Permanent safety net, keep it.
    thrust_dev_max: float = 0.06

    # ---- PD derivative on the vision error (react to the error GROWING) ----
    # Flight 3: u and v both ramped ~linearly (0.02->0.82) and P-only was always
    # behind. Add a derivative on u (-> bank) and v (-> thrust) so a growing error
    # is corrected early. Derivative is FILTERED (dirty-derivative, deriv_tau_s):
    # vision frames update slower than the control loop, so a raw frame-to-frame
    # d/dt would be spiky/zero. kd_* ~0.4x the matching P gain.
    kd_u: float = 0.4                 # bank derivative gain (on du/dt), ~0.4 * k_bank
    kd_v: float = 0.024               # thrust derivative gain (on dv/dt), ~0.4 * k_thrust_v
    deriv_tau_s: float = 0.15         # derivative low-pass time constant (s)

    # ---- attitude stabiliser (inner loop, from IMU) ----
    kp_att: float = 3.0               # rad/s of body rate per rad angle error (1.3->3.0 so
                                      # the commanded bank builds in ~0.3s, not ~0.8s)
    kd_att: float = 0.08              # damping on measured body rate (per rad/s) (was 0.35)
    kd_yaw: float = 0.15              # light yaw-rate damping
    # Hard cap on COMMANDED body rate (rad/s). 12 rad/s full deflection is far too
    # much authority for gate centring. Applied to the INTENDED rate (pre plant
    # calibration), so the vehicle really sees <= this. Raise as tuning firms up.
    max_cmd_rate_rad_s: float = 1.2

    # ---- anti-chase gating (first-flight fix: it grabbed distant gates) ----
    # While we have a RECENT gate, reject a detection that is too small (a distant
    # gate) or that jumped too far frame-to-frame (a real gate can't teleport).
    # Once gateless past reacquire_s, gating is dropped so we can re-lock anything.
    # Size Schmitt trigger (flight 4: sizes 0.040/0.041/0.042 against a single 0.04
    # edge thrashed thrust 0.320<->0.236 frame-to-frame). Two thresholds with
    # hysteresis: (re)lock only above accept, keep tracking down to reject.
    gate_size_accept: float = 0.035   # size_frac needed to (re)acquire the size lock
    gate_size_reject: float = 0.025   # drop the lock only below this
    max_u_jump: float = 0.5           # reject |u - last_u| above this while tracking

    # ---- lost-gate behaviour ----
    reacquire_s: float = 0.6          # coast straight-ish this long after losing gate
    search_yaw: float = 0.06          # then yaw toward last-seen side. 0.06*12=0.72 rad/s;
                                      # a literal halve (0.125) would still clamp to 1.2 and
                                      # not change, so set to ~half the clamp instead.
    lost_thrust_scale: float = 1.0    # NEVER below hover while blind. Flight 2 sank: 0.9 x
                                      # 0.265 = 0.239 was BELOW hover. Not-tracking thrust =
                                      # hover_cruise exactly (raise >1.0 if it still sinks).
    search_timeout_s: float = 2.0     # after this long with NO gate, stop searching: level
                                      # wings, zero yaw, level pitch, hold hover. A stationary
                                      # drone that can still see beats corkscrewing off-course.

    # bookkeeping (not a knob)
    name: str = "vq1-vision-servo-v0"


@dataclass
class GateDetection:
    found: bool
    u_err: float = 0.0     # horizontal centroid error, [-1,1], + = gate RIGHT
    v_err: float = 0.0     # vertical centroid error,   [-1,1], + = gate LOW (below centre)
    size_frac: float = 0.0 # sqrt(area)/img_h, apparent gate size
    area_frac: float = 0.0 # masked area / total pixels
    cx: float = 0.0        # pixel centroid (for viz/debug)
    cy: float = 0.0


def rgb_to_hsv_arrays(frame_rgb: np.ndarray):
    """Vectorised RGB->HSV, pure numpy. h in [0,360), s,v in [0,1].

    frame_rgb: (H,W,3) uint8 or float. Returns (h, s, v) float arrays (H,W)."""
    a = frame_rgb.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn

    v = mx
    s = np.where(mx > 1e-6, diff / np.maximum(mx, 1e-6), 0.0)

    h = np.zeros_like(mx)
    mask = diff > 1e-6
    # r is max
    idx = mask & (mx == r)
    h[idx] = (60.0 * ((g[idx] - b[idx]) / diff[idx]) + 360.0) % 360.0
    # g is max
    idx = mask & (mx == g) & (mx != r)
    h[idx] = 60.0 * ((b[idx] - r[idx]) / diff[idx]) + 120.0
    # b is max
    idx = mask & (mx == b) & (mx != r) & (mx != g)
    h[idx] = 60.0 * ((r[idx] - g[idx]) / diff[idx]) + 240.0
    return h, s, v


def _dilate1(mask):
    """1-px 8-connected binary dilation, pure numpy. Bridges the small breaks a
    hollow-square gate frame gets from JPEG/anti-aliasing so it labels as ONE
    blob rather than four disconnected sides."""
    d = mask.copy()
    d[:-1, :] |= mask[1:, :]; d[1:, :] |= mask[:-1, :]
    d[:, :-1] |= mask[:, 1:]; d[:, 1:] |= mask[:, :-1]
    d[:-1, :-1] |= mask[1:, 1:]; d[:-1, 1:] |= mask[1:, :-1]
    d[1:, :-1] |= mask[:-1, 1:]; d[1:, 1:] |= mask[:-1, :-1]
    return d


def _connected_components(mask):
    """8-connected components over the True pixels -> list of coord lists.
    Pure-numpy nonzero + iterative DFS over the pixel set. Intended for SMALL
    masks (gate pixels); callers must gate on mask size first (max_mask_frac)."""
    ys, xs = np.nonzero(mask)
    coords = set(zip(ys.tolist(), xs.tolist()))
    visited = set()
    comps = []
    NB = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    for seed in coords:
        if seed in visited:
            continue
        stack = [seed]
        visited.add(seed)
        comp = []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for dy, dx in NB:
                nb = (y + dy, x + dx)
                if nb in coords and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        comps.append(comp)
    return comps


class GateDetector:
    """Pure-numpy gate detector. No OpenCV/scipy (neither is installed).

    Strategy: colour-threshold to a binary mask, take the centroid of the masked
    pixels with a couple of outlier-trim passes so scattered noise doesn't pull
    the estimate off the dominant (gate) blob. Returns normalised centroid error
    and apparent size. This is deliberately simple and robust; it does not need
    connected-component labelling (which pure numpy makes painful) because the
    gate is expected to be the dominant thresholded object in the forward view.
    """

    def __init__(self, cfg: ServoConfig):
        self.cfg = cfg

    def _mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        h, s, v = rgb_to_hsv_arrays(frame_rgb)
        c = self.cfg
        if c.gate_hue_lo <= c.gate_hue_hi:
            hue_ok = (h >= c.gate_hue_lo) & (h <= c.gate_hue_hi)
        else:  # wrap-around band (e.g. red across 360/0)
            hue_ok = (h >= c.gate_hue_lo) | (h <= c.gate_hue_hi)
        colour = hue_ok & (s >= c.gate_sat_min) & (v >= c.gate_val_min)
        if c.gate_use_brightness_fallback:
            bright = (v >= c.bright_val_min) & (s <= c.bright_sat_max)
            return colour | bright
        return colour

    def _select_blob(self, mask, prefer=None):
        """Pick a gate blob. Dilate 1 px to bridge hollow-square gaps, label
        8-connected, score each blob by its ORIGINAL-mask pixel count.

        If prefer=(u, v, size) is given (the last ACCEPTED detection), prefer
        CONTINUITY: the non-trivial blob closest to it in a combined distance over
        horizontal (du), vertical (dv), AND apparent size (|log size-ratio|). On a
        straight course a far gate shares the near gate's u/v, so SIZE is the
        discriminator -- a gate that halves between frames is not the same gate.
        Otherwise (re-acquiring) pick the largest blob, tie-broken by lowest
        centroid (nearest in perspective). Returns (cx, cy, area_px) or None."""
        c = self.cfg
        H, W = mask.shape
        work = _dilate1(mask) if c.dilate_gaps else mask
        stats = []
        for comp in _connected_components(work):
            pts = [(y, x) for (y, x) in comp if mask[y, x]]  # count ORIGINAL px only
            if not pts:
                continue
            n = len(pts)
            cx = sum(p[1] for p in pts) / n
            cy = sum(p[0] for p in pts) / n
            stats.append((n, cx, cy))
        if not stats:
            return None
        if prefer is not None:
            ru, rv, rsize = prefer
            rsize = max(rsize, 1e-6)
            floor = c.min_area_frac * H * W

            def cont_score(s):
                n, cx, cy = s
                u = (cx - W / 2.0) / (W / 2.0)
                v = (cy - H / 2.0) / (H / 2.0)
                size = max(math.sqrt(n) / H, 1e-6)
                # size-ratio weighted 2x: it is the near/far discriminator here.
                return abs(u - ru) + abs(v - rv) + 2.0 * abs(math.log(size / rsize))

            near = [s for s in stats if s[0] >= floor]
            if near:
                n, cx, cy = min(near, key=cont_score)
                return float(cx), float(cy), float(n)
        max_area = max(s[0] for s in stats)
        cand = [s for s in stats if s[0] >= c.largest_blob_tol * max_area]
        n, cx, cy = max(cand, key=lambda s: s[2])   # lowest in frame = nearest
        return float(cx), float(cy), float(n)

    def detect(self, frame_rgb: np.ndarray, prefer=None) -> GateDetection:
        c = self.cfg
        H, W = frame_rgb.shape[0], frame_rgb.shape[1]
        total = H * W
        mask = self._mask(frame_rgb)
        n_true = int(mask.sum())
        # Empty, or the mask blew up (bad thresholds / whole-frame red) -> not a gate.
        if n_true == 0 or n_true > c.max_mask_frac_reject * total:
            return GateDetection(found=False)

        if c.use_largest_blob:
            sel = self._select_blob(mask, prefer=prefer)
            if sel is None:
                return GateDetection(found=False)
            cx, cy, area = sel
        else:
            # Legacy: trimmed centroid of ALL masked pixels (blends near+far gates).
            ys, xs = np.nonzero(mask)
            xs = xs.astype(np.float32); ys = ys.astype(np.float32)
            for _ in range(c.trim_iters):
                mx, my = xs.mean(), ys.mean()
                d = np.sqrt((xs - mx) ** 2 + (ys - my) ** 2)
                keep = d <= (d.mean() + c.trim_sigma * d.std() + 1e-6)
                if keep.sum() < max(4, c.min_area_frac * total * 0.5):
                    break
                xs, ys = xs[keep], ys[keep]
            cx, cy, area = float(xs.mean()), float(ys.mean()), float(xs.size)

        area_frac = area / total
        if area_frac < c.min_area_frac:
            return GateDetection(found=False)

        u_err = (cx - W / 2.0) / (W / 2.0)   # + = right
        v_err = (cy - H / 2.0) / (H / 2.0)   # + = low (below centre)
        size_frac = math.sqrt(area) / H
        return GateDetection(found=True, u_err=u_err, v_err=v_err,
                             size_frac=size_frac, area_frac=area_frac, cx=cx, cy=cy)


@dataclass
class TubeMeasurement:
    found: bool
    u_tube: float = 0.0     # lower-band horizontal centre, [-1,1], + = tube RIGHT of us
    curvature: float = 0.0  # u_upper - u_lower; + = course bends RIGHT ahead
    u_lower: float = 0.0
    u_upper: float = 0.0
    area_frac: float = 0.0


class TubeDetector:
    """MEASUREMENT-ONLY detector for the cyan guidance tube down the course centre.

    Not wired into guidance yet (held for the blend law, pending the rate-vs-angle
    verdict). Exposes a coarse course-following signal that is far more continuous
    than a distant gate blob:

      u_tube    -- horizontal centre of the tube in a LOWER band (nearest/widest,
                   most stable). Keep ~0 to stay centred on the course.
      curvature -- (upper-band centre) - (lower-band centre): the far part of the
                   tube shifting right of the near part means the course bends
                   right ahead. Anticipatory steering a single gate can't give.
    """

    def __init__(self, cfg: ServoConfig):
        self.cfg = cfg

    def _mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        h, s, v = rgb_to_hsv_arrays(frame_rgb)
        c = self.cfg
        return ((h >= c.tube_hue_lo) & (h <= c.tube_hue_hi)
                & (s >= c.tube_sat_min) & (v >= c.tube_val_min))

    def _band_center(self, mask, y0f, y1f):
        """Normalised horizontal centre of masked pixels in a horizontal band, or
        None if the band is too sparse to trust."""
        H, W = mask.shape
        y0, y1 = int(y0f * H), int(y1f * H)
        band = mask[y0:y1, :]
        n = int(band.sum())
        if band.size == 0 or n < self.cfg.tube_min_band_frac * band.size:
            return None
        cx = float(np.nonzero(band)[1].mean())
        return (cx - W / 2.0) / (W / 2.0)

    def measure(self, frame_rgb: np.ndarray) -> TubeMeasurement:
        c = self.cfg
        mask = self._mask(frame_rgb)
        area_frac = float(mask.sum()) / mask.size
        u_lower = self._band_center(mask, *c.tube_lower_band)
        u_upper = self._band_center(mask, *c.tube_upper_band)
        if u_lower is None:
            return TubeMeasurement(found=False, area_frac=area_frac)
        curv = (u_upper - u_lower) if u_upper is not None else 0.0
        return TubeMeasurement(found=True, u_tube=u_lower, curvature=curv,
                               u_lower=u_lower, u_upper=(u_upper or 0.0),
                               area_frac=area_frac)


def gravity_to_roll_pitch(gravity_frd):
    """gravity-down FRD [gx,gy,gz] -> (roll, pitch) rad. Level -> (0,0)."""
    gx, gy, gz = float(gravity_frd[0]), float(gravity_frd[1]), float(gravity_frd[2])
    roll = math.atan2(gy, gz)
    pitch = math.atan2(-gx, gz)
    return roll, pitch


class VisionServoController:
    """Hand-written outer (vision) + inner (IMU attitude) controller.

    Public API mirrors the model command source used by the adapter:
        command(frame_rgb, telemetry, active_gate) -> {throttle,roll,pitch,yaw}

    `detect()` is exposed for the offline dry-run harness / calibration tool.
    """

    def __init__(self, cfg: ServoConfig | None = None):
        self.cfg = cfg or ServoConfig()
        # Loud guard: cruise_pitch far from the spawn attitude makes the attitude
        # loop hold a constant pitch correction (flight-5 class error). Impossible
        # to make silently now.
        _dp = self.cfg.cruise_pitch_deg - SPAWN_PITCH_DEG
        if abs(_dp) > CRUISE_PITCH_TOL_DEG:
            logger.warning(
                "[servo] *** cruise_pitch_deg=%.1f is %+.1f deg from the ~%.1f deg "
                "SPAWN attitude -- the attitude loop will hold a CONSTANT ~%+.1f deg "
                "pitch correction (climb + back up, NOT fly the course). Set "
                "cruise_pitch_deg near %.1f. ***",
                self.cfg.cruise_pitch_deg, _dp, SPAWN_PITCH_DEG, _dp, SPAWN_PITCH_DEG)
        self.detector = GateDetector(self.cfg)
        self.tube = TubeDetector(self.cfg)   # measurement only; not yet in command()
        self._last_seen_t = -1e9
        self._last_u_err = 0.0
        self._last_v_err = 0.0
        self._last_size = 0.0     # last ACCEPTED detection (u,v,size) for continuity
        self._u_filt = 0.0
        self._v_filt = 0.0        # low-pass state for the filtered PD derivative
        self._last_cmd_t = -1e9
        self._size_locked = False  # size Schmitt-trigger state
        self._active_gate_prev = None
        self._t0 = None
        self._last_log_t = -1e9   # throttle for the per-frame tuning log

    def reset(self):
        """Clear per-race transient state. Call on the GO edge so a prior race's
        last-seen bearing / timers can't bleed into the first frames."""
        self._last_seen_t = -1e9
        self._last_u_err = 0.0
        self._last_v_err = 0.0
        self._last_size = 0.0
        self._u_filt = 0.0
        self._v_filt = 0.0
        self._last_cmd_t = -1e9
        self._size_locked = False
        self._t0 = None
        logger.info("[servo] reset for fresh race")

    # -- inner loop ---------------------------------------------------------
    def _attitude_rates(self, des_roll, des_pitch, des_yaw_rate_norm,
                        gravity_frd, gyro):
        """PD from measured (gravity-derived) roll/pitch to normalised body rates.
        des_* angles in rad; des_yaw_rate_norm already normalised [-1,1]."""
        c = self.cfg
        roll, pitch = gravity_to_roll_pitch(gravity_frd)
        p, q, r = float(gyro[0]), float(gyro[1]), float(gyro[2])

        roll_rate = c.kp_att * (des_roll - roll) - c.kd_att * p       # rad/s
        pitch_rate = c.kp_att * (des_pitch - pitch) - c.kd_att * q    # rad/s
        yaw_rate = des_yaw_rate_norm * MAX_BODY_RATE - c.kd_yaw * r   # rad/s

        # Clamp the COMMANDED rate to max_cmd_rate_rad_s (normalised limit), not the
        # full +/-1 (=+/-MAX_BODY_RATE). Caps authority for gate centring.
        lim = c.max_cmd_rate_rad_s / MAX_BODY_RATE
        roll_n = float(np.clip(roll_rate / MAX_BODY_RATE, -lim, lim))
        pitch_n = float(np.clip(pitch_rate / MAX_BODY_RATE, -lim, lim))
        yaw_n = float(np.clip(yaw_rate / MAX_BODY_RATE, -lim, lim))
        return roll_n, pitch_n, yaw_n

    # -- full step ----------------------------------------------------------
    def command(self, frame_rgb, telemetry, active_gate=None, now=None):
        c = self.cfg
        now = time.time() if now is None else now
        if self._t0 is None:
            self._t0 = now

        gravity = telemetry.get("gravity_frd", (0.0, 0.0, 9.81))
        gyro = telemetry.get("velocity", (0.0, 0.0, 0.0))

        # Gate-sequence bookkeeping: log advances, reset lost-timer on a fresh gate.
        if active_gate is not None and active_gate != self._active_gate_prev:
            if self._active_gate_prev is not None:
                logger.info("[servo] active_gate %s -> %s (advanced)",
                            self._active_gate_prev, active_gate)
            self._active_gate_prev = active_gate

        # Track continuity: while we've had a gate recently, bias the detector
        # toward the same blob and vet the detection; once gateless past
        # reacquire_s, drop the gating and re-lock anything.
        had_recent = (now - self._last_seen_t) <= c.reacquire_s
        prefer = ((self._last_u_err, self._last_v_err, self._last_size)
                  if had_recent else None)
        det = self.detector.detect(frame_rgb, prefer=prefer)
        cruise_pitch = math.radians(c.cruise_pitch_deg)

        # Anti-chase gating: reject a distant (small) or teleporting detection
        # while tracking, so a gate leaving frame doesn't make us chase far gates.
        # Size uses a Schmitt trigger (accept/reject thresholds) to stop boundary
        # thrash. When re-acquiring (not had_recent) we don't gate, but seed the
        # size lock for the frames that follow.
        accepted = det.found
        reject = ""
        if det.found:
            if had_recent:
                lo = c.gate_size_reject if self._size_locked else c.gate_size_accept
                size_ok = det.size_frac >= lo
                self._size_locked = size_ok
                if not size_ok:
                    accepted, reject = False, "small"
                elif abs(det.u_err - self._last_u_err) > c.max_u_jump:
                    accepted, reject = False, "jump"
            else:
                self._size_locked = det.size_frac >= c.gate_size_accept
        else:
            self._size_locked = False

        # LATERAL SIGN FIX (flights 6-7, root cause). The actuator is verified K=+1
        # and the vertical channel is correct, yet BOTH bank and yaw were inverted
        # -- the two lateral channels share exactly one input, u_err. Horizontal-
        # only inversion == a horizontally-MIRRORED FPV image (a flip inverts
        # left-right but not up-down, matching the data). The detector reports
        # IMAGE coords; convert to a PHYSICAL control error here so guidance uses
        # POSITIVE gains. u_ctrl>0 => gate physically RIGHT => bank/yaw RIGHT.
        # Continuity/gating deliberately stay in IMAGE coords (det.u_err) -- they
        # compare blob positions frame-to-frame, correct regardless of the mirror.
        u_ctrl = -det.u_err

        if accepted:
            # Filtered PD derivative on control-u, v (dirty-derivative: low-pass
            # then diff, robust to the vision stream updating slower than control).
            if not had_recent:                       # fresh lock -> seed, no D spike
                self._u_filt, self._v_filt = u_ctrl, det.v_err
                du_dt = dv_dt = 0.0
            else:
                dt = min(max(now - self._last_cmd_t, 1e-3), 0.5)
                a = dt / (c.deriv_tau_s + dt)
                self._u_filt += a * (u_ctrl - self._u_filt)
                self._v_filt += a * (det.v_err - self._v_filt)
                du_dt = (u_ctrl - self._u_filt) / c.deriv_tau_s
                dv_dt = (det.v_err - self._v_filt) / c.deriv_tau_s
            self._last_cmd_t = now
            self._last_seen_t = now
            self._last_u_err = det.u_err   # IMAGE u for continuity/gating
            self._last_v_err = det.v_err
            self._last_size = det.size_frac

            # BANK translates: P on control-u + D on d(control-u)/dt. Positive gains
            # (the mirror is already handled in u_ctrl).
            des_roll = float(np.clip(c.k_bank * u_ctrl + c.kd_u * du_dt,
                                     -math.radians(c.max_bank_deg),
                                     math.radians(c.max_bank_deg)))
            des_yaw = float(np.clip(c.k_yaw * u_ctrl, -1.0, 1.0))  # nose only, P
            des_pitch = cruise_pitch
            # gate LOW in frame (v_err>0) => we're too HIGH => descend => less thrust.
            # P on v + D on dv/dt so a growing vertical error is chased early.
            thrust = c.hover_cruise - (c.k_thrust_v * det.v_err + c.kd_v * dv_dt)
        else:
            # No usable gate. Coast+search, then GIVE UP: after search_timeout_s,
            # level + zero yaw + hold hover (a corkscrew off-course is worse).
            dt_lost = now - self._last_seen_t
            if dt_lost <= c.reacquire_s:             # coast straight, reacquire
                des_roll, des_pitch, des_yaw = 0.0, cruise_pitch, 0.0
                thrust = c.hover_cruise * c.lost_thrust_scale
            elif dt_lost <= c.search_timeout_s:      # gentle yaw toward last-seen side
                des_roll, des_pitch = 0.0, cruise_pitch
                # yaw toward where the gate PHYSICALLY was (last image-u is mirrored)
                des_yaw = math.copysign(c.search_yaw, (-self._last_u_err) or 1.0)
                thrust = c.hover_cruise * c.lost_thrust_scale
            else:                                    # give up: level + hold hover
                des_roll, des_pitch, des_yaw = 0.0, 0.0, 0.0
                thrust = c.hover_cruise

        # Hard vertical safety net: |thrust - hover| <= thrust_dev_max, ALWAYS,
        # regardless of branch/gain/sign (prevents the flight-4 thrust collapse).
        # Also kept within [min_thrust, max_thrust].
        lo_t = max(c.min_thrust, c.hover_cruise - c.thrust_dev_max)
        hi_t = min(c.max_thrust, c.hover_cruise + c.thrust_dev_max)
        thrust = float(np.clip(thrust, lo_t, hi_t))

        roll_n, pitch_n, yaw_n = self._attitude_rates(
            des_roll, des_pitch, des_yaw, gravity, gyro)

        # Per-frame tuning log (throttled ~10 Hz). g=active_gate (watch it tick to
        # see a pass the instant it happens); det=blob detected; use=acted (Y) or why
        # not (small/jump/n); thr vs hover makes SINKING obvious (dthr<0 => below hover).
        if now - self._last_log_t >= 0.1:
            self._last_log_t = now
            logger.info(
                "[servo] g=%s det=%s use=%s u=%+.3f v=%+.3f size=%.3f | intended rad/s "
                "roll=%+.2f pitch=%+.2f yaw=%+.2f | thr=%.3f (hov %.3f, d%+.3f)",
                active_gate, "Y" if det.found else "n",
                ("Y" if accepted else (reject or "n")),
                det.u_err, det.v_err, det.size_frac,
                roll_n * MAX_BODY_RATE, pitch_n * MAX_BODY_RATE, yaw_n * MAX_BODY_RATE,
                thrust, c.hover_cruise, thrust - c.hover_cruise)

        return {"throttle": thrust, "roll": roll_n, "pitch": pitch_n, "yaw": yaw_n,
                "_debug": {"found": det.found, "accepted": accepted, "reject": reject,
                           "u_err": det.u_err, "u_ctrl": u_ctrl, "v_err": det.v_err,
                           "size": det.size_frac, "des_roll": des_roll,
                           "des_pitch": des_pitch, "des_yaw": des_yaw}}
