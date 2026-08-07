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

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("vq1_vision_servo")

# Must match dcl_mavlink_adapter.MAX_BODY_RATE — normalised rate 1.0 == this rad/s.
MAX_BODY_RATE = 12.0

# ===========================================================================
# THRUST -> SINK-RATE MAP  (the ONE table to update from measurements)
# ---------------------------------------------------------------------------
# Points: (thrust_fraction, steady_sink_rate_m_s). sink > 0 = descending.
# Monotone: lower thrust -> more sink. Linearly interpolated (np.interp), so
# refining a point -- or adding a 5th -- is a ONE-LINE data change here; the
# control logic never changes.
#
# MEASURED on v3385 (2026-07-25, --hover-probe SINK RATE). Roughly linear, ~ -0.18
# m/s sink per +0.01 thrust. Note 0.29 already sinks +0.72 -- our "hover" is a
# gentle glide, which suits a descending course. To refine: re-measure and edit a
# row, or add a point (keep it monotone: lower thrust -> more sink).
THRUST_SINK_MAP = [
    (0.29, 0.72),
    (0.26, 1.13),
    (0.23, 1.79),
    (0.20, 2.30),
]


def thrust_for_sink(sink_rate: float) -> float:
    """Inverse-interpolate THRUST_SINK_MAP: needed sink rate (m/s, + = down) ->
    thrust fraction. Clamped to the map's range (np.interp returns the endpoints
    outside it). A needed CLIMB (sink < 0) maps to the highest-thrust point."""
    pts = sorted(THRUST_SINK_MAP, key=lambda p: p[1])   # ascending sink
    sinks = [p[1] for p in pts]
    thrusts = [p[0] for p in pts]
    return float(np.interp(sink_rate, sinks, thrusts))

# Measured resting/spawn body pitch on v3385 (nose-down). cruise_pitch_deg must
# stay near this or the attitude loop holds a constant pitch correction and the
# drone climbs/backs up instead of flying the course (flight 5). Guard below.
SPAWN_PITCH_DEG = -17.8
CRUISE_PITCH_TOL_DEG = 5.0

# LATERAL SIGN CONVENTION (was mislabeled "mirror" -- flights 6-7). CORRECTED:
# the DGX confirmed the FPV camera is NOT mirrored (a gate at image x=393 matches
# the +2.21 deg right-of-nose bearing; image-right = world-right). So u_err>0 IS a
# physically-right gate. The sign is instead the DCL sim's LATERAL FRAME: a
# positive commanded roll banks/translates the drone to its LEFT (the sim's body-y
# is left-positive). The forward/x axis is standard -- which is exactly why PITCH
# works with no sign and only the lateral (roll+yaw) channel needs one.
#
#   Sign trace, right gate (u_err > 0):
#     want physical RIGHT translation (to centre the gate)
#     -> in this sim a right translation is a NEGATIVE roll setpoint (body-y left+)
#     -> des_roll = k_bank * (LATERAL_SIGN * u_err),  LATERAL_SIGN = -1
#     -> attitude loop drives measured roll (atan2(gy,gz)) to des_roll; gy and the
#        roll gyro are self-consistent, so the loop converges
#     -> adapter wire_body_rate applies PLANT_RATE_CALIB (K_roll=-2.55: wire<->gyro
#        sign+scale -- a SEPARATE, correctly-placed correction, present on all axes)
#
# It is NOT a mirror, and it cannot move earlier: the inner loop's gravity-y and
# roll-gyro already agree, so flipping the roll ESTIMATE alone would invert the
# feedback and DIVERGE the loop. The minimal, correct home for the sim's
# left-handed lateral convention is here, at the vision->roll/yaw setpoint, where
# our right-positive vision meets the sim's left-positive roll.
LATERAL_SIGN = -1.0

# --- PROBE INSTRUMENTATION (branch gate3-pnp-probe, TEMPORARY) -----------------
# Set VQ1_PROBE_FRAME_SHAPE=1 to have GateDetector.detect() print the shape it is
# actually handed, ONCE per process. Default off => the live path is unchanged.
# Delete this and its use in detect() when the branch is dropped.
_PROBE_FRAME_SHAPE = bool(os.environ.get("VQ1_PROBE_FRAME_SHAPE"))
_PROBE_SHAPE_LOGGED = False


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
    dilate_gaps: bool = True            # dilation bridges hollow-square JPEG breaks
    gate_dilate_px: int = 1             # dilation RADIUS. 1 bridges a 2-px break. MEASURED
                                        # (vision_frame.png 7/27 19:16): the bright cyan tube
                                        # glow crossing gate 1's lower half opens a 6-px gap,
                                        # splitting the gate into two blobs (483 px @ cy=167
                                        # + 203 px @ cy=193) and biasing v_err to -0.071 when
                                        # the merged truth is -0.029 (2.4x). Radius 3 closes
                                        # it (686 px @ cy=174.7) and still leaves gate 2
                                        # (cy=219) a separate blob. Default stays 1 so the
                                        # legacy servo/hybrid paths are bit-identical.
    largest_blob_tol: float = 0.70      # blobs within this frac of max area are tie-broken
    max_mask_frac_reject: float = 0.20  # if the mask covers >20% it isn't gates -> reject

    # ---- guidance tube (cyan) -- WIRED into HybridController lateral (u_lower
    #      centring + k_tube_lead*curvature lead); measurement-only in VisionServo. --
    # Re-measured on vision_frame.png (real FPV): tube hue 193-206 (med 199), sat
    # med 0.65 (5th pct 0.41), val med 0.41 (5th pct 0.31). The OLD val_min=0.40
    # sat right on the val median, so half the tube was within 0.01 of the cutoff
    # -> any dimmer frame (far tube, nose-down attitude, dark course section)
    # collapsed the mask: that was the live 0.001-0.059 area flicker / found=0.
    # val_min=0.28 sits below the 5th pctile, so the mask survives dimming (verified:
    # found=True, area>=0.04 down to 60% brightness vs old band's 0.005/found=False).
    tube_hue_lo: float = 180.0
    tube_hue_hi: float = 215.0
    tube_sat_min: float = 0.30
    tube_val_min: float = 0.28
    tube_lower_band: tuple = (0.60, 0.85)  # frac of H: nearest/widest -> most stable u
    tube_upper_band: tuple = (0.35, 0.55)  # frac of H: curvature reference (bend ahead)
    tube_min_band_frac: float = 0.004      # min masked frac of a band to trust its centre

    # ---- RAIL extraction (the course path is two cyan LINES, not a fill) ----------
    # The old mask+area approach asked "how much cyan is there", which is the wrong
    # question for a THIN BRIGHT CURVE: a perfectly visible rail one pixel wide has
    # essentially zero area, so area_frac read ~0 and the tube was declared unseen.
    # That is the <4%-lock false negative. area_frac is still REPORTED (it is a useful
    # log column) but it never gates anything.
    #
    # THRESHOLD, measured on vision_frame.png. The rails are high-B, high-G, low-R, so
    # the discriminant is the chroma difference min(G,B) - R, which is ~0 on every grey
    # /white structure in the scene (buildings, the horizon grid) whatever its
    # brightness, and large only on saturated cyan. Measured over the whole frame that
    # score has p50 0, p90 45, p99 95; the rails sit at 100-255. 60 is above the
    # background's 90th percentile and below the rails, and holds 201 of 210 scanned
    # rows. A brightness floor is applied as well so dim cyan haze cannot masquerade
    # as a rail edge.
    rail_score_min: float = 60.0        # min(G,B) - R on the rails; grey scenery ~0
    rail_val_min: float = 90.0          # 0-255 floor on max(G,B): rails are bright
    rail_min_run_px: int = 2            # ignore 1-px speckle when picking a row's runs
    rail_merge_gap_px: int = 6          # runs closer than this are one feature
    rail_band: tuple = (0.30, 1.00)     # frac of H scanned for rails (below the horizon)
    #                                     bottom is ALREADY the last row: the band is not
    #                                     what runs out when the path sinks -- see
    #                                     rail_lookahead_below_conv.
    rail_lookahead: float = 0.75        # frac of H where the lateral offset is measured
    rail_lookahead_below_conv: float = 0.12
    #   Keep the look-ahead at least this far BELOW the convergence row. The rails
    #   diverge downward from the vanishing point, so above it they have crossed and the
    #   separation goes negative -- which fails the lock on a perfectly good fit. As the
    #   nose rises the path sinks and the convergence slides down past a fixed 0.75H,
    #   which is exactly that failure. 0 pins the look-ahead at rail_lookahead.
    rail_curv_row: float = 0.45         # frac of H used as the curvature (far) reference
    rail_min_rows: int = 20             # rows that must yield a rail point to fit at all
    rail_max_fit_rms: float = 6.0       # px; a worse fit is not a rail, it is clutter
    rail_min_sep_frac: float = 0.05     # min rail separation at look-ahead, frac of W
    rail_merge_recover: bool = False
    #   Recover rows where the two rails have MERGED into one narrow run because the path
    #   is distant. Those rows are otherwise discarded as "one rail, unusable alone",
    #   which on filt29's g1->g2 leg threw away 86.2% of ticks while the path was plainly
    #   in frame (area_frac 0.006, rows 0). With this on, such a run's CENTRE is used as
    #   the path centre. DEFAULT OFF: it has not been validated against real frames from
    #   that leg -- a merged run yields a bearing to a distant path, not a separation, so
    #   it is a weaker reference and must be proven before it steers.
    rail_wide_run_frac: float = 0.06    # a single run at least this wide (frac of W) is
    #                                     the corridor seen merged: use its two EDGES

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
    hover_cruise: float = 0.29        # slightly BELOW cruise-hover so the DEFAULT trajectory
                                      # descends gently (the whole course drops 26 m and the
                                      # drone starts above gate 1), instead of needing active
                                      # descent at every gate. Still above rest hover ~0.27.
    k_thrust_v: float = 0.16          # thrust per unit v_err. History: 0.027 too weak (flight
                                      # 3), 0.15 diverged at the OLD 0.06 clamp (flight 4), 0.10
                                      # never reached the wider clamp so the GAIN capped descent
                                      # (flight 9: thrust bottomed 0.185, floor is 0.170). 0.16:
                                      # v=0.5 -> 0.08 dev, v=0.75 saturates the 0.12 clamp -- the
                                      # gain can now reach the safety net instead of capping short.
    min_thrust: float = 0.05
    max_thrust: float = 0.60
    # Hard clamp on |thrust - hover_cruise|. The vertical channel must NEVER be
    # able to collapse the flight regardless of gain/sign (flight 4 ran to 0.134).
    # 0.06->0.12: flight 8 pinned the floor with v still growing = too little
    # descent authority. Wider band gives it; the clamp still blocks a collapse.
    thrust_dev_max: float = 0.12

    # ---- PD derivative on the vision error (react to the error GROWING) ----
    # Flight 3: u and v both ramped ~linearly (0.02->0.82) and P-only was always
    # behind. Add a derivative on u (-> bank) and v (-> thrust) so a growing error
    # is corrected early. Derivative is FILTERED (dirty-derivative, deriv_tau_s):
    # vision frames update slower than the control loop, so a raw frame-to-frame
    # d/dt would be spiky/zero. kd_* ~0.4x the matching P gain.
    kd_u: float = 0.4                 # bank derivative gain (on du/dt), ~0.4 * k_bank
    kd_v: float = 0.04                # thrust derivative gain (on dv/dt), scaled with k_thrust_v
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

    # ---- hybrid controller (three-tier blend: gate / tube / feedforward) ----
    cruise_speed_mps: float = 10.0    # forward-speed estimate for glide feedforward
                                      # (~161 m in ~15 s). Vision-trim absorbs the error.
    hybrid_gate_s_lo: float = 0.05    # gate size where gate-authority (w_gate) starts ramping
    hybrid_gate_s_hi: float = 0.15    # gate size for FULL gate authority (precise pass)
    hybrid_tube_area_min: float = 0.01  # tube masked-area frac to count as "visible"
    k_tube_lead: float = 0.5          # anticipatory lateral FF: weight on tube CURVATURE
                                      # (u_upper-u_lower, where the line LEADS) added to the
                                      # near-band centring. u_lower keeps you ON the line;
                                      # curvature steers toward where it GOES (e.g. gate 1 at
                                      # +2.21deg). Conservative start -- tune live.
    hybrid_v_trim_band: float = 0.03  # vision v-error may only move thrust +/- this much
                                      # around the feedforward glide. FEEDFORWARD carries the
                                      # descent (slope is shallow); vision trims residual only.
    hybrid_seg_timeout_mult: float = 2.5  # SAFETY: if active_gate hasn't advanced in
                                      # (segment_time * this), advance the schedule anyway so
                                      # a missed gate can't stall the controller forever.
                                      # active_gate (ground truth) still LEADS; this only
                                      # covers a stall, never an early drift-driven advance.

    # ---- open-loop flier (OpenLoopFlier: the anti-flip controller) ----
    # Every reactive flight died to a vertical/attitude RUNAWAY; the fixed-thrust
    # 0.40 hover probe flew straightest and never flipped. So: thrust is OPEN-LOOP
    # (sink schedule, NO vertical PID), attitude is a GENTLE hard-clamped hold (not
    # the stiff kp_att=3.0 winder), and tube-curvature steering is the ONLY closed
    # loop. A flip is structurally impossible: no positive-feedback source + a hard
    # small rate clamp = nothing can wind up.
    ol_thrust_lo: float = 0.18        # open-loop thrust hard floor (never collapse)
    ol_thrust_hi: float = 0.34        # open-loop thrust hard ceiling (never rocket up)
    ol_kp_att: float = 1.0            # GENTLE attitude-hold P (vs stiff kp_att=3.0)
    ol_kd_att: float = 0.05           # rate damping on the hold
    ol_max_rate_rad_s: float = 0.6    # HARD clamp on every commanded body rate -> no wind-up
    ol_max_bank_deg: float = 25.0     # lateral setpoint clamp (tighter than max_bank_deg)
    ol_fence_deg: float = 28.0        # attitude FENCE: past this tilt, stop steering + recover
    ol_gate_trim: float = 0.30        # max fraction gate-centring may pull lateral when gate is big

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


def _dilate1(mask, radius=1):
    """8-connected binary dilation by `radius` px, pure numpy. Bridges the small
    breaks a hollow-square gate frame gets from JPEG/anti-aliasing so it labels as
    ONE blob rather than four disconnected sides. A radius-r pass closes a gap of
    up to 2r px; see ServoConfig.gate_dilate_px for why gate 1 needs r=3."""
    d = mask
    for _ in range(max(1, int(radius))):
        s = d
        d = s.copy()
        d[:-1, :] |= s[1:, :]; d[1:, :] |= s[:-1, :]
        d[:, :-1] |= s[:, 1:]; d[:, 1:] |= s[:, :-1]
        d[:-1, :-1] |= s[1:, 1:]; d[:-1, 1:] |= s[1:, :-1]
        d[1:, :-1] |= s[:-1, 1:]; d[1:, 1:] |= s[:-1, :-1]
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
        work = _dilate1(mask, c.gate_dilate_px) if c.dilate_gaps else mask
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
        # --- PROBE INSTRUMENTATION (branch gate3-pnp-probe) ----------------------
        # Reports the shape the detector ACTUALLY consumes, once per process. Costs
        # one module-global bool test per call on the steady-state path -- no
        # formatting, no I/O, no allocation -- and is OFF unless the env var is set,
        # so it cannot spam or slow the control loop. Remove with the branch.
        global _PROBE_SHAPE_LOGGED
        if _PROBE_FRAME_SHAPE and not _PROBE_SHAPE_LOGGED:
            _PROBE_SHAPE_LOGGED = True
            print(f"[PROBE] GateDetector.detect() input shape = "
                  f"H={H} W={W} C={frame_rgb.shape[2] if frame_rgb.ndim > 2 else 1} "
                  f"dtype={frame_rgb.dtype}", flush=True)
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
    found: bool             # LOCK: both rails fitted. The only validity flag.
    u_tube: float = 0.0     # lateral offset at look-ahead, [-1,1], + = path RIGHT of us
    curvature: float = 0.0  # u_upper - u_lower; + = course bends RIGHT ahead
    u_lower: float = 0.0    # path centre at the look-ahead row
    u_upper: float = 0.0    # path centre at the far (curvature) row
    area_frac: float = 0.0  # REPORTED ONLY -- never gates anything (see rail_score_min)
    # --- rail geometry -------------------------------------------------------------
    v_converge: float = 0.0   # vanishing point row, [-1,1], + = BELOW frame centre
    converge_ok: bool = False  # the rails actually meet inside/near the frame
    rail_sep: float = 0.0     # rail separation at look-ahead, frac of W
    n_rows: int = 0           # rows that contributed a rail point
    fit_rms: float = 0.0      # worse of the two rails' fit residual, px
    y_look: float = 0.0       # the row the offset was measured at, px (may be ADAPTIVE:
    #                           it is pushed below the convergence as the path sinks)
    n_merged: int = 0         # rows where the two rails were MERGED into one narrow run
    #                           (a distant path): a bearing, not a separation -- see
    #                           ServoConfig.rail_merge_recover
    left_poly: tuple = ()     # x = f(y) quadratic coefficients, px (for the overlay)
    right_poly: tuple = ()


class TubeDetector:
    """RAIL detector for the cyan course path.

    THE PATH IS TWO LINES, NOT A REGION. The previous version masked cyan and asked
    how much of the frame it covered. That is the wrong question about a thin bright
    curve: a rail that is one or two pixels wide is fully visible and still has
    essentially zero area, so the area test read "not found" on frames where a human
    can see the path perfectly. That is the sub-4% lock rate -- a FALSE NEGATIVE, not
    a weak signal. Nothing here gates on area; area_frac is reported for the log only.

    METHOD, per frame:
      1. Threshold on CHROMA, not brightness: score = min(G,B) - R. The rails are the
         only saturated cyan in a scene made of grey buildings and a red gate, and
         this score is ~0 on grey whatever its brightness, so it separates the rails
         from bright white scenery that any value/luminance threshold would admit.
      2. Per row, group masked pixels into runs and take the OUTERMOST ones: the left
         rail is the first run, the right rail the last. The wide run in between is
         the hazy corridor floor, and taking outer runs discards it for free. A single
         WIDE run means the two rails have merged (close range / low altitude); its
         two EDGES are then the two boundaries, which is the same measurement.
      3. Fit x = f(y) as a quadratic per rail, ROBUSTLY -- iteratively rejecting
         points beyond 3 MAD. The rails are smooth curves; clutter is not, and a
         handful of bright pixels off a building must not bend the fit. Measured on
         vision_frame.png the fits land at 1.2 px RMS over ~200 rows.
      4. Report the path centre at a fixed LOOK-AHEAD row, and the row where the two
         fitted rails meet (the vanishing point).

    Outputs (all normalised [-1,1] unless noted):
      found       -- LOCK: both rails fitted and plausible. The only validity flag.
      u_tube      -- path centre minus frame centre at the look-ahead row. This is
                     the lateral offset: 0 = centred on the path.
      v_converge  -- vanishing-point row; + = below frame centre. Where the path
                     runs off to, i.e. the course's vertical position ahead.
      curvature   -- far centre minus near centre; + = the course bends RIGHT ahead.
    """

    def __init__(self, cfg: ServoConfig):
        self.cfg = cfg

    def _mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        """CHROMA mask. int16 so the G/B minus R difference can go negative without
        wrapping (uint8 would make grey scenery look like a huge positive score)."""
        f = frame_rgb.astype(np.int16)
        R, G, B = f[:, :, 0], f[:, :, 1], f[:, :, 2]
        gb = np.minimum(G, B)
        return (gb - R >= self.cfg.rail_score_min) & (gb >= self.cfg.rail_val_min)

    def _row_runs(self, row_mask):
        """Contiguous runs of masked pixels in one row as (x0, x1) inclusive. Runs
        separated by less than rail_merge_gap_px are joined: a rail broken by
        compression noise is one rail, not two."""
        x = np.nonzero(row_mask)[0]
        if x.size == 0:
            return []
        splits = np.nonzero(np.diff(x) > self.cfg.rail_merge_gap_px)[0]
        runs, start = [], 0
        for s in splits:
            runs.append((int(x[start]), int(x[s])))
            start = s + 1
        runs.append((int(x[start]), int(x[-1])))
        return [r for r in runs if r[1] - r[0] + 1 >= self.cfg.rail_min_run_px]

    def _rail_points(self, mask):
        """Per-row (y, x_left, x_right) for the two rail boundaries."""
        c = self.cfg
        H, W = mask.shape
        y0, y1 = int(c.rail_band[0] * H), int(c.rail_band[1] * H)
        wide = c.rail_wide_run_frac * W
        ys, xl, xr = [], [], []
        merged = 0
        for y in range(y0, y1):
            runs = self._row_runs(mask[y])
            if not runs:
                continue
            if len(runs) >= 2:
                # Outermost runs are the rails; anything between them is the floor.
                a, b = runs[0], runs[-1]
                left, right = 0.5 * (a[0] + a[1]), 0.5 * (b[0] + b[1])
            else:
                a = runs[0]
                if a[1] - a[0] + 1 < wide:
                    # ONE THIN RUN. Ambiguous: either a single rail (the other is out of
                    # frame or unlit -- unusable alone, no centre can be derived), or the
                    # two rails MERGED because the path is far enough away that they sit
                    # within rail_merge_gap_px of each other.
                    # MEASURED on filt29 leg 1 (g1->g2): 86.2% of ticks yielded ZERO rows
                    # while area_frac stayed at 0.006, i.e. the path was plainly in frame
                    # and every row was being discarded here. Reproduced synthetically: a
                    # path whose rails close to 8 px apart gives 0 rows at area 0.011.
                    # With recovery on, the run's CENTRE is taken as the path centre and
                    # the row contributes to steering but not to a separation estimate.
                    if not c.rail_merge_recover:
                        continue
                    mid = 0.5 * (a[0] + a[1])
                    left = right = float(mid)
                    merged += 1
                else:
                    left, right = float(a[0]), float(a[1])   # merged corridor -> edges
            ys.append(y)
            xl.append(left)
            xr.append(right)
        return (np.asarray(ys, float), np.asarray(xl, float), np.asarray(xr, float),
                merged)

    @staticmethod
    def _robust_quadfit(y, x, iters=5):
        """Quadratic x = f(y) with iterative outlier rejection at 3 MAD. Returns
        (coeffs, keep_mask, rms) or None if it cannot be fitted."""
        if y.size < 3:
            return None
        keep = np.ones(y.size, bool)
        coef = None
        for _ in range(iters):
            if keep.sum() < 3:
                return None
            coef = np.polyfit(y[keep], x[keep], 2)
            res = np.abs(x - np.polyval(coef, y))
            # MAD -> sigma, floored at 1 px so a near-perfect fit does not reject
            # everything on floating-point dust.
            sigma = max(1.4826 * float(np.median(res[keep])), 1.0)
            new = res < 3.0 * sigma
            if new.sum() < 3 or np.array_equal(new, keep):
                break
            keep = new
        rms = float(np.sqrt(np.mean((x[keep] - np.polyval(coef, y[keep])) ** 2)))
        return coef, keep, rms

    def measure(self, frame_rgb: np.ndarray) -> TubeMeasurement:
        c = self.cfg
        mask = self._mask(frame_rgb)
        H, W = mask.shape
        area_frac = float(mask.sum()) / mask.size      # REPORTED, never a gate
        ys, xl, xr, n_merged = self._rail_points(mask)
        if ys.size < c.rail_min_rows:
            return TubeMeasurement(found=False, area_frac=area_frac, n_rows=int(ys.size),
                                   n_merged=int(n_merged))
        fl = self._robust_quadfit(ys, xl)
        fr = self._robust_quadfit(ys, xr)
        if fl is None or fr is None:
            return TubeMeasurement(found=False, area_frac=area_frac, n_rows=int(ys.size),
                                   n_merged=int(n_merged))
        cl, keep_l, rms_l = fl
        cr, keep_r, rms_r = fr
        rms = max(rms_l, rms_r)

        def norm_x(px):
            return (px - W / 2.0) / (W / 2.0)

        y_far = c.rail_curv_row * H
        # VANISHING POINT: where the two fitted rails meet. Their difference is a
        # quadratic in y; take the real root nearest the top of the frame, which is
        # the convergence ahead rather than the spurious one far below.
        # Resolved BEFORE the look-ahead row, because the look-ahead has to sit below it.
        v_px, converge_ok = 0.0, False
        diff = np.asarray(cl, float) - np.asarray(cr, float)
        roots = np.roots(diff) if np.any(np.abs(diff[:2]) > 1e-12) else np.array([])
        real = [float(r.real) for r in np.atleast_1d(roots) if abs(r.imag) < 1e-6]
        # Accept only a convergence at or above the scanned band (the path runs AWAY
        # from us), within one frame height of the top -- anything else is the fit's
        # far branch, not the vanishing point.
        cand = [r for r in real if -H <= r <= c.rail_band[1] * H]
        if cand:
            v_px, converge_ok = max(cand), True
        else:
            v_px = float(min(ys))     # fall back to the highest row a rail was seen on

        # --- ADAPTIVE LOOK-AHEAD ------------------------------------------------------
        # The rails DIVERGE downward from the vanishing point, so a measurement row is
        # only meaningful BELOW it. As the nose rises the path sinks in frame and the
        # convergence row slides down past a fixed look-ahead; at that point the two
        # fitted curves have already CROSSED at the measurement row and the separation
        # goes NEGATIVE, which fails the lock test even though the fit is perfect.
        # MEASURED on vision_frame.png rolled down 150 px (the path sinking as the nose
        # relaxes): rows 50 and rms 0.40 -- both healthy -- but sep -0.050, no lock. It
        # is not the search band that runs out (that already reaches the bottom row);
        # it is the look-ahead ending up on the wrong side of the convergence.
        # So the look-ahead is pushed to stay at least `rail_lookahead_below_conv` of a
        # frame BELOW the convergence, which restores the lock (sep +0.108 at 0.85H) and
        # also measures the centre where the rails are well separated rather than nearly
        # coincident. Set the margin to 0 to pin the look-ahead at its fixed row.
        y_look = c.rail_lookahead * H
        if c.rail_lookahead_below_conv > 0.0:
            y_look = max(y_look, v_px + c.rail_lookahead_below_conv * H)
            y_look = min(y_look, H - 1.0)
        xl_look, xr_look = float(np.polyval(cl, y_look)), float(np.polyval(cr, y_look))
        sep = (xr_look - xl_look) / W
        u_lower = norm_x(0.5 * (xl_look + xr_look))
        u_upper = norm_x(0.5 * (float(np.polyval(cl, y_far))
                                + float(np.polyval(cr, y_far))))
        lock = (rms <= c.rail_max_fit_rms
                and sep >= c.rail_min_sep_frac
                and int(keep_l.sum()) >= c.rail_min_rows
                and int(keep_r.sum()) >= c.rail_min_rows)
        return TubeMeasurement(
            found=bool(lock), u_tube=u_lower, curvature=(u_upper - u_lower),
            u_lower=u_lower, u_upper=u_upper, area_frac=area_frac,
            v_converge=(v_px - H / 2.0) / (H / 2.0), converge_ok=converge_ok,
            rail_sep=float(sep), n_rows=int(min(keep_l.sum(), keep_r.sum())),
            fit_rms=rms, y_look=float(y_look), n_merged=int(n_merged),
            left_poly=tuple(float(v) for v in cl),
            right_poly=tuple(float(v) for v in cr))


def draw_rail_overlay(frame_rgb, m, cfg):
    """Render a TubeMeasurement back onto its frame: the two fitted rail curves, the
    measured path centre vs frame centre at the look-ahead row, the convergence row, and
    a lock/no-lock status band.

    THE POINT: the numbers cannot tell you whether the fit is on the real rails or on
    some bright clutter that happens to fit a parabola. Drawing it back is the check.

    Pure numpy -- no PIL, no fonts -- so the flight process can call it on a worker
    thread without pulling an image library into the control path. Returns a new array;
    the input is never modified.

    Colour key:
      green / red band across the top   LOCK / no lock
      green curves                      the two fitted rails (red when not locked, i.e.
                                        what was rejected and why it looked wrong)
      magenta tick                      measured path centre at the look-ahead row
      white tick                        frame centre -- the gap between them IS u_tube
      cyan dashes                       convergence (vanishing point) row
    """
    img = np.array(frame_rgb, dtype=np.uint8, copy=True)
    H, W = img.shape[:2]
    lock = bool(getattr(m, "found", False))
    rail_c = (0, 255, 0) if lock else (255, 60, 60)

    def put(x, y, colour, r=1):
        x, y = int(round(x)), int(round(y))
        if 0 <= x < W and 0 <= y < H:
            img[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = colour

    img[0:6, :] = rail_c                      # status band: visible at a glance
    if getattr(m, "left_poly", ()) and getattr(m, "right_poly", ()):
        y0, y1 = int(cfg.rail_band[0] * H), int(cfg.rail_band[1] * H)
        for y in range(y0, y1):
            put(np.polyval(m.left_poly, y), y, rail_c)
            put(np.polyval(m.right_poly, y), y, rail_c)
        y_look = cfg.rail_lookahead * H
        xc = 0.5 * (np.polyval(m.left_poly, y_look) + np.polyval(m.right_poly, y_look))
        for dy in range(-7, 8):               # measured centre vs frame centre
            put(xc, y_look + dy, (255, 0, 255), 1)
            put(W / 2.0, y_look + dy, (255, 255, 255), 0)
        yv = m.v_converge * (H / 2.0) + H / 2.0
        for x in range(0, W, 8):
            put(x, yv, (0, 255, 255), 0)
    return img


def gravity_to_roll_pitch(gravity_frd):
    """gravity-down FRD [gx,gy,gz] -> (roll, pitch) rad. Level -> (0,0)."""
    gx, gy, gz = float(gravity_frd[0]), float(gravity_frd[1]), float(gravity_frd[2])
    roll = math.atan2(gy, gz)
    pitch = math.atan2(-gx, gz)
    return roll, pitch


def attitude_rates(cfg, des_roll, des_pitch, des_yaw_rate_norm, gravity_frd, gyro):
    """PD from measured (gravity-derived) roll/pitch to normalised body rates.
    des_* angles in rad; des_yaw_rate_norm already normalised [-1,1]. Shared by
    VisionServoController and HybridController."""
    roll, pitch = gravity_to_roll_pitch(gravity_frd)
    p, q, r = float(gyro[0]), float(gyro[1]), float(gyro[2])

    roll_rate = cfg.kp_att * (des_roll - roll) - cfg.kd_att * p       # rad/s
    pitch_rate = cfg.kp_att * (des_pitch - pitch) - cfg.kd_att * q    # rad/s
    yaw_rate = des_yaw_rate_norm * MAX_BODY_RATE - cfg.kd_yaw * r     # rad/s

    # Clamp the COMMANDED rate to max_cmd_rate_rad_s (normalised limit).
    lim = cfg.max_cmd_rate_rad_s / MAX_BODY_RATE
    roll_n = float(np.clip(roll_rate / MAX_BODY_RATE, -lim, lim))
    pitch_n = float(np.clip(pitch_rate / MAX_BODY_RATE, -lim, lim))
    yaw_n = float(np.clip(yaw_rate / MAX_BODY_RATE, -lim, lim))
    return roll_n, pitch_n, yaw_n


def _warn_cruise_pitch(cfg):
    """Loud guard (flight-5 class error): cruise_pitch far from the spawn attitude
    makes the attitude loop hold a constant pitch correction. Shared by both
    controllers."""
    dp = cfg.cruise_pitch_deg - SPAWN_PITCH_DEG
    if abs(dp) > CRUISE_PITCH_TOL_DEG:
        logger.warning(
            "[servo] *** cruise_pitch_deg=%.1f is %+.1f deg from the ~%.1f deg SPAWN "
            "attitude -- the attitude loop will hold a CONSTANT ~%+.1f deg pitch "
            "correction (climb + back up, NOT fly the course). Set cruise_pitch_deg "
            "near %.1f. ***", cfg.cruise_pitch_deg, dp, SPAWN_PITCH_DEG, dp, SPAWN_PITCH_DEG)


_COURSE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "course_gates_cm.json")


class CourseSchedule:
    """Per-segment glide feedforward from the known course (course_gates_cm.json).

    The course descends as a STAIRCASE (level in, ~-12..-17 deg middle, ~-2 deg
    out). For the segment toward the current target gate (index = active_gate),
    returns the feedforward THRUST that produces that segment's descent, via the
    THRUST_SINK_MAP lookup: needed_sink = cruise_speed * tan(descent_angle).

    Slopes are computed from the gate geometry; only the thrust<->sink numbers are
    a calibration (THRUST_SINK_MAP), so refining them never touches this class.
    """

    def __init__(self, cruise_speed_mps=10.0, path=_COURSE_PATH):
        self.cruise_speed = float(cruise_speed_mps)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        spawn = np.asarray(data["spawn"]["pos_cm"], float) / 100.0
        gates = [np.asarray(g["pos_cm"], float) / 100.0 for g in data["gates"]]
        pts = [spawn] + gates
        # Per-segment geometry. Segment i = pts[i]->pts[i+1] ENDS at gate i, so
        # active_gate=i selects index i. slope<0 = descending.
        self.slopes = []      # rad, signed
        self.lengths = []     # m, 3D segment length (for the dead-reckon drift check)
        self.vdrops = []      # m, vertical drop (+ = descends)
        self.bearings = []    # deg, UE-XY heading of the segment (reference only)
        for i in range(len(pts) - 1):
            d = pts[i + 1] - pts[i]
            horiz = float(math.hypot(d[0], d[1]))
            self.slopes.append(math.atan2(float(d[2]), horiz) if horiz > 1e-6 else 0.0)
            self.lengths.append(float(np.linalg.norm(d)))
            self.vdrops.append(-float(d[2]))
            self.bearings.append(math.degrees(math.atan2(float(d[1]), float(d[0]))))
        self.n_segments = len(self.slopes)

    def _idx(self, active_gate):
        return int(np.clip(active_gate if active_gate is not None else 0,
                           0, self.n_segments - 1))

    def segment_slope_deg(self, active_gate):
        return math.degrees(self.slopes[self._idx(active_gate)])

    def segment_length(self, active_gate):
        return self.lengths[self._idx(active_gate)]

    def segment_vdrop(self, active_gate):
        return self.vdrops[self._idx(active_gate)]

    def bearing_deg(self, active_gate):
        return self.bearings[self._idx(active_gate)]

    def bearing_change_deg(self, active_gate):
        """Heading change from the previous segment into this one (the only
        commandable open-loop 'bearing' cue -- there is NO absolute heading
        reference, so this is a turn feedforward at gate transitions, ~0 on this
        near-straight course)."""
        i = self._idx(active_gate)
        if i == 0:
            return 0.0
        d = self.bearings[i] - self.bearings[i - 1]
        return (d + 180.0) % 360.0 - 180.0

    def ff_thrust(self, active_gate):
        """Feedforward thrust for the current segment (abs thrust fraction)."""
        slope = self.slopes[self._idx(active_gate)]           # rad, neg = descending
        needed_sink = self.cruise_speed * math.tan(-slope)    # + = descend
        return thrust_for_sink(needed_sink)


class VisionServoController:
    """Hand-written outer (vision) + inner (IMU attitude) controller.

    Public API mirrors the model command source used by the adapter:
        command(frame_rgb, telemetry, active_gate) -> {throttle,roll,pitch,yaw}

    `detect()` is exposed for the offline dry-run harness / calibration tool.
    """

    def __init__(self, cfg: ServoConfig | None = None):
        self.cfg = cfg or ServoConfig()
        _warn_cruise_pitch(self.cfg)
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
        return attitude_rates(self.cfg, des_roll, des_pitch, des_yaw_rate_norm,
                              gravity_frd, gyro)

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

        # LATERAL frame sign (see LATERAL_SIGN at module top -- the DGX confirmed the
        # camera is NOT mirrored; this is the sim's left-positive roll convention,
        # which is why pitch works and only the lateral channel needs a sign).
        # u_ctrl is the roll/yaw SETPOINT error in the sim's frame: right gate
        # (det.u_err>0) -> u_ctrl<0 -> negative roll setpoint -> right translation.
        # Continuity/gating stay in IMAGE coords (det.u_err) -- they compare blob
        # positions frame-to-frame and are correct regardless of the sign.
        u_ctrl = LATERAL_SIGN * det.u_err

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
            # (the sim's lateral sign is already in u_ctrl via LATERAL_SIGN).
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
                # yaw toward where the gate physically was (LATERAL_SIGN on last image-u)
                des_yaw = math.copysign(c.search_yaw, (LATERAL_SIGN * self._last_u_err) or 1.0)
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


def _kin_accel_frd(accel_frd, gravity_frd):
    """Kinematic acceleration in FRD = specific force + gravity-down. Rest -> 0."""
    return np.asarray(accel_frd, float) + np.asarray(gravity_frd, float)


class DeadReckoner:
    """PASSIVE IMU dead-reckoning -- DIAGNOSTIC ONLY, never in the control loop.

    Integrates kinematic accel -> velocity -> along-track distance + altitude
    drop, and is reset to ground truth on each active_gate advance (the sim's
    gate-pass is truth; integration only estimates 'how far into this segment am
    I'). We LOG the integrated distance vs the segment's KNOWN length so the
    grinder data shows how badly it drifts -- confirming active_gate must lead and
    integration can only ever be a within-segment fallback.

    Drift is expected: with no heading reference and a -17.8 deg tilt, gravity
    leaks into horizontal accel if the gravity estimate is even slightly off, and
    double integration compounds it. That is exactly why this never drives
    control -- it is here to be measured, not trusted.
    """

    def __init__(self):
        self.reset_segment()
        self.last_t = None
        self.last_completed_drift = 0.0   # integrated - known length, at last advance

    def reset_segment(self):
        self.fwd_vel = 0.0
        self.fwd_dist = 0.0
        self.vsink_vel = 0.0
        self.alt_drop = 0.0

    def on_gate_advance(self, completed_segment_length):
        self.last_completed_drift = self.fwd_dist - completed_segment_length
        self.reset_segment()              # ground-truth reset for the new segment

    def update(self, accel_frd, gravity_frd, now):
        a = _kin_accel_frd(accel_frd, gravity_frd)          # FRD kinematic accel
        g = np.asarray(gravity_frd, float)
        gmag = float(np.linalg.norm(g))
        up = -g / gmag if gmag > 1e-6 else np.array([0.0, 0.0, -1.0])
        fwd_a = float(a[0])                                 # body-forward (FRD x)
        vsink_a = -float(a @ up)                            # + = downward
        if self.last_t is not None:
            dt = min(max(now - self.last_t, 1e-4), 0.2)
            self.fwd_vel += fwd_a * dt
            self.fwd_dist += self.fwd_vel * dt
            self.vsink_vel += vsink_a * dt
            self.alt_drop += self.vsink_vel * dt
        self.last_t = now
        return self.fwd_dist, self.alt_drop


class HybridController:
    """Three-tier hybrid controller. Glide feedforward from the MEASURED
    THRUST_SINK_MAP; active_gate drives the segment schedule (NO dead-reckoning in
    the control path -- a passive DeadReckoner logs drift for diagnostics only).

    Authority, highest first (see the blend law):
      GATE  -- detected big + centred (w_gate ramps with size)  -> precise u,v trim
      TUBE  -- visible, continuous lane                         -> lateral u trim
      FEEDFORWARD -- known-path glide (thrust) + heading hold   -> backbone

        u_ref  = w_gate*gate_u + (1-w_gate)*w_tube*tube_u    (LATERAL_SIGN, not mirror)
        thrust = CourseSchedule.ff_thrust(active_gate) - w_gate*(k_thrust_v*gate_v + kd_v*dv)

    Gate detection, the lateral-sign convention, the attitude inner loop, the clamp, and
    the filtered PD derivative are reused from the vision servo. Same command
    contract: command(frame, telemetry, active_gate) -> {throttle,roll,pitch,yaw}.
    """

    def __init__(self, cfg: ServoConfig | None = None, cruise_speed_mps=None):
        self.cfg = cfg or ServoConfig()
        _warn_cruise_pitch(self.cfg)
        self.detector = GateDetector(self.cfg)
        self.tube = TubeDetector(self.cfg)
        self.schedule = CourseSchedule(cruise_speed_mps or self.cfg.cruise_speed_mps)
        self.dr = DeadReckoner()          # passive diagnostic (not in control)
        self._u_filt = 0.0
        self._v_filt = 0.0
        self._last_t = None
        self._sched_gate = 0              # internal schedule segment (active_gate LEADS it;
        self._sched_gate_t = None         # timeout only advances it, never retreats)
        self._last_log_t = -1e9

    def reset(self):
        self._u_filt = 0.0
        self._v_filt = 0.0
        self._last_t = None
        self.dr = DeadReckoner()
        self._sched_gate = 0
        self._sched_gate_t = None
        logger.info("[hybrid] reset for fresh race")

    def command(self, frame_rgb, telemetry, active_gate=None, now=None):
        c = self.cfg
        now = time.time() if now is None else now
        gravity = telemetry.get("gravity_frd", (0.0, 0.0, 9.81))
        gyro = telemetry.get("velocity", (0.0, 0.0, 0.0))

        if self._sched_gate_t is None:
            self._sched_gate_t = now

        # SEGMENT POINTER (Option A). active_gate = sim ground truth LEADS; the
        # internal _sched_gate only ADVANCES (never retreats, never early). A
        # ground-truth advance is adopted immediately; else a SAFETY TIMEOUT
        # advances one segment if active_gate has stalled past segment_time*mult
        # (covers a missed gate so the controller can't stall on segment N forever).
        prev_seg = self._sched_gate
        seg_time = (self.schedule.segment_length(self._sched_gate)
                    / max(c.cruise_speed_mps, 1e-3))
        adv_reason = None
        if active_gate is not None and active_gate > self._sched_gate:
            self._sched_gate = int(active_gate)
            adv_reason = "active_gate"
        elif ((now - self._sched_gate_t) > c.hybrid_seg_timeout_mult * seg_time
              and self._sched_gate < self.schedule.n_segments - 1):
            self._sched_gate += 1
            adv_reason = "TIMEOUT"
        if adv_reason is not None:
            self.dr.on_gate_advance(self.schedule.segment_length(prev_seg))
            self._u_filt = 0.0
            self._v_filt = 0.0
            self._sched_gate_t = now
            logger.info("[hybrid] segment %s -> %s (%s)  DR drift on last seg = %+.1f m",
                        prev_seg, self._sched_gate, adv_reason, self.dr.last_completed_drift)

        # Passive dead-reckoning (DIAGNOSTIC ONLY -- logged, never used for control).
        accel = telemetry.get("acceleration", (0.0, 0.0, 0.0))
        dr_fwd, dr_drop = self.dr.update(accel, gravity, now)
        seg = self._sched_gate                    # effective schedule segment
        seg_len = self.schedule.segment_length(seg)
        within_frac = dr_fwd / seg_len if seg_len > 1e-6 else 0.0   # integration = fraction only

        gate = self.detector.detect(frame_rgb)
        tube = self.tube.measure(frame_rgb)

        # --- confidence weights ---
        w_gate = 0.0
        if gate.found:
            w_gate = float(np.clip(
                (gate.size_frac - c.hybrid_gate_s_lo)
                / max(c.hybrid_gate_s_hi - c.hybrid_gate_s_lo, 1e-6), 0.0, 1.0))
        # tube.found IS the validity test now (both rails fitted). The old
        # `and area_frac >= hybrid_tube_area_min` conjunct is gone: area is the wrong
        # question about a thin bright curve -- a fully visible rail one pixel wide has
        # ~zero area -- so it rejected good locks. hybrid_tube_area_min is retained as a
        # config field only so existing tune files still load.
        w_tube = 1.0 if tube.found else 0.0

        # --- lateral reference (sim-frame roll setpoint; LATERAL_SIGN, NOT a mirror
        #     -- see module top): GATE -> TUBE -> hold ---
        gate_u = LATERAL_SIGN * gate.u_err if gate.found else 0.0
        # Tube lateral = near-band centring (u_lower, stay-ON-line) + anticipatory
        # curvature lead (where the line GOES). Near-band alone is ~0 when centred,
        # so it flies straight past an off-axis gate; curvature supplies the missing
        # lateral feedforward toward the next gate. (No absolute-heading reference on
        # v3385, so the visible corridor IS the heading source.)
        tube_u = (LATERAL_SIGN * (tube.u_tube + c.k_tube_lead * tube.curvature)
                  if tube.found else 0.0)
        u_ref = w_gate * gate_u + (1.0 - w_gate) * w_tube * tube_u

        # --- filtered PD derivative (dirty-derivative) on u_ref and gate v ---
        first = self._last_t is None
        dt = 0.0 if first else min(max(now - self._last_t, 1e-3), 0.5)
        a = 0.0 if first else dt / (c.deriv_tau_s + dt)
        if first:
            self._u_filt = u_ref
            self._v_filt = gate.v_err if gate.found else 0.0
            du = dv = 0.0
        else:
            self._u_filt += a * (u_ref - self._u_filt)
            du = (u_ref - self._u_filt) / c.deriv_tau_s
            if gate.found:
                self._v_filt += a * (gate.v_err - self._v_filt)
                dv = (gate.v_err - self._v_filt) / c.deriv_tau_s
            else:
                dv = 0.0
        self._last_t = now

        des_roll = float(np.clip(c.k_bank * u_ref + c.kd_u * du,
                                 -math.radians(c.max_bank_deg), math.radians(c.max_bank_deg)))
        des_yaw = float(np.clip(c.k_yaw * u_ref, -1.0, 1.0))
        des_pitch = math.radians(c.cruise_pitch_deg)

        # --- vertical: FEEDFORWARD glide dominates; vision v-error trims within a
        #     small band (+/- hybrid_v_trim_band) around the feedforward thrust. ---
        ff_thrust = self.schedule.ff_thrust(seg)
        if gate.found:
            v_trim = w_gate * (c.k_thrust_v * gate.v_err + c.kd_v * dv)
            v_trim = float(np.clip(v_trim, -c.hybrid_v_trim_band, c.hybrid_v_trim_band))
        else:
            v_trim = 0.0
        thrust = ff_thrust - v_trim
        # Hard safety net still applies (never collapse the flight).
        lo_t = max(c.min_thrust, c.hover_cruise - c.thrust_dev_max)
        hi_t = min(c.max_thrust, c.hover_cruise + c.thrust_dev_max)
        thrust = float(np.clip(thrust, lo_t, hi_t))

        roll_n, pitch_n, yaw_n = attitude_rates(c, des_roll, des_pitch, des_yaw, gravity, gyro)

        # --- authority split (log FEEDFORWARD vs TRIM for BOTH channels) ---
        # VERTICAL: ff_thrust (open-loop) vs trim_thrust (vision, +/- band).
        # LATERAL:  ff_yaw_rate = 0 (hold heading -- no absolute reference), so the
        #           whole lateral command is vision TRIM. bearing-change is a cue
        #           only. Logging both lets the histogram split SCHEDULE vs VISION.
        trim_thrust = -v_trim
        ff_yaw_rate = 0.0
        ff_bearing = self.schedule.bearing_change_deg(seg)

        if now - self._last_log_t >= 0.1:
            self._last_log_t = now
            logger.info(
                "[hybrid] g=%s seg=%d frac=%.2f wG=%.2f wT=%.2f | LAT uref=%+.3f "
                "gsz=%.3f uL=%+.3f crv=%+.3f "
                "ff_yaw=%+.2f trim(r=%+.2f y=%+.2f) ffB=%+.1fdeg | VERT ff=%.3f trim=%+.3f "
                "thr=%.3f | DR fwd=%.1f/%.1fm drop=%.1f/%.1fm",
                active_gate, seg, within_frac, w_gate, w_tube, u_ref,
                gate.size_frac, tube.u_lower, tube.curvature,
                ff_yaw_rate, roll_n * MAX_BODY_RATE, yaw_n * MAX_BODY_RATE, ff_bearing,
                ff_thrust, trim_thrust, thrust,
                dr_fwd, seg_len, dr_drop, self.schedule.segment_vdrop(seg))

        return {"throttle": thrust, "roll": roll_n, "pitch": pitch_n, "yaw": yaw_n,
                "_debug": {"active_gate": active_gate, "sched_gate": seg,
                           "within_frac": within_frac, "w_gate": w_gate, "w_tube": w_tube,
                           "u_ref": u_ref, "ff_thrust": ff_thrust, "trim_thrust": trim_thrust,
                           "ff_yaw_rate": ff_yaw_rate, "ff_bearing_change": ff_bearing,
                           "gate_found": gate.found, "gate_size": gate.size_frac,
                           "tube_found": tube.found, "tube_u_lower": tube.u_lower,
                           "tube_curvature": tube.curvature,
                           "des_roll": des_roll, "des_yaw": des_yaw,
                           "dr_fwd_dist": dr_fwd, "dr_alt_drop": dr_drop,
                           "seg_length": seg_len, "dr_last_drift": self.dr.last_completed_drift}}


# Knobs the semi-auto tuner (tools/tune_openloop.py) is allowed to override, so a
# tuning iteration never edits source. OpenLoopFlier loads them at construction and
# logs which it applied, so every flight log records the config it flew.
OPENLOOP_TUNE_KEYS = frozenset({
    "ol_thrust_lo", "ol_thrust_hi", "ol_kp_att", "ol_kd_att", "ol_max_rate_rad_s",
    "ol_max_bank_deg", "ol_fence_deg", "ol_gate_trim", "k_tube_lead", "cruise_pitch_deg",
})
_OPENLOOP_TUNE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "openloop_tune.json")


def apply_openloop_tune(cfg, path=None):
    """Override cfg's ol_* knobs from a JSON file, if present. Returns {applied}.

    Resolution: explicit `path` arg > $OPENLOOP_TUNE > repo-root openloop_tune.json.
    Only keys in OPENLOOP_TUNE_KEYS are honoured (unknown keys are ignored, logged)."""
    path = path or os.environ.get("OPENLOOP_TUNE") or _OPENLOOP_TUNE_PATH
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[openloop] tune file %s unreadable (%s) -- using defaults", path, exc)
        return {}
    applied = {}
    for k, v in (data or {}).items():
        if k in OPENLOOP_TUNE_KEYS:
            setattr(cfg, k, float(v))
            applied[k] = float(v)
        else:
            logger.warning("[openloop] tune key %r ignored (not a tunable knob)", k)
    return applied


class OpenLoopFlier:
    """Anti-flip hardcoded flier: the 0.40-probe trajectory + tube steering, MINUS
    the reactive vertical that flips every other controller.

    Design (see ServoConfig ol_* block):
      VERTICAL  -- OPEN-LOOP thrust from the measured sink schedule, hard-clamped to
                   [ol_thrust_lo, ol_thrust_hi]. NO vertical PID, NO thrust feedback.
                   Thrust is a pure function of the segment -- the runaway source is
                   simply gone.
      ATTITUDE  -- a GENTLE hold (ol_kp_att, ~1.0, vs the stiff kp_att=3.0 that winds
                   up) toward des_pitch=cruise / des_roll=steer, with EVERY commanded
                   body rate hard-clamped to +/-ol_max_rate_rad_s. A flip needs a
                   sustained high rate from a positive-feedback loop; there is no such
                   loop and the rate is clamped small, so a flip is structurally out.
      LATERAL   -- the ONLY closed loop: tube-curvature steering (u_lower centring +
                   k_tube_lead*curvature), with a gate-centring fine-trim when a gate
                   is big/close. Same LATERAL_SIGN convention as the others.
      FENCE     -- if the gravity-estimated tilt exceeds ol_fence_deg, steering is cut
                   and the hold recovers toward cruise/level. Belt-and-suspenders on
                   top of the rate clamp.

    Public API matches VisionServoController/HybridController: command(frame, telem,
    active_gate) -> {throttle, roll, pitch, yaw}; roll/pitch/yaw are NORMALISED body
    rates (1.0 == MAX_BODY_RATE), the adapter applies PLANT_RATE_CALIB.
    """

    def __init__(self, cfg: ServoConfig | None = None, cruise_speed_mps=None,
                 tune_path=None):
        self.cfg = cfg or ServoConfig()
        applied = apply_openloop_tune(self.cfg, tune_path)
        _warn_cruise_pitch(self.cfg)
        self.detector = GateDetector(self.cfg)
        self.tube = TubeDetector(self.cfg)
        self.schedule = CourseSchedule(cruise_speed_mps or self.cfg.cruise_speed_mps)
        self._last_log_t = -1e9
        if applied:
            logger.warning("[openloop] TUNE overrides applied: %s", applied)

    def reset(self):
        logger.info("[openloop] reset for fresh race")
        self._last_log_t = -1e9

    def command(self, frame_rgb, telemetry, active_gate=None, now=None):
        c = self.cfg
        now = time.time() if now is None else now
        gravity = telemetry.get("gravity_frd", (0.0, 0.0, 9.81))
        gyro = telemetry.get("velocity", (0.0, 0.0, 0.0))
        p, q, r = float(gyro[0]), float(gyro[1]), float(gyro[2])

        # --- VERTICAL: open-loop thrust from the sink schedule (NO feedback) ---
        seg = int(np.clip(active_gate if active_gate is not None else 0,
                          0, self.schedule.n_segments - 1))
        thrust = float(np.clip(self.schedule.ff_thrust(seg), c.ol_thrust_lo, c.ol_thrust_hi))

        # --- LATERAL: tube-curvature steering (the only closed loop) + gate fine-trim ---
        tube = self.tube.measure(frame_rgb)
        gate = self.detector.detect(frame_rgb)
        u_lat = (LATERAL_SIGN * (tube.u_tube + c.k_tube_lead * tube.curvature)
                 if tube.found else 0.0)
        if gate.found and gate.size_frac > c.hybrid_gate_s_lo:
            w = float(np.clip((gate.size_frac - c.hybrid_gate_s_lo)
                              / max(c.hybrid_gate_s_hi - c.hybrid_gate_s_lo, 1e-6), 0.0,
                              c.ol_gate_trim))
            u_lat = (1.0 - w) * u_lat + w * (LATERAL_SIGN * gate.u_err)

        des_roll = float(np.clip(c.k_bank * u_lat,
                                 -math.radians(c.ol_max_bank_deg),
                                 math.radians(c.ol_max_bank_deg)))
        des_pitch = math.radians(c.cruise_pitch_deg)
        des_yaw_norm = float(np.clip(c.k_yaw * u_lat, -1.0, 1.0))

        # --- ATTITUDE FENCE: past ol_fence_deg tilt, stop steering + recover ---
        est_roll, est_pitch = gravity_to_roll_pitch(gravity)
        fence = math.radians(c.ol_fence_deg)
        fenced = (abs(est_roll) > fence
                  or abs(est_pitch - des_pitch) > fence)
        if fenced:
            des_roll = 0.0          # level the wings, quit turning
            des_yaw_norm = 0.0      # (des_pitch stays at cruise: recover toward the glide)

        # --- GENTLE hold -> body rates, HARD-clamped small (no wind-up => no flip) ---
        roll_rate = c.ol_kp_att * (des_roll - est_roll) - c.ol_kd_att * p    # rad/s
        pitch_rate = c.ol_kp_att * (des_pitch - est_pitch) - c.ol_kd_att * q  # rad/s
        yaw_rate = des_yaw_norm * MAX_BODY_RATE - c.ol_kd_att * r            # rad/s
        lim = c.ol_max_rate_rad_s / MAX_BODY_RATE
        roll_n = float(np.clip(roll_rate / MAX_BODY_RATE, -lim, lim))
        pitch_n = float(np.clip(pitch_rate / MAX_BODY_RATE, -lim, lim))
        yaw_n = float(np.clip(yaw_rate / MAX_BODY_RATE, -lim, lim))

        if now - self._last_log_t >= 0.1:
            self._last_log_t = now
            logger.info(
                "[openloop] g=%s seg=%d ffthr=%.3f | LAT uLat=%+.3f uL=%+.3f crv=%+.3f "
                "gsz=%.3f desR=%+.1f | EST roll=%+.1f pitch=%+.1f %s | rate(r=%+.2f "
                "p=%+.2f y=%+.2f)rad/s",
                active_gate, seg, thrust, u_lat, tube.u_lower, tube.curvature,
                gate.size_frac, math.degrees(des_roll),
                math.degrees(est_roll), math.degrees(est_pitch), "FENCE" if fenced else "",
                roll_n * MAX_BODY_RATE, pitch_n * MAX_BODY_RATE, yaw_n * MAX_BODY_RATE)

        return {"throttle": thrust, "roll": roll_n, "pitch": pitch_n, "yaw": yaw_n,
                "_debug": {"active_gate": active_gate, "seg": seg, "ff_thrust": thrust,
                           "u_lat": u_lat, "tube_found": tube.found,
                           "tube_u_lower": tube.u_lower, "tube_curvature": tube.curvature,
                           "gate_found": gate.found, "gate_size": gate.size_frac,
                           "des_roll": des_roll, "des_pitch": des_pitch,
                           "est_roll": est_roll, "est_pitch": est_pitch, "fenced": fenced,
                           "roll_n": roll_n, "pitch_n": pitch_n, "yaw_n": yaw_n}}
