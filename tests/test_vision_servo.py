"""Logic tests for the hardcoded vision-servo controller (vq1_vision_servo.py).

These validate GEOMETRY and SIGN CONVENTIONS against synthetic frames — the
detector recovers the right centroid, and the guidance + inner attitude loop
push in the correct direction. They CANNOT validate tuning or real-gate colour
(no sim, no real frame here). Gains/thresholds are calibrated on the sim box;
these tests are the guardrail that a refactor doesn't flip a sign.

Run: pytest tests/test_vision_servo.py -v   (or: python tests/test_vision_servo.py)
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vq1_vision_servo import (  # noqa: E402
    ServoConfig, VisionServoController, GateDetector, TubeDetector,
    HybridController, CourseSchedule, DeadReckoner, thrust_for_sink, THRUST_SINK_MAP,
    gravity_to_roll_pitch, rgb_to_hsv_arrays,
)

# Bright RED gate, matching the calibrated red-wraparound band (hue ~0, high V).
RED_GATE = (230, 30, 30)
TUBE_CYAN = (30, 180, 220)   # hue ~193, in the tube band [180,215]
LEVEL_G = (0.0, 0.0, 9.81)
ZERO_GYRO = (0.0, 0.0, 0.0)


def add_square(frame, cx_frac, cy_frac, size_frac, colour=RED_GATE):
    h, w = frame.shape[0], frame.shape[1]
    half = int(size_frac * h / 2)
    cx, cy = int(cx_frac * w), int(cy_frac * h)
    frame[max(0, cy - half):cy + half, max(0, cx - half):cx + half] = colour
    return frame


def make_tube(x_lower_frac, x_upper_frac=None, halfwidth=40, w=640, h=360):
    """Vertical cyan stripe centred at x_lower_frac in the lower half and
    x_upper_frac in the upper half (default same) — for curvature tests."""
    if x_upper_frac is None:
        x_upper_frac = x_lower_frac
    frame = np.full((h, w, 3), 20, dtype=np.uint8)
    for y in range(h):
        frac = x_upper_frac if y < h // 2 else x_lower_frac
        cx = int(frac * w)
        frame[y, max(0, cx - halfwidth):cx + halfwidth] = TUBE_CYAN
    return frame


def make_frame(cx_frac, cy_frac, size_frac=0.18, colour=RED_GATE, w=640, h=360):
    """Dark frame with one solid coloured square centred at (cx_frac, cy_frac)."""
    frame = np.full((h, w, 3), 25, dtype=np.uint8)  # dark grey background
    half = int(size_frac * h / 2)
    cx, cy = int(cx_frac * w), int(cy_frac * h)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    frame[y0:y1, x0:x1] = colour
    return frame


def telem(gravity=LEVEL_G, gyro=ZERO_GYRO):
    return {"gravity_frd": gravity, "velocity": gyro}


# --------------------------------------------------------------------------
# HSV + detector geometry
# --------------------------------------------------------------------------
def test_hsv_red_gate_in_band():
    # Red gate hue sits at ~0/360 -> inside the wraparound band (>=340 or <=20).
    h, s, v = rgb_to_hsv_arrays(np.array([[RED_GATE]], dtype=np.uint8))
    hue = float(h[0, 0])
    assert hue >= 340.0 or hue <= 20.0
    assert float(s[0, 0]) > 0.30 and float(v[0, 0]) > 0.55


def test_detect_centered():
    det = GateDetector(ServoConfig()).detect(make_frame(0.5, 0.5))
    assert det.found
    assert abs(det.u_err) < 0.05 and abs(det.v_err) < 0.05


def test_detect_right_and_low():
    det = GateDetector(ServoConfig()).detect(make_frame(0.75, 0.70))
    assert det.found
    assert det.u_err > 0.3   # right of centre
    assert det.v_err > 0.2   # below centre


def test_detect_left_and_high():
    det = GateDetector(ServoConfig()).detect(make_frame(0.25, 0.30))
    assert det.found
    assert det.u_err < -0.3
    assert det.v_err < -0.2


def test_detect_none_on_dark_frame():
    frame = np.full((360, 640, 3), 25, dtype=np.uint8)
    assert not GateDetector(ServoConfig()).detect(frame).found


# --- largest-blob (nearest-gate) selection ---
def test_largest_blob_picks_nearest_not_blend():
    # Big near gate on the LEFT + small far gate on the RIGHT. The all-red
    # centroid would blend toward center; largest-blob must lock the big one.
    frame = make_frame(0.32, 0.45, size_frac=0.28)      # big, left
    add_square(frame, 0.70, 0.72, size_frac=0.05)       # small, right
    d = GateDetector(ServoConfig()).detect(frame)
    assert d.found
    assert d.u_err < -0.12                               # locked on the LEFT big gate
    assert abs(d.u_err - (0.32 - 0.5) * 2) < 0.12        # near the big gate's true u


def test_dilation_bridges_hollow_square():
    # A hollow RED square (ring) must detect as ONE blob centred on it, not fail.
    h, w = 360, 640
    frame = np.full((h, w, 3), 20, dtype=np.uint8)
    cx, cy, s = 320, 180, 40
    frame[cy - s:cy + s, cx - s:cx - s + 3] = RED_GATE   # left edge
    frame[cy - s:cy + s, cx + s - 3:cx + s] = RED_GATE   # right edge
    frame[cy - s:cy - s + 3, cx - s:cx + s] = RED_GATE   # top edge
    frame[cy + s - 3:cy + s, cx - s:cx + s] = RED_GATE   # bottom edge
    d = GateDetector(ServoConfig()).detect(frame)
    assert d.found and abs(d.u_err) < 0.05 and abs(d.v_err) < 0.05


def test_mask_blowup_rejected():
    # A frame flooded with red (mask > max_mask_frac_reject) is not a gate.
    frame = np.full((360, 640, 3), 0, dtype=np.uint8)
    frame[..., 0] = 230
    assert not GateDetector(ServoConfig()).detect(frame).found


# --- tube measurement (no guidance wiring) ---
def test_tube_center_straight():
    m = TubeDetector(ServoConfig()).measure(make_tube(0.5))
    assert m.found and abs(m.u_tube) < 0.05 and abs(m.curvature) < 0.05


def test_tube_bend_right():
    # upper band shifted RIGHT of lower -> curvature > 0 (course bends right ahead)
    m = TubeDetector(ServoConfig()).measure(make_tube(0.5, x_upper_frac=0.68))
    assert m.found and m.curvature > 0.1


def test_tube_bend_left():
    m = TubeDetector(ServoConfig()).measure(make_tube(0.5, x_upper_frac=0.32))
    assert m.found and m.curvature < -0.1


def test_tube_absent():
    dark = np.full((360, 640, 3), 20, dtype=np.uint8)
    assert not TubeDetector(ServoConfig()).measure(dark).found


def test_size_grows_with_gate():
    small = GateDetector(ServoConfig()).detect(make_frame(0.5, 0.5, size_frac=0.10))
    big = GateDetector(ServoConfig()).detect(make_frame(0.5, 0.5, size_frac=0.40))
    assert big.size_frac > small.size_frac


# --------------------------------------------------------------------------
# angle extraction
# --------------------------------------------------------------------------
def test_gravity_level_is_zero():
    roll, pitch = gravity_to_roll_pitch(LEVEL_G)
    assert abs(roll) < 1e-6 and abs(pitch) < 1e-6


def test_gravity_reproduces_rest_pitch():
    # measured rest accel [-2.999,-0.003,-9.340] => gravity-down = -accel
    g = (2.999, 0.003, 9.340)
    _, pitch = gravity_to_roll_pitch(g)
    assert math.degrees(pitch) == __import__("pytest").approx(-17.8, abs=0.3)


def test_gravity_rolled_right_positive():
    phi = math.radians(20.0)
    g = (0.0, 9.81 * math.sin(phi), 9.81 * math.cos(phi))
    roll, _ = gravity_to_roll_pitch(g)
    assert math.degrees(roll) == __import__("pytest").approx(20.0, abs=0.5)


# --------------------------------------------------------------------------
# guidance signs (outer loop)
# --------------------------------------------------------------------------
def test_lateral_convention_locked():
    # LOCK the lateral sign (cost five flights). Camera is NOT mirrored (DGX-
    # confirmed), so image-u == physical-u. The sign is the sim's LEFT-positive
    # roll: des_roll>0 translates LEFT. So a gate physically LEFT (u_err<0) needs
    # des_roll>0 (left); physically RIGHT (u_err>0) needs des_roll<0 (right).
    # u_ctrl = LATERAL_SIGN * u_err.
    # image-LEFT gate (u_err<0) -> u_ctrl>0 -> des_roll>0 (left translation, toward it):
    left_gate = VisionServoController().command(make_frame(0.20, 0.5), telem(), active_gate=0)
    assert left_gate["_debug"]["u_ctrl"] > 0
    assert left_gate["_debug"]["des_roll"] > 0.0
    assert left_gate["yaw"] > 0.0
    # image-RIGHT gate (u_err>0) -> u_ctrl<0 -> des_roll<0 (right translation):
    right_gate = VisionServoController().command(make_frame(0.80, 0.5), telem(), active_gate=0)
    assert right_gate["_debug"]["u_ctrl"] < 0
    assert right_gate["_debug"]["des_roll"] < 0.0
    assert right_gate["yaw"] < 0.0


def test_gate_low_reduces_thrust():
    ctl = VisionServoController()
    centred = ctl.command(make_frame(0.5, 0.5), telem(), active_gate=0)["throttle"]
    low = VisionServoController().command(
        make_frame(0.5, 0.85), telem(), active_gate=0)["throttle"]
    assert low < centred   # gate low => too high => descend


def test_gate_high_increases_thrust():
    ctl = VisionServoController()
    centred = ctl.command(make_frame(0.5, 0.5), telem(), active_gate=0)["throttle"]
    high = VisionServoController().command(
        make_frame(0.5, 0.15), telem(), active_gate=0)["throttle"]
    assert high > centred


def test_cruise_commands_nose_down_when_level():
    # Level body, desired forward cruise lean is nose-down => negative pitch rate.
    cmd = VisionServoController().command(make_frame(0.5, 0.5), telem(), active_gate=0)
    assert cmd["pitch"] < 0.0


# --------------------------------------------------------------------------
# inner attitude loop signs
# --------------------------------------------------------------------------
def test_rolled_right_corrects_left():
    # Body rolled right (gy>0), gate centred (des_roll~0) => command roll-left (neg).
    phi = math.radians(25.0)
    g = (0.0, 9.81 * math.sin(phi), 9.81 * math.cos(phi))
    cmd = VisionServoController().command(make_frame(0.5, 0.5), telem(gravity=g), active_gate=0)
    assert cmd["roll"] < 0.0


def test_yaw_rate_damped_by_gyro():
    # Centred gate but spinning fast in yaw => damping opposes the spin.
    fast_r = (0.0, 0.0, 5.0)  # r = +5 rad/s
    cmd = VisionServoController().command(
        make_frame(0.5, 0.5), telem(gyro=fast_r), active_gate=0)
    assert cmd["yaw"] < 0.0   # damping term -kd_yaw*r dominates a centred gate


# --------------------------------------------------------------------------
# lost-gate behaviour
# --------------------------------------------------------------------------
def test_lost_gate_coasts_then_searches():
    ctl = VisionServoController()
    # Gate seen at IMAGE-right (make_frame 0.80) = physically right; sim's left-
    # positive roll means the search yaws toward it via LATERAL_SIGN. Lose it.
    ctl.command(make_frame(0.80, 0.5), telem(), active_gate=0, now=100.0)
    dark = np.full((360, 640, 3), 25, dtype=np.uint8)
    coast = ctl.command(dark, telem(), active_gate=0, now=100.2)   # within reacquire
    assert abs(coast["_debug"]["des_yaw"]) < 1e-6
    search = ctl.command(dark, telem(), active_gate=0, now=101.5)  # past reacquire
    assert search["_debug"]["des_yaw"] < 0.0   # search toward PHYSICAL last-seen (left)
    assert not search["_debug"]["found"]


# --- anti-chase gating (first-flight fix) ---
def test_rejects_small_gate_while_tracking():
    ctl = VisionServoController()
    ctl.command(make_frame(0.5, 0.5, size_frac=0.20), telem(), active_gate=0, now=100.0)
    # a small distant gate right after (detected, but size < gate_size_reject 0.025)
    # -> rejected as "small", servo coasts
    out = ctl.command(make_frame(0.5, 0.5, size_frac=0.024), telem(), active_gate=0, now=100.1)
    assert out["_debug"]["found"] and not out["_debug"]["accepted"]
    assert out["_debug"]["reject"] == "small"


def test_size_schmitt_no_thrash_at_boundary():
    # Sizes hovering near the old 0.04 edge must NOT flip accept/reject once locked.
    ctl = VisionServoController()
    ctl.command(make_frame(0.5, 0.5, size_frac=0.06), telem(), active_gate=0, now=100.0)  # lock
    for i, sz in enumerate([0.030, 0.041, 0.030, 0.041], start=1):
        out = ctl.command(make_frame(0.5, 0.5, size_frac=sz), telem(),
                          active_gate=0, now=100.0 + i * 0.05)
        assert out["_debug"]["accepted"], f"thrashed at size {sz}"


def test_thrust_clamped_to_hover_band():
    cfg = ServoConfig()
    # extreme low gate would drive thrust way down; clamp holds it within +/- dev
    out = VisionServoController().command(
        make_frame(0.5, 0.98, size_frac=0.16), telem(), active_gate=0)
    assert out["throttle"] >= cfg.hover_cruise - cfg.thrust_dev_max - 1e-9
    out2 = VisionServoController().command(
        make_frame(0.5, 0.02, size_frac=0.16), telem(), active_gate=0)
    assert out2["throttle"] <= cfg.hover_cruise + cfg.thrust_dev_max + 1e-9


def test_rejects_teleport_jump_while_tracking():
    ctl = VisionServoController()
    ctl.command(make_frame(0.25, 0.5, size_frac=0.20), telem(), active_gate=0, now=100.0)
    # a big gate suddenly on the far side (|du| ~ 0.9 > max_u_jump) -> "jump"
    out = ctl.command(make_frame(0.80, 0.5, size_frac=0.20), telem(), active_gate=0, now=100.1)
    assert out["_debug"]["found"] and not out["_debug"]["accepted"]
    assert out["_debug"]["reject"] == "jump"


def test_small_gate_accepted_after_reacquire_window():
    ctl = VisionServoController()
    ctl.command(make_frame(0.5, 0.5, size_frac=0.20), telem(), active_gate=0, now=100.0)
    # gateless past reacquire_s -> gating dropped, small gate re-locks
    out = ctl.command(make_frame(0.5, 0.5, size_frac=0.03), telem(), active_gate=0, now=102.0)
    assert out["_debug"]["accepted"]


def test_continuity_prefers_last_tracked_blob():
    ctl = VisionServoController()
    # lock a gate on the LEFT
    ctl.command(make_frame(0.30, 0.5, size_frac=0.16), telem(), active_gate=0, now=100.0)
    # now two gates: left (near last pos) + a slightly bigger one on the right
    frame = make_frame(0.30, 0.5, size_frac=0.16)
    add_square(frame, 0.72, 0.5, size_frac=0.18)
    out = ctl.command(frame, telem(), active_gate=0, now=100.1)
    # continuity keeps the LEFT gate despite the right one being larger
    assert out["_debug"]["accepted"] and out["_debug"]["u_err"] < -0.1


# --- PD derivative on the vision error ---
def test_derivative_adds_bank_on_growing_u():
    ctl = VisionServoController()
    ctl.command(make_frame(0.50, 0.5), telem(), active_gate=0, now=100.0)
    out = None
    for i, cx in enumerate([0.53, 0.56, 0.59, 0.62], start=1):
        out = ctl.command(make_frame(cx, 0.5), telem(), active_gate=0, now=100.0 + i * 0.1)
    cfg = ServoConfig()
    u_ctrl = out["_debug"]["u_ctrl"]
    p_only = cfg.k_bank * u_ctrl
    # a growing offset (du/dt != 0) pushes bank BEYOND the proportional magnitude,
    # in the same direction.
    assert abs(out["_debug"]["des_roll"]) > abs(p_only) + 1e-3
    assert out["_debug"]["des_roll"] * p_only > 0


def test_derivative_zero_when_u_constant():
    ctl = VisionServoController()
    out = None
    for i in range(5):
        out = ctl.command(make_frame(0.58, 0.5), telem(), active_gate=0, now=100.0 + i * 0.1)
    cfg = ServoConfig()
    u_ctrl = out["_debug"]["u_ctrl"]
    p_only = cfg.k_bank * u_ctrl
    # steady u -> derivative decays to ~0 -> des_roll ~ proportional term
    assert abs(out["_debug"]["des_roll"] - p_only) < 0.03


# --- search timeout (anti-corkscrew) ---
def test_search_timeout_levels_and_holds():
    cfg = ServoConfig()
    ctl = VisionServoController()
    ctl.command(make_frame(0.80, 0.5), telem(), active_gate=0, now=100.0)  # lock right gate
    dark = np.full((360, 640, 3), 20, dtype=np.uint8)
    searching = ctl.command(dark, telem(), active_gate=0, now=101.5)   # within search window
    assert abs(searching["_debug"]["des_yaw"]) > 0.0
    giveup = ctl.command(dark, telem(), active_gate=0, now=103.0)      # past search_timeout_s
    assert giveup["_debug"]["des_yaw"] == 0.0
    assert giveup["_debug"]["des_pitch"] == 0.0                        # level pitch (hold)
    assert abs(giveup["throttle"] - cfg.hover_cruise) < 1e-9           # hover, not below


# --- cruise_pitch vs spawn attitude (flight-5 guard) ---
def test_cruise_pitch_matches_spawn_quiet_loop():
    # cruise_pitch -18 ~ spawn -17.8 -> near-zero pitch command (loop sits quiet),
    # NOT the persistent +0.5 rad/s nose-up flight 5 saw at cruise_pitch -8.
    g = (2.999, 0.003, 9.340)  # ~ -17.8 deg pitch
    cmd = VisionServoController().command(make_frame(0.5, 0.5), telem(gravity=g), active_gate=0)
    assert abs(cmd["pitch"] * 12.0) < 0.15


def test_cruise_pitch_guard_warns_when_far_from_spawn(caplog):
    import logging
    cfg = ServoConfig()
    cfg.cruise_pitch_deg = -8.0
    with caplog.at_level(logging.WARNING):
        VisionServoController(cfg)
    assert any("cruise_pitch" in r.message and "SPAWN" in r.message for r in caplog.records)


def test_cruise_pitch_guard_silent_on_default(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        VisionServoController()  # default -18 ~ spawn -17.8
    assert not any("cruise_pitch" in r.message for r in caplog.records)


def test_command_contract_keys_and_ranges():
    cmd = VisionServoController().command(make_frame(0.6, 0.4), telem(), active_gate=1)
    for k in ("throttle", "roll", "pitch", "yaw"):
        assert k in cmd
    assert 0.0 <= cmd["throttle"] <= 1.0
    for k in ("roll", "pitch", "yaw"):
        assert -1.0 <= cmd[k] <= 1.0


# --------------------------------------------------------------------------
# Hybrid controller: THRUST_SINK_MAP, CourseSchedule, three-tier blend
# --------------------------------------------------------------------------
def make_tube_frame(gate_cx=None, gate_sz=0.16, tube=True, w=640, h=360):
    f = np.full((h, w, 3), 20, dtype=np.uint8)
    if tube:
        f[:, 300:340] = TUBE_CYAN
    if gate_cx is not None:
        add_square(f, gate_cx, 0.5, gate_sz)
    return f


def test_thrust_for_sink_interp_and_clamp():
    # robust to the actual map values: exact at points, monotone, clamped, interp.
    pts = sorted(THRUST_SINK_MAP, key=lambda p: p[1])   # ascending sink
    for thrust, sink in pts:
        assert abs(thrust_for_sink(sink) - thrust) < 1e-9         # exact at map points
    assert thrust_for_sink(pts[0][1]) > thrust_for_sink(pts[-1][1])  # more sink -> less thrust
    assert thrust_for_sink(pts[-1][1] + 10) == pts[-1][0]        # max sink -> min thrust (clamp)
    assert thrust_for_sink(pts[0][1] - 10) == pts[0][0]          # below range -> max thrust (clamp)
    mid = (pts[0][1] + pts[1][1]) / 2
    assert pts[1][0] < thrust_for_sink(mid) < pts[0][0]          # interpolated between neighbours


def test_course_schedule_staircase():
    sch = CourseSchedule(cruise_speed_mps=10.0)
    assert sch.n_segments == 6
    slopes = [sch.segment_slope_deg(i) for i in range(6)]
    # level in, steep middle, level out
    assert slopes[0] > -3 and slopes[4] > -3 and slopes[5] > -3      # level segments
    assert slopes[1] < -10 and slopes[2] < -15 and slopes[3] < -14   # steep middle


def test_course_schedule_ff_thrust_descends_on_steep():
    sch = CourseSchedule(cruise_speed_mps=10.0)
    level = sch.ff_thrust(0)     # spawn->G1, ~level
    steep = sch.ff_thrust(2)     # G2->G3, steepest
    assert steep < level         # steeper segment commands LESS thrust (descend)
    assert level >= 0.28         # level ~ hover


def test_hybrid_gate_dominates_when_big():
    h = HybridController()
    # big gate physically RIGHT (image x=0.75) + tube -> gate wins; right gate needs
    # des_roll<0 (sim left-positive roll -> negative = right translation)
    o = h.command(make_tube_frame(gate_cx=0.75, gate_sz=0.18), telem(), active_gate=2)
    assert o["_debug"]["w_gate"] > 0.9
    assert o["_debug"]["u_ref"] < 0                  # LATERAL_SIGN * (u_err>0) -> <0
    assert o["_debug"]["des_roll"] < 0               # negative roll = right translation


def test_hybrid_tube_when_no_gate():
    h = HybridController()
    # tube offset to the image-right, no gate -> tube lane-keeps
    frame = np.full((360, 640, 3), 20, dtype=np.uint8)
    frame[:, 420:470] = TUBE_CYAN                     # tube right of center
    o = h.command(frame, telem(), active_gate=1)
    assert o["_debug"]["w_gate"] == 0.0
    assert o["_debug"]["w_tube"] == 1.0
    assert abs(o["_debug"]["u_ref"]) > 0.05           # follows the tube, not zero


def test_hybrid_ff_thrust_matches_segment():
    h = HybridController()
    # no gate -> thrust IS the segment feedforward (within the clamp)
    lvl = h.command(make_tube_frame(gate_cx=None), telem(), active_gate=0)["throttle"]
    stp = HybridController().command(make_tube_frame(gate_cx=None), telem(), active_gate=2)["throttle"]
    assert stp < lvl                                  # steep segment descends harder


def test_hybrid_v_trim_bounded_to_band():
    # FEEDFORWARD dominates: with a big gate driven to the frame bottom (large
    # v_err), the vision v-trim must stay within +/- hybrid_v_trim_band of ff.
    cfg = ServoConfig()
    frame = np.full((360, 640, 3), 20, dtype=np.uint8)
    frame[300:356, 292:348] = RED_GATE                 # big gate at the bottom (v_err large)
    o = HybridController().command(frame, telem(), active_gate=1)
    assert o["_debug"]["gate_size"] >= cfg.hybrid_gate_s_lo
    assert abs(o["throttle"] - o["_debug"]["ff_thrust"]) <= cfg.hybrid_v_trim_band + 1e-6


def test_hybrid_command_contract():
    o = HybridController().command(make_tube_frame(gate_cx=0.5), telem(), active_gate=1)
    for k in ("throttle", "roll", "pitch", "yaw"):
        assert k in o
    assert 0.0 <= o["throttle"] <= 1.0
    for k in ("roll", "pitch", "yaw"):
        assert -1.0 <= o[k] <= 1.0


def test_hybrid_inherits_lateral_sign():
    # CONFIRM (don't assume): the hybrid applies LATERAL_SIGN (u_ref = LATERAL_SIGN*u).
    # gate physically RIGHT (image x=0.78) -> u_ref<0 -> des_roll<0 (right translation).
    o = HybridController().command(make_tube_frame(gate_cx=0.78, gate_sz=0.18),
                                   telem(), active_gate=2)
    assert o["_debug"]["u_ref"] < 0
    assert o["_debug"]["des_roll"] < 0
    # (plant calib lives in dcl_mavlink_adapter.wire_body_rate, applied to every
    #  command_source incl. hybrid -- covered by tests/test_plant_calib.py.)


def test_hybrid_ff_trim_split_exposed():
    o = HybridController().command(make_tube_frame(gate_cx=0.5, gate_sz=0.18),
                                   telem(), active_gate=1)
    d = o["_debug"]
    for k in ("ff_thrust", "trim_thrust", "dr_fwd_dist", "dr_alt_drop", "seg_length"):
        assert k in d
    # thrust = ff_thrust + trim_thrust (trim_thrust = -v_trim), within the safety clamp
    assert abs(o["throttle"] - (d["ff_thrust"] + d["trim_thrust"])) < 1e-6


def test_hybrid_deadreckoner_is_passive():
    # DIFFERENT acceleration must NOT change the control output (DR is diagnostic).
    base = {"gravity_frd": (2.999, 0.0, 9.340), "velocity": (0.0, 0.0, 0.0)}
    t1 = dict(base, acceleration=(0.0, 0.0, -9.81))
    t2 = dict(base, acceleration=(5.0, 2.0, -14.0))   # wildly different accel
    frame = make_tube_frame(gate_cx=0.5, gate_sz=0.18)
    o1 = HybridController().command(frame, t1, active_gate=1, now=100.0)
    o2 = HybridController().command(frame, t2, active_gate=1, now=100.0)
    for k in ("throttle", "roll", "pitch", "yaw"):
        assert abs(o1[k] - o2[k]) < 1e-9, f"accel changed control on {k}"


def test_hybrid_active_gate_leads_schedule():
    # active_gate (ground truth) advances the schedule immediately; it never retreats.
    h = HybridController()
    o0 = h.command(make_tube_frame(gate_cx=None), telem(), active_gate=0, now=0.0)
    assert o0["_debug"]["sched_gate"] == 0
    o3 = h.command(make_tube_frame(gate_cx=None), telem(), active_gate=3, now=0.5)
    assert o3["_debug"]["sched_gate"] == 3          # adopts ground truth
    o1 = h.command(make_tube_frame(gate_cx=None), telem(), active_gate=1, now=0.6)
    assert o1["_debug"]["sched_gate"] == 3          # never retreats below what we've seen


def test_hybrid_safety_timeout_advances_when_active_gate_stalls():
    h = HybridController()
    cfg = ServoConfig()
    seg_time = h.schedule.segment_length(0) / cfg.cruise_speed_mps
    timeout = cfg.hybrid_seg_timeout_mult * seg_time
    h.command(make_tube_frame(gate_cx=None), telem(), active_gate=0, now=0.0)
    # active_gate stuck at 0. Before the timeout -> still segment 0.
    o = h.command(make_tube_frame(gate_cx=None), telem(), active_gate=0, now=timeout * 0.9)
    assert o["_debug"]["sched_gate"] == 0
    # Past the timeout -> schedule advances anyway (missed-gate safety).
    o2 = h.command(make_tube_frame(gate_cx=None), telem(), active_gate=0, now=timeout * 1.1)
    assert o2["_debug"]["sched_gate"] == 1


def test_deadreckoner_integrates_and_resets_on_advance():
    dr = DeadReckoner()
    g = (0.0, 0.0, 9.81)
    # constant forward kinematic accel (accel_x + gravity_x = 2.0) for 1 s
    for i in range(1, 101):
        dr.update((2.0, 0.0, -9.81), g, now=100.0 + i * 0.01)
    assert dr.fwd_dist > 0.5           # integrated some forward distance
    before = dr.fwd_dist
    dr.on_gate_advance(completed_segment_length=0.7)
    assert abs(dr.last_completed_drift - (before - 0.7)) < 1e-9   # drift recorded
    assert dr.fwd_dist == 0.0          # reset to ground truth for the new segment


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
