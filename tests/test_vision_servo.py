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
    # LOCK the lateral sign convention -- this class of bug cost five flights.
    # The DCL FPV is horizontally MIRRORED, so the physical control error
    # u_ctrl = -image_u. A gate PHYSICALLY to the RIGHT (u_ctrl>0) MUST command
    # bank AND yaw to the RIGHT (positive), moving the drone toward it.
    # An image-LEFT blob is physically RIGHT:
    right = VisionServoController().command(make_frame(0.20, 0.5), telem(), active_gate=0)
    assert right["_debug"]["u_ctrl"] > 0
    assert right["_debug"]["des_roll"] > 0.0    # bank right
    assert right["yaw"] > 0.0                    # yaw right
    # An image-RIGHT blob is physically LEFT -> command LEFT:
    left = VisionServoController().command(make_frame(0.80, 0.5), telem(), active_gate=0)
    assert left["_debug"]["u_ctrl"] < 0
    assert left["_debug"]["des_roll"] < 0.0
    assert left["yaw"] < 0.0


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
    # Gate seen at IMAGE-right (make_frame 0.80) -> physically LEFT (mirror). Lose it.
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


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
