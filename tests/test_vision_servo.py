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
    ServoConfig, VisionServoController, GateDetector, gravity_to_roll_pitch,
    rgb_to_hsv_arrays,
)

# Bright RED gate, matching the calibrated red-wraparound band (hue ~0, high V).
RED_GATE = (230, 30, 30)
LEVEL_G = (0.0, 0.0, 9.81)
ZERO_GYRO = (0.0, 0.0, 0.0)


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
def test_gate_right_yaws_and_banks_right():
    ctl = VisionServoController()
    cmd = ctl.command(make_frame(0.80, 0.5), telem(), active_gate=0)
    assert cmd["yaw"] > 0.05           # yaw right toward gate
    assert cmd["_debug"]["des_roll"] > 0.0  # bank right


def test_gate_left_yaws_and_banks_left():
    ctl = VisionServoController()
    cmd = ctl.command(make_frame(0.20, 0.5), telem(), active_gate=0)
    assert cmd["yaw"] < -0.05
    assert cmd["_debug"]["des_roll"] < 0.0


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
    # See a gate on the right, then lose it.
    ctl.command(make_frame(0.80, 0.5), telem(), active_gate=0, now=100.0)
    dark = np.full((360, 640, 3), 25, dtype=np.uint8)
    coast = ctl.command(dark, telem(), active_gate=0, now=100.2)   # within reacquire
    assert abs(coast["_debug"]["des_yaw"]) < 1e-6
    search = ctl.command(dark, telem(), active_gate=0, now=101.5)  # past reacquire
    assert search["_debug"]["des_yaw"] > 0.0   # search toward last-seen (right)
    assert not search["_debug"]["found"]


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
