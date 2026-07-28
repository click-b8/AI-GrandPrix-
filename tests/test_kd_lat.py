"""PRIORITY 1 acceptance: lateral derivative damping, replayed on filt5's ag=2 window.

WHAT filt5 SHOWS (the defect being fixed): on the gate-3 approach the lateral loop is
pure-P, so as the gate closes u_err sweeps -0.11 -> +0.98 and des_roll sits pinned at the
-11 deg clamp for 25 of the 40 close-range ticks (t=10.0-11.5, sz up to 0.3845). Nothing
in the law opposes the rush.

WHAT THE REPLAY MEASURES: filt5.csv already contains u_f (the shipped LPF output), so the
second stage u_f2 and the resulting command can be reconstructed exactly, at the real tick
timing, without re-running a flight.

HEADLINE RESULT -- the specified acceptance does NOT hold at the specified default:

    kd_lat   peak reversal rate   clamp ticks (close-range)
     0.00        24.8 deg/s            25/40      <- pure-P, today
     0.30        24.5 deg/s            19/40      <- largest kd that still damps
     0.80       187.6 deg/s            10/40      <- the requested default

kd_lat=0.8 puts D/P at 1.08 during the rush -- the derivative is LARGER than the
proportional term -- so the command gets more dynamic, not less. What 0.8 does deliver is
the other half of the goal: it takes the command off the clamp.
"""
import csv
import math
import os

import pytest

from vq1_vision_servo import LATERAL_SIGN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILT5 = os.path.join(ROOT, "filt5.csv")

TAU = 0.20                      # --gate-filter-tau as flown
K_BANK = 1.0                    # --k-gate-bank as flown
CLAMP = math.radians(11.0)      # --gate-max-bank-deg as flown
CLOSE = (10.0, 11.5)            # the close-range gate-3 window
PURE_P_PEAK = 24.8              # measured peak reversal rate at kd_lat=0


def _window():
    if not os.path.exists(FILT5):
        pytest.skip("filt5.csv not present")
    with open(FILT5, encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["active_gate"]) == 2]
    if not rows:
        pytest.skip("filt5.csv has no ag=2 window")
    return rows


def _replay(rows, kd_lat, freeze_second_stage=False):
    """Reconstruct des_roll_deg tick by tick. freeze_second_stage=True reproduces the
    naive 'mirror v_f2 exactly' variant, which freezes du through a hold."""
    u2 = None
    u2_loss = 0.0
    prev_t = None
    out = []
    for r in rows:
        t = float(r["t"])
        dt = 0.0 if prev_t is None else t - prev_t
        prev_t = t
        u_f, sig_k, live = float(r["u_f"]), float(r["sig_k"]), int(r["live"])
        a = dt / (TAU + dt) if dt > 0 else 0.0
        if live:
            u2 = u_f if u2 is None else u2 + a * (u_f - u2)
            u2_loss = u2
        elif u2 is not None:
            if sig_k >= 1.0 and not freeze_second_stage:
                u2 += a * (u_f - u2)        # HOLD: signal still -> derivative -> 0
                u2_loss = u2
            elif sig_k < 1.0:
                u2 = u2_loss * sig_k        # FADE: both stages fade together
        if u2 is None:
            out.append((t, 0.0, 0))
            continue
        u_lat = LATERAL_SIGN * u_f
        du_lat = LATERAL_SIGN * (u_f - u2) / TAU
        cmd = K_BANK * u_lat - kd_lat * du_lat
        des = max(-CLAMP, min(CLAMP, cmd)) if int(r["steer"]) else 0.0
        out.append((t, math.degrees(des), int(r["steer"])))
    return out


def _peak_rate(trace, lo=None, hi=None):
    """deg/s, only across tick pairs where steer stayed ON -- a steer 0->1 toggle steps
    the command by construction and says nothing about the control law."""
    seq = [x for x in trace if lo is None or lo <= x[0] <= hi]
    return max(abs(b[1] - a[1]) / (b[0] - a[0])
               for a, b in zip(seq, seq[1:])
               if a[2] and b[2] and b[0] - a[0] > 1e-6)


def _clamp_ticks(trace, lo, hi):
    return sum(1 for t, d, _ in trace if lo <= t <= hi and abs(abs(d) - 11.0) < 1e-6)


def test_pure_p_saturates_the_clamp_on_the_gate3_approach():
    """The defect: today's pure-P law rides the clamp through close range."""
    rows = _window()
    trace = _replay(rows, kd_lat=0.0)
    assert _clamp_ticks(trace, *CLOSE) == 25
    assert _peak_rate(trace, *CLOSE) == pytest.approx(PURE_P_PEAK, abs=0.5)


@pytest.mark.parametrize("kd_lat", [0.1, 0.2, 0.3])
def test_d_term_reduces_peak_reversal_rate(kd_lat):
    """THE SPECIFIED ACCEPTANCE -- holds for kd_lat <= 0.3."""
    rows = _window()
    peak = _peak_rate(_replay(rows, kd_lat), *CLOSE)
    assert peak <= PURE_P_PEAK + 2.5, (
        f"kd_lat={kd_lat} gave {peak:.1f} deg/s vs pure-P {PURE_P_PEAK}")


@pytest.mark.xfail(strict=True,
                   reason="kd_lat=0.8 RAISES the peak reversal rate 24.8 -> 187.6 deg/s "
                          "(D/P = 1.08 at the rush). Recorded so the shipped default's "
                          "behaviour is pinned; flip to a pass only by lowering the "
                          "default to <=0.3.")
def test_specified_default_meets_the_specified_acceptance():
    rows = _window()
    assert _peak_rate(_replay(rows, 0.8), *CLOSE) <= PURE_P_PEAK + 2.5


@pytest.mark.parametrize("kd_lat,expected", [(0.0, 25), (0.3, 19), (0.8, 10)])
def test_d_term_monotonically_relieves_clamp_saturation(kd_lat, expected):
    """The half of the goal the D term DOES deliver at every gain tested."""
    assert _clamp_ticks(_replay(_window(), kd_lat), *CLOSE) == expected


def test_frozen_second_stage_causes_a_command_slam():
    """Mirroring v_f2 exactly -- freezing BOTH stages through a hold -- pins du at its
    last value while the signal sits still, then slams when the gate returns.
    Measured on filt5 t=11.20-11.46: du frozen at 0.972 for 0.26 s."""
    rows = _window()
    frozen = _peak_rate(_replay(rows, 0.8, freeze_second_stage=True), *CLOSE)
    fixed = _peak_rate(_replay(rows, 0.8, freeze_second_stage=False), *CLOSE)
    assert frozen == pytest.approx(436.7, abs=1.0)
    assert fixed < frozen / 2, "converging second stage must more than halve the slam"


def test_d_term_is_zero_when_signal_is_steady():
    """A still signal must produce no derivative, whatever its magnitude."""
    u_f = 0.7
    u2 = u_f
    for _ in range(50):
        u2 += (0.03 / (TAU + 0.03)) * (u_f - u2)
    assert abs((u_f - u2) / TAU) < 1e-3
