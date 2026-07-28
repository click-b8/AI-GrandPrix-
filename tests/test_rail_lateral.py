"""RAIL SIGNAL properties, measured on filt9's gate-3 leg.

STATUS: the rail has NO steering authority on any leg, and is not to be given any.
    flt10  rail-primary everywhere -> broke gate 1 (tube solid on 1.7% of ticks)
    flt12  priority handoff, rail only while solid -> CRASHED gate 1; the curvature lead
           spiked the rail to -11 deg at close range at auth 0.95
RailSignal survives as INSTRUMENTATION: it produces the tube_solid / u_tube_f columns
that answer "did the altitude ladder put us on the line?" (claude/protocol.py). What this
file pins down is that signal's properties -- bounded where the gate saturates, its area
gate load-bearing, and a clean fade to zero on loss. Useful for reading logs; NOT a
standing argument for re-arming it. See test_commit_scope.py for the shipped lateral law.

WHAT flt9 SHOWS (the defect being fixed). On the gate-3 approach the lateral loop steered
on GATE u_err. A gate blob LUNGES sideways from parallax as you close on it, while the
tube -- a continuous line -- barely moves. Straight off filt9.csv:

    t=10.57   gate u_err +0.283    tube u +0.199
    t=10.81   gate u_err +0.526    tube u +0.249
    t=11.00   gate u_err +0.751    tube u +0.302
    peak      gate u_err +0.950    tube u ~+0.37

des_roll spent 30 of the 46 close-range ticks pinned to the -11 deg clamp, steering hard
off a decoy. The drone veered off before gate 3.

WHAT THIS TEST MEASURES: filt9.csv logs u_tube / curvature / area_frac every tick
alongside the gate signal and the des_roll that was actually commanded, so the rail law
can be replayed at the real tick timing, on the real flight, without flying again. The
RailSignal filter used here is the shipped class, not a copy.

HEADLINE: on that window the rail command never reaches its clamp (0/46 ticks) where the
gate command sat on its clamp for 65% of them. Note the rail's boundedness is EARNED by
the area gate, not free -- see test_area_gate_is_load_bearing: at --tube-steer-area-min
0.005 the rail saturates too, because sub-0.01 readings are slivers of tube glimpsed past
the gate frame at close range, not the rail.
"""
import csv
import math
import os

import pytest

from vq1_vision_servo import LATERAL_SIGN, ServoConfig
from tools.schedule_flier import RailSignal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "filt9.csv")

GATE3_LEG_AG = 2                    # the leg toward gate 3, i.e. after gate 2 was passed
CLOSE_LO, CLOSE_HI = 10.0, 11.6     # the close-approach window where the gate lunges

TAU, HOLD, DECAY = 0.2, 0.4, 0.5    # --gate-filter-* defaults, shared by the rail filter
RAIL_CLAMP_DEG = 15.0               # --tube-max-bank-deg
GATE_CLAMP_DEG = 11.0               # --gate-max-bank-deg as flt9 flew it
AREA_MIN = 0.015                    # --tube-steer-area-min

pytestmark = pytest.mark.skipif(not os.path.exists(CSV),
                                reason="filt9.csv (the flt9 flight log) not present")


def _leg():
    with open(CSV, encoding="utf-8") as fh:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    return [r for r in rows if r["active_gate"] == GATE3_LEG_AG]


def _rail_replay(rows, area_min=AREA_MIN, k_bank=1.0, clamp_deg=RAIL_CLAMP_DEG):
    """Replay the RAIL law over `rows` at their real timestamps. Returns
    [(t, des_roll_deg, rail_alive, row)]."""
    cfg = ServoConfig()
    rail = RailSignal(TAU, HOLD, DECAY)
    clamp = math.radians(clamp_deg)
    out, last_t = [], None
    for r in rows:
        t = r["t"]
        dt = min(max(t - last_t, 0.0), 0.5) if last_t is not None else 0.0
        last_t = t
        solid = bool(r["tube_found"] and r["area_frac"] >= area_min)
        rail.update(t, dt, solid, r["u_tube"], r["curvature"])
        u = LATERAL_SIGN * rail.control_u(cfg.k_tube_lead)
        deg = math.degrees(max(-clamp, min(clamp, k_bank * u))) if rail.alive else 0.0
        out.append((t, deg, rail.alive, r))
    return out


def _window(seq, lo=CLOSE_LO, hi=CLOSE_HI):
    return [x for x in seq if lo <= x[0] <= hi]


def _saturation(seq, clamp_deg):
    win = _window(seq)
    assert win, "close-approach window is empty -- filt9.csv timing changed"
    sat = sum(1 for _, d, _, _ in win if abs(d) > clamp_deg - 0.05)
    return sat / len(win), max(abs(d) for _, d, _, _ in win)


def test_the_gate_law_saturated_on_this_leg():
    """Baseline -- the defect, measured off what was actually commanded in flight."""
    flown = [(r["t"], r["des_roll_deg"], True, r) for r in _leg()]
    frac, peak = _saturation(flown, GATE_CLAMP_DEG)
    assert peak == pytest.approx(GATE_CLAMP_DEG, abs=0.05)
    assert frac > 0.5, f"expected the flown gate law to sit on its clamp; got {frac:.0%}"


def test_rail_command_stays_off_the_clamp_where_the_gate_law_saturated():
    """THE HEADLINE. Same ticks, same window, rail law: never reaches the clamp."""
    rail = _rail_replay(_leg())
    frac, peak = _saturation(rail, RAIL_CLAMP_DEG)
    assert frac == 0.0, f"rail law saturated on {frac:.0%} of the close-approach ticks"
    assert peak < RAIL_CLAMP_DEG, f"rail peak {peak:.1f} deg reached the clamp"


def test_the_gate_sweeps_far_more_than_the_rail_at_close_range():
    """The premise, stated as a measurement: over the close approach the gate's lateral
    signal travels several times further than the tube's."""
    win = [r for r in _leg() if CLOSE_LO <= r["t"] <= CLOSE_HI]
    solid = [r for r in win if r["tube_found"] and r["area_frac"] >= AREA_MIN]
    gate_span = max(r["u_err"] for r in win) - min(r["u_err"] for r in win)
    tube_span = max(r["u_tube"] for r in solid) - min(r["u_tube"] for r in solid)
    assert gate_span > 3.0 * tube_span, (
        f"gate span {gate_span:.3f} vs tube span {tube_span:.3f}")


def test_area_gate_is_load_bearing():
    """The rail's boundedness is EARNED, not free. Sub-0.01 area readings are slivers of
    tube seen past the gate frame at close range -- |u_tube| runs out to 0.67 on them.
    Admitting those saturates the rail law too, which is why --tube-steer-area-min
    defaults to 0.015 and not to the value that merely 'engages more often'."""
    loose = _rail_replay(_leg(), area_min=0.005)
    frac, _ = _saturation(loose, RAIL_CLAMP_DEG)
    assert frac > 0.3, ("expected a loose area gate to saturate -- if this stopped being "
                        "true the default could be relaxed")


def test_rail_coasts_straight_when_the_tube_is_weak_never_slams():
    """'Tube lost/weak: hold last command briefly, then coast straight. No slam.'

    Three separate claims, asserted separately:
      1. it does reach EXACTLY zero (coast straight, not a standing blind turn),
      2. it takes the specified hold+fade to get there, not one tick (no step down),
      3. it never steps (no slam) anywhere on the leg.
    """
    rail = _rail_replay(_leg())
    solid = [t for t, _, _, r in rail
             if r["tube_found"] and r["area_frac"] >= AREA_MIN]
    assert solid, "no solid tube readings on this leg at all"
    last_solid = max(solid)

    # 1. fully retired once the hold has expired and the fade has run ~5 tau
    dead = [x for x in rail if x[0] > last_solid + HOLD + 5.0 * DECAY]
    assert dead, "leg ends before the fade completes -- widen the log or the window"
    assert all(abs(d) < 1e-6 for _, d, _, _ in dead), \
        "rail still commanding bank long after the tube is gone"

    # 2. and it was still commanding something right after the loss: a HOLD, not a cut
    just_after = [d for t, d, _, _ in rail if last_solid < t <= last_solid + HOLD]
    assert just_after and max(abs(d) for d in just_after) > 1.0, \
        "rail dropped its command the moment the tube blinked -- that is a step, not a hold"

    # 3. no slam anywhere on the leg
    steps = [abs(b[1] - a[1]) for a, b in zip(rail, rail[1:])]
    assert max(steps) < 2.0, f"rail command stepped {max(steps):.1f} deg in one tick"


def test_gate_fine_trim_retires_itself_as_the_lunge_begins():
    """The anti-decoy rule: the gate may fine-trim only while LARGE and NEAR-CENTRED.
    On this leg that admits the early, honest ticks and rejects every sweep tick."""
    size_min, uerr_max = 0.15, 0.15
    leg = _leg()
    admitted = [r for r in leg
                if r["live"] and r["sz_f"] >= size_min and abs(r["u_f"]) <= uerr_max]
    rejected = [r for r in leg
                if r["live"] and r["sz_f"] >= size_min and abs(r["u_f"]) > uerr_max]
    assert admitted, "the fine-trim window never opens -- thresholds are too tight"
    assert rejected, "no sweep ticks on this leg -- the test is not measuring anything"
    # every admitted tick precedes every rejected one: the trim retires and stays retired
    assert max(r["t"] for r in admitted) <= min(r["t"] for r in rejected) + 1e-9
    # and the peak gate error is firmly outside the window
    assert max(abs(r["u_f"]) for r in leg) > 4.0 * uerr_max


def test_fine_trim_clamp_keeps_the_gate_subordinate_to_the_rail():
    """Even fully engaged, the gate trim is a nudge: 3 deg against the rail's 15."""
    gate_fine_max, rail_max = 3.0, RAIL_CLAMP_DEG
    assert gate_fine_max <= rail_max / 4.0
