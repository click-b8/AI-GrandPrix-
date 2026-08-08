"""RAIL AS THE PRIMARY REFERENCE (--rail-lateral-primary) and the ADAPTIVE LOOK-AHEAD.

The rail steers and the gate is the fallback -- the reverse of the default law, and the
"stay on the path from the start" behaviour: authority from ag == 0.

THE HISTORY THIS IS RETRYING. flt10 gave the rail authority on every leg and broke gate 1;
flt12 crashed into gate 1. Both steered on a detector that thresholded cyan AREA, so on
the legs where it mattered "steer on the tube" meant "steer on nothing" (solid on 1.7% of
ticks). The rebuilt detector fits the two rail curves and locks on fit quality, which is
why the retry is reasonable -- but it is UNPROVEN in flight, so what this file pins is the
things that make it survivable: a proportional command that never sits on its clamp, a
continuous fade back to gate-centring on loss, and no curvature lead at all.

THE ADAPTIVE LOOK-AHEAD is what keeps the rail usable as the nose rises. The rails
DIVERGE downward from the vanishing point, so a measurement row is only meaningful BELOW
it. As the path sinks the convergence slides down past a fixed 0.75H look-ahead, the two
fitted curves have already crossed there, and the separation goes NEGATIVE -- no lock, on
a fit with 0.40 px residual. It is not the search band running out: that already reaches
the last row of the frame.
"""
import math
import os

import numpy as np
import pytest
from PIL import Image

from dataclasses import replace

from vq1_vision_servo import LATERAL_SIGN, ServoConfig, TubeDetector
from tools.schedule_flier import (RailSignal, blend_lateral, build_parser, lag_step,
                                  rail_bank_cmd)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Fixture lives with the tests: the source frame sits in archive/frames/, which is
# gitignored bulk data, so a clone would otherwise skip every detector test.
FRAME = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "fixtures", "vision_frame.png")

K_LAT, CLAMP_DEG = 0.5, 11.0
TAU, HOLD, DECAY, AUTH_TAU = 0.2, 0.4, 0.5, 0.3
DT = 1.0 / 35.0

real_frame = pytest.mark.skipif(not os.path.exists(FRAME),
                                reason="vision_frame.png not present")


def frame():
    return np.asarray(Image.open(FRAME).convert("RGB"))


def sunk(px):
    """The real frame with the path sunk `px` rows lower, as when the nose rises."""
    f = np.roll(frame(), px, axis=0)
    if px > 0:
        f[:px] = 0
    return f


def primary_cmd(u_tube, k=K_LAT, clamp_deg=CLAMP_DEG):
    """The shipped primary law: pure band-centring, no lead."""
    return rail_bank_cmd(LATERAL_SIGN * u_tube, 0.0, k, 0.0,
                         math.radians(clamp_deg), 0.0)


# --------------------------------------------------------------------------------
# The primary lateral law
# --------------------------------------------------------------------------------
def test_it_is_off_by_default_and_engages_from_spawn_when_asked_for():
    d = vars(build_parser().parse_args([]))
    assert d["rail_lateral_primary"] is False, "must ship opt-in"
    assert d["rail_lateral_from_ag"] == 0, "the point of the flag is authority from spawn"
    assert d["rail_lat_k"] == K_LAT
    assert d["rail_lat_max_deg"] == CLAMP_DEG


def test_a_centred_path_commands_essentially_nothing():
    """u_tube reads about +-0.005 when centred, so the standing command must be a
    fraction of a degree -- not something that slowly walks the drone off the line."""
    assert abs(math.degrees(primary_cmd(0.005))) < 0.2


def test_the_command_is_signed_to_steer_back_toward_the_path():
    """Opposite offsets must give opposite banks, and they must be symmetric."""
    left, right = primary_cmd(-0.20), primary_cmd(+0.20)
    assert left * right < 0, "the two directions did not produce opposite banks"
    assert left == pytest.approx(-right, abs=1e-9)


def test_the_gain_is_proportional_across_the_whole_working_range():
    """The point of the gentle gain: a large-but-real offset lands AT the clamp rather
    than beyond it, so the rail never steers on a stop the way the gate law did."""
    assert abs(math.degrees(primary_cmd(0.384))) == pytest.approx(CLAMP_DEG, abs=0.2)
    for u in (0.05, 0.10, 0.20, 0.30):
        assert abs(math.degrees(primary_cmd(u))) < CLAMP_DEG - 0.5


def test_the_clamp_binds_on_an_absurd_offset():
    assert abs(math.degrees(primary_cmd(5.0))) == pytest.approx(CLAMP_DEG, abs=1e-6)


def test_no_curvature_lead_can_enter_the_primary_command():
    """THE flt12 TERM IS ABSENT BY CONSTRUCTION. Whatever the curvature does, the command
    depends only on u_tube -- that term spiked the rail to its clamp and flew into gate 1,
    and holding the path centred does not need it."""
    for curv in (-5.0, -1.0, 0.0, 1.0, 5.0):
        assert rail_bank_cmd(LATERAL_SIGN * 0.1, curv, K_LAT, 0.0,
                             math.radians(CLAMP_DEG), 0.0) == pytest.approx(
                                 primary_cmd(0.1), abs=1e-12)


# --------------------------------------------------------------------------------
# Fallback to the gate law
# --------------------------------------------------------------------------------
def _handoff(lock_pattern, u_tube=0.20, gate_cmd=math.radians(-6.0)):
    """Drive the rail filter with a lock/no-lock pattern and return the blended
    command each tick, as the flight computes it."""
    rail = RailSignal(TAU, HOLD, DECAY)
    out, auth, t = [], 0.0, 0.0
    for locked in lock_pattern:
        t += DT
        rail.update(t, DT, locked, u_tube, 0.0)
        target = rail.k if rail.alive else 0.0
        auth = lag_step(auth, DT, target, AUTH_TAU)
        if target == 0.0 and auth < 1e-4:
            auth = 0.0
        cmd = primary_cmd(u_tube) if auth > 0.0 else 0.0
        out.append((blend_lateral(auth, cmd, 0.0, 1.0, gate_cmd), auth))
    return out


def test_losing_the_rail_falls_back_to_the_gate_law():
    """Not to wings-level, and not to a held blind turn: to the gate command."""
    gate_cmd = math.radians(-6.0)
    seq = _handoff([True] * 60 + [False] * 200, gate_cmd=gate_cmd)
    assert seq[-1][1] == 0.0, "the rail kept authority after a long loss"
    assert seq[-1][0] == pytest.approx(gate_cmd, abs=1e-9)


def test_the_fallback_is_continuous_not_a_switch():
    """A hard handoff would step the bank the instant the lock dropped."""
    seq = _handoff([True] * 60 + [False] * 200)
    steps = [abs(math.degrees(b[0] - a[0])) for a, b in zip(seq, seq[1:])]
    assert max(steps) < 1.0, f"the handoff stepped {max(steps):.2f} deg in one tick"


def test_re_acquiring_the_rail_is_continuous_too():
    seq = _handoff([True] * 40 + [False] * 60 + [True] * 60)
    steps = [abs(math.degrees(b[0] - a[0])) for a, b in zip(seq, seq[1:])]
    assert max(steps) < 1.0, f"re-acquisition stepped {max(steps):.2f} deg"
    assert seq[-1][1] > 0.9, "the rail did not take authority back"


def test_a_blink_does_not_hand_control_back():
    """The rail's hold window carries a one-or-two-frame dropout, so a blink must not
    bounce the drone between two laws."""
    seq = _handoff([True] * 60 + [False] * 3 + [True] * 30)
    assert min(a for _, a in seq[60:]) > 0.9, "a 3-tick blink released authority"


# --------------------------------------------------------------------------------
# The adaptive look-ahead: keeping the lock as the path sinks
# --------------------------------------------------------------------------------
@real_frame
def test_the_search_band_already_reaches_the_bottom_row():
    """Stated as a test because it is the thing that is easy to get wrong: there is no
    room to 'extend the band downward' -- it is already the last row, so the band is not
    what runs out when the path sinks."""
    assert ServoConfig().rail_band[1] == 1.0


@real_frame
def test_a_fixed_lookahead_loses_the_lock_once_the_path_sinks():
    """THE DEFECT. With the adaptation disabled the lock dies at 150 px of sink -- and
    NOT because the fit went bad: rows and residual are both healthy. It dies because the
    look-ahead ends up above the convergence, where the rails have crossed."""
    cfg = replace(ServoConfig(), rail_lookahead_below_conv=0.0)
    m = TubeDetector(cfg).measure(sunk(150))
    assert not m.found
    assert m.rail_sep < 0, f"expected a NEGATIVE separation, got {m.rail_sep:+.3f}"
    assert m.n_rows > 20 and m.fit_rms < 2.0, (
        f"the fit itself should be fine (rows={m.n_rows}, rms={m.fit_rms:.2f})")


@real_frame
def test_the_adaptive_lookahead_holds_the_lock_as_the_path_sinks():
    """THE FIX, and how far it buys: locked out to 180 px of sink where a fixed
    look-ahead gave up at 120."""
    det = TubeDetector(ServoConfig())
    for px in (0, 60, 120, 150, 180):
        m = det.measure(sunk(px))
        assert m.found, f"lost the lock at {px} px of sink (sep={m.rail_sep:+.3f})"
        assert m.rail_sep > 0


@real_frame
def test_the_adaptation_leaves_the_nominal_frame_untouched():
    """It must only bind once the convergence is actually low, or it would perturb the
    measurement on every ordinary frame."""
    a = TubeDetector(ServoConfig()).measure(frame())
    b = TubeDetector(replace(ServoConfig(),
                             rail_lookahead_below_conv=0.0)).measure(frame())
    assert a.u_tube == pytest.approx(b.u_tube, abs=1e-12)
    assert a.y_look == pytest.approx(b.y_look, abs=1e-12)


@real_frame
def test_the_lookahead_is_pushed_below_the_convergence_only_when_needed():
    det = TubeDetector(ServoConfig())
    cfg = ServoConfig()
    fixed = cfg.rail_lookahead * 360
    assert det.measure(frame()).y_look == pytest.approx(fixed)      # not needed
    m = det.measure(sunk(150))
    assert m.y_look > fixed, "the look-ahead did not adapt when the path sank"
    y_conv = m.v_converge * 180 + 180
    assert m.y_look >= y_conv + cfg.rail_lookahead_below_conv * 360 - 1.0


@real_frame
def test_the_offset_stays_trustworthy_while_the_path_sinks():
    """The drone is laterally centred in this frame at every sink depth, so the steering
    command must stay near zero rather than drifting as the measurement row moves."""
    det = TubeDetector(ServoConfig())
    for px in (0, 60, 120, 150, 180):
        m = det.measure(sunk(px))
        assert abs(m.u_tube) < 0.03, f"{px} px sink: offset {m.u_tube:+.4f}"
        assert abs(math.degrees(primary_cmd(m.u_tube))) < 1.0


@real_frame
def test_a_path_genuinely_out_of_frame_still_fails_to_lock():
    """The adaptation must not manufacture a lock out of nothing -- when the path really
    has left the frame there is nothing to steer to and the gate law should take over."""
    m = TubeDetector(ServoConfig()).measure(sunk(230))
    assert not m.found and m.n_rows < 20
