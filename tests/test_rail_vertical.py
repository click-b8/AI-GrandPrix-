"""RAIL VERTICAL (--rail-vert): the path's vanishing point drives the thrust trim.

THE IDEA. v_converge is the frame row where the two fitted rails MEET. With pitch pinned
at the coast equilibrium (PERMANENT since flt8), that row is a direct readout of the
path's slope ahead relative to where the drone is actually going. So it answers "am I
descending at the course's rate?" continuously -- unlike the red gate, which is only a
usable reference in the last metre. Steering vertically toward the path is also what
keeps the path IN FRAME, which is why the rail lock collapses on leg 1 today.

THREE THINGS THIS FILE PINS, all of which are ways to get it badly wrong:

1. THE SIGN. Convergence BELOW target = the path dives away below us = we are riding
   high = DESCEND (negative thrust trim). Inverting this flies the drone into the ground
   or off the top of the course, and nothing else in the loop would contradict it.

2. THE TARGET IS NOT ZERO. MEASURED on vision_frame.png: when the gate is vertically
   CENTRED (v_err -0.03 -- the exact condition the working gate-vert loop drives to) the
   rail converges at v = -0.346. Targeting 0 would command a standing climb of k*0.346;
   at the gate loop's gain that is 0.055, nearly DOUBLE the up-authority clamp, so it
   would sit pinned to the clamp and fly the drone off the top of the course.

3. HOLD, THEN DECAY TO THE FEED-FORWARD -- NEVER TO LEVEL. The rail blinks out at close
   range and through the gate. A trim of 0.0 is not level flight: it is this leg's
   altitude-ladder baseline, already sized to the course slope. So the vertical command
   is always either the live rail or the slope-matched feed-forward.
"""
import math
import os

import numpy as np
import pytest
from PIL import Image

from tools.schedule_flier import RailVertical, build_parser, rail_vert_targets
from vq1_vision_servo import ServoConfig, TubeDetector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Fixture lives with the tests: the source frame sits in archive/frames/, which is
# gitignored bulk data, so a clone would otherwise skip every detector test.
FRAME = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "fixtures", "vision_frame.png")

TAU, HOLD, DECAY = 0.25, 0.4, 0.6
K, TARGET = 0.10, -0.35
UP, DOWN = 0.020, 0.040
DT = 1.0 / 35.0                      # a representative tick


def rv(**kw):
    p = dict(tau_s=TAU, hold_s=HOLD, decay_s=DECAY, k=K, target=TARGET,
             up_auth=UP, down_auth=DOWN)
    p.update(kw)
    return RailVertical(p["tau_s"], p["hold_s"], p["decay_s"], p["k"], p["target"],
                        p["up_auth"], p["down_auth"])


def settle(c, v, t0=0.0, n=60, locked=True):
    """Run the filter to steady state at a constant v_converge. Returns (trim, t)."""
    t = t0
    for _ in range(n):
        t += DT
        c.update(t, DT, locked, v)
    return c.trim, t


# ---------------------------------------------------------------------------------
# 1. THE SIGN
# ---------------------------------------------------------------------------------
def test_path_diving_away_below_commands_descent():
    """Convergence BELOW the target -> the path drops away -> less thrust."""
    trim, _ = settle(rv(), TARGET + 0.30)
    assert trim < 0, f"trim {trim:+.4f} should be negative (descend)"


def test_path_running_up_out_of_frame_commands_climb():
    """Convergence ABOVE the target -> the path climbs relative to us -> more thrust."""
    trim, _ = settle(rv(), TARGET - 0.30)
    assert trim > 0, f"trim {trim:+.4f} should be positive (climb)"


def test_on_target_commands_nothing():
    """At the target the rail asks for exactly the ladder feed-forward."""
    trim, _ = settle(rv(), TARGET)
    assert abs(trim) < 1e-6, f"trim {trim:+.6f} at the target"


def test_the_trim_is_proportional_and_signed_like_the_gate_law():
    """thrust = ff - k*(v - target), the same sign convention as the gate loop's
    thrust = ff - k*v_err, so the two can be blended without a sign surprise."""
    c = rv()
    trim, _ = settle(c, TARGET + 0.10)
    assert trim == pytest.approx(-K * 0.10, abs=1e-3)


# ---------------------------------------------------------------------------------
# 2. AUTHORITY LIMITS -- what a wrong target can cost
# ---------------------------------------------------------------------------------
def test_climb_authority_is_clamped_tighter_than_descent():
    """Climb is the dangerous direction: an over-high target commands a standing climb,
    and the clamp is the bound on how far that can go before a human sees it."""
    up, _ = settle(rv(), TARGET - 5.0)
    down, _ = settle(rv(), TARGET + 5.0)
    assert up == pytest.approx(UP, abs=1e-9)
    assert down == pytest.approx(-DOWN, abs=1e-9)
    assert UP < DOWN, "climb authority must be the tighter of the two"


def test_targeting_zero_would_have_pinned_the_climb_clamp():
    """The measurement that sets the default, stated as a test. At the equilibrium the
    real frame shows (v=-0.346), a target of 0 demands more climb than the loop is
    allowed to give -- i.e. it would sit on the clamp indefinitely."""
    demanded = K * abs(-0.346 - 0.0)
    assert demanded > UP, (
        f"target 0 demands {demanded:.4f} of climb trim against a {UP:.4f} clamp")
    at_measured = K * abs(-0.346 - TARGET)
    assert at_measured < 0.1 * UP, "the shipped target should sit essentially at rest"


# ---------------------------------------------------------------------------------
# 3. HOLD THEN DECAY TO THE LADDER FEED-FORWARD
# ---------------------------------------------------------------------------------
def test_the_trim_is_held_when_the_lock_drops():
    """Through the gate the rails are occluded. The descent must not evaporate."""
    c = rv()
    trim, t = settle(c, TARGET + 0.30)
    for _ in range(int(0.3 / DT)):                 # inside the hold window
        t += DT
        c.update(t, DT, False, 0.0)
    assert c.trim == pytest.approx(trim, abs=1e-9), "the held trim changed during HOLD"
    assert c.alive


def test_after_the_hold_it_decays_to_the_ladder_not_to_level():
    """0.0 IS the destination, and 0.0 means this leg's slope-sized ladder descent --
    the baseline the trim rides on -- not level flight."""
    c = rv()
    trim, t = settle(c, TARGET + 0.30)
    assert trim < -1e-3
    for _ in range(int(6.0 / DT)):
        t += DT
        c.update(t, DT, False, 0.0)
    assert c.trim == pytest.approx(0.0, abs=1e-6), (
        f"settled at {c.trim:+.5f}; it must reach exactly the feed-forward")
    assert not c.alive, "a fully decayed rail must release the vertical to the gate loop"


def test_the_decay_takes_the_specified_time_and_never_steps():
    """Not a cut: it holds, then eases, and never jumps."""
    c = rv()
    trim, t = settle(c, TARGET + 0.30)
    seq = []
    for _ in range(int(3.0 / DT)):
        t += DT
        seq.append(c.update(t, DT, False, 0.0))
    steps = [abs(b - a) for a, b in zip(seq, seq[1:])]
    assert max(steps) < 0.002, f"trim stepped {max(steps):.4f} in one tick"
    # still commanding most of the descent one hold-length in ...
    assert abs(seq[int(0.35 / DT)]) > 0.8 * abs(trim)
    # ... and essentially retired a few decay constants later
    assert abs(seq[int(2.5 / DT)]) < 0.15 * abs(trim)


def test_re_acquisition_resumes_without_a_step():
    """The rail coming back must not slam the thrust.

    REGRESSION. This caught a real defect: the decay shrank `trim` but left the filtered
    `v` at its pre-loss value, so the first locked tick recomputed the trim from a stale
    v and jumped straight back to full magnitude -- 0.025 of thrust in ONE tick, the
    exact slam the hold-then-decay exists to prevent, just relocated to the far end of
    the loss. Both must decay together so trim == -k*(v - target) holds at every instant.
    """
    c = rv()
    _, t = settle(c, TARGET + 0.30)
    for _ in range(int(1.5 / DT)):                 # lose it, partially decay
        t += DT
        c.update(t, DT, False, 0.0)
    before = c.trim
    assert abs(before) < 0.5 * K * 0.30, "the decay did not actually retire the trim"
    seq = [before]
    for _ in range(int(1.0 / DT)):                 # re-acquire and converge
        t += DT
        seq.append(c.update(t, DT, True, TARGET + 0.30))
    steps = [abs(b - a) for a, b in zip(seq, seq[1:])]
    assert max(steps) < 0.005, (
        f"re-acquisition stepped the trim {max(steps):.4f} in one tick")
    # and it does get back to the full command, just smoothly
    assert seq[-1] == pytest.approx(-K * 0.30, abs=2e-3)


def test_the_filter_state_and_the_trim_decay_together():
    """The invariant behind the regression above: at every point during a loss the
    reported trim is exactly what the reported v would produce. If these two ever drift
    apart, re-acquisition steps."""
    c = rv()
    settle(c, TARGET + 0.30)
    t = 2.0
    for _ in range(int(2.5 / DT)):
        t += DT
        trim = c.update(t, DT, False, 0.0)
        assert trim == pytest.approx(c._clamp(-K * (c.v - TARGET)), abs=1e-9), (
            f"trim {trim:+.5f} does not match v {c.v:+.5f} during the decay")


def test_it_commands_nothing_before_it_has_ever_locked():
    """No rail yet is not the same as a lost rail: it must sit at the feed-forward."""
    c = rv()
    t = 0.0
    for _ in range(50):
        t += DT
        c.update(t, DT, False, 0.0)
    assert c.trim == 0.0 and not c.alive


def test_the_filter_smooths_a_noisy_vanishing_point():
    """v_converge extrapolates two curve fits, so it is noisier than the fits. A jump
    must not pass straight through to the thrust."""
    c = rv()
    settle(c, TARGET)
    t = 1.0
    seq = []
    rng = np.random.default_rng(0)
    for _ in range(40):
        t += DT
        seq.append(c.update(t, DT, True, TARGET + 0.3 + rng.normal(0, 0.15)))
    steps = [abs(b - a) for a, b in zip(seq, seq[1:])]
    assert max(steps) < 0.006, f"noise moved the trim {max(steps):.4f} in one tick"


# ---------------------------------------------------------------------------------
# Per-leg targets
# ---------------------------------------------------------------------------------
def test_per_leg_targets_default_to_one_fixed_value():
    """The slope-to-frame-position scale has NOT been measured, so the default must be
    a single target rather than an invented table."""
    sched = [{"seg": i, "slope_deg": s} for i, s in
             enumerate((-1.3, 12.1, 17.1, 16.2, 1.9, 1.5))]
    t = rail_vert_targets(sched, -0.35, anchor_leg=2, per_deg=0.0)
    assert t == [-0.35] * 6


def test_per_leg_targets_put_steeper_legs_lower_in_frame():
    """With a scale supplied, a steeper leg's vanishing point sits LOWER in frame, so
    its target must be more positive than the anchor's."""
    sched = [{"seg": i, "slope_deg": s} for i, s in
             enumerate((-1.3, 12.1, 17.1, 16.2, 1.9, 1.5))]
    t = rail_vert_targets(sched, -0.35, anchor_leg=2, per_deg=0.03)
    assert t[2] == pytest.approx(-0.35)                     # the anchor is untouched
    assert t[1] < t[2], "a shallower leg (12.1 deg) should target HIGHER in frame"
    assert t[4] < t[1], "the near-flat legs should be higher still"


# ---------------------------------------------------------------------------------
# Wiring / discipline
# ---------------------------------------------------------------------------------
def test_rail_vert_is_off_by_default_and_engages_from_spawn_when_asked_for():
    d = vars(build_parser().parse_args([]))
    assert d["rail_vert"] is False, "--rail-vert must ship opt-in"
    # from_ag 0 = hold the path vertically on the gate-1 approach too, which is also
    # what keeps the rail in frame early enough to be worth having. The SAFETY here is
    # that --rail-vert is itself off by default, so this fence only applies once a run
    # has explicitly opted in; --rail-vert-from-ag 1 restores the old scoping.
    assert d["rail_vert_from_ag"] == 0, "the vertical should engage from spawn"
    assert d["rail_vert_target"] == TARGET
    assert d["rail_vert_k"] == K
    assert d["rail_vert_up_auth"] == UP and d["rail_vert_down_auth"] == DOWN
    assert d["rail_vert_target_per_deg"] == 0.0
    assert d["rail_full_res"] is False, "half res is the flight-safe default"
    # the rail must not be allowed more climb authority than the gate loop it blends with
    assert d["rail_vert_up_auth"] <= d["gate_vert_up_auth"]
    assert d["rail_vert_k"] <= d["k_thrust_v"]


@pytest.mark.skipif(not os.path.exists(FRAME), reason="vision_frame.png not present")
def test_the_shipped_target_matches_the_real_frame_at_vertical_equilibrium():
    """The default target is a MEASUREMENT, not a guess -- so it must keep matching the
    frame it was measured on. If the detector changes, this catches the drift."""
    frame = np.asarray(Image.open(FRAME).convert("RGB"))
    m = TubeDetector(ServoConfig()).measure(frame)
    assert m.found
    assert m.v_converge == pytest.approx(TARGET, abs=0.05), (
        f"the real frame now converges at {m.v_converge:+.3f} but --rail-vert-target "
        f"still ships {TARGET:+.2f}")
    # and on that frame the rail would ask for essentially nothing
    trim, _ = settle(rv(), m.v_converge)
    assert abs(trim) < 0.1 * UP
