"""GATE-CENTRE ALTITUDE FLOOR (--gate-vert-floor).

THE DEFECT. On a gate approach the filtered v_f runs +0.6 -> 0 -> -0.7: the drone
descends onto the gate line and straight through it. v_f > 0 means the gate is BELOW
frame centre (still above it); negative means the gate is ABOVE centre, i.e. the drone is
already under the bar. The climb reaction only fires once v_f has gone negative, and by
then there is sink velocity built up that the vertical trim cannot arrest in the metre
that remains. That is the consistent under-gate-2 (and gate-3) hit.

THE RULE. Once the drone has come down to a gate's level, it may not sink further until
it is through: while a gate is live and v_f <= vthresh, thrust is held at or above a
floor. A MAX, not an assignment -- hold or climb is allowed, only descending is denied.

WHAT MAKES OR BREAKS IT, and the reason this file exists: the floor has to beat the
THRUST SLEW LIMITER. On the steep legs the ladder baseline is 0.145-0.155 and the floor
is 0.250, so through the 0.15/s up-slew alone the floor takes 0.63-0.70 s to arrive -- at
8 m/s that is ~5 m, and the gate is long past. A floor applied only to the pre-slew
command would therefore do essentially nothing on exactly the two gates it was asked to
fix. It is re-applied after the limiter.
"""
import math

import pytest

from tools.schedule_flier import build_descent_ladder, build_bank_schedule, build_parser
from vq1_vision_servo import ServoConfig

FLOOR = 0.250
VTHRESH = 0.10
SLEW_UP, SLEW_DOWN = 0.15, 0.60
DT = 1.0 / 35.0


def leg_baselines():
    """base_thrust per segment exactly as the flight computes it: the sink-map schedule
    anchored to level thrust, clamped to the open-loop bounds, minus the altitude
    ladder's per-leg descent, AND CLAMPED AGAIN -- that second clamp is easy to forget
    and it changes the answer. On segs 1-3 the pre-clamp value is 0.145-0.155 but
    ol_thrust_lo is 0.18, so the real baseline there is 0.180."""
    cfg = ServoConfig()
    sched = build_bank_schedule(8.0, cfg=cfg, maneuver_frac=0.5)
    ladder = build_descent_ladder(sched, 0.035, anchor_leg=2)
    out = []
    for r in sched:
        sink = 8.0 * math.tan(math.radians(r["slope_deg"]))
        thr = max(cfg.ol_thrust_lo,
                  min(cfg.ol_thrust_hi, round(0.275 - max(sink, 0.0) / 17.6, 3)))
        base = thr - ladder[r["seg"]]
        out.append(max(cfg.ol_thrust_lo, min(cfg.ol_thrust_hi, base)))
    return out


def test_the_ladder_descent_is_swallowed_by_the_thrust_floor_on_the_steep_legs():
    """NOT about --gate-vert-floor, but found while sizing it and worth pinning: on segs
    1-3 the altitude ladder subtracts 0.025-0.035 from a baseline that is ALREADY at
    ol_thrust_lo (0.18), so the second clamp puts it straight back. The ladder therefore
    contributes nothing on the three steep legs -- the drone flies 0.180 there with the
    ladder on or off. If the descent on those legs ever needs to be real, it is
    ol_thrust_lo that has to move, not the ladder."""
    cfg = ServoConfig()
    assert leg_baselines()[1] == pytest.approx(cfg.ol_thrust_lo)
    assert leg_baselines()[2] == pytest.approx(cfg.ol_thrust_lo)
    assert leg_baselines()[3] == pytest.approx(cfg.ol_thrust_lo)
    # the flat legs are above the clamp, so there the ladder does bite
    assert leg_baselines()[4] > cfg.ol_thrust_lo


def fly(v_seq, base, floor_on=True, floor=FLOOR, vthresh=VTHRESH,
        past_slew=True, gate_live=True, vtrim=0.0):
    """Replay the shipped vertical chain over a v_f sequence: form thrust_cmd, apply the
    floor, slew-limit, then re-apply the floor past the limiter. Returns the commanded
    thrust per tick (what actually reaches the motors)."""
    cfg = ServoConfig()
    prev, out = None, []
    for v_f in v_seq:
        cmd = max(cfg.ol_thrust_lo, min(cfg.ol_thrust_hi, base + vtrim))
        active = bool(floor_on and gate_live and v_f <= vthresh)
        if active:
            cmd = max(cmd, floor)
        if prev is None:
            thrust = cmd
        else:
            rate = SLEW_DOWN if cmd < prev else SLEW_UP
            step = rate * DT
            thrust = prev + max(-step, min(step, cmd - prev))
        if active and past_slew:
            thrust = max(thrust, floor)
        prev = thrust
        out.append(thrust)
    return out


APPROACH = [0.6 - 0.02 * i for i in range(70)]      # v_f +0.6 -> -0.78, the flt failure


# ---------------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------------
def test_thrust_never_falls_below_the_floor_once_the_gate_is_reached():
    """THE HEADLINE, on the steepest leg."""
    base = leg_baselines()[2]
    out = fly(APPROACH, base)
    engaged = [t for v, t in zip(APPROACH, out) if v <= VTHRESH]
    assert engaged, "the floor never engaged on this approach"
    assert min(engaged) >= FLOOR - 1e-9, (
        f"thrust reached {min(engaged):.4f} below the {FLOOR} floor")


def test_it_engages_before_dead_centre_not_after():
    """A FLARE, not a catch: the sink must be stopped while the drone is still ABOVE the
    bar. The first floored tick has to come while v_f is still positive."""
    base = leg_baselines()[2]
    first = next(i for i, v in enumerate(APPROACH) if v <= VTHRESH)
    assert APPROACH[first] > 0.0, "the floor engaged only once already under the gate"
    out = fly(APPROACH, base)
    assert out[first] >= FLOOR - 1e-9


def test_it_is_a_max_so_the_drone_may_still_climb():
    """The floor forbids sinking; it does not command an altitude. Where the baseline
    plus up-trim exceeds the floor, the command must follow it UP rather than being
    pinned to the floor. Uses a flat leg, because on the steep legs the baseline is
    ol_thrust_lo (0.18) and the whole gate-vert up-authority (0.03) still lands below
    the 0.250 floor -- there the floor genuinely does dominate, which is the point of it.
    """
    flat = leg_baselines()[4]
    out = fly(APPROACH, flat, vtrim=+0.03)
    engaged = [t for v, t in zip(APPROACH, out) if v <= VTHRESH]
    assert max(engaged) > FLOOR + 1e-6, "the floor pinned the command and blocked a climb"
    assert max(engaged) == pytest.approx(min(flat + 0.03, 0.34), abs=1e-6)


def test_it_does_not_engage_while_the_drone_is_still_high():
    """Above the gate the ladder must be free to descend -- that is how it gets there."""
    base = leg_baselines()[2]
    high = [0.6, 0.5, 0.4, 0.3, 0.2]
    out = fly(high, base)
    assert max(out) < FLOOR, "the floor engaged while the gate was still well below"


def test_it_does_not_engage_without_a_live_gate():
    """Between gates the ladder owns the descent; a stale v_f must not floor the thrust
    for the whole leg."""
    base = leg_baselines()[2]
    out = fly(APPROACH, base, gate_live=False)
    assert max(out) < FLOOR


def test_off_by_default_changes_nothing():
    base = leg_baselines()[2]
    assert fly(APPROACH, base, floor_on=False) == fly(APPROACH, base, floor_on=False)
    assert max(fly(APPROACH, base, floor_on=False)) < FLOOR


# ---------------------------------------------------------------------------------
# Beating the slew limiter -- the part that makes it real
# ---------------------------------------------------------------------------------
def test_the_slew_limiter_alone_would_defeat_the_floor_on_every_steep_leg():
    """Quantifies why the floor is re-applied past the limiter.

    Left to the 0.15/s up-slew, the floor needs 0.47 s to arrive from the steep legs'
    0.180 baseline. Stated in DISTANCE, which is what decides whether it matters: at the
    8 m/s cruise that is ~3.7 m of travel, and the gate-2 approach does not have 3.7 m
    left once v_f has fallen to the engage threshold. So the clamp would arrive after
    the gate."""
    CRUISE = 8.0
    for seg in (1, 2, 3):
        base = leg_baselines()[seg]
        assert base < FLOOR
        t_to_floor = (FLOOR - base) / SLEW_UP
        assert t_to_floor * CRUISE > 2.0, (
            f"seg{seg}: the floor would arrive after only {t_to_floor * CRUISE:.1f} m "
            f"({t_to_floor:.2f}s) -- if the baselines moved this close to the floor, the "
            f"past-slew re-apply may no longer be needed")


def test_without_the_past_slew_reapply_the_drone_sinks_through_the_gate():
    """THE REGRESSION GUARD. Same approach, floor applied only BEFORE the limiter: the
    thrust spends the whole gate window under the floor, i.e. still sinking."""
    base = leg_baselines()[2]
    weak = fly(APPROACH, base, past_slew=False)
    engaged = [t for v, t in zip(APPROACH, weak) if v <= VTHRESH]
    assert min(engaged) < FLOOR - 0.05, (
        "the pre-slew-only floor held after all -- the slew rates must have changed")
    strong = fly(APPROACH, base, past_slew=True)
    s_eng = [t for v, t in zip(APPROACH, strong) if v <= VTHRESH]
    assert min(s_eng) >= FLOOR - 1e-9
    assert min(s_eng) > min(engaged), "the past-slew re-apply bought nothing"


def test_the_floor_is_reached_on_the_very_first_engaged_tick():
    """No ramp-in: the point is to arrest sink NOW, not 0.7 s from now."""
    base = leg_baselines()[2]
    out = fly(APPROACH, base)
    first = next(i for i, v in enumerate(APPROACH) if v <= VTHRESH)
    assert out[first] == pytest.approx(FLOOR, abs=1e-9)


def test_release_walks_the_thrust_back_down_rather_than_dropping_it():
    """Once the gate is passed the floor lets go; prev_thrust tracks the floored value,
    so the ordinary down-slew returns to the leg baseline smoothly."""
    base = leg_baselines()[2]
    v = APPROACH + [None] * 40                       # None = gate gone
    cfg = ServoConfig()
    prev, out = None, []
    for x in v:
        live = x is not None
        cmd = max(cfg.ol_thrust_lo, min(cfg.ol_thrust_hi, base))
        active = live and x <= VTHRESH
        if active:
            cmd = max(cmd, FLOOR)
        if prev is None:
            thrust = cmd
        else:
            rate = SLEW_DOWN if cmd < prev else SLEW_UP
            step = rate * DT
            thrust = prev + max(-step, min(step, cmd - prev))
        if active:
            thrust = max(thrust, FLOOR)
        prev = thrust
        out.append(thrust)
    tail = out[len(APPROACH):]
    steps = [abs(b - a) for a, b in zip(tail, tail[1:])]
    assert max(steps) <= SLEW_DOWN * DT + 1e-9, "release dropped the thrust in one step"
    assert tail[-1] == pytest.approx(base, abs=1e-3), "never returned to the baseline"


# ---------------------------------------------------------------------------------
# Precedence: it must beat the ladder and the VCOMMIT freeze
# ---------------------------------------------------------------------------------
def test_the_floor_overrides_the_altitude_ladder_descent():
    """The ladder lives in base_thrust, upstream of the floor, so its per-leg descent
    cannot dig under it -- checked on every descending leg."""
    for seg, base in enumerate(leg_baselines()):
        out = fly([0.05] * 20, base)
        assert min(out) >= FLOOR - 1e-9, f"seg{seg} ladder pulled under the floor"


def test_the_floor_overrides_a_vcommit_freeze_pulling_thrust_down():
    """VCOMMIT owns vtrim at exactly this range and eases it to the leg baseline; since
    the floor is applied to the fully-formed thrust_cmd, a negative held trim cannot dig
    under it either."""
    base = leg_baselines()[2]
    out = fly([0.05] * 20, base, vtrim=-0.06)        # the down-authority limit
    assert min(out) >= FLOOR - 1e-9


# ---------------------------------------------------------------------------------
# Discipline
# ---------------------------------------------------------------------------------
def test_defaults_are_off_and_as_specified():
    d = vars(build_parser().parse_args([]))
    assert d["gate_vert_floor"] is False, "the floor must ship opt-in"
    assert d["gate_vert_floor_vthresh"] == VTHRESH
    assert d["gate_vert_floor_thrust"] == FLOOR
    # engaging BEFORE dead centre is the whole design -- a non-positive default would
    # make it a catch instead of a flare
    assert d["gate_vert_floor_vthresh"] > 0.0
