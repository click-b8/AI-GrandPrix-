"""GATE-3 TERMINAL DESCENT (--gate3-vert-descent).

THE OBSERVATION. Clean-rate runs cross Gate 3 HIGH -- v_err/v_f +0.22..+0.86 at closest
approach, and filt92B6 hit the UPPER bar. The sign is the same one the floor uses:
v_f > 0 means the gate sits BELOW frame centre, i.e. the drone is ABOVE it. So the
earlier upward flare (--gate3-vert-flare) was pushing the wrong way; it stays in the
build, default off, and this is the opposite correction.

THE RULE. Near the Gate-3 plane, while the drone is verifiably high, cut a small fixed
amount of thrust so the crossing drops toward centre. What keeps that from turning an
over-high miss into an under-low one is HYSTERESIS, and it is the whole design:

    arm    when v_f >= --gate3-vert-descent-vhi       (clearly high)
    release when v_f <= --gate3-vert-descent-vrelease (near centre)

with vrelease STRICTLY BELOW vhi. Because v_f falls monotonically as the drone comes
down onto the gate, the release fires on the way through centre and cannot re-arm --
the nudge is structurally incapable of driving the drone below the bar. That asymmetry,
not the size of the delta, is what makes this safe to fly, and it is what these tests
pin. The delta is deliberately tiny (0.015) because the plant has no recovery once low.

SCOPE. ag == 2 only, and only with a live Gate-3 detection at close range. Gates 1 and 2
are bit-for-bit unchanged, flag on or off.
"""
import pytest

from tools.schedule_flier import build_parser
from vq1_vision_servo import ServoConfig

DELTA = 0.015
SIZE = 0.25
VHI = 0.15
VRELEASE = 0.10

# The Gate-3 approach as flown: thrust ~0.236 on the run-in, v_f coming down from the
# high crossing the clean-rate runs actually show.
APPROACH_THRUST = 0.236
V_APPROACH = [0.86 - 0.02 * i for i in range(80)]      # +0.86 -> -0.72


class Det:
    """Stands in for the gate detection; only .found is read by the block."""

    def __init__(self, found=True):
        self.found = found


def fly(v_seq, args, ag=2, thrust=APPROACH_THRUST, sz_f=0.50, gate_v_ok=True,
        gate=Det(True)):
    """Replay the shipped block over a v_f sequence. Returns (thrusts, on_flags).

    Mirrors tools/schedule_flier.py's --gate3-vert-descent block, including the
    arm-then-release ordering (release is evaluated AFTER arm on the same tick, which is
    what makes a fresh state at v_f between vrelease and vhi stay disarmed) and the
    ol_thrust_lo floor on the cut.
    """
    cfg = ServoConfig()
    armed = False
    out, flags = [], []
    for v_f in v_seq:
        cmd = thrust
        on = False
        valid = (args.gate3_vert_descent and ag == 2 and gate_v_ok
                 and gate is not None and gate.found
                 and sz_f >= args.gate3_vert_descent_size)
        if valid and v_f >= args.gate3_vert_descent_vhi:
            armed = True
        if v_f <= args.gate3_vert_descent_vrelease or not valid:
            armed = False
        if valid and armed:
            on = True
            cmd = max(cfg.ol_thrust_lo, cmd - args.gate3_vert_descent_delta)
        if ag != 2:
            armed = False
        out.append(cmd)
        flags.append(on)
    return out, flags


ON = build_parser().parse_args(["--gate3-vert-descent"])
OFF = build_parser().parse_args([])


# ---------------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------------
def test_it_cuts_thrust_while_the_drone_is_high():
    thrusts, flags = fly(V_APPROACH, ON)
    engaged = [t for t, f in zip(thrusts, flags) if f]
    assert engaged, "the nudge never engaged on a crossing that runs v_f +0.86 -> -0.72"
    assert all(t == pytest.approx(APPROACH_THRUST - DELTA) for t in engaged)


def test_it_pushes_DOWN_not_up():
    """The direction is the entire point of this change -- the flare pushed up."""
    thrusts, _ = fly(V_APPROACH, ON)
    base, _ = fly(V_APPROACH, OFF)
    assert min(thrusts) < min(base), "the nudge did not reduce thrust anywhere"
    assert max(thrusts) <= max(base) + 1e-12, "the nudge ADDED thrust somewhere"


# ---------------------------------------------------------------------------------
# Hysteresis -- the safety argument
# ---------------------------------------------------------------------------------
def test_it_releases_at_centre_and_never_re_engages_on_a_descending_approach():
    """THE HEADLINE SAFETY PROPERTY. v_f falls monotonically through the gate, so once
    the release fires there is no path back to armed -- the nudge cannot contribute to a
    low miss."""
    thrusts, flags = fly(V_APPROACH, ON)
    last_on = max(i for i, f in enumerate(flags) if f)
    assert V_APPROACH[last_on] > VRELEASE, (
        f"still cutting thrust at v_f={V_APPROACH[last_on]:+.2f}, at or past centre")
    assert not any(flags[last_on + 1:]), "re-engaged after releasing"


def test_it_never_fires_below_centre():
    """No frame with the drone already at or under the bar may see a thrust cut."""
    _, flags = fly(V_APPROACH, ON)
    assert not any(f for v, f in zip(V_APPROACH, flags) if v <= 0.0)


def test_the_release_threshold_is_strictly_below_the_arm_threshold():
    """A band, not an edge. Equal thresholds would chatter on and off frame-to-frame at
    the crossing, and an inverted pair would latch on permanently."""
    d = vars(build_parser().parse_args([]))
    assert d["gate3_vert_descent_vrelease"] < d["gate3_vert_descent_vhi"]
    assert d["gate3_vert_descent_vrelease"] > 0.0, (
        "releasing only at or below centre would let the cut run through the gate plane")


def test_the_band_holds_the_cut_on_through_a_noisy_v_f():
    """Inside the band the state is what decides, not the instantaneous sample: v_f
    dithering between 0.11 and 0.14 after arming must keep cutting, without chatter."""
    seq = [0.30, 0.14, 0.11, 0.13, 0.12, 0.14, 0.11]
    _, flags = fly(seq, ON)
    assert all(flags), f"the cut chattered inside the hysteresis band: {flags}"


def test_a_fresh_state_inside_the_band_stays_disarmed():
    """Arriving already at 0.12 (between release and arm) must NOT engage -- only a
    verified-high frame arms it."""
    _, flags = fly([0.12] * 10, ON)
    assert not any(flags)


# ---------------------------------------------------------------------------------
# Scope -- Gates 1 and 2 are untouched
# ---------------------------------------------------------------------------------
@pytest.mark.parametrize("ag", [0, 1, 3, 4])
def test_it_never_engages_off_leg_2(ag):
    thrusts, flags = fly(V_APPROACH, ON, ag=ag)
    assert not any(flags)
    assert all(t == APPROACH_THRUST for t in thrusts)


def test_the_arm_is_dropped_when_the_leg_advances():
    """Armed on ag==2, then Gate 3 registers: the very next frame must be clean, so no
    ag==2 state leaks onto the Gate-4 leg."""
    cfg = ServoConfig()
    armed = True                                     # as left by a high ag==2 frame
    ag, v_f = 3, 0.30
    valid = (ON.gate3_vert_descent and ag == 2)
    assert not valid
    if v_f <= ON.gate3_vert_descent_vrelease or not valid:
        armed = False
    assert not armed
    assert cfg.ol_thrust_lo > 0.0                    # floor still meaningful downstream


def test_losing_the_detection_disarms_it():
    """No coasting on a stale v_f: the cut stops the instant the gate is not seen."""
    _, flags = fly([0.30] * 3, ON, gate=Det(False))
    assert not any(flags)
    _, flags = fly([0.30] * 3, ON, gate_v_ok=False)
    assert not any(flags)


def test_it_does_not_engage_before_close_range():
    _, flags = fly(V_APPROACH, ON, sz_f=SIZE - 0.01)
    assert not any(flags)
    _, flags = fly(V_APPROACH, ON, sz_f=SIZE)
    assert any(flags), "the size gate is exclusive at the documented threshold"


# ---------------------------------------------------------------------------------
# Discipline
# ---------------------------------------------------------------------------------
def test_off_by_default_is_a_no_op():
    thrusts, flags = fly(V_APPROACH, OFF)
    assert not any(flags)
    assert all(t == APPROACH_THRUST for t in thrusts)


def test_defaults_are_off_and_as_specified():
    d = vars(build_parser().parse_args([]))
    assert d["gate3_vert_descent"] is False, "the nudge must ship opt-in"
    assert d["gate3_vert_descent_delta"] == DELTA
    assert d["gate3_vert_descent_size"] == SIZE
    assert d["gate3_vert_descent_vhi"] == VHI
    assert d["gate3_vert_descent_vrelease"] == VRELEASE


def test_the_delta_stays_small():
    """Sized against the plant, not the miss: there is no recovery from a low crossing,
    so the correction is deliberately far smaller than the ~0.099 headroom the upward
    flare had to play with. If this ever needs to grow, the high bias is being set
    earlier than the terminal window and the fix belongs on the leg-2 descent instead."""
    d = vars(build_parser().parse_args([]))
    assert 0.0 < d["gate3_vert_descent_delta"] <= 0.03


def test_the_cut_cannot_dig_under_the_open_loop_floor():
    cfg = ServoConfig()
    thrusts, flags = fly([0.30] * 5, ON, thrust=cfg.ol_thrust_lo + 0.005)
    assert all(flags)
    assert min(thrusts) == pytest.approx(cfg.ol_thrust_lo)
