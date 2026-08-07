"""GATE-3 BLIND DESCENT HOLD (--gate3-vert-descent-hold-s).

THE DEFECT. MEASURED on the clean descent-enabled runs: the visible arm->loss window is
only ~0.23-0.25 s. The gate goes away while v_f is still HIGH (+0.38..+0.45) and falling
at ~-0.5/s, and there is another ~0.54-0.75 s of BLIND coast to the gate plane. The
visible descent releases the instant the detection drops, so the drone stops sinking
while still high and half a second short of the plane -- the cut expires before it can
land. Carrying the same cut through the blind coast for a bounded time is the fix;
replay gives 0.30 s (crossing +0.28 -> ~+0.20). First flight test is 0.20 s.

WHY IT IS A SEPARATE STAGE. The visible block fires only on a VALID detection; this one
fires only on an INVALID one. They are mutually exclusive by construction, so the delta
can never be applied twice on one tick -- which is the thing that would turn a 0.015
nudge into a dive.

WHAT KEEPS IT SAFE:
  * it never increases the delta -- the same fixed 0.015, not scaled by the stale v_f;
  * the timer is bounded and cannot be restarted by detector flicker (g3_loss_t is set
    once at the transition and cleared only by a valid frame or by leaving the leg);
  * ANY valid reacquisition ends it and hands back to the live path;
  * leaving ag==2 -- including the advance to Gate 3 -- hard-resets every field.
"""
import pytest

from tools.schedule_flier import build_parser
from vq1_vision_servo import ServoConfig

DELTA = 0.015
SIZE = 0.25
HOLD_S = 0.20
DT = 1.0 / 35.0
BASE = 0.236


class Det:
    def __init__(self, found=True):
        self.found = found


def replay(frames, args, dt=DT, thrust=BASE):
    """Replay BOTH gate-3 descent stages over a frame sequence.

    frames: list of (ag, gate_v_ok, found, sz_f, v_f). Mirrors tools/schedule_flier.py.
    Returns per-tick dicts of the logged state.
    """
    cfg = ServoConfig()
    desc_armed = False
    g3hold_active, g3hold_t0 = False, None
    g3_last_vf, g3_last_sz, g3_loss_t = 0.0, 0.0, None
    g3_was_active = False
    now = 0.0
    out = []
    for ag, v_ok, found, sz_f, v_f in frames:
        now += dt
        cmd = thrust
        # --- visible descent -------------------------------------------------------
        desc_on, desc_raw = False, 0.0
        valid_d = (args.gate3_vert_descent and ag == 2 and v_ok and found
                   and sz_f >= args.gate3_vert_descent_size)
        if valid_d and v_f >= args.gate3_vert_descent_vhi:
            desc_armed = True
        if v_f <= args.gate3_vert_descent_vrelease or not valid_d:
            desc_armed = False
        if valid_d and desc_armed:
            desc_on = True
            desc_raw = -args.gate3_vert_descent_delta
            cmd = max(cfg.ol_thrust_lo, cmd - args.gate3_vert_descent_delta)
        if ag != 2:
            desc_armed = False
        # --- blind hold ------------------------------------------------------------
        g3hold_on = False
        if args.gate3_vert_descent_hold_s > 0.0 and ag == 2:
            valid = (v_ok and found and sz_f > 0.0)
            if valid:
                g3_last_vf, g3_last_sz = v_f, sz_f
                g3_was_active = bool(desc_on)
                if g3hold_active and v_f <= args.gate3_vert_descent_hold_release_vf:
                    g3hold_active = False
                g3hold_active = False
                g3_loss_t = None
            else:
                if (not g3hold_active and g3_loss_t is None and g3_was_active
                        and g3_last_sz >= args.gate3_vert_descent_size
                        and g3_last_vf > 0.0):
                    g3hold_active, g3hold_t0, g3_loss_t = True, now, now
                if g3hold_active:
                    if (now - g3hold_t0) >= args.gate3_vert_descent_hold_s:
                        g3hold_active = False
                    else:
                        g3hold_on = True
                        cmd = max(cfg.ol_thrust_lo,
                                  cmd - args.gate3_vert_descent_delta)
        if ag != 2:
            g3hold_active, g3hold_t0, g3_loss_t, g3_was_active = False, None, None, False
        out.append(dict(
            thrust=cmd, ag=ag, desc_on=desc_on,
            g3hold_armed=int(g3_was_active and ag == 2), g3hold_on=g3hold_on,
            g3hold_elapsed=(now - g3hold_t0) if g3hold_t0 is not None else 0.0,
            g3_last_vf=g3_last_vf, g3_last_sz=g3_last_sz,
            delta_app=(desc_raw - args.gate3_vert_descent_delta) if g3hold_on else desc_raw))
    return out


def cfg_args(*extra):
    return build_parser().parse_args([
        "--gate3-vert-descent", "--gate3-vert-descent-delta", str(DELTA),
        "--gate3-vert-descent-size", str(SIZE), *extra])


ON = cfg_args("--gate3-vert-descent-hold-s", str(HOLD_S))
OFF = cfg_args()

# A Gate-3 approach: visible and high, then the detection drops for the blind coast.
VISIBLE = [(2, True, True, 0.50, 0.40)] * 8
BLIND = [(2, False, False, 0.0, 0.0)] * 30


# ---------------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------------
def test_the_cut_continues_through_the_blind_coast():
    ticks = replay(VISIBLE + BLIND, ON)
    assert ticks[7]["desc_on"], "the visible descent never engaged"
    held = [t for t in ticks[8:] if t["g3hold_on"]]
    assert held, "the hold never fired after the detection dropped"
    assert all(t["thrust"] == pytest.approx(BASE - DELTA) for t in held)


def test_without_the_hold_the_cut_stops_at_gate_loss():
    """THE REGRESSION GUARD -- what the flag is for."""
    ticks = replay(VISIBLE + BLIND, OFF)
    assert all(t["thrust"] == BASE for t in ticks[8:]), (
        "thrust was still reduced after loss with the hold off")


def test_the_delta_is_never_doubled():
    """The two stages are mutually exclusive; a tick that cut twice would be a dive."""
    for t in replay(VISIBLE + BLIND, ON):
        assert t["thrust"] >= BASE - DELTA - 1e-12
        assert abs(t["delta_app"]) <= DELTA + 1e-12


def test_the_hold_is_bounded_by_the_timer():
    ticks = replay(VISIBLE + BLIND, ON)
    held = [i for i, t in enumerate(ticks) if t["g3hold_on"]]
    assert ticks[held[-1]]["g3hold_elapsed"] < HOLD_S
    assert not any(t["g3hold_on"] for t in ticks[held[-1] + 1:]), "fired past expiry"
    dur = len(held) * DT
    assert dur == pytest.approx(HOLD_S, abs=DT), f"held {dur:.3f}s, wanted {HOLD_S}"


# ---------------------------------------------------------------------------------
# Arm preconditions -- all four must hold
# ---------------------------------------------------------------------------------
@pytest.mark.parametrize("vis,why", [
    ([(2, True, True, 0.50, 0.05)] * 8, "not high (v_f below vhi, descent never armed)"),
    ([(2, True, True, 0.10, 0.40)] * 8, "too far (sz_f below the descent size)"),
    ([(2, False, True, 0.50, 0.40)] * 8, "gate_v_ok false, descent never active"),
])
def test_it_does_not_arm_without_an_active_high_close_descent(vis, why):
    ticks = replay(vis + BLIND, ON)
    assert not any(t["g3hold_on"] for t in ticks), f"armed anyway: {why}"


def test_it_requires_a_valid_to_invalid_transition():
    """Blind from the very start of the leg -- nothing was ever tracked, so nothing to
    hold."""
    assert not any(t["g3hold_on"] for t in replay(BLIND, ON))


def test_the_last_valid_state_is_what_the_arm_test_reads():
    ticks = replay(VISIBLE + BLIND, ON)
    assert ticks[-1]["g3_last_vf"] == pytest.approx(0.40)
    assert ticks[-1]["g3_last_sz"] == pytest.approx(0.50)


# ---------------------------------------------------------------------------------
# Release paths
# ---------------------------------------------------------------------------------
def test_advancing_to_gate_3_releases_immediately():
    seq = VISIBLE + [(2, False, False, 0.0, 0.0)] * 2 + [(3, False, False, 0.0, 0.0)] * 10
    ticks = replay(seq, ON)
    assert ticks[9]["g3hold_on"], "the hold was not running before the advance"
    assert not any(t["g3hold_on"] for t in ticks[10:]), "kept cutting on the Gate-4 leg"
    assert all(t["thrust"] == BASE for t in ticks[10:])


def test_a_valid_reacquisition_hands_back_to_the_live_path():
    seq = VISIBLE + [(2, False, False, 0.0, 0.0)] * 2 + [(2, True, True, 0.50, 0.05)] * 5
    ticks = replay(seq, ON)
    assert ticks[9]["g3hold_on"]
    tail = ticks[10:]
    assert not any(t["g3hold_on"] for t in tail), "blind hold survived a valid frame"
    assert all(t["thrust"] == BASE for t in tail), (
        "the low reacquisition must not keep descending")


def test_detector_flicker_cannot_restart_or_extend_the_timer():
    """THE LOAD-BEARING ONE. Alternating invalid ticks must not re-arm: the total cut
    must still be bounded by one window, not one window per dropout."""
    seq = VISIBLE + [(2, False, False, 0.0, 0.0)] * 40
    straight = sum(1 for t in replay(seq, ON) if t["g3hold_on"])
    # now with a single spurious HIGH reacquisition partway through the blind coast
    flick = VISIBLE + [(2, False, False, 0.0, 0.0)] * 4 \
        + [(2, True, True, 0.50, 0.40)] + [(2, False, False, 0.0, 0.0)] * 40
    ticks = replay(flick, ON)
    runs, cur = [], 0
    for t in ticks:
        if t["g3hold_on"]:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    assert all(r <= straight for r in runs), (
        f"a dropout restarted the timer: runs {runs} vs bound {straight}")
    assert len(runs) <= 2, (
        f"flicker produced {len(runs)} separate holds; a valid frame may hand back to "
        f"the live path but repeated dropouts must not each buy a fresh window")


# ---------------------------------------------------------------------------------
# Scope and discipline
# ---------------------------------------------------------------------------------
@pytest.mark.parametrize("ag", [0, 1, 3, 4])
def test_it_never_fires_off_leg_2(ag):
    seq = [(ag, True, True, 0.50, 0.40)] * 8 + [(ag, False, False, 0.0, 0.0)] * 20
    ticks = replay(seq, ON)
    assert not any(t["g3hold_on"] for t in ticks)
    assert all(t["thrust"] == BASE for t in ticks)
    assert all(t["g3hold_armed"] == 0 for t in ticks)


def test_off_by_default_is_a_no_op():
    on_off = replay(VISIBLE + BLIND, OFF)
    assert not any(t["g3hold_on"] for t in on_off)
    plain = replay(VISIBLE + BLIND, build_parser().parse_args([]))
    assert all(t["thrust"] == BASE for t in plain)


def test_defaults_are_off_and_as_specified():
    d = vars(build_parser().parse_args([]))
    assert d["gate3_vert_descent_hold_s"] == 0.0, "the hold must ship opt-in"
    assert d["gate3_vert_descent_hold_release_vf"] == 0.10


def test_the_hold_cannot_dig_under_the_open_loop_floor():
    cfg = ServoConfig()
    ticks = replay(VISIBLE + BLIND, ON, thrust=cfg.ol_thrust_lo + 0.005)
    assert min(t["thrust"] for t in ticks) == pytest.approx(cfg.ol_thrust_lo)


def test_the_first_flight_value_is_shorter_than_the_replay_value():
    """0.20 s for the first test, against replay's 0.30 -- the conservative direction:
    a short hold under-corrects, a long one risks the low miss the plant cannot recover
    from. Also comfortably under the 0.38 s earliest spurious reacquisition."""
    assert HOLD_S < 0.30
    assert HOLD_S < 0.38
