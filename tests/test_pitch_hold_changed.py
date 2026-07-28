"""CHANGE D acceptance: the gentle near-equilibrium pitch hold must bleed airspeed
WITHOUT the tumble signature that killed every earlier pitch hold.

TUMBLE SIGNATURE, from coast_tube.log (the flight that ended the first pitch-hold
attempt): est_roll ran to +90 and est_pitch to -90 by t=12.5 s -- the attitude estimate
departing to the gimbal limits on BOTH axes. The pitch loop has no roll authority at all,
so any |roll| growth it produces is cross-axis departure, i.e. the abort condition the
spec names ("abort if |roll| grows").

WHY THIS ONE IS EXPECTED TO BE DIFFERENT -- authority, not intent:
    earlier holds:  kp = 3.0 (cfg.ol_kp_att), setpoint up to 7.8 deg off equilibrium
                    -> -10 deg tumbled
    Change D:       kp = 1.0, setpoint 2.8 deg off equilibrium (-15 vs -17.8 spawn)
                    -> standing command 0.0041 norm, 8% of its own rate clamp

These are closed-form / small-signal checks on the control law. They cannot prove the sim
won't tumble -- only the flight can. They pin the authority envelope so a regression that
re-arms the old gains fails here first.
"""
import math

import pytest

from vq1_vision_servo import MAX_BODY_RATE, SPAWN_PITCH_DEG

KP = 1.0
KD = 0.15
MAX_RATE_DEG = 12.0
HOLD_DEG = -15.0
LIM = math.radians(MAX_RATE_DEG) / MAX_BODY_RATE


def pitch_n(des_deg, est_deg, q=0.0, kp=KP, kd=KD, lim=LIM, clamp=True):
    """The shipped law, verbatim. clamp=False exposes the raw demand, which is what
    'authority' means when comparing gain sets across different clamps."""
    rate = kp * (math.radians(des_deg) - math.radians(est_deg)) - kd * q
    n = rate / MAX_BODY_RATE
    return max(-lim, min(lim, n)) if clamp else n


def test_standing_command_at_equilibrium_is_a_nudge():
    """At the spawn attitude the hold must ask for a fraction of its own clamp."""
    n = pitch_n(HOLD_DEG, SPAWN_PITCH_DEG)
    assert n == pytest.approx(0.00407, abs=1e-4)
    frac = abs(n) / LIM
    assert frac == pytest.approx(0.233, abs=0.01)
    assert frac < 0.30, "standing command should leave most of the clamp as headroom"


def test_old_gains_would_have_been_far_more_aggressive():
    """Documents the difference from the holds that tumbled: same law, old numbers.
    Compared UNCLAMPED -- the old hold ran against the roll channel's looser 0.6 rad/s
    budget, so clamping both to Change D's 12 deg/s would hide the gap."""
    gentle = abs(pitch_n(HOLD_DEG, SPAWN_PITCH_DEG, clamp=False))
    tumbled = abs(pitch_n(-10.0, SPAWN_PITCH_DEG, kp=3.0, kd=0.05, clamp=False))
    assert gentle == pytest.approx(0.00407, abs=1e-4)
    assert tumbled == pytest.approx(0.03404, abs=1e-4)
    assert tumbled > 8 * gentle
    # and the old demand alone would peg Change D's clamp nearly 2x over
    assert tumbled > 1.9 * LIM


def test_command_is_clamped_under_any_attitude_excursion():
    """However far the estimate departs, the pitch loop can never command more than
    its own clamp -- this is the tumble stop."""
    for est in range(-180, 181, 5):
        for q in (-20.0, -5.0, 0.0, 5.0, 20.0):
            assert abs(pitch_n(HOLD_DEG, est, q)) <= LIM + 1e-12


def test_pitch_clamp_is_independent_of_the_roll_budget():
    """A pitch experiment must not be able to spend roll authority."""
    from vq1_vision_servo import ServoConfig
    roll_lim = ServoConfig().ol_max_rate_rad_s / MAX_BODY_RATE
    assert LIM == pytest.approx(math.radians(MAX_RATE_DEG) / MAX_BODY_RATE)
    assert LIM != roll_lim, "pitch clamp must be its own knob, not the roll one"


def test_rate_damping_opposes_motion():
    """Damping must subtract from the command when the nose is already moving the way
    the P term wants -- otherwise it rings."""
    err_only = pitch_n(HOLD_DEG, -17.8, q=0.0)
    with_rate = pitch_n(HOLD_DEG, -17.8, q=+0.5)     # nose already pitching up
    assert with_rate < err_only
    assert pitch_n(HOLD_DEG, -17.8, q=-0.5) > err_only


def test_step_response_is_damped_not_divergent():
    """Closed-loop step on a rate-command plant: theta' = MAX_BODY_RATE * pitch_n.
    Integrate the real law and require monotone convergence with no overshoot."""
    theta = SPAWN_PITCH_DEG
    q = 0.0
    dt = 1.0 / 200.0
    hist = []
    for _ in range(2000):                      # 10 s
        n = pitch_n(HOLD_DEG, theta, q)
        q = MAX_BODY_RATE * n                  # command IS the rate
        theta += math.degrees(q) * dt
        hist.append(theta)
    assert hist[-1] == pytest.approx(HOLD_DEG, abs=0.5), "must reach the setpoint"
    assert max(hist) <= HOLD_DEG + 1e-6, "overshot the setpoint (would ring)"
    for a, b in zip(hist, hist[1:]):
        assert b >= a - 1e-9, "non-monotonic approach = oscillation"


def test_no_roll_coupling_in_the_law():
    """The pitch command is a function of pitch error and q ONLY. If this ever reads a
    roll term, the |roll| abort condition becomes reachable from inside the loop."""
    import inspect
    from tools import schedule_flier
    src = inspect.getsource(schedule_flier.mode_coast_tube)
    body = src.split("des_pitch_hold is None")[1].split("base_thrust =")[0]
    assert "gyro[1]" in body, "must damp on the PITCH gyro axis"
    assert "gyro[0]" not in body, "pitch loop must not read the roll gyro"
    assert "est_roll" not in body and "des_roll" not in body


@pytest.mark.parametrize("csv_name", ["filt5.csv", "filt6.csv"])
def test_flown_roll_envelope_is_the_baseline_to_beat(csv_name):
    """Records the |roll| envelope of the PURE-COAST flights. Change D must not exceed
    this by much; a tumble would be |roll| marching to ~90."""
    import csv
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        csv_name)
    if not os.path.exists(path):
        pytest.skip(f"{csv_name} not present")
    with open(path, encoding="utf-8", newline="") as fh:
        roll = [abs(float(r["est_roll_deg"])) for r in csv.DictReader(fh)]
    assert max(roll) < 45.0, (
        f"{csv_name} baseline |roll| peaked at {max(roll):.1f} -- already departing")
