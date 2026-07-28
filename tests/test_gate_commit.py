"""CLOSE-RANGE COMMIT acceptance, replayed on filt9's gate-3 leg.

THE DEFECT BEING FIXED. A gate blob LUNGES sideways from parallax as you close on it, so
past a certain size its u_err stops meaning "you are off to one side" and starts meaning
"you are nearly through it". On filt9's gate-3 approach the filtered lateral error runs

    t=10.24  u_f +0.080   des_roll  -4.6      (honest: still far, still centring)
    t=10.60  u_f +0.219   des_roll -11.0      (clamp)
    t=10.90  u_f +0.451   des_roll -11.0
    t=11.40  u_f +0.674   des_roll -11.0

-- 30 of the 46 close-range ticks pinned to the -11 deg clamp, banking hard sideways at
the exact moment the drone should be flying straight through. It veered off before gate 3.

SCOPE. This file replays the ag=2 leg only, where the commit is in scope. The ag >= 2
fence itself -- and the guarantee that gates 1 & 2 are untouched -- is test_commit_scope.py.

THE FIX. Above --gate-commit-size the lateral command is slewed to wings-level over
--gate-commit-tau: hold the approach heading through the last stretch. It is a weight, not
a latch -- drop back below the size and the same lag ramps authority back in.

WHAT THIS MEASURES. filt9.csv logs u_f and sz_f every tick, and with kd_lat=0 the flown
law is exactly clamp(k_gate_bank * LATERAL_SIGN * u_f), which reproduces the logged
des_roll_deg. So the commit can be replayed at the real tick timing on the real flight.
The lag is the shipped lag_step().

HEADLINE at the 0.25 default: after the commit engages, 0 of 23 ticks reach the clamp
(against 23 of 23 without it), peak 8.9 deg, and 6.6 deg one tau in.
"""
import csv
import math
import os

import pytest

from vq1_vision_servo import LATERAL_SIGN
from tools.schedule_flier import lag_step

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "filt9.csv")

GATE3_LEG_AG = 2
CLOSE_LO, CLOSE_HI = 10.0, 11.6     # the gate-3 close approach
CLAMP_DEG = 11.0                    # --gate-max-bank-deg as flt9 flew it
COMMIT_SIZE = 0.25                  # --gate-commit-size default
COMMIT_TAU = 0.2                    # --gate-commit-tau default

pytestmark = pytest.mark.skipif(not os.path.exists(CSV),
                                reason="filt9.csv (the flt9 flight log) not present")


def _leg():
    with open(CSV, encoding="utf-8") as fh:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    return [r for r in rows if r["active_gate"] == GATE3_LEG_AG]


def _replay(rows, commit_size=COMMIT_SIZE, tau=COMMIT_TAU, k_gate=1.0,
            clamp_deg=CLAMP_DEG):
    """[(t, raw_deg, commanded_deg, commit_k, committed, row)] at real tick timing."""
    clamp = math.radians(clamp_deg)
    k, last_t, out = 1.0, None, []
    for r in rows:
        t = r["t"]
        dt = min(max(t - last_t, 0.0), 0.5) if last_t is not None else 0.0
        last_t = t
        committed = bool(commit_size > 0.0 and r["sz_f"] >= commit_size)
        k = lag_step(k, dt, 0.0 if committed else 1.0, tau)
        raw = max(-clamp, min(clamp, k_gate * LATERAL_SIGN * r["u_f"]))
        out.append((t, math.degrees(raw), math.degrees(k * raw), k, committed, r))
    return out


def _approach(seq):
    return [x for x in seq if CLOSE_LO <= x[0] <= CLOSE_HI]


def _pinned(xs, idx):
    return sum(1 for x in xs if abs(x[idx]) > CLAMP_DEG - 0.05)


def test_replay_reproduces_the_flown_command():
    """Guard on the METHOD: with the commit disabled the reconstruction must match what
    was actually commanded in flight, or nothing below measures the real law."""
    for _, raw, cmd, _, _, r in _approach(_replay(_leg(), commit_size=0.0)):
        assert raw == pytest.approx(r["des_roll_deg"], abs=0.05)
        assert cmd == pytest.approx(raw, abs=1e-9)


def test_the_uncommitted_law_pinned_the_clamp():
    """Baseline -- the defect, measured."""
    app = _approach(_replay(_leg(), commit_size=0.0))
    assert _pinned(app, 1) / len(app) > 0.5
    assert max(abs(x[1]) for x in app) == pytest.approx(CLAMP_DEG, abs=0.05)


def test_peak_command_after_commit_drops_well_below_the_clamp():
    """THE HEADLINE. Once sz_f crosses the commit size, the command comes off the clamp
    and keeps falling -- the drone stops chasing the gate sideways."""
    app = _approach(_replay(_leg()))
    engaged = [x for x in app if x[4]]
    assert engaged, "the commit never engages on this approach"
    t0 = min(x[0] for x in engaged)
    after = [x for x in app if x[0] >= t0]

    assert _pinned(after, 2) == 0, "commanded bank still reaches the clamp after commit"
    assert _pinned(after, 1) == len(after), \
        "without the commit these same ticks were NOT all pinned -- window drifted"
    assert max(abs(x[2]) for x in after) < 10.0

    # and one time-constant in, it is far below -- this is the 'over ~0.2 s' slew
    one_tau = [x for x in after if x[0] >= t0 + COMMIT_TAU]
    assert one_tau, "leg ends before one tau elapses"
    assert max(abs(x[2]) for x in one_tau) < 8.0


def test_commit_cuts_clamp_time_over_the_whole_approach():
    """Measured across the full close-approach window, not just post-engagement. The peak
    is unchanged (the clamp is hit BEFORE sz_f reaches 0.25 -- the commit is a terminal
    guard, not a fix for the mid-approach rush), but the time spent pinned collapses."""
    app_off = _approach(_replay(_leg(), commit_size=0.0))
    app_on = _approach(_replay(_leg()))
    assert _pinned(app_on, 2) < _pinned(app_off, 1) / 3.0


def _peak_rate(seq, idx=2):
    """deg/s, not deg/tick: the log has ticks from 8 ms to 47 ms, so a per-tick bound
    would just be measuring tick length."""
    return max(abs(b[idx] - a[idx]) / max(b[0] - a[0], 1e-6) for a, b in zip(seq, seq[1:]))


def test_commit_retires_the_bank_at_a_rate_tau_bounds():
    """The commit DOES move the command faster than the uncommitted law does -- it has
    to, it is retiring a bank that is sitting on the clamp (measured: 46 deg/s against
    the pinned law's 23). What matters is that the rate is BOUNDED and set by tau, not
    by how hard the gate happens to be sweeping. Ceiling is clamp/tau = 11/0.2 = 55 deg/s.
    """
    rate = _peak_rate(_approach(_replay(_leg())))
    assert rate < CLAMP_DEG / COMMIT_TAU, f"{rate:.0f} deg/s exceeds the clamp/tau ceiling"


def test_the_lag_is_what_makes_it_a_slew():
    """tau=0 is the same rule without the lag: a genuine STEP off the clamp. Measured,
    the lag takes 451 deg/s down to 46 -- very nearly an order of magnitude."""
    slewed = _peak_rate(_approach(_replay(_leg())))
    stepped = _peak_rate(_approach(_replay(_leg(), tau=0.0)))
    assert stepped > 8.0 * slewed, f"lag bought nothing: {stepped:.0f} vs {slewed:.0f} deg/s"


def test_a_longer_tau_retires_the_bank_more_gently():
    rates = [_peak_rate(_approach(_replay(_leg(), tau=t))) for t in (0.1, 0.2, 0.4, 0.8)]
    assert rates == sorted(rates, reverse=True), f"not monotonic in tau: {rates}"


def test_commit_weight_never_steps():
    app = _approach(_replay(_leg()))
    assert max(abs(b[3] - a[3]) for a, b in zip(app, app[1:])) < 0.25


def test_commit_is_reversible_not_a_latch():
    """Below the commit size the same lag must ramp authority back IN -- otherwise one
    close pass would disarm steering for the rest of the flight."""
    seq = _replay(_leg())
    engaged = [x for x in seq if x[4]]
    assert engaged
    t_last = max(x[0] for x in engaged)
    recovered = [x for x in seq if x[0] > t_last + 5.0 * COMMIT_TAU]
    assert recovered, "leg ends before the ramp-back can be observed"
    assert max(x[3] for x in recovered) > 0.95, "authority never came back"


def test_commit_weight_actually_reaches_near_zero():
    seq = _replay(_leg())
    assert min(x[3] for x in seq) < 0.2


def test_zero_commit_size_disables_the_feature():
    seq = _replay(_leg(), commit_size=0.0)
    assert all(x[3] == 1.0 for x in seq)
    assert not any(x[4] for x in seq)


def test_a_smaller_commit_size_commits_harder():
    """Monotonicity, so the knob behaves the way the tuning loop assumes. Measured as
    TIME ON THE CLAMP over the whole approach -- peak alone is not monotonic, because
    down to a commit size of 0.20 the clamp is still reached BEFORE the commit engages,
    so every one of those settings shares the same 11.0 deg peak."""
    pins = [_pinned(_approach(_replay(_leg(), commit_size=cs)), 2)
            for cs in (0.0, 0.30, 0.25, 0.20, 0.15)]
    assert pins == sorted(pins, reverse=True), f"not monotonic in commit size: {pins}"
    assert pins[0] > 25 and pins[-1] == 0


def test_lag_step_primitive():
    """The shared lag: symmetric, never steps, honours tau<=0 as instant."""
    assert lag_step(1.0, 0.0, 0.0, 0.2) == 1.0          # no time, no movement
    assert lag_step(1.0, 999.0, 0.0, 0.0) == 0.0        # tau 0 = instant
    k = 1.0
    for _ in range(25):                                  # 25 ticks x 8 ms = 0.2 s = 1 tau
        k = lag_step(k, 0.008, 0.0, 0.2)
    assert 0.30 < k < 0.42                               # ~exp(-1)
    assert lag_step(0.0, 0.008, 1.0, 0.2) > 0.0          # ramps back in symmetrically
