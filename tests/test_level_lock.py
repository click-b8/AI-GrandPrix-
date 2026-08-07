"""LEVEL-LOCK (--gate-vert-level-band): stop descending once the gate is centred.

THE DEFECT. The altitude ladder applies a CONSTANT descent per leg -- `base_thrust -=
descent_bias[leg]` -- and it never lets up. Once the gate is vertically centred (v_f ~ 0)
the drone is AT the gate's height and should hold it, but the ladder keeps sinking, so it
arrives at the gate plane still going down and passes below the middle. The only thing
opposing that is the v-trim, working against a descent that never yields, inside its own
bounded authority.

THE FIX. Scale the applied descent by how far the gate still sits below centre:

    descent_gain = min(1, |v_f| / band)      # 1 a full band below centre, 0 when centred
    base_thrust -= leg_bias * descent_gain

Full ladder descent while the drone is still high, easing linearly to LEVEL as v_f -> 0.

ONE-SIDED BY DESIGN. Applied only for v_f >= 0 (gate at or below centre). v_f < 0 means
the gate is ABOVE centre -- the drone is already low -- and cutting the descent there
would be suppressing lift the v-trim is separately trying to add, so that case keeps
normal behaviour.

SIGN REMINDER (this is the axis that has cost the most): v_err > 0 == gate BELOW frame
centre == camera looking DOWN at it == drone too HIGH.

DEFAULT OFF (band 0.0), which is also the divide-by-zero guard.
"""
import csv
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "filt9.csv")

BAND = 0.30

pytestmark = pytest.mark.skipif(not os.path.exists(REF), reason="filt9.csv not present")


def _rows():
    with open(REF, encoding="utf-8") as fh:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def _gain(v_f, gate_v_ok, descent_on, band=BAND):
    """The shipped level-lock gain, isolated."""
    if band > 0.0 and gate_v_ok and descent_on and v_f >= 0.0:
        return min(1.0, abs(v_f) / band)
    return 1.0


def _applied(row, band=BAND, leg_bias=0.026):
    """Descent actually applied on a logged tick."""
    gate_v_ok = bool(row["gate_v_ok"])
    g = _gain(row["v_f"], gate_v_ok, leg_bias > 0.0, band)
    return leg_bias * g, g


# --------------------------------------------------------------------- off by default

def test_band_zero_is_off_and_cannot_divide_by_zero():
    for r in _rows():
        applied, g = _applied(r, band=0.0)
        assert g == 1.0
        assert applied == pytest.approx(0.026)


def test_off_leaves_every_logged_tick_at_full_descent():
    assert all(_applied(r, band=0.0)[1] == 1.0 for r in _rows())


# ------------------------------------------------------------------ the gain curve

@pytest.mark.parametrize("v_f,expect", [
    (0.00, 0.0),        # centred -> level off completely
    (0.075, 0.25),
    (0.15, 0.5),        # half a band below centre -> half descent
    (0.30, 1.0),        # a full band below -> full ladder descent
    (0.90, 1.0),        # and it saturates, never exceeds 1
])
def test_gain_curve(v_f, expect):
    assert _gain(v_f, True, True) == pytest.approx(expect)


def test_gain_is_never_above_one_or_below_zero():
    for r in _rows():
        g = _applied(r)[1]
        assert 0.0 <= g <= 1.0


def test_gain_is_monotonic_in_v_f():
    xs = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    gs = [_gain(v, True, True) for v in xs]
    assert gs == sorted(gs)


def test_a_wider_band_holds_the_descent_off_further_out():
    """The knob's meaning: a larger band starts levelling earlier."""
    v = 0.2
    assert _gain(v, True, True, band=0.1) > _gain(v, True, True, band=0.4)


# --------------------------------------------------------- the one-sided condition

def test_negative_v_f_keeps_normal_behaviour():
    """Drone already low (gate above centre): do NOT cut the descent -- that would be
    suppressing lift the v-trim is separately adding."""
    for v in (-0.01, -0.2, -0.9):
        assert _gain(v, True, True) == 1.0


def test_zero_is_treated_as_centred_not_as_negative():
    assert _gain(0.0, True, True) == 0.0


def test_the_asymmetry_is_real_on_the_flight_log():
    """Both signs occur in flt9, so the one-sided rule is actually exercised."""
    rows = [r for r in _rows() if bool(r["gate_v_ok"])]
    assert any(r["v_f"] > 0.05 for r in rows), "no gate-below-centre ticks"
    assert any(r["v_f"] < -0.05 for r in rows), "no gate-above-centre ticks"
    faded = [r for r in rows if _applied(r)[1] < 1.0]
    assert faded, "the level-lock never engages on this log"
    assert all(r["v_f"] >= 0.0 for r in faded), "faded on a negative v_f"


# ----------------------------------------------------------------- gating and scope

def test_requires_a_live_gate():
    """No gate -> no idea where centre is -> fly the ladder as written."""
    assert _gain(0.0, False, True) == 1.0


def test_requires_a_descending_leg():
    """A flat leg has no descent to fade; the gain must not manufacture one."""
    assert _gain(0.0, True, False) == 1.0
    # and on a near-flat ladder leg the applied descent stays ~0 either way
    applied, _ = _applied({"gate_v_ok": 1.0, "v_f": 0.0}, leg_bias=0.0)
    assert applied == 0.0


@pytest.mark.parametrize("ag", [0, 1, 2])
def test_applies_on_every_leg_no_ag_fence(ag):
    """All active_gate values, unlike the lateral commit's ag>=2 fence."""
    rows = [r for r in _rows()
            if int(r["active_gate"]) == ag and bool(r["gate_v_ok"])]
    if not rows or not any(r["v_f"] >= 0.0 for r in rows):
        pytest.skip(f"ag={ag} has no gate-below-centre ticks")
    assert any(_applied(r)[1] < 1.0 for r in rows), f"never engaged at ag={ag}"


# ------------------------------------------------------ what it does to the flight

def test_it_removes_descent_exactly_where_the_gate_is_centred():
    """The headline behaviour, on real ticks: near centre the applied descent collapses
    while far from centre it is untouched."""
    rows = [r for r in _rows() if bool(r["gate_v_ok"]) and r["v_f"] >= 0.0]
    near = [r for r in rows if r["v_f"] < 0.05]
    far = [r for r in rows if r["v_f"] >= BAND]
    assert near and far, "need both near-centre and far-from-centre ticks"
    assert max(_applied(r)[0] for r in near) < 0.2 * 0.026
    assert all(_applied(r)[0] == pytest.approx(0.026) for r in far)


def test_total_descent_commanded_is_reduced_not_increased():
    """Sanity on direction: the lock can only ever REMOVE descent, never add any."""
    for r in _rows():
        assert _applied(r)[0] <= 0.026 + 1e-12
