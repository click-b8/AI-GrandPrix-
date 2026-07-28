"""--post-gate2-bank / --gate-max-bank-ag2: the ag==2 pre-turn, and its hard scope.

WHAT IT IS. A standing feed-forward bank on the ag==2 leg only, summed with the
gate-centring trim and then clamped -- the same mechanism --post-gate1-bank uses on ag==1
to line up the next gate. The point is to pre-turn toward the next gate's KNOWN position
so vision only trims the residual, instead of letting u_err run out to the -11 deg clamp
chasing the terminal parallax sweep.

WHY THE SCOPE IS KEYED TO RAW active_gate. --post-gate1-on-seg swaps the leg-1 predicate
over to the SEGMENT pointer, which the TIME fallback also advances -- mid-approach, while
the current gate is still dead ahead. That is exactly how the close-range commit ended up
firing on the ag==1 approach and clipping gate 2 (flt13). ag == 2 means gate 2 is
CONFIRMED passed, so no timing can let this term touch gates 1 or 2.

DEFAULTS ARE INERT: --post-gate2-bank 0 and --gate-max-bank-ag2 = --gate-max-bank-deg, so
an unflagged run is bit-identical to before on every leg.
"""
import csv
import math
import os

import pytest

from vq1_vision_servo import LATERAL_SIGN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "filt9.csv")

GLOBAL_CLAMP_DEG = 11.0

pytestmark = pytest.mark.skipif(not os.path.exists(REF), reason="filt9.csv not present")


def _rows():
    with open(REF, encoding="utf-8") as fh:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def _lateral(row, post_gate2_bank=0.0, clamp_ag2_deg=None):
    """The shipped lateral law for one tick. Returns (des_roll_deg, bank_bias_deg).

    The EXISTING feed-forward bank is read from the log (`ff_bank_deg`, which is what the
    flier writes for `bank_bias` -- it already contains the schedule backbone, the ag==1
    --post-gate1-bank that flt9 flew at +5, and the CHANGE A ff_lat_k decay). Only the NEW
    ag==2 term is added on top, so with defaults this reproduces the flight exactly.

    In flight the ag==2 term sits inside the same `ff_lat_k * (...)` product as the ag==1
    one, so it decays with CHANGE A identically; the tests below use ff_lat_k = 1 because
    what they measure is WHICH LEG the term lands on, not how it fades.
    """
    ag = int(row["active_gate"])
    leg2 = ag == 2
    clamp = math.radians(clamp_ag2_deg if (leg2 and clamp_ag2_deg is not None)
                         else GLOBAL_CLAMP_DEG)
    gate_cmd = (max(-clamp, min(clamp, LATERAL_SIGN * row["u_f"]))
                if row["steer"] else 0.0)
    bank_bias = math.radians(row["ff_bank_deg"]) + (math.radians(post_gate2_bank)
                                                    if leg2 else 0.0)
    return math.degrees(bank_bias + gate_cmd), math.degrees(bank_bias)


# ------------------------------------------------------------------ the scope

@pytest.mark.parametrize("ag", [0, 1, 2, 3, 4, 5])
def test_preturn_bank_applies_only_at_ag_2(ag):
    """The headline scope check, over every leg present in the log."""
    rows = [r for r in _rows() if int(r["active_gate"]) == ag]
    if not rows:
        pytest.skip(f"no ticks at ag={ag} in filt9.csv")
    deltas = {round(_lateral(r, post_gate2_bank=7.0)[1] - _lateral(r)[1], 6)
              for r in rows}
    if ag == 2:
        assert deltas == {7.0}, f"ag=2 did not get the pre-turn: {deltas}"
    else:
        assert deltas == {0.0}, f"ag={ag} leaked the ag==2 pre-turn: {deltas}"


def test_gates_1_and_2_are_byte_unchanged_by_the_preturn():
    """ag < 2 is the pair that passes today. Setting the leg-2 knobs must not move a
    single tick of it."""
    for r in _rows():
        if int(r["active_gate"]) >= 2:
            continue
        base = _lateral(r)[0]
        with_knobs = _lateral(r, post_gate2_bank=9.0, clamp_ag2_deg=4.0)[0]
        assert with_knobs == pytest.approx(base, abs=1e-12)
        assert base == pytest.approx(r["des_roll_deg"], abs=0.005)


def test_defaults_are_inert_on_every_leg():
    """Unflagged, the command is bit-identical to what flt9 flew, everywhere."""
    for r in _rows():
        assert _lateral(r)[0] == pytest.approx(r["des_roll_deg"], abs=0.005)


# ------------------------------------------------------- sign and summation

def test_positive_adds_bank_in_the_same_direction_as_post_gate1_bank():
    """Same sign convention as the knob it generalises. flt9 flew --post-gate1-bank 5 and
    its logged ag==1 bank bias is positive; a positive --post-gate2-bank must move the
    ag==2 bias the same way."""
    rows = _rows()
    leg1_bias = max(r["ff_bank_deg"] for r in rows if int(r["active_gate"]) == 1)
    assert leg1_bias > 0, "flt9's +5 leg-1 bank should log as a positive bias"
    r2 = next(r for r in rows if int(r["active_gate"]) == 2)
    assert _lateral(r2, post_gate2_bank=5.0)[1] - _lateral(r2)[1] == pytest.approx(5.0)


def test_preturn_is_summed_with_the_trim_then_clamped_not_instead_of_it():
    """It must ADD to vision, not replace it -- otherwise the leg flies open loop."""
    r = next(r for r in _rows() if int(r["active_gate"]) == 2 and r["steer"]
             and abs(r["u_f"]) > 0.05)
    without = _lateral(r)[0]
    with_ff = _lateral(r, post_gate2_bank=6.0)[0]
    assert with_ff == pytest.approx(without + 6.0, abs=1e-9)


# -------------------------------------------------------- the per-leg clamp

def test_ag2_clamp_bounds_vision_only_on_that_leg():
    tight = 4.0
    for r in _rows():
        ag = int(r["active_gate"])
        vision = _lateral(r, clamp_ag2_deg=tight)[0] - _lateral(r, clamp_ag2_deg=tight)[1]
        limit = tight if ag == 2 else GLOBAL_CLAMP_DEG
        assert abs(vision) <= limit + 1e-9, f"ag={ag} vision {vision:.2f} > {limit}"


def test_tight_ag2_clamp_lets_the_feedforward_lead():
    """The stated purpose, measured on the leg where vision saturated: with the clamp
    tightened, the vision term stops dominating and the pre-turn is the larger part of
    the command."""
    swept = [r for r in _rows()
             if int(r["active_gate"]) == 2 and r["steer"] and abs(r["u_f"]) > 0.3]
    assert swept, "no sweep ticks on the ag=2 leg"
    ff = 8.0
    for r in swept:
        total, bias = _lateral(r, post_gate2_bank=ff, clamp_ag2_deg=3.0)
        assert abs(bias) > abs(total - bias), "vision still dominates the pre-turn"


def test_default_ag2_clamp_equals_the_global_clamp():
    """--gate-max-bank-ag2 unset must change nothing, including on ag==2."""
    for r in (r for r in _rows() if int(r["active_gate"]) == 2):
        assert _lateral(r, clamp_ag2_deg=None)[0] == pytest.approx(
            _lateral(r, clamp_ag2_deg=GLOBAL_CLAMP_DEG)[0], abs=1e-12)
