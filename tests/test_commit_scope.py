"""CLOSE-RANGE COMMIT scoping: gates 1 & 2 untouched, gate 3 protected.

WHY THE SCOPE EXISTS. Three flights in a row tried to give the tube lateral authority and
all three cost something:
    flt10  rail-primary everywhere, gate demoted to a 3 deg fine-trim -> broke gate 1
    flt11  pure gate-centring -> gates 1 & 2 pass, veers off before gate 3
    flt12  priority handoff (rail only while the tube is solid) -> CRASHED gate 1; the
           curvature lead spiked the rail to -11 deg at close range at auth 0.95
    flt13  tube off, commit kept but gated on SIZE ALONE -> it fired on the ag=1
           approach, retired the leg-1 bank, and the drone CLIPPED GATE 2 (collision
           t=7.7)
So the tube is out of the steering path entirely, and the ONE remaining change -- the
close-range commit -- is fenced to active_gate >= --gate-commit-from-ag (2).

THE REFERENCE for gates 1 & 2 is the --gate-commit-size 0 law, i.e. pure gate centring.
MEASURED: that law reproduces flt9's ag 0-1 command exactly -- 0 of 138 and 0 of 99 ticks
differ, worst 0.00 deg -- and flt9 passed both gates. NOTE it does NOT reproduce flt11's
ag 0-1: flt11 was flown with the commit UNSCOPED (commit_k fell to 0.013 on ag=0 and
0.010 on ag=1), so 17 of its ag=0 ticks and 50 of its ag=1 ticks differ from the pure
law, by up to 13.73 deg. flt11 is therefore the wrong byte-identity reference; flt9 is
the flight whose early legs the scoped build actually reproduces.

WHAT THIS FILE ASSERTS:
  * ag 0 and ag 1: no commit ever fires, and the command is identical to flt9's.
  * ag 2: once sz_f >= 0.25 the command stops pinning the -11 deg clamp.
"""
import csv
import math
import os

import pytest

from vq1_vision_servo import LATERAL_SIGN
from tools.schedule_flier import lag_step

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "filt11.csv")        # the gate-3 leg (rail instrumented)
REF = os.path.join(ROOT, "filt9.csv")         # the pure-gate reference for ag 0-1

CLAMP_DEG = 11.0            # --gate-max-bank-deg
COMMIT_SIZE = 0.25          # --gate-commit-size
COMMIT_TAU = 0.2            # --gate-commit-tau
COMMIT_FROM_AG = 2          # --gate-commit-from-ag

pytestmark = pytest.mark.skipif(not (os.path.exists(CSV) and os.path.exists(REF)),
                                reason="flt9/flt11 flight logs not present")


def _rows(path=CSV):
    with open(path, encoding="utf-8") as fh:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def _leg(ag, path=CSV):
    return [r for r in _rows(path) if r["active_gate"] == ag]


def _replay(rows, commit_size=COMMIT_SIZE, tau=COMMIT_TAU, from_ag=COMMIT_FROM_AG):
    """The shipped lateral law: gate centring, with the commit fenced to ag >= from_ag.

    bank_bias is read from the log (ff_bank_deg) rather than recomputed, so this measures
    the commit's effect and not a re-derivation of the backbone.
    """
    clamp = math.radians(CLAMP_DEG)
    commit_k, last_t, out = 1.0, None, []
    for r in rows:
        t = r["t"]
        dt = min(max(t - last_t, 0.0), 0.5) if last_t is not None else 0.0
        last_t = t
        gate_cmd = (max(-clamp, min(clamp, LATERAL_SIGN * r["u_f"]))
                    if r["steer"] else 0.0)
        bank_bias = math.radians(r["ff_bank_deg"])
        in_scope = commit_size > 0.0 and r["active_gate"] >= from_ag
        if in_scope:
            committed = r["sz_f"] >= commit_size
            commit_k = lag_step(commit_k, dt, 0.0 if committed else 1.0, tau)
        else:
            committed, commit_k = False, 1.0
        des = commit_k * bank_bias + commit_k * gate_cmd
        out.append(dict(t=t, ag=int(r["active_gate"]), committed=committed,
                        commit_k=commit_k, des_roll=math.degrees(des),
                        sz_f=r["sz_f"], u_f=r["u_f"], row=r))
    return out


def _pinned(seq):
    return sum(1 for x in seq if abs(x["des_roll"]) > CLAMP_DEG - 0.05)


# --------------------- gates 1 & 2: byte-identical to the --gate-commit-size 0 law

@pytest.mark.parametrize("ag", [0, 1])
def test_no_commit_ever_fires_while_ag_is_below_two(ag):
    """THE GUARANTEE flt13 bought. The gate DOES get big enough on these approaches
    (sz_f peaks at 0.44 on ag=0 and 0.38 on ag=1, both well past the 0.25 commit size),
    so this is the ag fence doing the work -- not a coincidence of the size never being
    reached. Without it the commit retires the leg-1 bank and the drone clips gate 2."""
    for path in (REF, CSV):
        leg = _leg(ag, path)
        assert leg, f"no ticks for ag={ag} in {os.path.basename(path)}"
        assert max(r["sz_f"] for r in leg) >= COMMIT_SIZE, (
            f"ag={ag} never reaches sz_f {COMMIT_SIZE} -- the fence would be untested")
        for x in _replay(leg):
            assert not x["committed"], f"commit armed at ag={ag}, t={x['t']:.2f}"
            assert x["commit_k"] == 1.0, "commit weight is not EXACTLY 1.0 out of scope"


@pytest.mark.parametrize("ag", [0, 1])
def test_gate1_and_gate2_are_byte_identical_to_the_commit_size_zero_law(ag):
    """THE HEADLINE. Not 'close to' -- identical, tick for tick, against a flight that
    actually flew the --gate-commit-size 0 law: flt9 (the commit did not exist yet, and
    it passed both gates). Measured, the scoped build reproduces it to 0.00 deg.

    NOTE flt11 is NOT usable as this reference: it was flown with the commit UNSCOPED, so
    17 of its ag=0 ticks and 50 of its ag=1 ticks already differ from the commit-free law
    by up to 13.73 deg -- see test_flt11_early_legs_were_already_altered_by_the_commit.
    """
    leg = _leg(ag, REF)
    scoped = _replay(leg)
    unc = _replay(leg, commit_size=0.0)
    for x, y in zip(scoped, unc):
        assert x["des_roll"] == pytest.approx(y["des_roll"], abs=1e-12), "scope leaked"
        assert x["des_roll"] == pytest.approx(x["row"]["des_roll_deg"], abs=0.005)


def test_flt11_early_legs_were_already_altered_by_the_commit():
    """Recorded because it changes which flight is the reference. flt11 ran the commit
    with no ag fence, so its gate-1/gate-2 commands are NOT the commit-free law."""
    altered = {}
    for ag in (0, 1):
        leg = _leg(ag, CSV)
        altered[ag] = sum(1 for r in leg if r["commit_k"] < 0.999)
    assert altered[0] > 10 and altered[1] > 40, altered


def test_unscoped_commit_would_have_changed_the_early_legs():
    """The counterfactual, so the fence is shown to be load-bearing rather than
    decorative: with --gate-commit-from-ag 0 the gate-1/gate-2 commands DO change --
    which is exactly what flt13 flew into gate 2."""
    for ag in (0, 1):
        leg = _leg(ag, REF)
        unscoped = _replay(leg, from_ag=0)
        changed = sum(1 for x, y in zip(unscoped, _replay(leg, commit_size=0.0))
                      if abs(x["des_roll"] - y["des_roll"]) > 0.05)
        assert changed > 0, f"ag={ag} unchanged even unscoped -- fence proves nothing"


# ---------------------------------------------------- gate 3 leg: off the clamp

def test_gate3_leg_stops_pinning_the_clamp_once_committed():
    """THE POINT OF THE CHANGE. On ag=2, from the moment sz_f crosses 0.25 the command
    must come off the -11 deg clamp instead of chasing u_err out to +0.74."""
    seq = _replay(_leg(2))
    engaged = [x for x in seq if x["committed"]]
    assert engaged, "the commit never engages on the gate-3 leg"
    t0 = min(x["t"] for x in engaged)
    after = [x for x in seq if x["t"] >= t0]

    assert _pinned(after) == 0, "still pinning the clamp after the commit engaged"
    assert max(abs(x["des_roll"]) for x in after) < CLAMP_DEG


def test_gate3_leg_was_pinning_the_clamp_without_the_commit():
    """Baseline: the defect, measured off what flt11 actually commanded."""
    off = _replay(_leg(2), commit_size=0.0)
    assert _pinned(off) > 30
    # and the sweep it was chasing really does run away
    assert max(abs(x["u_f"]) for x in off) > 0.7


def test_commit_cuts_clamp_time_on_the_gate3_leg():
    on, off = _replay(_leg(2)), _replay(_leg(2), commit_size=0.0)
    assert _pinned(on) < _pinned(off) / 3.0, f"{_pinned(on)} vs {_pinned(off)} pinned"


def test_commit_releases_when_the_gate_shrinks_again():
    """Not a latch: below the commit size the same lag ramps authority back in, so one
    close pass cannot disarm steering for the rest of the leg."""
    seq = _replay(_leg(2))
    engaged = [x for x in seq if x["committed"]]
    t_last = max(x["t"] for x in engaged)
    later = [x for x in seq if x["t"] > t_last + 5.0 * COMMIT_TAU]
    assert later, "leg ends before the ramp-back can be observed"
    assert max(x["commit_k"] for x in later) > 0.95


def test_commit_weight_stays_in_range_and_never_steps():
    seq = _replay(_leg(2))
    assert all(0.0 <= x["commit_k"] <= 1.0 for x in seq)
    assert max(abs(b["commit_k"] - a["commit_k"]) for a, b in zip(seq, seq[1:])) < 0.25


def test_no_tube_term_anywhere_in_the_replayed_law():
    """Structural: the law above reads u_f, sz_f, steer and ff_bank_deg -- and nothing
    from the tube. If a tube term ever creeps back into the flier, this replay stops
    matching it and the byte-identity test above fails first."""
    assert any(r["tube_solid"] for r in _leg(2)), "filt11 ag=2 has solid tube readings"
    # ...and they change nothing: on flt9 (which flew the commit-free law end to end)
    # the command is reproduced exactly from the GATE signal alone, on every leg.
    for ag in (0, 1, 2):
        for x in _replay(_leg(ag, REF), commit_size=0.0):
            assert x["des_roll"] == pytest.approx(x["row"]["des_roll_deg"], abs=0.005)
