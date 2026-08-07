"""VERTICAL close-range commit (--gate-vert-commit-size): freeze the PD near the gate.

THE DEFECT. The vertical trim servos v_f all the way to the gate plane, and the kd_v*dv
term differentiates an already-noisy close-range signal. In the last metres that can spike
the trim into the down-clamp and fly the drone into the gate -- the vertical analogue of
the lateral parallax sweep the lateral COMMIT already handles. At that range the altitude
is made or missed; there is nothing useful left to servo on.

THE FIX. At sz_f >= --gate-vert-commit-size, stop recomputing the trim from v_f/dv: hold
what it had earned and ease it to the leg baseline (0.0 -- i.e. fly the altitude ladder's
own descent) over --gate-vert-commit-tau. Reversible: below the size the live PD resumes
from wherever the trim eased to, so re-engagement is continuous.

SCOPE. EVERY gate, including gate 1 -- deliberately unlike the lateral commit's ag>=2
fence. Gate 1 is exactly where a premature dive kills us, and the vertical axis has no
leg-1 bank for a fence to protect.

DEFAULT OFF (--gate-vert-commit-size 0.0), so an unflagged run is bit-identical.
"""
import csv
import math
import os

import pytest

from tools.schedule_flier import lag_step

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "filt9.csv")

K_THRUST_V, KD_V = 0.16, 0.25       # flt9's --k-thrust-v / --kd-v
DOWN_AUTH, UP_AUTH = 0.045, 0.10    # flt9's --gate-vert-down-auth / --gate-vert-up-auth
TAU_F = 0.2                         # --gate-filter-tau
COMMIT_TAU = 0.20                   # --gate-vert-commit-tau default

pytestmark = pytest.mark.skipif(not os.path.exists(REF), reason="filt9.csv not present")


def _rows():
    with open(REF, encoding="utf-8") as fh:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def _replay(rows, commit_size=0.0, commit_tau=COMMIT_TAU):
    """The shipped vertical-trim law. Returns per-tick dicts.

    `gate_v_ok` is read from the log, so the engagement/latch/fade machinery is exactly
    what flew. filt9 does not log v_f2 (the vertical second-stage EMA that supplies dv),
    so it is RECONSTRUCTED here from the logged v_f/live/sig_k using the flier's own
    rules -- seed on first sight, converge while live, FREEZE through the hold, fade with
    sig_k. test_default_reproduces_the_flown_trim is what proves the reconstruction is
    right: with the commit off it must reproduce the vtrim that actually flew.
    """
    gate_trim, last_t, out = 0.0, None, []
    v_f2, v2_loss, seen = 0.0, 0.0, False
    for r in rows:
        t = r["t"]
        dt = min(max(t - last_t, 0.0), 0.5) if last_t is not None else 0.0
        last_t = t
        live, sig_k = bool(r["live"]), r["sig_k"]
        if live:
            if not seen:
                v_f2, seen = r["v_f"], True
            else:
                a = dt / (TAU_F + dt)
                v_f2 += a * (r["v_f"] - v_f2)
            v2_loss = v_f2
        elif seen and sig_k < 1.0:
            v_f2 = v2_loss * sig_k          # fade; during the hold it stays frozen
        v_committed = commit_size > 0.0 and r["sz_f"] >= commit_size
        gate_v_ok = bool(r["gate_v_ok"])
        if gate_v_ok and v_committed:
            gate_trim = lag_step(gate_trim, dt, 0.0, commit_tau)
        elif gate_v_ok:
            dv = (r["v_f"] - v_f2) / TAU_F
            gate_trim = max(-DOWN_AUTH, min(UP_AUTH,
                                            -(K_THRUST_V * r["v_f"] + KD_V * dv)))
        out.append(dict(t=t, ag=int(r["active_gate"]), sz_f=r["sz_f"],
                        gate_v_ok=gate_v_ok, v_committed=v_committed,
                        trim=gate_trim, row=r))
    return out


def _commit_rate(seq):
    """Peak |d(trim)/dt| over ticks the COMMIT produced, INCLUDING the transition into it
    -- with tau=0 the whole give-up happens on that first tick, so excluding it would
    measure zero. The whole-sequence peak is not usable here: it is dominated by ordinary
    PD motion rather than by this feature."""
    return max((abs(b["trim"] - a["trim"]) / max(b["t"] - a["t"], 1e-6)
                for a, b in zip(seq, seq[1:])
                if b["v_committed"] and a["gate_v_ok"] and b["gate_v_ok"]
                and 0 < b["t"] - a["t"] < 0.1), default=0.0)


def _at_down_clamp(seq):
    return sum(1 for x in seq if x["trim"] <= -DOWN_AUTH + 1e-9)


# ------------------------------------------------------------------- default off

def test_default_is_off_and_never_commits():
    seq = _replay(_rows())
    assert not any(x["v_committed"] for x in seq)


def test_default_reproduces_the_flown_trim():
    """Guard on the METHOD: with the commit off, the replay must match the vtrim flt9
    actually commanded, or nothing below measures the real law."""
    seq = _replay(_rows())
    live = [x for x in seq if x["gate_v_ok"]]
    assert live, "no gate-visible ticks"
    worst = max(abs(x["trim"] - x["row"]["vtrim"]) for x in live)
    assert worst < 1e-4, f"replay diverges from the flown trim by {worst:.2e}"


# ------------------------------------------------------- the defect it addresses

def test_the_live_pd_reaches_the_down_clamp_at_close_range():
    """Baseline: the thing being guarded against, measured. The PD does hit the
    full-authority down-clamp while a gate is large in frame."""
    seq = _replay(_rows())
    close = [x for x in seq if x["gate_v_ok"] and x["sz_f"] >= 0.25]
    assert close, "no close-range gate-visible ticks in filt9"
    assert _at_down_clamp(close) > 0, "the PD never reaches the down-clamp here"


def test_commit_removes_the_down_clamp_at_close_range():
    """THE HEADLINE. Same ticks, commit on: the trim is eased to the baseline instead of
    being driven to full descent authority."""
    on = _replay(_rows(), commit_size=0.25)
    close = [x for x in on if x["gate_v_ok"] and x["sz_f"] >= 0.25]
    assert _at_down_clamp(close) == 0, "still hitting the down-clamp under the commit"
    # the PD did reach the clamp on these same ticks -- that is the whole contrast
    off = _replay(_rows())
    close_off = [x for x in off if x["gate_v_ok"] and x["sz_f"] >= 0.25]
    assert _at_down_clamp(close_off) > 0
    # NOTE the committed trim is not bounded by DOWN_AUTH in magnitude: whatever it had
    # earned before the commit (up to UP_AUTH on the climb side) is held and then eased.
    # What matters is that it is never DRIVEN further down.
    assert min(x["trim"] for x in close) > -DOWN_AUTH


def test_commit_moves_the_trim_toward_zero_not_somewhere_else():
    """'Slew toward the leg baseline' means 0.0 -- fly the ladder's own descent."""
    on = _replay(_rows(), commit_size=0.25)
    runs = [x for x in on if x["v_committed"] and x["gate_v_ok"]]
    assert runs
    for a, b in zip(runs, runs[1:]):
        if b["t"] - a["t"] > 0.5:
            continue                      # different commit episode
        assert abs(b["trim"]) <= abs(a["trim"]) + 1e-12, "trim moved away from baseline"


# ------------------------------------------------------------- scope: ALL gates

def test_commit_applies_on_gate_one():
    """The explicit scope difference from the lateral commit: it must engage at ag==0,
    the gate-1 approach, where a premature dive is fatal."""
    on = _replay(_rows(), commit_size=0.25)
    g1 = [x for x in on if x["ag"] == 0]
    assert g1, "no ag=0 ticks"
    assert any(x["v_committed"] for x in g1), "vertical commit never fired on gate 1"


@pytest.mark.parametrize("ag", [0, 1, 2])
def test_commit_is_not_fenced_by_active_gate(ag):
    """It fires on whichever legs actually reach the size -- no ag fence at all."""
    rows = [r for r in _rows() if int(r["active_gate"]) == ag]
    if not rows or max(r["sz_f"] for r in rows) < 0.25:
        pytest.skip(f"ag={ag} never reaches sz_f 0.25")
    assert any(x["v_committed"] for x in _replay(rows, commit_size=0.25))


# ----------------------------------------------------------------- reversibility

def test_commit_releases_and_the_live_pd_resumes():
    """Not a latch. After the gate shrinks again the trim must once more track the PD."""
    on = _replay(_rows(), commit_size=0.25)
    off = _replay(_rows())
    fired = [i for i, x in enumerate(on) if x["v_committed"]]
    assert fired
    after = [i for i in range(max(fired) + 1, len(on))
             if on[i]["gate_v_ok"] and not on[i]["v_committed"]]
    assert after, "never leaves the commit -- that would be a latch"
    # within a few ticks of release the committed run rejoins the live PD value
    i = after[min(20, len(after) - 1)]
    assert on[i]["trim"] == pytest.approx(off[i]["trim"], abs=1e-9)


def test_release_is_continuous_no_step():
    """Re-engagement resumes from wherever the trim eased to, so nothing jumps."""
    on = _replay(_rows(), commit_size=0.25)
    steps = [abs(b["trim"] - a["trim"]) for a, b in zip(on, on[1:])
             if b["t"] - a["t"] < 0.1]
    off = _replay(_rows())
    base = [abs(b["trim"] - a["trim"]) for a, b in zip(off, off[1:])
            if b["t"] - a["t"] < 0.1]
    assert max(steps) <= max(base) + 1e-9, "the commit made the trim jumpier"


# ----------------------------------------------------------------------- the lag

def test_tau_bounds_how_fast_the_trim_is_given_up():
    """tau=0 is the same rule as a STEP; the lag must be doing real work."""
    slew = _replay(_rows(), commit_size=0.25, commit_tau=0.20)
    step = _replay(_rows(), commit_size=0.25, commit_tau=0.0)

    assert _commit_rate(step) > 2.0 * _commit_rate(slew)


def test_longer_tau_gives_the_trim_up_more_gently():
    rates = [_commit_rate(_replay(_rows(), commit_size=0.25, commit_tau=tau))
             for tau in (0.05, 0.2, 0.8)]
    assert rates == sorted(rates, reverse=True), f"not monotonic in tau: {rates}"


def test_a_smaller_commit_size_commits_earlier():
    firsts = []
    for cs in (0.35, 0.25, 0.15):
        seq = _replay(_rows(), commit_size=cs)
        fired = [x["t"] for x in seq if x["v_committed"]]
        firsts.append(min(fired) if fired else math.inf)
    assert firsts == sorted(firsts, reverse=True), f"not monotonic: {firsts}"
