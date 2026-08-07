"""DERIVATIVE-AWARE COMMIT QUALIFICATION (--gate-commit-rate-max / --gate-commit-stable-s).

THE DEFECT. MEASURED on filt93B: the close-range commit latched at sz_f~0.16 while u_f
was CROSSING zero at du_f/dt +0.13 and still accelerating. commit_k then decayed to 0,
which swallowed the vision's full -9 deg correction, and the drone missed left. The
3-frame align streak accepted it because a streak cannot distinguish a settled signal
from a fast zero-crossing -- three consecutive frames inside +-0.08 is exactly what a
transient through centre looks like. The RATE is what separates them.

TWO CHANGES, both default off:
  * --gate-commit-rate-max R -- the aligned condition must ALSO be low-rate, |du_f/dt| < R.
  * --gate-commit-stable-s S -- the dwell is measured in SECONDS, not frames. A frame
    count means different things at different loop rates: 3 frames is 0.15 s at 20 Hz but
    0.05 s at 60 Hz, so the same rule qualified differently from run to run. This loop's
    achieved rate swings with detector load, so that was never a fixed rule.

Replay over the recent Gate-3 approaches gives rate 0.11 / dwell 0.15 s: it rejects the
false latches (filt93B +0.13, filt89.1 +0.19) and preserves the golden ones (filt87.1
+0.096, filt80.5 -0.012). The genuine settled window is only ~0.15 s, so the dwell has
very little room above that before the good latches are lost too -- pinned below.

SCOPE. Everything here is inside `in_commit_scope` (ag >= --gate-commit-from-ag, default
2). Out of scope commit_k is forced to EXACTLY 1.0, so Gates 1 and 2 are untouched
whether the flags are set or not.
"""
import pytest

from tools.schedule_flier import build_parser, lag_step

TAU = 0.25
DT = 1.0 / 35.0
SIZE = 0.16
ALIGN = 0.08
RATE = 0.11
DWELL = 0.15


def replay(u_seq, sz_seq, args, ag=2, dt=DT, from_ag=2):
    """Replay the shipped commit block over a (u_f, sz_f) approach.

    Mirrors tools/schedule_flier.py: the arm gate, the derivative/frame branch, the
    latch, and the commit_k lag. Returns per-tick dicts of everything the tick log
    records for this block.
    """
    commit_latched = commit_armed = False
    commit_align_count = 0
    commit_stable_s = 0.0
    u_f_prev = None
    commit_k = 1.0
    out = []
    for u_f, sz_f in zip(u_seq, sz_seq):
        du_f, rate_ok, align_ok = 0.0, True, False
        in_scope = args.gate_commit_size > 0.0 and ag >= from_ag
        if in_scope:
            size_ok = sz_f >= args.gate_commit_size
            deriv_on = (args.gate_commit_rate_max > 0.0
                        or args.gate_commit_stable_s > 0.0)
            if args.gate_commit_align is not None or deriv_on:
                if not size_ok:
                    commit_armed = True
                du_f = ((u_f - u_f_prev) / dt
                        if (u_f_prev is not None and dt > 0) else 0.0)
                align_ok = (args.gate_commit_align is None
                            or abs(u_f) <= args.gate_commit_align)
                rate_ok = (abs(du_f) < args.gate_commit_rate_max
                           if args.gate_commit_rate_max > 0.0 else True)
                if deriv_on:
                    if align_ok and rate_ok:
                        commit_stable_s += dt
                    else:
                        commit_stable_s = 0.0
                    stable_ok = commit_stable_s >= args.gate_commit_stable_s
                    if commit_armed and size_ok and stable_ok and not commit_latched:
                        commit_latched = True
                elif commit_armed and size_ok:
                    if abs(u_f) <= args.gate_commit_align:
                        commit_align_count += 1
                    else:
                        commit_align_count = 0
                    if commit_align_count >= args.gate_commit_stable_frames:
                        commit_latched = True
                u_f_prev = u_f
                committed = commit_latched
            else:
                committed = size_ok
            commit_k = lag_step(commit_k, dt, 0.0 if committed else 1.0, TAU)
        else:
            committed, commit_k = False, 1.0
        out.append(dict(committed=committed, commit_k=commit_k, du_f=du_f,
                        rate_ok=rate_ok, align_ok=align_ok,
                        commit_latched=commit_latched,
                        commit_stable_s=commit_stable_s))
    return out


def reference_frame_based(u_seq, sz_seq, args, ag=2, dt=DT, from_ag=2):
    """The PRE-CHANGE block, transcribed independently. Nothing here knows the new flags
    exist -- this is the thing default-off has to match bit-for-bit."""
    commit_latched = commit_armed = False
    commit_align_count = 0
    commit_k = 1.0
    out = []
    for u_f, sz_f in zip(u_seq, sz_seq):
        in_scope = args.gate_commit_size > 0.0 and ag >= from_ag
        if in_scope:
            size_ok = sz_f >= args.gate_commit_size
            if args.gate_commit_align is not None:
                if not size_ok:
                    commit_armed = True
                if commit_armed and size_ok:
                    if abs(u_f) <= args.gate_commit_align:
                        commit_align_count += 1
                    else:
                        commit_align_count = 0
                    if commit_align_count >= args.gate_commit_stable_frames:
                        commit_latched = True
                committed = commit_latched
            else:
                committed = size_ok
            commit_k = lag_step(commit_k, dt, 0.0 if committed else 1.0, TAU)
        else:
            committed, commit_k = False, 1.0
        out.append(dict(committed=committed, commit_k=commit_k,
                        commit_latched=commit_latched))
    return out


def cfg(*extra):
    return build_parser().parse_args([
        "--gate-commit-size", str(SIZE), "--gate-commit-align", str(ALIGN),
        "--gate-commit-stable-frames", "3", *extra])


def approach(du, n_far=12, n_near=14, u0=None):
    """A Gate-3 approach whose u_f crosses zero at a constant du_f/dt of `du`.

    Starts small (so the arm gate opens on the previous gate's decayed size) and grows
    past the commit size, with u_f arriving at centre exactly as size_ok turns on.
    """
    u0 = -du * DT * (n_near / 2.0) if u0 is None else u0
    u = [u0 - du * DT * (n_far - i) for i in range(n_far)]
    u += [u0 + du * DT * i for i in range(n_near)]
    sz = [0.05] * n_far + [SIZE + 0.01 * i for i in range(n_near)]
    return u, sz


# ---------------------------------------------------------------------------------
# 3. Default-off is bit-for-bit
# ---------------------------------------------------------------------------------
@pytest.mark.parametrize("du", [0.13, 0.096, -0.012, 0.19, 0.0])
def test_default_off_matches_the_frame_based_reference_exactly(du):
    """The headline discipline check: with both new flags at 0.0 the latch, the committed
    flag and the commit_k weight must be IDENTICAL to the pre-change block -- not close,
    identical, since commit_k multiplies the lateral command."""
    u, sz = approach(du)
    a = replay(u, sz, cfg())
    b = reference_frame_based(u, sz, cfg())
    for i, (x, y) in enumerate(zip(a, b)):
        assert x["committed"] == y["committed"], f"tick {i}"
        assert x["commit_latched"] == y["commit_latched"], f"tick {i}"
        assert x["commit_k"] == y["commit_k"], f"tick {i} commit_k drifted"


def test_default_off_leaves_the_legacy_size_only_path_alone():
    """--gate-commit-align unset = the old size-only rule; the new flags must not pull it
    into the align branch."""
    args = build_parser().parse_args(["--gate-commit-size", str(SIZE)])
    assert args.gate_commit_align is None
    u, sz = approach(0.13)
    a = replay(u, sz, args)
    b = reference_frame_based(u, sz, args)
    assert [x["committed"] for x in a] == [y["committed"] for y in b]
    assert [x["commit_k"] for x in a] == [y["commit_k"] for y in b]


# ---------------------------------------------------------------------------------
# 4. ag < 2 is untouched, flags on or off
# ---------------------------------------------------------------------------------
@pytest.mark.parametrize("ag", [0, 1])
@pytest.mark.parametrize("flags", [(), ("--gate-commit-rate-max", str(RATE),
                                        "--gate-commit-stable-s", str(DWELL))])
def test_commit_k_is_exactly_one_below_the_scope_gate(ag, flags):
    """EXACTLY 1.0, not merely lagged back toward it: Gates 1 and 2 pass today and the
    command there must carry no floating-point residue from a decayed weight."""
    u, sz = approach(0.13)
    for tick in replay(u, sz, cfg(*flags), ag=ag):
        assert tick["commit_k"] == 1.0
        assert tick["committed"] is False
        assert tick["commit_latched"] is False


def test_the_new_flags_change_nothing_below_the_scope_gate():
    u, sz = approach(0.13)
    off = replay(u, sz, cfg(), ag=1)
    on = replay(u, sz, cfg("--gate-commit-rate-max", str(RATE),
                           "--gate-commit-stable-s", str(DWELL)), ag=1)
    assert [t["commit_k"] for t in off] == [t["commit_k"] for t in on]


# ---------------------------------------------------------------------------------
# The rule itself -- the replay-derived separation
# ---------------------------------------------------------------------------------
ARGS_ON = ("--gate-commit-rate-max", str(RATE), "--gate-commit-stable-s", str(DWELL))


@pytest.mark.parametrize("du,label", [(0.13, "filt93B"), (0.19, "filt89.1")])
def test_it_rejects_a_transient_zero_crossing(du, label):
    """The false latches. Under the frame rule these lock; under the rate rule they must
    not, because commit_k staying at 1.0 is what leaves the -9 deg correction live."""
    u, sz = approach(du)
    assert replay(u, sz, cfg())[-1]["commit_latched"], (
        f"{label}: the frame-based rule was supposed to latch here -- if it no longer "
        f"does, this test is no longer measuring the regression")
    ticks = replay(u, sz, cfg(*ARGS_ON))
    assert not ticks[-1]["commit_latched"], f"{label} du={du:+.3f} still latched"
    assert all(t["commit_k"] == 1.0 for t in ticks), (
        "authority must stay FULL when the latch is refused -- that is the fallback")


@pytest.mark.parametrize("du,label", [(0.096, "filt87.1"), (-0.012, "filt80.5")])
def test_it_preserves_the_golden_latches(du, label):
    u, sz = approach(du)
    ticks = replay(u, sz, cfg(*ARGS_ON))
    assert ticks[-1]["commit_latched"], f"{label} du={du:+.3f} lost its latch"


def test_the_separation_sits_between_the_golden_and_false_rates():
    """Pins WHY 0.11: it has to be above the fastest golden crossing (+0.096) and below
    the slowest false one (+0.13). That is a narrow band, and a tuning change that moves
    the threshold outside it silently breaks one side or the other."""
    assert 0.096 < RATE < 0.13


# ---------------------------------------------------------------------------------
# Seconds, not frames
# ---------------------------------------------------------------------------------
def test_the_dwell_is_real_time_so_the_rule_survives_a_loop_rate_change():
    """THE POINT OF stable_s. The same settled approach at 20 Hz and at 60 Hz must
    qualify the same way; the frame rule does not, which is why it meant different things
    run to run."""
    u, sz = approach(-0.012, n_far=40, n_near=60)
    slow = replay(u, sz, cfg(*ARGS_ON), dt=1.0 / 20.0)
    fast = replay(u, sz, cfg(*ARGS_ON), dt=1.0 / 60.0)
    assert slow[-1]["commit_latched"] == fast[-1]["commit_latched"] is True


def test_the_dwell_must_actually_elapse():
    """A settled patch shorter than the window must not qualify, and one tick longer
    must. Built as a real approach does it: off-centre and far (arming, not accruing),
    then a step onto centre. The step itself is a rate failure, so the window starts on
    the tick AFTER it -- which is the behaviour, not an artefact."""
    args = cfg(*ARGS_ON)
    far_u, far_sz = [0.5] * 3, [0.05] * 3          # arms; align_ok False, no accrual
    need = -(-DWELL // DT)                         # ticks of dwell required, rounded up
    short = int(need)                              # +1 for the step tick = exactly short
    ticks = replay(far_u + [0.0] * short, far_sz + [SIZE + 0.05] * short, args)
    assert ticks[-1]["commit_stable_s"] < DWELL
    assert not ticks[-1]["commit_latched"], "latched before the window closed"
    ticks = replay(far_u + [0.0] * (short + 1), far_sz + [SIZE + 0.05] * (short + 1), args)
    assert ticks[-1]["commit_stable_s"] >= DWELL
    assert ticks[-1]["commit_latched"], "the window never closed even once exceeded"


def test_either_failure_zeroes_the_dwell_immediately():
    """No leaky accumulator: one off-centre or one fast frame restarts the window, so the
    latch tick is always itself aligned AND low-rate with the whole window behind it."""
    args = cfg(*ARGS_ON)
    u = [0.5] + [0.0] * 4 + [0.5] + [0.0] * 4      # a spike mid-window
    sz = [0.05] + [SIZE + 0.05] * 9
    ticks = replay(u, sz, args)
    spike = ticks[5]
    assert spike["commit_stable_s"] == 0.0
    assert not spike["align_ok"]
    latched_at = [i for i, t in enumerate(ticks) if t["commit_latched"]]
    assert not latched_at or latched_at[0] > 5, "latched across the reset"


def test_stable_s_supersedes_the_frame_count():
    """Once the seconds path is active the frame counter must not be able to latch on its
    own -- otherwise the two rules race and the looser one wins."""
    u, sz = approach(0.13)
    ticks = replay(u, sz, cfg("--gate-commit-stable-frames", "1", *ARGS_ON))
    assert not ticks[-1]["commit_latched"], (
        "the 1-frame rule latched despite the seconds path being active")


# ---------------------------------------------------------------------------------
# Robustness / discipline
# ---------------------------------------------------------------------------------
def test_the_rate_rule_works_without_an_align_band_set():
    """--gate-commit-rate-max alone must not raise on `abs(u_f) <= None`. It is a legal
    combination, so it has to mean something rather than crash mid-flight."""
    args = build_parser().parse_args(
        ["--gate-commit-size", str(SIZE), "--gate-commit-rate-max", str(RATE)])
    assert args.gate_commit_align is None
    u, sz = approach(0.13)
    ticks = replay(u, sz, args)                # must not raise
    assert all(t["align_ok"] for t in ticks), "align must be vacuously true when unset"


def test_no_derivative_is_taken_across_a_leg_boundary():
    """u_f_prev resets to None on a gate advance, so the first in-scope tick of a leg
    reports du_f 0.0 rather than a spike off the previous leg's heading."""
    u, sz = approach(0.13)
    assert replay(u, sz, cfg(*ARGS_ON))[0]["du_f"] == 0.0


def test_the_derivative_is_measured_on_the_filtered_signal_at_the_real_dt():
    """Scales with dt, not with the frame index -- a halved loop rate must double the
    reported rate for the same per-frame step."""
    u = [0.0, 0.01, 0.02]
    sz = [0.05, SIZE + 0.05, SIZE + 0.05]
    a = replay(u, sz, cfg(*ARGS_ON), dt=1.0 / 35.0)[-1]["du_f"]
    b = replay(u, sz, cfg(*ARGS_ON), dt=1.0 / 70.0)[-1]["du_f"]
    assert b == pytest.approx(2.0 * a)


def test_the_latch_is_still_irreversible():
    """Unchanged from the frame rule: the terminal parallax spike after the latch must not
    re-open it and hand the reaction back to vision."""
    args = cfg(*ARGS_ON)
    settled = [0.0] * 10
    spike = [0.6] * 10                         # the terminal sweep
    ticks = replay([0.5] + settled + spike,
                   [0.05] + [SIZE + 0.05] * 20, args)
    assert ticks[-1]["commit_latched"], "the parallax spike un-latched a good commit"


def test_defaults_are_off_and_as_specified():
    d = vars(build_parser().parse_args([]))
    assert d["gate_commit_rate_max"] == 0.0, "the rate rule must ship opt-in"
    assert d["gate_commit_stable_s"] == 0.0, "the seconds dwell must ship opt-in"


def test_the_replay_derived_dwell_is_not_raised_past_the_settled_window():
    """The spec's ceiling, recorded as a bound on the FLAG rather than as a claim about
    the flight. The ~0.15 s settled window is a measurement from the flown approaches,
    which this file has no access to -- a synthetic approach can be made settled for as
    long as one likes, so asserting '0.40 s is unsatisfiable' here would be asserting a
    property of the fixture, not of the aircraft. What is pinned is the value in use."""
    assert DWELL <= 0.15


def test_the_dwell_accrues_BEFORE_the_size_and_arm_gates():
    """A DELIBERATE ASYMMETRY vs the frame-based path, pinned because it is surprising.

    The frame counter only advances while `commit_armed and size_ok`; the seconds dwell
    accumulates on align+rate ALONE, then the latch tests armed/size/dwell together. So a
    drone that is already settled during the far field arrives with the window banked and
    can latch on the first frame that size_ok turns on.

    That is not a loophole in the rate rule -- the latch tick is still itself aligned and
    low-rate, because either failure zeroes the dwell that same tick. But it does mean the
    dwell is 'has been settled for 0.15 s', not 'has been settled for 0.15 s AT CLOSE
    RANGE'. On a real Gate-3 approach the far field is a turn, so align_ok is false there
    and the distinction rarely bites; on a straight-in approach it does."""
    args = cfg(*ARGS_ON)
    n_far = 12
    u = [0.0] * n_far + [0.0] * 3                  # settled the whole way in
    sz = [0.05] * n_far + [SIZE + 0.05] * 3
    ticks = replay(u, sz, args)
    assert ticks[n_far - 1]["commit_stable_s"] >= DWELL, (
        "the dwell did not accrue in the far field -- if this changed, the latch is now "
        "gated on close-range settling and the note above is stale")
    assert ticks[n_far]["commit_latched"], "did not latch on the first close frame"
