"""RAIL SIGNAL properties, measured on filt9's gate-3 leg.

STATUS: the rail steers under --tube-lateral (default OFF), from ag >= 2 only. Two
earlier attempts to arm it failed, and BOTH failed for reasons this build fences off:
    flt10  rail-primary on EVERY leg -> broke gate 1 (tube solid on 1.7% of ticks there:
           "steer on the tube" meant "steer on nothing")
           -> FENCED by --tube-lateral-from-ag (default 2). Out of scope the rail's
              authority is forced to EXACTLY 0.0, so legs 0-1 are bit-for-bit unchanged.
    flt12  priority handoff, rail only while solid -> CRASHED gate 1; the curvature lead
           spiked the rail to -11 deg at close range at auth 0.95
           -> FENCED by --tube-lead-max-deg (default 4), a clamp on the lead ALONE, so it
              can never dominate the u_tube term it is meant to lead.
The third guard is that authority is the rail's own decay weight, not a switch: where the
tube is not seen, the previous gate law fades back in, so a leg with no tube flies today's
proven law instead of flying blind.

This file covers both the SIGNAL's properties (bounded where the gate saturates, its area
gate load-bearing, a clean fade to zero on loss) and the SHIPPED law's guards, replayed on
the real flight. See test_commit_scope.py for the gate law the rail hands back to.

WHAT flt9 SHOWS (the defect being fixed). On the gate-3 approach the lateral loop steered
on GATE u_err. A gate blob LUNGES sideways from parallax as you close on it, while the
tube -- a continuous line -- barely moves. Straight off filt9.csv:

    t=10.57   gate u_err +0.283    tube u +0.199
    t=10.81   gate u_err +0.526    tube u +0.249
    t=11.00   gate u_err +0.751    tube u +0.302
    peak      gate u_err +0.950    tube u ~+0.37

des_roll spent 30 of the 46 close-range ticks pinned to the -11 deg clamp, steering hard
off a decoy. The drone veered off before gate 3.

WHAT THIS TEST MEASURES: filt9.csv logs u_tube / curvature / area_frac every tick
alongside the gate signal and the des_roll that was actually commanded, so the rail law
can be replayed at the real tick timing, on the real flight, without flying again. The
RailSignal filter used here is the shipped class, not a copy.

HEADLINE: on that window the rail command never reaches its clamp (0/46 ticks) where the
gate command sat on its clamp for 65% of them. Note the rail's boundedness is EARNED by
the area gate, not free -- see test_area_gate_is_load_bearing: at --tube-steer-area-min
0.005 the rail saturates too, because sub-0.01 readings are slivers of tube glimpsed past
the gate frame at close range, not the rail.
"""
import csv
import math
import os

import pytest

from vq1_vision_servo import LATERAL_SIGN, ServoConfig
from tools.schedule_flier import (RailSignal, blend_lateral, build_parser, gate_fine_trim,
                                  lag_step, rail_bank_cmd)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "filt9.csv")

GATE3_LEG_AG = 2                    # the leg toward gate 3, i.e. after gate 2 was passed
CLOSE_LO, CLOSE_HI = 10.0, 11.6     # the close-approach window where the gate lunges

TAU, HOLD, DECAY = 0.2, 0.4, 0.5    # --gate-filter-* defaults, shared by the rail filter
RAIL_CLAMP_DEG = 10.0               # --tube-max-bank-deg (same bound the gate law gets)
GATE_CLAMP_DEG = 11.0               # --gate-max-bank-deg as flt9 flew it
AREA_MIN = 0.015                    # --tube-steer-area-min
K_TUBE_BANK = 0.6                   # --k-tube-bank
LEAD_MAX_DEG = 3.0                  # --tube-lead-max-deg
FINE_MAX_DEG = 2.5                  # --gate-fine-max-deg
FINE_SIZE_MIN, FINE_UERR_MAX = 0.15, 0.15
AUTH_TAU = 0.3                      # --tube-auth-tau

pytestmark = pytest.mark.skipif(not os.path.exists(CSV),
                                reason="filt9.csv (the flt9 flight log) not present")


def _leg():
    with open(CSV, encoding="utf-8") as fh:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    return [r for r in rows if r["active_gate"] == GATE3_LEG_AG]


def _rail_replay(rows, area_min=AREA_MIN, k_bank=K_TUBE_BANK, clamp_deg=RAIL_CLAMP_DEG):
    """Replay the RAIL law over `rows` at their real timestamps. Returns
    [(t, des_roll_deg, rail_alive, row)]."""
    cfg = ServoConfig()
    rail = RailSignal(TAU, HOLD, DECAY)
    clamp = math.radians(clamp_deg)
    out, last_t = [], None
    for r in rows:
        t = r["t"]
        dt = min(max(t - last_t, 0.0), 0.5) if last_t is not None else 0.0
        last_t = t
        solid = bool(r["tube_found"] and r["area_frac"] >= area_min)
        rail.update(t, dt, solid, r["u_tube"], r["curvature"])
        u = LATERAL_SIGN * rail.control_u(cfg.k_tube_lead)
        deg = math.degrees(max(-clamp, min(clamp, k_bank * u))) if rail.alive else 0.0
        out.append((t, deg, rail.alive, r))
    return out


def _window(seq, lo=CLOSE_LO, hi=CLOSE_HI):
    return [x for x in seq if lo <= x[0] <= hi]


def _saturation(seq, clamp_deg):
    win = _window(seq)
    assert win, "close-approach window is empty -- filt9.csv timing changed"
    sat = sum(1 for _, d, _, _ in win if abs(d) > clamp_deg - 0.05)
    return sat / len(win), max(abs(d) for _, d, _, _ in win)


def test_the_gate_law_saturated_on_this_leg():
    """Baseline -- the defect, measured off what was actually commanded in flight."""
    flown = [(r["t"], r["des_roll_deg"], True, r) for r in _leg()]
    frac, peak = _saturation(flown, GATE_CLAMP_DEG)
    assert peak == pytest.approx(GATE_CLAMP_DEG, abs=0.05)
    assert frac > 0.5, f"expected the flown gate law to sit on its clamp; got {frac:.0%}"


def test_rail_command_stays_off_the_clamp_where_the_gate_law_saturated():
    """THE HEADLINE. Same ticks, same window, rail law: never reaches the clamp."""
    rail = _rail_replay(_leg())
    frac, peak = _saturation(rail, RAIL_CLAMP_DEG)
    assert frac == 0.0, f"rail law saturated on {frac:.0%} of the close-approach ticks"
    assert peak < RAIL_CLAMP_DEG, f"rail peak {peak:.1f} deg reached the clamp"


def test_the_gate_sweeps_far_more_than_the_rail_at_close_range():
    """The premise, stated as a measurement: over the close approach the gate's lateral
    signal travels several times further than the tube's."""
    win = [r for r in _leg() if CLOSE_LO <= r["t"] <= CLOSE_HI]
    solid = [r for r in win if r["tube_found"] and r["area_frac"] >= AREA_MIN]
    gate_span = max(r["u_err"] for r in win) - min(r["u_err"] for r in win)
    tube_span = max(r["u_tube"] for r in solid) - min(r["u_tube"] for r in solid)
    assert gate_span > 3.0 * tube_span, (
        f"gate span {gate_span:.3f} vs tube span {tube_span:.3f}")


def test_area_gate_is_load_bearing():
    """The rail's boundedness is EARNED, not free. Sub-0.01 area readings are slivers of
    tube seen past the gate frame at close range -- |u_tube| runs out to 0.67 on them.
    Admitting those saturates the rail law too, which is why --tube-steer-area-min
    defaults to 0.015 and not to the value that merely 'engages more often'."""
    loose = _rail_replay(_leg(), area_min=0.005)
    frac, _ = _saturation(loose, RAIL_CLAMP_DEG)
    assert frac > 0.3, ("expected a loose area gate to saturate -- if this stopped being "
                        "true the default could be relaxed")


def test_rail_coasts_straight_when_the_tube_is_weak_never_slams():
    """'Tube lost/weak: hold last command briefly, then coast straight. No slam.'

    Three separate claims, asserted separately:
      1. it does reach EXACTLY zero (coast straight, not a standing blind turn),
      2. it takes the specified hold+fade to get there, not one tick (no step down),
      3. it never steps (no slam) anywhere on the leg.
    """
    rail = _rail_replay(_leg())
    solid = [t for t, _, _, r in rail
             if r["tube_found"] and r["area_frac"] >= AREA_MIN]
    assert solid, "no solid tube readings on this leg at all"
    last_solid = max(solid)

    # 1. fully retired once the hold has expired and the fade has run ~5 tau
    dead = [x for x in rail if x[0] > last_solid + HOLD + 5.0 * DECAY]
    assert dead, "leg ends before the fade completes -- widen the log or the window"
    assert all(abs(d) < 1e-6 for _, d, _, _ in dead), \
        "rail still commanding bank long after the tube is gone"

    # 2. and it was still commanding something right after the loss: a HOLD, not a cut
    just_after = [d for t, d, _, _ in rail if last_solid < t <= last_solid + HOLD]
    assert just_after and max(abs(d) for d in just_after) > 1.0, \
        "rail dropped its command the moment the tube blinked -- that is a step, not a hold"

    # 3. no slam anywhere on the leg
    steps = [abs(b[1] - a[1]) for a, b in zip(rail, rail[1:])]
    assert max(steps) < 2.0, f"rail command stepped {max(steps):.1f} deg in one tick"


def test_gate_fine_trim_retires_itself_as_the_lunge_begins():
    """The anti-decoy rule: the gate may fine-trim only while LARGE and NEAR-CENTRED.
    On this leg that admits the early, honest ticks and rejects every sweep tick."""
    size_min, uerr_max = 0.15, 0.15
    leg = _leg()
    admitted = [r for r in leg
                if r["live"] and r["sz_f"] >= size_min and abs(r["u_f"]) <= uerr_max]
    rejected = [r for r in leg
                if r["live"] and r["sz_f"] >= size_min and abs(r["u_f"]) > uerr_max]
    assert admitted, "the fine-trim window never opens -- thresholds are too tight"
    assert rejected, "no sweep ticks on this leg -- the test is not measuring anything"
    # every admitted tick precedes every rejected one: the trim retires and stays retired
    assert max(r["t"] for r in admitted) <= min(r["t"] for r in rejected) + 1e-9
    # and the peak gate error is firmly outside the window
    assert max(abs(r["u_f"]) for r in leg) > 4.0 * uerr_max


def test_fine_trim_clamp_keeps_the_gate_subordinate_to_the_rail():
    """Even fully engaged, the gate trim is a nudge: 2.5 deg against the rail's 10. The
    gate is the signal that has already proven itself a decoy at close range, so it never
    gets to be more than a quarter of the authority of the reference it is trimming."""
    assert FINE_MAX_DEG <= RAIL_CLAMP_DEG / 4.0


# --------------------------------------------------------------------------------------
# THE SHIPPED LAW. Above replays the rail SIGNAL; below exercises the functions the flight
# actually calls (rail_bank_cmd / gate_fine_trim / blend_lateral), so the guards are
# pinned to the code that flies rather than to a paraphrase of it.
# --------------------------------------------------------------------------------------

def _shipped_replay(rows, area_min=AREA_MIN, lead_max_deg=LEAD_MAX_DEG, in_scope=True,
                    auth_tau=AUTH_TAU, k_bank=K_TUBE_BANK, clamp_deg=None):
    """Replay the SHIPPED lateral law over `rows` at their real timestamps: the rail
    filter, the lagged authority handoff, the separately-clamped rail command, the gate
    fine-trim, and the blend back to the flown gate command. Mirrors the block in
    mode_coast_tube and calls the same functions.

    Returns [(t, trim_rad, rail_auth, row)] -- RADIANS, deliberately: the fence test
    asserts bit-for-bit equality with the gate command, and a deg->rad->deg roundtrip
    would inject float error the flight never performs."""
    cfg = ServoConfig()
    rail = RailSignal(TAU, HOLD, DECAY)
    bank_max = math.radians(RAIL_CLAMP_DEG if clamp_deg is None else clamp_deg)
    lead_max = math.radians(lead_max_deg)
    fine_max = math.radians(FINE_MAX_DEG)
    out, last_t, auth_k, fine_k = [], None, 0.0, 0.0
    for r in rows:
        t = r["t"]
        dt = min(max(t - last_t, 0.0), 0.5) if last_t is not None else 0.0
        last_t = t
        solid = bool(r["tube_found"] and r["area_frac"] >= area_min)
        rail.update(t, dt, solid, r["u_tube"], r["curvature"])
        # gate_cmd as flown: filt9 logged the commanded des_roll for this leg.
        gate_cmd = math.radians(r["des_roll_deg"])
        target = rail.k if (in_scope and rail.alive) else 0.0
        auth_k = lag_step(auth_k, dt, target, auth_tau)
        if target == 0.0 and auth_k < 1e-4:
            auth_k = 0.0
        rail_cmd = fine_target = 0.0
        if auth_k > 0.0:
            rail_cmd = rail_bank_cmd(LATERAL_SIGN * rail.u, LATERAL_SIGN * rail.curv,
                                     k_bank, cfg.k_tube_lead, bank_max, lead_max)
            fine_target = gate_fine_trim(gate_cmd, r["sz_f"], r["u_f"],
                                         FINE_SIZE_MIN, FINE_UERR_MAX, fine_max)
        fine_k = lag_step(fine_k, dt, fine_target, auth_tau)
        out.append((t, blend_lateral(auth_k, rail_cmd, fine_k, 1.0, gate_cmd), auth_k, r))
    return out


def test_shipped_law_stays_bounded_where_the_gate_law_saturated():
    """THE HEADLINE, on the shipped code path, at the shipped defaults.

    Compare against test_the_gate_law_saturated_on_this_leg: over the same close-approach
    ticks the FLOWN gate command sat pinned to its 11 deg clamp for >50% of them. Where
    the rail owns the bank, the shipped command never reaches its clamp -- and its clamp
    is 10 deg, TIGHTER than the gate's 11, so this is not boundedness bought by widening
    the limit. It is bought by the gearing (--k-tube-bank 0.6): the rail stays
    proportional across its whole observed range instead of steering blind on a stop.
    """
    shipped = [(t, math.degrees(x), a, r) for t, x, a, r in _shipped_replay(_leg())]
    win = _window(shipped)
    assert win, "close-approach window is empty -- filt9.csv timing changed"
    # RAIL-OWNED ticks: where authority has actually transferred. On the remaining ticks
    # the blend is still (mostly) the gate law, and it is not this test's claim that a
    # fallback to the gate law is bounded -- it demonstrably is not, which is the whole
    # reason the rail exists.
    owned = [(t, d) for t, d, a, _ in win if a > 0.9]
    assert len(owned) > 0.5 * len(win), (
        f"the rail owned only {len(owned)}/{len(win)} close-approach ticks -- too few to "
        f"claim anything about the law that flew")
    peak = max(abs(d) for _, d in owned)
    assert peak < RAIL_CLAMP_DEG, (
        f"shipped command peaked at {peak:.1f} deg, reaching its own "
        f"{RAIL_CLAMP_DEG} deg clamp")
    assert peak < GATE_CLAMP_DEG, (
        f"shipped command peaked at {peak:.1f} deg, at/over the {GATE_CLAMP_DEG} deg the "
        f"gate law sat pinned to")


def test_the_rail_itself_never_slams():
    """Where the rail owns the bank it moves smoothly -- no step, including across the
    authority handoff, which is the transition --tube-auth-tau exists to smooth.

    Scoped to rail-owned ticks ON PURPOSE. The flown GATE law steps 11.0 deg in one tick
    on this leg (clamp to clamp), and wherever the blend has fallen back to it the
    shipped command inherits that step. Asserting 'the blended command never steps' would
    therefore be asserting something false about the gate law, not something true about
    the rail."""
    shipped = _shipped_replay(_leg())
    steps = [abs(math.degrees(b[1] - a[1])) for a, b in zip(shipped, shipped[1:])
             if a[2] > 0.9 and b[2] > 0.9]
    assert steps, "the rail never held authority for two consecutive ticks"
    assert max(steps) < 2.0, f"rail command stepped {max(steps):.1f} deg in one tick"


def test_the_shipped_law_reduces_the_worst_step_the_gate_law_flew():
    """End to end, over the WHOLE leg including every fallback tick: handing the bank to
    the rail cannot make the worst single-tick jump worse than the law it replaces."""
    leg = _leg()
    flown = max(abs(b["des_roll_deg"] - a["des_roll_deg"]) for a, b in zip(leg, leg[1:]))
    shipped = _shipped_replay(leg)
    got = max(abs(math.degrees(b[1] - a[1])) for a, b in zip(shipped, shipped[1:]))
    assert got < flown, f"shipped worst step {got:.1f} deg vs the flown {flown:.1f}"


def test_the_authority_lag_is_what_removes_the_handoff_step():
    """The lag is load-bearing, not decoration: at tau=0 the same replay slams."""
    instant = _shipped_replay(_leg(), auth_tau=0.0)
    steps = [abs(math.degrees(b[1] - a[1])) for a, b in zip(instant, instant[1:])]
    assert max(steps) > 5.0, ("an instant handoff no longer steps -- if this stopped "
                              "being true --tube-auth-tau could be relaxed")


def test_out_of_scope_the_law_is_bit_for_bit_the_gate_command():
    """THE LEG FENCE (the flt10 fix). On a leg below --tube-lateral-from-ag the rail's
    authority is forced to EXACTLY 0.0, so the command is the gate command to the bit --
    not merely close to it. This is what keeps gates 1 and 2 passing."""
    fenced = _shipped_replay(_leg(), in_scope=False)
    for t, trim_rad, auth, r in fenced:
        assert auth == 0.0
        assert trim_rad == math.radians(r["des_roll_deg"]), (
            f"t={t:.2f}: fenced command {trim_rad!r} != flown gate command "
            f"{math.radians(r['des_roll_deg'])!r}")


def test_the_lead_clamp_is_load_bearing():
    """THE flt12 FIX. The curvature lead is the term that crashed into gate 1. Unclamped
    it is free to reach the bank clamp on its own; clamped it cannot exceed its own bound,
    whatever curvature does."""
    cfg = ServoConfig()
    bank_max = math.radians(RAIL_CLAMP_DEG)
    lead_max = math.radians(LEAD_MAX_DEG)
    # A big curvature spike with the tube itself dead centred: the whole command is lead.
    spike = 2.0
    clamped = rail_bank_cmd(0.0, spike, 1.0, cfg.k_tube_lead, bank_max, lead_max)
    assert abs(clamped) == pytest.approx(lead_max), "the lead clamp did not bind"
    assert math.degrees(abs(clamped)) <= LEAD_MAX_DEG + 1e-9
    # Without its own clamp the same spike runs away to the full bank clamp -- i.e. the
    # separate bound is doing real work, it is not redundant with the outer one.
    unclamped = rail_bank_cmd(0.0, spike, 1.0, cfg.k_tube_lead, bank_max, bank_max)
    assert abs(unclamped) == pytest.approx(bank_max)
    assert abs(unclamped) > 3.0 * abs(clamped)


def test_the_lead_can_never_dominate_the_band_centring_term():
    """The lead is an anticipatory nudge onto the coming bend, not a steering term: at
    full opposing deflection the u_tube term still decides the sign of the command."""
    cfg = ServoConfig()
    bank_max, lead_max = math.radians(RAIL_CLAMP_DEG), math.radians(LEAD_MAX_DEG)
    # u_tube says hard LEFT at the clamp; curvature says hard RIGHT.
    cmd = rail_bank_cmd(math.radians(RAIL_CLAMP_DEG), -5.0, 1.0, cfg.k_tube_lead,
                        bank_max, lead_max)
    assert cmd > 0.0, "the curvature lead reversed the sign of the rail command"


def test_gate_fine_trim_admits_only_the_large_and_centred_gate():
    """The anti-decoy rule as the shipped function evaluates it."""
    fine_max = math.radians(FINE_MAX_DEG)
    big = math.radians(10.0)
    # large AND centred -> admitted, but clamped to a nudge
    assert gate_fine_trim(big, 0.30, 0.05, FINE_SIZE_MIN, FINE_UERR_MAX,
                          fine_max) == pytest.approx(fine_max)
    # large but SWEPT (the flt9 decoy, u_err +0.95) -> rejected outright
    assert gate_fine_trim(big, 0.30, 0.95, FINE_SIZE_MIN, FINE_UERR_MAX, fine_max) == 0.0
    # small/distant -> rejected
    assert gate_fine_trim(big, 0.05, 0.05, FINE_SIZE_MIN, FINE_UERR_MAX, fine_max) == 0.0


def test_the_rail_is_opt_in_and_fenced_by_default():
    """DISCIPLINE: a new behaviour ships OFF and cannot touch the legs that pass today.
    Also pins every constant this file replays against to the shipped default, so the
    measurements above cannot quietly stop describing the flown law."""
    d = vars(build_parser().parse_args([]))
    assert d["tube_lateral"] is False, "--tube-lateral must ship OFF (opt-in per run)"
    assert d["tube_lateral_from_ag"] == 2, (
        "the leg fence must default to 2 -- legs 0-1 are the gates that pass today")
    # Every constant this file replays against IS the shipped default, so the numbers
    # measured above cannot quietly stop describing the law that flies.
    assert d["k_tube_bank"] == K_TUBE_BANK
    assert d["tube_max_bank_deg"] == RAIL_CLAMP_DEG
    assert d["tube_lead_max_deg"] == LEAD_MAX_DEG
    assert d["gate_fine_max_deg"] == FINE_MAX_DEG
    assert d["gate_fine_size_min"] == FINE_SIZE_MIN
    assert d["gate_fine_uerr_max"] == FINE_UERR_MAX
    assert d["tube_auth_tau"] == AUTH_TAU
    assert d["tube_steer_area_min"] == AREA_MIN
    assert d["gate_filter_tau"] == TAU
    assert d["gate_filter_hold_s"] == HOLD
    assert d["gate_filter_decay_s"] == DECAY
    # The rail never gets more authority than the gate law it takes over from.
    assert d["tube_max_bank_deg"] <= d["gate_max_bank_deg"]


def test_blend_endpoints_are_exact():
    """The two endpoints of the handoff are exact, and that exactness is the safety
    property: auth 0 must reproduce the shipped gate expression to the bit."""
    gate_cmd, rail_cmd, fine, commit_k = 0.20, -0.05, 0.01, 0.37
    assert blend_lateral(0.0, rail_cmd, fine, commit_k, gate_cmd) == commit_k * gate_cmd
    assert blend_lateral(1.0, rail_cmd, fine, commit_k, gate_cmd) == rail_cmd + fine
    # and it is monotone in between -- no overshoot through the handoff
    mid = blend_lateral(0.5, rail_cmd, fine, commit_k, gate_cmd)
    lo, hi = sorted((commit_k * gate_cmd, rail_cmd + fine))
    assert lo <= mid <= hi
