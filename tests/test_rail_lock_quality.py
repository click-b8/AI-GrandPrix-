"""RAIL STEERING-QUALITY GATE, and the merged-distant-path diagnosis.

MEASURED ON REAL FLIGHT DATA (filt29.csv, flown on the rebuilt detector):

    leg 0 (spawn->g1) : LOCK 25.6% | rms p50 0.93 | rows p50 123 | sep p50 0.483
    leg 1 (g1->g2)    : LOCK  2.7% | rms p50 2.76 | rows p50  32 | sep p50 0.090

So the rail does NOT lock usefully on the g1->g2 leg, and the few ticks that do lock are
poorly conditioned: |u_tube| p90 0.623 there, which at the shipped gain is a FULL-CLAMP
bank taken off a distant sliver. Two consequences, both pinned here:

1. tube.found is NOT sufficient authority to steer. A separate quality gate (fit residual
   and rail separation) refuses those ticks, so the gate law keeps the bank instead of
   the rail banking hard on a bad fit.

2. WHY the leg fails: 86.2% of its ticks produced ZERO rail rows while area_frac stayed
   at 0.006 -- the path was in frame the whole time. The cause is that once the two rails
   converge to within rail_merge_gap_px they form ONE narrow run, which is discarded as
   "one rail, unusable alone". Reproduced synthetically below.
"""
import csv
import os

import numpy as np
import pytest

from dataclasses import replace

from tools.schedule_flier import build_parser
from vq1_vision_servo import ServoConfig, TubeDetector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "filt29.csv")

flight = pytest.mark.skipif(not os.path.exists(CSV), reason="filt29.csv not present")

MAX_RMS, MIN_SEP = 2.0, 0.15


def leg(n):
    with open(CSV, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    def f(r, k, d=0.0):
        try:
            return float(r[k])
        except (TypeError, ValueError):
            return d
    return [r for r in rows if int(f(r, "active_gate", -1)) == n], f


def distant_path(halfspan_px, h=180, w=320, y_conv=60):
    """Two rails converging to `halfspan_px` at the bottom -- a far-off path."""
    img = np.full((h, w, 3), 20, np.uint8)
    for y in range(y_conv, h):
        t = (y - y_conv) / (h - y_conv)
        for s in (-1, 1):
            x = int(w / 2 + s * halfspan_px * t)
            img[y, max(0, x - 1):x + 2] = (40, 230, 255)
    return img


# ---------------------------------------------------------------------------------
# 1. The quality gate
# ---------------------------------------------------------------------------------
def test_the_quality_gate_defaults_match_the_measurement():
    d = vars(build_parser().parse_args([]))
    assert d["rail_lat_max_rms"] == MAX_RMS
    assert d["rail_lat_min_sep"] == MIN_SEP
    # both must sit BETWEEN the good leg and the bad one, or they separate nothing
    assert 0.93 < MAX_RMS < 2.76, "the rms gate no longer splits leg 0 from leg 1"
    assert 0.090 < MIN_SEP < 0.483, "the sep gate no longer splits leg 0 from leg 1"


@flight
def test_the_gate_refuses_the_bad_leg_1_locks():
    """The ticks that would have steered on the g1->g2 leg are exactly the ones the gate
    is meant to reject."""
    rows, f = leg(1)
    locked = [r for r in rows if f(r, "tube_solid") > 0.5]
    assert locked, "no locked ticks on leg 1 at all"
    passing = [r for r in locked
               if f(r, "fit_rms") <= MAX_RMS and f(r, "rail_sep") >= MIN_SEP]
    assert len(passing) < 0.35 * len(locked), (
        f"the quality gate admitted {len(passing)}/{len(locked)} of leg 1's locks; it is "
        f"supposed to refuse the poorly-conditioned majority")


@flight
def test_the_gate_keeps_the_good_leg_0_locks():
    """It must not be so strict that it refuses a genuinely good rail."""
    rows, f = leg(0)
    locked = [r for r in rows if f(r, "tube_solid") > 0.5]
    assert locked
    passing = [r for r in locked
               if f(r, "fit_rms") <= MAX_RMS and f(r, "rail_sep") >= MIN_SEP]
    # MEASURED: 33/50 (66%). The gate is deliberately biased STRICT -- loosening it to
    # rms<=3.0 would admit 82% of these but also 4x more of leg 1's bad locks (4% -> 16%).
    # The asymmetry is intentional: refusing a good rail tick costs nothing, because the
    # fallback is the gate-centring law that already flies the course; admitting a bad
    # one costs a wrong bank. So this only asserts the gate is not gratuitously strict.
    assert len(passing) > 0.6 * len(locked), (
        f"the quality gate refused {len(locked) - len(passing)}/{len(locked)} of the "
        f"GOOD gate-1-leg locks -- it is too strict")


@flight
def test_the_rail_does_not_lock_usefully_on_the_g1_g2_leg():
    """THE STEP-1 VERDICT, recorded so it cannot be quietly forgotten. If a detector
    change ever raises this, the number to beat is here."""
    rows, f = leg(1)
    locked = [r for r in rows if f(r, "tube_solid") > 0.5]
    assert len(locked) / len(rows) < 0.10, (
        f"leg-1 lock rate is now {100 * len(locked) / len(rows):.1f}% -- if this really "
        f"improved, re-run the step-1 verification and consider arming the rail")


# ---------------------------------------------------------------------------------
# 2. Why it fails: merged rails on a distant path
# ---------------------------------------------------------------------------------
@flight
def test_the_leg_1_failure_is_zero_rows_while_the_path_is_in_frame():
    """Not a fit failure and not an absent path: the rows never get produced."""
    rows, f = leg(1)
    no_lock = [r for r in rows if f(r, "tube_solid") < 0.5]
    zero_rows = [r for r in no_lock if f(r, "rail_rows") == 0]
    assert len(zero_rows) > 0.8 * len(rows), (
        f"only {100 * len(zero_rows) / len(rows):.0f}% of leg-1 ticks were zero-row")
    # and the cyan really was there
    seen = [r for r in zero_rows if f(r, "area_frac") > 0.002]
    assert len(seen) > 0.95 * len(zero_rows), (
        "the zero-row ticks had no cyan either -- then the path was genuinely not in "
        "frame and the diagnosis in this file is wrong")


def test_merged_rails_are_discarded_by_default_and_that_is_the_defect():
    """Reproduces the flight signature: a path whose rails close to within the merge gap
    yields ZERO rows at a healthy area_frac."""
    det = TubeDetector(ServoConfig())
    m = det.measure(distant_path(4))
    assert m.n_rows == 0
    assert m.area_frac > 0.005, "the synthetic path should still mask plenty of cyan"


def test_merge_recovery_restores_the_rows_and_reads_the_offset():
    """With recovery on, the same frame produces rows again and the offset is right."""
    det = TubeDetector(replace(ServoConfig(), rail_merge_recover=True))
    m = det.measure(distant_path(4))
    assert m.n_rows > 50, f"recovery produced only {m.n_rows} rows"
    assert m.n_merged > 50, "the rows should be flagged as merged"
    assert abs(m.u_tube) < 0.02, f"centred path read {m.u_tube:+.4f}"


def test_merge_recovery_still_does_not_claim_a_two_rail_lock():
    """HONESTY OF THE FLAG. A merged run is a BEARING to a distant path, not a resolved
    corridor -- it cannot satisfy 'fitting the two OUTER rails', so it must not report
    found=True and quietly become a steering reference."""
    det = TubeDetector(replace(ServoConfig(), rail_merge_recover=True))
    assert not det.measure(distant_path(4)).found
    assert not det.measure(distant_path(2)).found


def test_merge_recovery_is_off_by_default_and_changes_nothing_when_off():
    assert ServoConfig().rail_merge_recover is False
    assert vars(build_parser().parse_args([]))["rail_merge_recover"] is False


def test_merge_recovery_barely_perturbs_a_well_resolved_path():
    """It only adds rows that were previously discarded. On a NEAR path that is just the
    handful of rows right at the convergence, where the rails genuinely do touch -- so
    the measurement shifts by a few 1e-5, not by anything a steering law would notice.
    (Asserting bit-equality here would be wrong: those rows are real and now counted.)"""
    a = TubeDetector(ServoConfig()).measure(distant_path(60))
    b = TubeDetector(replace(ServoConfig(),
                             rail_merge_recover=True)).measure(distant_path(60))
    assert a.found and b.found
    assert a.u_tube == pytest.approx(b.u_tube, abs=1e-3)
    assert a.rail_sep == pytest.approx(b.rail_sep, abs=1e-3)
    assert b.n_merged < 0.2 * b.n_rows, "a near path should be mostly UNmerged rows"
