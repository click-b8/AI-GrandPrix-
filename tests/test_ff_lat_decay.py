"""CHANGE A acceptance: the feed-forward lateral bank must fade to level when the gate
is lost, instead of holding an open-loop turn blind.

REGRESSION UNDER TEST -- measured in filt4.csv (flt4 flight, 2026-07-28):
    last live detection   t = 12.762 s
    run end               t = 20.568 s
    des_roll_deg over that entire 7.81 s tail: min +5.00, max +5.00, 1210 ticks
i.e. the drone was commanded a 5 deg left bank for 7.07 s after the signal was declared
lost, with nothing able to correct it. That is the veer this change kills.

The replay drives ff_lat_step with the REAL tick timestamps from filt4.csv's tail, so it
exercises the shipped per-tick path (dt distribution and all), not a reimplementation.
"""
import csv
import os

import pytest

from tools.schedule_flier import ff_lat_step

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILT4 = os.path.join(ROOT, "filt4.csv")

HOLD_S = 0.4        # --ff-lat-hold-s default
TAU_S = 0.4         # --ff-lat-decay-s default
BANK_DEG = 5.0      # --post-gate1-bank as flown in flt4
FADE_BY_S = 1.5     # acceptance: below 1 deg within this long of the loss
FADE_TO_DEG = 1.0


def _tail():
    """(t_loss, [t...]) for every tick from the last live detection to end of run."""
    if not os.path.exists(FILT4):
        pytest.skip("filt4.csv not present")
    with open(FILT4, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    live = [i for i, r in enumerate(rows) if int(r["live"])]
    if not live:
        pytest.skip("filt4.csv has no live detections")
    last = live[-1]
    return float(rows[last]["t"]), [float(r["t"]) for r in rows[last:]], rows[last:]


def _replay(times, hold_s=HOLD_S, tau_s=TAU_S):
    """Yield (t_since_loss, commanded_bank_deg) with the gate lost at times[0]."""
    t0 = times[0]
    k = 1.0
    out = []
    for prev, t in zip(times, times[1:]):
        gap = t - t0
        k = ff_lat_step(k, t - prev, gap <= hold_s, hold_s, tau_s)
        out.append((gap, k * BANK_DEG))
    return out


def test_filt4_tail_held_five_degrees_open_loop():
    """Documents the regression: the flown data really did hold +5 for >7 s."""
    t_loss, times, rows = _tail()
    dr = [float(r["des_roll_deg"]) for r in rows]
    assert max(dr) - min(dr) == 0.0, "flt4 tail was expected to be perfectly flat"
    assert abs(dr[0] - BANK_DEG) < 1e-9
    assert times[-1] - t_loss > 7.0, "flt4 tail should be >7 s of blind flight"


def test_ff_lat_decays_below_one_degree_within_1_5s():
    """THE ACCEPTANCE: +5 -> <1 deg within 1.5 s of the gate loss."""
    _, times, _ = _tail()
    trace = _replay(times)
    at_limit = [b for gap, b in trace if gap <= FADE_BY_S]
    assert at_limit, "no ticks inside the acceptance window"
    assert at_limit[-1] < FADE_TO_DEG, (
        f"bank was {at_limit[-1]:.3f} deg at {FADE_BY_S}s after loss, "
        f"needed < {FADE_TO_DEG}")
    # and it must still be there at the end -- a fade that rebounds is not a fade
    assert trace[-1][1] < FADE_TO_DEG


def test_ff_lat_holds_full_bank_through_the_hold_window():
    """No fade at all until --ff-lat-hold-s has elapsed: brief blinks must not
    dismantle the backbone."""
    _, times, _ = _tail()
    for gap, bank in _replay(times):
        if gap <= HOLD_S:
            assert abs(bank - BANK_DEG) < 1e-9, f"faded early at {gap:.3f}s"


def test_ff_lat_fade_is_monotonic_and_never_overshoots():
    _, times, _ = _tail()
    banks = [b for _, b in _replay(times)]
    for a, b in zip(banks, banks[1:]):
        assert b <= a + 1e-12, "fade must not rebound"
    assert min(banks) >= 0.0, "must not cross through level into a reverse bank"
    assert max(banks) <= BANK_DEG + 1e-12


def test_ff_lat_ramps_back_in_on_reacquisition():
    """Re-acquiring must restore the backbone WITHOUT a one-tick step."""
    k = 0.0
    dt = 1.0 / 30.0
    steps = []
    for _ in range(60):                       # 2 s of live gate
        k = ff_lat_step(k, dt, True, HOLD_S, TAU_S)
        steps.append(k)
    assert steps[0] < 0.10, "snapped back on in one tick"
    assert steps[-1] > 0.99, "never recovered"
    assert all(b >= a for a, b in zip(steps, steps[1:]))


def test_ff_lat_never_fades_before_first_acquisition():
    """det_t None (no gate seen yet) is NOT a lost gate -- the seg-0 backbone must
    survive the pre-acquisition phase at full strength."""
    k = 1.0
    for _ in range(300):                      # 10 s at 30 Hz, gate never yet seen
        k = ff_lat_step(k, 1.0 / 30.0, True, HOLD_S, TAU_S)
    assert k == pytest.approx(1.0)


@pytest.mark.parametrize("tau", [0.0, -1.0])
def test_ff_lat_tau_zero_is_instant(tau):
    assert ff_lat_step(1.0, 0.03, False, HOLD_S, tau) == 0.0
    assert ff_lat_step(0.0, 0.03, True, HOLD_S, tau) == 1.0


def test_ff_lat_dt_guards():
    """Same dt clamping as the thrust slew limiter: no negative or huge steps."""
    assert ff_lat_step(1.0, -5.0, False, HOLD_S, TAU_S) == 1.0     # dt<0 -> no move
    k = ff_lat_step(1.0, 1e9, False, HOLD_S, TAU_S)                # dt clamped to 0.5
    assert 0.0 <= k <= 1.0 and k == pytest.approx(1.0 - 0.5 / (TAU_S + 0.5))
