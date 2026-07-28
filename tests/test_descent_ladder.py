"""ALTITUDE LADDER acceptance: the per-leg vertical feed-forward table.

THE DEFECT BEING FIXED: --post-gate1-descent applied ONE value (0.026) to every leg from
active_gate>=1 onward. That value was tuned against the STEEP middle legs (17.1 / 16.2
deg). The final two legs are nearly FLAT (1.9 / 1.5 deg), so holding the steep-leg
descent across them commands a sink the course does not have -- which is what flies the
drone into the ground past gate 3.

WHAT THIS ASSERTS: the table is DERIVED from course_gates_cm.json's slopes (change the
geometry, the ladder changes), the anchor leg keeps exactly its proven value, and the
flat final legs come out near zero.
"""
import pytest

from tools.motion_schedule import build_bank_schedule
from tools.schedule_flier import build_descent_ladder

ANCHOR = 0.026          # the proven bias on the g1->g2 leg
ANCHOR_LEG = 2          # g1->g2, slope 17.1 deg

# Leg index == active_gate: leg i is flown while ag == i.
LEG_SPAWN_START, LEG_START_G1, LEG_G1_G2, LEG_G2_G3, LEG_G3_G4, LEG_G4_FIN = range(6)


@pytest.fixture(scope="module")
def sched():
    return build_bank_schedule(8.0)


@pytest.fixture(scope="module")
def ladder(sched):
    return build_descent_ladder(sched, ANCHOR, anchor_leg=ANCHOR_LEG)


def test_course_geometry_is_what_the_ladder_assumes(sched):
    """Guard on the INPUT: if the course file changes, the expectations below are stale
    and should fail loudly here rather than silently elsewhere."""
    slopes = [r["slope_deg"] for r in sched]
    assert slopes == pytest.approx([-1.3, 12.1, 17.1, 16.2, 1.9, 1.5], abs=0.05)


def test_flat_final_legs_are_near_level(ladder):
    """THE HEADLINE. G3->G4 (1.9 deg) and G4->FIN (1.5 deg) are essentially flat; the
    ladder must all but stop descending there instead of holding the steep-leg 0.026."""
    assert ladder[LEG_G3_G4] < 0.005, f"G3->G4 bias {ladder[LEG_G3_G4]} still over-descends"
    assert ladder[LEG_G4_FIN] < 0.005, f"G4->FIN bias {ladder[LEG_G4_FIN]} still over-descends"
    # and specifically: an order of magnitude below the flat-leg value that broke flt9
    assert ladder[LEG_G3_G4] < ANCHOR / 5.0
    assert ladder[LEG_G4_FIN] < ANCHOR / 5.0


def test_anchor_leg_keeps_the_proven_value(ladder):
    """G1->G2 is the leg 0.026 was tuned on. The ladder may not move it."""
    assert ladder[LEG_G1_G2] == pytest.approx(ANCHOR, abs=1e-4)


def test_steep_legs_still_descend(ladder):
    """The point is not 'descend less everywhere' -- the steep legs must be unchanged in
    character. G2->G3 (16.2) sits just under the anchor; START->G1 (12.1) proportionally
    below it."""
    assert ladder[LEG_G2_G3] == pytest.approx(0.025, abs=0.001)
    assert ladder[LEG_START_G1] == pytest.approx(0.018, abs=0.001)
    assert ladder[LEG_G2_G3] > ladder[LEG_START_G1] > ladder[LEG_G3_G4]


def test_spawn_to_start_leg_is_pinned_to_zero(ladder):
    """Leg 0 is the empirically-proven const-0.275 START pass. Nothing here touches it."""
    assert ladder[LEG_SPAWN_START] == 0.0


def test_no_leg_commands_a_climb(ladder):
    """A bias is a SINK. The vertical feed-forward is open loop; earning altitude back is
    --gate-vert's job, not a negative bias the ladder invents."""
    assert all(b >= 0.0 for b in ladder)


def test_table_is_derived_from_the_json_slopes_not_hardcoded(sched):
    """Move a leg's slope and its bias must move with it, proportionally."""
    import copy
    mutated = copy.deepcopy(sched)
    mutated[LEG_G3_G4]["slope_deg"] = 17.1        # make the flat leg as steep as the anchor
    lad = build_descent_ladder(mutated, ANCHOR, anchor_leg=ANCHOR_LEG)
    assert lad[LEG_G3_G4] == pytest.approx(ANCHOR, abs=1e-4)
    assert lad[LEG_G4_FIN] < 0.005                # the untouched flat leg is unaffected


def test_every_leg_follows_the_slope_ratio(sched, ladder):
    ref = sched[ANCHOR_LEG]["slope_deg"]
    for r, b in zip(sched, ladder):
        if r["seg"] == 0:
            continue
        expect = ANCHOR * max(r["slope_deg"], 0.0) / ref
        assert b == pytest.approx(expect, abs=1e-4)


def test_descent_scale_is_a_one_knob_trim(sched, ladder):
    half = build_descent_ladder(sched, ANCHOR, anchor_leg=ANCHOR_LEG, scale=0.5)
    for a, b in zip(ladder, half):
        assert b == pytest.approx(a / 2.0, abs=1e-4)


def test_overrides_are_absolute_and_ignore_the_scale(sched):
    """An override is an answer for that leg, not a starting point -- --descent-scale must
    not multiply it, or 'pin leg 1 to the value flt9 flew' would not mean that."""
    lad = build_descent_ladder(sched, ANCHOR, anchor_leg=ANCHOR_LEG, scale=0.5,
                               overrides={LEG_START_G1: 0.026})
    assert lad[LEG_START_G1] == pytest.approx(0.026, abs=1e-9)
    assert lad[LEG_G1_G2] == pytest.approx(ANCHOR / 2.0, abs=1e-4)   # scale still applies


def test_zero_anchor_disables_the_whole_ladder(sched):
    assert build_descent_ladder(sched, 0.0, anchor_leg=ANCHOR_LEG) == [0.0] * len(sched)


def test_non_descending_anchor_leg_is_rejected(sched):
    """Leg 0 climbs slightly (-1.3 deg). Anchoring there would divide the whole table by
    a negative slope and invert every sign -- fail loudly instead."""
    with pytest.raises(ValueError, match="not a descending leg"):
        build_descent_ladder(sched, ANCHOR, anchor_leg=0)


def test_flt9_regression_the_old_single_value_over_descends_the_flat_legs(ladder):
    """The quantified defect, so the reason for the change survives in the suite.

    flt9 flew 0.026 on every leg from ag>=1. On the 1.9 deg G3->G4 leg the course drops
    only 1.9/17.1 = 11% of what the anchor leg drops, so the old command was ~9x the
    descent that leg actually needs."""
    old = ANCHOR
    assert old / ladder[LEG_G3_G4] > 8.0
    assert old / ladder[LEG_G4_FIN] > 10.0


def test_ladder_length_matches_the_course(sched, ladder):
    assert len(ladder) == len(sched) == 6
