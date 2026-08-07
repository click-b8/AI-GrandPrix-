"""PER-LEG GATE-VERT AUTHORITY -- a TABLE indexed by active_gate, not a gate fence.

WHY PER-LEG AT ALL, measured on filt31:

  FIRST APPROACH (ag==0), near-level:
      in the last second the trim reaches +0.048 against a 0.06 clamp while |v_f| is
      only 0.024 -- already near the stop on a gate that is CENTRED. Raising the clamp
      globally therefore buys nothing but more over-climb into that gate's top bar.

  NEXT LEG (ag==1), an 8.6 m drop:
      the trim sat pinned to the DOWN clamp for 83% of the leg and v_err still read
      +0.5..+0.8 -- saturated at maximum descent and STILL riding above the line.

WHY A TABLE AND NOT AN ag<1 / ag>=1 FENCE: which leg is near-level and which is steep is
a property of the COURSE, not of the controller. Baking "gate 1" and "gate 2" into the
control law makes it wrong the moment the track or the gate count changes. The loop
indexes a vector the caller supplies and clamps past the end -- a different course is a
different table, not a different branch. Same shape as the descent ladder.
"""
import pytest

from tools.schedule_flier import build_parser, build_vertical_schedule


def sched(down=None, up=None, n=6, d_dflt=0.060, u_dflt=0.030):
    return build_vertical_schedule(n, down, up, d_dflt, u_dflt)


def auth(tbl, ag):
    """The shipped lookup: clamp to the last entry."""
    return tbl[min(ag, len(tbl) - 1)]


# ---------------------------------------------------------------------------------
# Discipline: unchanged unless a table is supplied
# ---------------------------------------------------------------------------------
def test_the_table_flags_default_to_unset():
    d = vars(build_parser().parse_args([]))
    assert d["vert_auth_down"] is None
    assert d["vert_auth_up"] is None


def test_no_table_means_the_scalar_authority_on_every_leg():
    tbl = sched()
    assert tbl == [(0.060, 0.030)] * 6


def test_one_axis_can_be_tabled_while_the_other_stays_global():
    tbl = sched(up="0.05,0.10")
    assert [u for _, u in tbl] == [0.05] + [0.10] * 5
    assert {d for d, _ in tbl} == {0.060}


# ---------------------------------------------------------------------------------
# The table's fill and clamp rules -- what makes it course-independent
# ---------------------------------------------------------------------------------
def test_a_short_table_is_extended_by_its_last_entry():
    """'0.045,0.08' must mean 'gentle first, aggressive thereafter' on a course of ANY
    length -- that is what lets one table serve a 6-gate and a 12-gate track."""
    tbl = sched(down="0.045,0.08", up="0.05,0.10", n=6)
    assert tbl[0] == (0.045, 0.05)
    for ag in range(1, 6):
        assert tbl[ag] == (0.08, 0.10)


def test_an_active_gate_past_the_end_clamps_to_the_last_entry():
    tbl = sched(down="0.045,0.08", up="0.05,0.10", n=3)
    assert auth(tbl, 99) == auth(tbl, len(tbl) - 1) == (0.08, 0.10)


def test_a_full_length_table_is_taken_entry_for_entry():
    """Per-gate tuning, which is the point of shipping a vector rather than two levels."""
    tbl = sched(down="0.01,0.02,0.03,0.04,0.05,0.06",
                up="0.11,0.12,0.13,0.14,0.15,0.16", n=6)
    for ag in range(6):
        assert tbl[ag] == (round(0.01 * (ag + 1), 3), round(0.11 + 0.01 * ag, 3))


def test_whitespace_and_a_single_value_are_accepted():
    assert sched(down=" 0.05 , 0.09 ", n=3) == [(0.05, 0.030), (0.09, 0.030),
                                                (0.09, 0.030)]
    assert sched(down="0.07", n=3) == [(0.07, 0.030)] * 3


def test_a_negative_authority_is_rejected():
    """These are magnitudes; a negative would silently invert the clamp."""
    with pytest.raises(ValueError):
        sched(down="0.045,-0.08")


def test_the_law_carries_no_gate_number():
    """The guard against regressing to a fence: nothing in the table build depends on a
    named gate, only on the index it is asked for."""
    a = sched(down="0.045,0.08", up="0.05,0.10", n=6)
    b = sched(down="0.045,0.08", up="0.05,0.10", n=12)
    assert a[:6] == b[:6]
    assert b[11] == b[1], "extending the course must not change any leg's meaning"


# ---------------------------------------------------------------------------------
# The starting tuning from the flight data
# ---------------------------------------------------------------------------------
FLIGHT_DOWN, FLIGHT_UP = "0.045,0.08", "0.05,0.10"


def test_the_two_regimes_are_genuinely_decoupled():
    """THE POINT: raising the arrest for the descending legs must not raise it for the
    first approach, because that is what was clipping its top bar."""
    tbl = sched(down=FLIGHT_DOWN, up=FLIGHT_UP)
    assert auth(tbl, 1)[1] > auth(tbl, 0)[1], "the descending legs must arrest harder"
    assert auth(tbl, 1)[0] > auth(tbl, 0)[0], "the descending legs must descend harder"
    assert auth(tbl, 0)[1] <= 0.06, (
        "the first approach's climb authority is back at the value that over-climbed on "
        "filt31 (+0.048 of trim against a 0.06 clamp with the gate already centred)")


def test_the_climb_authority_stays_inside_the_plant_ceiling():
    """A climb trim can only ever command up to cfg.ol_thrust_hi; authority beyond the
    headroom above a leg baseline is partly unreachable."""
    from vq1_vision_servo import ServoConfig
    tbl = sched(down=FLIGHT_DOWN, up=FLIGHT_UP)
    assert max(u for _, u in tbl) <= ServoConfig().ol_thrust_hi - 0.24
