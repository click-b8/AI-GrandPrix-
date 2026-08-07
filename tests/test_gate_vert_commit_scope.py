"""VERTICAL COMMIT SCOPE -- required at gate 1, fatal after it.

Both directions are proven by flights, so both are pinned here:

  flt41, commit OFF          -> GATE 1 CLIPS. The kd_v term reads the gate's geometric
                                fall in frame as sink, drives the trim to the +0.06
                                clamp and thrust to 0.340, and the drone climbs into the
                                top bar. Collision at t=4.6 s, tumble.
  flt40, commit ON everywhere -> GATE 2 PLUNGES. The PD is frozen through the crossing,
                                so the trim sits at ~0 (thrust 0.249) while v runs
                                +0.17 -> -0.90 and the arrest never fires.

Gate 1 is a near-level approach where the close-range D term is pure noise; the
descending legs arrive with real sink that something has to stop. Hence an UPPER bound
on active_gate, defaulting to gate 1 only -- the opposite shape to the LATERAL commit's
--gate-commit-from-ag lower bound, which is deliberate and easy to get backwards.
"""
from tools.schedule_flier import build_parser


def committed(ag, sz_f, args):
    """The shipped predicate, as mode_coast_tube evaluates it each tick."""
    return (args.gate_vert_commit_size > 0.0
            and ag <= args.gate_vert_commit_to_ag
            and sz_f >= args.gate_vert_commit_size)


def parse(*argv):
    return build_parser().parse_args(list(argv))


def test_the_scope_defaults_to_gate_1_only():
    assert vars(parse())["gate_vert_commit_to_ag"] == 0


def test_it_engages_on_the_gate_1_approach():
    """flt41: without this, gate 1 clips."""
    a = parse("--gate-vert-commit-size", "0.18")
    assert committed(0, 0.20, a)
    assert committed(0, 0.60, a)


def test_it_never_engages_on_the_descending_legs():
    """flt40: with it engaged here, gate 2 plunges -- the arrest can never fire."""
    a = parse("--gate-vert-commit-size", "0.18")
    for ag in (1, 2, 3, 4, 5):
        assert not committed(ag, 0.90, a), f"the commit engaged on ag={ag}"


def test_the_size_threshold_still_gates_within_scope():
    a = parse("--gate-vert-commit-size", "0.18")
    assert not committed(0, 0.17, a)
    assert committed(0, 0.18, a)


def test_size_zero_disables_it_everywhere():
    a = parse("--gate-vert-commit-size", "0")
    assert not committed(0, 0.99, a)


def test_the_scope_is_an_upper_bound_not_a_lower_one():
    """The lateral commit uses a FROM-ag lower bound and this uses a TO-ag upper bound;
    confusing them would apply the vertical commit exactly where it is fatal."""
    a = parse("--gate-vert-commit-size", "0.18", "--gate-vert-commit-to-ag", "1")
    assert committed(0, 0.5, a) and committed(1, 0.5, a)
    assert not committed(2, 0.5, a)
    d = vars(parse())
    assert d["gate_vert_commit_to_ag"] == 0 and d["gate_commit_from_ag"] == 2


def test_the_flight_config_parses_and_scopes_as_intended():
    """The shipped command line, end to end."""
    a = parse("--coast-tube", "--gate-vert", "--const-thrust", "0.275",
              "--descent-bias-leg0", "-0.005", "--descent-bias-leg1", "0.026",
              "--vert-auth-down", "0.045,0.08", "--vert-auth-up", "0.06,0.10",
              "--gate-vert-commit-size", "0.18", "--gate-vert-commit-tau", "0.20",
              "--gate-vert-commit-to-ag", "0",
              "--thrust-slew", "0.6", "--thrust-slew-down", "0.6",
              "--gate-max-bank-deg", "11", "--post-gate1-bank", "5",
              "--no-pitch-hold", "--tick-log", "filt42.csv")
    assert committed(0, 0.20, a) and not committed(1, 0.99, a)
    assert a.vert_auth_down == "0.045,0.08" and a.vert_auth_up == "0.06,0.10"
    assert a.const_thrust == 0.275 and a.no_pitch_hold is True
