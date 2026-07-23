"""Tests for the UE->NED setpoint conversion (nav_frames.py).

Locks the coordinate bookkeeping against the course JSON's own numbers so a
refactor can't silently flip an axis/sign. Does NOT validate the sim's true
axis convention — that is what tools/probe_position_target.py resolves live.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nav_frames import (  # noqa: E402
    climb_setpoint, gate_ned_setpoints, load_spawn_and_gates, ue_delta_to_ned,
)


def test_climb_is_negative_d_only():
    sp = climb_setpoint(3.0)
    assert sp[0] == 0.0 and sp[1] == 0.0
    assert sp[2] == -3.0   # NED down is negative for a climb


def test_start_delta_matches_json_bearing_check():
    # course_gates_cm.json meta.spawn_to_G1_bearing_check.vector_m = [-23.3,-0.9,0.51]
    spawn_cm, gates = load_spawn_and_gates()
    _, start_cm = gates[0]
    delta_ue_m = (start_cm - spawn_cm) / 100.0
    assert np.allclose(delta_ue_m, [-23.304, -0.9, 0.51], atol=1e-2)


def test_d_sign_follows_altitude():
    # START is slightly ABOVE spawn (dz>0 => D<0); FINISH is well BELOW (D>0).
    rows = gate_ned_setpoints(axes="world")
    assert rows[0]["ned"][2] < 0.0        # START above spawn
    assert rows[-1]["tag"] == "FINISH"
    assert rows[-1]["ned"][2] > 20.0      # FINISH ~25 m down


def test_world_negN_flips_north_only():
    d = np.array([10.0, 2.0, 3.0])
    w = ue_delta_to_ned(d, "world")
    n = ue_delta_to_ned(d, "world_negN")
    assert n[0] == -w[0]                   # North flipped
    assert n[1] == w[1] and n[2] == w[2]   # East, Down unchanged


def test_six_gates_ordered_start_to_finish():
    rows = gate_ned_setpoints(axes="world")
    assert len(rows) == 6
    assert rows[0]["tag"] == "START" and rows[-1]["tag"] == "FINISH"
    # monotonic in |N| (course marches steadily away from spawn)
    norths = [abs(r["ned"][0]) for r in rows]
    assert all(norths[i] <= norths[i + 1] + 1e-6 for i in range(len(norths) - 1))


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
