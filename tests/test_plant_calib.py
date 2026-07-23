"""Tests for the per-axis plant rate calibration (dcl_mavlink_adapter).

Locks the correction identity: for an intended normalised command, the WIRE
rate we send, run through the plant transfer gyro = K * wire, recovers exactly
the intended rate (command * MAX_BODY_RATE). Guards the measured K values and
the sign inversion too.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcl_mavlink_adapter import MAX_BODY_RATE, PLANT_RATE_CALIB, wire_body_rate  # noqa: E402


def test_calib_values_v3385():
    # Measured 2026-07-22 via tools/probe_rate_vs_angle.py on sim build v3385.
    assert PLANT_RATE_CALIB == {"roll": -2.55, "pitch": -2.49, "yaw": -2.20}


def test_correction_recovers_intended_rate():
    # wire = R/K ; plant delivers K*wire ; must equal intended R for every axis.
    for axis, K in PLANT_RATE_CALIB.items():
        for norm in (-1.0, -0.3, 0.0, 0.25, 1.0):
            intended = norm * MAX_BODY_RATE
            wire = wire_body_rate(axis, norm)
            plant_delivers = K * wire
            assert abs(plant_delivers - intended) < 1e-9, (axis, norm)


def test_wire_is_sign_inverted_and_scaled():
    # All axes have K<0, so the wire command opposes the normalised command...
    for axis in PLANT_RATE_CALIB:
        assert wire_body_rate(axis, 0.5) < 0.0
        assert wire_body_rate(axis, -0.5) > 0.0
    # ...and is scaled down by |K| (|wire| < |intended| since |K|>1).
    intended = 0.5 * MAX_BODY_RATE
    assert abs(wire_body_rate("pitch", 0.5)) < intended


def test_zero_command_zero_wire():
    for axis in PLANT_RATE_CALIB:
        assert wire_body_rate(axis, 0.0) == 0.0


def test_yaw_softer_than_roll_pitch():
    # yaw K magnitude is distinctly smaller -> a given command sends MORE wire
    # rate on yaw than on roll/pitch.
    assert abs(wire_body_rate("yaw", 0.5)) > abs(wire_body_rate("roll", 0.5))
    assert abs(wire_body_rate("yaw", 0.5)) > abs(wire_body_rate("pitch", 0.5))


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
