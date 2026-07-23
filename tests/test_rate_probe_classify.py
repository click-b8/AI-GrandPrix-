"""Tests for the rate-probe classifier (tools/probe_rate_vs_angle.classify).

Regression guard: the original classifier called the real v3385 pitch sweep
"RATE, no change needed" while the response was SIGN-INVERTED and 2.5x hot. The
fixed classifier must report sign and scale separately.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from probe_rate_vs_angle import classify  # noqa: E402


def test_real_pitch_sweep_flags_sign_and_scale():
    # measured pitch: cmd -0.30 -> +0.747 (inverted, 2.49x)
    r = classify(-0.30, first_q=0.70, last_q=0.747, measured=0.747)
    assert r["verdict"] == "RATE"
    assert not r["sign_ok"]
    assert r["K"] < 0                       # inverted
    assert "SIGN INVERTED" in r["issues"]
    assert any(i.startswith("SCALE") for i in r["issues"])
    assert abs(r["K"] + 2.49) < 0.05        # K ~ -2.49


def test_clean_rate_no_issues():
    r = classify(-0.30, first_q=-0.29, last_q=-0.30, measured=-0.30)
    assert r["verdict"] == "RATE" and r["sign_ok"] and r["issues"] == []
    assert abs(r["K"] - 1.0) < 0.02


def test_angle_decay_detected():
    r = classify(-0.30, first_q=-0.42, last_q=-0.02, measured=-0.05)
    assert r["verdict"] == "ANGLE"


def test_scale_only_reported():
    r = classify(-0.30, first_q=-0.14, last_q=-0.15, measured=-0.15)
    assert r["verdict"] == "RATE" and r["sign_ok"]
    assert r["issues"] == ["SCALE 0.50x"]
    assert abs(r["K"] - 0.5) < 0.02


def test_inconclusive_when_neither():
    # small first_q, collapses -> not a clean hold, not a clean decay-from-high
    r = classify(-0.30, first_q=0.05, last_q=0.02, measured=0.03)
    assert r["verdict"] == "INCONCLUSIVE"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
