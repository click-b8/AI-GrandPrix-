"""Unit tests for A2 GravityEstimator (attitude_filter.GravityEstimator).

Checks: (1) recovers gravity-down at known tilt angles, (2) rejects a synthetic
~138 g spike and stays stable, (3) output is finite with norm ~= 9.81 in FRD.
Conventions mirror tests/test_observation_contract.py (FRD, C = diag(1,-1,-1)).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from attitude_filter import GravityEstimator, GRAVITY
from dcl_adapter import _euler_to_rotmat  # body->world (z-up) ZYX, exists today

C = np.diag([1.0, -1.0, -1.0])
RATE = 115.0
DT = 1.0 / RATE


def _static_accel_frd(roll_deg, pitch_deg, yaw_deg, g=GRAVITY):
    """FRD accelerometer reading (specific force) and expected gravity-down dir
    for a static tilt: accel = -gravity_down_frd."""
    R = _euler_to_rotmat(np.radians(roll_deg), np.radians(pitch_deg), np.radians(yaw_deg))
    g_flu = R.T @ np.array([0.0, 0.0, -1.0])   # gravity-down, FLU unit
    g_frd = C @ g_flu                           # gravity-down, FRD unit
    return -g_frd * g, g_frd


def test_level_rest_is_down_plus_z():
    est = GravityEstimator(rate_hz=RATE)
    accel, _ = _static_accel_frd(0, 0, 0)
    est.reset()
    g = None
    for _ in range(300):
        g = est.update(accel, np.zeros(3), DT)
    assert np.all(np.isfinite(g))
    np.testing.assert_allclose(np.linalg.norm(g), GRAVITY, atol=0.05)
    np.testing.assert_allclose(g / GRAVITY, [0.0, 0.0, 1.0], atol=1e-3)


@pytest.mark.parametrize("roll,pitch,yaw", [(30, 30, 0), (-20, 15, 0), (0, -45, 0)])
def test_recovers_known_tilt(roll, pitch, yaw):
    est = GravityEstimator(rate_hz=RATE)
    est.reset()  # start LEVEL so convergence is genuinely exercised
    accel, g_frd = _static_accel_frd(roll, pitch, yaw)
    g = None
    for _ in range(3000):
        g = est.update(accel, np.zeros(3), DT)
    assert np.all(np.isfinite(g))
    np.testing.assert_allclose(np.linalg.norm(g), GRAVITY, atol=0.05)
    np.testing.assert_allclose(g / np.linalg.norm(g), g_frd, atol=1e-2)


def test_rejects_138g_spike_and_stays_stable():
    est = GravityEstimator(rate_hz=RATE)
    est.reset()
    accel, _ = _static_accel_frd(30, 30, 0)
    for _ in range(3000):
        est.update(accel, np.zeros(3), DT)
    before = est.gravity
    spike = np.array([-388.0, -838.0, -992.0])  # |a| ~ 1355 (~138 g)
    assert np.linalg.norm(spike) > 100
    for _ in range(50):
        est.update(spike, np.zeros(3), DT)  # gyro-only through the whole window
    after = est.gravity
    assert np.all(np.isfinite(after))
    np.testing.assert_allclose(after, before, atol=1e-6)  # spike ignored entirely


def test_gyro_only_rotation_preserves_norm_no_nan():
    est = GravityEstimator(rate_hz=RATE)
    est.reset()
    g = None
    for _ in range(1000):
        # |accel| = 0 -> outside gate -> gyro-only; spin about body-x at 2 rad/s.
        g = est.update(np.zeros(3), np.array([2.0, 0.0, 0.0]), DT)
    assert np.all(np.isfinite(g))
    np.testing.assert_allclose(np.linalg.norm(g), GRAVITY, atol=0.05)
