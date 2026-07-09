"""Unit tests for the UE->MuJoCo course conversion machinery.

Coordinate-independent: uses only synthetic inputs (no real/extracted course
coordinates), so no discarded data survives here as if it were ground truth.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from track import (ue_cm_to_env_m, ue_yaw_to_env_yaw, _horizontal_turn_signs,
                   assert_course_matches_source)


def test_position_cm_to_m_and_y_flip():
    # Pure synthetic case: /100 (cm->m) and Y negated (LH -> RH).
    np.testing.assert_allclose(ue_cm_to_env_m([-1000, 500, 250]), [-10.0, -5.0, 2.5])
    np.testing.assert_allclose(ue_cm_to_env_m([0, 0, 0]), [0.0, 0.0, 0.0])


def test_yaw_sign_and_offset():
    np.testing.assert_allclose(ue_yaw_to_env_yaw(0.0), 0.0, atol=1e-9)
    np.testing.assert_allclose(ue_yaw_to_env_yaw(90.0), -np.pi / 2, atol=1e-9)  # heading negates
    np.testing.assert_allclose(ue_yaw_to_env_yaw(-90.0, normal_offset_deg=90.0), 0.0, atol=1e-9)


def test_turn_sign_left_is_positive():
    path = [(0, 0, 0), (1, 0, 0), (1, 1, 0)]   # +X then +Y = left turn (CCW)
    assert _horizontal_turn_signs(path)[0] == 1.0


def test_sanity_check_passes_and_catches_mirror():
    # Synthetic raw UE positions (cm), left-handed Z-up.
    src_cm = np.array([(0, 0, 0), (1000, 0, 0), (1000, 1000, 0), (2000, 1000, 0)], float)
    env = np.array([ue_cm_to_env_m(p) for p in src_cm])       # correct conversion
    assert_course_matches_source(src_cm, env)                  # matches -> ok

    bad = env.copy()
    bad[:, 1] *= -1                                            # wrong extra Y flip (handedness)
    with pytest.raises(AssertionError):
        assert_course_matches_source(src_cm, bad)             # same spacing, flipped turns


def test_sanity_check_catches_scale_error():
    src_cm = np.array([(0, 0, 0), (1000, 0, 0), (1000, 1000, 0)], float)
    env = np.array([ue_cm_to_env_m(p) for p in src_cm])
    bad = env.copy()
    bad[:, 0] *= 10.0                                          # X scale slip
    with pytest.raises(AssertionError):
        assert_course_matches_source(src_cm, bad)
