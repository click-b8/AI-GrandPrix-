"""
Train == Deploy observation contract test.

Guards the 10-D vision+IMU observation defined in obsidian/observation-spec.md.
A divergence between the deploy builder (dcl_adapter) and the env builder
(drone_race_env) is the failure mode that has cost us weeks (z-axis sign,
FRD/FLU rates, faked frame stack). This test makes the contract executable.

Two layers:
  1. FRAME-CONVENTION ORACLE (runs now): asserts the contract's math —
     gravity-down at a TILTED attitude, FRD->FLU body-rate mapping, and the
     frame-stack channel order with a reversed-deque negative control. These
     encode the conventions and must hold before any builder is written.
  2. CROSS-BUILDER EQUALITY (skip-guarded until A1-A3 + B5 land): drives the
     deploy and env builders with one matched input and asserts byte-identical
     output. Converts from skip to hard assertion as the 10-D builders are
     implemented. MUST be green (un-skipped, passing) before GPU time.

Contract constants (obsidian/observation-spec.md):
  state = gravity_unit[3] (FLU) + body_rates[3] (FLU) + prev_action[4] = 10-D
  image = (FPV_FRAME_STACK*3, 48, 48) CHW uint8, channels oldest->newest, RGB
  C = diag(1, -1, -1)  maps FRD<->FLU (its own inverse)
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dcl_adapter import _euler_to_rotmat  # body->world (z-up) ZYX, exists today

C = np.diag([1.0, -1.0, -1.0]).astype(np.float64)

# The mandatory non-level test attitude and its contract gravity target.
TILT_ROLL_DEG = 30.0
TILT_PITCH_DEG = 30.0
TILT_YAW_DEG = 0.0
GRAVITY_UNIT_TILT = np.array([0.5, -0.433013, -0.75])  # FLU, from the spec


# ---------------------------------------------------------------------------
# Oracle helpers — the contract math, independent of any builder.
# ---------------------------------------------------------------------------

def _flu_gravity_unit(roll_deg, pitch_deg, yaw_deg):
    """Train-side gravity-down unit in FLU body: R.T @ [0,0,-1]."""
    R = _euler_to_rotmat(np.radians(roll_deg), np.radians(pitch_deg), np.radians(yaw_deg))
    return (R.T @ np.array([0.0, 0.0, -1.0])).astype(np.float64)


def _frd_static_accel_unit(roll_deg, pitch_deg, yaw_deg):
    """Deploy-side: a perfect static accelerometer in FRD measures specific
    force = -gravity in body. In FLU that is -gravity_unit; express in FRD via C."""
    return C @ (-_flu_gravity_unit(roll_deg, pitch_deg, yaw_deg))


def _distinguishable_frames(n=3, h=48, w=48):
    """n frames distinct across BOTH frame index and RGB channel, so a reversed
    stack or a swapped channel actually fails the assertion.
    frame_k[..., c] = 10*(k+1) + c   (k=0..n-1, c=0..2)."""
    frames = []
    for k in range(n):
        f = np.empty((h, w, 3), dtype=np.uint8)
        for c in range(3):
            f[..., c] = 10 * (k + 1) + c
        frames.append(f)
    return frames


def _oracle_stack_chw(frames_hwc):
    """Reference stacking exactly as the env does it: concatenate frames along
    the channel axis (axis=2, oldest->newest) to HWC, then VecTranspose to CHW."""
    stacked_hwc = np.concatenate(frames_hwc, axis=2)          # (H, W, 3*N)
    return np.transpose(stacked_hwc, (2, 0, 1)).copy()        # (3*N, H, W)


# ---------------------------------------------------------------------------
# Layer 1 — frame-convention oracle (runs now)
# ---------------------------------------------------------------------------

def test_gravity_contract_at_tilt_not_just_level():
    """gravity_unit agrees train-vs-deploy at a TILTED attitude (the case the
    old z-axis bug failed), and is genuinely non-level."""
    g_train = _flu_gravity_unit(TILT_ROLL_DEG, TILT_PITCH_DEG, TILT_YAW_DEG)

    # It must be the spec's tilted target, and NOT the level vector.
    np.testing.assert_allclose(g_train, GRAVITY_UNIT_TILT, atol=1e-6)
    assert not np.allclose(g_train, [0.0, 0.0, -1.0], atol=1e-3), \
        "test attitude is not actually tilted — would not catch the z-axis bug"
    np.testing.assert_allclose(np.linalg.norm(g_train), 1.0, atol=1e-6)

    # Deploy path: filter recovers gravity-down in FRD from the static accel
    # (= -accel_unit), then maps FRD->FLU with C. Must reproduce g_train.
    accel_frd_unit = _frd_static_accel_unit(TILT_ROLL_DEG, TILT_PITCH_DEG, TILT_YAW_DEG)
    g_deploy = C @ (-accel_frd_unit)
    np.testing.assert_allclose(g_deploy, g_train, atol=1e-6)


def test_gravity_contract_at_level_sanity():
    g = _flu_gravity_unit(0.0, 0.0, 0.0)
    np.testing.assert_allclose(g, [0.0, 0.0, -1.0], atol=1e-6)


def test_body_rates_frd_to_flu_negates_pitch_and_yaw():
    """C @ gyro_frd keeps roll, negates pitch & yaw. Distinct components so a
    missed negation fails."""
    gyro_frd = np.array([0.1, 0.2, 0.3])
    flu = C @ gyro_frd
    np.testing.assert_allclose(flu, [0.1, -0.2, -0.3], atol=1e-9)


def test_frame_stack_order_and_reversed_negative_control():
    """Channel order is oldest->newest; a reversed stack must differ. Uses
    distinguishable frames so the reversal is actually detectable."""
    frames = _distinguishable_frames(n=3)
    correct = _oracle_stack_chw(frames)
    reversed_stack = _oracle_stack_chw(list(reversed(frames)))

    assert correct.shape == (9, 48, 48)
    assert correct.dtype == np.uint8
    # Oldest frame occupies the first 3 channels, newest the last 3.
    assert correct[0, 0, 0] == 10   # frame0, channel R
    assert correct[6, 0, 0] == 30   # frame2, channel R
    # Negative control: with distinguishable frames, reversed != correct.
    assert not np.array_equal(correct, reversed_stack), \
        "reversed stack equals correct — frames not distinguishable enough"


def test_camera_fov_follows_intrinsics_not_stated_90():
    """Camera must follow the Elodin intrinsics (VFoV≈58.72°, HFoV≈90°, 16:9),
    NOT the stated 90° VFoV on a square render. Pins config against regression."""
    from config import (FPV_FOV, FPV_ASPECT, FPV_RENDER_W, FPV_RENDER_H,
                        FPV_RESOLUTION)
    # Vertical FoV is the intrinsics value, not the stated 90.
    assert abs(FPV_FOV - 58.72) < 0.1, "FPV_FOV must be the ~58.72° vertical FoV"
    assert FPV_FOV < 90.0
    # Render buffer is 16:9 so MuJoCo derives HFoV from fovy + aspect.
    np.testing.assert_allclose(FPV_RENDER_W / FPV_RENDER_H, 16.0 / 9.0, rtol=1e-3)
    np.testing.assert_allclose(FPV_ASPECT, 16.0 / 9.0, rtol=1e-3)
    # Horizontal FoV implied at 16:9 must be ~90°.
    hfov = 2.0 * np.degrees(np.arctan(np.tan(np.radians(FPV_FOV) / 2.0) * FPV_ASPECT))
    np.testing.assert_allclose(hfov, 90.0, atol=0.5)
    # Final policy image is a square 48x48 (16:9 squashed), matching deploy.
    assert FPV_RESOLUTION == 48


# ---------------------------------------------------------------------------
# Layer 2 — cross-builder equality (skip until 10-D builders exist)
# ---------------------------------------------------------------------------

def _builders_ready():
    """True once A2/A3 (deploy) and B5 (env) 10-D builders are implemented."""
    try:
        from config import VISION_STATE_DIM
    except Exception:
        return False
    if VISION_STATE_DIM != 10:
        return False
    try:
        import attitude_filter  # noqa: F401  (A2 GravityEstimator)
    except Exception:
        return False
    return True


_SKIP_REASON = ("10-D builders not implemented yet (A2 attitude_filter + A3 "
                "dcl_adapter 10-D + B5 env _get_minimal_state, VISION_STATE_DIM=10). "
                "Per the locked sequencing this test must be un-skipped and green "
                "before GPU time.")


@pytest.mark.skipif(not _builders_ready(), reason=_SKIP_REASON)
def test_deploy_state_equals_env_state_at_tilt():
    """Feed one matched input (tilted attitude, known gyro, prev_action) to both
    builders; assert byte-identical 10-D state (gravity + rates + prev_action).

    Implementation target once A2/A3/B5 land:
      - deploy: settle attitude_filter on the static FRD accel for the tilt +
        zero gyro, build state via dcl_adapter; map gyro via C.
      - env: set qpos quaternion for the tilt and qvel[3:6] = body rates,
        call _get_minimal_state().
      - np.testing.assert_allclose(deploy_state, env_state, atol=1e-5).
    """
    pytest.skip("activate body when A2/A3/B5 are implemented — see docstring")


@pytest.mark.skipif(not _builders_ready(), reason=_SKIP_REASON)
def test_deploy_image_stack_matches_oracle_order():
    """Push distinguishable consecutive frames through the deploy image builder
    and assert it equals the oracle stacker (oldest->newest CHW), and that a
    reversed input does NOT match — proving the deque order is correct."""
    pytest.skip("activate body when A3 frame-stack deque is implemented")
