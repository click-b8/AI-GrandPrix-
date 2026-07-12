"""Unit tests for the 10-D vision+IMU observation (B5 env builder).

state = gravity_unit(3, FLU) + body_rates(3, FLU) + prev_action(4) = 10-D
image = (FPV_RESOLUTION, FPV_RESOLUTION, 3*FPV_FRAME_STACK) HWC uint8
Skips gracefully if MuJoCo cannot create a GL context (headless without EGL).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import VISION_STATE_DIM, FPV_RESOLUTION, FPV_FRAME_STACK, SPAWN_PITCH_DEG


def _make_env_reset():
    from drone_race_env import DroneRaceEnv
    try:
        env = DroneRaceEnv(render_mode=None, domain_rand=False, difficulty=1.0,
                           vision_mode=True)
        obs, _ = env.reset()
    except Exception as e:  # GL context unavailable on a headless box
        pytest.skip(f"env render unavailable: {e}")
    return env, obs


def test_config_vision_state_dim_is_10():
    assert VISION_STATE_DIM == 10


def test_obs_dict_shapes_and_dtypes():
    env, obs = _make_env_reset()
    try:
        assert set(obs.keys()) == {"image", "state"}
        assert obs["state"].shape == (10,)
        assert obs["state"].dtype == np.float32
        assert obs["image"].shape == (FPV_RESOLUTION, FPV_RESOLUTION, 3 * FPV_FRAME_STACK)
        assert obs["image"].dtype == np.uint8
        assert np.all(np.isfinite(obs["state"]))
    finally:
        env.close()


def test_state_layout_and_ranges_at_rest():
    env, obs = _make_env_reset()
    try:
        state = obs["state"]
        gravity, body_rates, prev_action = state[0:3], state[3:6], state[6:10]
        # Gravity is a UNIT vector (contract), pointing down in FLU.
        np.testing.assert_allclose(np.linalg.norm(gravity), 1.0, atol=1e-4)
        # Spawn is intentionally NOT level: the drone rests SPAWN_PITCH_DEG (-17.8°)
        # nose-down (powered de-risk, hypothesis A). Gravity-down in FLU body is then
        # [sin|p|, 0, -cos|p|], which matches the deploy filter's rest value and closes
        # the train/deploy start-of-episode mismatch. (Level [0,0,-1] would be wrong now.)
        p = np.radians(abs(SPAWN_PITCH_DEG))
        np.testing.assert_allclose(gravity, [np.sin(p), 0.0, -np.cos(p)], atol=1e-2)
        # At rest: body rates ~0, prev_action zero, all finite.
        np.testing.assert_allclose(body_rates, [0.0, 0.0, 0.0], atol=1e-3)
        np.testing.assert_allclose(prev_action, np.zeros(4), atol=1e-6)
        assert np.all(np.isfinite(state))
    finally:
        env.close()


def test_gravity_unit_tracks_true_attitude_at_tilt():
    import mujoco
    from drone_race_env import _quat_to_rotmat
    env, _ = _make_env_reset()
    try:
        # A genuinely tilted attitude (roll=30, pitch=30) as a wxyz quaternion.
        r, p, y = np.radians(30.0), np.radians(30.0), 0.0
        cr, sr = np.cos(r / 2), np.sin(r / 2)
        cp, sp = np.cos(p / 2), np.sin(p / 2)
        cy, sy = np.cos(y / 2), np.sin(y / 2)
        quat = np.array([cr * cp * cy + sr * sp * sy,
                         sr * cp * cy - cr * sp * sy,
                         cr * sp * cy + sr * cp * sy,
                         cr * cp * sy - sr * sp * cy])
        env._data.qpos[3:7] = quat
        env._data.qvel[:] = 0.0
        mujoco.mj_forward(env._model, env._data)
        state = env._get_minimal_state()
        R = _quat_to_rotmat(*quat)
        expected = (R.T @ np.array([0.0, 0.0, -1.0])).astype(np.float32)
        np.testing.assert_allclose(state[0:3], expected, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(state[0:3]), 1.0, atol=1e-5)
        assert not np.allclose(state[0:3], [0.0, 0.0, -1.0], atol=1e-2)  # really tilted
        assert state.shape == (10,)
    finally:
        env.close()
