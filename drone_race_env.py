"""Gymnasium environment: Autonomous drone racing (AI Grand Prix aligned).

Architecture matches Swift (Nature 2023) and MonoRace (A2RL x DCL 2025):
- CTBR action space: [collective_thrust, roll_rate, pitch_rate, yaw_rate]
- Low-level rate controller runs at physics rate (simulates Betaflight)
- 24D observation: drone state + gate relative pose + previous action
- Domain randomization for sim-to-real transfer
"""

import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from collections import deque

from config import (
    SIM_FREQ, CTRL_FREQ, EPISODE_LENGTH_SEC,
    DRONE_MASS, DRONE_ARM_LENGTH, DRONE_INERTIA,
    MAX_COLLECTIVE_THRUST, HOVER_THRUST_NORMALIZED, GRAVITY,
    MAX_BODY_RATE, MAX_SPEED,
    OBS_DIM, ACT_DIM,
    RATE_KP, RATE_KD,
    REWARD_GATE_PASSED, REWARD_PROGRESS, REWARD_PERCEPTION,
    REWARD_CMD_SMOOTHNESS, REWARD_CRASH_PENALTY, REWARD_BODY_RATE_PENALTY,
    REWARD_TIME_PENALTY,
    GATE_TOLERANCE, TRACK_BOUNDS,
    DOMAIN_RAND, MASS_RANGE, THRUST_NOISE, LATENCY_STEPS, DRAG_COEFF_RANGE,
    MOTOR_TAU,
    OBS_NOISE_POS, OBS_NOISE_VEL, OBS_NOISE_RPY, OBS_NOISE_ANGVEL,
    OBS_NOISE_GATE_POS, OBS_NOISE_GATE_YAW, OBS_DELAY_STEPS,
    FPV_RESOLUTION, FPV_FOV, FPV_TILT_DEG, FPV_FRAME_STACK, VISION_STATE_DIM,
    VIO_GYRO_BIAS_INSTABILITY, VIO_GYRO_WHITE_NOISE,
    VIO_ACCEL_BIAS, VIO_ACCEL_NOISE,
    VIO_VEL_DRIFT_RATE, VIO_POS_DRIFT_RATE, VIO_UPDATE_RATE,
    MOTION_BLUR_SAMPLES,
    EVENT_CAMERA_ENABLED, EVENT_CONTRAST_THRESHOLD_POS, EVENT_CONTRAST_THRESHOLD_NEG,
    EVENT_CONTRAST_NOISE, EVENT_REFRACTORY_PERIOD, EVENT_FRAME_BINS,
)
from track import get_gate_positions, get_gate_yaws, build_gate_xml


def _quat_to_rpy(w, x, y, z):
    """Quaternion (w,x,y,z) -> roll, pitch, yaw."""
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return roll, pitch, yaw


def _quat_to_rotmat(w, x, y, z):
    """Quaternion (w,x,y,z) -> 3x3 rotation matrix (body-to-world)."""
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


def _rotmat_to_6d(R):
    """Rotation matrix -> 6D representation (first two columns).

    From event-sharp-nerf-drones (Zou et al. 2026): 6D rotation representation
    via Gram-Schmidt is continuous and avoids gimbal lock / quaternion
    discontinuities. Superior for CNN-based pose regression.
    See Zhou et al. "On the Continuity of Rotation Representations" (CVPR 2019).
    """
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


def _build_mjcf(mass, inertia):
    """Build MuJoCo XML. No actuators — we apply forces directly."""
    arm = DRONE_ARM_LENGTH
    gate_bodies = build_gate_xml()
    # FPV camera tilt axis (camera looks along -Z in MuJoCo camera frame)
    tilt_rad = np.radians(-FPV_TILT_DEG)
    cam_ax_y = np.sin(tilt_rad)
    cam_ax_z = np.cos(tilt_rad)

    xml = f"""
    <mujoco model="aigp_racer">
      <option timestep="{1.0 / SIM_FREQ}" gravity="0 0 -{GRAVITY}"/>
      <visual>
        <global offwidth="1280" offheight="960"/>
      </visual>

      <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.22 0.2" rgb2="0.26 0.28 0.26"
                 width="512" height="512"/>
        <material name="grid_mat" texture="grid" texrepeat="20 20" reflectance="0.02"/>
      </asset>

      <worldbody>
        <geom type="plane" size="50 50 0.1" material="grid_mat"/>
        <light directional="true" diffuse="0.9 0.9 0.9" pos="0 0 15" dir="0 0 -1"/>
        <light directional="true" diffuse="0.3 0.3 0.3" pos="15 15 10" dir="-1 -1 -1"/>

        <body name="drone" pos="0 -2 2.5">
          <joint type="free"/>
          <inertial mass="{mass}" pos="0 0 0"
                   diaginertia="{inertia[0]} {inertia[1]} {inertia[2]}"/>

          <!-- Frame (X config, 5.1" racing quad) -->
          <geom type="box" size="{arm*0.25} {arm*0.25} 0.012" rgba="0.15 0.15 0.15 1" mass="0"/>
          <geom type="capsule" fromto="{arm} {arm} 0 -{arm} -{arm} 0" size="0.006" rgba="0.6 0.05 0.05 1" mass="0"/>
          <geom type="capsule" fromto="{arm} -{arm} 0 -{arm} {arm} 0" size="0.006" rgba="0.6 0.05 0.05 1" mass="0"/>

          <!-- Prop discs -->
          <geom type="cylinder" pos="{arm} {arm} 0.005" size="0.065 0.002" rgba="0.7 0.7 0.7 0.4" mass="0"/>
          <geom type="cylinder" pos="-{arm} {arm} 0.005" size="0.065 0.002" rgba="0.7 0.7 0.7 0.4" mass="0"/>
          <geom type="cylinder" pos="-{arm} -{arm} 0.005" size="0.065 0.002" rgba="0.7 0.7 0.7 0.4" mass="0"/>
          <geom type="cylinder" pos="{arm} -{arm} 0.005" size="0.065 0.002" rgba="0.7 0.7 0.7 0.4" mass="0"/>

          <!-- Front indicator -->
          <geom type="sphere" pos="{arm*0.35} 0 0.015" size="0.008" rgba="0 1 0 1" mass="0"/>

          <!-- FPV camera: forward-facing, tilted down -->
          <camera name="fpv" pos="{arm*0.3} 0 0.02" xyaxes="0 -1 0 {cam_ax_y} 0 {cam_ax_z}" fovy="{FPV_FOV}"/>
          <!-- Gimbal camera: stabilized, dynamically aimed at next gate -->
          <camera name="gimbal" pos="0 0 0.02" xyaxes="0 -1 0 0 0 1" fovy="{FPV_FOV}"/>
        </body>

        {gate_bodies}
      </worldbody>
    </mujoco>
    """
    return xml


class VIOSimulator:
    """Simulated Visual-Inertial Odometry (VIO) pipeline.

    Models the noisy state estimation that a real drone gets from its
    IMU + camera VIO system (e.g. Intel RealSense T265, or custom VIO).
    Based on Swift (Nature 2023) sim-to-real transfer approach.

    IMU runs at physics rate: integrates gyro/accel with bias + noise.
    Visual updates run at camera rate (~30Hz): corrects drift using
    feature tracking (simulated as noisy ground-truth snapshots).
    """

    def __init__(self, difficulty=1.0, rng=None):
        self._difficulty = difficulty
        self._rng = rng or np.random.default_rng()
        # VIO visual update happens every N control steps (~30Hz at 100Hz ctrl)
        self._visual_update_interval = max(1, CTRL_FREQ // VIO_UPDATE_RATE)
        self._ctrl_step_counter = 0
        # State estimates
        self.est_pos = np.zeros(3, dtype=np.float64)
        self.est_vel = np.zeros(3, dtype=np.float64)
        self.est_angvel = np.zeros(3, dtype=np.float64)
        # Biases (random walk)
        self._gyro_bias = np.zeros(3, dtype=np.float64)
        self._accel_bias = np.zeros(3, dtype=np.float64)

    def reset(self, true_pos, true_vel, true_quat):
        """Initialize VIO state from ground truth at episode start."""
        self.est_pos = np.array(true_pos, dtype=np.float64)
        self.est_vel = np.array(true_vel, dtype=np.float64)
        self.est_angvel = np.zeros(3, dtype=np.float64)
        self._gyro_bias = np.zeros(3, dtype=np.float64)
        self._accel_bias = np.zeros(3, dtype=np.float64)
        self._ctrl_step_counter = 0

    def imu_update(self, true_angvel, true_accel, dt):
        """Simulate one IMU measurement + integration step.

        Called each physics substep. Adds gyro/accel bias + white noise,
        then dead-reckons velocity from accelerometer.
        """
        d = self._difficulty

        # Gyro bias random walk (bias instability)
        self._gyro_bias += self._rng.normal(
            0, VIO_GYRO_BIAS_INSTABILITY * d * np.sqrt(dt), 3
        )
        # Accel bias random walk
        self._accel_bias += self._rng.normal(
            0, VIO_ACCEL_BIAS * d * 0.1 * np.sqrt(dt), 3
        )

        # Noisy gyro reading
        gyro_noise = self._rng.normal(0, VIO_GYRO_WHITE_NOISE * d, 3)
        self.est_angvel = true_angvel + self._gyro_bias + gyro_noise

        # Noisy accelerometer reading + integrate for velocity
        accel_noise = self._rng.normal(0, VIO_ACCEL_NOISE * d, 3)
        measured_accel = true_accel + self._accel_bias + accel_noise
        self.est_vel += measured_accel * dt

        # Velocity drift (models VIO integration drift)
        self.est_vel += self._rng.normal(
            0, VIO_VEL_DRIFT_RATE * d * dt, 3
        )

        # Position dead-reckoning from velocity
        self.est_pos += self.est_vel * dt
        # Position drift
        self.est_pos += self._rng.normal(
            0, VIO_POS_DRIFT_RATE * d * dt, 3
        )

    def visual_update(self, true_pos, true_vel):
        """Periodic visual correction (simulates feature-tracking/VIO fusion).

        Called at control rate; only applies correction every N steps
        to simulate the slower camera frame rate (~30Hz).
        """
        self._ctrl_step_counter += 1
        if self._ctrl_step_counter % self._visual_update_interval != 0:
            return

        d = self._difficulty

        # Visual correction snaps estimates toward ground truth with noise
        # (simulates VIO loop closure / feature-tracking update)
        pos_correction_noise = self._rng.normal(0, 0.03 * d, 3)
        vel_correction_noise = self._rng.normal(0, 0.05 * d, 3)

        # Blend toward ground truth (0.7 = strong correction, like real VIO)
        alpha = 0.7
        self.est_pos = alpha * (true_pos + pos_correction_noise) + (1 - alpha) * self.est_pos
        self.est_vel = alpha * (true_vel + vel_correction_noise) + (1 - alpha) * self.est_vel

    def get_estimates(self):
        """Return current VIO estimates: velocity(3), angular_rates(3), position(3)."""
        return (
            self.est_vel.astype(np.float32),
            self.est_angvel.astype(np.float32),
            self.est_pos.astype(np.float32),
        )


class EventCameraSensor:
    """Simulated event camera co-located with the FPV RGB camera.

    Based on the Event Generation Model (EGM) from event-sharp-nerf-drones
    (Zou, Cannici & Scaramuzza, IEEE T-RO 2026, UZH RPG).

    An event camera fires an asynchronous event at pixel (x,y) when the
    log-luminance change exceeds a contrast threshold C:
        log(L(t)) - log(L(t_ref)) >= +C  -> positive event (ON)
        log(L(t)) - log(L(t_ref)) <= -C  -> negative event (OFF)

    At high speed, RGB frames suffer motion blur, but events remain sharp
    because they respond to instantaneous brightness changes. This gives
    the agent usable high-frequency perception during aggressive maneuvers.

    Output: accumulated event frames with shape (H, W, 2*EVENT_FRAME_BINS).
    Each temporal bin has 2 channels: positive event count, negative event count.
    This "event histogram" representation is standard for feeding events into CNNs
    (Maqueda et al., CVPR 2018; Rebecq et al., TPAMI 2020).
    """

    def __init__(self, height, width, C_pos=None, C_neg=None, C_noise=None):
        self._h = height
        self._w = width
        self._C_pos = C_pos or EVENT_CONTRAST_THRESHOLD_POS
        self._C_neg = C_neg or EVENT_CONTRAST_THRESHOLD_NEG
        self._C_noise = C_noise or EVENT_CONTRAST_NOISE
        self._num_bins = EVENT_FRAME_BINS
        # Per-pixel reference log-luminance (last event firing threshold)
        self._ref_log_luma = None
        self._initialized = False

    def reset(self, frame_rgb):
        """Initialize reference log-luminance from first RGB frame."""
        luma = self._rgb_to_luma(frame_rgb)
        self._ref_log_luma = np.log(luma + 1e-6)
        self._initialized = True

    def _rgb_to_luma(self, rgb):
        """RGB -> luminance using Rec.601 weights (same as event-sharp-nerf-drones)."""
        return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    def process_frame(self, frame_rgb, rng=None):
        """Generate event frame from a new RGB observation.

        Computes per-pixel log-luminance difference from the stored reference,
        fires events where threshold is exceeded, and accumulates into a
        histogram representation.

        Args:
            frame_rgb: (H, W, 3) uint8 RGB image from MuJoCo renderer
            rng: numpy random generator for threshold noise

        Returns:
            event_frame: (H, W, 2*num_bins) uint8 event histogram
                Channels [2*b, 2*b+1] = [positive_count, negative_count] for bin b
        """
        if not self._initialized:
            self.reset(frame_rgb)
            return np.zeros((self._h, self._w, 2 * self._num_bins), dtype=np.uint8)

        luma = self._rgb_to_luma(frame_rgb.astype(np.float32))
        log_luma = np.log(luma + 1e-6)
        diff = log_luma - self._ref_log_luma

        # Add per-pixel threshold noise for realism
        if rng is not None and self._C_noise > 0:
            noise_pos = rng.normal(0, self._C_noise, (self._h, self._w))
            noise_neg = rng.normal(0, self._C_noise, (self._h, self._w))
        else:
            noise_pos = noise_neg = 0.0

        C_pos = self._C_pos + noise_pos
        C_neg = self._C_neg + noise_neg

        # Count how many events fire at each pixel (can be multiple per step)
        pos_counts = np.maximum(0, np.floor(diff / C_pos)).astype(np.int32)
        neg_counts = np.maximum(0, np.floor(-diff / C_neg)).astype(np.int32)

        # Update reference: advance by the number of events fired * threshold
        fired = (pos_counts > 0) | (neg_counts > 0)
        self._ref_log_luma[fired] += (
            pos_counts[fired] * self._C_pos - neg_counts[fired] * self._C_neg
        )

        # Build event histogram with temporal bins
        # For single-step processing, distribute events uniformly across bins
        event_frame = np.zeros((self._h, self._w, 2 * self._num_bins), dtype=np.uint8)
        if self._num_bins == 1:
            event_frame[:, :, 0] = np.clip(pos_counts, 0, 255).astype(np.uint8)
            event_frame[:, :, 1] = np.clip(neg_counts, 0, 255).astype(np.uint8)
        else:
            # Split counts across temporal bins (approximate — exact would need
            # sub-frame timestamps, but this is good enough for RL training)
            for b in range(self._num_bins):
                frac = 1.0 / self._num_bins
                event_frame[:, :, 2 * b] = np.clip(
                    np.round(pos_counts * frac), 0, 255
                ).astype(np.uint8)
                event_frame[:, :, 2 * b + 1] = np.clip(
                    np.round(neg_counts * frac), 0, 255
                ).astype(np.uint8)

        return event_frame


class DroneRaceEnv(gym.Env):
    """Autonomous drone racing — AI Grand Prix competition simulator.

    Action (4D — CTBR):
        [normalized_thrust, roll_rate, pitch_rate, yaw_rate]
        thrust in [0, 1], body rates in [-1, 1] (scaled to MAX_BODY_RATE)

    Observation (24D):
        position(3), velocity(3), euler_rpy(3), angular_vel_world(3),
        angular_vel_body(3), relative_gate_position(3), relative_gate_yaw(1),
        previous_action(4), gates_remaining_normalized(1)
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, domain_rand=None, difficulty=1.0, vision_mode=False, fpv_view=False):
        super().__init__()
        self.render_mode = render_mode
        self._domain_rand = DOMAIN_RAND if domain_rand is None else domain_rand
        self._difficulty = np.clip(difficulty, 0.0, 1.0)  # 0=easy, 1=full Swift
        self._vision_mode = vision_mode
        self._fpv_view = fpv_view

        # Event camera (event-sharp-nerf-drones, Zou et al. 2026)
        self._use_events = self._vision_mode and EVENT_CAMERA_ENABLED
        self._event_channels = 2 * EVENT_FRAME_BINS if self._use_events else 0

        if self._vision_mode:
            # RGB channels + event histogram channels per stacked frame
            img_channels = (3 + self._event_channels) * FPV_FRAME_STACK
            self.observation_space = gym.spaces.Dict({
                "image": gym.spaces.Box(
                    low=0, high=255,
                    shape=(FPV_RESOLUTION, FPV_RESOLUTION, img_channels),
                    dtype=np.uint8,
                ),
                "state": gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(VISION_STATE_DIM,), dtype=np.float32,
                ),
            })
        else:
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32,
            )
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._sim_steps_per_ctrl = max(1, SIM_FREQ // CTRL_FREQ)
        self._max_steps = EPISODE_LENGTH_SEC * CTRL_FREQ
        self._dt_phys = 1.0 / SIM_FREQ  # physics timestep

        # Gate data
        self._gate_positions = get_gate_positions()
        self._gate_yaws = get_gate_yaws()
        self._num_gates = len(self._gate_positions)

        # State
        self._model = None
        self._data = None
        self._current_gate_idx = 0
        self._step_count = 0
        self._prev_dist = None
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)

        # Domain randomization params
        self._mass = DRONE_MASS
        self._inertia = DRONE_INERTIA.copy()
        self._thrust_scale = 1.0
        self._drag_coeff = 0.0
        self._action_delay = 0
        self._action_buffer = []

        # Motor dynamics: first-order lag (Swift: real motors don't respond instantly)
        # Scale motor tau by difficulty: 0 = instant response, 1 = full lag
        effective_tau = MOTOR_TAU * self._difficulty
        if effective_tau > 0:
            self._motor_alpha = 1.0 - np.exp(-1.0 / (SIM_FREQ * effective_tau))
        else:
            self._motor_alpha = 1.0  # instant response
        self._current_thrust = 0.0  # filtered thrust state

        # Observation noise + delay (Swift: VIO + CNN gate detection pipeline)
        self._obs_delay = 0
        self._obs_buffer = []

        self._viewer = None

        # Vision mode state
        self._renderer = None  # lazy-init mujoco.Renderer
        self._fpv_cam_id = None
        self._frame_buffer = deque(maxlen=FPV_FRAME_STACK)

        # VIO simulator (used in vision_mode to provide noisy state estimates)
        self._vio = VIOSimulator(difficulty=self._difficulty) if self._vision_mode else None

        # Event camera sensor (event-sharp-nerf-drones)
        self._event_sensor = (
            EventCameraSensor(FPV_RESOLUTION, FPV_RESOLUTION)
            if self._use_events else None
        )

        # Motion blur (event-sharp-nerf-drones: average multiple sub-exposure renders)
        self._motion_blur_enabled = False  # enabled via staged warmup
        self._motion_blur_samples = MOTION_BLUR_SAMPLES
        # Store previous qpos for sub-exposure interpolation
        self._prev_qpos = None

        self._build_model(DRONE_MASS, DRONE_INERTIA)

    def _build_model(self, mass, inertia):
        xml = _build_mjcf(mass, inertia)
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)
        if self._vision_mode or self._fpv_view:
            self._fpv_cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv")
            self._gimbal_cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "gimbal")
            # Reset renderer when model changes (domain randomization rebuilds model)
            self._renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._domain_rand and self.np_random is not None:
            d = self._difficulty  # 0 = no randomization, 1 = full
            # Scale mass range by difficulty
            mass_lo = 1.0 + (MASS_RANGE[0] - 1.0) * d
            mass_hi = 1.0 + (MASS_RANGE[1] - 1.0) * d
            self._mass = DRONE_MASS * self.np_random.uniform(mass_lo, mass_hi)
            self._inertia = DRONE_INERTIA * (self._mass / DRONE_MASS)
            self._thrust_scale = 1.0 + self.np_random.uniform(-THRUST_NOISE * d, THRUST_NOISE * d)
            self._drag_coeff = self.np_random.uniform(0.0, DRAG_COEFF_RANGE[1] * d)
            max_act_delay = int(round(LATENCY_STEPS[1] * d))
            self._action_delay = self.np_random.integers(0, max_act_delay + 1) if max_act_delay > 0 else 0
            max_obs_delay = int(round(OBS_DELAY_STEPS[1] * d))
            self._obs_delay = self.np_random.integers(0, max_obs_delay + 1) if max_obs_delay > 0 else 0
            self._build_model(self._mass, self._inertia)
        else:
            self._mass = DRONE_MASS
            self._inertia = DRONE_INERTIA.copy()
            self._thrust_scale = 1.0
            self._drag_coeff = 0.0
            self._action_delay = 0
            self._obs_delay = 0

        mujoco.mj_resetData(self._model, self._data)

        self._data.qpos[0:3] = [0.0, -2.0, 2.5]
        self._data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self._data.qvel[:] = 0.0

        mujoco.mj_forward(self._model, self._data)

        self._current_gate_idx = 0
        self._step_count = 0
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._action_buffer = [np.zeros(ACT_DIM) for _ in range(self._action_delay + 1)]
        self._current_thrust = 0.0

        # Reset VIO with ground-truth initial state
        if self._vio is not None:
            self._vio._difficulty = self._difficulty
            self._vio._rng = self.np_random if self.np_random is not None else np.random.default_rng()
            self._vio.reset(
                self._data.qpos[0:3].copy(),
                self._data.qvel[0:3].copy(),
                self._data.qpos[3:7].copy(),
            )

        obs_state = self._get_obs()
        # Fill obs delay buffer with initial observation (N entries for N-step delay)
        self._obs_buffer = [obs_state.copy() for _ in range(self._obs_delay)]
        self._prev_dist = np.linalg.norm(
            self._gate_positions[self._current_gate_idx] - obs_state[:3]
        )

        if self._vision_mode:
            # Initialize frame buffer with combined RGB+event frames
            first_frame = self._render_fpv()
            self._frame_buffer.clear()
            # Reset event sensor with first frame
            if self._event_sensor is not None:
                self._event_sensor.reset(first_frame)
                # First event frame is zeros (no change from reference yet)
                zero_events = np.zeros(
                    (FPV_RESOLUTION, FPV_RESOLUTION, self._event_channels),
                    dtype=np.uint8,
                )
                combined = np.concatenate([first_frame, zero_events], axis=2)
            else:
                combined = first_frame
            for _ in range(FPV_FRAME_STACK):
                self._frame_buffer.append(combined.copy())
            # Build obs from pre-filled buffer (avoids double _get_vision_obs)
            stacked = np.concatenate(list(self._frame_buffer), axis=2)
            state = self._get_minimal_state()
            return {"image": stacked, "state": state}, {}
        return obs_state, {}

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        action[0] = np.clip(action[0], 0.0, 1.0)
        action[1:] = np.clip(action[1:], -1.0, 1.0)

        # Action delay
        self._action_buffer.append(action.copy())
        delayed_action = self._action_buffer.pop(0)

        # Decode CTBR
        thrust_normalized = delayed_action[0]
        target_body_rates = delayed_action[1:] * MAX_BODY_RATE

        collective_thrust = thrust_normalized * MAX_COLLECTIVE_THRUST * self._thrust_scale

        # Run physics with rate controller at EACH substep
        for _ in range(self._sim_steps_per_ctrl):
            # Motor dynamics: first-order lag on thrust (Swift realism)
            self._current_thrust += self._motor_alpha * (collective_thrust - self._current_thrust)
            self._apply_forces(self._current_thrust, target_body_rates)
            mujoco.mj_step(self._model, self._data)

            # VIO IMU update each physics step
            if self._vio is not None:
                true_vel = self._data.qvel[0:3]
                true_angvel = self._data.qvel[3:6]
                # True acceleration: (F_applied / mass) — approximate from velocity change
                # Use gravity-compensated accel from xfrc_applied
                true_accel = self._data.xfrc_applied[1, :3] / self._mass - np.array([0, 0, GRAVITY])
                self._vio.imu_update(true_angvel, true_accel, self._dt_phys)

        # VIO visual update at control rate (corrects drift periodically)
        if self._vio is not None:
            self._vio.visual_update(
                self._data.qpos[0:3].copy(),
                self._data.qvel[0:3].copy(),
            )

        self._step_count += 1
        self._prev_action = action.copy()
        self._prev_qpos = self._data.qpos[:7].copy()  # for motion blur interpolation

        if self.render_mode == "human":
            self._render_frame()

        obs_clean = self._get_obs()

        # Observation noise (Swift: VIO + CNN gate detection are noisy)
        # Scaled by difficulty: 0 = no noise, 1 = full noise
        if self._domain_rand and self._difficulty > 0:
            d = self._difficulty
            noise = np.zeros_like(obs_clean)
            noise[0:3] = self.np_random.normal(0, OBS_NOISE_POS * d, 3)      # position
            noise[3:6] = self.np_random.normal(0, OBS_NOISE_VEL * d, 3)      # velocity
            noise[6:9] = self.np_random.normal(0, OBS_NOISE_RPY * d, 3)      # attitude
            noise[9:12] = self.np_random.normal(0, OBS_NOISE_ANGVEL * d, 3)  # ang vel world
            noise[12:15] = self.np_random.normal(0, OBS_NOISE_ANGVEL * d, 3) # ang vel body
            noise[15:18] = self.np_random.normal(0, OBS_NOISE_GATE_POS * d, 3)  # gate pos
            noise[18] = self.np_random.normal(0, OBS_NOISE_GATE_YAW * d)     # gate yaw
            obs_noisy = obs_clean + noise
        else:
            obs_noisy = obs_clean

        # Observation delay (Swift: camera + processing pipeline latency)
        if self._obs_delay > 0:
            self._obs_buffer.append(obs_noisy)
            obs = self._obs_buffer.pop(0)
        else:
            obs = obs_noisy

        # Use clean pos/rpy for reward computation (reward is ground truth)
        pos = obs_clean[:3]
        rpy = obs_clean[6:9]

        # === Rewards ===
        reward = 0.0
        terminated = False
        truncated = False

        dist_to_gate = np.linalg.norm(
            self._gate_positions[self._current_gate_idx] - pos
        )
        progress = self._prev_dist - dist_to_gate
        reward += REWARD_PROGRESS * progress
        self._prev_dist = dist_to_gate

        if dist_to_gate < GATE_TOLERANCE:
            # Progressive reward: later gates worth more (1.0x, 1.1x, ..., 1.7x)
            gate_multiplier = 1.0 + 0.1 * self._current_gate_idx
            # Alignment bonus: reward flying through gate straight (not from side)
            gate_yaw = self._gate_yaws[self._current_gate_idx]
            gate_forward = np.array([np.cos(gate_yaw), np.sin(gate_yaw), 0.0])
            drone_vel = self._data.qvel[0:3]
            speed = np.linalg.norm(drone_vel)
            if speed > 0.5:
                vel_dir = drone_vel / speed
                alignment = abs(np.dot(vel_dir[:2], gate_forward[:2]))  # 1.0 = perfect
                gate_multiplier *= (0.5 + 0.5 * alignment)  # 50-100% reward based on alignment
            reward += REWARD_GATE_PASSED * gate_multiplier
            self._current_gate_idx += 1
            if self._current_gate_idx >= self._num_gates:
                terminated = True
            else:
                self._prev_dist = np.linalg.norm(
                    self._gate_positions[self._current_gate_idx] - pos
                )

        # Perception penalty
        if not terminated:
            rel_gate = self._gate_positions[min(self._current_gate_idx, self._num_gates-1)] - pos
            gate_bearing = np.arctan2(rel_gate[1], rel_gate[0])
            bearing_error = abs((gate_bearing - rpy[2] + np.pi) % (2*np.pi) - np.pi)
            if bearing_error > np.pi/2:
                reward += REWARD_PERCEPTION

        # Command smoothness
        reward += REWARD_CMD_SMOOTHNESS * np.sum(np.abs(action[1:]))

        # Body rate penalty
        ang_vel = self._data.qvel[3:6]
        reward += REWARD_BODY_RATE_PENALTY * np.sum(ang_vel**2)

        # Time penalty (incentivize speed, no loitering)
        reward += REWARD_TIME_PENALTY

        # Crashes
        if pos[2] < 0.1:
            reward += REWARD_CRASH_PENALTY
            terminated = True
        elif np.linalg.norm(pos[:2]) > TRACK_BOUNDS:
            reward += REWARD_CRASH_PENALTY
            terminated = True
        elif pos[2] > 20.0:
            reward += REWARD_CRASH_PENALTY
            terminated = True
        elif abs(rpy[0]) > 2*np.pi/3 or abs(rpy[1]) > 2*np.pi/3:
            # 120° limit — racing drones bank aggressively through turns
            reward += REWARD_CRASH_PENALTY
            terminated = True

        if self._step_count >= self._max_steps:
            truncated = True

        info = {
            "gates_passed": self._current_gate_idx,
            "total_gates": self._num_gates,
            "speed": float(np.linalg.norm(self._data.qvel[0:3])),
        }

        if self._vision_mode:
            return self._get_vision_obs(), reward, terminated, truncated, info
        return obs, reward, terminated, truncated, info

    def _apply_forces(self, collective_thrust, target_body_rates):
        """Apply thrust + torques directly to the drone body each physics step.

        This simulates a Betaflight-style rate controller running at the
        flight controller frequency (same as physics rate).
        """
        quat_wxyz = self._data.qpos[3:7]
        ang_vel_world = self._data.qvel[3:6]

        # Rotation matrix (body-to-world)
        R = _quat_to_rotmat(*quat_wxyz)

        # --- Thrust: along body Z axis, in world frame ---
        thrust_world = R @ np.array([0, 0, collective_thrust])

        # --- Rate controller: compute body-frame torques ---
        # Convert world angular velocity to body frame
        ang_vel_body = R.T @ ang_vel_world

        # PD on body rates
        rate_error = target_body_rates - ang_vel_body
        # Torque = I * (Kp * error - Kd * omega_body)
        # Using direct inertia-scaled control for stability
        torque_body = self._inertia * (RATE_KP * rate_error - RATE_KD * ang_vel_body)

        # Convert torque to world frame
        torque_world = R @ torque_body

        # --- Drag ---
        drag_force = np.zeros(3)
        if self._drag_coeff > 0:
            vel = self._data.qvel[0:3]
            drag_force = -self._drag_coeff * vel * np.abs(vel)

        # Apply forces and torques to the drone body (body index 1, free joint)
        # xfrc_applied shape: (nbody, 6) — [fx, fy, fz, tx, ty, tz] in world frame
        self._data.xfrc_applied[1, :3] = thrust_world + drag_force
        self._data.xfrc_applied[1, 3:] = torque_world

    def _get_obs(self):
        """Build 24D observation vector."""
        pos = self._data.qpos[0:3].astype(np.float32)
        quat_wxyz = self._data.qpos[3:7]
        vel = self._data.qvel[0:3].astype(np.float32)
        ang_vel_world = self._data.qvel[3:6].astype(np.float32)

        roll, pitch, yaw = _quat_to_rpy(*quat_wxyz)
        rpy = np.array([roll, pitch, yaw], dtype=np.float32)

        R = _quat_to_rotmat(*quat_wxyz)
        ang_vel_body = (R.T @ ang_vel_world).astype(np.float32)

        gate_idx = min(self._current_gate_idx, self._num_gates - 1)
        rel_gate = (self._gate_positions[gate_idx] - pos).astype(np.float32)

        gate_yaw = self._gate_yaws[gate_idx]
        rel_yaw = np.float32((gate_yaw - yaw + np.pi) % (2 * np.pi) - np.pi)

        gates_remaining = np.float32(
            (self._num_gates - self._current_gate_idx) / self._num_gates
        )

        return np.concatenate([
            pos, vel, rpy, ang_vel_world, ang_vel_body,
            rel_gate, [rel_yaw], self._prev_action, [gates_remaining],
        ])

    def set_motion_blur(self, enabled):
        """Enable/disable motion blur rendering (staged warmup)."""
        self._motion_blur_enabled = enabled

    def set_difficulty(self, difficulty):
        """Update curriculum difficulty (0=easy, 1=full Swift)."""
        self._difficulty = np.clip(difficulty, 0.0, 1.0)
        # Update motor dynamics for new difficulty
        effective_tau = MOTOR_TAU * self._difficulty
        if effective_tau > 0:
            self._motor_alpha = 1.0 - np.exp(-1.0 / (SIM_FREQ * effective_tau))
        else:
            self._motor_alpha = 1.0
        # Update VIO noise scaling
        if self._vio is not None:
            self._vio._difficulty = self._difficulty

    def _render_fpv(self):
        """Render FPV image, optionally with motion blur.

        Motion blur (from event-sharp-nerf-drones, Zou et al. 2026):
        Averages N sub-exposure renders by interpolating drone pose between
        previous and current timestep. Simulates real camera integration
        during high-speed flight where shutter speed causes blur.
        """
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, FPV_RESOLUTION, FPV_RESOLUTION)

        if (not self._motion_blur_enabled
                or self._motion_blur_samples <= 1
                or self._prev_qpos is None):
            # No blur: single render
            self._renderer.update_scene(self._data, camera=self._fpv_cam_id)
            return self._renderer.render()

        # Motion blur: average N sub-exposure renders
        current_qpos = self._data.qpos[:7].copy()
        accum = np.zeros((FPV_RESOLUTION, FPV_RESOLUTION, 3), dtype=np.float32)

        for i in range(self._motion_blur_samples):
            alpha = i / (self._motion_blur_samples - 1)  # 0.0 to 1.0
            # Interpolate position linearly
            interp_pos = (1 - alpha) * self._prev_qpos[:3] + alpha * current_qpos[:3]
            # Interpolate quaternion via SLERP
            interp_quat = self._slerp(self._prev_qpos[3:7], current_qpos[3:7], alpha)
            # Temporarily set drone pose
            self._data.qpos[:3] = interp_pos
            self._data.qpos[3:7] = interp_quat
            mujoco.mj_forward(self._model, self._data)
            self._renderer.update_scene(self._data, camera=self._fpv_cam_id)
            accum += self._renderer.render().astype(np.float32)

        # Restore current pose
        self._data.qpos[:7] = current_qpos
        mujoco.mj_forward(self._model, self._data)

        return (accum / self._motion_blur_samples).astype(np.uint8)

    @staticmethod
    def _slerp(q0, q1, t):
        """Spherical linear interpolation between quaternions (w,x,y,z).

        From event-sharp-nerf-drones utils/interpolate.py.
        """
        q0 = q0 / np.linalg.norm(q0)
        q1 = q1 / np.linalg.norm(q1)
        dot = np.dot(q0, q1)
        if dot < 0:
            q1 = -q1
            dot = -dot
        dot = np.clip(dot, -1.0, 1.0)
        if dot > 0.9995:
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        return (np.sin((1 - t) * theta) / sin_theta) * q0 + (np.sin(t * theta) / sin_theta) * q1

    def _get_minimal_state(self):
        """Build 19D minimal state vector for hybrid vision policy.

        Uses 6D rotation representation (event-sharp-nerf-drones, Zhou CVPR 2019)
        which is continuous and avoids gimbal lock, superior for learning.

        Layout: rot_6d(6) + vio_velocity(3) + vio_angular_rates(3) +
                prev_action(4) + vio_position(3) = 19D
        """
        quat_wxyz = self._data.qpos[3:7]
        R = _quat_to_rotmat(*quat_wxyz)
        rot_6d = _rotmat_to_6d(R)

        if self._vio is not None:
            vio_vel, vio_angvel, vio_pos = self._vio.get_estimates()
            return np.concatenate([rot_6d, vio_vel, vio_angvel, self._prev_action, vio_pos])
        else:
            vel = self._data.qvel[0:3].astype(np.float32)
            ang_vel_body = (R.T @ self._data.qvel[3:6]).astype(np.float32)
            pos = self._data.qpos[0:3].astype(np.float32)
            return np.concatenate([rot_6d, vel, ang_vel_body, self._prev_action, pos])

    def _get_vision_obs(self):
        """Build Dict observation for vision mode: stacked FPV+event images + minimal state.

        With event camera enabled, each frame is (H, W, 3 + 2*EVENT_FRAME_BINS):
        RGB channels followed by event histogram channels (pos/neg per temporal bin).
        Stacked across FPV_FRAME_STACK frames along channel dim.
        """
        frame = self._render_fpv()

        if self._event_sensor is not None:
            rng = self.np_random if self.np_random is not None else None
            event_frame = self._event_sensor.process_frame(frame, rng=rng)
            # Concatenate RGB + event channels: (H, W, 3+event_channels)
            combined = np.concatenate([frame, event_frame], axis=2)
        else:
            combined = frame

        self._frame_buffer.append(combined)
        stacked = np.concatenate(list(self._frame_buffer), axis=2)
        state = self._get_minimal_state()
        return {"image": stacked, "state": state}

    def _render_frame(self):
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
            if self._fpv_view:
                # Onboard FPV camera — continuous, no cuts
                self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self._viewer.cam.fixedcamid = self._fpv_cam_id
            else:
                # Overhead track view
                self._viewer.cam.distance = 25.0
                self._viewer.cam.azimuth = 60
                self._viewer.cam.elevation = -25
                self._viewer.cam.lookat[:] = [0, 7, 2.5]
        self._viewer.sync()

    def _update_gimbal(self):
        """Position viewer camera on drone, looking at next gate."""
        drone_pos = self._data.qpos[0:3].copy()
        gate_idx = min(self._current_gate_idx, self._num_gates - 1)
        gate_pos = self._gate_positions[gate_idx]

        # Direction from drone to gate
        direction = gate_pos - drone_pos
        dist = np.linalg.norm(direction[:2])  # horizontal distance

        # Set camera at drone position, looking at gate
        # lookat = midpoint between drone and gate (viewer orbits around lookat)
        self._viewer.cam.lookat[:] = gate_pos
        self._viewer.cam.distance = max(dist, 1.0)
        self._viewer.cam.azimuth = np.degrees(np.arctan2(direction[1], direction[0])) - 90
        elev = np.degrees(np.arctan2(direction[2], dist)) if dist > 0.1 else 0
        self._viewer.cam.elevation = elev - 5

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
