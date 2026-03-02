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

from config import (
    SIM_FREQ, CTRL_FREQ, EPISODE_LENGTH_SEC,
    DRONE_MASS, DRONE_ARM_LENGTH, DRONE_INERTIA,
    MAX_COLLECTIVE_THRUST, HOVER_THRUST_NORMALIZED, GRAVITY,
    MAX_BODY_RATE, MAX_SPEED,
    OBS_DIM, ACT_DIM,
    RATE_KP, RATE_KD,
    REWARD_GATE_PASSED, REWARD_PROGRESS, REWARD_PERCEPTION,
    REWARD_CMD_SMOOTHNESS, REWARD_CRASH_PENALTY, REWARD_BODY_RATE_PENALTY,
    GATE_TOLERANCE, TRACK_BOUNDS,
    DOMAIN_RAND, MASS_RANGE, THRUST_NOISE, LATENCY_STEPS, DRAG_COEFF_RANGE,
    MOTOR_TAU,
    OBS_NOISE_POS, OBS_NOISE_VEL, OBS_NOISE_RPY, OBS_NOISE_ANGVEL,
    OBS_NOISE_GATE_POS, OBS_NOISE_GATE_YAW, OBS_DELAY_STEPS,
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


def _build_mjcf(mass, inertia):
    """Build MuJoCo XML. No actuators — we apply forces directly."""
    arm = DRONE_ARM_LENGTH
    gate_bodies = build_gate_xml()

    xml = f"""
    <mujoco model="aigp_racer">
      <option timestep="{1.0 / SIM_FREQ}" gravity="0 0 -{GRAVITY}"/>

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
        </body>

        {gate_bodies}
      </worldbody>
    </mujoco>
    """
    return xml


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

    def __init__(self, render_mode=None, domain_rand=None, difficulty=1.0):
        super().__init__()
        self.render_mode = render_mode
        self._domain_rand = DOMAIN_RAND if domain_rand is None else domain_rand
        self._difficulty = np.clip(difficulty, 0.0, 1.0)  # 0=easy, 1=full Swift

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
        self._build_model(DRONE_MASS, DRONE_INERTIA)

    def _build_model(self, mass, inertia):
        xml = _build_mjcf(mass, inertia)
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)

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

        obs = self._get_obs()
        # Fill obs delay buffer with initial observation (N entries for N-step delay)
        self._obs_buffer = [obs.copy() for _ in range(self._obs_delay)]
        self._prev_dist = np.linalg.norm(
            self._gate_positions[self._current_gate_idx] - obs[:3]
        )
        return obs, {}

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

        self._step_count += 1
        self._prev_action = action.copy()

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

    def set_difficulty(self, difficulty):
        """Update curriculum difficulty (0=easy, 1=full Swift)."""
        self._difficulty = np.clip(difficulty, 0.0, 1.0)
        # Update motor dynamics for new difficulty
        effective_tau = MOTOR_TAU * self._difficulty
        if effective_tau > 0:
            self._motor_alpha = 1.0 - np.exp(-1.0 / (SIM_FREQ * effective_tau))
        else:
            self._motor_alpha = 1.0

    def _render_frame(self):
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
            self._viewer.cam.distance = 25.0
            self._viewer.cam.azimuth = 60
            self._viewer.cam.elevation = -25
            self._viewer.cam.lookat[:] = [0, 7, 2.5]
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
