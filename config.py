"""Configuration for the drone race simulation.

Aligned with Anduril AI Grand Prix / A2RL x DCL competition specs.
Based on MonoRace (A2RL 2025 winner) and Swift (Nature 2023).

Drone: ~966g, 5.1" props, monocular camera + IMU
Action: Collective thrust + body rates (CTBR) — 4D
Observation: Drone state + gate relative pose — 24D
Training: PPO via Stable-Baselines3
"""

import numpy as np

# --- Simulation ---
SIM_FREQ = 200            # Physics steps per second
CTRL_FREQ = 100           # Agent control frequency (MonoRace: 500Hz, Swift: 125Hz)
EPISODE_LENGTH_SEC = 30   # Max episode duration

# --- Drone physical parameters (A2RL x DCL 2025 spec) ---
DRONE_MASS = 0.966        # kg (MonoRace competition drone)
DRONE_ARM_LENGTH = 0.13   # m (5.1" prop quad, ~260mm diagonal)
GRAVITY = 9.81

# Inertia for a 5" racing quad (approximated)
DRONE_INERTIA = np.array([0.003, 0.003, 0.005])  # Ixx, Iyy, Izz

# Thrust
THRUST_TO_WEIGHT = 4.5    # realistic for a ~1kg racing quad
MAX_COLLECTIVE_THRUST = THRUST_TO_WEIGHT * DRONE_MASS * GRAVITY  # Newtons total
HOVER_THRUST_NORMALIZED = 1.0 / THRUST_TO_WEIGHT  # ~0.22 of max thrust to hover
MAX_SPEED = 30.0          # m/s (~108 km/h, competition-realistic)

# --- Action space: CTBR (Collective Thrust + Body Rates) ---
# [normalized_thrust, roll_rate, pitch_rate, yaw_rate]
# This is the standard used by Swift (Nature 2023) and most racing systems.
# The low-level flight controller (Betaflight) converts these to motor PWMs.
ACT_DIM = 4
MAX_BODY_RATE = 12.0      # rad/s max body rate command (~690 deg/s)

# --- Observation space (24D, matches MonoRace) ---
# Drone state in gate frame:
#   position(3), velocity(3), euler_angles(3), angular_vel_world(3),
#   angular_vel_body(3), relative_gate_pos(3), relative_gate_yaw(1),
#   previous_action(4), gates_remaining(1)
# = 24 dims
OBS_DIM = 24

# --- Low-level rate controller (simulates Betaflight PID) ---
RATE_KP = 60.0     # P gain for body rate control (high for snappy response)
RATE_KD = 2.0      # D gain for body rate control (damping)

# --- Reward (matches Swift/MonoRace reward structure) ---
REWARD_GATE_PASSED = 100.0      # large gate bonus
REWARD_PROGRESS = 10.0          # distance progress toward gate
REWARD_PERCEPTION = -0.5        # penalty when drone can't "see" gate (too far off axis)
REWARD_CMD_SMOOTHNESS = -0.02   # penalty for jerky commands
REWARD_CRASH_PENALTY = -50.0    # collision/out-of-bounds
REWARD_BODY_RATE_PENALTY = -0.01  # penalty for high body rates (encourages smooth flight)

# --- Track ---
GATE_WIDTH = 1.5          # meters (MultiGP standard)
GATE_HEIGHT = 1.5         # meters
GATE_TOLERANCE = 1.0      # meters — pass radius around gate center
TRACK_BOUNDS = 40.0       # meters — arena boundary

# --- Motor dynamics (Swift: real motors have response lag) ---
MOTOR_TAU = 0.02                  # motor time constant in seconds (~50Hz bandwidth)

# --- Observation noise (simulates VIO + gate detection pipeline) ---
# Swift uses VIO (visual-inertial odometry) + CNN gate detector + Kalman filter.
# These are noisy — we inject noise to force robustness.
OBS_NOISE_POS = 0.05              # m, position estimate noise (VIO drift)
OBS_NOISE_VEL = 0.1               # m/s, velocity estimate noise
OBS_NOISE_RPY = 0.02              # rad, attitude estimate noise (~1.1 deg)
OBS_NOISE_ANGVEL = 0.05           # rad/s, gyro noise
OBS_NOISE_GATE_POS = 0.15         # m, gate position detection noise (CNN + projection)
OBS_NOISE_GATE_YAW = 0.05         # rad, gate yaw detection noise (~3 deg)
OBS_DELAY_STEPS = (0, 2)          # observation pipeline latency (camera + processing)

# --- Domain randomization (for sim-to-real transfer) ---
DOMAIN_RAND = True
MASS_RANGE = (0.9, 1.05)         # ±5-10% mass variation
THRUST_NOISE = 0.05               # ±5% thrust noise
LATENCY_STEPS = (0, 2)            # 0-2 step action delay
DRAG_COEFF_RANGE = (0.0, 0.3)    # aerodynamic drag coefficient
