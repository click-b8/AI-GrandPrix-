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
EPISODE_LENGTH_SEC = 45   # Max episode duration (Anduril-6 VQ1 course is ~160m long)

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
REWARD_CRASH_PENALTY = -200.0   # heavy crash penalty (prioritize safety)
REWARD_BODY_RATE_PENALTY = -0.01  # penalty for high body rates (encourages smooth flight)
REWARD_TIME_PENALTY = -0.1        # per-step penalty to incentivize speed (no loitering)

# --- Track ---
GATE_WIDTH = 1.5          # meters (MultiGP standard)
GATE_HEIGHT = 1.5         # meters
GATE_TOLERANCE = 1.0      # meters — pass radius around gate center
TRACK_BOUNDS = 160.0      # meters — arena boundary (sized to the Anduril-6 VQ1 course extent)
SPAWN_PITCH_DEG = -17.8   # measured nose-down rest attitude at spawn (powered de-risk 2026-07-11);
                          # confirmed hypothesis A (body attitude, not IMU mount). Baked into env spawn.

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

# --- FPV Camera (vision-based racing) ---
FPV_RESOLUTION = 48           # 48x48 pixels (smaller = faster rendering + CNN)
FPV_FOV = 58.72               # VERTICAL FoV (deg), from Elodin practice-rig intrinsics (NOT the stated 90). With a 16:9 render buffer MuJoCo derives HFoV≈90° from this fovy.
FPV_ASPECT = 16.0 / 9.0       # camera aspect: HFoV=90 with VFoV=58.72; deploy squashes 640x360 RGB -> 48x48
FPV_RENDER_W = 96             # 16:9 offscreen render width (96/54 = 16/9); resized down to FPV_RESOLUTION square
FPV_RENDER_H = 54             # 16:9 offscreen render height
FPV_TILT_DEG = 20             # VADR-TS-002 §3.8: camera tilted upwards 20°
FPV_FRAME_STACK = 3           # stacked frames -> image (9,48,48); N=3 adds an acceleration cue (not just velocity) for gate timing, and matches the contract-test oracle
VISION_STATE_DIM = 10         # gravity_unit(3, FLU) + body_rates(3, FLU) + prev_action(4)  — vision+IMU contract (observation-spec.md)

# --- Motion blur (from event-sharp-nerf-drones, Zou et al. 2026) ---
# At high speed, FPV frames are motion-blurred. Simulating this forces
# the CNN to be robust, matching real deployment conditions.
MOTION_BLUR_SAMPLES = 3       # sub-exposure renders averaged (1=no blur, 3=mild, 5=heavy)
MOTION_BLUR_WARMUP = 50_000   # timesteps before enabling blur (staged training)

# --- Event camera (from event-sharp-nerf-drones, Zou et al. 2026) ---
# Simulates a co-located event camera that fires per-pixel events when
# log-luminance changes exceed a contrast threshold C.
# At high speed, RGB frames blur but events remain sharp — giving the
# agent usable perception when the standard camera fails.
EVENT_CAMERA_ENABLED = False  # DCL provides RGB only
EVENT_CONTRAST_THRESHOLD_POS = 0.2    # C+ positive contrast threshold (log-luminance)
EVENT_CONTRAST_THRESHOLD_NEG = 0.2    # C- negative contrast threshold
EVENT_CONTRAST_NOISE = 0.03           # std of per-event threshold noise (realism)
EVENT_REFRACTORY_PERIOD = 1e-4        # seconds, minimum time between events at same pixel
EVENT_FRAME_BINS = 2                  # temporal bins for event representation (pos/neg per bin)
EVENT_WARMUP = 0                      # timesteps before enabling events (0=always on)

# --- VIO/IMU noise (simulates Visual-Inertial Odometry estimation errors) ---
# Based on typical MEMS IMU + stereo/mono VIO pipeline (Swift, Nature 2023)
VIO_GYRO_BIAS_INSTABILITY = 0.003   # rad/s, gyro bias random walk
VIO_GYRO_WHITE_NOISE = 0.01        # rad/s, gyro measurement noise
VIO_ACCEL_BIAS = 0.02              # m/s², accelerometer bias
VIO_ACCEL_NOISE = 0.05             # m/s², accelerometer measurement noise
VIO_VEL_DRIFT_RATE = 0.02          # m/s per second, velocity estimate drift
VIO_POS_DRIFT_RATE = 0.05          # m per second, position estimate drift
VIO_UPDATE_RATE = 30               # Hz, visual correction rate (camera-rate)

# --- Domain randomization (for sim-to-real transfer) ---
DOMAIN_RAND = True
MASS_RANGE = (0.9, 1.05)         # ±5-10% mass variation
THRUST_NOISE = 0.05               # ±5% thrust noise
LATENCY_STEPS = (0, 2)            # 0-2 step action delay
DRAG_COEFF_RANGE = (0.0, 0.3)    # aerodynamic drag coefficient
