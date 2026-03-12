# AI-GrandPrix-
The AI Grand Prix, founded by Anduril in partnership with the Drone Champions League (DCL), Neros Technologies, and JobsOhio, is a global autonomous drone racing competition challenging the world's best engineers to prove their autonomy software under real-world flight conditions.
  AI Grand Prix — Autonomous Drone Racing Simulator

  Autonomous drone racing simulation for the https://www.aigrandprix.com/ competition, founded by Anduril in partnership with
  the Drone Champions League (DCL), Neros Technologies, and JobsOhio.

  Architecture

  A vision-based autonomous racing system trained end-to-end with reinforcement learning (PPO), combining RGB FPV and
  simulated event camera inputs.

  Perception Pipeline

  - FPV Camera: 48×48 RGB forward-facing camera with configurable tilt and FOV
  - Event Camera: Simulated asynchronous brightness-change sensor based on the Event Generation Model (EGM) from
  https://github.com/uzh-rpg/event-sharp-nerf-drones (Zou, Cannici & Scaramuzza, IEEE T-RO 2026, UZH RPG). Events remain sharp
   during aggressive flight when RGB frames blur
  - Frame Stacking: Temporal context via stacked RGB+event frames (14 input channels)
  - Motion Blur Simulation: Sub-exposure averaging with SLERP pose interpolation and staged warmup

  Control

  - CTBR Action Space: Collective thrust + body rates — standard for racing drones (Swift, Nature 2023)
  - Betaflight-Style Rate Controller: PD body-rate tracking at physics rate
  - Motor Dynamics: First-order lag modeling real ESC/motor response

  State Estimation

  - Visual-Inertial Odometry (VIO): Simulated IMU integration with gyro/accelerometer bias, white noise, and drift. Periodic
  visual corrections at 30Hz
  - 6D Rotation Representation: Continuous rotation via first two columns of rotation matrix (Zhou et al., CVPR 2019),
  avoiding gimbal lock and quaternion discontinuities
  - SE(3)-Aware State Branch: 19D minimal state vector (6D rotation + VIO velocity/angular rates/position + previous action)

  Neural Network

  - Coarse-to-Fine CNN: Inspired by VoxelNeRF decomposition pattern
    - Coarse stage: 2 conv layers → 64D features (fast gate detection)
    - Fine stage: 2 conv layers on coarse feature maps → 128D features (precise localization)
  - State MLP: 2 FC layers → 64D features from VIO estimates
  - Combined: 256D feature vector → PPO policy/value heads

  Training

  - PPO via Stable-Baselines3 with curriculum learning
  - Curriculum: Medium difficulty (0–30%) → ramp (30–80%) → full (80–100%)
  - Domain Randomization: Mass (±10%), thrust noise (±5%), aerodynamic drag, action latency (0–2 steps), observation noise and
   delay
  - 200M timesteps target with checkpointing every 50K steps

  Environment

  MuJoCo-based physics simulation with:
  - 8-gate race track
  - 200Hz physics / 100Hz control
  - Configurable difficulty scaling (0 = easy, 1 = full Swift-level realism)
  - Reward: gate passage bonuses + progress shaping + crash penalties

  Quick Start

  # Create venv and install dependencies
  python3 -m venv venv && source venv/bin/activate
  pip install -r requirements.txt

  # Train vision policy with event camera
  python3 train_vision.py

  # View trained model in MuJoCo 3D viewer
  mjpython race.py --model trained_vision_events/best_model/best_model.zip

  # View FPV + event camera output
  python3 view_vision.py

  Project Structure

  ├── drone_race_env.py      # Gymnasium environment (physics, sensors, rewards)
  ├── config.py              # All hyperparameters and physical constants
  ├── track.py               # Gate positions and track layout
  ├── train_vision.py        # Vision+event training script (PPO)
  ├── train.py               # State-based training (baseline)
  ├── race.py                # 3D MuJoCo viewer for trained models
  ├── view_vision.py         # FPV + event camera visualization
  ├── train_hpc.slurm        # SLURM job script for HPC training
  └── trained_vision_events/ # Checkpoints, eval logs, TensorBoard

  References

  - Swift — Champion-level drone racing (Nature 2023). CTBR action space, domain randomization, sim-to-real transfer
  - event-sharp-nerf-drones — Event-Aided Sharp Radiance Field Reconstruction for Fast-Flying Drones (Zou et al., IEEE T-RO
  2026, UZH RPG). Event camera model, coarse-to-fine architecture, motion blur modeling
  - MonoRace — A2RL x DCL 2025 winner. Competition-spec drone parameters
  - Zhou et al. — On the Continuity of Rotation Representations in Neural Networks (CVPR 2019). 6D rotation representation

  License

  MIT

  ---
