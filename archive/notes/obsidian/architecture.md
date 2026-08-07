#architecture

# System architecture

The project has two regimes: training and deployment. They share the
perception / control code but diverge at the edges.

## Training loop (works, validated, produced the model we have)

```
┌────────────────────────────────────────────────────────────────┐
│                 DroneRaceEnv  (gymnasium.Env)                  │
│                                                                │
│  MuJoCo physics @ 200 Hz ── motor lag ── thrust+rate control   │
│       │                                                        │
│       ├─► VIOSimulator ── noisy state estimates (19D)          │
│       │                                                        │
│       ├─► mujoco.Renderer ── RGB 48×48 ── motion blur (SLERP)  │
│       │                                                        │
│       └─► EventCameraSensor ── log-luma Δ ── 2 pos/neg bins    │
│                                                                │
│  Stacked obs = {image: (14,48,48) uint8, state: (19,) float32} │
└──────────────────────────────┬─────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────┐
│              DroneVisionExtractor (features_dim=256)           │
│   image ─► coarse CNN (64D)                                    │
│   image ─► fine CNN on coarse maps (128D)                      │
│   state ─► state_mlp (64D)                                     │
│   concat(coarse, fine, state) = 256D                           │
└──────────────────────────────┬─────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────┐
│  SB3 MlpExtractor (256 ► 128 ► 64)  ─► action_net (64 ► 4)     │
│                                     └► value_net (64 ► 1)      │
│  Gaussian PPO head, action = CTBR [thrust, roll_rate,          │
│    pitch_rate, yaw_rate]                                       │
└────────────────────────────────────────────────────────────────┘
```

This is the actual structure of `drone-race-sim/trained_distilled/policy.pth`.
See [[vision-model]] for exact shapes and why.

## Deployment loop (intended; does not currently work end-to-end)

```
 DCL simulator (UDP, MAVLink v2, 120 Hz physics)
        │
        │  SET_ATTITUDE_TARGET ◄──── dcl_mavlink_client.py
        │                              │
        │  vision + telemetry ──────►  │  (asyncio loop @ 50 Hz)
        ▼                              ▼
                             SCUBALabAdapter (dcl_adapter.py)
                                 │
                                 ├─► DroneVisionExtractor (256D features)   ← weights load OK
                                 │
                                 └─► PolicyNet (Linear 275 ► 128 ► 64 ► 4)  ← weights DO NOT load
                                        │                                     (wrong prefix,
                                        ▼                                      wrong shape;
                                   [throttle, roll, pitch, yaw]                output is random)
                                        │
                                        ▼  dcl_mavlink_adapter.encode_set_attitude_target()
                                        │    (CRC zeroed, payload malformed,
                                        │     body rates written into attitude fields)
```

The three failure points (silent weight-load failure, malformed
MAVLink, semantic unit mismatch) each look correct locally. Together
they compose into a system that would appear to run and fly unusably.
See [[fragilities#Silent failure chain]].

## Where each concern lives

| Concern | Primary file | Notes |
|---|---|---|
| Physics, rewards, episode flow | `drone_race_env.py` | [[simulation]] |
| Hyperparameters, constants | `config.py` | [[simulation#Silent load-bearing constants]] |
| Gate layout | `track.py` | 8 gates, hardcoded coords |
| FPV + event camera | `drone_race_env.py` (`EventCameraSensor`, `_render_fpv`) | [[perception]] |
| VIO simulation | `drone_race_env.py` (`VIOSimulator`) | [[perception#VIO]] |
| Feature extractor | `vision_model.py` and duplicated in train scripts | [[vision-model]] |
| Training: state expert | `drone-race-sim/train_state.py` (archive) | [[training]] |
| Training: distillation | `drone-race-sim/train_distill.py` (archive) | [[training]] |
| Training: end-to-end vision (abandoned) | `train_vision.py` | [[training#The abandoned 200M vision run]] |
| Deployment adapter | `dcl_adapter.py` | [[deployment]], [[fragilities]] |
| MAVLink protocol | `dcl_mavlink_adapter.py` | [[deployment]], [[fragilities]] |
| Competition entry point | `dcl_mavlink_client.py` (root; moved 2026-04-21) | [[deployment]] |

## Pure / side-effectful seams

| Boundary | Pure above / effectful below |
|---|---|
| `DroneRaceEnv.step()` | Effectful: mutates MuJoCo state, renders, samples noise |
| `DroneVisionExtractor.forward()` | Pure |
| `SCUBALabAdapter.predict_action()` | Pure (CPU/GPU tensor math) |
| `SCUBALabAdapter.process_observation()` | Mostly pure (reshapes, pads event channels with zeros) |
| `SCUBALabMAVLinkAdapter.step()` | Constructs MAVLink frames (pure byte construction) |
| `dcl_mavlink_client.control_loop()` | Effectful: async I/O, UDP send via MAVSDK |

Testing note: everything above `control_loop` is easy to test. The
[[deployment]] failures start **at** the point where `control_loop`
ties everything together with a zero-placeholder vision frame and a
broken MAVSDK call; none of the unit-testable layers touch the real
failure surface.

## Repository layout (root = canonical, `drone-race-sim/` = thin archive)

Since `refactor/deduplicate-root-subdir` (2026-04-21), root holds the
single canonical copy of every VQ1-deployment file: `dcl_adapter.py`,
`vision_model.py`, `config.py`, `drone_race_env.py`, `track.py`,
`race.py`, `dcl_mavlink_adapter.py`, `dcl_mavlink_client.py`,
`test_dcl_adapter.py`, `measure_inference_time.py`,
`trained_distilled/`, `models_release/*.zip`, plus the docs.

`drone-race-sim/` is a thinner archive. It retains training scripts
(`train_state.py`, `train_distill.py`, `train_vision.py`, the `train_*`
HPC and experimentation variants), historical checkpoint directories
(`trained_swift_100m/` and the earlier-experiment siblings), HPC
`.slurm` job scripts, and a long tail of debug/demo/visualization
utilities that remain useful for archaeology but are not part of the
deployment package.

Caveat: `drone-race-sim/` is **still its own nested git repo**, with
its own history and an `origin` remote that happens to point at the
same GitHub project as the outer repo. Flattening the nested repo is
deferred post-VQ1, and the shared-remote situation is a known
footgun documented in
[[fragilities#Nested repo shares the outer repo's GitHub remote]].
The nested repo's pre-existing working-tree drift (10 modified
files, 30+ untracked scripts from past experiments) is also
explicitly out of scope for the VQ1 branches.

## See also

- [[simulation]] — the environment in detail
- [[perception]] — FPV and event camera modelling
- [[vision-model]] — the CNN architecture and its quirks
- [[deployment]] — how the trained model is supposed to reach the sim
- [[fragilities]] — where this fails
