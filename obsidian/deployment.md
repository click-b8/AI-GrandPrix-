#deployment #fragility

# Deployment

How the trained model is *supposed* to reach the DCL simulator, and
where the path is currently broken. This file describes intent plus
current state. For the operational punch list see
[[submission-readiness]]; for the long-form diagnoses see
[[fragilities]].

## Intended pipeline

```
DCL simulator (UDP, MAVLink v2, 120 Hz physics)
  ├── emits vision stream (48×48 FPV) + telemetry (attitude, velocity, position)
  └── accepts SET_ATTITUDE_TARGET or SET_POSITION_TARGET_LOCAL_NED at 50–120 Hz
            │
            ▼
drone-race-sim/dcl_mavlink_client.py
  ├── asyncio MAVSDK connection
  ├── subscribes to telemetry
  ├── polls vision frame (currently stubbed: np.zeros)
  ├── invokes SCUBALabAdapter.step()
  └── sends attitude command (currently: set_actuator_control — WRONG MESSAGE)
            │
            ▼
dcl_adapter.py  (SCUBALabAdapter)
  ├── process_observation: resize → (14,48,48) w/ event channels zero-padded
  ├── DroneVisionExtractor → 256D features (weights load OK)
  ├── PolicyNet → 4D action (weights DO NOT load; random output)
  └── action_to_dcl_command: clip to ranges
            │
            ▼
dcl_mavlink_adapter.py  (encode_set_attitude_target)
  ├── treats command[1:4] as attitude angles (WRONG — they are body rates)
  ├── packs a malformed payload (multiple reassigned struct.pack)
  └── appends b"\x00\x00" for CRC (WRONG — spec requires CRC-16-CCITT)
```

Every stage has a problem. Most of the problems fail silently — see
[[fragilities#Silent failure chain]] for why that combination is worse
than any single loud error.

## The code paths

### `dcl_adapter.py`

Purpose: load `policy.pth` directly (bypassing `PPO.load()`) and run
inference. It exists because the training HPC ran SB3 2.7.1 + torch 2.6
and the local Mac runs SB3 2.8 + torch 2.11; pickle-based loading
breaks across that gap. See [[decisions-log#Direct weight loading over PPO.load]].

Three concerns:
1. `SCUBALabAdapter.process_observation` — converts DCL telemetry dict
   and RGB image to the model's Dict observation format. Zero-pads
   event channels because DCL provides RGB only. See
   [[perception#The deployment gap — event channels are zero-padded at inference]].
2. `SCUBALabAdapter.predict_action` — runs the feature extractor and
   policy. Returns a 4D array clamped to `[0,1] × [-1,1]³`.
3. `SCUBALabAdapter.action_to_dcl_command` — wraps the array as a dict
   with named keys.

The load logic (`_load_weights`) is where the critical bug lives:

```python
feature_dict = {k: v for k, v in checkpoint.items() if k.startswith('features_extractor.')}
policy_dict  = {k: v for k, v in checkpoint.items() if k.startswith('pi_net.')}
```

The second filter matches zero keys. `load_state_dict(policy_dict,
strict=False)` silently leaves the handwritten `PolicyNet` at its
PyTorch default random init. Feature extractor loads fine; policy
head is untrained. See [[fragilities#The policy head is untrained at deployment]]
for confirmation steps and the replacement code.

### `dcl_mavlink_adapter.py`

Purpose: wrap the adapter with MAVLink byte-level encoding.

Four problems (each covered in [[fragilities]]):
- Payload structure does not match `SET_ATTITUDE_TARGET` as defined in
  MAVLink v2 common dialect.
- Output semantics swap body rates for attitude angles.
- CRC is zeroed.
- UDP transport is a stub (`# Send via UDP (not shown here)`).

Mitigating factor: `dcl_mavlink_client.py` (the real competition entry
point) bypasses this file entirely and uses MAVSDK instead — so
`dcl_mavlink_adapter.py` doesn't actually run in the happy path. But
its presence in the README package list ("deprecated — do not use") is
contradicted by the fact that it's duplicated to the root and
discussed in `DCL_INTEGRATION.md` as a primary component. Be explicit
about its status: deprecated, do not wire up, do not ship.

### `drone-race-sim/dcl_mavlink_client.py`

Purpose: the intended VQ1 entry point. Connects over UDP via MAVSDK,
subscribes to telemetry, runs the control loop at 50 Hz, emits control
commands.

Problems:
- Vision frames are placeholders (`np.zeros`), blocked on DCL sim release.
- Control command uses `set_actuator_control`, which is **not** the
  spec-required `SET_ATTITUDE_TARGET` / `SET_POSITION_TARGET_LOCAL_NED`.
- Two bare `except: pass` blocks swallow all send errors.
- Model path at line 287 points at the non-existent
  `AI_GrandPrix_Models/aigp_distill_final.zip`.

Also lives **only** in the `drone-race-sim/` subdirectory, not at
root. The root "curated" package is missing the intended entry point.
See [[fragilities#Canonical copy of each file]].

## Direct weight loading rationale and how it broke

`dcl_adapter.py`'s docstring says it was written to avoid SB3 version
conflicts. That's a correct read of the environment: HPC ran SB3
2.7.1; local venv runs 2.8.0; loads across that gap are fragile.

The direct-load pattern is fine in principle. The bug is that the
adapter's `PolicyNet` was written against an **assumed** topology
(`Linear(275, 128)`, state re-concatenated) that doesn't match the
**actual** trained topology (`Linear(256, 128)`, state already embedded
in features). The load-with-prefix-filter pattern is too silent when
the assumed topology is wrong: keys just don't match, nothing raises,
`strict=False` + `try/except` closes the observability gap.

The lesson is less "use `PPO.load`" and more "use `strict=True` in
load_state_dict". The adapter's use of `strict=False` was defensive
(in case of partial weight compatibility across versions) and became
the thing that hid a 100% mismatch.

See [[decisions-log#Direct weight loading over PPO.load]] for the decision,
and [[fragilities#The policy head is untrained at deployment]] for the fix.

## Hardware targets per `DCL_INTEGRATION.md`

- Minimum: Intel i5-10400F / Ryzen 5 3600, RTX 2060 Super / AMD 9060XT,
  16 GB RAM.
- Optimized for: ~100 TOPS embedded AI hardware.
- Python 3.14.2+ expected (per DCL spec; we train on 3.12).

We have no inference measurements on any of the target-class hardware.
The only number is 0.26 ms mean on Apple M4 CPU, which doesn't
generalize to embedded edge-AI class. See [[fragilities#Inference time]].

## What doesn't exist yet

- The DCL simulator (expected May 2026).
- Any end-to-end validated flight through our pipeline.
- A test harness that feeds non-zero images through the adapter and
  checks reasonable commands.

## See also

- [[fragilities]] — the details on each broken piece
- [[submission-readiness]] — the punch list to fix it
- [[competition]] — what DCL actually expects
- [[decisions-log]] — why we chose direct weight loading
