#deployment #competition

# Submission readiness

Operational checklist. If VQ1 submission were required tomorrow, this
is what would break, in priority order. Companion to [[fragilities]],
which has the long-form explanations. This file is a deadline artifact —
check items off here as they're fixed, and the wiki will accurately
describe the state of the deployment.

**Today's date:** 2026-04-21. **VQ1 deadline:** May 2026 (no specific
date known to us). **Net window:** a few weeks.

## Blocking issues (cannot submit without fixing)

### 1. ~~Policy weights do not load~~ — RESOLVED 2026-04-21

- **File:** `dcl_adapter.py`
- **Was:** `PolicyNet`'s `pi_net.` key-prefix filter matched zero keys
  in the real checkpoint; adapter ran random weights.
- **Done on branch `fix/policy-weight-load`:**
  - Rebuilt `PolicyNet` to the real topology
    (`mlp_extractor.policy_net: Linear(256,128)→Tanh→Linear(128,64)→Tanh`,
    `action_net: Linear(64,4)` bare). No state re-concat, no final
    Tanh. Activations verified against SB3 2.7.1 defaults.
  - Loader uses `strict=True`; any shape or key mismatch crashes
    loudly. Removed the `try/except` wrapper.
  - Post-load logs per-layer weight std/mean/|max|. Measured stats
    confirm trained weights (std ~0.09–0.14 on policy layers; SB3
    action_net init is 0.01, so post-training 0.09 = ~9× above init).
  - Rewrote `drone-race-sim/test_dcl_adapter.py` to import from root,
    point at real weights, and assert loaded-vs-Kaiming std.
- **Status:** ✅ Fixed on branch `fix/policy-weight-load`. Merge once
  eyeballed. See [[fragilities#The policy head is untrained at deployment]]
  for the resolution notes.

### 2. MAVLink output interprets body rates as attitude angles

- **File:** `dcl_mavlink_adapter.py:229-232`
- **What:** Policy emits rad/s body rates; adapter scales by π/4 and
  writes them into attitude fields. Drone would fly wrong even if
  every other piece worked.
- **Fix:** write body rates (scaled by `MAX_BODY_RATE = 12`) into
  `SET_ATTITUDE_TARGET`'s `body_roll_rate / body_pitch_rate /
  body_yaw_rate` fields, and set `type_mask` to ignore the quaternion
  (`type_mask = 0b10000000` per MAVLink — IGNORE_ATTITUDE).
- **Effort:** 1 hour including test.
- **Status:** ☐

### 3. MAVLink CRC is zero

- **File:** `dcl_mavlink_adapter.py:150`
- **What:** Literal `b"\x00\x00"` instead of CRC-16-CCITT. Any
  spec-compliant parser rejects every message.
- **Fix:** compute CRC-16-CCITT with the per-message-ID seed byte.
  Use `pymavlink.generator.mavcrc.x25crc` or a hand-rolled CRC table.
  Seed bytes for `HEARTBEAT` (239) and `SET_ATTITUDE_TARGET` (49) are
  fixed in the MAVLink spec.
- **Effort:** 1 hour including seed-table lookup.
- **Status:** ☐

### 4. MAVLink payload is malformed

- **File:** `dcl_mavlink_adapter.py:55-87`
- **What:** Three competing `struct.pack`s in sequence; final payload
  has wrong field ordering and no quaternion.
- **Fix:** replace with either (a) the canonical `struct.pack` for
  `SET_ATTITUDE_TARGET` as defined in the MAVLink v2 common dialect,
  or (b) `pymavlink.dialects.v20.common.MAVLink_set_attitude_target_message`
  and use its `.pack()` method. Option (b) also gets you the CRC seed
  for free.
- **Effort:** 1–2 hours, mostly reading the message definition.
- **Status:** ☐

### 5. Advertised model path is wrong everywhere

- **Files:** `README_SCUBA_LAB.md`, `COMPETITION_CHECKLIST.md`,
  `README_COMPETITION.md`, `dcl_mavlink_client.py`
- **What:** Docs point at `~/Desktop/AI GrandPrix/AI_GrandPrix_Models/aigp_distill_final.zip`.
  File is not there. Real model is at
  `drone-race-sim/models_release/aigp_distill_final.zip`.
- **Fix:** either copy the model to `AI_GrandPrix_Models/` (makes the
  docs true) or update every doc reference to point at
  `drone-race-sim/models_release/`. The copy is simpler and lets you
  delete four doc edits.
- **Effort:** 5 minutes.
- **Status:** ☐

### 6. Vision stream is a placeholder

- **File:** `drone-race-sim/dcl_mavlink_client.py:158`
- **What:** Every control cycle, `vision_frame = np.zeros((48,48,3))`.
  The model runs on black frames.
- **Fix:** blocked on DCL simulator release. Once the simulator exposes
  an image topic, replace the zero frame with the real stream. Budget
  2–4 hours for integration + validation once the simulator ships.
- **Effort:** blocked until May 2026; 2–4 h once unblocked.
- **Status:** ⏳ blocked on DCL sim

## Should-fix-before-submission (not strictly blocking but degrades everything)

### 7. `dcl_mavlink_client.py` uses the wrong MAVLink message

- **File:** `drone-race-sim/dcl_mavlink_client.py:215-225`
- **What:** Calls `set_actuator_control`, a different msg_id. DCL spec
  requires `SET_ATTITUDE_TARGET` or `SET_POSITION_TARGET_LOCAL_NED`.
- **Fix:** switch to `system.offboard.set_attitude_rate(...)` and
  `offboard.start()`, or emit raw frames via our own adapter once the
  adapter is fixed (#2–#4 above).
- **Effort:** 1 hour.
- **Status:** ☐

### 8. Silent `except: pass` in the MAVLink send

- **File:** `drone-race-sim/dcl_mavlink_client.py:226-231`
- **What:** Two bare excepts swallow all send errors.
- **Fix:** log + re-raise, or at least log + increment a
  failure counter. Observability floor.
- **Effort:** 10 minutes.
- **Status:** ☐

### 9. `test_dcl_adapter.py` is tautological

- **File:** `drone-race-sim/test_dcl_adapter.py`
- **What:** Points at the wrong model path, and even if it found the
  model it only checks "load doesn't throw" and "output is in [-1,1]".
  Both pass with zero-loaded random weights.
- **Fix:** replace with a test that loads `policy.pth` with
  `strict=True`, runs a small deterministic rollout from a fixed
  seed, and compares action sums against a recorded baseline.
- **Effort:** 2 hours.
- **Status:** ☐

### 10. `FPV_TILT_DEG` drift in `drone-race-sim/config.py`

- **What:** Resolved by deletion on `refactor/deduplicate-root-subdir`
  (2026-04-21) — the nested copy of `config.py` was removed entirely,
  so its uncommitted `FPV_TILT_DEG = 0` drift no longer exists.
  Root's `config.py` (with `-10`) is now the only copy.
- **Status:** ✅ subdir drift gone. Branch C
  (`fix/fpv-tilt-canonicalization`) still owns the *empirical*
  tilt validation — confirm the trained model expects `-10` against
  a held-out observation before pinning it with an import-time assert.

### Additional item resolved on `refactor/deduplicate-root-subdir` (2026-04-21)

- **Duplicate-file structure (root vs `drone-race-sim/`):** ✅ resolved.
  Root is now the single source of truth for VQ1-deployment code. All
  identical duplicates deleted from `drone-race-sim/`; diverged files
  resolved by root-wins rule; load-bearing files
  (`test_dcl_adapter.py`, `measure_inference_time.py`,
  `dcl_mavlink_client.py`, `trained_distilled/`, model zips) moved
  to root. `.gitignore` amended to track the deployment artifacts.
  See [[fragilities#Canonical copy of each file]].
- **`sys.path.insert` hack in `test_dcl_adapter.py`:** ✅ removed with
  the move to root.
- **Surfaced during this branch** (new fragility entry): the nested
  `drone-race-sim/` git repo shares its `origin` remote with the
  outer repo. Structural footgun. See [[fragilities#Nested repo shares the outer repo's GitHub remote]].
  Not in scope for VQ1; fix deferred.

## Safe to defer until after submission

- Empirical check of event-channel zero-padding's effect on performance.
- Measure inference time on competition-class hardware.
- Sim-to-sim validation of the 200 Hz → 120 Hz physics mismatch.
- Episode-length mismatch (30 s training vs 480 s race cap) — probably
  irrelevant since lap time is short, but worth a sanity check.

## Decision: what to ship for VQ1

Choose in order of preference:

1. **Preferred:** #1 (weight load) is done; fix #2–#5, use corrected
   `dcl_adapter.py`, emit spec-compliant `SET_ATTITUDE_TARGET` with
   body rates. Integrate with DCL sim when it ships. Estimated
   effort to readiness-after-sim-ships: 1 day.
2. **Fallback if time is short:** use `mavsdk`'s high-level offboard
   mode and accept that we're not fully exercising the CTBR pipeline.
   Depends on #1 (done). Estimated effort: half a day.
3. **Do not ship the current code in any form.** The three silent
   failures compose into a submission that burns an entry with nothing
   to learn from.

## What to ask DCL once the simulator ships

- Exact MAVLink message(s) accepted for control. The spec lists two
  options; confirm which is primary.
- The image topic, frame rate, and encoding for the FPV stream.
- Whether `TIMESYNC` handshake is required before control is accepted.
- Whether our heartbeat `MAV_TYPE` (currently set to
  `MAV_TYPE_FIXED_WING = 1`, wrong — should be `MAV_TYPE_QUADROTOR = 2`)
  affects anything.

## See also

- [[fragilities]] — long-form discussion of each item
- [[deployment]] — how the deployment layer is structured
- [[competition]] — the DCL spec
