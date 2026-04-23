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

### 2. ~~MAVLink output interprets body rates as attitude angles~~ — RESOLVED 2026-04-23

- **File:** `dcl_mavlink_adapter.py`
- **Was:** policy emitted rad/s body rates; adapter scaled by π/4 and
  wrote them into attitude fields.
- **Done on branch `fix/mavlink-complete`:** CTBR semantic rewrite —
  `body_roll_rate / body_pitch_rate / body_yaw_rate = policy_output *
  MAX_BODY_RATE (12 rad/s)`. `type_mask = 128` (IGNORE_ATTITUDE).
  Identity quaternion sent for the attitude field; FC ignores it.
- **Status:** ✅ Landed on main via `85d2a42` (feature) + `de6e87d`
  (PR #4 merge).

### 3. ~~MAVLink CRC is zero~~ — RESOLVED 2026-04-23

- **File:** `dcl_mavlink_adapter.py`
- **Was:** literal `b"\x00\x00"` instead of CRC-16-CCITT. Any
  spec-compliant parser rejects every message.
- **Done on branch `fix/mavlink-complete`:** CRC-16-CCITT computed
  with per-message `crc_extra` seed pulled from pymavlink's generated
  message classes. Verified via byte-exact roundtrip test against
  pymavlink's reference encoder.
- **Status:** ✅ Landed on main via `85d2a42` (feature) + `de6e87d`
  (PR #4 merge).

### 4. ~~MAVLink payload is malformed~~ — RESOLVED 2026-04-23

- **File:** `dcl_mavlink_adapter.py`
- **Was:** three competing `struct.pack`s in sequence; final payload
  had wrong field ordering and no quaternion.
- **Done on branch `fix/mavlink-complete`:** replaced with
  `pymavlink.dialects.v20.common.MAVLink_set_attitude_target_message`.
  Payload field ordering matches spec by construction; monotonically
  incrementing sequence byte; `MAV_TYPE_QUADROTOR` in heartbeat
  (was `FIXED_WING`).
- **Status:** ✅ Landed on main via `85d2a42` (feature) + `de6e87d`
  (PR #4 merge).

### 5. ~~Advertised model path is wrong everywhere~~ — RESOLVED 2026-04-23

- **Files:** `dcl_mavlink_client.py`, `DCL_INTEGRATION.md`.
- **Was:** docs pointed at `~/Desktop/AI GrandPrix/AI_GrandPrix_Models/aigp_distill_final.zip` —
  a file that never existed there. Real model was at
  `drone-race-sim/models_release/aigp_distill_final.zip`.
- **Done across two branches:** `refactor/deduplicate-root-subdir`
  moved the model to `./models_release/aigp_distill_final.zip`
  (canonical root path); `fix/mavlink-complete`'s `run_vq1.py` and
  updated `dcl_mavlink_client.py` reference the new canonical path
  with a fail-loudly `_check_model_path()` guard.
- **Status:** ✅ Landed on main via `582e28d` + `f74981a` (PR #3
  merge, path move) and `3c68ade` + `de6e87d` (PR #4 merge, client
  update).

### 6. Vision stream is a placeholder — PARTIALLY RESOLVED 2026-04-23

- **File:** `run_vq1.py`.
- **Was:** every control cycle, `vision_frame = np.zeros((48,48,3))`.
  The model ran on black frames and emitted a valid-looking MAVLink
  stream derived from garbage inference.
- **Done on branch `fix/vision-stub-guard`:** `_get_vision_frame()`
  now raises `NotImplementedError` unless `--allow-stub-vision` is
  passed. Running `run_vq1.py` without the flag fails loudly rather
  than silently emitting commands. The actual vision-integration
  (replace the stub with a real DCL image-API call) still cannot
  happen until the DCL simulator ships in May 2026.
- **Status:** ⏳ Guard landed on main via `21762c5` + `aae8af9`
  (PR #5 merge). Real vision-integration deferred until DCL sim
  ships; budget 2–4 h once unblocked.

## Should-fix-before-submission (not strictly blocking but degrades everything)

### 7. ~~`dcl_mavlink_client.py` uses the wrong MAVLink message~~ — RESOLVED 2026-04-23

- **File:** `dcl_mavlink_client.py`.
- **Was:** `set_actuator_control` (msg_id=139) — not spec-compliant.
  DCL requires `SET_ATTITUDE_TARGET` or `SET_POSITION_TARGET_LOCAL_NED`.
- **Done on branch `fix/mavlink-complete`:** client rewritten to
  delegate to `MAVLinkFrameBuilder` and emit `SET_ATTITUDE_TARGET`
  frames via a real UDP socket.
- **Status:** ✅ Landed on main via `3c68ade` (feature) + `de6e87d`
  (PR #4 merge).

### 8. ~~Silent `except: pass` in the MAVLink send~~ — RESOLVED 2026-04-23

- **File:** `dcl_mavlink_client.py`.
- **Was:** two bare excepts swallowing all send errors.
- **Done on branch `fix/mavlink-complete`:** removed. Send errors
  now logged at WARNING and counted in `self.send_errors`.
- **Status:** ✅ Landed on main via `3c68ade` (feature) + `de6e87d`
  (PR #4 merge).

### 9. ~~`test_dcl_adapter.py` is tautological~~ — RESOLVED 2026-04-23

- **File:** `test_dcl_adapter.py` (root; moved from `drone-race-sim/`).
- **Was:** pointed at non-existent path; only checked "load doesn't
  throw" and "output is in [-1,1]" — both passed with zero-loaded
  random weights.
- **Done across branches `fix/policy-weight-load` (initial rewrite),
  `refactor/deduplicate-root-subdir` (moved to root canonical path),
  `fix/mavlink-complete` (added `tests/test_mavlink_compliance.py`
  + `tests/test_vq1_readiness.py`), and `fix/vision-stub-guard`
  (added CLI subprocess test):** strict-load + Kaiming-floor checks,
  CTBR semantic assertions, byte-exact pymavlink roundtrip, 300-
  frame mock receiver, package-hygiene checks (no Desktop paths, no
  strict=False, no bare except:pass, FPV_TILT_DEG assertion), and
  end-to-end `--allow-stub-vision` CLI wiring test.
- **Status:** ✅ 28 tests pass on main. Landed via `fix/policy-weight-load`
  (earlier session), `582e28d` + `f74981a` (PR #3), `85d2a42` +
  `de6e87d` (PR #4), and `21762c5` + `aae8af9` (PR #5).

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
