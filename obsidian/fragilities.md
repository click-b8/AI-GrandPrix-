#fragility #deployment

# Fragilities

The most important file in the wiki. Read this before touching
anything in the deployment path.

## Silent failure chain (read this first)

**Status as of 2026-04-23:** all three pillars are now addressed on
main. Pillar #1 (wrong advertised model path) resolved via
`refactor/deduplicate-root-subdir` and `fix/mavlink-complete`
(commits `582e28d` → PR #3 merge `f74981a`; `3c68ade` → PR #4 merge
`de6e87d`). Pillar #2 (weight-load) resolved earlier on
`fix/policy-weight-load`. Pillar #3 (CTBR → attitude semantic
mismatch) resolved via `fix/mavlink-complete` (commit `85d2a42` →
PR #4 merge `de6e87d`). **Three of three fixed; diagnostic record
kept below because the composed-silent-failure pattern remains the
load-bearing lesson.**

**Note on `run_vq1.py` (2026-04-23, `fix/vision-stub-guard`):** uses a
module-level `_allow_stub_vision` flag; test isolation relies on
`try/finally`. If the flag grows additional toggle callers, prefer
dependency injection to avoid test-order-dependent flakes.

The deployment pipeline has three independent failure points, each of
which fails silently with an individually-reasonable-looking fallback.
The composition is a system that will appear to run end-to-end —
loading weights, processing frames, emitting bytes — and fly the drone
unusably. This is a worse failure mode than any single loud error,
because there is no grep-able symptom. The three are:

1. **The advertised model path does not exist.** Four documents
   (`README_SCUBA_LAB.md`, `COMPETITION_CHECKLIST.md`, `README_COMPETITION.md`,
   `dcl_mavlink_client.py:287`) point at
   `~/Desktop/AI GrandPrix/AI_GrandPrix_Models/aigp_distill_final.zip`.
   That file does not exist. Only `aigp_8gates_final.zip` lives at that
   path. The real distilled vision model is at
   `drone-race-sim/models_release/aigp_distill_final.zip`. The `test_dcl_adapter.py`
   script short-circuits at the missing path before ever exercising
   the adapter.

2. **The policy head is untrained at deployment.** The adapter's
   `PolicyNet` filter prefix (`pi_net.`) matches **zero** keys in the
   actual checkpoint. The real policy head lives under `mlp_extractor.policy_net.*`
   and `action_net.*`. Because `load_state_dict` is called with
   `strict=False` inside a `try/except`, nothing loads and the adapter
   prints "✓ Loaded policy weights". The running weights are Kaiming
   random init. Measured: `pi_net[0]` has std 0.0347, max 0.060 — exactly
   the torch default for a 275-fan-in layer.

3. **The MAVLink output semantics are wrong.** The trained policy emits
   CTBR — body rates in rad/s (scaled by `MAX_BODY_RATE = 12`). The
   adapter (`dcl_mavlink_adapter.py:229-232`) treats those same three
   outputs as **attitude angles** and scales them by `π/4`, packing them
   into the `roll/pitch/yaw` fields of `SET_ATTITUDE_TARGET`. Even if #1
   and #2 were fixed, the drone would be flown to ±45° attitude setpoints
   interpreted from values the network learned to mean "angular velocity".
   MAVLink doesn't validate semantic meaning; the bytes go out looking
   well-formed.

Any one of these is a loud-ish bug. Composed, they look like a working
system: load succeeds, inference succeeds, frames go out. The drone
would just fly terribly, with no traceable point of failure. **The
fragility isn't the individual bugs — it's the silence.** Wherever you
see `strict=False`, bare `except: pass`, try/except over a load, or a
"not shown here" comment in this project, treat it as a load-bearing
place we've decided to lie to ourselves about.

Below, each fragility gets its own entry with severity, measurement
status, and impact. See also [[submission-readiness]] for the operational
equivalent.

---

## Measurement status

For each fragility, the entry notes whether the drift was:
- **Resolved by measurement** — a concrete check was run; the truth is now pinned.
- **Resolved by choice** — the canonical answer was decided; the alternatives are historical.
- **Unresolved** — still open; marked as an [[open-questions]] candidate.

---

## The policy head is untrained at deployment

- **Severity:** ~~critical~~ → **RESOLVED 2026-04-21** on branch
  `fix/policy-weight-load`. See the "Resolution" subsection at the end
  of this entry for the fix details. Leaving the diagnostic record in
  place because it's the canonical example of the silent-failure-chain
  pattern and future-us should understand how it happened.
- **Measurement status:** resolved by measurement (see below) and then
  fixed in code.
- **Affected files:** `dcl_adapter.py` (root is canonical; the
  drone-race-sim/ duplicate remains stale — a structural concern
  deferred to a separate branch).

### What actually happens

`SCUBALabAdapter.__init__` unzips `policy.pth` from the model `.zip`
and calls `_load_weights()`. That function builds two filtered dicts:

```python
feature_dict = {k: v for k, v in checkpoint.items() if k.startswith('features_extractor.')}
policy_dict  = {k: v for k, v in checkpoint.items() if k.startswith('pi_net.')}
```

The checkpoint has:
- 16 keys starting with `features_extractor.` → `feature_dict` loads correctly.
- **0 keys starting with `pi_net.`** → `policy_dict` is empty.

The real policy-path weights in the checkpoint are:

| Key | Shape |
|---|---|
| `mlp_extractor.policy_net.0.weight` | `(128, 256)` |
| `mlp_extractor.policy_net.0.bias` | `(128,)` |
| `mlp_extractor.policy_net.2.weight` | `(64, 128)` |
| `mlp_extractor.policy_net.2.bias` | `(64,)` |
| `action_net.weight` | `(4, 64)` |
| `action_net.bias` | `(4,)` |

None match the `pi_net.` prefix. Even if the prefix were corrected,
`PolicyNet`'s first layer is `Linear(275, 128)` (because the adapter
re-concatenates state onto features), whereas the trained first layer
is `Linear(256, 128)` (because the trained feature extractor already
embeds state; the real policy head takes 256D features directly).
There is also an extra `Tanh()` at the end of `PolicyNet` that is
**not present during training** — training uses a Gaussian policy with
a bare linear `action_net`.

After `load_state_dict(policy_dict, strict=False)` returns with zero
matched keys, the adapter prints "✓ Loaded policy weights" because the
print is in the `try:` branch and no exception fired.

### How to confirm

```python
from dcl_adapter import SCUBALabAdapter
a = SCUBALabAdapter('drone-race-sim/models_release/aigp_distill_final.zip')
w = a.policy.pi_net[0].weight.detach()
print(w.shape, w.std().item(), w.abs().max().item())
# Observed: (128, 275) std=0.0347 max=0.060 — Kaiming default for fan_in=275.
```

### What a correct adapter looks like

Rebuild `PolicyNet` to match the actual checkpoint topology:

```python
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_extractor_policy_net = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.action_net = nn.Linear(64, 4)

    def forward(self, features):
        x = self.mlp_extractor_policy_net(features)
        return self.action_net(x)
```

And load with the corrected key mapping:

```python
policy_keys = {
    'mlp_extractor_policy_net.0.weight': ckpt['mlp_extractor.policy_net.0.weight'],
    'mlp_extractor_policy_net.0.bias':   ckpt['mlp_extractor.policy_net.0.bias'],
    'mlp_extractor_policy_net.2.weight': ckpt['mlp_extractor.policy_net.2.weight'],
    'mlp_extractor_policy_net.2.bias':   ckpt['mlp_extractor.policy_net.2.bias'],
    'action_net.weight':                 ckpt['action_net.weight'],
    'action_net.bias':                   ckpt['action_net.bias'],
}
self.policy.load_state_dict(policy_keys, strict=True)  # strict=True on purpose
```

And drop the Tanh — action clipping is applied downstream at
`action_to_dcl_command`. Alternatively, replace the handwritten
`PolicyNet` entirely with SB3's `MlpExtractor` plus the action net.

**Do not ship with `strict=False` anywhere in this path.** That flag is
exactly what let this bug live undetected.

See also: [[deployment#Direct weight loading rationale and how it broke]].

### Resolution (2026-04-21, branch `fix/policy-weight-load`)

`PolicyNet` and `_load_weights` in the root `dcl_adapter.py` were
rewritten to:

1. Match the real trained topology — `mlp_extractor.policy_net` as
   `Linear(256,128) → Tanh → Linear(128,64) → Tanh` and `action_net`
   as a bare `Linear(64,4)`. No state re-concatenation. No final
   Tanh (Gaussian-mean, not squashed). Activations verified against
   SB3 `MultiInputActorCriticPolicy` defaults at the SB3 2.7.1 used
   on the training HPC.
2. Load with `strict=True`. No `try/except` around the load. Any
   future drift between adapter topology and checkpoint will crash
   loudly.
3. Log per-layer weight statistics (std / mean / |max|) on every
   adapter init. Kaiming-uniform for the three policy-head layers
   would be std ≈ 0.051 / 0.072 / 0.102 for fan-in 256 / 128 / 64.
   Measured post-load: 0.100 / 0.138 / 0.090 — all meaningfully above
   Kaiming floors, confirming trained weights.

`drone-race-sim/test_dcl_adapter.py` was rewritten to:

- Force-import the canonical adapter from the project root via
  `sys.path`, instead of the stale subdir duplicate.
- Point at the real weights path
  (`drone-race-sim/models_release/aigp_distill_final.zip`).
- Assert each policy layer's std exceeds the Kaiming-uniform floor
  — distinguishing "loaded" from "silently initialized".
- Feed 10 distinct observations and confirm at least one action
  channel varies with input. Of note: the trained throttle channel
  saturates at `1.0` post-clip for every input (raw action_net
  output is 1.5 – 1.8, consistently above the clip ceiling). This is
  the trained policy's learned behavior for a race objective, not a
  bug. Roll/pitch/yaw all vary.

See [[submission-readiness]] item 1 for the operational close-out.

### Historical note on the silent-failure-chain

This bug is the canonical example of the silent-failure-chain pattern
discussed at the top of this file. Three independently-benign-looking
defensive patterns (`strict=False`, `try/except` around the load, a
"success" print inside the `try` branch) composed into a system that
printed "✓ Loaded policy weights" while loading zero weights. Every
one of those patterns was added for a reason — cross-version
robustness, error tolerance, user feedback — and each individually was
defensible. Together they erased the observability for a 100% load
failure.

The general lesson is visible throughout the project's deployment
layer: `except: pass`, `strict=False`, "not shown here" comments, and
fallback paths of any kind need a loud audit every time they go in.
Defensive programming and observability can point in opposite
directions; when they do, observability wins.

---

## MAVLink action semantics mismatch (CTBR → attitude)

- **Severity:** ~~critical~~ → **RESOLVED 2026-04-23** on branch
  `fix/mavlink-complete`. Commit `85d2a42` rewrote the CTBR semantic;
  on main via PR #4 merge `de6e87d`. Diagnostic record preserved
  below.
- **Measurement status:** resolved by inspection + test coverage in
  `tests/test_vq1_readiness.py::TestCTBRSemantics`.
- **Affected files:** `dcl_mavlink_adapter.py`

The trained policy emits [normalized_thrust, roll_rate, pitch_rate,
yaw_rate]. Rates are in [-1, 1] and are scaled inside the training env
by `MAX_BODY_RATE = 12.0 rad/s`. See `config.py:38` and `drone_race_env.py`
`step()` → `target_body_rates = delayed_action[1:] * MAX_BODY_RATE`.

`dcl_mavlink_adapter.py:229-232`:

```python
roll  = float(command["roll"])  * np.pi / 4  # Map [-1, 1] to [-π/4, π/4] radians
pitch = float(command["pitch"]) * np.pi / 4
yaw   = float(command["yaw"])   * np.pi / 4
```

Those values are then passed to `encode_set_attitude_target(roll=..., pitch=..., yaw=...)`,
which writes them into the attitude quaternion fields (the code path
that would write a proper quaternion; in the current broken encoding
it writes them as three floats). The `SET_ATTITUDE_TARGET` message has
separate fields for **attitude** (quaternion `q[4]`) and **body rates**
(`body_roll_rate, body_pitch_rate, body_yaw_rate`); selection is
controlled by the `type_mask`. Our adapter hardcodes `type_mask=0x04`
which is in any case wrong for both interpretations — see VADR-TS-001.

**The fix requires both a semantic decision and a protocol-correct
encoding**: choose whether the competition expects rates or attitude,
emit into the correct MAVLink fields, and set `type_mask` accordingly.
If rates: the output mapping is `body_*_rate = command[k] * MAX_BODY_RATE`
(not `* π/4`). If attitude: the policy was not trained for that
interface and a different model or a conversion layer is needed.

Per the DCL spec (VADR-TS-001; see [[competition]]), both
`SET_ATTITUDE_TARGET` and `SET_POSITION_TARGET_LOCAL_NED` are
acceptable. CTBR-style rate control maps naturally to
`SET_ATTITUDE_TARGET` with `type_mask` set to ignore the attitude
quaternion and use body rates + thrust.

---

## MAVLink CRC is a zeroed placeholder

- **Severity:** ~~critical for spec compliance; blocking for submission~~
  → **RESOLVED 2026-04-23** on branch `fix/mavlink-complete`. Commit
  `85d2a42` replaced the zeroed CRC with CRC-16-CCITT computed from
  per-message `crc_extra` seeds pulled from `pymavlink`'s message
  classes; on main via PR #4 merge `de6e87d`.
- **Measurement status:** resolved by inspection + byte-exact
  roundtrip test in `tests/test_mavlink_compliance.py::TestCRC`.
- **Affected files:** `dcl_mavlink_adapter.py:149-150`

```python
# CRC-16-CCITT (will compute if needed; for now, use zeros)
frame.extend(b"\x00\x00")  # Placeholder
```

MAVLink v2 frames carry a CRC-16-CCITT computed over the frame header
and payload with a per-message-ID seed byte. A spec-compliant parser
(pymavlink, MAVSDK, etc.) will discard any frame whose CRC does not
match. Our frames have a literal `\x00\x00` at the CRC position. Any
correctly-implemented receiver rejects 100% of our messages.

The CRC computation is ~20 lines of code and uses the well-known
message-ID-seeded CCITT variant. `pymavlink.generator.mavcrc.x25crc`
or a hand-rolled CRC table are both acceptable. The seed bytes for
`HEARTBEAT` and `SET_ATTITUDE_TARGET` are fixed constants in the
MAVLink message definitions.

---

## MAVLink payload is malformed

- **Severity:** ~~critical~~ → **RESOLVED 2026-04-23** on branch
  `fix/mavlink-complete`. Commit `85d2a42` replaced the handrolled
  `struct.pack` payload with `pymavlink`'s `MAVLink_set_attitude_target_message`
  class. Monotonic sequence byte and `MAV_TYPE_QUADROTOR`. On main
  via PR #4 merge `de6e87d`.
- **Measurement status:** resolved by inspection + byte-exact
  pymavlink-reference roundtrip test.
- **Affected files:** `dcl_mavlink_adapter.py:55-87`

The `encode_set_attitude_target` function has three successive
`struct.pack(...)` calls that each reassign `payload`, with comments
like `"Actually use proper format with quaternion + rates"` and
`"Simpler approach: just pack the essential data"`. The final form
packs: uint32 time_boot_ms, four uint8s (type_mask, target_sys,
target_comp, 0), three floats (roll, pitch, yaw), three floats
(roll_rate, pitch_rate, yaw_rate), one float (thrust).

The real `SET_ATTITUDE_TARGET` payload is defined as: uint32
time_boot_ms, 4× float quaternion q[4], float body_roll_rate, float
body_pitch_rate, float body_yaw_rate, float thrust, uint8 target_system,
uint8 target_component, uint8 type_mask, (MAVLink v2 extensions:
float thrust_body[3]). Our three floats labeled "roll/pitch/yaw" do
not correspond to any quaternion field, the field ordering is wrong,
and the target_system/component/type_mask fields are in the wrong
position.

Even if the CRC were correct, the receiver would interpret our bytes
as garbage.

---

## UDP transport is absent in `dcl_mavlink_adapter.py`

- **Severity:** ~~critical for that file~~ → **RESOLVED 2026-04-23**
  on branch `fix/mavlink-complete`. Commit `85d2a42` added real
  `socket.sendto(frame, (host, port))` in `SCUBALabMAVLinkAdapter._send`,
  with `send_errors` counter for logged failures. On main via PR #4
  merge `de6e87d`.
- **Affected files:** `dcl_mavlink_adapter.py:273, 288`

Both lines contain `# Send via UDP (not shown here)`. There is no
socket, no `sendto`. The `run_loop` async function is a stub. As
written this module cannot emit a single byte over a network.

`drone-race-sim/dcl_mavlink_client.py` is the intended entry point and
does wire up an actual transport via the `mavsdk` Python library — but
see the next entry.

---

## `dcl_mavlink_client.py` uses `set_actuator_control`, not the spec-mandated messages

- **Severity:** ~~high — submission would not satisfy VADR-TS-001~~
  → **RESOLVED 2026-04-23** on branch `fix/mavlink-complete`. Commit
  `3c68ade` rewrote the client to delegate to `MAVLinkFrameBuilder`
  and emit spec-compliant `SET_ATTITUDE_TARGET` frames over a real
  UDP socket. On main via PR #4 merge `de6e87d`.
- **Affected files:** `dcl_mavlink_client.py` (now at root after
  dedupe).

The DCL spec (VADR-TS-001) requires `SET_ATTITUDE_TARGET` or
`SET_POSITION_TARGET_LOCAL_NED`. Our client calls MAVSDK's
`system.action.set_actuator_control(0, [...])`, which is a different
message (`SET_ACTUATOR_CONTROL_TARGET`, msg_id=139) and is unlikely to
be what the DCL sim accepts. The `except AttributeError: pass` around
the call means the wrong message is emitted if MAVSDK supports it,
and silently nothing is emitted if MAVSDK doesn't. Either way the
caller never knows.

The right pattern is `system.offboard.set_attitude_rate(AttitudeRate(...))`
paired with `offboard.start()`, or a direct raw `mavsdk.mavlink`
write of a properly-formed `SET_ATTITUDE_TARGET`.

---

## Vision stream is a placeholder — PARTIALLY RESOLVED

- **Severity:** ~~high — model currently runs on all-black frames~~
  → **GUARD LANDED 2026-04-23** on branch `fix/vision-stub-guard`.
  Commit `21762c5` made `run_vq1._get_vision_frame()` raise
  `NotImplementedError` unless `--allow-stub-vision` CLI flag is
  passed; on main via PR #5 merge `aae8af9`. The runtime guard
  prevents silent emission of garbage-input MAVLink streams. **The
  actual vision-integration (replacing the stub with a real DCL
  image-API call) remains blocked until the DCL simulator ships in
  May 2026.**
- **Affected files:** `run_vq1.py` (root; replaces
  `drone-race-sim/dcl_mavlink_client.py`'s prior stub).

```python
# In actual competition, this comes from DCL simulator
# For now, use placeholder (will be replaced with real vision stream)
vision_frame = np.zeros((48, 48, 3), dtype=np.uint8)
```

The control loop currently feeds a zero image to the policy every
cycle. Until the DCL simulator ships and its image-stream API is known,
this cannot be fixed. But we can't even simulate a reasonable flight
against a non-zero stream today — we have no test harness that feeds
real imagery.

---

## Event channels are zero-padded at deployment

- **Severity:** medium — documented sim-to-real gap
- **Measurement status:** resolved by choice (intentional)
- **Affected files:** `dcl_adapter.py:138-143`

Training used a simulated event camera. At deployment, DCL provides
only RGB. `process_observation` builds the 14-channel input by
concatenating RGB with four zero channels for the event histogram
bins, twice (for the two stacked frames).

This is intentional — the design bet is that the learned policy uses
events as auxiliary regularization during training and can operate on
RGB alone at inference. **This bet has not been validated.** No test
has measured policy performance with real events vs zeros. Given
#policy-head-untrained, this channel is essentially irrelevant in the
current state of the adapter; but once the load bug is fixed, test
both with event-ablated training data (zeros) and with simulated
events to see if performance drops.

See [[perception#Event camera simulation]].

---

## `FPV_TILT_DEG` local edit does not reflect training

- **Severity:** medium — silent drift; affects any local vision
  rendering using the drone-race-sim copy of config
- **Measurement status:** resolved by investigation; canonical value is −10°

`drone-race-sim/` is its own git repo, nested inside the root repo.
`drone-race-sim/config.py` was committed with `FPV_TILT_DEG = -10`
(commit `be4f89d`, 2026-03-22). On **2026-03-24 19:07** the local copy
was edited to `FPV_TILT_DEG = 0` and **never committed** — it still
shows as unstaged in `git status`. The HPC training that produced
`trained_distilled/policy.pth` completed on 2026-03-25 09:23, running
from its own clone, so it would have seen the committed `−10` value.

**Canonical answer: the deployed model was trained with `FPV_TILT_DEG = -10`.**
The root `config.py` has `-10`. The nested `drone-race-sim/config.py`
has `0` as an uncommitted local drift. Anyone using
`drone-race-sim/config.py` to render the policy's view
(`view_vision.py`, `live_fpv.py`) is looking at the model under wrong
camera geometry and will observe degraded behavior and misattribute it.

Fix: either commit the nested config to match root (`-10`) or revert
the uncommitted edit. Do not leave the drift in place.

---

## Canonical copy of each file

- **Severity:** ~~medium~~ → **RESOLVED 2026-04-21** on branch
  `refactor/deduplicate-root-subdir`. Root is now the single source of
  truth for all VQ1-deployment files; the duplicates in `drone-race-sim/`
  were deleted and the load-bearing files migrated up.
- **Measurement status:** resolved by action.

### Resolution

Identical duplicates removed from `drone-race-sim/` (13 files plus
`models_release/MODELS.json`):
`dcl_mavlink_adapter.py`, `train_vision.py`, `track.py`, `vision_model.py`,
`race.py`, `requirements.txt`, `run.sh`, `launch_viewer.sh`,
`DCL_INTEGRATION.md`, `COMPETITION_STRATEGY.md`, `README_SCUBA_LAB.md`,
`README.md`, `MODELS.json`.

Diverged files resolved by root-wins rule (per audit: root is canonical
for all three):
- `config.py` — root's `FPV_TILT_DEG = -10` survives; subdir's
  uncommitted `= 0` is gone with the deletion. Branch C
  (`fix/fpv-tilt-canonicalization`) still owns the *empirical* tilt
  validation — it just does so against the single remaining config.
- `drone_race_env.py` — root's newer content (alignment-bonus reward,
  cleaner viewer init) survives.
- `dcl_adapter.py` — root's post-`fix/policy-weight-load` version
  survives; subdir's pre-fix copy is gone.
- `models_release/README.md` — root's `author={[ SCUBA LAB]}` survives.

Load-bearing files moved from subdir to root (no history preservation;
these were either untracked in the nested repo or straight copies):
- `dcl_mavlink_client.py` — now at root. Previously **not tracked
  anywhere**; this branch introduces it to version control for the
  first time.
- `test_dcl_adapter.py` — at root, with the `sys.path.insert` hack
  removed (no longer needed).
- `measure_inference_time.py` — at root, preserving the pre-existing
  multi-path lookup modification from the subdir working tree.
- `trained_distilled/` — whole directory (policy.pth and siblings) at
  root.
- `models_release/aigp_distill_final.zip` — at root alongside the
  other two zips (`aigp_8gates_final.zip`, `aigp_racer_final.zip`),
  matching `MODELS.json`'s canonical registry.

`.gitignore` amended to carve out negations for these paths
(`!trained_distilled/**`, `!models_release/*.zip`). Other `trained*/`
directories in `drone-race-sim/` (historical training runs) remain
ignored.

### Still open (carried to separate branches)

- Branch B (`fix/canonical-model-path`) will decide the fate of
  `AI_GrandPrix_Models/aigp_8gates_final.zip` — a pre-existing
  tracked duplicate at a third path that this branch intentionally
  left alone.
- The nested `drone-race-sim/` git repo still exists; flattening it
  is deferred. See also the new "Nested repo shares the outer repo's
  GitHub remote" entry below.

---

## Nested repo shares the outer repo's GitHub remote

- **Severity:** medium — footgun, not immediately blocking, but can
  silently break a submission workflow
- **Measurement status:** resolved by inspection; no current fix planned
- **Discovered during:** `refactor/deduplicate-root-subdir`

The nested `drone-race-sim/` is its own git repo AND its `origin`
remote points at the **same GitHub URL** as the outer repo
(`git@github.com:click-b8/AI-GrandPrix-.git`). This means any `git
push` executed from inside `drone-race-sim/` targets the outer repo's
GitHub project. The nested repo's `main` branch and the outer repo's
`main` branch both push to the same `origin/main` on GitHub.

Concretely, during the `refactor/deduplicate-root-subdir` branch, a
`git fetch origin` from inside `drone-race-sim/` pulled the outer
repo's recent commits (including `73f226e` which has nothing to do
with the nested repo's history) into the nested repo's tracking refs.
The two repos have diverged histories at the same remote, which is
a state GitHub didn't intend and which will misbehave in ways that
depend on exactly what operation is attempted from inside the nested
repo.

### Concrete risk examples

- Running `git push` from inside `drone-race-sim/` while on its
  `main` would attempt to push its stale history on top of the outer
  repo's tip. GitHub refuses non-fast-forward pushes to `main` by
  default, so this probably errors rather than silently overwrites —
  but the error message won't obviously identify the root cause.
- `git push --force` from inside the nested repo would overwrite the
  outer repo's `main`. No guardrail against accidental invocation.
- Any branch created inside the nested repo and pushed (e.g., if
  someone inside `drone-race-sim/` ran `git push -u origin my-branch`)
  would create a branch on the OUTER repo's GitHub project — which
  might be what they wanted or might not, and which wouldn't show up
  where they expected.
- Pull requests on the GitHub project would not distinguish "from
  the outer repo" versus "from the nested repo", since both share
  the same remote URL.

### Mitigation options (not in scope for current VQ1 work)

- **Flatten the nested repo:** `rm -rf drone-race-sim/.git` and
  move its working tree into the outer repo as plain tracked files.
  Deferred post-VQ1 per [[vq1-execution-plan#Fix-vs-defer triage]].
- **Repoint the nested repo to a different remote** (e.g., a new
  GitHub project for archival purposes, or an orphan branch of the
  outer repo). Would preserve the nested repo's structure while
  eliminating the footgun.
- **Add a pre-push hook inside `drone-race-sim/.git/hooks/`** that
  refuses pushes matching the outer repo's URL. Fast interim
  mitigation; ~10 lines of bash.

### How to avoid tripping the footgun until it's fixed

- Never run `git push` or `git fetch origin` from inside
  `drone-race-sim/` unless you specifically intend to reach the
  outer repo's GitHub project.
- When committing in the nested repo (as this branch did for the
  dedupe deletions), the commit stays local unless explicitly
  pushed. Let it stay local; the outer repo's gitlink is what
  connects the two.
- If `git status` from the outer repo shows `m drone-race-sim`
  (nested repo has pending changes), that is the canonical signal
  — operate on the nested repo only enough to address those
  specific changes; don't generalize to other housekeeping.

---

## SSH credentials on this machine come from macOS Keychain, not ssh-agent

- **Severity:** low — documentary, not a failure state
- **Surfaced during:** `docs/post-audit-followup` force-push
  diagnostics, 2026-04-23

`ssh-add -l` on this machine returns "The agent has no identities"
as a steady state. This is not a failure signal. SSH credentials
are served by macOS Keychain via `UseKeychain yes` (a macOS-specific
SSH option, not part of OpenSSH's portable spec) in `~/.ssh/config`
or system-wide ssh_config, which loads keys on demand from the
Keychain without requiring `ssh-add` to populate the agent.

When diagnosing SSH-related git failures, use positive-capability
signals:
- `git fetch origin` — if it succeeds, git's SSH path works.
- `ssh -T git@github.com` — if it returns
  "Hi <username>! You've successfully authenticated", the key is
  being served and accepted by GitHub.
- `git remote -v` — sanity-check the URL hasn't drifted.

Do NOT use `ssh-add -l` presence as a failure gate on this machine;
the negative result is baseline. A retry-authorization conditional
that required "ssh-add -l shows at least one identity" would never
be satisfied here and would force unnecessary stops on real
transients.

On a non-macOS machine (Linux, WSL), this entry does not apply —
`UseKeychain` is unrecognized outside Apple's ssh fork, and
`ssh-add -l` is a meaningful check there.

---

## SB3 version incompatibility across machines

- **Severity:** medium — the reason `dcl_adapter.py` exists in its
  current direct-load form
- **Measurement status:** resolved by choice

The HPC trained with SB3 `2.7.1`, PyTorch `2.6.0+cu124`, Python 3.12.4
on Linux. Local venv is SB3 `2.8.0`, PyTorch `2.11.0`, Python 3.12.13
on arm64 macOS. `PPO.load()` across this version gap is flaky, which
is why `dcl_adapter.py` opens the `.zip` manually and loads only
`policy.pth`. That design decision is **correct in principle** (and
explicitly noted in `dcl_adapter.py`'s docstring). It's what makes the
filtering logic load-bearing — and that's the same filtering logic
that silently fails. See #policy-head-untrained.

---

## Inference time

- **Severity:** low for the number itself; medium for the doc-claim drift
- **Measurement status:** resolved by measurement on Mac, still open for target hardware

Existing claims in the project docs:

| Claim | Source |
|---|---|
| "~50ms on CPU, <10ms on GPU" | `README_SCUBA_LAB.md:128` |
| "<5ms" | `COMPETITION_CHECKLIST.md:93, 118`, `DCL_INTEGRATION.md:106`, `README_COMPETITION.md:109` |
| "0.1–0.5ms on CPU, <1ms on GPU" | `README_COMPETITION.md:109` |
| "target <50ms" (as a pass/fail threshold) | `drone-race-sim/measure_inference_time.py:39` |

**Measured via `adapter.step()` end-to-end**, 500 samples, 20-call
warmup, Apple M4 CPU (10 cores, 16 GB, Darwin arm64, torch 2.11):

| Metric | Value (ms) |
|---|---|
| Mean | 0.26 |
| P50 | 0.26 |
| P95 | 0.30 |
| P99 | 0.35 |
| Min–max | 0.23 – 0.40 |

This is the cost of the current pipeline. Two caveats:

1. The policy head is random init (see #policy-head-untrained). Weight
   values don't affect forward-pass cost, so this measurement is still
   a valid cost for the architecture — but you're timing a network
   that doesn't do the task.
2. Apple M4 is a very fast desktop-class CPU. The competition target
   hardware is ~100 TOPS embedded (roughly RTX 2060 Super class per
   `DCL_INTEGRATION.md`). M4 CPU performance does not predict embedded
   edge-AI performance.

**Honest wiki entry: inference costs 0.26 ms per step on an M4 Mac
CPU. We have no trustworthy figure for competition-class hardware.**
Re-measure on the target when available. Until then, all claims of
"5 ms" / "10 ms" / "50 ms" in the project docs should be disregarded
as wrong on at least one hardware target.

---

## Training / deployment physics-rate mismatch

- **Severity:** medium — potentially affects control stability in DCL sim
- **Measurement status:** open

Training: `SIM_FREQ = 200` Hz, `CTRL_FREQ = 100` Hz (config.py).
`MOTOR_TAU = 0.02` was hand-tuned against 200 Hz substeps.

Deployment (DCL): 120 Hz physics, 50–120 Hz command rate (per
VADR-TS-001). The motor-lag and rate-controller PD tuning will see
different dynamics in the DCL sim than in our MuJoCo sim. We have no
numbers on whether this matters in practice.

A sim-to-sim validation harness (replay a trained rollout against the
DCL sim once it's available) is the way to find out. Not actionable
until the sim ships. Flagged in [[open-questions]].

---

## Episode length / run-duration mismatch

- **Severity:** low — mostly fine, flagged because it's silent
- `EPISODE_LENGTH_SEC = 30` in training. DCL run cap is 480 s (8 min).
  Training episodes are 16× shorter than competition runs. Matters for
  runs where the policy may accumulate VIO drift or otherwise degrade
  over long horizons; the trained policy has never been evaluated past
  30 s. If lap time is ~5 s × 8 gates × 3 laps = ~2 min, we're still
  inside the training-distribution horizon. If VQ1 asks for long
  hover-like behavior, we may be out of distribution.

---

## Silent `except: pass` in the control loop

- **Severity:** ~~medium — destroys observability~~ → **RESOLVED
  2026-04-23** on branch `fix/mavlink-complete`. Commit `3c68ade`
  removed the two bare `except: pass` blocks; send errors now
  logged at WARNING and counted in `self.send_errors`. On main
  via PR #4 merge `de6e87d`.
- **Affected files:** `dcl_mavlink_client.py` (root after dedupe).

Two naked `except` blocks swallow everything from the MAVLink send.
Whatever the real failure is — auth, serialization, disconnect — the
control loop will never report it. Replace with explicit logging at
minimum.

---

## Test suite does not exercise the real model

- **Severity:** ~~high~~ → **RESOLVED 2026-04-21** on branch
  `fix/policy-weight-load` (same branch that fixed the load bug the
  old test couldn't catch).
- **Affected files:** `drone-race-sim/test_dcl_adapter.py`.

Original problems:

1. The script's model path pointed to the non-existent
   `AI_GrandPrix_Models/aigp_distill_final.zip`, so it short-circuited
   before loading.
2. Even if the path were correct, the "validation" checks were:
   adapter loads without exception; output is in range. The adapter
   loaded "without exception" with zero weights matched. The output
   range was `[-1, 1] × [0, 1]` by construction (Tanh + clip), so
   range-validation passed on random init.

The test was tautological — a perfect illustration of why the load
bug never surfaced: the thing that should have failed was checked by
something designed to pass regardless.

### Resolution

Rewrote the test to:

- Force-import the canonical root adapter via `sys.path` (subdir
  duplicates can drift; the test now pins to whatever `dcl_adapter.py`
  lives at the project root).
- Point at the real weights artifact.
- **Assert post-load layer stds exceed the Kaiming-uniform floor** —
  this is the check that would have failed on the old code. A test
  that can't distinguish "loaded" from "random-init" is exactly the
  test that let this bug ship; the new check closes that gap.
- Feed 10 distinct observations, log raw commands, and require at
  least one channel to vary with input. Weaker than the std check
  (a random-init net also varies) but a reasonable end-to-end smoke
  signal.

Future tests that want to do more — deterministic-rollout diffs
against a recorded trajectory, reward-curve regression, etc. — are
worth adding later. For the 2026-04 submission window, this level of
coverage is enough.

---

## `train_fast_safe.py` instantiates curriculum but never wires it to model.learn

- Severity: low for the project today (the script is unused
  archive-side, classified (c) status unclear in the provenance
  audit). Latent for any future revival of the script.
- Surfaced during: training-script provenance audit, 2026-04-27.
  See [[training-script-provenance]] §8 and
  [[training-script-provenance-self-review]] §5.

The script's name implies it produces a fast, safe-to-train
model. Inspection shows the curriculum callback (which is the
part that's supposed to deliver the "safe" guarantee — gradual
difficulty progression with reset-on-failure) is instantiated
but never passed to the SB3 `model.learn()` call. Training
proceeds as a flat run with no curriculum effect. The script's
output (or its absence — `trained_fast_safe/` has a `best_model`
but no final, no checkpoints) cannot be trusted as "safely
trained" in the way the script's name implies.

This is the silent-failure-chain pattern in training code:
defensive-looking structure (a curriculum is defined!) composes
with a wiring gap (it's never actually used) into a script that
runs without errors and produces output, while delivering none
of the guarantees its name suggests.

Resolution: if the script is ever revived for active use, wire
the curriculum callback into the `model.learn(callback=...)`
argument. If the script is left in archive, this entry serves as
the warning for any future reader who would otherwise assume
"fast_safe" means what it sounds like.

---

## `train_domain_robust.py` wraps training in broad `except Exception`

- Severity: low for the project today (the script is unused
  archive-side, classified (c) status unclear in the provenance
  audit). Latent for any future revival.
- Surfaced during: training-script provenance audit, 2026-04-27.
  Wording corrected by self-review pass: this is *broad
  continue-on-error*, not silent-failure — the except block prints
  `❌ Training error: {e}` and a full traceback before continuing
  to the next model in the loop. Errors are visible.

The concern is the breadth and the continue-on-error shape, not
stealth. A NaN explosion, a divergence, an OOM, a corrupted
checkpoint — all get caught by the same handler, all get
printed, all get continued past. Real training failures that
warrant stopping the run don't stop the run; they print and the
loop moves on to the next model.

Per the project's "observability first; re-raise or return a
typed result" guardrail, the catch should be narrowed (e.g.,
`except FileNotFoundError`) or replaced with logging + re-raise.
If this script is ever revived for real use, that block needs to
be tightened.

---

## See also

- [[submission-readiness]] — the operational view of the same list
- [[deployment]] — where each broken piece lives in the code
- [[decisions-log]] — why `dcl_adapter.py` uses direct weight loading
- [[open-questions]] — the items flagged as unresolved above
