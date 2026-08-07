#competition #deployment #planning

# VQ1 execution plan

Sequenced plan to take this project from its current audited state to a
submittable Virtual Qualifier 1 entry. Produced on **2026-04-21** from
the audit findings in [[fragilities]] and the operational checklist in
[[submission-readiness]]. VQ1 deadline: **May 2026**.

**Status anchor:** branch `fix/policy-weight-load` (commit `96e3ae2`) is
pushed to `origin` and **awaiting your review — not merged to main**.
Every branch below assumes that branch lands on main before sequencing
begins. If review uncovers issues and the branch is rebased, the plan
shifts right by the rebase duration but is otherwise unaffected.

---

## Planning inputs (known constraints)

These were open questions in the first draft. User review has closed
them, either by decision or by acknowledging what the project docs
already state. Recording them here so every branch below is read
against a fixed premise.

- **DCL simulator: not available until May 2026.** Per
  [[project-overview]] and [[competition]] — the simulator ships
  concurrent with VQ1. Every MAVLink branch is therefore validated
  against a `pymavlink` reference parser only; there is no
  pre-submission opportunity to validate against the real target.
  Branches that depended on sim access (vision-stream integration,
  physics-rate replay) move to the explicit post-submission queue.
- **MAVLink control message: `SET_ATTITUDE_TARGET` with `IGNORE_ATTITUDE`
  in `type_mask`, writing body rates into the `body_*_rate` fields.**
  The trained policy emits CTBR body rates (rad/s, scaled by
  `MAX_BODY_RATE = 12`), which composes cleanly only with
  `SET_ATTITUDE_TARGET`. `SET_POSITION_TARGET_LOCAL_NED` would require
  a position-controller layer we don't have and that training never
  produced. If DCL rejects `SET_ATTITUDE_TARGET` at submission, the
  fix is a post-submission fork.
- **Encoding library: `pymavlink`.** Existing `dcl_mavlink_client.py`
  uses `mavsdk` for the connection/telemetry half, which we keep.
  The emission path switches to `pymavlink`'s message classes for
  correct CRC seeding, payload ordering, and dialect conformance.
  Adding `pymavlink` to `requirements.txt` is part of branch E.
- **Structural cleanup: in-scope for VQ1.** Duplicate files between
  root and `drone-race-sim/` get deduplicated in a dedicated branch
  before the MAVLink work (see §3). Load-bearing files currently
  reachable only via `sys.path` hacks move to root. The nested-git-repo
  issue stays deferred — one thing at a time.
- **Python runtime: 3.12.** `DCL_INTEGRATION.md:99` mentions "3.14.2+"
  but no DCL-side source confirms that requirement. If DCL enforces
  3.14 at submission, address post-submission.
- **Submission format: clean zip-able directory** with pinned
  `requirements.txt`, README, and one entry-point script. Exact format
  is unknown pre-submission; adapt post-submission if DCL wants
  something else.
- **Gate pass tolerance: assume 1.0 m** (our training value).
  `README_COMPETITION.md:122` mentions 0.5 m; if VQ1 enforces tighter,
  a ~50M-step fine-tune is a post-submission fix.
- **Scope: VQ1-only.** The physical qualifier (September) is a
  separate effort. The plan does not trade VQ1 work for physical-phase
  investment; that decision is re-opened after VQ1 submits.

## Decisions needed from user

Short list. Two real tradeoffs where I will not make the call unilaterally.

1. **Belt-and-suspenders MAVLink:** implement `SET_POSITION_TARGET_LOCAL_NED`
   as a secondary path behind a runtime toggle, in addition to
   `SET_ATTITUDE_TARGET`? Adds ~6–9 h rescaled. Insurance if DCL
   rejects `SET_ATTITUDE_TARGET` at submission. Plan currently
   **defaults to single-path**; flip on request.
2. **Post-submission calendar buffer:** §1 and §6 note that the first
   VQ1 submission is itself a measurement, not a validation — what
   the sim accepts can only be probed by submitting. How many days
   of buffer after the first submission do you want reserved for at
   least one fix-cycle? The answer affects how aggressively the plan
   can triage the deferred items in §4.

---

## 1. Definition of "VQ1 ready"

The plan is done when all of the following hold:

1. **Canonical model load is strict and loud.** Already done on
   `fix/policy-weight-load`. The plan retains this as a floor: no
   subsequent branch may introduce a new `strict=False`, a try/except
   around a load, or a fallback path that silently substitutes defaults
   for missing data. If an integration point can't satisfy this, it
   isn't done.

2. **Emitted MAVLink bytes are spec-compliant.** A reference
   `pymavlink` parser can decode every frame we emit, CRC passes,
   sequence increments monotonically, `MAV_TYPE` reflects reality
   (quadrotor), and payload field ordering matches the v2 common
   dialect for whichever message we target. No handrolled
   `struct.pack`. No zeroed CRC.

3. **Emitted MAVLink semantics match the trained policy.** The policy
   outputs CTBR body rates; the decoded `body_roll_rate` /
   `body_pitch_rate` / `body_yaw_rate` fields of emitted
   `SET_ATTITUDE_TARGET` messages equal `policy_output * MAX_BODY_RATE`
   within float tolerance, and `type_mask` is set to ignore the
   attitude-quaternion field. We do not assume DCL accepts this
   without confirmation — see [[vq1-execution-plan#Open questions that block planning]].

4. **End-to-end dry run against a mock receiver passes.** A standalone
   UDP + `pymavlink` listener, spawned from our test harness, accepts
   our adapter's emission stream for at least 30 seconds at 50 Hz with
   zero parse errors, sequence gaps, or dropped frames. This does not
   prove DCL acceptance; it proves spec compliance, which is the floor.

5. **Submission package is single-pathed.** One canonical model
   artifact, one adapter entry point, one documented `python3 <script>`
   command. No mention in any README of a path that doesn't exist. The
   docs agree with the code.

6. **Wiki accurately reflects state.** No "✓ READY" claim for anything
   untested against a real DCL simulator. Every fragility entry is
   marked resolved with a commit hash or remains open with a listed
   blocker. Submission-readiness's checkboxes reflect reality.

7. **A final integration checkpoint confirms nothing regressed.** The
   last branch in the sequence runs an end-to-end scripted rollout
   against the mock sim with all prior fixes composed. Pass/fail on
   this is the single gate that declares "VQ1 ready".

**The first submission is itself a measurement, not a validation.**
Spec-compliance is the floor we can reach pre-submission; the
ceiling is whatever DCL's sim actually accepts, and that ceiling can
only be probed by submitting. Plan the calendar with at least one
post-submission fix cycle reserved — if VQ1 accepts retries, we
re-submit; if it doesn't, the first submission becomes diagnostic
data for the subsequent physical qualifier. "VQ1 ready" as defined
above means "ready to submit and learn", not "ready to win".

**Gaps this definition deliberately doesn't require:**

- Validation against the actual DCL simulator. Per planning inputs,
  the sim ships concurrent with VQ1 — there is no pre-submission
  window to validate against the real target. Submission is
  spec-compliant-but-unverified; see the measurement-not-validation
  note above.
- Inference measurement on target hardware. Blocked on hardware
  access. See [[fragilities#Inference time]].
- Zero-padded event channel ablation. Risk but not blocker.

---

## 2. Remaining blockers ranked by severity

Tag key: **SF** = submission-failing (DCL will reject or our policy
will fly unusably). **SD** = submission-degrading (works but worse
than it should). **COS** = cosmetic (visible in logs / docs, no
behavior impact). **BLK** = currently blocked on external event.

### B1. MAVLink payload and semantics (SF)
- **What:** `dcl_mavlink_adapter.py:229-232` scales CTBR outputs by
  `π/4` and writes them to attitude fields; payload encoding at
  lines 55-87 is malformed; CRC at line 150 is `b"\x00\x00"`;
  heartbeat `MAV_TYPE` is `FIXED_WING`; sequence is hardcoded to 0.
- **Evidence:** [[fragilities#MAVLink action semantics mismatch (CTBR → attitude)]],
  [[fragilities#MAVLink CRC is a zeroed placeholder]],
  [[fragilities#MAVLink payload is malformed]].
- **Effort:** M (split into two branches per your constraint; total
  ~5-6 hours including tests).
- **Risk of ignoring:** catastrophic — submission either rejected at
  parse time or flies to attitude setpoints interpreted from rad/s
  commands.
- **Dependencies:** resolution of Open Question Q2 (which message DCL
  accepts). Does not depend on simulator access; we can implement
  against spec.

### B2. Client uses wrong MAVLink message (SF)
- **What:** `drone-race-sim/dcl_mavlink_client.py:215-225` emits
  `set_actuator_control` (msg_id 139), not the spec-required
  `SET_ATTITUDE_TARGET` or `SET_POSITION_TARGET_LOCAL_NED`.
- **Evidence:** [[fragilities#dcl_mavlink_client.py uses set_actuator_control, not the spec-mandated messages]].
- **Effort:** M (~2-3 hours).
- **Risk of ignoring:** catastrophic — emitted messages ignored by a
  spec-compliant sim.
- **Dependencies:** B1 landed first. The client becomes a thin wrapper
  over the corrected encoder from B1.

### B3. Advertised model path is wrong everywhere (SF)
- **What:** Four docs and one code path point at
  `~/Desktop/AI GrandPrix/AI_GrandPrix_Models/aigp_distill_final.zip`,
  which does not exist. Real model is at
  `drone-race-sim/models_release/aigp_distill_final.zip`.
- **Evidence:** [[fragilities#Silent failure chain]] item 1;
  [[submission-readiness]] item 5.
- **Effort:** S (~30 min).
- **Risk of ignoring:** submission blob fails to load at runtime if
  anyone tries to use `dcl_mavlink_client.py` as written.
- **Dependencies:** none. Should land first — it's a trivial fix
  that removes a source of confusion for every subsequent branch.

### B4. Silent `except: pass` in client control loop (SD)
- **What:** `drone-race-sim/dcl_mavlink_client.py:226-231` swallows
  every error from the MAVLink send.
- **Evidence:** [[fragilities#Silent except: pass in the control loop]].
- **Effort:** S (~15 min).
- **Risk of ignoring:** we lose all visibility into submission-day
  failures. If something goes wrong at VQ1, we won't know why.
- **Dependencies:** folds naturally into B2 (same file).

### B5. FPV_TILT_DEG uncommitted drift (SD → COS)
- **What:** `drone-race-sim/config.py` has an uncommitted local edit
  changing tilt from `-10` to `0`. Model was trained with `-10`.
  Root `config.py` is correct.
- **Evidence:** [[fragilities#FPV_TILT_DEG local edit does not reflect training]].
- **Effort:** S (~5 min + validation).
- **Risk of ignoring:** low for VQ1 specifically — the submission runs
  root code paths, which have the correct tilt. But anyone using
  `view_vision.py` / `live_fpv.py` from the subdir is seeing the model
  under wrong camera geometry and will misdiagnose any performance
  issue they observe.
- **Dependencies:** none.
- **Caveat:** the conclusion "trained with -10" is inferred from git
  history, not confirmed against HPC state at training time. See
  critique #3 in the red-team pass.

### B6. Vision stream is a zero placeholder (BLK → post-submission)
- **What:** `drone-race-sim/dcl_mavlink_client.py:158` feeds
  `np.zeros((48,48,3))` as the vision frame every control cycle.
- **Evidence:** [[fragilities#Vision stream is a placeholder]].
- **Effort:** unknown until DCL simulator API is available.
- **Risk of ignoring:** the submission runs the model on black frames
  — worse than random policy since the trained CNN produces systematic
  outputs for an OOD input. This IS submittable in the first-submit-
  as-measurement sense (we get back signal about sim acceptance
  without needing a working vision stream), but not in the "will fly
  well" sense.
- **Dependencies:** DCL simulator release. Per planning inputs, the
  sim is not available pre-submission, so this work moves explicitly
  to the post-submission queue. Any first-submission cycle burns
  without this resolved.

### B7. Training/deployment physics rate mismatch (BLK → post-submission)
- **What:** Training at 200 Hz physics / 100 Hz control; DCL runs at
  120 Hz / 50-120 Hz command.
- **Evidence:** [[fragilities#Training / deployment physics-rate mismatch]].
- **Effort:** 1-2 hours of replay validation once sim available
  (rescaled: 1.5-3 h).
- **Risk of ignoring:** degradation magnitude unknown. Could be
  nothing, could be 20% reward loss.
- **Dependencies:** DCL simulator release. Post-submission.

### B8. Deprecated `dcl_mavlink_adapter.py` still reachable (SD)
- **What:** The file is marked "Deprecated — do not use" in
  `README_COMPETITION.md` but is still duplicated to root and
  described in `DCL_INTEGRATION.md` as a primary component. After
  B1+B2 land, this file serves no purpose but remains on-disk in two
  places as a trap.
- **Evidence:** audit section on duplication;
  `README_COMPETITION.md:94`.
- **Effort:** S (~1 hour including grep for imports).
- **Risk of ignoring:** someone imports it, gets the broken version.
- **Dependencies:** B1 complete.

### Not separately listed (subsumed or deferred)

- **Duplicate file structure (root vs subdir):** structural; see
  [[vq1-execution-plan#Fix-vs-defer triage]].
- **Nested git repo:** same.
- **Curriculum unification across training scripts:** not VQ1-relevant
  (we aren't retraining).
- **MAV_TYPE, sequence field, CRC individually:** folded into B1.
- **Event-channel zero-padding ablation:** see triage.

---

## 3. Proposed branch sequence

Each branch: ≤ 4 hours focused work, evidence-investigate step before
coding, at least one test that would catch the bug it's fixing. No
branch may include a `strict=False`, a try/except around a load, or a
silent fallback.

### Sequence overview

```
main
 └─ fix/policy-weight-load  ← already open, awaiting merge
     └─ (merged)
        └─ refactor/deduplicate-root-subdir   (M,  ~1.5-2 h,    A, new)
            ├─ fix/canonical-model-path           (S,  ~0.75-1.5 h,  B, B3)
            └─ fix/fpv-tilt-canonicalization      (S,  ~1-1.5 h,    C, B5)
                └─ investigate/mavlink-interface            (S, ~1.5-3 h,  D)
                    └─ test/mavlink-mock-sim-harness        (M, ~3-4.5 h, E)
                        └─ fix/mavlink-protocol-compliance  (M, ~3-4.5 h, F, B1a)
                            └─ fix/mavlink-body-rate-semantics  (M, ~3-4.5 h, G, B1b)
                                └─ fix/client-entry-point           (M, ~3-4.5 h, H, B2+B4)
                                    └─ refactor/deprecate-mavlink-adapter-stub  (S, ~1.5 h, I, B8)
                                        └─ verify/vq1-readiness-integration     (M, ~3-4.5 h, J, final gate)

Post-submission queue (blocked on DCL sim release, which ships
concurrent with VQ1):
    fix/vision-stream-integration       (K, B6)
    verify/sim-physics-rate-replay      (L, B7)
```

Dedupe (A) lands first. B and C can then run in parallel against
deduplicated files — no lockstep tax. D–J are serial. Rationale for
dedupe-first (change from the previous revision): critique #6's
lockstep-editing concern applies to B and C as much as to F–I. Every
branch that edits duplicated files pays the same tax, and paying it
once up front is strictly cheaper than paying it per branch.

**Total focused-work estimate: ~24-26 h across 11 branches**, rescaled
1.5× from the first-draft estimate per critique #4. Real wall-clock
budget is ~35-40 h distributed across sessions, plus the investigation
hours, plus whatever schedule the review/merge cycle imposes. At
2026-04-21 with a May deadline, that's tight but achievable if work
starts within the week. The "decisions needed" items at top must be
resolved before branch D begins, since D's scope depends on them.

### Branch details

**(0) `fix/policy-weight-load`** — already open; merge before proceeding.
Not re-scoped here. Commit `96e3ae2`.

---

**(A) `refactor/deduplicate-root-subdir`** — addresses the structural
tax that critique #6 called out. Per your decision, moved in-scope
and now the first branch after weight-load merges. B and C both edit
files that exist in two copies today; paying the dedupe tax once up
front is strictly cheaper than paying it in every subsequent branch.
- **Scope:** make root the single source of truth for deployment
  code. Specifically:
  - **Delete identical duplicates** of deployment files in
    `drone-race-sim/`: `dcl_adapter.py`, `dcl_mavlink_adapter.py`,
    `vision_model.py`, `track.py`, `race.py`, `requirements.txt`,
    `run.sh`, `launch_viewer.sh`, and the duplicated READMEs
    (`README_SCUBA_LAB.md`, `DCL_INTEGRATION.md`,
    `COMPETITION_STRATEGY.md`).
  - **Move load-bearing files from subdir to root** so the existing
    `sys.path` hack in `drone-race-sim/test_dcl_adapter.py` becomes
    unnecessary:
    - `drone-race-sim/test_dcl_adapter.py` → `./test_dcl_adapter.py`
    - `drone-race-sim/measure_inference_time.py` →
      `./measure_inference_time.py`
    - `drone-race-sim/trained_distilled/policy.pth` →
      `./trained_distilled/policy.pth` (plus sibling files)
    - `drone-race-sim/models_release/aigp_distill_final.zip` →
      `./models_release/aigp_distill_final.zip` (superseding the
      empty `./models_release/` that currently only has
      `MODELS.json` + `README.md`)
  - **Reconcile the diverged files** (`config.py`,
    `drone_race_env.py`): root already has the newer content
    (alignment-bonus reward, `FPV_TILT_DEG = -10`). With A running
    first, the subdir copies simply become deletes; no coordination
    needed with B or C.
  - **Update imports** wherever code referenced subdir files.
- **Touches:** every duplicate under `drone-race-sim/`. The nested
  repo's working tree now has mass deletions; those don't affect the
  outer repo (which tracks `drone-race-sim` as a gitlink), but the
  nested repo's branch gets dirty. Option: commit the deletions in
  the nested repo's `main` and bump the outer repo's gitlink. Simpler
  option: leave deletions uncommitted in the nested repo; the outer
  repo doesn't care.
- **Does not touch:** the nested-git-repo structure itself. We do not
  flatten `drone-race-sim/` or remove its `.git/`. The dir still
  exists; it's just thinner. That's deferred.
- **Effort:** ~1.5-2 h rescaled. Mostly `git mv` / `rm`, plus
  updating a handful of import paths and `sys.path` inserts.
- **Evidence step:** produce an inventory of every duplicate before
  deleting. Run `diff` on each pair to confirm identical (or to
  identify which version survives for the diverged ones). Write
  the inventory into the PR description.
- **Test:** after the dedupe, `./test_dcl_adapter.py` (now at root)
  runs clean **without** any `sys.path.insert`. The existing
  `fix/policy-weight-load` test-suite assertions all pass. A fresh
  `grep -r "from dcl_adapter" --include="*.py"` shows no broken
  imports.
- **Why sequenced here:** first. Every subsequent branch edits files
  that would otherwise exist in two copies. Paying the dedupe cost
  once up front saves the lockstep-editing tax on B, C, F, G, H, I —
  six branches, not four. Done now, ~2 h; done later, fragmentation.

---

**(B) `fix/canonical-model-path`** — addresses B3.
- **Scope:** one canonical path for the distilled vision model. All
  code that reads it, all docs that cite it, and an adapter init-time
  assertion that fails loudly if the model is not at the canonical
  location. A has already moved the model to `./models_release/`;
  this branch updates the advertised-path references in the docs to
  match and adds the init-time assertion.
- **Touches:** `README_SCUBA_LAB.md`, `COMPETITION_CHECKLIST.md`,
  `README_COMPETITION.md`, `dcl_mavlink_client.py` (at root after A).
- **Does not touch:** MAVLink code, duplicate-file structure (A did
  that), any test scaffolding.
- **Effort:** ~0.75-1.5 h (rescaled).
- **Evidence step:** grep the project for every reference to
  `aigp_distill_final.zip` and `AI_GrandPrix_Models`. Write the list
  into the PR description.
- **Test:** `python3 -c "from dcl_adapter import SCUBALabAdapter; SCUBALabAdapter('<canonical path>')"` succeeds; the same call with a
  non-existent path raises a `FileNotFoundError` with a clear
  message (not a generic stack trace from deep in zipfile).
- **Why sequenced here:** after A; single-copy files only. Can run
  in parallel with C — both edit root-only files.

---

**(C) `fix/fpv-tilt-canonicalization`** — addresses B5.
- **Scope:** validate what tilt value the deployed policy actually
  expects, then commit a deliberate value in `config.py` (root, the
  only copy after A) with an import-time assertion.
- **Touches:** `config.py` (root, single copy after A).
- **Does not touch:** MAVLink code.
- **Effort:** ~1-1.5 h rescaled. The edit is trivial; the empirical
  validation is the actual work.
- **Evidence step (required — do not skip):** render an FPV frame
  against a held-out test observation at both tilts (`-10` and `0`).
  Run the trained adapter on each. Compare: (i) the action vectors
  the policy emits, (ii) which rendered image places the forward
  gate more centrally in frame. If tilt `-10` produces actions
  consistent with "fly toward the gate" and tilt `0` produces
  actions with visibly worse action-space sanity (e.g., pitching
  up or veering when the gate is centered), commit `-10`. If the
  result is ambiguous or reversed, the branch commits neither and
  instead documents the uncertainty in [[open-questions]]. A
  "default to -10 because git says so" commit is not acceptable
  without evidence.
- **Test:** (only if evidence supports `-10`) `config.FPV_TILT_DEG
  == -10` as an import-time `assert`. Plus a saved screenshot of
  the tilt-`-10` rendered view with the held-out observation, for
  the record.
- **Why sequenced here:** after A. Can run in parallel with B —
  both edit root-only files. The evidence check pays back the
  investigation time by turning an inferred "fix" into a verified one.

---

**(D) `investigate/mavlink-interface`** — per planning inputs, the
big MAVLink decisions (message choice, library choice) are closed. D
is now a confirmation-and-details branch, not a decision branch.
- **Scope:** research only. No code. Produce one doc at
  `obsidian/decisions-log.md` that records:
  - The exact `type_mask` bit pattern for "use body rates, ignore
    attitude" under pymavlink's common dialect. Should be
    `IGNORE_ROLL_RATE=0 | IGNORE_PITCH_RATE=0 | IGNORE_YAW_RATE=0 |
    IGNORE_THROTTLE=0 | IGNORE_ATTITUDE=128` — confirm from the
    generated message class, don't infer from the spec text.
  - The CRC seed byte for `SET_ATTITUDE_TARGET` (msg_id 82) and
    `HEARTBEAT` (msg_id 0). Pymavlink exposes these via
    `CRC_EXTRAS` or the generated class; record the exact values.
  - The target_system / target_component values DCL's SITL-class
    simulators conventionally use (1/1 is the usual default; we
    currently use 1/1; document why).
  - The `pymavlink` version we pin, plus any dialect module path.
- **Touches:** `obsidian/decisions-log.md` only.
- **Does not touch:** any code.
- **Effort:** ~1.5-3 h rescaled.
- **Evidence step:** for each of the four items above, provide the
  source (pymavlink class inspection, MAVLink message definition
  XML, or VADR-TS-001 section). If a detail can't be pinned,
  document the assumption and the fallback.
- **Test:** none. Doc review by user is the gate.
- **Why sequenced here:** E–H all need these exact values. The
  branch has shrunk because the big decisions (message type,
  library) are now planning inputs, not open questions.

---

**(E) `test/mavlink-mock-sim-harness`** — infrastructure for F/G/H.
- **Scope:** build a minimal UDP + `pymavlink` listener that acts as
  a spec-compliant receiver. Decodes incoming frames; asserts CRC;
  tracks sequence continuity; logs parse errors. Ships as a pytest
  fixture and a standalone script.
- **Touches:** new file at `tests/mock_dcl_receiver.py` and
  `tests/test_mavlink_emission.py`. New top-level `tests/` directory.
- **Does not touch:** production code. Entirely additive.
- **Effort:** ~3-4.5 h rescaled.
- **Evidence step:** none. This is infrastructure.
- **Test:** meta-test — feed the harness a frame built by pymavlink's
  own encoder and confirm our harness decodes it identically. Then
  feed it a frame with a deliberately corrupted CRC and confirm the
  harness rejects it.
- **Why sequenced here:** F/G/H all need this test target. Without
  it, their "tests" would be self-checking (we wrote the encoder AND
  the decoder). With pymavlink in the harness, we test against an
  external reference.

---

**(F) `fix/mavlink-protocol-compliance`** — protocol half of B1.
- **Scope:** replace the handrolled `struct.pack` encoding in
  `dcl_mavlink_adapter.py` with `pymavlink`'s message class. Correct
  CRC seeding, correct payload field ordering, incrementing sequence
  byte, correct `MAV_TYPE` in heartbeat. The semantics (which policy
  output goes in which field) stay wrong in this branch — that's
  the next branch's job.
- **Touches:** `dcl_mavlink_adapter.py` (single copy after A — no
  lockstep tax). Adds `pymavlink` to `requirements.txt`.
- **Does not touch:** `dcl_mavlink_client.py`. Does not change
  semantic mapping of CTBR outputs.
- **Effort:** ~3-4.5 h rescaled.
- **Evidence step:** diff pymavlink's generated `SET_ATTITUDE_TARGET`
  class against our handrolled payload layout. Document every
  divergence before writing new code.
- **Test:** (from E) the mock-sim harness decodes every frame our
  emitter produces without errors; sequence increments by 1 each
  frame; CRC validates.
- **Why sequenced here:** gives subsequent semantic work (G) a
  well-formed frame to place correct values into.

---

**(G) `fix/mavlink-body-rate-semantics`** — semantic half of B1.
- **Scope:** change the mapping so CTBR body rates go to the
  `body_roll_rate` / `body_pitch_rate` / `body_yaw_rate` fields of
  `SET_ATTITUDE_TARGET`, scaled by `MAX_BODY_RATE = 12.0` (not
  `π/4`). Set `type_mask` to ignore the attitude quaternion bit.
  Thrust maps to the thrust field unchanged.
- **Touches:** the emission function(s) in `dcl_mavlink_adapter.py`
  (single copy after A).
- **Does not touch:** encoding/transport (F did that). Does not
  touch the client entry point (H does that).
- **Effort:** ~3-4.5 h rescaled.
- **Evidence step:** write down the exact formula the training policy
  uses: `action[1:4] * MAX_BODY_RATE` → body rates in rad/s. Write
  down what the decoded frame should show for a known input. Only
  then edit.
- **Test:** feed the emitter a known CTBR input `[thrust=0.5,
  roll=0.3, pitch=-0.2, yaw=0.1]` and assert the mock-sim harness
  decodes the frame to `body_roll_rate=3.6`, `body_pitch_rate=-2.4`,
  `body_yaw_rate=1.2`, `thrust=0.5`, quaternion bit ignored in
  `type_mask`.
- **Why sequenced here:** depends on F's well-formed frames and E's
  test harness.

---

**(H) `fix/client-entry-point`** — addresses B2 and B4.
- **Scope:** rewrite `dcl_mavlink_client.py` to use the corrected
  encoder from F+G. Remove `set_actuator_control`. Replace both bare
  `except: pass` blocks with explicit logging at WARN level and a
  per-failure counter. Transport stays on MAVSDK per planning inputs
  (the telemetry path is already wired that way; swapping the transport
  isn't worth the risk with no sim to validate against).
- **Touches:** `dcl_mavlink_client.py` (at root after A).
- **Does not touch:** the encoder (F/G already did). Does not touch
  the vision stream placeholder (B6, post-submission).
- **Effort:** ~3-4.5 h rescaled.
- **Evidence step:** enumerate every function the client calls on
  `self.system` (MAVSDK's System). Note which are telemetry-in vs
  control-out. The control-out path is what we're replacing; the
  telemetry-in path stays.
- **Test:** spin up E's mock-sim as a UDP receiver on a local port;
  run the client control loop for 30 seconds at 50 Hz; assert 1500
  frames received, 0 parse errors, 0 CRC failures, monotonic
  sequence, body-rate values match the policy's outputs (verified by
  running the adapter directly on the same inputs and comparing).
- **Why sequenced here:** needs F+G to have a correct encoder.

---

**(I) `refactor/deprecate-mavlink-adapter-stub`** — addresses B8.
- **Scope:** remove `dcl_mavlink_adapter.py` (single copy at root
  after A), or replace it with a deprecation-shim that raises
  `ImportError` with a pointer to the new path. Update any doc that
  still lists it as a component.
- **Touches:** `dcl_mavlink_adapter.py`, `DCL_INTEGRATION.md`,
  README paragraphs.
- **Does not touch:** code that was already migrated off this file
  in F/G/H.
- **Effort:** ~1.5 h rescaled.
- **Evidence step:** `grep -r "dcl_mavlink_adapter"` across the entire
  repo (excluding venvs). Ensure no live import remains.
- **Test:** `python3 -c "import dcl_mavlink_adapter"` either fails
  with a clear deprecation error, or succeeds and the file is a shim
  containing only `raise ImportError(...)`.
- **Why sequenced here:** after the replacement is in place.

---

**(J) `verify/vq1-readiness-integration`** — final gate.
- **Scope:** runnable script that composes every prior fix and
  exercises the full path end-to-end against the mock sim. Outputs a
  pass/fail record plus decoded-frame statistics. No production-code
  changes.
- **Touches:** `tests/test_vq1_readiness.py` (new). Optional: GitHub
  Actions workflow at `.github/workflows/vq1-readiness.yml`.
- **Does not touch:** production code.
- **Effort:** ~3-4.5 h rescaled.
- **Evidence step:** enumerate every assertion the VQ1-ready
  definition (§1 above) makes. The test is a checklist iteration over
  them.
- **Test:** itself. Pass here means every §1 bullet passes.
- **Why sequenced here:** last, because it tests everything above.

---

**Blocked branches (executable only after DCL simulator release —
per planning inputs, this means post-submission):**

**(K) `fix/vision-stream-integration`** — addresses B6.
- Replace the zero-image placeholder with the real DCL camera
  stream. Effort unknown; bounds depend on DCL's image API format.
- Contains an evidence step: dump three frames from the sim, save
  them, inspect shape/dtype/encoding/range.
- Contains a test: playback the same three frames through the
  adapter, assert actions match a recorded baseline.

**(L) `verify/sim-physics-rate-replay`** — addresses B7.
- Replay a deterministic rollout from our MuJoCo sim (200Hz phys /
  100Hz ctrl) through DCL's 120Hz sim. Measure state divergence.
  Decide whether to retrain/fine-tune based on magnitude.

Per planning inputs, the DCL simulator ships concurrent with VQ1,
so K and L are post-submission.

---

## 4. Fix-vs-defer triage

**Moved in-scope on review (was deferred in first draft):**

- Duplicate-file structure (root vs `drone-race-sim/`) — now fixed by
  branch A `refactor/deduplicate-root-subdir`, which is the first
  branch after weight-load merges. The cost of paying it up front is
  lower than the lockstep-editing tax it would otherwise impose on B,
  C, F, G, H, and I.

**Explicitly deferred to post-VQ1:**

| Item | Why defer | What gets cut if schedule slips |
|---|---|---|
| Nested `drone-race-sim/` git repo | Structural; the dir becomes thin after A.5 but the nested `.git/` stays. Flattening this is its own branch worth ~1-2 h and has no VQ1-relevant payoff. | Keep deferring indefinitely. |
| Curriculum unification | We aren't retraining for VQ1; the trained model is fixed. | Keep deferring indefinitely unless we retrain. |
| CI setup (pytest + ruff in GH Actions) | A local `run_tests.sh` script covers the floor. CI is insurance; absent CI doesn't block submission. | Drop entirely. I-branch can ship as a manual checklist. |
| Event-channel-padding ablation | Unknown impact; the audit flagged it. Validating means running real non-zero events through the adapter, which requires the sim. | Accept as a known sim-to-real gap. |
| Inference timing on target hardware | No access to competition-class embedded hardware. | Document "measured on M4 = 0.26ms; target unknown" in the submission README. |
| Comprehensive test coverage beyond smoke | Shape check + mock-sim roundtrip is sufficient floor. Deterministic rollout diffs against recorded trajectories are better but can ship later. | Drop. |
| Documentation consolidation | Seven top-level READMEs still exist; the wiki is their replacement but they haven't been deleted. Harmless until someone reads them. | Drop. Add a one-liner to each stale README pointing at the wiki. |

**If schedule slips harder, cut in this order:**
1. I (`deprecate-mavlink-adapter-stub`) — the deprecated file is
   tolerable if no live code imports it.
2. J (integration verify) — replace with a written checklist the
   user runs manually.
3. C (FPV tilt) — assuming critique #3's empirical check comes out
   inconclusive, this branch becomes just a comment, not an edit.
4. B (canonical model path) — if we'd rather fix the docs only,
   this drops to a doc-patch commit with no code.

**Do not cut, at any schedule pressure:**
- A (dedupe) — every MAVLink branch pays a lockstep tax otherwise;
  cutting A means paying that tax six times over.
- D (mavlink investigation) — without this, every subsequent
  branch is guessing at type_mask / CRC seed / dialect details.
- F (protocol compliance) — submission fails without it.
- G (body-rate semantics) — submission flies wrong without it.
- H (client entry point) — the code path that actually sends bytes.
- E (mock-sim harness) — F/G/H can't be tested meaningfully without
  it.

---

## 5. Status of the first-draft open questions

The first draft had eight open questions (Q1-Q8) that needed answers
before the plan could be committed. User review closed all of them:

- **Q1 (sim availability), Q2 (which MAVLink message), Q3 (pymavlink
  vs mavsdk), Q4 (Python 3.14), Q5 (submission format), Q7 (hardware
  phase), Q8 (gate tolerance)** — all resolved as **planning inputs**
  at the top of this doc.
- **Q6 (is `fix/policy-weight-load` merged)** — practical
  precondition, noted in the status anchor and planning inputs.

The two remaining items that still need your call are captured in
[[vq1-execution-plan#Decisions needed from user]] at the top. Those
aren't open questions in the "blocks planning" sense; they're
tradeoffs where I declined to pick unilaterally.

---

## 6. Red-team pass

Honest critiques of the plan above. I tried to find real problems,
not perform self-criticism. Any one of these would collapse part
or all of the plan if true.

### Critique 1 — The investigation branch (C) has no success criterion and may terminate in "we don't know"

Branch C's goal is to decide Q2 and Q3. If VADR-TS-001 is ambiguous
(plausible — the audit already found it lists two acceptable
messages without priority) and the DCL simulator is unavailable
(current state as of 2026-04-21), C's honest output is: "could not
determine; defer until sim ships." That output does not unblock
D-G. The plan currently treats C as a forcing function, but it's
only effective if C has information to work with.

**Mitigation options:** (a) implement both `SET_ATTITUDE_TARGET` and
`SET_POSITION_TARGET_LOCAL_NED` paths behind a runtime toggle,
doubling E/F/G effort; (b) ship with `SET_ATTITUDE_TARGET` (our
best guess) and accept that the submission may need a one-line
toggle change if DCL rejects it; (c) wait for the simulator and
accept slipping past the deadline.

I'd lean (b), but the plan needs to name a default fallback
explicitly. Currently it doesn't.

### Critique 2 — "Spec-compliant but sim-unvalidated" is a weak success definition

The VQ1-ready definition (§1) says a `pymavlink` parser can decode
every frame. But `pymavlink` is Anthropic-level generic; it will
accept messages that are syntactically valid under the MAVLink v2
spec regardless of whether the DCL sim's parser accepts them. The
sim might require specific `target_system`/`target_component`
values, a specific stream rate, a TIMESYNC handshake before
accepting commands, a particular MAV_COMPONENT for the source
system — none of which a generic decoder checks.

The plan's bar of "pymavlink roundtrips clean" is the floor, not
the ceiling. Without access to the real simulator, we cannot lift
the floor. This isn't fixable within the plan; it's an unavoidable
limitation. But the plan should be explicit that passing I
(integration verify) does not mean "VQ1-accepted" — it means "spec-
floor achieved".

### Critique 3 — The FPV_TILT_DEG fix rests on inference, not measurement

The audit concluded the deployed model was trained with `-10` based
on git history: the root config had `-10` committed, the local Mac
subdir edit to `0` was never pushed, and the HPC clone would have
had `-10` at training time. This is circumstantial. If the HPC
clone was manually edited on the HPC to match the local Mac (or if
someone ran a different revision during the actual distill
training — recall the audit found no slurm script for distill),
the model could have been trained against tilt `0`. In that case,
the branch-B "fix" — asserting tilt must be `-10` — would actively
break deployment.

**Mitigation:** branch B's evidence step must be empirical, not
inference-based. Render the FPV view at both tilts. Run the
adapter against both. Check which produces saner actions on a
hand-built test observation that includes a visible gate. If
neither produces a clear signal, the branch becomes "document the
ambiguity, pick one, flag as known risk".

The plan lists this as an evidence step already, but doesn't
commit to the consequence of "inconclusive". It should.

### Critique 4 — Effort estimates ignore the cost of integration debugging

Every branch is quoted at its optimistic cost — the "I know
exactly what to do, write the code, tests pass" duration. Real
software engineering sees 1.5-3× on top of that from: environment
setup ("pymavlink isn't in venv312, let me pip install... oh,
conflict with numpy..."), subtle spec misreads (CRC seed table
has per-message-ID entries; picking the wrong seed is a 10-minute
debug that feels like an hour), Python version mismatches,
rebuilt-from-wrong-branch rollbacks, and the mid-session cwd-drift
pattern that tripped up the weight-load branch.

The plan totals ~16-17 hours of focused work. Honest wall-clock
budget before submission is probably **25-35 hours**, distributed
across multiple sessions because focus isn't linear. At 2026-04-21
with a May deadline, we have a few weeks. That's enough — but only
if we start soon and don't hit Q2's worst case (both message types
needed).

### Critique 5 — The plan has a single-point-of-failure on branch E

Every subsequent MAVLink branch (F, G, H, I) depends on E landing
correctly. If E introduces a subtle encoding bug — say, the wrong
endianness on a float, or a quaternion packed in XYZW order when
the spec wants WXYZ — F and G will look correct at the code level
while emitting garbage bytes. D's mock-sim harness would catch this
if (and only if) pymavlink's decoder is strict about encoding
variations; historically MAVLink libraries have been lenient.

**Mitigation:** E's test should include a byte-level fixture
comparison, not just "pymavlink decodes this OK". Capture the
expected bytes for a fixed input using pymavlink's own encoder and
assert E's output equals those bytes exactly. Otherwise we're
testing "pymavlink agrees with itself" which is vacuous.

The plan's test description for E currently says "mock-sim
harness decodes every frame without errors". That's weaker than
what we need. Strengthen.

### Critique 6 — The deferred "duplicate file structure" tax accumulates silently

Every MAVLink branch (E, F, H) must edit both `root/*.py` and
`drone-race-sim/*.py` in lockstep. That's fine once. Done seven
times across the plan's branches, it's 7× the chance of a
lockstep miss (one copy edited, the other forgotten). The
weight-load branch already ran this tax — and found that the
duplicate-source issue was real and non-trivial to manage.

**Mitigation option:** a brief deduplication branch before E. Replace
the `drone-race-sim/` copies with symlinks or delete them and make
anything in `drone-race-sim/` that needs them import from root via
`sys.path`. This is a ~1 hour branch that pays for itself within
two subsequent edits.

The plan currently has this at "deferred". Reconsider: moving it
up to before D might be net positive, not net negative.

### What would collapse the whole plan

Of the critiques, the most dangerous is **Critique 1 + Q2**: if
C terminates in "we don't know" AND the DCL simulator doesn't ship
in time, the plan has no way to converge on a correct MAVLink fix
without implementing both message types behind a toggle. That
doubles E/F/G effort and may push past May. The single decision
that most changes the plan's viability is whether DCL ships the
simulator before mid-May or not.

---

## Next actions

None are to be taken in this session. On your review:

1. Answer (or acknowledge uncertainty on) Q1-Q8.
2. Confirm or correct the VQ1-ready definition in §1.
3. Approve, revise, or reject the branch sequence in §3.
4. Rule on the fix-vs-defer triage in §4.
5. Respond to the red-team critiques — especially whether Critique 6
   (dedupe before E) should change the sequence.

Each approved branch becomes its own prompt, authored with the same
discipline as `fix/policy-weight-load`: evidence step first, tests
that distinguish loaded from default, no silent fallbacks, clear
scope boundary.

## See also

- [[fragilities]] — the long-form diagnostic for each blocker.
- [[submission-readiness]] — the operational checklist this plan
  sequences.
- [[open-questions]] — items flagged as unresolved during the audit.
- [[competition]] — the DCL spec references the plan assumes.
