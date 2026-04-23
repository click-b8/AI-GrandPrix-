#process

# Session prompt template

Use this for every prompt that instructs an agent (Claude Code or any
other) to execute a branch of work on this project. The template
exists because "follow the plan" without specifics produced one
plan-structure violation (three separate plan branches collapsed into
one, dedupe merged in retroactively; see
[[decisions-log#2026-04-23 — Accepted one-time plan-structure deviation]]).
Vague prompts produced correct outcomes by accident. This template
removes the accident.

## The rule

**If any of the Required sections below is blank or non-specific, the
prompt is not ready to send.** Vague instructions like "follow the
plan" or "implement per the audit" are explicitly disallowed.

## Template

Copy this block and fill in every Required field. Do not delete
fields you don't need — blank them with `— n/a, because X —` so the
reader can see the prompt was considered in full.

```
# Branch: <fix|refactor|test|verify|investigate|docs>/<kebab-case-name>

## Context
<Required. 2-5 sentences that stand alone. Why this branch exists,
what audit finding or plan reference it traces to, what the outcome
looks like when done. A reader who hasn't seen prior sessions must
be able to proceed from here without reading backward.>

Anchor refs:
- obsidian/vq1-execution-plan.md §<section> (if applicable)
- obsidian/fragilities.md §<entry> (if applicable)
- obsidian/decisions-log.md (if a decision is being applied)

## Scope

<Required. Numbered list. Each item states exactly what gets touched.
Name the files, the functions, the behavior change.

FORBIDDEN scope phrasings (if you find yourself writing any of these,
stop and rewrite the scope with specifics):
- "Follow the plan for branch X"
- "Execute the VQ1 plan"
- "Follow the execution-plan §3"
- "Implement per the audit"
- "Do what fix/Y did but for Z"
- "Plan branches F + G" (collapsing multiple named branches is itself
  a scope violation — see
  [[decisions-log#2026-04-23 — Accepted one-time plan-structure deviation]])

If the scope exceeds ~4 hours of focused work, split it into multiple
branches rather than bundling. If splitting is unclear, STOP and ask
the user to revise the plan before starting the branch.>

1. <Specific change. Include file names.>
2. <Specific change. Include file names.>
...

## Guardrails (non-negotiable)

- No `strict=False` in any `load_state_dict` call, no `try/except`
  around a weight load, no silent fallback that substitutes defaults
  for missing data.
- No bare `except: pass`. Log + count + re-raise or return a typed
  result; observability first.
- Stubs for blocked dependencies (vision stream, real sim, etc.)
  must raise `NotImplementedError` by default and require an
  explicit flag / env var / code edit to bypass. Log-only warnings
  are not sufficient.
- No structural "helpful" reorganization beyond the Scope above. If
  new problems surface, add them to obsidian/future-work.md or
  obsidian/open-questions.md — do not fix them here.
- Preserve git history with `git mv` wherever source + destination
  are in the same repo. Cross-repo moves (e.g., subdir ↔ root
  when the subdir is a nested repo) can't preserve history; accept
  that and note it in the commit message.

## Evidence step (required — STOP and report)

<Required. This is the single most load-bearing moment of the
branch. Specify exactly what evidence must be produced and shown
to the user BEFORE any destructive or irreversible action. The
evidence must be enough for the user to confirm that the scope
premise holds. If the evidence contradicts the premise, the
session stops, no commits are made, and the user decides what
to do.>

Concrete requirements:
- <Specific artifact: e.g., "inventory of every duplicate file with
  diff size", "output of `grep -r X`", "dry-run of Y".>
- <Checkpoint statement: "STOP here and wait for user approval
  before proceeding to scope item N.">

The prompt is not allowed to proceed past the evidence step on its
own judgment, even in auto mode. Auto mode authorizes the agent to
keep moving on low-risk work without asking; it does not authorize
it to skip a checkpoint that was explicitly put in the plan.

## Tests (required — must pass before opening PR)

<Required. Numbered list. Each test must: (a) be executable by a
single `python3` or `pytest` invocation from the project root;
(b) assert something specific that would catch a plausible
regression of the bug this branch fixes; (c) distinguish the loaded/
real state from the init/stub state when the fix is about silent
failure. A test that "adapter loads without exception" is
insufficient — the prior adapter loaded without exception while
loading zero weights.>

1. <Specific test, with expected-result criterion.>
2. <Specific test, with expected-result criterion.>
...

## Out of scope (explicitly)

<Required. List things that might look related but are NOT this
branch's job. Typical entries:>

- MAVLink adapter fixes (F/G/H branches).
- Duplicate-file structural reconciliation (branch A).
- FPV_TILT_DEG empirical validation (branch C).
- Any training code changes.
- Any change to obsidian/ files beyond what's needed to mark the
  branch's own fragility entries resolved.

## Commit identity

Commit and author with the identity that `git config user.name` and
`git config user.email` return from this machine. If either returns
empty, STOP and configure before running the branch. Do not fall
back to any other identity.

Runnable check before the first commit of the branch:

```bash
test -n "$(git config user.name)" && test -n "$(git config user.email)" \
  || { echo "STOP: git identity unset"; exit 1; }
```

**Do not use:**
- Co-Authored-By trailers.
- Non-person identities ("SCUBA Lab <scubalab@vq1.local>",
  "Bot <bot@...>", etc.). See
  [[decisions-log#2026-04-23 — Author identity uniformity]].
- `--amend` on any commit that has been pushed.
- `--no-verify` or any flag that skips commit hooks.
- Git's implicit hostname-derived fallback on a new machine. If
  `git config` is empty, configure explicitly — do not rely on
  `user@host.local` being "close enough".

## Deliverables (required)

1. Evidence checkpoint output (from the Evidence step above). Wait
   for user sign-off.
2. After sign-off: the Scope work.
3. Test run output: every pass/fail reported, no skips without a
   documented reason.
4. Wiki updates: mark resolved fragilities with date + commit hash,
   add new fragilities if any surfaced, update
   submission-readiness.md checkboxes that the branch affects.
5. Branch pushed to origin. NO merge to main.
6. Short closing summary: what's clean now, what's still fragile,
   anything that surfaced mid-work that should be logged in
   future-work.md or open-questions.md.

## Do NOT

- Merge to main. Merge happens only after explicit user approval,
  never as part of the branch's own completion.
- Force-push.
- Rebase a branch that has been pushed, unless explicitly authorized
  for an author-identity rewrite (and show the rebase command
  before executing).
- Retry any history-modifying command without explicit user
  re-authorization, even when the retry is technically idempotent
  under `--force-with-lease` or similar safety mechanisms. Transient
  failures are exactly when a second pair of eyes is cheap insurance.
  When the user authorizes a retry, they should specify the
  diagnostic evidence required (e.g., `ssh-add -l`, `git fetch origin`,
  `ssh -T git@host`) that would confirm the failure's cause before
  retrying.
- Create branches other than the named branch. If the scope truly
  requires splitting, STOP and ask the user to revise the plan.
  Collapsing multiple plan-branches into one, spawning side
  branches "to keep things tidy", or merging another plan-branch
  into this one retroactively, are all scope violations.
- Touch the nested `drone-race-sim/` git repo except for the narrow
  action this branch is explicitly scoped to take there. See
  [[fragilities#Nested repo shares the outer repos GitHub remote]].
- Install packages into the venv beyond what's in `requirements.txt`.
  If a new dependency is required, add it to `requirements.txt` as
  part of the branch's scope.
- Assume a decision that wasn't explicitly made. If the prompt
  doesn't specify an answer to a question the branch raises, STOP
  and ask.
```

## How to use this template

1. **Author fills every Required section** before sending to an
   agent. The template is the instrument of discipline; it works
   only if not skimped on.
2. **Agent treats each section as binding.** The Evidence step is a
   hard stop: auto mode doesn't override it. The Guardrails are
   non-negotiable; if the scope can't be accomplished without
   violating one, the agent stops and reports.
3. **Failure mode**: if the agent produces a branch that deviates
   from the template in any material way (wrong identity, skipped
   evidence step, silent fallback introduced, scope creep), the
   branch does not merge. The work may be technically correct
   (the MAVLink rewrite was), but the process failure is itself
   the thing being guarded against.

## Why this exists

The project's entire technical-quality thesis is that silent
failures compose into confident lies (see
[[fragilities#Silent failure chain]]). Vague prompts are a
category of silent failure in process: they produce output that
looks correct, that passes tests, that matches the plan's intent
— and whose process deviations only surface under audit. A prompt
that skips the evidence checkpoint and gets away with it is
indistinguishable from a prompt that skipped it and got away with
it by accident. The template makes deviation visible.

## See also

- [[vq1-execution-plan]] — the plan this template operationalizes.
- [[fragilities]] — the audit findings that motivated the
  discipline this template enforces.
- [[decisions-log]] — where process decisions like this one are
  recorded.
