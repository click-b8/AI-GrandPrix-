# Automation — the unattended qualification batch

`automation/run_qualifier_batch.ps1` (~1,200 lines, PowerShell 5.1) converts one manual flight into an unattended overnight campaign. It ended up being some of the hardest engineering in the project, because everything it automates is adversarial: a GUI simulator with an online login, a race countdown, exclusive-fullscreen input, and a flier that must be armed *before* the race starts.

## Simulator process lifecycle

The visible `FlightSim.exe` is a 167 KB **launcher**; the real simulator is `DCGame-Win64-Shipping.exe` (~92 MB), which owns both UDP sockets (14560 MAVLink, 5601 vision). The batch:

- discovers the shipping process by **PID-set differencing** around launch (exactly one new process, or refuse);
- terminates only the stored PID on restart, detects orphans/replacements, and verifies port release;
- asserts **pre-flight isolation** before every flight: exactly one shipping process, it is ours, and it is the *sole* owner of UDP 14560 — because two live sims merge their MAVLink streams into one indistinguishable source tuple. Any violation is `BROKEN_RESTART`: the batch stops rather than fly contaminated.

This fix retired a whole class of "mystery variance": stray sim instances silently merging vision streams.

## The race-reset problem

The sim needs a human to log in (online account; deliberately **not** automated — brittle and credential-sensitive) and to reset each run via the pause menu. The final design: log in **once**, keep the sim alive all night, and automate only the reset — `Esc`, `Down`, `Enter` (RESTART) sent via Win32 **`SendInput` with scancodes** (games ignore `SendKeys`; `Down` needs `KEYEVENTF_EXTENDEDKEY` or some builds read numpad-2).

Ordering is everything: the flier connects → arms → *waits for GO*; the countdown only counts a listener in. So each attempt is **Esc (pause holds the countdown) → launch flier → settle → RESTART → GO with an armed listener → fly → exit**. Attempt 1 is the same cycle. Two silent killers found on the way: PowerShell's `Start-Process -ArgumentList` does not quote arguments (a space in the repo path killed every launch — the flier died at argparse before ever arming), and Python block-buffers redirected stdout (the "waiting for GO" signal needed `-u`).

## Classification, stop conditions, disk

Every run: tick CSV + console log → `analysis/flight_report.py --json` → one classification (`FULL_COURSE`, `POST_G3_FAILURE`, `KEEP_DEEP`, `KEEP_UPSTREAM`, `KEEP_SHALLOW`, `DISCARD_RATE`, `DISCARD_DIRTY`, `BROKEN`, `BROKEN_RESTART`) → a row in `batch_summary.csv` (written *before* any purge, so the record is complete).

- **The only success stop is the sim's own race-finish signal** (`race_fin > 0`) — Gate 3 registering is a milestone, printed and continued, never a stop.
- Failure stops: `MaxAttempts` (default 1000) or **SIM LOST** (shipping PID gone / foreign port owner — re-login is human-only, so the batch stops loudly instead of burning attempts on a dead window).
- **Self-purging disk:** pure discards that never reached the Gate-3 leg are deleted after their summary row and JSON are written; anything deep, post-G3, or unreadable is kept. Uncertainty always resolves to *keep*.
- Keep-awake (`SetThreadExecutionState`), Ctrl-C-safe tally in `finally`, and a per-run lifecycle log (PIDs, ports, forced-kill flags) complete the unattended posture.

## What it achieved

The final overnight run: **546 attempts, 7 hours, zero operator interventions, zero runaway cycles, zero stray processes** — median 44 s per attempt including reset. Every attempt classified, every byte of evidence either kept or accounted for.
