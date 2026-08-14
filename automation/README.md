# automation/

`run_qualifier_batch.ps1` — the unattended campaign system (PowerShell 5.1).
Owns the simulator process lifecycle (PID-tracked shipping binary, UDP port
isolation), the in-game race reset (Win32 SendInput keystrokes with
arm-before-GO sequencing), per-attempt classification via
`analysis/flight_report.py --json`, self-purging disk management, and
stop conditions (course-complete signal, MaxAttempts, or SIM LOST).

Usage and environment assumptions: `../results/overnight-2026-08-03/REPRODUCE.md`.
Design details: `../docs/automation.md`. `-RepoPath` / `-SimExe` are parameters —
set them for your machine.
