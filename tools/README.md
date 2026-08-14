# tools/

- `schedule_flier.py` — **the flight controller** (`--coast-tube` mode). See
  `docs/controller.md` for the mechanism guide and `docs/flags.md` for the
  flag reference; the frozen competition command is in
  `results/overnight-2026-08-03/REPRODUCE.md`.
- `probe_*.py`, `imu_check.py`, `dump_race_stream.py`, `capture_encdata.py`,
  `tube_rail_check.py` — simulator-facing probes and diagnostics from the
  calibration era (require a live simulator).
- `guardian.py`, `grind.py`, `tune_openloop.py`, `motion_schedule.py`,
  `powered_derisk.py` — campaign utilities.
