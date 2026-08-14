# analysis/

- `flight_report.py` — one tick-log CSV in, one verdict line + JSON record out.
  The batch system's classification contract. Measures the *true* Gate-3
  crossing (deepest stable frame before sustained detection loss), flags dirty
  starts and rate anomalies. `--json`, `--descent-size`, `--log` options.
- `track.py`, `trajectory_viz.py`, `live_trajectory.py`, `trajectory_logger.py` —
  course-frame trajectory tooling.
- `summarize_attempts.py` — campaign-level aggregation.

Try it on shipped data:
`python analysis/flight_report.py results/overnight-2026-08-03/top10_traces/filt_auto_329.csv`
