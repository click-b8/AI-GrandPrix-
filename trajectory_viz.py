"""
VQ1 Trajectory Visualizer — debugging tool, not part of submission.

Reads trajectory_log.csv produced by TrajectoryLogger and saves trajectory_plot.png.

Usage:
    python3 trajectory_viz.py [--input trajectory_log.csv] [--output trajectory_plot.png]

Panels:
    1. Top-down 2D path  — X (North) vs Y (East) in NED frame
    2. Altitude over time — -Z (NED z is down, altitude is positive up)
    3. Ground speed over time — sqrt(vx² + vy²) m/s
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on headless/Windows
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: str) -> dict:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        print(f"[viz] {path} is empty — nothing to plot.")
        return {}
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description="VQ1 Trajectory Visualizer")
    parser.add_argument("--input",  default="trajectory_log.csv",
                        help="Input CSV from TrajectoryLogger")
    parser.add_argument("--output", default="trajectory_plot.png",
                        help="Output PNG path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[viz] ERROR: {args.input} not found. "
              "Run with --log-trajectory first.")
        sys.exit(1)

    data = load_csv(args.input)
    if not data:
        sys.exit(1)

    t   = data["time_s"]
    x   = data["x"]       # North  (m NED)
    y   = data["y"]       # East   (m NED)
    z   = data["z"]       # Down   (m NED)
    vx  = data["vx"]
    vy  = data["vy"]

    altitude     = -z                           # positive up
    ground_speed = np.sqrt(vx**2 + vy**2)      # horizontal m/s

    n_samples   = len(t)
    duration    = float(t.max() - t.min()) if n_samples > 1 else 0.0
    max_speed   = float(ground_speed.max()) if n_samples > 0 else 0.0
    max_alt     = float(altitude.max())     if n_samples > 0 else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"VQ1 Flight Trajectory  |  {n_samples} samples  |  {duration:.1f}s",
        fontsize=13,
    )

    # --- Panel 1: top-down path (NED: x=North, y=East) ---
    ax = axes[0]
    ax.plot(y, x, color="steelblue", linewidth=1.5, label="path")
    if n_samples > 0:
        ax.plot(y[0],  x[0],  "go", markersize=8, label="start")
        ax.plot(y[-1], x[-1], "rs", markersize=8, label="end")
    ax.set_xlabel("Y — East (m)")
    ax.set_ylabel("X — North (m)")
    ax.set_title("Top-Down Path (NED)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: altitude over time ---
    ax = axes[1]
    ax.plot(t, altitude, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(f"Altitude  (max {max_alt:.1f} m)")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: ground speed over time ---
    ax = axes[2]
    ax.plot(t, ground_speed, color="tomato", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ground Speed (m/s)")
    ax.set_title(f"Ground Speed  (max {max_speed:.1f} m/s)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(
        f"[viz] Saved {args.output} "
        f"({n_samples} samples, {duration:.1f}s, "
        f"max alt {max_alt:.1f}m, max speed {max_speed:.1f}m/s)"
    )


if __name__ == "__main__":
    main()
