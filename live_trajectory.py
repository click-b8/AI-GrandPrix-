"""
VQ1 Live Trajectory Dashboard — debugging tool, not part of submission.

Polls trajectory_log.csv while a flight is in progress and refreshes
a matplotlib figure every N seconds. Shows the path growing in real time.

Usage (start before or during a flight):
    python3 live_trajectory.py [--input trajectory_log.csv] [--interval 1.0]
                                [--save-frames frames/]

    --save-frames DIR   Instead of displaying an interactive window, save a
                        numbered PNG into DIR on each update. Useful on
                        headless machines or inside a remote session.

Requires a display unless --save-frames is used.
Press Ctrl+C to exit.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# CSV reader (robust — handles partial writes mid-flight)
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> dict | None:
    """Return dict of numpy arrays or None if file missing/empty/unreadable."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if all(r.values())]
        if not rows:
            return None
        return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Gate loader (optional)
# ---------------------------------------------------------------------------

def _load_gates():
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from track import get_gate_positions
        return get_gate_positions()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Plot update helper
# ---------------------------------------------------------------------------

def _update_axes(axes, data: dict, gates) -> None:
    """Redraw all axes with the latest data."""
    t     = data["time_s"]
    x     = data["x"]
    y     = data["y"]
    z     = data["z"]
    vx    = data["vx"]
    vy    = data["vy"]

    alt   = -z
    speed = np.sqrt(vx**2 + vy**2)
    n     = len(t)
    dur   = float(t.max() - t.min()) if n > 1 else 0.0

    ax_td, ax_alt, ax_spd = axes

    for ax in axes:
        ax.cla()

    # Top-down
    ax_td.set_facecolor("#1a1a2e")
    if n > 1:
        from matplotlib.collections import LineCollection
        import matplotlib.colors as mcolors

        vmax = float(speed.max()) if speed.max() > 0 else 1.0
        norm = mcolors.Normalize(0, vmax)
        pts  = np.array([y, x]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc   = LineCollection(segs, cmap="plasma", norm=norm, linewidth=2)
        lc.set_array(speed[:-1])
        ax_td.add_collection(lc)
        ax_td.autoscale()

    ax_td.plot(y[0],  x[0],  "o", color="lime", ms=8, zorder=5)
    ax_td.plot(y[-1], x[-1], "s", color="red",  ms=8, zorder=5)
    if gates is not None:
        ax_td.scatter(gates[:, 1], gates[:, 0], c="gold", marker="D",
                      s=80, zorder=6, label="gate")
    ax_td.set_xlabel("East — Y (m)", color="white")
    ax_td.set_ylabel("North — X (m)", color="white")
    ax_td.set_title(f"Top-Down  [{n} pts, {dur:.0f}s]", color="white")
    ax_td.set_aspect("equal", adjustable="datalim")
    ax_td.tick_params(colors="white")
    ax_td.grid(True, alpha=0.2, color="white")

    # Altitude
    ax_alt.plot(t, alt, color="deepskyblue", lw=1.5)
    if gates is not None:
        for i, gz in enumerate(gates[:, 2]):
            ax_alt.axhline(gz, color="gold", lw=0.5, ls="--", alpha=0.6)
    ax_alt.set_xlabel("Time (s)"); ax_alt.set_ylabel("Altitude (m)")
    ax_alt.set_title(f"Altitude  (max {alt.max():.1f} m)")
    ax_alt.grid(True, alpha=0.3)

    # Speed
    ax_spd.plot(t, speed, color="tomato", lw=1.5)
    if speed.mean() > 0:
        ax_spd.axhline(speed.mean(), color="orange", lw=1, ls="--",
                       label=f"mean {speed.mean():.1f}")
        ax_spd.legend(fontsize=8)
    ax_spd.set_xlabel("Time (s)"); ax_spd.set_ylabel("m/s")
    ax_spd.set_title(f"Ground Speed  (max {speed.max():.1f} m/s)")
    ax_spd.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VQ1 Live Trajectory Dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",       default="trajectory_log.csv")
    parser.add_argument("--interval",    type=float, default=1.0,
                        help="Refresh interval in seconds")
    parser.add_argument("--save-frames", default=None, metavar="DIR",
                        help="Save numbered PNGs here instead of showing a window")
    args = parser.parse_args()

    save_mode = args.save_frames is not None
    if save_mode:
        os.makedirs(args.save_frames, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    if not save_mode:
        plt.ion()

    fig, axes_arr = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0d0d1a")
    for ax in axes_arr:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    gates      = _load_gates()
    last_mtime = 0.0
    frame_idx  = 0
    waiting    = True

    print(f"[live] Watching {args.input!r} — interval {args.interval}s")
    if save_mode:
        print(f"[live] Saving frames to {args.save_frames!r}")
    else:
        print("[live] Press Ctrl+C to exit.")

    try:
        while True:
            mtime = os.path.getmtime(args.input) if os.path.exists(args.input) else 0.0

            if mtime != last_mtime:
                data = _read_csv(args.input)
                if data and len(data["time_s"]) > 1:
                    if waiting:
                        print("[live] Data arriving — dashboard active.")
                        waiting = False
                    try:
                        _update_axes(axes_arr, data, gates)
                        n   = len(data["time_s"])
                        dur = float(data["time_s"].max() - data["time_s"].min())
                        fig.suptitle(
                            f"LIVE  |  {n} pts  |  {dur:.0f}s",
                            color="white", fontsize=12,
                        )
                        plt.tight_layout(rect=[0, 0, 1, 0.95])

                        if save_mode:
                            path = os.path.join(args.save_frames,
                                                f"frame_{frame_idx:05d}.png")
                            plt.savefig(path, dpi=100, facecolor="#0d0d1a")
                            frame_idx += 1
                            print(f"[live] Frame {frame_idx} saved ({n} pts)")
                        else:
                            fig.canvas.draw()
                            fig.canvas.flush_events()
                    except Exception as exc:
                        print(f"[live] Plot error (will retry): {exc}")

                    last_mtime = mtime

            elif waiting:
                print(f"[live] Waiting for {args.input!r} ...", end="\r", flush=True)

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[live] Stopped.")
    finally:
        if not save_mode:
            plt.ioff()
            plt.close(fig)


if __name__ == "__main__":
    main()
