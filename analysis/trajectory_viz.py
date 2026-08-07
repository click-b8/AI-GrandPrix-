"""
VQ1 Trajectory Visualizer — debugging tool, not part of submission.

Reads trajectory_log.csv and produces:
  trajectory_plot.png  — 4-panel static figure (matplotlib)
  trajectory_plot.html — interactive figure (plotly)

Usage:
    python3 trajectory_viz.py [--input trajectory_log.csv]
                               [--png trajectory_plot.png]
                               [--html trajectory_plot.html]
                               [--no-html] [--no-gates]

Panels:
    1. 3D perspective path, colored by ground speed
    2. Top-down (NED: x=North, y=East) with speed color and direction arrows
    3. Altitude (-z) over time
    4. Ground speed over time with mean/max annotations

Gate positions are loaded from track.py (8-gate course) when available.
"""

import argparse
import csv
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: str) -> dict:
    """Return dict of numpy arrays keyed by CSV column name."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def load_gates() -> np.ndarray | None:
    """
    Try to load gate positions from track.py.
    Returns (N, 3) array of (x_north, y_east, altitude_m) or None.
    track.py uses z=altitude (positive up), matching our altitude convention.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from track import get_gate_positions
        gates = get_gate_positions()  # (N, 3): x, y, z_alt
        return gates
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

SPEED_CMAP = "plasma"


def _norm(values: np.ndarray) -> mcolors.Normalize:
    """Normalizer clamped to [0, max] so stub zeros still render."""
    vmax = float(values.max()) if values.max() > 0 else 1.0
    return mcolors.Normalize(vmin=0.0, vmax=vmax)


def _colored_line_2d(ax, x, y, values, cmap=SPEED_CMAP, lw=2.0):
    """Draw a 2-D line on ax colored per segment by values."""
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = _norm(values)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=lw)
    lc.set_array(values[:-1])
    ax.add_collection(lc)
    return lc, norm


def _colored_line_3d(ax, x, y, z, values, cmap=SPEED_CMAP, lw=1.5):
    """Draw a 3-D line on ax colored per segment by values."""
    pts = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = _norm(values)
    lc = Line3DCollection(segs, cmap=cmap, norm=norm, linewidth=lw)
    lc.set_array(values[:-1])
    ax.add_collection3d(lc)
    return lc, norm


def _colorbar(fig, ax, lc, norm, label="Speed (m/s)"):
    sm = cm.ScalarMappable(cmap=SPEED_CMAP, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=label, fraction=0.03, pad=0.04, shrink=0.6, aspect=20)


# ---------------------------------------------------------------------------
# Matplotlib static PNG (4 panels)
# ---------------------------------------------------------------------------

def make_png(data: dict, gates, output_path: str) -> None:
    t    = data["time_s"]
    x    = data["x"]          # North
    y    = data["y"]          # East
    z    = data["z"]          # NED down
    vx   = data["vx"]
    vy   = data["vy"]
    vz   = data["vz"]
    roll = data["roll"]
    pit  = data["pitch"]
    yaw  = data["yaw"]

    alt   = -z
    speed = np.sqrt(vx**2 + vy**2)
    n     = len(t)
    dur   = float(t.max() - t.min()) if n > 1 else 0.0
    is_stub = (speed.max() == 0.0 and alt.max() == 0.0)

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        f"VQ1 Flight Trajectory  |  {n} samples  |  {dur:.1f}s flight time"
        + ("  [STUB — all zeros]" if is_stub else ""),
        fontsize=13, y=0.98,
    )

    from matplotlib.gridspec import GridSpec
    gs     = GridSpec(2, 2, figure=fig, height_ratios=[1.8, 1])
    ax3d   = fig.add_subplot(gs[0, 0], projection="3d")
    ax_td  = fig.add_subplot(gs[0, 1])
    ax_alt = fig.add_subplot(gs[1, 0])
    ax_spd = fig.add_subplot(gs[1, 1])

    # --- 3-D perspective ---
    if n > 1:
        lc3, norm3 = _colored_line_3d(ax3d, y, x, alt, speed)
        _colorbar(fig, ax3d, lc3, norm3)
    ax3d.scatter(y[:1],  x[:1],  alt[:1],  c="lime",  s=60, zorder=5, label="start")
    ax3d.scatter(y[-1:], x[-1:], alt[-1:], c="red",   s=60, zorder=5, label="end")
    if gates is not None:
        gx, gy, gz = gates[:, 0], gates[:, 1], gates[:, 2]
        ax3d.scatter(gy, gx, gz, c="gold", marker="D", s=80, zorder=6, label="gate")
        for i, (gxi, gyi, gzi) in enumerate(zip(gx, gy, gz)):
            ax3d.text(gyi, gxi, gzi + 0.3, str(i), fontsize=7, color="gold", ha="center")
    ax3d.set_xlabel("East (m)"); ax3d.set_ylabel("North (m)"); ax3d.set_zlabel("Alt (m)")
    ax3d.set_title("3-D Path (speed color)")
    ax3d.legend(fontsize=7, loc="upper left")

    # --- Top-down 2D ---
    if n > 1:
        lc2, norm2 = _colored_line_2d(ax_td, y, x, speed)
        ax_td.autoscale()
        _colorbar(fig, ax_td, lc2, norm2)
    ax_td.plot(y[0],  x[0],  "o", color="lime", ms=8, zorder=5, label="start")
    ax_td.plot(y[-1], x[-1], "s", color="red",  ms=8, zorder=5, label="end")

    # Direction arrows — every ~5% of samples, skip if too few
    step = max(1, n // 20)
    for i in range(0, n - 1, step):
        dy, dx_ = y[i+1] - y[i], x[i+1] - x[i]
        dist = np.hypot(dy, dx_)
        if dist > 1e-6:
            ax_td.annotate("",
                xy=(y[i+1], x[i+1]), xytext=(y[i], x[i]),
                arrowprops=dict(arrowstyle="->", color="white", lw=0.8),
                zorder=4)

    if gates is not None:
        gx, gy = gates[:, 0], gates[:, 1]
        ax_td.scatter(gy, gx, c="gold", marker="D", s=100, zorder=6, label="gate")
        for i, (gxi, gyi) in enumerate(zip(gx, gy)):
            ax_td.text(gyi + 0.4, gxi, str(i), fontsize=8, color="gold")

    ax_td.set_xlabel("East — Y (m)"); ax_td.set_ylabel("North — X (m)")
    ax_td.set_title("Top-Down (speed color, arrows)")
    ax_td.set_aspect("equal", adjustable="datalim")
    ax_td.set_facecolor("#1a1a2e")
    ax_td.grid(True, alpha=0.2, color="white")
    ax_td.legend(fontsize=7)

    # --- Altitude ---
    ax_alt.plot(t, alt, color="deepskyblue", lw=1.5)
    if gates is not None:
        for gi, gz_i in enumerate(gates[:, 2]):
            ax_alt.axhline(gz_i, color="gold", lw=0.6, ls="--", alpha=0.6)
            ax_alt.text(t[-1], gz_i, f"G{gi}", fontsize=7, color="gold", va="bottom")
    ax_alt.set_xlabel("Time (s)"); ax_alt.set_ylabel("Altitude (m)")
    ax_alt.set_title(f"Altitude over Time  (max {alt.max():.1f} m)")
    ax_alt.grid(True, alpha=0.3)

    # --- Ground speed ---
    ax_spd.plot(t, speed, color="tomato", lw=1.5)
    mean_spd = float(speed.mean())
    ax_spd.axhline(mean_spd, color="orange", lw=1, ls="--",
                   label=f"mean {mean_spd:.1f} m/s")
    ax_spd.set_xlabel("Time (s)"); ax_spd.set_ylabel("Ground Speed (m/s)")
    ax_spd.set_title(f"Ground Speed  (max {speed.max():.1f} m/s)")
    ax_spd.legend(fontsize=8)
    ax_spd.grid(True, alpha=0.3)

    fig.subplots_adjust(left=0.06, right=0.94, top=0.93, bottom=0.10, hspace=0.48, wspace=0.38)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close(fig)
    print(f"[viz] PNG saved: {output_path}")


# ---------------------------------------------------------------------------
# Plotly interactive HTML
# ---------------------------------------------------------------------------

def make_html(data: dict, gates, output_path: str) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.colors as pc
    except ImportError:
        print("[viz] plotly not installed — skipping HTML output. "
              "Run: pip install plotly")
        return

    t    = data["time_s"]
    x    = data["x"]
    y    = data["y"]
    z    = data["z"]
    vx   = data["vx"]
    vy   = data["vy"]

    alt   = -z
    speed = np.sqrt(vx**2 + vy**2)
    n     = len(t)
    dur   = float(t.max() - t.min()) if n > 1 else 0.0

    # Speed → color strings for plotly
    vmax   = float(speed.max()) if speed.max() > 0 else 1.0
    s_norm = speed / vmax
    colors = pc.sample_colorscale("plasma", s_norm.tolist())

    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "scene", "colspan": 2}, None],
            [{"type": "xy"},                  {"type": "xy"}],
        ],
        subplot_titles=[
            "3-D Flight Path (colored by speed)",
            None,
            "Altitude over Time",
            "Ground Speed over Time",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # --- 3-D path (colored segments via individual line traces) ---
    # Plotly Line3D doesn't support per-point colors natively with a single trace;
    # use a scatter3d with markers=lines workaround: one trace per segment is too
    # slow. Instead use a single Scatter3d with a colorscale applied to the line.
    hover = [
        f"t={ti:.1f}s  spd={si:.2f} m/s  alt={ai:.1f}m"
        for ti, si, ai in zip(t, speed, alt)
    ]
    fig.add_trace(go.Scatter3d(
        x=y.tolist(), y=x.tolist(), z=alt.tolist(),
        mode="lines+markers",
        line=dict(color=speed.tolist(), colorscale="plasma", width=4,
                  colorbar=dict(title="m/s", x=0.48, len=0.45)),
        marker=dict(size=1.5, color=speed.tolist(), colorscale="plasma", opacity=0.8),
        hovertext=hover, hoverinfo="text",
        name="path",
    ), row=1, col=1)

    # Start / end markers
    fig.add_trace(go.Scatter3d(
        x=[float(y[0])], y=[float(x[0])], z=[float(alt[0])],
        mode="markers", marker=dict(size=10, color="lime", symbol="circle"),
        name="start",
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[float(y[-1])], y=[float(x[-1])], z=[float(alt[-1])],
        mode="markers", marker=dict(size=10, color="red", symbol="square"),
        name="end",
    ), row=1, col=1)

    # Gates in 3D
    if gates is not None:
        fig.add_trace(go.Scatter3d(
            x=gates[:, 1].tolist(), y=gates[:, 0].tolist(), z=gates[:, 2].tolist(),
            mode="markers+text",
            marker=dict(size=12, color="gold", symbol="diamond"),
            text=[f"G{i}" for i in range(len(gates))],
            textposition="top center", textfont=dict(color="gold", size=10),
            name="gates",
        ), row=1, col=1)

    # --- Altitude ---
    fig.add_trace(go.Scatter(
        x=t.tolist(), y=alt.tolist(),
        mode="lines", line=dict(color="deepskyblue", width=2),
        name="altitude", hovertemplate="t=%{x:.1f}s  alt=%{y:.2f}m<extra></extra>",
    ), row=2, col=1)
    if gates is not None:
        t_range = [float(t.min()), float(t.max())] if n > 1 else [0.0, 1.0]
        for i, gz_i in enumerate(gates[:, 2]):
            fig.add_trace(go.Scatter(
                x=t_range, y=[float(gz_i), float(gz_i)],
                mode="lines", line=dict(color="gold", width=0.8, dash="dash"),
                showlegend=(i == 0), name="gate alt",
                hovertemplate=f"G{i} alt={gz_i:.1f}m<extra></extra>",
            ), row=2, col=1)

    # --- Ground speed ---
    fig.add_trace(go.Scatter(
        x=t.tolist(), y=speed.tolist(),
        mode="lines", line=dict(color="tomato", width=2),
        name="ground speed", hovertemplate="t=%{x:.1f}s  spd=%{y:.2f}m/s<extra></extra>",
    ), row=2, col=2)
    mean_spd = float(speed.mean())
    t_range  = [float(t.min()), float(t.max())] if n > 1 else [0.0, 1.0]
    fig.add_trace(go.Scatter(
        x=t_range, y=[mean_spd, mean_spd],
        mode="lines", line=dict(color="orange", width=1, dash="dash"),
        name=f"mean {mean_spd:.1f} m/s",
        hovertemplate=f"mean speed={mean_spd:.2f}m/s<extra></extra>",
    ), row=2, col=2)

    fig.update_layout(
        title=dict(
            text=f"VQ1 Flight Trajectory  |  {n} samples  |  {dur:.1f}s",
            font=dict(size=15),
        ),
        paper_bgcolor="#0d0d1a",
        plot_bgcolor="#1a1a2e",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)"),
        scene=dict(
            xaxis_title="East (m)", yaxis_title="North (m)", zaxis_title="Alt (m)",
            bgcolor="#1a1a2e",
            xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
            zaxis=dict(gridcolor="#333"),
        ),
        height=900,
    )
    fig.update_xaxes(gridcolor="#333", row=2)
    fig.update_yaxes(gridcolor="#333", row=2)

    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"[viz] HTML saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VQ1 Trajectory Visualizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",    default="trajectory_log.csv",
                        help="Input CSV from TrajectoryLogger")
    parser.add_argument("--png",      default="trajectory_plot.png",
                        help="Output static PNG path")
    parser.add_argument("--output",   default=None,
                        help="Alias for --png (backwards compat)")
    parser.add_argument("--html",     default="trajectory_plot.html",
                        help="Output interactive HTML path")
    parser.add_argument("--no-html",  action="store_true",
                        help="Skip plotly HTML output")
    parser.add_argument("--no-gates", action="store_true",
                        help="Skip gate markers from track.py")
    args = parser.parse_args()

    png_path = args.output if args.output else args.png

    if not os.path.exists(args.input):
        print(f"[viz] ERROR: {args.input} not found — run with --log-trajectory first.")
        sys.exit(1)

    data = load_csv(args.input)
    if not data:
        print(f"[viz] {args.input} is empty.")
        sys.exit(1)

    n = len(data["time_s"])
    dur = float(data["time_s"].max() - data["time_s"].min()) if n > 1 else 0.0
    speed = np.sqrt(data["vx"]**2 + data["vy"]**2)
    print(f"[viz] Loaded {n} samples, {dur:.1f}s, max speed {speed.max():.1f} m/s")

    gates = None if args.no_gates else load_gates()
    if gates is not None:
        print(f"[viz] Loaded {len(gates)} gate positions from track.py")
    else:
        print("[viz] No gate positions (track.py unavailable or --no-gates)")

    make_png(data, gates, png_path)

    if not args.no_html:
        make_html(data, gates, args.html)


if __name__ == "__main__":
    main()
