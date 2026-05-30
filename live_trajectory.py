"""
VQ1 Live Trajectory Dashboard — debugging tool, not part of submission.

Runs a lightweight HTTP server. Open the printed URL in any browser and
the Plotly chart auto-polls /data every 2 seconds, so the flight path
builds in real time as trajectory_log.csv is written during a flight.

Usage:
    python3 live_trajectory.py [--csv trajectory_log.csv] [--port 8765] [--no-browser]

Then open http://127.0.0.1:8765 in your browser (opened automatically unless
--no-browser is passed).  Press Ctrl+C to stop the server.

Architecture:
  GET /        → HTML dashboard (plotly from CDN + JS polling)
  GET /data    → JSON snapshot of trajectory_log.csv  (cache-busted)
  Browser JS   → setInterval(fetchAndUpdate, 2000) → Plotly.react()
                  uirevision='static' preserves 3D camera between updates
"""

import argparse
import csv
import json
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> dict:
    """Return dict of float lists keyed by CSV column, or {} if unavailable."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("time_s", "").strip()]
        if not rows:
            return {}
        return {k: [float(r[k]) for r in rows] for k in rows[0]}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Gate loader
# ---------------------------------------------------------------------------

def _load_gates():
    """Return list-of-lists [[x,y,z], ...] from track.py, or None."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from track import get_gate_positions
        return get_gate_positions().tolist()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HTML dashboard
# The GATES_JSON placeholder is replaced at serve time with actual gate data.
# All JS curly braces use verbatim string, no Python f-string.
# ---------------------------------------------------------------------------

_DASHBOARD = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>VQ1 Live Trajectory</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0d0d1a;
      color: #ddd;
      font-family: 'Courier New', monospace;
      padding: 8px 12px;
      display: flex; flex-direction: column; height: 100vh;
    }
    #header { flex: 0 0 auto; margin-bottom: 6px; }
    #header h1 { font-size: 1em; color: #aaa; }
    #status {
      font-size: 0.82em; color: #5f9; margin-top: 2px;
      letter-spacing: 0.04em;
    }
    #chart { flex: 1 1 auto; min-height: 0; }
  </style>
</head>
<body>
  <div id="header">
    <h1>&#9652; VQ1 Live Trajectory Dashboard</h1>
    <div id="status">&#8226; Connecting to /data ...</div>
  </div>
  <div id="chart"></div>

<script>
// Gate positions injected by the server
const GATES = GATES_JSON;
const POLL_MS = 2000;
let initialized = false;

// ── helpers ──────────────────────────────────────────────────────────────

function groundSpeed(vx, vy) {
  return vx.map((v, i) => Math.sqrt(v * v + vy[i] * vy[i]));
}

function linspace(a, b, n) {
  if (n <= 1) return [a];
  return Array.from({length: n}, (_, i) => a + (b - a) * i / (n - 1));
}

// ── trace builders ───────────────────────────────────────────────────────

function buildTraces(d) {
  const n     = d.time_s ? d.time_s.length : 0;
  const alt   = (d.z  || []).map(v => -v);
  const speed = n ? groundSpeed(d.vx || [], d.vy || []) : [];
  const tArr  = d.time_s || [];
  const xArr  = d.x || [];   // North
  const yArr  = d.y || [];   // East

  const tRange = n > 1 ? [tArr[0], tArr[n - 1]] : [0, 1];
  const traces = [];

  // ── LEFT: 3-D flight path ────────────────────────────────────────
  traces.push({
    type: 'scatter3d', mode: 'lines+markers',
    x: yArr, y: xArr, z: alt,
    line:   { color: speed, colorscale: 'Plasma', width: 4 },
    marker: { size: 1.5, color: speed, colorscale: 'Plasma', opacity: 0.9 },
    customdata: tArr.map((t, i) => [t, speed[i] || 0]),
    hovertemplate: 't=%{customdata[0]:.1f}s  spd=%{customdata[1]:.2f}m/s  alt=%{z:.1f}m<extra></extra>',
    name: 'path',
  });

  if (n > 0) {
    traces.push({
      type: 'scatter3d', mode: 'markers',
      x: [yArr[0]], y: [xArr[0]], z: [alt[0]],
      marker: { size: 9, color: '#00ff88', symbol: 'circle' },
      name: 'start',
    });
    traces.push({
      type: 'scatter3d', mode: 'markers',
      x: [yArr[n-1]], y: [xArr[n-1]], z: [alt[n-1]],
      marker: { size: 9, color: '#ff4444', symbol: 'square' },
      name: 'end',
    });
  }

  if (GATES) {
    traces.push({
      type: 'scatter3d', mode: 'markers+text',
      x: GATES.map(g => g[1]),
      y: GATES.map(g => g[0]),
      z: GATES.map(g => g[2]),
      text: GATES.map((_, i) => 'G' + i),
      textposition: 'top center',
      textfont: { color: 'gold', size: 10 },
      marker: { size: 9, color: 'gold', symbol: 'diamond' },
      name: 'gates',
    });
  }

  // ── RIGHT-TOP: altitude over time ────────────────────────────────
  traces.push({
    type: 'scatter', mode: 'lines',
    x: tArr, y: alt,
    line: { color: '#00bfff', width: 2 },
    name: 'altitude',
    xaxis: 'x2', yaxis: 'y2',
    hovertemplate: 't=%{x:.1f}s  alt=%{y:.2f}m<extra></extra>',
  });

  if (GATES) {
    GATES.forEach((g, i) => {
      traces.push({
        type: 'scatter', mode: 'lines',
        x: tRange, y: [g[2], g[2]],
        line: { color: 'gold', width: 0.9, dash: 'dash' },
        name: 'G' + i + ' alt',
        showlegend: (i === 0),
        legendgroup: 'gatealt',
        xaxis: 'x2', yaxis: 'y2',
        hovertemplate: 'G' + i + ' alt=' + g[2].toFixed(1) + 'm<extra></extra>',
      });
    });
  }

  // ── RIGHT-BOTTOM: ground speed over time ─────────────────────────
  traces.push({
    type: 'scatter', mode: 'lines',
    x: tArr, y: speed,
    line: { color: '#ff6b6b', width: 2 },
    name: 'speed',
    xaxis: 'x3', yaxis: 'y3',
    hovertemplate: 't=%{x:.1f}s  spd=%{y:.2f}m/s<extra></extra>',
  });

  const mean = speed.length ? speed.reduce((a, b) => a + b, 0) / speed.length : 0;
  traces.push({
    type: 'scatter', mode: 'lines',
    x: tRange, y: [mean, mean],
    line: { color: 'orange', width: 1.2, dash: 'dot' },
    name: 'mean ' + mean.toFixed(1) + ' m/s',
    xaxis: 'x3', yaxis: 'y3',
    hovertemplate: 'mean=' + mean.toFixed(2) + 'm/s<extra></extra>',
  });

  return traces;
}

// ── layout ───────────────────────────────────────────────────────────────

function buildLayout() {
  const axStyle = { gridcolor: '#2a2a3e', zerolinecolor: '#444',
                    color: '#bbb', tickfont: {size: 9} };
  return {
    paper_bgcolor: '#0d0d1a',
    plot_bgcolor:  '#0d0d1a',
    font: { color: '#ccc', size: 10 },
    uirevision: 'keep',   // preserves 3D camera across Plotly.react() calls

    // 3-D scene occupies left half, full height
    scene: {
      domain: { x: [0, 0.47], y: [0, 1] },
      aspectmode: 'data',
      bgcolor: '#111122',
      xaxis: Object.assign({title: 'East (m)'},  axStyle),
      yaxis: Object.assign({title: 'North (m)'}, axStyle),
      zaxis: Object.assign({title: 'Alt (m)'},   axStyle),
    },

    // Right-top: altitude
    xaxis2: Object.assign({ domain: [0.52, 1.0], anchor: 'y2',
                             title: {text: 'Time (s)', font: {size: 9}} }, axStyle),
    yaxis2: Object.assign({ domain: [0.52, 1.0], anchor: 'x2',
                             title: {text: 'Altitude (m)', font: {size: 9}} }, axStyle),

    // Right-bottom: speed
    xaxis3: Object.assign({ domain: [0.52, 1.0], anchor: 'y3',
                             title: {text: 'Time (s)', font: {size: 9}} }, axStyle),
    yaxis3: Object.assign({ domain: [0.0, 0.44], anchor: 'x3',
                             title: {text: 'Speed (m/s)', font: {size: 9}} }, axStyle),

    legend: {
      bgcolor: 'rgba(10,10,30,0.7)',
      bordercolor: '#333', borderwidth: 1,
      x: 0.48, y: 0.98,
      font: { size: 9 },
    },
    margin: { t: 10, b: 35, l: 50, r: 20 },

    // Panel title annotations
    annotations: [
      { text: '3-D Flight Path', xref: 'paper', yref: 'paper',
        x: 0.23, y: 1.0, showarrow: false, font: {size: 10, color: '#888'} },
      { text: 'Altitude over Time', xref: 'paper', yref: 'paper',
        x: 0.76, y: 1.0, showarrow: false, font: {size: 10, color: '#888'} },
      { text: 'Ground Speed over Time', xref: 'paper', yref: 'paper',
        x: 0.76, y: 0.46, showarrow: false, font: {size: 10, color: '#888'} },
    ],
  };
}

// ── poll loop ─────────────────────────────────────────────────────────────

function fetchAndUpdate() {
  fetch('/data?t=' + Date.now())   // cache-bust
    .then(r => {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    })
    .then(d => {
      const n     = d.time_s ? d.time_s.length : 0;
      const dur   = n > 1 ? (d.time_s[n-1] - d.time_s[0]).toFixed(1) : '0.0';
      const speed = n ? groundSpeed(d.vx || [], d.vy || []) : [];
      const maxV  = speed.length ? Math.max(...speed).toFixed(2) : '0.00';

      document.getElementById('status').innerHTML =
        '&#9679; LIVE &nbsp;|&nbsp; ' + n + ' samples &nbsp;|&nbsp; ' +
        dur + 's &nbsp;|&nbsp; max speed ' + maxV + ' m/s &nbsp;|&nbsp; ' +
        new Date().toLocaleTimeString();

      const traces = buildTraces(d);
      const layout = buildLayout();

      if (!initialized) {
        Plotly.newPlot('chart', traces, layout, { responsive: true });
        initialized = true;
      } else {
        Plotly.react('chart', traces, layout);
      }
    })
    .catch(err => {
      document.getElementById('status').textContent =
        '\\u26a0 ' + err.message + ' \\u2014 retrying in ' + (POLL_MS/1000) + 's';
    });
}

fetchAndUpdate();
setInterval(fetchAndUpdate, POLL_MS);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    csv_path: str = "trajectory_log.csv"
    gates_json: str = "null"

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            body = _DASHBOARD.replace("GATES_JSON", self.gates_json).encode()
            self._respond(200, "text/html; charset=utf-8", body)
        elif self.path.startswith("/data"):
            data = _read_csv(self.csv_path)
            body = json.dumps(data).encode()
            self._respond(200, "application/json", body,
                          extra=[("Cache-Control", "no-store")])
        else:
            self.send_error(404)

    def _respond(self, code, ctype, body, extra=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # suppress per-request noise


# ---------------------------------------------------------------------------
# Demo mode — fake flight through the 8 gates
# ---------------------------------------------------------------------------

def _generate_demo_path(n_points: int = 300) -> np.ndarray:
    """
    Return (n_points, 3) NED path through all 8 race gates and back to start.
    Coordinates: x=North, y=East, z=NED-down (negative = altitude).
    Uses numpy.interp for smooth per-axis interpolation; no scipy needed.
    """
    # Gate positions from track.py RACE_TRACK, converted to NED z (z=-altitude).
    waypoints = np.array([
        [  0.0,   0.0, -1.0],   # launch
        [  8.0,   0.0, -2.5],   # G0
        [ 14.0,   7.0, -3.5],   # G1
        [ 10.0,  14.0, -4.5],   # G2
        [  0.0,  16.0, -3.5],   # G3
        [-10.0,  14.0, -2.0],   # G4
        [-14.0,   7.0, -1.5],   # G5
        [-10.0,   0.0, -2.5],   # G6
        [  0.0,  -2.0, -2.5],   # G7
        [  0.0,   0.0, -1.0],   # return to start
    ], dtype=float)

    u_wp   = np.linspace(0.0, 1.0, len(waypoints))
    u_fine = np.linspace(0.0, 1.0, n_points)
    return np.column_stack([
        np.interp(u_fine, u_wp, waypoints[:, i]) for i in range(3)
    ])


def _run_demo(csv_path: str, hz: float = 10.0, n_points: int = 300) -> None:
    """
    Write fake flight rows to csv_path at hz rate, one row at a time,
    so the live dashboard shows the path growing on each 2-second poll.
    Stops after one full lap (n_points rows).
    """
    import time as _time

    path = _generate_demo_path(n_points)
    dt   = 1.0 / hz

    # Velocities via central differences (m/s)
    vx = np.gradient(path[:, 0], dt)
    vy = np.gradient(path[:, 1], dt)
    vz = np.gradient(path[:, 2], dt)

    # Simplified attitude from velocity direction
    speed_h = np.sqrt(vx**2 + vy**2)
    yaw     = np.arctan2(vy, vx + 1e-9)
    pitch   = np.arctan2(-vz, speed_h + 1e-9) * 0.3   # gentle nose-up/down
    roll    = np.zeros(n_points)

    max_spd = float(np.sqrt(vx**2 + vy**2).max())
    print(f"[demo] {n_points} pts at {hz:.0f} Hz -> {n_points/hz:.0f}s  "
          f"max speed {max_spd:.1f} m/s")
    print(f"[demo] Writing to {csv_path!r} — browser polls every 2s")

    # Start fresh: write header only
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["time_s", "x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw"]
        )

    start = _time.monotonic()
    for i in range(n_points):
        row = [
            round(i * dt, 4),
            round(float(path[i, 0]), 4), round(float(path[i, 1]), 4),
            round(float(path[i, 2]), 4),
            round(float(vx[i]),      4), round(float(vy[i]),      4),
            round(float(vz[i]),      4),
            round(float(roll[i]),    4), round(float(pitch[i]),   4),
            round(float(yaw[i]),     4),
        ]
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        due  = start + (i + 1) * dt
        wait = due - _time.monotonic()
        if wait > 0:
            _time.sleep(wait)

    print(f"[demo] Lap complete — {n_points} rows written to {csv_path!r}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VQ1 Live Trajectory Dashboard — browser-based, polls every 2s",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv",        default="trajectory_log.csv",
                        help="Trajectory CSV to watch")
    parser.add_argument("--port",       type=int, default=8765,
                        help="HTTP server port")
    parser.add_argument("--host",       default="127.0.0.1",
                        help="HTTP server bind address")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open the browser")
    parser.add_argument("--demo", action="store_true",
                        help="Generate a fake flight through all 8 gates, writing to "
                             "--csv at 10 Hz. Stops after one full lap (~30s). "
                             "Use this to confirm the live dashboard works before "
                             "the real sim drops.")
    args = parser.parse_args()

    gates = _load_gates()
    _Handler.csv_path   = args.csv
    _Handler.gates_json = json.dumps(gates)

    gate_str = f"{len(gates)} gates from track.py" if gates else "no gates"
    url = f"http://{args.host}:{args.port}"

    if args.demo:
        import threading
        threading.Thread(
            target=_run_demo, args=(args.csv,),
            name="DemoFlight", daemon=True,
        ).start()

    server = HTTPServer((args.host, args.port), _Handler)
    print(f"[live] {url}  ({gate_str})")
    print(f"[live] Watching {args.csv!r} — browser polls every 2s")
    print("[live] Ctrl+C to stop.")

    if not args.no_browser:
        # Open slightly delayed so the server is ready
        import threading
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[live] Stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
