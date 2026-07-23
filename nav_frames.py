"""UE-world -> sim local-NED setpoint conversion for the position-target path.

Context
-------
VADR-TS-002 sec4.3 lists SET_POSITION_TARGET_LOCAL_NED as an accepted
Client->Simulator control message. If the sim's own controller flies to position
setpoints, the hardcoded qualifying script is just: send the 6 known gate
coordinates in sequence, advance on active_gate. NO position telemetry, NO
vision needed.

The gate coordinates in course_gates_cm.json are raw UNREAL WORLD space:
left-handed, Z-up, centimetres, X-forward, Y-right, NOT recentred. The sim's
control input is LOCAL NED (North-East-Down), metres, origin almost certainly at
the drone spawn/home. This module converts one to the other.

The unknown the probe must settle (tools/probe_position_target.py)
-----------------------------------------------------------------
Two things are NOT known until we test against the sim:
  (1) origin  -- is local-NED (0,0,0) the spawn point? (pure-climb probe answers)
  (2) axes    -- which UE axis is North vs East, and the sign. We default to the
                 world-aligned guess below, but the gate-1 probe confirms it.

Axis hypotheses (select with `axes=`):
  "world"        N=+dX_ue, E=+dY_ue, D=-dZ_ue   (local NED aligned to UE world)
  "world_negN"   N=-dX_ue, E=+dY_ue, D=-dZ_ue   (North = UE -X)
  "spawn_fwd"    x = -dX_ue (drone spawn faces UE -X, yaw -180), y=+dY_ue, D=-dZ_ue
Only the pure-climb + gate-1 probes tell us which is real; everything here is
translation/axis bookkeeping, deliberately explicit so the fix is a one-liner.

D sign is unambiguous regardless of the horizontal choice: UE Z is up, NED D is
down, so D = -(dZ_ue). A negative D setpoint (e.g. z=-3) commands a CLIMB.
"""
from __future__ import annotations

import json
import os

import numpy as np

_COURSE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "course_gates_cm.json")


def load_spawn_and_gates(path: str = _COURSE_PATH):
    """Return (spawn_cm[3], gates=[(tag, pos_cm[3]), ...]) from the course JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    spawn_cm = np.asarray(data["spawn"]["pos_cm"], dtype=np.float64)
    gates = [(g.get("tag"), np.asarray(g["pos_cm"], dtype=np.float64))
             for g in data["gates"]]
    return spawn_cm, gates


def ue_delta_to_ned(delta_ue_m: np.ndarray, axes: str = "world") -> np.ndarray:
    """Map a UE-world displacement (metres, X-fwd/Y-right/Z-up) to local NED.

    delta_ue_m: gate_ue - spawn_ue, already in METRES.
    Returns [N, E, D] metres. See module docstring for the axis hypotheses.
    """
    dx, dy, dz = float(delta_ue_m[0]), float(delta_ue_m[1]), float(delta_ue_m[2])
    if axes == "world":
        return np.array([dx, dy, -dz], dtype=np.float64)
    if axes == "world_negN":
        return np.array([-dx, dy, -dz], dtype=np.float64)
    if axes == "spawn_fwd":
        return np.array([-dx, dy, -dz], dtype=np.float64)  # same as world_negN for N;
        #   kept as a named hypothesis in case E also needs flipping after the probe.
    raise ValueError(f"unknown axes hypothesis: {axes!r}")


def gate_ned_setpoints(axes: str = "world", path: str = _COURSE_PATH):
    """Spawn-relative NED setpoints for every gate, in ORDER (START..FINISH).

    Returns list of dicts: {index, tag, ned:[N,E,D], delta_ue_m:[dx,dy,dz]}.
    """
    spawn_cm, gates = load_spawn_and_gates(path)
    out = []
    for i, (tag, pos_cm) in enumerate(gates):
        delta_ue_m = (pos_cm - spawn_cm) / 100.0
        ned = ue_delta_to_ned(delta_ue_m, axes=axes)
        out.append({"index": i, "tag": tag,
                    "ned": [round(float(v), 3) for v in ned],
                    "delta_ue_m": [round(float(v), 3) for v in delta_ue_m]})
    return out


def climb_setpoint(height_m: float = 3.0):
    """Pure vertical climb setpoint in LOCAL_NED: (N=0, E=0, D=-height).

    Origin-and-frame diagnostic: zero horizontal removes ALL axis ambiguity, so
    if the drone climbs ~height_m the sim honours position targets AND local-NED
    origin is spawn-relative. Vertical motion is unmistakable vs drift/noise."""
    return np.array([0.0, 0.0, -abs(height_m)], dtype=np.float64)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Print spawn-relative NED gate setpoints.")
    ap.add_argument("--axes", default="world",
                    choices=["world", "world_negN", "spawn_fwd"])
    args = ap.parse_args()
    spawn_cm, _ = load_spawn_and_gates()
    print(f"spawn UE (cm): {spawn_cm.tolist()}   axes hypothesis: {args.axes}")
    print(f"pure-climb probe setpoint (LOCAL_NED, 3 m up): "
          f"{climb_setpoint(3.0).tolist()}\n")
    print(f"{'idx':>3} {'tag':<8} {'N':>9} {'E':>9} {'D':>9}   (dx,dy,dz)_ue_m")
    for row in gate_ned_setpoints(axes=args.axes):
        n, e, d = row["ned"]
        print(f"{row['index']:>3} {str(row['tag'] or ''):<8} "
              f"{n:>9.2f} {e:>9.2f} {d:>9.2f}   {row['delta_ue_m']}")
