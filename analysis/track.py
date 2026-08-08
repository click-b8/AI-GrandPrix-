"""Race track definition — competition-scale FPV racing course.

Gates: 1.5m x 1.5m (MultiGP standard)
Course: the real Anduril-6 VQ1 export (course_gates_cm.json), 6 gates
START->FINISH descending ~29m -> ~3m over ~160m, loaded at import (see bottom).
Each gate has a position and a yaw (facing direction).
"""

import os

import numpy as np
from config import GATE_WIDTH, GATE_HEIGHT

# Each gate: (x, y, z, yaw_rad). yaw = fly-through heading (yaw=0 -> normal +X).
# DEAD / historical placeholder kept for reference only. The live course is the
# real Anduril-6 VQ1 export, loaded at import time below (see _COURSE_PATH). This
# tuple is NOT used anywhere; do not wire it back in as a fallback.
_PLACEHOLDER_TRACK = [
    (8.0,   0.0,  2.5,  0.0),           # Gate 0: straight ahead
    (14.0,  7.0,  3.5,  np.pi/4),       # Gate 1: climbing right turn
    (10.0, 14.0,  4.5,  np.pi/2),       # Gate 2: high point
    (0.0,  16.0,  3.5,  np.pi),         # Gate 3: turning back
    (-10.0, 14.0, 2.0, -3*np.pi/4),     # Gate 4: descending
    (-14.0,  7.0, 1.5, -np.pi/2),       # Gate 5: low fast section
    (-10.0,  0.0, 2.5, -np.pi/4),       # Gate 6: climbing back
    (0.0,  -2.0,  2.5,  0.0),           # Gate 7: back to start
]


def get_gate_positions(gate_list=None):
    """Return gate center positions as (N, 3) array."""
    if gate_list is None:
        gate_list = RACE_TRACK
    return np.array([(g[0], g[1], g[2]) for g in gate_list], dtype=np.float32)


def get_gate_yaws(gate_list=None):
    """Return gate yaw angles as (N,) array."""
    if gate_list is None:
        gate_list = RACE_TRACK
    return np.array([g[3] for g in gate_list], dtype=np.float32)


def build_gate_xml(gate_list=None):
    """Generate MuJoCo XML body elements for race gates."""
    if gate_list is None:
        gate_list = RACE_TRACK

    xml_parts = []
    hw = GATE_WIDTH / 2
    hh = GATE_HEIGHT / 2
    pole_r = 0.04

    for i, (gx, gy, gz, yaw) in enumerate(gate_list):
        cy, sy = np.cos(yaw), np.sin(yaw)

        # Left and right poles (offset perpendicular to gate facing)
        for side, color in [(-1, "1 0.15 0.15 1"), (1, "1 0.15 0.15 1")]:
            px = gx + side * hw * (-sy)  # perpendicular to yaw direction
            py = gy + side * hw * cy
            xml_parts.append(
                f'<body name="gate{i}_pole{side}" pos="{px:.3f} {py:.3f} {gz:.3f}">'
                f'  <geom type="cylinder" size="{pole_r} {hh}" rgba="{color}" contype="0" conaffinity="0"/>'
                f'</body>'
            )

        # Top bar
        bar_euler = f"0 90 {np.degrees(yaw)}"
        xml_parts.append(
            f'<body name="gate{i}_top" pos="{gx:.3f} {gy:.3f} {gz + hh:.3f}">'
            f'  <geom type="cylinder" size="{pole_r} {hw}" euler="{bar_euler}" rgba="0.15 1 0.15 1" contype="0" conaffinity="0"/>'
            f'</body>'
        )
        # Bottom bar
        xml_parts.append(
            f'<body name="gate{i}_bot" pos="{gx:.3f} {gy:.3f} {gz - hh:.3f}">'
            f'  <geom type="cylinder" size="{pole_r} {hw}" euler="{bar_euler}" rgba="0.15 0.15 1 1" contype="0" conaffinity="0"/>'
            f'</body>'
        )

    return "\n    ".join(xml_parts)


# ---------------------------------------------------------------------------
# UE -> MuJoCo course conversion + JSON loading (coordinate-independent)
#
# The extraction exports gates in Unreal Engine coords: LEFT-handed, Z-up,
# centimeters (X forward, Y RIGHT). Our env is MuJoCo: RIGHT-handed, Z-up,
# meters (X forward, Y LEFT). Keeping X-forward and Z-up, the only change is a
# Y flip. M = diag(1, -1, 1) has det = -1 -> exactly the handedness flip
# (the standard UE <-> glTF/MuJoCo convention).
# ---------------------------------------------------------------------------

UE_TO_ENV_AXES = np.diag([1.0, -1.0, 1.0]).astype(np.float64)  # LH Z-up -> RH Z-up


def ue_cm_to_env_m(pos_cm):
    """UE (left-handed, Z-up, cm) position -> MuJoCo env (right-handed, Z-up, m).
    cm -> m (/100), then negate Y (LH -> RH). No recentering (done at load time)."""
    return UE_TO_ENV_AXES @ (np.asarray(pos_cm, dtype=np.float64) / 100.0)


def _wrap_pi(angle_rad):
    """Wrap to (-pi, pi]."""
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def ue_yaw_to_env_yaw(yaw_deg, normal_offset_deg=0.0):
    """UE actor yaw (deg) -> env gate fly-through yaw (rad). env yaw=0 => gate
    normal points +X (build_gate_xml puts the poles at +/-Y at yaw=0).

    Convention math:
      * The Y flip negates a heading angle: (cos p, sin p) -> (cos p, -sin p)
        = (cos(-p), sin(-p)), so  phi_env = -phi_ue.
      * The exported ACTOR yaw may differ from the fly-through NORMAL by a fixed
        offset; fold it in first:  phi_ue_normal = yaw_deg + normal_offset_deg.
      => env_yaw = wrap( -radians(yaw_deg + normal_offset_deg) ).

    normal_offset_deg defaults to 0 (treat exported yaw as the normal heading).
    DO NOT TRUST this default -- the offset is confirmed EMPIRICALLY, either from
    the corrected JSON's rot_pyr (roll/pitch/yaw) data, or from the first real
    telemetry run: a gate face-on vs edge-on at approach is unmissable. An
    off-by-90 leaves the gates edge-on to the flight path.
    """
    return _wrap_pi(-np.radians(yaw_deg + normal_offset_deg))


def _horizontal_turn_signs(positions):
    """Signed z of cross(seg_i, seg_{i+1}) for consecutive horizontal segments.
    In our right-handed frame, +1 = left turn (CCW viewed from +Z). The formula
    is coordinate-agnostic, so it is applied identically to UE and env points."""
    p = np.asarray(positions, dtype=np.float64)
    seg = np.diff(p[:, :2], axis=0)
    return np.sign(np.array(
        [seg[i, 0] * seg[i + 1, 1] - seg[i, 1] * seg[i + 1, 0]
         for i in range(len(seg) - 1)]
    ))


def assert_course_matches_source(source_pos_ue_cm, env_gates, spacing_atol=1e-3):
    """Validate a converted course against the RAW UE source positions, via ONE
    code path -- no extraction-derived spacings/turn-signs ever enter.

    - SPACING: a rigid transform preserves distance, so the source (/100) spacing
      must equal the env spacing -> catches a cm/m scale slip or a dropped /
      duplicated axis.
    - SIGNED TURN DIRECTION: distance ALSO survives mirroring, so a wrong
      handedness (missing or extra Y flip) passes spacing but flips the dogleg
      chirality. We compute the horizontal turn signs of the raw UE points and of
      the env points with the SAME helper, then flip sign ONCE for the Y flip and
      assert equality (env == -ue). The extraction's own convention never enters.
    """
    src = np.asarray(source_pos_ue_cm, dtype=np.float64) / 100.0   # UE meters, pre-flip
    env = np.asarray(env_gates, dtype=np.float64)[:, :3]
    src_sp = np.linalg.norm(np.diff(src, axis=0), axis=1)
    env_sp = np.linalg.norm(np.diff(env, axis=0), axis=1)
    if not np.allclose(src_sp, env_sp, atol=spacing_atol):
        raise AssertionError(f"spacing mismatch: env={env_sp} source={src_sp}")
    ue_turns = _horizontal_turn_signs(src)
    env_turns = _horizontal_turn_signs(env)
    if not np.array_equal(env_turns, -ue_turns):     # one Y flip negates the turn sign
        raise AssertionError(f"turn-direction mismatch (handedness?): "
                             f"env={env_turns} expected(-ue)={-ue_turns}")


def load_course_json(path, recenter_to_start=True, start_altitude_m=3.0,
                     normal_offset_deg=0.0, validate=True):
    """Load a course JSON export into RACE_TRACK format. File drop + one call.

    Schema (extraction export):
        {
          "gates": [                       # ordered START -> FINISH
            {"pos_cm": [x, y, z],          # UE cm, left-handed Z-up
             "yaw_deg": <float>,           # UE actor yaw
             "tag": "START" | "FINISH" | null},
            ...
          ],
          "spawn": {"pos_cm": [x, y, z], "yaw_deg": <float>}   # optional
        }

    Returns (race_track, spawn_env):
      race_track: list of (x, y, z, yaw_rad) -> assign to RACE_TRACK.
      spawn_env:  np.ndarray (4,) env-frame spawn [x, y, z, yaw_rad], or None.

    If recenter_to_start, translates so the START gate is at
    (0, 0, start_altitude_m); the SAME translation is applied to the spawn.
    If validate, self-checks the conversion via assert_course_matches_source
    (spacing + signed-turn, computed from the JSON's own raw positions).
    """
    import json
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    gates = data["gates"]
    src_cm = [g["pos_cm"] for g in gates]
    env_pos = np.array([ue_cm_to_env_m(p) for p in src_cm])
    env_yaw = [ue_yaw_to_env_yaw(g["yaw_deg"], normal_offset_deg) for g in gates]

    translation = np.zeros(3)
    if recenter_to_start:
        start_i = next((i for i, g in enumerate(gates)
                        if str(g.get("tag", "")).upper() == "START"), 0)
        translation = np.array([0.0, 0.0, start_altitude_m]) - env_pos[start_i]
        env_pos = env_pos + translation

    if validate:
        # Distance + turn checks are translation-invariant, so post-recenter is fine.
        assert_course_matches_source(src_cm, env_pos)

    race_track = [(float(p[0]), float(p[1]), float(p[2]), float(y))
                  for p, y in zip(env_pos, env_yaw)]

    spawn_env = None
    spawn = data.get("spawn")
    if spawn:
        spawn_pos = ue_cm_to_env_m(spawn["pos_cm"]) + translation
        # The spawn actor yaw is the drone's LITERAL start heading, not a gate
        # normal, so normal_offset_deg is NOT applied here -- only the Y-flip
        # heading negation. (The raw export's own spawn->gate1 bearing check
        # confirms the unmodified spawn yaw already faces the first gate.)
        spawn_yaw = ue_yaw_to_env_yaw(spawn["yaw_deg"], normal_offset_deg=0.0)
        spawn_env = np.array([spawn_pos[0], spawn_pos[1], spawn_pos[2], spawn_yaw])

    return race_track, spawn_env


# ---------------------------------------------------------------------------
# Live course — the real Anduril-6 VQ1 export, loaded at import.
#
# normal_offset_deg=-90: the gate ACTOR yaw differs from the fly-through NORMAL
#   by a fixed -90 deg (confirmed against the render — gates come up face-on).
# start_altitude_m=29: the START gate sits ~29 m up; the whole course + spawn is
#   recentered so START is at (0, 0, 29).
#
# There is deliberately NO fallback to _PLACEHOLDER_TRACK: a failed load means
# the course file is missing/corrupt, and silently racing the wrong geometry is
# far worse than a hard, diagnostic stop.
# ---------------------------------------------------------------------------
# One level up: track.py lives in analysis/, but course_gates_cm.json stays at the
# repo root because nav_frames.py (still at the root) resolves it next to ITSELF.
# Moving or duplicating the file would trade one breakage for another.
_COURSE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "course_gates_cm.json")

try:
    RACE_TRACK, SPAWN = load_course_json(
        _COURSE_PATH, start_altitude_m=29, normal_offset_deg=-90)
except Exception as exc:  # noqa: BLE001 -- re-raised loudly with diagnostics
    raise RuntimeError(
        f"track.py: failed to load the live course from {_COURSE_PATH!r}: "
        f"{type(exc).__name__}: {exc}. This is fatal by design -- there is NO "
        f"placeholder fallback. Confirm course_gates_cm.json exists at the "
        f"repo root and passes the UE->MuJoCo self-validation."
    ) from exc
