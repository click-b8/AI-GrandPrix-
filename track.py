"""Race track definition — competition-scale FPV racing course.

Gates: 1.5m x 1.5m (MultiGP standard)
Course: 8 gates with 3D altitude variation in ~50m x 30m arena
Each gate has a position and a yaw (facing direction).
"""

import numpy as np
from config import GATE_WIDTH, GATE_HEIGHT

# Each gate: (x, y, z, yaw_rad)
# yaw defines which direction the drone should fly through the gate
RACE_TRACK = [
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
