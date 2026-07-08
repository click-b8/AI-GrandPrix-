"""A2 GravityEstimator — complementary filter: IMU (accel+gyro, FRD) -> gravity.

Recovers the gravity-DOWN direction in the FRD body frame from the deploy IMU so
the VQ1 policy's 10-D observation carries a clean gravity vector even though raw
in-flight accelerometer samples are wildly nonphysical (a logged sample read
|a| ~= 1355 m/s^2, ~138 g). Implements the contract in
obsidian/observation-spec.md "A2 GravityEstimator requirements".

Conventions (must match tests/test_observation_contract.py):
  - Inputs accel, gyro are FRD body (Forward-Right-Down), SI units.
  - Output is gravity-DOWN in FRD, magnitude ~= g (9.81). At rest & level the
    accelerometer reads specific force [0,0,-9.81] (zacc negative), so gravity-
    down in FRD is [0,0,+9.81] = -accel.
  - The 10-D observation builder maps FRD->FLU via C = diag(1,-1,-1) and unit-
    normalizes; this module stays entirely in FRD at magnitude g.

Algorithm (complementary filter):
  1. Propagate on the gyro:  g_dot = -omega x g  (a fixed world vector seen from
     a rotating body), renormalized to preserve magnitude.
  2. Correct toward accel-derived gravity (-accel) with a small blend weight
     alpha, but ONLY when |accel| is within a tight gate around 1 g; otherwise
     propagate on the gyro alone for that step (spike rejection).
"""

from __future__ import annotations

import numpy as np

GRAVITY = 9.81


class GravityEstimator:
    """Fuses FRD accel + gyro into a gravity-down estimate in FRD (norm ~= g)."""

    def __init__(self, alpha=0.02, accel_gate_g=(0.85, 1.15), g=GRAVITY,
                 rate_hz=115.0, R_mount=None):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.g_mag = float(g)
        self._lo = accel_gate_g[0] * self.g_mag
        self._hi = accel_gate_g[1] * self.g_mag
        self.default_dt = 1.0 / rate_hz
        # Fixed body->IMU mount rotation. Spec keeps this identity until the
        # de-risk flight resolves body-spawn vs mount-offset; do NOT bake a tilt
        # here. Applied to incoming accel/gyro when provided.
        self.R_mount = None if R_mount is None else np.asarray(R_mount, float)
        self.reset()

    def reset(self, gravity_frd=None):
        """Reset the estimate; defaults to level (gravity-down = +z in FRD)."""
        if gravity_frd is None:
            self._g = np.array([0.0, 0.0, self.g_mag], dtype=float)
        else:
            v = np.asarray(gravity_frd, float)
            self._g = v / np.linalg.norm(v) * self.g_mag
        return self._g.copy()

    @property
    def gravity(self):
        return self._g.copy()

    def _mount(self, v):
        return v if self.R_mount is None else self.R_mount @ v

    def update(self, accel_frd, gyro_frd, dt=None):
        """Advance one IMU step; return gravity-down in FRD (norm ~= g)."""
        dt = self.default_dt if dt is None else float(dt)
        accel = self._mount(np.asarray(accel_frd, float))
        gyro = self._mount(np.asarray(gyro_frd, float))

        # 1) Gyro propagation: g_dot = -omega x g. Renormalize to hold |g| fixed
        #    (first-order Euler integration otherwise drifts the magnitude).
        g_pred = self._g - dt * np.cross(gyro, self._g)
        n = np.linalg.norm(g_pred)
        if n > 1e-9:
            g_pred = g_pred / n * self.g_mag

        # 2) Magnitude-gated accel correction (gravity-down = -specific force).
        if self._lo <= np.linalg.norm(accel) <= self._hi:
            g_meas = -accel
            g_new = (1.0 - self.alpha) * g_pred + self.alpha * g_meas
            n2 = np.linalg.norm(g_new)
            if n2 > 1e-9:
                g_new = g_new / n2 * self.g_mag
        else:
            g_new = g_pred  # gyro-only fallback (spike rejected)

        self._g = g_new
        return self._g.copy()
