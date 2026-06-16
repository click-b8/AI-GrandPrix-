# State-vector fix spec — `process_observation()` rewrite

**Written from:** `drone_race_env._get_minimal_state()` + `_quat_to_rotmat()` + `_rotmat_to_6d()`  
**Date:** 2026-06-16  
**Status:** spec written before code changes — do not modify this file retroactively

---

## Training layout (19D, from `_get_minimal_state()`)

```
d0–5    rot_6d        6D rotation (first two columns of body→world R)   float32
d6–8    lin_vel       linear velocity, world frame, m/s                  float32
d9–11   ang_rates     body-frame angular rates, rad/s                    float32
d12–15  prev_action   last command [throttle, roll, pitch, yaw] normed   float32
d16–18  position      world frame, m                                     float32
```

Total: 6 + 3 + 3 + 4 + 3 = 19 ✓

Source in `drone_race_env.py` (no-VIO path, which is what deployment uses):
```python
rot_6d        = _rotmat_to_6d(R)                        # R from quaternion
lin_vel       = self._data.qvel[0:3]                    # world-frame linear vel
ang_vel_body  = R.T @ self._data.qvel[3:6]              # body-frame angular rates
prev_action   = self._prev_action                       # 4D, normalised [-1,1]
pos           = self._data.qpos[0:3]                    # world-frame position
```

---

## 6D rotation construction (Zhou CVPR 2019, via `_rotmat_to_6d`)

**Training code does:**
```python
quat_wxyz = self._data.qpos[3:7]           # MuJoCo quaternion (w,x,y,z)
R = _quat_to_rotmat(w, x, y, z)            # body→world 3×3 matrix
rot_6d = concat(R[:,0], R[:,1])            # first column then second column
```

**At deployment we have:** MAVLink `ATTITUDE.roll/pitch/yaw` (ZYX Euler, rad).

**Step 1 — build R from ZYX Euler** (aerospace convention, same as `_quat_to_rotmat`):

Let cr=cos(roll), sr=sin(roll), cp=cos(pitch), sp=sin(pitch), cy=cos(yaw), sy=sin(yaw).

```
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

     col 0           col 1                        col 2
R = [cy·cp,   cy·sp·sr − sy·cr,   cy·sp·cr + sy·sr]   row 0
    [sy·cp,   sy·sp·sr + cy·cr,   sy·sp·cr − cy·sr]   row 1
    [−sp,     cp·sr,              cp·cr            ]   row 2
```

**Step 2 — extract 6D:**
```
rot_6d[0:3] = R[:,0] = [cy·cp,  sy·cp,  −sp]
rot_6d[3:6] = R[:,1] = [cy·sp·sr − sy·cr,  sy·sp·sr + cy·cr,  cp·sr]
```

**Sanity checks:**

| Attitude | rot_6d |
|----------|--------|
| Level (r=p=y=0) | [1, 0, 0, 0, 1, 0] |
| Yaw 90° only    | [0, 1, 0, −1, 0, 0] |
| Pitch 30° only  | [√3/2, 0, −0.5, 0, 1, 0] |

These were hand-verified against `_quat_to_rotmat` outputs.

---

## Source field mapping (dim by dim)

| Dims | Value | MAVLink source | Available? |
|------|-------|----------------|-----------|
| 0–5  | rot_6d | `ATTITUDE.roll/pitch/yaw` → build R → extract cols 0,1 | ✅ |
| 6–8  | linear velocity (m/s) | `LOCAL_POSITION_NED.vx/vy/vz` | ✅ in `_latest_telemetry["linear_velocity"]` — **NOT wired through `process_telemetry()`** (fix needed) |
| 9–11 | body angular rates (rad/s) | `ATTITUDE.rollspeed/pitchspeed/yawspeed` (overridden by `HIGHRES_IMU.xgyro/ygyro/zgyro` if received) | ✅ stored as `"velocity"` key |
| 12–15 | prev_action [0..1, −1..1×3] | stored in adapter after each inference | ✅ **NOT stored anywhere yet** (add `_prev_action` field) |
| 16–18 | position (m) | `LOCAL_POSITION_NED.x/y/z` | ✅ stored as `"position"` key |

**No hard blockers.** Two wiring issues to fix:
1. `linear_velocity` exists in `run_vq1._latest_telemetry` but is dropped by
   `SCUBALabMAVLinkAdapter.process_telemetry()` — add a 4th parameter.
2. `_prev_action` needs a new field in `SCUBALabAdapter.__init__()` initialized
   to zeros and updated inside `predict_action()`.

---

## Frame convention note (known residual mismatch)

Training uses **MuJoCo world frame** (right-handed, z-up). DCL uses **NED** (x-North, y-East, z-Down, right-handed, z-down). Implications:

- **rot_6d:** The z-component of R[:,0] and R[:,1] will flip sign. Training saw z-up world; deployment uses z-down NED. Effect on rot_6d: d2 (= −sin(pitch)) and d5 (= cos(pitch)·sin(roll)) — signs are consistent between ZYX Euler and the MuJoCo quaternion as long as the drone's pitch convention is consistent. In practice, both use standard aerospace angles so this should be fine.
- **lin_vel d8 (vz):** training = z-up positive climbing; NED = z-down negative climbing. Sign of vertical velocity is flipped.
- **position d18 (z):** same sign flip.

Gate targeting is driven by the CNN on the FPV image — position and velocity are supplementary context. The model should tolerate the z-sign mismatch. Flag this if the drone consistently inverts its vertical behavior in-sim; fix would be to negate vz and z before feeding.

---

## What was wrong in the old `process_observation()`

| Dims | Training expects | Old code sent |
|------|-----------------|---------------|
| 0–5  | rot_6d          | `LOCAL_POSITION_NED.x/y/z` then `ATTITUDE.rollspeed/pitchspeed/yawspeed` |
| 6–8  | lin_vel         | `ATTITUDE.roll/pitch/yaw` (Euler angles — wrong quantity, wrong units) |
| 9–11 | ang_rates       | zeros |
| 12–15| prev_action     | zeros |
| 16–18| position        | zeros |

Root causes:
1. `telemetry["orientation"]` → sent as dims 6-8 (should be used to *build* rot_6d)
2. `telemetry["position"]` → sent as dims 0-2 (should be dims 16-18)
3. `telemetry["velocity"]` (ang_rates) → sent as dims 3-5 (should be dims 9-11)
4. `linear_velocity` never in `latest_telemetry` → always zeros in dims 6-8
5. `_prev_action` never stored → always zeros in dims 12-15
