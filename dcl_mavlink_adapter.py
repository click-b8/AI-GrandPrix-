"""
SCUBA LAB - DCL MAVLink Adapter
Bridges trained distilled vision model to DCL competition API via MAVLink v2.

Implements VADR-TS-001 specification:
- SET_ATTITUDE_TARGET with IGNORE_ATTITUDE type_mask (body-rate CTBR control)
- UDP MAVLink v2 transport via pymavlink
- Heartbeat at 2 Hz, control commands at 50-100 Hz
- Incrementing sequence byte, correct CRC-16-CCITT per message

Control semantic:
  Policy outputs [throttle in [0,1], roll in [-1,1], pitch in [-1,1], yaw in [-1,1]]
  These are CTBR body-rate commands scaled by MAX_BODY_RATE = 12.0 rad/s.
  Written into SET_ATTITUDE_TARGET body_roll/pitch/yaw_rate fields with
  type_mask = 0b10000000 = 128 (IGNORE_ATTITUDE), so the flight controller
  ignores the quaternion and tracks body rates + thrust directly.

Wire format (verified against pymavlink common dialect, msg_id=82, crc_extra=49):
  Payload 39 bytes (MAVLink native/wire field ordering, largest type first):
    uint32  time_boot_ms       4 bytes
    float   q[4]              16 bytes  (identity quaternion; ignored)
    float   body_roll_rate     4 bytes  rad/s
    float   body_pitch_rate    4 bytes  rad/s
    float   body_yaw_rate      4 bytes  rad/s
    float   thrust             4 bytes  [0,1]
    uint8   target_system      1 byte
    uint8   target_component   1 byte
    uint8   type_mask          1 byte
  Total: 39 bytes

CRC seed bytes confirmed from pymavlink MAVLink_*_message.crc_extra:
  SET_ATTITUDE_TARGET msg_id=82  crc_extra=49
  HEARTBEAT           msg_id=0   crc_extra=50
"""

import logging
import socket
import struct
import threading
import time
from typing import Dict

import numpy as np
from pymavlink.dialects.v20 import common as mav_common

from dcl_adapter import SCUBALabAdapter

# Body-rate scale: must match training config.py MAX_BODY_RATE
MAX_BODY_RATE = 12.0  # rad/s

# ---------------------------------------------------------------------------
# Per-axis PLANT calibration for the DCL sim (build v3385, measured 2026-07-22
# via tools/probe_rate_vs_angle.py -- constant-rate sweeps, gyro time-series).
#
# The sim is genuine RATE control (linear, holds) despite its "FLIGHT MODE:
# ANGLE" UI label, BUT every body-rate axis is SIGN-INVERTED and ~2.5x hot:
#
#   gyro_measured = K_axis * wire_rate
#     pitch: cmd -0.15/-0.30/-0.60 -> +0.371/+0.747/+1.572  => K = -2.49
#     roll : cmd -0.30/-0.15/+0.30 -> +0.763/+0.388/-0.760  => K = -2.55
#     yaw  : cmd -0.30/-0.15/+0.30 -> +0.667/+0.323/-0.667  => K = -2.20  (softer)
#
# To make the plant achieve an INTENDED body rate R, send wire = R / K_axis
# (this folds in BOTH the sign flip and the scale). Applied at the single point
# where a normalised command becomes a wire rate, so every controller -- the RL
# policy AND the vision servo -- is corrected in one place. Left uncorrected, a
# full command (norm 1.0 -> 12 rad/s wire) drove the plant to ~30 rad/s with the
# wrong sign: positive feedback -> the documented distill tumble.
#
# Re-measure and update these if the sim build changes. The rate-probe sends RAW
# (bypasses this), so post-fix it must still read K~-2.5 raw.
#
# NOTE: applies to the body-rate paths (SET_ATTITUDE_TARGET, control_mode
# attitude/rates). The 'actuator' path (SET_ACTUATOR_CONTROL_TARGET) is a
# different interface and is NOT calibrated here -- measure separately if used.
PLANT_RATE_CALIB = {"roll": -2.55, "pitch": -2.49, "yaw": -2.20}


def wire_body_rate(axis: str, command_norm: float) -> float:
    """Normalised command [-1,1] on `axis` -> calibrated wire body rate (rad/s).

    intended R = command_norm * MAX_BODY_RATE; wire = R / K_axis so the plant,
    which yields K_axis * wire, actually delivers R. See PLANT_RATE_CALIB."""
    return float(command_norm) * MAX_BODY_RATE / PLANT_RATE_CALIB[axis]

# --hover-probe diagnostic: after GO, hold a fixed thrust with zero body rates
# for this long while logging vz, then cut thrust to 0. Used to bracket DCL's
# hover-thrust fraction across separate races (vz ~= 0 => that thrust hovers).
HOVER_PROBE_DURATION_S = 2.0
HOVER_PROBE_HOVER_EPS = 0.1   # |climb rate| below this (m/s) reads as hover

# type_mask = 128 = 0b10000000
# bit 7 (0x80): IGNORE_ATTITUDE = 1  -> ignore quaternion
# bits 0-6: all 0 -> use body rates + thrust
TYPE_MASK_BODY_RATES_ONLY = 128

# MAVLink system/component IDs
OUR_SYSTEM_ID = 1
OUR_COMPONENT_ID = 1
TARGET_SYSTEM_ID = 1
TARGET_COMPONENT_ID = 1

# MAV_TYPE_QUADROTOR = 2
MAV_TYPE_QUADROTOR = 2
MAV_AUTOPILOT_INVALID = 8
MAV_STATE_ACTIVE = 4

logger = logging.getLogger(__name__)


def _compute_crc16(data: bytes, crc_extra: int) -> int:
    """CRC-16-CCITT over MAVLink header+payload bytes, then fold in crc_extra seed."""
    crc = 0xFFFF
    for byte in data:
        tmp = byte ^ (crc & 0xFF)
        tmp = (tmp ^ (tmp << 4)) & 0xFF
        crc = ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF
    tmp = crc_extra ^ (crc & 0xFF)
    tmp = (tmp ^ (tmp << 4)) & 0xFF
    crc = ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF
    return crc


class MAVLinkFrameBuilder:
    """Spec-compliant MAVLink v2 frame builder.

    Uses pymavlink message classes for payload encoding (correct field
    ordering guaranteed), our own CRC computation, and a monotonically
    incrementing sequence counter.
    """

    STX = 0xFD  # MAVLink v2 start byte

    def __init__(self, system_id: int = OUR_SYSTEM_ID, component_id: int = OUR_COMPONENT_ID):
        self.system_id = system_id
        self.component_id = component_id
        self._seq = 0
        self._mav = mav_common.MAVLink(None, srcSystem=system_id, srcComponent=component_id)

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq = (self._seq + 1) & 0xFF
        return seq

    def _build_frame(self, msg) -> bytes:
        """Build a complete MAVLink v2 frame from a pymavlink message object.

        pymavlink .pack() handles payload field ordering. We rebuild the
        header with our own sequence byte, then compute CRC ourselves.
        """
        packed_full = msg.pack(self._mav)
        # MAVLink v2 frame layout (after STX):
        #   len(1) incompat(1) compat(1) seq(1) sysid(1) compid(1) msgid(3) payload(N) crc(2)
        # Header = 10 bytes total (STX + 9 fields), CRC = last 2 bytes.
        payload_bytes = packed_full[10:-2]

        seq = self._next_seq()
        header = bytes([
            len(payload_bytes),   # payload length
            0,                    # incompat_flags
            0,                    # compat_flags
            seq,
            self.system_id,
            self.component_id,
        ]) + struct.pack('<I', msg.id)[:3]  # 3-byte little-endian msg_id

        crc = _compute_crc16(header + payload_bytes, msg.crc_extra)
        return bytes([self.STX]) + header + payload_bytes + struct.pack('<H', crc)

    def set_attitude_target(
        self,
        time_boot_ms: int,
        body_roll_rate: float,
        body_pitch_rate: float,
        body_yaw_rate: float,
        thrust: float,
        target_system: int = TARGET_SYSTEM_ID,
        target_component: int = TARGET_COMPONENT_ID,
    ) -> bytes:
        """Encode SET_ATTITUDE_TARGET using CTBR body rates.

        type_mask=128 (IGNORE_ATTITUDE): FC ignores quaternion, uses rates+thrust.

        body_*_rate values must already be in rad/s (policy_output * MAX_BODY_RATE).
        thrust must be in [0, 1].
        target_system / target_component are discovered from the sim's heartbeat;
        default to 1/1 for tests and stub mode.
        """
        msg = mav_common.MAVLink_set_attitude_target_message(
            time_boot_ms=time_boot_ms,
            target_system=target_system,
            target_component=target_component,
            type_mask=TYPE_MASK_BODY_RATES_ONLY,
            q=[1.0, 0.0, 0.0, 0.0],  # identity; ignored by FC
            body_roll_rate=body_roll_rate,
            body_pitch_rate=body_pitch_rate,
            body_yaw_rate=body_yaw_rate,
            thrust=thrust,
        )
        return self._build_frame(msg)

    def set_actuator_control_target(
        self,
        controls: list,
        target_system: int = TARGET_SYSTEM_ID,
        target_component: int = TARGET_COMPONENT_ID,
        group_mlx: int = 0,
    ) -> bytes:
        """Encode SET_ACTUATOR_CONTROL_TARGET matching PyAIPilotExample's update_motor_control().

        time_usec uses UNIX microseconds (int(time.time() * 1e6)) per the example.
        controls[8] normed to -1..+1; group 0 layout: [roll, pitch, yaw, throttle, 0, 0, 0, 0].

        Note: the example's set_actuator_control_target_send() call has positional
        args in the wrong order vs. pymavlink's actual signature. We use keyword args
        via MAVLink_set_actuator_control_target_message to get the correct field mapping.
        """
        msg = mav_common.MAVLink_set_actuator_control_target_message(
            time_usec=int(time.time() * 1e6),
            group_mlx=group_mlx,
            target_system=target_system,
            target_component=target_component,
            controls=controls,
        )
        return self._build_frame(msg)

    def heartbeat(self) -> bytes:
        """Encode HEARTBEAT. MAV_TYPE_QUADROTOR=2 (correct for our platform)."""
        msg = mav_common.MAVLink_heartbeat_message(
            type=MAV_TYPE_QUADROTOR,
            autopilot=MAV_AUTOPILOT_INVALID,
            base_mode=0,
            custom_mode=0,
            system_status=MAV_STATE_ACTIVE,
            mavlink_version=3,
        )
        return self._build_frame(msg)


class DCLTimesync:
    """Background thread sending TIMESYNC at 10 Hz over a pymavlink connection.

    Matches timesync.py from the PyAIPilotExample (tc1=now_ns, ts1=0).
    Call start() only after wait_heartbeat() so the connection has a known target.
    """

    TIMESYNC_HZ = 10

    def __init__(self, sim_conn):
        self._conn = sim_conn
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="DCLTimesync"
        )
        self._thread.start()
        logger.info("[DCL Timesync] Started at %d Hz", self.TIMESYNC_HZ)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        while self._running:
            now_ns = int(time.time_ns())
            try:
                self._conn.mav.timesync_send(now_ns, 0)
            except Exception as exc:
                logger.debug("[DCL Timesync] Send error: %s", exc)
            time.sleep(1.0 / self.TIMESYNC_HZ)


class SCUBALabMAVLinkAdapter:
    """MAVLink-compliant adapter wrapping the distilled vision policy.

    Loads the trained model, receives telemetry + vision frames, emits
    spec-compliant SET_ATTITUDE_TARGET MAVLink v2 frames over UDP.

    All send errors are logged at WARNING level and counted in self.send_errors.
    No silent swallowing of exceptions anywhere in the send path.
    """

    def __init__(
        self,
        model_path: str,
        udp_host: str = "127.0.0.1",
        udp_port: int = 14550,
        target_hz: float = 50.0,
        sim_conn=None,
        control_mode: str = "attitude",
        race_started_source=None,
        device: str = None,
        hover_probe: float = None,
        command_source=None,
    ):
        """
        Args:
            model_path:    Path to aigp_distill_final.zip. Canonical location:
                           ./models_release/aigp_distill_final.zip
            udp_host:      Fallback destination IP (stub mode only; real mode uses sim_conn).
            udp_port:      Fallback destination port (stub mode only).
            target_hz:     Control command rate. 250 Hz matches PyAIPilotExample CONTROL_HZ.
            sim_conn:      pymavlink MAVLink connection returned by mavutil.mavlink_connection()
                           after wait_heartbeat(). When provided, all sends go via
                           sim_conn.write() and target_system/component are discovered
                           from the heartbeat. Pass None in stub/test mode.
            control_mode:  'attitude' or 'rates' — SET_ATTITUDE_TARGET type_mask=128 (CTBR);
                           'actuator' — SET_ACTUATOR_CONTROL_TARGET group 0 at target_hz,
                           matching update_motor_control() from PyAIPilotExample.
            device:        'cuda', 'mps', or 'cpu'.  None = auto-select in SCUBALabAdapter.
            hover_probe:   Diagnostic. When set (a thrust in [0,1]), the model is
                           IGNORED: after GO the adapter commands this fixed thrust
                           with zero body rates for HOVER_PROBE_DURATION_S, logging
                           vz each frame, then cuts thrust. None = normal operation.
            command_source: Optional zero-arg callable returning a command dict
                           {throttle,roll,pitch,yaw}. When provided, the RL model is
                           NOT loaded and this is used as the control source instead
                           (the hardcoded vision-servo path, vq1_vision_servo.py).
                           It owns its own inputs (pulls its full-res frame + IMU +
                           active_gate). Same GO-gating/encoding/send apply.
        """
        if control_mode not in ("attitude", "rates", "actuator"):
            raise ValueError(f"control_mode must be 'attitude', 'rates', or 'actuator'; got {control_mode!r}")
        self.control_mode = control_mode
        self._race_started_source = race_started_source if race_started_source is not None else (lambda: True)
        self._race_started_prev = False   # for GO-edge detection (reset obs on False->True)

        # --hover-probe diagnostic state (None = disabled; model runs normally).
        self.hover_probe = hover_probe
        self._probe_start_time = None
        self._probe_summary_logged = False
        self._probe_vz_sum = 0.0
        self._probe_vz_n = 0
        self._probe_z0 = None
        self.udp_host = udp_host
        self.udp_port = udp_port
        self.target_hz = target_hz
        self.control_interval = 1.0 / target_hz

        self._sim_conn = sim_conn
        if sim_conn is not None:
            self.target_system   = sim_conn.target_system
            self.target_component = sim_conn.target_component
        else:
            self.target_system   = TARGET_SYSTEM_ID
            self.target_component = TARGET_COMPONENT_ID

        self._command_source = command_source
        if command_source is not None:
            self.adapter = None
            logger.info("[SCUBA Lab MAVLink] command_source provided -> RL model "
                        "NOT loaded; using hardcoded controller.")
        else:
            logger.info("[SCUBA Lab MAVLink] Loading model: %s", model_path)
            self.adapter = SCUBALabAdapter(model_path, device=device)
            logger.info("[SCUBA Lab MAVLink] Model loaded.")

        self.frame_builder = MAVLinkFrameBuilder()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.start_time = time.time()
        self.system_boot_ms = int(time.time() * 1000)
        self.frame_count = 0
        self.send_errors = 0  # never silenced; check this on submission day
        self._last_cmd_log = 0.0

        self.latest_telemetry: Dict = {
            "position":         [0.0, 0.0, 0.0],
            "velocity":         [0.0, 0.0, 0.0],   # body angular rates rad/s
            "orientation":      [0.0, 0.0, 0.0],   # Euler roll/pitch/yaw rad
            "linear_velocity":  [0.0, 0.0, 0.0],   # world/NED linear vel m/s
        }

    def get_time_boot_ms(self) -> int:
        return int((time.time() - self.start_time) * 1000)

    def process_telemetry(
        self,
        attitude: tuple,
        velocity: tuple,
        position: tuple,
        linear_velocity: tuple = (0.0, 0.0, 0.0),
    ):
        self.latest_telemetry["orientation"]     = list(attitude)
        self.latest_telemetry["velocity"]        = list(velocity)
        self.latest_telemetry["position"]        = list(position)
        self.latest_telemetry["linear_velocity"] = list(linear_velocity)

    def _send(self, frame: bytes) -> bool:
        """Send frame over UDP. Routes via sim_conn.write() when a real connection
        exists (address discovered from heartbeat), else falls back to raw sendto
        for stub/test mode. Logs + counts errors; never swallows silently."""
        try:
            if self._sim_conn is not None:
                self._sim_conn.write(frame)
            else:
                self._sock.sendto(frame, (self.udp_host, self.udp_port))
            return True
        except OSError as exc:
            self.send_errors += 1
            logger.warning(
                "[SCUBA Lab MAVLink] UDP send failed (#%d): %s",
                self.send_errors, exc,
            )
            return False

    def _hover_probe_command(self) -> Dict:
        """Diagnostic command source for --hover-probe (model ignored).

        Before GO: zero (the wire-gate also zeros pre-GO, but be explicit).
        For HOVER_PROBE_DURATION_S after GO: fixed thrust, zero body rates,
        logging vz each frame. After the window: cut thrust to 0 and log a
        one-time verdict. vz/z are NED (+down); climb rate = -vz (+up).
        """
        zero = {"throttle": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        if not self._race_started_source():
            return zero

        now = time.time()
        if self._probe_start_time is None:
            self._probe_start_time = now
            self._probe_z0 = float(self.latest_telemetry["position"][2])
            logger.info(
                "[HOVER PROBE] GO — holding thrust=%.3f, zero rates, for %.1fs. "
                "vz/z are NED (+down); climb = -vz (+up).",
                self.hover_probe, HOVER_PROBE_DURATION_S,
            )

        elapsed = now - self._probe_start_time
        if elapsed <= HOVER_PROBE_DURATION_S:
            vz = float(self.latest_telemetry["linear_velocity"][2])
            z = float(self.latest_telemetry["position"][2])
            self._probe_vz_sum += vz
            self._probe_vz_n += 1
            logger.info(
                "[HOVER PROBE] t=%.3fs thrust=%.3f  vz_ned=%+.4f m/s  "
                "climb=%+.4f m/s(+up)  z_ned=%+.3f m",
                elapsed, self.hover_probe, vz, -vz, z,
            )
            return {"throttle": float(self.hover_probe), "roll": 0.0, "pitch": 0.0, "yaw": 0.0}

        if not self._probe_summary_logged:
            self._probe_summary_logged = True
            mean_vz = self._probe_vz_sum / max(self._probe_vz_n, 1)
            mean_climb = -mean_vz
            net_climb = -(float(self.latest_telemetry["position"][2]) - (self._probe_z0 or 0.0))
            if abs(mean_climb) < HOVER_PROBE_HOVER_EPS:
                verdict = "~= HOVER (vz~0) -- thrust ~= DCL hover fraction"
            elif mean_climb > 0:
                verdict = "CLIMBING — thrust ABOVE hover"
            else:
                verdict = "SINKING — thrust BELOW hover"
            logger.info(
                "[HOVER PROBE] DONE thrust=%.3f over %.1fs (n=%d): mean climb=%+.4f m/s "
                "(mean vz_ned=%+.4f), net climb=%+.3f m  ->  %s. Thrust now cut to 0.",
                self.hover_probe, HOVER_PROBE_DURATION_S, self._probe_vz_n,
                mean_climb, mean_vz, net_climb, verdict,
            )
        return zero

    def step(self, vision_frame: np.ndarray) -> bytes:
        """Inference: vision + telemetry -> MAVLink control frame (sent + returned).

        attitude / rates: SET_ATTITUDE_TARGET type_mask=128 (CTBR body rates).
          body_roll_rate  = command['roll']  * MAX_BODY_RATE  (rad/s)
          body_pitch_rate = command['pitch'] * MAX_BODY_RATE  (rad/s)
          body_yaw_rate   = command['yaw']   * MAX_BODY_RATE  (rad/s)
          thrust          = command['throttle']               ([0,1])

        actuator: SET_ACTUATOR_CONTROL_TARGET group 0 (RPYT normed to [-1,1]).
          controls = [roll, pitch, yaw, throttle, 0, 0, 0, 0]
          No MAX_BODY_RATE scaling — actuator controls are already normed.

        When --hover-probe is active the model is ignored and a fixed-thrust
        command is substituted; everything downstream (gating, encoding, send)
        is identical, so only what goes on the wire changes.
        """
        # On the GO edge, clear stale per-race observation state (frame deque +
        # prev_action) so a prior race can't bleed into the first frames.
        race_now = self._race_started_source()
        if race_now and not self._race_started_prev and self.adapter is not None:
            self.adapter.reset_observation()
        self._race_started_prev = race_now

        if self.hover_probe is not None:
            command = self._hover_probe_command()
        elif self._command_source is not None:
            command = self._command_source()
        else:
            command = self.adapter.step(self.latest_telemetry, vision_frame)

        if self.control_mode == "actuator":
            frame = self.frame_builder.set_actuator_control_target(
                controls=[
                    float(command["roll"]),
                    float(command["pitch"]),
                    float(command["yaw"]),
                    float(command["throttle"]),
                    0.0, 0.0, 0.0, 0.0,
                ],
                target_system=self.target_system,
                target_component=self.target_component,
            )
            self._send(frame)
        elif self._sim_conn is not None:
            # Real sim mode: delegate encoding+send to pymavlink, matching
            # update_attitude_flight_control() in PyAIPilotExample controller.py.
            now_ms = int(time.time() * 1000)
            if self._race_started_source():
                # PLANT_RATE_CALIB: wire = intended / K_axis (sign + scale). See top.
                tx_roll   = wire_body_rate("roll",  command["roll"])
                tx_pitch  = wire_body_rate("pitch", command["pitch"])
                tx_yaw    = wire_body_rate("yaw",   command["yaw"])
                tx_thrust = float(command["throttle"])
            else:
                tx_roll = tx_pitch = tx_yaw = tx_thrust = 0.0
            self._sim_conn.mav.set_attitude_target_send(
                now_ms - self.system_boot_ms,
                self._sim_conn.target_system,
                self._sim_conn.target_component,
                128,           # ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
                [1, 0, 0, 0],  # identity quaternion (ignored)
                tx_roll, tx_pitch, tx_yaw, tx_thrust,
            )
            now = time.time()
            if self.frame_count == 0 or now - self._last_cmd_log >= 1.0:
                self._last_cmd_log = now
                logger.info(
                    "[CMD out] frame=%d  race_started=%s  thrust=%.4f  "
                    "roll=%.3f rad/s  pitch=%.3f rad/s  yaw=%.3f rad/s",
                    self.frame_count + 1,
                    self._race_started_source(),
                    tx_thrust, tx_roll, tx_pitch, tx_yaw,
                )
            frame = b""
        else:  # 'attitude' or 'rates', stub/test mode — use custom frame builder
            # Same PLANT_RATE_CALIB correction as the real-sim path (one rule).
            frame = self.frame_builder.set_attitude_target(
                time_boot_ms=self.get_time_boot_ms(),
                body_roll_rate=wire_body_rate("roll",  command["roll"]),
                body_pitch_rate=wire_body_rate("pitch", command["pitch"]),
                body_yaw_rate=wire_body_rate("yaw",   command["yaw"]),
                thrust=float(command["throttle"]),
                target_system=self.target_system,
                target_component=self.target_component,
            )
            self._send(frame)

        self.frame_count += 1
        return frame

    def send_heartbeat(self) -> bool:
        return self._send(self.frame_builder.heartbeat())

    async def run_loop(self, vision_stream_generator, telemetry_source):
        """Async control loop: heartbeat at 2 Hz, control at target_hz."""
        import asyncio
        heartbeat_interval = 0.5
        last_heartbeat = time.time()
        last_control = time.time()

        async for frame in vision_stream_generator:
            now = time.time()

            if now - last_heartbeat >= heartbeat_interval:
                if not self.send_heartbeat():
                    logger.warning("[SCUBA Lab MAVLink] Heartbeat send failed.")
                last_heartbeat = now

            telemetry = await telemetry_source()
            if telemetry:
                self.process_telemetry(
                    telemetry.get("attitude",        (0.0, 0.0, 0.0)),
                    telemetry.get("velocity",        (0.0, 0.0, 0.0)),
                    telemetry.get("position",        (0.0, 0.0, 0.0)),
                    telemetry.get("linear_velocity", (0.0, 0.0, 0.0)),
                )

            if now - last_control >= self.control_interval:
                self.step(frame)
                last_control = now

    def close(self):
        self._sock.close()
