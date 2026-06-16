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
        """
        if control_mode not in ("attitude", "rates", "actuator"):
            raise ValueError(f"control_mode must be 'attitude', 'rates', or 'actuator'; got {control_mode!r}")
        self.control_mode = control_mode
        self._race_started_source = race_started_source if race_started_source is not None else (lambda: True)
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

        logger.info("[SCUBA Lab MAVLink] Loading model: %s", model_path)
        self.adapter = SCUBALabAdapter(model_path)
        logger.info("[SCUBA Lab MAVLink] Model loaded.")

        self.frame_builder = MAVLinkFrameBuilder()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.start_time = time.time()
        self.system_boot_ms = int(time.time() * 1000)
        self.frame_count = 0
        self.send_errors = 0  # never silenced; check this on submission day
        self._last_cmd_log = 0.0

        self.latest_telemetry: Dict = {
            "position":    [0.0, 0.0, 0.0],
            "velocity":    [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        }

    def get_time_boot_ms(self) -> int:
        return int((time.time() - self.start_time) * 1000)

    def process_telemetry(self, attitude: tuple, velocity: tuple, position: tuple):
        self.latest_telemetry["orientation"] = list(attitude)
        self.latest_telemetry["velocity"]    = list(velocity)
        self.latest_telemetry["position"]    = list(position)

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
        """
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
                tx_roll   = float(command["roll"])   * MAX_BODY_RATE
                tx_pitch  = float(command["pitch"])  * MAX_BODY_RATE
                tx_yaw    = float(command["yaw"])    * MAX_BODY_RATE
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
            frame = self.frame_builder.set_attitude_target(
                time_boot_ms=self.get_time_boot_ms(),
                body_roll_rate=float(command["roll"])   * MAX_BODY_RATE,
                body_pitch_rate=float(command["pitch"]) * MAX_BODY_RATE,
                body_yaw_rate=float(command["yaw"])     * MAX_BODY_RATE,
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
                    telemetry.get("attitude",  (0.0, 0.0, 0.0)),
                    telemetry.get("velocity",  (0.0, 0.0, 0.0)),
                    telemetry.get("position",  (0.0, 0.0, 0.0)),
                )

            if now - last_control >= self.control_interval:
                self.step(frame)
                last_control = now

    def close(self):
        self._sock.close()
