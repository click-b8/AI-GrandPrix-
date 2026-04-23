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
    ) -> bytes:
        """Encode SET_ATTITUDE_TARGET using CTBR body rates.

        type_mask=128 (IGNORE_ATTITUDE): FC ignores quaternion, uses rates+thrust.

        body_*_rate values must already be in rad/s (policy_output * MAX_BODY_RATE).
        thrust must be in [0, 1].
        """
        msg = mav_common.MAVLink_set_attitude_target_message(
            time_boot_ms=time_boot_ms,
            target_system=TARGET_SYSTEM_ID,
            target_component=TARGET_COMPONENT_ID,
            type_mask=TYPE_MASK_BODY_RATES_ONLY,
            q=[1.0, 0.0, 0.0, 0.0],  # identity; ignored by FC
            body_roll_rate=body_roll_rate,
            body_pitch_rate=body_pitch_rate,
            body_yaw_rate=body_yaw_rate,
            thrust=thrust,
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
        udp_port: int = 14540,
        target_hz: float = 50.0,
    ):
        """
        Args:
            model_path: Path to aigp_distill_final.zip. Canonical location:
                        ./models_release/aigp_distill_final.zip
            udp_host:   Destination IP for MAVLink UDP frames.
            udp_port:   Destination port (DCL SITL default: 14540).
            target_hz:  Control command rate. VADR-TS-001 allows 50-120 Hz.
        """
        self.udp_host = udp_host
        self.udp_port = udp_port
        self.target_hz = target_hz
        self.control_interval = 1.0 / target_hz

        logger.info("[SCUBA Lab MAVLink] Loading model: %s", model_path)
        self.adapter = SCUBALabAdapter(model_path)
        logger.info("[SCUBA Lab MAVLink] Model loaded.")

        self.frame_builder = MAVLinkFrameBuilder()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.start_time = time.time()
        self.frame_count = 0
        self.send_errors = 0  # never silenced; check this on submission day

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
        """Send frame over UDP. Logs + counts errors; never swallows silently."""
        try:
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
        """Inference: vision + telemetry -> MAVLink SET_ATTITUDE_TARGET (sent + returned).

        Semantic mapping (CTBR):
          body_roll_rate  = command['roll']  * MAX_BODY_RATE  (rad/s)
          body_pitch_rate = command['pitch'] * MAX_BODY_RATE  (rad/s)
          body_yaw_rate   = command['yaw']   * MAX_BODY_RATE  (rad/s)
          thrust          = command['throttle']               ([0,1])
        """
        command = self.adapter.step(self.latest_telemetry, vision_frame)

        frame = self.frame_builder.set_attitude_target(
            time_boot_ms=self.get_time_boot_ms(),
            body_roll_rate=float(command["roll"])   * MAX_BODY_RATE,
            body_pitch_rate=float(command["pitch"]) * MAX_BODY_RATE,
            body_yaw_rate=float(command["yaw"])     * MAX_BODY_RATE,
            thrust=float(command["throttle"]),
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
